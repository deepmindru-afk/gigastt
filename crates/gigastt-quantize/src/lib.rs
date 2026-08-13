//! Dynamic INT8 (QOperator) quantization for ONNX encoder models.
//!
//! Native Rust replacement for `scripts/quantize.py`. Auto-invoked after
//! `gigastt download` and `gigastt serve` (see `src/main.rs`); also exposed
//! as the `gigastt quantize` subcommand.
//!
//! This emits the **dynamic INT8 (QOperator)** form that ONNX Runtime's
//! `quantize_dynamic(..., weight_type=QInt8)` produces:
//! `DynamicQuantizeLinear` on activations feeding integer compute kernels
//! (`MatMulInteger` / `ConvInteger`), with a per-channel float rescale on the
//! int32 output. This is fundamentally faster than weight-only `QDQ`
//! (`DequantizeLinear` → float `MatMul`/`Conv`), which stores int8 weights but
//! dequantizes them back to float at load and runs the heavy ops in FP32.
//!
//! The protobuf types come from `crate::onnx_proto`, which is generated at
//! build time from `proto/onnx.proto` via `prost-build` (see `build.rs`).
//! Fields that are `optional` in proto2 surface as `Option<T>` in prost
//! 0.13, so we lean on the generated accessor methods (`data_type()`,
//! `name()`, `op_type()`, …) for reads and wrap writes in `Some(_)`.

pub mod onnx_proto;

use anyhow::{Context, Result};
use prost::Message;
use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::onnx_proto::{AttributeProto, ModelProto, NodeProto, TensorProto};

/// ONNX data types (from onnx.proto `TensorProto.DataType`).
const FLOAT: i32 = 1;
const INT8: i32 = 3;
const INT64: i32 = 7;

/// ONNX attribute types (from onnx.proto `AttributeProto.AttributeType`).
const ATTR_INT: i32 = 2;

/// `Cast` `to` value for FLOAT (matches `TensorProto.DataType.FLOAT`).
const CAST_TO_FLOAT: i64 = FLOAT as i64;

/// Minimum opset (domain "") for `DynamicQuantizeLinear` (≥11).
/// `MatMulInteger` / `ConvInteger` need ≥10, so 11 covers everything we emit.
const MIN_OPSET: i64 = 11;

/// Node types whose weights benefit from INT8 quantization.
const QUANTIZABLE_OPS: &[&str] = &["MatMul", "Conv", "Gemm"];

/// Minimum number of elements in a tensor to quantize (skip small biases).
const MIN_ELEMENTS: usize = 1024;

/// A weight that has been quantized once and may be shared across ops.
struct QuantizedWeight {
    /// `{weight}_quantized` — INT8 initializer name.
    q_name: String,
    /// `{weight}_scale` — FLOAT [N] initializer name.
    s_name: String,
    /// `{weight}_zero_point` — INT8 [N] zeros initializer name.
    zp_name: String,
}

/// Quantize an ONNX model's float32 weights to dynamic INT8 (QOperator form).
///
/// For each quantizable weight tensor (MatMul/Conv/Gemm) the original float op
/// is **replaced** by a `DynamicQuantizeLinear` → `MatMulInteger`/`ConvInteger`
/// → `Cast` → per-channel `Mul` (+ optional bias `Add`) chain, with the weight
/// stored as a per-channel-symmetric INT8 initializer.
pub fn quantize_model(input: &Path, output: &Path) -> Result<()> {
    let model_bytes = std::fs::read(input).context("Failed to read ONNX model")?;
    let mut model =
        ModelProto::decode(&model_bytes[..]).context("Failed to decode ONNX protobuf")?;

    // Ensure opset (domain "") is high enough for the integer ops we emit.
    bump_opset(&mut model);

    let graph = model.graph.as_mut().context("Model has no graph")?;

    // Build map: initializer_name → index.
    let init_map: HashMap<String, usize> = graph
        .initializer
        .iter()
        .enumerate()
        .map(|(i, t)| (t.name().to_string(), i))
        .collect();

    // Collect quantization targets: (node_index, weight_input_index, weight_name, init_index).
    let mut targets = Vec::new();
    for (ni, node) in graph.node.iter().enumerate() {
        if !QUANTIZABLE_OPS.contains(&node.op_type()) {
            continue;
        }
        // Weight is input[1] for MatMul/Conv/Gemm.
        if node.input.len() < 2 {
            continue;
        }
        let weight_name = &node.input[1];
        if let Some(&init_idx) = init_map.get(weight_name) {
            let init = &graph.initializer[init_idx];
            if init.data_type() != FLOAT {
                continue;
            }
            let num_elements: i64 = init.dims.iter().product();
            if num_elements > 0 && num_elements as usize >= MIN_ELEMENTS {
                targets.push((ni, 1usize, weight_name.clone(), init_idx));
            }
        }
    }

    tracing::info!(
        "Found {} quantizable weight tensors in {} nodes",
        targets.len(),
        graph.node.len()
    );

    // First pass: quantize each distinct weight once (shared-weight dedup).
    let mut new_initializers = Vec::new();
    let mut quantized: HashMap<String, QuantizedWeight> = HashMap::new();

    for (node_idx, _input_idx, weight_name, init_idx) in &targets {
        if quantized.contains_key(weight_name) {
            continue;
        }

        let init = &graph.initializer[*init_idx];
        let float_data = extract_float_data(init)?;
        let dims = init.dims.clone();

        if dims.is_empty() {
            continue;
        }

        let expected_elements: usize = dims.iter().map(|&d: &i64| d.max(0) as usize).product();
        if expected_elements != float_data.len() {
            tracing::warn!(
                "Skipping tensor '{}': shape mismatch (dims={:?}, data={})",
                init.name(),
                dims,
                float_data.len()
            );
            continue;
        }

        // Pick the per-output-channel axis from the consuming op's semantics.
        // Quantizing along the wrong axis groups unrelated output channels under
        // one scale, silently inflating quantization error (and WER): a Conv
        // weight is `[out_channels, ...]` (axis 0), a MatMul weight is
        // `[..., K, N]` (output channel = last dim N), and a Gemm weight is
        // `[K, N]` or — when `transB=1` — `[N, K]`, so N's axis flips with it.
        let node = &graph.node[*node_idx];
        let axis = per_channel_axis(node.op_type(), node, dims.len());
        let channels = dims[axis].max(0) as usize;
        if channels == 0 {
            continue;
        }

        let (quantized_data, scales) = quantize_per_channel(&float_data, &dims, axis);
        let q_name = format!("{weight_name}_quantized");
        let s_name = format!("{weight_name}_scale");
        let zp_name = format!("{weight_name}_zero_point");

        // Quantized weight tensor (INT8).
        new_initializers.push(TensorProto {
            name: Some(q_name.clone()),
            dims,
            data_type: Some(INT8),
            raw_data: Some(quantized_data.iter().map(|&v| v as u8).collect()),
            ..Default::default()
        });

        // Per-channel scale (FLOAT [N]) — this carries the per-channel accuracy
        // and is applied as a float rescale on the integer op's int32 output.
        new_initializers.push(TensorProto {
            name: Some(s_name.clone()),
            dims: vec![channels as i64],
            data_type: Some(FLOAT),
            float_data: scales,
            ..Default::default()
        });

        // Weight zero-point: a per-TENSOR scalar INT8 zero. ONNX Runtime's CPU
        // `MatMulInteger` / `ConvInteger` kernels reject a per-channel weight
        // zero-point ("Non per-tensor quantization is not supported now"), and
        // for symmetric quantization the zero-point is 0 on every channel, so a
        // scalar 0 is numerically exact — the per-channel scale above keeps the
        // accuracy.
        new_initializers.push(TensorProto {
            name: Some(zp_name.clone()),
            dims: vec![],
            data_type: Some(INT8),
            raw_data: Some(vec![0u8]),
            ..Default::default()
        });

        quantized.insert(
            weight_name.clone(),
            QuantizedWeight {
                q_name,
                s_name,
                zp_name,
            },
        );
    }

    // Second pass: build the replacement op chain for every quantizable node.
    // `replacements[node_idx] = Vec<NodeProto>` substitutes the original node.
    let mut replacements: HashMap<usize, Vec<NodeProto>> = HashMap::new();
    let mut chain_initializers = Vec::new();

    for (node_idx, _input_idx, weight_name, _init_idx) in &targets {
        let Some(qw) = quantized.get(weight_name) else {
            continue; // weight was skipped above (shape mismatch / zero channels)
        };
        let node = &graph.node[*node_idx];
        let op_type = node.op_type();

        let chain = match op_type {
            "Conv" => build_conv_chain(node, qw, &graph.initializer, &mut chain_initializers),
            // MatMul and Gemm-as-MatMul: B is the per-channel weight on its N axis.
            _ => build_matmul_chain(node, qw),
        };
        replacements.insert(*node_idx, chain);
    }

    // Reassemble the node list: drop replaced float ops, splice in their chains,
    // and prepend any leftover quantizable nodes' chains in graph order.
    let original_nodes = std::mem::take(&mut graph.node);
    let mut rebuilt = Vec::with_capacity(original_nodes.len());
    for (idx, node) in original_nodes.into_iter().enumerate() {
        if let Some(chain) = replacements.remove(&idx) {
            rebuilt.extend(chain);
        } else {
            rebuilt.push(node);
        }
    }
    graph.node = rebuilt;

    // Remove original float weight initializers that we quantized.
    let quantized_weight_names: HashSet<&str> = quantized.keys().map(|s| s.as_str()).collect();
    graph
        .initializer
        .retain(|t| !quantized_weight_names.contains(t.name()));

    // Add quantized weight initializers + chain reshape/shape initializers.
    graph.initializer.extend(new_initializers);
    graph.initializer.extend(chain_initializers);

    // Write quantized model (atomic: write to partial, then rename).
    // Uses the `.partial` suffix convention shared with `src/model/mod.rs`
    // downloads so both pipelines leave identical breadcrumbs after a crash.
    let mut output_bytes = Vec::new();
    model
        .encode(&mut output_bytes)
        .context("Failed to encode quantized model")?;
    let mut partial_os: std::ffi::OsString = output.as_os_str().to_owned();
    partial_os.push(".partial");
    let partial = std::path::PathBuf::from(partial_os);
    std::fs::write(&partial, &output_bytes).context("Failed to write quantized model")?;
    std::fs::rename(&partial, output).context("Failed to finalize quantized model")?;

    let in_mb = model_bytes.len() as f64 / (1024.0 * 1024.0);
    let out_mb = output_bytes.len() as f64 / (1024.0 * 1024.0);
    tracing::info!(
        "Quantized: {in_mb:.0}MB → {out_mb:.0}MB ({:.1}x smaller)",
        in_mb / out_mb
    );

    Ok(())
}

/// Ensure the model imports opset (domain "") ≥ [`MIN_OPSET`]; bump it if lower
/// and add the default operator set if the model declares none.
fn bump_opset(model: &mut ModelProto) {
    let default = model
        .opset_import
        .iter_mut()
        .find(|o| o.domain() == "" || o.domain() == "ai.onnx");
    match default {
        Some(o) => {
            if o.version() < MIN_OPSET {
                o.version = Some(MIN_OPSET);
            }
        }
        None => {
            model
                .opset_import
                .push(crate::onnx_proto::OperatorSetIdProto {
                    domain: Some(String::new()),
                    version: Some(MIN_OPSET),
                });
        }
    }
}

/// Build the dynamic-INT8 replacement chain for a `MatMul`/`Gemm`:
/// `Y = (Cast(MatMulInteger(DynQ(A), W)) * (a_scale * W_scale))`.
fn build_matmul_chain(node: &NodeProto, qw: &QuantizedWeight) -> Vec<NodeProto> {
    let base = node_base_name(node);
    let a_input = node.input[0].clone();
    let y_output = node.output[0].clone();

    let a_q = format!("{base}_a_q");
    let a_scale = format!("{base}_a_scale");
    let a_zp = format!("{base}_a_zp");
    let mm_i32 = format!("{base}_mm_i32");
    let mm_f32 = format!("{base}_mm_f32");
    let scale_vec = format!("{base}_scale_vec");

    vec![
        // DynamicQuantizeLinear(A) → (a_q: uint8, a_scale: f32 scalar, a_zp: uint8 scalar)
        NodeProto {
            op_type: Some("DynamicQuantizeLinear".into()),
            input: vec![a_input],
            output: vec![a_q.clone(), a_scale.clone(), a_zp.clone()],
            name: Some(format!("{base}_dynq")),
            ..Default::default()
        },
        // MatMulInteger((A - a_zp) @ (W - 0)) → int32. A is uint8, W int8.
        NodeProto {
            op_type: Some("MatMulInteger".into()),
            input: vec![a_q, qw.q_name.clone(), a_zp, qw.zp_name.clone()],
            output: vec![mm_i32.clone()],
            name: Some(format!("{base}_matmulinteger")),
            ..Default::default()
        },
        // Cast int32 → float.
        cast_to_float(&mm_i32, &mm_f32, &format!("{base}_cast")),
        // Combined per-channel scale: a_scale (scalar) * W_scale ([N]) → [N].
        NodeProto {
            op_type: Some("Mul".into()),
            input: vec![a_scale, qw.s_name.clone()],
            output: vec![scale_vec.clone()],
            name: Some(format!("{base}_scale_mul")),
            ..Default::default()
        },
        // Rescale: mm_f32 ([..., N]) * scale_vec ([N]) → Y (original output name).
        NodeProto {
            op_type: Some("Mul".into()),
            input: vec![mm_f32, scale_vec],
            output: vec![y_output],
            name: Some(format!("{base}_rescale")),
            ..Default::default()
        },
    ]
}

/// Build the dynamic-INT8 replacement chain for a `Conv`:
/// `Y = Cast(ConvInteger(DynQ(A), W)) * reshape(a_scale * W_scale) [+ reshape(bias)]`.
///
/// The Conv's spatial attributes (`strides`, `pads`, `dilations`, `group`,
/// `kernel_shape`, `auto_pad`) are copied verbatim onto `ConvInteger`, which
/// takes no bias — any bias is folded back as a trailing `Add`.
fn build_conv_chain(
    node: &NodeProto,
    qw: &QuantizedWeight,
    initializers: &[TensorProto],
    chain_initializers: &mut Vec<TensorProto>,
) -> Vec<NodeProto> {
    let base = node_base_name(node);
    let a_input = node.input[0].clone();
    let y_output = node.output[0].clone();

    // Conv weight is [C_out, C_in/groups, *kernel]; the conv output is NCHW-like
    // with the channel on axis 1. The reshape rank matches the weight rank.
    let weight_init = initializers.iter().find(|t| t.name() == node.input[1]);
    let weight_rank = weight_init.map(|t| t.dims.len()).unwrap_or(4).max(2);
    let c_out = weight_init.map(|t| t.dims[0].max(0)).unwrap_or(0);

    let a_q = format!("{base}_a_q");
    let a_scale = format!("{base}_a_scale");
    let a_zp = format!("{base}_a_zp");
    let ci_i32 = format!("{base}_ci_i32");
    let ci_f32 = format!("{base}_ci_f32");
    let scale_c = format!("{base}_scale_c");
    let scale_reshaped = format!("{base}_scale_reshaped");
    let scaled = format!("{base}_scaled");

    let mut nodes = vec![
        // DynamicQuantizeLinear(A) → (a_q: uint8, a_scale: f32, a_zp: uint8)
        NodeProto {
            op_type: Some("DynamicQuantizeLinear".into()),
            input: vec![a_input],
            output: vec![a_q.clone(), a_scale.clone(), a_zp.clone()],
            name: Some(format!("{base}_dynq")),
            ..Default::default()
        },
        // ConvInteger(A, W, a_zp, W_zp) → int32, carrying the original conv attrs.
        NodeProto {
            op_type: Some("ConvInteger".into()),
            input: vec![a_q, qw.q_name.clone(), a_zp, qw.zp_name.clone()],
            output: vec![ci_i32.clone()],
            name: Some(format!("{base}_convinteger")),
            attribute: copy_conv_attrs(node),
            ..Default::default()
        },
        // Cast int32 → float.
        cast_to_float(&ci_i32, &ci_f32, &format!("{base}_cast")),
        // Combined per-channel scale: a_scale (scalar) * W_scale ([C_out]) → [C_out].
        NodeProto {
            op_type: Some("Mul".into()),
            input: vec![a_scale, qw.s_name.clone()],
            output: vec![scale_c.clone()],
            name: Some(format!("{base}_scale_mul")),
            ..Default::default()
        },
    ];

    // Reshape scale [C_out] → [1, C_out, 1, ...] so it broadcasts over the
    // channel axis (axis 1) of the conv output.
    let scale_shape = channel_broadcast_shape(c_out, weight_rank);
    chain_initializers.push(int64_shape_initializer(
        &format!("{base}_scale_shape"),
        &scale_shape,
    ));
    nodes.push(reshape_node(
        &scale_c,
        &format!("{base}_scale_shape"),
        &scale_reshaped,
        &format!("{base}_scale_reshape"),
    ));

    // Does the conv carry a bias (input[2] = float initializer)?
    let bias_name = node.input.get(2).filter(|n| !n.is_empty()).cloned();
    let has_bias = bias_name
        .as_deref()
        .map(|b| initializers.iter().any(|t| t.name() == b))
        .unwrap_or(false);

    if has_bias {
        // scaled = ci_f32 * reshaped_scale
        nodes.push(NodeProto {
            op_type: Some("Mul".into()),
            input: vec![ci_f32, scale_reshaped],
            output: vec![scaled.clone()],
            name: Some(format!("{base}_rescale")),
            ..Default::default()
        });
        // Reshape bias [C_out] → [1, C_out, 1, ...] and add.
        let bias = bias_name.expect("has_bias implies a bias name");
        let bias_reshaped = format!("{base}_bias_reshaped");
        let bias_shape = channel_broadcast_shape(c_out, weight_rank);
        chain_initializers.push(int64_shape_initializer(
            &format!("{base}_bias_shape"),
            &bias_shape,
        ));
        nodes.push(reshape_node(
            &bias,
            &format!("{base}_bias_shape"),
            &bias_reshaped,
            &format!("{base}_bias_reshape"),
        ));
        nodes.push(NodeProto {
            op_type: Some("Add".into()),
            input: vec![scaled, bias_reshaped],
            output: vec![y_output],
            name: Some(format!("{base}_bias_add")),
            ..Default::default()
        });
    } else {
        // No bias: the rescale produces Y directly.
        nodes.push(NodeProto {
            op_type: Some("Mul".into()),
            input: vec![ci_f32, scale_reshaped],
            output: vec![y_output],
            name: Some(format!("{base}_rescale")),
            ..Default::default()
        });
    }

    nodes
}

/// `[1, C_out, 1, ...]` broadcast shape for a conv output of the given rank.
fn channel_broadcast_shape(c_out: i64, rank: usize) -> Vec<i64> {
    let mut shape = vec![1i64; rank.max(2)];
    shape[1] = c_out;
    shape
}

/// Build an INT64 1-D shape initializer (for `Reshape`'s second input).
fn int64_shape_initializer(name: &str, shape: &[i64]) -> TensorProto {
    let mut raw = Vec::with_capacity(shape.len() * 8);
    for &v in shape {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    TensorProto {
        name: Some(name.into()),
        dims: vec![shape.len() as i64],
        data_type: Some(INT64),
        raw_data: Some(raw),
        ..Default::default()
    }
}

/// A `Reshape` node `out = Reshape(data, shape_init)`.
fn reshape_node(data: &str, shape_init: &str, out: &str, name: &str) -> NodeProto {
    NodeProto {
        op_type: Some("Reshape".into()),
        input: vec![data.into(), shape_init.into()],
        output: vec![out.into()],
        name: Some(name.into()),
        ..Default::default()
    }
}

/// A `Cast` node to FLOAT.
fn cast_to_float(input: &str, output: &str, name: &str) -> NodeProto {
    NodeProto {
        op_type: Some("Cast".into()),
        input: vec![input.into()],
        output: vec![output.into()],
        name: Some(name.into()),
        attribute: vec![AttributeProto {
            name: Some("to".into()),
            i: Some(CAST_TO_FLOAT),
            r#type: Some(ATTR_INT),
            ..Default::default()
        }],
        ..Default::default()
    }
}

/// Copy the spatial attributes a `ConvInteger` understands from a `Conv`.
/// `ConvInteger` shares Conv's `strides`/`pads`/`dilations`/`group`/
/// `kernel_shape`/`auto_pad`; everything else (e.g. fused activations) is
/// dropped because there is no float weight to fuse against anymore.
fn copy_conv_attrs(node: &NodeProto) -> Vec<AttributeProto> {
    const CONV_ATTRS: &[&str] = &[
        "strides",
        "pads",
        "dilations",
        "group",
        "kernel_shape",
        "auto_pad",
    ];
    node.attribute
        .iter()
        .filter(|a| CONV_ATTRS.contains(&a.name()))
        .cloned()
        .collect()
}

/// Stable base name for the nodes/tensors we synthesize for a quantized op.
/// Falls back to the op's first output (always present and unique) when the
/// node itself is unnamed, so generated tensor names never collide.
fn node_base_name(node: &NodeProto) -> String {
    let raw = if !node.name().is_empty() {
        node.name().to_string()
    } else {
        node.output
            .first()
            .cloned()
            .unwrap_or_else(|| node.op_type().to_string())
    };
    sanitize(&raw)
}

/// Make a string safe as part of an ONNX tensor name (no `/` or `:`).
fn sanitize(s: &str) -> String {
    s.replace(['/', ':'], "_")
}

/// Extract float32 data from a TensorProto initializer.
fn extract_float_data(tensor: &TensorProto) -> Result<Vec<f32>> {
    if !tensor.float_data.is_empty() {
        return Ok(tensor.float_data.clone());
    }
    if let Some(raw) = tensor.raw_data.as_deref()
        && !raw.is_empty()
    {
        anyhow::ensure!(
            raw.len().is_multiple_of(4),
            "Tensor '{}' raw_data length {} is not aligned to 4 bytes",
            tensor.name(),
            raw.len()
        );
        let num_floats = raw.len() / 4;
        let mut data = Vec::with_capacity(num_floats);
        for chunk in raw.chunks_exact(4) {
            data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        return Ok(data);
    }
    anyhow::bail!("Tensor '{}' has no float data", tensor.name());
}

/// Per-output-channel axis for a quantizable weight, chosen from the consuming
/// op's semantics. The scale tensor carries one entry per index along this axis,
/// so it must line up with the operator's *output* channels to keep
/// quantization error low:
/// - `Conv` weight `[out_channels, in/groups, *kernel]` → axis 0.
/// - `Gemm` weight `[K, N]` (`transB=0`) or `[N, K]` (`transB=1`) → N's axis.
/// - `MatMul` (and the fallback) weight `[..., K, N]` → the last axis (N).
fn per_channel_axis(op_type: &str, node: &NodeProto, rank: usize) -> usize {
    let last = rank.saturating_sub(1);
    match op_type {
        "Conv" => 0,
        "Gemm" => {
            if attr_int(node, "transB").unwrap_or(0) != 0 {
                0
            } else {
                last.min(1)
            }
        }
        // MatMul and any other matmul-shaped op: output channel is the last dim.
        _ => last,
    }
}

/// Read an integer attribute by name from a node, if present.
fn attr_int(node: &NodeProto, name: &str) -> Option<i64> {
    node.attribute
        .iter()
        .find(|a| a.name() == name)
        .and_then(|a| a.i)
}

/// Symmetric per-channel INT8 quantization of `data` (row-major, shaped `dims`)
/// along `axis`. Returns the quantized values in the original element order plus
/// one scale per channel (`dims[axis]` entries). All-zero channels get scale 1.0
/// to avoid division by zero. Quantizing along an arbitrary axis requires a
/// strided gather, so this generalises the previous axis-0-only contiguous-block
/// path.
fn quantize_per_channel(data: &[f32], dims: &[i64], axis: usize) -> (Vec<i8>, Vec<f32>) {
    let channels = (dims[axis].max(0) as usize).max(1);
    // Number of contiguous elements between successive indices along `axis`.
    let stride: usize = dims[axis + 1..]
        .iter()
        .map(|&d| d.max(0) as usize)
        .product::<usize>()
        .max(1);

    let mut abs_max = vec![0.0f32; channels];
    for (f, &v) in data.iter().enumerate() {
        let ch = (f / stride) % channels;
        abs_max[ch] = abs_max[ch].max(v.abs());
    }
    let scales: Vec<f32> = abs_max
        .iter()
        .map(|&m| if m == 0.0 { 1.0 } else { m / 127.0 })
        .collect();

    let mut quantized = vec![0i8; data.len()];
    for (f, &v) in data.iter().enumerate() {
        let ch = (f / stride) % channels;
        quantized[f] = (v / scales[ch]).round().clamp(-128.0, 127.0) as i8;
    }
    (quantized, scales)
}

#[cfg(test)]
mod tests;
