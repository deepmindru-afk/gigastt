//! ONNX graph rewrite: opset bump and integer MatMul/Conv chains.

use crate::onnx_proto::{AttributeProto, ModelProto, NodeProto, TensorProto};
use crate::weights::QuantizedWeight;
use crate::{ATTR_INT, CAST_TO_FLOAT, INT64, MIN_OPSET};

/// Ensure the model imports opset (domain "") ≥ [`MIN_OPSET`]; bump it if lower
/// and add the default operator set if the model declares none.
pub(crate) fn bump_opset(model: &mut ModelProto) {
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
pub(crate) fn build_matmul_chain(node: &NodeProto, qw: &QuantizedWeight) -> Vec<NodeProto> {
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
pub(crate) fn build_conv_chain(
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
