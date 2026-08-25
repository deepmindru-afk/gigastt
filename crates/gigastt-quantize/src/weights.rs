//! Weight extraction and per-channel INT8 quantization.

use anyhow::Result;

use crate::onnx_proto::{NodeProto, TensorProto};

pub(crate) struct QuantizedWeight {
    /// `{weight}_quantized` — INT8 initializer name.
    pub(crate) q_name: String,
    /// `{weight}_scale` — FLOAT [N] initializer name.
    pub(crate) s_name: String,
    /// `{weight}_zero_point` — INT8 [N] zeros initializer name.
    pub(crate) zp_name: String,
}

/// Extract float32 data from a TensorProto initializer.
pub(crate) fn extract_float_data(tensor: &TensorProto) -> Result<Vec<f32>> {
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
        for chunk in raw.as_chunks::<4>().0 {
            data.push(f32::from_le_bytes(*chunk));
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
pub(crate) fn per_channel_axis(op_type: &str, node: &NodeProto, rank: usize) -> usize {
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
pub(crate) fn quantize_per_channel(data: &[f32], dims: &[i64], axis: usize) -> (Vec<i8>, Vec<f32>) {
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
