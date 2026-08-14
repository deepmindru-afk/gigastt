//! Rotary Position Embeddings (RoPE) for the Candle Conformer encoder.

use candle_core::{Device, Result, Tensor};

// -----------------------------------------------------------------------
// Rotary Position Embedding (RoPE)
// -----------------------------------------------------------------------

/// Создать таблицу cos/sin для RoPE.
///
/// Возвращает два тензора (cos, sin) формы (max_len, 1, 1, dim).
pub(super) fn create_rope_table(
    dim: usize,
    max_len: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    // GigaAM v3 uses a RoPE base of 5000 (not the common 10000); verified by
    // matching the encoder's baked cos/sin table to 5.96e-6 over the first 100
    // positions. Using 10000 silently corrupts attention from the first layer.
    let base = 5_000f32;
    // inv_freq = 1 / (base^(2i/dim)) для i = 0, 2, 4, ..., dim-2
    let half_dim = dim / 2;
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / base.powf(2.0 * i as f32 / dim as f32))
        .collect();

    let inv_freq_t = Tensor::from_vec(inv_freq, half_dim, device)?;
    let positions: Vec<f32> = (0..max_len).map(|i| i as f32).collect();
    let positions_t = Tensor::from_vec(positions, max_len, device)?;

    // freqs = outer(positions, inv_freq) → (max_len, half_dim)
    let freqs = positions_t
        .unsqueeze(1)?
        .matmul(&inv_freq_t.unsqueeze(0)?)?;

    // emb = cat(freqs, freqs, dim=-1) → (max_len, dim)
    let emb = Tensor::cat(&[&freqs, &freqs], 1)?;

    let cos = emb.cos()?;
    let sin = emb.sin()?;

    // Формы: (max_len, 1, 1, dim) для broadcasting с (seq, batch, heads, d_k)
    let cos = cos.unsqueeze(1)?.unsqueeze(1)?;
    let sin = sin.unsqueeze(1)?.unsqueeze(1)?;

    Ok((cos, sin))
}

/// Применить RoPE к Q и K.
///
/// q, k: (seq_len, batch, n_heads, d_k)
/// cos, sin: (seq_len, 1, 1, d_k) — обрезанные до seq_len
pub(super) fn apply_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let q_rot = rotate_half(q)?;
    let k_rot = rotate_half(k)?;

    let q_embed = q.broadcast_mul(cos)?.add(&q_rot.broadcast_mul(sin)?)?;
    let k_embed = k.broadcast_mul(cos)?.add(&k_rot.broadcast_mul(sin)?)?;

    Ok((q_embed, k_embed))
}

/// Разделить последнее измерение пополам и повернуть: (-x2, x1).
fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let d = x.dim(candle_core::D::Minus1)?;
    let half = d / 2;
    let x1 = x.narrow(candle_core::D::Minus1, 0, half)?;
    let x2 = x.narrow(candle_core::D::Minus1, half, half)?;
    let neg_x2 = x2.neg()?;
    Tensor::cat(&[&neg_x2, &x1], candle_core::D::Minus1)
}
