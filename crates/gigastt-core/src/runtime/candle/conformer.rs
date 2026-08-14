// Vendored from askidmobile/RustASR (crates/model-gigaam, commit 33060b8d),
// dual-licensed MIT OR Apache-2.0. Copyright (c) the RustASR authors.
// Adapted for gigastt: import paths changed; compiled against upstream candle 0.9;
// CTC head not used here (only the v3 Conformer encoder).
#![allow(dead_code)] // wired into a RuntimeSession in a later task

//! Conformer-энкодер для GigaAM.
//!
//! Реализация архитектуры Conformer (Gulati et al., 2020)
//! с Rotary Position Embeddings (RoPE), Macaron-style FFN,
//! depthwise separable convolution, и стриженной субдискретизацией.
//!
//! Совместимость весов: ключи тензоров совпадают с PyTorch-реализацией
//! GigaAM (salute-developers/GigaAM), что позволяет загружать
//! сконвертированные safetensors напрямую через VarBuilder.

use candle_core::{Result, Tensor};
use candle_nn::{LayerNorm, Module, VarBuilder};

use super::config::EncoderConfig;

mod blocks;
mod rope;

use blocks::{ConformerConvolution, ConformerFeedForward, RotaryMHSA, StridingSubsampling};
use rope::create_rope_table;

// -----------------------------------------------------------------------
// Conformer Layer (Macaron-style)
// -----------------------------------------------------------------------

/// Один слой Conformer (Macaron-style):
/// FFN1 → Self-Attention → Convolution → FFN2 → LayerNorm
pub struct ConformerLayer {
    norm_feed_forward1: LayerNorm,
    feed_forward1: ConformerFeedForward,
    norm_self_att: LayerNorm,
    self_attn: RotaryMHSA,
    norm_conv: LayerNorm,
    conv: ConformerConvolution,
    norm_feed_forward2: LayerNorm,
    feed_forward2: ConformerFeedForward,
    norm_out: LayerNorm,
}

impl ConformerLayer {
    pub fn load(
        d_model: usize,
        d_ff: usize,
        n_heads: usize,
        conv_kernel_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let norm_feed_forward1 = candle_nn::layer_norm(d_model, 1e-5, vb.pp("norm_feed_forward1"))?;
        let feed_forward1 = ConformerFeedForward::load(d_model, d_ff, vb.pp("feed_forward1"))?;
        let norm_self_att = candle_nn::layer_norm(d_model, 1e-5, vb.pp("norm_self_att"))?;
        let self_attn = RotaryMHSA::load(d_model, n_heads, vb.pp("self_attn"))?;
        let norm_conv = candle_nn::layer_norm(d_model, 1e-5, vb.pp("norm_conv"))?;
        let conv = ConformerConvolution::load(d_model, conv_kernel_size, vb.pp("conv"))?;
        let norm_feed_forward2 = candle_nn::layer_norm(d_model, 1e-5, vb.pp("norm_feed_forward2"))?;
        let feed_forward2 = ConformerFeedForward::load(d_model, d_ff, vb.pp("feed_forward2"))?;
        let norm_out = candle_nn::layer_norm(d_model, 1e-5, vb.pp("norm_out"))?;

        Ok(Self {
            norm_feed_forward1,
            feed_forward1,
            norm_self_att,
            self_attn,
            norm_conv,
            conv,
            norm_feed_forward2,
            feed_forward2,
            norm_out,
        })
    }

    /// Прямой проход одного слоя Conformer.
    ///
    /// x: (batch, seq, d_model)
    /// cos_emb, sin_emb: RoPE таблицы (seq, 1, 1, d_k)
    /// att_mask: (batch, seq, seq) или None
    pub fn forward(
        &self,
        x: &Tensor,
        cos_emb: &Tensor,
        sin_emb: &Tensor,
        att_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        const FC_FACTOR: f64 = 0.5;

        // 1. FFN1 (с фактором 0.5)
        let h = self.norm_feed_forward1.forward(x)?;
        let h = self.feed_forward1.forward(&h)?;
        let residual = (x + (h * FC_FACTOR)?)?;

        // 2. Self-Attention
        let h = self.norm_self_att.forward(&residual)?;
        let h = self.self_attn.forward(&h, cos_emb, sin_emb, att_mask)?;
        let residual = (residual + h)?;

        // 3. Convolution
        let h = self.norm_conv.forward(&residual)?;
        let h = self.conv.forward(&h)?;
        let residual = (residual + h)?;

        // 4. FFN2 (с фактором 0.5)
        let h = self.norm_feed_forward2.forward(&residual)?;
        let h = self.feed_forward2.forward(&h)?;
        let residual = (residual + (h * FC_FACTOR)?)?;

        // 5. Финальная нормализация
        self.norm_out.forward(&residual)
    }
}

// -----------------------------------------------------------------------
// Conformer Encoder
// -----------------------------------------------------------------------

/// Полный Conformer-энкодер GigaAM:
/// Subsampling → Positional Encoding → N × ConformerLayer
pub struct ConformerEncoder {
    pre_encode: StridingSubsampling,
    layers: Vec<ConformerLayer>,
    /// Предвычисленная таблица cos для RoPE.
    rope_cos: Tensor,
    /// Предвычисленная таблица sin для RoPE.
    rope_sin: Tensor,
    /// Размерность одной головы (для пересоздания RoPE).
    d_k: usize,
    /// Количество входных mel-бинов.
    #[allow(dead_code)]
    feat_in: usize,
}

impl ConformerEncoder {
    pub fn load(config: &EncoderConfig, vb: VarBuilder) -> Result<Self> {
        let d_k = config.d_model / config.n_heads;
        let d_ff = config.d_model * config.ff_expansion_factor;

        // Субдискретизация
        let pre_encode = StridingSubsampling::load(
            config.feat_in,
            config.d_model,
            config.subs_kernel_size,
            config.subsampling_factor,
            vb.pp("pre_encode"),
        )?;

        // Создать таблицу RoPE
        let (rope_cos, rope_sin) = create_rope_table(d_k, config.pos_emb_max_len, vb.device())?;
        // Привести к тому же dtype, что и модель
        let rope_cos = rope_cos.to_dtype(vb.dtype())?;
        let rope_sin = rope_sin.to_dtype(vb.dtype())?;

        // Слои Conformer
        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let layer = ConformerLayer::load(
                config.d_model,
                d_ff,
                config.n_heads,
                config.conv_kernel_size,
                vb.pp(format!("layers.{i}")),
            )?;
            layers.push(layer);
        }

        Ok(Self {
            pre_encode,
            layers,
            rope_cos,
            rope_sin,
            d_k,
            feat_in: config.feat_in,
        })
    }

    /// Прямой проход энкодера.
    ///
    /// # Аргументы
    /// * `features` — mel-спектрограмма (batch, feat_in, seq_len)
    ///
    /// # Возвращает
    /// Тензор (batch, d_model, encoded_len) — закодированные фичи.
    pub fn forward(&self, features: &Tensor) -> Result<Tensor> {
        // 1. Субдискретизация: (batch, feat_in, seq) → (batch, seq/4, d_model)
        // Входной features в формате (batch, feat_in, seq_len)
        // Субдискретизация ожидает (batch, seq, feat_in)
        let x = features.transpose(1, 2)?;
        let x = self.pre_encode.forward(&x)?;

        let (_b, t, _d) = x.dims3()?;

        // 2. RoPE — обрезать cos/sin до текущей длины.
        //    Если последовательность длиннее предвычисленной таблицы,
        //    пересоздаём таблицу на лету.
        let (cos_emb, sin_emb) = if t <= self.rope_cos.dim(0)? {
            (
                self.rope_cos.narrow(0, 0, t)?,
                self.rope_sin.narrow(0, 0, t)?,
            )
        } else {
            tracing::warn!(
                "GigaAM: RoPE таблица расширена с {} до {} позиций",
                self.rope_cos.dim(0)?,
                t,
            );
            let (cos, sin) = create_rope_table(self.d_k, t, x.device())?;
            (cos.to_dtype(x.dtype())?, sin.to_dtype(x.dtype())?)
        };

        // 3. Прогоняем через все слои Conformer.
        // Маску не используем (batch_size=1 при инференсе).
        //
        // Metal workaround: каждые SYNC_EVERY слоёв вставляем device.synchronize()
        // для сброса Metal command buffer pool. Это предотвращает накопление
        // слишком большого количества буферов в in-flight состоянии, что может
        // вызвать краш AGXMetalG16X::fillBuffer на M4 / macOS 26.x.
        const SYNC_EVERY: usize = 4;
        let is_metal = x.device().is_metal();

        let mut h = x;
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &cos_emb, &sin_emb, None)?;

            if is_metal && (i + 1) % SYNC_EVERY == 0 {
                h.device().synchronize().map_err(|e| {
                    candle_core::Error::Msg(format!("Metal sync at layer {}: {e}", i + 1))
                })?;
            }
        }

        // 4. Выход: (batch, seq/4, d_model) → (batch, d_model, seq/4)
        h.transpose(1, 2)
    }
}
