//! Subsampling, rotary MHSA, feed-forward, and convolution blocks.

use candle_core::{Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, LayerNorm, Linear, Module, VarBuilder};

use super::rope::apply_rotary_pos_emb;

// -----------------------------------------------------------------------
// Strided Subsampling (conv1d)
// -----------------------------------------------------------------------

/// Страйденная субдискретизация через свёрточные слои.
///
/// Уменьшает длину последовательности в `subsampling_factor` раз.
/// Для factor=4 используется 2 слоя Conv1d с stride=2.
pub struct StridingSubsampling {
    /// Свёрточные слои (без ReLU — он применяется в forward).
    convs: Vec<Conv1d>,
    /// Фактор субдискретизации.
    factor: usize,
}

impl StridingSubsampling {
    pub fn load(
        feat_in: usize,
        d_model: usize,
        kernel_size: usize,
        factor: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let n_layers = (factor as f64).log2() as usize;
        let padding = (kernel_size - 1) / 2;
        let cfg = Conv1dConfig {
            padding,
            stride: 2,
            dilation: 1,
            groups: 1,
            ..Default::default()
        };

        let mut convs = Vec::with_capacity(n_layers);
        let mut in_ch = feat_in;

        for i in 0..n_layers {
            // Ключи: conv.0, conv.2 (рядом с ReLU на позициях 1, 3)
            let layer_idx = i * 2;
            let conv = candle_nn::conv1d(
                in_ch,
                d_model,
                kernel_size,
                cfg,
                vb.pp(format!("conv.{layer_idx}")),
            )?;
            convs.push(conv);
            in_ch = d_model;
        }

        Ok(Self { convs, factor })
    }

    /// Прямой проход: (batch, feat_in, seq_len) → (batch, seq_len/factor, d_model).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Вход: (batch, seq_len, feat_in) — транспонируем в (batch, feat_in, seq_len)
        let mut h = x.transpose(1, 2)?;

        for conv in &self.convs {
            h = conv.forward(&h)?;
            h = h.relu()?;
        }

        // Выход: (batch, d_model, seq_len/factor) → (batch, seq_len/factor, d_model)
        h.transpose(1, 2)
    }

    /// Вычислить длину после субдискретизации.
    pub fn output_length(&self, input_length: usize) -> usize {
        let mut length = input_length;
        let n_layers = (self.factor as f64).log2() as usize;
        for _ in 0..n_layers {
            length = length.div_ceil(2);
        }
        length
    }
}

// -----------------------------------------------------------------------
// Multi-Head Attention с Rotary Position Embeddings
// -----------------------------------------------------------------------

/// Multi-Head Self-Attention с RoPE (Rotary Position Embeddings).
pub struct RotaryMHSA {
    linear_q: Linear,
    linear_k: Linear,
    linear_v: Linear,
    linear_out: Linear,
    n_heads: usize,
    d_k: usize,
}

impl RotaryMHSA {
    pub fn load(d_model: usize, n_heads: usize, vb: VarBuilder) -> Result<Self> {
        let d_k = d_model / n_heads;
        let linear_q = candle_nn::linear(d_model, d_model, vb.pp("linear_q"))?;
        let linear_k = candle_nn::linear(d_model, d_model, vb.pp("linear_k"))?;
        let linear_v = candle_nn::linear(d_model, d_model, vb.pp("linear_v"))?;
        let linear_out = candle_nn::linear(d_model, d_model, vb.pp("linear_out"))?;

        Ok(Self {
            linear_q,
            linear_k,
            linear_v,
            linear_out,
            n_heads,
            d_k,
        })
    }

    /// Прямой проход MHSA с RoPE.
    ///
    /// # Аргументы
    /// * `x` — (batch, seq, d_model) — входной тензор (query=key=value=x)
    /// * `cos_emb`, `sin_emb` — RoPE таблицы, обрезанные до seq_len
    ///   формы (seq_len, 1, 1, d_k)
    /// * `att_mask` — маска внимания (batch, seq, seq) или None
    pub fn forward(
        &self,
        x: &Tensor,
        cos_emb: &Tensor,
        sin_emb: &Tensor,
        att_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, t, _d) = x.dims3()?;

        // 1. Применить RoPE к сырому входу (до проекции).
        //    GigaAM применяет RoPE ДО линейных проекций Q/K.
        //    x: (batch, seq, d_model) → reshape → (seq, batch, heads, d_k)
        let x_rope = x
            .transpose(0, 1)? // (seq, batch, d_model)
            .reshape((t, b, self.n_heads, self.d_k))?; // (seq, batch, heads, d_k)

        let (q_rope, k_rope) = apply_rotary_pos_emb(&x_rope, &x_rope, cos_emb, sin_emb)?;

        // Обратно: (seq, batch, heads, d_k) → (batch, seq, d_model)
        let q_in = q_rope
            .reshape((t, b, self.n_heads * self.d_k))?
            .transpose(0, 1)?; // (batch, seq, d_model)
        let k_in = k_rope
            .reshape((t, b, self.n_heads * self.d_k))?
            .transpose(0, 1)?; // (batch, seq, d_model)
        let v_in = x_rope
            .reshape((t, b, self.n_heads * self.d_k))?
            .transpose(0, 1)?; // (batch, seq, d_model)

        // 2. Проекция через линейные слои.
        let q = self
            .linear_q
            .forward(&q_in)? // (batch, seq, d_model)
            .reshape((b, t, self.n_heads, self.d_k))?
            .transpose(1, 2)? // (batch, heads, seq, d_k)
            .contiguous()?;
        let k = self
            .linear_k
            .forward(&k_in)?
            .reshape((b, t, self.n_heads, self.d_k))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .linear_v
            .forward(&v_in)?
            .reshape((b, t, self.n_heads, self.d_k))?
            .transpose(1, 2)?
            .contiguous()?;

        // 3. Scaled dot-product attention.
        let scale = (self.d_k as f64).sqrt();
        let mut scores = q.matmul(&k.transpose(2, 3)?)?;
        scores = (scores / scale)?;

        // Применить маску (если есть).
        if let Some(mask) = att_mask {
            // mask: (batch, seq, seq) → (batch, 1, seq, seq)
            let mask = mask.unsqueeze(1)?;
            // Маскированные позиции заполняем -10000.
            let fill_val =
                Tensor::new(-10_000f32, scores.device())?.broadcast_as(scores.shape())?;
            scores = mask.where_cond(&fill_val, &scores)?;
        }

        let attn = candle_nn::ops::softmax_last_dim(&scores)?;

        if let Some(mask) = att_mask {
            let mask = mask.unsqueeze(1)?;
            let zeros = Tensor::zeros_like(&attn)?;
            let attn = mask.where_cond(&zeros, &attn)?;

            // 4. Weighted sum и выходная проекция.
            let context = attn.matmul(&v)?; // (batch, heads, seq, d_k)
            let context = context
                .transpose(1, 2)? // (batch, seq, heads, d_k)
                .reshape((b, t, self.n_heads * self.d_k))?;
            return self.linear_out.forward(&context);
        }

        // 4. Weighted sum и выходная проекция (без маски).
        let context = attn.matmul(&v)?;
        let context = context
            .transpose(1, 2)?
            .reshape((b, t, self.n_heads * self.d_k))?;
        self.linear_out.forward(&context)
    }
}

// -----------------------------------------------------------------------
// Conformer Feed-Forward Module
// -----------------------------------------------------------------------

/// Conformer Feed-Forward: Linear → SiLU → Linear.
pub struct ConformerFeedForward {
    linear1: Linear,
    linear2: Linear,
}

impl ConformerFeedForward {
    pub fn load(d_model: usize, d_ff: usize, vb: VarBuilder) -> Result<Self> {
        let linear1 = candle_nn::linear(d_model, d_ff, vb.pp("linear1"))?;
        let linear2 = candle_nn::linear(d_ff, d_model, vb.pp("linear2"))?;
        Ok(Self { linear1, linear2 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.linear1.forward(x)?;
        let h = candle_nn::Activation::Silu.forward(&h)?;
        self.linear2.forward(&h)
    }
}

// -----------------------------------------------------------------------
// Conformer Convolution Module
// -----------------------------------------------------------------------

/// Conformer Convolution:
/// Pointwise Conv1d → GLU → Depthwise Conv1d → LayerNorm → SiLU → Pointwise Conv1d
pub struct ConformerConvolution {
    pointwise_conv1: Conv1d,
    depthwise_conv: Conv1d,
    norm: LayerNorm,
    pointwise_conv2: Conv1d,
    d_model: usize,
}

impl ConformerConvolution {
    pub fn load(d_model: usize, kernel_size: usize, vb: VarBuilder) -> Result<Self> {
        let padding = (kernel_size - 1) / 2;

        // Pointwise conv1: (d_model → 2*d_model, kernel=1) для GLU
        let pw1_cfg = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
            ..Default::default()
        };
        let pointwise_conv1 =
            candle_nn::conv1d(d_model, d_model * 2, 1, pw1_cfg, vb.pp("pointwise_conv1"))?;

        // Depthwise conv: (d_model → d_model, kernel=kernel_size, groups=d_model)
        let dw_cfg = Conv1dConfig {
            padding,
            stride: 1,
            dilation: 1,
            groups: d_model,
            ..Default::default()
        };
        let depthwise_conv = candle_nn::conv1d(
            d_model,
            d_model,
            kernel_size,
            dw_cfg,
            vb.pp("depthwise_conv"),
        )?;

        // LayerNorm (ключ "batch_norm" для совместимости с PyTorch)
        let norm = candle_nn::layer_norm(d_model, 1e-5, vb.pp("batch_norm"))?;

        // Pointwise conv2: (d_model → d_model, kernel=1)
        let pw2_cfg = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
            ..Default::default()
        };
        let pointwise_conv2 =
            candle_nn::conv1d(d_model, d_model, 1, pw2_cfg, vb.pp("pointwise_conv2"))?;

        Ok(Self {
            pointwise_conv1,
            depthwise_conv,
            norm,
            pointwise_conv2,
            d_model,
        })
    }

    /// Прямой проход: x (batch, seq, d_model) → (batch, seq, d_model).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: (batch, seq, d_model)

        // Транспонируем для Conv1d: (batch, d_model, seq)
        let h = x.transpose(1, 2)?;

        // Pointwise conv1: (batch, 2*d_model, seq)
        let h = self.pointwise_conv1.forward(&h)?;

        // GLU: разделить по каналам, sigmoid(вторая половина) * первая половина
        let h1 = h.narrow(1, 0, self.d_model)?;
        let h2 = h.narrow(1, self.d_model, self.d_model)?;
        let h = (h1 * candle_nn::ops::sigmoid(&h2)?)?;

        // Depthwise conv: (batch, d_model, seq)
        let h = self.depthwise_conv.forward(&h)?;

        // LayerNorm: нужно транспонировать в (batch, seq, d_model)
        let h = h.transpose(1, 2)?; // (batch, seq, d_model)
        let h = self.norm.forward(&h)?;
        let h = h.transpose(1, 2)?; // обратно (batch, d_model, seq)

        // SiLU
        let h = candle_nn::Activation::Silu.forward(&h)?;

        // Pointwise conv2
        let h = self.pointwise_conv2.forward(&h)?;

        // Обратно в (batch, seq, d_model)
        h.transpose(1, 2)
    }
}
