//! tokenizer.json parsing for the supported WordPiece scheme.

use std::collections::HashMap;

use anyhow::{Context, Result, anyhow, bail};
use serde::Deserialize;

use super::Normalizer;

// ---------------------------------------------------------------------------
// tokenizer.json parsing (only the fields the supported scheme uses)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub(super) struct TokenizerFile {
    #[serde(default)]
    pub(super) version: Option<String>,
    #[serde(default)]
    pub(super) truncation: Option<serde_json::Value>,
    #[serde(default)]
    pub(super) padding: Option<serde_json::Value>,
    #[serde(default)]
    pub(super) added_tokens: Vec<AddedTokenFile>,
    #[serde(default)]
    pub(super) normalizer: Option<serde_json::Value>,
    #[serde(default)]
    pub(super) pre_tokenizer: Option<serde_json::Value>,
    #[serde(default)]
    pub(super) post_processor: Option<serde_json::Value>,
    pub(super) model: ModelFile,
}

#[derive(Deserialize)]
pub(super) struct AddedTokenFile {
    pub(super) id: u32,
    pub(super) content: String,
    #[serde(default)]
    pub(super) single_word: bool,
    #[serde(default)]
    pub(super) lstrip: bool,
    #[serde(default)]
    pub(super) rstrip: bool,
    #[serde(default = "default_true")]
    pub(super) normalized: bool,
    // Parsed for completeness; special-ness does not change encoding here.
    #[serde(default)]
    #[allow(dead_code)]
    special: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Deserialize)]
pub(super) struct ModelFile {
    #[serde(rename = "type")]
    pub(super) kind: String,
    #[serde(default)]
    pub(super) unk_token: Option<String>,
    #[serde(default)]
    pub(super) continuing_subword_prefix: Option<String>,
    #[serde(default)]
    pub(super) max_input_chars_per_word: Option<usize>,
    pub(super) vocab: HashMap<String, u32>,
}

pub(super) fn parse_normalizer(value: Option<serde_json::Value>) -> Result<Normalizer> {
    let Some(value) = value else {
        return Ok(Normalizer::None);
    };
    #[derive(Deserialize)]
    struct BertNormalizerFile {
        #[serde(rename = "type")]
        kind: String,
        #[serde(default)]
        clean_text: Option<bool>,
        #[serde(default)]
        handle_chinese_chars: Option<bool>,
        #[serde(default)]
        strip_accents: Option<bool>,
        #[serde(default)]
        lowercase: Option<bool>,
    }
    let parsed: BertNormalizerFile =
        serde_json::from_value(value).context("failed to parse normalizer")?;
    if parsed.kind != "BertNormalizer" {
        bail!(
            "unsupported normalizer type '{}' (expected 'BertNormalizer' or null)",
            parsed.kind
        );
    }
    // Upstream defaults (`BertNormalizer::default`): clean_text and
    // handle_chinese_chars on, lowercase on (which also defaults
    // strip_accents). Anything requiring lowercase / accent stripping is
    // rejected rather than silently ignored.
    let lowercase = parsed.lowercase.unwrap_or(true);
    let strip_accents = parsed.strip_accents.unwrap_or(lowercase);
    if lowercase {
        bail!("BertNormalizer with lowercase=true is not supported");
    }
    if strip_accents {
        bail!("BertNormalizer with strip_accents=true is not supported");
    }
    Ok(Normalizer::Bert {
        clean_text: parsed.clean_text.unwrap_or(true),
        handle_chinese_chars: parsed.handle_chinese_chars.unwrap_or(true),
    })
}

/// Returns whether the (supported) `BertPreTokenizer` is active.
pub(super) fn parse_pre_tokenizer(value: Option<serde_json::Value>) -> Result<bool> {
    let Some(value) = value else {
        return Ok(false);
    };
    let kind = value
        .get("type")
        .and_then(|t| t.as_str())
        .context("pre_tokenizer is missing a 'type' field")?;
    if kind != "BertPreTokenizer" {
        bail!("unsupported pre_tokenizer type '{kind}' (expected 'BertPreTokenizer' or null)");
    }
    Ok(true)
}

/// Parse the `TemplateProcessing` post-processor. Only the exact
/// `[SpecialToken, Sequence A, SpecialToken]` single template (the BERT
/// `[CLS] A [SEP]` shape) is supported; pair templates are ignored because
/// this path never encodes pairs.
pub(super) fn parse_post_processor(
    value: Option<serde_json::Value>,
) -> Result<(Option<u32>, Option<u32>)> {
    let Some(value) = value else {
        return Ok((None, None));
    };

    #[derive(Deserialize)]
    struct TemplateProcessingFile {
        #[serde(rename = "type")]
        kind: String,
        single: Vec<TemplatePiece>,
        special_tokens: HashMap<String, SpecialTokenEntry>,
    }
    #[derive(Deserialize)]
    enum TemplatePiece {
        SpecialToken { id: String, type_id: u32 },
        Sequence { id: String, type_id: u32 },
    }
    #[derive(Deserialize)]
    struct SpecialTokenEntry {
        ids: Vec<u32>,
    }

    let parsed: TemplateProcessingFile =
        serde_json::from_value(value).context("failed to parse post_processor")?;
    if parsed.kind != "TemplateProcessing" {
        bail!(
            "unsupported post_processor type '{}' (expected 'TemplateProcessing' or null)",
            parsed.kind
        );
    }
    let [first, TemplatePiece::Sequence { id, type_id }, last] = parsed.single.as_slice() else {
        bail!("unsupported post_processor template (expected '[Special] A [Special]')");
    };
    if id != "A" || *type_id != 0 {
        bail!("unsupported post_processor template (expected sequence 'A' with type_id 0)");
    }
    let special_id = |piece: &TemplatePiece| -> Result<u32> {
        let TemplatePiece::SpecialToken { id, type_id } = piece else {
            bail!("unsupported post_processor template (expected '[Special] A [Special]')");
        };
        if *type_id != 0 {
            bail!("unsupported special token type_id {type_id} (expected 0)");
        }
        let entry = parsed
            .special_tokens
            .get(id)
            .ok_or_else(|| anyhow!("post_processor references unknown special token '{id}'"))?;
        let [token_id] = entry.ids.as_slice() else {
            bail!("special token '{id}' expands to more than one id, which is not supported");
        };
        Ok(*token_id)
    };
    Ok((Some(special_id(first)?), Some(special_id(last)?)))
}
