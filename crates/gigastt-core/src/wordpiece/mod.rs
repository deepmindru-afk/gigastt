//! In-tree WordPiece tokenizer for the RUPunct punctuation model.
//!
//! Replaces the `tokenizers` crate (0.23) on the punctuation path. The scope is
//! exactly the scheme shipped in the punct model's `tokenizer.json`:
//! `BertNormalizer { clean_text, handle_chinese_chars, strip_accents: false,
//! lowercase: false }` + `BertPreTokenizer` + greedy WordPiece (`##`
//! continuation, `[UNK]` fallback, 100-char word cap) + the `[CLS] A [SEP]`
//! template. Anything outside this scheme (lowercase / accent-stripping
//! normalizers, other pre-tokenizers or models, truncation / padding,
//! normalized or single-word added tokens) fails loudly at load time instead
//! of being silently mis-implemented.
//!
//! Byte-for-byte parity with the upstream `tokenizers` crate is enforced by
//! the golden fixtures in `tests/fixtures/wordpiece_golden.json` — produced by
//! the real crate over a corpus of Russian phrases and Unicode edges — which
//! the unit tests replay against the reduced vocab in
//! `tests/fixtures/wordpiece_tokenizer.json`.
//!
//! The `CONTROL_FORMAT_RANGES` / `PUNCTUATION_RANGES` tables are a compact
//! range encoding of the category tables in the `unicode_categories` 0.1.1
//! crate — the exact data source the upstream `tokenizers` 0.23 normalizer and
//! pre-tokenizer consult — so `is_other` / `is_punctuation` decisions match
//! the upstream crate char for char.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use serde::Deserialize;

/// WordPiece tokenizer loaded from a HuggingFace `tokenizer.json`.
///
/// Exposes only the surface the punctuation backends use: [`Tokenizer::from_file`],
/// [`Tokenizer::encode`], and the [`Encoding`] getters.
pub(crate) struct Tokenizer {
    vocab: HashMap<String, u32>,
    unk_id: u32,
    continuing_subword_prefix: String,
    max_input_chars_per_word: usize,
    normalizer: Normalizer,
    /// Whether the `BertPreTokenizer` split step is active (a null
    /// `pre_tokenizer` sends the whole normalized span to WordPiece as one
    /// word, as upstream does).
    bert_pre_tokenizer: bool,
    /// Added tokens extracted from the input before normalization, matched
    /// leftmost-longest (all `special: true, normalized: false` in the punct
    /// model's tokenizer.json).
    added_tokens: Vec<AddedToken>,
    /// `[CLS]` / `[SEP]` from the `TemplateProcessing` post-processor; `None`
    /// when the file declares no post-processor.
    cls: Option<u32>,
    sep: Option<u32>,
}

/// The result of [`Tokenizer::encode`]: token ids, an all-ones attention mask
/// (no padding is ever applied), and per-token word indices (`None` for the
/// `[CLS]` / `[SEP]` template tokens), matching the semantics of the upstream
/// `tokenizers::Encoding`.
pub(crate) struct Encoding {
    ids: Vec<u32>,
    attention_mask: Vec<u32>,
    word_ids: Vec<Option<u32>>,
}

impl Encoding {
    pub(crate) fn get_ids(&self) -> &[u32] {
        &self.ids
    }

    pub(crate) fn get_attention_mask(&self) -> &[u32] {
        &self.attention_mask
    }

    pub(crate) fn get_word_ids(&self) -> &[Option<u32>] {
        &self.word_ids
    }
}

/// One `added_tokens[]` entry: a literal string pulled out of the input before
/// normalization and emitted as a single token with the given id.
struct AddedToken {
    id: u32,
    content: String,
    lstrip: bool,
    rstrip: bool,
}

/// Normalizer configuration; only the modes the punct model uses are
/// supported (see module docs).
enum Normalizer {
    None,
    Bert {
        clean_text: bool,
        handle_chinese_chars: bool,
    },
}

impl Tokenizer {
    /// Load from a HuggingFace `tokenizer.json`. Fails loudly on any construct
    /// outside the supported scheme.
    pub(crate) fn from_file(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read tokenizer file {}", path.display()))?;
        let file: TokenizerFile = serde_json::from_str(&raw)
            .with_context(|| format!("failed to parse tokenizer file {}", path.display()))?;
        Self::from_parsed(file)
            .with_context(|| format!("unsupported tokenizer file {}", path.display()))
    }

    fn from_parsed(file: TokenizerFile) -> Result<Self> {
        if let Some(version) = &file.version
            && version != "1.0"
        {
            bail!("unknown tokenizer version '{version}'");
        }
        if file.truncation.is_some() {
            bail!("truncation is not supported (expected null)");
        }
        if file.padding.is_some() {
            bail!("padding is not supported (expected null)");
        }
        if file.model.kind != "WordPiece" {
            bail!(
                "unsupported model type '{}' (expected 'WordPiece')",
                file.model.kind
            );
        }
        if file.model.vocab.is_empty() {
            bail!("WordPiece vocabulary is empty");
        }

        let unk_token = file.model.unk_token.unwrap_or_else(|| "[UNK]".to_string());
        let unk_id =
            *file.model.vocab.get(&unk_token).ok_or_else(|| {
                anyhow!("WordPiece vocabulary is missing unk_token '{unk_token}'")
            })?;

        let normalizer = parse_normalizer(file.normalizer)?;
        let bert_pre_tokenizer = parse_pre_tokenizer(file.pre_tokenizer)?;
        let (cls, sep) = parse_post_processor(file.post_processor)?;

        let mut added_tokens = Vec::with_capacity(file.added_tokens.len());
        for token in &file.added_tokens {
            if token.content.is_empty() {
                continue;
            }
            if token.normalized {
                bail!(
                    "added token '{}' has normalized=true, which is not supported",
                    token.content
                );
            }
            if token.single_word {
                bail!(
                    "added token '{}' has single_word=true, which is not supported",
                    token.content
                );
            }
            // The upstream crate resolves the id by vocab lookup, falling back
            // to fresh ids; the JSON `id` field only feeds a mismatch warning.
            let id = file
                .model
                .vocab
                .get(&token.content)
                .copied()
                .unwrap_or(token.id);
            added_tokens.push(AddedToken {
                id,
                content: token.content.clone(),
                lstrip: token.lstrip,
                rstrip: token.rstrip,
            });
        }

        Ok(Self {
            vocab: file.model.vocab,
            unk_id,
            continuing_subword_prefix: file
                .model
                .continuing_subword_prefix
                .unwrap_or_else(|| "##".to_string()),
            max_input_chars_per_word: file.model.max_input_chars_per_word.unwrap_or(100),
            normalizer,
            bert_pre_tokenizer,
            added_tokens,
            cls,
            sep,
        })
    }

    /// Encode a single sequence. Mirrors `tokenizers::Tokenizer::encode(text,
    /// add_special_tokens)` for the supported scheme: added-token extraction,
    /// BertNormalizer, BertPreTokenizer, greedy WordPiece, `[CLS]`/`[SEP]`
    /// template. Infallible: every fallible condition is rejected at load.
    pub(crate) fn encode(&self, text: &str, add_special_tokens: bool) -> Encoding {
        let mut ids: Vec<u32> = Vec::new();
        let mut word_ids: Vec<Option<u32>> = Vec::new();

        if add_special_tokens && let Some(cls) = self.cls {
            ids.push(cls);
            word_ids.push(None);
        }

        // Word index per pre-tokenized split (added-token splits included),
        // matching `PreTokenizedString::into_encoding` upstream.
        let mut next_word: u32 = 0;
        for span in self.extract_added(text) {
            match span {
                Span::Added(idx) => {
                    ids.push(self.added_tokens[idx].id);
                    word_ids.push(Some(next_word));
                    next_word += 1;
                }
                Span::Text(slice) => {
                    let normalized = self.normalize(slice);
                    let words = if self.bert_pre_tokenizer {
                        bert_pre_tokenize(&normalized)
                    } else if normalized.is_empty() {
                        Vec::new()
                    } else {
                        vec![normalized.as_str()]
                    };
                    for word in words {
                        // tokenize_word always emits at least one token ([UNK]
                        // at worst), so the word index is always consumed.
                        self.tokenize_word(word, &mut ids);
                        word_ids.resize(ids.len(), Some(next_word));
                        next_word += 1;
                    }
                }
            }
        }

        if add_special_tokens && let Some(sep) = self.sep {
            ids.push(sep);
            word_ids.push(None);
        }

        let attention_mask = vec![1; ids.len()];
        Encoding {
            ids,
            attention_mask,
            word_ids,
        }
    }

    /// BertNormalizer: `clean_text` drops NUL / U+FFFD / control chars (any
    /// Unicode "Other" — Cc, Cf, Co — except \t \n \r) and maps all whitespace
    /// to ' '; `handle_chinese_chars` puts spaces around CJK ideographs.
    fn normalize(&self, text: &str) -> String {
        let (clean_text, handle_chinese_chars) = match &self.normalizer {
            Normalizer::None => return text.to_string(),
            Normalizer::Bert {
                clean_text,
                handle_chinese_chars,
            } => (*clean_text, *handle_chinese_chars),
        };

        let mut out = String::with_capacity(text.len());
        for c in text.chars() {
            if clean_text {
                if c == '\0' || c == '\u{FFFD}' || is_control(c) {
                    continue;
                }
                out.push(if c.is_whitespace() { ' ' } else { c });
            } else {
                out.push(c);
            }
        }
        if handle_chinese_chars {
            let mut spaced = String::with_capacity(out.len());
            for c in out.chars() {
                if is_chinese_char(c) {
                    spaced.push(' ');
                    spaced.push(c);
                    spaced.push(' ');
                } else {
                    spaced.push(c);
                }
            }
            out = spaced;
        }
        out
    }

    /// Greedy longest-match-first WordPiece over one pre-tokenized word,
    /// appending token ids to `out`. A word longer than
    /// `max_input_chars_per_word` chars, or one whose chars cannot all be
    /// covered by vocab entries, collapses to a single `[UNK]`.
    fn tokenize_word(&self, word: &str, out: &mut Vec<u32>) {
        if word.chars().count() > self.max_input_chars_per_word {
            out.push(self.unk_id);
            return;
        }

        let mut start = 0;
        let mut is_bad = false;
        // Checkpoint so a failed word rolls back only its own subtokens, not
        // tokens emitted for earlier words (upstream returns a per-word Vec).
        let checkpoint = out.len();
        while start < word.len() {
            let mut end = word.len();
            let mut found = None;
            while start < end {
                let substr = &word[start..end];
                let id = if start > 0 {
                    // Only continuation probes allocate; words are short
                    // (<= 100 chars) so this stays cheap.
                    let mut candidate =
                        String::with_capacity(self.continuing_subword_prefix.len() + substr.len());
                    candidate.push_str(&self.continuing_subword_prefix);
                    candidate.push_str(substr);
                    self.vocab.get(candidate.as_str()).copied()
                } else {
                    self.vocab.get(substr).copied()
                };
                if let Some(id) = id {
                    found = Some(id);
                    break;
                }
                end -= substr.chars().last().map_or(1, |c| c.len_utf8());
            }
            let Some(id) = found else {
                is_bad = true;
                break;
            };
            out.push(id);
            start = end;
        }

        if is_bad {
            // Any uncovered char dooms the whole word to a single [UNK].
            out.truncate(checkpoint);
            out.push(self.unk_id);
        }
    }

    /// Leftmost-longest scan for added tokens, splitting the input into text
    /// spans (to be normalized + tokenized) and added-token spans (emitted
    /// as-is). Mirrors `AddedVocabulary::find_matches` upstream.
    fn extract_added<'a>(&self, text: &'a str) -> Vec<Span<'a>> {
        let mut spans = Vec::new();
        if self.added_tokens.is_empty() {
            if !text.is_empty() {
                spans.push(Span::Text(text));
            }
            return spans;
        }

        let mut span_start = 0;
        let mut pos = 0;
        while pos < text.len() {
            // Longest match at this position (patterns are plain literals).
            let mut best: Option<usize> = None;
            for (i, token) in self.added_tokens.iter().enumerate() {
                if text[pos..].starts_with(token.content.as_str()) {
                    match best {
                        Some(b) if self.added_tokens[b].content.len() >= token.content.len() => {}
                        _ => best = Some(i),
                    }
                }
            }
            let Some(idx) = best else {
                pos += text[pos..].chars().next().map_or(1, |c| c.len_utf8());
                continue;
            };

            let token = &self.added_tokens[idx];
            let mut start = pos;
            let mut stop = pos + token.content.len();
            if token.lstrip {
                while start > span_start {
                    let prev = text[..start].chars().next_back();
                    match prev {
                        Some(c) if c.is_whitespace() => start -= c.len_utf8(),
                        _ => break,
                    }
                }
            }
            if token.rstrip {
                while let Some(c) = text[stop..].chars().next() {
                    if !c.is_whitespace() {
                        break;
                    }
                    stop += c.len_utf8();
                }
            }

            if start > span_start {
                spans.push(Span::Text(&text[span_start..start]));
            }
            spans.push(Span::Added(idx));
            pos = stop;
            span_start = stop;
        }
        if span_start < text.len() {
            spans.push(Span::Text(&text[span_start..]));
        }
        spans
    }
}

/// One piece of the input after added-token extraction.
enum Span<'a> {
    /// Raw text still to be normalized + tokenized.
    Text(&'a str),
    /// Index into `Tokenizer::added_tokens`.
    Added(usize),
}

/// BertPreTokenizer: split on whitespace (removed), then isolate every
/// punctuation char (ASCII punctuation or Unicode P* category) into its own
/// word. Borrows from `text`; never yields empty words.
fn bert_pre_tokenize(text: &str) -> Vec<&str> {
    let mut words = Vec::new();
    let mut word_start: Option<usize> = None;
    for (i, c) in text.char_indices() {
        if c.is_whitespace() {
            if let Some(s) = word_start.take() {
                words.push(&text[s..i]);
            }
        } else if is_bert_punc(c) {
            if let Some(s) = word_start.take() {
                words.push(&text[s..i]);
            }
            words.push(&text[i..i + c.len_utf8()]);
        } else if word_start.is_none() {
            word_start = Some(i);
        }
    }
    if let Some(s) = word_start {
        words.push(&text[s..]);
    }
    words
}

/// `char::is_ascii_punctuation` or Unicode punctuation category (P*), as in
/// the upstream `is_bert_punc`.
fn is_bert_punc(c: char) -> bool {
    c.is_ascii_punctuation() || in_ranges(c as u32, PUNCTUATION_RANGES)
}

/// Control per the upstream BertNormalizer: any Unicode "Other" (Cc, Cf, Co)
/// except \t \n \r, which count as whitespace instead.
fn is_control(c: char) -> bool {
    if matches!(c, '\t' | '\n' | '\r') {
        return false;
    }
    let cp = c as u32;
    // Category Co per unicode_categories 0.1.1 (hardcoded ranges there; the
    // OTHER_PRIVATE_USE table only repeats the range boundaries).
    if matches!(cp, 0xE000..=0xF8FF | 0xF0000..=0xFFFFD | 0x100000..=0x10FFFD) {
        return true;
    }
    in_ranges(cp, CONTROL_FORMAT_RANGES)
}

/// CJK Unified Ideographs ranges, verbatim from the upstream `is_chinese_char`.
fn is_chinese_char(c: char) -> bool {
    matches!(
        c as u32,
        0x4E00..=0x9FFF
            | 0x3400..=0x4DBF
            | 0x20000..=0x2A6DF
            | 0x2A700..=0x2B73F
            | 0x2B740..=0x2B81F
            | 0x2B920..=0x2CEAF
            | 0xF900..=0xFAFF
            | 0x2F800..=0x2FA1F
    )
}

/// Binary search over sorted, non-overlapping inclusive `(lo, hi)` ranges.
fn in_ranges(cp: u32, ranges: &[(u32, u32)]) -> bool {
    let i = ranges.partition_point(|&(lo, _)| lo <= cp);
    i > 0 && cp <= ranges[i - 1].1
}

/// Unicode categories Cc + Cf (control / format), from the OTHER_CONTROL and
/// OTHER_FORMAT tables of `unicode_categories` 0.1.1, merged into ranges.
const CONTROL_FORMAT_RANGES: &[(u32, u32)] = &[
    (0x0, 0x1F),
    (0x7F, 0x9F),
    (0xAD, 0xAD),
    (0x600, 0x605),
    (0x61C, 0x61C),
    (0x6DD, 0x6DD),
    (0x70F, 0x70F),
    (0x180E, 0x180E),
    (0x200B, 0x200F),
    (0x202A, 0x202E),
    (0x2060, 0x2064),
    (0x2066, 0x206F),
    (0xFEFF, 0xFEFF),
    (0xFFF9, 0xFFFB),
    (0x110BD, 0x110BD),
    (0x1BCA0, 0x1BCA3),
    (0x1D173, 0x1D17A),
    (0xE0001, 0xE0001),
    (0xE0020, 0xE007F),
];

/// Unicode punctuation categories Pc + Pd + Pe + Pf + Pi + Po + Ps, from the
/// PUNCTUATION_* tables of `unicode_categories` 0.1.1, merged into ranges.
const PUNCTUATION_RANGES: &[(u32, u32)] = &[
    (0x21, 0x23),
    (0x25, 0x2A),
    (0x2C, 0x2F),
    (0x3A, 0x3B),
    (0x3F, 0x40),
    (0x5B, 0x5D),
    (0x5F, 0x5F),
    (0x7B, 0x7B),
    (0x7D, 0x7D),
    (0xA1, 0xA1),
    (0xA7, 0xA7),
    (0xAB, 0xAB),
    (0xB6, 0xB7),
    (0xBB, 0xBB),
    (0xBF, 0xBF),
    (0x37E, 0x37E),
    (0x387, 0x387),
    (0x55A, 0x55F),
    (0x589, 0x58A),
    (0x5BE, 0x5BE),
    (0x5C0, 0x5C0),
    (0x5C3, 0x5C3),
    (0x5C6, 0x5C6),
    (0x5F3, 0x5F4),
    (0x609, 0x60A),
    (0x60C, 0x60D),
    (0x61B, 0x61B),
    (0x61E, 0x61F),
    (0x66A, 0x66D),
    (0x6D4, 0x6D4),
    (0x700, 0x70D),
    (0x7F7, 0x7F9),
    (0x830, 0x83E),
    (0x85E, 0x85E),
    (0x964, 0x965),
    (0x970, 0x970),
    (0xAF0, 0xAF0),
    (0xDF4, 0xDF4),
    (0xE4F, 0xE4F),
    (0xE5A, 0xE5B),
    (0xF04, 0xF12),
    (0xF14, 0xF14),
    (0xF3A, 0xF3D),
    (0xF85, 0xF85),
    (0xFD0, 0xFD4),
    (0xFD9, 0xFDA),
    (0x104A, 0x104F),
    (0x10FB, 0x10FB),
    (0x1360, 0x1368),
    (0x1400, 0x1400),
    (0x166D, 0x166E),
    (0x169B, 0x169C),
    (0x16EB, 0x16ED),
    (0x1735, 0x1736),
    (0x17D4, 0x17D6),
    (0x17D8, 0x17DA),
    (0x1800, 0x180A),
    (0x1944, 0x1945),
    (0x1A1E, 0x1A1F),
    (0x1AA0, 0x1AA6),
    (0x1AA8, 0x1AAD),
    (0x1B5A, 0x1B60),
    (0x1BFC, 0x1BFF),
    (0x1C3B, 0x1C3F),
    (0x1C7E, 0x1C7F),
    (0x1CC0, 0x1CC7),
    (0x1CD3, 0x1CD3),
    (0x2010, 0x2027),
    (0x2030, 0x2043),
    (0x2045, 0x2051),
    (0x2053, 0x205E),
    (0x207D, 0x207E),
    (0x208D, 0x208E),
    (0x2308, 0x230B),
    (0x2329, 0x232A),
    (0x2768, 0x2775),
    (0x27C5, 0x27C6),
    (0x27E6, 0x27EF),
    (0x2983, 0x2998),
    (0x29D8, 0x29DB),
    (0x29FC, 0x29FD),
    (0x2CF9, 0x2CFC),
    (0x2CFE, 0x2CFF),
    (0x2D70, 0x2D70),
    (0x2E00, 0x2E2E),
    (0x2E30, 0x2E42),
    (0x3001, 0x3003),
    (0x3008, 0x3011),
    (0x3014, 0x301F),
    (0x3030, 0x3030),
    (0x303D, 0x303D),
    (0x30A0, 0x30A0),
    (0x30FB, 0x30FB),
    (0xA4FE, 0xA4FF),
    (0xA60D, 0xA60F),
    (0xA673, 0xA673),
    (0xA67E, 0xA67E),
    (0xA6F2, 0xA6F7),
    (0xA874, 0xA877),
    (0xA8CE, 0xA8CF),
    (0xA8F8, 0xA8FA),
    (0xA8FC, 0xA8FC),
    (0xA92E, 0xA92F),
    (0xA95F, 0xA95F),
    (0xA9C1, 0xA9CD),
    (0xA9DE, 0xA9DF),
    (0xAA5C, 0xAA5F),
    (0xAADE, 0xAADF),
    (0xAAF0, 0xAAF1),
    (0xABEB, 0xABEB),
    (0xFD3E, 0xFD3F),
    (0xFE10, 0xFE19),
    (0xFE30, 0xFE52),
    (0xFE54, 0xFE61),
    (0xFE63, 0xFE63),
    (0xFE68, 0xFE68),
    (0xFE6A, 0xFE6B),
    (0xFF01, 0xFF03),
    (0xFF05, 0xFF0A),
    (0xFF0C, 0xFF0F),
    (0xFF1A, 0xFF1B),
    (0xFF1F, 0xFF20),
    (0xFF3B, 0xFF3D),
    (0xFF3F, 0xFF3F),
    (0xFF5B, 0xFF5B),
    (0xFF5D, 0xFF5D),
    (0xFF5F, 0xFF65),
    (0x10100, 0x10102),
    (0x1039F, 0x1039F),
    (0x103D0, 0x103D0),
    (0x1056F, 0x1056F),
    (0x10857, 0x10857),
    (0x1091F, 0x1091F),
    (0x1093F, 0x1093F),
    (0x10A50, 0x10A58),
    (0x10A7F, 0x10A7F),
    (0x10AF0, 0x10AF6),
    (0x10B39, 0x10B3F),
    (0x10B99, 0x10B9C),
    (0x11047, 0x1104D),
    (0x110BB, 0x110BC),
    (0x110BE, 0x110C1),
    (0x11140, 0x11143),
    (0x11174, 0x11175),
    (0x111C5, 0x111C9),
    (0x111CD, 0x111CD),
    (0x111DB, 0x111DB),
    (0x111DD, 0x111DF),
    (0x11238, 0x1123D),
    (0x112A9, 0x112A9),
    (0x114C6, 0x114C6),
    (0x115C1, 0x115D7),
    (0x11641, 0x11643),
    (0x1173C, 0x1173E),
    (0x12470, 0x12474),
    (0x16A6E, 0x16A6F),
    (0x16AF5, 0x16AF5),
    (0x16B37, 0x16B3B),
    (0x16B44, 0x16B44),
    (0x1BC9F, 0x1BC9F),
    (0x1DA87, 0x1DA8B),
];

// ---------------------------------------------------------------------------
// tokenizer.json parsing (only the fields the supported scheme uses)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct TokenizerFile {
    #[serde(default)]
    version: Option<String>,
    #[serde(default)]
    truncation: Option<serde_json::Value>,
    #[serde(default)]
    padding: Option<serde_json::Value>,
    #[serde(default)]
    added_tokens: Vec<AddedTokenFile>,
    #[serde(default)]
    normalizer: Option<serde_json::Value>,
    #[serde(default)]
    pre_tokenizer: Option<serde_json::Value>,
    #[serde(default)]
    post_processor: Option<serde_json::Value>,
    model: ModelFile,
}

#[derive(Deserialize)]
struct AddedTokenFile {
    id: u32,
    content: String,
    #[serde(default)]
    single_word: bool,
    #[serde(default)]
    lstrip: bool,
    #[serde(default)]
    rstrip: bool,
    #[serde(default = "default_true")]
    normalized: bool,
    // Parsed for completeness; special-ness does not change encoding here.
    #[serde(default)]
    #[allow(dead_code)]
    special: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Deserialize)]
struct ModelFile {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    unk_token: Option<String>,
    #[serde(default)]
    continuing_subword_prefix: Option<String>,
    #[serde(default)]
    max_input_chars_per_word: Option<usize>,
    vocab: HashMap<String, u32>,
}

fn parse_normalizer(value: Option<serde_json::Value>) -> Result<Normalizer> {
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
fn parse_pre_tokenizer(value: Option<serde_json::Value>) -> Result<bool> {
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
fn parse_post_processor(value: Option<serde_json::Value>) -> Result<(Option<u32>, Option<u32>)> {
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

#[cfg(test)]
mod tests;
