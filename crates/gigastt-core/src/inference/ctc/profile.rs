//! Private CTC benchmark seams. Instrumented and timed passes are separate.
use super::*;
use std::path::Path;

#[derive(Debug, Default, Clone, serde::Serialize)]
pub struct BeamStats {
    pub frames: usize,
    pub candidates_total: usize,
    pub candidates_max: usize,
    pub live_prefixes_max: usize,
    pub prefix_length_max: usize,
    pub candidate_buffers_max_bytes: usize,
}

impl BeamObserver for BeamStats {
    fn frame(&mut self, candidates: usize, next: &[(Vec<usize>, Hypothesis)]) {
        self.frames += 1;
        self.candidates_total += candidates;
        self.candidates_max = self.candidates_max.max(candidates);
        self.live_prefixes_max = self.live_prefixes_max.max(next.len());
        self.prefix_length_max = self.prefix_length_max.max(
            next.iter()
                .map(|(labels, _)| labels.len())
                .max()
                .unwrap_or(0),
        );
        self.candidate_buffers_max_bytes = self.candidate_buffers_max_bytes.max(
            next.iter()
                .map(|(labels, hyp)| {
                    labels.capacity() * std::mem::size_of::<usize>()
                        + hyp.tokens.capacity() * std::mem::size_of::<TokenInfo>()
                })
                .sum(),
        );
    }
}

pub struct BeamOutput(Vec<TokenInfo>);

impl PartialEq for BeamOutput {
    fn eq(&self, other: &Self) -> bool {
        self.0.len() == other.0.len()
            && self.0.iter().zip(&other.0).all(|(a, b)| {
                a.token_id == b.token_id
                    && a.frame_index == b.frame_index
                    && a.confidence.to_bits() == b.confidence.to_bits()
            })
    }
}
impl Eq for BeamOutput {}

impl BeamOutput {
    pub fn alignment(&self) -> Vec<(usize, usize, u32)> {
        self.0
            .iter()
            .map(|token| {
                (
                    token.token_id,
                    token.frame_index,
                    token.confidence.to_bits(),
                )
            })
            .collect()
    }
}

pub struct BeamProbe {
    tokenizer: Tokenizer,
    biaser: Option<Biaser>,
}

impl BeamProbe {
    pub fn new(vocab_path: &Path, phrases: &[(String, f32)], boost: f32) -> anyhow::Result<Self> {
        let tokenizer = Tokenizer::load(vocab_path)?;
        let biaser = Biaser::from_phrases(&tokenizer, phrases, boost);
        Ok(Self { tokenizer, biaser })
    }

    pub fn phrase_count(&self) -> usize {
        self.biaser.as_ref().map_or(0, Biaser::phrase_count)
    }

    pub fn decode(
        &self,
        logits: &[f32],
        frames: usize,
        abort: Option<&(dyn Fn() -> bool + Sync)>,
    ) -> BeamOutput {
        let vocab = self.tokenizer.vocab_size();
        let blank = self.tokenizer.blank_id();
        BeamOutput(match &self.biaser {
            Some(biaser) => {
                ctc_prefix_beam_decode_with_abort(logits, frames, vocab, blank, biaser, abort)
            }
            None => ctc_greedy_decode_with_abort(logits, frames, vocab, blank, abort),
        })
    }

    pub fn profile(&self, logits: &[f32], frames: usize) -> (BeamOutput, BeamStats) {
        let mut stats = BeamStats::default();
        let output = match &self.biaser {
            Some(biaser) => BeamOutput(beam_decode(
                logits,
                frames,
                self.tokenizer.vocab_size(),
                self.tokenizer.blank_id(),
                biaser,
                None,
                &mut stats,
            )),
            None => self.decode(logits, frames, None),
        };
        (output, stats)
    }

    pub fn text(&self, output: &BeamOutput) -> String {
        ctc_tokens_to_words(&self.tokenizer, &output.0, 0)
            .iter()
            .map(|word| word.word.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

pub fn file_sha256(path: &Path) -> std::io::Result<String> {
    use crate::sha256::{Sha256, hex_lower};
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0; 65536];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(hex_lower(&hash.finalize()))
}
