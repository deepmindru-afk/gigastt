//! Contextual hotword biasing for the greedy RNN-T decode loop.
//!
//! Shallow-fusion biasing steers the greedy transducer toward a curated set of
//! phrases (brands, names, domain terms) without a beam search. Each hotword is
//! tokenized to the id sequence the active head would emit (via
//! [`Tokenizer::encode_phrase`](super::tokenizer::Tokenizer::encode_phrase), so
//! it adapts to whichever vocab is loaded) and stored in a small prefix trie.
//!
//! During decode, a [`BiasState`] tracks which hotword prefixes are currently
//! "active" given the recently emitted tokens. Before the argmax over the
//! joiner logits, [`Biaser::boost_logits`] adds a fixed boost to the logits of
//! the token-ids that would extend an active prefix. A token that completes /
//! advances a prefix advances the state; anything else resets it (while still
//! letting a fresh hotword start). Blank frames leave the prefix state
//! unchanged — they emit no label, so a partially-matched hotword survives the
//! gaps between its tokens.
//!
//! The [`Biaser`] itself is immutable after construction and shared across the
//! session pool via `&Biaser`; the only mutable per-decode bookkeeping lives in
//! [`BiasState`], created fresh for each decode. When no hotwords are
//! configured the engine holds no biaser at all and the decode path is
//! byte-for-byte unchanged.

use super::tokenizer::Tokenizer;

/// Encode `phrase` in whatever spelling the active vocab can actually
/// represent, or `None` if none of them fit.
///
/// The heads disagree about spelling: the `e2e_rnnt` BPE vocab carries case,
/// the `rnnt` char vocab is 32 lowercase Cyrillic letters with no `ё`, and the
/// multilingual one adds Latin and `ё`. Users write a glossary the way the
/// words are written — `Гигаэм`, `AmoCRM`, `Пётр` — so the phrase is tried as
/// spelled first (the only form a cased vocab wants), then lowercased, then
/// with `ё` folded to the `е` a head without `ё` emits in its place. The first
/// spelling that encodes whole wins; nothing beyond that is guessed, so a Latin
/// brand still cannot be biased on a Cyrillic-only head.
fn encode_representable(tokenizer: &Tokenizer, phrase: &str) -> Option<Vec<usize>> {
    if let Some(ids) = tokenizer.encode_phrase(phrase) {
        return Some(ids);
    }
    let lowercased = phrase.to_lowercase();
    if lowercased != phrase
        && let Some(ids) = tokenizer.encode_phrase(&lowercased)
    {
        return Some(ids);
    }
    let folded = lowercased.replace('ё', "е");
    if folded != lowercased {
        return tokenizer.encode_phrase(&folded);
    }
    None
}

/// One node of the hotword prefix trie. The root is index 0.
struct TrieNode {
    /// Edges keyed by token id → child node index.
    children: std::collections::HashMap<usize, usize>,
    /// True when a hotword phrase ends here. Only the beam search reads it: a
    /// finished phrase keeps the boost it was granted, an abandoned one does
    /// not.
    is_end: bool,
    /// Scored length of the shortest phrase running through this node, used to
    /// spread one phrase's worth of boost across its tokens.
    shortest_phrase: usize,
    /// Distance from the root; 1 for a phrase's first token.
    depth: usize,
    /// True when this node is only the word-boundary marker that opens a
    /// phrase. Matched, never paid for — it is the same token for every hotword
    /// ever configured.
    is_entry: bool,
    /// Boost paid for entering this node under the beam search's per-phrase
    /// budget. Zero at the entry marker.
    grant: f32,
}

impl TrieNode {
    fn new(depth: usize) -> Self {
        Self {
            children: std::collections::HashMap::new(),
            is_end: false,
            shortest_phrase: usize::MAX,
            depth,
            is_entry: false,
            grant: 0.0,
        }
    }
}

/// Where one beam sits in the hotword trie, and how much boost it has been
/// granted for a phrase it has not finished.
///
/// The greedy transducer path cannot take a bonus back — it has already emitted
/// the token — so it lives with a rationed boost instead
/// ([`super::decode::greedy_decode`]). A beam search can: a hypothesis that
/// walked halfway into a hotword and then left is refunded, so a partial match
/// wins nothing and only a phrase actually spoken keeps its advantage.
#[derive(Clone, Copy, Default)]
pub(crate) struct BiasPath {
    /// Current trie node; 0 is the root.
    node: usize,
    /// Boost granted so far for the unfinished phrase under way.
    pending: f32,
}

impl BiasPath {
    /// Boost this path has been granted for a phrase it has not finished.
    ///
    /// Owed back: a hypothesis still mid-phrase when the audio runs out never
    /// earned it, so the final ranking has to discount it. During the search
    /// the amount stays credited — that is what keeps a half-matched phrase in
    /// the beam long enough to finish.
    pub(crate) fn pending(&self) -> f32 {
        self.pending
    }
}

/// Compiled hotword biaser: a prefix trie over hotword token-id sequences plus
/// the additive logit boost. Immutable and shareable across inference sessions.
pub struct Biaser {
    nodes: Vec<TrieNode>,
    /// Additive boost applied to a continuation token's logit.
    boost: f32,
    /// Number of distinct hotword phrases successfully compiled.
    phrase_count: usize,
}

impl Biaser {
    /// Build a biaser from hotword token-id sequences and a boost. Sequences
    /// must be non-empty; empty ones are skipped. Returns `None` if no sequence
    /// survives (so callers treat "no usable hotwords" as biasing-off).
    ///
    /// Test-only, and the raw sequences carry no word-boundary marker: this is
    /// how the decode-loop tests build a biaser without a tokenizer, so nothing
    /// here is treated as a phrase-entry precondition.
    #[cfg(test)]
    pub(crate) fn from_sequences(sequences: Vec<Vec<usize>>, boost: f32) -> Option<Self> {
        Self::build(sequences, boost, false)
    }

    /// Compile `sequences` into the trie.
    ///
    /// `leading_is_entry` says the first token of every sequence is the
    /// word-boundary marker that [`Tokenizer::encode_phrase`] prepends. That
    /// token is a *precondition* for the phrase, not part of what makes it
    /// distinctive — it is the same token for every hotword ever configured, so
    /// paying for it means paying at every word boundary in the audio no matter
    /// what the glossary says. It is matched but never scored.
    fn build(sequences: Vec<Vec<usize>>, boost: f32, leading_is_entry: bool) -> Option<Self> {
        let mut nodes = vec![TrieNode::new(0)];
        let mut phrase_count = 0;
        let entry_tokens = usize::from(leading_is_entry);
        for seq in sequences {
            if seq.is_empty() {
                continue;
            }
            phrase_count += 1;
            // Tokens the boost is actually spread over.
            let scored = seq.len().saturating_sub(entry_tokens).max(1);
            let mut node = 0usize;
            for tok in seq {
                node = match nodes[node].children.get(&tok) {
                    Some(&child) => child,
                    None => {
                        let depth = nodes[node].depth + 1;
                        let child = nodes.len();
                        nodes.push(TrieNode::new(depth));
                        nodes[node].children.insert(tok, child);
                        child
                    }
                };
                nodes[node].shortest_phrase = nodes[node].shortest_phrase.min(scored);
            }
            nodes[node].is_end = true;
        }
        if phrase_count == 0 {
            return None;
        }
        for node in nodes.iter_mut().skip(1) {
            node.is_entry = leading_is_entry && node.depth == 1;
            node.grant = if node.is_entry {
                0.0
            } else {
                boost / node.shortest_phrase.max(1) as f32
            };
        }
        Some(Self {
            nodes,
            boost,
            phrase_count,
        })
    }

    /// Build a biaser from `(phrase, weight)` pairs, tokenizing each phrase with
    /// the active [`Tokenizer`]. `weight` scales the base `boost` per phrase
    /// (use `1.0` for the default). Phrases the tokenizer can't represent are
    /// dropped. Returns `None` if no phrase compiles or `boost <= 0`.
    pub fn from_phrases(
        tokenizer: &Tokenizer,
        phrases: &[(String, f32)],
        boost: f32,
    ) -> Option<Self> {
        if boost <= 0.0 {
            return None;
        }
        // Per-phrase weights are folded into the boost by storing the *highest*
        // requested boost on each trie edge would complicate the immutable
        // node layout; instead we keep a single base boost and treat the weight
        // as a phrase-level filter (weight <= 0 drops the phrase). A future
        // per-edge weight can extend TrieNode without touching the decode loop.
        let mut sequences = Vec::new();
        let mut dropped: Vec<&str> = Vec::new();
        for (phrase, weight) in phrases {
            if *weight <= 0.0 {
                continue;
            }
            match encode_representable(tokenizer, phrase) {
                Some(ids) => sequences.push(ids),
                None => dropped.push(phrase),
            }
        }
        if !dropped.is_empty() {
            // Named, not just counted: which phrases fell out is the whole
            // actionable content — a Cyrillic-only head can never represent a
            // Latin brand, and the only way a user learns that is by reading
            // its name here.
            tracing::warn!(
                "{} hotword phrase(s) dropped, not representable in the active vocab: {}",
                dropped.len(),
                dropped.join(", ")
            );
        }
        Self::build(sequences, boost, true)
    }

    /// Number of hotword phrases compiled into the trie.
    pub fn phrase_count(&self) -> usize {
        self.phrase_count
    }

    /// Score emitting `tok` from `path`, returning the log-domain delta to add
    /// to the hypothesis and the path that follows.
    ///
    /// Three cases, and the middle one is why this exists:
    /// - the token continues the phrase under way — grant the boost;
    /// - it does not, but starts some phrase — refund everything the abandoned
    ///   partial match was granted, then grant the boost for the new start;
    /// - it is not a hotword token at all — refund and return to the root.
    ///
    /// A node that ends a phrase clears the pending amount: the phrase was
    /// spoken, so its boost is earned and is never taken back.
    pub(crate) fn score_token(&self, path: BiasPath, tok: usize) -> (f32, BiasPath) {
        // A phrase is worth `boost` however long it is, so each of its tokens
        // is granted a share rather than the whole amount.
        //
        // A character vocabulary makes the difference stark: paid per token, a
        // nine-letter phrase would earn nine times the boost and outrank
        // whatever was actually said — `любовницы` really did displace
        // `люк кейдж` that way. Clawing the excess back on completion is worse
        // still: the correction lands as one large negative step and the
        // hypothesis that just finished the phrase gets pruned for it. Granting
        // the right amount from the start keeps every step small.
        let enter = |from: BiasPath, child: usize| {
            let share = self.nodes[child].grant;
            (
                share,
                BiasPath {
                    node: child,
                    // A finished phrase owes nothing back; an unfinished one
                    // owes everything it has been granted so far.
                    pending: if self.nodes[child].is_end {
                        0.0
                    } else {
                        from.pending + share
                    },
                },
            )
        };

        if let Some(&child) = self.nodes[path.node].children.get(&tok) {
            return enter(path, child);
        }

        // The phrase under way dies here: take back what it was granted, and
        // let a fresh phrase start on the same token.
        let refund = -path.pending;
        match self.nodes[0].children.get(&tok) {
            Some(&child) => {
                let (delta, next) = enter(BiasPath::default(), child);
                (refund + delta, next)
            }
            None => (refund, BiasPath::default()),
        }
    }

    /// Token ids that would extend the phrase `path` is in, plus every phrase
    /// start. A beam search hands these to the candidate set so a boosted
    /// continuation can be considered even when the acoustic model ranks it
    /// below the pruning cut — which is the entire point of biasing.
    pub(crate) fn continuations(&self, path: BiasPath, out: &mut Vec<usize>) {
        out.extend(self.nodes[path.node].children.keys().copied());
        if path.node != 0 {
            out.extend(self.nodes[0].children.keys().copied());
        }
    }

    /// Create a fresh per-decode prefix-tracking state rooted at the trie root.
    pub(crate) fn new_state(&self) -> BiasState {
        BiasState {
            // The root is always active so a new hotword can start at any token.
            active: vec![0],
        }
    }

    /// Add the boost to `logits` for every token id that extends a currently
    /// active hotword prefix. No-op when no active node has children (i.e. no
    /// hotword could continue here), so non-hotword regions are untouched.
    pub(crate) fn boost_logits(&self, state: &BiasState, logits: &mut [f32]) {
        for &node in &state.active {
            for (&tok, &child) in &self.nodes[node].children {
                if tok < logits.len() && !self.nodes[child].is_entry {
                    // The full boost, not the beam's per-phrase share: a greedy
                    // argmax decides each step on its own, so the step delta is
                    // the whole mechanism. Splitting it across a phrase's
                    // characters — right when totals compete in a beam — leaves
                    // a long hotword too weak to win any single step, which is
                    // to say it turns biasing off.
                    logits[tok] += self.boost;
                }
            }
        }
    }

    /// Advance the prefix state after a non-blank token `tok` was emitted.
    ///
    /// New active set = the children reached by `tok` from any previously active
    /// node, plus the root (so a fresh hotword can begin on the next token).
    /// Deduplicated to keep the active set small.
    pub(crate) fn advance(&self, state: &mut BiasState, tok: usize) {
        let mut next = Vec::new();
        for &node in &state.active {
            if let Some(&child) = self.nodes[node].children.get(&tok)
                && !next.contains(&child)
            {
                next.push(child);
            }
        }
        // The root stays active so biasing can restart at the next token.
        if !next.contains(&0) {
            next.push(0);
        }
        state.active = next;
    }
}

/// Per-decode hotword prefix-tracking state. Holds the set of trie nodes whose
/// prefix has been matched by the recently emitted tokens. Cheap to create;
/// one per [`greedy_decode`](super::decode::greedy_decode) call.
pub(crate) struct BiasState {
    active: Vec<usize>,
}

#[cfg(test)]
mod tests;
