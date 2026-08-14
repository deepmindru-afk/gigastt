//! OpenAI-compatible transcript SSE events and the append-only assembler.

use axum::response::sse::Event;
use serde::Serialize;

/// OpenAI streaming event: incremental text.
#[derive(Debug, Serialize)]
pub struct OpenAITranscriptDelta {
    #[serde(rename = "type")]
    pub event_type: &'static str,
    pub delta: String,
}

/// OpenAI streaming event: full transcript at end of stream.
#[derive(Debug, Serialize)]
pub struct OpenAITranscriptDone {
    #[serde(rename = "type")]
    pub event_type: &'static str,
    pub text: String,
}

/// SSE `data:` payload for a text delta.
pub fn sse_delta_payload(delta: &str) -> String {
    serde_json::to_string(&OpenAITranscriptDelta {
        event_type: "transcript.text.delta",
        delta: delta.to_string(),
    })
    .unwrap_or_else(|_| r#"{"type":"transcript.text.delta","delta":""}"#.into())
}

/// SSE `data:` payload for the terminal done event.
pub fn sse_done_payload(text: &str) -> String {
    serde_json::to_string(&OpenAITranscriptDone {
        event_type: "transcript.text.done",
        text: text.to_string(),
    })
    .unwrap_or_else(|_| r#"{"type":"transcript.text.done","text":""}"#.into())
}

/// Build an axum SSE event for a delta / done / `[DONE]` marker.
pub fn sse_event_data(data: impl Into<String>) -> Event {
    Event::default().data(data.into())
}

/// Tracks append-only OpenAI text for progressive streaming.
///
/// Gigastt partials rewrite the *current* utterance; finals close it.
/// Deltas are emitted only when the cumulative view is a prefix extension of
/// what was already sent (OpenAI deltas are append-only — never retract).
#[derive(Debug, Default)]
pub struct OpenAIStreamAssembler {
    /// Text from completed (final) utterances.
    committed: String,
    /// Full cumulative text already sent as deltas.
    last_emitted: String,
}

impl OpenAIStreamAssembler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Full transcript for the terminal `transcript.text.done` event.
    pub fn text(&self) -> &str {
        // Prefer committed after finals; fall back to live-emitted partials.
        if self.committed.len() >= self.last_emitted.len() {
            &self.committed
        } else {
            &self.last_emitted
        }
    }

    /// Ingest one native segment; return an optional delta to send.
    pub fn push_segment(&mut self, text: &str, is_final: bool) -> Option<String> {
        let live = text.trim();
        let candidate = match (self.committed.is_empty(), live.is_empty()) {
            (_, true) => self.committed.clone(),
            (true, false) => live.to_string(),
            (false, false) => format!("{} {live}", self.committed),
        };

        let mut delta = None;
        if candidate.starts_with(&self.last_emitted) {
            let d = candidate[self.last_emitted.len()..].to_string();
            if !d.is_empty() {
                self.last_emitted.clone_from(&candidate);
                delta = Some(d);
            }
        }
        // else: partial rewrote earlier tokens — skip (cannot unsend)

        if is_final {
            if !live.is_empty() {
                self.committed = if self.committed.is_empty() {
                    live.to_string()
                } else {
                    format!("{} {live}", self.committed)
                };
            }
            // Align emission cursor with committed when it is a pure extension.
            if self.committed.starts_with(&self.last_emitted) {
                let d = self.committed[self.last_emitted.len()..].to_string();
                self.last_emitted.clone_from(&self.committed);
                if delta.is_none() && !d.is_empty() {
                    delta = Some(d);
                }
            } else {
                // Final disagrees with what we streamed — snap cursor for `done`
                // accuracy without inventing a retracting delta.
                self.last_emitted.clone_from(&self.committed);
            }
        }
        delta
    }
}
