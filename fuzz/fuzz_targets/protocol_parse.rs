//! Fuzz inbound WebSocket text-frame parsing.
//!
//! The server deserializes client control frames into
//! [`gigastt_core::protocol::ClientMessage`] via serde_json. This feeds the
//! raw frame bytes straight into `serde_json::from_slice` to prove the
//! deserializer never panics on malformed / hostile input — only returns a
//! clean `Err`.
#![no_main]

use gigastt_core::protocol::ClientMessage;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = serde_json::from_slice::<ClientMessage>(data);
});
