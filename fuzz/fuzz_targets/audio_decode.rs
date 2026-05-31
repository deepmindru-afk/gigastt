//! Fuzz the audio-file decode path (symphonia) with untrusted file bytes.
//!
//! `decode_audio_bytes` feeds the input through the crate-private
//! `BytesMediaSource` into symphonia's probe → decode → mono-mix → resample
//! pipeline — the exact path a REST upload of an attacker-controlled file
//! takes. All input is untrusted: any `Err` is fine, the property under test
//! is "no panic / no UB on arbitrary bytes".
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = gigastt_core::inference::audio::decode_audio_bytes(data);
});
