//! Transcribe an audio file using the gigastt engine directly.
//!
//! Reference snippet: the top-level `examples/` directory is not wired into
//! Cargo, so `cargo run --example transcribe_file` does not work. To run it,
//! copy this file under `crates/gigastt/examples/` (which is on the Cargo
//! example path) and use `cargo run --example transcribe_file -- path/to/audio.wav`
//! from the workspace root, or build it inside a scratch crate that depends on
//! `gigastt`. The model must already be downloaded (`cargo run -- download`).

use anyhow::Result;

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: transcribe_file <audio-file>");
        std::process::exit(1);
    });

    let model_dir = gigastt::model::default_model_dir();
    println!("Loading model from {model_dir}...");
    let engine = gigastt::inference::Engine::load(&model_dir)?;
    println!("Model loaded (INT8: {})", engine.is_int8());

    println!("Transcribing {path}...");
    let mut guard = engine
        .pool
        .checkout_blocking()
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let result = engine.transcribe_file(&path, &mut guard)?;
    println!("\nText: {}", result.text);
    println!("Duration: {:.1}s", result.duration_s);
    for word in &result.words {
        println!("  [{:.2}s] {}", word.start, word.word);
    }

    Ok(())
}
