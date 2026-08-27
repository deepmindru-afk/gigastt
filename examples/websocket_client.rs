//! WebSocket client that connects to a running gigastt server and streams audio.
//!
//! Reference snippet: the top-level `examples/` directory is not wired into
//! Cargo, so `cargo run --example websocket_client` does not work. To build it,
//! copy this file into a scratch crate (`cargo new ws-client`) with `anyhow`,
//! `futures-util`, `serde_json`, `tokio` (full), `tokio-tungstenite`, and
//! `tracing-subscriber` in `Cargo.toml`, or drop it under `crates/gigastt/examples/`.
//!
//! Usage:
//!   1. Start server: cargo run -- serve
//!   2. Run client:   cargo run -- path/to/audio.wav   (from the scratch crate)

use anyhow::Result;
use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::tungstenite::Message;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: websocket_client <16kHz-mono-PCM16.wav>");
        std::process::exit(1);
    });

    let url = std::env::var("GIGASTT_URL").unwrap_or_else(|_| "ws://127.0.0.1:9876/v1/ws".into());

    println!("Connecting to {url}...");
    let (ws, _) = tokio_tungstenite::connect_async(&url).await?;
    let (mut sink, mut stream) = ws.split();

    // Wait for Ready
    if let Some(Ok(msg)) = stream.next().await {
        let text = msg.into_text()?;
        let v: serde_json::Value = serde_json::from_str(&text)?;
        println!(
            "Server ready: model={}, sample_rate={}",
            v["model"], v["sample_rate"]
        );
    }

    // The session default is 48000 Hz; our WAV is 16 kHz PCM16, so declare the
    // real rate before the first audio frame (otherwise audio plays back 3x slow).
    sink.send(Message::Text(
        serde_json::to_string(&serde_json::json!({"type": "configure", "sample_rate": 16000}))
            .unwrap()
            .into(),
    ))
    .await?;

    // Strip the 44-byte WAV header: the WebSocket stream expects raw PCM16,
    // not a RIFF container (the other examples do the same).
    const WAV_HEADER_BYTES: usize = 44;
    let file_data = std::fs::read(&path)?;
    let audio_data = file_data.get(WAV_HEADER_BYTES..).unwrap_or(&[]);
    println!("Sending {} bytes of audio...", audio_data.len());
    for chunk in audio_data.chunks(32 * 1024) {
        sink.send(Message::Binary(chunk.to_vec().into())).await?;
    }

    // Send stop
    sink.send(Message::Text(
        serde_json::to_string(&serde_json::json!({"type": "stop"}))
            .unwrap()
            .into(),
    ))
    .await?;

    // Read responses until Final
    while let Some(Ok(msg)) = stream.next().await {
        if let Ok(text) = msg.into_text() {
            let v: serde_json::Value = serde_json::from_str(&text)?;
            match v["type"].as_str() {
                Some("partial") => println!("  Partial: {}", v["text"]),
                Some("final") => {
                    println!("  Final:   {}", v["text"]);
                    break;
                }
                Some("error") => {
                    eprintln!("  Error: {}", v["message"]);
                    break;
                }
                _ => {}
            }
        }
    }

    sink.close().await?;
    Ok(())
}
