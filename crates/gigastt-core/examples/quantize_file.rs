//! Dev helper: quantize an arbitrary ONNX to INT8 with gigastt's native quantizer
//! (`quantize::quantize_model`) — used to INT8 the GigaAM-Multilingual CTC export
//! for the WER fp32-vs-int8 comparison (roadmap 130/133). Not part of the shipped
//! CLI; the `serve`/`download`/`quantize` subcommands wire the same function to the
//! gigastt model filenames.
//!
//! Usage: cargo run -p gigastt-core --example quantize_file -- <input.onnx> <output.onnx>
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: quantize_file <input.onnx> <output.onnx>");
        std::process::exit(2);
    }
    let (input, output) = (Path::new(&args[1]), Path::new(&args[2]));
    let t = std::time::Instant::now();
    gigastt_core::quantize::quantize_model(input, output)?;
    eprintln!(
        "quantized in {:.1}s -> {}",
        t.elapsed().as_secs_f64(),
        output.display()
    );
    Ok(())
}
