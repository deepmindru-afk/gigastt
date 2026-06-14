# Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `protoc` not found during build | Missing Protocol Buffers compiler | `brew install protobuf` (macOS) or `apt install protobuf-compiler` (Debian/Ubuntu) |
| Model download hangs or fails | Network / HuggingFace availability | Retry `gigastt download`; check `~/.gigastt/models/` permissions |
| `Cannot quantize: FP32 encoder not found` | Partial download | Delete `~/.gigastt/models/` and re-run `gigastt download` |
| OOM on startup | Pool size too large for available RAM | Lower `--pool-size` (default 4); each session loads the full encoder |
| CoreML not used on macOS | Built without `--features coreml` | Re-build: `cargo build --release --features coreml` |
| `falling back to CPU execution provider` in logs | CoreML failed to compile/execute on this macOS/model combo | Transcription still works on CPU; clear `~/.gigastt/models/coreml_cache/` and retry, or file an issue with the warning text |
| CUDA not available on Linux | Built without `--features cuda` or missing CUDA 12+ | Re-build: `cargo build --release --features cuda`; verify `nvidia-smi` |
| WebSocket closes with 1008 | Session exceeded `--max-session-secs` | Increase `--max-session-secs` or send shorter streams |
| 429 Too Many Requests | Rate limiter enabled and bucket exhausted | Wait for `Retry-After`, or disable with `--rate-limit-per-minute 0` |
| Empty transcription for noisy audio | Input too quiet or wrong format | Ensure 16-bit PCM; normalize level; check supported formats |
