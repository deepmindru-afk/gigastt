# gigastt-quantize

Native Rust INT8 dynamic quantizer for ONNX encoder graphs, extracted from
[`gigastt`](https://github.com/ekhodzitsky/gigastt).

Produces `MatMulInteger` / `ConvInteger` integer-compute graphs, shrinking the
GigaAM v3 encoder from ~844 MB to ~215 MB.

Requires `protoc` in `PATH` at build time: the ONNX protobuf types are
regenerated from the vendored `proto/onnx.proto` via `prost-build`.

```rust
gigastt_quantize::quantize_model(&input_path, &output_path)?;
```

MIT.
