//! Model-free ORT allocation regression for reusable output buffers.
use gigastt_core::runtime_api::{Shape, Tensor, TensorData, cpu_factory};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct CountingAllocator;
thread_local! { static ALLOCATIONS: Cell<Option<(usize, usize)>> = const { Cell::new(None) }; }
// SAFETY: allocations are forwarded unchanged to System; counters are thread-local.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.with(|count| {
            if let Some((calls, bytes)) = count.get() {
                count.set(Some((calls + 1, bytes + layout.size())));
            }
        });
        // SAFETY: forward the caller's valid layout to System.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: pointer and layout originate from this allocator.
        unsafe { System.dealloc(ptr, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn measured<T>(f: impl FnOnce() -> T) -> (T, (usize, usize)) {
    ALLOCATIONS.with(|count| count.set(Some((0, 0))));
    let result = f();
    let count = ALLOCATIONS.with(|count| count.replace(None).unwrap());
    (result, count)
}

#[test]
#[cfg_attr(miri, ignore = "calls into onnxruntime FFI")]
fn test_ort_output_workspace_avoids_owned_output_allocations() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("identity.onnx");
    // ONNX IR 8, opset 13: Identity(x: float[1]) -> y: float[1].
    std::fs::write(&path, b"\x08\x08\x3a\x40\x0a\x10\x0a\x01x\x12\x01y\x22\x08Identity\x12\x0acache-test\x5a\x0f\x0a\x01x\x12\x0a\x0a\x08\x08\x01\x12\x04\x0a\x02\x08\x01\x62\x0f\x0a\x01y\x12\x0a\x0a\x08\x08\x01\x12\x04\x0a\x02\x08\x01\x42\x02\x10\x0d").unwrap();
    let runtime = cpu_factory().create(1).unwrap();
    let session = runtime.load_session(&path, false).unwrap();
    let input = [Tensor::new(Shape::new(vec![1]), TensorData::F32(vec![3.25])).unwrap()];
    let mut destination = vec![0.0];
    session
        .run_f32_into(&input, &mut [&mut destination])
        .unwrap();
    let pointer = destination.as_ptr();
    let (_, old) = measured(|| {
        for _ in 0..100 {
            let outputs = session.run(&input).unwrap();
            destination.clear();
            destination.extend_from_slice(outputs[0].view().data().as_f32().unwrap());
        }
    });
    let (_, new) = measured(|| {
        for _ in 0..100 {
            session
                .run_f32_into(&input, &mut [&mut destination])
                .unwrap();
        }
    });
    assert_eq!(destination, [3.25]);
    assert_eq!(pointer, destination.as_ptr());
    eprintln!("legacy={old:?} workspace={new:?} (Rust allocations, bytes), 100 calls");
    assert!(new.0 < old.0 && new.1 < old.1, "old={old:?}, new={new:?}");
}

#[test]
#[cfg(feature = "file-decode")]
#[ignore = "requires installed rnnt and e2e_rnnt models; synthetic per-stage allocation probe"]
fn benchmark_installed_runtime_output_workspace() {
    use gigastt_core::inference::{FeatureExtractor, audio::decode_audio_file};
    let model_dir =
        std::path::PathBuf::from(std::env::var_os("HOME").unwrap()).join(".gigastt/models");
    let samples = decode_audio_file(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../gigastt/tests/fixtures/golos_00.wav"
    ))
    .unwrap();
    let (mel, frames) = FeatureExtractor::new().compute(&samples);
    for variant in ["rnnt", "e2e_rnnt"] {
        let runtime = cpu_factory().create(4).unwrap();
        let encoder = runtime
            .load_session(
                &model_dir.join(format!("v3_{variant}_encoder_int8.onnx")),
                true,
            )
            .unwrap();
        let decoder = runtime
            .load_session(&model_dir.join(format!("v3_{variant}_decoder.onnx")), false)
            .unwrap();
        let joiner = runtime
            .load_session(&model_dir.join(format!("v3_{variant}_joint.onnx")), false)
            .unwrap();
        let encoder_inputs = [
            Tensor::new(
                Shape::new(vec![1, 64, frames]),
                TensorData::F32(mel.clone()),
            )
            .unwrap(),
            Tensor::new(Shape::new(vec![1]), TensorData::I64(vec![frames as i64])).unwrap(),
        ];
        let encoded = encoder.run(&encoder_inputs).unwrap();
        let encoded_frames = encoded[0].shape().dims()[2];
        let encoded_data = encoded[0].view().data().as_f32().unwrap();
        let decoder_inputs = [
            Tensor::new(
                Shape::new(vec![1, 1]),
                TensorData::I64(vec![if variant == "rnnt" { 33 } else { 1024 }]),
            )
            .unwrap(),
            Tensor::new(Shape::new(vec![1, 1, 320]), TensorData::F32(vec![0.0; 320])).unwrap(),
            Tensor::new(Shape::new(vec![1, 1, 320]), TensorData::F32(vec![0.0; 320])).unwrap(),
        ];
        let decoded = decoder.run(&decoder_inputs).unwrap();
        let joiner_inputs = [
            Tensor::new(
                Shape::new(vec![1, 768, 1]),
                TensorData::F32((0..768).map(|c| encoded_data[c * encoded_frames]).collect()),
            )
            .unwrap(),
            Tensor::new(
                Shape::new(vec![1, 320, 1]),
                TensorData::F32(decoded[0].view().data().as_f32().unwrap().to_vec()),
            )
            .unwrap(),
        ];
        for (name, session, inputs, iterations) in [
            ("encoder", &*encoder, encoder_inputs.as_slice(), 5),
            ("decoder", &*decoder, decoder_inputs.as_slice(), 200),
            ("joiner", &*joiner, joiner_inputs.as_slice(), 200),
        ] {
            let reference = session.run(inputs).unwrap();
            let copy_bytes: usize = reference
                .iter()
                .map(|t| {
                    t.shape().elements()
                        * if t.view().data().as_i64().is_some() {
                            8
                        } else {
                            4
                        }
                })
                .sum();
            let mut buffers: Vec<Vec<f32>> = reference
                .iter()
                .map(|t| t.view().data().as_f32().unwrap_or(&[]).to_vec())
                .collect();
            for legacy in [true, false] {
                let mut outputs: Vec<_> = buffers.iter_mut().collect();
                let start = std::time::Instant::now();
                let (_, allocations) = measured(|| {
                    for _ in 0..iterations {
                        if name == "encoder" {
                            std::hint::black_box(session.run(inputs).unwrap());
                        } else if legacy {
                            let values = session.run(inputs).unwrap();
                            for (value, target) in values.iter().zip(outputs.iter_mut()) {
                                target.clear();
                                target.extend_from_slice(value.view().data().as_f32().unwrap());
                            }
                        } else {
                            session.run_f32_into(inputs, &mut outputs).unwrap();
                        }
                    }
                });
                let elapsed = start.elapsed();
                eprintln!(
                    "{variant} {name} legacy={legacy}: calls={iterations} allocations/call={} bytes/call={} mean_us={} seam_copy_bytes/call={}",
                    allocations.0 / iterations,
                    allocations.1 / iterations,
                    elapsed.as_micros() / iterations as u128,
                    copy_bytes * if legacy && name != "encoder" { 2 } else { 1 }
                );
                if name != "encoder" {
                    for (value, actual) in reference.iter().zip(outputs.iter()) {
                        assert_eq!(value.view().data().as_f32().unwrap(), actual.as_slice());
                    }
                }
            }
        }
    }
}
