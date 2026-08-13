use super::*;

#[test]
fn test_calc_output_length_known_points() {
    // ×4 subsampling: matches T/4 for multiples of 4.
    assert_eq!(calc_output_length(768), 192);
    assert_eq!(calc_output_length(1500), 375);
    assert_eq!(calc_output_length(3000), 750);
    // Odd / non-multiple-of-4 values: ceil-div twice.
    assert_eq!(calc_output_length(593), 149);
    assert_eq!(calc_output_length(250), 63);
    assert_eq!(calc_output_length(769), 193);
    assert_eq!(calc_output_length(400), 100);
}

#[test]
fn test_select_bucket_fill_floor_cases() {
    let buckets = [512usize, 768, 1536, 3000];
    // 400/512 = 78% → smallest N≥400 meeting floor is 512.
    assert_eq!(select_bucket(400, &buckets, 0.5), Some(512));
    // 300/512 = 58.6% → 512.
    assert_eq!(select_bucket(300, &buckets, 0.5), Some(512));
    // 200/512 = 39% < floor → no bucket, fallback.
    assert_eq!(select_bucket(200, &buckets, 0.5), None);
    // 600 > 512 so 512 fails N≥T; 600/768 = 78% → 768.
    assert_eq!(select_bucket(600, &buckets, 0.5), Some(768));
    // 800 > 768 so 768 fails N≥T; smallest N≥800 is 1536 (800/1536=52%).
    assert_eq!(select_bucket(800, &buckets, 0.5), Some(1536));
    // 769 → 1536 (769/1536 = 50.06% ≥ floor).
    assert_eq!(select_bucket(769, &buckets, 0.5), Some(1536));
    // 4000 > max bucket → fallback.
    assert_eq!(select_bucket(4000, &buckets, 0.5), None);
    // Exact fit.
    assert_eq!(select_bucket(768, &buckets, 0.5), Some(768));
    assert_eq!(select_bucket(512, &buckets, 0.5), Some(512));
}

#[test]
fn test_select_bucket_fill_equal_to_floor_is_selected() {
    // 384/768 = exactly 0.5 == floor → the `>=` comparison must include the
    // boundary, so the bucket is selected (not rejected to the ort fallback).
    assert_eq!(select_bucket(384, &[768], 0.5), Some(768));
    // 256/512 = exactly 0.5.
    assert_eq!(select_bucket(256, &[512], 0.5), Some(512));
}

#[test]
fn test_select_bucket_unsorted_input() {
    let buckets = [3000usize, 512, 768, 1536];
    assert_eq!(select_bucket(300, &buckets, 0.5), Some(512));
    assert_eq!(select_bucket(400, &buckets, 0.5), Some(512));
    assert_eq!(select_bucket(600, &buckets, 0.5), Some(768));
    assert_eq!(select_bucket(800, &buckets, 0.5), Some(1536));
}

/// Streaming-sized windows (~249 mel frames @ 2.5 s) sit under the file-mode
/// 50% floor of bucket 512 but must select that bucket under the streaming
/// floor so live sessions can run on ANE.
#[test]
fn test_select_bucket_streaming_floor_accepts_underfilled_window() {
    let buckets = [512usize, 768, 1536, 3000];
    // ~2.5 s streaming window.
    const T: usize = 249;
    // File-mode floor still rejects (249/512 ≈ 48.6% < 0.5).
    assert_eq!(
        select_bucket(T, &buckets, FILL_FLOOR),
        None,
        "file-mode floor must still reject underfilled streaming-sized T"
    );
    // Streaming floor pads into the smallest eligible bucket.
    assert_eq!(
        select_bucket(T, &buckets, STREAMING_FILL_FLOOR),
        Some(512),
        "streaming floor must select bucket 512 for a 249-frame window"
    );
    // Over-max still rejected even with zero floor.
    assert_eq!(select_bucket(4000, &buckets, STREAMING_FILL_FLOOR), None);
}

#[test]
fn test_pad_time_appends_zeros_per_channel() {
    // 2 channels, t=2, n=4: each row [a,b] -> [a,b,0,0].
    let mel = vec![1.0, 2.0, 3.0, 4.0];
    let padded = pad_time(&mel, 2, 2, 4);
    assert_eq!(padded, vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0]);
    assert_eq!(padded.len(), 2 * 4);
}

#[test]
fn test_pad_time_noop_when_t_equals_n() {
    let mel = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let padded = pad_time(&mel, 2, 3, 3);
    assert_eq!(padded, mel);
}

#[test]
fn test_trim_time_keeps_leading_frames_per_channel() {
    // 2 channels, t_padded=4, t_keep=2: each row [a,b,c,d] -> [a,b].
    let out = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let trimmed = trim_time(&out, 2, 4, 2);
    assert_eq!(trimmed, vec![1.0, 2.0, 5.0, 6.0]);
    assert_eq!(trimmed.len(), 2 * 2);
}

#[test]
fn test_pad_then_trim_roundtrip_recovers_prefix() {
    // pad up then trim back to the original length recovers the real frames.
    let mel = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]; // 2 ch x 3
    let padded = pad_time(&mel, 2, 3, 5);
    let back = trim_time(&padded, 2, 5, 3);
    assert_eq!(back, mel);
}

/// File-mode smoke (no model / no ANE hardware): an `AneEncoderSession` with
/// no compiled bucket models routes a streaming-sized window through the ort
/// fallback on the file-mode path (`run` / [`FILL_FLOOR`]), because 250 mel
/// frames underfill bucket 512 at the 0.5 floor.
///
/// Pins that routing without a real `.mlpackage` by using a mock fallback
/// session and asserting it was invoked.
#[test]
fn test_file_mode_underfilled_window_routes_to_ort_fallback() {
    use crate::runtime::mock::MockSession;

    // Production buckets; none has a compiled model loaded here.
    const SHIPPED_BUCKETS: &[usize] = &[512, 768, 1536, 3000];
    // A streaming-sized window: 250 mel frames (the 2.5 s window cap).
    const T: usize = 250;

    // Sanity: file-mode floor rejects; streaming floor would accept 512.
    assert_eq!(
        select_bucket(T, SHIPPED_BUCKETS, FILL_FLOOR),
        None,
        "a 250-frame window must not select any shipped bucket at FILL_FLOOR"
    );
    assert_eq!(
        select_bucket(T, SHIPPED_BUCKETS, STREAMING_FILL_FLOOR),
        Some(512)
    );

    // Mock ort encoder fallback: accepts the [1,64,T] mel + [1] length pair
    // and returns the encoder's two-output contract ([encoded], [encoded_len]).
    let t_prime = calc_output_length(T);
    let fallback = MockSession::new(
        vec![Shape::new(vec![1, N_MELS, T]), Shape::new(vec![1])],
        vec![
            Tensor::new(
                Shape::new(vec![1, ENC_DIM, t_prime]),
                TensorData::F32(vec![0.0; ENC_DIM * t_prime]),
            )
            .unwrap(),
            Tensor::new(Shape::new(vec![1]), TensorData::I64(vec![t_prime as i64])).unwrap(),
        ],
    );

    // No bucket models -> every window falls back (file-mode or streaming).
    let session = AneEncoderSession::new(Vec::new(), Box::new(fallback));
    assert!(session.is_ane_encoder());

    let mel = Tensor::new(
        Shape::new(vec![1, N_MELS, T]),
        TensorData::F32(vec![0.0; N_MELS * T]),
    )
    .unwrap();
    let len = Tensor::new(Shape::new(vec![1]), TensorData::I64(vec![T as i64])).unwrap();
    let out = session.run(&[mel, len]).expect("fallback run succeeds");

    // The fallback's recorded contract flows straight through.
    assert_eq!(out.len(), 2, "encoder emits [encoded, encoded_len]");
    assert_eq!(out[0].shape().dims(), &[1, ENC_DIM, t_prime]);
    match out[1].view().data() {
        crate::runtime::tensor::TensorDataView::I64(v) => {
            assert_eq!(v[0], t_prime as i64, "fallback encoded_len passes through")
        }
        other => panic!("expected I64 encoded_len, got {other:?}"),
    }
}
