use super::*;

/// The loop `SliceWindows` replaced, verbatim, as the reference oracle.
fn legacy_windows(total: usize, window: usize, stride: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut start = 0usize;
    while start < total {
        let end = (start + window).min(total);
        out.push((start, end - start));
        if end == total {
            break;
        }
        start += stride;
    }
    out
}

fn observed(total: usize, spec: WindowSpec) -> Vec<(usize, usize)> {
    let samples = vec![0.0f32; total];
    let mut src = SliceWindows::new(&samples, spec);
    let mut out = Vec::new();
    while let Some(w) = src.next_window().expect("slice source never fails") {
        out.push((w.start_sample, w.samples.len()));
    }
    out
}

/// The ort long-form geometry, spelled out so this test pins the numbers the
/// engine feeds rather than following it.
fn ort_spec() -> WindowSpec {
    WindowSpec::new(16000 * 30, 16000 * 24, 16000 * 2)
}

#[test]
fn test_window_spec_stride_is_frame_aligned() {
    let spec = ort_spec();
    assert_eq!(spec.window(), 384_000);
    assert_eq!(spec.stride(), 352_000);
    assert_eq!(spec.stride() % FRAME_SAMPLES, 0);
    // 2 s overlap is already frame-aligned, so it survives the alignment.
    assert_eq!(spec.overlap(), 32_000);
    // ANE geometry: 30 s window, same 2 s overlap.
    let ane = WindowSpec::new(16000 * 30, 16000 * 30, 16000 * 2);
    assert_eq!(ane.stride(), 448_000);
    assert_eq!(ane.stride() % FRAME_SAMPLES, 0);
    assert_eq!(ane.overlap(), 32_000);
}

#[test]
fn test_window_spec_single_pass_boundary() {
    let spec = ort_spec();
    assert!(spec.is_single_pass(0));
    assert!(spec.is_single_pass(479_999));
    assert!(spec.is_single_pass(480_000)); // exactly 30 s stays single-pass
    assert!(!spec.is_single_pass(480_001));
}

#[test]
fn test_window_spec_degenerate_overlap_still_advances() {
    // overlap >= window would give a zero stride and a non-advancing source.
    let spec = WindowSpec::new(0, 1000, 4000);
    assert_eq!(spec.stride(), FRAME_SAMPLES);
    assert_eq!(observed(5000, spec).len(), 8);
}

#[test]
fn test_slice_windows_matches_legacy_loop_swept() {
    let spec = ort_spec();
    let (window, stride) = (spec.window(), spec.stride());
    let mut lengths: Vec<usize> = Vec::new();
    // Coarse sweep across 0..3x window.
    let mut n = 0usize;
    while n <= 3 * window {
        lengths.push(n);
        n += 4_001; // deliberately coprime with the stride/frame grid
    }
    // Exact boundaries: window/stride multiples ± 1, the single-pass
    // threshold, and the degenerate sub-frame tail band.
    for anchor in [
        0,
        1,
        FRAME_SAMPLES,
        window,
        stride,
        stride + window,
        2 * stride,
        2 * stride + window,
        480_000, // single-pass branch boundary
    ] {
        for d in [-1isize, 0, 1] {
            let v = anchor as isize + d;
            if v >= 0 {
                lengths.push(v as usize);
            }
        }
    }
    lengths.extend(704_000..=704_320); // degenerate band, every length
    lengths.sort_unstable();
    lengths.dedup();

    for total in lengths {
        assert_eq!(
            observed(total, spec),
            legacy_windows(total, window, stride),
            "window sequence diverged at total={total}"
        );
    }
}

#[test]
fn test_slice_windows_empty_yields_nothing() {
    assert!(observed(0, ort_spec()).is_empty());
}

#[test]
fn test_slice_windows_stop_exactly_at_the_end() {
    let spec = ort_spec();
    let total = 1_440_000; // 90 s
    let seq = observed(total, spec);
    assert!(!seq.is_empty());
    // Exactly one window reaches the end, and it is the last one emitted.
    assert_eq!(
        seq.iter()
            .filter(|(start, len)| start + len == total)
            .count(),
        1
    );
    let (start, len) = seq[seq.len() - 1];
    assert_eq!(start + len, total);
    // Every start is frame-aligned, so the frame offset is integral.
    for (start, _) in &seq {
        assert_eq!(start % FRAME_SAMPLES, 0);
    }
}
