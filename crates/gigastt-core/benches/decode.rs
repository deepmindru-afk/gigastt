//! Criterion micro-benchmark for WAVE / telephony decode (`ryf` path).
//!
//! Uses in-tree fixtures under `tests/fixtures/telephony/` (3 s @ 16 kHz)
//! plus G.711 WAVs built from the PCM fixture. No model required.
//!
//! ```sh
//! cargo bench -p gigastt-core --bench decode
//! ```

use audio_codec::Encoder;
use bytes::Bytes;
use criterion::{Criterion, criterion_group, criterion_main};
use gigastt_core::inference::audio::{
    TelephonyCodec, decode_audio_bytes_shared, decode_telephony_raw,
};
use std::hint::black_box;

const TONE_SRC: &[u8] = include_bytes!("../tests/fixtures/telephony/tone_src.wav");
const G722_WAV: &[u8] = include_bytes!("../tests/fixtures/telephony/g722_tone.wav");

fn find_riff_chunk<'a>(data: &'a [u8], want: &[u8; 4]) -> &'a [u8] {
    let mut pos = 12usize;
    while pos + 8 <= data.len() {
        let id = &data[pos..pos + 4];
        let size = u32::from_le_bytes(
            data[pos + 4..pos + 8]
                .try_into()
                .expect("fmt size is 4 bytes"),
        ) as usize;
        let start = pos + 8;
        let end = start.saturating_add(size).min(data.len());
        if id == want {
            return &data[start..end];
        }
        pos = start.saturating_add(size).saturating_add(size & 1);
    }
    panic!(
        "missing {} chunk",
        std::str::from_utf8(want).unwrap_or("????")
    );
}

fn pcm16_from_wav(wav: &[u8]) -> Vec<i16> {
    find_riff_chunk(wav, b"data")
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| i16::from_le_bytes(*c))
        .collect()
}

fn encode_into<E: Encoder>(encoder: &mut E, samples: &[i16]) -> Vec<u8> {
    let mut out = vec![0u8; encoder.max_encode_bytes(samples.len())];
    let n = encoder
        .encode_into(samples, &mut out)
        .unwrap_or_else(|e| panic!("encode: {e}"));
    out.truncate(n);
    out
}

fn compressed_wav(tag: u16, sample_rate: u32, byte_rate: u32, payload: &[u8]) -> Vec<u8> {
    let data_size = payload.len() as u32;
    let mut buf = Vec::with_capacity(46 + payload.len());
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&(38 + data_size).to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&18u32.to_le_bytes());
    buf.extend_from_slice(&tag.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes());
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes());
    buf.extend_from_slice(&8u16.to_le_bytes());
    buf.extend_from_slice(&0u16.to_le_bytes());
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());
    buf.extend_from_slice(payload);
    buf
}

fn bench_decode(c: &mut Criterion) {
    let pcm16 = pcm16_from_wav(TONE_SRC);
    // Naive 16 kHz → 8 kHz for a realistic G.711 upsample path (8 k → 16 k).
    let pcm8k: Vec<i16> = pcm16.iter().copied().step_by(2).collect();
    let alaw = encode_into(&mut audio_codec::pcma::PcmaEncoder::new(), &pcm8k);
    let ulaw = encode_into(&mut audio_codec::pcmu::PcmuEncoder::new(), &pcm8k);
    let alaw_wav = compressed_wav(0x0006, 8_000, 8_000, &alaw);
    let ulaw_wav = compressed_wav(0x0007, 8_000, 8_000, &ulaw);
    let g722_payload = find_riff_chunk(G722_WAV, b"data").to_vec();

    let pcm_bytes = Bytes::from_static(TONE_SRC);
    let g722_bytes = Bytes::from_static(G722_WAV);
    let alaw_bytes = Bytes::from(alaw_wav);
    let ulaw_bytes = Bytes::from(ulaw_wav);

    // Sanity: one decode each so a broken fixture fails before timing.
    for (name, buf) in [
        ("pcm", &pcm_bytes),
        ("g722", &g722_bytes),
        ("alaw", &alaw_bytes),
        ("ulaw", &ulaw_bytes),
    ] {
        let n = decode_audio_bytes_shared(buf.clone()).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(!n.is_empty(), "{name} decoded empty");
    }

    let mut group = c.benchmark_group("decode_audio");
    group.bench_function("pcm16_wav_3s", |b| {
        b.iter(|| {
            black_box(decode_audio_bytes_shared(black_box(pcm_bytes.clone())).unwrap());
        });
    });
    group.bench_function("g722_wav_3s", |b| {
        b.iter(|| {
            black_box(decode_audio_bytes_shared(black_box(g722_bytes.clone())).unwrap());
        });
    });
    group.bench_function("g711_alaw_wav_8k", |b| {
        b.iter(|| {
            black_box(decode_audio_bytes_shared(black_box(alaw_bytes.clone())).unwrap());
        });
    });
    group.bench_function("g711_mulaw_wav_8k", |b| {
        b.iter(|| {
            black_box(decode_audio_bytes_shared(black_box(ulaw_bytes.clone())).unwrap());
        });
    });
    group.finish();

    let mut group = c.benchmark_group("decode_telephony_raw");
    group.bench_function("raw_g722_3s", |b| {
        b.iter(|| {
            black_box(
                decode_telephony_raw(black_box(&g722_payload), TelephonyCodec::G722, 8_000)
                    .unwrap(),
            );
        });
    });
    group.bench_function("raw_pcmu_8k", |b| {
        b.iter(|| {
            black_box(decode_telephony_raw(black_box(&ulaw), TelephonyCodec::Pcmu, 8_000).unwrap());
        });
    });
    group.finish();
}

criterion_group!(benches, bench_decode);
criterion_main!(benches);
