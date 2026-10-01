# Model-content hashing and startup

Every engine load hashes the encoder before selecting its optimized ORT graph.
This preserves correctness when weights change without changing the file name,
length, or timestamp. Pool slots share the digest for that load; there is no
process-global or filesystem-metadata shortcut. Cold graph publication rechecks
source content before its atomic rename.

The former portable, in-tree SHA-256 loop and byte-by-byte external-data scan
added substantial debug startup cost for the 225,250,603-byte default encoder.
Its INT8 filename is outside the older FP32 checksum table, so it takes the
actual-content hashing path even when installed from the published bundle.
The graph cache itself was hitting normally. This overhead lengthened the
model E2E suite enough to exhaust its existing 45-minute budget.

SHA-256 now uses RustCrypto `sha2` with default features disabled. The exact
`location` byte marker uses `memchr::memmem::find`; the seven-byte overlap across
read buffers is unchanged. Possible external tensor data still disables caching.
The accepted model contents are unchanged: local quantization can produce a
canonical INT8 filename with weights different from the published bundle.
No expected digest is substituted for hashing the actual file. Content identity
is distinct from publisher authentication: only files covered by the existing
startup checksum table must match its pins. Published INT8 downloads are verified
when installed, while subsequent local RNN-T INT8 replacements remain supported. CTC files
covered by the startup checksum table still require their pinned contents.

## Measurements

Local Linux x86_64, AMD Ryzen AI 9 HX 370 (SHA-NI), Rust 1.98.1,
ORT `2.0.0-rc.13`, default core features, CPU pool size one. The source before
this change was `de86a157`; both engine measurements were rebuilt from source,
used the same installed model and warm graph cache, and ran without concurrent
compilation by the measuring process. Other host activity was not controlled.
The model SHA-256 was
`c52665e9d96c4ca3a153c063d2ee9af6c567fe2975ca50fd038b75bbf2f60e7f`.

| Warm engine load | Four successive samples (seconds) |
|---|---|
| Before, debug | 8.057, 8.071, 8.057, 8.305 |
| After, debug | 0.372, 0.374, 0.358, 0.354 |

The timed call was `Engine::load_with_pool_size(model_dir, 1)`; engines were
dropped between samples. Model download, compilation, and transcription were
outside the timing. These are startup measurements on one host, not inference
RTF improvements or universal latency guarantees.

An isolated file-hashing probe with three samples measured the original full
hash plus marker scan at 6.160–6.173 seconds in debug and 0.613–0.615 seconds in
release. The candidate accelerated implementation measured about 0.112 seconds
in debug and 0.105 seconds in release; complete digests matched. The preliminary
probe used `sha2` 0.11.0 and `memchr` 2.8.3; the repository retains its existing
`memchr` 2.8.0 lock entry, used by the engine measurements above.

## Dependencies and build profiles

The workspace lock adds only `sha2` 0.11.0; unrelated locked versions stay fixed.
`memchr` already belongs to the minimal core dependency closure. The lean core
normal/build closure additionally includes these seven packages:

- `sha2` 0.11.0
- `digest` 0.11.3
- `block-buffer` 0.12.0
- `crypto-common` 0.2.1
- `hybrid-array` 0.4.11
- `cpufeatures` 0.3.0
- `typenum` 1.20.0

The six transitive packages already existed elsewhere in the workspace lock.
This avoids adding a native C/assembly build requirement to lean consumers.
[Upstream SHA-2 documentation](https://docs.rs/sha2/0.11.0/sha2/#backends)
describes runtime x86/ARM feature detection and the portable fallback.

Workspace dev/test builds optimize only `sha2` and `memchr`: unoptimized CPU
intrinsics otherwise retain significant debug overhead. Release builds already
optimize them. Cargo profiles belong to the root workspace, so consumers using
this library in another workspace choose their own debug optimization policy.
NIST vectors, ragged streaming boundaries, external-marker read boundaries,
and same-size/same-timestamp replacement tests remain mandatory.
