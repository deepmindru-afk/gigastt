# gigastt Android (AAR)

Android library for [gigastt](https://github.com/ekhodzitsky/gigastt) —
on-device Russian speech-to-text (GigaAM v3) — via the UniFFI Kotlin bindings.

> **Status: experimental.** The Rust cross-build is proven (CI cross-compiles the
> native library via cargo-ndk), but the Gradle/Maven AAR assembly and publish
> have not yet been validated end-to-end on a real Android toolchain. Verify with
> a local Android SDK/NDK before relying on a published artifact.

## What the AAR contains

- `jniLibs/<abi>/libgigastt_uniffi.so` + `libonnxruntime.so` for `arm64-v8a`,
  `armeabi-v7a`, `x86_64`. onnxruntime is dynamically linked: each ABI folder
  carries its own `libonnxruntime.so` (from the official Microsoft
  onnxruntime-android AAR), which Android's dynamic linker resolves from the
  same `jniLibs/<abi>/` directory at load time.
- The UniFFI-generated Kotlin bindings (idiomatic `Engine` / `Stream` + typed
  exceptions).
- A JNA dependency (`net.java.dev.jna:jna@aar`) — UniFFI Kotlin calls the native
  library through JNA.

The ~215 MB INT8 model is **not** bundled; side-load it at runtime (ship the
model directory with the app or download it) and pass its path to `Engine`.

## Build

The native libs + Kotlin are generated before assembling (not committed):

```sh
# Per-ABI native libs. A single `cargo ndk -t arm64-v8a -t armeabi-v7a -t x86_64`
# call does NOT work: ort's pyke prebuilts cover only aarch64-linux-android, so
# the other two ABIs have no prebuilt. Instead (mirrors android-aar.yml):
# 1. Fetch the official Microsoft onnxruntime-android AAR and unpack one
#    libonnxruntime.so per ABI:
ORT_VER=1.24.2
curl -fsSL -o ort-android.aar \
  "https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/${ORT_VER}/onnxruntime-android-${ORT_VER}.aar"
for abi in arm64-v8a armeabi-v7a x86_64; do
  mkdir -p "ort-lib/$abi"
  unzip -p ort-android.aar "jni/$abi/libonnxruntime.so" > "ort-lib/$abi/libonnxruntime.so"
done
# 2. Build each ABI separately with ORT_LIB_LOCATION pointing at its own
#    onnxruntime .so (dynamic link), then copy the .so next to our cdylib:
export ORT_PREFER_DYNAMIC_LINK=1
for abi in arm64-v8a armeabi-v7a x86_64; do
  ORT_LIB_LOCATION="$PWD/ort-lib/$abi" \
  cargo ndk -t "$abi" \
    -o packaging/android/gigastt/src/main/jniLibs build --release -p gigastt-uniffi
  cp "ort-lib/$abi/libonnxruntime.so" \
     "packaging/android/gigastt/src/main/jniLibs/$abi/"
done
# Kotlin bindings (from a host build of the cdylib; metadata is arch-independent)
cargo build --release -p gigastt-uniffi
cargo run --release -p gigastt-uniffi --bin uniffi-bindgen -- generate \
  --library target/release/libgigastt_uniffi.* --language kotlin \
  --out-dir packaging/android/gigastt/src/main/kotlin
# assemble
cd packaging/android && gradle :gigastt:assembleRelease
```

CI: `.github/workflows/android-aar.yml` (`workflow_dispatch`) runs the same
per-ABI flow above (fetch the onnxruntime-android AAR, build each ABI with
`ORT_LIB_LOCATION`, copy `libonnxruntime.so` into each `jniLibs/<abi>/`) and,
with `publish: true` + Maven credentials, publishes the AAR.

## Usage

```kotlin
val engine = Engine("/path/to/models")     // side-loaded model dir
val t = engine.transcribeFile("recording.wav")
println(t.text)
```

## License

MIT.
