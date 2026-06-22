# Runtime Abstraction Layer Over `ort`

**Status:** Design approved, awaiting spec review  
**Date:** 2026-06-22  
**Author:** AI design session  
**Scope:** `crates/gigastt-core` inference engine

## 1. Context and Problem

`gigastt-core` currently depends directly on `ort = "2.0.0-rc.12"`. `ort` types (`Session`, `Value`, `RunOptions`, etc.) leak into `Engine`, `SessionPool`, and the RNN-T decode loop. Because `ort` 2.0 is still a release candidate, this creates several risks:

- **API churn:** a future `ort` release may change the types we depend on, forcing wide-spread edits.
- **Vendor lock-in:** replacing `ort` with another backend (custom bindings, a different runtime, etc.) would require rewriting the inference engine.
- **Testability:** unit tests for `Engine`, `SessionPool`, and decode logic must either use real ONNX models (~850 MB) or avoid the code paths that touch `ort`.
- **Workaround scattering:** `ort`-specific quirks (e.g., `Send`/`Sync` limitations, error wrapping) are spread across the codebase.

This design introduces a small, internal runtime abstraction layer. `ort` becomes an implementation detail of a single adapter module.

## 2. Goals

1. **Isolate `ort`:** no `ort` type is imported outside `runtime/ort/`.
2. **Preserve behavior:** all existing unit, e2e, WER, load, and soak tests pass unchanged.
3. **Enable mock testing:** provide a `MockRuntime` so `Engine`, `SessionPool`, and decode logic can be unit-tested without ONNX models.
4. **Keep all execution providers:** CPU, CoreML, CUDA, and NNAPI must continue to work.
5. **Leave the door open:** the internal API should be clean enough to become public later or to host a custom backend.

## 3. Non-Goals

1. **Reimplement ONNX:** we do not build a generic tensor library, allocator abstraction, or full ONNX operator set. We abstract only what `gigastt` actually uses.
2. **Public API change:** the abstraction stays internal in this phase. `Engine` public API remains the same.
3. **Async runtime API:** `RuntimeSession::run` stays synchronous; async scheduling remains the responsibility of `Engine`/`SessionPool` via `spawn_blocking`.
4. **Replace feature flags:** existing `--features coreml` / `cuda` / `nnapi` / `diarization` are retained.

## 4. Design Overview

Introduce a `runtime` module in `gigastt-core` with:

- **Core traits:** `Runtime`, `RuntimeSession`, `RuntimeFactory`.
- **Value types:** `Tensor`, `TensorView`, `Shape`, `ElementType`.
- **Error type:** `RuntimeError`, converted to `GigasttError` at the boundary.
- **Ort adapter:** `OrtRuntime`, `OrtSession`, `OrtTensor`, and provider-specific `OrtFactory`.
- **Mock adapter:** `MockRuntime`, `MockSession` for tests.

`Engine` is refactored to hold a `Box<dyn Runtime>` and `SessionTriplet` of `Box<dyn RuntimeSession>`. The rest of the inference code operates on `Tensor` instead of `ort::Value`.

## 5. Module Structure

```
crates/gigastt-core/src/
  runtime/
    mod.rs              # public (internal) exports: traits + value types
    tensor.rs           # Tensor, TensorView, Shape, ElementType
    error.rs            # RuntimeError
    factory.rs          # RuntimeFactory trait + provider selection helpers
    session.rs          # RuntimeSession trait
    ort/                # ort adapter (depends on `ort` crate)
      mod.rs
      session.rs
      tensor.rs
      factory.rs
    mock/               # mock adapter for tests
      mod.rs
      session.rs
      tensor.rs
```

## 6. Core API

### 6.1 Traits

```rust
/// Creates a `Runtime` configured for a specific execution provider.
pub trait RuntimeFactory: Send + Sync + 'static {
    fn create(&self, intra_threads: usize) -> Result<Box<dyn Runtime>, RuntimeError>;
}

/// Owns loaded sessions. One runtime per `Engine`.
pub trait Runtime: Send + Sync + 'static {
    fn load_session(
        &self,
        model_path: &std::path::Path,
    ) -> Result<Box<dyn RuntimeSession>, RuntimeError>;
}

/// One ONNX session (encoder, decoder, or joiner).
pub trait RuntimeSession: Send + Sync + 'static {
    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, RuntimeError>;
}
```

### 6.2 Tensor

```rust
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    shape: Shape,
    data: TensorData,
}

pub struct TensorView<'a> {
    shape: Shape,
    data: TensorDataView<'a>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TensorData {
    F32(Vec<f32>),
    I32(Vec<i32>),
    I64(Vec<i64>),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TensorDataView<'a> {
    F32(&'a [f32]),
    I32(&'a [i32]),
    I64(&'a [i64]),
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: Vec<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElementType {
    F32,
    I32,
    I64,
}
```

- `Tensor` is owned and cheaply cloneable (data is `Vec`).
- `TensorView` is a zero-copy borrow used for reading outputs without moving data.
- `Shape` normalizes dimensions so callers do not depend on `ort` shape types.

## 7. Ort Adapter

### 7.1 OrtRuntime

```rust
pub struct OrtRuntime {
    environment: ort::Environment,
    intra_threads: usize,
    provider: OrtExecutionProvider,
}

impl Runtime for OrtRuntime {
    fn load_session(
        &self,
        model_path: &std::path::Path,
    ) -> Result<Box<dyn RuntimeSession>, RuntimeError> {
        let session = self.environment
            .new_session_builder()?
            .with_intra_threads(self.intra_threads)?
            .with_execution_provider(self.provider.to_ort())?
            .commit_from_file(model_path)?;
        Ok(Box::new(OrtSession { session }))
    }
}
```

### 7.2 OrtSession

```rust
pub struct OrtSession {
    session: ort::Session,
}

impl RuntimeSession for OrtSession {
    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, RuntimeError> {
        let ort_inputs: Vec<ort::Value> = inputs
            .into_iter()
            .map(Tensor::into_ort_value)
            .collect::<Result<_, _>>()?;
        let outputs = self.session.run(ort_inputs).map_err(wrap_ort_error)?;
        outputs
            .into_iter()
            .map(OrtTensor::try_from)
            .collect()
    }
}
```

### 7.3 Tensor Conversion

Conversion between `Tensor` and `ort::Value` lives entirely in `runtime/ort/tensor.rs`. All `ort` memory and shape quirks are encapsulated here.

## 8. Provider Factories

Existing feature flags are preserved. Each flag selects a factory internally.

```rust
pub fn default_factory(_intra_threads: usize) -> Box<dyn RuntimeFactory> {
    #[cfg(feature = "coreml")]
    return Box::new(OrtFactory::coreml());
    #[cfg(feature = "cuda")]
    return Box::new(OrtFactory::cuda());
    #[cfg(feature = "nnapi")]
    return Box::new(OrtFactory::nnapi());
    #[cfg(not(any(feature = "coreml", feature = "cuda", feature = "nnapi")))]
    Box::new(OrtFactory::cpu())
}
```

`OrtFactory` holds only provider configuration; `intra_threads` is supplied by `Engine` at runtime:

```rust
pub struct OrtFactory {
    provider: OrtExecutionProvider,
}

impl RuntimeFactory for OrtFactory {
    fn create(&self, intra_threads: usize) -> Result<Box<dyn Runtime>, RuntimeError> {
        let environment = ort::Environment::builder()
            .with_name("gigastt")
            .build()?;
        Ok(Box::new(OrtRuntime {
            environment,
            intra_threads,
            provider: self.provider.clone(),
        }))
    }
}
```

## 9. Mock Runtime

```rust
pub struct MockFactory {
    sessions: HashMap<String, Arc<MockSession>>,
}

impl RuntimeFactory for MockFactory {
    fn create(&self, _intra_threads: usize) -> Result<Box<dyn Runtime>, RuntimeError> {
        Ok(Box::new(MockRuntime {
            sessions: self.sessions.clone(),
        }))
    }
}

pub struct MockRuntime {
    sessions: HashMap<String, Arc<MockSession>>,
}

impl Runtime for MockRuntime {
    fn load_session(
        &self,
        model_path: &std::path::Path,
    ) -> Result<Box<dyn RuntimeSession>, RuntimeError> {
        let key = model_path.file_stem()
            .ok_or_else(|| RuntimeError::LoadFailed("empty path".into()))?
            .to_string_lossy()
            .to_string();
        let session = self.sessions
            .get(&key)
            .ok_or_else(|| RuntimeError::LoadFailed(format!("no mock for {key}")))?
            .clone();
        Ok(Box::new((*session).clone()))
    }
}

#[derive(Clone)]
pub struct MockSession {
    expected_inputs: Vec<Shape>,
    outputs: Vec<Tensor>,
    call_count: AtomicUsize,
}

impl RuntimeSession for MockSession {
    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, RuntimeError> {
        // Validate input shapes match expectations.
        for (actual, expected) in inputs.iter().zip(self.expected_inputs.iter()) {
            if actual.shape() != expected {
                return Err(RuntimeError::InvalidShape {
                    expected: expected.clone(),
                    got: actual.shape().clone(),
                });
            }
        }
        self.call_count.fetch_add(1, Ordering::Relaxed);
        Ok(self.outputs.clone())
    }
}
```

Mock sessions return pre-recorded tensors, allowing unit tests for `Engine`, `SessionPool`, streaming state machine, and decoder logic without model files.

## 10. Integration with Engine and SessionPool

### 10.1 SessionTriplet

Current:

```rust
struct SessionTriplet {
    encoder: ort::Session,
    decoder: ort::Session,
    joiner: ort::Session,
}
```

New:

```rust
struct SessionTriplet {
    encoder: Box<dyn RuntimeSession>,
    decoder: Box<dyn RuntimeSession>,
    joiner: Box<dyn RuntimeSession>,
}
```

### 10.2 Engine Load

`Engine::load_with_pools_threads` accepts a `Box<dyn RuntimeFactory>` instead of constructing `ort` sessions directly:

```rust
pub fn load_with_pools_threads(
    model_dir: &Path,
    pool_size: usize,
    pool_min_size: usize,
    batch_pool_size: usize,
    intra_threads: usize,
) -> Result<Self> {
    let factory = default_factory(intra_threads);
    Self::load_with_factory(
        model_dir,
        pool_size,
        pool_min_size,
        batch_pool_size,
        factory,
    )
}

pub(crate) fn load_with_factory(
    model_dir: &Path,
    pool_size: usize,
    pool_min_size: usize,
    batch_pool_size: usize,
    factory: Box<dyn RuntimeFactory>,
) -> Result<Self> {
    // ... spawn_blocking, load sessions via factory, build pools
}
```

A package-private `load_with_factory` is used by tests to inject `MockFactory`.

### 10.3 Decode Loop

`decode.rs` currently operates on raw float slices from `ort::Value`. It is updated to accept `TensorView`:

```rust
fn decode_step(
    encoder_out: &TensorView,
    decoder_state: &mut DecoderState,
) -> Result<Option<TokenId>, RuntimeError>;
```

This removes the last `ort` dependency from the decode path.

## 11. Error Handling

```rust
#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("failed to load model {path}: {message}")]
    LoadFailed { path: PathBuf, message: String },

    #[error("inference failed: {0}")]
    InferenceFailed(String),

    #[error("invalid tensor shape: expected {expected:?}, got {got:?}")]
    InvalidShape { expected: Shape, got: Shape },

    #[error("unsupported element type")]
    UnsupportedElementType,

    #[error("invalid input count: expected {expected}, got {got}")]
    InvalidInputCount { expected: usize, got: usize },
}
```

`runtime/ort/` maps `ort::Error` to `RuntimeError::InferenceFailed`. `GigasttError` gains a `Runtime` variant or maps `RuntimeError` to existing `Inference`/`ModelLoad` variants. Internal errors are sanitized before reaching clients, preserving current behavior.

## 12. Testing Strategy

1. **Adapter tests:** verify `Tensor` <-> `ort::Value` round-trip in `runtime/ort/tensor.rs`.
2. **Mock tests:** build an `Engine` with `MockFactory` and assert decode output.
3. **Parity tests:** run existing e2e, WER, load, and soak tests with the `ort` adapter unchanged.
4. **Isolation test:** a CI lint job ensures no `ort` import exists outside `runtime/ort/`.

## 13. Migration Plan

We deliver the refactor in small PRs to keep CI green and reviews focused.

### PR 1: Runtime module skeleton
- Create `runtime/mod.rs`, `runtime/tensor.rs`, `runtime/error.rs`, `runtime/factory.rs`, `runtime/session.rs`.
- Add `Tensor`, `Shape`, `ElementType`, `RuntimeError`, `RuntimeFactory`, `Runtime`, `RuntimeSession`.
- Add unit tests for tensor shape helpers and error conversions.

### PR 2: Ort adapter
- Implement `runtime/ort/` adapter.
- Add `OrtRuntime`, `OrtSession`, `OrtTensor`, `OrtFactory`, provider factories.
- Verify `cargo check --features coreml/cuda/diarization` passes.

### PR 3: Integrate Engine and SessionPool
- Replace `ort::Session` in `SessionTriplet` with `Box<dyn RuntimeSession>`.
- Add `Engine::load_with_factory`.
- Update `decode.rs` to use `TensorView`.
- Ensure all existing tests pass.

### PR 4: Mock runtime and new tests
- Implement `runtime/mock/`.
- Rewrite selected unit tests to use `MockFactory`.
- Add CI check that `ort` is not imported outside `runtime/ort/`.

## 14. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Performance regression from trait objects | Benchmark with Criterion before/after; if overhead is measurable, use `Box<dyn RuntimeSession>` only at pool boundaries and keep hot path monomorphized. |
| `ort` `Send`/`Sync` quirks | Isolate in adapter; use `+ Send + Sync` bounds on traits. |
| Memory layout differences between `ort::Value` and `Tensor` | Centralize conversion code and add round-trip tests for all tensor types used by the models. |
| Large PR hard to review | Split into 4 PRs as described in Migration Plan. |
| Behavior drift | Require all e2e/WER/load/soak tests to pass unchanged on each PR. |

## 15. Open Questions

1. Should `Tensor` use `Arc<[f32]>` instead of `Vec<f32>` to make clones cheaper when the same tensor is passed to multiple consumers?
2. Should we expose `RuntimeFactory` as a public API under an unstable feature flag?

These questions will be resolved during implementation based on concrete usage in `Engine`.

## 16. Decision Log

| Decision | Choice | Rationale |
|---|---|---|
| Abstraction depth | Session + tensor level | Deep enough to swap backend, shallow enough to avoid reimplementing ONNX. |
| Sync/async boundary | Runtime sync, Engine async | ONNX is blocking; async wrapper stays where it already is. |
| API visibility | Internal first | Avoid backward-compatibility commitments until design is battle-tested. |
| Feature flags | Keep existing | No user-facing change; providers map to factories internally. |
| Execution providers | All from day one | Preserve parity with current `coreml`/`cuda`/`nnapi` support. |
| Mock runtime | First-class | One of the primary goals is testability without models. |
