# Runtime Abstraction Layer Over `ort` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce an internal runtime abstraction layer in `gigastt-core` so that `ort` types are isolated to a single adapter module, enabling mock-based unit tests and future backend swaps while preserving all existing behavior.

**Architecture:** Define core traits (`Runtime`, `RuntimeSession`, `RuntimeFactory`) and value types (`Tensor`, `Shape`, `RuntimeError`) in a new `runtime` module. Provide an `ort` adapter that implements these traits using the existing `ort` crate, and a `mock` adapter for tests. Refactor `Engine`/`SessionPool` to use `Box<dyn RuntimeSession>` instead of `ort::Session` directly. Deliver in 4 sequential PRs.

**Tech Stack:** Rust 2024, `ort` 2.0.0-rc.12, `thiserror`, `anyhow`, `tokio`.

---

## File Structure

### New files

| File | Responsibility |
|---|---|
| `crates/gigastt-core/src/runtime/mod.rs` | Public (internal) exports: traits and value types. |
| `crates/gigastt-core/src/runtime/tensor.rs` | `Tensor`, `TensorView`, `Shape`, `ElementType`. |
| `crates/gigastt-core/src/runtime/error.rs` | `RuntimeError` enum + conversions. |
| `crates/gigastt-core/src/runtime/factory.rs` | `RuntimeFactory` trait + provider selection helpers. |
| `crates/gigastt-core/src/runtime/session.rs` | `RuntimeSession` trait. |
| `crates/gigastt-core/src/runtime/ort/mod.rs` | Ort adapter public exports. |
| `crates/gigastt-core/src/runtime/ort/session.rs` | `OrtRuntime`, `OrtSession`. |
| `crates/gigastt-core/src/runtime/ort/tensor.rs` | `Tensor` <-> `ort::Value` conversions. |
| `crates/gigastt-core/src/runtime/ort/factory.rs` | `OrtFactory` + provider factories. |
| `crates/gigastt-core/src/runtime/mock/mod.rs` | Mock adapter public exports. |
| `crates/gigastt-core/src/runtime/mock/session.rs` | `MockRuntime`, `MockSession`, `MockFactory`. |

### Modified files

| File | Change |
|---|---|
| `crates/gigastt-core/src/lib.rs` | Add `pub(crate) mod runtime;` (internal only in this phase). |
| `crates/gigastt-core/src/inference/mod.rs` | Replace `ort::Session` in `SessionTriplet` with `Box<dyn RuntimeSession>`; add `load_with_factory`. |
| `crates/gigastt-core/src/inference/decode.rs` | Accept `TensorView` instead of raw `ort::Value` slices. |
| `crates/gigastt-core/src/error.rs` | Add `From<RuntimeError> for GigasttError`. |

---

## Phase 1: Runtime Module Skeleton

**Dependency:** None. Can start immediately.

### Task 1.1: Create `runtime/tensor.rs`

**Files:**
- Create: `crates/gigastt-core/src/runtime/tensor.rs`

- [ ] **Step 1: Write the types**

```rust
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    shape: Shape,
    data: TensorData,
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

impl<'a> TensorDataView<'a> {
    pub fn as_f32(&self) -> Option<&'a [f32]> {
        match self {
            TensorDataView::F32(v) => Some(v),
            _ => None,
        }
    }
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

impl Tensor {
    pub fn new(shape: Shape, data: TensorData) -> Self {
        // Verify data length matches shape product.
        let expected = shape.elements();
        let actual = data.len();
        assert_eq!(expected, actual, "tensor data length mismatch: expected {expected}, got {actual}");
        Self { shape, data }
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn element_type(&self) -> ElementType {
        match &self.data {
            TensorData::F32(_) => ElementType::F32,
            TensorData::I32(_) => ElementType::I32,
            TensorData::I64(_) => ElementType::I64,
        }
    }

    pub fn view(&self) -> TensorView<'_> {
        TensorView {
            shape: self.shape.clone(),
            data: match &self.data {
                TensorData::F32(v) => TensorDataView::F32(v.as_slice()),
                TensorData::I32(v) => TensorDataView::I32(v.as_slice()),
                TensorData::I64(v) => TensorDataView::I64(v.as_slice()),
            },
        }
    }

    pub fn into_data(self) -> TensorData {
        self.data
    }
}

impl TensorData {
    pub fn len(&self) -> usize {
        match self {
            TensorData::F32(v) => v.len(),
            TensorData::I32(v) => v.len(),
            TensorData::I64(v) => v.len(),
        }
    }
}

impl Shape {
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }

    pub fn elements(&self) -> usize {
        self.dims.iter().product::<usize>().max(1)
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }
}
```

- [ ] **Step 2: Add unit tests at the bottom of the file**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_shape_and_data_match() {
        let t = Tensor::new(Shape::new(vec![2, 3]), TensorData::F32(vec![0.0; 6]));
        assert_eq!(t.shape().dims(), &[2, 3]);
        assert_eq!(t.element_type(), ElementType::F32);
    }

    #[test]
    #[should_panic(expected = "tensor data length mismatch")]
    fn test_tensor_rejects_mismatched_data() {
        Tensor::new(Shape::new(vec![2, 3]), TensorData::F32(vec![0.0; 5]));
    }

    #[test]
    fn test_shape_elements() {
        assert_eq!(Shape::new(vec![2, 3, 4]).elements(), 24);
        assert_eq!(Shape::new(vec![]).elements(), 1);
    }
}
```

- [ ] **Step 3: Verify**

Run:
```bash
cargo test -p gigastt-core runtime::tensor
```
Expected: PASS

Run:
```bash
cargo clippy -p gigastt-core --all-targets
```
Expected: zero warnings

- [ ] **Step 4: Commit**

```bash
git add crates/gigastt-core/src/runtime/tensor.rs
git commit -m "feat(runtime): add Tensor, Shape, and ElementType"
```

---

### Task 1.2: Create `runtime/error.rs`

**Files:**
- Create: `crates/gigastt-core/src/runtime/error.rs`

- [ ] **Step 1: Write the error type**

```rust
use std::path::PathBuf;
use thiserror::Error;

use super::tensor::Shape;

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("failed to load model {path}: {message}")]
    LoadFailed { path: PathBuf, message: String },

    #[error("inference failed: {0}")]
    InferenceFailed(String),

    #[error("invalid tensor shape: expected {expected:?}, got {got:?}")]
    InvalidShape { expected: Shape, got: Shape },

    #[error("unsupported element type: {0:?}")]
    UnsupportedElementType(super::tensor::ElementType),

    #[error("invalid input count: expected {expected}, got {got}")]
    InvalidInputCount { expected: usize, got: usize },
}
```

- [ ] **Step 2: Add unit tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_failed_display() {
        let e = RuntimeError::LoadFailed {
            path: PathBuf::from("encoder.onnx"),
            message: "not found".into(),
        };
        assert_eq!(e.to_string(), "failed to load model encoder.onnx: not found");
    }
}
```

- [ ] **Step 3: Verify**

Run:
```bash
cargo test -p gigastt-core runtime::error
cargo clippy -p gigastt-core --all-targets
```

- [ ] **Step 4: Commit**

```bash
git add crates/gigastt-core/src/runtime/error.rs
git commit -m "feat(runtime): add RuntimeError"
```

---

### Task 1.3: Create `runtime/session.rs`

**Files:**
- Create: `crates/gigastt-core/src/runtime/session.rs`

- [ ] **Step 1: Write the trait**

```rust
use super::{error::RuntimeError, tensor::Tensor};

/// One loaded model session: encoder, decoder, or joiner.
pub trait RuntimeSession: Send + Sync + 'static {
    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, RuntimeError>;
}
```

- [ ] **Step 2: Verify**

Run:
```bash
cargo clippy -p gigastt-core --all-targets
```

- [ ] **Step 3: Commit**

```bash
git add crates/gigastt-core/src/runtime/session.rs
git commit -m "feat(runtime): add RuntimeSession trait"
```

---

### Task 1.4: Create `runtime/factory.rs`

**Files:**
- Create: `crates/gigastt-core/src/runtime/factory.rs`

- [ ] **Step 1: Write the traits**

```rust
use super::{error::RuntimeError, session::RuntimeSession};

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
```

- [ ] **Step 2: Verify**

Run:
```bash
cargo clippy -p gigastt-core --all-targets
```

- [ ] **Step 3: Commit**

```bash
git add crates/gigastt-core/src/runtime/factory.rs
git commit -m "feat(runtime): add Runtime and RuntimeFactory traits"
```

---

### Task 1.5: Create `runtime/mod.rs` and wire into `lib.rs`

**Files:**
- Create: `crates/gigastt-core/src/runtime/mod.rs`
- Modify: `crates/gigastt-core/src/lib.rs`

- [ ] **Step 1: Write `runtime/mod.rs`**

```rust
pub mod error;
pub mod factory;
pub mod session;
pub mod tensor;

pub use error::RuntimeError;
pub use factory::{Runtime, RuntimeFactory};
pub use session::RuntimeSession;
pub use tensor::{ElementType, Shape, Tensor, TensorData, TensorDataView, TensorView};
```

- [ ] **Step 2: Modify `lib.rs`**

Add after `pub mod protocol;`:

```rust
pub(crate) mod runtime;
```

- [ ] **Step 3: Verify**

Run:
```bash
cargo test -p gigastt-core --lib
cargo clippy -p gigastt-core --all-targets
```

- [ ] **Step 4: Commit**

```bash
git add crates/gigastt-core/src/runtime/mod.rs crates/gigastt-core/src/lib.rs
git commit -m "feat(runtime): wire runtime module into gigastt-core"
```

---

## Phase 2: Ort Adapter

**Dependency:** Phase 1 must be complete and committed.

### Task 2.1: Create `runtime/ort/tensor.rs`

**Files:**
- Create: `crates/gigastt-core/src/runtime/ort/tensor.rs`

- [ ] **Step 1: Implement `Tensor` -> `ort::Value`**

```rust
use ort::value::Value;

use crate::runtime::{error::RuntimeError, tensor::{ElementType, Shape, Tensor, TensorData}};

impl Tensor {
    pub fn into_ort_value(self) -> Result<Value, RuntimeError> {
        let shape = self.shape.dims().iter().map(|&d| d as i64).collect::<Vec<_>>();
        match self.data {
            TensorData::F32(data) => Value::from_array((shape.as_slice(), data))
                .map_err(|e| RuntimeError::InferenceFailed(e.to_string())),
            TensorData::I32(data) => Value::from_array((shape.as_slice(), data))
                .map_err(|e| RuntimeError::InferenceFailed(e.to_string())),
            TensorData::I64(data) => Value::from_array((shape.as_slice(), data))
                .map_err(|e| RuntimeError::InferenceFailed(e.to_string())),
        }
    }
}
```

- [ ] **Step 2: Implement `ort::Value` -> `Tensor`**

```rust
pub fn value_to_tensor(value: Value) -> Result<Tensor, RuntimeError> {
    let tensor = value.try_extract_tensor::<f32>()
        .or_else(|_| value.try_extract_tensor::<i32>().map(|t| t.map(|x| *x as f32)))
        .or_else(|_| value.try_extract_tensor::<i64>().map(|t| t.map(|x| *x as f32)))
        .map_err(|e| RuntimeError::InferenceFailed(e.to_string()))?;

    let shape = Shape::new(tensor.shape().iter().map(|&d| d as usize).collect());
    let data = tensor.view().iter().copied().collect::<Vec<_>>();
    Ok(Tensor::new(shape, TensorData::F32(data)))
}
```

**Note:** The exact `ort::Value` API may differ. Adjust `try_extract_tensor`, `from_array`, and shape extraction to match `ort 2.0.0-rc.12` signatures. The principle is: convert `TensorData` enum to the matching typed `Value::from_array`, and convert output `Value` back to `TensorData::F32` (or detect element type if `ort` exposes it).

- [ ] **Step 3: Add round-trip test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_ort_roundtrip() {
        let tensor = Tensor::new(Shape::new(vec![2, 3]), TensorData::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        let value = tensor.clone().into_ort_value().unwrap();
        let recovered = value_to_tensor(value).unwrap();
        assert_eq!(tensor, recovered);
    }
}
```

- [ ] **Step 4: Verify**

Run:
```bash
cargo test -p gigastt-core runtime::ort::tensor
cargo clippy -p gigastt-core --all-targets
```

- [ ] **Step 5: Commit**

```bash
git add crates/gigastt-core/src/runtime/ort/tensor.rs
git commit -m "feat(runtime): add ort tensor conversion"
```

---

### Task 2.2: Create `runtime/ort/session.rs`

**Files:**
- Create: `crates/gigastt-core/src/runtime/ort/session.rs`

- [ ] **Step 1: Implement `OrtRuntime` and `OrtSession`**

```rust
use std::path::Path;
use ort::{Environment, Session};

use crate::runtime::{
    error::RuntimeError,
    factory::{Runtime, RuntimeFactory},
    session::RuntimeSession,
    tensor::Tensor,
};
use super::tensor::value_to_tensor;

pub struct OrtRuntime {
    environment: Environment,
    intra_threads: usize,
    provider: super::factory::OrtExecutionProvider,
}

pub struct OrtSession {
    session: Session,
}

impl Runtime for OrtRuntime {
    fn load_session(&self, model_path: &Path) -> Result<Box<dyn RuntimeSession>, RuntimeError> {
        let session = self.environment
            .new_session_builder()
            .map_err(|e| RuntimeError::LoadFailed { path: model_path.into(), message: e.to_string() })?
            .with_intra_threads(self.intra_threads)
            .map_err(|e| RuntimeError::LoadFailed { path: model_path.into(), message: e.to_string() })?
            .with_execution_provider(self.provider.to_ort())
            .map_err(|e| RuntimeError::LoadFailed { path: model_path.into(), message: e.to_string() })?
            .commit_from_file(model_path)
            .map_err(|e| RuntimeError::LoadFailed { path: model_path.into(), message: e.to_string() })?;
        Ok(Box::new(OrtSession { session }))
    }
}

impl RuntimeSession for OrtSession {
    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, RuntimeError> {
        let ort_inputs: Vec<Value> = inputs
            .into_iter()
            .map(Tensor::into_ort_value)
            .collect::<Result<_, _>>()?;
        let outputs = self.session.run(ort_inputs)
            .map_err(|e| RuntimeError::InferenceFailed(e.to_string()))?;
        outputs.into_iter().map(value_to_tensor).collect()
    }
}
```

- [ ] **Step 2: Verify**

Run:
```bash
cargo clippy -p gigastt-core --all-targets
```

- [ ] **Step 3: Commit**

```bash
git add crates/gigastt-core/src/runtime/ort/session.rs
git commit -m "feat(runtime): add OrtRuntime and OrtSession"
```

---

### Task 2.3: Create `runtime/ort/factory.rs`

**Files:**
- Create: `crates/gigastt-core/src/runtime/ort/factory.rs`
- Modify: `crates/gigastt-core/src/runtime/ort/mod.rs`

- [ ] **Step 1: Define execution provider wrapper and `OrtFactory`**

```rust
use ort::{ExecutionProvider, Environment};

use crate::runtime::{error::RuntimeError, factory::{Runtime, RuntimeFactory}};
use super::session::OrtRuntime;

#[derive(Clone)]
pub enum OrtExecutionProvider {
    Cpu,
    #[cfg(feature = "coreml")]
    CoreML,
    #[cfg(feature = "cuda")]
    Cuda,
    #[cfg(feature = "nnapi")]
    Nnapi,
}

impl OrtExecutionProvider {
    pub fn to_ort(&self) -> ExecutionProvider {
        match self {
            OrtExecutionProvider::Cpu => ExecutionProvider::CPU(Default::default()),
            #[cfg(feature = "coreml")]
            OrtExecutionProvider::CoreML => ExecutionProvider::CoreML(Default::default()),
            #[cfg(feature = "cuda")]
            OrtExecutionProvider::Cuda => ExecutionProvider::CUDA(Default::default()),
            #[cfg(feature = "nnapi")]
            OrtExecutionProvider::Nnapi => ExecutionProvider::NNAPI(Default::default()),
        }
    }
}

pub struct OrtFactory {
    provider: OrtExecutionProvider,
}

impl OrtFactory {
    pub fn cpu() -> Self {
        Self { provider: OrtExecutionProvider::Cpu }
    }

    #[cfg(feature = "coreml")]
    pub fn coreml() -> Self {
        Self { provider: OrtExecutionProvider::CoreML }
    }

    #[cfg(feature = "cuda")]
    pub fn cuda() -> Self {
        Self { provider: OrtExecutionProvider::Cuda }
    }

    #[cfg(feature = "nnapi")]
    pub fn nnapi() -> Self {
        Self { provider: OrtExecutionProvider::Nnapi }
    }
}

impl RuntimeFactory for OrtFactory {
    fn create(&self, intra_threads: usize) -> Result<Box<dyn Runtime>, RuntimeError> {
        let environment = Environment::builder()
            .with_name("gigastt")
            .build()
            .map_err(|e| RuntimeError::InferenceFailed(e.to_string()))?;
        Ok(Box::new(OrtRuntime {
            environment,
            intra_threads,
            provider: self.provider.clone(),
        }))
    }
}

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

**Note:** Adjust `ExecutionProvider::CPU/CoreML/CUDA/NNAPI` constructors to match `ort` API exactly.

- [ ] **Step 2: Write `runtime/ort/mod.rs`**

```rust
pub mod factory;
pub mod session;
pub mod tensor;

pub use factory::{default_factory, OrtExecutionProvider, OrtFactory};
pub use session::{OrtRuntime, OrtSession};
```

- [ ] **Step 3: Verify feature builds**

Run:
```bash
cargo check -p gigastt-core
cargo check -p gigastt-core --features coreml
cargo check -p gigastt-core --features cuda
cargo check -p gigastt-core --features diarization
```

- [ ] **Step 4: Commit**

```bash
git add crates/gigastt-core/src/runtime/ort/
git commit -m "feat(runtime): add ort provider factories"
```

---

## Phase 3: Engine and SessionPool Integration

**Dependency:** Phase 2 must be complete and committed.

### Task 3.1: Refactor `SessionTriplet` in `inference/mod.rs`

**Files:**
- Modify: `crates/gigastt-core/src/inference/mod.rs`

- [ ] **Step 1: Remove direct `ort` imports and update `SessionTriplet`**

Remove:
```rust
use ort::session::Session;
use ort::value::TensorRef;
```

Change:
```rust
struct SessionTriplet {
    encoder: Box<dyn crate::runtime::session::RuntimeSession>,
    decoder: Box<dyn crate::runtime::session::RuntimeSession>,
    joiner: Box<dyn crate::runtime::session::RuntimeSession>,
}
```

- [ ] **Step 2: Update all places that construct or use `SessionTriplet`**

Search for `ort::Session` and `Session` usages in `inference/mod.rs`. Replace construction with calls to `runtime.load_session(path)`. Update any method signatures that previously took `ort::Session` or `ort::Value`.

- [ ] **Step 3: Verify compilation**

Run:
```bash
cargo check -p gigastt-core
cargo clippy -p gigastt-core --all-targets
```

- [ ] **Step 4: Commit**

```bash
git add crates/gigastt-core/src/inference/mod.rs
git commit -m "refactor(inference): use RuntimeSession in SessionTriplet"
```

---

### Task 3.2: Add `Engine::load_with_factory` and update `load_with_pools_threads`

**Files:**
- Modify: `crates/gigastt-core/src/inference/mod.rs`

- [ ] **Step 1: Update `Engine::load_with_pools_threads` signature and add internal factory-based loader**

Find the existing `load_with_pools_threads` function. Keep the public signature identical. Inside, build the default factory and delegate:

```rust
pub fn load_with_pools_threads(
    model_dir: &Path,
    pool_size: usize,
    pool_min_size: usize,
    batch_pool_size: usize,
    intra_threads: usize,
) -> anyhow::Result<Self> {
    let factory = crate::runtime::ort::default_factory(intra_threads);
    Self::load_with_factory(
        model_dir,
        pool_size,
        pool_min_size,
        batch_pool_size,
        factory,
        intra_threads,
    )
}

pub(crate) fn load_with_factory(
    model_dir: &Path,
    pool_size: usize,
    pool_min_size: usize,
    batch_pool_size: usize,
    factory: Box<dyn crate::runtime::factory::RuntimeFactory>,
    intra_threads: usize,
) -> anyhow::Result<Self> {
    let runtime = factory.create(intra_threads)
        .map_err(|e| anyhow::anyhow!(e))?;
    // ... existing logic, but load sessions via runtime.load_session(path)
}
```

- [ ] **Step 2: Update session loading logic**

Where the code previously built `ort::Session::commit_from_file`, use:

```rust
let encoder = runtime.load_session(&encoder_path)?;
let decoder = runtime.load_session(&decoder_path)?;
let joiner = runtime.load_session(&joiner_path)?;
```

- [ ] **Step 3: Verify**

Run:
```bash
cargo test -p gigastt-core --lib
cargo clippy -p gigastt-core --all-targets
```

- [ ] **Step 4: Commit**

```bash
git add crates/gigastt-core/src/inference/mod.rs
git commit -m "feat(inference): add Engine::load_with_factory"
```

---

### Task 3.3: Update `decode.rs` to use `TensorView`

**Files:**
- Modify: `crates/gigastt-core/src/inference/decode.rs`

- [ ] **Step 1: Find all `ort::Value` / raw slice usage**

Search for `TensorRef`, `try_extract_tensor`, `.view()`, and raw float slices in `decode.rs`.

- [ ] **Step 2: Replace with `TensorView`**

Change function signatures from raw slices or `ort::Value` to `TensorView`. For example:

```rust
fn decode_step(
    encoder_out: &crate::runtime::tensor::TensorView,
    decoder_state: &mut DecoderState,
) -> anyhow::Result<Option<TokenId>> {
    let data = match encoder_out.data() {
        crate::runtime::tensor::TensorDataView::F32(v) => v,
        _ => anyhow::bail!("unexpected tensor type"),
    };
    // ... existing logic using data
}
```

- [ ] **Step 3: Update callers in `inference/mod.rs`**

Where `decode.rs` functions are called, pass `tensor.view()` instead of extracting raw slices.

- [ ] **Step 4: Verify**

Run:
```bash
cargo test -p gigastt-core --lib
cargo clippy -p gigastt-core --all-targets
```

- [ ] **Step 5: Commit**

```bash
git add crates/gigastt-core/src/inference/decode.rs crates/gigastt-core/src/inference/mod.rs
git commit -m "refactor(decode): operate on TensorView instead of ort values"
```

---

### Task 3.4: Add `From<RuntimeError>` for `GigasttError`

**Files:**
- Modify: `crates/gigastt-core/src/error.rs`

- [ ] **Step 1: Add conversion**

At the bottom of `error.rs`, outside the `#[cfg(test)]` module:

```rust
impl From<crate::runtime::RuntimeError> for GigasttError {
    fn from(err: crate::runtime::RuntimeError) -> Self {
        match err {
            crate::runtime::RuntimeError::LoadFailed { path, message } => GigasttError::ModelLoad {
                path: path.to_string_lossy().into_owned(),
                source: Some(message.into()),
            },
            other => GigasttError::Inference {
                source: Box::new(other),
            },
        }
    }
}
```

- [ ] **Step 2: Update `Engine` to use `?` conversion**

Where `inference/mod.rs` currently wraps runtime errors with `ort_err` or `anyhow!`, replace with direct `?` so `RuntimeError` converts to `GigasttError`.

- [ ] **Step 3: Verify**

Run:
```bash
cargo test -p gigastt-core --lib
cargo clippy -p gigastt-core --all-targets
```

- [ ] **Step 4: Commit**

```bash
git add crates/gigastt-core/src/error.rs crates/gigastt-core/src/inference/mod.rs
git commit -m "feat(error): map RuntimeError to GigasttError"
```

---

### Task 3.5: Ensure no `ort` imports leak outside `runtime/ort/`

**Files:**
- Modify: any file still importing `ort` outside `runtime/ort/`

- [ ] **Step 1: Search for leaks**

Run:
```bash
rg "^use ort::" crates/gigastt-core/src --type rust
rg "ort::" crates/gigastt-core/src --type rust | grep -v "runtime/ort"
```

- [ ] **Step 2: Fix any leaks**

Replace direct `ort` usage with `crate::runtime::*` types. If a file legitimately needs `ort`, move the code into `runtime/ort/`.

- [ ] **Step 3: Verify**

Run:
```bash
cargo test -p gigastt-core --lib
cargo clippy -p gigastt-core --all-targets
```

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(runtime): isolate ort usage to runtime/ort module"
```

---

## Phase 4: Mock Runtime and Tests

**Dependency:** Phase 3 must be complete and committed.

### Task 4.1: Create `runtime/mock/session.rs`

**Files:**
- Create: `crates/gigastt-core/src/runtime/mock/session.rs`
- Create: `crates/gigastt-core/src/runtime/mock/mod.rs`

- [ ] **Step 1: Implement mock types**

```rust
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::runtime::{
    error::RuntimeError,
    factory::{Runtime, RuntimeFactory},
    session::RuntimeSession,
    tensor::{Shape, Tensor},
};

#[derive(Clone, Default)]
pub struct MockFactory {
    sessions: HashMap<String, Arc<MockSession>>,
}

impl MockFactory {
    pub fn new(sessions: HashMap<String, Arc<MockSession>>) -> Self {
        Self { sessions }
    }
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
    fn load_session(&self, model_path: &Path) -> Result<Box<dyn RuntimeSession>, RuntimeError> {
        let key = model_path.file_stem()
            .ok_or_else(|| RuntimeError::LoadFailed { path: model_path.into(), message: "empty path".into() })?
            .to_string_lossy()
            .to_string();
        let session = self.sessions
            .get(&key)
            .ok_or_else(|| RuntimeError::LoadFailed { path: model_path.into(), message: format!("no mock for {key}") })?
            .clone();
        Ok(Box::new((*session).clone()))
    }
}

#[derive(Clone)]
pub struct MockSession {
    pub expected_inputs: Vec<Shape>,
    pub outputs: Vec<Tensor>,
}

impl RuntimeSession for MockSession {
    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, RuntimeError> {
        for (actual, expected) in inputs.iter().zip(self.expected_inputs.iter()) {
            if actual.shape() != expected {
                return Err(RuntimeError::InvalidShape {
                    expected: expected.clone(),
                    got: actual.shape().clone(),
                });
            }
        }
        Ok(self.outputs.clone())
    }
}
```

- [ ] **Step 2: Write `runtime/mock/mod.rs`**

```rust
pub mod session;

pub use session::{MockFactory, MockRuntime, MockSession};
```

- [ ] **Step 3: Verify**

Run:
```bash
cargo test -p gigastt-core --lib
cargo clippy -p gigastt-core --all-targets
```

- [ ] **Step 4: Commit**

```bash
git add crates/gigastt-core/src/runtime/mock/
git commit -m "feat(runtime): add mock runtime adapter"
```

---

### Task 4.2: Add Engine unit tests using `MockFactory`

**Files:**
- Modify: `crates/gigastt-core/src/inference/mod.rs` (add tests at bottom) or create `crates/gigastt-core/tests/runtime_mock.rs`

- [ ] **Step 1: Write a test that loads Engine with mock sessions**

```rust
#[cfg(test)]
mod runtime_tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use crate::inference::Engine;
    use crate::runtime::mock::{MockFactory, MockSession};
    use crate::runtime::tensor::{Shape, Tensor, TensorData};

    #[test]
    fn test_engine_loads_with_mock_runtime() {
        let mut sessions = HashMap::new();
        sessions.insert("encoder".into(), Arc::new(MockSession {
            expected_inputs: vec![Shape::new(vec![1, 64, 100])],
            outputs: vec![Tensor::new(Shape::new(vec![1, 25, 768]), TensorData::F32(vec![0.0; 25 * 768]))],
        }));
        sessions.insert("decoder".into(), Arc::new(MockSession {
            expected_inputs: vec![Shape::new(vec![1, 1])],
            outputs: vec![Tensor::new(Shape::new(vec![1, 1, 320]), TensorData::F32(vec![0.0; 320]))],
        }));
        sessions.insert("joiner".into(), Arc::new(MockSession {
            expected_inputs: vec![Shape::new(vec![1, 768]), Shape::new(vec![1, 320])],
            outputs: vec![Tensor::new(Shape::new(vec![1, 1, 34]), TensorData::F32(vec![0.0; 34]))],
        }));

        let factory = Box::new(MockFactory::new(sessions));
        let result = Engine::load_with_factory(
            std::path::Path::new("/tmp/mock-models"),
            1, 1, 0,
            factory,
            1,
        );
        assert!(result.is_ok());
    }
}
```

**Note:** Adjust shapes to match the actual model dimensions (`ENC_DIM=768`, `PRED_HIDDEN=320`, vocab size). The test above is illustrative.

- [ ] **Step 2: Verify**

Run:
```bash
cargo test -p gigastt-core --lib runtime_tests
cargo clippy -p gigastt-core --all-targets
```

- [ ] **Step 3: Commit**

```bash
git add crates/gigastt-core/src/inference/mod.rs
git commit -m "test(runtime): add Engine mock runtime test"
```

---

### Task 4.3: Add CI check for `ort` import isolation

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Add a lint step**

Add a new job or step in `ci.yml`:

```yaml
  runtime-isolation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Check ort is isolated to runtime/ort
        run: |
          if rg "^use ort::" crates/gigastt-core/src --type rust | grep -v "runtime/ort"; then
            echo "ERROR: ort import found outside runtime/ort/"
            exit 1
          fi
          if rg "ort::" crates/gigastt-core/src --type rust | grep -v "runtime/ort"; then
            echo "ERROR: ort usage found outside runtime/ort/"
            exit 1
          fi
          echo "OK: ort is isolated"
```

- [ ] **Step 2: Verify locally**

Run:
```bash
if rg "^use ort::" crates/gigastt-core/src --type rust | grep -v "runtime/ort"; then echo "LEAK"; else echo "OK"; fi
```

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: enforce ort isolation to runtime/ort"
```

---

## Final Verification

After all tasks are complete, run the full verification suite:

```bash
cargo test -p gigastt-core --lib --bins
cargo test -p gigastt --lib --bins
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check
```

If model files are available locally, also run:

```bash
cargo test -p gigastt --test e2e_rest --test e2e_ws -- --ignored --test-threads=1
```

---

## Self-Review Checklist

- [ ] Every spec goal has at least one implementing task.
- [ ] No placeholder text ("TBD", "TODO", "implement later") remains.
- [ ] File paths are exact and exist in the repo.
- [ ] Type names (`RuntimeSession`, `RuntimeError`, `Tensor`, etc.) are consistent across all tasks.
- [ ] Each task ends with a verification command and a commit.
- [ ] Dependencies between phases are explicit.
