use super::{error::RuntimeError, tensor::Tensor};

/// One loaded model session: encoder, decoder, or joiner.
pub trait RuntimeSession: Send + Sync + 'static {
    fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, RuntimeError>;

    /// Run into caller-owned, flattened f32 output buffers in model output order.
    ///
    /// The number of destinations must match the model outputs. Output count and
    /// types are validated before changing any destination. Buffers retain their
    /// capacity across calls and remain owned by the caller after this returns.
    /// Shapes are intentionally omitted; use `run` when output dimensions matter.
    /// The default preserves existing backends by copying their owned outputs;
    /// optimized backends may copy directly from runtime-owned output storage.
    fn run_f32_into(
        &self,
        inputs: &[Tensor],
        destinations: &mut [&mut Vec<f32>],
    ) -> Result<(), RuntimeError> {
        let outputs = self.run(inputs)?;
        if outputs.len() != destinations.len() {
            return Err(RuntimeError::InferenceFailed(format!(
                "expected {} output buffers, got {}",
                outputs.len(),
                destinations.len()
            )));
        }
        for output in &outputs {
            if output.view().data().as_f32().is_none() {
                return Err(RuntimeError::UnsupportedElementType(output.element_type()));
            }
        }
        for (output, destination) in outputs.iter().zip(destinations) {
            if let Some(data) = output.view().data().as_f32() {
                destination.clear();
                destination.extend_from_slice(data);
            }
        }
        Ok(())
    }

    /// Low-latency encoder path used by streaming windows.
    ///
    /// Default delegates to [`Self::run`]. The ANE encoder overrides this to
    /// accept a lower pad fill floor so short streaming windows can pad into the
    /// smallest eligible bucket instead of falling back to ort.
    fn run_low_latency(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, RuntimeError> {
        self.run(inputs)
    }

    /// True when this encoder runs on the ANE fixed-shape pad-up path.
    ///
    /// Used to pick a longer long-form chunk window (30s fills ANE bucket 3000
    /// nearly full; ort keeps 24s for peak activation memory). Default false.
    fn is_ane_encoder(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::tensor::{Shape, TensorData};

    struct LegacySession(Vec<Tensor>);
    impl RuntimeSession for LegacySession {
        fn run(&self, _: &[Tensor]) -> Result<Vec<Tensor>, RuntimeError> {
            Ok(self.0.clone())
        }
    }

    #[test]
    fn test_f32_workspace_default_preserves_capacity_and_validates_before_write() {
        let float = Tensor::new(Shape::new(vec![2]), TensorData::F32(vec![1.0, 2.0])).unwrap();
        let integer = Tensor::new(Shape::new(vec![1]), TensorData::I64(vec![3])).unwrap();
        let mut first = vec![9.0; 4];
        let mut second = vec![8.0; 4];
        let pointer = first.as_ptr();
        LegacySession(vec![float.clone()])
            .run_f32_into(&[], &mut [&mut first])
            .unwrap();
        assert_eq!(first, [1.0, 2.0]);
        assert_eq!(first.as_ptr(), pointer);
        let original = first.clone();
        assert!(
            LegacySession(vec![float.clone()])
                .run_f32_into(&[], &mut [&mut first, &mut second])
                .is_err()
        );
        assert_eq!(first, original);
        assert_eq!(second, [8.0; 4]);
        assert!(matches!(
            LegacySession(vec![float, integer]).run_f32_into(&[], &mut [&mut first, &mut second]),
            Err(RuntimeError::UnsupportedElementType(_))
        ));
        assert_eq!(first, original);
        assert_eq!(second, [8.0; 4]);
        LegacySession(vec![]).run_f32_into(&[], &mut []).unwrap();
    }
}
