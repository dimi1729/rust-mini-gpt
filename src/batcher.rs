use crate::dataset::GPTItem;
use crate::model::TrainingBatch;
use burn::{
    data::dataloader::batcher::Batcher,
    tensor::{backend::Backend, Int, Tensor},
};

/// Batcher for GPT training that combines individual sequence items into batches
#[derive(Clone, Debug)]
pub struct GPTBatcher<B: Backend> {
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> GPTBatcher<B> {
    pub fn new(_device: B::Device) -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> Batcher<B, GPTItem<B>, TrainingBatch<B>> for GPTBatcher<B> {
    fn batch(&self, items: Vec<GPTItem<B>>, _device: &B::Device) -> TrainingBatch<B> {
        let batch_size = items.len();

        if batch_size == 0 {
            panic!("Cannot create batch from empty items");
        }

        let _sequence_length = items[0].inputs.dims()[0];

        // Collect all input and target tensors
        let inputs: Vec<Tensor<B, 1, Int>> = items.iter().map(|item| item.inputs.clone()).collect();
        let targets: Vec<Tensor<B, 1, Int>> =
            items.iter().map(|item| item.targets.clone()).collect();

        // Stack them into batch tensors [batch_size, sequence_length]
        let batched_inputs = Tensor::stack(inputs, 0);
        let batched_targets = Tensor::stack(targets, 0);

        TrainingBatch::new(batched_inputs, batched_targets)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_batcher() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let batcher = GPTBatcher::new(device.clone());

        // Create test items
        let sequence_length = 4;
        let item1 = GPTItem::new(
            Tensor::from_data([0, 1, 2, 3], &device),
            Tensor::from_data([1, 2, 3, 4], &device),
        );
        let item2 = GPTItem::new(
            Tensor::from_data([4, 5, 6, 7], &device),
            Tensor::from_data([5, 6, 7, 8], &device),
        );

        let items = vec![item1, item2];
        let batch = batcher.batch(items, &device);

        // Check batch dimensions
        assert_eq!(batch.inputs.shape().dims, [2, sequence_length]); // [batch_size, seq_len]
        assert_eq!(batch.targets.shape().dims, [2, sequence_length]);

        // Verify batch contents
        let inputs_data = batch.inputs.into_data().to_vec::<i32>().unwrap();
        let targets_data = batch.targets.into_data().to_vec::<i32>().unwrap();

        // First sequence
        assert_eq!(inputs_data[0..4], [0, 1, 2, 3]);
        assert_eq!(targets_data[0..4], [1, 2, 3, 4]);

        // Second sequence
        assert_eq!(inputs_data[4..8], [4, 5, 6, 7]);
        assert_eq!(targets_data[4..8], [5, 6, 7, 8]);
    }
}
