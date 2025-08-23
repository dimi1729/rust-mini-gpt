use burn::{
    data::dataset::Dataset,
    tensor::{backend::Backend, Int, Tensor},
};

/// A dataset item containing input and target token sequences
#[derive(Debug, Clone)]
pub struct GPTItem<B: Backend> {
    pub inputs: Tensor<B, 1, Int>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> GPTItem<B> {
    pub fn new(inputs: Tensor<B, 1, Int>, targets: Tensor<B, 1, Int>) -> Self {
        Self { inputs, targets }
    }
}

/// Dataset for GPT training that generates sequences from tokens
pub struct GPTDataset<B: Backend> {
    tokens: Vec<u32>,
    sequence_length: usize,
    device: B::Device,
    items: Vec<GPTItem<B>>,
}

impl<B: Backend> GPTDataset<B> {
    pub fn new(tokens: Vec<u32>, sequence_length: usize, device: B::Device) -> Self {
        let mut items = Vec::new();

        // Create overlapping sequences from the token stream
        // Each item is a sequence of tokens and its corresponding targets (next tokens)
        for i in 0..tokens.len().saturating_sub(sequence_length) {
            let input_tokens: Vec<i32> = tokens[i..i + sequence_length]
                .iter()
                .map(|&t| t as i32)
                .collect();

            let target_tokens: Vec<i32> = tokens[i + 1..i + sequence_length + 1]
                .iter()
                .map(|&t| t as i32)
                .collect();

            if input_tokens.len() == sequence_length && target_tokens.len() == sequence_length {
                let inputs = Tensor::from_data(
                    burn::tensor::TensorData::new(input_tokens, [sequence_length]),
                    &device,
                );
                let targets = Tensor::from_data(
                    burn::tensor::TensorData::new(target_tokens, [sequence_length]),
                    &device,
                );

                items.push(GPTItem::new(inputs, targets));
            }
        }

        Self {
            tokens,
            sequence_length,
            device,
            items,
        }
    }

    /// Split the dataset into train and validation sets
    pub fn split(self, train_ratio: f32) -> (Self, Self) {
        let split_idx = (self.tokens.len() as f32 * train_ratio) as usize;

        let train_tokens = self.tokens[..split_idx].to_vec();
        let valid_tokens = self.tokens[split_idx..].to_vec();

        let train_dataset = GPTDataset::new(train_tokens, self.sequence_length, self.device.clone());
        let valid_dataset = GPTDataset::new(valid_tokens, self.sequence_length, self.device);

        (train_dataset, valid_dataset)
    }
}

impl<B: Backend> Dataset<GPTItem<B>> for GPTDataset<B> {
    fn get(&self, index: usize) -> Option<GPTItem<B>> {
        self.items.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_dataset_creation() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let tokens = (0..100).map(|i| i as u32).collect::<Vec<_>>();
        let sequence_length = 8;

        let dataset = GPTDataset::<TestBackend>::new(tokens.clone(), sequence_length, device);

        // Should have tokens.len() - sequence_length items
        assert_eq!(dataset.len(), tokens.len() - sequence_length);

        // Test first item
        let first_item = dataset.get(0).unwrap();
        assert_eq!(first_item.inputs.shape().dims, [sequence_length]);
        assert_eq!(first_item.targets.shape().dims, [sequence_length]);

        // Verify the first few input/target pairs
        let input_data = first_item.inputs.into_data().to_vec::<i32>().unwrap();
        let target_data = first_item.targets.into_data().to_vec::<i32>().unwrap();

        for i in 0..sequence_length {
            assert_eq!(input_data[i], i as i32);
            assert_eq!(target_data[i], (i + 1) as i32);
        }
    }

    #[test]
    fn test_dataset_split() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let tokens = (0..100).map(|i| i as u32).collect::<Vec<_>>();
        let sequence_length = 8;

        let dataset = GPTDataset::<TestBackend>::new(tokens, sequence_length, device);
        let (train_dataset, valid_dataset) = dataset.split(0.8);

        // Check that split worked
        assert!(train_dataset.len() > 0);
        assert!(valid_dataset.len() > 0);
        assert!(train_dataset.len() > valid_dataset.len());

        // Check that we can get items from both datasets
        let train_item = train_dataset.get(0).unwrap();
        let valid_item = valid_dataset.get(0).unwrap();

        assert_eq!(train_item.inputs.shape().dims, [sequence_length]);
        assert_eq!(valid_item.inputs.shape().dims, [sequence_length]);
    }
}
