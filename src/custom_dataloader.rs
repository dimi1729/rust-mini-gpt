// Burn needs to use its proper Dataset + Dataloader traits to
// actually use the LearnerBuilder TUI so this code is
// unnessecary, but kept to show how to implement a simple dataloader
// manually
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

#[derive(Clone)]
pub struct DataLoader<B: Backend> {
    pub device: B::Device,
    b: usize, // batch size
    t: usize, // tokens per batch
    process_rank: usize,
    num_processes: usize,
    pub current_position: usize,
    pub tokens: Vec<u32>,
}

impl<B: Backend> DataLoader<B> {
    pub fn new(
        device: B::Device,
        tokens: Vec<u32>,
        b: usize,
        t: usize,
        process_rank: usize,
        num_processes: usize,
    ) -> Self {
        assert!(
            tokens.len() > b * t * num_processes * 2,
            "Not enough tokens for batch size and tokens per batch"
        );

        return Self {
            device,
            b,
            t,
            process_rank,
            num_processes,
            current_position: b * t * process_rank,
            tokens,
        };
    }

    pub fn next_batch(&mut self) -> (Tensor<B, 2, Int>, Tensor<B, 2, Int>) {
        let end = self.current_position + self.b * self.t;

        // Check if we need to wrap around
        if end + 1 > self.tokens.len() {
            self.current_position = self.b * self.t * self.process_rank;
        }

        let start = self.current_position;
        self.current_position += self.b * self.t * self.num_processes;

        // Create input (x) and target (y) tensors
        // x contains tokens from start to start + b*t
        // y contains the next tokens (shifted by 1) for each position in x
        let mut x_data = Vec::with_capacity(self.b * self.t);
        let mut y_data = Vec::with_capacity(self.b * self.t);

        for batch_idx in 0..self.b {
            let batch_start = start + batch_idx * self.t;
            for seq_idx in 0..self.t {
                let pos = batch_start + seq_idx;
                x_data.push(self.tokens[pos]);
                y_data.push(self.tokens[pos + 1]);
            }
        }

        let x = Tensor::from_data(
            burn::tensor::TensorData::new(x_data, [self.b, self.t]),
            &self.device,
        );
        let y = Tensor::from_data(
            burn::tensor::TensorData::new(y_data, [self.b, self.t]),
            &self.device,
        );

        (x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom_tokenizer::CustomTokenizer;
    use burn_ndarray::NdArray;
    use std::fs;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_dataloader_single_device() {
        // Load the tokenizer
        let tokenizer = CustomTokenizer::load("data/tokenizer.ron")
            .expect("Failed to load tokenizer. Make sure to run main.rs first to create it.");

        // Load and tokenize the data
        let text = fs::read_to_string("data/tiny_shakespeare.txt")
            .expect("Failed to read tiny_shakespeare.txt");

        // Use a subset for testing
        let text = text.chars().take(5000).collect::<String>();
        let tokens = tokenizer.encode(&text);

        println!("Loaded {} tokens for testing", tokens.len());

        // Create dataloader for single device
        let device: <TestBackend as burn::tensor::backend::Backend>::Device = Default::default();
        let batch_size = 4;
        let seq_length = 32;
        let mut dataloader = DataLoader::<TestBackend>::new(
            device, tokens, batch_size, seq_length, 0, // process_rank
            1, // num_processes
        );

        // Get a batch
        let (x, y) = dataloader.next_batch();

        // Check dimensions
        assert_eq!(x.shape().dims, [batch_size, seq_length]);
        assert_eq!(y.shape().dims, [batch_size, seq_length]);

        // Get another batch to ensure position advances correctly
        let (x2, y2) = dataloader.next_batch();
        assert_eq!(x2.shape().dims, [batch_size, seq_length]);
        assert_eq!(y2.shape().dims, [batch_size, seq_length]);

        println!("Single device test passed!");
    }

    #[test]
    fn test_dataloader_multiple_devices() {
        // Load the tokenizer
        let tokenizer = CustomTokenizer::load("data/tokenizer.ron")
            .expect("Failed to load tokenizer. Make sure to run main.rs first to create it.");

        // Load and tokenize the data
        let text = fs::read_to_string("data/tiny_shakespeare.txt")
            .expect("Failed to read tiny_shakespeare.txt");

        // Use a subset for testing
        let text = text.chars().take(10000).collect::<String>();
        let tokens = tokenizer.encode(&text);

        println!("Loaded {} tokens for testing", tokens.len());

        let device: <TestBackend as burn::tensor::backend::Backend>::Device = Default::default();
        let batch_size = 2;
        let seq_length = 16;
        let num_processes = 2;

        // Create dataloaders for multiple processes
        let mut dataloader_0 = DataLoader::<TestBackend>::new(
            device,
            tokens.clone(),
            batch_size,
            seq_length,
            0, // process_rank
            num_processes,
        );

        let mut dataloader_1 = DataLoader::<TestBackend>::new(
            device,
            tokens,
            batch_size,
            seq_length,
            1, // process_rank
            num_processes,
        );

        // Get batches from both dataloaders
        let (x0, y0) = dataloader_0.next_batch();
        let (x1, y1) = dataloader_1.next_batch();

        // Check dimensions
        assert_eq!(x0.shape().dims, [batch_size, seq_length]);
        assert_eq!(y0.shape().dims, [batch_size, seq_length]);
        assert_eq!(x1.shape().dims, [batch_size, seq_length]);
        assert_eq!(y1.shape().dims, [batch_size, seq_length]);

        // Verify that the two processes are getting different data
        // (they should start at different positions)
        assert_eq!(
            dataloader_0.current_position,
            batch_size * seq_length * num_processes
        );
        assert_eq!(
            dataloader_1.current_position,
            batch_size * seq_length * (1 + num_processes)
        );

        println!("Multiple devices test passed!");
    }
}
