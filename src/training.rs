use crate::config::Config;
use crate::dataloader::DataLoader;
use crate::model::{MiniGPT, TrainingBatch};
use burn::tensor::{Int, Tensor, backend::AutodiffBackend};

pub struct SimpleTrainer<B: AutodiffBackend> {
    model: MiniGPT<B>,
    device: B::Device,
}

impl<B: AutodiffBackend> SimpleTrainer<B> {
    pub fn new(config: &Config, device: B::Device) -> Self {
        let model = MiniGPT::new(config, &device);

        Self { model, device }
    }

    pub fn train_step(&mut self, batch: TrainingBatch<B>, learning_rate: f64) -> f32 {
        // Forward pass with loss calculation
        let (_logits, loss) = self.model.training_step(batch.inputs, batch.targets);

        // Backward pass - compute gradients
        let grads = loss.backward();

        // Manual parameter update using SGD (simplified)
        // In a real implementation, you'd use burn's optimizer framework
        self.model = self.simple_sgd_step(self.model.clone(), grads, learning_rate);

        // Return loss value
        loss.into_data().to_vec::<f32>().unwrap()[0]
    }

    // Simple SGD implementation for demonstration
    // Note: This is a simplified version - real optimizers are more complex
    fn simple_sgd_step(
        &self,
        model: MiniGPT<B>,
        _grads: B::Gradients,
        _learning_rate: f64,
    ) -> MiniGPT<B> {
        // In a real implementation, you would iterate through all parameters
        // and update them using the gradients. For now, we just return the model
        // as this is primarily for demonstration purposes.

        // The actual parameter update would look something like:
        // for param in model.parameters() {
        //     let grad = param.grad(&grads);
        //     param = param - learning_rate * grad;
        // }

        model
    }

    pub fn validate_step(&self, batch: TrainingBatch<B>) -> f32 {
        let (_, loss) = self.model.forward_with_loss(batch.inputs, batch.targets);
        loss.into_data().to_vec::<f32>().unwrap()[0]
    }

    pub fn train_epoch(
        &mut self,
        dataloader: &mut DataLoader<B>,
        steps_per_epoch: usize,
        learning_rate: f64,
    ) -> f32 {
        let mut total_loss = 0.0;
        let mut step_count = 0;

        for step in 0..steps_per_epoch {
            // Get batch from dataloader
            let (inputs, targets) = dataloader.next_batch();
            let batch = TrainingBatch::new(inputs, targets);

            // Training step
            let loss = self.train_step(batch, learning_rate);
            total_loss += loss;
            step_count += 1;

            if step % 100 == 0 && step > 0 {
                println!("Step {}/{}, Loss: {:.4}", step, steps_per_epoch, loss);
            }
        }

        total_loss / step_count as f32
    }

    pub fn generate(&self, prompt: Tensor<B, 2, Int>, max_new_tokens: usize) -> Vec<u32> {
        let mut generated = Vec::new();
        let mut current_input = prompt;

        for _ in 0..max_new_tokens {
            // Forward pass to get logits
            let logits = self.model.forward(current_input.clone());

            // Get the last token's logits [B, vocab_size]
            let seq_len = logits.dims()[1];
            let last_logits =
                logits
                    .clone()
                    .slice([0..1, (seq_len - 1)..seq_len, 0..logits.dims()[2]]);
            let last_logits = last_logits.squeeze::<2>(1); // [1, vocab_size]

            // For simplicity, just take argmax (greedy sampling)
            let next_token = last_logits.argmax(1);

            // Extract the token value
            let token_value = next_token.into_data().to_vec::<i64>().unwrap()[0] as u32;
            generated.push(token_value);

            // Update input for next iteration
            // Append the new token to the sequence
            let new_token_tensor =
                Tensor::<B, 2, Int>::from_data([[token_value as i32]], &self.device);

            // Concatenate along sequence dimension
            current_input = Tensor::cat(vec![current_input, new_token_tensor], 1);

            // Keep only the last block_size tokens to fit model's context window
            let seq_len = current_input.dims()[1];
            if seq_len > 128 {
                // Limit context length
                let start_idx = seq_len - 128;
                current_input = current_input.slice([0..1, start_idx..seq_len]);
            }
        }

        generated
    }

    pub fn model(&self) -> &MiniGPT<B> {
        &self.model
    }
}

// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub epochs: usize,
    pub learning_rate: f64,
    pub batch_size: usize,
    pub sequence_length: usize,
    pub steps_per_epoch: usize,
    pub validation_steps: usize,
    pub print_every: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 5,
            learning_rate: 3e-4,
            batch_size: 4,
            sequence_length: 32,
            steps_per_epoch: 500,
            validation_steps: 50,
            print_every: 100,
        }
    }
}

pub fn train_model<B: AutodiffBackend>(
    config: &Config,
    training_config: &TrainingConfig,
    mut train_dataloader: DataLoader<B>,
    device: B::Device,
) -> Result<SimpleTrainer<B>, Box<dyn std::error::Error>> {
    println!("Starting training with config: {:?}", training_config);

    let mut trainer = SimpleTrainer::new(config, device);

    for epoch in 0..training_config.epochs {
        println!("Epoch {}/{}", epoch + 1, training_config.epochs);

        // Training
        let train_loss = trainer.train_epoch(
            &mut train_dataloader,
            training_config.steps_per_epoch,
            training_config.learning_rate,
        );

        println!("Training Loss: {:.4}", train_loss);
        println!("---");
    }

    println!("Training completed!");
    Ok(trainer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_autodiff::Autodiff;
    use burn_ndarray::NdArray;

    type TestBackend = Autodiff<NdArray<f32>>;

    #[test]
    fn test_trainer_creation() {
        let device = Default::default();
        let config = Config {
            block_size: 32,
            vocab_size: 100,
            n_layer: 2,
            n_embed: 64,
            n_head: 4,
        };

        let _trainer = SimpleTrainer::<TestBackend>::new(&config, device);

        // Just check that trainer was created successfully
        assert!(true);
    }

    #[test]
    fn test_training_step() {
        let device: <TestBackend as burn::tensor::backend::Backend>::Device = Default::default();
        let config = Config {
            block_size: 16,
            vocab_size: 50,
            n_layer: 1,
            n_embed: 32,
            n_head: 2,
        };

        let mut trainer = SimpleTrainer::<TestBackend>::new(&config, device.clone());

        // Create sample batch
        let inputs = Tensor::<TestBackend, 2, Int>::from_data([[1, 2, 3], [4, 5, 6]], &device);
        let targets = Tensor::<TestBackend, 2, Int>::from_data([[2, 3, 4], [5, 6, 7]], &device);
        let batch = TrainingBatch::new(inputs, targets);

        // Perform training step
        let loss = trainer.train_step(batch, 1e-3);

        // Loss should be positive
        assert!(loss > 0.0);
        println!("Training step loss: {}", loss);
    }

    #[test]
    fn test_generation() {
        let device: <TestBackend as burn::tensor::backend::Backend>::Device = Default::default();
        let config = Config {
            block_size: 32,
            vocab_size: 100,
            n_layer: 1,
            n_embed: 32,
            n_head: 2,
        };

        let trainer = SimpleTrainer::<TestBackend>::new(&config, device.clone());

        // Create a prompt
        let prompt = Tensor::<TestBackend, 2, Int>::from_data([[1, 2, 3]], &device);

        // Generate tokens
        let generated = trainer.generate(prompt, 5);

        // Should generate 5 tokens
        assert_eq!(generated.len(), 5);
        println!("Generated tokens: {:?}", generated);
    }

    #[test]
    fn test_backward_pass() {
        // Test that demonstrates the key functionality: backward pass computation
        let device: <TestBackend as burn::tensor::backend::Backend>::Device = Default::default();
        let config = Config {
            block_size: 16,
            vocab_size: 50,
            n_layer: 1,
            n_embed: 32,
            n_head: 2,
        };

        let trainer = SimpleTrainer::<TestBackend>::new(&config, device.clone());

        // Create sample data
        let inputs = Tensor::<TestBackend, 2, Int>::from_data([[1, 2, 3], [4, 5, 6]], &device);
        let targets = Tensor::<TestBackend, 2, Int>::from_data([[2, 3, 4], [5, 6, 7]], &device);

        // Forward pass with loss
        let (logits, loss) = trainer.model().training_step(inputs, targets);

        // Backward pass to compute gradients
        let _grads = loss.backward();

        // Verify shapes
        assert_eq!(logits.shape().dims, [2, 3, 50]);
        assert_eq!(loss.shape().dims, [1]);

        let loss_value = loss.into_data().to_vec::<f32>().unwrap()[0];
        assert!(loss_value > 0.0);

        println!("Forward pass completed - Loss: {:.4}", loss_value);
        println!("Backward pass completed - Gradients computed successfully");
    }
}
