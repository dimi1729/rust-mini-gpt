use crate::batcher::GPTBatcher;
use crate::config::Config;
use crate::dataset::GPTDataset;
use crate::model::{MiniGPT, TrainingBatch};
use burn::{
    config::Config as BurnConfig,
    data::{dataloader::DataLoaderBuilder, dataset::Dataset},
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::{backend::AutodiffBackend, Int, Tensor},
    train::{
        metric::{AccuracyMetric, LossMetric},
        ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep,
    },
};

// Training configuration using Burn's Config system
#[derive(BurnConfig)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 5)]
    pub num_epochs: usize,
    #[config(default = 4)]
    pub batch_size: usize,
    #[config(default = 32)]
    pub sequence_length: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 3e-4)]
    pub learning_rate: f64,
    #[config(default = 500)]
    pub steps_per_epoch: usize,
    #[config(default = 50)]
    pub validation_steps: usize,
}

// Model configuration
#[derive(BurnConfig)]
pub struct ModelConfig {
    #[config(default = 128)]
    pub block_size: usize,
    #[config(default = 50257)]
    pub vocab_size: usize,
    #[config(default = 12)]
    pub n_layer: usize,
    #[config(default = 768)]
    pub n_embed: usize,
    #[config(default = 12)]
    pub n_head: usize,
}

impl ModelConfig {
    pub fn create_new(vocab_size: usize, n_embed: usize) -> Self {
        let mut config = ModelConfig::new();
        config.vocab_size = vocab_size;
        config.n_embed = n_embed;
        config.block_size = 128;
        config.n_layer = 6;
        config.n_head = 8;
        config
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> MiniGPT<B> {
        let config = Config {
            block_size: self.block_size,
            vocab_size: self.vocab_size,
            n_layer: self.n_layer,
            n_embed: self.n_embed,
            n_head: self.n_head,
        };
        MiniGPT::new(&config, device)
    }
}

impl Default for TrainingConfig {
    fn default() -> Self {
        let mut config =
            TrainingConfig::new(ModelConfig::create_new(50257, 768), AdamConfig::new());
        config.num_epochs = 5;
        config.batch_size = 4;
        config.sequence_length = 32;
        config.num_workers = 1;
        config.seed = 42;
        config.learning_rate = 3e-4;
        config.steps_per_epoch = 500;
        config.validation_steps = 50;
        config
    }
}

// Add forward_classification method to MiniGPT for training compatibility
impl<B: Backend> MiniGPT<B> {
    pub fn forward_classification(
        &self,
        inputs: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
    ) -> ClassificationOutput<B> {
        let logits = self.forward(inputs);
        let (_, loss) = self.forward_with_loss_from_logits(logits.clone(), targets.clone());

        // For language modeling, we need to reshape logits and targets for the classification output
        let [batch_size, seq_len, vocab_size] = logits.dims();
        let logits_flat = logits.reshape([batch_size * seq_len, vocab_size]);
        let targets_flat = targets.reshape([batch_size * seq_len]);

        ClassificationOutput::new(loss, logits_flat, targets_flat)
    }

    fn forward_with_loss_from_logits(
        &self,
        logits: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
    ) -> (Tensor<B, 3>, Tensor<B, 1>) {
        let loss = self.calculate_loss_from_logits(logits.clone(), targets);
        (logits, loss)
    }

    fn calculate_loss_from_logits(
        &self,
        logits: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
    ) -> Tensor<B, 1> {
        let (batch_size, seq_len, vocab_size) = logits.dims().into();

        // Reshape logits from [B, T, V] to [B*T, V]
        let logits_flat = logits.clone().reshape([batch_size * seq_len, vocab_size]);

        // Reshape targets from [B, T] to [B*T]
        let targets_flat = targets.reshape([batch_size * seq_len]);

        // Calculate cross entropy loss
        let loss_config = burn::nn::loss::CrossEntropyLossConfig::new();
        let loss_fn = loss_config.init(&logits.device());

        loss_fn.forward(logits_flat, targets_flat)
    }
}

// Implement TrainStep for the model
impl<B: AutodiffBackend> TrainStep<TrainingBatch<B>, ClassificationOutput<B>> for MiniGPT<B> {
    fn step(&self, batch: TrainingBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.inputs, batch.targets);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

// Implement ValidStep for the inner (non-autodiff) model
impl<B: Backend> ValidStep<TrainingBatch<B>, ClassificationOutput<B>> for MiniGPT<B> {
    fn step(&self, batch: TrainingBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.inputs, batch.targets)
    }
}

// Create artifact directory helper
fn create_artifact_dir(artifact_dir: &str) {
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).expect("Should be able to create artifact dir");
}

pub fn train_with_burn_tui<B: AutodiffBackend>(
    tokens: Vec<u32>,
    vocab_size: usize,
    device: B::Device,
    artifact_dir: &str,
) -> Result<MiniGPT<B>, Box<dyn std::error::Error>>
where
    B::InnerBackend: Clone,
    <B as AutodiffBackend>::InnerBackend: Backend,
{
    println!("Start training");

    // Create artifact directory
    create_artifact_dir(artifact_dir);

    // Create configuration
    let mut config =
        TrainingConfig::new(ModelConfig::create_new(vocab_size, 256), AdamConfig::new());
    config.batch_size = 8;
    config.sequence_length = 64;
    config.num_epochs = 3;
    config.learning_rate = 1e-3;

    // Save config
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    // Set random seed
    B::seed(config.seed);

    println!("Creating datasets from {} tokens...", tokens.len());

    // Create training dataset with AutodiffBackend
    let train_dataset =
        GPTDataset::<B>::new(tokens.clone(), config.sequence_length, device.clone());

    // Create validation dataset with regular backend (no autodiff needed for validation)
    // We'll use the same tokens for simplicity, but in practice you'd use separate validation data
    let valid_tokens = tokens.clone();
    // For InnerBackend, we typically use the same device but need to ensure type compatibility
    let valid_device: <B::InnerBackend as Backend>::Device = device.clone();
    let valid_dataset =
        GPTDataset::<B::InnerBackend>::new(valid_tokens, config.sequence_length, valid_device);

    println!(
        "Created {} training samples and {} validation samples",
        Dataset::len(&train_dataset),
        Dataset::len(&valid_dataset)
    );

    // Create batchers for both backends
    let train_batcher = GPTBatcher::<B>::new(device.clone());
    let valid_device_for_batcher: <B::InnerBackend as Backend>::Device = device.clone();
    let valid_batcher = GPTBatcher::<B::InnerBackend>::new(valid_device_for_batcher);

    // Create dataloaders with proper backend types
    let train_dataloader = DataLoaderBuilder::new(train_batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_dataset);

    let valid_dataloader = DataLoaderBuilder::new(valid_batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(valid_dataset);

    // Init model
    let model = config.model.init::<B>(&device);

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary() // 🎉 This enables the beautiful TUI!
        .build(model, config.optimizer.init(), config.learning_rate);

    let model_trained = learner.fit(train_dataloader, valid_dataloader);

    // Save final model
    model_trained
        .clone()
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");

    println!("Training completed successfully!");
    println!(
        "Model saved to: {}/model, logs saved to: {}/",
        artifact_dir, artifact_dir
    );

    Ok(model_trained)
}

// Simple generation function for testing
pub fn generate<B: AutodiffBackend>(
    model: &MiniGPT<B>,
    prompt: Tensor<B, 2, Int>,
    max_new_tokens: usize,
    device: &B::Device,
) -> Vec<u32> {
    use burn::module::AutodiffModule;

    let mut generated = Vec::new();
    let mut current_input = prompt;
    let model = model.valid();

    for _ in 0..max_new_tokens {
        // Forward pass to get logits
        let logits = model.forward(current_input.clone().inner());

        // Get the last token's logits [B, vocab_size]
        let seq_len = logits.dims()[1];
        let last_logits = logits
            .clone()
            .slice([0..1, (seq_len - 1)..seq_len, 0..logits.dims()[2]]);
        let last_logits = last_logits.squeeze::<2>(1); // [1, vocab_size]

        // For simplicity, just take argmax (greedy sampling)
        let next_token = last_logits.argmax(1);

        // Extract the token value
        let token_value = next_token.into_data().to_vec::<i64>().unwrap()[0] as u32;
        generated.push(token_value);

        // Update input for next iteration
        let new_token_tensor = Tensor::<B, 2, Int>::from_data([[token_value as i32]], device);

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

#[cfg(test)]
mod tests {
    use super::*;
    use burn_autodiff::Autodiff;
    use burn_ndarray::NdArray;

    type TestBackend = Autodiff<NdArray<f32>>;

    #[test]
    fn test_training_config() {
        let config = TrainingConfig::default();
        assert_eq!(config.num_epochs, 5);
        assert_eq!(config.batch_size, 4);
        assert!(config.learning_rate > 0.0);
    }

    #[test]
    fn test_model_config() {
        let config = ModelConfig::create_new(1000, 256);
        assert_eq!(config.vocab_size, 1000);
        assert_eq!(config.n_embed, 256);

        let device = Default::default();
        let model = config.init::<TestBackend>(&device);

        // Test model creation
        let input = Tensor::<TestBackend, 2, Int>::from_data([[1, 2, 3]], &device);
        let output = model.forward(input.inner());
        assert_eq!(output.shape().dims[2], 1000); // vocab_size
    }

    #[test]
    fn test_classification_output() {
        let device: <TestBackend as burn::tensor::backend::Backend>::Device = Default::default();
        let loss = Tensor::<TestBackend, 1>::from_data([2.5], &device);
        let logits = Tensor::<TestBackend, 2>::from_data([[1.0, 2.0, 3.0]], &device);
        let targets = Tensor::<TestBackend, 1, Int>::from_data([1], &device);

        let _output = ClassificationOutput::new(loss.clone(), logits, targets);

        // Should be able to extract loss value
        let loss_value = loss.into_data().to_vec::<f32>().unwrap()[0];
        assert!((loss_value - 2.5).abs() < 1e-6);
    }
}
