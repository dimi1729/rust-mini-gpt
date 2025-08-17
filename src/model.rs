use crate::config::Config;
use crate::primitives::{Mlp, SelfAttention};
use burn::{
    module::Module,
    nn::{
        Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig,
        loss::CrossEntropyLossConfig,
    },
    tensor::{
        Int, Tensor,
        backend::{AutodiffBackend, Backend},
    },
};

#[derive(Module, Debug)]
pub struct Block<B: Backend> {
    ln_1: LayerNorm<B>,
    ln_2: LayerNorm<B>,
    attn: SelfAttention<B>,
    mlp: Mlp<B>,
}

impl<B: Backend> Block<B> {
    pub fn new(config: &Config, device: &B::Device) -> Self {
        return Self {
            ln_1: LayerNormConfig::new(config.n_embed).init(device),
            ln_2: LayerNormConfig::new(config.n_embed).init(device),
            attn: SelfAttention::new(config, device),
            mlp: Mlp::new(config, device),
        };
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        // Residual connections with layer norm
        let x = input.clone() + self.attn.forward(self.ln_1.forward(input.clone()));
        let x = x.clone() + self.mlp.forward(self.ln_2.forward(x));
        return x;
    }
}

#[derive(Module, Debug)]
pub struct MiniGPT<B: Backend> {
    wte: Embedding<B>,
    wpe: Embedding<B>,
    blocks: Vec<Block<B>>,
    ln_f: LayerNorm<B>,
    lm_head: Linear<B>,
}

impl<B: Backend> MiniGPT<B> {
    pub fn new(config: &Config, device: &B::Device) -> Self {
        // Use a vector instead of nn.ModuleDict
        let mut blocks = Vec::new();
        for _ in 0..config.n_layer {
            blocks.push(Block::new(config, device));
        }

        Self {
            wte: EmbeddingConfig::new(config.vocab_size, config.n_embed).init(device),
            wpe: EmbeddingConfig::new(config.block_size, config.n_embed).init(device),
            blocks,
            ln_f: LayerNormConfig::new(config.n_embed).init(device),
            lm_head: LinearConfig::new(config.n_embed, config.vocab_size).init(device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let (batch_size, seq_len) = (input.dims()[0], input.dims()[1]);

        // Token embeddings: [B, T] -> [B, T, C]
        let tok_emb = self.wte.forward(input.clone());

        // Position embeddings: [T] -> [B, T, C]
        let pos = Tensor::arange(0..seq_len as i64, &input.device()).unsqueeze::<2>(); // [T, 1]
        let pos_emb = self.wpe.forward(pos).unsqueeze::<3>(); // [1, T, C]

        // Add embeddings
        let mut x = tok_emb.clone() + pos_emb.expand([batch_size, seq_len, tok_emb.dims()[2]]);

        // Pass through transformer blocks
        for block in &self.blocks {
            x = block.forward(x);
        }

        // Final layer norm
        x = self.ln_f.forward(x);

        let logits = self.lm_head.forward(x);

        return logits;
    }

    pub fn forward_with_loss(
        &self,
        input: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
    ) -> (Tensor<B, 3>, Tensor<B, 1>) {
        let logits = self.forward(input);
        let loss = self.calculate_loss(logits.clone(), targets);
        (logits, loss)
    }

    fn calculate_loss(&self, logits: Tensor<B, 3>, targets: Tensor<B, 2, Int>) -> Tensor<B, 1> {
        let (batch_size, seq_len, vocab_size) = logits.dims().into();

        // Reshape logits from [B, T, V] to [B*T, V]
        let logits_flat = logits.clone().reshape([batch_size * seq_len, vocab_size]);

        // Reshape targets from [B, T] to [B*T]
        let targets_flat = targets.reshape([batch_size * seq_len]);

        // Calculate cross entropy loss
        let loss_config = CrossEntropyLossConfig::new();
        let loss_fn = loss_config.init(&logits.device());

        loss_fn.forward(logits_flat, targets_flat)
    }
}

// For training with autodiff backend
impl<B: AutodiffBackend> MiniGPT<B> {
    pub fn training_step(
        &self,
        input: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
    ) -> (Tensor<B, 3>, Tensor<B, 1>) {
        self.forward_with_loss(input, targets)
    }
}

// Training batch structure
#[derive(Clone, Debug)]
pub struct TrainingBatch<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
}

impl<B: Backend> TrainingBatch<B> {
    pub fn new(inputs: Tensor<B, 2, Int>, targets: Tensor<B, 2, Int>) -> Self {
        Self { inputs, targets }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_autodiff::Autodiff;
    use burn_ndarray::NdArray;

    #[test]
    fn test_mini_gpt_output_dimensions() {
        type Backend = NdArray<f32>;
        let device = Default::default();

        let n_layer = 2;
        let n_embed = 128;
        let n_head = 4;
        let vocab_size = 1000;
        let block_size = 64;

        let config = Config {
            block_size,
            vocab_size,
            n_layer,
            n_embed,
            n_head,
        };

        let model = MiniGPT::new(&config, &device);

        // Create input: batch_size=2, seq_len=8
        let data: [[i32; 8]; 2] = [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]];

        let input = Tensor::<Backend, 2, Int>::from_data(data, &device);

        let output = model.forward(input);

        // Check output shape: [batch_size=2, seq_len=8, vocab_size=1000]
        assert_eq!(output.shape().dims, [2, 8, 1000]);
    }

    #[test]
    fn test_mini_gpt_loss_calculation() {
        type Backend = Autodiff<NdArray<f32>>;
        let device = Default::default();

        let config = Config {
            block_size: 32,
            vocab_size: 100,
            n_layer: 1,
            n_embed: 64,
            n_head: 2,
        };

        let model = MiniGPT::new(&config, &device);

        // Create input and target tensors
        let inputs_data = [[1, 2, 3, 4], [5, 6, 7, 8]];
        let targets_data = [[2, 3, 4, 5], [6, 7, 8, 9]];

        let inputs = Tensor::<Backend, 2, Int>::from_data(inputs_data, &device);
        let targets = Tensor::<Backend, 2, Int>::from_data(targets_data, &device);

        let (logits, loss) = model.forward_with_loss(inputs, targets);

        // Check shapes
        assert_eq!(logits.shape().dims, [2, 4, 100]);
        assert_eq!(loss.shape().dims, [1]);

        // Loss should be positive
        let loss_value = loss.into_data().to_vec::<f32>().unwrap()[0];
        assert!(loss_value > 0.0);
    }

    #[test]
    fn test_training_step_with_gradients() {
        type Backend = Autodiff<NdArray<f32>>;
        let device = Default::default();

        let config = Config {
            block_size: 16,
            vocab_size: 50,
            n_layer: 1,
            n_embed: 32,
            n_head: 2,
        };

        let model = MiniGPT::new(&config, &device);

        // Create training batch
        let inputs_data = [[1, 2, 3], [4, 5, 6]];
        let targets_data = [[2, 3, 4], [5, 6, 7]];

        let inputs = Tensor::<Backend, 2, Int>::from_data(inputs_data, &device);
        let targets = Tensor::<Backend, 2, Int>::from_data(targets_data, &device);

        // Execute training step
        let (logits, loss) = model.training_step(inputs, targets);

        // Check that we have a valid loss
        let loss_value = loss.clone().into_data().to_vec::<f32>().unwrap()[0];
        assert!(loss_value > 0.0);

        // Check output shapes
        assert_eq!(logits.shape().dims, [2, 3, 50]);
        assert_eq!(loss.shape().dims, [1]);

        // Test backward pass
        let _grads = loss.backward();

        // Verify gradients were computed (this should not panic)
        println!("Backward pass completed successfully");
        println!("Loss: {:.4}", loss_value);
    }
}
