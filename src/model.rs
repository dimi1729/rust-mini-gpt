use crate::primitives::{Mlp, SelfAttention};
use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig},
    tensor::{Int, Tensor, backend::Backend},
};

#[derive(Module, Debug)]
pub struct Block<B: Backend> {
    ln_1: LayerNorm<B>,
    ln_2: LayerNorm<B>,
    attn: SelfAttention<B>,
    mlp: Mlp<B>,
}

impl<B: Backend> Block<B> {
    pub fn new(n_embed: usize, n_head: usize, device: &B::Device) -> Self {
        return Self {
            ln_1: LayerNormConfig::new(n_embed).init(device),
            ln_2: LayerNormConfig::new(n_embed).init(device),
            attn: SelfAttention::new(n_embed, n_head, device),
            mlp: Mlp::new(n_embed, device),
        };
    }

    pub fn forward(&self, mut input: Tensor<B, 3>) -> Tensor<B, 3> {
        input = self.ln_1.forward(input);
        input = self.attn.forward(input);
        input = self.ln_2.forward(input);
        input = self.mlp.forward(input);

        return input;
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
    pub fn new(
        n_layer: usize,
        n_embed: usize,
        n_head: usize,
        vocab_size: usize,
        block_size: usize,
        device: &B::Device,
    ) -> Self {
        // Use a vector instead of nn.ModuleDict
        let mut blocks = Vec::new();
        for _ in 0..n_layer {
            blocks.push(Block::new(n_embed, n_head, device));
        }

        Self {
            wte: EmbeddingConfig::new(vocab_size, n_embed).init(device),
            wpe: EmbeddingConfig::new(block_size, n_embed).init(device),
            blocks,
            ln_f: LayerNormConfig::new(n_embed).init(device),
            lm_head: LinearConfig::new(n_embed, vocab_size).init(device),
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
}

#[cfg(test)]
mod tests {
    use super::*;
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

        let model = MiniGPT::new(n_layer, n_embed, n_head, vocab_size, block_size, &device);

        // Create input: batch_size=2, seq_len=8
        let data: [[i32; 8]; 2] = [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]];

        let input = Tensor::<Backend, 2, Int>::from_data(data, &device);

        let output = model.forward(input);

        // Check output shape: [batch_size=2, seq_len=8, vocab_size=128]
        assert_eq!(output.shape().dims, [2, 8, 1000]);
    }
}
