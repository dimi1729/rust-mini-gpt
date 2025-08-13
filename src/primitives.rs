use crate::config::Config;
use burn::{
    module::Module,
    nn::{Gelu, Linear, LinearConfig},
    tensor::{Tensor, activation::softmax, backend::Backend},
};

#[derive(Module, Debug)]
pub struct SelfAttention<B: Backend> {
    qkv_attn: Linear<B>,
    c_proj: Linear<B>,
    n_embed: usize,
    n_head: usize,
}

impl<B: Backend> SelfAttention<B> {
    pub fn new(config: &Config, device: &B::Device) -> Self {
        return Self {
            // qkv attn is basically q, k, and v tensors concatenated
            qkv_attn: LinearConfig::new(config.n_embed, config.n_embed * 3).init(device),
            c_proj: LinearConfig::new(config.n_embed, config.n_embed).init(device),
            n_embed: config.n_embed,
            n_head: config.n_head,
        };
    }

    /// Forward pass through the self-attention layer
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let shape = input.shape(); // [B, L, C]
        let batch_size = shape.dims[0];
        let seq_len = shape.dims[1];
        let n_embed = shape.dims[2];

        // Get q, k, v projections: [B, L, C] -> [B, L, 3C]
        let qkv = self.qkv_attn.forward(input);

        // Split into q, k, v: [B, L, 3C] -> 3 x [B, L, C]
        let qkv_splits = qkv.chunk(3, 2);
        let q = qkv_splits[0].clone();
        let k = qkv_splits[1].clone();
        let v = qkv_splits[2].clone();

        // Reshape for multi-head attention: [B, L, C] -> [B, L, nh, hs] -> [B, nh, L, hs]
        let head_size = n_embed / self.n_head;
        let q = q
            .reshape([batch_size, seq_len, self.n_head, head_size])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch_size, seq_len, self.n_head, head_size])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch_size, seq_len, self.n_head, head_size])
            .swap_dims(1, 2);

        // Scaled dot-product attention
        // attn = (q @ k.T) * (1.0 / sqrt(head_size))
        let k_transposed = k.swap_dims(2, 3); // [B, nh, hs, L]
        let mut attn = q.matmul(k_transposed); // [B, nh, L, L]

        // Scale by sqrt(head_size) as per gpt2 paper
        let scale = 1.0 / (head_size as f32).sqrt();
        attn = attn * scale;

        // TODO: Apply causal mask here
        // attn = attn.masked_fill(self.mask[:,:,:L,:L] == 0, float('-inf'))

        // Apply softmax
        attn = softmax(attn, 3); // softmax over last dimension (L)

        // [B, nh, L, L] @ [B, nh, L, hs] -> [B, nh, L, hs]
        let y = attn.matmul(v);

        // [B, nh, L, hs] -> [B, L, nh, hs] -> [B, L, C]
        let y = y.swap_dims(1, 2).reshape([batch_size, seq_len, n_embed]);

        // Final projection
        self.c_proj.forward(y)
    }
}

#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    c_fc: Linear<B>,
    c_proj: Linear<B>,
    activation: Gelu,
}

impl<B: Backend> Mlp<B> {
    pub fn new(config: &Config, device: &B::Device) -> Self {
        return Self {
            c_fc: LinearConfig::new(config.n_embed, config.n_embed * 4).init(device),
            c_proj: LinearConfig::new(config.n_embed * 4, config.n_embed).init(device),
            activation: Gelu::new(),
        };
    }

    /// Forward pass through the MLP
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.c_fc.forward(input);
        let x = self.activation.forward(x);
        let x = self.c_proj.forward(x);
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    #[test]
    fn test_self_attention() {
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

        let attention = SelfAttention::new(&config, &device);

        let batch_size = 2;
        let seq_len = 64;

        // Create input: [batch_size, seq_len, n_embed]
        let input = Tensor::<Backend, 3>::random(
            [batch_size, seq_len, n_embed],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        // Forward pass
        let output = attention.forward(input);

        // Check output shape matches input
        assert_eq!(output.shape().dims, [batch_size, seq_len, n_embed]);
    }

    #[test]
    fn test_mlp() {
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

        let seq_len = 10;
        let batch_size = 2;

        let mlp = Mlp::new(&config, &device);

        // Create input: [batch_size, seq_len, n_embed]
        let input = Tensor::<Backend, 3>::random(
            [batch_size, seq_len, n_embed],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        // Forward pass
        let output = mlp.forward(input);

        // Check output shape matches input
        assert_eq!(output.shape().dims, [batch_size, seq_len, n_embed]);
    }
}
