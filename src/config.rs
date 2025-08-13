#[derive(Debug, Clone)]
pub struct Config {
    pub block_size: usize,
    pub vocab_size: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_embed: usize,
}
impl Config {
    pub fn new(
        block_size: usize,
        vocab_size: usize,
        n_layer: usize,
        n_head: usize,
        n_embed: usize,
    ) -> Config {
        return Config {
            block_size,
            vocab_size,
            n_layer,
            n_head,
            n_embed,
        };
    }
}
