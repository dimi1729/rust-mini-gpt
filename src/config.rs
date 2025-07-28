pub struct Config {
    block_size: i32,
    vocab_size: i32,
    n_layer: i8,
    n_head: i8,
    n_embed: i32,
}
impl Config {
    pub fn new(block_size: i32, vocab_size: i32, n_layer: i8, n_head: i8, n_embed: i32) -> Self {
        Self {
            block_size,
            vocab_size,
            n_layer,
            n_head,
            n_embed,
        }
    }
}
