mod custom_tokenizer;

use custom_tokenizer::CustomTokenizer;
use std::fs;

fn main() {
    let text = fs::read_to_string("data/tiny_shakespeare.txt")
        .expect("Failed to read tiny_shakespeare.txt");

    // for now, only train the tokenizer on a subset
    let text = text.chars().take(10000).collect::<String>();

    println!("Text loaded: {} characters", text.len());

    // Create and train the tokenizer
    let mut tokenizer = CustomTokenizer::new();
    let vocab_size = 400;

    println!("Training tokenizer with vocab size {}...", vocab_size);
    tokenizer.train(&text, vocab_size);

    println!("Training complete");

    // Test the tokenizer with a sample text
    let sample_text = "Hello world! This is a test.";
    println!("\nTesting with: '{}'", sample_text);

    let encoded = tokenizer.encode(sample_text);
    println!("Encoded: {:?}", encoded);

    let decoded = tokenizer.decode(encoded.clone());
    println!("Decoded: '{}'", decoded);

    let num_chars = sample_text.chars().count();
    let num_tokens = encoded.len();
    println!(
        "Went from {} characters to {} tokens",
        num_chars, num_tokens
    );

    let save_path = "data/tokenizer.ron";
    println!("Saving tokenizer to {}", save_path);
    tokenizer.save(save_path).expect("Failed to save tokenizer");

    // let loaded_tokenizer = CustomTokenizer::load(save_path).expect("Failed to load tokenizer");
}
