use rust_mini_gpt::config::Config;
use rust_mini_gpt::custom_tokenizer::CustomTokenizer;
use rust_mini_gpt::dataloader::DataLoader;
use rust_mini_gpt::model::{MiniGPT, TrainingBatch};
use rust_mini_gpt::training::SimpleTrainer;

use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Int, Tensor};
use burn_autodiff::Autodiff;
use burn_ndarray::NdArray;
use std::fs;

fn main() {
    let text = fs::read_to_string("data/tiny_shakespeare.txt")
        .expect("Failed to read tiny_shakespeare.txt");

    // Use a subset for this example
    let text = text.chars().take(10000).collect::<String>();
    println!("Text loaded: {} characters", text.len());

    // Load the tokenizer
    let save_path = "data/tokenizer.ron";
    let tokenizer = CustomTokenizer::load(save_path).expect("Failed to load tokenizer");
    let tokens = tokenizer.encode(&text);
    println!("Tokenized to {} tokens", tokens.len());

    // Use autodiff backend for training
    type TrainingBackend = Autodiff<NdArray<f32>>;
    let device: <TrainingBackend as burn::tensor::backend::Backend>::Device = Default::default();

    // Model configuration
    let config = Config::new(32, tokenizer.final_vocab_size.unwrap() as usize, 2, 4, 64);

    println!("Model config: {:?}", config);

    // Create dataloader
    let mut dataloader = DataLoader::<TrainingBackend>::new(device.clone(), tokens, 2, 16, 0, 1);

    // Create trainer
    let mut trainer = SimpleTrainer::<TrainingBackend>::new(&config, device.clone());

    // Train model
    let num_steps = 10;
    let lr = 3e-4;
    for step in 1..(num_steps + 1) {
        let (inputs, targets) = dataloader.next_batch();
        let batch = TrainingBatch::new(inputs, targets);

        let loss = trainer.train_step(batch, lr);
        println!("Step {}: Loss = {:.4}", step, loss);
    }

    // Generate an output from the input prompt
    let prompt_text = "How are thou";
    let prompt_vec = tokenizer.encode(prompt_text);
    let prompt_tokens: Vec<i32> = prompt_vec.iter().map(|&x| x as i32).collect();
    let prompt_tensor = Tensor::<TrainingBackend, 2, Int>::from_data(
        burn::tensor::TensorData::new(prompt_tokens.clone(), [1, prompt_vec.len()]),
        &device,
    );

    let response = trainer.generate(prompt_tensor, 10);
    println!("Generated response: {:?}", tokenizer.decode(response));
}
