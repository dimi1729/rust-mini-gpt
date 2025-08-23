use rust_mini_gpt::custom_tokenizer::CustomTokenizer;
use rust_mini_gpt::training::{generate, train_with_burn_tui};

use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Int, Tensor};
use burn_autodiff::Autodiff;
use burn_ndarray::NdArray;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Mini GPT Training");

    // Load and prepare data
    let text = fs::read_to_string("data/tiny_shakespeare.txt")
        .expect("Failed to read tiny_shakespeare.txt");

    // Use a subset for this example
    let text = text.chars().take(10000).collect::<String>();
    println!("Text loaded: {} characters", text.len());

    // Load the tokenizer
    let save_path = "data/tokenizer.ron";
    let tokenizer = CustomTokenizer::load(save_path).expect("Failed to load tokenizer");
    let tokens = tokenizer.encode(&text);
    let vocab_size = tokenizer.final_vocab_size.unwrap() as usize;
    println!(
        "Tokenized to {} tokens, using vocab size of {}",
        tokens.len(),
        vocab_size
    );

    // Use autodiff backend for training
    type TrainingBackend = Autodiff<NdArray<f32>>;
    let device: <TrainingBackend as Backend>::Device = Default::default();

    // ( if using GPU)
    // type TrainingBackend = Autodiff<Wgpu<f32, i32>>;
    // let device = WgpuDevice::default();

    // Start training with Burn's LearnerBuilder and TUI
    println!("\nStarting training with Burn's LearnerBuilder TUI");

    let artifact_dir = "/tmp/mini-gpt-burn-tui";

    match train_with_burn_tui::<TrainingBackend>(
        tokens.clone(),
        vocab_size,
        device.clone(),
        artifact_dir,
    ) {
        Ok(trained_model) => {
            println!("Training completed");

            // Test the trained model with generation
            println!("\nTesting text generation");
            test_generation(trained_model, &tokenizer, &device)?;
        }
        Err(e) => {
            eprintln!("Training failed: {}", e);
            return Err(e);
        }
    }

    Ok(())
}

fn test_generation<B: AutodiffBackend>(
    model: rust_mini_gpt::model::MiniGPT<B>,
    tokenizer: &CustomTokenizer,
    device: &B::Device,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Test text samples");

    let prompts = vec!["HAMLET:", "To be or not to be", "ROMEO:"];

    for prompt in prompts {
        println!("\nPrompt: \"{}\"", prompt);

        let prompt_tokens = tokenizer.encode(prompt);
        if prompt_tokens.is_empty() {
            panic!("Empty prompt tokens");
        }

        let prompt_tensor = Tensor::<B, 2, Int>::from_data(
            burn::tensor::TensorData::new(
                prompt_tokens
                    .iter()
                    .map(|&x| x as i32)
                    .collect::<Vec<i32>>(),
                [1, prompt_tokens.len()],
            ),
            device,
        );

        // Generate with the model (we need to add a generation method)
        let generated_tokens = generate(&model, prompt_tensor, 20, device);
        let generated_text = tokenizer.decode(generated_tokens);

        println!("Generated: \"{}{}\"", prompt, generated_text);
    }

    Ok(())
}
