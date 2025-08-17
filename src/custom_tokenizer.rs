use regex::Regex;
use std::collections::HashMap;

pub struct CustomTokenizer {
    original_vocab_size: u32,
    merge_dict: HashMap<(u32, u32), u32>,
    vocab: HashMap<u32, String>,
    pub final_vocab_size: Option<u32>,
    merge_order: Vec<(u32, u32)>,
}

impl CustomTokenizer {
    pub fn new() -> CustomTokenizer {
        let vocab: HashMap<u32, String> = (0..256)
            .map(|idx| {
                let byte_char = char::from(idx as u8);
                (idx, byte_char.to_string())
            })
            .collect();

        CustomTokenizer {
            original_vocab_size: 256,
            merge_dict: HashMap::new(),
            vocab: vocab,
            final_vocab_size: None,
            merge_order: Vec::new(),
        }
    }

    fn get_stats(&self, tokens: Vec<u32>) -> Vec<((u32, u32), u32)> {
        let mut counts: HashMap<(u32, u32), u32> = HashMap::new();

        for window in tokens.windows(2) {
            if let [first, second] = window {
                let pair = (*first, *second);
                *counts.entry(pair).or_insert(0) += 1;
            }
        }

        let mut result: Vec<((u32, u32), u32)> = counts.into_iter().collect();
        result.sort_by(|a, b| b.1.cmp(&a.1)); // sort decreasing

        return result;
    }

    fn merge_tokens(&self, pair: (u32, u32), tokens: Vec<u32>, new_id: u32) -> Vec<u32> {
        let mut new_tokens = Vec::new();
        let mut i = 0;

        while i < tokens.len() {
            // Check if we can form the pair at current position
            if i < tokens.len() - 1 && tokens[i] == pair.0 && tokens[i + 1] == pair.1 {
                // Found the pair, replace with new_id
                new_tokens.push(new_id);
                i += 2; // Skip both tokens in the pair
            } else {
                // No pair found, keep the current token
                new_tokens.push(tokens[i]);
                i += 1;
            }
        }

        return new_tokens;
    }

    fn split_words(
        &self,
        text: &str,
        regex_type: Option<&str>,
    ) -> Result<Vec<String>, regex::Error> {
        let regex_type = regex_type.unwrap_or("gpt4");

        let pattern = match regex_type {
            "gpt2" => r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+",
            "gpt4" => {
                r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s+$|\s*[\r\n]|\s+"
            }
            _ => return Err(regex::Error::Syntax("Invalid regex_type".to_string())),
        };

        let regex = Regex::new(pattern)?;
        let matches: Vec<String> = regex
            .find_iter(text)
            .map(|m| m.as_str().to_string())
            .collect();

        return Ok(matches);
    }

    fn text_to_bytes(&self, text: &str) -> Vec<Vec<u32>> {
        let chunks = self.split_words(text, None).unwrap();
        let bytes: Vec<Vec<u32>> = chunks
            .iter()
            .map(|chunk| chunk.chars().map(|c| c as u32).collect())
            .collect();

        return bytes;
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        if self.merge_order.is_empty() {
            panic!("Merge order is empty, train or load file first");
        }

        let mut chunk_tokens = self
            .text_to_bytes(text)
            .into_iter()
            .map(|chunk| chunk.into_iter().collect::<Vec<u32>>())
            .collect::<Vec<Vec<u32>>>();

        for pair in &self.merge_order {
            if let Some(&new_id) = self.merge_dict.get(pair) {
                for i in 0..chunk_tokens.len() {
                    chunk_tokens[i] = self.merge_tokens(*pair, chunk_tokens[i].clone(), new_id);
                }
            }
        }

        return chunk_tokens.into_iter().flatten().collect();
    }

    pub fn decode(&self, tokens: Vec<u32>) -> String {
        if self.vocab.is_empty() {
            panic!("Vocab is empty, train or load file first");
        }

        // Map tokens to their string representations
        let mut result = String::new();
        for &token_id in &tokens {
            if let Some(token_str) = self.vocab.get(&token_id) {
                result.push_str(token_str);
            } else {
                eprintln!("Warning: Token {} not found in vocab", token_id);
                // For debugging, you might want to add the token ID as a placeholder
                result.push_str(&format!("[UNK:{}]", token_id));
            }
        }

        return result;
    }

    pub fn train(&mut self, text: &str, final_vocab_size: u32) {
        // Train BPE encoder on given text
        self.final_vocab_size = Some(final_vocab_size);

        let mut chunk_tokens = self.text_to_bytes(text);
        let num_merges = self.final_vocab_size.unwrap() - self.original_vocab_size;

        if !self.merge_order.is_empty() || !self.merge_dict.is_empty() {
            println!("Merge order or merge dictionary is not empty, train or load file first");
        }

        println!("Training BPE with {} merges", num_merges);

        for i in 0..num_merges {
            // Chunk to respect word boundaries
            let chunk_stats: HashMap<(u32, u32), u32> = chunk_tokens
                .iter()
                .flat_map(|chunk| self.get_stats(chunk.clone()))
                .fold(HashMap::new(), |mut acc, (pair, count)| {
                    *acc.entry(pair).or_insert(0) += count;
                    acc
                });

            let top_pair: ((u32, u32), u32) = chunk_stats
                .iter()
                .max_by_key(|&(_pair, count)| count)
                .map(|(&pair, &count)| (pair, count))
                .expect("No pairs found");

            if top_pair.1 <= 1 {
                println!(
                    "Stop merging early since top hit only has {} occurences",
                    top_pair.1
                );
                break;
            }

            let pair: (u32, u32) = top_pair.0;
            let new_id: u32 = self.original_vocab_size + i;

            for j in 0..chunk_tokens.len() {
                chunk_tokens[j] = self.merge_tokens(pair, chunk_tokens[j].clone(), new_id);
            }

            self.merge_order.push(pair);
            self.merge_dict.insert(pair, new_id);
        }

        // Build vocab in merge order to ensure dependencies exist
        for &(first, second) in &self.merge_order {
            if let Some(&new_id) = self.merge_dict.get(&(first, second)) {
                match (self.vocab.get(&first), self.vocab.get(&second)) {
                    (Some(first_token), Some(second_token)) => {
                        let combined = format!("{}{}", first_token, second_token);
                        self.vocab.insert(new_id, combined);
                    }
                    _ => {
                        eprintln!("Failed to merge tokens {}, {}", first, second);
                    }
                }
            }
        }
    }

    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        use std::fs;

        #[derive(serde::Serialize)]
        struct TokenizerData {
            original_vocab_size: u32,
            merge_dict: HashMap<(u32, u32), u32>,
            vocab: HashMap<u32, String>,
            final_vocab_size: Option<u32>,
            merge_order: Vec<(u32, u32)>,
        }

        let data = TokenizerData {
            original_vocab_size: self.original_vocab_size,
            merge_dict: self.merge_dict.clone(),
            vocab: self.vocab.clone(),
            final_vocab_size: self.final_vocab_size,
            merge_order: self.merge_order.clone(),
        };

        let ron_string = ron::to_string(&data)?;
        fs::write(path, ron_string)?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<CustomTokenizer, Box<dyn std::error::Error>> {
        use std::fs;

        #[derive(serde::Deserialize)]
        struct TokenizerData {
            original_vocab_size: u32,
            merge_dict: HashMap<(u32, u32), u32>,
            vocab: HashMap<u32, String>,
            final_vocab_size: Option<u32>,
            merge_order: Vec<(u32, u32)>,
        }

        let ron_string = fs::read_to_string(path)?;
        let data: TokenizerData = ron::from_str(&ron_string)?;

        Ok(CustomTokenizer {
            original_vocab_size: data.original_vocab_size,
            merge_dict: data.merge_dict,
            vocab: data.vocab,
            final_vocab_size: data.final_vocab_size,
            merge_order: data.merge_order,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_encode_decode() {
        let mut tokenizer = CustomTokenizer::new();

        // Train on a simple text
        let training_text = "hello world hello world test test";
        tokenizer.train(training_text, 270); // 256 + 14 merges

        // Test encoding and decoding
        let test_text = "hello world";
        let encoded = tokenizer.encode(test_text);
        let decoded = tokenizer.decode(encoded);

        assert_eq!(test_text, decoded);
    }

    #[test]
    fn test_round_trip_various_texts() {
        let mut tokenizer = CustomTokenizer::new();

        // Train on more complex text with more repetition to ensure good merges
        let training_text = "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. Hello world test.";
        tokenizer.train(training_text, 280);

        let test_cases = vec![
            "The quick brown fox",
            "jumps over the lazy dog",
            "Hello, world!",
            "Test 123",
            "Special chars: !@#$%",
        ];

        for test_text in test_cases {
            let encoded = tokenizer.encode(test_text);
            let decoded = tokenizer.decode(encoded);
            assert_eq!(test_text, decoded, "Failed round trip for: {}", test_text);
        }
    }

    #[test]
    fn test_save_and_load() {
        let mut tokenizer = CustomTokenizer::new();

        // Train the tokenizer
        let training_text = "hello world test save load functionality";
        tokenizer.train(training_text, 280);

        // Test text
        let test_text = "hello world test";
        let original_encoded = tokenizer.encode(test_text);

        // Save tokenizer
        let temp_path = "/tmp/test_tokenizer.ron";
        tokenizer.save(temp_path).expect("Failed to save tokenizer");

        // Load tokenizer
        let loaded_tokenizer = CustomTokenizer::load(temp_path).expect("Failed to load tokenizer");

        // Test that loaded tokenizer produces same results
        let loaded_encoded = loaded_tokenizer.encode(test_text);
        let loaded_decoded = loaded_tokenizer.decode(loaded_encoded.clone());

        assert_eq!(original_encoded, loaded_encoded);
        assert_eq!(test_text, loaded_decoded);

        // Clean up
        std::fs::remove_file(temp_path).ok();
    }
}
