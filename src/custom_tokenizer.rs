use regex::Regex;
use std::collections::HashMap;

pub struct CustomTokenizer {
    original_vocab_size: i32,
    merge_dict: HashMap<i32, i32>,
    vocab: HashMap<i32, String>,
    final_vocab_size: Option<i32>,
    merge_order: Vec<(i32, i32)>,
}

impl CustomTokenizer {
    pub fn new() -> CustomTokenizer {
        let mut vocab: HashMap<i32, String> = (0..256)
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

    fn get_stats(&self, tokens: Vec<i32>) -> Vec<((i32, i32), i32)> {
        let mut counts: HashMap<(i32, i32), i32> = HashMap::new();

        for window in tokens.windows(2) {
            if let [first, second] = window {
                let pair = (*first, *second);
                *counts.entry(pair).or_insert(0) += 1;
            }
        }

        let mut result: Vec<((i32, i32), i32)> = counts.into_iter().collect();
        result.sort_by(|a, b| b.1.cmp(&a.1)); // sort decreasing

        return result;
    }

    fn merge_tokens(&self, pair: (i32, i32), tokens: Vec<i32>, new_id: i32) -> Vec<i32> {
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
            "gpt2" => r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
            "gpt4" => {
                r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"
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

    pub fn encode(&self, text: &str) -> Vec<i32> {
        if self.merge_order.is_empty() {
            panic!("Merge order is empty, train or load file first");
        }

        let mut chunk_tokens = text_to_bytes(text)
            .into_iter()
            .map(|chunk| {
                chunk
                    .into_iter()
                    .map(|byte| byte as i32)
                    .collect::<Vec<i32>>()
            })
            .collect::<Vec<Vec<i32>>>();

        for pair in &self.merge_order {
            if let Some(&new_id) = self.merge_dict.get(pair) {
                for i in 0..chunk_tokens.len() {
                    chunk_tokens[i] = self.merge_tokens(*pair, chunk_tokens[i].clone(), new_id);
                }
            }
        }

        return chunk_tokens.into_iter().flatten().collect();
    }

    pub fn decode(&self, tokens: Vec<i32>) -> String {
        if self.vocab.is_empty() {
            panic!("Vocab is empty, train or load file first");
        }

        // Map tokens to bytes via vocab
        let bytes: Vec<u8> = tokens
            .iter()
            .filter_map(|&tokens_id| self.vocab.get(&token_id))
            .flat_map(|token_str| token_str.as_bytes())
            .collect();

        return String::from_utf8_lossy(&bytes).to_string();
    }

    pub fn train(&mut self, text: &str, final_vocab_size: i32) {
        // Train BPE encoder on given text
        self.final_vocab_size = final_vocab_size;

        let chunk_tokens = self.text_to_bytes(text);
        let num_merges = self.final_vocab_size - self.original_vocab_size;

        if !self.merge_order.is_empty() || !self.merge_dict.is_empty() {
            println!("Merge order or merge dictionary is not empty, train or load file first");
        }

        println!("Training BPE with {} merges", num_merges);

        for i in 0..num_merges {
            // Chunk to respect word boundaries
            let chunk_stats: HashMap<(i32, i32), i32> = chunk_tokens
                .iter()
                .flat_map(|chunk| self.get_stats(chunk.clone()))
                .fold(HashMap::new(), |mut acc, (pair, count)| {
                    *acc.entry(pair).or_insert(0) += count;
                    acc
                });

            let top_pair: Option<((i32, i32), i32)> = chunk_stats
                .iter()
                .max_by_key(|&(pair, count)| count)
                .map(|&pair, &count| (pair, count))
                .expect("No pairs found");

            if top_pair.is_none() {
                println!("No more pairs to merge");
                break;
            }
            if top_pair[1] <= 2 {
                println!(
                    "Stop merging early since top hit only has {} occurences",
                    top_pair[1]
                );
            }

            let pair: (i32, i32) = top_pair[0];
            let new_id: i32 = self.original_vocab_size + i;

            for j in 0..chunk_tokens.len() {
                chunk_tokens[j] = self.merge_tokens(*pair, chunk_tokens[j].clone(), new_id);
            }

            self.merge_order.push(pair);
            self.merge_dict.insert(pair, new_id);
        }

        // Build vocab
        for (&(first, second), &new_id) in &self.merge_dict {
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
