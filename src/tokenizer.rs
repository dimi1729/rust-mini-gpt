use tokenizers::Tokenizer;

fn tokenize(sentence: &str) -> Vec<i64> {
    let tokenizer = Tokenizer::new("gpt2");
    let tokens = tokenizer.encode(sentence, true).unwrap();

    return tokens.get_ids().iter().map(|id| *id as f64).collect();
}

fn decode(tokens: Vec<i64>) -> String {
    let tokenizer = Tokenizer::new("gpt2");
    let decoded = tokenizer.decode(tokens, true).unwrap();

    return decoded;
}
