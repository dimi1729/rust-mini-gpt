use tokenizers::tokenizer::{Result, Tokenizer};

pub fn tokenize(sentence: &str) -> Result<Vec<u32>> {
    let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None)?;
    let tokens = tokenizer.encode(sentence, false)?;

    return Ok(tokens.get_ids().iter().copied().collect());
}

pub fn decode(tokens: Vec<u32>) -> Result<String> {
    let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None)?;
    let decoded = tokenizer.decode(&tokens, true)?;

    return Ok(decoded);
}
