from transformers import AutoTokenizer

def get_tokenizer(tokenizer_path, legacy=False, trust_remote_code=False):
    """Get tokenizer from path"""
    print(f"Loading tokenizer from: {tokenizer_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=trust_remote_code
        )
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        raise
