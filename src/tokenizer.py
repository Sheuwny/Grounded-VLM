from transformers import AutoTokenizer

def get_tokenizer(model_name="microsoft/Phi-3-vision-128k-instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # 1000 location tokens add kar rahe hain (0 se 999)
    new_tokens = [f"<loc{i}>" for i in range(1000)]
    tokenizer.add_tokens(new_tokens)
    
    # Kuch special tokens
    special_tokens = ["<box>", "</box>", "<p>"]
    tokenizer.add_tokens(special_tokens)
    
    print(f"Tokenizer updated! New vocab size: {len(tokenizer)}")
    return tokenizer

if __name__ == "__main__":
    get_tokenizer()
