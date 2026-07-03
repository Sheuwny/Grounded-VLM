import torch
from src.model import GroundedVLM
from src.tokenizer import get_tokenizer
from src.dataset import get_data

def prepare_training():
    tokenizer = get_tokenizer()
    model = GroundedVLM(vocab_size=len(tokenizer))
    dataset = get_data()
    
    print("Everything is ready for training!")
    print(f"Model Vocab Size: {len(tokenizer)}")
    
    return model, tokenizer, dataset

if __name__ == "__main__":
    prepare_training()
