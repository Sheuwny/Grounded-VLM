import torch
import torch.nn as nn
from transformers import AutoModelForVision2Seq

class GroundedVLM(nn.Module):
    def __init__(self, model_name="microsoft/Phi-3-vision-128k-instruct", vocab_size=None):
        super().__init__()
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name, 
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        # Agar naye tokens add kiye hain, toh embedding resize karo
        if vocab_size:
            self.model.resize_token_embeddings(vocab_size)
        
    def forward(self, images, input_ids, labels):
        return self.model(images=images, input_ids=input_ids, labels=labels)

print("Model architecture updated with embedding resizing!")
