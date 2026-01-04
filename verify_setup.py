import torch
from datasets import load_dataset
from transformers import CLIPProcessor
import sys
sys.path.append('/workspace')

from universal_compositional_embedder.models import UniversalCompositionalEmbedder
from universal_compositional_embedder.config import ModelConfig

def verify_setup():
    print("Verifying setup...")
    
    # Initialize our Universal Compositional Embedder model with minimal parameters
    config = ModelConfig(
        text_encoder_name="openai/clip-vit-base-patch32",
        image_encoder_name="openai/clip-vit-base-patch32",
        video_encoder_name="openai/clip-vit-base-patch32",
        audio_encoder_name="openai/clip-vit-base-patch32",
        embed_dim=512,
        compositional_dim=512,
        cone_dim=64,
        max_length=77
    )
    
    print("Creating model...")
    model = UniversalCompositionalEmbedder(config)
    
    print("Loading processor...")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    print("Loading dataset...")
    ds = load_dataset("nlphuji/flickr30k", streaming=True)
    
    print("Setup verified successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
if __name__ == "__main__":
    verify_setup()