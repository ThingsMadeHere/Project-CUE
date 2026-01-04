import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel, CLIPTokenizer, CLIPTextModel
import numpy as np
from PIL import Image
import os
import json
from tqdm import tqdm
import sys
sys.path.append('/workspace')

from universal_compositional_embedder.models import UniversalCompositionalEmbedder
from universal_compositional_embedder.config import ModelConfig

class Flickr30kDataset(Dataset):
    def __init__(self, dataset, processor, max_length=77):
        self.dataset = dataset
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = Image.open(item['image']).convert('RGB')
        caption = item['caption']
        
        # Process image and text
        inputs = self.processor(
            text=caption,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        
        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0)
        }

def train_model():
    # Load the Flickr30k dataset
    print("Loading Flickr30k dataset...")
    ds = load_dataset("nlphuji/flickr30k")
    
    # Initialize our Universal Compositional Embedder model
    config = ModelConfig(
        text_encoder_name="openai/clip-vit-base-patch32",
        image_encoder_name="openai/clip-vit-base-patch32",
        video_encoder_name="openai/clip-vit-base-patch32",  # Using same architecture for video
        audio_encoder_name="openai/clip-vit-base-patch32",  # Using same architecture for audio
        embed_dim=512,
        compositional_dim=512,
        cone_dim=64,
        max_length=77
    )
    
    model = UniversalCompositionalEmbedder(config)
    
    # Initialize processor
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create datasets
    train_dataset = Flickr30kDataset(ds['train'], processor)
    val_dataset = Flickr30kDataset(ds['test'], processor)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Set model to training mode
    model.train()
    
    # Define optimizer
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.02)
    
    # Training loop
    num_epochs = 3
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
            # Move batch to device
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass - get embeddings from both image and text encoders
            image_embeddings = model.encode_image(pixel_values)
            text_embeddings = model.encode_text(input_ids, attention_mask)
            
            # Compute contrastive loss to align image and text embeddings
            # Using a contrastive loss function to make image and text embeddings similar
            logits_per_image = image_embeddings @ text_embeddings.t()
            logits_per_text = text_embeddings @ image_embeddings.t()
            
            labels = torch.arange(len(pixel_values), device=device)
            
            loss_img = nn.CrossEntropyLoss()(logits_per_image, labels)
            loss_txt = nn.CrossEntropyLoss()(logits_per_text, labels)
            
            loss = (loss_img + loss_txt) / 2
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Print loss every 100 batches
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")):
                pixel_values = batch['pixel_values'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                # Get embeddings from both image and text encoders
                image_embeddings = model.encode_image(pixel_values)
                text_embeddings = model.encode_text(input_ids, attention_mask)
                
                # Compute contrastive loss
                logits_per_image = image_embeddings @ text_embeddings.t()
                logits_per_text = text_embeddings @ image_embeddings.t()
                
                labels = torch.arange(len(pixel_values), device=device)
                
                loss_img = nn.CrossEntropyLoss()(logits_per_image, labels)
                loss_txt = nn.CrossEntropyLoss()(logits_per_text, labels)
                
                loss = (loss_img + loss_txt) / 2
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # Save the trained model
    model.save_pretrained("./flickr30k_universal_embedder")
    print("Model saved to ./flickr30k_universal_embedder")

if __name__ == "__main__":
    train_model()