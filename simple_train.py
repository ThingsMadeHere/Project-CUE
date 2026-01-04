import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPVisionModel, CLIPTextModel
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

class SimpleImageTextModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Use pre-trained CLIP encoders
        self.image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32", torch_dtype=torch.float32)
        
        # Projection layers to ensure same embedding dimension
        self.image_projection = nn.Linear(512, 512)
        self.text_projection = nn.Linear(512, 512)
    
    def encode_image(self, pixel_values):
        image_features = self.image_encoder(pixel_values=pixel_values).pooler_output
        image_embeddings = self.image_projection(image_features)
        # Normalize embeddings
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        return image_embeddings
    
    def encode_text(self, input_ids, attention_mask):
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use the pooled output (CLS token representation)
        text_features = text_outputs.pooler_output
        text_embeddings = self.text_projection(text_features)
        # Normalize embeddings
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        return text_embeddings

class Flickr30kDataset(Dataset):
    def __init__(self, dataset, processor, max_length=77, size_limit=500):  # Small size limit
        self.dataset = dataset
        self.processor = processor
        self.max_length = max_length
        self.size_limit = size_limit

    def __len__(self):
        return min(len(self.dataset), self.size_limit)

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
    print("Loading Flickr30k dataset (first 500 samples only)...")
    ds = load_dataset("nlphuji/flickr30k")
    
    # Initialize processor
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Initialize simple model
    model = SimpleImageTextModel()
    
    # Use CPU to save memory
    device = torch.device("cpu")
    model = model.to(device)
    
    # Create dataset with very small size
    train_dataset = Flickr30kDataset(ds['train'], processor, size_limit=100)  # Very small dataset
    val_dataset = Flickr30kDataset(ds['test'], processor, size_limit=50)     # Very small validation set
    
    # Create dataloaders with minimal batch size
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=1)  # Minimal batch size
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=1)     # Minimal batch size
    
    # Set model to training mode
    model.train()
    
    # Define optimizer with a smaller learning rate
    optimizer = optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0.01)  # Smaller LR
    
    # Training loop with just 1 epoch
    num_epochs = 1
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        batch_count = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
            # Move batch to device
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass - get embeddings from both image and text encoders
            image_embeddings = model.encode_image(pixel_values)
            text_embeddings = model.encode_text(input_ids, attention_mask)
            
            # Compute contrastive loss to align image and text embeddings
            logits_per_image = image_embeddings @ text_embeddings.t()
            logits_per_text = text_embeddings @ image_embeddings.t()
            
            labels = torch.arange(len(pixel_values), device=device)
            
            loss_img = nn.CrossEntropyLoss()(logits_per_image, labels)
            loss_txt = nn.CrossEntropyLoss()(logits_per_text, labels)
            
            loss = (loss_img + loss_txt) / 2
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            batch_count += 1
            
            # Print loss every 10 batches
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            # Limit to first 20 batches for efficiency
            if batch_idx >= 20:
                break
        
        avg_train_loss = train_loss / min(len(train_loader), 20)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batch_count = 0
        
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
                val_batch_count += 1
                
                # Limit validation to first 5 batches
                if batch_idx >= 5:
                    break
        
        avg_val_loss = val_loss / min(len(val_loader), 5)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # Save the trained model
    print("Training completed successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

if __name__ == "__main__":
    train_model()