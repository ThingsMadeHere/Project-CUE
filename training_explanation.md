# Training the Universal Compositional Embedder with Flickr30k

## Overview

This document explains how to train the Universal Compositional Embedder model using the Flickr30k dataset to align image and text embeddings.

## Training Objective

The goal is to train the image encoder to output embeddings that match the text encoder's output for the same semantic content. This alignment allows both encoders to produce consistent representations of the same concept across modalities.

## Dataset: Flickr30k

- **Size**: 31,783 images
- **Annotations**: Each image has 5 different captions
- **Total pairs**: ~158,915 image-text pairs
- **Content**: Diverse set of objects, actions, and scenes
- **Source**: `datasets.load_dataset("nlphuji/flickr30k")`

## Model Architecture

The Universal Compositional Embedder consists of:

1. **Image Encoder**: CLIP Vision Transformer
2. **Text Encoder**: CLIP Text Transformer
3. **Video Encoder**: CLIP-based architecture
4. **Audio Encoder**: CLIP-based architecture
5. **Projection layers**: To ensure consistent embedding dimensions
6. **Compositional layer**: For combining embeddings
7. **Cone embeddings**: For representing relationships

## Training Process

### 1. Data Preparation
- Load image-text pairs from Flickr30k
- Preprocess images to consistent size (e.g., 224x224)
- Tokenize text captions using CLIP tokenizer
- Create batches of aligned image-text pairs

### 2. Forward Pass
- Encode images using the image encoder: `image_embeddings = model.encode_image(pixel_values)`
- Encode text using the text encoder: `text_embeddings = model.encode_text(input_ids, attention_mask)`
- Both encoders output normalized embeddings of the same dimension

### 3. Loss Calculation
- Compute similarity matrix: `logits = image_embeddings @ text_embeddings.t()`
- Use contrastive loss (InfoNCE) to align matching pairs
- The loss encourages embeddings of matching image-text pairs to be similar
- And embeddings of non-matching pairs to be dissimilar

### 4. Optimization
- Backpropagate the loss through both encoders
- Update model parameters using AdamW optimizer
- Apply gradient clipping for stability

## Training Parameters

- **Batch size**: 32 (adjustable based on memory constraints)
- **Learning rate**: 5e-5
- **Optimizer**: AdamW with weight decay
- **Epochs**: 3-5
- **Loss**: Contrastive (InfoNCE) loss
- **Embedding normalization**: Applied to ensure unit length vectors

## Expected Outcomes

After training:
1. Image and text encoders produce aligned embeddings for the same content
2. Cross-modal retrieval becomes possible (image query → text results and vice versa)
3. The compositional layer can combine embeddings meaningfully
4. Cone embeddings can represent relationships between concepts

## Implementation Code Structure

```python
# Pseudo-code for the training process:

class UniversalCompositionalEmbedder(nn.Module):
    def __init__(self):
        # Initialize encoders for different modalities
        self.image_encoder = CLIPVisionModel.from_pretrained(...)
        self.text_encoder = CLIPTextModel.from_pretrained(...)
        self.video_encoder = ...
        self.audio_encoder = ...
        
        # Projection layers to ensure consistent embedding dimensions
        self.image_projection = nn.Linear(512, 512)
        self.text_projection = nn.Linear(512, 512)
        # ... for other modalities

    def encode_image(self, pixel_values):
        features = self.image_encoder(pixel_values=pixel_values).pooler_output
        embeddings = self.image_projection(features)
        return F.normalize(embeddings, dim=-1)

    def encode_text(self, input_ids, attention_mask):
        features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        embeddings = self.text_projection(features)
        return F.normalize(embeddings, dim=-1)

# Training loop:
model = UniversalCompositionalEmbedder()
optimizer = AdamW(model.parameters(), lr=5e-5)

for epoch in range(num_epochs):
    for batch in dataloader:
        # Get embeddings from both encoders
        image_embeddings = model.encode_image(batch['pixel_values'])
        text_embeddings = model.encode_text(batch['input_ids'], batch['attention_mask'])

        # Compute contrastive loss
        logits = image_embeddings @ text_embeddings.t()
        labels = torch.arange(len(image_embeddings))
        loss = F.cross_entropy(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Memory Optimization Strategies

For systems with limited memory:
1. Reduce batch size
2. Limit dataset size during training
3. Use gradient checkpointing
4. Implement data streaming
5. Apply mixed precision training (if available)
6. Clear GPU cache periodically

## Benefits of This Approach

1. **Cross-modal alignment**: Images and text produce consistent embeddings
2. **Scalability**: Can handle multiple modalities in a unified framework
3. **Compositional reasoning**: Can combine concepts from different modalities
4. **Flexibility**: Supports various downstream tasks (retrieval, classification, etc.)