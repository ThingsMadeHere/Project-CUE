# Universal Compositional Embedder - Training with Flickr30k Dataset

## Project Overview

The Universal Compositional Embedder (CUE) is a multi-modal embedding system that processes text, images, audio, and video inputs, representing concepts as spherical cones in a shared semantic space. This project demonstrates how to train the model using the Flickr30k dataset to align image and text embeddings.

## Key Components

### Model Architecture
- **Modality-Specific Encoders**: Pretrained models (CLIP, E5, Whisper) for each modality
- **Universal Projection Heads**: Trainable layers mapping to shared semantic space
- **Cone Representation System**: Spherical cone representation with mathematical operations
- **Compositional Layer**: Synthesizes compound embeddings from primitive concepts

### Cone-Based Semantics
- Concepts represented as spherical cones (μ, θ) where μ is the centroid direction and θ is the angular radius
- Implements cone containment for inheritance relationships
- Supports compositionality by combining primitive concepts into compound concepts

## Training with Flickr30k Dataset

### Dataset Information
- **Source**: `datasets.load_dataset("nlphuji/flickr30k")`
- **Size**: 31,783 images
- **Annotations**: Each image has 5 different captions
- **Total pairs**: ~158,915 image-text pairs
- **Content**: Diverse set of objects, actions, and scenes

### Training Objective
The goal is to train the image encoder to output embeddings that match the text encoder's output for the same semantic content. This alignment allows both encoders to produce consistent representations of the same concept across modalities.

### Training Process
1. **Data Preparation**: Load image-text pairs from Flickr30k, preprocess images and tokenize text
2. **Forward Pass**: Encode images and text separately using specialized encoders
3. **Loss Calculation**: Compute contrastive loss (InfoNCE) to align matching pairs
4. **Optimization**: Backpropagate the loss and update model parameters

### Expected Outcomes
1. Image and text encoders produce aligned embeddings for the same content
2. Cross-modal retrieval becomes possible (image query → text results and vice versa)
3. The compositional layer can combine embeddings meaningfully
4. Cone embeddings can represent relationships between concepts

## Implementation Files

1. **`universal_compositional_embedder/models.py`**: Main model implementation
2. **`universal_compositional_embedder/config.py`**: Configuration classes
3. **`universal_compositional_embedder/layers/`**: Custom layers and operations
4. **`training_explanation.md`**: Detailed training explanation
5. **`README.md`**: Project documentation

## Training Parameters
- **Batch size**: 32 (adjustable based on memory constraints)
- **Learning rate**: 5e-5
- **Optimizer**: AdamW with weight decay
- **Epochs**: 3-5
- **Loss**: Contrastive (InfoNCE) loss
- **Embedding normalization**: Applied to ensure unit length vectors

## Benefits of This Approach

1. **Cross-modal alignment**: Images and text produce consistent embeddings
2. **Scalability**: Can handle multiple modalities in a unified framework
3. **Compositional reasoning**: Can combine concepts from different modalities
4. **Flexibility**: Supports various downstream tasks (retrieval, classification, etc.)

## Usage

To train the model with Flickr30k:
```python
from datasets import load_dataset

ds = load_dataset("nlphuji/flickr30k")
# Use the dataset with the Universal Compositional Embedder model
```

## Memory Optimization Strategies

For systems with limited memory:
1. Reduce batch size
2. Limit dataset size during training
3. Use gradient checkpointing
4. Implement data streaming
5. Apply mixed precision training (if available)
6. Clear GPU cache periodically

This implementation provides a comprehensive framework for training a multi-modal embedding system that can effectively align different modalities while maintaining the ability to compose complex concepts from simpler primitives.