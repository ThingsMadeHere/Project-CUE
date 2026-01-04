# Implementation Plan: Universal Compositional Embedder with Cone-Based Semantics

## Project Overview
This document outlines the implementation plan for a Universal Compositional Embedder with Cone-Based Semantics that can process multiple modalities (text, images, audio, video) and represent concepts as spherical cones in a shared semantic space.

## Phase 1: Project Setup and Dependencies

### 1.1 Repository Structure
```
project-cue/
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoders.py          # Modality-specific encoders
│   │   ├── cone_embeddings.py   # Cone representation logic
│   │   ├── projection_heads.py  # Universal projection layers
│   │   └── compositional_layer.py # Composition operations
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loaders.py           # Multi-modal data loaders
│   │   └── preprocessors.py     # Modality-specific preprocessing
│   ├── training/
│   │   ├── __init__.py
│   │   ├── losses.py            # Training objectives
│   │   └── trainer.py           # Training loop
│   └── utils/
│       ├── __init__.py
│       ├── cone_operations.py   # Cone math operations
│       └── evaluation.py        # Evaluation metrics
├── configs/
│   └── model_config.yaml
├── tests/
├── requirements.txt
└── README.md
```

### 1.2 Dependencies
- PyTorch (deep learning framework)
- Transformers (pretrained models)
- OpenCV (video processing)
- librosa (audio processing)
- Pillow (image processing)
- scikit-learn (utilities)
- numpy, scipy (mathematical operations)

## Phase 2: Core Components Implementation

### 2.1 Modality-Specific Encoders (Frozen)
- Text encoder: E5 or SBERT pretrained model
- Image encoder: CLIP-ViT pretrained model
- Audio encoder: Whisper encoder or Audio Spectrogram Transformer
- Video handler: Shot boundary detection + frame extraction + image encoder

### 2.2 Cone Representation System
- Implement spherical cone representation (μ, θ) where:
  - μ ∈ ℝᴰ: unit vector (centroid direction)
  - θ ∈ (0, π/2]: angular radius (specificity)
- Implement cone operations (intersection, containment, similarity)

### 2.3 Universal Projection Heads
- Lightweight trainable projection layers
- Map modality-specific outputs to shared semantic space (D=256)
- Optionally up-project to 4096-D for compatibility

### 2.4 Compositional Layer
- Implement compound embedding synthesis
- Centroid averaging and radius reduction for intersection
- Primitive concept handling

## Phase 3: Training Objectives Implementation

### 3.1 Cross-modal Alignment Loss
- Contrastive loss to align embeddings of same concept from different modalities

### 3.2 Cone Containment Loss
- Differentiable loss to enforce cone inclusion for known parent-child relationships

### 3.3 Primitive Grounding Loss
- Stability and irreducibility constraints for primitive concepts

### 3.4 Contrastive Discrimination Loss
- Ensure cones for unrelated concepts are disjoint

## Phase 4: Data Pipeline

### 4.1 Multi-modal Data Loaders
- Handle text, image, audio, and video files
- Support common formats (.txt, .pdf, .docx, .html, .jpg, .png, .webp, .wav, .mp3, .flac, .mp4, .mov, .avi)

### 4.2 Preprocessing
- Text: tokenization, cleaning
- Images: normalization, resizing
- Audio: feature extraction, normalization
- Video: shot detection, frame extraction

## Phase 5: Model Architecture Integration

### 5.1 Main Model Class
- Combine all components into unified model
- Handle different input modalities
- Output cone representations

### 5.2 Forward Pass Implementation
- Route inputs to appropriate encoders
- Apply projection heads
- Output spherical cones

## Phase 6: Training and Evaluation

### 6.1 Training Loop
- Implement training procedure with multiple objectives
- Validation metrics for cone properties

### 6.2 Evaluation Metrics
- Similarity scoring accuracy
- Cone containment verification
- Cross-modal alignment quality

## Phase 7: Testing and Validation

### 7.1 Unit Tests
- Individual component tests
- Cone operation correctness
- Cross-modal functionality

### 7.2 Integration Tests
- End-to-end functionality
- Multi-modal input processing
- Composition operations

## Phase 8: Documentation and Examples

### 8.1 API Documentation
- Clear documentation for all classes and functions

### 8.2 Usage Examples
- Simple usage examples
- Advanced composition examples

## Technical Considerations

### Cone Mathematics
- Efficient implementation of cone intersection
- Numerical stability for cone containment checks
- Angular distance calculations

### Memory Management
- Batch processing for different modalities
- Efficient handling of video sequences

### Performance Optimization
- CUDA support for GPU acceleration
- Efficient cone operations

## Timeline Estimate
- Phase 1-2: 2-3 weeks (Core architecture)
- Phase 3-4: 2-3 weeks (Training pipeline)
- Phase 5-6: 2-3 weeks (Integration and training)
- Phase 7-8: 1-2 weeks (Testing and documentation)

Total: 7-11 weeks