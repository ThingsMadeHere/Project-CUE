# Universal Compositional Embedder with Cone-Based Semantics (Project CUE)

This project implements a Universal Compositional Embedder that processes multiple modalities (text, images, audio, video) and represents concepts as spherical cones in a shared semantic space.

## Features

- **Multi-modal Support**: Handles text, images, audio, and video inputs
- **Cone-Based Representations**: Concepts represented as spherical cones (μ, θ) where μ is the centroid direction and θ is the angular radius
- **Cross-Modal Alignment**: Aligns embeddings from different modalities for the same concept
- **Compositionality**: Combines primitive concepts to form compound concepts
- **Semantic Properties**: Implements cone containment for inheritance relationships

## Architecture

The model consists of:

1. **Modality-Specific Encoders** (frozen pretrained models):
   - Text: E5/SBERT
   - Image: CLIP-ViT
   - Audio: Whisper encoder
   - Video: Key frame extraction + image encoder

2. **Universal Projection Heads**: Trainable layers mapping to shared semantic space

3. **Cone Representation System**: Spherical cone representation with mathematical operations

4. **Compositional Layer**: Synthesizes compound embeddings from primitive concepts

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from src.models import UniversalCompositionalEmbedder

# Create the model
model = UniversalCompositionalEmbedder(
    output_dim=256,
    enable_composition=True
)

# Encode different modalities
text_embeddings = model.encode_text(["sample text"])
image_embeddings = model.encode_image([pil_image])
audio_embeddings = model.encode_audio([audio_data])
video_embeddings = model.encode_video([video_path])

# Perform cone operations
similarity = model.compute_cone_similarity(cone1, cone2)
containment = model.check_cone_containment(outer_cone, inner_cone)
membership = model.test_membership(embedding, cone)

# Compose concepts
compound_cone = model.compose_concepts([cone1, cone2, cone3])
```

## Project Structure

```
project-cue/
├── src/
│   ├── models/
│   │   ├── encoders.py              # Modality-specific encoders
│   │   ├── cone_embeddings.py       # Cone representation system
│   │   ├── projection_heads.py      # Universal projection layers
│   │   ├── compositional_layer.py   # Composition operations
│   ├── data/
│   │   ├── loaders.py               # Multi-modal data loaders
│   ├── training/
│   │   ├── losses.py                # Training objectives
│   │   └── trainer.py               # Training loop
│   └── utils/
│       ├── cone_operations.py       # Cone math operations
├── configs/
├── tests/
├── requirements.txt
└── README.md
```

## Training

The model supports multiple training objectives:
- Cross-modal alignment loss
- Cone containment loss
- Primitive grounding loss
- Contrastive discrimination loss

## Semantic Properties

- **Inheritance via Cone Containment**: More specific concepts are geometrically contained within more general ones
- **Embedding Primitives**: Irreducible concepts that form the basis for composition
- **Compositionality**: Complex concepts built from combinations of primitive cones

## Example

See `example_usage.py` for a complete example demonstrating the model's capabilities.