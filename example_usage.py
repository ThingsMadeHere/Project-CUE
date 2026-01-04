"""
Example usage of the Universal Compositional Embedder with Cone-Based Semantics
"""
import torch
from src.models import UniversalCompositionalEmbedder
from src.data.loaders import MultiModalDataLoader, create_multimodal_dataset_from_directory
from PIL import Image
import numpy as np


def main():
    print("Creating Universal Compositional Embedder...")
    
    # Create the model
    model = UniversalCompositionalEmbedder(
        output_dim=256,
        enable_composition=True,
        num_primitives=100
    )
    
    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print("Model created successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Example 1: Text encoding
    print("\n--- Example 1: Text Encoding ---")
    text = ["This is a sample text for embedding", "Another example text"]
    mu_text, theta_text = model.encode_text(text)
    print(f"Text embeddings shape: mu={mu_text.shape}, theta={theta_text.shape}")
    
    # Example 2: Image encoding (using a dummy image)
    print("\n--- Example 2: Image Encoding ---")
    # Create a dummy image for demonstration
    dummy_image = Image.new('RGB', (224, 224), color='red')
    mu_img, theta_img = model.encode_image([dummy_image])
    print(f"Image embeddings shape: mu={mu_img.shape}, theta={theta_img.shape}")
    
    # Example 3: Audio encoding (using a dummy audio signal)
    print("\n--- Example 3: Audio Encoding ---")
    # Create a dummy audio signal for demonstration
    dummy_audio = np.random.randn(16000)  # 1 second at 16kHz
    mu_audio, theta_audio = model.encode_audio([dummy_audio], sampling_rate=16000)
    print(f"Audio embeddings shape: mu={mu_audio.shape}, theta={theta_audio.shape}")
    
    # Example 4: Cone operations
    print("\n--- Example 4: Cone Operations ---")
    # Check similarity between text and image embeddings
    similarity = model.compute_cone_similarity((mu_text[0], theta_text[0]), (mu_text[1], theta_text[1]))
    print(f"Similarity between two text embeddings: {similarity.item():.4f}")
    
    # Check containment (this would make more sense with semantically related concepts)
    containment = model.check_cone_containment((mu_text[0], theta_text[0]), (mu_text[1], theta_text[1]))
    print(f"Is text[0] contained in text[1]: {containment.item()}")
    
    # Example 5: Composition (if enabled)
    if model.enable_composition:
        print("\n--- Example 5: Composition ---")
        # Compose multiple concepts
        cones_list = [
            (mu_text[0], theta_text[0]),
            (mu_img[0], theta_img[0])
        ]
        mu_compound, theta_compound = model.compose_concepts(cones_list)
        print(f"Compound concept shape: mu={mu_compound.shape}, theta={theta_compound.shape}")
        
        # Extract primitives
        top_similarities, top_indices = model.extract_primitives(mu_text[0], k=3)
        print(f"Top 3 primitive concepts: similarities={top_similarities}, indices={top_indices}")
    
    print("\n--- Example 6: Membership Test ---")
    # Test if an embedding is inside a cone
    membership = model.test_membership(mu_text[0], (mu_text[1], theta_text[1]))
    print(f"Is text[0] inside text[1] cone: {membership.item()}")
    
    print("\nAll examples completed successfully!")


def create_sample_dataset():
    """
    Create a sample dataset directory structure for demonstration
    """
    import os
    
    # Create sample directories
    os.makedirs('sample_data/text', exist_ok=True)
    os.makedirs('sample_data/images', exist_ok=True)
    os.makedirs('sample_data/audio', exist_ok=True)
    
    # Create a sample text file
    with open('sample_data/text/sample.txt', 'w') as f:
        f.write("This is a sample text file for the Universal Compositional Embedder.")
    
    # Create a sample image
    img = Image.new('RGB', (224, 224), color='blue')
    img.save('sample_data/images/sample.jpg')
    
    print("Sample dataset created in 'sample_data/' directory")


if __name__ == "__main__":
    # Create sample dataset first
    create_sample_dataset()
    
    # Run main example
    main()