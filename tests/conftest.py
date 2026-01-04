"""
Pytest configuration for Universal Compositional Embedder tests
"""
import pytest


def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU (deselect with '-m \"not gpu\"')"
    )


@pytest.fixture
def sample_text():
    """Sample text for testing"""
    return ["This is a sample text for embedding", "Another example text"]


@pytest.fixture
def sample_image():
    """Sample image for testing"""
    from PIL import Image
    import numpy as np
    # Create a dummy image
    img_array = (255 * np.random.rand(224, 224, 3)).astype('uint8')
    return Image.fromarray(img_array)


@pytest.fixture
def sample_audio():
    """Sample audio for testing"""
    import numpy as np
    # Create a dummy audio signal (1 second at 16kHz)
    return np.random.randn(16000)


@pytest.fixture
def sample_model_config():
    """Sample model configuration for testing"""
    return {
        'text_encoder_model': 'sentence-transformers/all-MiniLM-L6-v2',
        'image_encoder_model': 'google/vit-base-patch16-224',
        'audio_encoder_model': 'openai/whisper-tiny',
        'output_dim': 64,  # Smaller for faster tests
        'up_project_dim': None,
        'enable_composition': True,
        'num_primitives': 10  # Smaller for faster tests
    }


@pytest.fixture
def device():
    """Return appropriate device for testing"""
    try:
        import torch
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    except ImportError:
        # If torch is not available, return cpu
        return 'cpu'