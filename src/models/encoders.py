"""
Modality-Specific Encoders (Frozen)
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import torchvision.models as tv_models
import torchaudio
from PIL import Image
import numpy as np


class TextEncoder(nn.Module):
    """
    Text encoder using a pretrained language model (e.g., E5 or SBERT)
    """
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', freeze=True):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def forward(self, texts):
        """
        Args:
            texts: List of strings or a single string
        Returns:
            embeddings: Tensor of shape (batch_size, hidden_size)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.model(**inputs)
            # Use mean pooling for sentence embeddings
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings


class ImageEncoder(nn.Module):
    """
    Image encoder using a pretrained vision model (e.g., CLIP-ViT)
    """
    def __init__(self, model_name='google/vit-base-patch16-224', freeze=True):
        super().__init__()
        from transformers import ViTModel, ViTImageProcessor
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name)
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def forward(self, images):
        """
        Args:
            images: PIL Images or list of PIL Images
        Returns:
            embeddings: Tensor of shape (batch_size, hidden_size)
        """
        if isinstance(images, Image.Image):
            images = [images]
        
        inputs = self.processor(images=images, return_tensors='pt')
        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling of patch embeddings
        
        return embeddings


class AudioEncoder(nn.Module):
    """
    Audio encoder using a pretrained model (e.g., Whisper encoder or AST)
    """
    def __init__(self, model_name='openai/whisper-tiny', freeze=True):
        super().__init__()
        from transformers import WhisperModel, WhisperProcessor
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperModel.from_pretrained(model_name)
        
        # Use only the encoder part
        self.encoder = self.model.encoder
        
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def forward(self, audios, sampling_rate=16000):
        """
        Args:
            audios: Raw audio tensors or list of audio tensors
        Returns:
            embeddings: Tensor of shape (batch_size, hidden_size)
        """
        if isinstance(audios, (list, tuple)):
            # Process multiple audio files
            input_features = self.processor(audios, sampling_rate=sampling_rate, return_tensors="pt").input_features
        else:
            # Process single audio file
            input_features = self.processor(audios, sampling_rate=sampling_rate, return_tensors="pt").input_features
        
        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.encoder(input_features=input_features)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling of sequence
        
        return embeddings


class VideoEncoder(nn.Module):
    """
    Video handler: Shot boundary detection + frame extraction + image encoder
    """
    def __init__(self, image_encoder=None, num_frames=6):
        super().__init__()
        self.image_encoder = image_encoder or ImageEncoder()
        self.num_frames = num_frames  # Extract 4-8 representative frames
    
    def extract_frames(self, video_path):
        """
        Extract representative frames from video
        Args:
            video_path: Path to video file
        Returns:
            frames: List of PIL Images
        """
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        frames = []
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if i in frame_indices:
                # Convert BGR to RGB and then to PIL Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)
        
        cap.release()
        return frames
    
    def forward(self, video_paths):
        """
        Args:
            video_paths: List of paths to video files or single path
        Returns:
            embeddings: Tensor of shape (batch_size, hidden_size)
        """
        if isinstance(video_paths, str):
            video_paths = [video_paths]
        
        all_embeddings = []
        
        for video_path in video_paths:
            frames = self.extract_frames(video_path)
            frame_embeddings = self.image_encoder(frames)
            
            # Aggregate frame embeddings via mean pooling (simple approach)
            # In practice, you might want to use learned attention pooling
            video_embedding = frame_embeddings.mean(dim=0, keepdim=True)
            all_embeddings.append(video_embedding)
        
        return torch.cat(all_embeddings, dim=0)