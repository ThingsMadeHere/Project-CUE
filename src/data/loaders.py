"""
Multi-modal Data Loaders
Handles loading of different file types: text, images, audio, video
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import librosa
import numpy as np


class MultiModalDataset(Dataset):
    """
    Dataset class that handles multiple modalities
    """
    def __init__(self, data_paths, labels=None, max_length=512):
        """
        Args:
            data_paths: List of tuples (file_path, modality_type) where modality_type is one of 
                       'text', 'image', 'audio', 'video'
            labels: Optional list of labels for each data sample
            max_length: Maximum length for text sequences
        """
        self.data_paths = data_paths
        self.labels = labels
        self.max_length = max_length
        
        # Validate file paths exist
        for path, modality in data_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File does not exist: {path}")
            if modality not in ['text', 'image', 'audio', 'video']:
                raise ValueError(f"Invalid modality: {modality}")
    
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        file_path, modality = self.data_paths[idx]
        
        if modality == 'text':
            data = self._load_text(file_path)
        elif modality == 'image':
            data = self._load_image(file_path)
        elif modality == 'audio':
            data = self._load_audio(file_path)
        elif modality == 'video':
            data = self._load_video(file_path)
        else:
            raise ValueError(f"Unknown modality: {modality}")
        
        result = {
            'data': data,
            'modality': modality,
            'path': file_path
        }
        
        if self.labels is not None:
            result['label'] = self.labels[idx]
        
        return result
    
    def _load_text(self, file_path):
        """Load text from file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Truncate if necessary
        if len(text) > self.max_length:
            text = text[:self.max_length]
        
        return text
    
    def _load_image(self, file_path):
        """Load image from file"""
        image = Image.open(file_path).convert('RGB')
        return image
    
    def _load_audio(self, file_path):
        """Load audio from file"""
        # Load audio with librosa
        audio, sr = librosa.load(file_path, sr=None)
        return audio, sr
    
    def _load_video(self, file_path):
        """Load video from file - return path as processing is done by encoder"""
        # For video, we just return the path since the VideoEncoder handles frame extraction
        return file_path


class MultiModalDataLoader:
    """
    Wrapper for creating DataLoader with multi-modal data
    """
    def __init__(self, batch_size=1, shuffle=False, num_workers=0, **kwargs):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.kwargs = kwargs
    
    def create_loader(self, data_paths, labels=None):
        """
        Create a DataLoader for multi-modal data
        Args:
            data_paths: List of tuples (file_path, modality_type)
            labels: Optional list of labels
        """
        dataset = MultiModalDataset(data_paths, labels, **self.kwargs)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers
        )
        return loader


def get_file_modality(file_path):
    """
    Determine modality type based on file extension
    Args:
        file_path: Path to the file
    Returns:
        modality: String indicating the modality type
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    text_extensions = {'.txt', '.pdf', '.docx', '.html', '.htm', '.md', '.rst'}
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.gif'}
    audio_extensions = {'.wav', '.mp3', '.flac', '.aac', '.m4a', '.ogg'}
    video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.wmv', '.webm', '.m4v'}
    
    if ext in text_extensions:
        return 'text'
    elif ext in image_extensions:
        return 'image'
    elif ext in audio_extensions:
        return 'audio'
    elif ext in video_extensions:
        return 'video'
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def create_multimodal_dataset_from_directory(directory_path, recursive=True):
    """
    Create a multi-modal dataset from a directory of files
    Args:
        directory_path: Path to directory containing mixed modality files
        recursive: Whether to search recursively in subdirectories
    Returns:
        data_paths: List of tuples (file_path, modality_type)
    """
    data_paths = []
    
    if recursive:
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    modality = get_file_modality(file_path)
                    data_paths.append((file_path, modality))
                except ValueError:
                    # Skip unsupported file types
                    continue
    else:
        for file in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file)
            if os.path.isfile(file_path):
                try:
                    modality = get_file_modality(file_path)
                    data_paths.append((file_path, modality))
                except ValueError:
                    # Skip unsupported file types
                    continue
    
    return data_paths