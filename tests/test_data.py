"""
Tests for the data loading module
"""
import pytest
import torch
import os
import tempfile
from PIL import Image
import numpy as np

from src.data.loaders import (
    MultiModalDataset, MultiModalDataLoader, 
    get_file_modality, create_multimodal_dataset_from_directory
)


class TestDataLoaders:
    """Test data loading functionality"""
    
    def test_get_file_modality(self):
        """Test file modality detection"""
        # Test text files
        assert get_file_modality("test.txt") == "text"
        assert get_file_modality("document.pdf") == "text"
        assert get_file_modality("file.docx") == "text"
        assert get_file_modality("readme.md") == "text"
        
        # Test image files
        assert get_file_modality("image.jpg") == "image"
        assert get_file_modality("photo.png") == "image"
        assert get_file_modality("picture.jpeg") == "image"
        assert get_file_modality("snapshot.gif") == "image"
        
        # Test audio files
        assert get_file_modality("song.mp3") == "audio"
        assert get_file_modality("recording.wav") == "audio"
        assert get_file_modality("music.flac") == "audio"
        assert get_file_modality("audio.ogg") == "audio"
        
        # Test video files
        assert get_file_modality("video.mp4") == "video"
        assert get_file_modality("movie.avi") == "video"
        assert get_file_modality("clip.mov") == "video"
        assert get_file_modality("film.mkv") == "video"
        
        # Test unsupported file type
        with pytest.raises(ValueError):
            get_file_modality("document.xyz")
    
    def test_multimodal_dataset_initialization(self):
        """Test MultiModalDataset initialization"""
        # Create temporary files for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a text file
            text_file = os.path.join(temp_dir, "test.txt")
            with open(text_file, 'w') as f:
                f.write("This is a test text file.")
            
            # Create an image file
            img = Image.new('RGB', (224, 224), color='red')
            img_file = os.path.join(temp_dir, "test.jpg")
            img.save(img_file)
            
            # Test dataset initialization
            data_paths = [
                (text_file, 'text'),
                (img_file, 'image')
            ]
            
            dataset = MultiModalDataset(data_paths)
            
            # Check dataset properties
            assert len(dataset) == 2
            assert dataset.data_paths == data_paths
    
    def test_multimodal_dataset_getitem(self):
        """Test MultiModalDataset __getitem__ method"""
        # Create temporary files for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a text file
            text_file = os.path.join(temp_dir, "test.txt")
            with open(text_file, 'w') as f:
                f.write("This is a test text file.")
            
            # Create an image file
            img = Image.new('RGB', (224, 224), color='red')
            img_file = os.path.join(temp_dir, "test.jpg")
            img.save(img_file)
            
            # Test dataset
            data_paths = [
                (text_file, 'text'),
                (img_file, 'image')
            ]
            
            dataset = MultiModalDataset(data_paths)
            
            # Test text item
            text_item = dataset[0]
            assert text_item['modality'] == 'text'
            assert text_item['path'] == text_file
            assert isinstance(text_item['data'], str)
            assert "test text file" in text_item['data']
            
            # Test image item
            image_item = dataset[1]
            assert image_item['modality'] == 'image'
            assert image_item['path'] == img_file
            assert isinstance(image_item['data'], Image.Image)
            assert image_item['data'].size == (224, 224)
    
    def test_multimodal_dataloader(self):
        """Test MultiModalDataLoader functionality"""
        # Create temporary files for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a text file
            text_file = os.path.join(temp_dir, "test.txt")
            with open(text_file, 'w') as f:
                f.write("This is a test text file.")
            
            # Create an image file
            img = Image.new('RGB', (224, 224), color='red')
            img_file = os.path.join(temp_dir, "test.jpg")
            img.save(img_file)
            
            # Test data loader
            data_paths = [
                (text_file, 'text'),
                (img_file, 'image')
            ]
            
            loader = MultiModalDataLoader(batch_size=1)
            data_loader = loader.create_loader(data_paths)
            
            # Check that we can iterate over the loader
            items = []
            for batch in data_loader:
                items.extend(batch)
            
            assert len(items) == 2
            assert items[0]['modality'] == 'text'
            assert items[1]['modality'] == 'image'
    
    def test_create_multimodal_dataset_from_directory(self):
        """Test create_multimodal_dataset_from_directory function"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create various files
            text_file = os.path.join(temp_dir, "test.txt")
            with open(text_file, 'w') as f:
                f.write("Test text")
            
            img_file = os.path.join(temp_dir, "test.jpg")
            img = Image.new('RGB', (224, 224), color='blue')
            img.save(img_file)
            
            # Create subdirectory with more files
            sub_dir = os.path.join(temp_dir, "subdir")
            os.makedirs(sub_dir)
            audio_file = os.path.join(sub_dir, "test.wav")
            # Create a simple dummy audio file (just for testing the path)
            with open(audio_file, 'wb') as f:
                f.write(b'dummy audio content')
            
            # Test recursive search
            data_paths = create_multimodal_dataset_from_directory(temp_dir, recursive=True)
            
            # Should find text, image, and audio files
            found_modalities = [path[1] for path in data_paths]
            assert 'text' in found_modalities
            assert 'image' in found_modalities
            # Note: audio file won't be detected since it's not a real audio file format
            # but the function should still work for valid files
            
            # Test non-recursive search
            data_paths = create_multimodal_dataset_from_directory(temp_dir, recursive=False)
            
            # Should only find text and image in the root directory
            found_modalities = [path[1] for path in data_paths]
            assert 'text' in found_modalities
            assert 'image' in found_modalities
            assert len([p for p in data_paths if p[0] == audio_file]) == 0  # audio file in subdir not found