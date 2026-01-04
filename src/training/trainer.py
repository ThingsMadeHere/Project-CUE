"""
Training Loop Implementation
Handles the training procedure with multiple objectives
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
from src.models import UniversalCompositionalEmbedder
from src.training.losses import CombinedTrainingLoss


class Trainer:
    """
    Main trainer class for the Universal Compositional Embedder
    """
    def __init__(self, model, train_loader, val_loader=None, 
                 learning_rate=1e-4, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=learning_rate
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Use the combined loss function
        self.criterion = CombinedTrainingLoss(
            cross_modal_weight=1.0,
            containment_weight=1.0,
            primitive_weight=0.5,  # Lower weight initially
            discrimination_weight=1.0,
            temperature=0.07
        )
        
        # Move model to device
        self.model.to(self.device)
    
    def train_epoch(self, epoch):
        """
        Train for one epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch}")
        
        for batch in progress_bar:
            # Extract data and move to device
            data_list = []
            modality_list = []
            labels_list = []
            
            for item in batch:
                data = item['data']
                modality = item['modality']
                # For simplicity, using the path as a label (in practice, you'd have semantic labels)
                label = hash(item['path']) % 1000  # Simple hash-based label
                
                data_list.append(data)
                modality_list.append(modality)
                labels_list.append(label)
            
            # Convert labels to tensor
            labels_tensor = torch.tensor(labels_list, dtype=torch.long, device=self.device)
            
            # Process each modality and collect embeddings
            mu_list = []
            theta_list = []
            
            for data, modality in zip(data_list, modality_list):
                # Convert data to appropriate format and encode
                try:
                    if modality == 'text':
                        # For text, we can pass the string directly
                        mu, theta = self.model.encode_text([data])
                    elif modality == 'image':
                        # For image, pass the PIL image
                        mu, theta = self.model.encode_image([data])
                    elif modality == 'audio':
                        # For audio, pass the audio data
                        audio_data, sr = data
                        mu, theta = self.model.encode_audio([audio_data], sampling_rate=sr)
                    elif modality == 'video':
                        # For video, pass the path
                        mu, theta = self.model.encode_video([data])
                    
                    mu_list.append(mu)
                    theta_list.append(theta)
                except Exception as e:
                    print(f"Error processing {modality} data: {e}")
                    continue
            
            if len(mu_list) == 0:
                continue  # Skip if no valid data processed
            
            # Combine all embeddings
            all_mu = torch.cat(mu_list, dim=0)
            all_theta = torch.cat(theta_list, dim=0)
            
            # Replicate labels for each modality
            expanded_labels = labels_tensor[:all_mu.size(0)]
            
            # Compute loss
            loss, loss_components = self.criterion(
                [all_mu], [all_theta], expanded_labels
            )
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def validate(self):
        """
        Validate the model
        """
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Similar processing as in training
                data_list = []
                modality_list = []
                labels_list = []
                
                for item in batch:
                    data = item['data']
                    modality = item['modality']
                    label = hash(item['path']) % 1000
                    
                    data_list.append(data)
                    modality_list.append(modality)
                    labels_list.append(label)
                
                labels_tensor = torch.tensor(labels_list, dtype=torch.long, device=self.device)
                
                mu_list = []
                theta_list = []
                
                for data, modality in zip(data_list, modality_list):
                    try:
                        if modality == 'text':
                            mu, theta = self.model.encode_text([data])
                        elif modality == 'image':
                            mu, theta = self.model.encode_image([data])
                        elif modality == 'audio':
                            audio_data, sr = data
                            mu, theta = self.model.encode_audio([audio_data], sampling_rate=sr)
                        elif modality == 'video':
                            mu, theta = self.model.encode_video([data])
                        
                        mu_list.append(mu)
                        theta_list.append(theta)
                    except Exception:
                        continue
                
                if len(mu_list) == 0:
                    continue
                
                all_mu = torch.cat(mu_list, dim=0)
                all_theta = torch.cat(theta_list, dim=0)
                
                expanded_labels = labels_tensor[:all_mu.size(0)]
                
                loss, _ = self.criterion(
                    [all_mu], [all_theta], expanded_labels
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def train(self, num_epochs=10, save_dir='./checkpoints', patience=10):
        """
        Main training loop
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss = self.train_epoch(epoch+1)
            
            # Validate
            val_loss = self.validate()
            
            # Update learning rate based on validation loss
            self.scheduler.step(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save model checkpoint
                checkpoint_path = os.path.join(save_dir, f'best_model_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, checkpoint_path)
                
                print(f"Saved best model checkpoint to {checkpoint_path}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement")
                break
        
        print(f"\nTraining completed. Best validation loss: {best_val_loss:.4f}")
    
    def save_model(self, path):
        """
        Save the trained model
        """
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """
        Load a trained model
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")


def create_trainer(config, model, train_loader, val_loader=None):
    """
    Factory function to create a trainer with specified configuration
    """
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config.get('learning_rate', 1e-4),
        device=config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    return trainer