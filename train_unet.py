#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import argparse

class UNet(nn.Module):
    """U-Net architecture for heatmap prediction"""
    
    def __init__(self, in_channels=3, out_channels=1, features=[4, 6, 8, 10]):
        super(UNet, self).__init__()
        
        # Encoder (downsampling)
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # First encoder block
        self.encoder.append(self._make_conv_block(in_channels, features[0]))
        
        # Remaining encoder blocks
        for i in range(len(features) - 1):
            self.encoder.append(self._make_conv_block(features[i], features[i + 1]))
        
        # Bottleneck
        self.bottleneck = self._make_conv_block(features[-1], features[-1] * 2)
        
        # Decoder (upsampling)
        self.decoder = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        
        # Reverse features for decoder
        features_reversed = features[::-1]
        
        for i in range(len(features_reversed)):
            # Upconvolutional layer
            if i == 0:
                self.upconvs.append(
                    nn.ConvTranspose2d(features_reversed[0] * 2, features_reversed[0], 
                                     kernel_size=2, stride=2)
                )
            else:
                self.upconvs.append(
                    nn.ConvTranspose2d(features_reversed[i-1], features_reversed[i], 
                                     kernel_size=2, stride=2)
                )
            
            # Decoder conv block
            self.decoder.append(
                self._make_conv_block(features_reversed[i] * 2, features_reversed[i])
            )
        
        # Final output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def _make_conv_block(self, in_channels, out_channels):
        """Create a convolutional block with two conv layers"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder path
        skip_connections = []
        
        for encoder_block in self.encoder:
            x = encoder_block(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path
        skip_connections = skip_connections[::-1]  # Reverse for decoder
        
        for i, (upconv, decoder_block) in enumerate(zip(self.upconvs, self.decoder)):
            x = upconv(x)
            
            # Handle size mismatch by cropping or padding
            skip_connection = skip_connections[i]
            if x.shape != skip_connection.shape:
                # Resize x to match skip connection
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=False)
            
            # Concatenate skip connection
            x = torch.cat([skip_connection, x], dim=1)
            x = decoder_block(x)
        
        # Final output
        x = self.final_conv(x)
        
        # Ensure output is exactly 64x64
        if x.shape[2:] != (64, 64):
            x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
        
        return torch.sigmoid(x)  # Output in [0, 1] range

class TangramDataset(Dataset):
    """Dataset for tangram vertex heatmap prediction"""
    
    def __init__(self, metadata_path, input_size=(64, 64), augment=False):
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.input_size = input_size
        self.augment = augment
        
        # Filter out any missing files
        self.valid_samples = []
        for sample in self.metadata:
            crop_path = Path(sample['crop_path'])
            heatmap_path = Path(sample['heatmap_path'])
            if crop_path.exists() and heatmap_path.exists():
                self.valid_samples.append(sample)
        
        print(f"Loaded {len(self.valid_samples)} valid samples")
    
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        sample = self.valid_samples[idx]
        
        # Load crop image
        crop = cv2.imread(sample['crop_path'])
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        # Load heatmap
        heatmap = np.load(sample['heatmap_path'])
        
        # Resize crop to input size
        crop_resized = cv2.resize(crop, self.input_size)
        
        # Normalize crop to [0, 1]
        crop_normalized = crop_resized.astype(np.float32) / 255.0
        
        # Normalize heatmap to [0, 1]
        heatmap_normalized = heatmap.astype(np.float32) / 255.0
        
        # Simple augmentation (optional)
        if self.augment and np.random.rand() > 0.5:
            # Random horizontal flip
            crop_normalized = np.fliplr(crop_normalized)
            heatmap_normalized = np.fliplr(heatmap_normalized)
        
        # Convert to PyTorch tensors
        # Crop: (H, W, C) -> (C, H, W)
        crop_tensor = torch.from_numpy(crop_normalized).permute(2, 0, 1)
        
        # Heatmap: (H, W) -> (1, H, W)
        heatmap_tensor = torch.from_numpy(heatmap_normalized).unsqueeze(0)
        
        return crop_tensor, heatmap_tensor

def combined_loss(pred, target, alpha=0.5):
    """Combined MSE and structural loss"""
    # MSE loss
    mse_loss = F.mse_loss(pred, target)
    
    # Structural similarity loss (simple version)
    # Compare local patches
    pred_patches = F.unfold(pred, kernel_size=3, padding=1)
    target_patches = F.unfold(target, kernel_size=3, padding=1)
    
    # Cosine similarity between patches
    pred_patches_norm = F.normalize(pred_patches, dim=1)
    target_patches_norm = F.normalize(target_patches, dim=1)
    
    cosine_sim = (pred_patches_norm * target_patches_norm).sum(dim=1)
    structural_loss = 1 - cosine_sim.mean()
    
    return alpha * mse_loss + (1 - alpha) * structural_loss

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch_idx, (images, heatmaps) in enumerate(tqdm(dataloader, desc="Training")):
        images = images.to(device)
        heatmaps = heatmaps.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Calculate loss
        loss = criterion(outputs, heatmaps)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, heatmaps in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            heatmaps = heatmaps.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, heatmaps)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def visualize_predictions(model, dataset, device, num_samples=4, save_path=None):
    """Visualize model predictions"""
    model.eval()
    
    fig, axes = plt.subplots(3, num_samples, figsize=(4 * num_samples, 12))
    
    with torch.no_grad():
        for i in range(num_samples):
            # Get random sample
            idx = np.random.randint(0, len(dataset))
            image, target = dataset[idx]
            
            # Make prediction
            image_batch = image.unsqueeze(0).to(device)
            prediction = model(image_batch)
            
            # Convert to numpy for visualization
            image_np = image.permute(1, 2, 0).cpu().numpy()
            target_np = target.squeeze(0).cpu().numpy()
            prediction_np = prediction.squeeze(0).squeeze(0).cpu().numpy()
            
            # Plot
            axes[0, i].imshow(image_np)
            axes[0, i].set_title(f'Input {i+1}')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(target_np, cmap='hot')
            axes[1, i].set_title(f'Target {i+1}')
            axes[1, i].axis('off')
            
            axes[2, i].imshow(prediction_np, cmap='hot')
            axes[2, i].set_title(f'Prediction {i+1}')
            axes[2, i].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train U-Net for vertex heatmap prediction')
    parser.add_argument('--data_dir', type=str, default='unet_data',
                       help='Directory containing training data')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--input_size', type=int, default=64,
                       help='Input image size (will be resized to this)')
    parser.add_argument('--save_dir', type=str, default='unet_models',
                       help='Directory to save models')
    parser.add_argument('--augment', action='store_true',
                       help='Use data augmentation')
    
    args = parser.parse_args()
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    metadata_path = Path(args.data_dir) / 'metadata.json'
    full_dataset = TangramDataset(metadata_path, input_size=(args.input_size, args.input_size), 
                                 augment=args.augment)
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    model = UNet(in_channels=3, out_channels=1).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function and optimizer
    criterion = combined_loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, save_dir / 'best_model.pth')
            print(f"New best model saved! Val Loss: {val_loss:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, save_dir / f'checkpoint_epoch_{epoch+1}.pth')
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / 'training_curves.png')
    plt.close()
    
    # Generate predictions visualization
    print("\nGenerating prediction visualizations...")
    visualize_predictions(model, val_dataset.dataset, device, num_samples=6, 
                         save_path=save_dir / 'predictions.png')
    
    print(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved to: {save_dir}")

if __name__ == "__main__":
    main()
