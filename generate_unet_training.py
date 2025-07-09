#!/usr/bin/env python3
import argparse
import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

# Configuration
DEFAULT_YOLO_MODEL = os.path.expanduser("~/data-main-1/sam2/yolo_tangram_model.pt")
YOLO_CONFIDENCE_THRESHOLD = 0.30

# Class mapping
CLASS_MAPPING = {
    0: 'pink_triangle',
    1: 'red_triangle',
    2: 'orange_triangle',
    3: 'blue_triangle',
    4: 'green_triangle',
    5: 'yellow_square',
    6: 'purple_parallelogram'
}

def load_vertex_labels(label_path, img_width, img_height):
    """Load vertex labels from YOLO format text file."""
    vertices_by_class = {}
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
                
            class_id = int(parts[0])
            cx, cy = float(parts[1]), float(parts[2])
            
            # Convert normalized coordinates to pixel coordinates
            x = cx * img_width
            y = cy * img_height
            
            if class_id not in vertices_by_class:
                vertices_by_class[class_id] = []
            vertices_by_class[class_id].append((x, y))
    
    return vertices_by_class



def create_quantized_heatmap(vertices, crop_shape, bbox_in_crop, target_size=(64, 64), sigma_pixels=15):
    """
    Create a quantized heatmap at target resolution.
    
    Args:
        vertices: List of (x, y) vertex coordinates in original image space
        crop_shape: (height, width) of the crop
        bbox_in_crop: [x1, y1, x2, y2] bbox position within the crop
        target_size: (height, width) of the target heatmap
        sigma_pixels: Gaussian sigma in pixels (will be scaled to target resolution)
    
    Returns:
        heatmap: Quantized heatmap at target resolution
    """
    target_h, target_w = target_size
    orig_h, orig_w = crop_shape[:2]
    
    # Calculate scaling factors
    scale_x = target_w / orig_w
    scale_y = target_h / orig_h
    
    # Scale sigma to match target resolution
    sigma_scaled = sigma_pixels * min(scale_x, scale_y)
    
    heatmap = np.zeros((target_h, target_w), dtype=np.float32)
    
    # Create coordinate grids for target resolution
    y_grid, x_grid = np.ogrid[:target_h, :target_w]
    
    # Get crop origin in original image
    x1_crop, y1_crop = bbox_in_crop[0], bbox_in_crop[1]
    
    for vx, vy in vertices:
        # Convert vertex from image coordinates to crop coordinates
        vx_crop = vx - x1_crop
        vy_crop = vy - y1_crop
        
        # Scale to target resolution
        vx_scaled = vx_crop * scale_x
        vy_scaled = vy_crop * scale_y
        
        # Calculate Gaussian centered at scaled vertex position
        gaussian = np.exp(-((x_grid - vx_scaled)**2 + (y_grid - vy_scaled)**2) / (2 * sigma_scaled**2))
        
        # Add to heatmap (taking maximum to handle overlapping Gaussians)
        heatmap = np.maximum(heatmap, gaussian)
    
    # Normalize to [0, 1]
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    # Quantize to 8-bit (0-255)
    heatmap_quantized = (heatmap * 255).astype(np.uint8)
    
    return heatmap_quantized

def generate_training_data(image_path, label_path, model, output_dir, sigma_pixels=15):
    """
    Generate training data for UNET model with no padding.
    
    Args:
        image_path: Path to the image
        label_path: Path to the vertex labels
        model: YOLO model
        output_dir: Directory to save training data
        sigma_pixels: Gaussian sigma in pixels
    """
    # Create output directories
    crops_dir = output_dir / 'crops'
    heatmaps_dir = output_dir / 'heatmaps'
    visualizations_dir = output_dir / 'visualizations'
    
    for dir in [crops_dir, heatmaps_dir, visualizations_dir]:
        dir.mkdir(parents=True, exist_ok=True)
    
    # Load image
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    
    # Load vertex labels
    vertices_by_class = load_vertex_labels(label_path, w, h)
    
    # Run YOLO detection
    results = model(img, verbose=False, conf=YOLO_CONFIDENCE_THRESHOLD)
    
    base_name = image_path.stem
    processed_pieces = []
    
    # Process each detection
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = CLASS_MAPPING.get(class_id, f'class_{class_id}')
        
        if class_id not in vertices_by_class:
            continue
        
        bbox = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, bbox)
        
        # Extract crop with NO padding
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        
        # Generate quantized heatmap
        vertices = vertices_by_class[class_id]
        bbox_in_crop = [x1, y1, x2, y2]
        
        heatmap_quantized = create_quantized_heatmap(
            vertices,
            crop.shape,
            bbox_in_crop,
            target_size=(64, 64),
            sigma_pixels=sigma_pixels
        )
        
        # Save outputs
        crop_filename = f'{base_name}_{class_name}.png'
        heatmap_filename = f'{base_name}_{class_name}.npy'
        
        cv2.imwrite(str(crops_dir / crop_filename), crop)
        np.save(str(heatmaps_dir / heatmap_filename), heatmap_quantized)
        
        processed_pieces.append({
            'class_name': class_name,
            'crop_path': str(crops_dir / crop_filename),
            'heatmap_path': str(heatmaps_dir / heatmap_filename),
            'bbox': [x1, y1, x2, y2],
            'vertices': vertices,
            'crop_size': crop.shape[:2],
            'heatmap_size': heatmap_quantized.shape[:2],
            'sigma_pixels': sigma_pixels
        })
    
    return processed_pieces

def visualize_training_sample(crop_path, heatmap_path, info, save_path=None):
    """Visualize a single training sample."""
    # Load crop and heatmap
    crop = cv2.imread(crop_path)
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    heatmap = np.load(heatmap_path)
    
    # Get bbox info
    x1, y1, x2, y2 = info['bbox']
    
    # Create visualization with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Show crop with vertices
    axes[0].imshow(crop_rgb)
    axes[0].set_title(f"{info['class_name']} - Crop\n{crop.shape[1]}x{crop.shape[0]} pixels")
    
    # Plot vertices on crop
    for vx, vy in info['vertices']:
        # Convert to crop coordinates (no padding offset)
        vx_crop = vx - x1
        vy_crop = vy - y1
        axes[0].plot(vx_crop, vy_crop, 'yo', markersize=8, markeredgecolor='red')
    
    axes[0].axis('off')
    
    # Show target heatmap with pixelated display
    axes[1].imshow(heatmap, cmap='hot', interpolation='nearest')
    axes[1].set_title(f'Target Heatmap\n{heatmap.shape[1]}x{heatmap.shape[0]} pixels\nValues: 0-255')
    axes[1].axis('off')
    
    # Show heatmap overlaid on crop
    axes[2].imshow(crop_rgb)
    
    # Resize heatmap to match crop size for overlay
    heatmap_resized = cv2.resize(heatmap, (crop.shape[1], crop.shape[0]))
    heatmap_normalized = heatmap_resized / 255.0
    
    # Create colored overlay
    heatmap_colored = plt.cm.hot(heatmap_normalized)
    heatmap_colored[:, :, 3] = heatmap_normalized * 0.7  # Set alpha based on intensity
    
    axes[2].imshow(heatmap_colored, alpha=0.7)
    axes[2].set_title('Heatmap Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def visualize_dataset_statistics(metadata, output_dir):
    """Generate dataset statistics visualization."""
    # Extract crop sizes
    crop_sizes = [(info['crop_size'][1], info['crop_size'][0]) for info in metadata]
    
    # Create histogram of crop dimensions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    widths = [s[0] for s in crop_sizes]
    heights = [s[1] for s in crop_sizes]
    
    ax1.hist(widths, bins=20, alpha=0.7, label='Width')
    ax1.hist(heights, bins=20, alpha=0.7, label='Height')
    ax1.set_xlabel('Pixels')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Crop Dimensions (No Padding)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Scatter plot of dimensions
    ax2.scatter(widths, heights, alpha=0.5)
    ax2.set_xlabel('Width (pixels)')
    ax2.set_ylabel('Height (pixels)')
    ax2.set_title('Crop Dimensions Scatter')
    ax2.grid(True, alpha=0.3)
    
    # Add diagonal line for square crops
    max_dim = max(max(widths), max(heights))
    ax2.plot([0, max_dim], [0, max_dim], 'r--', alpha=0.5, label='Square')
    ax2.legend()
    
    plt.tight_layout()
    stats_path = output_dir / 'dataset_statistics.png'
    plt.savefig(stats_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print statistics
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(metadata)}")
    print(f"  Width range: {min(widths)}-{max(widths)} pixels")
    print(f"  Height range: {min(heights)}-{max(heights)} pixels")
    print(f"  Mean size: {np.mean(widths):.1f} x {np.mean(heights):.1f} pixels")

def main():
    parser = argparse.ArgumentParser(description='Generate UNET training data with full-resolution heatmaps')
    parser.add_argument('--images_dir', type=str, default='labeled_data_vertex/images',
                       help='Directory containing images')
    parser.add_argument('--labels_dir', type=str, default='labeled_data_vertex/labels',
                       help='Directory containing vertex labels')
    parser.add_argument('--yolo_model', type=str, default=DEFAULT_YOLO_MODEL,
                       help='Path to YOLO model')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for training data')
    parser.add_argument('--sigma', type=float, default=15.0,
                       help='Gaussian sigma in pixels (default: 15.0)')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations for each sample')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to process (for testing)')
    
    args = parser.parse_args()
    
    # Load YOLO model
    print("Loading YOLO model...")
    model = YOLO(args.yolo_model)
    
    # Process all images
    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    output_dir = Path(args.output_dir)
    
    image_files = sorted(images_dir.glob('*.png'))
    if args.max_samples:
        image_files = image_files[:args.max_samples]
    
    print(f"Found {len(image_files)} images to process")
    print(f"Using sigma={args.sigma} pixels for Gaussian kernels")
    print(f"Using NO padding around bounding boxes")
    
    all_metadata = []
    
    for image_path in tqdm(image_files, desc="Processing images"):
        label_path = labels_dir / (image_path.stem + '.txt')
        
        if not label_path.exists():
            print(f"Skipping {image_path.name} - no label file found")
            continue
        
        # Generate training data
        pieces_info = generate_training_data(
            image_path, label_path, model, output_dir,
            sigma_pixels=args.sigma
        )
        
        # Visualize if requested
        if args.visualize and pieces_info:
            viz_dir = output_dir / 'visualizations'
            for info in pieces_info:
                viz_path = viz_dir / f"{image_path.stem}_{info['class_name']}_viz.png"
                visualize_training_sample(
                    info['crop_path'], 
                    info['heatmap_path'], 
                    info, 
                    viz_path
                )
        
        # Store metadata
        all_metadata.extend(pieces_info)
    
    # Save metadata
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(all_metadata, f, indent=2)
    
    # Generate dataset statistics
    if all_metadata:
        visualize_dataset_statistics(all_metadata, output_dir)
    
    print(f"\nProcessing complete!")
    print(f"Generated {len(all_metadata)} training samples")
    print(f"Outputs saved to: {output_dir}")

if __name__ == "__main__":
    main()
