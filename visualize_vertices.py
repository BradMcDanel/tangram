#!/usr/bin/env python3
import argparse
import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Configuration
DEFAULT_YOLO_MODEL = os.path.expanduser("~/data-main-1/sam2/yolo_tangram_model.pt")
YOLO_CONFIDENCE_THRESHOLD = 0.30

# Class mapping (0-indexed in YOLO labels)
CLASS_MAPPING = {
    0: 'pink_triangle',
    1: 'red_triangle',
    2: 'orange_triangle',
    3: 'blue_triangle',
    4: 'green_triangle',
    5: 'yellow_square',
    6: 'purple_parallelogram'
}

# Color mapping for visualization (RGB format for matplotlib)
PIECE_COLORS = {
    'pink_triangle': (1.0, 0.75, 0.8),       # Pink
    'red_triangle': (1.0, 0.0, 0.0),         # Red
    'orange_triangle': (1.0, 0.65, 0.0),     # Orange
    'blue_triangle': (0.12, 0.56, 1.0),      # Blue
    'green_triangle': (0.0, 1.0, 0.0),       # Green
    'yellow_square': (1.0, 1.0, 0.0),        # Yellow
    'purple_parallelogram': (0.5, 0.0, 0.5)  # Purple
}

def load_vertex_labels(label_path, img_width, img_height):
    """
    Load vertex labels from YOLO format text file.
    
    Args:
        label_path: Path to the label file
        img_width: Image width for denormalizing coordinates
        img_height: Image height for denormalizing coordinates
    
    Returns:
        Dictionary mapping class_id to list of vertex coordinates
    """
    vertices_by_class = {}
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
                
            class_id = int(parts[0])
            # YOLO format: center_x, center_y, width, height (all normalized)
            cx, cy = float(parts[1]), float(parts[2])
            
            # Convert normalized coordinates to pixel coordinates
            x = cx * img_width
            y = cy * img_height
            
            if class_id not in vertices_by_class:
                vertices_by_class[class_id] = []
            vertices_by_class[class_id].append((x, y))
    
    return vertices_by_class

def get_best_detection_per_class(results, model):
    """Extract best detection per class from YOLO results."""
    best_detections = {}
    
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        
        if class_name not in best_detections or confidence > best_detections[class_name]['conf']:
            best_detections[class_name] = {
                'bbox': box.xyxy[0].cpu().numpy(),
                'class_id': class_id,
                'conf': confidence
            }
    
    return best_detections

def create_vertex_heatmap(vertices, bbox, heatmap_size=(64, 64), sigma=2, padding=5):
    """
    Create a heatmap for vertices within a bounding box.
    
    Args:
        vertices: List of (x, y) vertex coordinates
        bbox: Bounding box [x1, y1, x2, y2]
        heatmap_size: Size of the output heatmap
        sigma: Gaussian kernel standard deviation
        padding: Padding to add around bbox for vertex inclusion
    
    Returns:
        Heatmap as numpy array
    """
    heatmap = np.zeros(heatmap_size)
    x1, y1, x2, y2 = bbox
    
    # Add padding to bbox for vertex detection
    x1_pad = x1 - padding
    y1_pad = y1 - padding
    x2_pad = x2 + padding
    y2_pad = y2 + padding
    
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    
    for vx, vy in vertices:
        # Check if vertex is within padded bbox
        if x1_pad <= vx <= x2_pad and y1_pad <= vy <= y2_pad:
            # Normalize to heatmap coordinates (relative to original bbox)
            hx = int((vx - x1) / bbox_width * heatmap_size[1])
            hy = int((vy - y1) / bbox_height * heatmap_size[0])
            
            # Clamp to heatmap bounds
            hx = max(0, min(heatmap_size[1] - 1, hx))
            hy = max(0, min(heatmap_size[0] - 1, hy))
            
            # Create gaussian around vertex
            for i in range(max(0, hy-3*sigma), min(heatmap_size[0], hy+3*sigma+1)):
                for j in range(max(0, hx-3*sigma), min(heatmap_size[1], hx+3*sigma+1)):
                    dist_sq = (i - hy)**2 + (j - hx)**2
                    heatmap[i, j] += np.exp(-dist_sq / (2 * sigma**2))
    
    # Normalize heatmap
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    return heatmap

def visualize_single_image(image_path, label_path, model, save_path=None):
    """
    Visualize vertices overlaid on YOLO detections for a single image.
    
    Args:
        image_path: Path to the image
        label_path: Path to the vertex labels
        model: YOLO model
        save_path: Optional path to save the visualization
    """
    # Load image
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    # Load vertex labels
    vertices_by_class = load_vertex_labels(label_path, w, h)
    
    # Run YOLO detection
    results = model(img, verbose=False, conf=YOLO_CONFIDENCE_THRESHOLD)
    detections = get_best_detection_per_class(results, model)
    
    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'Vertex Overlay Visualization: {os.path.basename(image_path)}', fontsize=16)
    
    # Main image with all detections
    ax_main = axes[0, 0]
    ax_main.imshow(img_rgb)
    ax_main.set_title('Original with Detections')
    ax_main.axis('off')
    
    # Draw all bounding boxes and vertices
    for class_name, det in detections.items():
        bbox = det['bbox']
        class_id = det['class_id']
        color = PIECE_COLORS.get(class_name, (1, 1, 1))
        
        # Draw bbox
        x1, y1, x2, y2 = bbox
        rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                        linewidth=2, edgecolor=color, facecolor='none')
        ax_main.add_patch(rect)
        
        # Draw vertices if available
        if class_id in vertices_by_class:
            vertices = vertices_by_class[class_id]
            for vx, vy in vertices:
                ax_main.plot(vx, vy, 'o', color=color, markersize=4)
    
    # Individual piece visualizations
    plot_idx = 1
    padding = 10  # Add padding to ensure vertices are visible
    
    for class_id, class_name in CLASS_MAPPING.items():
        if class_name in detections and class_id in vertices_by_class:
            ax = axes[plot_idx // 4, plot_idx % 4]
            
            # Extract bbox region with padding
            bbox = detections[class_name]['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Add padding (but stay within image bounds)
            x1_pad = max(0, x1 - padding)
            y1_pad = max(0, y1 - padding)
            x2_pad = min(w, x2 + padding)
            y2_pad = min(h, y2 + padding)
            
            piece_img = img_rgb[y1_pad:y2_pad, x1_pad:x2_pad]
            
            if piece_img.size > 0:
                ax.imshow(piece_img)
                
                # Plot vertices relative to padded bbox
                vertices = vertices_by_class[class_id]
                vertices_in_view = 0
                for vx, vy in vertices:
                    if x1_pad <= vx <= x2_pad and y1_pad <= vy <= y2_pad:
                        # Convert to padded-bbox-relative coordinates
                        rel_x = vx - x1_pad
                        rel_y = vy - y1_pad
                        ax.plot(rel_x, rel_y, 'ro', markersize=6)
                        vertices_in_view += 1
                
                # Draw original bbox boundary
                bbox_x1 = x1 - x1_pad
                bbox_y1 = y1 - y1_pad
                bbox_w = x2 - x1
                bbox_h = y2 - y1
                rect = Rectangle((bbox_x1, bbox_y1), bbox_w, bbox_h,
                               linewidth=1, edgecolor='white', facecolor='none', linestyle='--')
                ax.add_patch(rect)
                
                ax.set_title(f'{class_name}\n({vertices_in_view}/{len(vertices)} vertices)')
            else:
                ax.text(0.5, 0.5, 'Empty bbox', ha='center', va='center')
                ax.set_title(class_name)
            
            ax.axis('off')
            plot_idx += 1
    
    # Fill remaining subplots
    while plot_idx < 8:
        ax = axes[plot_idx // 4, plot_idx % 4]
        ax.axis('off')
        plot_idx += 1
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()

def generate_training_data(image_path, label_path, model, output_dir, padding=10):
    """
    Generate training data for UNET model (crops and heatmaps).
    
    Args:
        image_path: Path to the image
        label_path: Path to the vertex labels
        model: YOLO model
        output_dir: Directory to save training data
        padding: Padding to add around bounding boxes
    """
    # Create output directories
    crops_dir = output_dir / 'crops'
    heatmaps_dir = output_dir / 'heatmaps'
    metadata_dir = output_dir / 'metadata'
    crops_dir.mkdir(parents=True, exist_ok=True)
    heatmaps_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    # Load image
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    
    # Load vertex labels
    vertices_by_class = load_vertex_labels(label_path, w, h)
    
    # Run YOLO detection
    results = model(img, verbose=False, conf=YOLO_CONFIDENCE_THRESHOLD)
    detections = get_best_detection_per_class(results, model)
    
    base_name = image_path.stem
    
    # Process each detection
    for class_name, det in detections.items():
        class_id = det['class_id']
        if class_id not in vertices_by_class:
            continue
            
        bbox = det['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        
        # Add padding to bbox
        x1_pad = max(0, x1 - padding)
        y1_pad = max(0, y1 - padding)
        x2_pad = min(w, x2 + padding)
        y2_pad = min(h, y2 + padding)
        
        # Extract crop with padding
        crop = img[y1_pad:y2_pad, x1_pad:x2_pad]
        if crop.size == 0:
            continue
            
        # Generate heatmap (vertices are relative to original bbox, not padded)
        vertices = vertices_by_class[class_id]
        heatmap = create_vertex_heatmap(vertices, bbox, padding=padding)
        
        # Save crop and heatmap
        crop_path = crops_dir / f'{base_name}_{class_name}.png'
        heatmap_path = heatmaps_dir / f'{base_name}_{class_name}.npy'
        
        cv2.imwrite(str(crop_path), crop)
        np.save(str(heatmap_path), heatmap)
        
        # Save metadata about padding and original bbox
        metadata = {
            'original_bbox': bbox.tolist(),
            'padded_bbox': [x1_pad, y1_pad, x2_pad, y2_pad],
            'padding': padding,
            'vertices': vertices
        }
        metadata_path = metadata_dir / f'{base_name}_{class_name}.json'
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved {class_name}: {crop_path.name}")

def main():
    parser = argparse.ArgumentParser(description='Visualize vertex labels on YOLO detections')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--labels_dir', type=str, default='labeled_data_vertex/labels',
                       help='Directory containing vertex labels')
    parser.add_argument('--images_dir', type=str, default='labeled_data_vertex/images',
                       help='Directory containing images')
    parser.add_argument('--yolo_model', type=str, default=DEFAULT_YOLO_MODEL,
                       help='Path to YOLO model')
    parser.add_argument('--output_dir', type=str, help='Output directory for training data')
    parser.add_argument('--visualize_all', action='store_true',
                       help='Visualize all images in the directory')
    parser.add_argument('--generate_training', action='store_true',
                       help='Generate training data (crops and heatmaps)')
    
    args = parser.parse_args()
    
    # Load YOLO model
    print("Loading YOLO model...")
    model = YOLO(args.yolo_model)
    
    if args.image:
        # Process single image
        image_path = Path(args.image)
        label_name = image_path.stem + '.txt'
        label_path = Path(args.labels_dir) / label_name
        
        if not label_path.exists():
            print(f"Warning: Label file not found: {label_path}")
            return
            
        if args.generate_training and args.output_dir:
            output_dir = Path(args.output_dir)
            generate_training_data(image_path, label_path, model, output_dir)
        else:
            visualize_single_image(image_path, label_path, model)
    
    elif args.visualize_all:
        # Process all images
        images_dir = Path(args.images_dir)
        labels_dir = Path(args.labels_dir)
        output_dir = Path(args.output_dir) if args.output_dir else None
        
        for image_path in sorted(images_dir.glob('*.png')):
            label_path = labels_dir / (image_path.stem + '.txt')
            
            if not label_path.exists():
                print(f"Skipping {image_path.name} - no label file found")
                continue
            
            print(f"Processing {image_path.name}...")
            
            if args.generate_training and output_dir:
                generate_training_data(image_path, label_path, model, output_dir)
            else:
                save_path = output_dir / f'viz_{image_path.name}' if output_dir else None
                visualize_single_image(image_path, label_path, model, save_path)

if __name__ == "__main__":
    main()
