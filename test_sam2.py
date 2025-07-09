#!/usr/bin/env python3
"""
A test script for running SAM 2 segmentation on a local image.

This script loads a pre-trained SAM 2 model, processes a specified input image,
runs segmentation with various prompts (points and boxes), and saves the
visualized result to an output file.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2 # Used in a helper function

# --- 1. Configuration ---
SAM2_CHECKPOINT = os.path.expanduser("~/data-main-1/sam2/sam2.1_hiera_large.pt")
INPUT_IMAGE_PATH = os.path.expanduser("~/code/tangram/data/images/calibration.png")
OUTPUT_IMAGE_PATH = "sam2_calibration_output.png"
MODEL_CONFIG_NAME = "configs/sam2.1/sam2.1_hiera_l.yaml"

# --- 2. Helper Functions (from the notebook) ---
def show_mask(mask, ax, random_color=False):
    """Draws a mask on the given matplotlib axis."""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
    # Draw contours for borders
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        ax.plot(contour[:, 0, 0], contour[:, 0, 1], color='white', linewidth=1.5, alpha=0.8)


def show_points(coords, labels, ax, marker_size=200):
    """Draws points on the given matplotlib axis."""
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    """Draws a box on the given matplotlib axis."""
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))


def main():
    """Main execution function."""
    # --- 3. Setup Device and Model ---
    print("--- Setting up environment ---")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if not os.path.exists(SAM2_CHECKPOINT):
        print(f"\nError: Model checkpoint not found at: {SAM2_CHECKPOINT}")
        return

    print(f"Loading SAM 2 model from {os.path.basename(SAM2_CHECKPOINT)}...")
    print(f"Using model config: {MODEL_CONFIG_NAME}")
    
    sam2_model = build_sam2(MODEL_CONFIG_NAME, SAM2_CHECKPOINT, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    print("Model loaded successfully.")

    # --- 4. Load Image and Set Embeddings ---
    print(f"\n--- Loading and processing image: {INPUT_IMAGE_PATH} ---")
    try:
        image = Image.open(INPUT_IMAGE_PATH).convert("RGB")
        image = np.array(image)
    except FileNotFoundError:
        print(f"Error: Input image not found at {INPUT_IMAGE_PATH}")
        return
        
    predictor.set_image(image)
    print("Image embeddings calculated.")
    H, W, _ = image.shape
    print(f"Image dimensions: {W}x{H}")

    # --- 5. Define Prompts and Predict ---
    print("\n--- Defining prompts and running predictions ---")
    point_prompt_1 = {"point_coords": np.array([[W*0.5, H*0.25]]), "point_labels": np.array([1])}
    box_prompt_1 = np.array([W*0.4, H*0.4, W*0.6, H*0.6])
    box_prompt_2 = np.array([W*0.6, H*0.6, W*0.9, H*0.9])
    point_prompt_2 = {"point_coords": np.array([[W*0.75, H*0.75]]), "point_labels": np.array([0])}

    print("Predicting for Test Case 1 (single point)...")
    masks1, _, _ = predictor.predict(**point_prompt_1, multimask_output=False)

    print("Predicting for Test Case 2 (center box)...")
    masks2, _, _ = predictor.predict(box=box_prompt_1[None, :], multimask_output=False)
    
    print("Predicting for Test Case 3 (box with negative point)...")
    masks3, _, _ = predictor.predict(
        point_coords=point_prompt_2["point_coords"],
        point_labels=point_prompt_2["point_labels"],
        box=box_prompt_2[None, :],
        multimask_output=False
    )
    
    # --- THIS IS THE CORRECTED LINE ---
    all_masks = [masks1[0], masks2[0], masks3[0]]
    print("All predictions are complete.")

    # --- 6. Visualize and Save Results ---
    print(f"\n--- Visualizing and saving results to {OUTPUT_IMAGE_PATH} ---")
    plt.figure(figsize=(12, 12))
    plt.imshow(image)
    for mask in all_masks:
        show_mask(mask, plt.gca(), random_color=True)
    show_points(point_prompt_1["point_coords"], point_prompt_1["point_labels"], plt.gca())
    show_box(box_prompt_1, plt.gca())
    show_box(box_prompt_2, plt.gca())
    show_points(point_prompt_2["point_coords"], point_prompt_2["point_labels"], plt.gca())
    
    plt.title("SAM 2 Segmentation Results on calibration.png", fontsize=16)
    plt.axis('off')
    plt.savefig(OUTPUT_IMAGE_PATH, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"\nSuccess! Output saved to {OUTPUT_IMAGE_PATH}")

if __name__ == '__main__':
    main()
