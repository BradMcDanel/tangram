#!/usr/bin/env python3
import os
import numpy as np
import torch
import cv2
import argparse
import matplotlib.pyplot as plt
from ultralytics import YOLO

# --- Configuration ---
DEFAULT_SAM2_CHECKPOINT = os.path.expanduser("~/data-main-1/sam2/sam2.1_hiera_large.pt")
DEFAULT_MODEL_CONFIG_NAME = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEFAULT_YOLO_MODEL = os.path.expanduser("~/data-main-1/sam2/yolo_tangram_model.pt")
YOLO_CONFIDENCE_THRESHOLD = 0.4

def detect_table_roi(image):
    print("Attempting to auto-detect table ROI...")
    TABLE_LOWER_HSV = np.array([100, 0, 115])
    TABLE_UPPER_HSV = np.array([179, 35, 190])
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, TABLE_LOWER_HSV, TABLE_UPPER_HSV)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("WARNING: No table contour found. Will process the whole image.")
        return None
    table_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(table_contour)
    print(f"INFO: Auto-detected table ROI at [x={x}, y={y}, w={w}, h={h}]")
    return (x, y, w, h)

def load_models(yolo_path, sam_checkpoint_path, sam_config_path):
    # This function is unchanged
    print("--- Setting up models ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading custom YOLO model from: {yolo_path}")
    if not os.path.exists(yolo_path): raise FileNotFoundError(f"YOLO model not found at: {yolo_path}")
    yolo_model = YOLO(yolo_path)
    print(yolo_model.names)
    print("YOLO model loaded successfully.")
    print(f"Loading SAM 2 model from: {os.path.basename(sam_checkpoint_path)}")
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    if not os.path.exists(sam_checkpoint_path): raise FileNotFoundError(f"SAM checkpoint not found at: {sam_checkpoint_path}")
    sam2_model = build_sam2(sam_config_path, sam_checkpoint_path, device=device)
    sam_predictor = SAM2ImagePredictor(sam2_model)
    print("SAM 2 model loaded successfully.")
    return yolo_model, sam_predictor

def segment_with_yolo_sam(image, yolo_model, sam_predictor, roi=None):
    print("\n--- Running Inference ---")
    print("Stage 1: Detecting tangrams with YOLO...")
    yolo_results = yolo_model(image, verbose=False)
    all_detections = yolo_results[0].boxes.data
    print(f"YOLO found {len(all_detections)} total potential objects.")

    # --- NEW: Select only the single best detection per class ---
    best_detections_per_class = {}
    for det in all_detections:
        confidence = det[4].item()
        if confidence < YOLO_CONFIDENCE_THRESHOLD:
            continue

        if roi is not None:
            box = det[:4].cpu().numpy()
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            rx, ry, rw, rh = roi
            if not (rx < cx < rx + rw and ry < cy < ry + rh):
                continue # Skip detection outside the table ROI
        
        class_id = int(det[5].item())
        
        # If we haven't seen this class yet, or if this detection is more confident, store it.
        if class_id not in best_detections_per_class or confidence > best_detections_per_class[class_id][4].item():
            best_detections_per_class[class_id] = det
    
    # The final list of detections to process is the values from our dictionary
    final_detections = list(best_detections_per_class.values())
    print(f"Selected {len(final_detections)} best detections (one per class) passing filters.")

    if not final_detections:
        return []

    print("Stage 2: Generating precise masks with SAM...")
    sam_predictor.set_image(image)
    
    final_segmentations = []
    for det in final_detections:
        xyxy = det[:4].cpu().numpy()
        class_id = int(det[5].item())
        class_name = yolo_model.names[class_id]

        masks, scores, _ = sam_predictor.predict(box=xyxy[None, :], multimask_output=True)
        best_mask = masks[np.argmax(scores)]

        final_segmentations.append({
            'class_id': class_id, 'class_name': class_name,
            'confidence': det[4].item(), 'mask': best_mask.squeeze()
        })
    
    print(f"Generated {len(final_segmentations)} high-quality masks.")
    return final_segmentations

def visualize_results(image, segmentations, yolo_model, roi=None):
    # This function is unchanged
    display_image = image.copy()
    if roi is not None:
        rx, ry, rw, rh = roi
        cv2.rectangle(display_image, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)
    for seg in segmentations:
        mask = seg['mask']
        class_name = seg['class_name']
        mask_bool = mask.astype(bool)
        avg_rgb_color = np.mean(image[mask_bool], axis=0) if np.any(mask_bool) else np.array([255, 255, 255])
        avg_bgr_color = avg_rgb_color[::-1].astype(np.uint8)
        roi_pixels = display_image[mask_bool]
        if roi_pixels.size == 0: continue
        color_block = np.full(roi_pixels.shape, avg_bgr_color, dtype=np.uint8)
        blended_roi = cv2.addWeighted(roi_pixels, 0.5, color_block, 0.5, 0)
        display_image[mask_bool] = blended_roi
        mask_uint8 = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(display_image, contours, -1, (255, 255, 255), 2)
        M = cv2.moments(mask_uint8)
        if M["m00"] > 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            font_scale = 0.4; font_thickness = 1
            text = class_name.replace('_', ' ')
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            cv2.putText(display_image, text, (cX - w//2 + 1, cY + h//2 + 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness + 1, cv2.LINE_AA)
            cv2.putText(display_image, text, (cX - w//2, cY + h//2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    return display_image

def main():
    # This function is unchanged
    parser = argparse.ArgumentParser(description="Segment tangrams using a YOLO+SAM two-stage pipeline.")
    parser.add_argument("--image", required=True, help="Path to the input image to process.")
    parser.add_argument("--yolo-model", default=DEFAULT_YOLO_MODEL, help=f"Path to your custom-trained YOLO model.")
    parser.add_argument("--output", default="segmented_output.png", help="Path to save the final segmented image.")
    parser.add_argument("--sam-checkpoint", default=DEFAULT_SAM2_CHECKPOINT, help="Path to SAM 2 model checkpoint.")
    parser.add_argument("--sam-config", default=DEFAULT_MODEL_CONFIG_NAME, help="Path to SAM 2 model config yaml.")
    args = parser.parse_args()
    try:
        yolo_model, sam_predictor = load_models(args.yolo_model, args.sam_checkpoint, args.sam_config)
    except FileNotFoundError as e:
        print(f"\nError: {e}"); return
    if not os.path.exists(args.image):
        print(f"Error: Input image not found at {args.image}"); return
    image_bgr = cv2.imread(args.image)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    table_roi = detect_table_roi(image_rgb)
    segmentations = segment_with_yolo_sam(image_rgb, yolo_model, sam_predictor, roi=table_roi)
    output_image_bgr = visualize_results(image_bgr, segmentations, yolo_model, roi=table_roi)
    cv2.imwrite(args.output, output_image_bgr)
    print(f"\nSUCCESS: Final segmentation saved to {args.output}")
    cv2.imshow("YOLO-SAM Segmentation", output_image_bgr)
    print("Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
