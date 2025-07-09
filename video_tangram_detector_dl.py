#!/usr/bin/env python3
import cv2
import numpy as np
import json
import argparse
import time
import datetime
import os
import torch
from ultralytics import YOLO
from tqdm import tqdm

# ==============================================================================
# --- Configuration Constants ---
# ==============================================================================
DEFAULT_YOLO_MODEL = os.path.expanduser("~/data-main-1/sam2/yolo_tangram_model.pt")
DEFAULT_SAM2_CHECKPOINT = os.path.expanduser("~/data-main-1/sam2/sam2.1_hiera_large.pt")
DEFAULT_MODEL_CONFIG_NAME = "configs/sam2.1/sam2.1_hiera_l.yaml"
YOLO_CONFIDENCE_THRESHOLD = 0.1
TABLE_OUTLINE_COLOR_BGR = (0, 192, 255)
PIECE_LOCKED_COLOR_BGR = (0, 255, 0)
PIECE_VERTEX_COLOR_BGR = (0, 0, 255)
PIECE_LOCKED_THICKNESS = 2
PIECE_VERTEX_RADIUS = 2

# ==============================================================================
# --- Core Functions ---
# ==============================================================================

def load_models():
    """Initializes and loads the YOLO and SAM video models once."""
    print("--- Setting up models ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    yolo_model = YOLO(DEFAULT_YOLO_MODEL)
    print("YOLO model loaded successfully.")
    from sam2.build_sam import build_sam2_video_predictor
    if not os.path.exists(DEFAULT_SAM2_CHECKPOINT):
        raise FileNotFoundError(f"SAM checkpoint not found at: {DEFAULT_SAM2_CHECKPOINT}")
    sam_video_predictor = build_sam2_video_predictor(DEFAULT_MODEL_CONFIG_NAME, DEFAULT_SAM2_CHECKPOINT, device=device)
    print("SAM 2 Video Predictor loaded successfully.")
    return yolo_model, sam_video_predictor

def find_initial_detections(video_path, yolo_model, config_data):
    """Finds the first occurrence of each tangram piece using YOLO by sampling a video."""
    print("--- Finding initial object detections with YOLO ---")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")
    
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_indices = sorted(list(set([0, num_frames // 4, num_frames // 2, (num_frames * 3) // 4, num_frames - 1])))
    
    first_seen = {}
    for frame_idx in tqdm(sample_indices, desc="Scanning for objects"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret: continue
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yolo_results = yolo_model(frame_rgb, verbose=False, conf=YOLO_CONFIDENCE_THRESHOLD)
        
        for det in yolo_results[0].boxes.data:
            class_id = int(det[5].item())
            if class_id not in first_seen:
                class_name = yolo_model.names[class_id]
                pid = next((pid for pid, attrs in config_data.items() if attrs.get("class_name") == class_name), None)
                if pid:
                    first_seen[class_id] = {
                        'pid': pid, 'class_name': class_name, 'frame_idx': frame_idx,
                        'box': det[:4].cpu().numpy()}
    cap.release()

    print(f"Found initial detections for {len(first_seen)} unique classes.")
    for class_id, data in first_seen.items():
        print(f"  - Found '{data['class_name']}' on frame {data['frame_idx']}")
    return first_seen

def main():
    parser = argparse.ArgumentParser(description="Tangram tracking with YOLO and SAM2 Video Predictor.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--no_display", action="store_true")
    args = parser.parse_args()

    # --- 1. Initialization Phase ---
    yolo_model, sam_predictor = load_models()
    with open(args.config, 'r') as f: tangram_config_data = json.load(f)

    # --- 2. Detection Phase (on video file path) ---
    initial_detections = find_initial_detections(args.video, yolo_model, tangram_config_data)
    if not initial_detections:
        print("No tangram pieces were detected by YOLO in the video."); return

    # --- 3. SAM Initialization and Seeding ---
    print("\n--- Initializing SAM state and seeding prompts ---")
    # THE FIX: Pass the video_path directly to init_state
    inference_state = sam_predictor.init_state(video_path=args.video)
    
    for class_id, data in initial_detections.items():
        sam_predictor.add_new_points_or_box(
            inference_state=inference_state, frame_idx=data['frame_idx'],
            obj_id=class_id, box=data['box'])

    # --- 4. Propagation Phase ---
    print("--- Propagating masks through video with SAM ---")
    all_video_segments = {}
    total_frames = inference_state['video_info']['num_frames']
    for out_frame_idx, out_obj_ids, out_mask_logits in tqdm(sam_predictor.propagate_in_video(inference_state), total=total_frames, desc="Tracking"):
        frame_masks = {obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, obj_id in enumerate(out_obj_ids)}
        all_video_segments[out_frame_idx] = frame_masks

    # --- 5. Final Output Generation ---
    print("--- Generating final output JSON and visualization ---")
    all_frames_data = []
    cap = cv2.VideoCapture(args.video) # Re-open video for visualization
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        display_frame = frame.copy()
        current_frame_pieces = []
        
        frame_masks = all_video_segments.get(frame_idx, {})
        for obj_id, mask in frame_masks.items():
            mask_uint8 = mask.astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: continue
            
            main_contour = max(contours, key=cv2.contourArea)
            poly = cv2.approxPolyDP(main_contour, 0.02 * cv2.arcLength(main_contour, True), True)
            
            class_name = yolo_model.names[obj_id]
            pid_for_class = next((pid for pid, attrs in tangram_config_data.items() if attrs.get("class_name") == class_name), None)

            if pid_for_class:
                current_frame_pieces.append({"piece_id": pid_for_class, "class_name": class_name,
                                             "vertices": poly.reshape(-1, 2).tolist()})
                cv2.drawContours(display_frame, [poly], -1, PIECE_LOCKED_COLOR_BGR, PIECE_LOCKED_THICKNESS)
                for x, y in poly.reshape(-1, 2):
                    cv2.circle(display_frame, (int(x), int(y)), PIECE_VERTEX_RADIUS, PIECE_VERTEX_COLOR_BGR, -1)

        all_frames_data.append({"frame_index": frame_idx, "pieces": current_frame_pieces})
        
        if not args.no_display:
            cv2.imshow("Tangram Tracking (YOLO+SAM Video)", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        frame_idx += 1

    # --- Cleanup and Save ---
    cap.release()
    if not args.no_display: cv2.destroyAllWindows()
    
    output_json_content = {"frames_data": all_frames_data}
    with open(args.output_json, 'w') as f_out:
        json.dump(output_json_content, f_out, indent=2)
    print(f"\nProcessed {frame_idx} frames. Output saved to {args.output_json}")

if __name__ == "__main__":
    main()
