#!/usr/bin/env python3
import argparse
import os
import sys
import cv2
from tqdm import tqdm
from ultralytics import YOLO

# --- Configuration ---
DEFAULT_VERTEX_MODEL = os.path.expanduser("~/data-main-1/sam2/yolo_tangram_vertex_model.pt") 
VERTEX_CONFIDENCE_THRESHOLD = 0.01
VERTEX_RADIUS = 2

# --- Color Mapping (BGR format for OpenCV) ---
# Maps the BASE piece name to a color.
PIECE_COLORS = {
    'pink_triangle': (203, 192, 255),
    'red_triangle': (0, 0, 255),
    'orange_triangle': (0, 165, 255),
    'blue_triangle': (255, 144, 30),
    'green_triangle': (0, 255, 0),
    'yellow_square': (0, 255, 255),
    'purple_parallelogram': (128, 0, 128)
}

def main():
    parser = argparse.ArgumentParser(description="Barebones Tangram vertex detection and display.")
    parser.add_argument("--video", required=True, help="Path to the video file.")
    parser.add_argument("--model", default=DEFAULT_VERTEX_MODEL, help="Path to the YOLO model trained to detect vertices.")
    parser.add_argument("--conf", type=float, default=VERTEX_CONFIDENCE_THRESHOLD, help="Confidence threshold for vertex detection.")
    args = parser.parse_args()
    
    print("--- Initializing Vertex Model ---")
    if not os.path.exists(args.model):
        sys.exit(f"FATAL ERROR: Model file not found at: {args.model}")
        
    vertex_model = YOLO(args.model)
    print("Model loaded successfully.")
    
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"FATAL ERROR: Could not open video file {args.video}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("\n--- Processing video... Press 'q' in the display window to quit. ---")
    
    for _ in tqdm(range(total_frames), desc="Processing", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break
        
        # --- Run Vertex Detection ---
        vertex_results = vertex_model(frame, verbose=False, conf=args.conf)
        
        display_frame = frame.copy()

        # --- Draw All Found Vertices Directly ---
        for box in vertex_results[0].boxes:
            # Get the name of the detected vertex class (e.g., "vertex_green_triangle")
            vertex_class_name = vertex_model.names[int(box.cls[0])]
            
            # Derive the base piece name to look up its color
            base_class_name = vertex_class_name.replace("vertex_", "")
            
            # Get the corresponding color, defaulting to white if not found
            color = PIECE_COLORS.get(base_class_name, (255, 255, 255))
            
            # Calculate the center of the vertex's bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # Draw a filled circle at the center of the detected vertex
            cv2.circle(display_frame, (center_x, center_y), VERTEX_RADIUS, color, -1)
            # Add a black outline for better visibility
            cv2.circle(display_frame, (center_x, center_y), VERTEX_RADIUS, (0, 0, 0), 1)

        cv2.imshow("Raw Vertex Detection", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("--- Processing complete. ---")

if __name__ == "__main__":
    main()
