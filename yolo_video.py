#!/usr/bin/env python3
import argparse
import os
import sys
import cv2
import numpy as np
import torch
import math
from collections import deque
from tqdm import tqdm
from ultralytics import YOLO
from scipy.ndimage import maximum_filter
from train_unet import UNet

# Model paths
DEFAULT_YOLO_MODEL = os.path.expanduser("yolo.pt")
DEFAULT_UNET_MODEL = os.path.expanduser("unet.pth")

# Detection parameters
YOLO_CONFIDENCE_THRESHOLD = 0.45   # YOLO confidence threshold
MAX_PIECE_SIZE = 40                # Maximum width or height for a piece in pixels
HEATMAP_THRESHOLD = 0.05           # Threshold for peak detection in heatmap
MIN_VERTEX_DISTANCE = 20           # Minimum distance between vertices in pixels

# Stabilization parameters
STABILIZATION_FRAMES = 45          # k frames for stabilization
STABILIZATION_WINDOW = 60          # Window size to check detection rate
MIN_DETECTION_RATE = 0.95          # Minimum detection rate (95%) within window
MOVEMENT_THRESHOLD = 3             # Pixels - threshold to consider piece moved to new position
MISSING_FRAMES_THRESHOLD = 60      # Remove piece after this many missing frames
MIN_VERTEX_VALIDATION_FRAMES = 45  # Minimum frames with correct vertices to stabilize
VERTEX_HEATMAP_WINDOW = 60         # Sliding window size for heatmap accumulation

# UI
BOX_THICKNESS = 1
TRANSPARENCY_ALPHA = 0.6

# Color mapping for each piece (BGR format for OpenCV)
PIECE_COLORS = {
    'pink_triangle': (203, 192, 255),
    'red_triangle': (0, 0, 255),
    'orange_triangle': (0, 165, 255),
    'blue_triangle': (255, 144, 30),
    'green_triangle': (0, 255, 0),
    'yellow_square': (0, 255, 255),
    'purple_parallelogram': (128, 0, 128)
}

# Map piece type to its expected number of vertices
PIECE_VERTEX_COUNT = {
    'pink_triangle': 3,
    'red_triangle': 3,
    'orange_triangle': 3,
    'blue_triangle': 3,
    'green_triangle': 3,
    'yellow_square': 4,
    'purple_parallelogram': 4
}


class PieceTracker:
    """Tracks a single piece through stabilization phases"""

    def __init__(self, class_name):
        self.class_name = class_name
        self.is_stable = False
        self.frames_missing = 0

        # Data for initial stabilization
        self._reset_stabilization_data()

        # Data for tracking a potential new position for an already-stable piece
        self._reset_new_position_tracking()

        # Final stable data
        self.stable_bbox = None
        self.stable_vertices = []

    def add_detection(self, bbox, heatmap=None, crop_offset=None, crop_shape=None):
        """Add a new detection, handling stabilization and movement."""
        self.frames_missing = 0

        if self.is_stable:
            if self._is_different_position(bbox):
                # Detected far from stable spot. Start/continue tracking this new position.
                self._accumulate_for_new_position(bbox, heatmap, crop_offset, crop_shape)
                if self._should_stabilize_new_position():
                    # New position is confirmed. Commit it.
                    self._commit_new_stable_position()
            else:
                # Detected near the stable spot. This confirms the current position.
                # Reset any tracking of a potential new (and likely spurious) position.
                self._reset_new_position_tracking()
        else:
            # Not yet stable. Accumulate data for initial stabilization.
            self._accumulate_for_initial_stabilization(bbox, heatmap, crop_offset, crop_shape)
            if self._should_stabilize():
                self._stabilize()

    def mark_missing(self):
        """Mark that the piece was not detected in this frame"""
        self.frames_missing += 1

        if self.is_stable:
            # If we're tracking a new position, a miss invalidates that attempt.
            self._reset_new_position_tracking()
        else:
            # A miss counts against the detection rate when trying to stabilize.
            self.detection_window.append(0)
            self.total_frames_processed += 1
            # Reset vertex validation, as we need consecutive frames.
            self.valid_vertex_frames = 0
            self.current_vertices = []

    def should_remove(self):
        """Check if the piece should be removed due to prolonged absence"""
        return self.frames_missing > MISSING_FRAMES_THRESHOLD

    def _accumulate_for_initial_stabilization(self, bbox, heatmap, crop_offset, crop_shape):
        """Accumulate data for a piece that has not yet been stabilized."""
        self.detection_window.append(1)
        self.total_frames_processed += 1
        self.frames_collected += 1
        self.bbox_history.append(bbox)

        if heatmap is not None:
            self.heatmap_history.append(heatmap)
            self.crop_offset_history.append(crop_offset)
            self.crop_shape_history.append(crop_shape)
            vertices = self._extract_vertices_from_heatmaps(
                self.heatmap_history, self.crop_offset_history, self.crop_shape_history
            )
            expected_verts = PIECE_VERTEX_COUNT.get(self.class_name, 4)
            if len(vertices) == expected_verts:
                self.valid_vertex_frames += 1
                self.current_vertices = vertices
            else:
                self.valid_vertex_frames = 0
                self.current_vertices = []

    def _accumulate_for_new_position(self, bbox, heatmap, crop_offset, crop_shape):
        """Accumulate data for a potential new position of an already-stable piece."""
        self.new_detection_window.append(1)
        self.new_total_frames_processed += 1
        self.new_position_frames += 1
        self.new_bbox_history.append(bbox)

        if heatmap is not None:
            self.new_heatmap_history.append(heatmap)
            self.new_crop_offset_history.append(crop_offset)
            self.new_crop_shape_history.append(crop_shape)
            vertices = self._extract_vertices_from_heatmaps(
                self.new_heatmap_history, self.new_crop_offset_history, self.new_crop_shape_history
            )
            expected_verts = PIECE_VERTEX_COUNT.get(self.class_name, 4)
            if len(vertices) == expected_verts:
                self.new_valid_vertex_frames += 1
            else:
                self.new_valid_vertex_frames = 0

    def _should_stabilize(self):
        if self.frames_collected < STABILIZATION_FRAMES: return False
        if self.total_frames_processed < STABILIZATION_WINDOW: return False
        if not self.detection_window: return False
        rate = sum(self.detection_window) / len(self.detection_window)
        return rate >= MIN_DETECTION_RATE and self.valid_vertex_frames >= MIN_VERTEX_VALIDATION_FRAMES

    def _should_stabilize_new_position(self):
        if self.new_position_frames < STABILIZATION_FRAMES: return False
        if self.new_total_frames_processed < STABILIZATION_WINDOW: return False
        if not self.new_detection_window: return False
        rate = sum(self.new_detection_window) / len(self.new_detection_window)
        return rate >= MIN_DETECTION_RATE and self.new_valid_vertex_frames >= MIN_VERTEX_VALIDATION_FRAMES

    def _stabilize(self):
        """Finalize stable values from initial accumulated data."""
        self.stable_bbox = max(self.bbox_history, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        self.stable_vertices = self.current_vertices[:]
        self.is_stable = True
        self._reset_stabilization_data()

    def _commit_new_stable_position(self):
        """Promote the validated new position to be the current stable position."""
        self.stable_bbox = max(self.new_bbox_history, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        # Re-extract vertices from the full history of the new position for best accuracy
        self.stable_vertices = self._extract_vertices_from_heatmaps(
            self.new_heatmap_history, self.new_crop_offset_history, self.new_crop_shape_history
        )
        # Reset the tracking for any future moves
        self._reset_new_position_tracking()

    def _reset_stabilization_data(self):
        """Resets variables used for initial stabilization."""
        self.frames_collected = 0
        self.bbox_history = []
        self.heatmap_history = deque(maxlen=VERTEX_HEATMAP_WINDOW)
        self.crop_offset_history = deque(maxlen=VERTEX_HEATMAP_WINDOW)
        self.crop_shape_history = deque(maxlen=VERTEX_HEATMAP_WINDOW)
        self.detection_window = deque(maxlen=STABILIZATION_WINDOW)
        self.total_frames_processed = 0
        self.valid_vertex_frames = 0
        self.current_vertices = []

    def _reset_new_position_tracking(self):
        """Resets variables used for tracking a potential new position."""
        self.new_position_frames = 0
        self.new_bbox_history = []
        self.new_heatmap_history = deque(maxlen=VERTEX_HEATMAP_WINDOW)
        self.new_crop_offset_history = deque(maxlen=VERTEX_HEATMAP_WINDOW)
        self.new_crop_shape_history = deque(maxlen=VERTEX_HEATMAP_WINDOW)
        self.new_detection_window = deque(maxlen=STABILIZATION_WINDOW)
        self.new_total_frames_processed = 0
        self.new_valid_vertex_frames = 0

    def _is_different_position(self, new_bbox):
        """Check if bbox is in a significantly different position from the stable one."""
        if self.stable_bbox is None: return False
        old_cx = (self.stable_bbox[0] + self.stable_bbox[2]) / 2
        old_cy = (self.stable_bbox[1] + self.stable_bbox[3]) / 2
        new_cx = (new_bbox[0] + new_bbox[2]) / 2
        new_cy = (new_bbox[1] + new_bbox[3]) / 2
        return math.hypot(new_cx - old_cx, new_cy - old_cy) > MOVEMENT_THRESHOLD

    def _extract_vertices_from_heatmaps(self, heatmaps, crop_offsets, crop_shapes):
        if not heatmaps: return []
        try:
            avg_heatmap = np.mean(np.array(list(heatmaps)), axis=0)
            crop_offset, crop_shape = crop_offsets[-1], crop_shapes[-1]
            expected_verts = PIECE_VERTEX_COUNT.get(self.class_name, 4)
            return self._heatmap_to_vertices(avg_heatmap, crop_offset, crop_shape, expected_verts)
        except Exception as e:
            print(f"Warning: Vertex extraction failed: {e}")
            return []

    def _heatmap_to_vertices(self, heatmap, crop_offset, crop_shape,
                           expected_vertex_count,
                           threshold=HEATMAP_THRESHOLD, min_distance=MIN_VERTEX_DISTANCE):
        footprint = np.ones((min_distance, min_distance))
        local_max = maximum_filter(heatmap, footprint=footprint)
        maxima_mask = (heatmap == local_max) & (heatmap > threshold)
        peak_coords = np.argwhere(maxima_mask)
        if peak_coords.shape[0] == 0: return []
        peak_intensities = heatmap[peak_coords[:, 0], peak_coords[:, 1]]
        top_peaks = peak_coords[np.argsort(peak_intensities)[::-1][:expected_vertex_count]]
        vertices = []
        for y_heat, x_heat in top_peaks:
            x_img = (x_heat * crop_shape[1] / 64.0) + crop_offset[0]
            y_img = (y_heat * crop_shape[0] / 64.0) + crop_offset[1]
            vertices.append((int(x_img), int(y_img)))
        return vertices

    def get_display_data(self):
        """Get data for display only if the piece is stable."""
        return {'bbox': self.stable_bbox, 'vertices': self.stable_vertices, 'class_name': self.class_name} if self.is_stable else None


# The rest of the file (BBoxStabilizer, VertexPredictor, main, etc.) remains unchanged.
# I'm including it here for completeness.

class BBoxStabilizer:
    """Simple stabilizer that accumulates k frames before displaying"""
    
    def __init__(self, vertex_predictor):
        self.trackers = {}
        self.vertex_predictor = vertex_predictor
        
    def update(self, detections, frame):
        """Update trackers with new detections"""
        # Process new detections
        for class_name, det in detections.items():
            if class_name not in self.trackers:
                self.trackers[class_name] = PieceTracker(class_name)
            
            tracker = self.trackers[class_name]
            bbox = det['bbox']
            
            # Always get heatmap if a piece is not stable or might have moved.
            # The tracker itself will decide if it needs it.
            heatmap, crop_offset, crop_shape = None, None, None
            crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            if crop.size > 0:
                heatmap = self._get_heatmap(crop)
                crop_offset = (bbox[0], bbox[1])
                crop_shape = (bbox[3] - bbox[1], bbox[2] - bbox[0])
            
            tracker.add_detection(bbox, heatmap, crop_offset, crop_shape)
        
        # Mark trackers as missing if they weren't detected
        detected_classes = set(detections.keys())
        for class_name, tracker in self.trackers.items():
            if class_name not in detected_classes:
                tracker.mark_missing()
        
        # Remove trackers that have been missing too long
        to_remove = [name for name, tracker in self.trackers.items() 
                    if tracker.should_remove()]
        for name in to_remove:
            del self.trackers[name]
        
        # Return stable detections
        stable_detections = {}
        for class_name, tracker in self.trackers.items():
            display_data = tracker.get_display_data()
            if display_data:
                stable_detections[class_name] = display_data
                
        return stable_detections
    
    def _get_heatmap(self, crop):
        """Get heatmap prediction for a crop"""
        try:
            # Resize crop to model input size (64x64)
            crop_resized = cv2.resize(crop, (64, 64))
            crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1]
            crop_normalized = crop_rgb.astype(np.float32) / 255.0
            
            # Convert to tensor: (H, W, C) -> (1, C, H, W)
            crop_tensor = torch.from_numpy(crop_normalized).permute(2, 0, 1).unsqueeze(0)
            crop_tensor = crop_tensor.to(self.vertex_predictor.device)
            
            # Predict heatmap
            with torch.no_grad():
                heatmap_tensor = self.vertex_predictor.model(crop_tensor)
                heatmap = heatmap_tensor.squeeze().cpu().numpy()
            
            return heatmap
            
        except Exception as e:
            print(f"Warning: Heatmap prediction failed: {e}")
            return None


def is_valid_piece_size(bbox, max_size=MAX_PIECE_SIZE):
    """Check if bounding box dimensions are within size constraints"""
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return width <= max_size and height <= max_size


def order_vertices_for_polygon(vertices):
    """Sorts 4 vertices to form a convex polygon"""
    center_x = sum(v[0] for v in vertices) / 4
    center_y = sum(v[1] for v in vertices) / 4
    sorted_vertices = sorted(
        vertices,
        key=lambda v: math.atan2(v[1] - center_y, v[0] - center_x)
    )
    return sorted_vertices


class VertexPredictor:
    """Handles vertex prediction using trained U-Net model"""
    
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load U-Net model
        self.model = UNet(in_channels=3, out_channels=1).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    args = parser.parse_args()
    
    model = YOLO(DEFAULT_YOLO_MODEL)
    vertex_predictor = VertexPredictor(DEFAULT_UNET_MODEL)
    
    stabilizer = BBoxStabilizer(vertex_predictor)
    
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"FATAL ERROR: Could not open video file {args.video}")
        sys.exit(1)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("--- Press 'q' in the display window to quit. ---")
    
    for _ in tqdm(range(total_frames), desc="Processing", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w, _ = frame.shape
        reconstruction_view = np.zeros((h, w, 3), dtype=np.uint8)

        results = model(frame, verbose=False, conf=YOLO_CONFIDENCE_THRESHOLD)
        
        best_detections_by_class = {}
        for box in results[0].boxes:
            class_name = model.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            bbox = tuple(map(int, box.xyxy[0]))
            
            # Apply size constraint - skip detections that are too large
            if not is_valid_piece_size(bbox, MAX_PIECE_SIZE):
                continue
            
            if class_name not in best_detections_by_class or \
               confidence > best_detections_by_class[class_name]['conf']:
                
                best_detections_by_class[class_name] = {
                    'bbox': bbox, 
                    'class_name': class_name,
                    'conf': confidence
                }
        
        stabilized_detections = stabilizer.update(best_detections_by_class, frame)
        
        # --- Draw on both views ---
        for det in stabilized_detections.values():
            bbox = det['bbox']
            class_name = det['class_name']
            color = PIECE_COLORS.get(class_name, (255, 255, 255))
            
            # Get the vertices to be drawn
            drawable_vertices = det.get('vertices', [])
            
            if class_name in ['yellow_square', 'purple_parallelogram'] and len(drawable_vertices) == 4:
                drawable_vertices = order_vertices_for_polygon(drawable_vertices)
            
            # Draw on the original frame
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, BOX_THICKNESS)
            for vx, vy in drawable_vertices:
                cv2.circle(frame, (vx, vy), 3, color, -1)

            # Draw on the reconstruction view
            if len(drawable_vertices) >= 3:
                overlay = reconstruction_view.copy()
                pts = np.array(drawable_vertices, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(overlay, [pts], color)
                cv2.addWeighted(overlay, TRANSPARENCY_ALPHA, reconstruction_view, 1 - TRANSPARENCY_ALPHA, 0, reconstruction_view)
                
                # Draw the vertices on top (fully opaque)
                for vx, vy in drawable_vertices:
                    cv2.circle(reconstruction_view, (vx, vy), 1, color, -1)
        
        combined_view = np.hstack((frame, reconstruction_view))
        cv2.imshow("Tangram Detection", combined_view)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("--- Processing complete. ---")

if __name__ == "__main__":
    main()
