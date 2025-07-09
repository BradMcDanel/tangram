#!/usr/bin/env python3
import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import argparse
import matplotlib.colors as mcolors
from tqdm import tqdm
import sys
import copy
import shutil

# --- Configuration (can be overridden by args) ---
DEFAULT_SAM2_CHECKPOINT = os.path.expanduser("~/data-main-1/sam2/sam2.1_hiera_large.pt")
DEFAULT_MODEL_CONFIG_NAME = "configs/sam2.1/sam2.1_hiera_l.yaml"

# --- Main Application Class ---
class AnnotationTool:
    def __init__(self, frames_to_label, output_dir, predictor, class_map):
        self.frames_to_label = frames_to_label
        self.output_dir = output_dir
        self.predictor = predictor
        self.class_map = class_map
        self.colors = plt.get_cmap('gist_rainbow', len(class_map))
        self.colors = [
            '#FF69B4',  # 1: pink_triangle (HotPink)
            '#DC143C',  # 2: red_triangle (Crimson)
            '#FFA500',  # 3: orange_triangle (Orange)
            '#1E90FF',  # 4: blue_triangle (DodgerBlue)
            '#32CD32',  # 5: green_triangle (LimeGreen)
            '#FFD700',  # 6: yellow_square (Gold)
            '#9400D3'   # 7: purple_parallelogram (DarkViolet)
        ]
        self.is_running = True
        self.next_image_flag = False
        self.current_image = None
        self.fig, self.ax = None, None
        self._reset_state()

    def run(self):
        if not self.frames_to_label:
            print("\nAll frames have been labeled! Nothing to do.")
            return

        self.fig, self.ax = plt.subplots(1, figsize=(12, 8))
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        self.fig.canvas.mpl_connect('close_event', self._on_close)

        total_to_label = len(self.frames_to_label)
        for i, img_path in enumerate(self.frames_to_label):
            if not self.is_running: break
            
            print_instructions(self.class_map)
            print(f"\n--- Loading image {i+1}/{total_to_label}: {os.path.basename(img_path)} ---")
            self._load_new_image(img_path)
            self.next_image_flag = False
            while not self.next_image_flag and self.is_running:
                plt.pause(0.1)
        
        plt.close(self.fig)

    def _load_new_image(self, img_path):
        self.current_img_path = img_path
        self._reset_state()
        self.current_image = np.array(Image.open(img_path).convert("RGB"))
        self.predictor.set_image(self.current_image)
        self.fig.canvas.manager.set_window_title(f"Labeling: {os.path.basename(img_path)}")
        self._redraw_canvas()
        self._push_state_to_history()

    def _reset_state(self):
        self.points, self.point_labels, self.generated_masks = [], [], {}
        self.history, self.redo_stack = [], []

    def _push_state_to_history(self):
        state = {'points': copy.deepcopy(self.points), 'point_labels': copy.deepcopy(self.point_labels),
                 'generated_masks': copy.deepcopy(self.generated_masks)}
        self.history.append(state)
        self.redo_stack = []

    def _redraw_canvas(self):
        self.ax.clear()
        display_image = self.current_image.copy()

        if self.generated_masks:
            for class_id, data in self.generated_masks.items():
                mask = data['mask']
                if isinstance(mask, torch.Tensor): mask = mask.cpu().numpy()
                mask_2d = mask.squeeze().astype(bool)

                color_hex = self.colors[data['class_id']]
                # Convert hex to an RGB tuple (0-1 range) then to a 0-255 range for OpenCV
                class_color_rgb = (np.array(mcolors.to_rgb(color_hex)) * 255).astype(np.uint8)

                # Apply the color fill
                roi = display_image[mask_2d]
                color_block = np.full(roi.shape, class_color_rgb, dtype=np.uint8)
                blended_roi = cv2.addWeighted(roi, 0.5, color_block, 0.5, 0)
                display_image[mask_2d] = blended_roi
        
        self.ax.imshow(display_image)
        
        if self.generated_masks:
            for class_id, data in self.generated_masks.items():
                mask = data['mask']
                if isinstance(mask, torch.Tensor): mask = mask.cpu().numpy()
                mask_2d = mask.squeeze().astype(np.uint8)
                contours, _ = cv2.findContours(mask_2d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                # --- CHANGED: Get outline color from our new list ---
                outline_color = self.colors[data['class_id']]

                for contour in contours:
                    self.ax.plot(contour[:, 0, 0], contour[:, 0, 1], color=outline_color, linewidth=0.8, alpha=0.9)

        for i, (point, label) in enumerate(zip(self.points, self.point_labels)):
            if label == -1:
                self.ax.scatter(point[0], point[1], color='white', marker='+', s=15, linewidth=0.6)
            else:
                # --- CHANGED: Get point color from our new list ---
                point_color = self.colors[label]
                self.ax.scatter(point[0], point[1], color=point_color, marker='o', s=15, edgecolor='white', linewidth=0.8)
        
        self.ax.axis('off')
        self.fig.canvas.draw_idle()

    def _on_click(self, event):
        if event.inaxes != self.ax: return
        self._push_state_to_history()
        self.points.append([int(event.xdata), int(event.ydata)])
        self.point_labels.append(-1)
        self._redraw_canvas()

    def _on_key_press(self, event):
        if event.key in self.class_map: self._assign_class_and_update_masks(int(event.key))
        elif event.key == 's': self._save_labels()
        elif event.key == 'r': self._reset_frame()
        elif event.key == 'u': self._undo()
        elif event.key == 'y': self._redo()
        elif event.key in ['q', 'n']: self.next_image_flag = True

    def _on_close(self, event):
        self.is_running = False; self.next_image_flag = True

    def _assign_class_and_update_masks(self, class_id_to_assign):
        indices_to_classify = [i for i, label in enumerate(self.point_labels) if label == -1]
        if not indices_to_classify: return
        self._push_state_to_history()
        print(f"Assigning class '{self.class_map[str(class_id_to_assign)]}' to {len(indices_to_classify)} point(s) and regenerating all masks.")
        for idx in indices_to_classify: self.point_labels[idx] = class_id_to_assign - 1
        self._update_all_masks(); self._redraw_canvas()

    def _update_all_masks(self):
        if not self.points: return
        all_points_np = np.array(self.points)
        classified_points_indices = [i for i, cid in enumerate(self.point_labels) if cid != -1]
        if not classified_points_indices:
            self.generated_masks = {}; return
        points_by_class = {}
        for i in classified_points_indices:
            class_id = self.point_labels[i]
            if class_id not in points_by_class: points_by_class[class_id] = []
            points_by_class[class_id].append(self.points[i])
        
        new_masks = {}
        for class_id, points_in_class in points_by_class.items():
            prompt_labels, current_prompt_points = [], []
            prompt_labels.extend([1] * len(points_in_class))
            current_prompt_points.extend(points_in_class)
            for other_class_id, other_points in points_by_class.items():
                if other_class_id != class_id:
                    prompt_labels.extend([0] * len(other_points))
                    current_prompt_points.extend(other_points)
            masks, scores, _ = self.predictor.predict(
                point_coords=np.array(current_prompt_points), point_labels=np.array(prompt_labels), multimask_output=True)
            new_masks[class_id] = {'mask': masks[np.argmax(scores)], 'class_id': class_id}
        self.generated_masks = new_masks

    def _save_labels(self):
        if not self.generated_masks: print("No masks generated. Classify points first."); return
        base_name = os.path.splitext(os.path.basename(self.current_img_path))[0]
        label_path = os.path.join(self.output_dir, "labels", f"{base_name}.txt")
        labeled_image_path = os.path.join(self.output_dir, "images", f"{base_name}.png")
        
        H, W, _ = self.current_image.shape
        yolo_labels = []
        for class_id, data in self.generated_masks.items():
            mask_np = data['mask']
            if isinstance(mask_np, torch.Tensor): mask_np = mask_np.cpu().numpy().squeeze()
            contours, _ = cv2.findContours(mask_np.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: continue
            all_points = np.vstack(contours)
            x, y, w, h = cv2.boundingRect(all_points)
            x_c, y_c, norm_w, norm_h = (x+w/2)/W, (y+h/2)/H, w/W, h/H
            yolo_labels.append(f"{class_id} {x_c:.6f} {y_c:.6f} {norm_w:.6f} {norm_h:.6f}")

        with open(label_path, 'w') as f: f.write("\n".join(yolo_labels))
        shutil.move(self.current_img_path, labeled_image_path)
        print(f"\nSUCCESS: Saved {len(yolo_labels)} merged labels to {label_path}")
        print("Progress is saved. Advancing to next image.")
        self.next_image_flag = True

    def _reset_frame(self):
        self._push_state_to_history(); self.points, self.point_labels, self.generated_masks = [], [], {}
        self._redraw_canvas(); print("Annotations for this frame have been reset.")
        
    def _undo(self):
        if len(self.history) > 1:
            self.redo_stack.append(self.history.pop()); self._restore_state_from_history()
            print("Undo successful.")
        else: print("Nothing to undo.")

    def _redo(self):
        if self.redo_stack:
            self.history.append(self.redo_stack.pop()); self._restore_state_from_history()
            print("Redo successful.")
        else: print("Nothing to redo.")

    def _restore_state_from_history(self):
        last_state = self.history[-1]
        self.points = copy.deepcopy(last_state['points'])
        self.point_labels = copy.deepcopy(last_state['point_labels'])
        self.generated_masks = copy.deepcopy(last_state['generated_masks'])
        self._redraw_canvas()

def prepare_frames_for_labeling(video_dir, frame_dir, images_dir, num_frames_per_video, force_rerun=False):
    os.makedirs(frame_dir, exist_ok=True)

    # --- ROBUST CHECK: Only extract if it's a completely fresh start ---
    # A fresh start means both the 'to-do' and final 'images' directories are empty.
    is_fresh_start = not os.listdir(frame_dir) and not os.listdir(images_dir)
    if not force_rerun and not is_fresh_start:
        print(f"Frames already exist in {frame_dir} or {images_dir}. Skipping extraction.")
    else:
        print(f"Extracting frames from videos in {video_dir}...")
        video_paths = [p for p in glob.glob(os.path.join(video_dir, "*.mp4")) if "post" not in os.path.basename(p)]

        for video_path in tqdm(video_paths, desc="Processing Videos"):
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 0: continue

            middle_start_percent = 0.5
            middle_end_percent = 0.9

            start_frame = int(frame_count * middle_start_percent)
            end_frame = int(frame_count * middle_end_percent)
            
            # Ensure the window has frames to sample from
            available_frames_in_window = end_frame - start_frame
            if available_frames_in_window <= 0:
                print(f"Warning: Video '{os.path.basename(video_path)}' is too short to sample from the middle 40%. Skipping.")
                continue

            # Determine how many frames to sample, ensuring it's not more than available in our window
            num_to_sample = min(num_frames_per_video, available_frames_in_window)

            # Use np.linspace to get evenly spaced frame indices within the middle window
            if num_to_sample > 0:
                # np.linspace generates floats, so we round and convert to unique integers
                selected_indices = np.unique(np.linspace(start_frame, end_frame, num=num_to_sample).round()).astype(int)
            else:
                selected_indices = []

            for frame_idx in selected_indices:
                output_path = os.path.join(frame_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_frame_{frame_idx}.png")
                # Only write if it doesn't exist to avoid re-writing on force_rerun
                if not os.path.exists(output_path):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if ret: cv2.imwrite(output_path, frame)
            cap.release()

    return sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith(('.png', '.jpg'))])

def print_instructions(class_map):
    print("\n--- Tangram Annotation Tool ---")
    print("1. Click one or more times on tangram pieces.")
    print("2. Press a number key (1-7) to assign a class to ALL unclassified points.")
    for key, name in class_map.items(): print(f"   '{key}': {name}")
    print("\n--- Other Commands ---")
    print("  's': Save labels and go to next image.")
    print("  'r': Reset all points on current image.")
    print("  'u': Undo last action.  'y': Redo last undone action.")
    print("  'n' or 'q': Skip to next image.  CLOSE WINDOW: Quit.")
    print("---------------------------------\n")

def main():
    parser = argparse.ArgumentParser(description="Interactive SAM2 labeling pipeline for YOLO.")
    parser.add_argument("--video_dir", default="data/videos", help="Directory with input .mp4 files.")
    parser.add_argument("--output_dir", default="yolo_dataset", help="Directory to save the final YOLO dataset.")
    parser.add_argument("--frames_per_video", type=int, default=10, help="Number of frames to extract per video.")
    parser.add_argument("--sam_checkpoint", default=DEFAULT_SAM2_CHECKPOINT, help="Path to SAM 2 model checkpoint.")
    parser.add_argument("--sam_config", default=DEFAULT_MODEL_CONFIG_NAME, help="Path to SAM 2 model config yaml.")
    parser.add_argument("--force_extract", action="store_true", help="Force re-extraction of frames.")
    args = parser.parse_args()

    class_map = {'1': 'pink_triangle', '2': 'red_triangle', '3': 'orange_triangle', '4': 'blue_triangle',
                 '5': 'green_triangle', '6': 'yellow_square', '7': 'purple_parallelogram'}

    frames_to_label_dir = os.path.join(args.output_dir, "frames_to_label")
    images_dir = os.path.join(args.output_dir, "images")
    labels_dir = os.path.join(args.output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    frames_to_label = prepare_frames_for_labeling(args.video_dir, frames_to_label_dir, images_dir, args.frames_per_video, args.force_extract)
    if not frames_to_label: print("No frames found to process or all frames are already labeled. Exiting."); return

    print("\n--- Setting up SAM 2 environment ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    if not os.path.exists(args.sam_checkpoint): print(f"Error: Model checkpoint not found at: {args.sam_checkpoint}"); return

    sam2_model = build_sam2(args.sam_config, args.sam_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    print("Model loaded successfully.")

    tool = AnnotationTool(frames_to_label, args.output_dir, predictor, class_map)
    tool.run()
    
    print("\n--- Labeling pipeline complete or exited early. ---")
    print(f"Your YOLO dataset is located in: {args.output_dir}")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")
        sys.exit(0)
