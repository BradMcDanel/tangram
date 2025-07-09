#!/usr/bin/env python3
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import argparse
import matplotlib.colors as mcolors
from tqdm import tqdm
import sys
import copy
import shutil

# --- Configuration ---
VERTEX_BBOX_SIZE = 4  # Size in pixels (e.g., 4 means a 4x4 box). An even number is best.
DEFAULT_VIDEO_DIR = "data/videos"
DEFAULT_OUTPUT_DIR = "data/training/vertex"
DEFAULT_FRAMES_PER_VIDEO = 10

# --- NEW: Configuration for Annotation View ---
# Hardcoded crop rectangle [y1, y2, x1, x2] for the area of interest.
# This isolates the top part of the table to make labeling easier.
CROP_RECT = [140, 335, 200, 550] 
UPSCALE_FACTOR = 3.0  # How much to zoom in on the cropped area.

# --- Main Application Class ---
class VertexAnnotationTool:
    def __init__(self, frames_to_label, output_dir, class_map):
        self.frames_to_label = frames_to_label
        self.output_dir = output_dir
        self.class_map = class_map
        self.colors = {
            1: '#FF69B4', 2: '#DC143C', 3: '#FFA500', 4: '#1E90FF',
            5: '#32CD32', 6: '#FFD700', 7: '#9400D3',
        }
        
        # Application state
        self.is_running = True
        self.next_image_flag = False
        self.current_image_to_display = None
        self.original_image_shape = None
        self.fig, self.ax = None, None
        
        self._reset_annotation_state()

    def run(self):
        if not self.frames_to_label:
            print("\nAll frames have been labeled or no frames found. Nothing to do.")
            return

        # --- FIXED: Create figure with better initial sizing ---
        self.fig, self.ax = plt.subplots(1, figsize=(12, 9))
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        self.fig.canvas.mpl_connect('close_event', self._on_close)
        
        # Enable interactive mode for better responsiveness
        plt.ion()

        total_to_label = len(self.frames_to_label)
        for i, img_path in enumerate(self.frames_to_label):
            if not self.is_running: break
            
            self._print_instructions()
            print(f"\n--- Loading image {i+1}/{total_to_label}: {os.path.basename(img_path)} ---")
            self._load_new_image(img_path)
            
            self.next_image_flag = False
            while not self.next_image_flag and self.is_running:
                plt.pause(0.1)
        
        plt.ioff()  # Turn off interactive mode
        if self.fig: plt.close(self.fig)

    def _load_new_image(self, img_path):
        self.current_img_path = img_path
        self._reset_annotation_state()
        
        # Load the full, original image
        full_image = np.array(Image.open(img_path).convert("RGB"))
        self.original_image_shape = full_image.shape
        
        # --- FIXED: Crop and Upscale for Display ---
        y1, y2, x1, x2 = CROP_RECT
        cropped_image = full_image[y1:y2, x1:x2]
        
        new_width = int(cropped_image.shape[1] * UPSCALE_FACTOR)
        new_height = int(cropped_image.shape[0] * UPSCALE_FACTOR)
        print(f"Original crop size: {cropped_image.shape[1]}x{cropped_image.shape[0]}")
        print(f"Upscaled size: {new_width}x{new_height}")
        
        self.current_image_to_display = cv2.resize(
            cropped_image, (new_width, new_height), interpolation=cv2.INTER_NEAREST
        )
        
        # --- FIXED: Adjust figure size based on upscaled image ---
        # Calculate aspect ratio and set appropriate figure size
        aspect_ratio = new_width / new_height
        base_height = 8  # Base height in inches
        fig_width = base_height * aspect_ratio
        fig_height = base_height
        
        # Update figure size
        self.fig.set_size_inches(fig_width, fig_height)
        
        self.fig.canvas.manager.set_window_title(f"Labeling: {os.path.basename(img_path)} (Zoomed {UPSCALE_FACTOR}x)")
        self._redraw_canvas()

    def _reset_annotation_state(self):
        self.current_class_id = None
        self.vertices = []
        self.history, self.redo_stack = [], []

    def _push_state_to_history(self):
        state = {'vertices': copy.deepcopy(self.vertices), 'current_class_id': self.current_class_id}
        self.history.append(state)
        self.redo_stack = []

    def _redraw_canvas(self):
        self.ax.clear()
        
        # --- FIXED: Set explicit limits to show full upscaled image ---
        self.ax.imshow(self.current_image_to_display)
        
        # Ensure the axis shows the full image
        height, width = self.current_image_to_display.shape[:2]
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(height, 0)  # Flip Y axis for image coordinates
        
        # --- FIXED: Scale vertex display elements properly ---
        for (x, y, class_id) in self.vertices:
            color = self.colors.get(class_id, '#FFFFFF')
            # Scale the bbox size for display
            half_size = (VERTEX_BBOX_SIZE / 2) * UPSCALE_FACTOR
            top_left = (x - half_size, y - half_size)
            rect = plt.Rectangle(top_left, half_size * 2, half_size * 2,
                                 linewidth=1.2, edgecolor=color, facecolor='none', alpha=0.9)
            self.ax.add_patch(rect)
            
            # Scale the marker size too
            marker_size = 25 * UPSCALE_FACTOR
            self.ax.scatter(x, y, color=color, marker='+', s=marker_size, linewidth=1.2)
        
        self.ax.axis('off')
        
        # Make sure the plot is tight
        self.ax.set_aspect('equal')
        plt.tight_layout()

        if self.current_class_id:
            class_name = self.class_map.get(str(self.current_class_id), "Unknown")
            class_color = self.colors.get(self.current_class_id, '#FFFFFF')
            self.ax.set_title(f"CURRENTLY LABELING: {class_name.upper()} (Zoom: {UPSCALE_FACTOR}x)",
                              color=class_color, weight='bold', fontsize=14)
        else:
            self.ax.set_title(f"Select a class (1-7) to start labeling (Zoom: {UPSCALE_FACTOR}x)", color='white')

        self.fig.canvas.draw_idle()

    def _on_click(self, event):
        if event.inaxes != self.ax or self.current_class_id is None:
            if self.current_class_id is None: print("Please select a class (1-7) before clicking.")
            return

        self._push_state_to_history()
        x, y = int(event.xdata), int(event.ydata)
        self.vertices.append((x, y, self.current_class_id))
        print(f"Added vertex for '{self.class_map[str(self.current_class_id)]}' at ({x}, {y}) in zoomed view.")
        self._redraw_canvas()

    def _on_key_press(self, event):
        if event.key in self.class_map:
            self.current_class_id = int(event.key)
            print(f"\nSwitched to labeling class: '{self.class_map[event.key]}'")
            self._redraw_canvas()
        elif event.key == 's': self._save_labels()
        elif event.key == 'r': self._reset_frame()
        elif event.key == 'u': self._undo()
        elif event.key == 'y': self._redo()
        elif event.key in ['q', 'n']: self.next_image_flag = True
        elif event.key == 'escape':
            self.current_class_id = None
            print("\nDe-selected class. Choose a number to resume labeling.")
            self._redraw_canvas()

    def _on_close(self, event):
        self.is_running = False
        self.next_image_flag = True

    def _save_labels(self):
        if not self.vertices:
            print("No vertices labeled. Nothing to save."); return
            
        base_name = os.path.splitext(os.path.basename(self.current_img_path))[0]
        label_path = os.path.join(self.output_dir, "labels", f"{base_name}.txt")
        labeled_image_path = os.path.join(self.output_dir, "images", f"{base_name}.png")
        
        H, W, _ = self.original_image_shape
        crop_y1, _, crop_x1, _ = CROP_RECT
        yolo_labels = []

        for (x_zoomed, y_zoomed, class_id) in self.vertices:
            # --- NEW: Transform coordinates back to the original frame ---
            # 1. Reverse the upscaling
            x_cropped = x_zoomed / UPSCALE_FACTOR
            y_cropped = y_zoomed / UPSCALE_FACTOR
            
            # 2. Reverse the crop by adding the top-left offset
            x_original = x_cropped + crop_x1
            y_original = y_cropped + crop_y1

            # YOLO format: class_id x_center y_center width height (normalized)
            yolo_class_id = class_id - 1
            norm_x_c = x_original / W
            norm_y_c = y_original / H
            norm_w = VERTEX_BBOX_SIZE / W
            norm_h = VERTEX_BBOX_SIZE / H
            
            yolo_labels.append(f"{yolo_class_id} {norm_x_c:.6f} {norm_y_c:.6f} {norm_w:.6f} {norm_h:.6f}")

        with open(label_path, 'w') as f: f.write("\n".join(yolo_labels))
        shutil.move(self.current_img_path, labeled_image_path)
        
        print(f"\nSUCCESS: Saved {len(yolo_labels)} vertex labels to {label_path}")
        print("Progress is saved. Advancing to next image.")
        self.next_image_flag = True

    def _reset_frame(self):
        self._push_state_to_history(); self.vertices = []; self._redraw_canvas()
        print("Annotations for this frame have been reset.")
        
    def _undo(self):
        if self.history:
            last_state = self.history.pop()
            self.redo_stack.append({'vertices': copy.deepcopy(self.vertices), 'current_class_id': self.current_class_id})
            self._restore_state(last_state); print("Undo successful.")
        else: print("Nothing to undo.")

    def _redo(self):
        if self.redo_stack:
            state_to_restore = self.redo_stack.pop()
            self._push_state_to_history()
            self._restore_state(state_to_restore); print("Redo successful.")
        else: print("Nothing to redo.")

    def _restore_state(self, state):
        self.vertices = state['vertices']
        self.current_class_id = state['current_class_id']
        self._redraw_canvas()

    def _print_instructions(self):
        print("\n--- Vertex Annotation Tool (Zoomed View) ---")
        print("1. Press a number key (1-7) to SELECT a class to label.")
        for key, name in self.class_map.items(): print(f"   '{key}': {name}")
        print("2. CLICK on the image to place vertices for the selected class.")
        print("\n--- Other Commands ---")
        print("  's': Save labels and move to next image.")
        print("  'r': Reset all vertices on current image.")
        print("  'u': Undo last action.  'y': Redo last undone action.")
        print("  'n' or 'q': Skip to next image (without saving).")
        print("  'esc': De-select current class.")
        print("  CLOSE WINDOW: Quit the application.")
        print("---------------------------------\n")


def prepare_frames_for_labeling(video_dir, frame_dir, images_dir, num_frames_per_video, force_rerun=False):
    os.makedirs(frame_dir, exist_ok=True)
    
    is_fresh_start = not os.listdir(frame_dir) and not os.listdir(images_dir)
    if not force_rerun and not is_fresh_start:
        print(f"Frames already exist in {frame_dir} or {images_dir}. Skipping extraction.")
    else:
        if force_rerun:
            print("Forcing re-extraction...")
            for f in os.listdir(frame_dir): os.remove(os.path.join(frame_dir, f))

        print(f"Extracting frames from videos in {video_dir}...")
        video_paths = [p for p in glob.glob(os.path.join(video_dir, "*.mp4")) if "post" not in os.path.basename(p)]

        for video_path in tqdm(video_paths, desc="Processing Videos"):
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened(): continue
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 0: continue
            
            num_to_sample = min(num_frames_per_video, frame_count)
            if num_to_sample > 0:
                selected_indices = np.unique(np.linspace(0, frame_count - 1, num=num_to_sample).round()).astype(int)
            else:
                selected_indices = []

            for frame_idx in selected_indices:
                output_path = os.path.join(frame_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_frame_{frame_idx}.png")
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret: cv2.imwrite(output_path, frame)
            cap.release()

    return sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith(('.png', '.jpg'))])


def main():
    parser = argparse.ArgumentParser(description="Interactive vertex labeling tool for YOLO.")
    parser.add_argument("--video_dir", default=DEFAULT_VIDEO_DIR, help="Directory with input .mp4 files.")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Directory to save the final YOLO dataset.")
    parser.add_argument("--frames_per_video", type=int, default=DEFAULT_FRAMES_PER_VIDEO, help="Number of frames to extract per video.")
    parser.add_argument("--force_extract", action="store_true", help="Force re-extraction of frames even if folders are not empty.")
    args = parser.parse_args()

    class_map = {
        '1': 'vertex_pink_triangle', '2': 'vertex_red_triangle', '3': 'vertex_orange_triangle',
        '4': 'vertex_blue_triangle', '5': 'vertex_green_triangle', '6': 'vertex_yellow_square',
        '7': 'vertex_purple_parallelogram'
    }

    frames_to_label_dir = os.path.join(args.output_dir, "frames_to_label")
    images_dir = os.path.join(args.output_dir, "images")
    labels_dir = os.path.join(args.output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    frames_to_label = prepare_frames_for_labeling(args.video_dir, frames_to_label_dir, images_dir, args.frames_per_video, args.force_extract)
    if not frames_to_label:
        print("No frames found to process or all frames are already labeled. Exiting."); return

    yaml_path = os.path.join(args.output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"path: {os.path.abspath(args.output_dir)}\n")
        f.write("train: images\n"); f.write("val: images\n"); f.write("\n")
        f.write("names:\n")
        for i in range(1, 8): f.write(f"  {i-1}: {class_map[str(i)]}\n")
    print(f"Created YOLO dataset config: {yaml_path}")

    tool = VertexAnnotationTool(frames_to_label, args.output_dir, class_map)
    tool.run()
    
    print("\n--- Labeling pipeline complete or exited early. ---")
    print(f"Your YOLO vertex dataset is located in: {args.output_dir}")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")
        sys.exit(0)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)
