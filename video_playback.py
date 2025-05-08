import cv2
import numpy as np
import json
import argparse
import os
import time

COLOR_MAP = {
    "red": (0, 0, 255), "blue": (255, 0, 0), "green": (0, 255, 0),
    "yellow": (0, 255, 255), "orange": (0, 165, 255), "purple": (128, 0, 128),
    "pink": (203, 192, 255), "cyan": (255, 255, 0), "magenta": (255, 0, 255),
    "lime": (0, 255, 128), "teal": (128, 128, 0), "brown": (42, 42, 165),
    "default": (200, 200, 200)
}
OUTLINE_COLOR = (50, 50, 50)
OUTLINE_THICKNESS = 2

def get_piece_color(color_name_str):
    return COLOR_MAP.get(color_name_str.lower(), COLOR_MAP["default"])

def main():
    parser = argparse.ArgumentParser(description="Replay tangram detections from an output JSON file.")
    parser.add_argument("--json_file", required=True, help="Path to the output JSON file from the detector.")
    parser.add_argument("--video_bg", type=str, default=None, help="Optional: Path to the original video for background. If not provided, a blank canvas is used, and dimensions/FPS must be in the JSON.")
    parser.add_argument("--speed_factor", type=float, default=1.0, help="Playback speed factor (e.g., 1.0 for normal, 2.0 for 2x speed, 0.5 for half speed).")
    parser.add_argument("--start_frame", type=int, default=0, help="Frame index to start replay from.")
    args = parser.parse_args()

    if not os.path.exists(args.json_file):
        print(f"Error: JSON file not found at {args.json_file}")
        return

    with open(args.json_file, 'r') as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    frames_data = data.get("frames_data", [])

    if not frames_data:
        print("Error: No frame data found in the JSON file.")
        return

    canvas_width = None
    canvas_height = None
    video_fps = None
    cap_bg = None

    if args.video_bg:
        if os.path.exists(args.video_bg):
            cap_bg = cv2.VideoCapture(args.video_bg)
            if not cap_bg.isOpened():
                print(f"Error: Could not open background video {args.video_bg}.")
                return
            
            canvas_width = int(cap_bg.get(cv2.CAP_PROP_FRAME_WIDTH))
            canvas_height = int(cap_bg.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_fps = cap_bg.get(cv2.CAP_PROP_FPS)

            if canvas_width <= 0 or canvas_height <= 0:
                print(f"Error: Invalid dimensions ({canvas_width}x{canvas_height}) from video file {args.video_bg}.")
                if cap_bg: cap_bg.release()
                return
            if video_fps <= 0:
                print(f"Error: Invalid FPS ({video_fps}) from video file {args.video_bg}.")
                if cap_bg: cap_bg.release()
                return
        else:
            print(f"Error: Background video file not found at {args.video_bg}.")
            return
    else: # No --video_bg, rely on JSON metadata
        if "image_width" not in metadata or "image_height" not in metadata:
            print("Error: 'image_width' and/or 'image_height' not found in JSON metadata, and no --video_bg provided.")
            return
        canvas_width = metadata["image_width"]
        canvas_height = metadata["image_height"]

        if "video_fps" not in metadata:
            print("Error: 'video_fps' not found in JSON metadata, and no --video_bg provided.")
            return
        video_fps = metadata["video_fps"]

        if not isinstance(canvas_width, int) or canvas_width <= 0 or \
           not isinstance(canvas_height, int) or canvas_height <= 0:
            print(f"Error: Invalid 'image_width' or 'image_height' ({canvas_width}x{canvas_height}) in JSON metadata.")
            return
        if not isinstance(video_fps, (int, float)) or video_fps <= 0:
            print(f"Error: Invalid 'video_fps' ({video_fps}) in JSON metadata.")
            return

    delay_ms = int((1.0 / (video_fps * args.speed_factor)) * 1000)
    if delay_ms <= 0: delay_ms = 1

    current_json_frame_index = 0
    
    if cap_bg and args.start_frame > 0:
        cap_bg.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    while current_json_frame_index < len(frames_data):
        frame_info = frames_data[current_json_frame_index]
        
        if frame_info.get("frame_index", -1) < args.start_frame:
            current_json_frame_index +=1
            continue

        if cap_bg:
            ret_bg, bg_frame = cap_bg.read()
            if not ret_bg:
                print("End of background video or error reading frame.")
                break 
            display_frame = bg_frame.copy()
        else:
            display_frame = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        pieces_in_frame = frame_info.get("pieces", [])
        for piece in pieces_in_frame:
            vertices = piece.get("vertices", [])
            if vertices:
                pts = np.array(vertices, dtype=np.int32).reshape((-1, 1, 2))
                color_name = piece.get("color_name", "default")
                fill_color = get_piece_color(color_name)

                cv2.fillPoly(display_frame, [pts], fill_color)
                cv2.polylines(display_frame, [pts], isClosed=True, color=OUTLINE_COLOR, thickness=OUTLINE_THICKNESS)

        cv2.imshow("Tangram Replay", display_frame)
        key = cv2.waitKey(delay_ms) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '): 
            cv2.waitKey(0) 
        
        current_json_frame_index +=1

    if cap_bg:
        cap_bg.release()
    cv2.destroyAllWindows()
    print("Replay finished.")

if __name__ == "__main__":
    main()
