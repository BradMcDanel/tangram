import cv2
import numpy as np
import json
import argparse
import os
from visualization_utils import draw_piece_filled, draw_piece_on_frame, create_reconstruction_view, get_piece_color

PIECE_ALPHA = 0.6 # Adjust as needed

def main():
    parser = argparse.ArgumentParser(description="Replay tangram detections from an output JSON file.")
    parser.add_argument("--json_file", required=True, help="Path to the output JSON file from the detector.")
    parser.add_argument("--video_bg", type=str, default=None, help="Optional: Path to the original video for background. If not provided, a blank canvas is used, and dimensions/FPS must be in the JSON.")
    parser.add_argument("--speed_factor", type=float, default=1.0, help="Playback speed factor for display, or speed adjustment for output video.")
    parser.add_argument("--start_frame", type=int, default=0, help="Frame index to start replay/processing from.")
    parser.add_argument("--output_video_path", type=str, default=None, help="Optional: Path to save the output as a video file. If provided, display is skipped.")
    args = parser.parse_args()

    if not os.path.exists(args.json_file):
        print(f"Error: JSON file not found at {args.json_file}")
        return

    if args.speed_factor <= 0:
        print(f"Error: --speed_factor must be positive. Got {args.speed_factor}")
        return

    with open(args.json_file, 'r') as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    frames_data = data.get("frames_data", [])

    if not frames_data:
        print("Error: No frame data found in the JSON file.")
        return

    canvas_width, canvas_height, base_video_fps = None, None, None
    cap_bg, video_out = None, None

    if args.video_bg:
        if os.path.exists(args.video_bg):
            cap_bg = cv2.VideoCapture(args.video_bg)
            if not cap_bg.isOpened():
                print(f"Error: Could not open background video {args.video_bg}.")
                return
            canvas_width = int(cap_bg.get(cv2.CAP_PROP_FRAME_WIDTH))
            canvas_height = int(cap_bg.get(cv2.CAP_PROP_FRAME_HEIGHT))
            base_video_fps = cap_bg.get(cv2.CAP_PROP_FPS)
            if not (canvas_width > 0 and canvas_height > 0 and base_video_fps > 0):
                print(f"Error: Invalid video properties from {args.video_bg}.")
                if cap_bg: cap_bg.release()
                return
        else:
            print(f"Error: Background video file not found at {args.video_bg}.")
            return
    else:
        required_meta = ["original_image_width", "original_image_height", "video_fps"]
        if not all(k in metadata for k in required_meta):
            print(f"Error: Missing required metadata ({', '.join(required_meta)}) in JSON, and no --video_bg provided.")
            return
        canvas_width = metadata["original_image_width"]
        canvas_height = metadata["original_image_height"]
        base_video_fps = metadata["video_fps"]
        if not (isinstance(canvas_width, int) and canvas_width > 0 and
                isinstance(canvas_height, int) and canvas_height > 0 and
                isinstance(base_video_fps, (int, float)) and base_video_fps > 0):
            print("Error: Invalid metadata values for dimensions or FPS.")
            return

    display_delay_ms = 1
    if args.output_video_path:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(args.output_video_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        output_fps = base_video_fps * args.speed_factor
        if output_fps <= 0:
            print(f"Error: Calculated output FPS ({output_fps}) is not positive.")
            if cap_bg: cap_bg.release()
            return
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if args.output_video_path.lower().endswith(".avi"):
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_out = cv2.VideoWriter(args.output_video_path, fourcc, output_fps, (canvas_width * 2, canvas_height))
        if not video_out.isOpened():
            print(f"Error: Could not open video writer for {args.output_video_path}")
            if cap_bg: cap_bg.release()
            return
        print(f"Outputting to video: {args.output_video_path} at {output_fps:.2f} FPS.")
    else:
        effective_display_fps = base_video_fps * args.speed_factor
        display_delay_ms = int((1.0 / effective_display_fps) * 1000)
        if display_delay_ms <= 0: display_delay_ms = 1
        print(f"Displaying replay at effective {effective_display_fps:.2f} FPS (delay: {display_delay_ms}ms).")

    current_json_frame_index = 0
    processed_frames_count = 0
    
    if cap_bg and args.start_frame > 0:
        if not cap_bg.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame):
            print(f"Warning: Could not seek background video to frame {args.start_frame}.")

    while current_json_frame_index < len(frames_data):
        frame_info = frames_data[current_json_frame_index]
        json_frame_id = frame_info.get("frame_index", -1)
        
        if json_frame_id < args.start_frame:
            current_json_frame_index += 1
            continue

        # 1. Get Base Background for the current frame iteration
        base_frame_this_iteration = None
        if cap_bg:
            current_video_frame_pos = int(cap_bg.get(cv2.CAP_PROP_POS_FRAMES))
            # Skip video frames if JSON is ahead
            while current_video_frame_pos < json_frame_id:
                ret_skip, _ = cap_bg.read()
                if not ret_skip: # Video ended while trying to skip
                    print(f"End of background video while skipping to sync with JSON frame {json_frame_id}.")
                    cap_bg.release() # Ensure it's released
                    cap_bg = None # Mark as unusable
                    break 
                current_video_frame_pos = int(cap_bg.get(cv2.CAP_PROP_POS_FRAMES))
            
            if cap_bg: # Check if video didn't end during skip
                ret_bg, bg_frame_from_video = cap_bg.read()
                if not ret_bg:
                    print(f"End of background video or error reading frame for JSON frame {json_frame_id}.")
                    break 
                base_frame_this_iteration = bg_frame_from_video.copy()
            else: # Video ended during skip, cannot proceed with background
                base_frame_this_iteration = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        else: # No background video specified
            base_frame_this_iteration = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        # 2. Initialize Final Display Frame for this iteration
        # This starts as a copy of the base background and will have pieces blended onto it.
        display_frame = base_frame_this_iteration.copy()
        
        # Create reconstruction view
        reconstruction_view = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        pieces_in_frame = frame_info.get("pieces", [])
        for piece in pieces_in_frame:
            # Draw piece on main frame (with bounding boxes and vertices)
            draw_piece_on_frame(display_frame, piece)
            
        # Create reconstruction view
        reconstruction_view = create_reconstruction_view((canvas_height, canvas_width, 3), pieces_in_frame)
        
        # Combine both views horizontally like video_tangram_detector.py
        combined_view = np.hstack((display_frame, reconstruction_view))
        final_display_frame = combined_view

        if video_out:
            video_out.write(final_display_frame)
        else:
            cv2.imshow("Tangram Replay", final_display_frame)
        
        processed_frames_count += 1
        if processed_frames_count % 100 == 0 and args.output_video_path:
            print(f"Processed {processed_frames_count} frames for video output...")

        key_wait_duration = 1 if video_out else display_delay_ms
        key = cv2.waitKey(key_wait_duration) & 0xFF
        
        if key == ord('q'):
            print("Quitting...")
            break
        elif not video_out and key == ord(' '): 
            print("Paused. Press any key to continue...")
            cv2.waitKey(0) 
        
        current_json_frame_index +=1

    if cap_bg:
        cap_bg.release()
    if video_out:
        video_out.release()
        print(f"Video successfully saved to {args.output_video_path}")
    if not args.output_video_path: 
        cv2.destroyAllWindows()
        
    print(f"Processing finished. {processed_frames_count} frames handled.")

if __name__ == "__main__":
    main()
