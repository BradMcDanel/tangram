#!/usr/bin/env python3
import argparse
import os
import subprocess
from tqdm import tqdm

def batch_process_videos(input_dir, output_dir):
    """Processes all videos in a directory using video_tangram_detector.py."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_files = [f for f in os.listdir(input_dir) if f.endswith(('.mp4', '.avi', '.mov'))]

    for video_file in tqdm(video_files, desc="Batch Processing Videos"):
        video_path = os.path.join(input_dir, video_file)
        output_filename = os.path.splitext(video_file)[0] + '.json'
        output_path = os.path.join(output_dir, output_filename)

        command = [
            './video_tangram_detector.py',
            '--video',
            video_path,
            '--output-json',
            output_path,
            '--no-display'
        ]

        print(f"\nProcessing {video_file}...")
        # Note: We need to add argument parsing for output json to video_tangram_detector.py
        # For now, this will just run the script and display the output.
        subprocess.run(command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process videos using video_tangram_detector.py.")
    parser.add_argument("--input-dir", required=True, help="Directory containing videos to process.")
    parser.add_argument("--output-dir", required=True, help="Directory to save output JSON files.")
    args = parser.parse_args()

    batch_process_videos(args.input_dir, args.output_dir)
