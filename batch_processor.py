#!/usr/bin/env python3
import argparse
import fnmatch
import os
import subprocess
import time
from tqdm import tqdm

def batch_process_videos(input_dir, output_dir, pattern='robot'):
    """Processes all videos in a directory using video_tangram_detector.py."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_files = [f for f in os.listdir(input_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    video_files = sorted([f for f in all_files if fnmatch.fnmatch(f, f'*{pattern}*')])

    start_time = time.time()
    times = []
    
    with tqdm(video_files, desc="Batch Processing Videos", position=0, leave=True) as pbar:
        for i, video_file in enumerate(pbar):
            video_start = time.time()
            
            video_path = os.path.join(input_dir, video_file)
            output_filename = os.path.splitext(video_file)[0] + '.json'
            output_path = os.path.join(output_dir, output_filename)

            command = [
                'python', './video_tangram_detector.py',
                '--video',
                video_path,
                '--output-json',
                output_path,
                '--no-display'
            ]

            subprocess.run(command)
            
            # Calculate timing statistics
            video_time = time.time() - video_start
            times.append(video_time)
            
            # Update progress bar with timing info
            if len(times) > 1:
                avg_time = sum(times) / len(times)
                remaining_videos = len(video_files) - (i + 1)
                eta_seconds = avg_time * remaining_videos
                eta_str = f"{int(eta_seconds//60)}:{int(eta_seconds%60):02d}"
                pbar.set_postfix({
                    'current': f"{video_time:.1f}s",
                    'avg': f"{avg_time:.1f}s", 
                    'ETA': eta_str
                })
            else:
                pbar.set_postfix({'current': f"{video_time:.1f}s"})
    
    total_time = time.time() - start_time
    print(f"\nBatch processing complete! Total time: {total_time:.1f}s ({total_time/60:.1f}min)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process videos using video_tangram_detector.py.")
    parser.add_argument("--input-dir", required=True, help="Directory containing videos to process.")
    parser.add_argument("--output-dir", required=True, help="Directory to save output JSON files.")
    parser.add_argument("--pattern", default="robot", help="Filename pattern to filter files (default: 'robot')")
    args = parser.parse_args()

    batch_process_videos(args.input_dir, args.output_dir, args.pattern)
