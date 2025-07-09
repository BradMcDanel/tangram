import os
import sys
import glob
import argparse
import subprocess
from multiprocessing import Pool, cpu_count
import time

# --- Configuration ---
VIDEO_DETECTOR_SCRIPT_NAME = "video_tangram_detector.py"
SUPPORTED_VIDEO_EXTENSIONS = ("*.mp4", "*.avi", "*.mov", "*.mkv") # Add more if needed

def run_detection_task(task_args):
    """
    Worker function to run a single detection task.
    task_args is a tuple: (video_file, config_file, output_json_file, python_executable, detector_script_path)
    """
    video_file, config_file, output_json_file, python_executable, detector_script_path = task_args
    
    command = [
        python_executable,
        detector_script_path,
        "--video", video_file,
        "--config", config_file,
        "--output_json", output_json_file,
        "--no_display" # Essential for batch processing
        # Add any other flags you want to pass to all instances, e.g., "--verbose"
    ]

    print(f"Starting: {' '.join(command)}")
    start_time = time.time()
    try:
        # Ensure the directory for the output JSON exists
        os.makedirs(os.path.dirname(output_json_file), exist_ok=True)
        
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        duration = time.time() - start_time
        print(f"SUCCESS: Processed {os.path.basename(video_file)} in {duration:.2f}s. Output: {output_json_file}")
        # if result.stdout: # Uncomment for debugging detector script's stdout
        #     print(f"Stdout for {os.path.basename(video_file)}:\n{result.stdout}")
        return True, video_file, None
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        error_message = (
            f"FAILURE: Error processing {os.path.basename(video_file)} in {duration:.2f}s.\n"
            f"  Command: {' '.join(e.cmd)}\n"
            f"  Return code: {e.returncode}\n"
            f"  Stdout:\n{e.stdout}\n"
            f"  Stderr:\n{e.stderr}"
        )
        print(error_message)
        return False, video_file, error_message
    except FileNotFoundError:
        duration = time.time() - start_time
        error_message = f"FAILURE: Detector script '{detector_script_path}' or python executable '{python_executable}' not found for {os.path.basename(video_file)}."
        print(error_message)
        return False, video_file, error_message
    except Exception as e:
        duration = time.time() - start_time
        error_message = f"FAILURE: An unexpected error occurred with {os.path.basename(video_file)} in {duration:.2f}s: {str(e)}"
        print(error_message)
        return False, video_file, error_message


def main():
    parser = argparse.ArgumentParser(description="Batch process video files for tangram detection.")
    parser.add_argument(
        "--videos_dir", 
        required=True, 
        help="Directory containing input video files."
    )
    parser.add_argument(
        "--output_dir", 
        required=True, 
        help="Directory where output JSON files will be saved."
    )
    parser.add_argument(
        "--config", 
        required=True, 
        help="Path to the tangram_config.json file."
    )
    parser.add_argument(
        "--workers", 
        type=int, 
        default=max(1, cpu_count() - 1), 
        help="Number of parallel worker processes. Defaults to (CPU cores - 1)."
    )
    parser.add_argument(
        "--python_executable",
        default=sys.executable,
        help="Path to the python interpreter to run the detector script (e.g., if using a venv). Defaults to current python."
    )
    parser.add_argument(
        "--detector_script_path",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), VIDEO_DETECTOR_SCRIPT_NAME),
        help=f"Path to the {VIDEO_DETECTOR_SCRIPT_NAME} script. Defaults to being in the same directory as this batch script."
    )

    args = parser.parse_args()

    if not os.path.isdir(args.videos_dir):
        print(f"Error: Videos directory '{args.videos_dir}' not found.")
        sys.exit(1)
    if not os.path.isfile(args.config):
        print(f"Error: Config file '{args.config}' not found.")
        sys.exit(1)
    if not os.path.isfile(args.detector_script_path):
        print(f"Error: Detector script '{args.detector_script_path}' not found.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    video_files_to_process = []
    for ext in SUPPORTED_VIDEO_EXTENSIONS:
        video_files_to_process.extend(glob.glob(os.path.join(args.videos_dir, ext)))
    
    # Also search in subdirectories (optional, remove if not needed)
    for ext in SUPPORTED_VIDEO_EXTENSIONS:
        video_files_to_process.extend(glob.glob(os.path.join(args.videos_dir, "**", ext), recursive=True))
    
    # Remove duplicates if recursive search found same files
    video_files_to_process = sorted(list(set(video_files_to_process)))


    if not video_files_to_process:
        print(f"No video files found in '{args.videos_dir}' with extensions {SUPPORTED_VIDEO_EXTENSIONS}.")
        sys.exit(0)

    print(f"Found {len(video_files_to_process)} video files to process.")
    
    tasks = []
    for video_file_path in video_files_to_process:
        base_name = os.path.basename(video_file_path)
        name_without_ext, _ = os.path.splitext(base_name)
        output_json_file = os.path.join(args.output_dir, f"{name_without_ext}.json")
        tasks.append((video_file_path, args.config, output_json_file, args.python_executable, args.detector_script_path))

    num_workers = min(args.workers, len(tasks))
    if num_workers <= 0:
        num_workers = 1
    
    print(f"Using {num_workers} worker processes.")
    
    overall_start_time = time.time()
    
    with Pool(processes=num_workers) as pool:
        results = pool.map(run_detection_task, tasks)

    overall_duration = time.time() - overall_start_time
    
    successful_tasks = 0
    failed_tasks_info = []

    for success, video_file, error_msg in results:
        if success:
            successful_tasks += 1
        else:
            failed_tasks_info.append((video_file, error_msg))

    print("\n--- Batch Processing Summary ---")
    print(f"Total videos processed: {len(tasks)}")
    print(f"Successful: {successful_tasks}")
    print(f"Failed: {len(failed_tasks_info)}")
    print(f"Total processing time: {overall_duration:.2f} seconds.")

    if failed_tasks_info:
        print("\n--- Details for Failed Tasks ---")
        for video_file, error_msg in failed_tasks_info:
            print(f"\nVideo: {video_file}")
            print(f"Error: {error_msg if error_msg else 'Unknown error during subprocess execution.'}")
            # You might want to log these errors to a file as well
    
    print("\nBatch processing finished.")

if __name__ == "__main__":
    main()
