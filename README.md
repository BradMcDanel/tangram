# Tangram Detection Pipeline

A system for detecting tangram pieces in videos using YOLO object detection and U-Net vertex prediction.

## Setup

```bash
pip install -r requirements.txt
```

Ensure you have the required model files:
- `yolo.pt` - YOLO model for piece detection
- `unet.pth` - U-Net model for vertex prediction

## Usage

### 1. Process a Video

```bash
python video_tangram_detector.py --video input.mp4 --output-json results.json
```

**Options:**
- `--video`: Input video file (required)
- `--output-json`: Save detection results as JSON
- `--no-display`: Disable visualization during processing

### 2. Visualize Results

```bash
python video_playback.py --json_file results.json
```

**Options:**
- `--json_file`: JSON file from detector (required)
- `--video_bg`: Overlay on original video
- `--output_video_path`: Save visualization as video
- `--speed_factor`: Playback speed multiplier
- `--start_frame`: Start from specific frame

### 3. Batch Process Multiple Videos

```bash
python batch_processor.py --input-dir videos/ --output-dir results/
```

Processes all video files in the input directory and saves JSON results.

## Output Format

Detection results are saved as JSON with frame-by-frame data:

```json
{
  "metadata": {
    "video_file": "input.mp4",
    "total_frames_processed": 1000,
    "video_fps": 30.0
  },
  "frames_data": [
    {
      "frame_index": 0,
      "pieces": [
        {
          "class_name": "orange_triangle",
          "vertices": [[430, 245], [447, 241], [425, 233]],
          "bbox": [420, 230, 450, 250]
        }
      ]
    }
  ]
}
```

## Detected Pieces

- `pink_triangle`, `red_triangle` - Large triangles
- `orange_triangle` - Medium triangle  
- `blue_triangle`, `green_triangle` - Small triangles
- `yellow_square` - Square
- `purple_parallelogram` - Parallelogram