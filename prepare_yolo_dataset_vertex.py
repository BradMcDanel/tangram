import os
import glob
import random
import shutil
import yaml
import argparse

def create_yolo_dataset(source_dir, train_ratio=0.8):
    """
    Organizes vertex-labeled data from the annotation tool into a
    YOLOv8 compatible dataset structure with train/validation splits.

    Args:
        source_dir (str): The path to the dataset directory which contains
                          the flat 'images' and 'labels' folders.
        train_ratio (float): The proportion of data to be used for training.
    """
    print(f"--- Preparing YOLO vertex dataset from source: {source_dir} ---")

    images_src_path = os.path.join(source_dir, 'images')
    labels_src_path = os.path.join(source_dir, 'labels')

    if not os.path.isdir(images_src_path) or not os.path.isdir(labels_src_path):
        print(f"Error: Source directory '{source_dir}' must contain 'images' and 'labels' subdirectories.")
        print("Please run the vertex annotation script first.")
        return

    # Create destination directories for train/validation splits
    train_img_path = os.path.join(source_dir, 'train', 'images')
    valid_img_path = os.path.join(source_dir, 'valid', 'images')
    train_lbl_path = os.path.join(source_dir, 'train', 'labels')
    valid_lbl_path = os.path.join(source_dir, 'valid', 'labels')

    os.makedirs(train_img_path, exist_ok=True)
    os.makedirs(valid_img_path, exist_ok=True)
    os.makedirs(train_lbl_path, exist_ok=True)
    os.makedirs(valid_lbl_path, exist_ok=True)

    # Get all image files from the source directory
    all_images = [f for f in os.listdir(images_src_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not all_images:
        print("No images found in the source directory. Nothing to split.")
        return
        
    random.shuffle(all_images)

    # Split into training and validation sets
    split_index = int(len(all_images) * train_ratio)
    train_images = all_images[:split_index]
    valid_images = all_images[split_index:]

    print(f"Total labeled images: {len(all_images)}")
    print(f"  -> Training set size: {len(train_images)}")
    print(f"  -> Validation set size: {len(valid_images)}")

    # Helper function to move image-label pairs
    def move_files(file_list, dest_img_path, dest_lbl_path):
        for img_file in file_list:
            base_name = os.path.splitext(img_file)[0]
            lbl_file = f"{base_name}.txt"

            src_img = os.path.join(images_src_path, img_file)
            src_lbl = os.path.join(labels_src_path, lbl_file)

            if os.path.exists(src_lbl):
                shutil.move(src_img, os.path.join(dest_img_path, img_file))
                shutil.move(src_lbl, os.path.join(dest_lbl_path, lbl_file))
            else:
                print(f"Warning: Label file '{lbl_file}' not found for image '{img_file}'. Skipping this pair.")
    
    # Move files to their final destinations
    print("\nMoving training files...")
    move_files(train_images, train_img_path, train_lbl_path)
    print("Moving validation files...")
    move_files(valid_images, valid_img_path, valid_lbl_path)

    # Clean up original flat directories if they are now empty
    try:
        if not os.listdir(images_src_path): os.rmdir(images_src_path)
        if not os.listdir(labels_src_path): os.rmdir(labels_src_path)
        print("Cleaned up original flat 'images' and 'labels' directories.")
    except OSError as e:
        print(f"Could not remove source directories (they might not be empty): {e}")
    
    # --- Create the YAML file ---
    # This class map MUST match the one in the vertex annotation script.
    class_map = {
        0: 'vertex_pink_triangle', 1: 'vertex_red_triangle', 2: 'vertex_orange_triangle',
        3: 'vertex_blue_triangle', 4: 'vertex_green_triangle', 5: 'vertex_yellow_square',
        6: 'vertex_purple_parallelogram'
    }
    
    # Convert class_map to the list of names format that YOLO expects
    class_names = [class_map[i] for i in sorted(class_map.keys())]

    yaml_data = {
        'path': os.path.abspath(source_dir),  # Absolute path to the dataset root
        'train': os.path.join('train', 'images'),
        'val': os.path.join('valid', 'images'),
        'nc': len(class_names),
        'names': class_names
    }

    yaml_path = os.path.join(source_dir, 'tangram_vertices.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False, default_flow_style=False)

    print(f"\nSUCCESS: YOLO dataset created.")
    print(f"YAML configuration file saved to: {yaml_path}")
    print("\nYou are now ready to train your YOLO model!")
    print(f"Example training command: yolo train data='{yaml_path}' model=yolov8n.pt epochs=100 imgsz=640")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Organize a vertex-labeled dataset for YOLOv8 training.")
    parser.add_argument(
        "--source_dir", 
        default="yolo_vertex_dataset", 
        help="Path to the directory containing the flat 'images' and 'labels' folders from the annotation tool."
    )
    parser.add_argument(
        "--train_split", 
        type=float, 
        default=0.8, 
        help="Fraction of data to use for training (e.g., 0.8 for 80%% train, 20%% valid)."
    )
    args = parser.parse_args()

    # Basic validation for train_split
    if not 0.0 < args.train_split < 1.0:
        print("Error: --train_split must be a value between 0.0 and 1.0 (exclusive).")
    else:
        create_yolo_dataset(args.source_dir, args.train_split)
