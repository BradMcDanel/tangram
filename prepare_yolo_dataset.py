import os
import glob
import random
import shutil
import yaml
import argparse

def create_yolo_dataset(source_dir, train_ratio=0.8):
    """
    Organizes labeled data into a YOLOv8 compatible dataset structure.

    Args:
        source_dir (str): The path to the directory containing 'images' and 'labels' folders.
        train_ratio (float): The proportion of data to be used for training (e.g., 0.8 for 80%).
    """
    print(f"--- Preparing YOLO dataset from source: {source_dir} ---")

    images_src_path = os.path.join(source_dir, 'images')
    labels_src_path = os.path.join(source_dir, 'labels')

    if not os.path.isdir(images_src_path) or not os.path.isdir(labels_src_path):
        print(f"Error: Source directory '{source_dir}' must contain 'images' and 'labels' subdirectories.")
        return

    # Create destination directories
    train_img_path = os.path.join(source_dir, 'train', 'images')
    valid_img_path = os.path.join(source_dir, 'valid', 'images')
    train_lbl_path = os.path.join(source_dir, 'train', 'labels')
    valid_lbl_path = os.path.join(source_dir, 'valid', 'labels')

    os.makedirs(train_img_path, exist_ok=True)
    os.makedirs(valid_img_path, exist_ok=True)
    os.makedirs(train_lbl_path, exist_ok=True)
    os.makedirs(valid_lbl_path, exist_ok=True)

    # Get all image files
    all_images = [f for f in os.listdir(images_src_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(all_images)

    # Split into training and validation sets
    split_index = int(len(all_images) * train_ratio)
    train_images = all_images[:split_index]
    valid_images = all_images[split_index:]

    print(f"Total images: {len(all_images)}")
    print(f"Training set size: {len(train_images)}")
    print(f"Validation set size: {len(valid_images)}")

    # Function to move files
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
                print(f"Warning: Label file not found for {img_file}. Skipping.")
    
    # Move files to their final destinations
    print("\nMoving training files...")
    move_files(train_images, train_img_path, train_lbl_path)
    print("Moving validation files...")
    move_files(valid_images, valid_img_path, valid_lbl_path)

    # Clean up original flat directories if they are now empty
    if not os.listdir(images_src_path): os.rmdir(images_src_path)
    if not os.listdir(labels_src_path): os.rmdir(labels_src_path)
    print("Cleaned up original flat directories.")
    
    # --- Create the YAML file ---
    class_map = {
        0: 'pink_triangle', 1: 'red_triangle', 2: 'orange_triangle', 3: 'blue_triangle',
        4: 'green_triangle', 5: 'yellow_square', 6: 'purple_parallelogram'
    }
    
    # Convert class_map to the format YOLO expects (just a list of names)
    class_names = [class_map[i] for i in sorted(class_map.keys())]

    yaml_data = {
        'path': os.path.abspath(source_dir),  # Absolute path to the dataset root
        'train': os.path.join('train', 'images'),
        'val': os.path.join('valid', 'images'),
        'nc': len(class_names),
        'names': class_names
    }

    yaml_path = os.path.join(source_dir, 'tangrams.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False)

    print(f"\nSUCCESS: YOLO dataset created.")
    print(f"YAML configuration file saved to: {yaml_path}")
    print("\nYou are now ready to train your YOLO model!")
    print(f"Example training command: yolo train data={yaml_path} model=yolov8n.pt epochs=100 imgsz=640")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Organize a labeled dataset for YOLOv8 training.")
    parser.add_argument("--source_dir", default="labeled_data", help="Path to the directory containing 'images' and 'labels' folders.")
    parser.add_argument("--train_split", type=float, default=0.8, help="Fraction of data to use for training (0.0 to 1.0).")
    args = parser.parse_args()

    create_yolo_dataset(args.source_dir, args.train_split)
