import cv2
import numpy as np
import json
import argparse
import os

def detect_table_roi(image):
    """
    Automatically detects the table in the image to create a Region of Interest.
    This function is adapted from the color-based detection in the provided script.

    Args:
        image (np.ndarray): The input image in BGR format.

    Returns:
        list or None: A list [x1, y1, x2, y2] representing the bounding box of the
                      table, or None if no table is found.
    """
    # HSV color range for the table, lifted from the provided script.
    TABLE_LOWER_HSV = np.array([100, 0, 115])
    TABLE_UPPER_HSV = np.array([179, 35, 190])

    # Convert the image to HSV and create a mask for the table color
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, TABLE_LOWER_HSV, TABLE_UPPER_HSV)
    
    # Clean up the mask to remove noise and fill holes in the table
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Find contours of the table
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("WARNING: No table contour found.")
        return None

    # Assume the largest contour is the table
    table_contour = max(contours, key=cv2.contourArea)
    
    # Get the simple bounding box of the table contour. This is our ROI.
    # This is more robust than trying to find exact corners (approxPolyDP).
    x, y, w, h = cv2.boundingRect(table_contour)
    
    print(f"INFO: Auto-detected table ROI at [x={x}, y={y}, w={w}, h={h}]")
    return [x, y, x + w, y + h]

def find_tangram_centers(image, config):
    """
    Detects tangram centers, first by auto-detecting a table ROI.
    """
    output_image = image.copy()
    
    # --- DYNAMIC ROI DETECTION ---
    table_roi = detect_table_roi(image)
    
    roi_mask = np.zeros(image.shape[:2], dtype="uint8")
    if table_roi is not None:
        x1, y1, x2, y2 = table_roi
        # Create a mask for the detected table area
        cv2.rectangle(roi_mask, (x1, y1), (x2, y2), 255, -1)
        # Draw the detected ROI for visualization
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 192, 0), 2)
    else:
        # Fallback to searching the entire image if no table is found
        roi_mask.fill(255)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blurred_image = cv2.GaussianBlur(hsv_image, (7, 7), 0)
    detected_centers = {}

    for piece_id, params in config.items():
        color_name = params['color_name']
        class_name = params['class_name']
        piece_name = f"{color_name}_{class_name}"
        
        lower_bound = np.array(params['hsv_lower'])
        upper_bound = np.array(params['hsv_upper'])
        color_mask = cv2.inRange(blurred_image, lower_bound, upper_bound)
        
        # Apply the ROI mask to the color mask
        final_mask = cv2.bitwise_and(color_mask, color_mask, mask=roi_mask)

        kernel = np.ones((5, 5), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            continue
            
        main_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(main_contour)
        
        min_area_threshold = params.get('min_area', 20)
        if area > min_area_threshold:
            M = cv2.moments(main_contour)
            if M["m00"] > 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                detected_centers[piece_name] = (cX, cY)

                # Smaller circles for visualization
                dot_color = image[cY, cX].tolist()
                cv2.drawContours(output_image, [main_contour], -1, (0, 255, 0), 1)
                cv2.circle(output_image, (cX, cY), 4, (0, 0, 0), -1) 
                cv2.circle(output_image, (cX, cY), 3, dot_color, -1)
                
                print(f"SUCCESS [{piece_name}]: Found center at ({cX}, {cY})")
            
    return detected_centers, output_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tangram detector with automatic table ROI detection.")
    parser.add_argument('-i', '--image', type=str, required=True, help="Path to input image.")
    parser.add_argument('-c', '--config', type=str, required=True, help="Path to tangram JSON config.")
    parser.add_argument('-o', '--output', type=str, help="Path to save the output image.")
    args = parser.parse_args()

    # The file handling logic remains the same
    if not os.path.exists(args.config): print(f"FATAL: Config file not found at {args.config}"); exit()
    if not os.path.exists(args.image): print(f"FATAL: Image file not found at {args.image}"); exit()
    
    output_path = args.output
    if not output_path:
        base, ext = os.path.splitext(args.image)
        output_path = f"{base}_detected{ext}"

    with open(args.config, 'r') as f:
        config = json.load(f)
    image = cv2.imread(args.image)
    
    centers, result_image = find_tangram_centers(image, config)

    print("\n--- Detection Summary ---")
    if centers:
        for name, coords in sorted(centers.items()):
            print(f"{name}: {coords}")
    else:
        print("No tangram pieces were detected.")
    print("-------------------------\n")
    
    cv2.imwrite(output_path, result_image)
    print(f"✅ Output image with detections saved to: {output_path}")
