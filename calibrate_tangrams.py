import cv2
import numpy as np
import json
import os
import sys
import argparse

# Placeholder function for trackbar callbacks
def nothing(x):
    pass

def calibrate_tangram_from_image(input_json_data, image_path, color_space='hsv'):
    """
    Performs Phase 0: Calibration using a static image, with choice of color space.

    Args:
        input_json_data (list): Loaded JSON data describing tangram pieces.
        image_path (str): Path to the static calibration image file.
        color_space (str): 'hsv' or 'lab'.

    Returns:
        dict: Calibrated parameters keyed by object_id. Includes chosen color_space.
              Returns None if calibration fails or is aborted.
    """
    if not os.path.exists(image_path):
        print(f"Error: Calibration image file not found at {image_path}")
        return None

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image file {image_path}")
        return None

    # --- Preprocessing ---
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0) # Apply blur

    # --- Color Space Conversion ---
    if color_space == 'hsv':
        converted_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
        ch1_name, ch2_name, ch3_name = 'H', 'S', 'V'
        ch1_max = 179 # OpenCV Hue max
    elif color_space == 'lab':
        converted_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2LAB)
        ch1_name, ch2_name, ch3_name = 'L', 'a', 'b'
        ch1_max = 255
    else:
        print(f"Error: Invalid color space '{color_space}'. Choose 'hsv' or 'lab'.")
        return None

    print(f"Using image '{os.path.basename(image_path)}' ({frame.shape[1]}x{frame.shape[0]}) for calibration in {color_space.upper()} space.")

    calibrated_params = {}

    # --- UI Setup ---
    cv2.namedWindow('Calibration View (Masked)')
    cv2.namedWindow('Controls')

    # Create trackbars - Ranges are 0-255 for S, V, L, a, b. Only H is 0-179.
    cv2.createTrackbar(f'{ch1_name}_min', 'Controls', 0, ch1_max, nothing)
    cv2.createTrackbar(f'{ch2_name}_min', 'Controls', 0, 255, nothing)
    cv2.createTrackbar(f'{ch3_name}_min', 'Controls', 0, 255, nothing)
    cv2.createTrackbar(f'{ch1_name}_max', 'Controls', ch1_max, ch1_max, nothing)
    cv2.createTrackbar(f'{ch2_name}_max', 'Controls', 255, 255, nothing)
    cv2.createTrackbar(f'{ch3_name}_max', 'Controls', 255, 255, nothing)

    print("\n--- Calibration Start ---")
    print(f"Using {color_space.upper()} color space.")
    print("Adjust sliders to isolate the target piece (shown highlighted).")
    print("Press 'n' or ENTER to confirm settings for the current piece.")
    print("Press 'q' to quit calibration.")

    for piece_info in input_json_data:
        required_keys = ['object_id', 'color_name', 'class_name', 'num_vertices']
        if not all(key in piece_info for key in required_keys):
            print(f"Error: Input JSON entry missing required keys: {piece_info}")
            cv2.destroyAllWindows()
            return None

        object_id = piece_info['object_id']
        color_name = piece_info['color_name']
        class_name = piece_info['class_name']
        num_vertices = piece_info['num_vertices']

        print(f"\nCalibrating: ID {object_id} - {color_name} {class_name}")
        print("Adjust sliders until only this piece is clearly highlighted.")

        # Reset trackbars (adjust starting points maybe, esp. for Lab)
        cv2.setTrackbarPos(f'{ch1_name}_min', 'Controls', 0)
        cv2.setTrackbarPos(f'{ch2_name}_min', 'Controls', 50 if color_space == 'hsv' else 0) # Start S at 50, a/b at 0
        cv2.setTrackbarPos(f'{ch3_name}_min', 'Controls', 50 if color_space == 'hsv' else 0) # Start V at 50, a/b at 0
        cv2.setTrackbarPos(f'{ch1_name}_max', 'Controls', ch1_max)
        cv2.setTrackbarPos(f'{ch2_name}_max', 'Controls', 255)
        cv2.setTrackbarPos(f'{ch3_name}_max', 'Controls', 255)

        while True:
            # Get current trackbar positions
            ch1_min = cv2.getTrackbarPos(f'{ch1_name}_min', 'Controls')
            ch2_min = cv2.getTrackbarPos(f'{ch2_name}_min', 'Controls')
            ch3_min = cv2.getTrackbarPos(f'{ch3_name}_min', 'Controls')
            ch1_max = cv2.getTrackbarPos(f'{ch1_name}_max', 'Controls')
            ch2_max = cv2.getTrackbarPos(f'{ch2_name}_max', 'Controls')
            ch3_max = cv2.getTrackbarPos(f'{ch3_name}_max', 'Controls')

            # Fail Fast: Ensure min <= max
            if ch1_min > ch1_max or ch2_min > ch2_max or ch3_min > ch3_max:
                 display_frame = frame.copy() # Show original frame on error
                 cv2.putText(display_frame, "INVALID RANGE (min > max)", (10, 30),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                 cv2.imshow('Calibration View (Masked)', display_frame)
                 key = cv2.waitKey(30) & 0xFF
                 if key == ord('q'):
                     print("Calibration aborted by user.")
                     cv2.destroyAllWindows()
                     return None
                 continue

            lower_bound = np.array([ch1_min, ch2_min, ch3_min])
            upper_bound = np.array([ch1_max, ch2_max, ch3_max])

            # Create mask in the chosen color space
            mask = cv2.inRange(converted_frame, lower_bound, upper_bound)

            # Noise Reduction
            kernel = np.ones((3,3), np.uint8)
            mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask_processed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel, iterations=1)

            # --- UI Change: Apply mask to original frame for preview ---
            result_display = cv2.bitwise_and(frame, frame, mask=mask_processed)
            cv2.imshow('Calibration View (Masked)', result_display)
            # --- End UI Change ---

            key = cv2.waitKey(30) & 0xFF

            if key == ord('n') or key == 13: # 'n' or Enter key
                # Find contours on the FINAL processed mask
                contours, _ = cv2.findContours(mask_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                print(f"Found {len(contours)} contours with current settings.") # Feedback

                # Fail Fast Check: Expect exactly one significant contour
                if len(contours) == 0:
                    print("Error: No contours found. Please refine the mask.")
                    continue
                elif len(contours) > 1:
                    areas = [cv2.contourArea(c) for c in contours]
                    sorted_areas = sorted(areas, reverse=True)
                    # Heuristic: largest is 5x bigger than next, and next is reasonably small
                    if len(sorted_areas) > 1 and sorted_areas[0] > sorted_areas[1] * 5 and sorted_areas[1] < 50:
                         print("Warning: Multiple contours found, selecting the largest (others small).")
                         contours = [contours[areas.index(sorted_areas[0])]]
                    else:
                        print(f"Error: Found {len(contours)} significant contours. Please refine the mask to isolate only ONE piece.")
                        # Draw contours on the result display to help
                        temp_display = result_display.copy()
                        cv2.drawContours(temp_display, contours, -1, (0, 0, 255), 2)
                        cv2.imshow('Calibration View (Masked)', temp_display)
                        continue

                target_contour = contours[0]
                area = cv2.contourArea(target_contour)

                MIN_ACCEPTABLE_AREA = 10
                if area < MIN_ACCEPTABLE_AREA:
                     print(f"Error: Contour area ({area:.1f}) is below minimum ({MIN_ACCEPTABLE_AREA}). Likely noise. Please refine the mask.")
                     continue

                min_area = int(area * 0.6)
                max_area = int(area * 1.4)

                # Store the bounds used (important!) and the color space
                calibrated_params[object_id] = {
                    f'{color_space}_lower': lower_bound.tolist(),
                    f'{color_space}_upper': upper_bound.tolist(),
                    'min_area': min_area,
                    'max_area': max_area,
                    'num_vertices': num_vertices,
                    'color_name': color_name,
                    'class_name': class_name,
                    'color_space': color_space # Store which space was used
                }
                print(f"Saved: ID {object_id}, Area={area:.0f}, MinArea={min_area}, MaxArea={max_area}")
                print(f"   {color_space.upper()} Lower: {lower_bound}, Upper: {upper_bound}")

                # Draw accepted contour briefly on the result display
                temp_display = result_display.copy()
                cv2.drawContours(temp_display, [target_contour], -1, (0, 255, 0), 2)
                cv2.imshow('Calibration View (Masked)', temp_display)
                cv2.waitKey(500)
                # No need to restore view, loop continues or exits

                break # Move to next piece

            elif key == ord('q'):
                print("Calibration aborted by user.")
                cv2.destroyAllWindows()
                return None

    # --- Calibration Finished ---
    print("\n--- Calibration Complete ---")
    cv2.destroyAllWindows()

    if len(calibrated_params) != len(input_json_data):
         print(f"Error: Calibration was not completed for all pieces. Expected {len(input_json_data)}, got {len(calibrated_params)}.")
         return None

    print("Successfully calibrated all pieces.")
    return calibrated_params

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate Color/Area thresholds for Tangram pieces using a static image.")
    parser.add_argument("-i", "--image", required=True, help="Path to the static calibration image file.")
    parser.add_argument("-c", "--config", required=True, help="Path to the input JSON config file defining the pieces.")
    parser.add_argument("-o", "--output", default="calibrated_tangram_params.json", help="Path to save the calibrated parameters JSON file.")
    parser.add_argument("-s", "--colorspace", default="hsv", choices=['hsv', 'lab'], help="Color space to use for calibration (hsv or lab). Default: hsv.")

    args = parser.parse_args()

    # File existence checks
    if not os.path.exists(args.image):
        print(f"Error: Calibration image file not found: {args.image}")
        sys.exit(1)
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    # Load input JSON
    try:
        with open(args.config, 'r') as f:
            input_data = json.load(f)
        if not isinstance(input_data, list):
             raise ValueError("JSON config root must be a list.")
    except Exception as e:
        print(f"Error reading or parsing config file {args.config}: {e}")
        sys.exit(1)

    # Run calibration
    calibrated_data = calibrate_tangram_from_image(input_data, args.image, args.colorspace)

    # Process results
    if calibrated_data:
        print("\nCalibrated Parameters:")
        print(json.dumps(calibrated_data, indent=2))
        try:
            with open(args.output, "w") as f:
                json.dump(calibrated_data, f, indent=2)
            print(f"\nCalibrated parameters saved to '{args.output}'")
        except Exception as e:
            print(f"\nError saving calibrated parameters to '{args.output}': {e}")
            sys.exit(1)
    else:
        print("\nCalibration failed or was aborted. No output file saved.")
        sys.exit(1)

    print("\nCalibration script finished successfully.")
