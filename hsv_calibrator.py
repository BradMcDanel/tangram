import cv2
import numpy as np
import argparse

def nothing(x):
    pass

def main():
    parser = argparse.ArgumentParser(description="HSV Color Range Calibrator for an image.")
    parser.add_argument("--image", required=True, help="Path to the input image for calibration.")
    args = parser.parse_args()

    # Load the image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image from {args.image}")
        return

    # Create a window for trackbars
    cv2.namedWindow('HSV Trackbars')
    cv2.resizeWindow('HSV Trackbars', 600, 300) # Adjust size as needed

    # Create trackbars for color change
    # Hue is 0-179 in OpenCV
    cv2.createTrackbar('H_min', 'HSV Trackbars', 0, 179, nothing)
    cv2.createTrackbar('H_max', 'HSV Trackbars', 179, 179, nothing)
    # Saturation is 0-255
    cv2.createTrackbar('S_min', 'HSV Trackbars', 0, 255, nothing)
    cv2.createTrackbar('S_max', 'HSV Trackbars', 255, 255, nothing)
    # Value is 0-255
    cv2.createTrackbar('V_min', 'HSV Trackbars', 0, 255, nothing)
    cv2.createTrackbar('V_max', 'HSV Trackbars', 255, 255, nothing)

    # Set default values (optional, you can start from scratch)
    # Example: A common starting point for a generic blue
    # cv2.setTrackbarPos('H_min', 'HSV Trackbars', 90)
    # cv2.setTrackbarPos('H_max', 'HSV Trackbars', 130)
    # cv2.setTrackbarPos('S_min', 'HSV Trackbars', 50)
    # cv2.setTrackbarPos('S_max', 'HSV Trackbars', 255)
    # cv2.setTrackbarPos('V_min', 'HSV Trackbars', 50)
    # cv2.setTrackbarPos('V_max', 'HSV Trackbars', 255)

    print("Adjust trackbars to isolate the desired color.")
    print("Press 'q' to quit and print the HSV values.")

    while True:
        # Get current positions of all trackbars
        h_min = cv2.getTrackbarPos('H_min', 'HSV Trackbars')
        h_max = cv2.getTrackbarPos('H_max', 'HSV Trackbars')
        s_min = cv2.getTrackbarPos('S_min', 'HSV Trackbars')
        s_max = cv2.getTrackbarPos('S_max', 'HSV Trackbars')
        v_min = cv2.getTrackbarPos('V_min', 'HSV Trackbars')
        v_max = cv2.getTrackbarPos('V_max', 'HSV Trackbars')

        # Form lower and upper HSV arrays
        lower_bound = np.array([h_min, s_min, v_min])
        upper_bound = np.array([h_max, s_max, v_max])

        # Convert the BGR image to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Create a mask using the inRange function
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

        # Apply the mask to the original image
        result_image = cv2.bitwise_and(image, image, mask=mask)

        # Display the original image, the mask, and the result
        cv2.imshow('Original Image', image)
        cv2.imshow('Mask', mask)
        cv2.imshow('Result (Masked Image)', result_image)

        # Wait for a key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Print the final HSV values
    print("\n--- Final HSV Values ---")
    print(f"TABLE_BLUE_HSV_LOWER = np.array([{h_min}, {s_min}, {v_min}])")
    print(f"TABLE_BLUE_HSV_UPPER = np.array([{h_max}, {s_max}, {v_max}])")

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
