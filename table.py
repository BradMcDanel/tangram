import cv2
import numpy as np
import os

# --- Configuration Constants ---
# HSV Color Range for the blue table (User-tuned)
LOWER_BLUE_HSV = np.array([100, 0, 115])
UPPER_BLUE_HSV = np.array([179, 35, 190])

# Morphological Operations
MORPH_KERNEL_SIZE = (5, 5)
MORPH_OPEN_ITERATIONS = 1
MORPH_CLOSE_ITERATIONS = 2

# Contour Approximation
APPROX_POLY_EPSILON_FACTOR = 0.03 # Percentage of arc length for approxPolyDP

# Table Region Definition
TARGET_UPPER_PERCENTAGE = 0.3  # Percentage of table height from top for the region
TARGET_EXTRUSION_PIXELS = 5    # Pixels to extrude detected table corners outwards

# Drawing Parameters
OUTLINE_COLOR_BGR = (0, 0, 255)  # Red
OUTLINE_THICKNESS = 3
DEBUG_CONTOUR_COLOR_BGR = (0, 255, 255) # Yellow for failed approximation

# --- Helper Functions ---
def normalize_vector(v):
    """Normalizes a 2D vector."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def order_points(pts):
    """
    Orders the 4 points of a quadrilateral in top-left, top-right,
    bottom-right, bottom-left order. Input pts must be (4,2).
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right

    # diff = pts[:, 1] - pts[:, 0] # y - x
    # rect[1] = pts[np.argmin(diff)] # Top-right
    # rect[3] = pts[np.argmax(diff)] # Bottom-left
    # The above diff method can be unstable if the quadrilateral is skewed.
    # A more robust method for the remaining two points after tl and br are found:
    
    # Sort by x-coordinate to distinguish left and right pairs
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]

    # Sort left_most by y-coordinate to get tl, bl
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    (tl_check, bl_check) = left_most # tl_check should match rect[0]

    # The remaining points are tr and br. Use the diff for these.
    # Create a diff array for y-x. The point with smallest y-x is tr.
    # The point with largest y-x is bl.
    # However, we are using the sum method which is generally more robust for perspective.
    # Let's stick to the sum/diff method fully for consistency.
    # diff = pts[:,1] - pts[:,0] # y-x
    # For OpenCV, positive y is down.
    # Top-right: smaller y-x, larger x relative to y
    # Bottom-left: larger y-x, smaller x relative to y

    # Alternative for tr and bl using difference of coordinates (y-x)
    # This might be more stable than the simple argmin/argmax of np.diff(pts, axis=1)
    # which calculates pts[:,1] - pts[:,0] for each point and then finds min/max of that.
    # Let's stick to the widely used sum/diff method:
    diff_coords = np.array([pt[1] - pt[0] for pt in pts]) # y - x for each point
    
    # Exclude tl and br already found from candidates for tr and bl
    remaining_pts_indices = [i for i, pt in enumerate(pts) if not (np.array_equal(pt, rect[0]) or np.array_equal(pt, rect[2]))]
    
    if len(remaining_pts_indices) == 2:
        pt1_idx, pt2_idx = remaining_pts_indices
        if diff_coords[pt1_idx] < diff_coords[pt2_idx]:
            rect[1] = pts[pt1_idx] # Top-right
            rect[3] = pts[pt2_idx] # Bottom-left
        else:
            rect[1] = pts[pt2_idx] # Top-right
            rect[3] = pts[pt1_idx] # Bottom-left
    else: # Fallback if rect[0] or rect[2] matched multiple points or other issue
          # This fallback is less robust and assumes approxPolyDP gave distinct points.
        diff_direct = np.diff(pts, axis=1).reshape(-1) # Calculates pts[i,1]-pts[i,0]
        rect[1] = pts[np.argmin(diff_direct)]
        rect[3] = pts[np.argmax(diff_direct)]
        # Re-check to ensure tl, tr, br, bl are unique and correctly assigned.
        # If rect[0] (tl) ended up as the one with min diff, that's wrong.
        # This part of order_points can be tricky. The sum/diff method is generally good.
        # For rect[1] (top-right) and rect[3] (bottom-left)
        # Top-right has smallest (y-x)
        # Bottom-left has largest (y-x)
        # So we should be using the `diff_coords` computed above directly on all points.
        # Let's re-implement the diff part correctly:
        temp_diff = np.array([pt[1] - pt[0] for pt in pts])
        rect[1] = pts[np.argmin(temp_diff)] # Top-right should have smallest y-x
        rect[3] = pts[np.argmax(temp_diff)] # Bottom-left should have largest y-x

        # Re-assign based on the rule, ensuring uniqueness:
        all_pts = list(map(tuple, pts))
        s_vals = {tuple(p): p.sum() for p in pts}
        d_vals = {tuple(p): p[1]-p[0] for p in pts}

        sorted_by_s = sorted(all_pts, key=lambda p: s_vals[p])
        rect[0] = np.array(sorted_by_s[0]) # tl
        rect[2] = np.array(sorted_by_s[-1]) # br

        sorted_by_d = sorted(all_pts, key=lambda p: d_vals[p])
        # Ensure TR and BL are not TL or BR
        tr_candidate = np.array(sorted_by_d[0])
        bl_candidate = np.array(sorted_by_d[-1])

        if not np.array_equal(tr_candidate, rect[0]) and not np.array_equal(tr_candidate, rect[2]):
            rect[1] = tr_candidate
        else: # if tr_candidate is tl or br, pick the next one
            rect[1] = np.array(sorted_by_d[1])

        if not np.array_equal(bl_candidate, rect[0]) and not np.array_equal(bl_candidate, rect[2]):
            rect[3] = bl_candidate
        else: # if bl_candidate is tl or br, pick the second to last
            rect[3] = np.array(sorted_by_d[-2])

    return rect


def get_table_region_and_warp(image_path):
    """
    Identifies the table, optionally extrudes its corners, draws the outline
    of its upper part, and creates a bird's-eye view of this region.

    Args:
        image_path (str): Path to the input image.

    Returns:
        tuple: (output_image_with_outline, warped_region, upper_region_corners)
               Returns (original_image, None, None) or (None, None, None) on failure.
    """
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not load image from {image_path}")
        return None, None, None

    output_image_with_outline = original_image.copy()
    hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, LOWER_BLUE_HSV, UPPER_BLUE_HSV)
    kernel = np.ones(MORPH_KERNEL_SIZE, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=MORPH_OPEN_ITERATIONS)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_CLOSE_ITERATIONS)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No contours found for the table color.")
        return original_image, None, None

    table_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(table_contour)
    perimeter_hull = cv2.arcLength(hull, True)
    epsilon = APPROX_POLY_EPSILON_FACTOR * perimeter_hull
    approx_poly_table = cv2.approxPolyDP(hull, epsilon, True)

    if len(approx_poly_table) == 4:
        table_corners = approx_poly_table.reshape(4, 2).astype(np.float32)
        ordered_table_corners = order_points(table_corners)

        # --- Extrude Table Corners ---
        if TARGET_EXTRUSION_PIXELS != 0:
            extruded_corners_list = []
            num_corners = len(ordered_table_corners)
            for i in range(num_corners):
                P = ordered_table_corners[i]
                prev_P = ordered_table_corners[(i - 1 + num_corners) % num_corners]
                next_P = ordered_table_corners[(i + 1) % num_corners]
                vec1 = P - prev_P
                vec2 = P - next_P
                norm_vec1 = normalize_vector(vec1)
                norm_vec2 = normalize_vector(vec2)
                direction_outward = normalize_vector(norm_vec1 + norm_vec2)
                if np.all(direction_outward == 0):
                    extruded_P = P
                else:
                    extruded_P = P + TARGET_EXTRUSION_PIXELS * direction_outward
                extruded_corners_list.append(extruded_P)
            ordered_table_corners = np.array(extruded_corners_list, dtype=np.float32)

        (tl_table, tr_table, br_table, bl_table) = ordered_table_corners

        # --- Calculate corners for the upper region (the red outline) ---
        p_left_new_x = tl_table[0] + TARGET_UPPER_PERCENTAGE * (bl_table[0] - tl_table[0])
        p_left_new_y = tl_table[1] + TARGET_UPPER_PERCENTAGE * (bl_table[1] - tl_table[1])
        bl_region = np.array([int(p_left_new_x), int(p_left_new_y)], dtype=np.float32) # Bottom-left of region

        p_right_new_x = tr_table[0] + TARGET_UPPER_PERCENTAGE * (br_table[0] - tr_table[0])
        p_right_new_y = tr_table[1] + TARGET_UPPER_PERCENTAGE * (br_table[1] - tr_table[1])
        br_region = np.array([int(p_right_new_x), int(p_right_new_y)], dtype=np.float32) # Bottom-right of region
        
        tl_region = tl_table.astype(np.float32)
        tr_region = tr_table.astype(np.float32)

        upper_region_corners = np.array([tl_region, tr_region, br_region, bl_region], dtype=np.int32)
        
        # Draw the outline on the output image
        cv2.polylines(output_image_with_outline, [upper_region_corners], isClosed=True,
                      color=OUTLINE_COLOR_BGR, thickness=OUTLINE_THICKNESS)

        # --- Perform Perspective Warp for Bird's-Eye View of the Region ---
        # Source points are the corners of the red region, ensure float32
        src_pts_warp = upper_region_corners.astype(np.float32)
        
        # tl_region, tr_region, br_region, bl_region for warp
        P1_tl = src_pts_warp[0] # Top-left of region
        P2_tr = src_pts_warp[1] # Top-right of region
        P3_br = src_pts_warp[2] # Bottom-right of region (p_right_new)
        P4_bl = src_pts_warp[3] # Bottom-left of region (p_left_new)

        # Calculate width and height for the destination warped image
        width_top = np.linalg.norm(P2_tr - P1_tl)
        width_bottom = np.linalg.norm(P3_br - P4_bl)
        maxWidth = int(max(width_top, width_bottom))

        height_left = np.linalg.norm(P4_bl - P1_tl)
        height_right = np.linalg.norm(P3_br - P2_tr)
        maxHeight = int(max(height_left, height_right))

        if maxWidth <=0 or maxHeight <=0:
            print("Warning: Calculated maxWidth or maxHeight for warp is zero or negative. Skipping warp.")
            return output_image_with_outline, None, upper_region_corners.astype(np.int32)


        dst_pts_warp = np.array([
            [0, 0],                         # P1_tl maps here
            [maxWidth - 1, 0],              # P2_tr maps here
            [maxWidth - 1, maxHeight - 1],  # P3_br maps here
            [0, maxHeight - 1]              # P4_bl maps here
        ], dtype="float32")
        print(src_pts_warp, dst_pts_warp)

        # Get the perspective transformation matrix
        M = cv2.getPerspectiveTransform(src_pts_warp, dst_pts_warp)
        # Warp the original image (not the one with outlines)
        warped_region = cv2.warpPerspective(original_image, M, (maxWidth, maxHeight))

        return output_image_with_outline, warped_region, upper_region_corners.astype(np.int32)
    else:
        print(f"Table contour (after hull) approximated to {len(approx_poly_table)} points, not 4.")
        cv2.drawContours(output_image_with_outline, [approx_poly_table], -1, DEBUG_CONTOUR_COLOR_BGR, 2)
        return output_image_with_outline, None, None


# --- Main Execution ---
if __name__ == "__main__":
    image_file = 'data/images/calibration.png' # Make sure this is your original image

    if not os.path.exists(image_file):
        print(f"Error: Image file not found at '{image_file}'")
        print("Please ensure the path is correct and the image exists.")
        exit()

    img_with_outline, warped_img, region_corners = get_table_region_and_warp(image_file)

    if img_with_outline is not None:
        cv2.imshow("Original with Upper Table Region Outline", img_with_outline)

    if warped_img is not None:
        cv2.imshow(f"Warped Top Region (Bird's-Eye View)", warped_img)
    
    if img_with_outline is not None or warped_img is not None:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Optionally save the results
    # if img_with_outline is not None:
    #     cv2.imwrite("output_table_region_outlined.jpg", img_with_outline)
    # if warped_img is not None:
    #     cv2.imwrite("output_warped_table_region.jpg", warped_img)
