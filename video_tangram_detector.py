import cv2
import numpy as np
import json
import argparse
import time
import datetime

# --- Configuration Constants ---

# --- Table Detection & Warping Parameters ---
TABLE_LOWER_HSV = np.array([100, 0, 115])
TABLE_UPPER_HSV = np.array([179, 35, 190])
TABLE_MORPH_KERNEL_SIZE = (5, 5)
TABLE_MORPH_OPEN_ITERATIONS = 1
TABLE_MORPH_CLOSE_ITERATIONS = 2
TABLE_APPROX_POLY_EPSILON_FACTOR = 0.03
TABLE_REGION_UPPER_PERCENTAGE = 0.45
TABLE_CORNER_EXTRUSION_PIXELS = 5
TABLE_OUTLINE_COLOR_BGR = (255, 192, 0) # Light Blue
TABLE_OUTLINE_THICKNESS = 1

# --- Tangram Piece Detection Parameters ---
PIECE_BLUR_KERNEL_SIZE = (5, 5)
PIECE_MORPH_KERNEL_SIZE = (3, 3)
PIECE_MORPH_OPEN_ITERATIONS = 1
PIECE_MORPH_CLOSE_ITERATIONS = 1
PIECE_CORNER_EXTRUSION_PIXELS = 3
PIECE_EPSILON_SEARCH_START = 0.01
PIECE_EPSILON_SEARCH_END = 0.1
PIECE_EPSILON_SEARCH_STEP = 0.001
FALLBACK_MAX_AREA_PX = 150

# --- Shape Regularity Parameters ---
MIN_ALLOWED_ANGLE_DEG = 30.0
MAX_ALLOWED_ANGLE_DEG = 130.0
SQUARE_TARGET_ANGLE_DEG = 90.0
SQUARE_ANGLE_TOLERANCE_DEG = 15.0
PARALLELOGRAM_OPPOSITE_ANGLE_DIFF_TOLERANCE_DEG = 30.0
PARALLELOGRAM_ADJACENT_ANGLE_SUM_TARGET_DEG = 180.0
PARALLELOGRAM_ADJACENT_ANGLE_SUM_TOLERANCE_DEG = 30.0

# --- Relaxation Parameters for Piece Detection ---
MAX_RELAXATION_STEPS = 5
RELAX_PERCENT_INCREMENT_CH0 = 0.001 # For Hue
RELAX_PERCENT_INCREMENT_CH12 = 0.002 # For Saturation & Value
AREA_RELAX_FACTOR_PER_STEP = 0.01

# --- Scoring Parameters for Piece Detection (NEW) ---
WEIGHT_COLOR_CLOSENESS = 0.6
WEIGHT_AREA_CLOSENESS = 0.4

# --- Temporal Smoothing and Persistence Parameters ---
CONFIRM_STABLE_DURATION_S = 0.5
CONFIRM_MIN_PERCENT_VISIBLE = 30.0
POSITION_STABILITY_PX = 3
AREA_STABILITY_RATIO = 0.15

# --- Drawing Parameters ---
PIECE_LOCKED_COLOR_BGR = (0, 255, 0) # Green
PIECE_LOCKED_THICKNESS = 1
PIECE_VERTEX_COLOR_BGR = (0, 0, 255) # Red
PIECE_VERTEX_RADIUS = 1


# --- Helper Functions ---
def regularize_shape(polygon, shape_class_name):
    """
    Takes a detected polygon and returns a geometrically "perfect" version of it
    (e.g., a true square, a true isosceles right triangle) with the same
    centroid, area, and orientation.
    """
    if polygon is None or len(polygon) < 3:
        return polygon

    # Ensure polygon is in (N, 2) format
    poly_2d = polygon.reshape(-1, 2).astype(np.float32)
    
    if shape_class_name == "Square" and len(poly_2d) == 4:
        # For a square, the most robust method is to use its minimum area bounding rectangle.
        # This gives us a stable center, size, and rotation angle.
        # We then create a perfect square with the same area and orientation.
        rect = cv2.minAreaRect(poly_2d)
        (center, (width, height), angle) = rect
        
        # Preserve the original area
        area = cv2.contourArea(poly_2d)
        if area > 0:
            side_length = np.sqrt(area)
            # Create a new rectangle definition for a perfect square
            square_rect = (center, (side_length, side_length), angle)
            # Get the 4 corners of this new perfect square
            box_pts = cv2.boxPoints(square_rect)
            # Return it in the same format as the input
            return box_pts.reshape(-1, 1, 2).astype(np.float32)
        
    elif shape_class_name == "Triangle" and len(poly_2d) == 3:
        # For tangram triangles (isosceles right triangles).
        angles = calculate_internal_angles(poly_2d)
        if not angles: return polygon # Should not happen

        # Find the apex (the corner with the ~90-degree angle)
        apex_index = np.argmin(np.abs(np.array(angles) - 90.0))
        p_apex = poly_2d[apex_index]
        
        # The other two points form the hypotenuse
        p_base1 = poly_2d[(apex_index + 1) % 3]
        p_base2 = poly_2d[(apex_index + 2) % 3]

        # Preserve original area to calculate ideal leg length
        area = cv2.contourArea(poly_2d)
        if area > 0:
            # Area of right triangle = 0.5 * leg1 * leg2. For isosceles, leg1=leg2.
            # So, Area = 0.5 * leg^2  => leg = sqrt(2 * Area)
            ideal_leg_length = np.sqrt(2 * area)

            # Vectors from the detected apex to the other two vertices
            vec1 = p_base1 - p_apex
            vec2 = p_base2 - p_apex
            
            # Average the length of the two legs from the detection for stability
            avg_detected_leg_length = (np.linalg.norm(vec1) + np.linalg.norm(vec2)) / 2.0
            if avg_detected_leg_length == 0: return polygon # Avoid division by zero

            # Create new, perfectly perpendicular vectors with the ideal length
            # We align them with the original detected vectors to preserve orientation
            v1_new = (vec1 / np.linalg.norm(vec1)) * ideal_leg_length
            v2_new = (vec2 / np.linalg.norm(vec2)) * ideal_leg_length

            # To make them perfectly 90 degrees, we can average their direction and then
            # rotate +/- 45 degrees. This is more robust to skewed inputs.
            bisector_vec = normalize_vector(normalize_vector(vec1) + normalize_vector(vec2))
            
            # Rotation matrix for +45 and -45 degrees
            cos45, sin45 = np.cos(np.pi/4), np.sin(np.pi/4)
            rot_plus_45 = np.array([[cos45, -sin45], [sin45, cos45]])
            rot_minus_45 = np.array([[cos45, sin45], [-sin45, cos45]])
            
            leg_vec1 = rot_minus_45 @ bisector_vec * ideal_leg_length
            leg_vec2 = rot_plus_45 @ bisector_vec * ideal_leg_length

            # Re-create the triangle from the original apex
            new_p_base1 = p_apex + leg_vec1
            new_p_base2 = p_apex + leg_vec2

            regularized_triangle = np.array([p_apex, new_p_base1, new_p_base2], dtype=np.float32)
            
            # Final check: The new centroid should be close to the old one.
            # If it drifted too much, fall back to the original polygon.
            M_old = cv2.moments(poly_2d)
            M_new = cv2.moments(regularized_triangle)
            if M_old['m00'] > 0 and M_new['m00'] > 0:
                cx_old, cy_old = M_old['m10']/M_old['m00'], M_old['m01']/M_old['m00']
                cx_new, cy_new = M_new['m10']/M_new['m00'], M_new['m01']/M_new['m00']
                if np.hypot(cx_old-cx_new, cy_old-cy_new) < 10.0: # Allow 10px drift
                    return regularized_triangle.reshape(-1, 1, 2)

    # For Parallelogram or other shapes, regularization is more complex.
    # For now, we return the original polygon for them.
    # A potential improvement for parallelogram would be to average opposite side lengths
    # and angles.
    return polygon

def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0: return v
    return v / norm

def order_points_for_quad(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = pts[:, 1] - pts[:, 0] # y - x
    temp_pts_with_diff = []
    for i in range(len(pts)):
        if not np.array_equal(pts[i], rect[0]) and not np.array_equal(pts[i], rect[2]):
            temp_pts_with_diff.append((pts[i], diff[i]))
    if len(temp_pts_with_diff) == 2:
        temp_pts_with_diff.sort(key=lambda x: x[1]) # Sort by y-x; smallest is TR, largest is BL
        rect[1] = temp_pts_with_diff[0][0] # Top-right
        rect[3] = temp_pts_with_diff[1][0] # Bottom-left
    else:
        print(f"WARNING: order_points_for_quad found {len(temp_pts_with_diff)} remaining points, expected 2. Input pts: {pts.tolist()}")
        return None
    return rect

def extrude_polygon_corners(polygon, extrusion_pixels):
    if polygon is None or extrusion_pixels == 0: return polygon
    if polygon.ndim == 3 and polygon.shape[1] == 1: poly_2d = polygon.reshape(-1, 2).astype(np.float32)
    elif polygon.ndim == 2: poly_2d = polygon.astype(np.float32)
    else: return polygon
    num_corners = len(poly_2d)
    if num_corners < 3: return polygon
    extruded_corners_list = []
    for i in range(num_corners):
        P = poly_2d[i]; prev_P = poly_2d[(i - 1 + num_corners) % num_corners]; next_P = poly_2d[(i + 1) % num_corners]
        vec1 = P - prev_P; vec2 = P - next_P
        norm_vec1 = normalize_vector(vec1); norm_vec2 = normalize_vector(vec2)
        direction_outward = normalize_vector(norm_vec1 + norm_vec2)
        extruded_P = P + extrusion_pixels * direction_outward if not np.all(direction_outward == 0) else P
        extruded_corners_list.append(extruded_P)
    if polygon.ndim == 3 and polygon.shape[1] == 1: return np.array(extruded_corners_list, dtype=np.float32).reshape(-1,1,2)
    return np.array(extruded_corners_list, dtype=np.float32)

def calculate_internal_angles(polygon):
    if polygon.ndim == 3 and polygon.shape[1] == 1: pts = polygon.reshape(-1, 2)
    elif polygon.ndim == 2: pts = polygon
    else: return []
    num_vertices = len(pts)
    if num_vertices < 3: return []
    angles = []
    for i in range(num_vertices):
        p_prev = pts[(i - 1 + num_vertices) % num_vertices]; p_curr = pts[i]; p_next = pts[(i + 1) % num_vertices]
        v1 = p_prev - p_curr; v2 = p_next - p_curr
        dot_product = np.dot(v1, v2); mag_v1 = np.linalg.norm(v1); mag_v2 = np.linalg.norm(v2)
        if mag_v1 * mag_v2 == 0: angles.append(0); continue
        cos_angle = np.clip(dot_product / (mag_v1 * mag_v2), -1.0, 1.0)
        angles.append(np.degrees(np.arccos(cos_angle)))
    return angles

def is_shape_regular(polygon, shape_class_name, num_vertices_expected):
    if polygon is None or len(polygon) != num_vertices_expected: return False
    angles = calculate_internal_angles(polygon)
    if not angles or len(angles) != num_vertices_expected: return False
    for angle in angles:
        if not (MIN_ALLOWED_ANGLE_DEG <= angle <= MAX_ALLOWED_ANGLE_DEG): return False
    if shape_class_name == "Triangle": return True
    elif shape_class_name == "Square":
        for angle in angles:
            if not (SQUARE_TARGET_ANGLE_DEG - SQUARE_ANGLE_TOLERANCE_DEG <= angle <= \
                    SQUARE_TARGET_ANGLE_DEG + SQUARE_ANGLE_TOLERANCE_DEG): return False
        return True
    elif shape_class_name == "Parallelogram":
        if abs(angles[0] - angles[2]) > PARALLELOGRAM_OPPOSITE_ANGLE_DIFF_TOLERANCE_DEG: return False
        if abs(angles[1] - angles[3]) > PARALLELOGRAM_OPPOSITE_ANGLE_DIFF_TOLERANCE_DEG: return False
        for i in range(4):
            if abs((angles[i] + angles[(i+1)%4]) - PARALLELOGRAM_ADJACENT_ANGLE_SUM_TARGET_DEG) > PARALLELOGRAM_ADJACENT_ANGLE_SUM_TOLERANCE_DEG: return False
        return True
    return True

def initialize_table_warp_parameters(first_frame):
    hsv = cv2.cvtColor(first_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, TABLE_LOWER_HSV, TABLE_UPPER_HSV)
    kernel = np.ones(TABLE_MORPH_KERNEL_SIZE, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=TABLE_MORPH_OPEN_ITERATIONS)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=TABLE_MORPH_CLOSE_ITERATIONS)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: print("ERROR: No table contours found."); return None, None, 0, 0, None
    table_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(table_contour)
    perimeter_hull = cv2.arcLength(hull, True)
    epsilon = TABLE_APPROX_POLY_EPSILON_FACTOR * perimeter_hull
    approx_poly_table = cv2.approxPolyDP(hull, epsilon, True)
    if len(approx_poly_table) != 4: print(f"ERROR: Table approx to {len(approx_poly_table)} points."); return None, None, 0, 0, None
    table_corners = approx_poly_table.reshape(4, 2).astype(np.float32)
    ordered_table_corners = order_points_for_quad(table_corners)
    if ordered_table_corners is None: print("ERROR: Could not order table corners."); return None, None, 0, 0, None
    ordered_table_corners = extrude_polygon_corners(ordered_table_corners, TABLE_CORNER_EXTRUSION_PIXELS)
    (tl_table, tr_table, br_table, bl_table) = ordered_table_corners
    p_left_new_y = tl_table[1] + TABLE_REGION_UPPER_PERCENTAGE * (bl_table[1] - tl_table[1]); p_left_new_x = tl_table[0] + TABLE_REGION_UPPER_PERCENTAGE * (bl_table[0] - tl_table[0])
    bl_region = np.array([p_left_new_x, p_left_new_y], dtype=np.float32)
    p_right_new_y = tr_table[1] + TABLE_REGION_UPPER_PERCENTAGE * (br_table[1] - tr_table[1]); p_right_new_x = tr_table[0] + TABLE_REGION_UPPER_PERCENTAGE * (br_table[0] - tr_table[0])
    br_region = np.array([p_right_new_x, p_right_new_y], dtype=np.float32)
    upper_region_corners_orig_float = np.array([tl_table.astype(np.float32), tr_table.astype(np.float32), br_region, bl_region], dtype=np.float32)
    upper_region_corners_orig_int = upper_region_corners_orig_float.astype(np.int32)
    width_top = np.linalg.norm(tr_table - tl_table); width_bottom = np.linalg.norm(br_region - bl_region)
    warp_width = int(max(width_top, width_bottom))
    height_left = np.linalg.norm(bl_region - tl_table); height_right = np.linalg.norm(br_region - tr_table)
    warp_height = int(max(height_left, height_right))
    if warp_width <= 0 or warp_height <= 0: print(f"ERROR: Invalid warp dimensions ({warp_width}x{warp_height})."); return None, None, 0, 0, None
    dst_pts_warp = np.array([[0, 0], [warp_width - 1, 0], [warp_width - 1, warp_height - 1], [0, warp_height - 1]], dtype="float32")
    perspective_M = cv2.getPerspectiveTransform(upper_region_corners_orig_float, dst_pts_warp)
    ret_invert, inverse_perspective_M = cv2.invert(perspective_M)
    if not ret_invert: print("ERROR: Could not invert perspective matrix."); return None, None, 0, 0, None
    return perspective_M, inverse_perspective_M, warp_width, warp_height, upper_region_corners_orig_int

def get_poly_metrics(poly):
    if poly is None or len(poly) == 0: return None, 0
    if poly.ndim == 2: poly_for_moments = poly.reshape(-1,1,2).astype(np.int32)
    else: poly_for_moments = poly.astype(np.int32)
    M = cv2.moments(poly_for_moments)
    area = M['m00']
    current_poly_pts = poly.reshape(-1,2)
    if area == 0:
        if len(current_poly_pts) == 0: return None, 0
        cx = np.mean(current_poly_pts[:,0]); cy = np.mean(current_poly_pts[:,1])
    else:
        cx = M['m10'] / area; cy = M['m01'] / area
    return (int(cx), int(cy)), area

def compare_polys_for_stability(poly1, poly2):
    if poly1 is None or poly2 is None: return False
    c1, a1 = get_poly_metrics(poly1); c2, a2 = get_poly_metrics(poly2)
    if c1 is None or c2 is None: return False
    centroid_dist = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
    if a1 == 0 and a2 == 0: area_diff_ok = True
    elif a1 == 0 or a2 == 0 or max(a1,a2) == 0: area_diff_ok = False
    else: area_diff_ok = (abs(a1 - a2) / max(a1, a2)) < AREA_STABILITY_RATIO
    return centroid_dist < POSITION_STABILITY_PX and area_diff_ok

def get_hue_distance(h1, h2, max_hue_val=179):
    """Calculates the shortest distance between two hue values on a circle (0-max_hue_val)."""
    diff = abs(h1 - h2)
    return min(diff, (max_hue_val + 1) - diff)

def process_tangrams_in_warped_frame(warped_frame, config_data, debug_pid_str=None, show_cv_windows=True, frame_num_debug=0):
    if warped_frame is None or warped_frame.size == 0:
        print("ERROR: Received empty warped_frame.")
        return {}

    blur_warped_frame = cv2.GaussianBlur(warped_frame, PIECE_BLUR_KERNEL_SIZE, 0)
    conv_img_masking = cv2.cvtColor(blur_warped_frame, cv2.COLOR_BGR2HSV)
    ch0_max, ch1_max, ch2_max = 179, 255, 255

    morph_k_piece = np.ones(PIECE_MORPH_KERNEL_SIZE, np.uint8)
    raw_detected_piece_data = {}

    for pid, p_attrs in config_data.items():
        p_color_space_key = p_attrs.get('color_space', 'hsv')
        if p_color_space_key != 'hsv':
            print(f"WARNING: Piece {pid} has unsupported color_space '{p_color_space_key}'.")

        orig_low_thresh = np.array(p_attrs[f'{p_color_space_key}_lower'])
        orig_up_thresh = np.array(p_attrs[f'{p_color_space_key}_upper'])
        tgt_verts = p_attrs['num_vertices']
        min_area, max_area = p_attrs.get('min_area', 0), p_attrs.get('max_area', float('inf'))
        shape_class = p_attrs.get("class_name", "Unknown")

        best_poly_in_warped_coords = None
        best_combined_score = float('inf')

        # --- Nested Helper Function for Clarity ---
        def _process_contour(cnt, relax_step):
            """Processes a single contour, returning a candidate polygon and its score, or None."""
            area = cv2.contourArea(cnt)
            min_area_mult = 1 - AREA_RELAX_FACTOR_PER_STEP * relax_step
            max_area_mult = 1 + AREA_RELAX_FACTOR_PER_STEP * relax_step
            relaxed_min_a = max(0.0, min_area * min_area_mult)
            relaxed_max_a = max_area * max_area_mult

            if not (relaxed_min_a <= area <= relaxed_max_a):
                return None, None

            # --- Primary Detection Method: approxPolyDP ---
            poly_found = False
            candidate_poly = None
            peri = cv2.arcLength(cnt, True)
            if peri > 0:
                for eps_factor in np.arange(PIECE_EPSILON_SEARCH_START, PIECE_EPSILON_SEARCH_END, PIECE_EPSILON_SEARCH_STEP):
                    poly = cv2.approxPolyDP(cnt, eps_factor * peri, True)
                    if len(poly) == tgt_verts:
                        candidate_poly = poly
                        poly_found = True
                        break
                    elif len(poly) < tgt_verts:
                        break

            # --- Fallback Detection Method for Small/Noisy Contours ---
            if not poly_found and area < FALLBACK_MAX_AREA_PX:
                if tgt_verts == 3: # Specific fallback for triangles
                    ret, triangle = cv2.minEnclosingTriangle(cnt)
                    if ret and triangle is not None:
                        candidate_poly = triangle.astype(np.int32)
                        poly_found = True
                elif tgt_verts == 4: # Generic fallback for quadrilaterals
                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect)
                    candidate_poly = box.astype(np.int32).reshape(4, 1, 2)
                    poly_found = True

            # --- If a polygon was found by any method, score it ---
            if poly_found and candidate_poly is not None:
                if is_shape_regular(candidate_poly, shape_class, tgt_verts):
                    # Scoring logic (same as before)
                    tgt_area_mid = (min_area + max_area) / 2.0 if max_area != float('inf') and (min_area + max_area) > 0 else min_area * 1.5 if min_area > 0 else 1.0
                    original_color_center = (orig_low_thresh + orig_up_thresh) / 2.0
                    original_color_range_half = (orig_up_thresh - orig_low_thresh) / 2.0
                    original_color_range_half[original_color_range_half == 0] = 1.0
                    
                    area_diff_abs = abs(area - tgt_area_mid)
                    normalized_area_distance = area_diff_abs / tgt_area_mid if tgt_area_mid > 0 else (0.0 if area_diff_abs == 0 else 1.0)
                    
                    contour_mask_for_avg_color = np.zeros(conv_img_masking.shape[:2], dtype=np.uint8)
                    cv2.drawContours(contour_mask_for_avg_color, [cnt], -1, 255, -1)
                    avg_color_tuple = cv2.mean(conv_img_masking, mask=contour_mask_for_avg_color)
                    avg_color_piece = np.array(avg_color_tuple[:3])
                    
                    color_distances_per_channel = np.zeros(3)
                    color_distances_per_channel[0] = get_hue_distance(avg_color_piece[0], original_color_center[0], ch0_max)
                    color_distances_per_channel[1] = abs(avg_color_piece[1] - original_color_center[1])
                    color_distances_per_channel[2] = abs(avg_color_piece[2] - original_color_center[2])
                    
                    normalized_color_distances = np.nan_to_num(color_distances_per_channel / original_color_range_half, nan=1.0)
                    normalized_color_distance_avg = np.mean(np.clip(normalized_color_distances, 0.0, 1.0))

                    score = (WEIGHT_COLOR_CLOSENESS * normalized_color_distance_avg +
                             WEIGHT_AREA_CLOSENESS * normalized_area_distance)
                    
                    return candidate_poly, score
            
            return None, None
        # --- End of Nested Helper Function ---

        for relax_step in range(MAX_RELAXATION_STEPS + 1):
            curr_low_thresh, curr_up_thresh = orig_low_thresh.copy(), orig_up_thresh.copy()
            if relax_step > 0:
                ch0_adj = int(ch0_max * RELAX_PERCENT_INCREMENT_CH0 * relax_step)
                ch1_adj = int(ch1_max * RELAX_PERCENT_INCREMENT_CH12 * relax_step)
                ch2_adj = int(ch2_max * RELAX_PERCENT_INCREMENT_CH12 * relax_step)
                curr_low_thresh -= np.array([ch0_adj, ch1_adj, ch2_adj])
                curr_up_thresh += np.array([ch0_adj, ch1_adj, ch2_adj])
                np.clip(curr_low_thresh, 0, None, out=curr_low_thresh)
                curr_up_thresh[0] = min(ch0_max, curr_up_thresh[0])
                curr_up_thresh[1] = min(ch1_max, curr_up_thresh[1])
                curr_up_thresh[2] = min(ch2_max, curr_up_thresh[2])

            color_mask = cv2.inRange(conv_img_masking, curr_low_thresh, curr_up_thresh)
            opened_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, morph_k_piece, iterations=PIECE_MORPH_OPEN_ITERATIONS)
            proc_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, morph_k_piece, iterations=PIECE_MORPH_CLOSE_ITERATIONS)

            if show_cv_windows and debug_pid_str == pid and warped_frame.size > 0:
                debug_mask_disp = warped_frame.copy()
                debug_mask_disp[proc_mask == 0] = 0
                cv2.imshow(f"F{frame_num_debug} P_WARPED P{pid} R{relax_step}", debug_mask_disp)

            contours, _ = cv2.findContours(proc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            found_in_this_step = False
            for cnt in contours:
                poly, score = _process_contour(cnt, relax_step)
                if poly is not None and score is not None:
                    if score < best_combined_score:
                        best_combined_score = score
                        regularized_poly = regularize_shape(poly, shape_class)
                        best_poly_in_warped_coords = extrude_polygon_corners(regularized_poly, PIECE_CORNER_EXTRUSION_PIXELS)
                        found_in_this_step = True
            
            # If we found a good candidate in this relaxation step, we don't need to relax further
            if found_in_this_step:
                break

        if best_poly_in_warped_coords is not None:
            raw_detected_piece_data[pid] = {
                'poly': best_poly_in_warped_coords,
                'score': best_combined_score
            }
            
    return raw_detected_piece_data

def get_default_piece_state():
    return {'status': 'UNSEEN', 'candidate_poly': None, 'candidate_recent_detections': [], 'locked_poly': None, 'frames_unseen_at_locked_pos': 0}

def main():
    parser = argparse.ArgumentParser(description="Tangram detection with table warping and temporal persistence.")
    parser.add_argument("--video", required=True); parser.add_argument("--config", required=True)
    parser.add_argument("--output_json", required=True); parser.add_argument("--debug_piece_id", type=str, default=None)
    parser.add_argument("--no_display", action="store_true"); parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    args = parser.parse_args()
    start_processing_time = time.time()
    with open(args.config, 'r') as f: tangram_config_data = json.load(f)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened(): print(f"Error: Cannot open video {args.video}"); return
    fps = cap.get(cv2.CAP_PROP_FPS); fps = fps if fps > 0 else 30.0
    frame_width_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); frame_height_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ret_first, first_frame = cap.read()
    if not ret_first: print("Error: Could not read the first frame."); cap.release(); return
    perspective_M, inverse_perspective_M, warp_width, warp_height, table_outline_pts_orig = initialize_table_warp_parameters(first_frame)
    if perspective_M is None or inverse_perspective_M is None: print("Exiting: table warp init failed."); cap.release(); return
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Rewind to start
    confirmation_window_frames = int(CONFIRM_STABLE_DURATION_S * fps)
    confirmation_min_positive_detections = 0
    if confirmation_window_frames > 0:
        confirmation_min_positive_detections = int(confirmation_window_frames * (CONFIRM_MIN_PERCENT_VISIBLE / 100.0))
        if confirmation_min_positive_detections == 0 and CONFIRM_MIN_PERCENT_VISIBLE > 0: confirmation_min_positive_detections = 1
    elif CONFIRM_STABLE_DURATION_S > 0 and CONFIRM_MIN_PERCENT_VISIBLE > 0: confirmation_min_positive_detections = 1 # If duration_s leads to 0 frames, still need 1 detection
    
    frames_data_for_json = []; piece_states = {pid: get_default_piece_state() for pid in tangram_config_data}
    frame_idx = 0; total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while cap.isOpened():
        loop_start_time = time.time(); ret, current_frame_orig = cap.read()
        if not ret: break
        current_frame_timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if args.verbose or (frame_idx > 0 and frame_idx % 30 == 0):
            print(f"Processing Frame {frame_idx}" + (f" / {total_frames_video -1}" if total_frames_video > 0 else ""))
        
        current_warped_frame = cv2.warpPerspective(current_frame_orig, perspective_M, (warp_width, warp_height))
        raw_detections_in_warped = process_tangrams_in_warped_frame(current_warped_frame, tangram_config_data, args.debug_piece_id, not args.no_display, frame_idx)
        
        display_original_output = current_frame_orig.copy()
        if table_outline_pts_orig is not None: cv2.polylines(display_original_output, [table_outline_pts_orig], True, TABLE_OUTLINE_COLOR_BGR, TABLE_OUTLINE_THICKNESS)
        
        current_frame_output_pieces_list = []
        for piece_id, piece_cfg in tangram_config_data.items():
            state = piece_states[piece_id]; raw_poly_data = raw_detections_in_warped.get(piece_id)
            raw_poly_warped = raw_poly_data['poly'] if raw_poly_data else None
            output_poly_for_this_piece_original_coords = None

            if state['status'] == 'UNSEEN':
                if raw_poly_warped is not None:
                    state.update({'status': 'CONFIRMING', 'candidate_poly': raw_poly_warped, 'candidate_recent_detections': [True]})
                    if confirmation_window_frames == 0 and 1 >= confirmation_min_positive_detections: # Immediate lock if no confirmation window needed
                        state.update({'status': 'LOCKED', 'locked_poly': state['candidate_poly'], 'frames_unseen_at_locked_pos': 0, 'candidate_poly': None, 'candidate_recent_detections': []})
            elif state['status'] == 'CONFIRMING':
                is_stable_candidate = False
                if raw_poly_warped is not None:
                    if state['candidate_poly'] is None or not compare_polys_for_stability(raw_poly_warped, state['candidate_poly']):
                        state.update({'candidate_poly': raw_poly_warped, 'candidate_recent_detections': [True]}) # New candidate
                    else:
                        state['candidate_recent_detections'].append(True) # Stable candidate seen again
                        is_stable_candidate = True
                elif state['candidate_poly'] is not None: # Candidate not seen this frame
                    state['candidate_recent_detections'].append(False)
                
                # Trim buffer
                while len(state['candidate_recent_detections']) > confirmation_window_frames and confirmation_window_frames > 0:
                    state['candidate_recent_detections'].pop(0)

                # Check for confirmation
                candidate_confirmed = False
                if state['candidate_poly'] is not None: # Only confirm if there's an active candidate
                    if confirmation_window_frames == 0: # No temporal window, 1 good detection is enough if min_pos_detect is 1
                         candidate_confirmed = (1 >= confirmation_min_positive_detections)
                    elif len(state['candidate_recent_detections']) == confirmation_window_frames: # Window full
                         candidate_confirmed = (sum(state['candidate_recent_detections']) >= confirmation_min_positive_detections)
                
                if candidate_confirmed:
                    state.update({'status': 'LOCKED', 'locked_poly': state['candidate_poly'], 'frames_unseen_at_locked_pos': 0, 'candidate_poly': None, 'candidate_recent_detections': []})
                elif state['candidate_poly'] is not None: # Has candidate, but not confirmed
                    # Reset if window full and not confirmed, or if window is 0 and not immediately confirmed
                    window_full_not_confirmed = (confirmation_window_frames > 0 and len(state['candidate_recent_detections']) == confirmation_window_frames and not candidate_confirmed)
                    no_window_not_confirmed = (confirmation_window_frames == 0 and not candidate_confirmed and confirmation_min_positive_detections > 0) # e.g. if min_pos_detect was > 1 (unusual for window 0)

                    if window_full_not_confirmed or no_window_not_confirmed:
                         state.update(get_default_piece_state()); state['status'] = 'UNSEEN' # Reset

            elif state['status'] == 'LOCKED':
                locked_poly_warped = state['locked_poly']
                seen_at_locked_pos_this_frame = False
                
                if raw_poly_warped is not None:
                    if compare_polys_for_stability(raw_poly_warped, locked_poly_warped):
                        seen_at_locked_pos_this_frame = True
                        state.update({'frames_unseen_at_locked_pos': 0, 'candidate_poly': None, 'candidate_recent_detections': []}) # Reset any relocation attempt
                    else: # Detected a piece, but not at the locked position. Start confirming new position.
                        if state['candidate_poly'] is None or not compare_polys_for_stability(raw_poly_warped, state['candidate_poly']):
                            state.update({'candidate_poly': raw_poly_warped, 'candidate_recent_detections': [True]})
                        else:
                            state['candidate_recent_detections'].append(True)
                        
                        while len(state['candidate_recent_detections']) > confirmation_window_frames and confirmation_window_frames > 0:
                            state['candidate_recent_detections'].pop(0)

                        reloc_candidate_confirmed = False
                        if state['candidate_poly'] is not None:
                            if confirmation_window_frames == 0:
                                reloc_candidate_confirmed = (1 >= confirmation_min_positive_detections)
                            elif len(state['candidate_recent_detections']) == confirmation_window_frames:
                                reloc_candidate_confirmed = (sum(state['candidate_recent_detections']) >= confirmation_min_positive_detections)

                        if reloc_candidate_confirmed:
                            state.update({'locked_poly': state['candidate_poly'], 'frames_unseen_at_locked_pos': 0, 'candidate_poly': None, 'candidate_recent_detections': []})
                            locked_poly_warped = state['locked_poly'] # Update for drawing this frame
                            seen_at_locked_pos_this_frame = True
                        # If relocation candidate not confirmed yet, do nothing with locked_poly, it remains as is.
                
                elif state['candidate_poly'] is not None: # No raw detection, but had a relocation candidate
                    state['candidate_recent_detections'].append(False) # Relocation candidate not seen
                    while len(state['candidate_recent_detections']) > confirmation_window_frames and confirmation_window_frames > 0:
                        state['candidate_recent_detections'].pop(0)
                    
                    # If relocation window is full and not confirmed, abandon relocation attempt
                    if (confirmation_window_frames > 0 and len(state['candidate_recent_detections']) == confirmation_window_frames and sum(state['candidate_recent_detections']) < confirmation_min_positive_detections) or \
                       (confirmation_window_frames == 0 and confirmation_min_positive_detections > 0 and not any(state['candidate_recent_detections'])): # for window 0, if it was 1 true, it would have confirmed
                        state.update({'candidate_poly': None, 'candidate_recent_detections': []})

                if not seen_at_locked_pos_this_frame:
                    state['frames_unseen_at_locked_pos'] += 1
                    # Note: Add logic here if you want to unlock after 'X' frames_unseen_at_locked_pos without a new candidate confirming.
                    # For now, it stays locked indefinitely until a new position is confirmed.

                if locked_poly_warped is not None:
                    poly_to_transform = locked_poly_warped.reshape(-1,1,2).astype(np.float32) if locked_poly_warped.ndim == 2 else locked_poly_warped.astype(np.float32)
                    output_poly_for_this_piece_original_coords = cv2.perspectiveTransform(poly_to_transform, inverse_perspective_M)

            piece_states[piece_id] = state # Persist state changes

            if output_poly_for_this_piece_original_coords is not None:
                current_frame_output_pieces_list.append({"piece_id": piece_id, "color_name": piece_cfg.get("color_name", "N/A"), "class_name": piece_cfg.get("class_name", "N/A"), "vertices": output_poly_for_this_piece_original_coords.reshape(-1, 2).tolist()})
                cv2.drawContours(display_original_output, [output_poly_for_this_piece_original_coords.astype(np.int32)], -1, PIECE_LOCKED_COLOR_BGR, PIECE_LOCKED_THICKNESS)
                for v_x, v_y in output_poly_for_this_piece_original_coords.reshape(-1, 2): cv2.circle(display_original_output, (int(v_x), int(v_y)), PIECE_VERTEX_RADIUS, PIECE_VERTEX_COLOR_BGR, -1)
        
        frames_data_for_json.append({"frame_index": frame_idx, "frame_timestamp_ms": current_frame_timestamp_ms, "pieces": current_frame_output_pieces_list})
        
        if not args.no_display:
            cv2.imshow("Tangram Detection on Original Frame", display_original_output)
            display_wait_ms = 1
            if args.realtime:
                proc_time_sec = time.time() - loop_start_time; target_interval_sec = 1.0 / fps if fps > 0 else 0.033
                wait_time_sec = target_interval_sec - proc_time_sec
                if wait_time_sec > 0: display_wait_ms = int(wait_time_sec * 1000)
            key = cv2.waitKey(max(1,display_wait_ms)) & 0xFF
            if key == ord('q'): break
            if key == ord('p') and args.debug_piece_id: cv2.waitKey(0) # Pause if 'p' is pressed and debugging a piece
        frame_idx += 1

    cap.release();
    if not args.no_display: cv2.destroyAllWindows()
    end_processing_time = time.time()

    output_json_content = {
        "metadata": {"processing_start_utc": datetime.datetime.fromtimestamp(start_processing_time, datetime.timezone.utc).isoformat(), "processing_duration_seconds": round(end_processing_time - start_processing_time, 3),
                     "video_file": args.video, "config_file": args.config, "output_file": args.output_json, "total_frames_processed": frame_idx, "video_fps": round(fps,2),
                     "original_image_width": frame_width_orig, "original_image_height": frame_height_orig, "warped_region_width": warp_width, "warped_region_height": warp_height,
                     "command_line_args": vars(args),
                     "table_detection_params_used": {"LOWER_HSV": TABLE_LOWER_HSV.tolist(), "UPPER_HSV": TABLE_UPPER_HSV.tolist(), "UPPER_REGION_PERCENTAGE": TABLE_REGION_UPPER_PERCENTAGE, "CORNER_EXTRUSION_PIXELS": TABLE_CORNER_EXTRUSION_PIXELS},
                     "piece_detection_params_used":{"CORNER_EXTRUSION_PIXELS": PIECE_CORNER_EXTRUSION_PIXELS, "SHAPE_REGULARITY_MIN_ANGLE_DEG": MIN_ALLOWED_ANGLE_DEG, "SHAPE_REGULARITY_MAX_ANGLE_DEG": MAX_ALLOWED_ANGLE_DEG, "WEIGHT_COLOR_CLOSENESS": WEIGHT_COLOR_CLOSENESS, "WEIGHT_AREA_CLOSENESS": WEIGHT_AREA_CLOSENESS},
                     "temporal_params_used": {"CONFIRM_STABLE_DURATION_S": CONFIRM_STABLE_DURATION_S, "CONFIRM_MIN_PERCENT_VISIBLE": CONFIRM_MIN_PERCENT_VISIBLE, "POSITION_STABILITY_PX": POSITION_STABILITY_PX, "AREA_STABILITY_RATIO": AREA_STABILITY_RATIO}},
        "tangram_config_data_used": tangram_config_data, "frames_data": frames_data_for_json}

    with open(args.output_json, 'w') as f_out: json.dump(output_json_content, f_out, indent=2)
    print(f"\nProcessed {frame_idx} frames. Output saved to {args.output_json}")

if __name__ == "__main__":
    main()
