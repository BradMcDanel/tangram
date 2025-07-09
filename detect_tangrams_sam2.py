#!/usr/bin/env python3
"""
True Hybrid Tangram Detector: Guided ROI, Perspective Warp, and SAM2 Injection.

This script fuses the best of all previous methods into a robust pipeline:
1.  (User) A human-guided ROI provides a perfect starting point.
2.  (CV) The ROI is perspective-warped to a top-down view, normalizing shapes.
3.  (CV) A classical color blob search on the warped image proposes candidates.
4.  (SAM2) A box prompt from the blob is used to get a pixel-perfect mask from SAM2.
5.  (CV) The clean mask is verified for shape and sharpened into a perfect polygon.
6.  (CV) The final polygon is projected back to the original image.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import json

# --- 1. Configuration ---
SAM2_CHECKPOINT = os.path.expanduser("~/data-main-1/sam2/sam2.1_hiera_large.pt")
INPUT_IMAGE_PATH = os.path.expanduser("~/code/tangram/data/images/calibration.png")
CONFIG_PATH = os.path.expanduser("~/code/tangram/tangram_config.json")
OUTPUT_DIR = "output/images"
OUTPUT_IMAGE_PATH = os.path.join(OUTPUT_DIR, "true_hybrid_output.png")
MODEL_CONFIG_NAME = "configs/sam2.1/sam2.1_hiera_l.yaml"

# --- Parameters ---
MIN_PIECE_AREA_WARPED = 50  # Min area in the warped, top-down view
SHAPE_APPROX_EPSILON = 0.03

# --- Table Detection & Warping Parameters (from video_tangram_detector.py) ---
TABLE_LOWER_HSV = np.array([100, 0, 115])
TABLE_UPPER_HSV = np.array([179, 35, 190])
TABLE_MORPH_KERNEL_SIZE = (5, 5)
TABLE_MORPH_OPEN_ITERATIONS = 1
TABLE_MORPH_CLOSE_ITERATIONS = 2
TABLE_APPROX_POLY_EPSILON_FACTOR = 0.03
TABLE_REGION_UPPER_PERCENTAGE = 0.45
TABLE_CORNER_EXTRUSION_PIXELS = 5

# --- Tangram Piece Detection Parameters (from video_tangram_detector.py) ---
PIECE_BLUR_KERNEL_SIZE = (5, 5)
PIECE_MORPH_KERNEL_SIZE = (3, 3)
PIECE_MORPH_OPEN_ITERATIONS = 1
PIECE_MORPH_CLOSE_ITERATIONS = 1
PIECE_CORNER_EXTRUSION_PIXELS = 3
PIECE_EPSILON_SEARCH_START = 0.01
PIECE_EPSILON_SEARCH_END = 0.1
PIECE_EPSILON_SEARCH_STEP = 0.001

# --- Shape Regularity Parameters (from video_tangram_detector.py) ---
MIN_ALLOWED_ANGLE_DEG = 30.0
MAX_ALLOWED_ANGLE_DEG = 130.0
SQUARE_TARGET_ANGLE_DEG = 90.0
SQUARE_ANGLE_TOLERANCE_DEG = 15.0
PARALLELOGRAM_OPPOSITE_ANGLE_DIFF_TOLERANCE_DEG = 30.0
PARALLELOGRAM_ADJACENT_ANGLE_SUM_TARGET_DEG = 180.0
PARALLELOGRAM_ADJACENT_ANGLE_SUM_TOLERANCE_DEG = 30.0

# --- Relaxation Parameters for Piece Detection (from video_tangram_detector.py) ---
MAX_RELAXATION_STEPS = 5
RELAX_PERCENT_INCREMENT_CH0 = 0.001 # For Hue
RELAX_PERCENT_INCREMENT_CH12 = 0.002 # For Saturation & Value
AREA_RELAX_FACTOR_PER_STEP = 0.01

# --- Scoring Parameters for Piece Detection (from video_tangram_detector.py) ---
WEIGHT_COLOR_CLOSENESS = 0.6
WEIGHT_AREA_CLOSENESS = 0.4

# --- 2. Helper Functions ---

# --- Helper Functions from video_tangram_detector.py ---
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

def get_hue_distance(h1, h2, max_hue_val=179):
    """Calculates the shortest distance between two hue values on a circle (0-max_hue_val)."""
    diff = abs(h1 - h2)
    return min(diff, (max_hue_val + 1) - diff)

def process_tangrams_in_warped_frame(warped_frame, config_data, debug_pid_str=None, save_debug_images=False):
    if warped_frame is None or warped_frame.size == 0: print("ERROR: Received empty warped_frame."); return {}

    blur_warped_frame = cv2.GaussianBlur(warped_frame, PIECE_BLUR_KERNEL_SIZE, 0)
    conv_img_masking = cv2.cvtColor(blur_warped_frame, cv2.COLOR_BGR2HSV)
    ch0_max, ch1_max, ch2_max = 179, 255, 255 # Max values for H, S, V

    morph_k_piece = np.ones(PIECE_MORPH_KERNEL_SIZE, np.uint8)
    raw_detected_piece_data = {}

    for pid, p_attrs in config_data.items():
        p_color_space_key = p_attrs.get('color_space', 'hsv')
        if p_color_space_key != 'hsv':
            print(f"WARNING: Piece {pid} has color_space '{p_color_space_key}', but only 'hsv' is fully supported in this simplified version. Attempting to use hsv_lower/hsv_upper keys.")

        orig_low_thresh = np.array(p_attrs[f'{p_color_space_key}_lower'])
        orig_up_thresh = np.array(p_attrs[f'{p_color_space_key}_upper'])

        tgt_verts = p_attrs['num_vertices']
        min_area, max_area = p_attrs.get('min_area', 0), p_attrs.get('max_area', float('inf'))
        tgt_area_mid = (min_area + max_area) / 2.0 if max_area != float('inf') and (min_area + max_area) > 0 else min_area * 1.5 if min_area > 0 else 1.0

        original_color_center = (orig_low_thresh + orig_up_thresh) / 2.0
        original_color_range_half = (orig_up_thresh - orig_low_thresh) / 2.0
        original_color_range_half[original_color_range_half == 0] = 1.0 # Avoid division by zero; treats exact value matches correctly

        best_poly_in_warped_coords = None
        best_combined_score = float('inf')

        for relax_step in range(MAX_RELAXATION_STEPS + 1):
            curr_low_thresh, curr_up_thresh = orig_low_thresh.copy(), orig_up_thresh.copy()
            if relax_step > 0:
                ch0_adj = int(ch0_max * RELAX_PERCENT_INCREMENT_CH0 * relax_step) # Hue
                ch1_adj = int(ch1_max * RELAX_PERCENT_INCREMENT_CH12 * relax_step) # Saturation
                ch2_adj = int(ch2_max * RELAX_PERCENT_INCREMENT_CH12 * relax_step) # Value

                curr_low_thresh -= np.array([ch0_adj, ch1_adj, ch2_adj])
                curr_up_thresh += np.array([ch0_adj, ch1_adj, ch2_adj])
                np.clip(curr_low_thresh, 0, None, out=curr_low_thresh)
                curr_up_thresh[0] = min(ch0_max, curr_up_thresh[0])
                curr_up_thresh[1] = min(ch1_max, curr_up_thresh[1])
                curr_up_thresh[2] = min(ch2_max, curr_up_thresh[2])

            if conv_img_masking.size == 0: continue
            color_mask = cv2.inRange(conv_img_masking, curr_low_thresh, curr_up_thresh)
            opened_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, morph_k_piece, iterations=PIECE_MORPH_OPEN_ITERATIONS)
            proc_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, morph_k_piece, iterations=PIECE_MORPH_CLOSE_ITERATIONS)

            contours, _ = cv2.findContours(proc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                min_area_mult = 1 - AREA_RELAX_FACTOR_PER_STEP * relax_step
                max_area_mult = 1 + AREA_RELAX_FACTOR_PER_STEP * relax_step
                relaxed_min_a = max(0.0, min_area * min_area_mult)
                relaxed_max_a = max_area * max_area_mult
                if not (relaxed_min_a <= area <= relaxed_max_a):
                    continue

                peri = cv2.arcLength(cnt, True)
                if peri > 0:
                    for eps_factor in np.arange(PIECE_EPSILON_SEARCH_START, PIECE_EPSILON_SEARCH_END, PIECE_EPSILON_SEARCH_STEP):
                        poly = cv2.approxPolyDP(cnt, eps_factor * peri, True)
                        if len(poly) == tgt_verts:
                            shape_class = p_attrs.get("class_name", "Unknown")
                            if is_shape_regular(poly, shape_class, tgt_verts):
                                area_diff_abs = abs(area - tgt_area_mid)
                                normalized_area_distance = area_diff_abs / tgt_area_mid if tgt_area_mid > 0 else (0.0 if area_diff_abs == 0 else 1.0)

                                contour_mask_for_avg_color = np.zeros(proc_mask.shape[:2], dtype=np.uint8)
                                cv2.drawContours(contour_mask_for_avg_color, [cnt], -1, 255, -1)
                                avg_color_tuple = cv2.mean(conv_img_masking, mask=contour_mask_for_avg_color)
                                avg_color_piece = np.array(avg_color_tuple[:3])

                                color_distances_per_channel = np.zeros(3)
                                color_distances_per_channel[0] = get_hue_distance(avg_color_piece[0], original_color_center[0], ch0_max)
                                color_distances_per_channel[1] = abs(avg_color_piece[1] - original_color_center[1])
                                color_distances_per_channel[2] = abs(avg_color_piece[2] - original_color_center[2])

                                normalized_color_distances_per_channel = color_distances_per_channel / original_color_range_half
                                normalized_color_distances_per_channel = np.nan_to_num(normalized_color_distances_per_channel, nan=1.0)
                                normalized_color_distances_per_channel = np.clip(normalized_color_distances_per_channel, 0.0, 1.0)
                                normalized_color_distance_avg = np.mean(normalized_color_distances_per_channel)

                                current_combined_score = (WEIGHT_COLOR_CLOSENESS * normalized_color_distance_avg +\
                                                          WEIGHT_AREA_CLOSENESS * normalized_area_distance)

                                if current_combined_score < best_combined_score:
                                    best_combined_score = current_combined_score
                                    best_poly_in_warped_coords = extrude_polygon_corners(poly, PIECE_CORNER_EXTRUSION_PIXELS)
                                break
                        elif len(poly) < tgt_verts:
                            break

        if best_poly_in_warped_coords is not None:
            raw_detected_piece_data[pid] = {
                'poly': best_poly_in_warped_coords,
                'score': best_combined_score
            }
    return raw_detected_piece_data

# --- 2. Helper Functions ---

def load_tangram_config(config_path):
    with open(config_path, 'r') as f: config = json.load(f)
    for _, p in config.items():
        p['hsv_lower'] = np.array(p['hsv_lower']); p['hsv_upper'] = np.array(p['hsv_upper'])
    return config



def verify_shape(polygon, num_vertices_expected):
    """Verifies a polygon has the correct number of vertices."""
    return len(polygon) == num_vertices_expected

def load_sam_predictor_model(device):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    print("Loading SAM 2 model for Image Predictor...")
    sam_model = build_sam2(MODEL_CONFIG_NAME, SAM2_CHECKPOINT, device=device)
    return SAM2ImagePredictor(sam_model)

def get_color_map():
    color_map_bgr = {"Pink": (203, 192, 255), "Red": (0, 0, 255), "Orange": (0, 165, 255), "Blue": (255, 0, 0), "Green": (0, 255, 0), "Yellow": (0, 255, 255), "Purple": (128, 0, 128)}
    return {name: (bgr[2]/255, bgr[1]/255, bgr[0]/255) for name, bgr in color_map_bgr.items()}

def visualize_results(full_image_rgb, final_polys_orig, color_map, output_path):
    plt.figure(figsize=(12, 12)); plt.imshow(full_image_rgb); ax = plt.gca(); ax.set_autoscale_on(False)
    for poly_data in final_polys_orig:
        poly, config = poly_data['poly'], poly_data['config']
        highlight_color = color_map.get(config['color_name'], (1, 1, 1))
        patch = plt.Polygon(poly.reshape(-1, 2), color=highlight_color, alpha=0.9)
        ax.add_patch(patch)
    plt.axis('off'); plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
    print(f"\nSuccess! Fused detector output saved to {output_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    tangram_config = load_tangram_config(CONFIG_PATH)
    image_rgb = np.array(Image.open(INPUT_IMAGE_PATH).convert("RGB"))

    # Step 1 & 2: Dynamically detect table and warp to a top-down view
    print("--- Step 1&2: Dynamically detecting table and warping ROI ---")
    # initialize_table_warp_parameters expects BGR, so convert image_rgb to BGR
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    perspective_M, inverse_M, warp_width, warp_height, table_outline_pts_orig = initialize_table_warp_parameters(image_bgr)
    if perspective_M is None: print("Could not create warp matrices."); return
    
    warped_rgb = cv2.warpPerspective(image_rgb, perspective_M, (warp_width, warp_height))
    warped_hsv = cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2HSV)

    # Step 3: Load SAM2 Predictor and set the warped image
    print("\n--- Step 3: Initializing SAM2 Predictor on Warped ROI ---")
    predictor = load_sam_predictor_model(device)
    predictor.set_image(warped_rgb)
    
    final_polys_in_orig_coords = []
    
    # Step 4, 5, 6, 7: Propose, Segment, Verify, Project
    print("\n--- Step 4-7: Proposing, Segmenting, and Verifying each piece ---")
    raw_detections_in_warped = process_tangrams_in_warped_frame(warped_rgb, tangram_config, debug_pid_str=None, save_debug_images=False)
    
    for piece_id, config in tangram_config.items():
        print(f"  - Searching for '{config['color_name']}' piece...")
        
        raw_poly_data = raw_detections_in_warped.get(piece_id)
        if raw_poly_data is None:
            print("    -> No color blob found.")
            continue
        
        main_contour = raw_poly_data['poly']
        
        # B. Create a box prompt from the blob
        x_b, y_b, w_b, h_b = cv2.boundingRect(main_contour)
        box_prompt = np.array([x_b, y_b, x_b + w_b, y_b + h_b])
        
        # C. Use SAM2 to get a precision mask from the box prompt
        masks, scores, _ = predictor.predict(box=box_prompt[None, :], multimask_output=True)
        sam_mask = masks[np.argmax(scores)].astype(bool)
        
        # D. Sharpen the SAM2 mask into a clean polygon and verify its shape
        sam_contours, _ = cv2.findContours(sam_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not sam_contours:
            print("    -> SAM2 returned an empty mask.")
            continue
            
        sharp_poly = cv2.approxPolyDP(max(sam_contours, key=cv2.contourArea), SHAPE_APPROX_EPSILON * cv2.arcLength(max(sam_contours, key=cv2.contourArea), True), True)
        
        if verify_shape(sharp_poly, config['num_vertices']):
            print(f"    -> SUCCESS: Found and verified '{config['color_name']}'.")
            poly_in_orig = cv2.perspectiveTransform(sharp_poly.astype(np.float32), inverse_M)
            final_polys_in_orig_coords.append({'poly': poly_in_orig, 'config': config})
        else:
            print(f"    -> FAILED: Final shape verification failed.")

    print(f"\nFinal detection count: {len(final_polys_in_orig_coords)} pieces.")
    color_map = get_color_map()
    visualize_results(image_rgb, final_polys_in_orig_coords, color_map, OUTPUT_IMAGE_PATH)

if __name__ == '__main__':
    main()
