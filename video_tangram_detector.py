import cv2
import numpy as np
import json
import argparse
import time
import datetime

# --- Global Detection Parameters ---
MAX_RELAXATION_STEPS = 10
RELAX_PERCENT_INCREMENT_CH0 = 0.001
RELAX_PERCENT_INCREMENT_CH12 = 0.002
AREA_RELAX_FACTOR_PER_STEP = 0.025
MIN_PIECES_FOR_ROI_DEFINITION = 3
ROI_DIM_PERCENT_OF_IMAGE = 0.45

# --- Temporal Smoothing and Persistence Parameters ---
CONFIRM_STABLE_DURATION_S = 0.25
CONFIRM_MIN_PERCENT_VISIBLE = 30.0
POSITION_STABILITY_PX = 1
AREA_STABILITY_RATIO = 0.01


def get_poly_metrics(poly):
    if poly is None or len(poly) == 0: return None, 0
    M = cv2.moments(poly)
    area = M['m00']
    if area == 0:
        x_coords = poly[:, 0, 0]
        y_coords = poly[:, 0, 1]
        if len(x_coords) == 0: return None, 0
        cx = np.mean(x_coords)
        cy = np.mean(y_coords)
    else:
        cx = M['m10'] / area
        cy = M['m01'] / area
    return (int(cx), int(cy)), area

def compare_polys_for_stability(poly1, poly2):
    if poly1 is None or poly2 is None: return False
    c1, a1 = get_poly_metrics(poly1)
    c2, a2 = get_poly_metrics(poly2)
    if c1 is None or c2 is None: return False

    centroid_dist = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

    if a1 == 0 and a2 == 0: area_diff_ok = True
    elif a1 == 0 or a2 == 0: area_diff_ok = False
    elif max(a1, a2) == 0 : area_diff_ok = True
    else: area_diff_ok = (abs(a1 - a2) / max(a1, a2)) < AREA_STABILITY_RATIO

    return centroid_dist < POSITION_STABILITY_PX and area_diff_ok

def process_single_frame(frame, config_data, debug_pid_str=None, show_cv_windows=True, frame_num_debug=0):
    img_h, img_w = frame.shape[:2]
    if not config_data: return {}, frame.copy()
    first_pid = next(iter(config_data), None)
    if not first_pid: return {}, frame.copy()

    g_color_space = config_data[first_pid]['color_space']
    blur_frame = cv2.GaussianBlur(frame, (5, 5), 0)

    if g_color_space == 'hsv':
        conv_img_masking = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2HSV)
        ch0_max = 179
    elif g_color_space == 'lab':
        conv_img_masking = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2LAB)
        ch0_max = 255
    else: return {}, frame.copy()

    ch1_max, ch2_max = 255, 255
    morph_k = np.ones((3,3), np.uint8)
    output_display_img = (frame * 0.7).astype(np.uint8)

    initial_detections = {}
    for pid, p_attrs in config_data.items():
        p_color_space = p_attrs['color_space']
        low_thresh = np.array(p_attrs[f'{p_color_space}_lower'])
        up_thresh = np.array(p_attrs[f'{p_color_space}_upper'])
        tgt_verts = p_attrs['num_vertices']
        min_area, max_area = p_attrs.get('min_area', 0), p_attrs.get('max_area', float('inf'))
        tgt_area_mid = (min_area + max_area) / 2.0 if max_area != float('inf') and (min_area + max_area) > 0 else min_area * 1.5 if min_area > 0 else 1.0

        color_mask = cv2.inRange(conv_img_masking, low_thresh, up_thresh)
        opened_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, morph_k, iterations=1)
        proc_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, morph_k, iterations=1)

        if show_cv_windows and debug_pid_str == pid:
            debug_mask_disp = frame.copy(); debug_mask_disp[proc_mask == 0] = 0
            cv2.imshow(f"F{frame_num_debug} P1 Raw P{pid}", debug_mask_disp)

        contours, _ = cv2.findContours(proc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_cnt, best_poly, min_area_diff = None, None, float('inf')
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (min_area <= area <= max_area): continue
            peri = cv2.arcLength(cnt, True)
            if peri > 0:
                for eps in np.arange(0.01, 1, 0.00125):
                    poly = cv2.approxPolyDP(cnt, eps * peri, True)
                    if len(poly) == tgt_verts:
                        area_diff = abs(area - tgt_area_mid) if tgt_area_mid > 0 else area
                        if area_diff < min_area_diff:
                            min_area_diff, best_cnt, best_poly = area_diff, cnt, poly
                        break
        if best_cnt is not None: initial_detections[pid] = {'contour': best_cnt, 'poly': best_poly}

    roi_x, roi_y, roi_w, roi_h = 0, 0, img_w, img_h
    roi_defined = False
    if len(initial_detections) >= MIN_PIECES_FOR_ROI_DEFINITION:
        centroids_x, centroids_y = [], []
        for data in initial_detections.values():
            M = cv2.moments(data['contour'])
            if M["m00"] != 0:
                centroids_x.append(int(M["m10"] / M["m00"])); centroids_y.append(int(M["m01"] / M["m00"]))
        if centroids_x and centroids_y:
            med_cx, med_cy = int(np.median(centroids_x)), int(np.median(centroids_y))
            roi_dim_w_tgt = int(img_w * ROI_DIM_PERCENT_OF_IMAGE)
            roi_dim_h_tgt = int(img_h * ROI_DIM_PERCENT_OF_IMAGE)
            roi_x = max(0, int(med_cx - roi_dim_w_tgt / 2)); roi_y = max(0, int(med_cy - roi_dim_h_tgt / 2))
            roi_w = min(roi_dim_w_tgt, img_w - roi_x); roi_h = min(roi_dim_h_tgt, img_h - roi_y)
            if roi_w > 0 and roi_h > 0: roi_defined = True
            else: roi_x, roi_y, roi_w, roi_h = 0, 0, img_w, img_h
    if roi_defined: cv2.rectangle(output_display_img, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 255), 1)

    raw_detected_piece_data = {}
    search_img_mask_base = conv_img_masking[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    frame_debug_base = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    off_x, off_y = roi_x, roi_y

    for pid, p_attrs in config_data.items():
        p_color_space = p_attrs['color_space']
        orig_low_thresh = np.array(p_attrs[f'{p_color_space}_lower'])
        orig_up_thresh = np.array(p_attrs[f'{p_color_space}_upper'])
        tgt_verts = p_attrs['num_vertices']
        min_area, max_area = p_attrs.get('min_area', 0), p_attrs.get('max_area', float('inf'))
        tgt_area_mid = (min_area + max_area) / 2.0 if max_area != float('inf') and (min_area + max_area) > 0 else min_area * 1.5 if min_area > 0 else 1.0
        best_poly_overall, best_mask_overall, best_score, best_relax_step = None, None, float('inf'), float('inf')

        for relax_step in range(MAX_RELAXATION_STEPS + 1):
            curr_low_thresh, curr_up_thresh = orig_low_thresh.copy(), orig_up_thresh.copy()
            if relax_step > 0:
                ch0_adj = int(ch0_max * RELAX_PERCENT_INCREMENT_CH0 * relax_step)
                ch1_adj = int(ch1_max * RELAX_PERCENT_INCREMENT_CH12 * relax_step)
                ch2_adj = int(ch2_max * RELAX_PERCENT_INCREMENT_CH12 * relax_step)
                curr_low_thresh -= np.array([ch0_adj, ch1_adj, ch2_adj])
                curr_up_thresh += np.array([ch0_adj, ch1_adj, ch2_adj])
                np.clip(curr_low_thresh, 0, None, out=curr_low_thresh)
                curr_up_thresh[0] = min(ch0_max, curr_up_thresh[0]); curr_up_thresh[1] = min(ch1_max, curr_up_thresh[1]); curr_up_thresh[2] = min(ch2_max, curr_up_thresh[2])

            if search_img_mask_base.size == 0: continue
            color_mask_roi = cv2.inRange(search_img_mask_base, curr_low_thresh, curr_up_thresh)
            opened_mask_roi = cv2.morphologyEx(color_mask_roi, cv2.MORPH_OPEN, morph_k, iterations=1)
            proc_mask_roi = cv2.morphologyEx(opened_mask_roi, cv2.MORPH_CLOSE, morph_k, iterations=1)

            if show_cv_windows and debug_pid_str == pid and frame_debug_base.size > 0 :
                debug_mask_disp_p3 = frame_debug_base.copy(); debug_mask_disp_p3[proc_mask_roi == 0] = 0
                title = f"F{frame_num_debug} P3 P{pid} R{relax_step}" + (" (ROI)" if roi_defined else "")
                cv2.imshow(title, debug_mask_disp_p3)

            contours_roi, _ = cv2.findContours(proc_mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt_roi in contours_roi:
                area_roi = cv2.contourArea(cnt_roi)
                min_area_mult = 1 - AREA_RELAX_FACTOR_PER_STEP * relax_step
                max_area_mult = 1 + AREA_RELAX_FACTOR_PER_STEP * relax_step
                relaxed_min_a = max(0.0, min_area * min_area_mult); relaxed_max_a = max_area * max_area_mult
                if not (relaxed_min_a <= area_roi <= relaxed_max_a): continue
                peri_roi = cv2.arcLength(cnt_roi, True)
                if peri_roi > 0:
                    for eps in np.arange(0.01, 1.0, 0.000625):
                        poly_roi = cv2.approxPolyDP(cnt_roi, eps * peri_roi, True)
                        if len(poly_roi) == tgt_verts:
                            area_diff_curr = abs(area_roi - tgt_area_mid) if tgt_area_mid > 0 else area_roi
                            if relax_step < best_relax_step or (relax_step == best_relax_step and area_diff_curr < best_score):
                                best_score, best_relax_step = area_diff_curr, relax_step
                                best_poly_overall = poly_roi + np.array([off_x, off_y])
                            break
        if best_poly_overall is not None:
            raw_detected_piece_data[pid] = {'poly': best_poly_overall}
            if show_cv_windows and debug_pid_str == pid and best_mask_overall is not None:
                 output_display_img[best_mask_overall > 0] = frame[best_mask_overall > 0]
    return raw_detected_piece_data, output_display_img

def get_default_piece_state():
    return {
        'status': 'UNSEEN',
        'candidate_poly': None,
        'candidate_recent_detections': [],
        'locked_poly': None,
        'frames_unseen_at_locked_pos': 0
    }

def main():
    parser = argparse.ArgumentParser(description="Tangram detection with revised temporal persistence.")
    parser.add_argument("--video", required=True, help="Input video file.")
    parser.add_argument("--config", required=True, help="JSON config file.")
    parser.add_argument("--output_json", required=True, help="Output JSON file.")
    parser.add_argument("--debug_piece_id", type=str, default=None, help="ID for debug masks.")
    parser.add_argument("--no_display", action="store_true", help="Disable OpenCV display.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose console logs.")
    parser.add_argument("--realtime", action="store_true", help="Playback video in real-time (if display is enabled).")
    args = parser.parse_args()

    start_processing_time = time.time()
    with open(args.config, 'r') as f: config_data_content = json.load(f)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened(): print(f"Error: Cannot open video {args.video}"); return

    fps = cap.get(cv2.CAP_PROP_FPS); fps = fps if fps > 0 else 30.0

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    confirmation_window_frames = int(CONFIRM_STABLE_DURATION_S * fps)

    confirmation_min_positive_detections = 0
    if confirmation_window_frames > 0 :
        confirmation_min_positive_detections = int(confirmation_window_frames * (CONFIRM_MIN_PERCENT_VISIBLE / 100.0))
        if confirmation_min_positive_detections == 0 and CONFIRM_MIN_PERCENT_VISIBLE > 0:
             confirmation_min_positive_detections = 1
    elif CONFIRM_STABLE_DURATION_S > 0 and CONFIRM_MIN_PERCENT_VISIBLE > 0 :
        confirmation_min_positive_detections = 1


    frames_data_for_json = []
    piece_states = {pid: get_default_piece_state() for pid in config_data_content}
    frame_idx = 0
    total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        loop_start_time = time.time()
        ret, frame = cap.read()
        if not ret: break
        current_frame_timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

        if args.verbose or (frame_idx > 0 and frame_idx % 30 == 0):
            prog_str = f"Processing Frame {frame_idx}"
            if total_frames_video > 0: prog_str += f" / {total_frames_video -1}"
            print(prog_str)

        raw_detections, output_display_img_base = process_single_frame(
            frame.copy(), config_data_content, args.debug_piece_id, not args.no_display, frame_idx
        )

        current_frame_output_pieces_list = []

        for piece_id, piece_cfg in config_data_content.items():
            state = piece_states[piece_id]
            raw_poly_data = raw_detections.get(piece_id)
            raw_poly = raw_poly_data['poly'] if raw_poly_data else None
            output_poly_for_this_piece = None

            if state['status'] == 'UNSEEN':
                if raw_poly is not None:
                    state['status'] = 'CONFIRMING'
                    state['candidate_poly'] = raw_poly
                    state['candidate_recent_detections'] = [True]
                    if confirmation_window_frames == 0 and 1 >= confirmation_min_positive_detections:
                        state['status'] = 'LOCKED'
                        state['locked_poly'] = state['candidate_poly']
                        state['frames_unseen_at_locked_pos'] = 0
                        state['candidate_poly'] = None
                        state['candidate_recent_detections'] = []

            if state['status'] == 'CONFIRMING':
                if raw_poly is not None:
                    if state['candidate_poly'] is None:
                        state['candidate_poly'] = raw_poly
                        state['candidate_recent_detections'] = [True]
                    elif compare_polys_for_stability(raw_poly, state['candidate_poly']):
                        state['candidate_recent_detections'].append(True)
                    else:
                        state['candidate_poly'] = raw_poly
                        state['candidate_recent_detections'] = [True]
                else:
                    if state['candidate_poly'] is not None:
                        state['candidate_recent_detections'].append(False)

                while len(state['candidate_recent_detections']) > confirmation_window_frames and confirmation_window_frames > 0:
                    state['candidate_recent_detections'].pop(0)

                if confirmation_window_frames == 0 and state['candidate_poly'] is not None:
                    if 1 >= confirmation_min_positive_detections:
                        state['status'] = 'LOCKED'
                        state['locked_poly'] = state['candidate_poly']
                        state['frames_unseen_at_locked_pos'] = 0
                        state['candidate_poly'] = None
                        state['candidate_recent_detections'] = []
                    else:
                        state.update(get_default_piece_state())
                        state['status'] = 'UNSEEN'
                elif len(state['candidate_recent_detections']) == confirmation_window_frames and \
                     state['candidate_poly'] is not None and confirmation_window_frames > 0:
                    positive_detections_count = sum(state['candidate_recent_detections'])
                    if positive_detections_count >= confirmation_min_positive_detections:
                        state['status'] = 'LOCKED'
                        state['locked_poly'] = state['candidate_poly']
                        state['frames_unseen_at_locked_pos'] = 0
                        state['candidate_poly'] = None
                        state['candidate_recent_detections'] = []
                    else:
                        if state['locked_poly'] is not None:
                            state['status'] = 'LOCKED'
                            state['candidate_poly'] = None
                            state['candidate_recent_detections'] = []
                        else:
                            state.update(get_default_piece_state())
                            state['status'] = 'UNSEEN'

            if state['status'] == 'LOCKED':
                output_poly_for_this_piece = state['locked_poly']
                seen_at_locked_pos_this_frame = False

                if raw_poly is not None:
                    if compare_polys_for_stability(raw_poly, state['locked_poly']):
                        seen_at_locked_pos_this_frame = True
                        state['frames_unseen_at_locked_pos'] = 0
                        state['candidate_poly'] = None
                        state['candidate_recent_detections'] = []
                    else:
                        if state['candidate_poly'] is None or \
                           not compare_polys_for_stability(raw_poly, state['candidate_poly']):
                            state['candidate_poly'] = raw_poly
                            state['candidate_recent_detections'] = [True]
                        else:
                            state['candidate_recent_detections'].append(True)
                        
                        while len(state['candidate_recent_detections']) > confirmation_window_frames and confirmation_window_frames > 0:
                            state['candidate_recent_detections'].pop(0)

                        reloc_candidate_confirmed = False
                        if confirmation_window_frames == 0 and state['candidate_poly'] is not None:
                            if 1 >= confirmation_min_positive_detections:
                                reloc_candidate_confirmed = True
                        elif len(state['candidate_recent_detections']) == confirmation_window_frames and \
                             state['candidate_poly'] is not None and confirmation_window_frames > 0:
                            if sum(state['candidate_recent_detections']) >= confirmation_min_positive_detections:
                                reloc_candidate_confirmed = True
                        
                        if reloc_candidate_confirmed:
                            state['locked_poly'] = state['candidate_poly']
                            output_poly_for_this_piece = state['locked_poly']
                            state['frames_unseen_at_locked_pos'] = 0
                            state['candidate_poly'] = None
                            state['candidate_recent_detections'] = []
                            seen_at_locked_pos_this_frame = True

                else:
                    if state['candidate_poly'] is not None:
                        state['candidate_recent_detections'].append(False)
                        while len(state['candidate_recent_detections']) > confirmation_window_frames and confirmation_window_frames > 0:
                            state['candidate_recent_detections'].pop(0)
                        
                        if len(state['candidate_recent_detections']) == confirmation_window_frames and confirmation_window_frames > 0:
                            if sum(state['candidate_recent_detections']) < confirmation_min_positive_detections:
                                state['candidate_poly'] = None
                                state['candidate_recent_detections'] = []
                        elif confirmation_window_frames == 0 and 1 < confirmation_min_positive_detections:
                                state['candidate_poly'] = None 
                                state['candidate_recent_detections'] = []


                if not seen_at_locked_pos_this_frame:
                    state['frames_unseen_at_locked_pos'] += 1
            
            piece_states[piece_id] = state
            if output_poly_for_this_piece is not None:
                current_frame_output_pieces_list.append({
                    "piece_id": piece_id,
                    "color_name": piece_cfg.get("color_name", "N/A"),
                    "class_name": piece_cfg.get("class_name", "N/A"),
                    "vertices": output_poly_for_this_piece.reshape(-1, 2).tolist()
                })
                cv2.drawContours(output_display_img_base, [output_poly_for_this_piece], -1, (0, 255, 0), 2)
                for v_x, v_y in output_poly_for_this_piece.reshape(-1, 2):
                    cv2.circle(output_display_img_base, (int(v_x), int(v_y)), 3, (0, 0, 255), -1)

        frames_data_for_json.append({
            "frame_index": frame_idx,
            "frame_timestamp_ms": current_frame_timestamp_ms,
            "pieces": current_frame_output_pieces_list
        })

        if not args.no_display:
            display_wait_ms = 1
            if args.realtime:
                loop_end_time = time.time()
                processing_time_sec = loop_end_time - loop_start_time
                target_frame_interval_sec = 1.0 / fps if fps > 0 else 0.033
                wait_time_sec = target_frame_interval_sec - processing_time_sec
                if wait_time_sec > 0:
                    display_wait_ms = int(wait_time_sec * 1000)
                display_wait_ms = max(1, display_wait_ms)

            cv2.imshow("Tangram Detection", output_display_img_base)
            key = cv2.waitKey(display_wait_ms) & 0xFF
            if key == ord('q'): break
            if key == ord('p') and args.debug_piece_id:
                if args.verbose: print(f"Paused on frame {frame_idx}. Press any key to resume.")
                cv2.waitKey(0)

        frame_idx += 1

    cap.release()
    if not args.no_display: cv2.destroyAllWindows()
    end_processing_time = time.time()

    output_json_content = {
        "metadata": {
            "processing_start_utc": datetime.datetime.fromtimestamp(start_processing_time, datetime.timezone.utc).isoformat(),
            "processing_duration_seconds": round(end_processing_time - start_processing_time, 3),
            "video_file": args.video,
            "config_file": args.config,
            "output_file": args.output_json,
            "total_frames_processed": frame_idx,
            "video_fps": round(fps,2),
            "image_width": frame_width,
            "image_height": frame_height,
            "command_line_args": vars(args),
            "temporal_params": {
                "CONFIRM_STABLE_DURATION_S": CONFIRM_STABLE_DURATION_S,
                "CONFIRM_MIN_PERCENT_VISIBLE": CONFIRM_MIN_PERCENT_VISIBLE,
                "POSITION_STABILITY_PX": POSITION_STABILITY_PX,
                "AREA_STABILITY_RATIO": AREA_STABILITY_RATIO,
                "confirmation_window_frames_calc": confirmation_window_frames,
                "confirmation_min_positive_detections_calc": confirmation_min_positive_detections
            }
        },
        "config_data_used": config_data_content,
        "frames_data": frames_data_for_json
    }

    with open(args.output_json, 'w') as f_out:
        json.dump(output_json_content, f_out, indent=2)
    print(f"\nProcessed {frame_idx} frames. Output saved to {args.output_json}")

if __name__ == "__main__":
    main()
