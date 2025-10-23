import time
from collections import deque
import math

import cv2
import mediapipe as mp
import numpy as np

# ---------------- Configuration ----------------
CAM_ID = 0
FRAME_W, FRAME_H = 1280, 720

# Toolbar config (swatches displayed at top)
SWATCHES = [
    {"name": "Red",   "bgr": (0, 0, 255)},
    {"name": "Green", "bgr": (0, 255, 0)},
    {"name": "Blue",  "bgr": (255, 0, 0)},
    {"name": "Yellow","bgr": (0, 255, 255)},
    {"name": "Purple","bgr": (255, 0, 255)},
]
SWATCH_SIZE = (100, 60)  # (width, height)
SWATCH_PADDING = 15

# Gesture hold thresholds (seconds)
HOLD_SELECT = 0.6
HOLD_ERASE = 0.6
HOLD_CLEAR = 1.2

# Drawing defaults
DEFAULT_COLOR = SWATCHES[0]["bgr"]
DEFAULT_THICKNESS = 8
MIN_THICKNESS = 1
MAX_THICKNESS = 60

# ------------------------------------------------

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

# Setup camera
cap = cv2.VideoCapture(CAM_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

# Canvas + mask
canvas = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)  # color drawing
mask = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)      # 255 where drawing exists

# State
pen_color = DEFAULT_COLOR
thickness = DEFAULT_THICKNESS
eraser_mode = False
dashed_mode = False

prev_point = None
drawing = False

# For selection hold timers
last_over_swatch = None
swatch_start_time = 0

# For gesture holds
two_fingers_start = 0
fist_start = 0

# FPS smoothing
fps_deque = deque(maxlen=8)
prev_time = time.time()

# Utility functions ----------------------------------------------------------
def landmarks_to_pixels(landmarks, w, h):
    """Return list of (x,y) pixel coordinates from mediapipe landmarks."""
    pts = []
    for lm in landmarks.landmark:
        x = int(lm.x * w)
        y = int(lm.y * h)
        pts.append((x, y))
    return pts

def count_extended_fingers(landmarks, img_w, img_h):
    """
    Heuristic finger counting excluding thumb complexities.
    Returns number of extended fingers (index, middle, ring, pinky).
    landmarks: mediapipe landmark object
    """
    pts = landmarks_to_pixels(landmarks, img_w, img_h)
    # landmark indices: tips: 8(index), 12(mid), 16(ring), 20(pinky),
    # PIP joints: 6(index pip), 10(mid pip), 14(ring pip), 18(pinky pip)
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    count = 0
    for tip, pip in zip(tips, pips):
        if pts[tip][1] < pts[pip][1] - 8:  # tip is above pip in image coords (smaller y)
            count += 1
    # thumb check: compare x positions relative to wrist for simplicity (not used in count)
    return count

def draw_dashed_line(img, p1, p2, color, thickness, dash_len=20, gap_len=10):
    """Draw dashed line between p1 and p2 by segmenting the line."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    dist = math.hypot(dx, dy)
    if dist == 0:
        return
    vx = dx / dist
    vy = dy / dist
    pos = 0.0
    draw = True
    while pos < dist:
        seg_len = dash_len if draw else gap_len
        start_x = int(x1 + vx * pos)
        start_y = int(y1 + vy * pos)
        pos += seg_len
        end_x = int(x1 + vx * min(pos, dist))
        end_y = int(y1 + vy * min(pos, dist))
        if draw:
            cv2.line(img, (start_x, start_y), (end_x, end_y), color, thickness)
        draw = not draw

# Toolbar geometry
def get_swatches_layout():
    swatches = []
    total_w = len(SWATCHES) * SWATCH_SIZE[0] + (len(SWATCHES) - 1) * SWATCH_PADDING
    start_x = (FRAME_W - total_w) // 2
    y = 10
    x = start_x
    for s in SWATCHES:
        rect = (x, y, SWATCH_SIZE[0], SWATCH_SIZE[1])
        swatches.append({"name": s["name"], "bgr": s["bgr"], "rect": rect})
        x += SWATCH_SIZE[0] + SWATCH_PADDING
    return swatches

swatches_layout = get_swatches_layout()

# Main loop -----------------------------------------------------------------
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't access camera. Exiting.")
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Run mediapipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands.process(rgb)
        rgb.flags.writeable = True

        # Draw toolbar
        toolbar = frame.copy()
        for sw in swatches_layout:
            x, y, sw_w, sw_h = sw["rect"]
            cv2.rectangle(toolbar, (x, y), (x+sw_w, y+sw_h), (50, 50, 50), -1)
            cv2.rectangle(toolbar, (x+6, y+6), (x+sw_w-6, y+sw_h-6), sw["bgr"], -1)
            cv2.putText(toolbar, sw["name"], (x+6, y+sw_h+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        # Overlay canvas onto frame using mask (only where mask==255)
        composed = frame.copy()
        drawn_pixels = mask == 255
        if np.any(drawn_pixels):
            composed[drawn_pixels] = canvas[drawn_pixels]

        display = composed.copy()
        # Blend toolbar semi-transparently
        alpha_toolbar = 0.9
        display[0:SWATCH_SIZE[1]+40, :] = cv2.addWeighted(toolbar[0:SWATCH_SIZE[1]+40, :], alpha_toolbar,
                                                           display[0:SWATCH_SIZE[1]+40, :], 1 - alpha_toolbar, 0)

        # Process hand detection/gesture logic
        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            pts = landmarks_to_pixels(lm, w, h)
            index_tip = pts[8]
            index_mcp = pts[5]  # base of index
            # draw small cursor circle
            cv2.circle(display, index_tip, 6, (255,255,255), -1)
            cv2.circle(display, index_tip, 10, (0,0,0), 2)

            # Count extended fingers (index..pinky)
            extended = count_extended_fingers(lm, w, h)

            # Determine if hovering over any swatch
            over = None
            ix, iy = index_tip
            for i, sw in enumerate(swatches_layout):
                x, y, sw_w, sw_h = sw["rect"]
                if x <= ix <= x+sw_w and y <= iy <= y+sw_h:
                    over = i
                    break

            # Selection mode: if user shows all 4 fingers (index+middle+ring+pink) - i.e., extended >=4
            if extended >= 4:
                # If hovering on the same swatch for HOLD_SELECT seconds, select it
                if over is not None:
                    if last_over_swatch != over:
                        last_over_swatch = over
                        swatch_start_time = time.time()
                    else:
                        if time.time() - swatch_start_time > HOLD_SELECT:
                            # select swatch color
                            pen_color = swatches_layout[over]["bgr"]
                            eraser_mode = False
                            print(f"Selected color: {swatches_layout[over]['name']}")
                            # small visual feedback: draw border
                            x, y, sw_w, sw_h = swatches_layout[over]["rect"]
                            cv2.rectangle(display, (x-3, y-3), (x+sw_w+3, y+sw_h+3), (255,255,255), 3)
                else:
                    last_over_swatch = None
                    swatch_start_time = 0
                # Not drawing while in selection mode
                prev_point = None
                drawing = False
            else:
                # Not in selection mode -> drawing / gesture detection
                last_over_swatch = None
                swatch_start_time = 0

                # Eraser toggle gesture: if exactly 2 fingers extended (index+middle), hold to toggle
                if extended == 2:
                    if two_fingers_start == 0:
                        two_fingers_start = time.time()
                    else:
                        if time.time() - two_fingers_start > HOLD_ERASE:
                            eraser_mode = not eraser_mode
                            two_fingers_start = 0
                            print("Eraser mode:", eraser_mode)
                else:
                    two_fingers_start = 0

                # Clear gesture: no fingers (fist) held to clear
                if extended == 0:
                    if fist_start == 0:
                        fist_start = time.time()
                    else:
                        if time.time() - fist_start > HOLD_CLEAR:
                            # clear canvas
                            canvas[:] = 0
                            mask[:] = 0
                            prev_point = None
                            print("Canvas cleared (gesture).")
                            fist_start = 0
                else:
                    fist_start = 0

                # Drawing behaviour: draw when index finger is up (at least 1 finger)
                # We'll consider drawing active when index is up (extended >=1)
                if extended >= 1:
                    # Use index tip as pen
                    cur_point = index_tip

                    if prev_point is None:
                        prev_point = cur_point

                    # Draw on canvas (either erasing or drawing)
                    if eraser_mode:
                        # Erase: remove pixels within a circle around cur_point
                        eraser_radius = max(12, thickness * 2)
                        eraser_mask = np.zeros_like(mask, dtype=np.uint8)
                        cv2.circle(eraser_mask, cur_point, eraser_radius, 255, -1)
                        # Clear canvas where eraser_mask is 255
                        canvas[eraser_mask == 255] = 0
                        mask[eraser_mask == 255] = 0
                        # small visual for eraser
                        cv2.circle(display, cur_point, eraser_radius, (0,0,0), 2)
                    else:
                        # Drawing: either dashed or solid lines
                        color = pen_color
                        if dashed_mode:
                            # draw dashed segments both on canvas and mask
                            # draw dashed on a temporary image then merge
                            tmp_canvas = np.zeros_like(canvas)
                            tmp_mask = np.zeros_like(mask)
                            draw_dashed_line(tmp_canvas, prev_point, cur_point, color, thickness)
                            draw_dashed_line(tmp_mask, prev_point, cur_point, (255,), thickness)
                            # Where tmp_mask has lines (non-zero), update canvas & mask
                            drawn = cv2.cvtColor(tmp_mask, cv2.COLOR_GRAY2BGR)
                            drawn_pixels = tmp_mask > 0
                            canvas[drawn_pixels] = tmp_canvas[drawn_pixels]
                            mask[drawn_pixels] = 255
                        else:
                            cv2.line(canvas, prev_point, cur_point, color, thickness)
                            cv2.line(mask, prev_point, cur_point, 255, thickness)
                    prev_point = cur_point
                    drawing = True
                else:
                    prev_point = None
                    drawing = False

        else:
            # No hand detected
            last_over_swatch = None
            swatch_start_time = 0
            two_fingers_start = 0
            fist_start = 0
            prev_point = None
            drawing = False

        # Draw HUD: current mode & controls
        hud_y = FRAME_H - 70
        mode_text = f"Color: {pen_color} | Thickness: {thickness} | Eraser: {eraser_mode} | Dashed: {dashed_mode}"
        cv2.putText(display, mode_text, (10, hud_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(display, "Gestures: 4-fingers=select color | 2-fingers=toggle eraser | fist(hold)=clear",
                    (10, hud_y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)
        cv2.putText(display, "Keyboard: q=quit | c=clear | e=eraser toggle | d=dashed toggle | +/- thickness",
                    (10, hud_y+45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)

        # Show resulting frame
        cv2.imshow("CV Drawing Pad (press 'q' in this window to quit)", display)
        # cv2.imshow("Canvas", canvas) for canvas debug
        key = cv2.waitKey(1) & 0xFF

        # Keyboard controls (work only when the display window has focus)
        if key == ord('q'):
            print("Exit key pressed.")
            break
        elif key == ord('c'):
            canvas[:] = 0
            mask[:] = 0
            print("Canvas cleared (keyboard).")
        elif key == ord('e'):
            eraser_mode = not eraser_mode
            print("Eraser mode:", eraser_mode)
        elif key == ord('d'):
            dashed_mode = not dashed_mode
            print("Dashed mode:", dashed_mode)
        elif key == ord('+') or key == ord('='):
            thickness = min(MAX_THICKNESS, thickness + 1)
        elif key == ord('-') or key == ord('_'):
            thickness = max(MIN_THICKNESS, thickness - 1)
        elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
            idx = int(chr(key)) - 1
            if 0 <= idx < len(SWATCHES):
                pen_color = SWATCHES[idx]["bgr"]
                eraser_mode = False
                print(f"Selected color: {SWATCHES[idx]['name']} (keyboard)")

        # FPS (for information)
        now = time.time()
        fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
        prev_time = now
        fps_deque.append(fps)
        if len(fps_deque) > 0:
            fps_smooth = sum(fps_deque) / len(fps_deque)
            cv2.putText(display, f"FPS: {fps_smooth:.1f}", (FRAME_W - 120, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
