import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

# --- Configuration ---
CAMERA_ID = 0
FRAME_WIDTH = 500
FRAME_HEIGHT = 500
MAX_FPS_AVG = 10  # moving average window for FPS smoothing

# Mediapipe init
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,       # video stream
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

# OpenCV capture
cap = cv2.VideoCapture(CAMERA_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# FPS smoothing
fps_deque = deque(maxlen=MAX_FPS_AVG)
prev_time = time.time()

def landmarks_to_numpy(landmark_list, image_w, image_h):
    """Convert mediapipe landmark list to Nx2 numpy array in pixel coords."""
    pts = []
    for lm in landmark_list.landmark:
        x_px = int(lm.x * image_w)
        y_px = int(lm.y * image_h)
        pts.append((x_px, y_px))
    return np.array(pts, dtype=np.int32)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting.")
            break

        # Flip for mirror-like view (optional)
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Convert BGR to RGB for mediapipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = hands.process(rgb)
        rgb.flags.writeable = True

        # Draw and compute mask/contour for each detected hand
        mask = np.zeros((h, w), dtype=np.uint8)  # single channel mask
        if result.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                # Draw landmarks and connections nicely
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Convert landmarks to numpy points and build convex hull
                pts = landmarks_to_numpy(hand_landmarks, w, h)
                if pts.shape[0] >= 3:
                    hull = cv2.convexHull(pts)

                    # Fill hull on mask
                    cv2.fillConvexPoly(mask, hull, 255)

                    # Draw hull contour on frame
                    cv2.polylines(frame, [hull], isClosed=True, color=(0, 255, 255), thickness=2)

                    # Bounding box
                    x, y, bw, bh = cv2.boundingRect(hull)
                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 0, 0), 2)

                    # Handedness label (Left/Right)
                    label = handedness.classification[0].label if handedness.classification else "Hand"
                    score = handedness.classification[0].score if handedness.classification else 0.0
                    cv2.putText(frame, f"{label} {score:.2f}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

        # Optionally: show mask as overlay (semi-transparent)
        colored_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        overlay = frame.copy()
        overlay[mask == 255] = (0, 128, 0)  # color region where hand mask exists
        alpha = 0.35
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Compute and display FPS (smoothed)
        now = time.time()
        dt = now - prev_time
        prev_time = now
        fps = 1.0 / dt if dt > 0 else 0.0
        fps_deque.append(fps)
        fps_smooth = sum(fps_deque) / len(fps_deque)
        cv2.putText(frame, f"FPS: {fps_smooth:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        # Show windows
        cv2.imshow("Hand Detection", frame)
        # Also show mask window if you want to debug 
        #cv2.imshow("Hand Mask (binary)", mask)
        # wrote this for fun 
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exit key pressed. Closing...")
            break


finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
