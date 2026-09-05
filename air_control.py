"""
AirControl v4 — Volume, Brightness, Cursor & Slide Tanpa Sentuh
Python + MediaPipe (Tasks API)
==========================================================
GANTI MODE:
  - Pinch JEMPOL + KELINGKING -> cycle mode: Volume -> Brightness -> Cursor -> Slide -> ulang

DI DALAM MODE:
  - VOLUME / BRIGHTNESS : jarak jempol-telunjuk = atur nilai
  - CURSOR:
      * ujung telunjuk       -> gerakkan kursor
      * pinch jempol+telunjuk-> klik kiri (kalau sambil gerak = DRAG)
      * pinch jempol+tengah  -> klik kanan
      * pinch jempol+manis, TAHAN + gerak tangan naik/turun -> SCROLL
  - SLIDE:
      * gerakkan SELURUH tangan cepat ke kiri/kanan -> ganti slide presentasi

FITUR GLOBAL (aktif di mode manapun):
  - Kepalkan tangan (fist) -> ambil SCREENSHOT otomatis, disimpan di folder "airsnaps"

CARA PAKAI
----------
1. Install dependency (sekali saja):
   python -m pip install opencv-python mediapipe numpy pycaw comtypes screen-brightness-control pyautogui

2. Jalankan:
   python air_control.py

   Saat pertama kali dijalankan, script otomatis mendownload file model
   "hand_landmarker.task" (~8 MB). Pastikan internet nyala sekali saja.

3. Tekan 'q' untuk keluar.
"""

import os
import urllib.request
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp
import pyautogui
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from system_control import VolumeController, BrightnessController

pyautogui.FAILSAFE = False

# ============== KONFIGURASI (silakan diubah-ubah) ==============
BAR_COLOR = (255, 60, 220)
CURSOR_COLOR = (60, 230, 255)
SLIDE_COLOR = (120, 255, 120)
BG_DARKEN = 0.35
MAX_HANDS = 1

PINCH_MIN_NORM = 0.15
PINCH_MAX_NORM = 1.3
ACTION_PINCH_NORM = 0.35

VOLUME_UPDATE_EVERY_N_FRAMES = 3
CURSOR_MARGIN = 80
CURSOR_SMOOTHING = 0.35
SCROLL_SENSITIVITY = 0.4

SWIPE_THRESHOLD_PX = 80
SWIPE_FRAME_WINDOW = 8
SWIPE_COOLDOWN_FRAMES = 20
SWIPE_FLASH_FRAMES = 15

SCREENSHOT_DIR = "airsnaps"
SCREENSHOT_FLASH_FRAMES = 15

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

MODE_NAMES = ["VOLUME", "BRIGHTNESS", "CURSOR", "SLIDE"]
MODE_COLORS = [BAR_COLOR, BAR_COLOR, CURSOR_COLOR, SLIDE_COLOR]


def ensure_model_downloaded():
    if not os.path.exists(MODEL_PATH):
        print("Model belum ada, mendownload hand_landmarker.task (sekali saja)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download selesai.")


def dist(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def map_range(val, in_min, in_max, out_min, out_max):
    val = max(in_min, min(in_max, val))
    return out_min + (val - in_min) / (in_max - in_min) * (out_max - out_min)


def pinch_to_percent(pinch_norm):
    pct = (pinch_norm - PINCH_MIN_NORM) / (PINCH_MAX_NORM - PINCH_MIN_NORM) * 100
    return max(0, min(100, pct))


def is_curled(pts, tip_idx, pip_idx):
    return pts[tip_idx][1] > pts[pip_idx][1]


def is_fist(pts):
    return (is_curled(pts, 8, 6) and is_curled(pts, 12, 10) and
            is_curled(pts, 16, 14) and is_curled(pts, 20, 18))


class SwipeDetector:
    def __init__(self):
        self.history = deque(maxlen=SWIPE_FRAME_WINDOW)
        self.cooldown = 0

    def update(self, wrist_x):
        event = None
        if self.cooldown > 0:
            self.cooldown -= 1
        self.history.append(wrist_x)
        if len(self.history) == SWIPE_FRAME_WINDOW and self.cooldown == 0:
            delta = self.history[-1] - self.history[0]
            if delta > SWIPE_THRESHOLD_PX:
                event = "next"
                self.cooldown = SWIPE_COOLDOWN_FRAMES
                self.history.clear()
            elif delta < -SWIPE_THRESHOLD_PX:
                event = "prev"
                self.cooldown = SWIPE_COOLDOWN_FRAMES
                self.history.clear()
        return event

    def reset(self):
        self.history.clear()


def draw_hand_dots(frame, landmarks_px, color):
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
    ]
    for a, b in connections:
        cv2.line(frame, landmarks_px[a], landmarks_px[b], color, 1, cv2.LINE_AA)
    for (x, y) in landmarks_px:
        cv2.circle(frame, (x, y), 3, (255, 255, 255), -1, cv2.LINE_AA)
    return frame


def draw_bar(frame, percent, label, color):
    h, w = frame.shape[:2]
    bar_x, bar_y, bar_w, bar_h = 40, 100, 40, h - 200
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), 2)
    fill_h = int(bar_h * (percent / 100))
    cv2.rectangle(frame, (bar_x, bar_y + bar_h - fill_h), (bar_x + bar_w, bar_y + bar_h), color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)
    cv2.putText(frame, f"{int(percent)}%", (bar_x - 5, bar_y + bar_h + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, label, (bar_x - 15, bar_y - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return frame


def draw_mode_hint(frame, current_mode):
    h, w = frame.shape[:2]
    y = 30
    for i, name in enumerate(MODE_NAMES):
        active = (i == current_mode)
        color = MODE_COLORS[i] if active else (120, 120, 120)
        cv2.putText(frame, name, (w - 180, y + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2 if active else 1, cv2.LINE_AA)
    cv2.putText(frame, "Pinch jempol+kelingking = ganti mode", (w - 340, y + 4 * 26 + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
    return frame


def take_screenshot():
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    filename = datetime.now().strftime("snap_%Y%m%d_%H%M%S.png")
    path = os.path.join(SCREENSHOT_DIR, filename)
    try:
        img = pyautogui.screenshot()
        img.save(path)
        print(f"[Screenshot] Disimpan: {path}")
        return True
    except Exception as e:
        print(f"[Screenshot] Gagal: {e}")
        return False


def main():
    ensure_model_downloaded()

    base_options = BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=MAX_HANDS,
        running_mode=vision.RunningMode.VIDEO,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    volume_ctrl = VolumeController()
    brightness_ctrl = BrightnessController()
    swipe_detector = SwipeDetector()

    screen_w, screen_h = pyautogui.size()
    smooth_x, smooth_y = screen_w / 2, screen_h / 2

    was_pinky_pinching = False
    was_index_pinching = False
    was_middle_pinching = False
    was_ring_pinching = False
    was_fist = False
    last_scroll_y = None

    current_mode = 0
    swipe_flash, swipe_flash_text = 0, ""
    snap_flash = 0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Tidak bisa mengakses webcam. Pastikan kamera tersambung & tidak dipakai aplikasi lain.")
        return

    frame_timestamp_ms = 0
    frame_counter = 0
    print("AirControl siap. Pinch jempol+kelingking=ganti mode. Kepal=screenshot. 'q'=keluar.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        output = (frame.astype(np.float32) * BG_DARKEN).astype(np.uint8)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        frame_timestamp_ms += 33
        frame_counter += 1
        result = detector.detect_for_video(mp_image, frame_timestamp_ms)

        current_percent = (volume_ctrl.get_volume_percent() if current_mode == 0
                            else brightness_ctrl.get_brightness_percent() if current_mode == 1 else 0)

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
            color = MODE_COLORS[current_mode]
            output = draw_hand_dots(output, pts, color)

            wrist, thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip, middle_mcp = \
                pts[0], pts[4], pts[8], pts[12], pts[16], pts[20], pts[9]
            hand_size = dist(wrist, middle_mcp)

            index_norm = dist(thumb_tip, index_tip) / hand_size if hand_size > 0 else 999
            middle_norm = dist(thumb_tip, middle_tip) / hand_size if hand_size > 0 else 999
            ring_norm = dist(thumb_tip, ring_tip) / hand_size if hand_size > 0 else 999
            pinky_norm = dist(thumb_tip, pinky_tip) / hand_size if hand_size > 0 else 999

            # --- GLOBAL: fist -> screenshot ---
            fist_now = is_fist(pts)
            if fist_now and not was_fist:
                if take_screenshot():
                    snap_flash = SCREENSHOT_FLASH_FRAMES
            was_fist = fist_now

            # --- GLOBAL: pinch jempol+kelingking -> ganti mode ---
            is_pinky_pinching = pinky_norm < ACTION_PINCH_NORM
            if is_pinky_pinching and not was_pinky_pinching:
                current_mode = (current_mode + 1) % len(MODE_NAMES)
                swipe_detector.reset()
                if was_index_pinching:
                    try:
                        pyautogui.mouseUp()
                    except Exception:
                        pass
                was_index_pinching = False
                was_middle_pinching = False
                was_ring_pinching = False
                last_scroll_y = None
            was_pinky_pinching = is_pinky_pinching

            if current_mode == 0:  # VOLUME
                current_percent = pinch_to_percent(index_norm)
                if frame_counter % VOLUME_UPDATE_EVERY_N_FRAMES == 0:
                    volume_ctrl.nudge_to_percent(current_percent)
                cv2.line(output, thumb_tip, index_tip, (255, 255, 255), 2, cv2.LINE_AA)

            elif current_mode == 1:  # BRIGHTNESS
                current_percent = pinch_to_percent(index_norm)
                brightness_ctrl.set_brightness_percent(current_percent)
                cv2.line(output, thumb_tip, index_tip, (255, 255, 255), 2, cv2.LINE_AA)

            elif current_mode == 2:  # CURSOR
                target_x = map_range(index_tip[0], CURSOR_MARGIN, w - CURSOR_MARGIN, 0, screen_w)
                target_y = map_range(index_tip[1], CURSOR_MARGIN, h - CURSOR_MARGIN, 0, screen_h)
                smooth_x += (target_x - smooth_x) * CURSOR_SMOOTHING
                smooth_y += (target_y - smooth_y) * CURSOR_SMOOTHING
                try:
                    pyautogui.moveTo(smooth_x, smooth_y)
                except Exception:
                    pass

                # klik kiri / DRAG: mouseDown saat mulai pinch, mouseUp saat lepas
                is_index_pinching = index_norm < ACTION_PINCH_NORM
                if is_index_pinching and not was_index_pinching:
                    try:
                        pyautogui.mouseDown(button="left")
                    except Exception:
                        pass
                elif not is_index_pinching and was_index_pinching:
                    try:
                        pyautogui.mouseUp(button="left")
                    except Exception:
                        pass
                was_index_pinching = is_index_pinching

                # klik kanan (edge-triggered, sekali per pinch)
                is_middle_pinching = middle_norm < ACTION_PINCH_NORM
                if is_middle_pinching and not was_middle_pinching:
                    try:
                        pyautogui.click(button="right")
                    except Exception:
                        pass
                was_middle_pinching = is_middle_pinching

                # scroll: pinch jempol+manis, TAHAN + gerak naik/turun
                is_ring_pinching = ring_norm < ACTION_PINCH_NORM
                if is_ring_pinching:
                    if last_scroll_y is not None:
                        delta_y = index_tip[1] - last_scroll_y
                        scroll_amount = int(-delta_y * SCROLL_SENSITIVITY)
                        if scroll_amount != 0:
                            try:
                                pyautogui.scroll(scroll_amount)
                            except Exception:
                                pass
                    last_scroll_y = index_tip[1]
                else:
                    last_scroll_y = None
                was_ring_pinching = is_ring_pinching

                click_color = (0, 255, 0) if (is_index_pinching or is_middle_pinching) else CURSOR_COLOR
                cv2.circle(output, index_tip, 10, click_color, -1, cv2.LINE_AA)
                if is_ring_pinching:
                    cv2.putText(output, "SCROLL", (index_tip[0] + 15, index_tip[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, SLIDE_COLOR, 2, cv2.LINE_AA)

            else:  # SLIDE
                event = swipe_detector.update(wrist[0])
                if event == "next":
                    pyautogui.press("right")
                    swipe_flash, swipe_flash_text = SWIPE_FLASH_FRAMES, "NEXT SLIDE >>"
                elif event == "prev":
                    pyautogui.press("left")
                    swipe_flash, swipe_flash_text = SWIPE_FLASH_FRAMES, "<< PREV SLIDE"
        else:
            was_pinky_pinching = False
            was_middle_pinching = False
            was_ring_pinching = False
            was_fist = False
            last_scroll_y = None
            if was_index_pinching:
                try:
                    pyautogui.mouseUp(button="left")
                except Exception:
                    pass
            was_index_pinching = False

        if current_mode in (0, 1):
            output = draw_bar(output, current_percent, MODE_NAMES[current_mode], BAR_COLOR)
        elif current_mode == 2:
            cv2.putText(output, "CURSOR: jempol+telunjuk=klik/drag, +tengah=klik kanan, +manis(tahan)=scroll",
                        (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, CURSOR_COLOR, 2, cv2.LINE_AA)
        else:
            cv2.putText(output, "SLIDE: gerakkan tangan cepat ke kiri/kanan",
                        (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, SLIDE_COLOR, 2, cv2.LINE_AA)

        if swipe_flash > 0:
            swipe_flash -= 1
            cv2.putText(output, swipe_flash_text, (w // 2 - 150, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, SLIDE_COLOR, 3, cv2.LINE_AA)

        if snap_flash > 0:
            snap_flash -= 1
            cv2.rectangle(output, (0, 0), (w, h), (255, 255, 255), 8)
            cv2.putText(output, "SCREENSHOT DISIMPAN", (w // 2 - 170, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        output = draw_mode_hint(output, current_mode)
        cv2.putText(output, "'q'=Keluar", (15, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow("AirControl v4 - Python + MediaPipe", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()