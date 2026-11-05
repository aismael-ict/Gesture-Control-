"""
Hand-gesture control for a laptop, using the webcam.

    Thumbs up              -> volume up    (repeats while held)
    Thumbs down            -> volume down  (repeats while held)
    Index finger -> right  -> Right arrow  (one press, slow repeat if held)
    Index finger -> left   -> Left arrow   (one press, slow repeat if held)
    Pinch (thumb + index)  -> Space bar    (one press per pinch)

A preview window shows the mirrored camera image with the 21-point hand
skeleton drawn on it. Keys in the preview window: P = pause, Q or Esc = quit.

Hand tracking is MediaPipe's Hand Landmarker; everything runs locally.
Run:  python hand_control.py            (add --dry-run to test without key presses)
      python hand_control.py --reel     (tall 9:16 window for Reels / TikTok recordings;
                                         shows only the picture and the skeleton, no text)
"""

import argparse
import math
import os
import sys
import time
import urllib.request
from pathlib import Path

# Redirect stdout/stderr to devnull if None (e.g. in windowed/noconsole PyInstaller build)
if sys.stdout is None:
    sys.stdout = open(os.devnull, 'w')
if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w')

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

# Gesture definition: gesture_name -> (action_name, hold_seconds, repeat_seconds)
GESTURES = {
    'THUMB_UP': ('VOLUME UP', 0.3, 0.12),
    'THUMB_DOWN': ('VOLUME DOWN', 0.3, 0.12),
    'POINT_RIGHT': ('RIGHT ARROW', 0.3, 1.2),
    'POINT_LEFT': ('LEFT ARROW', 0.3, 1.2),
    'PINCH': ('SPACE', 0.2, None),
}

RELEASE_S = 0.25               # Grace period before an active gesture drops out
CURLED_BELOW_DEG = 130         # Joint angle below which a finger is considered curled
STRAIGHT_ABOVE_DEG = 145       # Joint angle above which index finger is considered straight
THUMB_OUT_RATIO = 1.05         # Ratio of tip-to-pinky vs joint-to-pinky distance for extended thumb
DIRECTION_CONE_DEG = 35        # Acceptable angular cone around 0, 90, 180, -90 degrees
MIN_INDEX_LEN = 0.6            # Minimum index finger length relative to palm size
PINCH_CLOSE = 0.2              # Normalized gap to trigger a pinch
PINCH_OPEN = 0.3               # Normalized gap to release a pinch (hysteresis)
PINCH_MIN_INDEX_BEND = 90      # Minimum index finger joint angle to allow pinch (avoids closed fist)

MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task'
MODEL_PATH = Path(__file__).with_name('hand_landmarker.task')
WINDOW = 'Hand control  (P = pause, Q = quit)'

# Landmark indices
WRIST = 0
THUMB_MCP, THUMB_IP, THUMB_TIP = 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP = 9
PINKY_MCP = 17

# Fingers defined by (knuckle, joint, tip)
FINGERS = [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]

# Skeleton bones for visualization
BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]


def _joint_angle(a, b, c):
    """Angle at point b (degrees) between the segments b->a and b->c."""
    u = a - b
    v = c - b
    cos = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)
    return math.degrees(math.acos(float(np.clip(cos, -1.0, 1.0))))


def _direction(vec):
    """'RIGHT' / 'UP' / 'LEFT' / 'DOWN' for a 2D image vector, or None if diagonal."""
    angle = math.degrees(math.atan2(-vec[1], vec[0]))
    for name, centre in (('RIGHT', 0), ('UP', 90), ('LEFT', 180), ('DOWN', -90)):
        if abs((angle - centre + 180) % 360 - 180) <= DIRECTION_CONE_DEG:
            return name
    return None


def _thumb_out(px):
    """True if the thumb sticks out from the hand instead of lying across the fist.

    Compares how far the thumb tip and the thumb's middle joint are from the
    pinky knuckle: an extended thumb has its tip clearly farther away.
    """
    tip = np.linalg.norm(px[THUMB_TIP] - px[PINKY_MCP])
    joint = np.linalg.norm(px[THUMB_IP] - px[PINKY_MCP])
    return (tip / (joint + 1e-9)) > THUMB_OUT_RATIO


def _pinching(px, index_bend, was_pinching):
    """True if the thumb and index fingertips are touching.

    Once a pinch has started, the fingertips must separate a bit further before
    it ends, so a pinch held near the limit cannot flicker into two presses.
    """
    palm = np.linalg.norm(px[MIDDLE_MCP] - px[WRIST])
    gap = np.linalg.norm(px[THUMB_TIP] - px[INDEX_TIP]) / (palm + 1e-9)
    limit = PINCH_OPEN if was_pinching else PINCH_CLOSE
    return (gap < limit) and (index_bend > PINCH_MIN_INDEX_BEND)


def classify(px, world, was_pinching=False):
    """Return a gesture name from GESTURES, or None.

    px:    (21, 2) landmarks in pixels of the mirrored image (for directions)
    world: (21, 3) landmarks in metres, hand-centred (for finger bending,
           which is unreliable in 2D when fingers point towards the camera)
    was_pinching: True if the previous frames were already a pinch
    """
    bend = [_joint_angle(world[k], world[j], world[t]) for k, j, t in FINGERS]
    curled = [b < CURLED_BELOW_DEG for b in bend]
    index_straight = bend[0] > STRAIGHT_ABOVE_DEG

    if _pinching(px, bend[0], was_pinching):
        return 'PINCH'

    if all(curled) and _thumb_out(px):
        way = _direction(px[THUMB_TIP] - px[THUMB_MCP])
        if way == 'UP':
            return 'THUMB_UP'
        if way == 'DOWN':
            return 'THUMB_DOWN'
        return None

    if index_straight and all(curled[1:]):
        finger = px[INDEX_TIP] - px[INDEX_MCP]
        palm = np.linalg.norm(px[MIDDLE_MCP] - px[WRIST])
        if np.linalg.norm(finger) < MIN_INDEX_LEN * palm:
            return None
        way = _direction(finger)
        if way == 'RIGHT':
            return 'POINT_RIGHT'
        if way == 'LEFT':
            return 'POINT_LEFT'

    return None


class Trigger:
    """Turns a per-frame gesture name into key-press events (hold, then repeat)."""

    def __init__(self):
        self.gesture = None
        self.since = 0.0
        self.last_seen = 0.0
        self.next_fire = 0.0

    def update(self, gesture, now):
        # Grace period: keep current gesture if hand tracking dropped briefly
        if gesture is None and self.gesture and (now - self.last_seen < RELEASE_S):
            return None

        if gesture != self.gesture:
            self.gesture = gesture
            self.since = now
            if gesture:
                hold_s = GESTURES[gesture][1]
                self.next_fire = now + hold_s

        if self.gesture is None:
            return None

        self.last_seen = now
        if now >= self.next_fire:
            action, _, repeat_s = GESTURES[self.gesture]
            if repeat_s is None:
                self.next_fire = math.inf
            else:
                self.next_fire = now + repeat_s
            return action

        return None

    def hold_progress(self, now):
        if self.gesture is None:
            return 0.0
        hold_s = GESTURES[self.gesture][1]
        return min(1.0, (now - self.since) / hold_s)


def fail(message):
    """Stop with an error message. The packaged app has no console, so it shows a pop-up."""
    if getattr(sys, 'frozen', False) and sys.platform == 'win32':
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, message, 'Hand control', 16)
    sys.exit(message)


def _windows_key_presser():
    """Key presses sent through Windows itself (keybd_event), with no extra library."""
    import ctypes
    user32 = ctypes.windll.user32
    extended = 1
    key_up = 2

    keys = {
        'VOLUME UP': (175, extended),    # VK_VOLUME_UP
        'VOLUME DOWN': (174, extended),  # VK_VOLUME_DOWN
        'RIGHT ARROW': (39, extended),   # VK_RIGHT
        'LEFT ARROW': (37, extended),    # VK_LEFT
        'SPACE': (32, 0),                # VK_SPACE
    }

    def press(action):
        code, flags = keys[action]
        scan = user32.MapVirtualKeyW(code, 0)
        user32.keybd_event(code, scan, flags, 0)
        user32.keybd_event(code, scan, flags | key_up, 0)

    return press


def make_key_presser(dry_run):
    """Return press(action_name). With dry_run, nothing is sent to the system."""
    if dry_run:
        return lambda action: None

    if sys.platform == 'win32':
        return _windows_key_presser()

    from pynput.keyboard import Controller, Key
    keyboard = Controller()
    keys = {
        'VOLUME UP': Key.media_volume_up,
        'VOLUME DOWN': Key.media_volume_down,
        'RIGHT ARROW': Key.right,
        'LEFT ARROW': Key.left,
        'SPACE': Key.space,
    }

    def press(action):
        keyboard.press(keys[action])
        keyboard.release(keys[action])

    return press


def ensure_model():
    """Download MediaPipe hand landmarker model if not present."""
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000:
        return

    print(f'Downloading hand model (one time) to {MODEL_PATH} ...')
    partial = MODEL_PATH.with_name(MODEL_PATH.name + '.part')
    try:
        urllib.request.urlretrieve(MODEL_URL, partial)
        partial.replace(MODEL_PATH)
    except Exception as err:
        partial.unlink(missing_ok=True)
        fail(f'Could not download the model ({err}).\nDownload it manually from:\n  {MODEL_URL}\nand save it as:\n  {MODEL_PATH}')


def _open_webcam(index):
    cap = None
    if sys.platform == 'win32':
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(index)
    return cap


def open_camera(source, hd=False):
    """Open camera by index or video file path."""
    if source.isdigit():
        cap = _open_webcam(int(source))
        if hd and cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            if not cap.read()[0]:
                cap.release()
                cap = _open_webcam(int(source))
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        fail(
            'Cannot open camera.\n\n'
            'Close other apps using the camera (Zoom, Teams, Camera app)\n'
            'and verify webcam permissions in Windows Settings.'
        )
    return cap


def reel_size(requested_height):
    """(width, height) in pixels of an exact 9:16 picture that fits on this screen."""
    height = requested_height or 960
    if sys.platform == 'win32':
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
        if not requested_height:
            try:
                height = int(ctypes.windll.user32.GetSystemMetrics(1) * 0.85)
            except Exception:
                pass
    height = max(320, (height // 16) * 16)
    return (height * 9) // 16, height


def crop_to_9x16(frame):
    height, width = frame.shape[:2]
    crop_w = int(round(height * 9 / 16))
    if crop_w <= width:
        x0 = (width - crop_w) // 2
        return frame[:, x0:x0 + crop_w]
    else:
        crop_h = int(round(width * 16 / 9))
        y0 = (height - crop_h) // 2
        return frame[y0:y0 + crop_h, :]


def draw_skeleton(frame, px, colour, size=1.0):
    pts = px.astype(int)
    dot = max(2, round(2 * size))
    line = max(5, round(5 * size))

    for a, b in BONES:
        cv2.line(frame, tuple(pts[a]), tuple(pts[b]), colour, line, cv2.LINE_AA)

    for x, y in pts:
        cv2.circle(frame, (int(x), int(y)), dot, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (int(x), int(y)), dot, colour, line, cv2.LINE_AA)


def draw_text(frame, text, y, colour=(255, 255, 255), scale=0.7):
    # Outline in black
    cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    # Foreground text
    cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 2, cv2.LINE_AA)


def self_test():
    """Verify that MediaPipe HandLandmarker initializes and runs correctly."""
    ensure_model()
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=MODEL_PATH.read_bytes()),
        num_hands=1
    )
    try:
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            blank = np.zeros((480, 640, 3), np.uint8)
            landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=blank))
        print('Self-test passed.')
    except BaseException:
        import traceback
        Path('self_test_error.txt').write_text(traceback.format_exc())
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Control volume, arrow keys and space with hand gestures.')
    parser.add_argument('--camera', default='0', help='camera index (default 0) or a video file')
    parser.add_argument('--dry-run', action='store_true', help='show gestures but do not press keys')
    parser.add_argument('--reel', action='store_true', help='tall 9:16 window (Instagram Reels / TikTok shape) for screen recording')
    parser.add_argument('--reel-height', type=int, default=0, help='height of the reel window in pixels (default: fit the screen)')
    parser.add_argument('--self-test', action='store_true', help='check that the hand model loads, then exit (no camera needed)')
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    ensure_model()
    press = make_key_presser(args.dry_run)
    reel_w, reel_h = reel_size(args.reel_height) if args.reel else (0, 0)
    cap = open_camera(args.camera, hd=args.reel)

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=MODEL_PATH.read_bytes()),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1
    )

    green = (80, 220, 80)
    orange = (0, 165, 255)
    grey = (160, 160, 160)

    trigger = Trigger()
    paused = False
    last_action = ''
    last_action_time = 0.0
    start = time.monotonic()
    last_stamp = -1

    cv2.namedWindow(WINDOW)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_TOPMOST, 1)

    if args.reel:
        cv2.moveWindow(WINDOW, 40, 0)
        print(f'Reel mode: the picture is {reel_w} x {reel_h} pixels (exactly 9:16).')

    print('Running. Click the app you want to control so it receives the key presses.')

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        was_pinching = False
        while True:
            ret, frame = cap.read()
            if not ret:
                fail('The camera is not sending a picture.\n\nClose other apps that use the camera and open its privacy shutter, then start again.')

            frame = cv2.flip(frame, 1)
            now = time.monotonic()
            stamp_ms = max(int((now - start) * 1000), last_stamp + 1)
            last_stamp = stamp_ms

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, stamp_ms)

            gesture = None
            px = None
            if result.hand_landmarks:
                marks = result.hand_landmarks[0]
                world_marks = result.hand_world_landmarks[0]
                px = np.array([[m.x * w, m.y * h] for m in marks])
                world = np.array([[m.x, m.y, m.z] for m in world_marks])

                if not paused:
                    gesture = classify(px, world, was_pinching=was_pinching)
                    was_pinching = (gesture == 'PINCH')
                    fired = trigger.update(gesture, now)
                    if fired:
                        press(fired)
                        last_action = fired
                        last_action_time = now

            if args.reel:
                frame = crop_to_9x16(frame)
                if (frame.shape[1], frame.shape[0]) != (reel_w, reel_h):
                    frame = cv2.resize(frame, (reel_w, reel_h), interpolation=cv2.INTER_AREA)
                if px is not None:
                    # In reel mode: minimal clean display with no text
                    draw_skeleton(frame, px, green if trigger.gesture else grey, size=0.6)
            else:
                if px is not None:
                    draw_skeleton(frame, px, green if trigger.gesture else grey)

                if paused:
                    draw_text(frame, 'PAUSED  (press P to resume)', 30, (0, 0, 255))
                elif trigger.gesture:
                    action_label = GESTURES[trigger.gesture][0]
                    draw_text(frame, f'{trigger.gesture} -> {action_label}', 30, orange)
                    bar = int(200 * trigger.hold_progress(now))
                    cv2.rectangle(frame, (12, 42), (212, 52), (0, 0, 0), -1)
                    cv2.rectangle(frame, (12, 42), (12 + bar, 52), orange, -1)
                else:
                    draw_text(frame, 'Show a gesture' if px is not None else 'No hand in view', 30)

                if last_action and (now - last_action_time < 1.0):
                    draw_text(frame, f'Sent: {last_action}', 70, green, 0.9)

                if args.dry_run:
                    draw_text(frame, 'DRY RUN (no keys sent)', h - 20, grey, 0.6)

            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
            elif key in (ord('p'), ord('P')):
                paused = not paused

            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

