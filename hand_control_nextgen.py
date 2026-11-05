"""
=============================================================================
                  HAND CONTROL NEXT-GEN 2.0
      Next-Generation AI Hand Tracking & Gesture Control Platform
=============================================================================
Features:
  - Multi-Hand Tracking (up to 2 hands simultaneously with handedness)
  - Air Mouse & Virtual Touchpad (1€ filter jitter-smoothing, air-clicks, drag, scroll)
  - Natural Gesture Vocabulary (Volume, Media, Slides, Screenshots, Fullscreen, Mute)
  - Dynamic Motion Trajectory Gestures (Swipes for App Switch & Show Desktop)
  - Native Windows Input Engine (user32.dll direct scan-codes & mouse events)
  - Tactile Sound Feedback (subtle non-blocking synthesizer audio cues)
  - Cyberpunk Glowing HUD (neon skeleton, circular charge dials, active reticles)
  - Hot-Reloadable JSON Profiles & Config
=============================================================================
"""

import bootstrap # Mounts bundled PyInstaller environment if running in standard Python
import argparse
import os
import sys
import time
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from config import load_config, save_config
from filters import PointFilter2D, TrajectoryTracker
from gestures import (
    GestureClassifier,
    WRIST,
    THUMB_TIP,
    INDEX_TIP,
    MIDDLE_TIP,
    MIDDLE_MCP,
)
from hud import CyberHUD
from input_controller import InputController

# Window Title
WINDOW_TITLE = "Hand Control Next-Gen  [M: Air Mouse | P: Pause | D: Dark Mode | Q: Quit]"


class GestureTrigger:
    """Manages hold duration, auto-repeat intervals, and dropout grace periods."""

    def __init__(self, gesture_configs, release_grace_s=0.25):
        self.configs = gesture_configs
        self.release_grace_s = release_grace_s

        self.current_gesture = None
        self.since = 0.0
        self.last_seen = 0.0
        self.next_fire = 0.0

    def update(self, gesture, now):
        # Grace period for temporary tracking dropout
        if gesture is None and self.current_gesture and (now - self.last_seen < self.release_grace_s):
            return None

        # Gesture changed
        if gesture != self.current_gesture:
            self.current_gesture = gesture
            self.since = now
            if gesture and gesture in self.configs:
                hold_s = self.configs[gesture]["hold_s"]
                self.next_fire = now + hold_s

        if not self.current_gesture or self.current_gesture not in self.configs:
            return None

        self.last_seen = now
        if now >= self.next_fire:
            cfg = self.configs[self.current_gesture]
            action = cfg["action"]
            repeat_s = cfg.get("repeat_s")
            if repeat_s is None:
                self.next_fire = float('inf') # Single shot
            else:
                self.next_fire = now + repeat_s
            return action

        return None

    def progress(self, now):
        if not self.current_gesture or self.current_gesture not in self.configs:
            return 0.0
        hold_s = self.configs[self.current_gesture]["hold_s"]
        if hold_s <= 0.001:
            return 1.0
        return min(1.0, max(0.0, (now - self.since) / hold_s))


def execute_action(action_name, input_ctrl):
    """Executes mapped system actions via the InputController."""
    if not action_name:
        return

    if action_name == "VOLUME_UP":
        input_ctrl.press_key('VOLUME_UP')
    elif action_name == "VOLUME_DOWN":
        input_ctrl.press_key('VOLUME_DOWN')
    elif action_name == "MUTE":
        input_ctrl.press_key('MUTE')
    elif action_name == "RIGHT":
        input_ctrl.press_key('RIGHT')
    elif action_name == "LEFT":
        input_ctrl.press_key('LEFT')
    elif action_name == "SPACE":
        input_ctrl.press_key('SPACE')
    elif action_name == "SCREENSHOT":
        input_ctrl.action_screenshot()
    elif action_name == "F11":
        input_ctrl.action_toggle_fullscreen()
    elif action_name == "APP_SWITCH":
        input_ctrl.action_switch_app_next()
    elif action_name == "DESKTOP":
        input_ctrl.action_show_desktop()


def open_camera_stream(camera_idx=0, width=1280, height=720):
    """Opens webcam with DirectShow for low latency and sets 720p HD resolution."""
    cap = None
    if sys.platform == 'win32':
        cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(camera_idx)

    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return cap


def main():
    parser = argparse.ArgumentParser(description="Hand Control Next-Gen AI System")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--dry-run", action="store_true", help="Visualize without sending keystrokes or mouse clicks")
    parser.add_argument("--mouse", action="store_true", help="Start with Air Mouse enabled immediately")
    parser.add_argument("--dark", action="store_true", help="Start with Dark Cyber canvas background")
    args = parser.parse_args()

    cfg = load_config()
    air_mouse_enabled = args.mouse or cfg["air_mouse"]["enabled"]
    dark_canvas_enabled = args.dark or cfg["general"]["dark_canvas"]
    show_hud = cfg["general"]["show_hud"]

    # Locate Model
    model_path = Path(__file__).with_name("hand_landmarker.task")
    if not model_path.exists():
        internal_model = Path(__file__).parent / "_internal" / "hand_landmarker.task"
        if internal_model.exists():
            model_path = internal_model
        else:
            print("Error: hand_landmarker.task model not found.")
            sys.exit(1)

    # Initialize Modules
    input_ctrl = InputController(dry_run=args.dry_run, sound_enabled=cfg["general"]["sound_feedback"])
    classifier = GestureClassifier()
    hud = CyberHUD(theme=cfg["general"]["hud_theme"])
    trigger = GestureTrigger(cfg["gestures"], release_grace_s=cfg["general"]["release_grace_s"])

    # Air Mouse smoothing filters & trajectory
    mouse_filter = PointFilter2D(
        min_cutoff=cfg["air_mouse"]["one_euro_min_cutoff"],
        beta=cfg["air_mouse"]["one_euro_beta"]
    )
    trajectory = TrajectoryTracker()

    # MediaPipe HandLandmarker Setup
    num_hands = min(2, max(1, cfg["general"].get("num_hands", 2)))
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=model_path.read_bytes()),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=num_hands
    )

    cap = open_camera_stream(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}. Check if another application is using it.")
        sys.exit(1)

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_TITLE, cv2.WND_PROP_TOPMOST, 1)

    print("\n" + "=" * 60)
    print(" HAND CONTROL NEXT-GEN 2.0 INITIALIZED")
    print("=" * 60)
    print(" -> Hotkeys in preview window:")
    print("    [M] Toggle Air Mouse Mode")
    print("    [P] Pause / Resume Gesture Engine")
    print("    [D] Toggle Dark Cyber Canvas")
    print("    [H] Toggle HUD Overlay")
    print("    [Q] or [Esc] Quit")
    print("=" * 60 + "\n")

    paused = False
    last_action_text = ""
    last_action_time = 0.0
    prev_scroll_y = None
    pinch_drag_active = False

    start_time = time.monotonic()
    last_stamp_ms = -1
    fps_time = start_time
    fps_frames = 0
    current_fps = 30.0

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera feed lost or interrupted.")
                break

            # Mirror image horizontally for natural interaction
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            now = time.monotonic()

            # FPS calculation
            fps_frames += 1
            if now - fps_time >= 0.5:
                current_fps = fps_frames / (now - fps_time)
                fps_frames = 0
                fps_time = now

            # Video timestamp in ms
            stamp_ms = max(int((now - start_time) * 1000), last_stamp_ms + 1)
            last_stamp_ms = stamp_ms

            # Optional Dark Cyber Canvas (pure aesthetic, hides cluttered room background)
            display_frame = frame
            if dark_canvas_enabled:
                dark_canvas = np.full_like(frame, (18, 16, 24))
                # Soft blend camera image behind dark canvas
                display_frame = cv2.addWeighted(frame, 0.15, dark_canvas, 0.85, 0)

            # Inference
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection = landmarker.detect_for_video(mp_img, stamp_ms)

            num_detected_hands = len(detection.hand_landmarks) if detection.hand_landmarks else 0
            primary_gesture = None
            primary_hand_center = None

            if num_detected_hands > 0 and not paused:
                # Iterate detected hands (Primary hand for Air Mouse, secondary for gestures)
                for idx, (lms, world_lms) in enumerate(zip(detection.hand_landmarks, detection.hand_world_landmarks)):
                    handedness = "Right"
                    if detection.handedness and idx < len(detection.handedness):
                        handedness = detection.handedness[idx][0].category_name

                    px = np.array([[m.x * w, m.y * h] for m in lms])
                    world = np.array([[m.x, m.y, m.z] for m in world_lms])

                    analysis = classifier.classify_hand(px, world, handedness)
                    pose = analysis["pose"]
                    primary_hand_center = px[MIDDLE_MCP]

                    # Visual Skeleton
                    if show_hud:
                        skeleton_color = hud.NEON_CYAN if handedness == "Right" else hud.NEON_MAGENTA
                        hud.draw_skeleton(display_frame, px, colour=skeleton_color, glow=True)

                    # Dynamic Trajectory Tracking (Swipes) for primary hand
                    if idx == 0:
                        norm_palm_x, norm_palm_y = px[MIDDLE_MCP][0] / w, px[MIDDLE_MCP][1] / h
                        trajectory.add(norm_palm_x, norm_palm_y, now)
                        swipe = trajectory.detect_swipe()
                        if swipe:
                            pose = swipe # Dynamic swipe overrides static pose

                        primary_gesture = pose

                    # ----------------------------------------------------
                    # AIR MOUSE MODE
                    # ----------------------------------------------------
                    if air_mouse_enabled and idx == 0:
                        box = cfg["air_mouse"]["active_box"]
                        idx_tip = px[INDEX_TIP]
                        norm_x = (idx_tip[0] / w - box["x_min"]) / max(1e-4, box["x_max"] - box["x_min"])
                        norm_y = (idx_tip[1] / h - box["y_min"]) / max(1e-4, box["y_max"] - box["y_min"])

                        # Clamp and scale with sensitivity
                        norm_x = max(0.0, min(1.0, norm_x))
                        norm_y = max(0.0, min(1.0, norm_y))

                        # Apply 1€ Filter
                        smooth_norm_x, smooth_norm_y = mouse_filter.filter(norm_x, norm_y, timestamp=now)
                        target_screen_x = smooth_norm_x * input_ctrl.screen_w
                        target_screen_y = smooth_norm_y * input_ctrl.screen_h

                        input_ctrl.move_mouse(target_screen_x, target_screen_y)

                        # Air Click (Thumb + Index Pinch)
                        is_pinching_idx = analysis["is_pinch_index"]
                        is_pinching_mid = analysis["is_pinch_middle"]

                        if is_pinching_idx:
                            if not pinch_drag_active:
                                input_ctrl.mouse_left_down()
                                pinch_drag_active = True
                        else:
                            if pinch_drag_active:
                                input_ctrl.mouse_left_up()
                                pinch_drag_active = False

                        # Air Right Click (Thumb + Middle Pinch)
                        if is_pinching_mid and not is_pinching_idx:
                            input_ctrl.mouse_right_click()

                        # Two-Finger Vertical Scroll
                        if pose == 'TWO_FINGER_POINT':
                            curr_y = idx_tip[1]
                            if prev_scroll_y is not None:
                                dy = prev_scroll_y - curr_y
                                if abs(dy) > 4:
                                    clicks = 1 if dy > 0 else -1
                                    input_ctrl.mouse_scroll(clicks)
                            prev_scroll_y = curr_y
                        else:
                            prev_scroll_y = None

                        if show_hud:
                            hud.draw_air_mouse_reticle(
                                display_frame,
                                idx_tip,
                                is_clicking=is_pinching_idx,
                                is_right_clicking=is_pinching_mid
                            )

            else:
                # No hands or paused: release any mouse drag and reset filters
                if pinch_drag_active:
                    input_ctrl.mouse_left_up()
                    pinch_drag_active = False
                prev_scroll_y = None
                mouse_filter.reset()
                trajectory.reset()

            # ----------------------------------------------------
            # GESTURE TRIGGER & ACTION DISPATCH
            # ----------------------------------------------------
            if not paused and not (air_mouse_enabled and primary_gesture in ('PINCH_INDEX', 'TWO_FINGER_POINT')):
                fired_action = trigger.update(primary_gesture, now)
                if fired_action:
                    execute_action(fired_action, input_ctrl)
                    last_action_text = f"{primary_gesture} -> {fired_action}"
                    last_action_time = now

            # ----------------------------------------------------
            # CYBERPUNK HUD RENDERING
            # ----------------------------------------------------
            if show_hud:
                # Active Air Mouse Bounding Box
                if air_mouse_enabled:
                    hud.draw_active_box(display_frame, cfg["air_mouse"]["active_box"])

                # Circular Progress Ring on Active Hand
                if trigger.current_gesture and primary_hand_center is not None and not paused:
                    progress = trigger.progress(now)
                    hud.draw_circular_progress(display_frame, primary_hand_center, progress)

                # Top Glassmorphism Status Bar
                hud.draw_header_hud(
                    display_frame,
                    fps=current_fps,
                    air_mouse_active=air_mouse_enabled,
                    hands_count=num_detected_hands,
                    paused=paused
                )

                # Bottom Action Confirmation Banner
                if last_action_text:
                    hud.draw_action_banner(display_frame, last_action_text, now - last_action_time)

                # Hotkey Legend
                hud.draw_footer_hotkeys(display_frame)

            # Display
            cv2.imshow(WINDOW_TITLE, display_frame)

            # Key Handling
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27): # Quit
                break
            elif key in (ord('p'), ord('P')):   # Pause
                paused = not paused
                input_ctrl.play_sound('toggle_off' if paused else 'toggle_on')
            elif key in (ord('m'), ord('M')):   # Toggle Air Mouse
                air_mouse_enabled = not air_mouse_enabled
                input_ctrl.play_sound('toggle_on' if air_mouse_enabled else 'toggle_off')
            elif key in (ord('d'), ord('D')):   # Toggle Dark Canvas
                dark_canvas_enabled = not dark_canvas_enabled
            elif key in (ord('h'), ord('H')):   # Toggle HUD
                show_hud = not show_hud

            # Close button clicked
            if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

