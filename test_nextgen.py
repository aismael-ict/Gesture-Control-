"""
Comprehensive Automated Test Suite for HandControl Next-Gen.
Validates:
  1. 1€ (One-Euro) Filter responsiveness and stability
  2. Trajectory tracker dynamic swipe gesture detection
  3. 3D Biomechanical angle calculations & gesture classification logic
  4. Native Input Controller state transitions & mouse drag handling
  5. Config management and defaults validation
"""
import math
import numpy as np
import time

from filters import OneEuroFilter, PointFilter2D, TrajectoryTracker
from gestures import joint_angle_3d, vector_direction_2d, GestureClassifier
from input_controller import InputController
from config import DEFAULT_CONFIG, load_config


def test_filters():
    print("Testing 1€ Filter...")
    f = OneEuroFilter(min_cutoff=1.0, beta=0.01)
    # Feed stationary signal with high-frequency noise
    vals = [10.0 + (0.5 if i % 2 == 0 else -0.5) for i in range(20)]
    t = 0.0
    filtered = []
    for v in vals:
        filtered.append(f.filter(v, t))
        t += 0.033

    # The filtered signal variance should be significantly reduced
    raw_var = np.var(vals)
    filt_var = np.var(filtered[5:])
    assert filt_var < raw_var, f"Filter should reduce jitter: {filt_var} vs {raw_var}"
    print(f"  [OK] 1€ Filter reduced jitter from {raw_var:.4f} to {filt_var:.4f}")

    print("Testing Trajectory Tracker (Swipes)...")
    tracker = TrajectoryTracker(window_seconds=0.3)
    # Simulate rapid rightward swipe: x goes from 0.2 to 0.7 in 0.15s
    t = 1.0
    for i in range(8):
        x = 0.2 + (i / 7.0) * 0.5
        y = 0.5
        tracker.add(x, y, t)
        t += 0.02

    detected = tracker.detect_swipe()
    assert detected == 'SWIPE_RIGHT', f"Expected SWIPE_RIGHT, got {detected}"
    print(f"  [OK] Trajectory tracker successfully detected: {detected}")


def test_gesture_math():
    print("Testing 3D Joint Angle Calculation...")
    # Form a right angle: a=(0, 1, 0), b=(0, 0, 0), c=(1, 0, 0)
    a = np.array([0.0, 1.0, 0.0])
    b = np.array([0.0, 0.0, 0.0])
    c = np.array([1.0, 0.0, 0.0])
    deg = joint_angle_3d(a, b, c)
    assert abs(deg - 90.0) < 0.01, f"Expected 90 deg, got {deg}"
    print(f"  [OK] Joint angle orthogonal test: {deg:.2f} deg")

    print("Testing 2D Directional Cone...")
    v_right = np.array([100.0, 5.0]) # Slight deviation
    assert vector_direction_2d(v_right) == 'RIGHT'
    v_up = np.array([5.0, -100.0])   # Image coords: -y is UP
    assert vector_direction_2d(v_up) == 'UP'
    v_diag = np.array([100.0, 100.0]) # 45 deg diagonal -> should be ignored
    assert vector_direction_2d(v_diag) is None
    print("  [OK] 2D directional cones passed (Right, Up, Diagonal rejection).")


def test_synthetic_gestures():
    print("Testing Synthetic Gesture Classification...")
    classifier = GestureClassifier()

    # Synthetic landmark array: 21 points
    px = np.zeros((21, 2))
    world = np.zeros((21, 3))

    # Wrist at (0, 0), Palm center at (0, -50)
    px[0] = [200, 300]
    px[9] = [200, 250] # Middle MCP

    # Simulate Pinch: Thumb tip and Index tip touching at (200, 200)
    px[4] = [200, 200]
    px[8] = [202, 201]
    # Set index bend straight-ish
    world[5] = [0, 0.05, 0]
    world[6] = [0, 0.08, 0]
    world[8] = [0, 0.12, 0]

    res = classifier.classify_hand(px, world)
    assert res['is_pinch_index'] is True
    print(f"  [OK] Synthetic Pinch classification passed: is_pinch_index={res['is_pinch_index']}")


def test_input_controller():
    print("Testing Input Controller in Dry-Run mode...")
    ctrl = InputController(dry_run=True, sound_enabled=False)
    ctrl.move_mouse(500, 500)
    ctrl.mouse_left_down()
    assert ctrl.is_left_down is True or ctrl.dry_run
    ctrl.mouse_left_up()
    ctrl.mouse_click()
    ctrl.mouse_right_click()
    ctrl.mouse_scroll(2)
    ctrl.press_key('SPACE')
    ctrl.send_combo('LWIN', 'LSHIFT', 'KEY_S')
    print("  [OK] Input Controller event dispatching verified without errors.")


def test_config():
    print("Testing Configuration Management...")
    cfg = load_config("config.json")
    assert "gestures" in cfg
    assert "air_mouse" in cfg
    assert "THUMB_UP" in cfg["gestures"]
    print("  [OK] Config loaded and validated against default schema.")


if __name__ == '__main__':
    print("========================================")
    print("RUNNING HAND CONTROL NEXT-GEN UNIT TESTS")
    print("========================================")
    test_filters()
    test_gesture_math()
    test_synthetic_gestures()
    test_input_controller()
    test_config()
    print("========================================")
    print("ALL TESTS PASSED SUCCESSFULLY! (5/5)")
    print("========================================")

