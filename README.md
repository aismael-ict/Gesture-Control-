# 🖐️ Gesture Control Next-Gen 2.0
### Touchless AI Hand Tracking, Air Mouse & Natural Gesture Interface

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange.svg)](https://developers.google.com/mediapipe)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-green.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An advanced, low-latency, touchless computer control platform powered by Google MediaPipe's Hand Landmarker, 3D biomechanical joint geometry, and an adaptive **$1€$ (One-Euro)** tremor filter.

Turn any standard laptop webcam into a spatial computing controller: smooth touchless mouse navigation, natural air clicks, volume control, media scrubbing, slide navigation, and dynamic velocity swipes.

---

## ⚡ Key Highlights

* **🖱️ Air Mouse & Virtual Touchpad**: Sub-pixel cursor tracking using 1€ filtering. Pinch index to left-click, hold to drag-and-drop, pinch middle finger to right-click, and use two fingers to scroll.
* **🌐 3D Biomechanical Invariant Tracking**: Evaluates angles in hand-centric metric 3D space (`world_landmarks`), avoiding camera perspective distortion.
* **⚡ Dynamic Velocity Swipes**: Trajectory tracker detects rapid hand motion to switch apps (`Alt+Tab`) or minimize to desktop (`Win+D`).
* **👥 Dual-Hand Intelligence**: Tracks up to 2 hands simultaneously. Use the left hand for media/volume and the right hand for mouse navigation.
* **🔊 Non-blocking Audio Feedback**: Subtle auditory click and action tones confirm gesture execution without requiring visual confirmation.
* **🎨 Cyberpunk Neon HUD**: Real-time skeleton visualization with glowing dual-pass bones, circular radial charge dials, active reticles, and dark canvas mode.
* **⚙️ JSON Profile Management**: Customize key bindings, hold times, sensitivities, and gesture mappings on the fly via `config.json`.

---

## 📸 System Architecture

```
                                  [ Webcam Feed ]
                                         │
                                  (Mirrored 720p)
                                         ▼
                           [ MediaPipe Hand Landmarker ]
                           ┌─────────────┴─────────────┐
                           ▼                           ▼
                 2D Image Landmarks          3D Metric World Space
                 (Screen projections)        (Biomechanical angles)
                           │                           │
                           └─────────────┬─────────────┘
                                         ▼
                         [ Gesture & Motion Classifier ]
                                         │
                 ┌───────────────────────┼───────────────────────┐
                 ▼                       ▼                       ▼
          [ Air Mouse Engine ]   [ Static Gestures ]    [ Dynamic Swipes ]
          • 1€ Adaptive Filter   • Joint Angle Bends    • Velocity Tracking
          • Pinch Click & Drag   • Directional Cones    • Rapid App Switching
          • Two-Finger Scroll    • Debounce & Hold      • Desktop Peek
                 │                       │                       │
                 └───────────────────────┼───────────────────────┘
                                         ▼
                            [ Native Input Injector ]
                             (user32.dll / pynput)
                                         ▼
                            [ Windows OS Actions ]
```

---

## 🎯 Gesture Reference Table

| Gesture | Physical Pose | Action Triggered | Default Hold | Behavior |
| :--- | :--- | :--- | :--- | :--- |
| **`AIR MOUSE`** | Index finger pointing | **Cursor Move** | Instant | Smoothed with $1€$ Filter |
| **`PINCH_INDEX`** | Thumb tip + Index tip | **Left Click / Drag** | `0.18s` | Click or Drag & Drop |
| **`PINCH_MIDDLE`** | Thumb tip + Middle tip | **Right Click** | Instant | Context menu click |
| **`SCROLL`** | Index + Middle extended | **Mouse Wheel** | Instant | Move vertically to scroll |
| **`THUMB_UP`** | Fist with extended thumb up | **Volume Up** | `0.28s` | Continuous ramp while held |
| **`THUMB_DOWN`** | Fist with extended thumb down | **Volume Down** | `0.28s` | Continuous ramp while held |
| **`POINT_RIGHT`** | Index pointing right | **Right Arrow** | `0.28s` | Next Slide / Fast-forward |
| **`POINT_LEFT`** | Index pointing left | **Left Arrow** | `0.28s` | Prev Slide / Rewind |
| **`FIST`** | All fingers curled tight | **Mute Audio** | `0.40s` | Toggle Mute/Unmute |
| **`PEACE (V)`** | Index + Middle straight | **Screenshot** | `0.45s` | Snipping Tool (`Win+Shift+S`) |
| **`ROCK ON`** | Index + Pinky straight | **Fullscreen** | `0.45s` | Toggle Fullscreen (`F11`) |
| **`OPEN PALM`** | 5 fingers extended spread | **Freeze / Stop** | Instant | Neutral pose / pointer freeze |
| **`SWIPE L/R`** | Fast horizontal hand motion | **App Switcher** | Dynamic | `Alt + Tab` window switch |
| **`SWIPE DOWN`** | Fast downward hand motion | **Show Desktop** | Dynamic | `Win + D` minimize all |

---

## 🚀 Quick Start

### 1. Prerequisites & Installation

Clone the repository and install requirements:

```bash
git clone https://github.com/aismael-ict/Gesture-Control-.git
cd Gesture-Control-
pip install -r requirements.txt
```

### 2. Launching

Run the Next-Gen platform:

```bash
# Windows one-click launcher
run_nextgen.bat

# Or via Python
python hand_control_nextgen.py
```

### Optional Command-Line Flags:
* `--mouse`: Start with Air Mouse mode enabled immediately.
* `--dark`: Enable Dark Cyber Canvas mode (replaces webcam background).
* `--dry-run`: Visualize gestures and landmarks without injecting keystrokes or mouse clicks.
* `--camera <id>`: Select camera index (default `0`) or video file path.

---

## ⌨️ Interactive In-App Hotkeys

| Key | Function |
| :--- | :--- |
| **`M`** | Toggle **Air Mouse Mode** On / Off |
| **`P`** | **Pause / Resume** gesture tracking |
| **`D`** | Toggle **Dark Cyber Canvas** (dims camera feed) |
| **`H`** | Toggle **HUD Overlays** (minimal clean feed) |
| **`Q`** or **`Esc`** | Exit cleanly |

---

## 🛠️ Configuration (`config.json`)

All gesture bindings, hold durations, repeat intervals, and Air Mouse bounding boxes can be customized without writing code:

```json
{
  "general": {
    "sound_feedback": true,
    "show_hud": true,
    "hud_theme": "cyberpunk",
    "num_hands": 2
  },
  "air_mouse": {
    "enabled": false,
    "speed": 1.4,
    "one_euro_min_cutoff": 1.2,
    "one_euro_beta": 0.05
  },
  "gestures": {
    "THUMB_UP": { "action": "VOLUME_UP", "hold_s": 0.28, "repeat_s": 0.10 },
    "PEACE": { "action": "SCREENSHOT", "hold_s": 0.45, "repeat_s": null }
  }
}
```

---

## 🧪 Testing & Verification

A built-in test suite verifies filter stability, gesture classification, and trajectory detection:

```bash
python test_nextgen.py
```

---

## 🔒 Privacy & Local Processing

* **100% Offline**: All computer vision, MediaPipe inference, and input injection run entirely locally on your CPU/GPU.
* **Zero Telemetry**: No camera frames, images, or tracking coordinates are ever saved, recorded, or transmitted.

---

## 📄 License

Distributed under the MIT License. See [LICENSE](LICENSE) for more details.
