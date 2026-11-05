"""
Configuration loader and profile management for HandControl Next-Gen.
Persists settings into config.json with hot-reloading capability.
"""
import json
import os

DEFAULT_CONFIG = {
    "version": "2.0-nextgen",
    "general": {
        "sound_feedback": True,
        "show_hud": True,
        "hud_theme": "cyberpunk",   # "cyberpunk", "emerald", "minimal"
        "dark_canvas": False,       # Replace webcam background with dark canvas
        "num_hands": 2,             # Track up to 2 hands simultaneously
        "fps_limit": 60,
        "release_grace_s": 0.25
    },
    "air_mouse": {
        "enabled": False,           # Can be toggled on/off on the fly (hotkey 'M')
        "speed": 1.4,
        "one_euro_min_cutoff": 1.2,
        "one_euro_beta": 0.05,
        "scroll_speed": 1.2,
        "active_box": {            # Central interaction box inside camera view (percentages)
            "x_min": 0.15,
            "x_max": 0.85,
            "y_min": 0.15,
            "y_max": 0.80
        }
    },
    "gestures": {
        "THUMB_UP": {
            "action": "VOLUME_UP",
            "hold_s": 0.28,
            "repeat_s": 0.10,
            "description": "Increase system volume"
        },
        "THUMB_DOWN": {
            "action": "VOLUME_DOWN",
            "hold_s": 0.28,
            "repeat_s": 0.10,
            "description": "Decrease system volume"
        },
        "POINT_RIGHT": {
            "action": "RIGHT",
            "hold_s": 0.28,
            "repeat_s": 1.0,
            "description": "Next slide / Fast-forward"
        },
        "POINT_LEFT": {
            "action": "LEFT",
            "hold_s": 0.28,
            "repeat_s": 1.0,
            "description": "Previous slide / Rewind"
        },
        "PINCH_INDEX": {
            "action": "SPACE",
            "hold_s": 0.18,
            "repeat_s": None,
            "description": "Space (Play/Pause/Click)"
        },
        "FIST": {
            "action": "MUTE",
            "hold_s": 0.40,
            "repeat_s": None,
            "description": "Mute / Unmute audio"
        },
        "PEACE": {
            "action": "SCREENSHOT",
            "hold_s": 0.45,
            "repeat_s": None,
            "description": "Take snippet screenshot"
        },
        "ROCK_ON": {
            "action": "F11",
            "hold_s": 0.45,
            "repeat_s": None,
            "description": "Toggle Fullscreen"
        },
        "SWIPE_LEFT": {
            "action": "APP_SWITCH",
            "hold_s": 0.0,
            "repeat_s": None,
            "description": "Switch window (Alt+Tab)"
        },
        "SWIPE_RIGHT": {
            "action": "APP_SWITCH",
            "hold_s": 0.0,
            "repeat_s": None,
            "description": "Switch window"
        },
        "SWIPE_DOWN": {
            "action": "DESKTOP",
            "hold_s": 0.0,
            "repeat_s": None,
            "description": "Minimize All / Show Desktop"
        }
    }
}


def load_config(path="config.json"):
    if not os.path.exists(path):
        save_config(DEFAULT_CONFIG, path)
        return DEFAULT_CONFIG

    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            # Merge with defaults for any missing keys
            merged = DEFAULT_CONFIG.copy()
            for k, v in cfg.items():
                if isinstance(v, dict) and k in merged:
                    merged[k].update(v)
                else:
                    merged[k] = v
            return merged
    except Exception as e:
        print(f"Warning: Failed to parse {path} ({e}), using default configuration.")
        return DEFAULT_CONFIG


def save_config(cfg, path="config.json"):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4)
    except Exception as e:
        print(f"Failed to write configuration: {e}")

