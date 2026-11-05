"""
Native input injection and audio feedback engine.
Provides low-level Windows API integration for mouse movement, clicks, scrolling,
keystrokes, hotkey combinations, and tactile audio feedback.
"""
import ctypes
import os
import sys
import threading
import time

try:
    import winsound
except ImportError:
    winsound = None


class InputController:
    def __init__(self, dry_run=False, sound_enabled=True):
        self.dry_run = dry_run
        self.sound_enabled = sound_enabled and (winsound is not None)
        self.is_windows = sys.platform == 'win32'

        if self.is_windows:
            self.user32 = ctypes.windll.user32
            # Set process DPI awareness for precise screen coordinates
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2) # Per-monitor DPI aware
            except Exception:
                try:
                    ctypes.windll.user32.SetProcessDPIAware()
                except Exception:
                    pass
            self.screen_w = self.user32.GetSystemMetrics(0)
            self.screen_h = self.user32.GetSystemMetrics(1)
        else:
            self.screen_w = 1920
            self.screen_h = 1080

        # Mouse button states
        self.is_left_down = False
        self.is_right_down = False

        # Virtual Key Codes
        self.VK = {
            'VOLUME_UP': (175, 1),
            'VOLUME_DOWN': (174, 1),
            'MUTE': (173, 1),
            'MEDIA_PLAY_PAUSE': (179, 1),
            'MEDIA_NEXT': (176, 1),
            'MEDIA_PREV': (177, 1),
            'RIGHT': (39, 1),
            'LEFT': (37, 1),
            'UP': (38, 1),
            'DOWN': (40, 1),
            'SPACE': (32, 0),
            'ESCAPE': (27, 0),
            'RETURN': (13, 0),
            'TAB': (9, 0),
            'LWIN': (91, 1),
            'LSHIFT': (160, 0),
            'LCONTROL': (162, 0),
            'LMENU': (164, 0), # Alt
            'KEY_D': (68, 0),
            'KEY_S': (83, 0),
            'KEY_F': (70, 0),
            'F11': (122, 0),
        }

        # Mouse flags
        self.MOUSEEVENTF_MOVE = 0x0001
        self.MOUSEEVENTF_LEFTDOWN = 0x0002
        self.MOUSEEVENTF_LEFTUP = 0x0004
        self.MOUSEEVENTF_RIGHTDOWN = 0x0008
        self.MOUSEEVENTF_RIGHTUP = 0x0010
        self.MOUSEEVENTF_MIDDLEDOWN = 0x0020
        self.MOUSEEVENTF_MIDDLEUP = 0x0040
        self.MOUSEEVENTF_WHEEL = 0x0800
        self.MOUSEEVENTF_ABSOLUTE = 0x8000

    def play_sound(self, sound_type='click'):
        """Plays subtle, non-blocking feedback tones."""
        if not self.sound_enabled or winsound is None:
            return

        def _beep():
            try:
                if sound_type == 'click':
                    winsound.Beep(988, 30)      # High crisp click (B5, 30ms)
                elif sound_type == 'action':
                    winsound.Beep(1318, 45)     # Confirmation chirp (E6, 45ms)
                elif sound_type == 'toggle_on':
                    winsound.Beep(880, 40)
                    winsound.Beep(1175, 50)
                elif sound_type == 'toggle_off':
                    winsound.Beep(1175, 40)
                    winsound.Beep(880, 50)
                elif sound_type == 'swipe':
                    winsound.Beep(1760, 35)     # Bright swoosh (A6, 35ms)
            except Exception:
                pass

        threading.Thread(target=_beep, daemon=True).start()

    # --- Mouse Control ---

    def move_mouse(self, screen_x, screen_y):
        """Move cursor directly to absolute pixel coordinates."""
        if self.dry_run or not self.is_windows:
            return
        ix = int(max(0, min(self.screen_w - 1, screen_x)))
        iy = int(max(0, min(self.screen_h - 1, screen_y)))
        self.user32.SetCursorPos(ix, iy)

    def mouse_left_down(self):
        if self.dry_run or not self.is_windows or self.is_left_down:
            return
        self.user32.mouse_event(self.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        self.is_left_down = True
        self.play_sound('click')

    def mouse_left_up(self):
        if self.dry_run or not self.is_windows or not self.is_left_down:
            return
        self.user32.mouse_event(self.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        self.is_left_down = False

    def mouse_click(self):
        """Standard left click."""
        if self.dry_run or not self.is_windows:
            return
        self.user32.mouse_event(self.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        self.user32.mouse_event(self.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        self.play_sound('click')

    def mouse_right_click(self):
        """Right click."""
        if self.dry_run or not self.is_windows:
            return
        self.user32.mouse_event(self.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
        self.user32.mouse_event(self.MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
        self.play_sound('action')

    def mouse_double_click(self):
        self.mouse_click()
        time.sleep(0.05)
        self.mouse_click()

    def mouse_scroll(self, clicks):
        """Vertical scroll (positive = up, negative = down)."""
        if self.dry_run or not self.is_windows:
            return
        wheel_delta = int(clicks * 120)
        self.user32.mouse_event(self.MOUSEEVENTF_WHEEL, 0, 0, wheel_delta, 0)

    # --- Keyboard & Macro Control ---

    def _send_key(self, vk_code, flags=0, is_up=False):
        if self.dry_run or not self.is_windows:
            return
        scan = self.user32.MapVirtualKeyW(vk_code, 0)
        key_flags = flags | (2 if is_up else 0)
        self.user32.keybd_event(vk_code, scan, key_flags, 0)

    def press_key(self, key_name):
        """Press and release a single key."""
        if self.dry_run or not self.is_windows:
            return
        vk, flags = self.VK.get(key_name, (32, 0))
        self._send_key(vk, flags, is_up=False)
        self._send_key(vk, flags, is_up=True)
        self.play_sound('action')

    def send_combo(self, *key_names):
        """Press and hold keys in order, then release in reverse order (e.g. Win+Shift+S)."""
        if self.dry_run or not self.is_windows:
            return
        vks = [self.VK[k] for k in key_names if k in self.VK]
        for vk, flags in vks:
            self._send_key(vk, flags, is_up=False)
        time.sleep(0.04)
        for vk, flags in reversed(vks):
            self._send_key(vk, flags, is_up=True)
        self.play_sound('action')

    # Common Next-Gen Action Shortcuts
    def action_screenshot(self):
        """Windows Snipping Tool (Win + Shift + S)."""
        self.send_combo('LWIN', 'LSHIFT', 'KEY_S')

    def action_switch_app_next(self):
        """Alt + Tab switch."""
        self.send_combo('LMENU', 'TAB')
        self.play_sound('swipe')

    def action_show_desktop(self):
        """Win + D."""
        self.send_combo('LWIN', 'KEY_D')
        self.play_sound('swipe')

    def action_toggle_fullscreen(self):
        """F11."""
        self.press_key('F11')

