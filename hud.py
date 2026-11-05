"""
Futuristic Cyberpunk HUD & Visualization Renderer for HandControl Next-Gen.
Renders neon glowing skeletons, circular charge-up rings, motion trails,
glassmorphism cards, and interactive targeting reticles.
"""
import math
import time
import cv2
import numpy as np

from gestures import BONES, MIDDLE_MCP, INDEX_TIP, THUMB_TIP


class CyberHUD:
    def __init__(self, theme="cyberpunk"):
        self.theme = theme
        # BGR Palettes
        self.NEON_CYAN = (255, 220, 0)
        self.NEON_MAGENTA = (255, 60, 220)
        self.EMERALD_GREEN = (100, 255, 120)
        self.ELECTRIC_AMBER = (0, 180, 255)
        self.WHITE = (255, 255, 255)
        self.DARK_BG = (18, 16, 22)
        self.CARD_BG = (35, 28, 40)

        # Trails history for fingertip
        self.trails = [] # [(x, y, timestamp)]

    def draw_skeleton(self, frame, px, colour=None, glow=True, size=1.0):
        """Draws glowing skeleton with anti-aliased dual-pass lines and landmark dots."""
        if colour is None:
            colour = self.NEON_CYAN

        pts = px.astype(int)
        line_w = max(2, int(2.5 * size))
        glow_w = max(4, int(6.0 * size))
        dot_r = max(3, int(4.0 * size))

        # Optional soft glow pass
        if glow:
            overlay = frame.copy()
            for a, b in BONES:
                cv2.line(overlay, tuple(pts[a]), tuple(pts[b]), colour, glow_w, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.40, frame, 0.60, 0, frame)

        # Crisp inner bone lines
        for a, b in BONES:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), self.WHITE, line_w, cv2.LINE_AA)

        # Landmark points
        for x, y in pts:
            cv2.circle(frame, (int(x), int(y)), dot_r + 1, colour, -1, cv2.LINE_AA)
            cv2.circle(frame, (int(x), int(y)), dot_r - 1, self.WHITE, -1, cv2.LINE_AA)

    def draw_circular_progress(self, frame, center_pt, progress, radius=38, colour=None):
        """Draws a smooth circular progress dial around the hand center."""
        if colour is None:
            colour = self.ELECTRIC_AMBER

        cx, cy = int(center_pt[0]), int(center_pt[1])
        # Background dark ring
        cv2.circle(frame, (cx, cy), radius, (50, 50, 50), 3, cv2.LINE_AA)

        if progress > 0.01:
            end_angle = -90 + int(progress * 360)
            cv2.ellipse(frame, (cx, cy), (radius, radius), 0, -90, end_angle, colour, 5, cv2.LINE_AA)
            # Glowing dot at the leading tip of the dial
            rad = math.radians(end_angle)
            tip_x = int(cx + radius * math.cos(rad))
            tip_y = int(cy + radius * math.sin(rad))
            cv2.circle(frame, (tip_x, tip_y), 5, self.WHITE, -1, cv2.LINE_AA)

    def draw_air_mouse_reticle(self, frame, pos, is_clicking=False, is_right_clicking=False):
        """Draws futuristic targeting reticle on cursor position."""
        cx, cy = int(pos[0]), int(pos[1])
        color = self.NEON_MAGENTA if is_right_clicking else (self.EMERALD_GREEN if is_clicking else self.NEON_CYAN)
        r = 16 if is_clicking else 20

        # Reticle circle
        cv2.circle(frame, (cx, cy), r, color, 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 3, self.WHITE, -1, cv2.LINE_AA)

        # Crosshair ticks
        cv2.line(frame, (cx - r - 6, cy), (cx - r + 2, cy), color, 2, cv2.LINE_AA)
        cv2.line(frame, (cx + r - 2, cy), (cx + r + 6, cy), color, 2, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - r - 6), (cx, cy - r + 2), color, 2, cv2.LINE_AA)
        cv2.line(frame, (cx, cy + r - 2), (cx, cy + r + 6), color, 2, cv2.LINE_AA)

        # Click indicator text
        if is_clicking:
            cv2.putText(frame, "CLICK", (cx + 22, cy + 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, self.EMERALD_GREEN, 1, cv2.LINE_AA)
        elif is_right_clicking:
            cv2.putText(frame, "R-CLICK", (cx + 22, cy + 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, self.NEON_MAGENTA, 1, cv2.LINE_AA)

    def draw_active_box(self, frame, box_cfg):
        """Draws corner brackets indicating the active interaction bounding box."""
        h, w = frame.shape[:2]
        x1 = int(box_cfg["x_min"] * w)
        x2 = int(box_cfg["x_max"] * w)
        y1 = int(box_cfg["y_min"] * h)
        y2 = int(box_cfg["y_max"] * h)

        bracket_len = 24
        color = (120, 100, 100)

        # 4 Corners
        # Top-Left
        cv2.line(frame, (x1, y1), (x1 + bracket_len, y1), color, 2, cv2.LINE_AA)
        cv2.line(frame, (x1, y1), (x1, y1 + bracket_len), color, 2, cv2.LINE_AA)
        # Top-Right
        cv2.line(frame, (x2, y1), (x2 - bracket_len, y1), color, 2, cv2.LINE_AA)
        cv2.line(frame, (x2, y1), (x2, y1 + bracket_len), color, 2, cv2.LINE_AA)
        # Bottom-Left
        cv2.line(frame, (x1, y2), (x1 + bracket_len, y2), color, 2, cv2.LINE_AA)
        cv2.line(frame, (x1, y2), (x1, y2 - bracket_len), color, 2, cv2.LINE_AA)
        # Bottom-Right
        cv2.line(frame, (x2, y2), (x2 - bracket_len, y2), color, 2, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2, y2 - bracket_len), color, 2, cv2.LINE_AA)

    def draw_header_hud(self, frame, fps=0, air_mouse_active=False, hands_count=0, paused=False):
        """Draws top glassmorphism status bar with pills and active indicators."""
        h, w = frame.shape[:2]

        # Top banner background (semi-transparent dark)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 52), self.DARK_BG, -1)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

        # Title
        cv2.putText(frame, "HAND CONTROL", (16, 26), cv2.FONT_HERSHEY_DUPLEX, 0.65, self.WHITE, 1, cv2.LINE_AA)
        cv2.putText(frame, "NEXT-GEN 2.0", (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.NEON_CYAN, 1, cv2.LINE_AA)

        # Status Badges
        x_badge = 190
        # Air Mouse status
        mouse_color = self.NEON_CYAN if air_mouse_active else (100, 100, 100)
        mouse_text = "[AIR MOUSE: ON]" if air_mouse_active else "[AIR MOUSE: OFF (M)]"
        cv2.putText(frame, mouse_text, (x_badge, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, mouse_color, 1, cv2.LINE_AA)

        # Pause status
        if paused:
            cv2.putText(frame, "[PAUSED]", (x_badge + 220, 32), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)
        else:
            hands_text = f"Hands: {hands_count}"
            cv2.putText(frame, hands_text, (x_badge + 220, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.EMERALD_GREEN if hands_count > 0 else (120, 120, 120), 1, cv2.LINE_AA)

        # FPS indicator
        fps_str = f"{int(fps)} FPS"
        cv2.putText(frame, fps_str, (w - 85, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.50, self.WHITE, 1, cv2.LINE_AA)

    def draw_action_banner(self, frame, action_text, elapsed_s, duration=1.0):
        """Draws large glowing action badge in the bottom center that smoothly fades out."""
        if elapsed_s >= duration or not action_text:
            return

        h, w = frame.shape[:2]
        alpha = max(0.0, 1.0 - (elapsed_s / duration))

        banner_text = f"ACTION: {action_text}"
        (tw, th), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_DUPLEX, 0.85, 2)
        cx = (w - tw) // 2
        cy = h - 55

        # Background card
        pad_x, pad_y = 20, 12
        overlay = frame.copy()
        cv2.rectangle(overlay, (cx - pad_x, cy - th - pad_y), (cx + tw + pad_x, cy + pad_y), self.CARD_BG, -1)
        cv2.rectangle(overlay, (cx - pad_x, cy - th - pad_y), (cx + tw + pad_x, cy + pad_y), self.EMERALD_GREEN, 2)
        cv2.putText(overlay, banner_text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.85, self.EMERALD_GREEN, 2, cv2.LINE_AA)

        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

    def draw_footer_hotkeys(self, frame):
        """Draws sleek footer hotkey reminders."""
        h, w = frame.shape[:2]
        footer_text = "Hotkeys: [M] Air Mouse | [P] Pause | [D] Dark Mode | [Q/Esc] Quit"
        cv2.putText(frame, footer_text, (16, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)

