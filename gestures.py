"""
Next-Generation Hand Gesture Engine.
Combines 3D biomechanical joint angle analysis, spatial fingertip distance geometry,
and dynamic motion trajectory tracking.
"""
import math
import numpy as np

# Landmark Indices
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# 4 Fingers (knuckle, joint, tip)
FINGERS = [
    (INDEX_MCP, INDEX_PIP, INDEX_TIP),
    (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
    (RING_MCP, RING_PIP, RING_TIP),
    (PINKY_MCP, PINKY_PIP, PINKY_TIP),
]

# Bones for skeleton rendering
BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]


def joint_angle_3d(a, b, c):
    """Calculates angle at joint b (in degrees) between segments b->a and b->c in 3D world space."""
    u = a - b
    v = c - b
    norm_prod = (np.linalg.norm(u) * np.linalg.norm(v)) + 1e-9
    cos = np.dot(u, v) / norm_prod
    return math.degrees(math.acos(float(np.clip(cos, -1.0, 1.0))))


def vector_direction_2d(vec, cone_deg=35):
    """Identifies primary direction ('RIGHT', 'UP', 'LEFT', 'DOWN') within a cone, or None if diagonal."""
    angle = math.degrees(math.atan2(-vec[1], vec[0])) # Image coordinates (y increases downward)
    targets = (('RIGHT', 0), ('UP', 90), ('LEFT', 180), ('DOWN', -90))
    for name, centre in targets:
        diff = abs((angle - centre + 180) % 360 - 180)
        if diff <= cone_deg:
            return name
    return None


class GestureClassifier:
    def __init__(self, config=None):
        self.config = config or {}
        # Thresholds
        self.curled_below_deg = self.config.get('curled_below_deg', 125)
        self.straight_above_deg = self.config.get('straight_above_deg', 145)
        self.pinch_close = self.config.get('pinch_close', 0.22)
        self.pinch_open = self.config.get('pinch_open', 0.32)
        self.thumb_out_ratio = self.config.get('thumb_out_ratio', 1.05)

        # Internal state for hysteresis
        self.was_pinching_index = False
        self.was_pinching_middle = False

    def classify_hand(self, px, world, handedness="Right"):
        """Classify pose of a single hand.
        px: (21, 2) image pixel landmarks
        world: (21, 3) 3D camera-independent metric landmarks
        Returns a dictionary with parsed gesture information."""

        # 1. 3D Finger Bend Angles (index, middle, ring, pinky)
        bends = [joint_angle_3d(world[k], world[j], world[t]) for k, j, t in FINGERS]
        curled = [b < self.curled_below_deg for b in bends]
        straight = [b > self.straight_above_deg for b in bends]

        # Thumb bend (knuckle -> IP -> tip)
        thumb_bend = joint_angle_3d(world[THUMB_MCP], world[THUMB_IP], world[THUMB_TIP])
        thumb_straight = thumb_bend > 140

        # Palm size in 2D for scale normalization
        palm_size = float(np.linalg.norm(px[MIDDLE_MCP] - px[WRIST])) + 1e-9

        # Fingertip to Thumb gaps (normalized)
        gap_index = float(np.linalg.norm(px[THUMB_TIP] - px[INDEX_TIP])) / palm_size
        gap_middle = float(np.linalg.norm(px[THUMB_TIP] - px[MIDDLE_TIP])) / palm_size

        # 2. Pinch with Hysteresis
        limit_idx = self.pinch_open if self.was_pinching_index else self.pinch_close
        is_pinch_index = (gap_index < limit_idx) and (bends[0] > 75)
        self.was_pinching_index = is_pinch_index

        limit_mid = self.pinch_open if self.was_pinching_middle else self.pinch_close
        is_pinch_middle = (gap_middle < limit_mid) and (bends[1] > 75) and not is_pinch_index
        self.was_pinching_middle = is_pinch_middle

        # Extended Thumb Check
        tip_pinky = np.linalg.norm(px[THUMB_TIP] - px[PINKY_MCP])
        joint_pinky = np.linalg.norm(px[THUMB_IP] - px[PINKY_MCP])
        thumb_extended = (tip_pinky / (joint_pinky + 1e-9)) > self.thumb_out_ratio

        # Recognized Gesture Name
        pose = None

        # A) Pinch Gestures
        if is_pinch_index:
            pose = 'PINCH_INDEX'
        elif is_pinch_middle:
            pose = 'PINCH_MIDDLE'

        # B) Thumb Up / Thumb Down (Fist with extended thumb)
        elif all(curled) and thumb_extended:
            thumb_dir = vector_direction_2d(px[THUMB_TIP] - px[THUMB_MCP], cone_deg=35)
            if thumb_dir == 'UP':
                pose = 'THUMB_UP'
            elif thumb_dir == 'DOWN':
                pose = 'THUMB_DOWN'

        # C) Open Palm (All 5 fingers straight)
        elif all(straight) and thumb_extended:
            pose = 'OPEN_PALM'

        # D) Fist (All fingers tightly curled)
        elif all(curled) and not thumb_extended:
            pose = 'FIST'

        # E) Victory / Peace Sign (Index & Middle straight, Ring & Pinky curled)
        elif straight[0] and straight[1] and curled[2] and curled[3]:
            pose = 'PEACE'

        # F) Rock On (Index & Pinky straight, Middle & Ring curled)
        elif straight[0] and straight[3] and curled[1] and curled[2]:
            pose = 'ROCK_ON'

        # G) Two-Finger Point / Scroll (Index & Middle straight together)
        elif straight[0] and straight[1] and curled[2] and curled[3]:
            # Hand index vector
            tip_dist = np.linalg.norm(px[INDEX_TIP] - px[MIDDLE_TIP]) / palm_size
            if tip_dist < 0.28:
                pose = 'TWO_FINGER_POINT'

        # H) Single Index Point (Right / Left / Up)
        elif straight[0] and all(curled[1:]):
            finger_vec = px[INDEX_TIP] - px[INDEX_MCP]
            if np.linalg.norm(finger_vec) >= 0.55 * palm_size:
                f_dir = vector_direction_2d(finger_vec, cone_deg=35)
                if f_dir == 'RIGHT':
                    pose = 'POINT_RIGHT'
                elif f_dir == 'LEFT':
                    pose = 'POINT_LEFT'
                elif f_dir == 'UP':
                    pose = 'POINT_UP'

        return {
            'pose': pose,
            'is_pinch_index': is_pinch_index,
            'is_pinch_middle': is_pinch_middle,
            'gap_index': gap_index,
            'gap_middle': gap_middle,
            'index_pos': px[INDEX_TIP],
            'palm_center': px[MIDDLE_MCP],
            'handedness': handedness,
            'bends': bends,
            'curled': curled,
            'straight': straight,
        }

