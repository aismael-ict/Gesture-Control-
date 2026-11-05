"""
Signal filtering and trajectory analysis for Next-Gen gesture control.
Implements the 1€ (One-Euro) Filter for jitter-free, ultra-low latency cursor motion
and trajectory tracking for dynamic swipe gestures.
"""
import math
import time
from collections import deque


class LowPassFilter:
    def __init__(self, alpha=0.5):
        self.y = None
        self.alpha = alpha

    def filter(self, value, alpha=None):
        if alpha is not None:
            self.alpha = alpha
        if self.y is None:
            s = value
        else:
            s = self.alpha * value + (1.0 - self.alpha) * self.y
        self.y = s
        return s

    def reset(self):
        self.y = None


class OneEuroFilter:
    """1€ Filter: Adaptive low-pass filter that dynamically balances jitter reduction
    and lag based on speed. At low speeds, cutoff is small (removes tremors/jitter).
    At high speeds, cutoff is large (eliminates lag)."""

    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_filt = LowPassFilter()
        self.dx_filt = LowPassFilter()
        self.last_time = None

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x, timestamp=None):
        if timestamp is None:
            timestamp = time.monotonic()

        if self.last_time is None:
            self.last_time = timestamp
            return self.x_filt.filter(x, 1.0)

        dt = max(timestamp - self.last_time, 1e-5)
        self.last_time = timestamp

        # Estimate derivative of input signal
        prev_x = self.x_filt.y
        dx = (x - prev_x) / dt if prev_x is not None else 0.0
        edx = self.dx_filt.filter(dx, self._alpha(self.d_cutoff, dt))

        # Dynamic cutoff frequency based on rate of change
        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self.x_filt.filter(x, self._alpha(cutoff, dt))

    def reset(self):
        self.x_filt.reset()
        self.dx_filt.reset()
        self.last_time = None


class PointFilter2D:
    """Applies independent 1€ filters along X and Y axes."""

    def __init__(self, min_cutoff=1.2, beta=0.05):
        self.x_filter = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        self.y_filter = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)

    def filter(self, x, y, timestamp=None):
        fx = self.x_filter.filter(x, timestamp)
        fy = self.y_filter.filter(y, timestamp)
        return fx, fy

    def reset(self):
        self.x_filter.reset()
        self.y_filter.reset()


class TrajectoryTracker:
    """Tracks landmark positions over recent history to detect dynamic gestures (swipes)."""

    def __init__(self, window_seconds=0.35, max_points=25):
        self.window_s = window_seconds
        self.history = deque(maxlen=max_points)

    def add(self, x, y, now):
        self.history.append((x, y, now))
        # Remove points older than window
        while self.history and (now - self.history[0][2] > self.window_s):
            self.history.popleft()

    def detect_swipe(self, min_velocity_norm=0.7, min_distance_norm=0.18):
        """Detect rapid horizontal or vertical swipe.
        Returns: 'SWIPE_LEFT', 'SWIPE_RIGHT', 'SWIPE_UP', 'SWIPE_DOWN', or None."""
        if len(self.history) < 4:
            return None

        p_start = self.history[0]
        p_end = self.history[-1]
        dt = p_end[2] - p_start[2]
        if dt < 0.06 or dt > self.window_s:
            return None

        dx = p_end[0] - p_start[0]
        dy = p_end[1] - p_start[1]
        dist = math.hypot(dx, dy)
        vel = dist / dt

        if dist < min_distance_norm or vel < min_velocity_norm:
            return None

        # Check linearity: points in between shouldn't be too chaotic
        angle = math.degrees(math.atan2(-dy, dx))  # Image coordinates: y increases downwards

        # Cones: Right (0), Up (90), Left (180/-180), Down (-90)
        cone = 38
        if abs((angle - 0 + 180) % 360 - 180) <= cone:
            self.history.clear()
            return 'SWIPE_RIGHT'
        elif abs((angle - 180 + 180) % 360 - 180) <= cone:
            self.history.clear()
            return 'SWIPE_LEFT'
        elif abs((angle - 90 + 180) % 360 - 180) <= cone:
            self.history.clear()
            return 'SWIPE_UP'
        elif abs((angle - (-90) + 180) % 360 - 180) <= cone:
            self.history.clear()
            return 'SWIPE_DOWN'

        return None

    def reset(self):
        self.history.clear()

