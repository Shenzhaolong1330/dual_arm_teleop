"""Synchronous GN01 aperture feedback using direct force, with an input watchdog.

Changing Move goals repeatedly interrupts motion on the tested GN01/controller.
Intermediate aperture goals therefore use bounded force feedback. Endpoint goals
use one Move to retain the configured grasp force and speed. Public goals remain
apertures in metres, accepted only after the corresponding SDK call succeeds.
"""
import math
import threading
import time


class GripperApertureController:
    def __init__(self, gripper, *, min_width, max_width, speed, force_limit,
                 free_force_limit=5.0, timeout=0.25):
        self.gripper = gripper
        self.min_width, self.max_width = min_width, max_width
        self.speed, self.force_limit = speed, force_limit
        self.free_force_limit = min(free_force_limit, force_limit)
        self.timeout = timeout
        if not all(math.isfinite(v) for v in (
                min_width, max_width, speed, force_limit, free_force_limit, timeout)):
            raise ValueError("Gripper feedback parameters must be finite")
        if min_width >= max_width or min(speed, self.free_force_limit, timeout) <= 0:
            raise ValueError("Invalid gripper feedback limits")
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._error = None
        self._closed = False
        self._clear()
        self._thread = threading.Thread(target=self._watch, name="gn01-input-watchdog", daemon=True)
        self._thread.start()

    def _clear(self):
        self._deadline = None
        self._last_time = None
        self._last_width = None
        self._target = None
        self._mode = None
        self._velocity = self._integral = self._error_previous = 0.0

    def update(self, target):
        with self._lock:
            if self._closed:
                raise RuntimeError("Gripper aperture controller is closed")
            if self._error is not None:
                raise RuntimeError("Gripper watchdog failed to stop") from self._error
            if not math.isfinite(target) or not self.min_width <= target <= self.max_width:
                raise ValueError("Gripper aperture target outside configured limits")
            now = time.monotonic()
            try:
                if self._last_time is not None and now - self._last_time > self.timeout:
                    self.gripper.Stop()
                    self._clear()
                width = float(self.gripper.states().width)
                if not (math.isfinite(width)
                        and self.min_width - 0.002 <= width <= self.max_width + 0.002):
                    raise RuntimeError("Invalid GN01 width feedback")
                dt = min(0.1, max(1 / 120, now - self._last_time)) if self._last_time is not None else 1 / 30
                if self._last_width is not None:
                    self._velocity = 0.75 * self._velocity + 0.25 * (width - self._last_width) / dt
                endpoint = target <= self.min_width + 0.0002 or target >= self.max_width - 0.0002
                if endpoint:
                    if self._mode != "endpoint" or abs(target - self._target) >= 0.0002:
                        self.gripper.Move(target, self.speed, self.force_limit)
                    self._mode = "endpoint"
                    self._integral = self._error_previous = 0.0
                else:
                    error = width - target  # positive Grasp force closes
                    if error * self._error_previous < 0:
                        self._integral = 0.0
                    self._integral = max(-1.5, min(1.5, self._integral + 400 * error * dt))
                    force = max(-self.free_force_limit, min(self.free_force_limit,
                                300 * error + 10 * self._velocity + self._integral))
                    if abs(error) < 0.0008 and abs(self._velocity) < 0.003:
                        if self._mode != "hold":
                            self.gripper.Stop()
                        self._mode = "hold"
                        self._integral = 0.0
                    else:
                        self.gripper.Grasp(force)
                        self._mode = "force"
                    self._error_previous = error
                # No queue or unacknowledged goals. Only successful SDK calls
                # extend the watchdog and accept the new aperture setpoint.
                self._last_width, self._target = width, target
                self._last_time = time.monotonic()
                self._deadline = self._last_time + self.timeout
                return target
            except BaseException:
                try:
                    self.gripper.Stop()
                finally:
                    self._clear()
                raise

    def _check_watchdog(self, now):
        with self._lock:
            if self._deadline is not None and now >= self._deadline:
                try:
                    self.gripper.Stop()
                except Exception as exc:
                    self._error = exc
                finally:
                    self._clear()

    def _watch(self):
        while not self._stop.wait(0.02):
            self._check_watchdog(time.monotonic())

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        with self._lock:
            self._closed = True
            try:
                self.gripper.Stop()
            finally:
                self._clear()
