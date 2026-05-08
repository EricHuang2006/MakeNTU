import time

from config import (
    LOG_ANGLE_EVENTS,
    LOG_API_EVENTS,
    LOG_DETECTION_EVENTS,
    LOG_ENABLED,
    LOG_ERROR_EVENTS,
    LOG_STATE_EVENTS,
    LOG_THROTTLE_SECONDS,
)


_LAST_LOG_TIMES = {}

_CATEGORY_ENABLED = {
    "state": LOG_STATE_EVENTS,
    "detect": LOG_DETECTION_EVENTS,
    "angle": LOG_ANGLE_EVENTS,
    "api": LOG_API_EVENTS,
    "error": LOG_ERROR_EVENTS,
    "system": True,
}


def log_event(category, message, throttle_key=None, throttle_seconds=None):
    if not LOG_ENABLED:
        return
    if not _CATEGORY_ENABLED.get(category, True):
        return

    key = throttle_key or f"{category}:{message}"
    delay = LOG_THROTTLE_SECONDS if throttle_seconds is None else throttle_seconds
    now = time.monotonic()
    last_time = _LAST_LOG_TIMES.get(key)

    if last_time is not None and (now - last_time) < delay:
        return

    _LAST_LOG_TIMES[key] = now
    print(f"[{category.upper()}] {message}")


def log_once_per_change(category, change_key, value, message):
    if not LOG_ENABLED:
        return
    if not _CATEGORY_ENABLED.get(category, True):
        return

    cache_key = f"change:{change_key}"
    last_value = _LAST_LOG_TIMES.get(cache_key)
    if last_value == value:
        return

    _LAST_LOG_TIMES[cache_key] = value
    print(f"[{category.upper()}] {message}")
