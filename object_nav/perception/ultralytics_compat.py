"""
Compatibility helpers for Ultralytics APIs.

Ultralytics changes internal module layout across versions. This file provides
stable imports for the rest of the project.
"""

from __future__ import annotations

# LOGGER moved around across versions
try:
    # Newer versions commonly expose LOGGER here
    from ultralytics.utils import LOGGER  # type: ignore
except Exception:  # pragma: no cover
    try:
        # Fallback: some versions keep logging elsewhere
        from ultralytics.utils.loggers import LOGGER  # type: ignore
    except Exception:  # pragma: no cover
        LOGGER = None  # last-resort fallback

# Ops functions moved/changed signatures sometimes
try:
    from ultralytics.utils.ops import xywh2xyxy  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError("Failed to import xywh2xyxy from ultralytics.utils.ops") from e

# nms_rotated may not exist in all versions; keep a safe fallback
try:
    from ultralytics.utils.ops import nms_rotated  # type: ignore
except Exception:  # pragma: no cover
    nms_rotated = None  # your code must handle None if it relies on rotated NMS
