from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import cv2


# OpenCV exposes extended keys differently across GUI backends. These values
# cover the common GTK/Win32, X11, Qt, macOS, and low-byte arrow codes.
UP_ARROW_KEY_CODES = (82, 63232, 65362, 2490368, 16777235)
DOWN_ARROW_KEY_CODES = (84, 63233, 65364, 2621440, 16777237)


@dataclass(frozen=True)
class KeyboardControls:
    """Keyboard bindings for manual Habitat navigation."""

    forward_key: str = "w"
    left_key: str = "a"
    right_key: str = "d"
    stop_key: str = "f"

    def action_for_key(self, key_code: int) -> Optional[str]:
        """Return the action bound to an OpenCV key code."""
        if key_code in UP_ARROW_KEY_CODES:
            return "look_up"
        if key_code in DOWN_ARROW_KEY_CODES:
            return "look_down"

        bindings: Mapping[int, str] = {
            ord(self.forward_key): "move_forward",
            ord(self.left_key): "turn_left",
            ord(self.right_key): "turn_right",
            ord(self.stop_key): "stop",
        }
        return bindings.get(key_code)


@dataclass
class InteractiveKeyboardAgent:
    """Manual control agent driven by OpenCV keyboard events."""

    controls: KeyboardControls = KeyboardControls()
    wait_ms: int = 0

    def reset(self) -> None:
        """Reset per-episode state."""

    def act(self, observations: object | None = None) -> str:
        """Wait for a valid keypress and return the corresponding action."""
        del observations
        while True:
            key_code = cv2.waitKeyEx(self.wait_ms)
            action = self.controls.action_for_key(key_code)
            if action is not None:
                return action
            print("Invalid key. Use W/A/D to move, arrows to look, F to stop.")
