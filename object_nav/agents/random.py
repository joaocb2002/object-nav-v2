from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class RandomActionAgent:
    """Select a navigation action uniformly at random."""

    actions: Sequence[str] = ("move_forward", "turn_left", "turn_right")
    rng: random.Random = field(default_factory=random.SystemRandom)

    def reset(self) -> None:
        """Reset per-episode state."""

    def act(self, observations: object | None = None) -> str:
        """Return the next action."""
        del observations
        return self.rng.choice(self.actions)


RandomAgent = RandomActionAgent
