"""Optional tests for repository-local Habitat task actions."""

from __future__ import annotations

import importlib.util
import unittest
from types import SimpleNamespace


HABITAT_AVAILABLE = importlib.util.find_spec("habitat") is not None


@unittest.skipUnless(HABITAT_AVAILABLE, "Habitat-Lab is required")
class HabitatActionCompatibilityTest(unittest.TestCase):
    def test_config_selects_compatible_look_actions(self) -> None:
        from object_nav.agents.habitat_actions import enable_compatible_look_actions

        actions = SimpleNamespace(
            look_up=SimpleNamespace(type="LookUpAction"),
            look_down=SimpleNamespace(type="LookDownAction"),
        )
        config = SimpleNamespace(
            habitat=SimpleNamespace(task=SimpleNamespace(actions=actions))
        )

        enable_compatible_look_actions(config)

        self.assertEqual(actions.look_up.type, "CompatibleLookUpAction")
        self.assertEqual(actions.look_down.type, "CompatibleLookDownAction")

    def test_private_habitat_sim_sensor_suite_is_supported(self) -> None:
        from object_nav.agents.habitat_actions import _agent_sensors

        sensors = {"rgb": object(), "depth": object()}

        self.assertIs(_agent_sensors(SimpleNamespace(_sensors=sensors)), sensors)


if __name__ == "__main__":
    unittest.main()
