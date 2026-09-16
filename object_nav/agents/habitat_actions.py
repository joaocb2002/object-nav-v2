"""Repository-local Habitat action compatibility helpers."""

from __future__ import annotations

from typing import Any, Mapping

import magnum as mn
from habitat.core.registry import registry
from habitat.tasks.nav.nav import NavigationMovementAgentAction


def enable_compatible_look_actions(habitat_config: Any) -> None:
    """Select look actions compatible with current and older Habitat-Sim APIs."""
    actions = habitat_config.habitat.task.actions
    if not hasattr(actions, "look_up") or not hasattr(actions, "look_down"):
        raise ValueError("The Habitat task must define look_up and look_down actions.")
    actions.look_up.type = "CompatibleLookUpAction"
    actions.look_down.type = "CompatibleLookDownAction"


class _CompatibleVerticalLookAction(NavigationMovementAgentAction):
    def _move_camera_vertical(self, amount: float) -> None:
        agents = self._sim.agents
        if len(agents) != 1:
            raise RuntimeError("Navigation look actions require exactly one agent.")

        sensors = _agent_sensors(agents[0])
        rotation = mn.Quaternion.rotation(mn.Deg(amount), mn.Vector3.x_axis())
        for sensor in sensors.values():
            sensor.node.rotation = sensor.node.rotation * rotation


@registry.register_task_action
class CompatibleLookUpAction(_CompatibleVerticalLookAction):
    """Tilt all simulator sensors upward."""

    def step(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self._move_camera_vertical(self._tilt_angle)


@registry.register_task_action
class CompatibleLookDownAction(_CompatibleVerticalLookAction):
    """Tilt all simulator sensors downward."""

    def step(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self._move_camera_vertical(-self._tilt_angle)


def _agent_sensors(agent: Any) -> Mapping[str, Any]:
    sensors = getattr(agent, "sensors", None)
    if sensors is None:
        sensors = getattr(agent, "_sensors", None)
    if sensors is None:
        scene_node = getattr(agent, "scene_node", None)
        sensors = getattr(scene_node, "subtree_sensors", None)
    if sensors is None:
        raise RuntimeError("Could not find the Habitat-Sim agent sensor suite.")
    return sensors
