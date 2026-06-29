import habitat
import numpy as np
from habitat.core.simulator import Observations
from habitat.sims.habitat_simulator.actions import HabitatSimActions


class RandomAgent(habitat.Agent):
    def __init__(self, success_distance: float, goal_sensor_uuid: str) -> None:
        self.dist_threshold_to_stop = success_distance
        self.goal_sensor_uuid = goal_sensor_uuid

    def reset(self) -> None:
        pass

    def is_goal_reached(self, observations: Observations) -> bool:
        dist = observations[self.goal_sensor_uuid][0]
        return dist <= self.dist_threshold_to_stop

    def act(self, observations: Observations) -> dict[str, int]:
        if self.is_goal_reached(observations):
            action = HabitatSimActions.stop
        else:
            action = np.random.choice(
                [
                    HabitatSimActions.move_forward,
                    HabitatSimActions.turn_left,
                    HabitatSimActions.turn_right,
                ]
            )
        return {"action": action}
