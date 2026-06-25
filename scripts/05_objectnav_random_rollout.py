import os
import random
import cv2
import habitat
from habitat import get_config

os.chdir("../habitat-lab")
CONFIG = "habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_hm3d.yaml"
ACTIONS = ["move_forward", "turn_left", "turn_right"]

def rgb_to_bgr(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def print_episode(env):
    ep = env.current_episode
    goal = getattr(ep.goals[0], "object_category", "unknown")

    print("\nEpisode:", ep.episode_id)
    print("Scene:", ep.scene_id)
    print("Goal:", goal)

cfg = get_config(CONFIG)

with habitat.Env(config=cfg) as env:
    for _ in range(3):
        obs = env.reset()
        print_episode(env)

        for step in range(100):
            cv2.imshow("RGB", rgb_to_bgr(obs["rgb"]))
            cv2.waitKey(250)

            action = random.choice(ACTIONS)
            print("Action:", action)

            obs = env.step(action)

            if env.episode_over:
                break

        print("Metrics:", env.get_metrics())

cv2.destroyAllWindows()
