import os
import cv2
import random
import habitat
from habitat import get_config
from habitat.config import read_write

os.chdir("../habitat-lab")
CONFIG = "habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_hm3d.yaml"
FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
FINISH="f"

def rgb_to_bgr(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def print_episode(env):
    ep = env.current_episode
    goal = getattr(ep.goals[0], "object_category", "unknown")

    print("\nEpisode:", ep.episode_id)
    print("Scene:", ep.scene_id)
    print("Goal:", goal)

SCENE = "92vYG1q49FY"
cfg = get_config(CONFIG)

with read_write(cfg):
    cfg.habitat.dataset.content_scenes = [SCENE]
    cfg.habitat.environment.iterator_options.shuffle = False
    cfg.habitat.environment.iterator_options.group_by_scene = False
    cfg.habitat.environment.iterator_options.cycle = False

with habitat.Env(config=cfg) as env:
    #env.episode_iterator.set_next_episode_by_id(EPISODE_ID)
    obs = env.reset()
    cv2.imshow("RGB", rgb_to_bgr(obs["rgb"]))
    print_episode(env)

    while not env.episode_over:
        keystroke = cv2.waitKey(0)

        if keystroke == ord(FORWARD_KEY):
            action = "move_forward"
        elif keystroke == ord(LEFT_KEY):
            action = "turn_left"
        elif keystroke == ord(RIGHT_KEY):
            action = "turn_right"
        elif keystroke == ord(FINISH):
            action = "stop"
        else:
            print("INVALID KEY")
            continue
        
        print("Action:", action)
        obs = env.step(action)
        cv2.imshow("RGB", rgb_to_bgr(obs["rgb"]))

    print("Metrics:", env.get_metrics())

cv2.destroyAllWindows()