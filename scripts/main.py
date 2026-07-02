import os
import time
import habitat
from habitat import get_config
from habitat.config import read_write
from object_nav.agents import InteractiveKeyboardAgent
from object_nav.mapping.habitat import HabitatVoxelMapper, enable_topdown_map_measure, show_habitat_topdown_map
from object_nav.mapping.visualization import show_navigation_maps
from object_nav.perception import YoloConfig, build_yolo_detector, close_perception_windows, print_observations, show_depth_rgb_detections
from object_nav.utils import print_env, print_episode

os.chdir("../habitat-lab")
CONFIG = "habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_hm3d.yaml"
SCENE = "92vYG1q49FY"
NUM_EPISODES = 1

cfg = get_config(CONFIG)
with read_write(cfg):
    cfg.habitat.seed = time.time_ns() % (2**32)
    cfg.habitat.dataset.content_scenes = [SCENE]
    cfg.habitat.environment.iterator_options.num_episode_sample = NUM_EPISODES
    enable_topdown_map_measure(cfg)

agent = InteractiveKeyboardAgent()
yolo_detector = build_yolo_detector(YoloConfig())
voxel_mapper = HabitatVoxelMapper(cfg)

with habitat.Env(config=cfg) as env:
    print_env(env)

    for _ in range(NUM_EPISODES):
        obs = env.reset()
        print(f"\n\n--- Running episode {env.current_episode.episode_id} ---")
        agent.reset()
        voxel_mapper.reset()
        step = 0
        print_episode(env.current_episode)

        while not env.episode_over:
            print(f"\nStep {step}")
            print_observations(obs)
            voxel_mapper.integrate(env, obs, step)
            detections = yolo_detector.detect(obs["rgb"])
            show_depth_rgb_detections(obs["rgb"], obs["depth"], detections)
            show_navigation_maps(voxel_mapper.render_maps(env, output_height=obs["rgb"].shape[0]))
            show_habitat_topdown_map(env, output_height=obs["rgb"].shape[0])

            action = agent.act(obs)
            print("Action:", action)
            obs = env.step(action)
            step += 1

        metrics = dict(env.get_metrics())
        metrics.pop("top_down_map", None)
        print("Metrics:", metrics)

close_perception_windows()
