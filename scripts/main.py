import os
import random
import time
from pathlib import Path

import habitat
from habitat import get_config
from habitat.config import read_write

from object_nav.agents import InteractiveKeyboardAgent
from object_nav.mapping.habitat import (
    HabitatVoxelMapper,
    enable_topdown_map_measure,
    render_habitat_topdown_map,
)
from object_nav.perception import (
    SegFormerConfig,
    assert_segformer_camera,
    build_segformer_segmenter,
    colorize_segmentation_bgr,
    depth_to_bgr,
    print_observations,
)
from object_nav.utils import (
    DashboardConfig,
    OpenCVDashboard,
    choose_random_objectnav_scene,
    print_env,
    print_episode,
    rgb_to_bgr,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HABITAT_LAB_ROOT = PROJECT_ROOT.parent / "habitat-lab"
CONFIG = (
    HABITAT_LAB_ROOT
    / "habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_hm3d.yaml"
)
SCENE_CONTENT_DIR = HABITAT_LAB_ROOT / "data/datasets/objectnav/hm3d/v2/train/content"
NUM_EPISODES = 1
DISPLAY = DashboardConfig(
    enabled_panels=(
        "RGB",
        "Depth",
        "SegFormer",
        "Voxel world",
        "Ground truth map",
    )
)
RUN_SEED = time.time_ns() % (2**32)
SCENE = choose_random_objectnav_scene(SCENE_CONTENT_DIR, rng=random.Random(RUN_SEED))

# Habitat dataset paths in the composed config are relative to this adjacent checkout.
os.chdir(HABITAT_LAB_ROOT)
cfg = get_config(str(CONFIG))
with read_write(cfg):
    cfg.habitat.seed = RUN_SEED
    cfg.habitat.dataset.content_scenes = [SCENE]
    cfg.habitat.environment.iterator_options.num_episode_sample = NUM_EPISODES
    enable_topdown_map_measure(cfg)

agent = InteractiveKeyboardAgent()
segmenter = build_segformer_segmenter(SegFormerConfig())
assert_segformer_camera(
    segmenter,
    CONFIG,
    habitat_lab_root=HABITAT_LAB_ROOT,
)
voxel_mapper = HabitatVoxelMapper(cfg)
dashboard = OpenCVDashboard(DISPLAY)

print(f"Run seed: {RUN_SEED}")
print(f"Selected scene: {SCENE}")
print(f"SegFormer checkpoint: {segmenter.checkpoint}")
print(f"SegFormer device: {segmenter.device}")
print(f"SegFormer temperature: {segmenter.temperature}")

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
            output_height = obs["rgb"].shape[0]
            dashboard.show(
                {
                    "RGB": lambda: rgb_to_bgr(obs["rgb"]),
                    "Depth": lambda: (
                        depth_to_bgr(obs["depth"]) if "depth" in obs else None
                    ),
                    "SegFormer": lambda: colorize_segmentation_bgr(
                        segmenter(obs["rgb"])["labels"]
                    ),
                    "Voxel world": lambda: voxel_mapper.render_maps(
                        env, output_height=output_height
                    ),
                    "Ground truth map": lambda: render_habitat_topdown_map(
                        env, output_height=output_height
                    ),
                }
            )

            action = agent.act(obs)
            print("Action:", action)
            obs = env.step(action)
            step += 1

        metrics = dict(env.get_metrics())
        metrics.pop("top_down_map", None)
        print("\n\n")
        print_episode(env.current_episode)
        print("Metrics:", metrics)

dashboard.close()
