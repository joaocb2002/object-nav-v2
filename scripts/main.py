import os
import random
import time
from functools import cache

import habitat
from habitat import get_config
from habitat.config import read_write

from object_nav.agents import InteractiveKeyboardAgent, enable_compatible_look_actions
from object_nav.mapping.habitat import (HabitatVoxelMapper, enable_topdown_map_measure, render_habitat_topdown_map)
from object_nav.perception import (SegFormerConfig, annotate_semantic_islands_bgr, assert_segformer_camera, build_segformer_segmenter, colorize_segmentation_bgr, depth_to_bgr, print_observations)
from object_nav.utils import (DashboardConfig, OpenCVDashboard, blend_bgr_overlay, choose_random_objectnav_scene, print_env, print_episode, rgb_to_bgr)
from main_config import HABITAT_LAB_ROOT, OBJECTNAV_CONFIG, SCENE_CONTENT_DIR

NUM_EPISODES = 1
SEGMENTATION_OVERLAY_OPACITY = 0.3
SHOW_SEGMENTATION_LABELS = True
DISPLAY = DashboardConfig(
    enabled_panels=(
        "RGB + segmentation",
        "Depth",
        "SegFormer",
        "3D voxel view",
        "Voxel top-down map",
        "Ground truth map",
    )
)
RUN_SEED = time.time_ns() % (2**32)
SCENE = choose_random_objectnav_scene(SCENE_CONTENT_DIR, rng=random.Random(RUN_SEED))

# Habitat dataset paths in the composed config are relative to this adjacent checkout.
os.chdir(HABITAT_LAB_ROOT)
cfg = get_config(str(OBJECTNAV_CONFIG))
with read_write(cfg):
    cfg.habitat.seed = RUN_SEED
    cfg.habitat.dataset.content_scenes = [SCENE]
    cfg.habitat.environment.iterator_options.num_episode_sample = NUM_EPISODES
    enable_compatible_look_actions(cfg)
    enable_topdown_map_measure(cfg)

agent = InteractiveKeyboardAgent()
dashboard = OpenCVDashboard(DISPLAY)
dashboard.open()
segmenter = build_segformer_segmenter(SegFormerConfig())
assert_segformer_camera(segmenter, OBJECTNAV_CONFIG, habitat_lab_root=HABITAT_LAB_ROOT)
voxel_mapper = HabitatVoxelMapper(cfg)

print(f"\nRun seed: {RUN_SEED}")
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
            segmentation = cache(lambda: segmenter(obs["rgb"]))
            segmentation_bgr = cache(lambda: colorize_segmentation_bgr(segmentation()["labels"]))
            segmentation_panel = cache(lambda: annotate_semantic_islands_bgr(segmentation()["labels"], segmentation_bgr()) if SHOW_SEGMENTATION_LABELS else segmentation_bgr())
            dashboard.show(
                {
                    "RGB + segmentation": lambda: blend_bgr_overlay(
                        rgb_to_bgr(obs["rgb"]),
                        segmentation_bgr(),
                        opacity=SEGMENTATION_OVERLAY_OPACITY,
                    ),
                    "Depth": lambda: (
                        depth_to_bgr(obs["depth"]) if "depth" in obs else None
                    ),
                    "SegFormer": segmentation_panel,
                    "3D voxel view": lambda: voxel_mapper.render_camera_view(
                        env, output_height=output_height
                    ),
                    "Voxel top-down map": lambda: voxel_mapper.render_topdown_map(
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
