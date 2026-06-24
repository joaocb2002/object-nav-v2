# scripts/02_load_env.py
import os
import habitat
from habitat import get_config

# Change working directory to a specific path
os.chdir("../habitat-lab")

config_path = "habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_hm3d.yaml"
cfg = get_config(config_path)

with habitat.Env(config=cfg) as env:
    print("\n\n\nEnvironment loaded successfully.")
    
    obs = env.reset()
    print("\n\n\nObservation keys:", obs)

    print("\n\n\nCurrent episode:", env.current_episode)
    print("\n\n\n****************ENV.DICT**************\n", env.__dict__)
    print("\n\n\n****************ENV.SIM**************\n", env.sim.__dict__)
    print("\n\n\n****************ENV.TASK**************\n", env.task.__dict__)
    print("\n\n\n****************ENV.EPISODES**************\n", env.episodes.__dict__)

    print("\n\n\n****************ENV**************\n", env)
    print("\n\n\n****************ENV.SIM**************\n", env.sim)
    print("\n\n\n****************ENV.TASK**************\n", env.task)
    print("\n\n\n****************ENV.EPISODES**************\n", env.episodes)
    print("\n\n\n****************ENV.ACTION_SPACE**************\n", env.action_space)




    