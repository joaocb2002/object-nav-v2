import habitat_sim

backend_cfg = habitat_sim.SimulatorConfiguration()
backend_cfg.scene_id = "../habitat-lab/data/scene_datasets/hm3d_v0.2/minival/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"
backend_cfg.scene_dataset_config_file = "../habitat-lab/data/scene_datasets/hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json"

print("Simulator backend configuration:", backend_cfg)

sem_cfg = habitat_sim.CameraSensorSpec()
sem_cfg.uuid = "semantic"
sem_cfg.sensor_type = habitat_sim.SensorType.SEMANTIC

print("Semantic sensor configuration:", sem_cfg)

agent_cfg = habitat_sim.agent.AgentConfiguration()
agent_cfg.sensor_specifications = [sem_cfg]

print("Agent configuration:", agent_cfg)

sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
sim = habitat_sim.Simulator(sim_cfg)

print("Simulator created successfully.")