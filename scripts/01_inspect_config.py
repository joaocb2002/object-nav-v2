from habitat import get_config
from omegaconf import DictConfig, ListConfig, OmegaConf

config_path = "../habitat-lab/habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_hm3d.yaml"

cfg = get_config(config_path)

def print_config_fields(node, path="cfg"):
    if isinstance(node, DictConfig):
        for key, value in node.items():
            current_path = f"{path}.{key}"
            print(current_path)
            print_config_fields(value, current_path)
    elif isinstance(node, ListConfig):
        for index, value in enumerate(node):
            current_path = f"{path}[{index}]"
            print(current_path)
            print_config_fields(value, current_path)
    else:
        print(f"{path} = {node!r}")


print("Resolved config:")
print(OmegaConf.to_yaml(cfg, resolve=True))

print("All fields and keys:")
print_config_fields(cfg)

