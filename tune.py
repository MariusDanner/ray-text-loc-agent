import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from text_localization_environment import TextLocEnv
from ray import tune
import numpy as np

def create_environment(conf):
    from text_localization_environment import TextLocEnv
    import numpy as np
    import os
    imagefile= conf.get('image', '/Users/mariusdanner/Desktop/rl_text_stuff/last_year/data/simple_generated_data/image_locations.txt')
    boxfile=conf.get('image', '/Users/mariusdanner/Desktop/rl_text_stuff/last_year/data/simple_generated_data/bounding_boxes.npy')
    gpu=conf.get('gpu', 0)
    relative_paths = np.loadtxt(imagefile, dtype=str)
    images_base_path = os.path.dirname(imagefile)
    absolute_paths = [images_base_path + i.strip('.') for i in relative_paths]
    bboxes = np.load(boxfile)
    return TextLocEnv(absolute_paths, bboxes, gpu)

register_env('TextLocEnv-v0', create_environment)

ray.init()

tune.run(
    "PPO",
    stop={
            "timesteps_total": 10000,
        },
    config={
            "env": 'TextLocEnv-v0',
            "model": {
                "conv_filters": [224,224,224,224],
                "fcnet_hiddens": [1024,1024],
                "framestack": False,
                "grayscale": True,
                "fcnet_activation":"relu"
            },
            "lr": 0.005,
            "num_workers": 0,
            "env_config": {}
        }
)
