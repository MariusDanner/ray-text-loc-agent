import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from text_localization_environment import TextLocEnv
import numpy as np

def create_environment(conf):
    from text_localization_environment import TextLocEnv
    import numpy as np
    import os
    imagefile= conf.get('image', '../last_year/data/simple_generated_data/image_locations.txt')
    boxfile=conf.get('image', '../last_year/data/simple_generated_data/bounding_boxes.npy')
    gpu=conf.get('gpu', 0)
    relative_paths = np.loadtxt(imagefile, dtype=str)
    images_base_path = os.path.dirname(imagefile)
    absolute_paths = [images_base_path + i.strip('.') for i in relative_paths]
    bboxes = np.load(boxfile)
    return TextLocEnv(absolute_paths, bboxes, gpu)

register_env('TextLocEnv-v0', create_environment)

ray.init()
config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 0
config["model"]['conv_filters'] = [224,224,224,224]
config["model"]['fcnet_hiddens'] = [1024,1024]
config["model"]['framestack'] = False
config["model"]['grayscale'] = True
config["model"]['fcnet_activation'] = 'relu'
trainer = ppo.PPOTrainer(config=config, env="TextLocEnv-v0")
print('starting training')
for i in range(10):
   # Perform one iteration of training the policy with PPO
   result = trainer.train()
   print(pretty_print(result))
   checkpoint = trainer.save()
   print("checkpoint saved at", checkpoint)
# result = trainer.train()
# print(pretty_print(result))
