import torch
import gymnasium as gym

print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Device:", torch.cuda.get_device_name(0))

env = gym.make("Ant-v5", render_mode="rgb_array")
obs, info = env.reset()
print("MuJoCo Environment Loaded Successfully! Obs shape:", obs.shape)
env.close()
