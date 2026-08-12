import torch
import gymnasium as gym


def test_installation():
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU Device:", torch.cuda.get_device_name(0))

    env = gym.make("HalfCheetah-v5")
    obs, info = env.reset()
    assert obs.shape == (17,), f"Gymnasium installation has a problem"
    print("MuJoCo Environment Loaded Successfully! Obs shape:", obs.shape)
    env.close()


if __name__ == "__main__":
    test_installation()
