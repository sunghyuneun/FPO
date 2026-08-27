import os
import torch
import yaml
import numpy as np
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import sys
from pathlib import Path

target_dir = Path(__file__).resolve().parent.parent / "src/pytorch"
sys.path.append(str(target_dir))

from buffer import Buffer
from network import ActorCritic
from ppo_update import ppo_update


def load_configs(config_path="../configs/ppo_torch_halfcheetah.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train():
    cfg = load_configs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    seed = cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make(cfg["env_name"], render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    model = ActorCritic(state_dim, action_dim)
    optimizer = optim.Adam(
        model.parameters(), lr=float(cfg["learning_rate"]), eps=float(cfg["adam_eps"])
    )
    buffer = Buffer(cfg["buffer_size"], state_dim, action_dim, device)

    # _ is info here, a dict
    state, _ = env.reset(seed=seed)
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device)

    global_step = 0
    episode_reward = 0.0
    completed_episode_rewards = []
    num_updates = cfg["total_timesteps"] // cfg["batch_size"]

    for update in range(1, num_updates + 1):
        for step in range(cfg["batch_size"]):
            global_step += 1

            with torch.no_grad():
                action, log_prob, entropy, value = model.get_action_value(
                    state_tensor.unsqueeze(0)
                )
                env_action = np.clip(
                    action.squeeze(0).cpu().numpy(),
                    env.action_space.low,
                    env.action_space.high,
                )

            next_state, reward, terminated, truncated, _ = env.step(env_action)
            done = terminated or truncated
            episode_reward += reward

            buffer.add(
                state_tensor,
                action.squeeze(0),
                value.squeeze(0),
                log_prob.squeeze(0),
                done,
                reward,
            )

            if done:
                completed_episode_rewards.append(episode_reward)
                episode_reward = 0.0
                next_state, _ = env.reset()

            state_tensor = torch.tensor(next_state, dtype=torch.float32, device=device)

        buffer.gae(cfg["gamma"], cfg["gae_lambda"])
        buffer.normalize_advantages()
        buffer.clear()

        ppo_update(
            model,
            optimizer,
            buffer,
            cfg["batch_size"],
            cfg["epoch_count"],
            cfg["clip_eps"],
            float(cfg["ent_coef"]),
        )

    save_path = f"../models/{cfg['env_name']}_torch_ppo.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Training finished, model saved to {save_path}")

    env.close()


if __name__ == "__main__":
    train()
