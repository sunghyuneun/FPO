import os
import torch
import yaml
import numpy as np
import torch.nn as nn
import torch.optim as optim
from itertools import count
from torch.distributions.normal import Normal
from collections import namedtuple

import gymnasium


def load_configs(config_path="configs/ppo_halfcheetah.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train():
    config = load_configs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")


if __name__ == "__main__":
    train()
