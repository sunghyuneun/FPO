import sys
from pathlib import Path
import torch
import torch.optim as optim

target_dir = Path(__file__).resolve().parent.parent / "src/PPO"
sys.path.append(str(target_dir))

from ppo_buffer import Buffer
from ppo_network import ActorCritic
from ppo_update import ppo_update

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")


def test_ppo_update():
    # Test Network
    state_dim = 8
    action_dim = 4

    batch_size = 64

    model = ActorCritic(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    # Test Buffer
    size = 8
    gamma = 0.99
    lam = 0.95
    batch_size = 4

    test_buffer = Buffer(size, state_dim, action_dim, device)

    for i in range(size):
        state = torch.randn(state_dim)
        action = torch.randn(action_dim)
        value = torch.tensor(1)
        log_prob = torch.tensor(-1)
        done = torch.tensor(1) if i == 4 else 0
        reward = torch.tensor(1)

        test_buffer.add(state, action, value, log_prob, done, reward)

    test_buffer.gae(gamma, lam)
    test_buffer.normalize_advantages()

    # Test PPO Params
    epoch_count = 4
    clip_epsilon = 0.2
    entropy_coefficient = 0.01

    weights_before = [param.clone().detach() for param in model.actor_mean.parameters()]

    ppo_update(
        model,
        optimizer,
        test_buffer,
        batch_size,
        epoch_count,
        clip_epsilon,
        entropy_coefficient,
    )

    # Test 1: Weights actually changed
    for i, param in enumerate(model.actor_mean.parameters()):
        # Calculate the norm of the difference
        weight_diff = torch.norm(weights_before[i] - param)

        assert weight_diff > 0, f"Layer index {i} weights did not change"
    print("Change in weight test passed!")


if __name__ == "__main__":
    test_ppo_update()
