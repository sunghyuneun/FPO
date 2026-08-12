import sys
from pathlib import Path
import torch
import torch.optim as optim

target_dir = Path(__file__).resolve().parent.parent / "src/pytorch"
sys.path.append(str(target_dir))

from buffer import Buffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")


def test_buffer():
    size = 8
    state_dim = 4
    action_dim = 2
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

    # Test 1: GAE calculation errors
    assert test_buffer.returns.shape == (size,), (
        f"Expected return shape {(size,)}, got {test_buffer.returns.shape}"
    )
    assert not torch.isnan(test_buffer.advantages).any(), "NaN detected in advantages!"
    print("GAE calculation test passed!")

    # Test 2: Minibatch errors
    generator = test_buffer.minibatch_generator(batch_size)
    states, actions, log_probs, returns, advantages = next(generator)
    assert states.shape == (batch_size, state_dim), (
        f"Expected return shape {(batch_size, state_dim)}, got {states.shape}"
    )
    assert actions.shape == (batch_size, action_dim), (
        f"Expected return shape {(batch_size, action_dim)}, got {actions.shape}"
    )
    assert log_probs.shape == (batch_size,), (
        f"Expected return shape {(batch_size,)}, got {log_probs.shape}"
    )
    print("Minibatch test passed!")


if __name__ == "__main__":
    test_buffer()
