import sys
from pathlib import Path
import torch
import torch.optim as optim

target_dir = Path(__file__).resolve().parent.parent / "src/pytorch"
sys.path.append(str(target_dir))

from network import ActorCritic


def test_network():
    state_dim = 17
    action_dim = 6

    batch_size = 64

    model = ActorCritic(state_dim, action_dim)
    test_batch = torch.randn(batch_size, state_dim)
    action, log_prob, entropy, value = model.get_action_value(test_batch, None)
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    weights_before = [param.clone().detach() for param in model.actor_mean.parameters()]

    loss = log_prob.mean() + value.mean()
    loss.backward()
    optimizer.step()

    # Test 1: Shapes from Forward Pass
    assert action.shape == (batch_size, action_dim), (
        f"Expected action shape {(batch_size, action_dim)}, got {action.shape}"
    )
    assert log_prob.shape == (batch_size,), (
        f"Expected log_prob shape {(batch_size,)}, got {log_prob.shape}"
    )
    assert entropy.shape == (batch_size,), (
        f"Expected entropy shape {(batch_size,)}, got {entropy.shape}"
    )
    assert value.shape == (batch_size,), (
        f"Expected value shape {(batch_size,)}, got {value.shape}"
    )

    print("Forward pass shape test passed!")

    # Test 2: Back Propagation
    for name, param in model.actor_mean.named_parameters():
        assert param.grad is not None, f"Actor gradient is None for {name}"
        assert torch.sum(torch.abs(param.grad)) > 0, (
            f"Actor gradient is zero for {name}"
        )

    for name, param in model.critic.named_parameters():
        assert param.grad is not None, f"Critic gradient is None for {name}"
        assert torch.sum(torch.abs(param.grad)) > 0, (
            f"Critic gradient is zero for {name}"
        )

    print("Back propagation test passed!")

    # Test 3: Weights actually changed
    for i, param in enumerate(model.actor_mean.parameters()):
        # Calculate the norm of the difference
        weight_diff = torch.norm(weights_before[i] - param)

        assert weight_diff > 0, f"Layer index {i} weights did not change"
    print("Change in weight test passed!")


if __name__ == "__main__":
    test_network()
