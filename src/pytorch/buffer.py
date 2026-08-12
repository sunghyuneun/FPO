import torch


class Buffer:
    def __init__(self, size, state_dim, action_dim, device):
        self.states = torch.zeros((size, state_dim), dtype=torch.float32).to(device)
        self.actions = torch.zeros((size, action_dim), dtype=torch.float32).to(device)
        self.values = torch.zeros(size, dtype=torch.float32).to(device)
        self.log_probs = torch.zeros(size, dtype=torch.float32).to(device)
        self.dones = torch.zeros(size, dtype=torch.float32).to(device)
        self.rewards = torch.zeros(size, dtype=torch.float32).to(device)
        self.returns = torch.zeros(size, dtype=torch.float32).to(device)
        self.advantages = torch.zeros(size, dtype=torch.float32).to(device)

        self.count = 0
        self.size = size

    def add(self, state, action, value, log_prob, done, reward):
        self.states[self.count] = state
        self.actions[self.count] = action
        self.values[self.count] = value
        self.log_probs[self.count] = log_prob
        self.dones[self.count] = done
        self.rewards[self.count] = reward

        self.count += 1

    def gae(self, gamma, lam):
        # Inside a batch, there may be multiple episodes. I need to stop bleed over.
        # if it's done, it's 1. if it's not done, it's 0.

        # base case
        A_t1 = 0
        V_t1 = 0

        for t in reversed(range(self.size)):
            # If it's done at that point, future value/advantage is all 0.
            delta_t = (
                self.rewards[t] + gamma * V_t1 * (1 - self.dones[t]) - self.values[t]
            )
            self.advantages[t] = delta_t + gamma * lam * A_t1 * (1 - self.dones[t])

            V_t1 = self.values[t]
            A_t1 = self.advantages[t]

        self.returns = self.advantages + self.values

    def minibatch_generator(self, batch_size):

        indices = torch.randperm(self.size)

        for start_index in range(0, self.size, batch_size):
            index = indices[start_index : start_index + batch_size]
            yield (
                self.states[index],
                self.actions[index],
                self.log_probs[index],
                self.returns[index],
                self.advantages[index],
            )


# YOU GOTTTTTT WHAT I WANTTTTTTTTTT
# GIRL YOU GOT WHAT I NEEDDDDDDDDDDDDDDDD
# 37 27 42
