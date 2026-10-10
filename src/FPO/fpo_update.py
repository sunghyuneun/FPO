import torch


def ppo_update(
    model, optimizer, buffer, batch_size, epoch_count, clip_epsilon, entropy_coefficient
):
    for epoch in range(epoch_count):
        for (
            states,
            actions,
            log_probs,
            returns,
            advantages,
        ) in buffer.minibatch_generator(batch_size):
            # First one should be new_actions but unused for now
            _, new_log_probs, entropies, new_values = model.get_action_value(
                states, actions
            )

            log_ratios = new_log_probs - log_probs
            ratios = torch.exp(log_ratios)

            # self note: negative because it's a minimizer. also this is actor loss
            L_clip = -torch.min(
                ratios * advantages,
                torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon) * advantages,
            ).mean()
            critic_loss = 0.5 * ((new_values - returns) ** 2).mean()
            entropy_loss = entropies.mean()

            total_loss = L_clip + critic_loss + entropy_coefficient * entropy_loss
            optimizer.zero_grad()
            total_loss.backward()
            # clips the gradient norms to 0.5 to avoid gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
