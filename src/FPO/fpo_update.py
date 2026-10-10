import torch
import copy


def FPO_Update(model, optimizer, buffer, batch_size, epoch_count, clip_epsilon):
    old_model = copy.deepcopy(model)
    old_model.eval()
    # turn it off

    for epoch in range(epoch_count):
        for (
            states,
            actions,
            returns,
            advantages,
        ) in buffer.minibatch_generator(batch_size):
            # First one should be new_actions but unused for now
            new_actions, new_values = model.get_action_value(states, actions)

            new_loss = model.cfm_loss(states, actions)
            with torch.no_grad():
                old_loss = old_model.cfm_loss(states, actions)

            log_ratios = old_loss - new_loss
            ratios = torch.exp(log_ratios)

            # self note: negative because it's a minimizer. also this is actor loss
            L_clip = -torch.min(
                ratios * advantages,
                torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon) * advantages,
            ).mean()
            critic_loss = 0.5 * ((new_values - returns) ** 2).mean()

            total_loss = L_clip + critic_loss
            optimizer.zero_grad()
            total_loss.backward()
            # clips the gradient norms to 0.5 to avoid gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
