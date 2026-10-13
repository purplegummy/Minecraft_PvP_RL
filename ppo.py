import numpy as np
import torch


def estimate_value(value_net, state):
    with torch.no_grad():
        return value_net(torch.tensor(state, dtype=torch.float32)).item()


def sample_action(policy, state):
    """One sample per head; the action's log prob is the sum over heads (independent choices multiply)."""
    with torch.no_grad():
        head_probs = policy(torch.tensor(state, dtype=torch.float32))
    action, log_prob = [], 0.0
    for probs in head_probs:
        a = torch.multinomial(probs, 1).item()
        action.append(a)
        log_prob += torch.log(probs[a] + 1e-8).item()
    return action, log_prob


def sample_actions(policy, states):
    """sample_action for a batch of states at once: returns a list of actions and a list of log probs."""
    with torch.no_grad():
        head_probs = policy(torch.tensor(np.asarray(states), dtype=torch.float32))
    picks = [torch.multinomial(probs, 1).squeeze(1) for probs in head_probs]
    log_probs = sum(torch.log(probs.gather(1, pick.unsqueeze(1)).squeeze(1) + 1e-8)
                    for probs, pick in zip(head_probs, picks))
    actions = torch.stack(picks, dim=1).tolist()
    return actions, log_probs.tolist()


def estimate_values(value_net, states):
    with torch.no_grad():
        return value_net(torch.tensor(np.asarray(states), dtype=torch.float32)).reshape(-1).tolist()


def collect_rollout(N, env, policy, value_net, state, episode):
    # `episode` holds {"return", "length"} of the in-progress episode so it survives across rollouts
    states, actions, log_probs, rewards, values, ends, end_values = [], [], [], [], [], [], []
    finished = []  # (return, length, final info) of every episode that ended during this rollout

    for i in range(N):
        a_t, log_prob = sample_action(policy, state)
        value = estimate_value(value_net, state)
        next_state, reward, terminated, truncated, info = env.step(a_t)
        episode["return"] += reward
        episode["length"] += 1

        # the return chain breaks when the episode ends or the rollout runs out of steps.
        # died -> nothing comes after, so restart from 0.
        # cut off by a time limit -> the future still exists, so restart from the critic's estimate.
        end = terminated or truncated or i == N - 1
        if terminated:
            end_value = 0.0
        elif end:
            end_value = estimate_value(value_net, next_state)
        else:
            end_value = None

        states.append(state)
        actions.append(a_t)
        log_probs.append(log_prob)
        rewards.append(reward)
        values.append(value)
        ends.append(end)
        end_values.append(end_value)

        state = next_state
        if terminated or truncated:
            finished.append((episode["return"], episode["length"], info))
            episode["return"], episode["length"] = 0.0, 0
            state, info = env.reset()

    return (states, actions, log_probs, rewards, values, ends, end_values), state, finished


def calculate_advantages(rewards, values, ends, end_values, gamma=0.99, lam=0.95):
    # GAE: delta_t = r_t + γ·V(next) − V(s_t) is how much better step t went than the critic expected;
    # advantages sum those surprises forward, fading by γ·λ, instead of using the whole noisy rest of the fight
    n = len(rewards)
    advantages = np.zeros(n)
    gae = 0.0
    # walk backwards so each advantage can reuse the one after it, like the old running G
    for t in reversed(range(n)):
        if ends[t]:
            next_value = end_values[t]  # 0 if knocked out, the critic's estimate if cut off
            gae = 0.0                   # nothing after this point belongs to the same fight
        else:
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae

    returns = advantages + np.array(values)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return returns.tolist(), advantages.tolist()


def ppo_loss(states, actions, old_log_probs, advantages, returns, policy, value_net, eps=0.2, entropy_coef=0.002):
    states_t = torch.tensor(np.array(states), dtype=torch.float32)
    actions_t = torch.tensor(np.array(actions), dtype=torch.long)  # (N, heads)
    old_log_probs_t = torch.tensor(old_log_probs, dtype=torch.float32)
    advantages_t = torch.tensor(advantages, dtype=torch.float32)
    returns_t = torch.tensor(returns, dtype=torch.float32)

    head_probs = policy(states_t)
    rows = torch.arange(len(actions_t))
    # each head is an independent choice, so log probs and entropies add up across heads
    new_log_probs = sum(torch.log(p[rows, actions_t[:, h]] + 1e-8) for h, p in enumerate(head_probs))
    predicted_values = value_net(states_t)

    # do the division in log space
    ratio = torch.exp(new_log_probs - old_log_probs_t)
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps)
    policy_loss = -torch.mean(torch.min(ratio * advantages_t, clipped * advantages_t))

    value_loss = torch.mean((returns_t - predicted_values) ** 2)

    # summed over all 8 heads, so the per-head coefficient is kept small or it drowns weak rewards
    entropy = sum(-torch.mean(torch.sum(p * torch.log(p + 1e-8), dim=-1)) for p in head_probs)

    return policy_loss + 0.5 * value_loss - entropy_coef * entropy
