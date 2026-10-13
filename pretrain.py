"""Imitation warm-start: record a hand-coded aimbot fighting, train the networks to copy it, save pretrained.pt.

train.py starts from pretrained.pt when there is no checkpoint to resume, so PPO begins with a bot that already
turns toward the opponent, closes in and swings when the sword is charged.

usage: python pretrain.py [steps to record]   (needs the server and nothing else connected to it)
"""
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

import obs_layout as L
from networks import PolicyNetwork, ValueNetwork
from pvp_env import PvpEnv

OUTPUT = "pretrained.pt"
RECORD_STEPS = 16000
# opponents to record against: the first curriculum stages, plus ranged Theo so arrows come in to be blocked
OPPONENTS = [("scripted", 0.0, None), ("scripted", 0.1, None), ("scripted", 0.2, None),
             ("carpet", "trainee1", "smp"), ("carpet", "trainee1", "ranged"), ("carpet", "trainee2", "ranged")]
# share of steps where the executed action is random; the label is still the teacher's choice there
NOISE = 0.2
GAMMA = 0.99
EPOCHS = 30
BATCH = 256
LABEL_SMOOTHING = 0.1

YAW_BINS = [-30, -12, -4, 0, 4, 12, 30]
PITCH_BINS = [-20, -6, -2, 0, 2, 6, 20]


def nearest(bins, value):
    return int(np.argmin([abs(b - value) for b in bins]))


def incoming_arrow(obs):
    for i in range(L.N_PROJECTILES):
        base = L.PROJECTILES + i * L.PROJECTILE_SIZE
        if obs[base + L.PROJECTILE_PRESENT] < 0.5 or obs[base + L.PROJECTILE_ARROW] < 0.5:
            continue
        pos, vel = obs[base:base + 3] * 16, obs[base + 3:base + 6] * 3
        if np.linalg.norm(pos) < 12 and np.dot(pos, vel) < 0:  # close and closing
            return True
    return False


def teacher(obs):
    """A simple, sensible fighter for the network to copy before PPO takes over. In priority order:
    burning -> water at its feet; arrow incoming or opponent drawing -> shield up; hurt with room -> back off and eat;
    far away -> bow; otherwise walk in and swing the sword only when it's charged and on target."""
    yaw_err = np.degrees(np.arctan2(obs[L.OPP_YAW_SIN], obs[L.OPP_YAW_COS]))
    pitch_err = obs[L.OPP_PITCH_ERROR] * 90
    dist = obs[L.OPP_DISTANCE] * 16
    held = L.ITEM_KINDS[int(obs[L.HELD].argmax())]
    look = [nearest(YAW_BINS, yaw_err), nearest(PITCH_BINS, pitch_err)]

    def select(item):
        return 0 if held == item else L.HOTBAR.index(item) + 1

    if obs[L.ON_FIRE] > 0.5 and obs[L.WATER_BUCKETS] > 0.5:
        ready = held == "water bucket" and obs[L.PITCH] * 90 > 60
        return [0, 0, 0, nearest(YAW_BINS, 0), nearest(PITCH_BINS, 20), 0, int(ready), select("water bucket")]

    # a shield takes ~5 ticks to come up, so guard as soon as the opponent draws, not when the arrow is close
    opp_drawing = (L.ITEM_KINDS[int(obs[L.OPP_HELD].argmax())] == "bow" and obs[L.OPP_USING] > 0.5
                   and obs[L.OPP_USE_PROGRESS] > 0.25)
    if incoming_arrow(obs) or (opp_drawing and dist > 4):
        guarding = held in ("sword", "axe")  # either raises the off-hand shield
        return [0, 0, 0, *look, 0, int(guarding), 0 if guarding else select("sword")]

    if obs[L.HEALTH] < 0.4 and dist > 5 and obs[L.APPLES] > 0:
        return [2, 0, 0, *look, 0, int(held == "food"), select("food")]

    # stick with the bow down to 8 blocks once it's out, so a draw isn't cancelled by distance wobbling around 12
    if obs[L.ARROWS] > 0 and (dist > 12 or (held == "bow" and dist > 8)):
        aim_pitch = pitch_err - 1.5 * dist / 15  # a little high for arrow drop (negative pitch is up)
        aimed = abs(yaw_err) < 6 and abs(aim_pitch) < 6
        drawn = obs[L.USING] > 0.5 and obs[L.USE_PROGRESS] >= 1.0
        use = int(held == "bow" and not (drawn and aimed))  # letting go of a full draw on target fires
        return [0, 0, 0, nearest(YAW_BINS, yaw_err), nearest(PITCH_BINS, aim_pitch), 0, use, select("bow")]

    forward = dist > 2.5 and abs(yaw_err) < 60
    attack = held == "sword" and obs[L.CROSSHAIR_ON_OPP] > 0.5 and obs[L.COOLDOWN] >= 0.9
    return [1 if forward else 0, 0, 1 if forward else 0, *look, int(attack), 0, select("sword")]


def record(env, steps):
    states, labels, rewards, ends = [], [], [], []
    wins = fights = 0

    def reset():
        kind, skill, style = random.choice(OPPONENTS)
        return (env.reset_carpet(skill, style) if kind == "carpet" else env.reset(skill))[0]

    obs = reset()
    for i in range(steps):
        label = teacher(obs)
        action = [random.randrange(n) for n in env.head_sizes] if random.random() < NOISE else label
        next_obs, _, reward, _, terminated, truncated, info = env.step(action)
        states.append(obs)
        labels.append(label)
        rewards.append(reward)
        done = terminated or truncated
        ends.append(done)
        if done:
            fights += 1
            wins += info["outcome"] == 1
            obs = reset()
        else:
            obs = next_obs
        if (i + 1) % 2000 == 0:
            print(f"  recorded {i + 1}/{steps} steps, teacher won {wins}/{fights} fights so far")
    return np.array(states, dtype=np.float32), np.array(labels), rewards, ends


def discounted_returns(rewards, ends):
    returns, g = [], 0.0
    for r, end in zip(reversed(rewards), reversed(ends)):
        if end:
            g = 0.0
        g = r + GAMMA * g
        returns.append(g)
    return np.array(returns[::-1], dtype=np.float32)


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else RECORD_STEPS
    env = PvpEnv()
    print(f"recording {steps} steps of the teacher ({NOISE:.0%} random actions mixed in)...")
    states, labels, rewards, ends = record(env, steps)
    env.close()
    returns = discounted_returns(rewards, ends)

    policy = PolicyNetwork(env.obs_size, env.head_sizes)
    value_net = ValueNetwork(env.obs_size)
    optimizer = torch.optim.Adam(list(policy.parameters()) + list(value_net.parameters()), lr=1e-3)
    x = torch.tensor(states)
    y = torch.tensor(labels, dtype=torch.long)
    g = torch.tensor(returns)

    for epoch in range(EPOCHS):
        order = torch.randperm(len(x))
        total_policy = total_value = 0.0
        for start in range(0, len(x), BATCH):
            idx = order[start:start + BATCH]
            head_probs = policy(x[idx])
            # cross-entropy per head against the teacher's choice; the network outputs probabilities, so use their log
            policy_loss = sum(F.nll_loss(torch.log(p + 1e-8), y[idx, h]) for h, p in enumerate(head_probs))
            smoothing = sum(-torch.log(p + 1e-8).mean() for p in head_probs)
            value_loss = F.mse_loss(value_net(x[idx]), g[idx])
            loss = (1 - LABEL_SMOOTHING) * policy_loss + LABEL_SMOOTHING * smoothing + 0.5 * value_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_policy += policy_loss.item() * len(idx)
            total_value += value_loss.item() * len(idx)
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            with torch.no_grad():
                agree = [(p.argmax(-1) == y[:, h]).float().mean().item() for h, p in enumerate(policy(x))]
            print(f"epoch {epoch}: policy loss {total_policy / len(x):.3f}, value loss {total_value / len(x):.3f}, "
                  f"matches teacher on aim {agree[3]:.0%}/{agree[4]:.0%}, attack {agree[5]:.0%}, move {agree[0]:.0%}")

    torch.save({"policy": policy.state_dict(), "value": value_net.state_dict(),
                "obs_size": env.obs_size, "head_sizes": env.head_sizes}, OUTPUT)
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
