import math
import os
import random
from collections import deque

import numpy as np
import torch
import torch.optim as optim

from networks import PolicyNetwork, ValueNetwork
from ppo import calculate_advantages, estimate_value, estimate_values, ppo_loss, sample_action, sample_actions
from pvp_env import PvpEnv

CHECKPOINT = "checkpoint.pt"
# made by pretrain.py; used instead of a random network when there is no checkpoint to resume
PRETRAINED = "pretrained.pt"
ITERATIONS = 5000
# steps per arena per iteration; all arenas step together, so one iteration is STEPS_PER_ARENA world steps
STEPS_PER_ARENA = 512
EPOCHS = 10
# entropy bonus: higher keeps the policy trying different actions instead of settling early (ppo.py default 0.002)
ENTROPY_COEF = 0.01

# curriculum: an ordered list of opponents. The bot moves to the next stage once it wins PROMOTE_WIN_RATE of its
# recent fights against the current one. A few easy scripted stages teach the basics, then the Carpet PvP bot
# ("Theo", needs the CarpetPvp plugin) at its five difficulty presets.
# the trainee levels are easier than Theo's own beginner preset (slower reactions, more misses, fewer clicks)
STAGES = [("scripted", 0.0), ("scripted", 0.1), ("scripted", 0.2),
          ("carpet", "trainee1"), ("carpet", "trainee2"), ("carpet", "trainee3"),
          ("carpet", "beginner"), ("carpet", "casual"), ("carpet", "average"), ("carpet", "skilled"), ("carpet", "expert")]
PROMOTE_WIN_RATE = 0.6
MIN_FIGHTS_TO_PROMOTE = 20
# the server runs several arenas at once. Arena 0 always fights the current stage's opponent (the anchor: the Carpet
# bot only exists there), which keeps the bot grounded and drives promotion. The other arenas are self-play:
# a mirror match (both sides learn) or a frozen past self
P_MIRROR = 0.5
# share of Theo fights where he plays his ranged (bow) style instead of smp, so arrows actually come at the bot
P_RANGED = 0.3
SNAPSHOT_EVERY = 20
POOL_SIZE = 20

# starting probabilities for a fresh network, by head index (move, jump, sprint, yaw, pitch, attack, use, slot).
# without this a random policy spins its view to the sky, switches items ~85% of steps and spams use/attack;
# yaw/pitch index 3 is "no turn"
HEAD_START_PROBS = {3: {3: 0.5}, 4: {3: 0.5}, 5: {1: 0.25}, 6: {1: 0.2}, 7: {0: 0.95}}


def calm_start(policy):
    with torch.no_grad():
        for head in policy.heads:
            head.weight.mul_(0.01)
            head.bias.zero_()
        for index, probs in HEAD_START_PROBS.items():
            head = policy.heads[index]
            for choice, p in probs.items():
                rest = (1 - p) / (head.out_features - 1)
                head.bias[choice] = math.log(p / rest)


class Trajectory:
    def __init__(self):
        self.states, self.actions, self.log_probs = [], [], []
        self.rewards, self.values, self.ends, self.end_values = [], [], [], []

    def add(self, state, action, log_prob, reward, value, end, end_value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.ends.append(end)
        self.end_values.append(end_value)

    def extend(self, other):
        for name in vars(self):
            getattr(self, name).extend(getattr(other, name))


class Opponent:
    def __init__(self, kind, skill=None, policy=None, style="smp"):
        self.kind = kind  # "scripted", "carpet", "mirror" or "past"
        self.skill = skill  # scripted skill (0..1) or carpet difficulty name
        self.policy = policy
        self.style = style  # carpet only: "smp" (sword, shield, water) or "ranged" (bow at a distance)


def stage_name(stage):
    kind, skill = stage
    return f"scripted {skill:.1f}" if kind == "scripted" else f"Theo {skill}"


class Curriculum:
    def __init__(self, stage=0):
        self.stage = stage
        self.at_stage = deque(maxlen=50)

    @property
    def current(self):
        return STAGES[self.stage]

    @property
    def name(self):
        return stage_name(self.current)

    def is_current(self, opponent):
        return (opponent.kind, opponent.skill) == self.current

    def record(self, opponent, outcome):
        if self.is_current(opponent):
            self.at_stage.append(outcome)

    def win_rate(self):
        return self.at_stage.count(1) / len(self.at_stage) if self.at_stage else 0.0

    def maybe_promote(self):
        if (self.stage < len(STAGES) - 1 and len(self.at_stage) >= MIN_FIGHTS_TO_PROMOTE
                and self.win_rate() >= PROMOTE_WIN_RATE):
            self.stage += 1
            self.at_stage.clear()
            return True
        return False


def pick_opponent(pool, curriculum, env, arena):
    if arena == 0:
        kind, skill = curriculum.current
        style = "ranged" if kind == "carpet" and random.random() < P_RANGED else "smp"
        return Opponent(kind, skill=skill, style=style)
    if random.random() < P_MIRROR or not pool:
        return Opponent("mirror")
    past = PolicyNetwork(env.obs_size, env.head_sizes)
    past.load_state_dict(random.choice(pool))
    past.eval()
    return Opponent("past", policy=past)


def reset(env, opponent, arena):
    if opponent.kind == "carpet":
        return env.reset_carpet(opponent.skill, opponent.style)  # always arena 0
    return env.reset(opponent.skill if opponent.kind == "scripted" else None, arena=arena)


def new_fight(env, pool, curriculum, arena):
    opp = pick_opponent(pool, curriculum, env, arena)
    obs_a, obs_b = reset(env, opp, arena)
    return {"opponent": opp, "obs_a": obs_a, "obs_b": obs_b, "length": 0, "dealt": 0.0, "taken": 0.0}


def end_value(value_net, next_state, terminated, end):
    # died -> nothing comes after (0); cut off by time limit or rollout end -> the critic's estimate
    if terminated:
        return 0.0
    return estimate_value(value_net, next_state) if end else None


def collect(env, policy, value_net, fights, n_steps, pool, curriculum, results):
    """Plays n_steps in every arena at once. A always learns; B also learns in mirror matches.
    `fights` holds each arena's fight in progress, so fights carry across rollouts."""
    n = len(fights)
    trajs_a, trajs_b = [Trajectory() for _ in range(n)], [Trajectory() for _ in range(n)]

    for i in range(n_steps):
        # one batched forward pass for every arena's A, and one for the B sides of mirror matches
        obs_a = [f["obs_a"] for f in fights]
        acts_a, lps_a = sample_actions(policy, obs_a)
        vals_a = estimate_values(value_net, obs_a)
        acts_b, lps_b, vals_b = [None] * n, [None] * n, [None] * n
        mirror = [k for k, f in enumerate(fights) if f["opponent"].kind == "mirror"]
        if mirror:
            obs_b = [fights[k]["obs_b"] for k in mirror]
            acts, lps = sample_actions(policy, obs_b)
            vals = estimate_values(value_net, obs_b)
            for j, k in enumerate(mirror):
                acts_b[k], lps_b[k], vals_b[k] = acts[j], lps[j], vals[j]
        for k, f in enumerate(fights):
            if f["opponent"].kind == "past":
                acts_b[k], _ = sample_action(f["opponent"].policy, f["obs_b"])

        stepped = env.step_all([(k, acts_a[k], acts_b[k]) for k in range(n)])
        for k, (next_a, next_b, r_a, r_b, terminated, truncated, info) in enumerate(stepped):
            f = fights[k]
            opp = f["opponent"]
            f["length"] += 1
            f["dealt"] += info["damage_to_b"]
            f["taken"] += info["damage_to_a"]
            done = terminated or truncated
            end = done or i == n_steps - 1
            trajs_a[k].add(f["obs_a"], acts_a[k], lps_a[k], r_a, vals_a[k], end,
                           end_value(value_net, next_a, terminated, end))
            if opp.kind == "mirror":
                trajs_b[k].add(f["obs_b"], acts_b[k], lps_b[k], r_b, vals_b[k], end,
                               end_value(value_net, next_b, terminated, end))
            if done:
                results.append((opp.kind, opp.skill, info["outcome"], f["length"], f["dealt"], f["taken"]))
                curriculum.record(opp, info["outcome"])
                fights[k] = new_fight(env, pool, curriculum, k)
            else:
                f["obs_a"], f["obs_b"] = next_a, next_b

    # every trajectory ends with end=True, so each one's chain of returns starts fresh after the one before it
    traj = Trajectory()
    for t in trajs_a + trajs_b:
        traj.extend(t)
    return traj


def summary(results, curriculum):
    def rates(rows):
        outcomes = [r[2] for r in rows]
        n = len(outcomes)
        return f"won {outcomes.count(1) / n:.0%} lost {outcomes.count(-1) / n:.0%} draw {outcomes.count(0) / n:.0%} ({n})"

    at_stage = [r for r in results if (r[0], r[1]) == curriculum.current]
    past = [r for r in results if r[0] == "past"]
    mirror = [r for r in results if r[0] == "mirror"]
    parts = [f"stage {curriculum.stage + 1}/{len(STAGES)} {curriculum.name}"]
    if at_stage:
        parts.append(f"vs it: {rates(at_stage)}, dealt {np.mean([r[4] for r in at_stage]):.1f} "
                     f"taken {np.mean([r[5] for r in at_stage]):.1f}, "
                     f"recent win rate {curriculum.win_rate():.0%} of last {len(curriculum.at_stage)} (needs {PROMOTE_WIN_RATE:.0%})")
    if past:
        parts.append(f"vs past selves: {rates(past)}")
    if mirror:
        parts.append(f"mirror fights {len(mirror)}")
    parts.append(f"fight {np.mean([r[3] for r in results]):.0f} steps")
    return " | ".join(parts)


def train():
    env = PvpEnv()
    policy = PolicyNetwork(env.obs_size, env.head_sizes)
    value_net = ValueNetwork(env.obs_size)
    all_params = list(policy.parameters()) + list(value_net.parameters())
    optimizer = optim.Adam(all_params, lr=3e-4)
    pool = deque(maxlen=POOL_SIZE)
    curriculum = Curriculum()
    start = 0

    resumed = False
    if os.path.exists(CHECKPOINT):
        ckpt = torch.load(CHECKPOINT)
        if ckpt.get("obs_size") == env.obs_size and ckpt.get("head_sizes") == env.head_sizes and "stage" in ckpt:
            policy.load_state_dict(ckpt["policy"])
            value_net.load_state_dict(ckpt["value"])
            optimizer.load_state_dict(ckpt["optimizer"])
            pool.extend(ckpt.get("pool", []))
            curriculum.stage = ckpt["stage"]
            start = ckpt["iteration"] + 1
            resumed = True
            print(f"resumed from iteration {ckpt['iteration']} at stage {curriculum.name}, {len(pool)} past selves")
        else:
            print("checkpoint doesn't match this setup, starting fresh")
    if not resumed:
        pretrained = torch.load(PRETRAINED) if os.path.exists(PRETRAINED) else None
        if pretrained and pretrained["obs_size"] == env.obs_size and pretrained["head_sizes"] == env.head_sizes:
            policy.load_state_dict(pretrained["policy"])
            value_net.load_state_dict(pretrained["value"])
            print(f"starting from {PRETRAINED} (imitation of the aimbot)")
        else:
            calm_start(policy)

    print(f"{env.n_arenas} arenas, {STEPS_PER_ARENA} steps each per iteration")
    fights = [new_fight(env, pool, curriculum, k) for k in range(env.n_arenas)]
    results = deque(maxlen=500)

    for iteration in range(start, ITERATIONS):
        traj = collect(env, policy, value_net, fights, STEPS_PER_ARENA, pool, curriculum, results)
        returns, advantages = calculate_advantages(traj.rewards, traj.values, traj.ends, traj.end_values)
        for epoch in range(EPOCHS):
            loss = ppo_loss(traj.states, traj.actions, traj.log_probs, advantages, returns, policy, value_net,
                            entropy_coef=ENTROPY_COEF)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 0.5)
            optimizer.step()

        if iteration % SNAPSHOT_EVERY == 0:
            pool.append({k: v.clone() for k, v in policy.state_dict().items()})

        def save(path):
            torch.save({
                "policy": policy.state_dict(),
                "value": value_net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "pool": list(pool),
                "stage": curriculum.stage,
                "iteration": iteration,
                "obs_size": env.obs_size,
                "head_sizes": env.head_sizes,
            }, path)

        if results:
            print(f"iteration {iteration} | {summary(results, curriculum)}")
        passed = curriculum.name
        if curriculum.maybe_promote():
            # keep the bot that beat each stage, so a later bad strategy can't overwrite a good one
            save(f"checkpoint_stage{curriculum.stage}.pt")
            print(f"  -> beat {passed}; next opponent: {curriculum.name} | saved checkpoint_stage{curriculum.stage}.pt")

        if iteration % 10 == 0:
            save(CHECKPOINT)

    env.close()


if __name__ == "__main__":
    train()
