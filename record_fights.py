"""Records fights for the replay viewer in replay/: every step's positions, look direction, items, health and
projectiles, plus the bot's own estimate of how the fight is going (its critic). Every fight played is kept,
wins and losses alike. Opponents are the Carpet bot (Theo) or the bot itself (self-play).

usage: python record_fights.py   (needs the server and nothing else connected to it)
"""
import json
import time

import torch

from networks import PolicyNetwork, ValueNetwork
from ppo import estimate_value, sample_action
from pvp_env import PvpEnv

OUTPUT = "replay/fights.js"
# (checkpoint, label, opponent, fights). Opponent: ("theo", difficulty, style) or ("self", checkpoint, name)
PLAN = [
    ("checkpoint.pt", "Final bot", ("theo", "expert", "smp"), 8),
    ("checkpoint.pt", "Final bot", ("theo", "expert", "ranged"), 3),
    ("checkpoint_stage8.pt", "Mid-training (stage 8)", ("theo", "expert", "smp"), 2),
    ("pretrained.pt", "Imitation start (before RL)", ("theo", "expert", "smp"), 2),
    ("checkpoint.pt", "Final bot", ("self", "checkpoint.pt", "Mirror (final bot)"), 8),
    ("checkpoint.pt", "Final bot", ("self", "checkpoint_stage10.pt", "Past self (update 302)"), 4),
]
FIGHTER_KEYS = ["x", "y", "z", "yaw", "pitch", "health", "absorption", "max"]
FLAG_KEYS = ["blocking", "using", "burning"]


def load(env, path):
    ckpt = torch.load(path, weights_only=False)
    policy = PolicyNetwork(env.obs_size, env.head_sizes)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    value = ValueNetwork(env.obs_size)
    value.load_state_dict(ckpt["value"])
    value.eval()
    return policy, value, ckpt.get("iteration")


def snapshot(env, items):
    """Compact frame: each fighter as [x, y, z, yaw, pitch, health, absorption, max health, held item, flags]."""
    state = json.loads(env._request("state"))

    def fighter(f):
        if f["held"] not in items:
            items.append(f["held"])
        flags = sum(1 << i for i, k in enumerate(FLAG_KEYS) if f[k])
        return [round(f[k], 2) for k in FIGHTER_KEYS] + [items.index(f["held"]), flags]

    return {"a": fighter(state["a"]), "b": fighter(state["b"]),
            "p": [[p[0]] + [round(v, 2) for v in p[1:]] for p in state["projectiles"]],
            "l": state["liquids"]}


def main():
    env = PvpEnv()
    items, fights = [], []
    for path, label, opponent, count in PLAN:
        policy, value, iteration = load(env, path)
        if opponent[0] == "theo":
            _, difficulty, style = opponent
            opp_label, opp_name, rival = f"Theo {difficulty} ({style})", "Theo", None
        else:
            _, rival_path, opp_name = opponent
            rival, _, rival_iteration = load(env, rival_path)
            opp_label = f"self-play: {opp_name}"
        for n in range(count):
            if rival is None:
                obs, _ = env.reset_carpet(difficulty, style)
            else:
                obs, obs_b = env.reset(None)
            frames, dealt, taken, done = [], 0.0, 0.0, False
            while True:
                frame = snapshot(env, items)
                frame["v"] = round(estimate_value(value, obs), 3)
                frames.append(frame)
                if done:
                    break
                action_b = sample_action(rival, obs_b)[0] if rival is not None else None
                obs, obs_b, _, _, terminated, truncated, info = env.step(sample_action(policy, obs)[0], action_b)
                frame["hitA"], frame["hitB"] = round(info["damage_to_a"], 2), round(info["damage_to_b"], 2)
                dealt += info["damage_to_b"]
                taken += info["damage_to_a"]
                done = terminated or truncated
            outcome = info["outcome"]
            fights.append({"label": label, "checkpoint": path, "iteration": iteration,
                           "opponent": opp_label, "opponentName": opp_name, "selfPlay": rival is not None,
                           "outcome": outcome,
                           "dealt": round(dealt, 1), "taken": round(taken, 1), "frames": frames})
            print(f"{label} vs {opp_label} #{n + 1}: "
                  f"{({1: 'won', -1: 'lost', 0: 'draw'})[outcome]} in {len(frames) - 1} steps, "
                  f"dealt {dealt:.1f} taken {taken:.1f}", flush=True)
    env.close()

    data = {"recorded": time.strftime("%Y-%m-%d"), "stepSeconds": 0.1, "items": items,
            "fighterKeys": FIGHTER_KEYS + ["held", "flags"], "flagKeys": FLAG_KEYS, "fights": fights}
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write("window.REPLAY = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print(f"saved {len(fights)} fights to {OUTPUT}")


if __name__ == "__main__":
    main()
