"""Records fights for the replay viewer in replay/: every step's positions, look direction, items, health and
projectiles, plus the bot's own estimate of how the fight is going (its critic). Every fight played is kept,
wins and losses alike.

usage: python record_fights.py   (needs the server and nothing else connected to it)
"""
import json
import time

import torch

from networks import PolicyNetwork, ValueNetwork
from ppo import estimate_value, sample_action
from pvp_env import PvpEnv

OUTPUT = "replay/fights.js"
# (checkpoint, label, Theo difficulty, Theo style, fights)
PLAN = [
    ("checkpoint.pt", "Final bot", "expert", "smp", 8),
    ("checkpoint.pt", "Final bot", "expert", "ranged", 3),
    ("checkpoint_stage8.pt", "Mid-training (stage 8)", "expert", "smp", 2),
    ("pretrained.pt", "Imitation start (before RL)", "expert", "smp", 2),
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
    for path, label, difficulty, style, count in PLAN:
        policy, value, iteration = load(env, path)
        for n in range(count):
            obs, _ = env.reset_carpet(difficulty, style)
            frames, dealt, taken, done = [], 0.0, 0.0, False
            while True:
                frame = snapshot(env, items)
                frame["v"] = round(estimate_value(value, obs), 3)
                frames.append(frame)
                if done:
                    break
                obs, _, _, _, terminated, truncated, info = env.step(sample_action(policy, obs)[0])
                frame["hitA"], frame["hitB"] = round(info["damage_to_a"], 2), round(info["damage_to_b"], 2)
                dealt += info["damage_to_b"]
                taken += info["damage_to_a"]
                done = terminated or truncated
            outcome = info["outcome"]
            fights.append({"label": label, "checkpoint": path, "iteration": iteration,
                           "opponent": f"Theo {difficulty} ({style})", "outcome": outcome,
                           "dealt": round(dealt, 1), "taken": round(taken, 1), "frames": frames})
            print(f"{label} vs Theo {difficulty} {style} #{n + 1}: "
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
