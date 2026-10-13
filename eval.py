"""Plays the saved bot and shows where its reward comes from and what it does.

usage: python eval.py [opponent] [fights] [--sample]
  opponent: a scripted skill like 0.4, or a Carpet difficulty like carpet:average
"""
import os
import sys

import numpy as np
import torch

import obs_layout as L
from fight import best_action
from networks import PolicyNetwork
from ppo import sample_action
from pvp_env import PvpEnv

args = [a for a in sys.argv[1:] if not a.startswith("--")]
opponent = args[0] if args else "0.4"
carpet = opponent.startswith("carpet:")
skill = opponent.split(":", 1)[1] if carpet else float(opponent)
fights = int(args[1]) if len(args) > 1 else 10
sample = "--sample" in sys.argv

env = PvpEnv()
# your own training checkpoint if there is one, otherwise the trained bot that ships with the repo
ckpt = torch.load("checkpoint.pt" if os.path.exists("checkpoint.pt") else "models/pvpbot.pt")
policy = PolicyNetwork(env.obs_size, env.head_sizes)
policy.load_state_dict(ckpt["policy"])
policy.eval()
print(f"bot from iteration {ckpt['iteration']} vs {'Theo ' + skill if carpet else f'scripted skill {skill}'}, "
      f"{'sampling' if sample else 'best guess'}\n")

rows = []
for fight in range(fights):
    obs, _ = env.reset_carpet(skill) if carpet else env.reset(skill)
    total = dealt = taken = 0.0
    steps = attacks = uses = switches = hooked = opp_burning = blocking = ignored_attacks = 0
    lava_placed_at = None
    held = np.zeros(len(L.ITEM_KINDS))
    last_held = None
    distances, pitches, yaw_errors, pitch_errors = [], [], [], []
    done = False
    while not done:
        action = sample_action(policy, obs)[0] if sample else best_action(policy, obs)
        kind = int(obs[L.HELD].argmax())
        held[kind] += 1
        switches += last_held is not None and kind != last_held
        last_held = kind
        attacks += action[5]
        blocking += obs[L.BLOCKING] > 0.5
        ignored_attacks += action[5] == 1 and obs[L.USING] > 0.5
        uses += action[6]
        hooked += obs[L.HOOKING_OPP] > 0.5
        opp_burning += obs[L.OPP_ON_FIRE] > 0.5 or obs[L.OPP_IN_LAVA] > 0.5
        if lava_placed_at is None and obs[L.LAVA_BUCKETS] < 0.5:
            lava_placed_at = steps
        distances.append(obs[L.OPP_DISTANCE] * 16)
        pitches.append(obs[L.PITCH] * 90)
        yaw_errors.append(abs(np.degrees(np.arctan2(obs[L.OPP_YAW_SIN], obs[L.OPP_YAW_COS]))))
        pitch_errors.append(abs(obs[L.OPP_PITCH_ERROR] * 90))
        obs, _, reward, _, terminated, truncated, info = env.step(action)
        total += reward
        dealt += info["damage_to_b"]
        taken += info["damage_to_a"]
        steps += 1
        done = terminated or truncated
    outcome = info["outcome"]
    damage_part = (dealt - taken) / 20
    aim_part = total - damage_part - outcome
    rows.append((outcome, total, damage_part, aim_part, dealt, taken))
    top_item = L.ITEM_KINDS[int(held.argmax())]
    print(f"fight {fight}: {({1: 'won', -1: 'lost', 0: 'draw'})[outcome]:4s} {steps:3d} steps | "
          f"reward {total:+.2f} = damage {damage_part:+.2f} + win/loss {outcome:+d} + aim/tiebreak {aim_part:+.2f} | "
          f"dealt {dealt:4.1f} taken {taken:4.1f} | attack {attacks / steps:.0%} use {uses / steps:.0%} "
          f"switch {switches / steps:.0%} | shield up {blocking / steps:.0%}, attacks ignored {ignored_attacks}/{attacks} | "
          f"mostly {top_item} ({held.max() / steps:.0%}) | "
          f"opponent hooked {hooked / steps:.0%} | opponent burning {opp_burning / steps:.0%} | "
          f"lava placed {'at step ' + str(lava_placed_at) if lava_placed_at is not None else 'never'} | "
          f"avg distance {np.mean(distances):.1f} | "
          f"pitch {np.mean(pitches):+.0f} (negative = up) | aim off by {np.mean(yaw_errors):.0f} deg sideways, "
          f"{np.mean(pitch_errors):.0f} deg vertically")

outcomes = [r[0] for r in rows]
print(f"\nwon {outcomes.count(1)}, lost {outcomes.count(-1)}, draw {outcomes.count(0)} | "
      f"avg reward {np.mean([r[1] for r in rows]):+.2f} (damage {np.mean([r[2] for r in rows]):+.2f}, "
      f"aim {np.mean([r[3] for r in rows]):+.2f}) | avg dealt {np.mean([r[4] for r in rows]):.1f} "
      f"taken {np.mean([r[5] for r in rows]):.1f}")
env.close()
