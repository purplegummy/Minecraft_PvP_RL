"""Fight the trained bot yourself. Join the server first, then run: python fight.py [--sample]

By default the bot takes its most likely choice for every part of the action. --sample makes it pick
randomly by its probabilities like during training, which adds a lot of exploration noise.
Stop train.py before running this; training freezes the world, this runs it in real time.
"""
import os
import sys
import time

import torch

import obs_layout as L
from networks import PolicyNetwork
from ppo import sample_action
from pvp_env import PvpEnv

# your own training checkpoint if there is one, otherwise the trained bot that ships with the repo
CHECKPOINT = "checkpoint.pt" if os.path.exists("checkpoint.pt") else "models/pvpbot.pt"
SECONDS_BETWEEN_ROUNDS = 3


def best_action(policy, obs):
    with torch.no_grad():
        return [int(p.argmax()) for p in policy(torch.tensor(obs, dtype=torch.float32))]


class Habits:
    """How often the bot attacks, presses use and switches items, and what it holds."""

    def __init__(self):
        self.steps = self.attacks = self.uses = self.switches = 0
        self.held = [0] * len(L.ITEM_KINDS)
        self.last_held = None

    def record(self, action, obs):
        self.steps += 1
        self.attacks += action[5]
        self.uses += action[6]
        held = int(obs[L.HELD].argmax())
        self.switches += self.last_held is not None and held != self.last_held
        self.last_held = held
        self.held[held] += 1

    def report(self):
        n = max(self.steps, 1)
        held = ", ".join(f"{name} {count / n:.0%}" for name, count in zip(L.ITEM_KINDS, self.held) if count)
        return (f"  bot habits: attack {self.attacks / n:.0%} of steps, use {self.uses / n:.0%}, "
                f"item switch {self.switches / n:.0%} | held: {held}")


def wait_for_player(env):
    """reset_human fails while nobody is online, so keep trying until you've joined."""
    waiting = False
    while True:
        try:
            return env.reset_human()[0]
        except RuntimeError as e:
            if "no real player" not in str(e):
                raise
            if not waiting:
                print("waiting for you to join the server...")
                waiting = True
            time.sleep(2)


def main():
    env = PvpEnv()
    ckpt = torch.load(CHECKPOINT)
    if ckpt.get("obs_size") != env.obs_size or ckpt.get("head_sizes") != env.head_sizes:
        raise SystemExit("checkpoint.pt doesn't match the current plugin; train a new one first")
    policy = PolicyNetwork(env.obs_size, env.head_sizes)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    sample = "--sample" in sys.argv
    print(f"loaded bot from iteration {ckpt['iteration']} (curriculum stage {ckpt.get('stage', 0) + 1}), "
          f"{'sampling like training' if sample else 'playing its best guess'}. Ctrl+C to stop.")

    score = {1: 0, -1: 0, 0: 0}
    try:
        while True:
            obs = wait_for_player(env)
            habits = Habits()
            done = False
            while not done:
                action = sample_action(policy, obs)[0] if sample else best_action(policy, obs)
                habits.record(action, obs)
                obs, _, _, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            score[info["outcome"]] += 1
            print(f"bot {score[1]} - you {score[-1]} - draws {score[0]}")
            print(habits.report())
            time.sleep(SECONDS_BETWEEN_ROUNDS)
    except KeyboardInterrupt:
        pass
    env.close()


if __name__ == "__main__":
    main()
