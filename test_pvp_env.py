"""Smoke test: random actions against the scripted opponent and in a random-vs-random self-play fight.

usage: python test_pvp_env.py [scripted skill] [episodes] [--shutdown]
"""
import sys
import time

import numpy as np

from pvp_env import PvpEnv

args = [a for a in sys.argv[1:] if not a.startswith("--")]
skill = float(args[0]) if args else 0.0
episodes = int(args[1]) if len(args) > 1 else 2

env = PvpEnv()
print(f"obs size {env.obs_size}, heads {env.head_sizes}, max {env.max_steps} steps")


def random_action():
    return [np.random.randint(n) for n in env.head_sizes]


for label, scripted in [(f"vs scripted skill {skill}", skill), ("self-play random vs random", None)]:
    for ep in range(episodes):
        obs_a, obs_b = env.reset(scripted)
        total_a = total_b = 0.0
        steps, done, start = 0, False, time.time()
        while not done:
            obs_a, obs_b, r_a, r_b, terminated, truncated, info = env.step(
                random_action(), None if scripted is not None else random_action())
            total_a += r_a
            total_b += r_b
            steps += 1
            done = terminated or truncated
        result = {1: "A won", -1: "A lost", 0: "draw"}[info["outcome"]]
        print(f"{label}, episode {ep}: {result} after {steps} steps, rewards A {total_a:.2f} B {total_b:.2f}, "
              f"{steps / (time.time() - start):.0f} steps/s, obs ok: {obs_a.shape == obs_b.shape == (env.obs_size,)}")

if "--shutdown" in sys.argv:
    env.shutdown_server()
env.close()
