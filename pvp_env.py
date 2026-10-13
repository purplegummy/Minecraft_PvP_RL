import socket

import numpy as np


class PvpEnv:
    """Wrapper around the PvpEnv Paper plugin: fighter A ("PvpBot") vs fighter B ("Sparring").

    B is either the plugin's scripted opponent or controlled from Python (self-play).
    An action is one integer per head: move, jump, sprint, yaw, pitch, attack, use, slot.
    info["outcome"] on the final step is 1 (A won), -1 (A lost) or 0 (time ran out);
    info["damage_to_a"] / info["damage_to_b"] are this step's damage after armor.

    The server runs n_arenas fights side by side. reset/step use arena 0 unless told otherwise; step_all advances
    several arenas in one go (the world ticks once for all of them), which is much faster than one at a time.
    The Carpet bot and human fights only run in arena 0.
    """

    def __init__(self, host="127.0.0.1", port=5555):
        self.sock = socket.create_connection((host, port))
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.file = self.sock.makefile("rw", encoding="utf-8", newline="\n")

        obs_size, heads, max_steps, *arenas = self._request("info").split(";")
        self.obs_size = int(obs_size)
        self.head_sizes = [int(h) for h in heads.split(",")]
        self.max_steps = int(max_steps)
        self.n_arenas = int(arenas[0]) if arenas else 1

    def _request(self, command):
        self.file.write(command + "\n")
        self.file.flush()
        line = self.file.readline().strip()
        if not line or line.startswith("error"):
            raise RuntimeError(f"PvpEnv plugin: {line or 'connection closed'}")
        return line

    def _send(self, command):
        return self._parse(self._request(command))

    @staticmethod
    def _parse(result):
        obs_a, obs_b, r_a, r_b, terminated, truncated, outcome, dmg_a, dmg_b = result.split(";")
        return (
            np.array(obs_a.split(","), dtype=np.float32),
            np.array(obs_b.split(","), dtype=np.float32),
            float(r_a),
            float(r_b),
            terminated == "1",
            truncated == "1",
            {"outcome": int(outcome), "damage_to_a": float(dmg_a), "damage_to_b": float(dmg_b)},
        )

    def reset(self, scripted_skill=None, arena=0):
        """scripted_skill in [0, 1] uses the scripted opponent; None means Python controls B."""
        command = "reset external" if scripted_skill is None else f"reset script {scripted_skill:.3f}"
        obs_a, obs_b, *_ = self._send(f"@{arena} {command}")
        return obs_a, obs_b

    def reset_carpet(self, difficulty, style="smp"):
        """B becomes a Carpet PvP bot (trainee1-3, beginner .. expert; style smp or ranged). Needs the CarpetPvp plugin."""
        obs_a, obs_b, *_ = self._send(f"reset carpet {difficulty} {style}")
        return obs_a, obs_b

    def reset_human(self):
        """B becomes the real player online; the world runs in real time until the next training reset."""
        obs_a, obs_b, *_ = self._send("reset human")
        return obs_a, obs_b

    @staticmethod
    def _actions(action_a, action_b):
        text = " ".join(str(int(x)) for x in action_a)
        if action_b is not None:
            text += " | " + " ".join(str(int(x)) for x in action_b)
        return text

    def step(self, action_a, action_b=None):
        return self._send("step " + self._actions(action_a, action_b))

    def step_all(self, entries):
        """entries: list of (arena, action_a, action_b or None). Returns one step() result per entry, in order."""
        command = "stepall " + " / ".join(f"{arena}:{self._actions(a, b)}" for arena, a, b in entries)
        return [self._parse(part) for part in self._request(command).split("#")]

    def shutdown_server(self):
        self._request("shutdown")

    def close(self):
        self.sock.close()
