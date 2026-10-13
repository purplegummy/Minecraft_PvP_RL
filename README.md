# PvpBot: a Minecraft PvP agent trained with PPO

A neural network that learns to fight in Minecraft. It is trained with PPO (written in PyTorch, no RL library)
inside a real Paper server, against scripted opponents, the [Carpet PvP](https://github.com/AndrewCTF/Carpet-PvP)
fighting AI ("Theo") and itself.

**Replays:** open [`replay/index.html`](replay/index.html) in a browser to watch recorded fights in 3D.

## Results

Final bot (PPO update 1140) against a freshly spawned Theo at expert difficulty, sword style:

| Evaluation | Won | Lost | Draw |
|---|---|---|---|
| 30 fights, actions sampled as in training | 30 | 0 | 0 |
| 30 fights, most likely action | 29 | 0 | 1 |

The same Theo beats the earlier checkpoints (0–10 before reinforcement learning, 7–3 at update 302), so he is
really fighting. Logs are in [`logs/`](logs).

Limits:
- The bot wins mostly with the bow at close range. It beats this opponent reliably; it is not all-round PvP skill.
- Theo fights with the bot's kit (no totems or potions), so this is not Carpet's full expert setup.
- Against Theo's bow style it is much weaker: in 3 recorded fights it won twice by setting him on fire with lava,
  and drew once.

## How it works

- **Environment.** A Paper plugin ([`plugin/`](plugin)) spawns server-side fake players, freezes the world and steps
  it 2 ticks at a time on request, over a TCP line protocol. Ten arenas run side by side and step together.
- **Observation and actions.** 284 numbers per step (own state, opponent, nearby projectiles, a 7×7×3 grid of lava,
  water and blocks). The action is 8 independent choices: movement, jump, sprint, turn left/right, look up/down,
  attack, use item, hotbar slot.
- **PPO** ([`ppo.py`](ppo.py)): one shared body with a softmax head per action choice, a separate value network,
  GAE (γ 0.99, λ 0.95), clipped objective (ε 0.2), entropy bonus, Adam.
- **Rewards.** Damage dealt minus damage taken, ±1 for a knockout, a health tiebreak on timeout, and potential-based
  aim shaping. Fire and lava damage earn nothing, after the bot learned to farm them.
- **Imitation warm start** ([`pretrain.py`](pretrain.py)). PPO from a random policy never learned to land a hit
  ([log](logs/no-pretraining-run-train.txt)), so the networks first copy a simple hand-written fighter.
- **Curriculum and self-play** ([`train.py`](train.py)). 11 opponents of rising difficulty (scripted, then Theo
  from weakened "trainee" settings up to expert), with 9 arenas of self-play against itself and frozen past versions.

## Layout

| Path | What it is |
|---|---|
| `plugin/` | Java Paper plugin: the environment (arenas, fake players, observations, rewards) |
| `thirdparty/carpet-pvp-freeze.patch` | Patch so Theo only thinks on ticks the frozen world actually runs |
| `pvp_env.py` | Python client for the plugin |
| `networks.py`, `ppo.py` | Policy/value networks and PPO |
| `pretrain.py`, `train.py` | Imitation warm start, then PPO training |
| `eval.py`, `fight.py` | Test the bot against an opponent, or fight it yourself |
| `record_fights.py`, `replay/` | Record fights and the 3D replay viewer |
| `models/pvpbot.pt` | The trained bot |
| `logs/` | Training and evaluation logs |
| `docs/ppo-notes.md` | My notes on the PPO loss and an early NumPy version of the networks |

## Running it

Needs Python 3.11+, Java 25 and a [Paper](https://papermc.io) 26.2 server.

1. Put the Paper jar in `server/` as `paper.jar` and run it once (`java -jar paper.jar --nogui`) to accept the EULA
   and create the files. In `server.properties` set `level-type=minecraft:flat`, `max-players=50`,
   `view-distance=4` and `simulation-distance=3`.
2. Build the plugin: `powershell -File plugin/build.ps1` (compiles against the server's jars and copies
   `PvpEnv.jar` into `server/plugins`).
3. For Theo: build Carpet PvP's Paper plugin with `thirdparty/carpet-pvp-freeze.patch` applied and put the jar in
   `server/plugins`.
4. Start the server, then:

```
pip install -r requirements.txt
python eval.py carpet:expert 10     # watch the trained bot's stats against Theo
python fight.py                     # join the server and fight it yourself
python pretrain.py                  # imitation warm start -> pretrained.pt
python train.py                     # PPO training -> checkpoint.pt
python record_fights.py             # record fights for the replay viewer
```
