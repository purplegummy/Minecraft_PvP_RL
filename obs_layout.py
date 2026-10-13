"""Where things live in a fighter's observation. Mirrors Fighter.observe() in the plugin; keep the two in sync."""

ITEM_KINDS = ["empty", "sword", "axe", "mace", "bow", "crossbow", "rod", "wind charge",
              "lava bucket", "water bucket", "bucket", "pearl", "snowball", "food", "shield", "other"]
_N = len(ITEM_KINDS)

# hotbar slot index -> slot-head action is index + 1 (0 keeps the current item)
HOTBAR = ["sword", "bow", "rod", "wind charge", "lava bucket", "pearl", "axe", "food", "water bucket"]

# own state
HEALTH = 0
ABSORPTION = 1
COOLDOWN = 6
USING = 7
USE_PROGRESS = 8
BLOCKING = 9
HELD = slice(10, 10 + _N)
ARROWS = 10 + _N
WIND_CHARGES = 11 + _N
PEARLS = 12 + _N
APPLES = 13 + _N
LAVA_BUCKETS = 14 + _N
WATER_BUCKETS = 15 + _N
SHIELD_COOLDOWN = 18 + _N
PITCH = 19 + _N
ON_FIRE = 22 + _N
IN_LAVA = 23 + _N
IN_WATER = 24 + _N
SELF_SIZE = 27 + _N

# opponent, as seen by this fighter
_O = SELF_SIZE
OPP_DISTANCE = _O + 3
OPP_HEALTH = _O + 7
OPP_USING = _O + 10
OPP_USE_PROGRESS = _O + 11
OPP_BLOCKING = _O + 12
OPP_SHIELD_COOLDOWN = _O + 13
OPP_HELD = slice(_O + 14, _O + 14 + _N)
OPP_ON_FIRE = _O + 16 + _N
OPP_IN_LAVA = _O + 17 + _N
OPP_IN_WATER = _O + 18 + _N
OPP_YAW_SIN = _O + 19 + _N
OPP_YAW_COS = _O + 20 + _N
OPP_PITCH_ERROR = _O + 21 + _N
CROSSHAIR_ON_OPP = _O + 23 + _N
HOOKING_OPP = _O + 24 + _N
OPP_SIZE = 26 + _N

# incoming projectiles (nearest first): right, forward, up (/16), velocity right, forward, up (/3),
# kind one-hot (arrow, wind charge, hook, pearl, snowball, other), present
PROJECTILES = SELF_SIZE + OPP_SIZE
PROJECTILE_SIZE = 13
N_PROJECTILES = 4
PROJECTILE_ARROW = 6
PROJECTILE_PRESENT = 12
