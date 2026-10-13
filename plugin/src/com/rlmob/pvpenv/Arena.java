package com.rlmob.pvpenv;

import org.bukkit.Location;
import org.bukkit.Material;
import org.bukkit.World;

import java.util.Random;

// a flat square: glass walls you can see, invisible barrier above them, and a barrier ceiling, so nobody can pearl
// or wind-charge their way out to run the clock down. Arena 0 (the Carpet bot's) sits at the world origin; the
// self-play arenas 1.. form a grid next to it, side by side, so you can watch them all at once
final class Arena {
    // grid arenas are this far apart centre to centre: their walls are 3 blocks apart
    static final int SPACING = 30;
    static final int GRID_COLUMNS = 3;
    // the grid starts further out: the Carpet bot in arena 0 goes for the nearest player, so every fighter in the grid
    // must be further from it than its own opponent can ever be (~34 blocks corner to corner)
    static final int GRID_OFFSET = 64;
    static final int RADIUS = 12;
    private static final int WALL_HEIGHT = 4;
    static final int CEILING = 24;
    private static final double FAR_SPAWN_CHANCE = 0.3;
    // the old pig arena reached x/z ±49 and 48 blocks up; cleared once so its terrain doesn't linger
    private static final int CLEAR_RADIUS = 50;
    private static final int CLEAR_HEIGHT = 48;

    final World world;
    final int baseY;
    final int originX;
    final int originZ;
    final Location center;

    Arena(World world, int index) {
        this.world = world;
        // sample the untouched flat world well outside anything ever built
        this.baseY = world.getHighestBlockYAt(-CLEAR_RADIUS - 40, CLEAR_RADIUS + 40);
        // grid rows run along z, centred on z = 0
        int cell = index - 1;
        this.originX = index == 0 ? 0 : GRID_OFFSET + (cell % GRID_COLUMNS) * SPACING;
        this.originZ = index == 0 ? 0 : (cell / GRID_COLUMNS - 1) * SPACING;
        this.center = new Location(world, originX + 0.5, baseY + 1, originZ + 0.5);
    }

    void prepare() {
        // only arena 0 has old pig-arena terrain around it; the others just need their own footprint
        int clear = originX == 0 ? CLEAR_RADIUS : SPACING / 2;
        int lo = Math.floorDiv(-clear, 16), hi = Math.floorDiv(clear, 16);
        for (int cx = lo; cx <= hi; cx++) {
            for (int cz = lo; cz <= hi; cz++) {
                world.getChunkAt(Math.floorDiv(originX, 16) + cx, Math.floorDiv(originZ, 16) + cz).setForceLoaded(true);
            }
        }
        for (int x = originX - clear; x <= originX + clear; x++) {
            for (int z = originZ - clear; z <= originZ + clear; z++) {
                for (int y = baseY + 1; y <= baseY + CLEAR_HEIGHT; y++) {
                    if (world.getBlockAt(x, y, z).getType() != Material.AIR) {
                        world.getBlockAt(x, y, z).setType(Material.AIR, false);
                    }
                }
            }
        }
        int r = RADIUS + 1;
        for (int x = -r; x <= r; x++) {
            for (int z = -r; z <= r; z++) {
                boolean wall = Math.abs(x) == r || Math.abs(z) == r;
                world.getBlockAt(originX + x, baseY, originZ + z).setType(wall ? Material.POLISHED_ANDESITE : Material.SMOOTH_STONE, false);
                world.getBlockAt(originX + x, baseY + CEILING, originZ + z).setType(Material.BARRIER, false);
                if (!wall) continue;
                for (int y = baseY + 1; y < baseY + CEILING; y++) {
                    world.getBlockAt(originX + x, y, originZ + z).setType(y <= baseY + WALL_HEIGHT ? Material.GLASS : Material.BARRIER, false);
                }
            }
        }
    }

    // earlier layouts put arenas in a row along +x (128 apart, then 30 apart): knock down whatever of them is
    // outside the current grid, walls and ceiling only (the floor is part of the flat world)
    static void removeOldLayouts(World world, int baseY) {
        int gridEnd = GRID_OFFSET + (GRID_COLUMNS - 1) * SPACING + SPACING / 2;
        for (int x = gridEnd + 1; x <= 7 * 128 + RADIUS + 1; x++) {
            for (int z = -RADIUS - 1; z <= RADIUS + 1; z++) {
                world.getBlockAt(x, baseY, z).setType(Material.GRASS_BLOCK, false);
                for (int y = baseY + 1; y <= baseY + CEILING; y++) {
                    if (world.getBlockAt(x, y, z).getType() != Material.AIR) world.getBlockAt(x, y, z).setType(Material.AIR, false);
                }
            }
        }
    }

    boolean contains(Location loc) {
        return Math.abs(loc.getX() - center.getX()) <= RADIUS + 0.5 && Math.abs(loc.getZ() - center.getZ()) <= RADIUS + 0.5
                && loc.getY() > baseY && loc.getY() < baseY + CEILING;
    }

    // wipes lava and anything else placed during the previous round, all the way up to the ceiling
    // (a bucket aimed at the invisible barrier walls or ceiling leaves lava hanging up there)
    void resetInterior() {
        for (int x = -RADIUS; x <= RADIUS; x++) {
            for (int z = -RADIUS; z <= RADIUS; z++) {
                world.getBlockAt(originX + x, baseY, originZ + z).setType(Material.SMOOTH_STONE, false);
                for (int y = baseY + 1; y < baseY + CEILING; y++) {
                    if (world.getBlockAt(originX + x, y, originZ + z).getType() != Material.AIR) {
                        world.getBlockAt(originX + x, y, originZ + z).setType(Material.AIR, false);
                    }
                }
            }
        }
    }

    // two spots facing each other across the centre: usually 6-14 blocks apart, sometimes 18-23 so ranged play pays
    Location[] spawnPair(Random rng) {
        double angle = rng.nextDouble() * 2 * Math.PI;
        double distance = rng.nextDouble() < FAR_SPAWN_CHANCE ? 18 + rng.nextDouble() * 5 : 6 + rng.nextDouble() * 8;
        double half = distance / 2;
        double ox = Math.cos(angle) * half, oz = Math.sin(angle) * half;
        Location a = center.clone().add(ox, 0, oz);
        Location b = center.clone().add(-ox, 0, -oz);
        face(a, b, rng);
        face(b, a, rng);
        return new Location[]{a, b};
    }

    private static void face(Location from, Location to, Random rng) {
        from.setDirection(to.toVector().subtract(from.toVector()));
        from.setYaw(from.getYaw() + (float) (rng.nextGaussian() * 40));
        from.setPitch(0);
    }
}
