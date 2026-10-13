package com.rlmob.pvpenv;

import org.bukkit.Location;
import org.bukkit.Material;
import org.bukkit.util.Vector;

import java.util.Random;

// a hand-written sparring partner that picks actions from the same action space as the bot.
// skill 0 wanders in and flails; higher skill adds sprinting, strafing, timed hits and crits, then
// unlocks bow (0.3), fishing rod (0.4), wind charges (0.6) and lava (0.7).
// at every skill it steps around lava and puts itself out with water, so lava alone never beats it
final class ScriptedOpponent {
    private enum Mode { SWORD, BOW, ROD, WIND, LAVA, WATER }

    private final Random rng;
    private final double skill;
    private int strafeDir = 1;
    private int strafeTimer = 0;
    private Mode mode = Mode.SWORD;
    private int modeSteps = 0;
    private boolean pressed;

    ScriptedOpponent(Random rng, double skill) {
        this.rng = rng;
        this.skill = skill;
    }

    int[] act(Fighter me, Fighter target) {
        int[] a = new int[Fighter.HEADS.length];
        if (--strafeTimer <= 0) {
            strafeDir = -strafeDir;
            strafeTimer = 5 + rng.nextInt(15);
        }
        double dist = me.bukkit.getLocation().distance(target.bukkit.getLocation());
        if (mode == Mode.SWORD || mode == Mode.BOW) chooseMode(me, target, dist);
        modeSteps++;

        switch (mode) {
            case SWORD -> sword(me, target, dist, a);
            case BOW -> bow(me, target, a);
            case ROD -> rod(me, target, a);
            case WIND -> throwAt(me, target, Material.WIND_CHARGE, target.bukkit.getEyeLocation(), a);
            case LAVA -> throwAt(me, target, Material.LAVA_BUCKET, target.bukkit.getLocation(), a);
            case WATER -> extinguish(me, a);
        }
        // never walk forward into lava: sidestep instead
        if (lavaAhead(me) && (a[0] == 1 || a[0] == 5 || a[0] == 6)) a[0] = strafeDir > 0 ? 3 : 4;
        pressed = a[6] == 1;
        return a;
    }

    private static boolean lavaAhead(Fighter me) {
        Location feet = me.bukkit.getLocation();
        Vector dir = feet.getDirection().setY(0);
        if (dir.lengthSquared() < 1e-6) return false;
        dir.normalize();
        for (double d = 0.5; d <= 2.0; d += 0.5) {
            Location probe = feet.clone().add(dir.clone().multiply(d));
            if (probe.getBlock().getType() == Material.LAVA
                    || probe.clone().subtract(0, 1, 0).getBlock().getType() == Material.LAVA) return true;
        }
        return false;
    }

    private void chooseMode(Fighter me, Fighter target, double dist) {
        Mode next;
        if ((me.onFire() || me.bukkit.isInLava()) && me.slotOf(Material.WATER_BUCKET) >= 0) next = Mode.WATER;
        else if (skill >= 0.7 && dist > 1.5 && dist < 4 && me.slotOf(Material.LAVA_BUCKET) >= 0 && rng.nextDouble() < 0.05) next = Mode.LAVA;
        else if (skill >= 0.6 && dist > 4 && dist < 10 && me.count(Material.WIND_CHARGE) > 0
                && me.bukkit.getCooldown(Material.WIND_CHARGE) == 0 && rng.nextDouble() < 0.05) next = Mode.WIND;
        else if (skill >= 0.4 && dist > 4 && dist < 9 && rng.nextDouble() < 0.05) next = Mode.ROD;
        else if (skill >= 0.3 && dist > 9 && me.count(Material.ARROW) > 0) next = Mode.BOW;
        else next = Mode.SWORD;
        if (next != mode) {
            mode = next;
            modeSteps = 0;
        }
    }

    private void aim(Fighter me, Location point, int[] a) {
        double noise = 25 * (1 - skill);
        a[3] = nearestBin(Fighter.YAW_BINS, me.yawErrorTo(point) + rng.nextGaussian() * noise);
        a[4] = nearestBin(Fighter.PITCH_BINS, me.pitchErrorTo(point) + rng.nextGaussian() * noise * 0.5);
    }

    // returns true once the item is in hand; until then requests the switch
    private boolean select(Fighter me, Material item, int[] a) {
        int slot = me.slotOf(item);
        if (slot < 0) {
            mode = Mode.SWORD;
            return false;
        }
        if (me.heldSlot() == slot) return true;
        a[7] = slot + 1;
        return false;
    }

    private void sword(Fighter me, Fighter target, double dist, int[] a) {
        aim(me, target.bukkit.getEyeLocation(), a);
        if (me.heldSlot() != Fighter.SWORD) a[7] = Fighter.SWORD + 1;
        boolean strafing = rng.nextDouble() < skill;
        if (dist > 2.2) a[0] = strafing ? (strafeDir > 0 ? 5 : 6) : 1;
        else a[0] = strafeDir > 0 ? 3 : 4;
        if (rng.nextDouble() < 0.3 * (1 - skill)) a[0] = 0;
        a[2] = rng.nextDouble() < 0.2 + 0.8 * skill ? 1 : 0;

        boolean inReach = dist < Fighter.REACH && Math.abs(me.yawErrorTo(target)) < 20;
        if (inReach && me.cooldown() >= 0.9) a[5] = 1;
        else if (rng.nextDouble() < 0.15 * (1 - skill)) a[5] = 1;
        if (inReach && skill > 0.5 && rng.nextDouble() < 0.25 * skill) a[1] = 1;
    }

    private void bow(Fighter me, Fighter target, int[] a) {
        aim(me, target.bukkit.getEyeLocation(), a);
        if (!select(me, Material.BOW, a)) return;
        a[0] = strafeDir > 0 ? 3 : 4;
        boolean releaseNow = me.bowDraw() >= 1.0 && Math.abs(me.yawErrorTo(target)) < 6;
        a[6] = releaseNow ? 0 : 1;
    }

    // cast, wait for the hook to land, then reel in to pull the target closer
    private void rod(Fighter me, Fighter target, int[] a) {
        aim(me, target.bukkit.getEyeLocation(), a);
        if (!select(me, Material.FISHING_ROD, a)) return;
        if (!me.hookOut()) {
            if (modeSteps > 12) mode = Mode.SWORD;
            else a[6] = pressed ? 0 : 1;
        } else if (me.hooking(target) || modeSteps > 10) {
            a[6] = pressed ? 0 : 1;
            if (a[6] == 1) mode = Mode.SWORD;
        }
    }

    private void throwAt(Fighter me, Fighter target, Material item, Location point, int[] a) {
        aim(me, point, a);
        if (modeSteps > 15) {
            mode = Mode.SWORD;
            return;
        }
        if (!select(me, item, a)) return;
        boolean aimed = Math.abs(me.yawErrorTo(point)) < 8 && Math.abs(me.pitchErrorTo(point)) < 8;
        if (aimed && !pressed) {
            a[6] = 1;
            mode = Mode.SWORD;
        }
    }

    // look straight down and pour water at its own feet, like a player putting themselves out
    private void extinguish(Fighter me, int[] a) {
        a[4] = Fighter.PITCH_BINS.length - 1;
        if (modeSteps > 15 || !(me.onFire() || me.bukkit.isInLava())) {
            mode = Mode.SWORD;
            return;
        }
        if (!select(me, Material.WATER_BUCKET, a)) return;
        if (me.bukkit.getLocation().getPitch() > 70 && !pressed) {
            a[6] = 1;
            mode = Mode.SWORD;
        }
    }

    private static int nearestBin(float[] bins, double value) {
        int best = 0;
        for (int i = 1; i < bins.length; i++) {
            if (Math.abs(bins[i] - value) < Math.abs(bins[best] - value)) best = i;
        }
        return best;
    }
}
