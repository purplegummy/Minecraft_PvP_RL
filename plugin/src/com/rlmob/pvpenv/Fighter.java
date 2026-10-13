package com.rlmob.pvpenv;

import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.InteractionHand;
import org.bukkit.GameMode;
import org.bukkit.Location;
import org.bukkit.Material;
import org.bukkit.World;
import org.bukkit.attribute.Attribute;
import org.bukkit.attribute.AttributeInstance;
import org.bukkit.block.Block;
import org.bukkit.entity.AbstractArrow;
import org.bukkit.entity.AbstractWindCharge;
import org.bukkit.entity.EnderPearl;
import org.bukkit.entity.Entity;
import org.bukkit.entity.FishHook;
import org.bukkit.entity.Player;
import org.bukkit.entity.Projectile;
import org.bukkit.entity.Snowball;
import org.bukkit.inventory.ItemStack;
import org.bukkit.inventory.PlayerInventory;
import org.bukkit.util.RayTraceResult;
import org.bukkit.util.Vector;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

// one combatant: loadout, applying an action, and what it observes
final class Fighter {
    static final double REACH = 3.0;
    static final float[] YAW_BINS = {-30f, -12f, -4f, 0f, 4f, 12f, 30f};
    static final float[] PITCH_BINS = {-20f, -6f, -2f, 0f, 2f, 6f, 20f};
    // hotbar slots 0..8; the slot head picks "keep" (0) or one of these (1..9). arrows sit in the main inventory
    static final Material[] HOTBAR = {
            Material.IRON_SWORD, Material.BOW, Material.FISHING_ROD, Material.WIND_CHARGE,
            Material.LAVA_BUCKET, Material.ENDER_PEARL, Material.IRON_AXE, Material.GOLDEN_APPLE, Material.WATER_BUCKET};
    private static final int[] HOTBAR_COUNTS = {1, 1, 1, 8, 1, 4, 1, 2, 1};
    static final int SWORD = 0, BOW = 1, ROD = 2, WIND = 3, LAVA = 4, PEARL = 5, AXE = 6, APPLE = 7, WATER = 8;
    private static final int ARROWS = 16;
    // action heads: move, jump, sprint, yaw, pitch, attack, use, slot
    static final int[] HEADS = {9, 2, 2, YAW_BINS.length, PITCH_BINS.length, 2, 2, 1 + HOTBAR.length};
    // {forward, strafe}; positive strafe is to the left, like the A key
    private static final float[][] MOVES = {
            {0, 0}, {1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}};

    // held items are described by kind, not hotbar slot, so opponents with other kits (e.g. Carpet PvP bots) make sense
    enum ItemKind {
        EMPTY, SWORD, AXE, MACE, BOW, CROSSBOW, ROD, WIND_CHARGE,
        LAVA_BUCKET, WATER_BUCKET, BUCKET, PEARL, SNOWBALL, FOOD, SHIELD, OTHER
    }

    static final int ITEM_KINDS = ItemKind.values().length;
    private static final int PROJECTILE_KINDS = 6;
    private static final int OBS_PROJECTILES = 4;
    private static final int HAZARD = 7;
    private static final double VIEW = 16.0;
    static final int SELF_FEATURES = 27 + ITEM_KINDS;
    static final int OPP_FEATURES = 26 + ITEM_KINDS;
    static final int OBS_SIZE = SELF_FEATURES + OPP_FEATURES
            + OBS_PROJECTILES * (7 + PROJECTILE_KINDS) + HAZARD * HAZARD * 3;

    final ServerPlayer nms;
    // the bot-controlled player, or null when this fighter is a real person (human mode)
    final FakePlayer fake;
    final Player bukkit;
    private boolean lastUse;

    Fighter(ServerPlayer nms) {
        this.nms = nms;
        this.fake = nms instanceof FakePlayer f ? f : null;
        this.bukkit = nms.getBukkitEntity();
    }

    boolean alive() {
        return bukkit.isValid() && !bukkit.isDead();
    }

    // the sword and armor material; the curriculum hands an easier opponent weaker gear
    record Gear(Material sword, String armor) {
        static final Gear IRON = new Gear(Material.IRON_SWORD, "IRON");
    }

    void equip() {
        equip(Gear.IRON);
    }

    void equip(Gear gear) {
        if (nms.isUsingItem()) nms.stopUsingItem();
        // game mode first: a real player switched from spectator in the same tick as the kit was handed out
        // could end up without the off-hand shield on their client
        bukkit.setGameMode(GameMode.SURVIVAL);
        PlayerInventory inv = bukkit.getInventory();
        inv.clear();
        for (int slot = 0; slot < HOTBAR.length; slot++) inv.setItem(slot, new ItemStack(HOTBAR[slot], HOTBAR_COUNTS[slot]));
        inv.setItem(9, new ItemStack(Material.ARROW, ARROWS));
        inv.setItemInOffHand(new ItemStack(Material.SHIELD));
        inv.setItem(SWORD, new ItemStack(gear.sword()));
        inv.setHelmet(new ItemStack(Material.valueOf(gear.armor() + "_HELMET")));
        inv.setChestplate(new ItemStack(Material.valueOf(gear.armor() + "_CHESTPLATE")));
        inv.setLeggings(new ItemStack(Material.valueOf(gear.armor() + "_LEGGINGS")));
        inv.setBoots(new ItemStack(Material.valueOf(gear.armor() + "_BOOTS")));
        inv.setHeldItemSlot(SWORD);
        for (Material m : new Material[]{Material.WIND_CHARGE, Material.ENDER_PEARL, Material.SHIELD, Material.IRON_AXE}) {
            bukkit.setCooldown(m, 0);
        }

        if (fake == null) bukkit.updateInventory();  // push the whole kit to a real client
        bukkit.setHealth(maxHealth());
        bukkit.setAbsorptionAmount(0);
        bukkit.clearActivePotionEffects();
        bukkit.setFoodLevel(20);
        bukkit.setSaturation(5);
        bukkit.setFireTicks(0);
        bukkit.setFallDistance(0);
        bukkit.setArrowsInBody(0);
        bukkit.setVelocity(new Vector());
        if (fake != null) {
            fake.forward = 0;
            fake.strafe = 0;
            fake.jump = false;
            fake.sprint = false;
        }
        lastUse = false;
    }

    void apply(int[] a, Fighter target) {
        float[] move = MOVES[a[0]];
        fake.forward = move[0];
        fake.strafe = move[1];
        fake.jump = a[1] == 1;
        fake.sprint = a[2] == 1;

        float yaw = nms.getYRot() + YAW_BINS[a[3]];
        float pitch = Math.max(-90f, Math.min(90f, nms.getXRot() + PITCH_BINS[a[4]]));
        nms.setYRot(yaw);
        nms.setYHeadRot(yaw);
        nms.setYBodyRot(yaw);
        nms.setXRot(pitch);

        if (a[7] != 0 && heldSlot() != a[7] - 1) {
            if (nms.isUsingItem()) nms.stopUsingItem();
            bukkit.getInventory().setHeldItemSlot(a[7] - 1);
            lastUse = false;
        }

        // vanilla right-click: the main hand is used if its item has a use, otherwise the off-hand (shield).
        // bow, food and shield are held down; rod, wind charge, bucket, pearl and snowball act once per click
        boolean use = a[6] == 1;
        InteractionHand holdHand = holdHand();
        if (holdHand != null) {
            if (use && !nms.isUsingItem()) useItem(holdHand);
            else if (!use && nms.isUsingItem()) nms.releaseUsingItem();
        } else if (use && !lastUse) {
            useItem(InteractionHand.MAIN_HAND);
        }
        lastUse = use;

        if (a[5] == 1) attack(target);
    }

    // which hand a held-down use applies to for the current item, or null if the item acts on a click
    private InteractionHand holdHand() {
        Material main = held();
        if (main == Material.BOW || main.isEdible()) return InteractionHand.MAIN_HAND;
        if ((main.name().endsWith("_SWORD") || main.name().endsWith("_AXE")) && bukkit.getInventory().getItemInOffHand().getType() == Material.SHIELD) {
            return InteractionHand.OFF_HAND;
        }
        return null;
    }

    // the same server path a real client's right-click takes
    private void useItem(InteractionHand hand) {
        nms.gameMode.useItem(nms, nms.level(), nms.getItemInHand(hand), hand);
    }

    private void attack(Fighter target) {
        // the vanilla client ignores attack clicks while an item is in use (drawing, eating, blocking)
        if (nms.isUsingItem()) return;
        nms.swing(InteractionHand.MAIN_HAND);
        if (crosshairOn(target)) nms.attack(target.nms);
        else nms.resetAttackStrengthTicker();
    }

    boolean crosshairOn(Fighter target) {
        Location eye = bukkit.getEyeLocation();
        Vector dir = eye.getDirection();
        World world = bukkit.getWorld();
        RayTraceResult hit = world.rayTraceEntities(eye, dir, REACH, 0.0, e -> e.equals(target.bukkit));
        if (hit == null) return false;
        RayTraceResult block = world.rayTraceBlocks(eye, dir, REACH);
        return block == null || block.getHitPosition().distanceSquared(eye.toVector())
                > hit.getHitPosition().distanceSquared(eye.toVector());
    }

    int heldSlot() {
        return bukkit.getInventory().getHeldItemSlot();
    }

    Material held() {
        return bukkit.getInventory().getItemInMainHand().getType();
    }

    boolean holdingBow() {
        return held() == Material.BOW;
    }

    double cooldown() {
        return nms.getAttackStrengthScale(0.5f);
    }

    double bowDraw() {
        if (!nms.isUsingItem() || !holdingBow()) return 0;
        return Math.min(1.0, nms.getTicksUsingItem() / 20.0);
    }

    // how far along the current hold is: a full bow draw is 20 ticks, eating is 32
    double useProgress() {
        if (!nms.isUsingItem()) return 0;
        Material main = held();
        if (main == Material.BOW) return bowDraw();
        if (main.isEdible()) return Math.min(1.0, nms.getTicksUsingItem() / 32.0);
        return 0;
    }

    int count(Material material) {
        int n = 0;
        for (ItemStack stack : bukkit.getInventory().getContents()) {
            if (stack != null && stack.getType() == material) n += stack.getAmount();
        }
        return n;
    }

    // hotbar slot currently holding this material, or -1
    int slotOf(Material material) {
        for (int slot = 0; slot < 9; slot++) {
            ItemStack stack = bukkit.getInventory().getItem(slot);
            if (stack != null && stack.getType() == material) return slot;
        }
        return -1;
    }

    boolean onFire() {
        return bukkit.getFireTicks() > 0;
    }

    boolean hooking(Fighter target) {
        for (FishHook hook : bukkit.getWorld().getEntitiesByClass(FishHook.class)) {
            if (bukkit.equals(hook.getShooter()) && target.bukkit.equals(hook.getHookedEntity())) return true;
        }
        return false;
    }

    boolean hookOut() {
        for (FishHook hook : bukkit.getWorld().getEntitiesByClass(FishHook.class)) {
            if (bukkit.equals(hook.getShooter())) return true;
        }
        return false;
    }

    static ItemKind kindOf(Material m) {
        if (m.isAir()) return ItemKind.EMPTY;
        String name = m.name();
        if (name.endsWith("_SWORD")) return ItemKind.SWORD;
        if (name.endsWith("_AXE")) return ItemKind.AXE;
        if (m == Material.LAVA_BUCKET) return ItemKind.LAVA_BUCKET;
        if (m == Material.WATER_BUCKET) return ItemKind.WATER_BUCKET;
        if (name.endsWith("BUCKET")) return ItemKind.BUCKET;
        if (m.isEdible()) return ItemKind.FOOD;
        return switch (m) {
            case MACE -> ItemKind.MACE;
            case BOW -> ItemKind.BOW;
            case CROSSBOW -> ItemKind.CROSSBOW;
            case FISHING_ROD -> ItemKind.ROD;
            case WIND_CHARGE -> ItemKind.WIND_CHARGE;
            case ENDER_PEARL -> ItemKind.PEARL;
            case SNOWBALL -> ItemKind.SNOWBALL;
            case SHIELD -> ItemKind.SHIELD;
            default -> ItemKind.OTHER;
        };
    }

    // 1 when looking straight at the target's head, 0 when facing directly away. Half of it falls off smoothly so
    // turning toward the target pays from any angle; the other half is sharp so precise aim pays most
    double aimQuality(Fighter target) {
        Location eye = bukkit.getEyeLocation();
        Vector toTarget = target.bukkit.getEyeLocation().toVector().subtract(eye.toVector());
        if (toTarget.lengthSquared() < 1e-6) return 1;
        double cos = eye.getDirection().dot(toTarget.normalize());
        return 0.5 * (cos + 1) / 2 + 0.5 * Math.pow(Math.max(0, cos), 8);
    }

    // signed horizontal angle (degrees) the fighter must turn to face a point
    double yawErrorTo(Location point) {
        Location me = bukkit.getEyeLocation();
        double desired = Math.toDegrees(Math.atan2(-(point.getX() - me.getX()), point.getZ() - me.getZ()));
        return wrap(desired - me.getYaw());
    }

    double pitchErrorTo(Location point) {
        Location me = bukkit.getEyeLocation();
        double dx = point.getX() - me.getX(), dy = point.getY() - me.getY(), dz = point.getZ() - me.getZ();
        return -Math.toDegrees(Math.atan2(dy, Math.hypot(dx, dz))) - me.getPitch();
    }

    double yawErrorTo(Fighter target) {
        return yawErrorTo(target.bukkit.getEyeLocation());
    }

    double pitchErrorTo(Fighter target) {
        return pitchErrorTo(target.bukkit.getEyeLocation());
    }

    static double wrap(double degrees) {
        double d = degrees % 360;
        if (d > 180) d -= 360;
        if (d < -180) d += 360;
        return d;
    }

    double maxHealth() {
        AttributeInstance maxHp = bukkit.getAttribute(Attribute.MAX_HEALTH);
        return maxHp != null ? maxHp.getValue() : 20;
    }

    private static final class Obs {
        final double[] values = new double[OBS_SIZE];
        int k = 0;

        void add(double v) {
            values[k++] = v;
        }

        void add(boolean b) {
            values[k++] = b ? 1 : 0;
        }

        void oneHot(int index, int size) {
            for (int i = 0; i < size; i++) values[k++] = i == index ? 1 : 0;
        }
    }

    private static int projectileKind(Entity e) {
        if (e instanceof AbstractArrow) return 0;
        if (e instanceof AbstractWindCharge) return 1;
        if (e instanceof FishHook) return 2;
        if (e instanceof EnderPearl) return 3;
        if (e instanceof Snowball) return 4;
        return 5;
    }

    // everything relative to this fighter: right = +x, forward = +z in its own frame.
    // the layout is mirrored in obs_layout.py; keep the two in sync
    double[] observe(Fighter opp, Location center, double arenaRadius) {
        Obs o = new Obs();
        Location me = bukkit.getLocation();
        double yaw = Math.toRadians(me.getYaw());
        double fx = -Math.sin(yaw), fz = Math.cos(yaw);
        double rx = -fz, rz = fx;
        Vector v = bukkit.getVelocity();
        double cx = center.getX() - me.getX(), cz = center.getZ() - me.getZ();

        // self: 27 + item kinds
        o.add(bukkit.getHealth() / maxHealth());
        o.add(bukkit.getAbsorptionAmount() / 4.0);
        o.add(bukkit.isOnGround());
        o.add((v.getX() * rx + v.getZ() * rz) / 0.5);
        o.add((v.getX() * fx + v.getZ() * fz) / 0.5);
        o.add(v.getY() / 0.5);
        o.add(cooldown());
        o.add(nms.isUsingItem());
        o.add(useProgress());
        o.add(nms.isBlocking());
        o.oneHot(kindOf(held()).ordinal(), ITEM_KINDS);
        o.add(count(Material.ARROW) / (double) ARROWS);
        o.add(count(Material.WIND_CHARGE) / 8.0);
        o.add(count(Material.ENDER_PEARL) / 4.0);
        o.add(count(Material.GOLDEN_APPLE) / 2.0);
        o.add(count(Material.LAVA_BUCKET));
        o.add(count(Material.WATER_BUCKET));
        o.add(bukkit.getCooldown(Material.WIND_CHARGE) / 10.0);
        o.add(bukkit.getCooldown(Material.ENDER_PEARL) / 20.0);
        o.add(bukkit.getCooldown(Material.SHIELD) / 100.0);
        o.add(me.getPitch() / 90.0);
        o.add(bukkit.isSprinting());
        o.add(nms.hurtTime / 10.0);
        o.add(onFire());
        o.add(bukkit.isInLava());
        o.add(bukkit.isInWater());
        o.add((cx * rx + cz * rz) / arenaRadius);
        o.add((cx * fx + cz * fz) / arenaRadius);

        // opponent: 26 + item kinds
        Location them = opp.bukkit.getLocation();
        double dx = them.getX() - me.getX(), dy = them.getY() - me.getY(), dz = them.getZ() - me.getZ();
        Vector ov = opp.bukkit.getVelocity().subtract(v);
        double yawErr = Math.toRadians(yawErrorTo(opp));
        Location oppEye = opp.bukkit.getEyeLocation();
        Vector toMe = bukkit.getEyeLocation().toVector().subtract(oppEye.toVector());
        o.add((dx * rx + dz * rz) / VIEW);
        o.add((dx * fx + dz * fz) / VIEW);
        o.add(dy / VIEW);
        o.add(Math.sqrt(dx * dx + dy * dy + dz * dz) / VIEW);
        o.add((ov.getX() * rx + ov.getZ() * rz) / 0.5);
        o.add((ov.getX() * fx + ov.getZ() * fz) / 0.5);
        o.add(ov.getY() / 0.5);
        o.add(opp.bukkit.getHealth() / opp.maxHealth());
        o.add(opp.bukkit.getAbsorptionAmount() / 4.0);
        o.add(opp.cooldown());
        o.add(opp.nms.isUsingItem());
        o.add(opp.useProgress());
        o.add(opp.nms.isBlocking());
        // an axe hit on a raised shield disables it for 100 ticks
        o.add(opp.bukkit.getCooldown(Material.SHIELD) / 100.0);
        o.oneHot(kindOf(opp.held()).ordinal(), ITEM_KINDS);
        o.add(opp.bukkit.isOnGround());
        o.add(opp.nms.hurtTime / 10.0);
        o.add(opp.onFire());
        o.add(opp.bukkit.isInLava());
        o.add(opp.bukkit.isInWater());
        o.add(Math.sin(yawErr));
        o.add(Math.cos(yawErr));
        o.add(pitchErrorTo(opp) / 90.0);
        o.add(toMe.lengthSquared() > 1e-6 ? oppEye.getDirection().dot(toMe.normalize()) : 1);
        o.add(crosshairOn(opp));
        o.add(hooking(opp));
        o.add(opp.hooking(this));

        // nearest projectiles that aren't ours: position, velocity, kind, present
        List<Entity> projectiles = new ArrayList<>();
        for (Entity e : bukkit.getWorld().getNearbyEntities(me, VIEW, VIEW, VIEW)) {
            if (!(e instanceof Projectile p) || bukkit.equals(p.getShooter())) continue;
            if (e instanceof AbstractArrow arrow && arrow.isInBlock()) continue;
            // arenas sit side by side, so ignore whatever is flying around next door
            Location pl = e.getLocation();
            if (Math.abs(pl.getX() - center.getX()) > arenaRadius + 1 || Math.abs(pl.getZ() - center.getZ()) > arenaRadius + 1) continue;
            projectiles.add(e);
        }
        projectiles.sort(Comparator.comparingDouble(e -> e.getLocation().distanceSquared(me)));
        for (int i = 0; i < OBS_PROJECTILES; i++) {
            if (i >= projectiles.size()) {
                for (int j = 0; j < 7 + PROJECTILE_KINDS; j++) o.add(0);
                continue;
            }
            Entity e = projectiles.get(i);
            Location el = e.getLocation();
            Vector ev = e.getVelocity();
            double ax = el.getX() - me.getX(), az = el.getZ() - me.getZ();
            o.add((ax * rx + az * rz) / VIEW);
            o.add((ax * fx + az * fz) / VIEW);
            o.add((el.getY() - me.getY()) / VIEW);
            o.add((ev.getX() * rx + ev.getZ() * rz) / 3.0);
            o.add((ev.getX() * fx + ev.getZ() * fz) / 3.0);
            o.add(ev.getY() / 3.0);
            o.oneHot(projectileKind(e), PROJECTILE_KINDS);
            o.add(1);
        }

        // hazard grid around the feet, in this fighter's frame: lava, water, solid block at feet level
        int feetY = me.getBlockY();
        int half = HAZARD / 2;
        for (int i = 0; i < HAZARD; i++) {
            for (int j = 0; j < HAZARD; j++) {
                double right = i - half, forward = j - half;
                int bx = (int) Math.floor(me.getX() + right * rx + forward * fx);
                int bz = (int) Math.floor(me.getZ() + right * rz + forward * fz);
                Block feet = bukkit.getWorld().getBlockAt(bx, feetY, bz);
                Block below = bukkit.getWorld().getBlockAt(bx, feetY - 1, bz);
                o.add(feet.getType() == Material.LAVA || below.getType() == Material.LAVA);
                o.add(feet.getType() == Material.WATER || below.getType() == Material.WATER);
                o.add(!feet.isPassable());
            }
        }

        if (o.k != OBS_SIZE) throw new IllegalStateException("observation has " + o.k + " values, expected " + OBS_SIZE);
        return o.values;
    }
}
