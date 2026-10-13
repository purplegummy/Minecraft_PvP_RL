package com.rlmob.pvpenv;

import org.bukkit.Bukkit;
import org.bukkit.GameMode;
import org.bukkit.Location;
import org.bukkit.Material;
import org.bukkit.ServerTickManager;
import net.kyori.adventure.text.Component;
import org.bukkit.World;
import org.bukkit.craftbukkit.entity.CraftPlayer;
import org.bukkit.entity.Entity;
import org.bukkit.entity.Player;
import org.bukkit.entity.Item;
import org.bukkit.entity.Projectile;
import org.bukkit.event.EventHandler;
import org.bukkit.event.EventPriority;
import org.bukkit.event.Listener;
import org.bukkit.event.entity.CreatureSpawnEvent;
import org.bukkit.event.entity.EntityDamageEvent;
import org.bukkit.event.player.PlayerTeleportEvent;
import org.bukkit.plugin.java.JavaPlugin;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.StringJoiner;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

public final class PvpEnvPlugin extends JavaPlugin implements Listener {
    private static final int PORT = 5555;
    // separate fights running side by side; one "stepall" advances all of them by the same ticks.
    // arena 0 (Carpet bot, humans) plus a 3x3 grid of self-play arenas
    private static final int ARENAS = 1 + 3 * Arena.GRID_COLUMNS;
    private static final int TICKS_PER_STEP = 2;
    // 600 steps * 2 ticks = 60 seconds of fight before it's called a draw
    private static final int MAX_STEPS = 600;
    private static final float TICK_RATE = 1000f;
    private static final double WIN_REWARD = 1.0;
    // potential-based aim shaping: reward = AIM_SHAPING * (GAMMA * quality_now - quality_before).
    // turning toward the opponent pays and turning away costs the same, so staring can't be farmed;
    // GAMMA must match the discount used in training for this to leave the best strategy unchanged
    private static final double AIM_SHAPING = 0.2;
    // when time runs out, whoever has more health (plus absorption) gets +TIEBREAK and the other -TIEBREAK,
    // so stalling while behind costs something; a real knockout (WIN_REWARD) is still worth twice as much
    private static final double TIEBREAK = 0.5;
    private static final Set<EntityDamageEvent.DamageCause> BURNING = EnumSet.of(
            EntityDamageEvent.DamageCause.LAVA, EntityDamageEvent.DamageCause.FIRE,
            EntityDamageEvent.DamageCause.FIRE_TICK, EntityDamageEvent.DamageCause.HOT_FLOOR,
            EntityDamageEvent.DamageCause.CAMPFIRE);
    private static final double GAMMA = 0.99;

    private World world;
    private ServerTickManager tickManager;
    private ServerSocket serverSocket;
    private final Random rng = new Random();
    private final List<Match> matches = new ArrayList<>();

    // the step in flight: which matches it covers and where their results go once the stepped ticks are used up
    private CompletableFuture<String> pendingStep;
    private List<Match> pendingMatches;
    // human mode runs the world in real time for one person against arena 0's bot
    private boolean humanMode;

    // carpet mode: B is a Carpet PvP bot (separate plugin) that thinks for itself on the ticks we step.
    // there is only one of it, so carpet fights happen in arena 0 only
    private static final String CARPET_BOT = "Theo";
    private static final Set<String> CARPET_PRESETS = Set.of("beginner", "casual", "average", "skilled", "expert");
    // easier than the plugin's own beginner preset: beginner plus these /bot option overrides
    private static final Map<String, String[]> CARPET_TRAINEES = Map.of(
            "trainee1", new String[]{"skill 0.0", "reactiondelay 12", "pingticks 4", "clickspersecond 2", "misschance 80",
                    "shieldplay false", "critical false", "strafe false", "wtap false"},
            "trainee2", new String[]{"skill 0.05", "reactiondelay 8", "pingticks 3", "clickspersecond 3", "misschance 65",
                    "shieldplay false", "critical false", "strafe false", "wtap false"},
            "trainee3", new String[]{"skill 0.1", "reactiondelay 6", "pingticks 2", "clickspersecond 4", "misschance 55",
                    "shieldplay true", "critical false", "strafe false", "wtap false"});
    // the trainee levels also fight in worse gear than the bot's iron, closing the gap step by step
    private static final Map<String, Fighter.Gear> CARPET_GEAR = Map.of(
            "trainee1", new Fighter.Gear(Material.WOODEN_SWORD, "LEATHER"),
            "trainee2", new Fighter.Gear(Material.STONE_SWORD, "CHAINMAIL"));
    // a preset decides its numbers, but the switches a trainee turned off have to be put back on
    private static final String[] CARPET_SWITCHES_ON = {"shieldplay true", "critical true", "strafe true", "wtap true"};
    private static final int CARPET_SPAWN_TIMEOUT_TICKS = 200;
    private String carpetDifficulty;
    private String carpetStyle;
    private static final Set<String> CARPET_STYLES = Set.of("smp", "ranged");
    private static final double LOW_HEALTH_START_CHANCE = 0.15;
    private static final double ON_FIRE_START_CHANCE = 0.1;

    // one fight: A is always the learning bot ("PvpBot", "PvpBot1", ...), B is scripted, Python-driven (self-play),
    // a real person or the Carpet bot
    private final class Match {
        final int index;
        final Arena arena;
        Fighter a;
        Fighter b;
        ScriptedOpponent script;
        boolean carpet;
        int steps;
        double damageToA;
        double damageToB;
        double aimA;
        double aimB;
        Fighter knockedOut;

        Match(int index) {
            this.index = index;
            this.arena = new Arena(world, index);
        }

        String suffix() {
            return index == 0 ? "" : Integer.toString(index);
        }
    }

    @Override
    public void onEnable() {
        world = Bukkit.getWorlds().get(0);
        // force-loading is saved with the world: drop whatever an older arena layout left loaded, then load ours
        for (org.bukkit.Chunk chunk : world.getForceLoadedChunks()) chunk.setForceLoaded(false);
        java.io.File cleaned = new java.io.File(getDataFolder(), "old-layouts-removed");
        if (!cleaned.exists()) {
            Arena.removeOldLayouts(world, new Arena(world, 0).baseY);
            getDataFolder().mkdirs();
            try {
                cleaned.createNewFile();
            } catch (IOException e) {
                getLogger().warning("couldn't write " + cleaned + ": " + e.getMessage());
            }
        }
        for (int i = 0; i < ARENAS; i++) {
            Match m = new Match(i);
            m.arena.prepare();
            matches.add(m);
        }
        tickManager = Bukkit.getServerTickManager();
        tickManager.setTickRate(TICK_RATE);
        tickManager.setFrozen(true);
        getServer().getPluginManager().registerEvents(this, this);
        Bukkit.getScheduler().runTaskTimer(this, this::tick, 1L, 1L);

        Thread socketThread = new Thread(this::serve, "PvpEnv-socket");
        socketThread.setDaemon(true);
        socketThread.start();
    }

    @Override
    public void onDisable() {
        if (tickManager != null) {
            tickManager.setFrozen(false);
            tickManager.setTickRate(20f);
        }
        for (Match m : matches) {
            if (m.a != null) m.a.fake.despawn();
            if (m.b != null && m.b.fake != null) m.b.fake.despawn();
        }
        try {
            if (serverSocket != null) serverSocket.close();
        } catch (IOException ignored) {
        }
    }

    @EventHandler
    public void onCreatureSpawn(CreatureSpawnEvent event) {
        if (event.getSpawnReason() != CreatureSpawnEvent.SpawnReason.CUSTOM) event.setCancelled(true);
    }

    // a lethal hit ends the round instead of killing: fake players have no client to press "respawn"
    @EventHandler(priority = EventPriority.HIGHEST, ignoreCancelled = true)
    public void onDamage(EntityDamageEvent event) {
        Match m = matchOf(event.getEntity());
        if (m == null) return;
        Fighter victim = m.a.bukkit.equals(event.getEntity()) ? m.a : m.b;
        double health = victim.bukkit.getHealth();
        double damage = Math.min(event.getFinalDamage(), health);
        // burning damage earns no reward either way, so pouring lava on someone can't be farmed;
        // a knockout from it still ends the fight
        if (!BURNING.contains(event.getCause())) {
            if (victim == m.a) m.damageToA += damage;
            else m.damageToB += damage;
        }
        if (event.getFinalDamage() >= health) {
            event.setCancelled(true);
            if (m.knockedOut == null) m.knockedOut = victim;
        }
    }

    // walls and ceiling already stop this; it's the safety net in case a pearl lands somewhere odd
    @EventHandler(ignoreCancelled = true)
    public void onPearl(PlayerTeleportEvent event) {
        if (event.getCause() != PlayerTeleportEvent.TeleportCause.ENDER_PEARL) return;
        Match m = matchOf(event.getPlayer());
        if (m != null && !m.arena.contains(event.getTo())) event.setCancelled(true);
    }

    private Match matchOf(Entity entity) {
        for (Match m : matches) {
            if (m.a != null && entity.equals(m.a.bukkit)) return m;
            if (m.b != null && entity.equals(m.b.bukkit)) return m;
        }
        return null;
    }

    private void tick() {
        // the scheduler keeps running while the world is frozen; a step is done once its stepped ticks are used up
        if (!humanMode && pendingStep != null && !tickManager.isStepping()) {
            CompletableFuture<String> done = pendingStep;
            List<Match> stepped = pendingMatches;
            pendingStep = null;
            pendingMatches = null;
            StringJoiner results = new StringJoiner("#");
            for (Match m : stepped) results.add(finishStep(m));
            done.complete(results.toString());
        }
    }

    private void ensureBot(Match m) {
        if (m.a == null || !m.a.alive()) {
            if (m.a != null) m.a.fake.despawn();
            m.a = new Fighter(FakePlayer.spawn(world, "PvpBot" + m.suffix()));
        }
    }

    // B is the "Sparring" fake player in training, a real person in human mode, or the Carpet bot in carpet mode
    private void setOpponent(Match m, boolean human, Player carpetBot) {
        if (carpetBot != null) {
            if (m.b != null && m.b.fake != null) m.b.fake.despawn();
            // compare the entity, not Player.equals: a respawned bot keeps its name and UUID
            if (m.b == null || m.b.nms != ((CraftPlayer) carpetBot).getHandle()) m.b = new Fighter(((CraftPlayer) carpetBot).getHandle());
            return;
        }
        // outside carpet mode the Carpet bot must not be around arena 0, or it would join the fight. It is logged out
        // rather than parked: switching its combat off and back on left it standing still without fighting
        Player stray = Bukkit.getPlayerExact(CARPET_BOT);
        if (m.index == 0 && stray != null) {
            Bukkit.dispatchCommand(Bukkit.getConsoleSender(), "bot " + CARPET_BOT + " disconnect");
            carpetDifficulty = null;
            carpetStyle = null;
        }
        if (human) {
            Player person = null;
            for (Player p : Bukkit.getOnlinePlayers()) {
                if (!(((CraftPlayer) p).getHandle() instanceof FakePlayer)) person = p;
            }
            if (person == null) throw new IllegalStateException("no real player is online to fight the bot");
            if (m.b != null && m.b.fake != null) m.b.fake.despawn();
            if (m.b == null || m.b.fake != null || !m.b.bukkit.equals(person)) m.b = new Fighter(((CraftPlayer) person).getHandle());
        } else if (m.b == null || m.b.fake == null || !m.b.alive()) {
            if (m.b != null && m.b.fake != null) m.b.fake.despawn();
            m.b = new Fighter(FakePlayer.spawn(world, "Sparring" + m.suffix()));
        }
    }

    // the Carpet bot attacks the nearest player in survival, so a watching player would pull it off PvpBot
    private void moveSpectatorsOutOfTheWay() {
        for (Player p : Bukkit.getOnlinePlayers()) {
            boolean fake = ((CraftPlayer) p).getHandle() instanceof FakePlayer || p.getName().equals(CARPET_BOT);
            GameMode mode = p.getGameMode();
            if (fake || (mode != GameMode.SURVIVAL && mode != GameMode.ADVENTURE)) continue;
            p.setGameMode(GameMode.SPECTATOR);
            p.sendMessage(Component.text("Switched you to spectator so the Carpet bot fights PvpBot instead of you. "
                    + "fight.py puts you back in survival."));
        }
    }

    private void applyCarpetDifficulty(String difficulty) {
        if (difficulty.equals(carpetDifficulty)) return;
        String[] trainee = CARPET_TRAINEES.get(difficulty);
        carpetOption("difficulty " + (trainee != null ? "beginner" : difficulty));
        for (String option : trainee != null ? trainee : CARPET_SWITCHES_ON) carpetOption(option);
        carpetDifficulty = difficulty;
    }

    private void carpetOption(String option) {
        Bukkit.dispatchCommand(Bukkit.getConsoleSender(), "bot option " + CARPET_BOT + " " + option);
    }

    // spawns the Carpet bot in arena 0 in the requested style if needed (a style change means logging it out and back
    // in, and a login can take a few ticks), sets its difficulty, then resets
    private void resetCarpet(Match m, String difficulty, String style, CompletableFuture<String> out) {
        tickManager.setTickRate(TICK_RATE);
        tickManager.setFrozen(true);
        ensureBot(m);
        moveSpectatorsOutOfTheWay();
        Player bot = Bukkit.getPlayerExact(CARPET_BOT);
        if (bot != null && style.equals(carpetStyle)) {
            applyCarpetDifficulty(difficulty);
            out.complete(reset(m, -1, false, bot));
            return;
        }
        if (bot != null) Bukkit.dispatchCommand(Bukkit.getConsoleSender(), "bot " + CARPET_BOT + " disconnect");
        Location c = m.arena.center;
        // the Carpet plugin finishes a login from its per-tick task, which (patched) only runs on simulated ticks,
        // so let the world run until the bot is in; reset() re-freezes it and puts everyone in place.
        // the other arenas run a few ticks on their last actions meanwhile, which only happens on a style change
        tickManager.setFrozen(false);
        carpetDifficulty = null;
        boolean[] requested = {false};
        int[] waited = {0};
        Bukkit.getScheduler().runTaskTimer(this, task -> {
            Player online = Bukkit.getPlayerExact(CARPET_BOT);
            if (!requested[0]) {
                // the old one has to be gone first, or the name is still taken
                if (online == null) {
                    Bukkit.dispatchCommand(Bukkit.getConsoleSender(), String.format(Locale.ROOT,
                            "bot spawn %s %s beginner at %.1f %.1f %.1f", CARPET_BOT, style, c.getX(), c.getY(), c.getZ()));
                    carpetStyle = style;
                    requested[0] = true;
                }
            } else if (online != null) {
                task.cancel();
                applyCarpetDifficulty(difficulty);
                out.complete(reset(m, -1, false, online));
                return;
            }
            if (++waited[0] > CARPET_SPAWN_TIMEOUT_TICKS) {
                task.cancel();
                tickManager.setFrozen(true);
                out.completeExceptionally(new IllegalStateException(
                        "the Carpet bot didn't log in; is the CarpetPvp plugin installed?"));
            }
        }, 1L, 1L);
    }

    // some fights start hurt or burning, so golden apples and the water bucket have something to do
    private void varyStart(Fighter f) {
        if (rng.nextDouble() < LOW_HEALTH_START_CHANCE) f.bukkit.setHealth(6 + rng.nextInt(7));
        if (rng.nextDouble() < ON_FIRE_START_CHANCE) f.bukkit.setFireTicks(100);
    }

    // scriptSkill < 0 means B is controlled externally (self-play, a human or the Carpet bot)
    private String reset(Match m, double scriptSkill, boolean human, Player carpetBot) {
        humanMode = human;
        m.carpet = carpetBot != null;
        tickManager.setTickRate(human ? 20f : TICK_RATE);
        tickManager.setFrozen(!human);
        ensureBot(m);
        setOpponent(m, human, carpetBot);
        for (Entity e : world.getNearbyEntities(m.arena.center, Arena.RADIUS + 3, 16, Arena.RADIUS + 3)) {
            if (e instanceof Projectile || e instanceof Item) e.remove();
        }
        m.arena.resetInterior();
        world.setTime(6000);
        m.a.equip();
        m.b.equip(m.carpet ? CARPET_GEAR.getOrDefault(carpetDifficulty, Fighter.Gear.IRON) : Fighter.Gear.IRON);
        Location[] spawns = m.arena.spawnPair(rng);
        m.a.bukkit.teleport(spawns[0]);
        m.b.bukkit.teleport(spawns[1]);
        if (!human) {
            varyStart(m.a);
            varyStart(m.b);
        }

        m.script = scriptSkill >= 0 ? new ScriptedOpponent(rng, scriptSkill) : null;
        m.steps = 0;
        m.damageToA = 0;
        m.damageToB = 0;
        m.aimA = m.a.aimQuality(m.b);
        m.aimB = m.b.aimQuality(m.a);
        m.knockedOut = null;
        return observations(m) + ";0;0;0;0;0;0;0";
    }

    // applies one match's actions for the coming step; null if fine, otherwise the problem
    private String applyActions(Match m, int[] actionA, int[] actionB) {
        if (m.a == null) return "call reset before step (arena " + m.index + ")";
        if (m.carpet) {
            if (!m.b.alive()) return "the Carpet bot is gone; reset again";
            // the Carpet bot picks its own moves on the ticks we step
            m.a.apply(actionA, m.b);
            return null;
        }
        if (m.script != null) actionB = m.script.act(m.b, m.a);
        else if (actionB == null) return "self-play step needs actions for both fighters (arena " + m.index + ")";
        m.a.apply(actionA, m.b);
        m.b.apply(actionB, m.a);
        return null;
    }

    // steps every listed match together: all actions are applied, then the world advances TICKS_PER_STEP once
    private void startStep(List<Match> stepping, List<int[]> actionsA, List<int[]> actionsB, CompletableFuture<String> out) {
        if (humanMode) {
            Match m = stepping.get(0);
            if (stepping.size() != 1 || m.index != 0) {
                out.completeExceptionally(new IllegalStateException("human mode only steps arena 0"));
                return;
            }
            if (!m.b.bukkit.isOnline()) {
                out.completeExceptionally(new IllegalStateException("the human player left"));
                return;
            }
            // the person plays live; just apply the bot's action and report back after the step's real ticks
            m.a.apply(actionsA.get(0), m.b);
            Bukkit.getScheduler().runTaskLater(this, () -> out.complete(finishStep(m)), TICKS_PER_STEP);
            return;
        }
        for (int i = 0; i < stepping.size(); i++) {
            String problem = applyActions(stepping.get(i), actionsA.get(i), actionsB.get(i));
            if (problem != null) {
                out.completeExceptionally(new IllegalStateException(problem));
                return;
            }
        }
        pendingStep = out;
        pendingMatches = stepping;
        tickManager.stepGameIfFrozen(TICKS_PER_STEP);
    }

    // response: obsA;obsB;rewardA;rewardB;terminated;truncated;outcome;damageToA;damageToB
    // (outcome 1 A won, -1 A lost, 0 otherwise; damage is this step's, after armor)
    private String finishStep(Match m) {
        m.steps++;
        boolean terminated = m.knockedOut != null;
        // after the fight ends there is no future aim, so the final potential is 0
        double nextAimA = terminated ? 0 : m.a.aimQuality(m.b);
        double nextAimB = terminated ? 0 : m.b.aimQuality(m.a);
        double rewardA = (m.damageToB - m.damageToA) / 20.0 + AIM_SHAPING * (GAMMA * nextAimA - m.aimA);
        double rewardB = (m.damageToA - m.damageToB) / 20.0 + AIM_SHAPING * (GAMMA * nextAimB - m.aimB);
        m.aimA = nextAimA;
        m.aimB = nextAimB;
        String damage = m.damageToA + ";" + m.damageToB;
        m.damageToA = 0;
        m.damageToB = 0;

        int outcome = 0;
        if (terminated) {
            outcome = m.knockedOut == m.b ? 1 : -1;
            rewardA += outcome * WIN_REWARD;
            rewardB -= outcome * WIN_REWARD;
        }
        boolean truncated = !terminated && m.steps >= MAX_STEPS;
        if (truncated) {
            double lead = (m.a.bukkit.getHealth() + m.a.bukkit.getAbsorptionAmount())
                    - (m.b.bukkit.getHealth() + m.b.bukkit.getAbsorptionAmount());
            double tiebreak = TIEBREAK * Math.signum(lead);
            rewardA += tiebreak;
            rewardB -= tiebreak;
        }
        if (humanMode && (terminated || truncated)) {
            String result = outcome == 1 ? "PvpBot wins this round." : outcome == -1 ? "You win this round!" : "Draw: time ran out.";
            Bukkit.broadcast(Component.text(result));
        }
        return observations(m) + ";" + rewardA + ";" + rewardB + ";" + (terminated ? 1 : 0) + ";"
                + (truncated ? 1 : 0) + ";" + outcome + ";" + damage;
    }

    // a snapshot of one arena for replays, as JSON with positions relative to the arena centre (floor level):
    // {"a": fighter, "b": fighter, "projectiles": [[kind, x, y, z, vx, vy, vz], ...], "liquids": [[x, y, z, lava?], ...]}
    private String state(Match m) {
        if (m.a == null || m.b == null) throw new IllegalStateException("call reset before state");
        Location c = m.arena.center;
        StringBuilder sb = new StringBuilder("{\"a\":").append(fighterState(m.a, c)).append(",\"b\":").append(fighterState(m.b, c));
        sb.append(",\"projectiles\":[");
        boolean first = true;
        for (Entity e : world.getNearbyEntities(c, Arena.RADIUS + 1, Arena.CEILING, Arena.RADIUS + 1)) {
            if (!(e instanceof Projectile)) continue;
            Location l = e.getLocation();
            org.bukkit.util.Vector v = e.getVelocity();
            if (!first) sb.append(',');
            first = false;
            sb.append(String.format(Locale.ROOT, "[\"%s\",%.2f,%.2f,%.2f,%.2f,%.2f,%.2f]", e.getType().name().toLowerCase(Locale.ROOT),
                    l.getX() - c.getX(), l.getY() - c.getY(), l.getZ() - c.getZ(), v.getX(), v.getY(), v.getZ()));
        }
        sb.append("],\"liquids\":[");
        first = true;
        int cx = c.getBlockX(), cz = c.getBlockZ(), y0 = m.arena.baseY + 1;
        for (int x = -Arena.RADIUS; x <= Arena.RADIUS; x++) {
            for (int z = -Arena.RADIUS; z <= Arena.RADIUS; z++) {
                for (int y = 0; y < 4; y++) {
                    Material type = world.getBlockAt(cx + x, y0 + y, cz + z).getType();
                    if (type != Material.LAVA && type != Material.WATER) continue;
                    if (!first) sb.append(',');
                    first = false;
                    sb.append('[').append(x).append(',').append(y).append(',').append(z).append(',').append(type == Material.LAVA ? 1 : 0).append(']');
                }
            }
        }
        return sb.append("]}").toString();
    }

    private static String fighterState(Fighter f, Location c) {
        Location l = f.bukkit.getLocation();
        return String.format(Locale.ROOT,
                "{\"name\":\"%s\",\"x\":%.3f,\"y\":%.3f,\"z\":%.3f,\"yaw\":%.1f,\"pitch\":%.1f,\"health\":%.2f,\"absorption\":%.2f,"
                        + "\"max\":%.1f,\"held\":\"%s\",\"offhand\":\"%s\",\"blocking\":%b,\"using\":%b,\"burning\":%b,\"sneaking\":%b}",
                f.bukkit.getName(), l.getX() - c.getX(), l.getY() - c.getY(), l.getZ() - c.getZ(), l.getYaw(), l.getPitch(),
                f.bukkit.getHealth(), f.bukkit.getAbsorptionAmount(), f.maxHealth(),
                f.bukkit.getInventory().getItemInMainHand().getType().name().toLowerCase(Locale.ROOT),
                f.bukkit.getInventory().getItemInOffHand().getType().name().toLowerCase(Locale.ROOT),
                f.bukkit.isBlocking(), f.nms.isUsingItem(), f.bukkit.getFireTicks() > 0, f.bukkit.isSneaking());
    }

    private String observations(Match m) {
        return join(m.a.observe(m.b, m.arena.center, Arena.RADIUS)) + ";" + join(m.b.observe(m.a, m.arena.center, Arena.RADIUS));
    }

    private static String join(double[] values) {
        StringJoiner joiner = new StringJoiner(",");
        for (double d : values) joiner.add(Double.toString(d));
        return joiner.toString();
    }

    private void serve() {
        try (ServerSocket ss = new ServerSocket(PORT, 1, InetAddress.getLoopbackAddress())) {
            serverSocket = ss;
            getLogger().info("Listening for Python on 127.0.0.1:" + PORT);
            while (!ss.isClosed()) {
                try (Socket socket = ss.accept();
                     BufferedReader in = new BufferedReader(
                             new InputStreamReader(socket.getInputStream(), StandardCharsets.UTF_8));
                     PrintWriter out = new PrintWriter(
                             new OutputStreamWriter(socket.getOutputStream(), StandardCharsets.UTF_8), true)) {
                    socket.setTcpNoDelay(true);
                    getLogger().info("Python connected");
                    String line;
                    while ((line = in.readLine()) != null) {
                        out.println(handle(line.trim()));
                    }
                    getLogger().info("Python disconnected");
                } catch (IOException e) {
                    if (!ss.isClosed()) getLogger().warning("Connection dropped: " + e.getMessage());
                }
            }
        } catch (IOException e) {
            if (serverSocket == null || !serverSocket.isClosed()) {
                getLogger().severe("Could not open port " + PORT + ": " + e.getMessage());
            }
        }
    }

    private static int[] parseAction(String text) {
        String[] parts = text.trim().split("\\s+");
        if (parts.length != Fighter.HEADS.length) {
            throw new IllegalArgumentException("expected " + Fighter.HEADS.length + " action values, got " + parts.length);
        }
        int[] action = new int[parts.length];
        for (int i = 0; i < parts.length; i++) {
            action[i] = Integer.parseInt(parts[i]);
            if (action[i] < 0 || action[i] >= Fighter.HEADS[i]) {
                throw new IllegalArgumentException("action " + i + " out of range: " + action[i]);
            }
        }
        return action;
    }

    private Match arena(String index) {
        int i = Integer.parseInt(index.trim());
        if (i < 0 || i >= ARENAS) throw new IllegalArgumentException("arena must be 0.." + (ARENAS - 1) + ", got " + i);
        return matches.get(i);
    }

    // protocol: "info" -> obs_size;head sizes (comma separated);max_steps;number of arenas
    //           "reset script <skill>" | "reset external" | "reset human" (an online player fights the bot live)
    //             | "reset carpet <trainee1|trainee2|trainee3|beginner|casual|average|skilled|expert> [smp|ranged]"
    //             any reset can be prefixed with "@<arena> " (default arena 0); carpet and human are arena 0 only
    //           "step <8 values>" (scripted B, Carpet bot or human) | "step <8 values> | <8 values>" (self-play): arena 0
    //             -> obsA;obsB;rewardA;rewardB;terminated;truncated;outcome;damageToA;damageToB
    //           "stepall <arena>:<8 values>[ | <8 values>] / <arena>:..." steps those arenas together in one go
    //             -> one result per listed arena, in the same order, separated by "#"
    //           "state" (optionally "@<arena> state") -> JSON snapshot of that arena, for replays
    //           "shutdown" stops the server (used by automated tests)
    private String handle(String line) {
        if (line.equals("info")) {
            StringJoiner heads = new StringJoiner(",");
            for (int h : Fighter.HEADS) heads.add(Integer.toString(h));
            return Fighter.OBS_SIZE + ";" + heads + ";" + MAX_STEPS + ";" + ARENAS;
        }
        if (line.equals("shutdown")) {
            Bukkit.getScheduler().runTask(this, Bukkit::shutdown);
            return "ok";
        }

        CompletableFuture<String> result = new CompletableFuture<>();
        Runnable task;
        try {
            Match target = matches.get(0);
            if (line.startsWith("@")) {
                String[] split = line.substring(1).split("\\s+", 2);
                target = arena(split[0]);
                line = split.length > 1 ? split[1].trim() : "";
            }
            Match m = target;
            String[] words = line.split("\\s+", 2);
            if (words[0].equals("reset")) {
                String arg = words.length > 1 ? words[1].trim() : "script 0";
                if (arg.startsWith("carpet")) {
                    if (m.index != 0) return "error carpet fights only run in arena 0";
                    String[] carpetArgs = arg.substring(6).trim().toLowerCase(Locale.ROOT).split("\\s+");
                    String difficulty = carpetArgs[0];
                    String style = carpetArgs.length > 1 ? carpetArgs[1] : "smp";
                    if (!CARPET_STYLES.contains(style)) return "error carpet style must be one of " + CARPET_STYLES;
                    if (!CARPET_PRESETS.contains(difficulty) && !CARPET_TRAINEES.containsKey(difficulty)) {
                        return "error carpet difficulty must be one of " + CARPET_TRAINEES.keySet() + " or " + CARPET_PRESETS;
                    }
                    task = () -> resetCarpet(m, difficulty, style, result);
                } else {
                    double skill;
                    boolean human = arg.equals("human");
                    if (human && m.index != 0) return "error human fights only run in arena 0";
                    if (arg.equals("external") || human) skill = -1;
                    else if (arg.startsWith("script")) skill = Math.max(0, Math.min(1, Double.parseDouble(arg.substring(6).trim())));
                    else return "error reset expects 'script <skill>', 'external', 'human' or 'carpet <difficulty>'";
                    task = () -> result.complete(reset(m, skill, human, null));
                }
            } else if (words[0].equals("step") && words.length > 1) {
                String[] sides = words[1].split("\\|");
                List<int[]> actionsA = new ArrayList<>(), actionsB = new ArrayList<>();
                actionsA.add(parseAction(sides[0]));
                actionsB.add(sides.length > 1 ? parseAction(sides[1]) : null);
                task = () -> startStep(List.of(m), actionsA, actionsB, result);
            } else if (words[0].equals("state")) {
                task = () -> result.complete(state(m));
            } else if (words[0].equals("stepall") && words.length > 1) {
                List<Match> stepping = new ArrayList<>();
                List<int[]> actionsA = new ArrayList<>(), actionsB = new ArrayList<>();
                for (String entry : words[1].split("/")) {
                    String[] parts = entry.split(":", 2);
                    if (parts.length != 2) return "error stepall entries look like <arena>:<actions>";
                    Match stepped = arena(parts[0]);
                    if (stepping.contains(stepped)) return "error arena " + stepped.index + " listed twice";
                    String[] sides = parts[1].split("\\|");
                    stepping.add(stepped);
                    actionsA.add(parseAction(sides[0]));
                    actionsB.add(sides.length > 1 ? parseAction(sides[1]) : null);
                }
                task = () -> startStep(stepping, actionsA, actionsB, result);
            } else {
                return "error unknown command: " + line;
            }
        } catch (IllegalArgumentException e) {
            return "error " + e.getMessage();
        }

        String command = line;
        Bukkit.getScheduler().runTask(this, () -> {
            try {
                task.run();
            } catch (Exception e) {
                getLogger().severe("Command failed: " + command + " -> " + e);
                result.completeExceptionally(e);
            }
        });
        try {
            return result.get(30, TimeUnit.SECONDS);
        } catch (Exception e) {
            return "error " + e;
        }
    }
}
