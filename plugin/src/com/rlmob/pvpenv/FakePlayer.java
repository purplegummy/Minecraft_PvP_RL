package com.rlmob.pvpenv;

import com.mojang.authlib.GameProfile;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ClientInformation;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.server.network.CommonListenerCookie;
import org.bukkit.Bukkit;
import org.bukkit.World;
import org.bukkit.craftbukkit.CraftServer;
import org.bukkit.craftbukkit.CraftWorld;

import java.nio.charset.StandardCharsets;
import java.util.UUID;

// a server-side player with no client: the server runs its physics and Python supplies its inputs
final class FakePlayer extends ServerPlayer {
    // keyboard-style inputs, re-applied every tick because aiStep decays them
    float forward;
    float strafe;
    boolean jump;
    boolean sprint;

    private FakePlayer(MinecraftServer server, ServerLevel level, GameProfile profile) {
        super(server, level, profile, ClientInformation.createDefault());
    }

    static FakePlayer spawn(World world, String name) {
        MinecraftServer server = ((CraftServer) Bukkit.getServer()).getServer();
        ServerLevel level = ((CraftWorld) world).getHandle();
        // a fixed UUID per name so the same player data file is reused across restarts
        UUID id = UUID.nameUUIDFromBytes(("rlmob-pvp:" + name).getBytes(StandardCharsets.UTF_8));
        GameProfile profile = new GameProfile(id, name);
        FakePlayer player = new FakePlayer(server, level, profile);
        FakeConnection connection = new FakeConnection();
        server.getPlayerList().placeNewPlayer(connection, player, CommonListenerCookie.createInitial(profile, false));
        return player;
    }

    void despawn() {
        ((CraftServer) Bukkit.getServer()).getServer().getPlayerList().remove(this);
    }

    // vanilla players are moved by their client; returning false makes the server simulate this one
    @Override
    public boolean isClientAuthoritative() {
        return false;
    }

    @Override
    public void tick() {
        // players are exempt from /tick freeze, so skip ticks the rest of the world isn't simulating
        if (!level().tickRateManager().runsNormally()) return;
        zza = forward;
        xxa = strafe;
        setJumping(jump);
        setSprinting(sprint && forward > 0 && getFoodData().getFoodLevel() > 6 && !isUsingItem());
        super.tick();
        doTick();
        level().getChunkSource().move(this);
    }
}
