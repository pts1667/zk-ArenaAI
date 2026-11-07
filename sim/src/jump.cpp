// Jumping and digging in.
//
// unit_jumpjets.lua, for every mobile unit with canjump.  A jump's arc has a height, which is worked out for what a blast or a bomb's reach needs and
// nowhere else: the sim is flat, and a jumper is shot at where it is on the map.  Left out: units landing at the same place spreading out (separate()
// does that once they're down), bouncing off a structure and jump modules' changes to range and reload.
//
// The scripts' GG.SetWantedCloaked for what has idle_cloak: Imp, Snitch, Flea and Blastwing.
#include <algorithm>
#include <cmath>

#include "zksim/world.hpp"

namespace zk {

namespace {
constexpr float WALL_STEP = 0.015f;         // Jump's test along the arc
constexpr float WALL_LEEWAY = 30.0f;        // ... which ground this far above the arc doesn't fail
constexpr float CANCEL_FACTOR = 2.0f;       // DISARM_CANCEL_FACTOR: disarmed for this many times its delay, a take-off is off
}  // namespace

float World::jump_altitude(int u) const {
    const JumpState& j = jump[size_t(u)];
    if (!j.flying) return skid[size_t(u)].y;
    const float bend = 2.0f * j.at - 1.0f;
    return (1.0f - bend * bend) * j.height;
}

// AllowCommand's ValidJumpLocation, which is stricter here: it also lets a unit land in water too deep for it, where the sim has nothing for it to do next.
// A structure doesn't count, as the map's terrain doesn't know of one.
bool World::can_land_at(int u, float gx, float gz) const {
    const int cls = type_of(u).move_class;
    return cls < 0 || map->speed_at(cls, gx, gz) > 0.0f;
}

// Jump: it won't leave if the ground rises above the arc.  AttemptJump leaves the order there to try again.
bool World::arc_is_clear(int u, float gx, float gz) const {
    const float px = x[size_t(u)], pz = z[size_t(u)], start = map->height_at(px, pz), rise = map->height_at(gx, gz) - start;
    const float height = type_of(u).jump_height;
    for (float i = 0.0f; i <= 1.0f; i += WALL_STEP) {
        const float bend = 2.0f * i - 1.0f;
        if (map->height_at(px + (gx - px) * i, pz + (gz - pz) * i) - WALL_LEEWAY > start + rise * i + (1.0f - bend * bend) * height) return false;
    }
    return true;
}

// AttemptJump: walks into range (wants_to_walk), then goes as soon as it's charged and can.
void World::try_jump(int u) {
    const size_t i = size_t(u);
    JumpState& j = jump[i];
    const auto& d = type_of(u);
    if (j.busy || order[i] != ZK_ORDER_JUMP || !active(u) || disarmed(u) || j.charge < 1.0f) return;
    const float gx = goal_x[i], gz = goal_z[i], line = std::hypot(gx - x[i], gz - z[i]);
    if (line >= d.jump_range || !arc_is_clear(u, gx, gz)) return;
    // A flight's length and so its time go by the two straight lines through the top of the arc.
    const float rise = map->height_at(gx, gz) - map->height_at(x[i], z[i]);
    const float flight = std::hypot(line * 0.5f, rise * 0.5f + d.jump_height) + std::hypot(line * 0.5f, d.jump_height - rise * 0.5f);
    j.busy = true;
    j.flying = d.jump_delay <= 0.0f;
    j.delay = d.jump_delay;
    j.timeout = d.jump_delay * CANCEL_FACTOR;
    j.at = 0.0f;
    j.step = 1.0f / std::ceil(flight / d.jump_speed);
    j.from_x = x[i];
    j.from_z = z[i];
    j.to_x = gx;
    j.to_z = gz;
    j.charge = 0.0f;
    j.height = d.jump_height;
    j.sideways = 0.0f;
    j.thrown_by = -1;
    skid[i] = Skid{};
    if (line > 0.0f) heading[i] = std::atan2(gx - x[i], gz - z[i]);
    vx[i] = vz[i] = speed[i] = 0.0f;
}

void World::land(int u) {
    const size_t i = size_t(u);
    JumpState& j = jump[i];
    const auto& d = type_of(u);
    j.busy = j.flying = false;
    x[i] = j.to_x;
    z[i] = j.to_z;
    vx[i] = vz[i] = speed[i] = 0.0f;
    const bool thrown_here = j.thrown_by >= 0;
    j.thrown_by = -1;
    if (d.landing_weapon >= 0 && !thrown_here) {
        show_for_action(u);
        explode(d.landing_weapon, x[i], z[i], u, side[i], false);
    }
    // Landing where it was told to finishes the order.
    if (order[i] == ZK_ORDER_JUMP && std::abs(goal_x[i] - j.to_x) < 1.0f && std::abs(goal_z[i] - j.to_z) < 1.0f) order[i] = ZK_ORDER_NONE;
    // A game's Pyro told to move, fight or attack in the air stood where it landed for as long as it was watched, 10 s.  The move order lapsed within 12 s, and
    // that's taken as at once here.  The others stay as they are until it's told something else.
    if (j.stranded && order[i] == ZK_ORDER_MOVE) {
        order[i] = ZK_ORDER_NONE;
        j.stranded = false;
    }
}

void World::step_jump(int u) {
    const size_t i = size_t(u);
    JumpState& j = jump[i];
    const float frames = float(ZK_TICK_FRAMES);
    const auto& d = type_of(u);
    if (!j.busy) {
        // Stun stops the reload, and disarm and slow take it with everything else that reloads.
        if (j.charge < 1.0f && active(u) && !disarmed(u)) j.charge = std::min(1.0f, j.charge + frames / d.jump_reload * (1.0f - slow_now(u)));
        return;
    }
    if (!j.flying) {
        if (!stunned(u)) {
            const float rate = disarmed(u) ? 0.0f : 1.0f - slow_now(u);
            j.delay -= frames * rate;
            if (rate <= 0.0f) j.timeout -= frames;
        }
        if (j.timeout <= 0.0f) {
            j.busy = false;     // The order stays, for when it has charged again.
            return;
        }
        if (j.delay > 0.0f) return;
        j.flying = true;
    }
    // A flight isn't stopped by anything, and a jumper shows itself all the way, but for what has its own cloak (no script of those calls PokeDecloakUnit) or lands
    // with a blast, which does so only then.
    // What's thrown is shown whatever it is (unit_thrower.lua).
    if (j.thrown_by >= 0 || (!(d.flags & ZK_U_CAN_CLOAK) && d.landing_weapon < 0)) show_for_action(u);
    j.at = std::min(1.0f, j.at + j.step * frames);
    // The sideways push is given while it's going up, and is lost if the thrower can't finish it.  It comes back down where it was.
    if (j.thrown_by >= 0 && j.at < j.sideways && (!alive[size_t(j.thrown_by)] || hp[size_t(j.thrown_by)] <= 0.0f || stunned(j.thrown_by) || disarmed(j.thrown_by))) {
        j.to_x = j.from_x;
        j.to_z = j.from_z;
    }
    const float across = j.sideways > 0.0f ? std::clamp((j.at - j.sideways) / (1.0f - j.sideways), 0.0f, 1.0f) : j.at;
    const float nx = j.from_x + (j.to_x - j.from_x) * across, nz = j.from_z + (j.to_z - j.from_z) * across;
    vx[i] = (nx - x[i]) / frames;
    vz[i] = (nz - z[i]) / frames;
    speed[i] = std::hypot(vx[i], vz[i]);
    x[i] = nx;
    z[i] = nz;
    if (j.at >= 1.0f) land(u);
}

void World::update_jumps() {
    const bool slow_update = tick_count % 3 == 0;
    for (int u : jumpers) {
        if (!alive[size_t(u)]) continue;
        if (slow_update) try_jump(u);
        step_jump(u);
    }
    for (int u : thrown)
        if (alive[size_t(u)]) step_jump(u);
    thrown.erase(std::remove_if(thrown.begin(), thrown.end(), [this](int u) { return !alive[size_t(u)] || !jump[size_t(u)].busy; }), thrown.end());
}

// A ground unit stops (GG.StartStopMovingControl), and digs in a moment later.  Starting to move brings it straight back up.  A Blastwing polls for being slow
// and near the ground, and with no orders it's pushed down to it; there's no height here, so slow and idle is what it comes to.
void World::update_burrows() {
    for (int u : burrowers) {
        const size_t i = size_t(u);
        if (!alive[i]) continue;
        const auto& d = type_of(u);
        const bool still = speed[i] < d.burrow_speed && (!(d.flags & ZK_U_AIR) || (order[i] == ZK_ORDER_NONE && ai[i].chasing < 0));
        if (still) {
            if (frame >= burrow_at[i]) want_cloak[i] = 1;
        } else {
            want_cloak[i] = 0;
            burrow_at[i] = frame + BURROW_FRAMES;
        }
    }
}

}  // namespace zk
