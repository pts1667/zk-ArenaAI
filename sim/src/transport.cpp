// Air transports: Charon and Hercules.  A transport flies to what it's told to load, comes down over it and picks it up; flies
// it to where it's told to unload, comes down and sets it down; or drops it from where it is, and it falls.
//
// The engine's part is MobileCAI.cpp (ExecuteLoadUnits, ExecuteUnloadUnit) and CUnit::CanTransport; Zero-K's is
// unit_transport_load_unload.lua (how near it must be, who can be picked up, the classes of load), unit_transport_speed.lua (a
// loaded transport is slower by its load's class) and the Drop Cargo button (oneclick_weapon_defs.lua, ForceDropUnit, which a
// crashing transport's script calls too).  The fall and its damage are unit_fall_damage.lua's, as for anything thrown or pushed
// (abilities.cpp step_skid).
//
// What's carried is out of play, as in the engine, where it's stunned and isn't a firing platform: it does nothing, and here
// nothing can shoot it either (Zero-K's modrules let a transported unit be targeted: left out, 01-simulator.md).  It's off the
// grid, so nothing finds it to shoot, push or blow up, and it's seen exactly when its transport is.
#include <algorithm>
#include <cmath>

#include "zksim/world.hpp"

namespace zk {

namespace {
constexpr float LOADING_RADIUS = 220.0f;      // UnitDef's default loadingRadius: this near, it comes down to its load's height
constexpr float UNLOADING_SHARE = 0.9f;       // ... and to unload, within this much of it
constexpr float LOAD_REACH = 16.0f;           // AllowUnitTransportLoad: this near where it picks up, height counted
constexpr float UNLOAD_REACH = 12.0f;         // AllowUnitTransportUnload: the same, to set down
constexpr float UNLOAD_REACH_FLAT = 8.0f;     // ... or this near across, and all but still
constexpr float UNLOAD_STILL = 0.25f;         // elmos a frame
constexpr float STANDS_STILL_WITHIN = 80.0f;  // a load of its own allyteam is told to wait once its transport is this near
constexpr float ENEMY_STILL = 0.5f;           // elmos a frame: an enemy faster than this can't be picked up
constexpr int DROP_RELOAD = 210;              // frames: Drop Cargo's reloadTime, 7 s
constexpr float ON_THE_GROUND = 1.0f;         // let go this low, there's no fall
}  // namespace

// CUnit::CanTransport and AllowUnitTransport: a class it lifts, and a footprint that fits.
bool World::can_lift(int u, int t) const {
    const auto& d = type_of(u);
    const auto& load = type_of(t);
    const float share = load.load_class == 1 ? d.lift_light : load.load_class == 2 ? d.lift_medium : load.load_class == 3 ? d.lift_heavy : 0.0f;
    return share > 0.0f && load.load_size <= d.lift_size && build[size_t(t)] >= 1.0f && u != t;
}

// Its top speed with what it carries, as a share of its own.
float World::loaded_speed(int u) const {
    const int c = cargo[size_t(u)];
    if (c < 0) return 1.0f;
    const auto& d = type_of(u);
    const int k = type_of(c).load_class;
    return k == 1 ? d.lift_light : k == 2 ? d.lift_medium : d.lift_heavy;
}

void World::pick_up(int u, int t) {
    const size_t i = size_t(t);
    carrier[i] = u;
    cargo[size_t(u)] = t;
    x[i] = x[size_t(u)];
    z[i] = z[size_t(u)];
    vx[i] = vz[i] = speed[i] = 0.0f;
    skid[i] = Skid{};
    // It does nothing until it's down again, and what it was told stays for then.
    ai[i].chasing = -1;
    ai[i].dodging = false;
    for (int m = 0; m < ZK_MAX_MOUNTS; ++m) {
        weapon_target[i * ZK_MAX_MOUNTS + size_t(m)] = -1;
        salvo_left[i * ZK_MAX_MOUNTS + size_t(m)] = 0;
    }
}

// Set down where it was told to, or let go in the air to fall with the transport's own speed.
void World::let_go(int u, bool falls) {
    const int c = cargo[size_t(u)];
    if (c < 0) return;
    const size_t i = size_t(c);
    const float height = carried_altitude(c);
    cargo[size_t(u)] = -1;
    carrier[i] = -1;
    x[i] = x[size_t(u)];
    z[i] = z[size_t(u)];
    skid[i] = Skid{};
    vx[i] = vz[i] = speed[i] = 0.0f;
    if (!falls || height <= ON_THE_GROUND) return;
    Skid& s = skid[i];
    s.on = true;
    s.vx = vx[size_t(u)];
    s.vz = vz[size_t(u)];
    s.y = height;
}

// Drop Cargo: now, from where it is, and not again for a while.
void World::drop_cargo(int u) {
    const size_t i = size_t(u);
    if (cargo[i] < 0 || frame < drop_ready[i] || !active(u)) return;
    drop_ready[i] = frame + DROP_RELOAD;
    let_go(u, true);
    if (order[i] == ZK_ORDER_UNLOAD) order[i] = ZK_ORDER_NONE;
}

void World::update_transports() {
    for (int u : transports) {
        const size_t i = size_t(u);
        if (!alive[i]) continue;
        AirState& a = air[i];
        a.want = -1.0f;
        if (!active(u)) continue;
        if (order[i] == ZK_ORDER_LOAD) {
            const int t = order_target[i];
            if (t < 0 || !alive[size_t(t)] || carrier[size_t(t)] >= 0 || cargo[i] >= 0 || !can_lift(u, t)) {
                order[i] = ZK_ORDER_NONE;
                continue;
            }
            const float across = std::hypot(x[size_t(t)] - x[i], z[size_t(t)] - z[i]);
            const bool friendly = side[size_t(t)] == side[i];
            if (friendly && across <= STANDS_STILL_WITHIN) held_for_lift[size_t(t)] = frame;
            if (across > LOADING_RADIUS) continue;
            const float low = type_of(t).load_height;
            a.want = low;
            // An enemy: only one that's in sight, shows, and has all but stopped.
            const bool can = friendly || (!cloaked[size_t(t)] && in_sight(side[i], t) && std::hypot(vx[size_t(t)], vz[size_t(t)]) <= ENEMY_STILL);
            const bool grounded = !jump[size_t(t)].busy && skid[size_t(t)].y <= 0.0f;
            if (can && grounded && a.phase == AIR_FLYING && std::hypot(across, a.height - low) <= LOAD_REACH) {
                pick_up(u, t);
                order[i] = ZK_ORDER_NONE;
            }
        } else if (order[i] == ZK_ORDER_UNLOAD) {
            const int c = cargo[i];
            if (c < 0) {
                order[i] = ZK_ORDER_NONE;
                continue;
            }
            const float across = std::hypot(goal_x[i] - x[i], goal_z[i] - z[i]);
            if (across >= LOADING_RADIUS * UNLOADING_SHARE) continue;
            const float low = type_of(c).load_height;
            a.want = low;
            const bool near = std::hypot(across, a.height - low) <= UNLOAD_REACH || (across <= UNLOAD_REACH_FLAT && speed[i] <= UNLOAD_STILL);
            if (near && a.height - low <= UNLOAD_REACH) {
                let_go(u, false);
                // Where it was told: the order was only taken for a place its load can stand.
                x[size_t(c)] = goal_x[i];
                z[size_t(c)] = goal_z[i];
                order[i] = ZK_ORDER_NONE;
            }
        }
    }
}

}  // namespace zk
