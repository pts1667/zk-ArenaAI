// Water (01-simulator.md, "Water"): who can aim at whom in it and fire from it, as the engine has it (Weapon.cpp TestTarget and
// CanFire, TorpedoLauncher.cpp, BombDropper.cpp, CSolidObject::UpdatePhysicalState), and the amphibious units that float up to
// fire (unit_impulsefloat_toggle.lua with LuaRules/Configs/float_defs.lua and the units' scripts).
#include <algorithm>
#include <cmath>

#include "zksim/world.hpp"

namespace zk {

namespace {
// The engine slows what the gadget has a floating unit do, in water: each frame it goes its speed less this times the speed's
// square.  Fitted to the engine's rises of five types from 150 deep, frame by frame (tools/sim_engine.py water): 0.069 to 0.076
// each, Archer's 5 to 9.5 elmos a frame among them.
constexpr float WATER_DRAG = 0.07f;
// The gadget looks at a unit's orders, and whether it's stunned, every 16 frames, at these phases (gadget:GameFrame).
constexpr int FLOAT_EVERY = 16, FLOAT_ORDER_PHASE = 12, FLOAT_STUN_PHASE = 4;
}  // namespace

// The engine's pos.y: on the ground or the sea bed, riding the water (GroundMoveType::GetGroundHeight, StaticMoveType), wherever
// it has floated to, or in the air, which is over the water for what flies or is carried.
float World::base_y(int u) const {
    const size_t i = size_t(u);
    if (floating[i]) return float_y[i];
    const float g = map->height_at(x[i], z[i]);
    if (carrier[i] >= 0) return std::max(g, 0.0f) + flight_height(u) + carried_altitude(u);
    return std::max(g, type_of(u).water_y) + flight_height(u) + jump_altitude(u);
}

void World::update_water() {
    for (const int u : live) {
        const size_t i = size_t(u);
        const auto& d = type_of(u);
        base_at[i] = base_y(u);
        wet[i] = uint8_t((base_at[i] <= 0.0f ? WET_IN : 0) | (base_at[i] + d.mid_height + d.body_radius < 0.0f ? WET_UNDER : 0));
    }
}

// Weapon.cpp TestTarget: what isn't a water weapon aims at nothing under water from above it, and from under water at nothing in
// it.  TorpedoLauncher.cpp, and BombDropper.cpp for a plane's: a torpedo aims at what's in water, whether its muzzle is or not.
// surfaced: as if the shooter were up at its float point, for what floats up to fire (may_float).
bool World::water_allows(int u, const zk_weapon_rec& w, int t, bool surfaced) const {
    if (!has_water) return true;
    const uint8_t at = wet[size_t(t)];
    if (w.flags & ZK_W_TORPEDO) return (at & WET_IN) != 0;
    if (w.flags & ZK_W_WATER) return true;
    if (!surfaced && (wet[size_t(u)] & WET_UNDER)) return !(at & WET_IN);
    return !(at & WET_UNDER);
}

// What a shot meets on its way: one that isn't a water weapon goes off at the surface, over what's under it, and a torpedo keeps
// under the surface (weapon_torpedo_stay_underwater.lua), below what's out of the water.
bool World::water_hits(const zk_weapon_rec& w, int t) const {
    if (!has_water) return true;
    const uint8_t at = wet[size_t(t)];
    return (w.flags & ZK_W_TORPEDO) ? (at & WET_IN) != 0 : (w.flags & ZK_W_WATER) || !(at & WET_UNDER);
}

// CWeapon::CanFire: what isn't a water weapon doesn't fire with its aim-from point at or under the surface.  And amphimpulse.lua's
// BlockShot holds its fire while the unit is deep.
bool World::fires_in_water(int u, int mount, const zk_weapon_rec& w) const {
    if (!has_water) return true;
    const float b = base_at[size_t(u)];
    if (!(w.flags & ZK_W_WATER) && b + type_of(u).mount_aim[mount] <= 0.0f) return false;
    const zk_float_rec* f = cat->float_def(type[size_t(u)]);
    return !f || b >= f->block_below;
}

// What floats up to fire aims as if it were up already: it's floating, or would go up from here, not told to move and deeper than
// its float's depth (addFloat).  The Buoy's, Grizzly's and Bulkhead's fake weapons, water weapons that aim with the real one, are
// what float those in a game: aiming the real one as from the surface comes to the same.
bool World::may_float(int u) const {
    const size_t i = size_t(u);
    if (floating[i]) return true;
    const zk_float_rec* f = cat->float_def(type[i]);
    return f && order[i] != ZK_ORDER_MOVE && carrier[i] < 0 && base_at[i] < f->depth;
}

// GG.Floating_AimWeapon, as the scripts call it: a weapon has a target, and for some only once the first weapon has nearly
// reloaded, or has fired since the gadget last looked.
bool World::float_aims(int u, const zk_float_rec& f) const {
    const auto& d = type_of(u);
    bool aims = false;
    for (int m = 0; m < d.n_mounts && !aims; ++m) {
        const int t = weapon_target[size_t(u) * ZK_MAX_MOUNTS + size_t(m)];
        aims = t >= 0 && alive[size_t(t)];
    }
    if (!aims || f.reload_frames <= 0.0f) return aims;
    const float left = cooldown[size_t(u) * ZK_MAX_MOUNTS];
    return left < f.reload_frames || left > cat->weapons[size_t(d.mount_weapon[0])].reload - float(FLOAT_EVERY);
}

// unit_impulsefloat_toggle.lua's GameFrame, a frame at a time: what aims, isn't told to move and stands deep enough leaves the sea
// bed at its initial speed, rises to its float point, where it treads water, and sinks back once it hasn't aimed for a look of the
// gadget's, or is told to move, or is stunned if it sinks then.  Back on the sea bed it's let go.  It stays where it is meanwhile.
void World::update_floats() {
    for (const int u : floaters) {
        const size_t i = size_t(u);
        if (!alive[i]) continue;
        if (carrier[i] >= 0) {      // UnitLoaded: lifted out of the water
            floating[i] = 0;
            continue;
        }
        const zk_float_rec& f = *cat->float_def(type[i]);
        const bool moving = order[i] == ZK_ORDER_MOVE;
        const float ground = map->height_at(x[i], z[i]);
        if (!floating[i]) {
            if (build[i] < 1.0f || stunned(u) || moving || ground >= f.depth || !float_aims(u, f)) continue;
            floating[i] = 1;
            float_up[i] = 1;
            float_y[i] = ground;
            float_speed[i] = f.init_speed;
            float_since[i] = -1;
        }
        float& y = float_y[i];
        float& v = float_speed[i];
        for (int fr = frame + 1; fr <= frame + ZK_TICK_FRAMES; ++fr) {
            const int phase = fr % FLOAT_EVERY;
            if (phase == FLOAT_STUN_PHASE && f.sink_on_para && stunned(u)) float_up[i] = 0;
            if (phase == FLOAT_ORDER_PHASE) {
                if (!stunned(u)) float_up[i] = uint8_t(!moving && float_aims(u, f));
                else if (f.sink_on_para) float_up[i] = 0;
            }
            // In water up to its float point, and out of it a surface_dampen above, the two blended between.
            float in = 1.0f;
            if (y > f.float_point) in = f.surface_dampen > 0.0f && y - f.float_point <= f.surface_dampen ? 1.0f - (y - f.float_point) / f.surface_dampen : 0.0f;
            // Near the surface for a while, what sinks it first pushes it up, then less, then sinks it.
            float push = 1.0f;
            if (f.wait_max > 0.0f) {
                if (y <= f.wait_y) float_since[i] = -1;
                else if (float_since[i] < 0) float_since[i] = fr;
                const float t = float(fr - float_since[i]);
                if (float_since[i] >= 0 && t < f.wait_max)
                    push = t < f.wait_min ? -0.5f : 1.5f * (t - f.wait_min) / (f.wait_max - f.wait_min) - 0.5f;
            }
            float water = 0.0f, air = 0.0f;
            if (in != 0.0f)
                water = float_up[i] ? (v + f.rise_accel) * (v > 0.0f ? f.rise_up_drag : f.rise_down_drag)
                                    : (v + push * f.sink_accel) * (v > 0.0f ? f.sink_up_drag : f.sink_down_drag);
            if (in != 1.0f) air = (v + f.air_accel) * f.air_drag;
            v = in * water + (1.0f - in) * air;
            // Falling back into the water from above its float point.
            const float above = y - f.float_point;
            if (v < 0.0f && above > 0.0f && above < -v) v *= f.water_hit_drag;
            y += y <= 0.0f ? v - WATER_DRAG * v * std::fabs(v) : v;
            if (y <= ground) {      // Float_stopOnFloor
                floating[i] = 0;
                break;
            }
        }
    }
}

}  // namespace zk
