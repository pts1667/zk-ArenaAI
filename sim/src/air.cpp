// Aircraft.  Planes fly on wings: they keep moving, turn in a circle, and fly over what they attack and come round again.
// Gunships hover, as ground units without the ground (movement.cpp), and those that strafe circle what they attack
// (unit_ai.cpp walk_to_target).  What a plane factory makes lands when it has nothing to do (unit_air_plants.lua).  Bombers
// carry a salvo, then fly to a pad, rearm and repair there (unit_bomber_command.lua, unit_refuel_pad_handler.lua).
//
// The engine's plane is 3D, with rudder, elevator and bank (StrafeAirMoveType.cpp).  This one turns at a fixed rate and slows
// in a hard turn, both fitted to the engine (tools/sim_engine.py air; 01-simulator.md, "Aircraft").
#include <algorithm>
#include <cmath>

#include "zksim/world.hpp"

namespace zk {

namespace {
constexpr float PI = 3.14159265358979f;
constexpr float LANDED_HEIGHT = 10.0f;        // lower than this an aircraft is on the ground, for what can hit it
constexpr int PLANE_TAKEOFF_FRAMES = 20;      // a plane climbs this long before it moves off: 8 to 34 measured
constexpr float HOVER_TAKEOFF_SHARE = 0.9f;   // a gunship climbs to this much of its height first: 30 to 50 frames measured
constexpr float PLANE_ARRIVES = 100.0f;       // a plane's move is over this far from its goal: it lands 41 to 142 short
constexpr float TURN_SLOWS = 0.5f;            // radians off its goal: past this a plane slows to its turning speed
constexpr float DRAG_SPEED = 1.1f;            // StrafeAirMoveType.cpp: drag is maxAcc over 1.1 times top speed
constexpr float PAD_SNAP = 80.0f;             // unit_refuel_pad_handler.lua padSnapRangeSqr
constexpr float REPAIR_COST_FACTOR = 0.666f;  // unitdefs_post.lua's default for Game.repairEnergyCostFactor
constexpr float FRAMES_PER_SECOND = 30.0f;
constexpr float BOMB_NEAR = 8.0f;             // BombDropper.cpp: within this it drops whatever the spread
// StrafeAirMoveType.cpp UpdateFlying: a goal this close and this far round has the plane turn the other way, so it doesn't
// circle the goal for ever.
constexpr float NEAR_GOAL_SHARE = 0.5f, NEAR_GOAL_DOT = 0.7f, BEHIND_DOT = -0.1f;

float wrap(float a) {
    while (a > PI) a -= 2.0f * PI;
    while (a < -PI) a += 2.0f * PI;
    return a;
}
}  // namespace

bool World::airborne(int u) const { return (type_of(u).flags & ZK_U_AIR) && air[size_t(u)].height > LANDED_HEIGHT; }

// Where an aircraft wants to go, if anywhere.  A plane flies at what it attacks, or at anything its weapons have, rather than
// stopping in range, unless the tactical AI has given it a move to make first (a fighter told to loop back, unit_ai.cpp); one
// with no ammo makes for its pad.
bool World::air_goal(int u, float& gx, float& gz) {
    const auto& d = type_of(u);
    const AirState& a = air[size_t(u)];
    if (!active(u)) return false;
    if (out_of_ammo(u)) {
        // It finishes its run first, going straight on (bomberdisarm.lua reloads at the end of one).
        for (int m = 0; m < d.n_mounts; ++m)
            if (salvo_left[size_t(u) * ZK_MAX_MOUNTS + size_t(m)] > 0) {
                gx = x[size_t(u)] + std::sin(heading[size_t(u)]) * PLANE_ARRIVES * 2.0f;
                gz = z[size_t(u)] + std::cos(heading[size_t(u)]) * PLANE_ARRIVES * 2.0f;
                return true;
            }
        if (a.pad < 0 || !alive[size_t(a.pad)]) return false;
        gx = x[size_t(a.pad)];
        gz = z[size_t(a.pad)];
        return true;
    }
    if ((d.flags & ZK_U_PLANE) && !ai[size_t(u)].dodging) {
        int t = order[size_t(u)] == ZK_ORDER_ATTACK ? order_target[size_t(u)] : ai[size_t(u)].chasing;
        for (int m = 0; m < d.n_mounts && (t < 0 || !alive[size_t(t)]); ++m) t = weapon_target[size_t(u) * ZK_MAX_MOUNTS + size_t(m)];
        if (t >= 0 && alive[size_t(t)]) {
            gx = x[size_t(t)];
            gz = z[size_t(t)];
            return true;
        }
    }
    return wants_to_walk(u, gx, gz);
}

// One frame of going up or down.  Returns whether it can move across this frame.
bool World::climb_or_land(int u, bool wants_to_go) {
    const auto& d = type_of(u);
    AirState& a = air[size_t(u)];
    const float climb = std::max(d.vertical_speed, 0.1f);
    switch (a.phase) {
        case AIR_ON_PAD:
            return false;
        case AIR_LANDED:
            if (!wants_to_go || !active(u)) return false;
            a.phase = AIR_TAKEOFF;
            a.takeoff_left = (d.flags & ZK_U_PLANE) ? PLANE_TAKEOFF_FRAMES : int(std::ceil(HOVER_TAKEOFF_SHARE * d.cruise_height / climb));
            [[fallthrough]];
        case AIR_TAKEOFF:
            a.height = std::min(d.cruise_height, a.height + climb);
            if (--a.takeoff_left > 0) return false;
            a.phase = AIR_FLYING;
            return true;
        case AIR_LANDING:
            if (wants_to_go) {
                a.phase = AIR_FLYING;
                return true;
            }
            a.height = std::max(0.0f, a.height - climb);
            if (a.height <= 0.0f) a.phase = AIR_LANDED;
            return false;
        default:
            if (!wants_to_go && (d.flags & ZK_U_LANDS)) {
                a.phase = AIR_LANDING;
                return false;
            }
            // A transport comes down to what it loads or unloads (transport.cpp).
            a.height += std::clamp((a.want >= 0.0f ? a.want : d.cruise_height) - a.height, -climb, climb);
            return true;
    }
}

void World::fly_plane(int u) {
    const size_t i = size_t(u);
    const auto& d = type_of(u);
    AirState& a = air[i];
    float gx = x[i], gz = z[i];
    bool go = air_goal(u, gx, gz);
    // A move, or a fight with nothing to fight, is over near its goal: a plane can't stop on a point.
    const int o = order[i];
    if (go && !out_of_ammo(u) && (o == ZK_ORDER_MOVE || (o == ZK_ORDER_FIGHT && ai[i].chasing < 0)) &&
        std::hypot(gx - x[i], gz - z[i]) < PLANE_ARRIVES) {
        order[i] = ZK_ORDER_NONE;
        go = false;
    }
    float h = heading[i], v = speed[i], px = x[i], pz = z[i];
    const float radius = d.turn_speed / std::max(d.turn_rate, 1e-3f);
    const float drag = d.accel / (DRAG_SPEED * std::max(d.speed, 0.1f));
    const float slow_mult = 1.0f - slow_now(u);
    for (int f = 0; f < ZK_TICK_FRAMES; ++f) {
        // What doesn't land flies round where it is.
        if (!climb_or_land(u, go || !(d.flags & ZK_U_LANDS))) {
            v = 0.0f;
            continue;
        }
        float diff = d.turn_rate;
        if (go) {
            const float dx = gx - px, dz = gz - pz, dist = std::hypot(dx, dz);
            diff = wrap(std::atan2(dx, dz) - h);
            const float dot = std::cos(diff);
            if ((dist < radius * NEAR_GOAL_SHARE && dot < NEAR_GOAL_DOT) || (dist < radius && dot < BEHIND_DOT)) diff = -diff;
        }
        h = wrap(h + std::clamp(diff, -d.turn_rate, d.turn_rate));
        const float top = (std::fabs(diff) > TURN_SLOWS ? d.turn_speed : d.speed) * slow_mult;
        v += (top - v) * (v < top ? drag : std::min(1.0f, d.brake / std::max(v, 0.1f)));
        px += std::sin(h) * v;
        pz += std::cos(h) * v;
    }
    px = std::clamp(px, 0.0f, map->width() - 1.0f);
    pz = std::clamp(pz, 0.0f, map->depth() - 1.0f);
    const float frames = float(ZK_TICK_FRAMES);
    vx[i] = (px - x[i]) / frames;
    vz[i] = (pz - z[i]) / frames;
    x[i] = px;
    z[i] = pz;
    heading[i] = h;
    speed[i] = a.phase == AIR_FLYING ? v : 0.0f;
}

// The nearest pad of its allyteam with room, or the nearest if none has (unit_bomber_command.lua).
int World::find_pad(int u) const {
    const int s = side[size_t(u)];
    int best = -1, best_full = -1;
    float bd = 0.0f, bd_full = 0.0f;
    for (int p : pads) {
        if (!alive[size_t(p)] || side[size_t(p)] != s || build[size_t(p)] < 1.0f) continue;
        int using_it = 0;
        for (int o : aircraft)
            if (o != u && alive[size_t(o)] && air[size_t(o)].pad == p) ++using_it;
        const float dd = std::hypot(x[size_t(p)] - x[size_t(u)], z[size_t(p)] - z[size_t(u)]);
        if (using_it < type_of(p).pad_count) {
            if (best < 0 || dd < bd) best = p, bd = dd;
        } else if (best_full < 0 || dd < bd_full) {
            best_full = p, bd_full = dd;
        }
    }
    return best >= 0 ? best : best_full;
}

// A salvo began.  The last one sends it back to rearm, and an attack order is over (scripts/bombers.lua Reload).
void World::used_ammo(int u) {
    AirState& a = air[size_t(u)];
    if (type_of(u).ammo <= 0 || --a.ammo > 0) return;
    a.pad = -1;
    if (order[size_t(u)] == ZK_ORDER_ATTACK) order[size_t(u)] = ZK_ORDER_NONE;
    ai[size_t(u)].chasing = -1;
}

// Rearming, each tick: finding a pad, settling on it, and the time and energy that take.
void World::update_air() {
    for (int u : aircraft) {
        const size_t i = size_t(u);
        if (!alive[i]) continue;
        const auto& d = type_of(u);
        AirState& a = air[i];
        if (d.ammo <= 0 || (a.ammo > 0 && a.phase != AIR_ON_PAD)) continue;
        if (a.phase == AIR_ON_PAD) {
            const int p = a.pad;
            if (p < 0 || !alive[size_t(p)]) {
                a.pad = -1;
                a.phase = AIR_LANDED;
                continue;
            }
            if (a.ammo <= 0) {
                // unit_refuel_pad_handler.lua: so much energy a frame, without which it waits.  The pad's team pays, an ally's or not.
                const float cost = d.reammo_energy * float(ZK_TICK_FRAMES);
                if (!economy_on || pay_energy(team[size_t(p)], cost)) a.rearmed += float(ZK_TICK_FRAMES);
                if (a.rearmed >= d.reammo_frames) {
                    a.ammo = d.ammo;
                    a.rearmed = 0.0f;
                }
                continue;
            }
            // Rearmed, it repairs while it's down, and then it's done with the pad.
            if (hp[i] < d.hp) {
                const float per_frame = type_of(p).pad_bp * d.hp / (REPAIR_COST_FACTOR * FRAMES_PER_SECOND * std::max(d.cost, 1.0f));
                hp[i] = std::min(d.hp, hp[i] + per_frame * float(ZK_TICK_FRAMES));
                continue;
            }
            a.pad = -1;
            a.phase = AIR_LANDED;
            continue;
        }
        if (a.pad < 0 || !alive[size_t(a.pad)]) a.pad = find_pad(u);
        const int p = a.pad;
        if (p < 0 || std::hypot(x[i] - x[size_t(p)], z[i] - z[size_t(p)]) > PAD_SNAP) continue;
        int on = 0;
        for (int o : aircraft)
            if (o != u && alive[size_t(o)] && air[size_t(o)].pad == p && air[size_t(o)].phase == AIR_ON_PAD) ++on;
        if (on >= type_of(p).pad_count) continue;
        x[i] = x[size_t(p)];
        z[i] = z[size_t(p)];
        vx[i] = vz[i] = speed[i] = 0.0f;
        a.height = 0.0f;
        a.phase = AIR_ON_PAD;
        a.rearmed = 0.0f;
    }
}

// BombDropper.cpp GetPredictedImpactTime, from level flight.
float World::bomb_fall_frames(int u, const zk_weapon_rec& w, int t) const {
    const float drop = flight_height(u) + map->height_at(x[size_t(u)], z[size_t(u)]) - map->height_at(x[size_t(t)], z[size_t(t)]);
    return drop > 0.0f ? std::sqrt(2.0f * drop / std::max(w.gravity, 1e-3f)) : 0.0f;
}

// BombDropper.cpp TestRange: where it would land is within the salvo's spread of the target.
bool World::bomb_due(int u, const zk_weapon_rec& w, int t) const {
    if (!airborne(u)) return false;
    const float fall = bomb_fall_frames(u, w, t);
    const float px = x[size_t(u)] + vx[size_t(u)] * fall, pz = z[size_t(u)] + vz[size_t(u)] * fall;
    const float spread = std::max(1.0f, w.salvo - 1.0f) * w.salvo_delay * speed[size_t(u)] * 0.5f;
    // A bomb of one has no spread: it goes as near as a tick's flight gets it, since this looks once a tick.
    const float reach = std::max({spread, speed[size_t(u)] * float(ZK_TICK_FRAMES) * 0.5f, BOMB_NEAR});
    return std::hypot(px - x[size_t(t)], pz - z[size_t(t)]) < reach;
}

// bomberdisarm.lua: where its fake bomb would drop, the script fires a run of what targets nothing.
void World::start_run(int u) {
    const auto& d = type_of(u);
    for (int m = 0; m < d.n_mounts; ++m) {
        const auto& w = cat->weapons[size_t(d.mount_weapon[m])];
        if (!(w.flags & ZK_W_DROPPED)) continue;
        const size_t i = size_t(u) * ZK_MAX_MOUNTS + size_t(m);
        salvo_left[i] = int(w.salvo);
        salvo_timer[i] = 0.0f;
    }
}

// One shot of a run, straight down from where the plane is.
void World::drop_shot(int u, int slot, int frame_offset) {
    const int weapon = type_of(u).mount_weapon[slot];
    const auto& w = cat->weapons[size_t(weapon)];
    const float px = x[size_t(u)] + vx[size_t(u)] * float(frame_offset), pz = z[size_t(u)] + vz[size_t(u)] * float(frame_offset);
    ++ai[size_t(u)].shots_fired;
    show_for_action(u);
    const Shot s{weapon, u, side[size_t(u)], -1, px, pz, false, false, 1.0f, px, pz, false};
    if (w.kind == ZK_WEAPON_INSTANT) {
        apply_shot(s);
        return;
    }
    const int slots = int(ring.size());
    const int down = int(flight_height(u) / std::max(w.speed, 1.0f));
    const int ahead = std::clamp((frame_offset + down + ZK_TICK_FRAMES / 2) / ZK_TICK_FRAMES, 1, slots - 1);
    ring[size_t((tick_count + ahead) % slots)].push_back(s);
}

}  // namespace zk
