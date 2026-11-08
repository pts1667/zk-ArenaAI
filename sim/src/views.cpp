// What each side can see, as the raw frame the live bot also produces.  It's an approximation of the engine's LOS: one ray per
// pair on the height map, not its mip-mapped circles, and no jammers yet.  Fog only runs for worlds that emit frames.
#include <algorithm>
#include <cmath>

#include "zksim/profile.hpp"
#include "zksim/world.hpp"

namespace zk {

namespace {
constexpr float SIGHT_STEP = 32.0f;          // elmos between samples of the height map along a line of sight
constexpr float TARGET_HEIGHT = 12.0f;       // how far above the ground a unit can be seen, roughly its middle
constexpr int BLIP_WINDOW_FRAMES = 60;       // blips.py's SPEED_WINDOW
constexpr int BLIP_MIN_SPAN_FRAMES = 36;     // ... and 0.6 of it before a track counts
constexpr float AIR_ALTITUDE = 150.0f;
constexpr float FRAMES_PER_SECOND = 30.0f;

inline float surface(const Map& m, float x, float z) { return std::max(0.0f, m.height_at(x, z)); }
}  // namespace

void World::ensure_views() {
    const size_t n = size_t(count());
    for (auto& v : views) {
        if (v.in_los.size() == n) continue;
        v.in_los.resize(n, 0);
        v.in_radar.resize(n, 0);
        v.known.resize(n, 0);
        v.remembered.resize(n, 0);
        v.seen_frame.resize(n, 0);
        v.ghost_hp.resize(n, 0.0f);
        v.ghost_build.resize(n, 0.0f);
        v.tracks.resize(n);
    }
    last_damage.resize(n, 0.0f);
}

bool World::clear_sight(int from, int to) const { return sees_point(from, x[size_t(to)], z[size_t(to)]); }

bool World::sees_point(int from, float tx, float tz) const {
    const float fx = x[size_t(from)], fz = z[size_t(from)];
    const float eye = surface(*map, fx, fz) + type_of(from).los_height;
    const float seen = surface(*map, tx, tz) + TARGET_HEIGHT;
    const float dist = std::hypot(tx - fx, tz - fz);
    const int steps = int(dist / SIGHT_STEP);
    for (int i = 1; i < steps; ++i) {
        const float t = float(i) / float(steps);
        if (surface(*map, fx + (tx - fx) * t, fz + (tz - fz) * t) > eye + (seen - eye) * t) return false;
    }
    return true;
}

bool World::point_seen_by(int s, float px, float pz) const {
    bool seen = false;
    for_near(px, pz, max_los, [&](int o) {
        if (seen || side[size_t(o)] != s || !alive[size_t(o)] || build[size_t(o)] < 1.0f) return;
        const float r = type_of(o).los_radius, dx = x[size_t(o)] - px, dz = z[size_t(o)] - pz;
        if (dx * dx + dz * dz <= r * r) seen = true;
    });
    return seen;
}

void World::update_view(int s) {
    ZK_PROF(FOG);
    SideView& v = views[size_t(s)];
    if (v.updated_frame == frame) return;     // done for targeting, or by a frame just asked for
    v.updated_frame = frame;
    const int n = count();
    radar_units.clear();
    for (int o = 0; o < n; ++o)
        if (side[size_t(o)] == s && alive[size_t(o)] && build[size_t(o)] >= 1.0f && type_of(o).radar_radius > 0.0f && !unpaid[size_t(o)]) radar_units.push_back(o);
    bool contact = false;
    for (int e = 0; e < n; ++e) {
        const size_t i = size_t(e);
        if (side[i] == s) continue;
        if (!alive[i]) {
            // A remembered building is forgotten once its place is back in sight.
            if (v.remembered[i] && point_seen_by(s, x[i], z[i])) v.remembered[i] = 0;
            v.in_los[i] = v.in_radar[i] = v.known[i] = 0;
            continue;
        }
        const auto& d = type_of(e);
        const bool flying = (d.flags & ZK_U_AIR) != 0;
        bool los = false;
        // LosHandler.cpp: a cloaked enemy isn't in sight, and unit_stealth.lua keeps it off radar too.
        if (!cloaked[i]) for_near(x[i], z[i], max_los, [&](int o) {
            if (los || side[size_t(o)] != s || !alive[size_t(o)] || build[size_t(o)] < 1.0f) return;
            const auto& od = type_of(o);
            const float r = flying ? od.air_los_radius : od.los_radius;
            const float dx = x[size_t(o)] - x[i], dz = z[size_t(o)] - z[i];
            if (dx * dx + dz * dz > r * r) return;
            if (flying || (od.flags & ZK_U_AIR) || clear_sight(o, e)) los = true;
        });
        bool radar = false;
        if (!cloaked[i] && !los)
            for (int o : radar_units) {
                const float r = type_of(o).radar_radius, dx = x[size_t(o)] - x[i], dz = z[size_t(o)] - z[i];
                if (dx * dx + dz * dz <= r * r) {
                    radar = true;
                    break;
                }
            }
        v.in_los[i] = los;
        v.in_radar[i] = radar;
        // AICallback.cpp: a blip's type is known if it's in LOS, or was seen and has stayed in radar contact since.
        v.known[i] = los || (radar && v.known[i]);
        contact = contact || los || radar;
        if (los && !(d.flags & ZK_U_MOBILE)) {
            v.remembered[i] = 1;
            v.seen_frame[i] = frame;
            v.ghost_hp[i] = hp[i] / d.hp;
            v.ghost_build[i] = build[i];
        }
        if (radar) {
            BlipTrack& t = v.tracks[i];
            const int slots = int(sizeof(t.speeds) / sizeof(t.speeds[0]));
            t.speeds[t.head] = speed[i] * FRAMES_PER_SECOND;
            t.frames[t.head] = frame;
            t.head = (t.head + 1) % slots;
            t.filled = std::min(t.filled + 1, slots);
            float slowest = 1e30f;
            int oldest = frame;
            for (int k = 0; k < t.filled; ++k) {
                if (frame - t.frames[k] > BLIP_WINDOW_FRAMES) continue;
                slowest = std::min(slowest, t.speeds[k]);
                oldest = std::min(oldest, t.frames[k]);
            }
            if (frame - oldest >= BLIP_MIN_SPAN_FRAMES) t.top = std::max(t.top, slowest);
        }
    }
    if (contact) v.last_contact = frame;
}

int World::emit_view(int s, UnitRecord* out, int cap, float* glob, int commander, const float* base) {
    SideView& v = views[size_t(s)];
    const int n = count();
    int w = 0;
    int unfinished = 0, idle_builders = 0;
    auto common = [&](int u, int knowledge, int flags) -> UnitRecord& {
        const size_t i = size_t(u);
        const auto& d = type_of(u);
        UnitRecord& r = out[w++];
        r.uid = u;
        r.ord_target = -1;
        r.def_id = d.def_id;
        r.x = x[i];
        r.z = z[i];
        const bool water_top = d.move_class == 4 || d.move_class == 5 || (d.flags & ZK_U_FLOATS);   // hovers, ships and what floats ride on the water
        r.y = (water_top ? surface(*map, x[i], z[i]) : map->height_at(x[i], z[i])) + ((d.flags & ZK_U_AIR) ? AIR_ALTITUDE : 0.0f) + jump_altitude(u);
        r.vx = vx[i];
        r.vz = vz[i];
        r.hp_frac = hp[i] / d.hp;
        r.build_frac = build[i];
        r.experience = 0.0f;
        r.blip_top_speed = 0.0f;
        r.ord_age = 0.0f;
        r.ord_x = 0.0f;
        r.ord_z = 0.0f;
        r.knowledge = uint8_t(knowledge);
        // One player to an alliance, for now: a side is both.
        r.player = int16_t(side[i]);
        r.ally_team = int16_t(side[i]);
        r.pad = 0;
        r.ord_kind = -1;
        if (build[i] < 1.0f) flags |= UF_BEING_BUILT;
        if (stunned(u)) flags |= UF_PARALYZED;
        if (cloaked[i] && side[i] == s) flags |= UF_CLOAKED;
        r.flags = uint16_t(flags);
        return r;
    };
    // Own units first, then what's seen, then radar, then memory, then what's hidden: each in handle order.
    for (int u = 0; u < n && w < cap; ++u) {
        const size_t i = size_t(u);
        if (side[i] != s || !alive[i]) continue;
        const auto& d = type_of(u);
        int flags = UF_IN_LOS | UF_TYPE_KNOWN;
        const bool aims = order[i] == ZK_ORDER_ATTACK || order[i] == ZK_ORDER_MANUAL_FIRE;
        // The engine's queue holds what a unit gives itself as well.
        const bool order_done = (order[i] == ZK_ORDER_NONE && ai[i].chasing < 0 && !ai[i].dodging) || (aims && order_target[i] >= 0 && !alive[size_t(order_target[i])]);
        if (order_done && (d.flags & ZK_U_MOBILE) && build[i] >= 1.0f) flags |= UF_IDLE;
        if (damage_taken[i] > last_damage[i]) flags |= UF_DAMAGED;
        // Whatever it does with it: throwing's the same to whoever's deciding.
        if ((d.jump_range > 0.0f && jump[i].charge >= 1.0f && !jump[i].busy) || throw_ready(u)) flags |= UF_JUMP_READY;
        if (pulling[i]) flags |= UF_PULLING;
        last_damage[i] = damage_taken[i];
        if (!(d.flags & ZK_U_MOBILE) && build[i] < 1.0f) ++unfinished;
        if (order_done && (d.flags & ZK_U_MOBILE) && d.build_speed > 0.0f && build[i] >= 1.0f) ++idle_builders;
        UnitRecord& r = common(u, KNOW_LOS, flags);
        if (mem_kind[i] >= 0) {
            r.ord_kind = int8_t(mem_kind[i]);
            r.ord_age = float(frame - mem_frame[i]);
            r.ord_x = mem_x[i];
            r.ord_z = mem_z[i];
            // What a factory or silo makes is a type: the frame gives it as a def id, as it gives every other type.
            r.ord_target = mem_kind[i] == ZK_ORDER_PRODUCE && mem_target[i] >= 0 ? cat->types[size_t(mem_target[i])].def_id : mem_target[i];
        }
    }
    for (int u = 0; u < n && w < cap; ++u)
        if (side[size_t(u)] != s && alive[size_t(u)] && v.in_los[size_t(u)]) common(u, KNOW_LOS, UF_IN_LOS | UF_TYPE_KNOWN);
    for (int u = 0; u < n && w < cap; ++u) {
        const size_t i = size_t(u);
        if (side[i] == s || !alive[i] || v.in_los[i] || !v.in_radar[i]) continue;
        UnitRecord& r = common(u, KNOW_RADAR, v.known[i] ? UF_TYPE_KNOWN : 0);
        if (!v.known[i]) r.def_id = -1;
        r.hp_frac = std::nanf("");
        r.build_frac = 0.0f;
        r.blip_top_speed = v.tracks[i].top;
    }
    for (int u = 0; u < n && w < cap; ++u) {
        const size_t i = size_t(u);
        if (side[i] == s || !v.remembered[i] || v.in_los[i] || (alive[i] && v.in_radar[i])) continue;
        UnitRecord& r = common(u, KNOW_MEMORY, UF_TYPE_KNOWN);
        r.hp_frac = v.ghost_hp[i];
        r.build_frac = v.ghost_build[i];
        r.vx = r.vz = 0.0f;
        r.ord_age = float(frame - v.seen_frame[i]);
    }
    for (int u = 0; u < n && w < cap; ++u) {
        const size_t i = size_t(u);
        if (side[i] != s && alive[i] && !v.in_los[i] && !v.in_radar[i] && !v.remembered[i]) common(u, KNOW_HIDDEN, 0);
    }
    if (glob) {
        std::fill(glob, glob + G_COUNT, 0.0f);
        glob[G_FRAME] = float(frame);
        glob[G_PLAYER] = float(s);
        glob[G_ALLY_TEAM] = float(s);
        glob[G_MAP_W] = map->width();
        glob[G_MAP_H] = map->depth();
        glob[G_COMMANDER_UID] = -1.0f;
        if (commander >= 0 && alive[size_t(commander)]) {
            glob[G_COMMANDER_UID] = float(commander);
            glob[G_COMMANDER_HP] = hp[size_t(commander)] / type_of(commander).hp;
        }
        glob[G_BASE_X] = base[0];
        glob[G_BASE_Z] = base[1];
        glob[G_FRAMES_SINCE_CONTACT] = float(frame - v.last_contact);
        glob[G_UNFINISHED_STRUCTURES] = float(unfinished);
        if (economy_on) {
            const Purse& p = purses[size_t(s)];
            glob[G_METAL] = p.metal;
            glob[G_METAL_INCOME] = p.metal_income;
            glob[G_METAL_STORAGE] = p.storage;
            glob[G_ENERGY] = p.energy;
            glob[G_ENERGY_INCOME] = p.energy_income;
            glob[G_ENERGY_STORAGE] = p.storage;
            glob[G_IDLE_BUILDERS] = float(idle_builders);
        }
    }
    return w;
}

// In ground sight of the side, as a feature is (AICallback.cpp).  One ray per wreck and unit, as for units.
int World::emit_features(int s, FeatureRecord* out, int cap) const {
    int w = 0;
    for (int id : live_wrecks) {
        if (w >= cap) break;
        const Wreck& wreck = wrecks[size_t(id)];
        bool seen = false;
        for_near(wreck.x, wreck.z, max_los, [&](int o) {
            if (seen || side[size_t(o)] != s || !alive[size_t(o)] || build[size_t(o)] < 1.0f) return;
            const auto& od = type_of(o);
            const float dx = x[size_t(o)] - wreck.x, dz = z[size_t(o)] - wreck.z;
            if (dx * dx + dz * dz > od.los_radius * od.los_radius) return;
            if ((od.flags & ZK_U_AIR) || sees_point(o, wreck.x, wreck.z)) seen = true;
        });
        if (!seen) continue;
        FeatureRecord& r = out[w++];
        r.feat_id = id;
        r.x = wreck.x;
        r.z = wreck.z;
        r.metal = wreck.metal * wreck.left;
        r.def_id = wreck.feature_def;
    }
    return w;
}

void World::side_stats(int s, float* out) const {
    // The reward's tracked value (03-training.md 1.3): cost x build progress, real units only.
    float value = 0.0f;
    int units = 0;
    for (int u = 0; u < count(); ++u) {
        const size_t i = size_t(u);
        if (side[i] != s || !alive[i]) continue;
        value += type_of(u).cost * build[i];
        ++units;
    }
    out[0] = value;
    out[1] = float(units);
}

}  // namespace zk
