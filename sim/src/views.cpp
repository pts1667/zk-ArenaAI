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
constexpr float FRAMES_PER_SECOND = 30.0f;
constexpr float GRID_CELL = 128.0f;          // as the unit grid and wreck_cells

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

void World::update_view(int s) {
    ZK_PROF(FOG);
    SideView& v = views[size_t(s)];
    if (v.updated_frame == frame) return;     // done for targeting, or by a frame just asked for
    v.updated_frame = frame;
    const int n = count();
    radar_units.clear();
    for (int o = 0; o < n; ++o)
        if (side[size_t(o)] == s && alive[size_t(o)] && build[size_t(o)] >= 1.0f && type_of(o).radar_radius > 0.0f && !unpaid[size_t(o)]) radar_units.push_back(o);
    // Who's in sight, from each of the side's eyes outward: it visits only the grid cells eyes can reach, where going from each enemy
    // would look through every cell within the game's longest sight.  Whether something is seen doesn't depend on which eye looks first.
    std::vector<uint8_t>& sighted = scratch_seen;
    std::vector<int>& eyes = scratch_eyes;
    sighted.assign(size_t(n), 0);
    eyes.clear();
    for (int o = 0; o < n; ++o)
        if (side[size_t(o)] == s && alive[size_t(o)] && build[size_t(o)] >= 1.0f) eyes.push_back(o);
    ZK_PROF(FOG_SWEEP);
    for (int o : eyes) {
        const size_t eye = size_t(o);
        const auto& od = type_of(o);
        const bool from_air = (od.flags & ZK_U_AIR) != 0;
        const float ex = x[eye], ez = z[eye], reach = std::max(od.los_radius, od.air_los_radius);
        for_side(1 - s, ex, ez, reach + SEARCH_PAD, [&](int e) {
            const size_t i = size_t(e);
            if (sighted[i] || cloaked[i] || !alive[i]) return;
            const float dx = ex - x[i], dz = ez - z[i], d2 = dx * dx + dz * dz;
            if (d2 > reach * reach) return;     // beyond either radius: r <= reach, so r * r <= reach * reach
            const bool flying = (type_of(e).flags & ZK_U_AIR) != 0;
            const float r = flying ? od.air_los_radius : od.los_radius;
            if (d2 > r * r) return;
            if (flying || from_air || clear_sight(o, e)) sighted[i] = 1;
        });
    }
    bool contact = false;
    for (int e = 0; e < n; ++e) {
        const size_t i = size_t(e);
        if (side[i] == s) continue;
        if (!alive[i]) {
            // A remembered building is forgotten once its place is back in sight.
            if (v.remembered[i]) {
                ZK_PROF(FOG_GONE);
                for (int o : eyes) {
                    const float r = type_of(o).los_radius, dx = x[size_t(o)] - x[i], dz = z[size_t(o)] - z[i];
                    if (dx * dx + dz * dz <= r * r) {
                        v.remembered[i] = 0;
                        break;
                    }
                }
            }
            v.in_los[i] = v.in_radar[i] = v.known[i] = 0;
            continue;
        }
        const auto& d = type_of(e);
        // LosHandler.cpp: a cloaked enemy isn't in sight, and unit_stealth.lua keeps it off radar too.
        const bool los = sighted[i] != 0;
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
        r.y = (water_top ? surface(*map, x[i], z[i]) : map->height_at(x[i], z[i])) + flight_height(u) + jump_altitude(u);
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
    // Own units first, then what's seen, then radar, then memory, then what's hidden: each in handle order.  One pass sorts them
    // into those five, which don't overlap.
    auto& [own, seen, blips, memory, hidden] = emit_lists;
    own.clear();
    seen.clear();
    blips.clear();
    memory.clear();
    hidden.clear();
    for (int u = 0; u < n; ++u) {
        const size_t i = size_t(u);
        if (side[i] == s) {
            if (alive[i]) own.push_back(u);
        } else if (v.in_los[i]) {
            if (alive[i]) seen.push_back(u);
        } else if (alive[i] && v.in_radar[i]) {
            blips.push_back(u);
        } else if (v.remembered[i]) {
            memory.push_back(u);
        } else if (alive[i]) {
            hidden.push_back(u);
        }
    }
    for (int u : own) {
        if (w >= cap) break;
        const size_t i = size_t(u);
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
    for (int u : seen) {
        if (w >= cap) break;
        common(u, KNOW_LOS, UF_IN_LOS | UF_TYPE_KNOWN);
    }
    for (int u : blips) {
        if (w >= cap) break;
        const size_t i = size_t(u);
        UnitRecord& r = common(u, KNOW_RADAR, v.known[i] ? UF_TYPE_KNOWN : 0);
        if (!v.known[i]) r.def_id = -1;
        r.hp_frac = std::nanf("");
        r.build_frac = 0.0f;
        r.blip_top_speed = v.tracks[i].top;
    }
    for (int u : memory) {
        if (w >= cap) break;
        const size_t i = size_t(u);
        UnitRecord& r = common(u, KNOW_MEMORY, UF_TYPE_KNOWN);
        r.hp_frac = v.ghost_hp[i];
        r.build_frac = v.ghost_build[i];
        r.vx = r.vz = 0.0f;
        r.ord_age = float(frame - v.seen_frame[i]);
    }
    for (int u : hidden) {
        if (w >= cap) break;
        common(u, KNOW_HIDDEN, 0);
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

void World::index_wrecks() {
    if (!wreck_index_stale) return;
    wreck_index_stale = false;
    const size_t cells = grid_cells();
    wreck_start.assign(cells + 1, 0);
    auto cell_of = [&](int id) {
        const Wreck& w = wrecks[size_t(id)];
        const int gx = std::clamp(int(w.x / GRID_CELL), 0, grid_cols - 1), gz = std::clamp(int(w.z / GRID_CELL), 0, grid_rows - 1);
        return size_t(gz) * size_t(grid_cols) + size_t(gx);
    };
    for (int id : live_wrecks) ++wreck_start[cell_of(id)];
    for (size_t c = 1; c <= cells; ++c) wreck_start[c] += wreck_start[c - 1];
    wreck_ids.resize(live_wrecks.size());
    wreck_px.resize(live_wrecks.size());
    wreck_pz.resize(live_wrecks.size());
    wreck_slot.resize(wrecks.size());
    for (int id : live_wrecks) {
        const size_t k = size_t(--wreck_start[cell_of(id)]);
        wreck_ids[k] = id;
        wreck_slot[size_t(id)] = int(k);
        wreck_px[k] = wrecks[size_t(id)].x;
        wreck_pz[k] = wrecks[size_t(id)].z;
    }
}

// In ground sight of the side, as a feature is (AICallback.cpp).  One ray per wreck and unit, as for units.  From each of the side's
// eyes outward, a row of the wreck grid at a time, as for units in update_view.
int World::emit_features(int s, FeatureRecord* out, int cap) {
    if (live_wrecks.empty() || cap <= 0) return 0;
    {
        ZK_PROF(FEATURES_INDEX);
        index_wrecks();
    }
    ZK_PROF(FEATURES_SWEEP);
    std::vector<uint8_t>& sighted = scratch_seen;      // by place in the index
    sighted.assign(wreck_ids.size(), 0);
    const int n = count();
    for (int o = 0; o < n; ++o) {
        const size_t eye = size_t(o);
        if (side[eye] != s || !alive[eye] || build[eye] < 1.0f) continue;
        const auto& od = type_of(o);
        const float ex = x[eye], ez = z[eye], reach2 = od.los_radius * od.los_radius;
        const bool from_air = (od.flags & ZK_U_AIR) != 0;
        const CellBox b = cell_box(ex, ez, od.los_radius + SEARCH_PAD);
        for (int gz = b.z0; gz <= b.z1; ++gz) {
            const size_t row = size_t(gz) * size_t(grid_cols);
            const size_t end = size_t(wreck_start[row + size_t(b.x1) + 1]);
            for (size_t k = size_t(wreck_start[row + size_t(b.x0)]); k < end; ++k) {
                if (sighted[k]) continue;
                const float dx = ex - wreck_px[k], dz = ez - wreck_pz[k];
                if (dx * dx + dz * dz > reach2) continue;
                if (from_air || sees_point(o, wreck_px[k], wreck_pz[k])) sighted[k] = 1;
            }
        }
    }
    int w = 0;
    for (int id : live_wrecks) {
        if (w >= cap) break;
        if (!sighted[size_t(wreck_slot[size_t(id)])]) continue;
        const Wreck& wreck = wrecks[size_t(id)];
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
    // The reward's tracked value (03-training.md, "Reward"): cost x build progress, real units only.
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
    out[2] = purses[s].reclaim_income;
}

}  // namespace zk
