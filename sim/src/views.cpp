// What each allyteam can see, and each team's raw frame of it, as the live bot also produces.  It's an approximation of the engine's
// LOS: one ray per pair on the height map, not its mip-mapped circles, and no jammers yet.  Fog only runs for worlds that emit frames.
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
    for (int s = 0; s < n_allies; ++s) {
        SideView& v = views[s];
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
    // Who's in sight, from each of the allyteam's eyes outward: it visits only the grid cells eyes can reach, where going from each
    // enemy would look through every cell within the game's longest sight.  Whether something is seen doesn't depend on which eye
    // looks first.
    std::vector<uint8_t>& sighted = scratch_seen;
    std::vector<int>& eyes = scratch_eyes;
    sighted.assign(size_t(count()), 0);
    radar_units.clear();
    eyes.clear();
    float farthest = 0.0f;          // ground sight of any eye
    for (const int o : live) {
        if (side[size_t(o)] != s || !alive[size_t(o)] || build[size_t(o)] < 1.0f) continue;
        eyes.push_back(o);
        farthest = std::max(farthest, type_of(o).los_radius);
        if (type_of(o).radar_radius > 0.0f && !unpaid[size_t(o)]) radar_units.push_back(o);
    }
    ZK_PROF(FOG_SWEEP);
    for (int o : eyes) {
        const size_t eye = size_t(o);
        const auto& od = type_of(o);
        const bool from_air = (od.flags & ZK_U_AIR) != 0;
        const float ex = x[eye], ez = z[eye], reach = std::max(od.los_radius, od.air_los_radius);
        for_enemies(s, ex, ez, reach + SEARCH_PAD, [&](int e) {
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
    for (const int e : live) {
        const size_t i = size_t(e);
        if (side[i] == s || !alive[i]) continue;
        const auto& d = type_of(e);
        // LosHandler.cpp: a cloaked enemy isn't in sight, and unit_stealth.lua keeps it off radar too.  One the reveal rule shows is
        // in everyone's sight (check_reveal).
        const bool los = sighted[i] != 0 || ((revealed >> side[i]) & 1u) != 0;
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
    // The dead: out of sight and radar from their first look on, and a building among them remembered until its place is back in
    // sight.  Only the newly dead and the remembered are looked at.
    for (; v.fallen_seen < fallen.size(); ++v.fallen_seen) {
        const size_t i = size_t(fallen[v.fallen_seen]);
        if (side[i] == s) continue;
        v.in_los[i] = v.in_radar[i] = v.known[i] = 0;
        if (v.remembered[i]) v.memories.push_back(int(i));
    }
    if (!v.memories.empty()) {
        // Any eye near enough will do, so only those in the grid round its place are asked.  The grid was made since anyone
        // came or went, as the sweep above needs too, so its finished units of this allyteam are the eyes.
        ZK_PROF(FOG_GONE);
        size_t kept = 0;
        for (const int m : v.memories) {
            const size_t i = size_t(m);
            bool seen = false;
            for_side(s, x[i], z[i], farthest + SEARCH_PAD, [&](int o) {
                if (seen || !alive[size_t(o)] || build[size_t(o)] < 1.0f) return;
                const float r = type_of(o).los_radius, dx = x[size_t(o)] - x[i], dz = z[size_t(o)] - z[i];
                seen = dx * dx + dz * dz <= r * r;
            });
            if (seen) v.remembered[i] = 0;
            else v.memories[kept++] = m;
        }
        v.memories.resize(kept);
    }
}

int World::emit_view(int viewer, UnitRecord* out, int cap, float* glob, int commander, const float* base) {
    const int s = ally_of[viewer];
    SideView& v = views[size_t(s)];
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
        r.player = int16_t(team[i]);
        r.ally_team = int16_t(side[i]);
        r.pad = 0;
        r.ord_kind = -1;
        if (build[i] < 1.0f) flags |= UF_BEING_BUILT;
        if (stunned(u)) flags |= UF_PARALYZED;
        if (cloaked[i] && side[i] == s) flags |= UF_CLOAKED;     // an ally knows its friends are hidden
        r.flags = uint16_t(flags);
        return r;
    };
    // Own units first, then allies', then what's seen, then radar, then memory, then what's hidden: each in handle order.  One pass
    // over the living sorts them into those six, which don't overlap; the dead are only ever remembered.
    auto& [own, allies, seen, blips, memory, hidden] = emit_lists;
    own.clear();
    allies.clear();
    seen.clear();
    blips.clear();
    memory.clear();
    hidden.clear();
    for (const int u : live) {
        const size_t i = size_t(u);
        if (!alive[i]) continue;
        if (side[i] == s) (team[i] == viewer ? own : allies).push_back(u);
        else if (v.in_los[i]) seen.push_back(u);
        else if (v.in_radar[i]) blips.push_back(u);
        else if (v.remembered[i]) memory.push_back(u);
        else hidden.push_back(u);
    }
    // The remembered dead, and any that died since the view was last brought up to date, which it still has as they were.
    const size_t living = memory.size();
    for (const int u : v.memories)
        if (!v.in_los[size_t(u)] && v.remembered[size_t(u)]) memory.push_back(u);
    for (size_t k = v.fallen_seen; k < fallen.size(); ++k) {
        const size_t i = size_t(fallen[k]);
        if (side[i] != s && !v.in_los[i] && v.remembered[i]) memory.push_back(int(i));
    }
    if (memory.size() > living) std::sort(memory.begin(), memory.end());
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
    // An ally's units are always in sight, and show what a seen unit does: not its orders, nor whether it's idle.
    for (int u : allies) {
        if (w >= cap) break;
        common(u, KNOW_LOS, UF_IN_LOS | UF_TYPE_KNOWN);
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
        glob[G_PLAYER] = float(viewer);
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
            const Purse& p = purses[size_t(viewer)];
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
    for (int id : live_wrecks) {
        const size_t k = size_t(--wreck_start[cell_of(id)]);
        wreck_ids[k] = id;
        wreck_px[k] = wrecks[size_t(id)].x;
        wreck_pz[k] = wrecks[size_t(id)].z;
    }
}

// In ground sight of the allyteam, as a feature is (AICallback.cpp).  One ray per wreck and unit, as for units.  From each of the
// allyteam's eyes outward, a row of the wreck grid at a time, as for units in update_view.
int World::emit_features(int s, FeatureRecord* out, int cap) {
    if (live_wrecks.empty() || cap <= 0) return 0;
    {
        ZK_PROF(FEATURES_INDEX);
        index_wrecks();
    }
    ZK_PROF(FEATURES_SWEEP);
    std::vector<uint8_t>& sighted = scratch_seen;      // by place in the index
    sighted.assign(wreck_ids.size(), 0);
    std::vector<int>& found = scratch_found;
    found.clear();
    for (const int o : live) {
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
                if (from_air || sees_point(o, wreck_px[k], wreck_pz[k])) {
                    sighted[k] = 1;
                    found.push_back(wreck_ids[k]);
                }
            }
        }
    }
    // By id, as the live wrecks are.  When there's more in sight than a frame holds, the ones with the most metal left: what a frame
    // shows features for is reclaiming them.  Ties go to the older.
    std::sort(found.begin(), found.end());
    if (int(found.size()) > cap) {
        const auto richer = [this](int a, int b) {
            const float ma = wrecks[size_t(a)].metal * wrecks[size_t(a)].left, mb = wrecks[size_t(b)].metal * wrecks[size_t(b)].left;
            return ma != mb ? ma > mb : a < b;
        };
        std::nth_element(found.begin(), found.begin() + cap, found.end(), richer);
        found.resize(size_t(cap));
        std::sort(found.begin(), found.end());
    }
    int w = 0;
    for (const int id : found) {
        if (w >= cap) break;
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

void World::team_stats(int t, float* out) const {
    // The reward's tracked value (03-training.md, "Reward"): cost x build progress, real units only.  A chicken costs nothing; its
    // build time stands in, which for every unit that does cost something is the same as its metal cost.
    float value = 0.0f;
    int units = 0, counting = 0;
    for (const int u : live) {
        const size_t i = size_t(u);
        if (team[i] != t || !alive[i]) continue;
        const auto& d = type_of(u);
        value += (d.cost > 0.0f ? d.cost : d.build_time) * build[i];
        ++units;
        // What keeps a team in the game (game_over.lua): finished units, but for those that don't count.
        counting += build[i] >= 1.0f && !(d.flags & ZK_U_DONT_COUNT);
    }
    out[0] = value;
    out[1] = float(units);
    out[2] = purses[t].reclaim_income;
    out[3] = float(counting);
}

}  // namespace zk
