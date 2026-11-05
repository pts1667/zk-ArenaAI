// The C ABI, and the threaded replay of logged fights.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

#include "zksim/profile.hpp"
#include "zksim/world.hpp"

using namespace zk;

namespace {

// Logs have no heading, so everything starts facing its nearest enemy.
void face_nearest_enemy(World& w, int first, int last, const int* side, const float* x, const float* z) {
    for (int e = first; e < last; ++e) {
        float cx = 0.0f, cz = 0.0f, best = 1e30f;
        for (int o = first; o < last; ++o) {
            if (side[o] == side[e]) continue;
            const float d = (x[o] - x[e]) * (x[o] - x[e]) + (z[o] - z[e]) * (z[o] - z[e]);
            if (d < best) {
                best = d;
                cx = x[o];
                cz = z[o];
            }
        }
        if (best < 1e30f) w.heading[size_t(e - first)] = std::atan2(cx - x[e], cz - z[e]);
    }
}

}  // namespace

extern "C" {

int zk_abi_version(void) { return ZK_ABI_VERSION; }
int zk_sizeof_weapon(void) { return int(sizeof(zk_weapon_rec)); }
int zk_sizeof_type(void) { return int(sizeof(zk_type_rec)); }
int zk_sizeof_state(void) { return int(sizeof(zk_unit_state)); }

void* zk_catalog_new(const zk_type_rec* types, int n_types, const zk_weapon_rec* weapons, int n_weapons,
                     const float* priority) {
    auto* c = new Catalog;
    c->types.assign(types, types + n_types);
    c->weapons.assign(weapons, weapons + n_weapons);
    c->priority.assign(priority, priority + size_t(n_weapons) * size_t(n_types));
    return c;
}

void zk_catalog_free(void* catalog) { delete static_cast<Catalog*>(catalog); }

void* zk_map_new(int cols, int rows, float cell, const float* height, const float* speed, int classes,
                 const float* slope, const float* normal_x, const float* normal_z, const float* slope_mod,
                 const short* segments) {
    auto* m = new Map;
    m->cols = cols;
    m->rows = rows;
    m->classes = classes;
    m->cell = cell;
    const size_t n = size_t(cols) * size_t(rows);
    m->height.assign(height, height + n);
    m->speed.assign(speed, speed + n * size_t(classes));
    m->slope.assign(slope, slope + n);
    m->normal_x.assign(normal_x, normal_x + n);
    m->normal_z.assign(normal_z, normal_z + n);
    m->slope_mod.assign(slope_mod, slope_mod + classes);
    m->set_segments(segments);
    return m;
}

void zk_map_set_routes(void* map, int move_class, int nodes, const float* centres, int edges, const int* edge_to,
                       const float* gates, const int* next_edge) {
    Routes& r = static_cast<Map*>(map)->routes[size_t(move_class)];
    r.nodes = nodes;
    r.centre_x.resize(size_t(nodes));
    r.centre_z.resize(size_t(nodes));
    for (int i = 0; i < nodes; ++i) {
        r.centre_x[size_t(i)] = centres[2 * i];
        r.centre_z[size_t(i)] = centres[2 * i + 1];
    }
    r.edge_to.assign(edge_to, edge_to + edges);
    r.gate_x.resize(size_t(edges));
    r.gate_z.resize(size_t(edges));
    for (int i = 0; i < edges; ++i) {
        r.gate_x[size_t(i)] = gates[2 * i];
        r.gate_z[size_t(i)] = gates[2 * i + 1];
    }
    r.next_edge.assign(next_edge, next_edge + size_t(nodes) * size_t(nodes));
}

void zk_map_free(void* map) { delete static_cast<Map*>(map); }

void* zk_world_new(const void* catalog, const void* map, uint64_t seed, int behaviour) {
    return new World(static_cast<const Catalog*>(catalog), static_cast<const Map*>(map), seed, behaviour);
}

void zk_world_free(void* world) { delete static_cast<World*>(world); }

int zk_world_add(void* world, int type, int side, float x, float z, float heading, float hp_frac, float build_frac) {
    return static_cast<World*>(world)->add(type, side, x, z, heading, hp_frac, build_frac);
}

void zk_world_order(void* world, int unit, int kind, float x, float z, int target) {
    static_cast<World*>(world)->order_unit(unit, kind, x, z, target);
}

void zk_world_step(void* world, int ticks) { static_cast<World*>(world)->step(ticks); }

int zk_world_frame(const void* world) { return static_cast<const World*>(world)->frame; }

void zk_world_set_frame(void* world, int frame) { static_cast<World*>(world)->frame = frame; }

void zk_world_ready_weapons(void* world) { static_cast<World*>(world)->ready_weapons(); }

void zk_world_kill(void* world, int unit) { static_cast<World*>(world)->hp[size_t(unit)] = 0.0f; }

void zk_world_get(const void* world, int unit, zk_unit_state* out) {
    const auto* w = static_cast<const World*>(world);
    const size_t u = size_t(unit);
    out->x = w->x[u];
    out->z = w->z[u];
    out->vx = w->vx[u];
    out->vz = w->vz[u];
    out->heading = w->heading[u];
    out->speed = w->speed[u];
    out->hp = w->hp[u];
    out->damage_taken = w->damage_taken[u];
    out->para = w->para[u];
    out->shield = w->shield[u];
    out->alive = w->alive[u];
    out->order = w->order[u];
    out->target = w->order_target[u];
}

void zk_replay(const void* catalog, const void* const* maps, const int* snapshot_map, int n_snapshots,
               const int* offsets, const int* type, const int* side, const float* x, const float* z,
               const float* hp_frac, const float* build_frac, const uint8_t* tracked, int n_seeds, uint64_t seed,
               const int* horizons, int n_horizons, int behaviour, int threads, float* out) {
    const auto* cat = static_cast<const Catalog*>(catalog);
    std::atomic<int> next{0};
    auto worker = [&]() {
        for (;;) {
            const int i = next.fetch_add(1);
            if (i >= n_snapshots) return;
            const int first = offsets[i], last = offsets[i + 1];
            const auto* map = static_cast<const Map*>(maps[snapshot_map[i]]);
            for (int s = 0; s < n_seeds; ++s) {
                World w(cat, map, seed ^ (uint64_t(i) << 20) ^ uint64_t(s), behaviour);
                for (int e = first; e < last; ++e) w.add(type[e], side[e], x[e], z[e], 0.0f, hp_frac[e], build_frac[e]);
                face_nearest_enemy(w, first, last, side, x, z);
                int done = 0;
                for (int h = 0; h < n_horizons; ++h) {
                    w.step(horizons[h] - done);
                    done = horizons[h];
                    float lost[2] = {0.0f, 0.0f};
                    for (int e = first; e < last; ++e) {
                        if (!tracked[e]) continue;
                        const auto& d = cat->types[size_t(type[e])];
                        // As the bot's logger: cost x min(hp at the start, damage / max HP).
                        const float frac = std::min(hp_frac[e], w.damage_taken[size_t(e - first)] / d.hp);
                        lost[side[e]] += d.cost * std::max(0.0f, frac);
                    }
                    float* row = out + (size_t(i) * size_t(n_seeds) + size_t(s)) * size_t(n_horizons) * 2 + size_t(h) * 2;
                    row[0] = lost[0];
                    row[1] = lost[1];
                }
            }
        }
    };
    std::vector<std::thread> pool;
    for (int t = 0; t < std::max(1, threads); ++t) pool.emplace_back(worker);
    for (auto& t : pool) t.join();
}

}  // extern "C"

#ifdef ZKSIM_PROFILE
extern "C" {
int zk_prof_count() { return zk::prof::PHASE_COUNT; }
const char* zk_prof_name(int phase) { return zk::prof::NAMES[phase]; }
void zk_prof_read(long long* nanos, long long* calls, int reset) {
    for (int i = 0; i < zk::prof::PHASE_COUNT; ++i) {
        nanos[i] = zk::prof::nanos[i];
        calls[i] = zk::prof::calls[i];
        if (reset) zk::prof::nanos[i] = zk::prof::calls[i] = 0;
    }
}
}
#endif
