// The C ABI, and the threaded replay of logged fights.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <memory>
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

struct Game {
    std::unique_ptr<World> world;
    int commander[2] = {-1, -1};
    float base[4] = {0.0f, 0.0f, 0.0f, 0.0f};
};

struct Batch {
    const Catalog* cat = nullptr;
    int threads = 1;
    std::vector<Game> games;
};

}  // namespace

extern "C" {

int zk_abi_version(void) { return ZK_ABI_VERSION; }
int zk_sizeof_weapon(void) { return int(sizeof(zk_weapon_rec)); }
int zk_sizeof_type(void) { return int(sizeof(zk_type_rec)); }
int zk_sizeof_state(void) { return int(sizeof(zk_unit_state)); }

void* zk_catalog_new(const zk_type_rec* types, int n_types, const zk_weapon_rec* weapons, int n_weapons,
                     const float* priority, const zk_tactics_rec* tactics, const uint8_t* relations) {
    auto* c = new Catalog;
    c->types.assign(types, types + n_types);
    c->weapons.assign(weapons, weapons + n_weapons);
    c->priority.assign(priority, priority + size_t(n_weapons) * size_t(n_types));
    c->tactics.assign(tactics, tactics + n_types);
    c->relations.assign(relations, relations + size_t(n_types) * size_t(n_types));
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

void zk_world_set_hp(void* world, int unit, float hp) { static_cast<World*>(world)->hp[size_t(unit)] = hp; }

void zk_world_step(void* world, int ticks) { static_cast<World*>(world)->step(ticks); }

int zk_world_frame(const void* world) { return static_cast<const World*>(world)->frame; }

void zk_world_set_frame(void* world, int frame) { static_cast<World*>(world)->frame = frame; }

void zk_world_ready_weapons(void* world) { static_cast<World*>(world)->ready_weapons(); }

void zk_world_kill(void* world, int unit) { static_cast<World*>(world)->hp[size_t(unit)] = 0.0f; }

void zk_world_set_shield(void* world, int unit, float charge) { static_cast<World*>(world)->shield[size_t(unit)] = charge; }

void zk_world_economy(void* world, int n_spots, const float* spots, float metal, float energy, float innate_metal, float innate_energy, float storage) {
    static_cast<World*>(world)->set_economy(n_spots, spots, metal, energy, innate_metal, innate_energy, storage);
}

int zk_world_add_feature(void* world, float x, float z, float metal, float hp, float radius, int feature_def) {
    return static_cast<World*>(world)->add_wreck(x, z, metal, hp, radius, feature_def, -1);
}

int zk_world_features(void* world, int side, void* records, int cap) {
    auto* w = static_cast<World*>(world);
    w->rebuild_index();
    return w->emit_features(side, static_cast<FeatureRecord*>(records), cap);
}

void zk_world_resources(const void* world, int side, float* out) {
    const Purse& p = static_cast<const World*>(world)->purses[size_t(side)];
    out[0] = p.metal;
    out[1] = p.energy;
    out[2] = p.metal_income;
    out[3] = p.energy_income;
    out[4] = p.overdrive_metal;
    out[5] = p.overdrive_energy;
}

int zk_world_count(const void* world) { return static_cast<const World*>(world)->count(); }

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
    out->disarm = w->disarm[u];
    out->build = w->build[u];
    out->stock = w->stock[u];
    out->cloaked = w->cloaked[u];
    out->heat = w->ai[u].heat;
    out->jump_charge = w->jump[u].charge;
    out->jumping = w->jump[u].flying ? 2 : (w->jump[u].busy ? 1 : 0);
    {
        const auto& flight = w->jump[u];
        const float bend = 2.0f * flight.at - 1.0f;
        out->altitude = flight.flying ? (1.0f - bend * bend) * flight.height : w->skid[u].y;
    }
    out->skidding = w->skid[u].on;
    out->low_power = w->power_off(unit);
    out->pulling = w->pulling[u];
    {
        const Teleporter* t = w->teleporter_of(unit);
        out->deployed = t && t->deployed;
        out->beacon = t ? t->beacon : -1;
    }
    out->chasing = w->ai[u].chasing;
    out->dodging = w->ai[u].dodging;
    out->shots_fired = w->ai[u].shots_fired;
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

int zk_sizeof_unit_record(void) { return int(sizeof(UnitRecord)); }
int zk_sizeof_site_record(void) { return int(sizeof(SiteRecord)); }
int zk_sizeof_feature_record(void) { return int(sizeof(FeatureRecord)); }
int zk_global_count(void) { return G_COUNT; }

int zk_world_view(void* world, int side, void* records, int cap, float* glob, int commander, const float* base) {
    auto* w = static_cast<World*>(world);
    w->ensure_views();
    w->rebuild_index();
    w->update_view(side);
    return w->emit_view(side, static_cast<UnitRecord*>(records), cap, glob, commander, base);
}

void* zk_batch_new(const void* catalog, int games, int threads) {
    auto* b = new Batch;
    b->cat = static_cast<const Catalog*>(catalog);
    b->games.resize(size_t(games));
    b->threads = std::max(1, threads);
    return b;
}

void zk_batch_free(void* batch) { delete static_cast<Batch*>(batch); }

void zk_batch_reset(void* batch, int g, const void* map, uint64_t seed, int behaviour, int n, const int* type, const int* side,
                    const float* x, const float* z, const float* heading, const float* hp_frac, const float* build_frac,
                    const int* commander, const float* base) {
    auto* b = static_cast<Batch*>(batch);
    Game& game = b->games[size_t(g)];
    game.world = std::make_unique<World>(b->cat, static_cast<const Map*>(map), seed, behaviour);
    for (int i = 0; i < n; ++i) game.world->add(type[i], side[i], x[i], z[i], heading[i], hp_frac[i], build_frac[i]);
    game.world->ready_weapons();
    game.world->ensure_views();
    game.commander[0] = commander[0];
    game.commander[1] = commander[1];
    std::copy(base, base + 4, game.base);
}

void* zk_batch_world(void* batch, int g) { return static_cast<Batch*>(batch)->games[size_t(g)].world.get(); }

void zk_batch_step(void* batch, const int* games, int n_games, const zk_order_rec* orders, const int* order_offsets, int ticks,
                   void* records, int cap, int* counts, float* globs, float* stats, void* features, int feature_cap, int* feature_counts) {
    auto* b = static_cast<Batch*>(batch);
    auto* out = static_cast<UnitRecord*>(records);
    auto* out_features = static_cast<FeatureRecord*>(features);
    auto worker = [&](int slice) {
        for (int i = slice; i < n_games; i += b->threads) {
            Game& game = b->games[size_t(games[i])];
            World& w = *game.world;
            for (int o = order_offsets[i]; o < order_offsets[i + 1]; ++o)
                if (w.alive[size_t(orders[o].unit)]) w.order_unit(orders[o].unit, orders[o].kind, orders[o].x, orders[o].z, orders[o].target);
            if (ticks > 0) w.step(ticks);
            w.ensure_views();
            w.rebuild_index();
            for (int s = 0; s < 2; ++s) {
                const size_t slot = size_t(i) * 2 + size_t(s);
                w.update_view(s);
                counts[slot] = w.emit_view(s, out + slot * size_t(cap), cap, globs + slot * G_COUNT, game.commander[s], game.base + 2 * s);
                w.side_stats(s, stats + slot * ZK_STATS);
                feature_counts[slot] = w.emit_features(s, out_features + slot * size_t(feature_cap), feature_cap);
            }
        }
    };
    if (b->threads == 1 || n_games <= 1) {
        for (int t = 0; t < b->threads; ++t) worker(t);
        return;
    }
    std::vector<std::thread> pool;
    for (int t = 0; t < b->threads; ++t) pool.emplace_back(worker, t);
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
