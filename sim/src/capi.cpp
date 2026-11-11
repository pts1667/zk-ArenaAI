// The C ABI, and the threaded replay of logged fights.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <unistd.h>

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
    int commander[ZK_MAX_TEAMS] = {-1, -1, -1, -1, -1, -1, -1, -1};      // per team
    float base[2 * ZK_MAX_TEAMS] = {};
    long long cost = 0;     // nanoseconds its last step took: what the schedule goes by, never the results
};

// What zk_batch_step was given: the caller keeps it alive until the step is over.
struct StepArgs {
    const int* games = nullptr;
    int n_games = 0;
    const zk_order_rec* orders = nullptr;
    const int* order_offsets = nullptr;
    const zk_group_rec* groups = nullptr;      // may be null
    const int* group_offsets = nullptr;
    int ticks = 0;
    UnitRecord* records = nullptr;
    int cap = 0;
    int* counts = nullptr;
    float* globs = nullptr;
    float* stats = nullptr;
    FeatureRecord* features = nullptr;
    int feature_cap = 0;
    int* feature_counts = nullptr;
};

// A batch's games and the threads that step them.  The threads live as long as the batch and wait between steps.  A step
// hands out games one at a time, costliest first, to whichever thread is free; each game is stepped whole by one thread
// and has its own RNG and output slots, so the results are the same for any thread count and any schedule.  Batches share
// nothing but the read-only catalog and maps, so several can step at once from different threads.
class Batch {
public:
    const Catalog* cat = nullptr;
    std::vector<Game> games;
    int slots = 2;                        // teams a game has room for in a step's arrays
    std::vector<int8_t> role_of_type;     // zk_batch_set_roles: each new game's
    bool reveal = true;                   // zk_batch_set_reveal: each new game's

    Batch(const Catalog* c, int n, int threads_, int teams) : cat(c), games(size_t(n)), slots(std::clamp(teams, 1, ZK_MAX_TEAMS)) {
        set_threads(threads_);
    }

    ~Batch() {
        if (forked()) return;       // the threads are the parent's, and there's no one here to stop
        wait();
        {
            std::lock_guard<std::mutex> lock(sync->mutex);
            stopping = true;
        }
        sync->wake.notify_all();
        for (auto& t : workers) t.join();
    }

    void set_threads(int n) {
        wait();
        threads = std::clamp(n, 1, 256);
#ifdef ZKSIM_PROFILE
        if (counters.size() < size_t(threads) + 1) counters.resize(size_t(threads) + 1);
#endif
    }

    // Steps on the calling thread and threads - 1 workers, and returns when every game is done.
    void step(const StepArgs& a) {
        wait();
        prepare(a);
        if (threads == 1 || a.n_games <= 1) {
            run(0);
            return;
        }
        launch(threads - 1);
        run(0);
        wait();
    }

    // Steps on `threads` workers and returns at once; wait() is when it's over.
    void begin(const StepArgs& a) {
        wait();
        prepare(a);
        launch(threads);
    }

    void wait() {
        if (forked()) adopt();
        std::unique_lock<std::mutex> lock(sync->mutex);
        sync->done.wait(lock, [&] { return running == 0; });
    }

#ifdef ZKSIM_PROFILE
    void read_profile(long long* nanos, long long* calls, bool reset) {
        wait();
        std::fill(nanos, nanos + prof::PHASE_COUNT, 0LL);
        std::fill(calls, calls + prof::PHASE_COUNT, 0LL);
        for (auto& c : counters) {
            c.add_to(nanos, calls);
            if (reset) c.clear();
        }
    }
#endif

private:
    int threads = 1;
    StepArgs args;
    std::vector<int> schedule;          // positions in args.games, costliest game first
    std::atomic<int> next{0};
    std::vector<std::thread> workers;
    struct Sync {
        std::mutex mutex;
        std::condition_variable wake, done;
    };
    std::unique_ptr<Sync> sync = std::make_unique<Sync>();
    pid_t owner = getpid();             // the process the workers belong to
    unsigned long long generation = 0;  // of the step the workers are told about
    int taking_part = 0;                // workers with an index below this step the current one
    int running = 0;                    // of those, the ones not yet finished
    bool stopping = false;
#ifdef ZKSIM_PROFILE
    std::vector<prof::Counters> counters;   // the caller's, then each worker's
#endif

    bool forked() const { return owner != getpid(); }

    // A child of fork() has the parent's batch but none of its threads.  Their handles can't be joined or destroyed, nor the
    // locks trusted, so all of them are let go (a little memory, once) and the batch starts again with its own.  A fork in the
    // middle of a step leaves that step's games half done.
    void adopt() {
        new std::vector<std::thread>(std::move(workers));
        workers.clear();
        (void)sync.release();
        sync = std::make_unique<Sync>();
        owner = getpid();
        generation = 0;
        taking_part = running = 0;
        stopping = false;
    }

    void prepare(const StepArgs& a) {
        args = a;
        schedule.resize(size_t(a.n_games));
        for (int i = 0; i < a.n_games; ++i) schedule[size_t(i)] = i;
        std::sort(schedule.begin(), schedule.end(), [&](int p, int q) {
            const long long cp = games[size_t(a.games[p])].cost, cq = games[size_t(a.games[q])].cost;
            return cp != cq ? cp > cq : p < q;
        });
        next.store(0, std::memory_order_relaxed);
    }

    void launch(int helpers) {
        while (int(workers.size()) < helpers) {
            const int index = int(workers.size());
            const unsigned long long now = generation;      // only this thread changes it
            workers.emplace_back([this, index, now] { work(index, now); });
        }
        {
            std::lock_guard<std::mutex> lock(sync->mutex);
            taking_part = helpers;
            running = helpers;
            ++generation;
        }
        sync->wake.notify_all();
    }

    void work(int index, unsigned long long seen) {
        Sync& s = *sync;
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(s.mutex);
                s.wake.wait(lock, [&] { return stopping || generation != seen; });
                if (stopping) return;
                seen = generation;
                if (index >= taking_part) continue;
            }
            run(index + 1);
            bool last;
            {
                std::lock_guard<std::mutex> lock(s.mutex);
                last = --running == 0;
            }
            if (last) s.done.notify_all();
        }
    }

    void run(int slot) {
#ifdef ZKSIM_PROFILE
        prof::Redirect into(&counters[size_t(slot)]);
#else
        (void)slot;
#endif
        for (;;) {
            const int k = next.fetch_add(1, std::memory_order_relaxed);
            if (k >= args.n_games) return;
            step_game(schedule[size_t(k)]);
        }
    }

    void step_game(int i) {
        const auto start = std::chrono::steady_clock::now();
        const StepArgs& a = args;
        Game& game = games[size_t(a.games[i])];
        World& w = *game.world;
        {
            ZK_PROF(ORDERS);
            if (a.groups)
                for (int o = a.group_offsets[i]; o < a.group_offsets[i + 1]; ++o) w.order_group(a.groups[o]);
            for (int o = a.order_offsets[i]; o < a.order_offsets[i + 1]; ++o)
                if (w.alive[size_t(a.orders[o].unit)]) w.order_unit(a.orders[o].unit, a.orders[o].kind, a.orders[o].x, a.orders[o].z, a.orders[o].target);
        }
        if (a.ticks > 0) w.step(a.ticks);
        w.ensure_views();
        w.rebuild_index();
        for (int t = 0; t < slots; ++t) {
            const size_t slot = size_t(i) * size_t(slots) + size_t(t);
            if (t >= w.n_teams) {
                // A slot the game doesn't use: an empty frame.
                a.counts[slot] = a.feature_counts[slot] = 0;
                std::fill_n(a.globs + slot * G_COUNT, G_COUNT, 0.0f);
                std::fill_n(a.stats + slot * ZK_STATS, ZK_STATS, 0.0f);
                continue;
            }
            const int s = w.ally_of[t];
            w.update_view(s);
            {
                ZK_PROF(EMIT);
                a.counts[slot] = w.emit_view(t, a.records + slot * size_t(a.cap), a.cap, a.globs + slot * G_COUNT, game.commander[t], game.base + 2 * t);
            }
            {
                ZK_PROF(STATS);
                w.team_stats(t, a.stats + slot * ZK_STATS);
            }
            {
                ZK_PROF(FEATURES);
                // An allyteam's features are the same for each of its teams: the first's are copied.
                int first = 0;
                while (w.ally_of[first] != s) ++first;
                FeatureRecord* out = a.features + slot * size_t(a.feature_cap);
                if (first == t) {
                    a.feature_counts[slot] = w.emit_features(s, out, a.feature_cap);
                } else {
                    const size_t from = size_t(i) * size_t(slots) + size_t(first);
                    a.feature_counts[slot] = a.feature_counts[from];
                    std::copy_n(a.features + from * size_t(a.feature_cap), a.feature_counts[from], out);
                }
            }
        }
        game.cost = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
    }
};

StepArgs step_args(const int* games, int n_games, const zk_order_rec* orders, const int* order_offsets, const zk_group_rec* groups,
                   const int* group_offsets, int ticks, void* records, int cap,
                   int* counts, float* globs, float* stats, void* features, int feature_cap, int* feature_counts) {
    StepArgs a;
    a.games = games;
    a.n_games = n_games;
    a.orders = orders;
    a.order_offsets = order_offsets;
    a.groups = groups;
    a.group_offsets = group_offsets;
    a.ticks = ticks;
    a.records = static_cast<UnitRecord*>(records);
    a.cap = cap;
    a.counts = counts;
    a.globs = globs;
    a.stats = stats;
    a.features = static_cast<FeatureRecord*>(features);
    a.feature_cap = feature_cap;
    a.feature_counts = feature_counts;
    return a;
}

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

void zk_catalog_reveal(void* catalog, float value_ratio, int every_frames) {
    auto* c = static_cast<Catalog*>(catalog);
    c->reveal_ratio = value_ratio;
    c->reveal_every = every_frames;
}

void zk_catalog_chickens(void* catalog, int difficulty, const zk_chicken_rules* rules) {
    auto* c = static_cast<Catalog*>(catalog);
    if (difficulty < 0 || difficulty >= 6) return;
    c->chickens[difficulty] = *rules;
    c->has_chickens[difficulty] = true;
}

int zk_sizeof_chicken_rules(void) { return int(sizeof(zk_chicken_rules)); }

void zk_catalog_eggs(void* catalog, const zk_egg_rec* eggs) {
    auto* c = static_cast<Catalog*>(catalog);
    c->eggs.assign(eggs, eggs + c->types.size());
}

void* zk_map_new(int cols, int rows, float cell, const float* height, const float* speed, int classes,
                 const float* slope, const float* normal_x, const float* normal_z, const float* slope_mod,
                 const short* segments) {
    auto* m = new Map;
    m->cols = cols;
    m->rows = rows;
    m->classes = classes;
    m->set_cell(cell);
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

void zk_world_set_teams(void* world, int n_teams, const int32_t* ally_of_team) { static_cast<World*>(world)->set_teams(n_teams, ally_of_team); }

int zk_world_add(void* world, int type, int team, float x, float z, float heading, float hp_frac, float build_frac) {
    auto* w = static_cast<World*>(world);
    if (team < 0 || team >= w->n_teams) return -1;
    return w->add(type, team, x, z, heading, hp_frac, build_frac);
}

void zk_world_order(void* world, int unit, int kind, float x, float z, int target) {
    static_cast<World*>(world)->order_unit(unit, kind, x, z, target);
}

void zk_world_set_roles(void* world, const int32_t* role_of_type, int n_types) {
    static_cast<World*>(world)->role_of_type.assign(role_of_type, role_of_type + n_types);
}

int zk_world_order_group(void* world, const zk_group_rec* group) { return static_cast<World*>(world)->order_group(*group); }

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

int zk_world_features(void* world, int team, void* records, int cap) {
    auto* w = static_cast<World*>(world);
    w->rebuild_index();
    return w->emit_features(w->ally_of[team], static_cast<FeatureRecord*>(records), cap);
}

void zk_world_set_reveal(void* world, int on) { static_cast<World*>(world)->reveal_on = on != 0; }

void zk_world_chickens(void* world, int team, int difficulty) { static_cast<World*>(world)->start_chickens(team, difficulty); }

void zk_world_chicken_state(const void* world, float* out) { static_cast<const World*>(world)->chicken_state(out); }

void zk_world_resources(const void* world, int team, float* out) {
    const Purse& p = static_cast<const World*>(world)->purses[size_t(team)];
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
        out->altitude = flight.flying ? (1.0f - bend * bend) * flight.height : w->skid[u].y + w->air[u].height;
    }
    out->ammo = w->air[u].ammo;
    out->air_state = (w->type_of(unit).flags & ZK_U_AIR) ? w->air[u].phase : 0;
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

int zk_world_view(void* world, int team, void* records, int cap, float* glob, int commander, const float* base) {
    auto* w = static_cast<World*>(world);
    w->ensure_views();
    w->rebuild_index();
    w->update_view(w->ally_of[team]);
    return w->emit_view(team, static_cast<UnitRecord*>(records), cap, glob, commander, base);
}

void* zk_batch_new(const void* catalog, int games, int threads, int teams) {
    return new Batch(static_cast<const Catalog*>(catalog), games, threads, teams);
}

void zk_batch_free(void* batch) { delete static_cast<Batch*>(batch); }

void zk_batch_set_threads(void* batch, int threads) { static_cast<Batch*>(batch)->set_threads(threads); }

void zk_batch_reset(void* batch, int g, const void* map, uint64_t seed, int behaviour, int n_teams, const int32_t* ally_of_team,
                    int n, const int* type, const int* team, const float* x, const float* z, const float* heading,
                    const float* hp_frac, const float* build_frac, const int* commander, const float* base) {
    auto* b = static_cast<Batch*>(batch);
    Game& game = b->games[size_t(g)];
    game.world = std::make_unique<World>(b->cat, static_cast<const Map*>(map), seed, behaviour);
    World& w = *game.world;
    w.set_teams(std::min(n_teams, b->slots), ally_of_team);
    w.role_of_type = b->role_of_type;
    w.reveal_on = b->reveal;
    for (int i = 0; i < n; ++i)
        if (team[i] >= 0 && team[i] < w.n_teams) w.add(type[i], team[i], x[i], z[i], heading[i], hp_frac[i], build_frac[i]);
    w.ready_weapons();
    w.ensure_views();
    for (int t = 0; t < ZK_MAX_TEAMS; ++t) {
        game.commander[t] = t < w.n_teams ? commander[t] : -1;
        game.base[2 * t] = t < w.n_teams ? base[2 * t] : 0.0f;
        game.base[2 * t + 1] = t < w.n_teams ? base[2 * t + 1] : 0.0f;
    }
}

void* zk_batch_world(void* batch, int g) { return static_cast<Batch*>(batch)->games[size_t(g)].world.get(); }

void zk_batch_set_roles(void* batch, const int32_t* role_of_type, int n_types) {
    auto* b = static_cast<Batch*>(batch);
    b->wait();
    b->role_of_type.assign(role_of_type, role_of_type + n_types);
    for (auto& game : b->games)
        if (game.world) game.world->role_of_type = b->role_of_type;
}

void zk_batch_set_reveal(void* batch, int on) {
    auto* b = static_cast<Batch*>(batch);
    b->wait();
    b->reveal = on != 0;
    for (auto& game : b->games)
        if (game.world) game.world->reveal_on = b->reveal;
}

void zk_batch_chickens(void* batch, int g, int team, int difficulty) {
    auto* b = static_cast<Batch*>(batch);
    b->wait();
    if (World* w = b->games[size_t(g)].world.get()) w->start_chickens(team, difficulty);
}

void zk_batch_chicken_state(const void* batch, int g, float* out) {
    auto* b = const_cast<Batch*>(static_cast<const Batch*>(batch));
    b->wait();
    if (const World* w = b->games[size_t(g)].world.get()) w->chicken_state(out);
    else std::fill_n(out, ZK_CHICKEN_STATE, 0.0f);
}

void zk_batch_step(void* batch, const int* games, int n_games, const zk_order_rec* orders, const int* order_offsets,
                   const zk_group_rec* groups, const int* group_offsets, int ticks,
                   void* records, int cap, int* counts, float* globs, float* stats, void* features, int feature_cap, int* feature_counts) {
    static_cast<Batch*>(batch)->step(step_args(games, n_games, orders, order_offsets, groups, group_offsets, ticks, records, cap, counts, globs, stats,
                                               features, feature_cap, feature_counts));
}

void zk_batch_step_begin(void* batch, const int* games, int n_games, const zk_order_rec* orders, const int* order_offsets,
                         const zk_group_rec* groups, const int* group_offsets, int ticks,
                         void* records, int cap, int* counts, float* globs, float* stats, void* features, int feature_cap, int* feature_counts) {
    static_cast<Batch*>(batch)->begin(step_args(games, n_games, orders, order_offsets, groups, group_offsets, ticks, records, cap, counts, globs, stats,
                                                features, feature_cap, feature_counts));
}

void zk_batch_step_wait(void* batch) { static_cast<Batch*>(batch)->wait(); }

}  // extern "C"

#ifdef ZKSIM_PROFILE
extern "C" {
int zk_prof_count() { return zk::prof::PHASE_COUNT; }
const char* zk_prof_name(int phase) { return zk::prof::NAMES[phase]; }
void zk_prof_read(long long* nanos, long long* calls, int reset) {
    auto& own = zk::prof::own;
    for (int i = 0; i < zk::prof::PHASE_COUNT; ++i) {
        nanos[i] = own.nanos[i];
        calls[i] = own.calls[i];
    }
    if (reset) own.clear();
}
void zk_batch_prof_read(void* batch, long long* nanos, long long* calls, int reset) { static_cast<Batch*>(batch)->read_profile(nanos, calls, reset != 0); }
}
#endif
