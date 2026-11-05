// Struct-of-arrays state of one game, and the read-only catalog and map it shares with others.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "zksim/capi.h"

namespace zk {

struct Catalog {
    std::vector<zk_type_rec> types;
    std::vector<zk_weapon_rec> weapons;
    std::vector<float> priority;
    float target_priority(int weapon, int type) const { return priority[size_t(weapon) * types.size() + size_t(type)]; }
};

// Whether a walking unit is getting anywhere, and what it's trying if it isn't.
struct WalkProgress {
    float anchor_x = 0.0f, anchor_z = 0.0f;  // where it was when the window began
    int ticks = 0;                           // into the window
    int attempts = 0;                        // windows without progress since it last changed segment
    int segment = -1;                        // where those attempts began
    int escape_ticks = 0;                    // left of aiming straight at its way out
};

// One class's way around a map: tools/sim_data.py works this out, so a run needs no pathfinding.
struct Routes {
    int nodes = 0;                      // segments
    std::vector<float> centre_x, centre_z;
    std::vector<int> edge_to;           // per gate between two segments
    std::vector<float> gate_x, gate_z;
    std::vector<int> next_edge;         // nodes x nodes: the gate to take from a segment towards a goal's, or -1
};

struct Map {
    int cols = 0, rows = 0, classes = 0;
    float cell = 16.0f;
    std::vector<float> height;
    std::vector<float> speed;
    std::vector<float> slope, normal_x, normal_z, slope_mod;
    std::vector<short> nearest_segment;           // per class and cell: the segment there, or the nearest
    std::vector<Routes> routes;                   // per class
    void set_segments(const short* labels);
    int segment_at(int cls, float x, float z) const { return nearest_segment[size_t(cls) * size_t(rows) * size_t(cols) + size_t(cell_index(x, z))]; }
    float width() const { return float(cols) * cell; }
    float depth() const { return float(rows) * cell; }
    int cell_index(float x, float z) const {
        int cx = int(x / cell), cz = int(z / cell);
        cx = cx < 0 ? 0 : (cx >= cols ? cols - 1 : cx);
        cz = cz < 0 ? 0 : (cz >= rows ? rows - 1 : cz);
        return cz * cols + cx;
    }
    float height_at(float x, float z) const { return height[size_t(cell_index(x, z))]; }
    const float* speed_grid(int cls) const { return &speed[size_t(cls) * size_t(rows) * size_t(cols)]; }
    float speed_at(int cls, float x, float z) const { return speed_grid(cls)[size_t(cell_index(x, z))]; }
    // CMoveMath's speed for a move in this direction: climbing slows a unit, descending doesn't.
    float speed_along(int cls, float x, float z, float dir_x, float dir_z) const {
        const size_t i = size_t(cell_index(x, z));
        const float base = speed_grid(cls)[i], mod = slope_mod[size_t(cls)], s = slope[i];
        if (base <= 0.0f || mod <= 0.0f || s <= 0.0f || height[i] < 0.0f) return base;
        const float climb = -(dir_x * normal_x[i] + dir_z * normal_z[i]);
        const float directed = climb > 0.0f ? s * climb : 0.0f;
        return base * (1.0f + s * mod) / (1.0f + directed * mod);
    }
};

// xoshiro256**: one per world, so a result doesn't depend on how worlds share threads.
struct Rng {
    uint64_t s[4];
    explicit Rng(uint64_t seed);
    uint64_t next();
    float uniform() { return float(next() >> 40) * (1.0f / float(1ull << 24)); }
};

// A shot in flight: damage is scheduled, there is no projectile.
struct Shot {
    int weapon, source, source_side, target;
    float x, z, pending;
    bool guided, air;
};

// Ground fire left by a shot.
struct FireArea {
    const zk_weapon_rec* weapon;
    float x, z, damage_per_frame;
    int frames_left;
};

struct World {
    const Catalog* cat;
    const Map* map;
    Rng rng;
    int behaviour;
    int frame = 0;
    int tick_count = 0;

    // Units.
    std::vector<int> type, side, order, order_target, nearest;
    std::vector<uint8_t> alive;
    std::vector<float> x, z, vx, vz, heading, speed, hp, build, para, slow, damage_taken, pending;
    std::vector<float> goal_x, goal_z, shield;
    std::vector<int> order_wait;          // frames until a new move order takes effect
    std::vector<int> order_lag;           // frames before the unit starts to brake for it
    std::vector<WalkProgress> progress;
    std::vector<float> burn_rate, burn_frames;
    std::vector<FireArea> areas;
    // Weapons, ZK_MAX_MOUNTS per unit.
    std::vector<float> cooldown, salvo_timer;
    std::vector<int> salvo_left, weapon_target;

    // Spatial index, rebuilt each tick.
    int grid_cols = 0, grid_rows = 0;
    std::vector<int> cell_start, cell_units, cell_fill;

    // Shots by the tick they land.
    std::vector<std::vector<Shot>> ring;

    World(const Catalog* c, const Map* m, uint64_t seed, int behaviour_);
    int add(int type, int side, float x, float z, float heading, float hp_frac, float build_frac);
    void order_unit(int unit, int kind, float gx, float gz, int target);
    void ready_weapons();
    void step(int ticks);
    int count() const { return int(type.size()); }
    const zk_type_rec& type_of(int u) const { return cat->types[size_t(type[size_t(u)])]; }
    bool stunned(int u) const { return para[size_t(u)] > 1.0f; }
    bool active(int u) const { return build[size_t(u)] >= 1.0f && !stunned(u); }

    template <class F>
    void for_near(float cx, float cz, float radius, F&& f) const;

private:
    void tick();
    void rebuild_grid();
    void find_nearest();
    void pick_targets();
    void fire_weapons();
    void fire_one(int u, int slot, int frame_offset);
    void resolve_shots();
    void apply_shot(const Shot& s);
    void explode(int weapon, float px, float pz, int source, int source_side, bool air);
    void damage_unit(int u, const zk_weapon_rec& w, float mult);
    bool absorbed_by_shield(const zk_weapon_rec& w, float px, float pz, const Shot& s);
    void move_units();
    void track_progress(int u, bool walking);
    void move_one(int u);
    bool wants_to_walk(int u, float& gx, float& gz);
    void separate();
    void finish_deaths();
    void update_status();
    void burn_areas();
    float target_score(int u, int slot, int target) const;
    bool in_range(int u, const zk_weapon_rec& w, int target) const;
    bool faces(int u, const zk_weapon_rec& w, int target) const;
    void waypoint(int u, float gx, float gz, float& wx, float& wz) const;
};

template <class F>
void World::for_near(float cx, float cz, float radius, F&& f) const {
    constexpr float CELL = 128.0f;
    int x0 = int((cx - radius) / CELL), x1 = int((cx + radius) / CELL);
    int z0 = int((cz - radius) / CELL), z1 = int((cz + radius) / CELL);
    x0 = x0 < 0 ? 0 : x0;
    z0 = z0 < 0 ? 0 : z0;
    x1 = x1 >= grid_cols ? grid_cols - 1 : x1;
    z1 = z1 >= grid_rows ? grid_rows - 1 : z1;
    for (int gz = z0; gz <= z1; ++gz)
        for (int gx = x0; gx <= x1; ++gx) {
            const size_t c = size_t(gz) * size_t(grid_cols) + size_t(gx);
            for (int i = cell_start[c]; i < cell_start[c + 1]; ++i) f(cell_units[size_t(i)]);
        }
}

}  // namespace zk
