// Setup, tick order, the spatial index and status timers.
#include <algorithm>
#include <cmath>

#include "zksim/profile.hpp"
#include "zksim/world.hpp"

namespace zk {

namespace {
constexpr float GRID_CELL = 128.0f;
constexpr int RING_SLOTS = 256;                 // 42 s of flight, more than any weapon needs
constexpr float PARA_DECLINE_FRAMES = 1200.0f;  // Game.paralyzeDeclineRate 40 s at 30 frames
constexpr float SLOW_DECLINE_FRAMES = 750.0f;   // timeslow_defs.lua: 0.04 of health per second
constexpr float NEAREST_RADII[] = {600.0f, 2000.0f, 8000.0f};
constexpr int SLOW_UPDATE_FRAMES = 15;          // UNIT_SLOWUPDATE_RATE; staggered across units
constexpr int SLOW_UPDATE_PHASE = 4;            // the first unit's, as the battery measures it
constexpr int ORDER_BRAKE_LAG = 2;              // the battery's frames from a move order to braking

uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
}  // namespace

Rng::Rng(uint64_t seed) {
    // splitmix64 expands the seed.
    for (auto& v : s) {
        uint64_t z = (seed += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        v = z ^ (z >> 31);
    }
}

uint64_t Rng::next() {
    const uint64_t result = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 45);
    return result;
}

World::World(const Catalog* c, const Map* m, uint64_t seed, int behaviour_)
    : cat(c), map(m), rng(seed), behaviour(behaviour_), ring(RING_SLOTS) {
    grid_cols = std::max(1, int(std::ceil(m->width() / GRID_CELL)));
    grid_rows = std::max(1, int(std::ceil(m->depth() / GRID_CELL)));
    cell_start.resize(size_t(grid_cols) * size_t(grid_rows) + 1);
}

int World::add(int t, int s, float px, float pz, float head, float hp_frac, float build_frac) {
    const auto& d = cat->types[size_t(t)];
    const int u = count();
    type.push_back(t);
    side.push_back(s);
    order.push_back(ZK_ORDER_NONE);
    order_target.push_back(-1);
    nearest.push_back(-1);
    alive.push_back(1);
    x.push_back(px);
    z.push_back(pz);
    vx.push_back(0.0f);
    vz.push_back(0.0f);
    heading.push_back(head);
    speed.push_back(0.0f);
    hp.push_back(std::max(0.01f, hp_frac) * d.hp);
    build.push_back(build_frac);
    para.push_back(0.0f);
    slow.push_back(0.0f);
    damage_taken.push_back(0.0f);
    pending.push_back(0.0f);
    goal_x.push_back(px);
    goal_z.push_back(pz);
    order_wait.push_back(0);
    order_lag.push_back(0);
    progress.push_back(WalkProgress{px, pz, 0, 0, -1, 0});
    burn_rate.push_back(0.0f);
    burn_frames.push_back(0.0f);
    float power = 0.0f;
    for (int i = 0; i < ZK_MAX_MOUNTS; ++i) {
        float start = 0.0f;
        if (i < d.n_mounts) {
            const auto& w = cat->weapons[size_t(d.mount_weapon[i])];
            if (w.kind == ZK_WEAPON_SHIELD) power = std::max(power, w.shield_power);
            // A snapshot catches a weapon somewhere in its reload.
            start = rng.uniform() * w.reload;
        }
        cooldown.push_back(start);
        salvo_timer.push_back(0.0f);
        salvo_left.push_back(0);
        weapon_target.push_back(-1);
    }
    shield.push_back(power);
    return u;
}

void World::ready_weapons() { std::fill(cooldown.begin(), cooldown.end(), 0.0f); }

void World::order_unit(int u, int kind, float gx, float gz, int target) {
    order[size_t(u)] = kind;
    goal_x[size_t(u)] = gx;
    goal_z[size_t(u)] = gz;
    order_target[size_t(u)] = target;
    // Battery: a plain move order stops the unit until its next slow update.  A fight order doesn't.
    if (kind == ZK_ORDER_MOVE) {
        const int until = (SLOW_UPDATE_PHASE + u - frame) % SLOW_UPDATE_FRAMES;
        const int wrapped = until < 0 ? until + SLOW_UPDATE_FRAMES : until;
        order_wait[size_t(u)] = wrapped == 0 ? SLOW_UPDATE_FRAMES : wrapped;
        order_lag[size_t(u)] = ORDER_BRAKE_LAG;
    }
}

void World::step(int ticks) {
    for (int i = 0; i < ticks; ++i) tick();
}

void World::tick() {
    { ZK_PROF(GRID); rebuild_grid(); }
    if (tick_count % 3 == 0) {  // the engine's 15-frame weapon update
        if (behaviour == ZK_BEHAVIOUR_ADVANCE) { ZK_PROF(NEAREST); find_nearest(); }
        { ZK_PROF(TARGETS); pick_targets(); }
    }
    { ZK_PROF(FIRE); fire_weapons(); }
    { ZK_PROF(SHOTS); resolve_shots(); }
    { ZK_PROF(BURN); burn_areas(); }
    { ZK_PROF(MOVE); move_units(); }
    { ZK_PROF(DEATHS); finish_deaths(); }
    { ZK_PROF(STATUS); update_status(); }
    frame += ZK_TICK_FRAMES;
    ++tick_count;
}

void World::rebuild_grid() {
    const int n = count();
    std::fill(cell_start.begin(), cell_start.end(), 0);
    auto cell_of = [&](int u) {
        int gx = int(x[size_t(u)] / GRID_CELL), gz = int(z[size_t(u)] / GRID_CELL);
        gx = gx < 0 ? 0 : (gx >= grid_cols ? grid_cols - 1 : gx);
        gz = gz < 0 ? 0 : (gz >= grid_rows ? grid_rows - 1 : gz);
        return size_t(gz) * size_t(grid_cols) + size_t(gx);
    };
    int live = 0;
    for (int u = 0; u < n; ++u)
        if (alive[size_t(u)]) {
            ++cell_start[cell_of(u) + 1];
            ++live;
        }
    for (size_t c = 1; c < cell_start.size(); ++c) cell_start[c] += cell_start[c - 1];
    cell_units.resize(size_t(live));
    cell_fill.assign(cell_start.begin(), cell_start.end() - 1);
    for (int u = 0; u < n; ++u)
        if (alive[size_t(u)]) cell_units[size_t(cell_fill[cell_of(u)]++)] = u;
}

void World::find_nearest() {
    const int n = count();
    for (int u = 0; u < n; ++u) {
        nearest[size_t(u)] = -1;
        if (!alive[size_t(u)] || !(type_of(u).flags & ZK_U_MOBILE)) continue;
        // Fights are small and the grid coarse: widen until something turns up.
        float best = 1e18f;
        for (float r : NEAREST_RADII) {
            for_near(x[size_t(u)], z[size_t(u)], r, [&](int e) {
                if (side[size_t(e)] == side[size_t(u)]) return;
                const float dx = x[size_t(e)] - x[size_t(u)], dz = z[size_t(e)] - z[size_t(u)];
                const float d = dx * dx + dz * dz;
                if (d < best) {
                    best = d;
                    nearest[size_t(u)] = e;
                }
            });
            if (nearest[size_t(u)] >= 0) break;
        }
    }
}

void World::update_status() {
    const int n = count();
    const float frames = float(ZK_TICK_FRAMES);
    for (int u = 0; u < n; ++u) {
        if (!alive[size_t(u)]) continue;
        const auto& d = type_of(u);
        para[size_t(u)] = std::max(0.0f, para[size_t(u)] - frames / PARA_DECLINE_FRAMES);
        slow[size_t(u)] = std::max(0.0f, slow[size_t(u)] - frames / SLOW_DECLINE_FRAMES);
        if (burn_frames[size_t(u)] > 0.0f) {
            const float amount = burn_rate[size_t(u)] * std::min(frames, burn_frames[size_t(u)]);
            hp[size_t(u)] -= amount;
            damage_taken[size_t(u)] += amount;
            burn_frames[size_t(u)] -= frames;
        }
        if (d.auto_heal > 0.0f && build[size_t(u)] >= 1.0f)
            hp[size_t(u)] = std::min(d.hp, hp[size_t(u)] + d.auto_heal * frames);
        for (int m = 0; m < d.n_mounts; ++m) {
            const auto& w = cat->weapons[size_t(d.mount_weapon[m])];
            if (w.kind == ZK_WEAPON_SHIELD)
                shield[size_t(u)] = std::min(w.shield_power, shield[size_t(u)] + w.shield_regen * frames);
        }
    }
}

}  // namespace zk
