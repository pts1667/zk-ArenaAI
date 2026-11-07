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
constexpr float RECENT_DAMAGE_KEPT = 0.59049f;  // per tick: 0.9 a frame, Unit.cpp

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
    disarm.push_back(0.0f);
    stock.push_back(0);
    stock_progress.push_back(0.0f);
    stun_attackers.push_back(0);
    slow.push_back(0.0f);
    damage_taken.push_back(0.0f);
    recent_damage.push_back(0.0f);
    last_hit.push_back(-1000000);
    closing_since.push_back(-1000000);
    regen_wait.push_back(0.0f);
    regen_seen_damage.push_back(0.0f);
    {
        UnitAI a;
        a.move_state = d.move_state;
        a.fire_state = d.fire_state;
        a.heat = d.heat_initial;
        a.line_x = a.line_end_x = px;
        a.line_z = a.line_end_z = pz;
        ai.push_back(a);
    }
    incoming.emplace_back();
    views[0].updated_frame = views[1].updated_frame = -1;
    goal_x.push_back(px);
    goal_z.push_back(pz);
    order_wait.push_back(0);
    order_lag.push_back(0);
    progress.push_back(WalkProgress{px, pz, 0, 0, -1, 0});
    max_los = std::max(max_los, std::max(d.los_radius, d.air_los_radius));
    mem_kind.push_back(-1);
    mem_frame.push_back(0);
    mem_target.push_back(-1);
    mem_x.push_back(0.0f);
    mem_z.push_back(0.0f);
    burn_rate.push_back(0.0f);
    burn_frames.push_back(0.0f);
    work_on.push_back(-1);
    produces.push_back(-1);
    extracts.push_back(d.metal_make + (d.extract_mult > 0.0f ? d.extract_mult * spot_income(px, pz) : 0.0f));
    overdrive.push_back(0.0f);
    unpaid.push_back(0);
    want_cloak.push_back((d.flags & ZK_U_CAN_CLOAK) && (d.flags & ZK_U_STARTS_CLOAKED) ? 1 : 0);
    area_on.push_back(d.area_cloak_radius > 0.0f && (d.flags & ZK_U_AREA_CLOAK_ON) ? 1 : 0);
    area_cloaked.push_back(0);
    cloaked.push_back(want_cloak.back() && build_frac >= 1.0f ? 1 : 0);
    cloak_blocked_until.push_back(0);
    if ((d.flags & ZK_U_CAN_CLOAK) || d.burrow_speed > 0.0f) personal_cloakers.push_back(u);
    if (d.burrow_speed > 0.0f) burrowers.push_back(u);
    burrow_at.push_back(frame + BURROW_FRAMES);
    jump.emplace_back();
    if (d.jump_range > 0.0f) jumpers.push_back(u);
    if (d.drop_weapon >= 0) droppers.push_back(u);
    if (d.area_cloak_radius > 0.0f) area_cloakers.push_back(u);
    if (d.energy_upkeep > 0.0f || d.area_cloak_upkeep > 0.0f) upkeepers.push_back(u);
    if (d.pylon_range > 0.0f) pylons.push_back(u);
    if (d.build_speed > 0.0f) builders.push_back(u);
    if (d.metal_make > 0.0f || d.energy_make > 0.0f || d.extract_mult > 0.0f) earners.push_back(u);
    float power = 0.0f;
    const zk_weapon_rec* shield_weapon = nullptr;
    for (int i = 0; i < ZK_MAX_MOUNTS; ++i) {
        float start = 0.0f;
        if (i < d.n_mounts) {
            const auto& w = cat->weapons[size_t(d.mount_weapon[i])];
            if (w.kind == ZK_WEAPON_SHIELD) {
                power = std::max(power, w.shield_power);
                shield_weapon = &w;
            }
            // A snapshot catches a weapon somewhere in its reload.
            start = rng.uniform() * w.reload;
        }
        cooldown.push_back(start);
        salvo_timer.push_back(0.0f);
        salvo_left.push_back(0);
        weapon_target.push_back(-1);
    }
    shield.push_back(power);
    if (shield_weapon) shields.push_back({u, shield_weapon, {}});
    return u;
}

void World::ready_weapons() {
    std::fill(cooldown.begin(), cooldown.end(), 0.0f);
    for (int u = 0; u < count(); ++u)
        if (type_of(u).stockpile_frames > 0.0f) stock[size_t(u)] = std::max(stock[size_t(u)], 1);
}

void World::order_unit(int u, int kind, float gx, float gz, int target) {
    if (kind == ZK_ORDER_MANUAL_FIRE && manual_mount(u) < 0) return;
    const auto& d = type_of(u);
    if (kind == ZK_ORDER_PRODUCE) {
        if (d.flags & ZK_U_FACTORY) produces[size_t(u)] = target;
        return;
    }
    if (kind == ZK_ORDER_MOVE_STATE) {
        ai[size_t(u)].move_state = std::clamp(target, 0, 2);
        return;
    }
    if (kind == ZK_ORDER_FIRE_STATE) {
        ai[size_t(u)].fire_state = std::clamp(target, 0, 2);
        return;
    }
    if (kind == ZK_ORDER_UNIT_AI) {
        ai[size_t(u)].tactics_off = target == 0;
        return;
    }
    if (kind == ZK_ORDER_SELF_DESTRUCT) {
        if (d.kamikaze_distance > 0.0f && !stunned(u)) self_destruct(u);
        return;
    }
    if (kind == ZK_ORDER_CLOAK_ON || kind == ZK_ORDER_CLOAK_OFF) {
        if (d.flags & ZK_U_CAN_CLOAK) want_cloak[size_t(u)] = kind == ZK_ORDER_CLOAK_ON;
        if (d.area_cloak_radius > 0.0f) area_on[size_t(u)] = kind == ZK_ORDER_CLOAK_ON;
        return;
    }
    if (kind == ZK_ORDER_JUMP && (d.jump_range <= 0.0f || !can_land_at(u, gx, gz))) return;
    if ((kind == ZK_ORDER_BUILD || kind == ZK_ORDER_ASSIST) && d.build_speed <= 0.0f) return;
    if (kind == ZK_ORDER_RECLAIM && (d.reclaim_speed <= 0.0f || (d.flags & ZK_U_FACTORY))) return;
    if (!(d.flags & ZK_U_FACTORY)) work_on[size_t(u)] = -1;
    jump[size_t(u)].stranded = jump[size_t(u)].busy && kind != ZK_ORDER_JUMP;
    order[size_t(u)] = kind;
    goal_x[size_t(u)] = gx;
    goal_z[size_t(u)] = gz;
    order_target[size_t(u)] = target;
    forget_commands(u);
    // GiveCommand runs the command AI at once, so an order at what can't be seen has used one of its four updates already.
    if (kind == ZK_ORDER_ATTACK && behaviour == ZK_BEHAVIOUR_ORDERS && target >= 0 && alive[size_t(target)]) {
        ensure_views();     // an order can come before the first tick
        target_lost(u, target);
    }
    if (kind == ZK_ORDER_JUMP) {
        // The map's edge, as Jump has it.  GiveCommand runs the command AI at once, so it goes now if it can.
        goal_x[size_t(u)] = gx = std::clamp(gx, 1.0f, map->width() - 1.0f);
        goal_z[size_t(u)] = gz = std::clamp(gz, 1.0f, map->depth() - 1.0f);
        try_jump(u);
    }
    mem_kind[size_t(u)] = kind;
    mem_frame[size_t(u)] = frame;
    // A build's target is a type, an assist's is one of ours and a reclaim's is a feature: none is what the net means by an aim.
    mem_target[size_t(u)] = kind == ZK_ORDER_BUILD || kind == ZK_ORDER_ASSIST || kind == ZK_ORDER_RECLAIM ? -1 : target;
    mem_x[size_t(u)] = gx;
    mem_z[size_t(u)] = gz;
    // Battery: a plain move order stops the unit until its next slow update.  A fight order doesn't.
    if (kind == ZK_ORDER_MOVE) {
        const int until = (SLOW_UPDATE_PHASE + u - frame) % SLOW_UPDATE_FRAMES;
        const int wrapped = until < 0 ? until + SLOW_UPDATE_FRAMES : until;
        // Too close to the update to catch it: the unit waits for the one after (battery, kbots at 0.5 s).
        order_wait[size_t(u)] = wrapped <= ORDER_BRAKE_LAG ? wrapped + SLOW_UPDATE_FRAMES : wrapped;
        order_lag[size_t(u)] = ORDER_BRAKE_LAG;
    }
}

void World::step(int ticks) {
    for (int i = 0; i < ticks; ++i) tick();
}

void World::tick() {
    { ZK_PROF(GRID); rebuild_grid(); }
    if (tick_count % 3 == 0) {  // the engine's 15-frame weapon update
        { ZK_PROF(CLOAK); update_cloak(); }
        if (behaviour == ZK_BEHAVIOUR_ADVANCE) { ZK_PROF(NEAREST); find_nearest(); }
        if (behaviour == ZK_BEHAVIOUR_ORDERS) {
            // Targeting goes by what a side knows, so fog runs for every world with orders.
            ensure_views();
            update_view(0);
            update_view(1);
            ZK_PROF(UNIT_AI);
            update_unit_ai();
        }
        if (behaviour == ZK_BEHAVIOUR_ORDERS || behaviour == ZK_BEHAVIOUR_SCRIPTED) update_kamikaze();
        { ZK_PROF(TARGETS); pick_targets(); }
    }
    if (behaviour == ZK_BEHAVIOUR_ORDERS || behaviour == ZK_BEHAVIOUR_SCRIPTED) update_droppers();
    { ZK_PROF(FIRE); fire_weapons(); }
    { ZK_PROF(SHOTS); resolve_shots(); }
    { ZK_PROF(BURN); burn_areas(); }
    { ZK_PROF(MOVE); update_jumps(); move_units(); }
    { ZK_PROF(DEATHS); finish_deaths(); }
    // After deaths: a build step adds health, and mustn't bring back what this tick killed.
    if (economy_on) { ZK_PROF(ECONOMY); run_economy(); }
    { ZK_PROF(STATUS); update_status(); update_shields(); }
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
                if (side[size_t(e)] == side[size_t(u)] || !can_be_aimed_at(side[size_t(u)], e)) return;
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
        recent_damage[size_t(u)] *= RECENT_DAMAGE_KEPT;
        para[size_t(u)] = std::max(0.0f, para[size_t(u)] - frames / PARA_DECLINE_FRAMES);
        disarm[size_t(u)] = std::max(0.0f, disarm[size_t(u)] - frames / PARA_DECLINE_FRAMES);
        // Stops for stun, disarm and an unfinished unit, and is slowed like everything else it does (unit_stockpile.lua).
        if (d.stockpile_frames > 0.0f && active(u) && !disarmed(u)) {
            const float made = frames * (1.0f - slow_now(u));
            // Both resources, and all of a step or none of it.  A world with no economy has nothing to pay with, and stockpiles free.
            const float cost = economy_on ? d.stockpile_cost * made / d.stockpile_frames : 0.0f;
            Purse& purse = purses[size_t(side[size_t(u)])];
            if (purse.metal >= cost && purse.energy >= cost) {
                purse.metal -= cost;
                purse.energy -= cost;
                purse.energy_spent += cost;
                stock_progress[size_t(u)] += made;
                if (stock_progress[size_t(u)] >= d.stockpile_frames) {
                    stock_progress[size_t(u)] -= d.stockpile_frames;
                    ++stock[size_t(u)];
                }
            }
        }
        slow[size_t(u)] = std::max(0.0f, slow[size_t(u)] - frames / SLOW_DECLINE_FRAMES);
        if (burn_frames[size_t(u)] > 0.0f) {
            const float amount = burn_rate[size_t(u)] * std::min(frames, burn_frames[size_t(u)]);
            hp[size_t(u)] -= amount;
            damage_taken[size_t(u)] += amount;
            recent_damage[size_t(u)] += amount;
            burn_frames[size_t(u)] -= frames;
        }
        // unit_regeneration.lua.  It adds half the rate every 15 frames; this adds it evenly.
        if (damage_taken[size_t(u)] > regen_seen_damage[size_t(u)]) {
            regen_seen_damage[size_t(u)] = damage_taken[size_t(u)];
            regen_wait[size_t(u)] = d.idle_time;
        } else {
            regen_wait[size_t(u)] = std::max(0.0f, regen_wait[size_t(u)] - frames);
        }
        if (d.idle_regen > 0.0f && regen_wait[size_t(u)] <= 0.0f && hp[size_t(u)] > 0.0f && build[size_t(u)] >= 1.0f && !stunned(u) && !disarmed(u))
            hp[size_t(u)] = std::max(hp[size_t(u)], std::min(d.hp, hp[size_t(u)] + d.idle_regen * frames));
    }
}

}  // namespace zk
