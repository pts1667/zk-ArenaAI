// Chicken defense: LuaRules/Gadgets/unit_spawner.lua, with its settings from LuaRules/Configs/spawn_defs.lua as tools/sim_rules.py
// exports them (sim/data/rules.json, given to the catalog).  The chicken team is an ordinary team in its own allyteam with no one
// to play it: the world makes its burrows and chickens and gives them their orders, on the gadget's schedule, with its formulas and
// in its order, drawing from the world's own random numbers.  The numbers here are the gadget's code (its schedule of frames and the
// like), not its settings.  Times are game seconds, as the gadget's formulas have them.
//
// Left out or simplified, each where it's done:
// * Chicken games where the players aren't all allied (the gadget's PvP mode: burrows' target caches, the endgame's miniqueen
//   waves).  The players are one alliance.
// * Where things go: the engine's build test and blocked ground (TestBuildOrder, GetGroundBlocked) are ground the unit could stand
//   on, with no building there (for a burrow: dry land a kbot could walk, with no unit on it), and distances are on the map, not 3D.
// * The digger's ambush, which comes out of the ground, comes on it.  The queen's morph between her forms is a new unit of the other
//   form in her place, with her share of health and her order.  Her extra health is damage taken divided by the gadget's factor.
// * No lava, eggs mode (eggs for metal instead of extractors), endless or speed mode, custom difficulty or mod options.  The queen's
//   experience is commented out in the gadget, and the sim has none.  spawn_defs' alwaysEggs is on, as there.
// * The gadget's loops over Lua tables (pairs) go in no set order; here kinds go by name and burrows and births by when they came.
// * When the queen dies, the chickens' alliance goes (GG.DestroyAlliance) without blasts or eggs.
#include <algorithm>
#include <cmath>

#include "zksim/world.hpp"

namespace zk {

namespace {
constexpr double FPS = 30.0;
constexpr int SECOND_FRAMES = 30, SECOND_PHASE = 17;    // GameFrame: (n + 17) % 30, burrows and queens
constexpr int SLOW_FRAMES = 90, SLOW_PHASE = 29;        // (n + 29) % 90: old age, eggs' decay, idle chickens' orders
constexpr double BURROW_SPAWN_SHARE = 0.25;             // burrowSpawnRate * 0.25 * (burrows + 1) between burrows
constexpr int MORPH_POSTPONED = 60;                     // MorphQueen: frames a stunned or disarmed queen waits
constexpr float NEAREST_ENEMY_RANGE = 20000.0f;         // GetUnitNearestEnemy(burrowID, 20000, false)
constexpr int TARGET_TRIES = 100;                       // ChooseTarget
constexpr int EGG_SCATTER = 30;                         // SpawnEggs: random(-30, 30)
constexpr int OLD_AGE_SPREAD = 10;                      // KillOldChicken: maxAge + random(10)
constexpr int SQUAD_SPREAD_MIN = 75, SQUAD_SPREAD_MAX = 125;    // a turret's or supporter's squad: random(75, 125) / 100
constexpr int ROAM = 2;                                 // CMD.MOVESTATE_ROAM
constexpr float NORTH = 3.14159265f;                    // CreateUnit's facing "n", towards smaller z
constexpr int KBOT_CLASS = 0;                           // tools/sim_data.py MOVE_CLASS_NAMES

const zk_chicken_kind* find_kind(const zk_chicken_rules& r, int role, int type) {
    for (int k = 0; k < r.n_kinds; ++k)
        if (r.kinds[k].role == role && r.kinds[k].type == type) return &r.kinds[k];
    return nullptr;
}

// UnitDestroyed: what of the chickens' leaves an egg that decays.
bool leaves_egg(const zk_chicken_rules& r, int type) {
    return find_kind(r, ZK_CHICKEN_OFFENSE, type) || find_kind(r, ZK_CHICKEN_DEFENSE, type) || type == r.mini_queen;
}
}  // namespace

void World::start_chickens(int t, int difficulty) {
    if (t < 0 || t >= n_teams || difficulty < 0 || difficulty >= 6 || !cat->has_chickens[difficulty]) return;
    ChickenGame& c = chickens;
    c = ChickenGame{};
    const zk_chicken_rules& r = cat->chickens[difficulty];
    c.rules = &r;
    c.team = t;
    c.ally = ally_of[t];
    c.difficulty = difficulty;
    // Every other team is a player's (humanTeams), and the chickens are the one computer team.  What the gadget adjusts for them:
    c.players = n_teams - 1;
    const double players = double(std::max(c.players, 1));
    c.malus = std::pow(players, double(r.player_malus));
    c.burrow_regress_time = r.burrow_regress_time / players;
    c.aggro_per_burrow = r.human_aggro_per_burrow / players;
    c.aggro_defense_factor = r.human_aggro_defense_factor * players;
    c.defense_per_wave = r.defense_per_wave * players;
    c.burrow_spawn_rate = r.burrow_spawn_rate / (c.malus * 0.8 + 0.2);
    c.grace_period = std::max(double(r.grace_period) - double(r.grace_penalty) * (players - 1.0), double(r.grace_period_min));
    c.queen_time = c.base_queen_time = r.queen_time;
    c.queen_health_mod = r.queen_health_mod;
    c.wave_schedule = c.grace_period * FPS;     // GameStart: the first wave once the grace period is over
    c.fallen_seen = fallen.size();
}

void World::chicken_state(float* out) const {
    std::fill_n(out, ZK_CHICKEN_STATE, 0.0f);
    out[ZK_CS_TEAM] = -1.0f;
    const ChickenGame& c = chickens;
    if (!c.rules) return;
    out[ZK_CS_QUEEN] = c.victory ? 2.0f : c.queen >= 0 ? 1.0f : 0.0f;
    out[ZK_CS_UNTIL_QUEEN] = c.endgame ? 0.0f : float(std::max(0.0, c.queen_time - frame / FPS));
    out[ZK_CS_BURROWS] = float(c.burrows.size());
    out[ZK_CS_BURROWS_KILLED] = float(c.burrows_killed);
    int alive_chickens = 0;
    for (const int u : live) alive_chickens += alive[size_t(u)] && team[size_t(u)] == c.team && type[size_t(u)] != c.rules->burrow;
    out[ZK_CS_CHICKENS] = float(alive_chickens);
    out[ZK_CS_WAVES] = float(c.waves);
    out[ZK_CS_AGGRO] = float(c.aggro);
    out[ZK_CS_TECH] = float(c.tech);
    out[ZK_CS_QUEEN_TIME] = float(c.queen_time);
    out[ZK_CS_QUEEN_HP] = c.queen >= 0 && alive[size_t(c.queen)] ? hp[size_t(c.queen)] / type_of(c.queen).hp : 0.0f;
    out[ZK_CS_DIFFICULTY] = float(c.difficulty);
    out[ZK_CS_TEAM] = float(c.team);
}

// After a tick: the deaths it saw, then the gadget's GameFrame for each frame it went through.
void World::run_chickens() {
    chicken_deaths();
    for (int f = frame + 1; f <= frame + ZK_TICK_FRAMES && !chickens.victory; ++f) chicken_frame(f);
}

void World::chicken_frame(int f) {
    ChickenGame& c = chickens;
    const zk_chicken_rules& r = *c.rules;
    const double t = f / FPS;
    if (double(f) > c.wave_schedule) {
        c.wave_schedule = double(f) + FPS * r.chicken_spawn_rate;
        chicken_wave(t);
    }
    if ((f + SECOND_PHASE) % SECOND_FRAMES == 0) chicken_second(t, f);
    if ((f + SLOW_PHASE) % SLOW_FRAMES == 0) chicken_slow(t);
    if (f == c.morph_frame) morph_queen(f);
}

// UnitDestroyed, for the units that died this tick.
void World::chicken_deaths() {
    ChickenGame& c = chickens;
    const zk_chicken_rules& r = *c.rules;
    const double now = (frame + ZK_TICK_FRAMES) / FPS;
    for (; c.fallen_seen < fallen.size(); ++c.fallen_seen) {
        const int u = fallen[c.fallen_seen];
        if (c.victory) continue;
        if (u == c.queen) {
            // KillAllComputerUnits: the players have won, and the chickens' alliance is destroyed.
            c.queen = -1;
            c.victory = true;
            kill_all_chickens();
            continue;
        }
        if (team[size_t(u)] != c.team) continue;
        if (type[size_t(u)] == r.burrow) burrow_killed(u, now);
        if (leaves_egg(r, type[size_t(u)])) lay_egg(type[size_t(u)], x[size_t(u)], z[size_t(u)], now + r.egg_decay_time);
    }
}

void World::kill_all_chickens() {
    for (size_t k = 0; k < live.size(); ++k)
        if (const int u = live[k]; alive[size_t(u)] && team[size_t(u)] == chickens.team) died(u);
    chickens.burrows.clear();
}

void World::burrow_killed(int u, double now) {
    ChickenGame& c = chickens;
    const zk_chicken_rules& r = *c.rules;
    c.burrows.erase(std::remove(c.burrows.begin(), c.burrows.end(), u), c.burrows.end());
    ++c.burrows_killed;
    // The queen comes sooner by how angry the players have made the chickens, and they're angrier for it.
    const double aggro = std::min(std::max(c.aggro, double(r.human_aggro_queen_time_min)), double(r.human_aggro_queen_time_max));
    const double reduction = std::max(double(r.burrow_queen_time) * r.human_aggro_queen_time_factor * aggro, 0.0);
    c.queen_time = std::max(c.queen_time - reduction, 1.0);
    const double before = c.aggro;
    c.aggro = std::min(c.aggro + c.aggro_per_burrow, double(r.human_aggro_max));
    c.aggro_delta += c.aggro - before;
    c.tech -= c.burrow_regress_time;
    c.defense_pool += r.defense_per_burrow_kill;
    // SpawnEggs: eggs of the kinds that have come out by now, around where it stood.  These don't decay.
    int choices[ZK_CHICKEN_KINDS], n = 0;
    for (int k = 0; k < r.n_kinds; ++k)
        if ((r.kinds[k].role == ZK_CHICKEN_OFFENSE || r.kinds[k].role == ZK_CHICKEN_DEFENSE) && r.kinds[k].time <= now) choices[n++] = r.kinds[k].type;
    if (n > 0)
        for (int i = 0; i < int(r.burrow_eggs); ++i) {
            const int choice = choices[rng.between(1, n) - 1];
            const int rx = rng.between(-EGG_SCATTER, EGG_SCATTER), rz = rng.between(-EGG_SCATTER, EGG_SCATTER);
            lay_egg(choice, x[size_t(u)] + float(rx), z[size_t(u)] + float(rz), -1.0);
        }
    if (rng.unit() < r.burrow_respawn_chance) spawn_burrow();
}

// A negative decay for one that stays.
void World::lay_egg(int t, float px, float pz, double decay) {
    if (cat->eggs.empty()) return;
    const zk_egg_rec& e = cat->eggs[size_t(t)];
    if (e.feature < 0) return;
    px = std::clamp(px, 0.0f, map->width() - 1.0f);
    pz = std::clamp(pz, 0.0f, map->depth() - 1.0f);
    const int id = add_wreck(px, pz, e.metal, e.hp, e.radius, e.feature, -1);
    if (decay >= 0.0) chickens.eggs.push_back({id, decay});
}

int World::add_chicken(int t, float px, float pz) {
    return add(t, chickens.team, std::clamp(px, 0.0f, map->width() - 1.0f), std::clamp(pz, 0.0f, map->depth() - 1.0f), NORTH, 1.0f, 1.0f);
}

void World::chicken_second(double t, int f) {
    ChickenGame& c = chickens;
    const zk_chicken_rules& r = *c.rules;
    const int burrows = int(c.burrows.size());
    if (c.burrow_spawn_rate * BURROW_SPAWN_SHARE * (burrows + 1) < t - c.time_of_last_spawn && burrows < r.max_burrows) {
        spawn_burrow();
        c.time_of_last_spawn = t;
    }
    if (t >= c.queen_time && !c.endgame) {
        c.queen = spawn_queen(f);
        c.endgame = true;
    }
    if (c.mini_queens_sent < r.n_mini_queen_times && t >= r.mini_queen_time[c.mini_queens_sent] * c.queen_time) {
        for (int i = 0; i < c.players; ++i) spawn_mini_queen();
        ++c.mini_queens_sent;
    }
}

void World::chicken_slow(double t) {
    ChickenGame& c = chickens;
    const zk_chicken_rules& r = *c.rules;
    // KillOldChicken: what came in a wave dies of old age, as anything dies, and leaves its egg.
    size_t kept = 0;
    for (const ChickenGame::Birth& b : c.births) {
        if (!alive[size_t(b.unit)]) continue;
        if (t - b.second > r.max_age + rng.between(1, OLD_AGE_SPREAD)) hp[size_t(b.unit)] = 0.0f;
        else c.births[kept++] = b;
    }
    c.births.resize(kept);
    // DecayEggs.  One reclaimed or destroyed first is gone already.
    kept = 0;
    for (const ChickenGame::Egg& e : c.eggs) {
        if (t < e.decay) c.eggs[kept++] = e;
        else if (wreck_exists(e.wreck)) remove_wreck(e.wreck);
    }
    c.eggs.resize(kept);
    // targetCache, where idle chickens are sent.  The queen walks there rather than fighting her way, above Beginner.
    c.have_target = choose_target(-1, c.target_x, c.target_z);
    if (!c.have_target) return;
    for (size_t k = 0; k < live.size(); ++k) {
        const int u = live[k];
        if (!alive[size_t(u)] || team[size_t(u)] != c.team || order[size_t(u)] != ZK_ORDER_NONE || !(type_of(u).flags & ZK_U_MOBILE)) continue;
        order_unit(u, r.level > 1 && u == c.queen ? ZK_ORDER_MOVE : ZK_ORDER_FIGHT, c.target_x, c.target_z, -1);
    }
}

void World::chicken_wave(double t) {
    ChickenGame& c = chickens;
    const zk_chicken_rules& r = *c.rules;
    --c.special_cooldown;
    special_powers(t);
    ++c.waves;
    const int burrows = int(c.burrows.size());
    // The chickens' tech goes on with the players' count, back with their anger and on while they're calm.
    const double decel = c.aggro > 0.0 ? c.aggro * r.human_aggro_tech_time_regress : c.aggro * r.human_aggro_tech_time_progress;
    c.tech = c.tech - decel + r.tech_accel_per_player * (c.players - 1);
    int other = -1;
    const int first = choose_kind(ZK_CHICKEN_OFFENSE, true, t, &other);
    const int turret = choose_kind(ZK_CHICKEN_DEFENSE, false, t, nullptr);
    const int support = choose_kind(ZK_CHICKEN_SUPPORT, true, t, nullptr);
    double squad = (t * r.time_spawn_bonus + r.wave_size_mult) *
                   (r.base_wave_size + std::min(std::max(c.aggro * r.human_aggro_wave_factor, 0.0), double(r.human_aggro_wave_max)) +
                    burrows * double(r.burrow_wave_size)) / std::max(burrows, 1);
    if (t < r.ramp_up_time) squad = squad * t / r.ramp_up_time;
    if (first < 0) return;      // no kind out by the chickens' tech time: the gadget would stop here with an error
    const zk_chicken_kind &one = r.kinds[first], &two = r.kinds[other];
    const int n1 = int(std::ceil(r.wave_ratio * squad * one.squad_size));
    const int n2 = int(std::floor((1.0 - r.wave_ratio) * squad * two.squad_size));
    if (c.queen >= 0) {
        spawn_chickens(c.queen, int(n1 * r.queen_spawn_mult), one.type, t);
        spawn_chickens(c.queen, int(n2 * r.queen_spawn_mult), two.type, t);
    }
    double delta = c.aggro * c.aggro_defense_factor;
    if (turret >= 0) delta += c.defense_per_wave;
    c.defense_pool += std::max(delta, 0.0);
    c.defense_quota = c.defense_pool / double(burrows);
    const bool defend = turret >= 0 && r.kinds[turret].cost < c.defense_quota;
    const std::vector<int> each = c.burrows;
    for (const int b : each) {
        spawn_chickens(b, n1, one.type, t);
        spawn_chickens(b, n2, two.type, t);
        if (defend && !c.endgame) spawn_turret(b, r.kinds[turret].type, 0.0, false, t);
        spawn_support(b, support >= 0 ? r.kinds[support].type : -1, t);
    }
    c.aggro = std::min(std::max(c.aggro - r.human_aggro_decay, double(r.human_aggro_min)), double(r.human_aggro_max));
    c.aggro_delta = 0.0;
}

// ChooseChicken: a kind of the role that's come and not gone by the chickens' tech time, picked twice (the gadget always draws both).
// Every kind's chance is 1: no kind sets initialChance or timeChanceMult.
int World::choose_kind(int role, bool use_tech, double t, int* other) {
    const zk_chicken_rules& r = *chickens.rules;
    int choices[ZK_CHICKEN_KINDS], n = 0;
    double tech = t + (use_tech ? chickens.tech : 0.0);
    tech = std::min(std::max(tech, t * r.tech_time_floor_factor), double(r.tech_time_max));
    for (int k = 0; k < r.n_kinds; ++k) {
        const zk_chicken_kind& kind = r.kinds[k];
        if (kind.role == role && tech >= kind.time && !(kind.obsolete < tech)) choices[n++] = k;
    }
    if (n == 0) return -1;
    const int a = choices[rng.between(1, n) - 1], b = choices[rng.between(1, n) - 1];
    if (other) *other = b;
    return a;
}

void World::special_powers(double t) {
    ChickenGame& c = chickens;
    const zk_chicken_rules& r = *c.rules;
    if (c.special_cooldown > 0) return;
    // The last of the list that's due, its time come and the players calm enough.
    const zk_chicken_kind* chosen = nullptr;
    for (int k = r.n_kinds - 1; k >= 0 && !chosen; --k) {
        const zk_chicken_kind& s = r.kinds[k];
        if (s.role == ZK_CHICKEN_SPECIAL && s.time < t && s.obsolete > t && s.max_aggro > c.aggro) chosen = &s;
    }
    if (!chosen) return;
    const double count = chosen->count >= 0.0f ? chosen->count : chosen->burrow_ratio >= 0.0f ? chosen->burrow_ratio * double(c.burrows.size()) : 1.0;
    const int n = int(std::ceil(count));
    if (chosen->tie_to_burrow) {
        if (!c.burrows.empty()) spawn_turret(c.burrows[size_t(rng.between(1, int(c.burrows.size())) - 1)], chosen->type, double(n), true, t);
    } else {
        float target[2];
        const bool aimed = chosen->target_human && choose_target(-1, target[0], target[1]);
        spawn_anywhere(chosen->type, n, chosen->min_dist, chosen->max_dist, aimed ? target : nullptr);
    }
    c.special_cooldown = int(chosen->cooldown);
}

// SpawnChicken: n of a kind round a burrow (or the queen), sent at a random unit of whoever owns what's nearest it.
void World::spawn_chickens(int burrow, int n, int t, double now) {
    ChickenGame& c = chickens;
    const zk_chicken_rules& r = *c.rules;
    const float bx = x[size_t(burrow)], bz = z[size_t(burrow)];
    float tx = c.target_x, tz = c.target_z;
    bool sent = c.have_target;
    const int near = nearest_enemy(burrow);
    if (near >= 0) sent = choose_target(near, tx, tz);
    float s = r.spawn_square;
    int tries = 0;
    for (int i = 0; i < n; ++i) {
        float px, pz;
        do {
            px = float(rng.between(int(std::floor(bx - s)), int(std::floor(bx + s))));
            pz = float(rng.between(int(std::floor(bz - s)), int(std::floor(bz + s))));
            s += r.spawn_square_increment;
            ++tries;
        } while (ground_blocked(t, px, pz) && tries <= n + r.max_tries_small);
        const int u = add_chicken(t, px, pz);
        order_unit(u, ZK_ORDER_MOVE_STATE, 0.0f, 0.0f, ROAM);
        if (sent) order_unit(u, ZK_ORDER_FIGHT, tx, tz, -1);
        c.births.push_back({u, now});
    }
}

// SpawnTurret: a defender at a burrow, from the defense pool, by chance unless forced.  A spire sprout forces it.
void World::spawn_turret(int burrow, int t, double number, bool force, double now) {
    ChickenGame& c = chickens;
    const zk_chicken_rules& r = *c.rules;
    (void)now;
    if (t < 0 || !alive[size_t(burrow)] || c.victory) return;
    const zk_chicken_kind* kind = find_kind(r, ZK_CHICKEN_DEFENSE, t);
    const double cost = kind ? kind->cost : 1.0;
    const double squad = (kind ? kind->squad_size : 1.0) * c.defense_quota / cost * rng.between(SQUAD_SPREAD_MIN, SQUAD_SPREAD_MAX) / 100.0;
    if (!force && rng.unit() > squad) return;
    const double spawned = number > 0.0 ? number : std::max(squad, 1.0);
    if (!force) c.defense_pool -= cost * spawned;
    const float bx = x[size_t(burrow)], bz = z[size_t(burrow)];
    float s = r.spawn_square;
    int tries = 0;
    for (int i = 1; i <= spawned; ++i) {     // Lua's for with a limit that needn't be whole
        float px, pz;
        do {
            px = float(rng.between(int(std::floor(bx - s)), int(std::floor(bx + s))));
            pz = float(rng.between(int(std::floor(bz - s)), int(std::floor(bz + s))));
            s += r.spawn_square_increment;
            ++tries;
        } while (ground_blocked(t, px, pz) && tries <= spawned + r.max_tries_small);
        const int u = add_chicken(t, px, pz);
        if (!(type_of(u).flags & ZK_U_MOBILE)) continue;
        const int near = nearest_enemy(burrow);
        float tx, tz;
        if (near >= 0 && choose_target(near, tx, tz)) order_unit(u, ZK_ORDER_FIGHT, tx, tz, -1);
    }
}

// SpawnSupport: by chance, with a wave.
void World::spawn_support(int burrow, int t, double now) {
    ChickenGame& c = chickens;
    const zk_chicken_rules& r = *c.rules;
    if (t < 0 || !alive[size_t(burrow)]) return;
    const zk_chicken_kind* kind = find_kind(r, ZK_CHICKEN_SUPPORT, t);
    double squad = (kind ? kind->squad_size : 1.0) * r.wave_size_mult * rng.between(SQUAD_SPREAD_MIN, SQUAD_SPREAD_MAX) / 100.0;
    if (now < r.ramp_up_time) squad = squad * now / r.ramp_up_time;
    if (rng.unit() > squad) return;
    const int n = std::max(int(std::floor(squad)), 1);
    const float bx = x[size_t(burrow)], bz = z[size_t(burrow)];
    float s = r.spawn_square;
    int tries = 0;
    for (int i = 0; i < n; ++i) {
        float px, pz;
        do {
            px = float(rng.between(int(std::floor(bx - s)), int(std::floor(bx + s))));
            pz = float(rng.between(int(std::floor(bz - s)), int(std::floor(bz + s))));
            s += r.spawn_square_increment;
            ++tries;
        } while (ground_blocked(t, px, pz) && tries <= n + r.max_tries_small);
        const int u = add_chicken(t, px, pz);
        const int near = nearest_enemy(burrow);
        float tx, tz;
        if (near >= 0 && choose_target(near, tx, tz)) order_unit(u, ZK_ORDER_FIGHT, tx, tz, -1);
    }
}

// SpawnUnit: a special power's units, anywhere or near a player's, preferring ground between the distances.  The gadget's test
// stops at the first ground that isn't blocked, so the distances seldom matter.
void World::spawn_anywhere(int t, int number, float min_dist, float max_dist, const float* target) {
    ChickenGame& c = chickens;
    const zk_chicken_rules& r = *c.rules;
    if (c.victory) return;
    const float lo = min_dist >= 0.0f ? min_dist : r.min_base_distance, hi = max_dist >= 0.0f ? max_dist : r.max_base_distance;
    const int square = int(r.spawn_square);
    float px, pz;
    int tries = 0;
    bool kept_off;
    do {
        if (!target) {
            px = float(rng.between(square, int(map->width()) - square));
            pz = float(rng.between(square, int(map->depth()) - square));
        } else {
            px = float(rng.between(int(std::floor(target[0] - hi)), int(std::floor(target[0] + hi))));
            pz = float(rng.between(int(std::floor(target[1] - hi)), int(std::floor(target[1] + hi))));
        }
        ++tries;
        kept_off = player_within(px, pz, lo) || !player_within(px, pz, hi);
    } while (ground_blocked(t, px, pz) && kept_off && tries <= number + r.max_tries);
    for (int i = 0; i < number; ++i) {
        const int dx = rng.between(-square, square), dz = rng.between(-square, square);
        order_unit(add_chicken(t, px + float(dx), pz + float(dz)), ZK_ORDER_MOVE_STATE, 0.0f, 0.0f, ROAM);
    }
}

void World::spawn_burrow() {
    ChickenGame& c = chickens;
    const zk_chicken_rules& r = *c.rules;
    if (c.victory || c.endgame) return;
    // Anywhere it could be built that's neither close to a player's unit nor far from all of them; after so many tries, the last.
    const int square = int(r.spawn_square);
    float px, pz;
    int tries = 0;
    bool free;
    do {
        px = float(rng.between(square, int(map->width()) - square));
        pz = float(rng.between(square, int(map->depth()) - square));
        ++tries;
        free = can_place(r.test_building, px, pz) &&
               !(player_within(px, pz, r.min_base_distance) || !player_within(px, pz, r.max_base_distance));
    } while (!free && tries <= r.max_tries_small);
    c.burrows.push_back(add_chicken(r.burrow, px, pz));
}

// SpawnQueen and SpawnMiniQueen place alike: anywhere one could stand that's not close to a player's unit.
bool World::place_queen(float& px, float& pz) {
    const zk_chicken_rules& r = *chickens.rules;
    int tries = 0;
    bool free;
    do {
        px = float(rng.between(0, int(map->width())));
        pz = float(rng.between(0, int(map->depth())));
        ++tries;
        free = can_place(r.test_building_queen, px, pz) && !player_within(px, pz, r.min_base_distance);
    } while (!free && tries <= r.max_tries);
    return free;
}

int World::spawn_queen(int f) {
    ChickenGame& c = chickens;
    const zk_chicken_rules& r = *c.rules;
    float px, pz;
    place_queen(px, pz);
    // The sooner she comes, the less health she has; and the more players, the more.
    c.queen_health_mod *= (0.5 * (c.queen_time / c.base_queen_time) + 0.5) * (c.players / 2.0 + 0.5);
    if (r.queen_morph >= 0) c.morph_frame = f + rng.between(int(r.queen_morph_frames[0]), int(r.queen_morph_frames[1]));
    const int q = add_chicken(r.queen, px, pz);
    damage_mult[size_t(q)] = float(1.0 / c.queen_health_mod);
    return q;
}

void World::spawn_mini_queen() {
    float px, pz;
    place_queen(px, pz);
    const int m = add_chicken(chickens.rules->mini_queen, px, pz);
    const int near = nearest_enemy(m);
    float tx, tz;
    if (near >= 0 && choose_target(near, tx, tz)) order_unit(m, ZK_ORDER_MOVE, tx, tz, -1);
}

// MorphQueen: the other form in her place, with her share of health and what she was told.  Not while she's stunned or disarmed.
void World::morph_queen(int f) {
    ChickenGame& c = chickens;
    const zk_chicken_rules& r = *c.rules;
    const int q = c.queen;
    if (q < 0 || !alive[size_t(q)]) return;
    if (stunned(q) || disarmed(q)) {
        c.morph_frame += MORPH_POSTPONED;
        return;
    }
    const size_t i = size_t(q);
    const float share = hp[i] / type_of(q).hp, px = x[i], pz = z[i], head = heading[i], para_level = para[i];
    const int kind = order[i], target = order_target[i];
    const float gx = goal_x[i], gz = goal_z[i];
    died(q);        // DestroyUnit(..., reclaimed): no explosion, nothing left
    const int next = add(c.morphed ? r.queen : r.queen_morph, c.team, px, pz, head, 1.0f, 1.0f);
    c.morphed = !c.morphed;
    c.morph_frame = f + rng.between(int(r.queen_morph_frames[0]), int(r.queen_morph_frames[1]));
    damage_mult[size_t(next)] = float(1.0 / c.queen_health_mod);
    hp[size_t(next)] = share * type_of(next).hp;
    para[size_t(next)] = para_level;
    if (kind != ZK_ORDER_NONE) order_unit(next, kind, gx, gz, target);
    c.queen = next;
}

// ChooseTarget: a random unit of the team that owns `unit`, or of a random player's team with one if there's no unit; not one of
// what chickens aren't sent at, if there's a choice.
bool World::choose_target(int unit, float& tx, float& tz) {
    ChickenGame& c = chickens;
    const zk_chicken_rules& r = *c.rules;
    std::vector<int>& units = c.pick;
    units.clear();
    const auto collect = [&](int owner) {
        for (const int u : live)
            if (alive[size_t(u)] && team[size_t(u)] == owner) units.push_back(u);
    };
    if (unit >= 0) {
        collect(team[size_t(unit)]);
    } else {
        if (c.players <= 0) return false;
        for (int tries = 0; units.empty() && tries < TARGET_TRIES; ++tries) {
            int pick = rng.between(1, c.players);
            for (int t = 0; t < n_teams; ++t)
                if (t != c.team && --pick == 0) {
                    collect(t);
                    break;
                }
        }
    }
    if (units.empty()) return false;
    const auto unwanted = [&](int u) {
        for (int k = 0; k < ZK_CHICKEN_NO_TARGET && r.no_target[k] >= 0; ++k)
            if (type[size_t(u)] == r.no_target[k]) return true;
        return false;
    };
    int chosen = units[0];
    if (units.size() > 1) {
        int tries = 0;
        do {
            chosen = units[size_t(rng.between(1, int(units.size())) - 1)];
            ++tries;
        } while (unwanted(chosen) && tries < TARGET_TRIES);
    }
    tx = x[size_t(chosen)];
    tz = z[size_t(chosen)];
    return true;
}

// GetUnitNearestEnemy, without needing sight of it.
int World::nearest_enemy(int u) const {
    int best = -1;
    float best_d2 = NEAREST_ENEMY_RANGE * NEAREST_ENEMY_RANGE;
    for (const int o : live) {
        if (!alive[size_t(o)] || side[size_t(o)] == side[size_t(u)]) continue;
        const float dx = x[size_t(o)] - x[size_t(u)], dz = z[size_t(o)] - z[size_t(u)], d2 = dx * dx + dz * dz;
        if (d2 < best_d2) best = o, best_d2 = d2;
    }
    return best;
}

bool World::player_within(float px, float pz, float r) const {
    for (const int o : live) {
        if (!alive[size_t(o)] || team[size_t(o)] == chickens.team) continue;
        const float dx = x[size_t(o)] - px, dz = z[size_t(o)] - pz;
        if (dx * dx + dz * dz <= r * r) return true;
    }
    return false;
}

// GetGroundBlocked, as ground this type can't stand on or a building stands on.
bool World::ground_blocked(int t, float px, float pz) const {
    if (px < 0.0f || pz < 0.0f || px >= map->width() || pz >= map->depth()) return true;
    const auto& d = cat->types[size_t(t)];
    if ((d.flags & ZK_U_MOBILE) && !(d.flags & ZK_U_AIR) && d.move_class >= 0 && map->speed_at(d.move_class, px, pz) <= 0.0f) return true;
    bool blocked = false;
    for_near(px, pz, cat_max_footprint + SEARCH_PAD, [&](int o) {
        if (blocked || !alive[size_t(o)] || (type_of(o).flags & ZK_U_MOBILE)) return;
        const float r = type_of(o).footprint_radius, dx = x[size_t(o)] - px, dz = z[size_t(o)] - pz;
        blocked = dx * dx + dz * dz < r * r;
    });
    return blocked;
}

// TestBuildOrder: for what moves, ground it can stand on; for a building, dry land a kbot could walk, with nothing on it.
bool World::can_place(int t, float px, float pz) const {
    if (px < 0.0f || pz < 0.0f || px >= map->width() || pz >= map->depth()) return false;
    const auto& d = cat->types[size_t(t)];
    if (d.flags & ZK_U_MOBILE) return d.move_class < 0 || map->speed_at(d.move_class, px, pz) > 0.0f;
    if (map->height_at(px, pz) < 0.0f || map->speed_at(KBOT_CLASS, px, pz) <= 0.0f) return false;
    bool clear = true;
    for_near(px, pz, d.footprint_radius + cat_max_footprint + SEARCH_PAD, [&](int o) {
        if (!clear || !alive[size_t(o)]) return;
        const float reach = d.footprint_radius + type_of(o).footprint_radius, dx = x[size_t(o)] - px, dz = z[size_t(o)] - pz;
        clear = dx * dx + dz * dz >= reach * reach;
    });
    return clear;
}

}  // namespace zk
