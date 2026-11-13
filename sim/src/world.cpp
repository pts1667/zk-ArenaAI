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
    : cat(c), map(m), rng(seed), behaviour(behaviour_), turn_salt(uint32_t((seed * 0x9E3779B97F4A7C15ull) >> 32)), ring(RING_SLOTS) {
    grid_cols = std::max(1, int(std::ceil(m->width() / GRID_CELL)));
    grid_rows = std::max(1, int(std::ceil(m->depth() / GRID_CELL)));
    cell_start.resize(size_t(n_allies) * grid_cells() + 1);
    for (const auto& t : c->types) cat_max_footprint = std::max(cat_max_footprint, t.footprint_radius);
}

void World::set_teams(int n, const int32_t* ally_of_team) {
    n_teams = std::clamp(n, 1, ZK_MAX_TEAMS);
    n_allies = 0;
    std::fill(std::begin(ally_size), std::end(ally_size), 0);
    for (int t = 0; t < ZK_MAX_TEAMS; ++t) {
        ally_of[t] = t < n_teams && ally_of_team ? std::clamp(int(ally_of_team[t]), 0, ZK_MAX_TEAMS - 1) : t;
        if (t >= n_teams) continue;
        ++ally_size[ally_of[t]];
        n_allies = std::max(n_allies, ally_of[t] + 1);
    }
    sharing = false;
    for (int s = 0; s < n_allies; ++s) sharing = sharing || ally_size[s] > 1;
    cell_start.assign(size_t(n_allies) * grid_cells() + 1, 0);
    // Anything already here goes with its team's new allyteam.
    for (int u = 0; u < count(); ++u) side[size_t(u)] = ally_of[team[size_t(u)]];
}

int World::add(int t, int owner, float px, float pz, float head, float hp_frac, float build_frac) {
    const auto& d = cat->types[size_t(t)];
    const int u = count();
    type.push_back(t);
    side.push_back(ally_of[owner]);
    team.push_back(owner);
    order.push_back(ZK_ORDER_NONE);
    order_target.push_back(-1);
    nearest.push_back(-1);
    alive.push_back(1);
    live.push_back(u);
    grid_slot.push_back(-1);
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
        // unit_tactical_ai.lua draws each unit's sides at random.
        a.jink_side = rng.between(0, 1) ? 1.0f : -1.0f;
        a.strafe_side = rng.between(0, 1) ? 1.0f : -1.0f;
        a.tactics_off = (cat->tactics[size_t(t)].flags & ZK_T_STARTS_OFF) != 0;
        ai.push_back(a);
    }
    incoming.emplace_back();
    for (int s = 0; s < n_allies; ++s) views[s].updated_frame = -1;
    goal_x.push_back(px);
    goal_z.push_back(pz);
    order_wait.push_back(0);
    order_lag.push_back(0);
    progress.push_back(WalkProgress{px, pz, 0, 0, -1, 0});
    mem_kind.push_back(-1);
    mem_frame.push_back(0);
    mem_target.push_back(-1);
    mem_x.push_back(0.0f);
    mem_z.push_back(0.0f);
    burn_rate.push_back(0.0f);
    burn_frames.push_back(0.0f);
    burn_team.push_back(-1);
    damage_mult.push_back(1.0f);
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
    skid.emplace_back();
    {
        // An aircraft starts on the ground, as one the engine makes does, and takes off when it has somewhere to be.
        AirState a;
        a.phase = AIR_LANDED;
        a.ammo = d.ammo;
        air.push_back(a);
        if (d.flags & ZK_U_AIR) aircraft.push_back(u);
        if (d.pad_count > 0) pads.push_back(u);
    }
    low_power.push_back(d.needed_link > 0.0f ? 1 : 0);     // until the first look at the grids
    pulling.push_back(0);
    wait_since.push_back(-1);
    carrier.push_back(-1);
    cargo.push_back(-1);
    drop_ready.push_back(0);
    held_for_lift.push_back(-1);
    if (d.lift_light > 0.0f || d.lift_medium > 0.0f || d.lift_heavy > 0.0f) transports.push_back(u);
    if (d.throw_radius > 0.0f) throwers.push_back(u);
    if (d.teleport_rate > 0.0f) teleporters.push_back(Teleporter{u});
    if (d.drop_weapon >= 0) droppers.push_back(u);
    if (d.area_cloak_radius > 0.0f) area_cloakers.push_back(u);
    if (d.energy_upkeep > 0.0f || d.area_cloak_upkeep > 0.0f) upkeepers.push_back(u);
    if (d.pylon_range > 0.0f || d.needed_link > 0.0f) pylons.push_back(u);
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

int World::order_group(const zk_group_rec& g) {
    if (g.anchor < 0 || g.anchor >= count() || !alive[size_t(g.anchor)]) return 0;
    const float r2 = g.radius * g.radius, cx = x[size_t(g.anchor)], cz = z[size_t(g.anchor)];
    // A player's box selects only its own units, never an ally's.
    const int owner = team[size_t(g.anchor)];
    int n = 0;
    for (size_t k = 0; k < live.size(); ++k) {
        const int u = live[k];
        if (!alive[size_t(u)]) continue;
        if (u != g.anchor) {
            if (!alive[size_t(u)] || team[size_t(u)] != owner || build[size_t(u)] < 1.0f || !(type_of(u).flags & ZK_U_MOBILE)) continue;
            const size_t t = size_t(type[size_t(u)]);
            const int role = t < role_of_type.size() ? role_of_type[t] : -1;
            if (role < 0 || role >= 32 || !((g.roles >> role) & 1u)) continue;
            const float dx = x[size_t(u)] - cx, dz = z[size_t(u)] - cz;
            if (dx * dx + dz * dz > r2) continue;
        }
        order_unit(u, g.kind, g.x, g.z, g.target);
        ++n;
    }
    return n;
}

void World::order_unit(int u, int kind, float gx, float gz, int target) {
    if (kind == ZK_ORDER_MANUAL_FIRE && manual_mount(u) < 0) return;
    const auto& d = type_of(u);
    if (kind == ZK_ORDER_PRODUCE) {
        if (d.flags & ZK_U_FACTORY) {
            produces[size_t(u)] = target;
            // As the net sees it: what it's making, since when.  The frame turns the type into a def id.
            mem_kind[size_t(u)] = kind;
            mem_frame[size_t(u)] = frame;
            mem_target[size_t(u)] = target;
            mem_x[size_t(u)] = x[size_t(u)];
            mem_z[size_t(u)] = z[size_t(u)];
        }
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
    if (kind == ZK_ORDER_ON_OFF) {
        pulling[size_t(u)] = target == 0;
        return;
    }
    if (kind == ZK_ORDER_DROP) {
        drop_cargo(u);
        return;
    }
    // A transport with room, and something it can lift that nothing carries; or one with a load, and a place that load can stand.
    if (kind == ZK_ORDER_LOAD && (target < 0 || target >= count() || !alive[size_t(target)] || cargo[size_t(u)] >= 0 || carrier[size_t(target)] >= 0 || !can_lift(u, target))) return;
    if (kind == ZK_ORDER_UNLOAD && (cargo[size_t(u)] < 0 || !can_land_at(cargo[size_t(u)], gx, gz))) return;
    if (kind == ZK_ORDER_PLACE_BEACON && d.teleport_rate <= 0.0f) return;
    if (kind == ZK_ORDER_WAIT_AT_BEACON) {
        // Either the beacon or who it belongs to.  Nothing that stands still can go, nor anything to another allyteam's.
        const Teleporter* via = target >= 0 && target < count() ? teleporter_of(target) : nullptr;
        if (via) target = via->beacon;
        bool beacon = false;
        for (const Teleporter& t : teleporters) beacon = beacon || (target >= 0 && t.beacon == target);
        if (!beacon || !alive[size_t(target)] || side[size_t(target)] != side[size_t(u)] || !(d.flags & ZK_U_MOBILE)) return;
        if (order[size_t(u)] != kind || order_target[size_t(u)] != target) wait_since[size_t(u)] = frame;   // told again, it keeps its place
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
    // An ally's frame or factory can be helped, as in the game, and no enemy's.
    if (kind == ZK_ORDER_ASSIST && target >= 0 && target < count() && side[size_t(target)] != side[size_t(u)]) return;
    if (kind == ZK_ORDER_RECLAIM && (d.reclaim_speed <= 0.0f || (d.flags & ZK_U_FACTORY))) return;
    if (!(d.flags & ZK_U_FACTORY)) work_on[size_t(u)] = -1;
    // Thrown, it's only falling, and does what it was told once it's down.
    jump[size_t(u)].stranded = jump[size_t(u)].busy && jump[size_t(u)].thrown_by < 0 && kind != ZK_ORDER_JUMP;
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
    // The net has kinds for what it can tell a unit to do, and these aren't among them: it sees a unit going somewhere.
    const bool goes = kind == ZK_ORDER_PLACE_BEACON || kind == ZK_ORDER_WAIT_AT_BEACON || kind == ZK_ORDER_LOAD || kind == ZK_ORDER_UNLOAD;
    mem_kind[size_t(u)] = goes ? int(ZK_ORDER_MOVE) : kind;
    mem_frame[size_t(u)] = frame;
    // A build's target is a type, an assist's is one of ours and a reclaim's is a feature: none is what the net means by an aim.
    mem_target[size_t(u)] = kind == ZK_ORDER_BUILD || kind == ZK_ORDER_ASSIST || kind == ZK_ORDER_RECLAIM || kind == ZK_ORDER_WAIT_AT_BEACON || kind == ZK_ORDER_LOAD ? -1 : target;
    // What it is to load is where it goes.
    mem_x[size_t(u)] = kind == ZK_ORDER_LOAD ? x[size_t(target)] : gx;
    mem_z[size_t(u)] = kind == ZK_ORDER_LOAD ? z[size_t(target)] : gz;
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
    { ZK_PROF(GRID); drop_the_dead(); rebuild_grid(); }
    if (tick_count % 3 == 0) {  // the engine's 15-frame weapon update
        { ZK_PROF(CLOAK); update_cloak(); }
        if (behaviour == ZK_BEHAVIOUR_ADVANCE) { ZK_PROF(NEAREST); find_nearest(); }
        if (behaviour == ZK_BEHAVIOUR_ORDERS) {
            // Targeting goes by what an allyteam knows, so fog runs for every world with orders.
            ensure_views();
            for (int s = 0; s < n_allies; ++s) update_view(s);
            ZK_PROF(UNIT_AI);
            update_unit_ai();
        }
        if (behaviour == ZK_BEHAVIOUR_ORDERS || behaviour == ZK_BEHAVIOUR_SCRIPTED) { ZK_PROF(KAMIKAZE); update_kamikaze(); }
        { ZK_PROF(TARGETS); pick_targets(); }
    }
    if (behaviour == ZK_BEHAVIOUR_ORDERS) { ZK_PROF(UNIT_AI); update_tactics(); }
    if (behaviour == ZK_BEHAVIOUR_ORDERS || behaviour == ZK_BEHAVIOUR_SCRIPTED) { ZK_PROF(KAMIKAZE); update_droppers(); }
    { ZK_PROF(FIRE); fire_weapons(); }
    { ZK_PROF(SHOTS); resolve_shots(); }
    { ZK_PROF(BURN); burn_areas(); }
    {
        ZK_PROF(MOVE);
        {
            ZK_PROF(MOVE_ABILITIES);
            update_throwers();
            update_teleporters();
            update_jumps();
            update_air();
            update_transports();
        }
        move_units();
    }
    // After the units: each projectile's path this tick is tried against where they went in it.
    { ZK_PROF(SHOTS); fly_projectiles(); }
    { ZK_PROF(DEATHS); finish_deaths(); }
    // After deaths: a build step adds health, and mustn't bring back what this tick killed.
    if (economy_on) { ZK_PROF(ECONOMY); run_economy(); }
    { ZK_PROF(STATUS); update_status(); update_shields(); }
    // The gadgets, for the frames this tick has gone through.
    if (chickens.rules) {
        ZK_PROF(CHICKENS);
        run_chickens();
    } else if (reveal_on && !revealed && cat->reveal_every > 0) {
        for (int f = frame + 1; f <= frame + ZK_TICK_FRAMES; ++f)
            if (f % cat->reveal_every == 0) check_reveal();
    }
    check_vital();
    frame += ZK_TICK_FRAMES;
    ++tick_count;
}

// game_over.lua's ProcessLastAlly: with two alliances alive, one worth more than so many times the other has the other's units always
// visible from then on, cloaked or not (modrules alwaysVisibleOverridesCloaked).  The gadget also destroys what it has outside the
// map, where nothing can be here.  It doesn't look again once it has revealed.  Alive and worth: finished units that count, by metal.
void World::check_reveal() {
    double value[ZK_MAX_TEAMS] = {};
    int units[ZK_MAX_TEAMS] = {};
    for (const int u : live) {
        const size_t i = size_t(u);
        const auto& d = type_of(u);
        if (!alive[i] || build[i] < 1.0f || (d.flags & ZK_U_DONT_COUNT)) continue;
        value[side[i]] += double(d.metal_cost);
        ++units[side[i]];
    }
    int alliance[2] = {-1, -1}, n = 0;
    for (int s = 0; s < n_allies; ++s)
        if (units[s] > 0 && n++ < 2) alliance[n - 1] = s;
    if (n != 2) return;
    const double ratio = double(cat->reveal_ratio), one = value[alliance[0]], other = value[alliance[1]];
    const int supreme = one > ratio * other ? alliance[0] : other > ratio * one ? alliance[1] : -1;
    if (supreme < 0) return;
    for (int s = 0; s < n_allies; ++s)
        if (s != supreme) revealed |= 1u << s;
}

// game_over.lua's resignation of AI teams: an alliance whose teams are all AIs, as every team in the sim stands for, is destroyed when
// it has no finished builder or factory left that counts (ZK_U_VITAL; sim/data/rules.json "vital").  So it's out (stats[3] 0) and
// the game goes on without it, or ends.  The gadget looks as each unit that counts goes, and blows the alliance's units up over the
// next 50 frames; here they die at the next tick.  Not the chickens.  An alliance that has never had such a unit, an army set down
// in a test or a scenario, isn't looked at: in a game every team starts with its commander.
void World::check_vital() {
    uint32_t has = 0;
    for (const int u : live) {
        const size_t i = size_t(u);
        const uint32_t f = type_of(u).flags;
        if (alive[i] && build[i] >= 1.0f && (f & ZK_U_VITAL) && !(f & ZK_U_DONT_COUNT)) has |= 1u << side[i];
    }
    had_vital |= has;
    uint32_t gone = had_vital & ~has;
    if (chickens.rules && chickens.ally >= 0) gone &= ~(1u << chickens.ally);
    if (!gone) return;
    for (const int u : live)
        if (alive[size_t(u)] && ((gone >> side[size_t(u)]) & 1u)) hp[size_t(u)] = 0.0f;
}

void World::died(int u) {
    alive[size_t(u)] = 0;
    fallen.push_back(u);
    // A dead transport lets its load fall from where it was, as its script does when it crashes (ForceDropUnit); and a dead
    // load leaves its transport empty.
    if (cargo[size_t(u)] >= 0) let_go(u, true);
    if (carrier[size_t(u)] >= 0) {
        cargo[size_t(carrier[size_t(u)])] = -1;
        carrier[size_t(u)] = -1;
    }
}

// The lists by kind lose their dead, keeping their order, when someone has died since they last did.  Not the area cloakers, whose
// being there at all makes update_cloak look at everyone, nor teleporters or silos' missiles, which keep more than a handle.  Shields
// keep links by place in their list, and lose their dead only as the links are made again (update_shields).
void World::drop_the_dead() {
    if (fallen_pruned == fallen.size()) return;
    fallen_pruned = fallen.size();
    const auto dead = [this](int u) { return !alive[size_t(u)]; };
    for (std::vector<int>* list : {&builders, &earners, &upkeepers, &aircraft, &pads, &personal_cloakers, &burrowers, &jumpers, &throwers, &droppers, &transports})
        list->erase(std::remove_if(list->begin(), list->end(), dead), list->end());
}

void World::rebuild_grid() {
    const size_t cells = grid_cells();
    std::fill(cell_start.begin(), cell_start.end(), 0);
    unit_key.resize(size_t(count()));
    // The living go on in handle order, and the dead leave the list for good.
    size_t kept = 0, on_grid = 0;
    for (size_t k = 0; k < live.size(); ++k) {
        const int u = live[k];
        if (!alive[size_t(u)]) continue;
        live[kept++] = u;
        // What a transport carries is off the grid: nothing that looks around a place finds it (transport.cpp).
        if (carrier[size_t(u)] >= 0) {
            unit_key[size_t(u)] = -1;
            grid_slot[size_t(u)] = -1;
            continue;
        }
        ++on_grid;
        int gx = int(x[size_t(u)] / GRID_CELL), gz = int(z[size_t(u)] / GRID_CELL);
        gx = gx < 0 ? 0 : (gx >= grid_cols ? grid_cols - 1 : gx);
        gz = gz < 0 ? 0 : (gz >= grid_rows ? grid_rows - 1 : gz);
        const int key = int(size_t(side[size_t(u)]) * cells + size_t(gz) * size_t(grid_cols) + size_t(gx));
        unit_key[size_t(u)] = key;
        ++cell_start[size_t(key)];
    }
    live.resize(kept);
    // Each key's end, then filled backwards so that each key's begin is left behind with its units in handle order.  The sum is
    // kept in a register: through memory, each add would wait on the store before it.
    int sum = 0;
    for (int& c : cell_start) c = sum += c;
    cell_units.resize(on_grid);
    for (size_t k = kept; k-- > 0;) {
        const int u = live[k];
        if (unit_key[size_t(u)] < 0) continue;
        const int slot = --cell_start[size_t(unit_key[size_t(u)])];
        cell_units[size_t(slot)] = u;
        grid_slot[size_t(u)] = slot;
    }
}

void World::find_nearest() {
    for (size_t k = 0, n = live.size(); k < n; ++k) {
        const int u = live[k];
        nearest[size_t(u)] = -1;
        if (!alive[size_t(u)] || !(type_of(u).flags & ZK_U_MOBILE)) continue;
        // Fights are small and the grid coarse: widen until something turns up.
        float best = 1e18f;
        for (float r : NEAREST_RADII) {
            for_enemies(side[size_t(u)], x[size_t(u)], z[size_t(u)], r, [&](int e) {
                if (!can_be_aimed_at(side[size_t(u)], e)) return;
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
    const float frames = float(ZK_TICK_FRAMES);
    for (size_t k = 0, n = live.size(); k < n; ++k) {
        const int u = live[k];
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
            Purse& purse = purses[size_t(team[size_t(u)])];
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
            const float amount = burn_rate[size_t(u)] * std::min(frames, burn_frames[size_t(u)]) * damage_mult[size_t(u)];
            count_damage(u, amount, burn_team[size_t(u)]);
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
