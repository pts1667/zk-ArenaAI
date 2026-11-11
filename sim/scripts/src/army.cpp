// The army script (tools/policy_army.py ArmyScript), its production (tools/policy_mix.py Mix) and the aimer
// (tools/policy_eco.py Aimer).  Each function follows the Python one of the same name; the comments say only where the C++
// had to choose how to do what Python does implicitly.
#include <algorithm>
#include <climits>
#include <cmath>
#include <tuple>

#include "scripts.hpp"

namespace zks {

namespace {

constexpr int FIRST_STRAIGHT = 3;
constexpr double PATROL_SHARE = 0.25;
constexpr int PATROL_MIN = 2;
constexpr double PATROL_MIN_FROM = 90.0;
constexpr double PATROL_REACH = 900.0;
constexpr double PATROL_OFFSET = 150.0;
constexpr int SQUAD_SIZE = 3;
constexpr double COMMIT = 1.0;
constexpr double DEFENCE_MARGIN = 100.0;
constexpr double THREAT_REACH = 500.0;
constexpr double OUTRANGE_MARGIN = 50.0;
constexpr double OUTRANGE_WEIGHT = 4.0;
constexpr double HEAVY_RIOT_REACH = 800.0;
constexpr double PROBE = 300.0;
constexpr double PROBE_STEP = 100.0;
constexpr double TRAVEL_SPEED = 20.0;
constexpr double GIVE_UP_SLACK = 60.0;
constexpr double PATIENCE_AT_TARGET = 300.0;
constexpr double GIVE_UP_FOR = 300.0;
constexpr double NEAR_TARGET = 300.0;
constexpr double MERGE_REACH = 900.0;
constexpr double REORDER = 200.0;
constexpr double AT_GOAL = 150.0;
constexpr double BEHIND = 150.0;
// An assault that hasn't moved STILL_REACH elmos in STILL_FOR seconds doesn't lead: riots once held for 30 minutes behind two
// stuck Thugs.
constexpr double STILL_REACH = 100.0;
constexpr double STILL_FOR = 60.0;
constexpr double KITE_RANGE = 0.6;
constexpr double KITE_STEP = 250.0;
constexpr double ARTY_STAND = 0.75;
constexpr double SCOUT_LEG = 250.0;
constexpr double FAR = 1e5;
// policy_script.py
constexpr double RETREAT_BELOW = 0.4;
constexpr double RECOVERED = 0.9;
constexpr double ARRIVED = 500.0;
constexpr double GROUPS_FROM = 180.0;
constexpr double GROUP_FIRST_METAL = 400.0;
constexpr double GROUP_METAL_PER_MINUTE = 200.0;
constexpr double GROUP_MOST_METAL = 3000.0;
constexpr double LONGEST_WAIT = 180.0;
// featurize.py: the PRODUCE commit lock
constexpr float COMMIT_FRAMES = 6 * 15;
// policy_mix.py
const double CONSTRUCTORS[7][2] = {{0.0, 0.0}, {1.0, 1.0}, {3.0, 2.0}, {5.0, 3.0}, {7.0, 4.0}, {10.0, 5.0}, {14.0, 6.0}};
constexpr double INCOME_PER_CONSTRUCTOR = 3.0;
constexpr int MOST_CONSTRUCTORS = 20;
constexpr int FLOAT_CONSTRUCTORS = 2;   // more while the economy script finds metal floating: build power is what's short
const double RAID_X[5] = {0.0, 4.0, 10.0, 20.0, 60.0}, RAID_Y[5] = {1.0, 1.0, 0.5, 0.4, 0.35};
const double RIOT_X[4] = {0.0, 4.0, 10.0, 60.0}, RIOT_Y[4] = {0.0, 0.0, 0.3, 0.3};
const double ASSAULT_X[4] = {0.0, 6.0, 12.0, 60.0}, ASSAULT_Y[4] = {0.0, 0.0, 0.25, 0.25};
const double ARTY_X[4] = {0.0, 12.0, 18.0, 60.0}, ARTY_Y[4] = {0.0, 0.0, 0.1, 0.1};
constexpr double SKIRM_SHARE = 0.25, AA_SHARE = 0.12, SNIPE_SHARE = 0.10, SNIPE_FROM = 480.0;
// An air plant's mix, of what it has: a plane plant's bombers (Ravens, Phoenixes), a gunship plant's raiders, assaults and
// skirmishers; AA once the enemy has been seen with air.  It comes late, so there's no schedule.
constexpr double AIR_RAID = 0.35, AIR_RIOT = 0.2, AIR_ASSAULT = 0.25, AIR_SKIRM = 0.2, AIR_BOMB = 0.4, AIR_AA = 0.3;
// Chicken defense (unit_spawner.lua, with spawn_defs.lua's numbers).  A burrow killed brings the queen sooner by
// burrowQueenTime (15 s) times the chickens' anger (humanAggro, taken between 0 and 8), and angers them by 1 / players; each
// wave calms them by humanAggroDecay (0.25).  So a kill while the anger is at most 0 costs no time, and the alliance's kills
// then settle at what the waves calm, players / 4 a wave.  Every burrow left alive makes each wave bigger (burrowWaveSize
// squads), but each kill past that brings the queen sooner by more than the one before, and her coming is the game's end one
// way or the other; and below 0 the chickens' kinds come sooner each wave, by humanAggroTechTimeProgress (20 s) times the
// calm.  So the army hunts burrows only while a kill is free: first of all while the anger is below 0, else when no chicken is
// near our buildings, as it goes for those first.  With nothing to do it holds at its rally.  But the chickens' tech outgrows
// the scripts' economy before her full time, so from QUEEN_CALL the army hunts whatever the anger, to bring her on while
// it's strongest against them, and each kill makes the waves smaller too.  Once she's come it goes for her; plants make AA for
// her (she flies and lands by turns) from QUEEN_PREP before she's due, not sooner: AA is no use against the waves.
constexpr double HUNT_ANGER = 0.0;      // the army hunts a burrow while the anger is at most this
constexpr double QUEEN_CALL = 1800.0;   // game seconds
constexpr double DEFEND_REACH = 800.0;  // a chicken this near one of our buildings is one the army goes for
constexpr double QUEEN_PREP = 300.0;    // seconds before the queen is due
constexpr double QUEEN_AA = 0.35;       // AA's share of a plant's mix from then, on top of any it has
// Transports (Seat::lifts)
constexpr int MOST_TRANSPORTS = 4;          // of one kind
constexpr int LOADS_A_TRANSPORT = 2;        // a plant makes one for every two units of ours it could carry
constexpr double DROP_LOSS = 0.15;          // of its health: a load the fall costs at most this is dropped, not set down
constexpr double DROP_NEAR = 200.0;         // elmos from the target: inside it a transport comes down to set its load down
constexpr double SET_DOWN_SHORT = 120.0;    // a load is set down this far short of the target
constexpr double WING_WAIT = 45.0;          // seconds the loaded wait for those still fetching
constexpr double LANDED_REACH = 900.0;      // come down this near the target, a load goes for it
constexpr double NEXT_REACH = 1000.0;       // ... and then for the buildings this near it, one after another
constexpr double NOT_IN_A_FIGHT = 600.0;    // a unit with an armed enemy this near isn't fetched
// policy_eco.py Aimer
constexpr double FIRE_AGAIN = 60.0;
constexpr float NUKE_VALUE_REACH = 960.0f;

double group_metal(double seconds) {
    if (seconds < GROUPS_FROM) return 0.0;
    return std::min(GROUP_MOST_METAL, GROUP_FIRST_METAL + GROUP_METAL_PER_MINUTE * (seconds - GROUPS_FROM) / 60.0);
}

int scout_quota(double seconds) { return (seconds >= 120.0) + (seconds >= 600.0); }

int constructor_quota(double seconds, double metal_income, bool floating) {
    double by_time = 0.0;
    bool any = false;
    for (auto& p : CONSTRUCTORS)
        if (seconds >= p[0] * 60.0 && (!any || p[1] > by_time)) by_time = p[1], any = true;
    int64_t by_income = (int64_t)(metal_income / INCOME_PER_CONSTRUCTOR);
    int64_t want = std::max((int64_t)by_time, by_income) + (floating ? FLOAT_CONSTRUCTORS : 0);
    return (int)std::min((int64_t)MOST_CONSTRUCTORS, want);
}

bool scout_or_raid(int kind) { return kind == R_RAID || kind == R_SCOUT; }

zk_order_rec order(int64_t uid, int kind, double x, double z, int64_t target) {
    zk_order_rec o;
    o.unit = (int32_t)uid;
    o.kind = kind;
    o.x = (float)x;
    o.z = (float)z;
    o.target = (int32_t)target;
    return o;
}

}  // namespace

zks_style standard_style() {
    zks_style s{};
    for (double& w : s.role_weight) w = 1.0;
    s.commit = COMMIT, s.gather = 1.0, s.patrol_share = PATROL_SHARE, s.patience = 1.0, s.retreat_below = RETREAT_BELOW;
    return s;
}

void Seat::observe() {
    const Data& d = *data;
    for (size_t e = 0; e < enemies.size(); e++) {
        int r = enemy_rows[e];
        if (d.has(r, ZKS_MOBILE) && d.has(r, ZKS_AIR)) seen_air = true;
        if (d.has(r, ZKS_KITABLE)) seen_kitable = true;
        if (d.has(r, ZKS_SLOW_HEAVY)) seen_heavy = true;
    }
}

// _births, then _forget
void Seat::births_and_forget() {
    const Data& d = *data;
    for (size_t i = 0; i < own.size(); i++) {
        int r = own_rows[i];
        if (!d.has(r, ZKS_FIGHTS) || d.has(r, ZKS_TRANSPORT)) continue;     // a transport isn't in the mix (transport_due)
        int64_t uid = own[i]->uid;
        if (!born[uid]) {
            born[uid] = 1;
            made[d.role[r]]++;
            if (d.has(r, ZKS_AIR)) made_air[d.role[r]]++;
        }
    }
    // Per-uid memory of what's gone.  `stamp` was set for every own uid by Seat::run.
    for (size_t u = 0; u < state.size(); u++) {
        if (stamp[u] == call) continue;
        state[u] = S_NEW;
        ordered_kind[u] = 0;
        way[u] = -1;
        squad_of[u] = -1;
        current[u] = -1;
        patrol[u] = 0;
        still_since[u] = NAN;
        lift_load[u] = -1;
        lifted[u] = 0;
        drop_goal[u] = -1;
    }
    std::vector<uint8_t> used(squad_live.size(), 0);
    for (size_t u = 0; u < squad_of.size(); u++)
        if (squad_of[u] >= 0) used[squad_of[u]] = 1;
    for (size_t s = 0; s < squad_live.size(); s++)
        if (squad_live[s] && !used[s]) squad_live[s] = 0, since_has[s] = 0;
}

int Seat::choose(int factory_row, double seconds, int constructors, double metal_income, bool queen_due, int transport) const {
    const Data& d = *data;
    int builder = d.menu_of(factory_row, R_BUILDER);
    if (builder >= 0 && constructors < constructor_quota(seconds, metal_income, floating)) return builder;
    if (transport >= 0) return transport;
    int scout = d.menu_of(factory_row, R_SCOUT);
    if (scout >= 0 && made[d.role_ids[R_SCOUT]] < scout_quota(seconds)) return scout;
    // shares(), in its dict's order: the schedule's roles, then those added once seen
    double minutes = seconds / 60.0;
    int roles[R_COUNT];
    double weights[R_COUNT];
    int n = 0;
    auto add = [&](int r, double w) {
        if (r < ZKS_ROLE_KNOBS) w *= style.role_weight[r];
        if (w > 0.0 && d.menu_of(factory_row, r) >= 0) roles[n] = r, weights[n] = w, n++;
    };
    if (d.air_plant[factory_row]) {
        add(R_RAID, AIR_RAID);
        add(R_RIOT, AIR_RIOT);
        add(R_ASSAULT, AIR_ASSAULT);
        add(R_SKIRM, AIR_SKIRM);
        add(R_BOMB, AIR_BOMB);
        if (seen_air || queen_due) add(R_AA, (seen_air ? AA_SHARE + AIR_AA : 0.0) + (queen_due ? QUEEN_AA : 0.0));
    } else {
    add(R_RAID, interp(minutes, RAID_X, RAID_Y, 5));
    add(R_RIOT, interp(minutes, RIOT_X, RIOT_Y, 4));
    add(R_ASSAULT, interp(minutes, ASSAULT_X, ASSAULT_Y, 4));
    add(R_ARTY, interp(minutes, ARTY_X, ARTY_Y, 4));
    if (seen_kitable) add(R_SKIRM, SKIRM_SHARE);
    if (seen_air || queen_due) add(R_AA, (seen_air ? AA_SHARE : 0.0) + (queen_due ? QUEEN_AA : 0.0));
    if (seen_heavy && seconds >= SNIPE_FROM) add(R_SNIPE, SNIPE_SHARE);
    }
    if (!n) return -1;
    PySum sum;     // sum(share.values())
    for (int k = 0; k < n; k++) sum.add(weights[k]);
    double total = sum.value();
    // An air plant's mix counts what flies, and a ground plant's what doesn't: each keeps its own shares.
    const bool air = d.air_plant[factory_row] != 0;
    auto count = [&](int r) -> int64_t { return air ? made_air[d.role_ids[r]] : made[d.role_ids[r]] - made_air[d.role_ids[r]]; };
    int64_t started = 0;
    for (int r = R_RAID; r <= R_SNIPE; r++)
        if (d.role_ids[r] >= 0) started += count(r);
    if (air && d.role_ids[R_BOMB] >= 0) started += count(R_BOMB);
    int best = -1;
    double best_behind = 0.0;
    for (int k = 0; k < n; k++) {
        int64_t have = count(roles[k]);
        double behind = weights[k] / total * (double)(started + 1) - (double)have;
        if (best < 0 || behind > best_behind || (behind == best_behind && roles[k] < roles[best])) best = k, best_behind = behind;
    }
    return d.menu_of(factory_row, roles[best]);
}

void Seat::produce(const Ctx& c, double metal_income) {
    const Data& d = *data;
    int constructors = 0;
    for (int r : own_rows) constructors += d.has(r, ZKS_CONSTRUCTOR);
    const bool queen_due = c.chickens.known && (c.chickens.queen || c.chickens.until_queen <= QUEEN_PREP);
    auto held = [&](const Rec* u) {
        return u->ord_kind == K_PRODUCE && u->ord_age < COMMIT_FRAMES && (u->flags & (zk::UF_DAMAGED | zk::UF_IDLE)) == 0;
    };
    for (size_t k = 0; k < own.size(); k++) {
        const Rec* u = own[k];
        int r = own_rows[k];
        if ((u->flags & zk::UF_BEING_BUILT) || !d.has(r, ZKS_FACTORY)) continue;
        int frame_on = u->ord_kind == K_PRODUCE ? d.row_of(u->ord_target) : -1;
        int on = current[u->uid] >= 0 ? current[u->uid] : (frame_on >= 0 && frame_on != d.unknown ? frame_on : -1);
        int due = choose(r, c.seconds, constructors, metal_income, queen_due, transport_due(r));
        if (due < 0 || due == on || held(u)) {
            if (on >= 0) current[u->uid] = on;
            continue;
        }
        current[u->uid] = due;
        army_out.push_back(order(u->uid, K_PRODUCE, 0.0, 0.0, d.catalog_row[due]));
    }
    for (size_t i = 0; i < own.size(); i++) {
        const Rec* u = own[i];
        int r = own_rows[i];
        if ((u->flags & zk::UF_BEING_BUILT) || d.has(r, ZKS_FACTORY)) continue;
        int due = d.best_missile[r];
        if (due < 0 || current[u->uid] == due || held(u)) continue;
        current[u->uid] = due;
        army_out.push_back(order(u->uid, K_PRODUCE, 0.0, 0.0, d.catalog_row[due]));
    }
}

void Seat::make_view() {
    const Data& d = *data;
    View& v = view;
    size_t n = enemies.size();
    v.uid.resize(n);
    v.x.resize(n);
    v.z.resize(n);
    v.cost.resize(n);
    v.reach.resize(n);
    v.building.resize(n);
    v.defence.resize(n);
    v.threat.resize(n);
    v.heavy_riot.resize(n);
    v.slow_heavy.resize(n);
    v.extractor.resize(n);
    for (size_t e = 0; e < n; e++) {
        const Rec* u = enemies[e];
        int r = enemy_rows[e];
        bool known = r != d.unknown, mobile = d.has(r, ZKS_MOBILE), defence = d.has(r, ZKS_DEFENCE);
        v.uid[e] = u->uid;
        v.x[e] = u->x;
        v.z[e] = u->z;
        v.cost[e] = d.cost[r];
        v.reach[e] = d.range[r];
        v.building[e] = known && !mobile && !defence;
        v.defence[e] = known && defence;
        v.threat[e] = known && mobile && d.has(r, ZKS_ARMED);
        v.heavy_riot[e] = d.has(r, ZKS_HEAVY_RIOT);
        v.slow_heavy[e] = d.has(r, ZKS_SLOW_HEAVY);
        v.extractor[e] = d.has(r, ZKS_EXTRACTOR);
        if (u->uid >= 0 && u->uid < (int64_t)view_stamp.size()) view_stamp[u->uid] = call, view_index[u->uid] = (int32_t)e;
    }
}

void Seat::army(const Ctx& c) {
    const Data& d = *data;
    double metal_income = c.glob[zk::G_METAL_INCOME];
    observe();
    births_and_forget();
    produce(c, metal_income);
    make_view();
    lifts(c);
    std::vector<Fighter> fighters;
    std::vector<P2> extractors;
    for (size_t i = 0; i < own.size(); i++) {
        const Rec* u = own[i];
        int r = own_rows[i];
        if (u->flags & zk::UF_BEING_BUILT) continue;
        // Not the wing, nor what it carries or has brought somewhere: those are Seat::lifts'.
        const bool lift = d.has(r, ZKS_TRANSPORT) || (u->flags & zk::UF_CARRIED) || drop_goal[u->uid] >= 0;
        if (d.has(r, ZKS_FIGHTS) && !lift)
            fighters.push_back({u->uid, d.kind[r], r, (double)u->x, (double)u->z, (double)u->hp_frac, (double)d.cost[r],
                                (double)d.range[r], (u->flags & zk::UF_IDLE) != 0});
        if (d.has(r, ZKS_EXTRACTOR)) extractors.push_back({(double)u->x, (double)u->z});
    }
    if (!fighters.empty()) fight(c, fighters, extractors);
    if (aimer) aim(c);
}

// A unit of ours a transport might carry for the army: a finished assault or riot that walks.
bool Seat::load_for_lift(size_t i) const {
    const Data& d = *data;
    const int r = own_rows[i];
    return (d.kind[r] == R_ASSAULT || d.kind[r] == R_RIOT) && d.has(r, ZKS_FIGHTS) && !d.has(r, ZKS_AIR) && d.load_class[r] > 0 &&
           !(own[i]->flags & zk::UF_BEING_BUILT);
}

// The transport this plant should make now, -1 for none: one for every LOADS_A_TRANSPORT units of ours that it could carry
// and no cheaper transport of the plant's could, up to MOST_TRANSPORTS of a kind, counting those being built.  So a gunship
// plant makes Charons for Jacks and Ogres, and a Hercules only for what a Charon can't lift.  None against the chickens.
int Seat::transport_due(int factory_row) const {
    const Data& d = *data;
    if (!d.n_transports || game.chickens() >= 0) return -1;
    std::vector<int> kinds;
    for (int k = d.option_offsets[factory_row]; k < d.option_offsets[factory_row + 1]; k++)
        if (d.has(d.options[k], ZKS_TRANSPORT)) kinds.push_back(d.options[k]);
    std::sort(kinds.begin(), kinds.end(), [&](int a, int b) { return d.cost[a] < d.cost[b] || (d.cost[a] == d.cost[b] && a < b); });
    for (size_t k = 0; k < kinds.size(); k++) {
        int loads = 0, have = 0;
        for (size_t i = 0; i < own.size(); i++) {
            have += own_rows[i] == kinds[k];
            if (!load_for_lift(i) || !d.lifts(kinds[k], own_rows[i])) continue;
            bool cheaper = false;
            for (size_t j = 0; j < k; j++) cheaper |= d.lifts(kinds[j], own_rows[i]);
            loads += !cheaper;
        }
        if (have < std::min(MOST_TRANSPORTS, (loads + LOADS_A_TRANSPORT - 1) / LOADS_A_TRANSPORT)) return kinds[k];
    }
    return -1;
}

// One unit's order by `_emit`'s rule: not again while it's on its way to the same.  Whether it was sent.
bool Seat::tell(const Rec* u, int kind, double x, double z, int64_t target) {
    const int64_t uid = u->uid;
    if (ordered_kind[uid] == kind && py_hypot(ordered_x[uid] - x, ordered_z[uid] - z) <= REORDER) {
        if (!(u->flags & zk::UF_IDLE) || py_hypot(u->x - x, u->z - z) <= AT_GOAL) return false;
    }
    ordered_kind[uid] = (int8_t)kind;
    ordered_x[uid] = x;
    ordered_z[uid] = z;
    army_out.push_back(order(uid, kind, x, z, target));
    return true;
}

// Where a load of this move class can be set down for `at` by a transport coming from `from`: SET_DOWN_SHORT short of it, or
// else the first of a ring of points about it, on ground the class can stand on.  false for nowhere.
bool Seat::landing(int move_class, P2 at, P2 from, P2& out) const {
    const Ground& g = *ground;
    auto stands = [&](P2 p) {
        if (p.x < 0.0 || p.z < 0.0 || p.x >= g.map_w || p.z >= g.map_h) return false;
        if (move_class < 0 || move_class >= ZKS_MOVE_CLASSES || g.labels[move_class].empty()) return true;
        const int i = std::min((int)(p.z / g.cell), g.rows - 1), j = std::min((int)(p.x / g.cell), g.cols - 1);
        return g.labels[move_class][(size_t)i * g.cols + j] >= 0;
    };
    const P2 first = along(at, from, SET_DOWN_SHORT);
    if (stands(first)) {
        out = first;
        return true;
    }
    for (int k = 0; k < 8; k++) {
        const double a = k * (3.14159265358979 / 4.0);
        const P2 p{at.x + SET_DOWN_SHORT * std::cos(a), at.z + SET_DOWN_SHORT * std::sin(a)};
        if (stands(p)) {
            out = p;
            return true;
        }
    }
    return false;
}

// Transports (the sim's are sim/src/transport.cpp: Charon and Hercules).  A plant that can make them makes a few once the side
// has assaults or riots to carry (transport_due).  Those it has are a wing.  Each free one fetches one of those units, the
// dearest first, by the cheapest transport that lifts it.  The wing goes for the enemy building worth most (as the aimer
// counts worth) among those whose cover the metal it carries can take, as a raiding squad's target is taken, and brings its
// loads there: dropped in flight (Drop Cargo) when the fall costs the load at most DROP_LOSS of its health, as it does a Jack
// or a Dante, and set down beside the target otherwise, as a Reaver or an Ogre is.  A load is dropped at the last look
// before its transport would be within DROP_NEAR of the target, so it lands short of it by up to a look's flight less its
// fall's.  The loaded wait up to WING_WAIT for those still fetching, so that they come together.  A unit brought there goes for the target, then for the buildings near it, and
// is the army's again when none is left.  With nothing the wing could take it fetches nothing, and what's aboard is set down
// at the rally.
//
// Left out: only the target's cover is weighed, not what could shoot the wing down on its way, which is a straight line; it
// carries no ally's unit and picks up no enemy.
void Seat::lifts(const Ctx& c) {
    const Data& d = *data;
    if (!d.n_transports) return;
    struct Wing {
        const Rec* u;
        int row;
        bool loaded;
    };
    std::vector<Wing> wing;
    for (size_t i = 0; i < own.size(); i++)
        if (d.has(own_rows[i], ZKS_TRANSPORT) && !(own[i]->flags & zk::UF_BEING_BUILT))
            wing.push_back({own[i], own_rows[i], (own[i]->flags & zk::UF_LOADED) != 0});
    int target = view_of(lift_target);
    // Who's up, and who has come down: near what the wing went for, it goes for that.
    for (size_t i = 0; i < own.size(); i++) {
        const Rec* u = own[i];
        const int64_t uid = u->uid;
        if (u->flags & zk::UF_CARRIED) {
            lifted[uid] = 1;
        } else if (lifted[uid]) {
            lifted[uid] = 0;
            for (const Wing& w : wing)
                if (lift_load[w.u->uid] == uid) lift_load[w.u->uid] = -1;
            if (target >= 0 && std::hypot(view.x[target] - u->x, view.z[target] - u->z) <= LANDED_REACH) {
                drop_goal[uid] = lift_target;
                ordered_kind[uid] = 0;
            }
        }
    }
    for (size_t i = 0; i < own.size(); i++) {
        const Rec* u = own[i];
        const int64_t uid = u->uid;
        if (drop_goal[uid] < 0 || (u->flags & zk::UF_CARRIED)) continue;
        int e = view_of(drop_goal[uid]);
        if (e < 0) {
            e = nearest(view.building, {u->x, u->z}, NEXT_REACH);
            ordered_kind[uid] = 0;
            if (e < 0) {
                drop_goal[uid] = -1;
                state[uid] = S_SENT;
                continue;
            }
            drop_goal[uid] = view.uid[e];
        }
        tell(u, K_ATTACK_UNIT, view.x[e], view.z[e], view.uid[e]);
    }
    if (wing.empty()) {
        lift_target = -1;
        lift_waiting = NAN;
        return;
    }
    // What each has, aboard or to fetch, as an index into `own`.
    std::vector<int> load(wing.size(), -1), fetch(wing.size(), -1);
    std::vector<uint8_t> taken(own.size(), 0);
    auto ours = [&](int64_t uid) { return uid >= 0 && uid < (int64_t)stamp.size() && stamp[uid] == call; };
    for (size_t k = 0; k < wing.size(); k++) {
        const Wing& w = wing[k];
        int32_t& l = lift_load[w.u->uid];
        if (l >= 0 && !ours(l)) l = -1;
        const bool aboard = l >= 0 && (own[own_at[l]]->flags & zk::UF_CARRIED);
        if (w.loaded && !aboard) {
            // Not what it was sent for: whatever of ours hangs where it is.
            l = -1;
            for (size_t i = 0; i < own.size() && l < 0; i++)
                if ((own[i]->flags & zk::UF_CARRIED) && own[i]->x == w.u->x && own[i]->z == w.u->z) l = (int32_t)own[i]->uid;
        } else if (!w.loaded && l >= 0 && (aboard || drop_goal[l] >= 0 || state[l] == S_RETREATING)) {
            l = -1;     // another's, brought already, or hurt
        }
        if (l >= 0) load[k] = own_at[l], taken[own_at[l]] = 1;
    }
    const bool guards = game.chickens() >= 0;       // against the chickens the army guards, and the wing fetches nothing
    if (!guards) {
        std::vector<int> wanted;
        for (size_t i = 0; i < own.size(); i++) {
            const int64_t uid = own[i]->uid;
            if (taken[i] || !load_for_lift(i) || (own[i]->flags & zk::UF_CARRIED) || drop_goal[uid] >= 0 || state[uid] == S_RETREATING) continue;
            if (nearest(view.threat, {own[i]->x, own[i]->z}, NOT_IN_A_FIGHT) >= 0) continue;
            wanted.push_back((int)i);
        }
        std::stable_sort(wanted.begin(), wanted.end(), [&](int a, int b) { return d.cost[own_rows[a]] > d.cost[own_rows[b]]; });
        for (int i : wanted) {
            int best = -1;
            double best_cost = 0.0, best_away = 0.0;
            for (size_t k = 0; k < wing.size(); k++) {
                if (wing[k].loaded || load[k] >= 0 || fetch[k] >= 0 || !d.lifts(wing[k].row, own_rows[i])) continue;
                const double cost = d.cost[wing[k].row], away = std::hypot(wing[k].u->x - own[i]->x, wing[k].u->z - own[i]->z);
                if (best < 0 || cost < best_cost || (cost == best_cost && away < best_away)) best = (int)k, best_cost = cost, best_away = away;
            }
            if (best >= 0) fetch[best] = i;
        }
    }
    // What the wing carries and would fetch, and what that can take.
    double metal = 0.0, reach = 0.0;
    for (size_t k = 0; k < wing.size(); k++) {
        const int i = load[k] >= 0 ? load[k] : fetch[k];
        if (i >= 0) metal += d.cost[own_rows[i]], reach = std::max(reach, (double)d.range[own_rows[i]]);
    }
    if (guards || metal <= 0.0) {
        lift_target = -1;
    } else if (!acceptable(lift_target, metal, reach)) {
        lift_target = -1;
        std::vector<int> at;
        std::vector<double> ax, az, cv;
        std::vector<uint8_t> hv;
        for (int e = 0; e < view.size(); e++)
            if (view.building[e]) at.push_back(e), ax.push_back(view.x[e]), az.push_back(view.z[e]);
        cover(ax.data(), az.data(), (int)at.size(), reach, true, cv, hv);
        int best = -1;
        double best_worth = 0.0, best_away = 0.0;
        for (size_t k = 0; k < at.size(); k++) {
            if (hv[k] || cv[k] > metal / style.commit) continue;
            const double worth = view.cost[at[k]] * (d.has(enemy_rows[at[k]], ZKS_VALUABLE) ? 4.0 : 1.0);
            const double away = std::hypot(ax[k] - rally.x, az[k] - rally.z);
            if (best < 0 || worth > best_worth || (worth == best_worth && away < best_away)) best = (int)k, best_worth = worth, best_away = away;
        }
        if (best >= 0) lift_target = view.uid[at[best]];
    }
    target = view_of(lift_target);
    // The loaded that haven't been sent wait for those still fetching, up to WING_WAIT.
    int held = 0, fetching = 0;
    for (size_t k = 0; k < wing.size(); k++) {
        if (wing[k].loaded) held += load[k] >= 0 && ordered_kind[wing[k].u->uid] == K_LOAD;
        else fetching += target >= 0 && (load[k] >= 0 || fetch[k] >= 0);
    }
    if (!(held && fetching)) lift_waiting = NAN;
    else if (std::isnan(lift_waiting)) lift_waiting = c.seconds;
    const bool go = !fetching || c.seconds - lift_waiting >= WING_WAIT;
    for (size_t k = 0; k < wing.size(); k++) {
        const Rec* u = wing[k].u;
        const int64_t uid = u->uid;
        const P2 at{u->x, u->z};
        if (!wing[k].loaded) {
            const int i = target >= 0 ? (load[k] >= 0 ? load[k] : fetch[k]) : -1;
            lift_load[uid] = i >= 0 ? (int32_t)own[i]->uid : -1;
            if (i < 0) {
                tell(u, K_MOVE, rally.x, rally.z, -1);      // nothing to fetch: home
            } else if (ordered_kind[uid] != K_LOAD || ordered_x[uid] != (double)own[i]->uid || (u->flags & zk::UF_IDLE)) {
                ordered_kind[uid] = (int8_t)K_LOAD;
                ordered_x[uid] = (double)own[i]->uid;
                ordered_z[uid] = 0.0;
                army_out.push_back(order(uid, K_LOAD, own[i]->x, own[i]->z, own[i]->uid));
            }
            continue;
        }
        if (load[k] < 0) {      // nothing of ours: let go of it
            army_out.push_back(order(uid, K_DROP, 0.0, 0.0, -1));
            continue;
        }
        const int row = own_rows[load[k]];
        P2 down;
        if (target < 0) {       // nowhere to take it: home, and down
            if (landing(d.move_class[row], rally, at, down)) tell(u, K_UNLOAD, down.x, down.z, -1);
            else tell(u, K_MOVE, rally.x, rally.z, -1);
            continue;
        }
        if (ordered_kind[uid] == K_LOAD && !go) continue;
        const P2 goal{view.x[target], view.z[target]};
        // It flies over and drops its load whatever the fall costs where the load can't be set down, or the sim wouldn't have
        // it (told to set down and idle), and once that's so it stays so.
        bool over = !landing(d.move_class[row], goal, at, down);
        over = over || (ordered_kind[uid] == K_UNLOAD && (u->flags & zk::UF_IDLE) && py_hypot(ordered_x[uid] - down.x, ordered_z[uid] - down.z) <= REORDER);
        over = over || (ordered_kind[uid] == K_MOVE && py_hypot(ordered_x[uid] - goal.x, ordered_z[uid] - goal.z) <= REORDER);
        const double ahead = std::hypot(u->vx, u->vz) * c.army_frames;      // how far it flies before the next look
        if (std::hypot(goal.x - at.x, goal.z - at.z) <= DROP_NEAR + ahead && (over || d.dropped(wing[k].row, row) <= DROP_LOSS)) {
            ordered_kind[uid] = (int8_t)K_DROP;
            army_out.push_back(order(uid, K_DROP, 0.0, 0.0, -1));
        } else if (over) {
            tell(u, K_MOVE, goal.x, goal.z, -1);
        } else {
            tell(u, K_UNLOAD, down.x, down.z, -1);
        }
    }
}

int Seat::nearest(const std::vector<uint8_t>& mask, P2 p, double below) const {
    int best = -1;
    double bd = 0.0;
    for (int e = 0; e < view.size(); e++) {
        if (!mask[e]) continue;
        double dd = std::hypot(view.x[e] - p.x, view.z[e] - p.z);
        if (best < 0 || dd < bd) best = e, bd = dd;
    }
    if (best >= 0 && !(bd < below)) return -1;
    return best;
}

void Seat::fight(const Ctx& c, std::vector<Fighter>& f, const std::vector<P2>& extractors) {
    const double seconds = c.seconds;
    int n = (int)f.size();
    std::vector<uint8_t> retreat(n);
    for (int k = 0; k < n; k++) {
        int64_t uid = f[k].uid;
        int back = (scout_or_raid(f[k].kind) || seconds < GROUPS_FROM) ? S_SENT : S_WAITING;
        int s = state[uid];
        if (s == S_NEW) s = back;
        if (data->has(f[k].row, ZKS_REGENERATES) && f[k].hp < style.retreat_below)
            s = S_RETREATING;
        else if (s == S_RETREATING && f[k].hp >= RECOVERED)
            s = back;
        state[uid] = (int8_t)s;
        retreat[k] = s == S_RETREATING;
    }
    // _release
    {
        PySum waiting;
        for (int k = 0; k < n; k++)
            if (state[f[k].uid] == S_WAITING && !scout_or_raid(f[k].kind)) waiting.add(f[k].cost);
        bool any = waiting.any;
        double sum = waiting.value();
        if (!any) {
            has_waiting_since = false;
        } else {
            if (!has_waiting_since) has_waiting_since = true, waiting_since = seconds;
            if (sum >= group_metal(seconds) * style.gather || seconds - waiting_since >= LONGEST_WAIT) {
                for (int k = 0; k < n; k++)
                    if (state[f[k].uid] == S_WAITING && !scout_or_raid(f[k].kind)) state[f[k].uid] = S_SENT;
                has_waiting_since = false;
            }
        }
    }
    std::vector<int> sent;
    for (int k = 0; k < n; k++)
        if (state[f[k].uid] == S_SENT) sent.push_back(k);
    // The army walks to the start of the nearest enemy still in the game (`foe`).  A team the scripts don't play (the chickens)
    // has no base there to walk to: against it the army guards (guard_goal), and the raiders go with it rather than raid.
    const bool walk = foe >= 0 && game.player(foe);
    const bool guard = !walk && game.chickens() >= 0;
    march = walk ? enemy_base : rally;
    P2 centre = rally;
    if (!sent.empty()) {
        // pos[sent].mean(axis=0): an axis-0 sum, which numpy does row after row
        double sx = 0.0, sz = 0.0;
        for (int k : sent) sx += f[k].x, sz += f[k].z;
        centre = {sx / (double)sent.size(), sz / (double)sent.size()};
        if (walk && !reached_start)
            for (int k : sent)
                if (std::hypot(f[k].x - enemy_base.x, f[k].z - enemy_base.z) < ARRIVED) reached_start = true;
    }
    // _goal: having been at an enemy's start, the army goes for the nearest building of any enemy, then the nearest armed unit.
    P2 goal = march;
    if (guard) {
        goal = guard_goal(c, centre);
    } else if (reached_start || !walk) {
        int i = nearest(view.building, centre);
        if (i < 0) i = nearest(view.threat, centre);
        if (i >= 0) goal = {view.x[i], view.z[i]};
    }
    // The rest hold behind the assaults that are on the move; with none, they go for the goal.
    for (int k = 0; k < n; k++) {
        const int64_t uid = f[k].uid;
        if (std::isnan(still_since[uid]) || std::hypot(f[k].x - still_x[uid], f[k].z - still_z[uid]) > STILL_REACH)
            still_x[uid] = f[k].x, still_z[uid] = f[k].z, still_since[uid] = seconds;
    }
    P2 lead = goal;
    {
        std::vector<int> assault;
        for (int k : sent)
            if (!scout_or_raid(f[k].kind) && f[k].kind == R_ASSAULT && seconds - still_since[f[k].uid] < STILL_FOR)
                assault.push_back(k);
        if (!assault.empty()) {
            double m = (double)assault.size();
            lead = {np_sum<double>([&](int64_t i) { return f[assault[i]].x; }, (int64_t)assault.size()) / m,
                    np_sum<double>([&](int64_t i) { return f[assault[i]].z; }, (int64_t)assault.size()) / m};
        }
    }
    std::vector<Plan> plans(n);
    for (int k = 0; k < n; k++) {
        Plan& p = plans[k];
        p.has = true;
        int kind = f[k].kind;
        if (kind == R_SCOUT) {
            int64_t uid = f[k].uid;
            int i = way[uid] >= 0 ? way[uid] : (int)(uid % 4);
            if (py_hypot(f[k].x - waypoints[i].x, f[k].z - waypoints[i].z) < SCOUT_LEG) i = (i + 1) % 4;
            way[uid] = (int8_t)i;
            p.kind = K_MOVE, p.x = waypoints[i].x, p.z = waypoints[i].z;
            continue;
        }
        if (kind == R_RAID && !guard) {
            p.has = false;
            continue;
        }
        // _army_order
        int s = state[f[k].uid];
        P2 pos{f[k].x, f[k].z};
        if (s == S_RETREATING) {
            p.kind = K_MOVE, p.x = rally.x, p.z = rally.z;
            continue;
        }
        if (s != S_SENT) {
            p.kind = K_FIGHT, p.x = rally.x, p.z = rally.z;
            continue;
        }
        P2 behind = along(lead, rally, std::min(BEHIND, py_hypot(rally.x - lead.x, rally.z - lead.z)));
        p.kind = K_FIGHT, p.x = behind.x, p.z = behind.z;
        if (kind == R_ASSAULT) {
            p.x = goal.x, p.z = goal.z;
        } else if (kind == R_AA) {
            const P2 at = guard && c.chickens.queen ? goal : centre;     // the queen is the goal: to her
            p.x = at.x, p.z = at.z;
        } else if (kind == R_SKIRM) {
            int near = nearest(view.threat, pos, KITE_RANGE * f[k].reach);
            if (near >= 0) {
                P2 away = along(pos, {view.x[near], view.z[near]}, -KITE_STEP);
                p.kind = K_MOVE, p.x = away.x, p.z = away.z;
            }
        } else if (kind == R_ARTY) {
            int b = nearest(view.building, lead);
            if (b >= 0) {
                P2 stand = along({view.x[b], view.z[b]}, rally, ARTY_STAND * f[k].reach);
                p.x = stand.x, p.z = stand.z;
            }
        } else if (kind == R_SNIPE) {
            int h = nearest(view.slow_heavy, pos);
            if (h >= 0) p.kind = K_ATTACK_UNIT, p.x = view.x[h], p.z = view.z[h], p.target = view.uid[h];
        } else if (kind == R_BOMB && data->has(f[k].row, ZKS_AIR)) {
            // A bomber goes for what stands in the army's way, a defence before any other building, and is back on its pad
            // after each run (the sim's unit_bomber_command.lua).
            int b = nearest(view.defence, lead);
            if (b < 0) b = nearest(view.building, lead);
            if (b >= 0) p.kind = K_ATTACK_UNIT, p.x = view.x[b], p.z = view.z[b], p.target = view.uid[b];
        }
    }
    std::vector<int> raiders;
    for (int k = 0; k < n; k++)
        if (f[k].kind == R_RAID) raiders.push_back(k);
    if (!guard) raids(c, f, raiders, retreat, extractors, plans);
    // _emit
    const size_t first = army_out.size();
    for (int k = 0; k < n; k++) {
        const Plan& p = plans[k];
        if (!p.has) continue;
        int64_t uid = f[k].uid;
        if (ordered_kind[uid] == p.kind && py_hypot(ordered_x[uid] - p.x, ordered_z[uid] - p.z) <= REORDER) {
            if (!f[k].idle || py_hypot(f[k].x - p.x, f[k].z - p.z) <= AT_GOAL) continue;
        }
        ordered_kind[uid] = (int8_t)p.kind;
        ordered_x[uid] = p.x;
        ordered_z[uid] = p.z;
        army_out.push_back(order(uid, p.kind, p.x, p.z, p.target));
    }
    group_plans(c, first);
}

// Where the army goes in a chicken game: the queen, wherever she is, as her death is the game; else, while the anger is below 0
// or from QUEEN_CALL, the burrow nearest our base (or what guards it); else the chicken nearest the army of those near our
// buildings; else, while a kill is free, that burrow; else the rally.  The army sees every chicken, as it sees every enemy.
P2 Seat::guard_goal(const Ctx& c, P2 centre) const {
    const Data& d = *data;
    int queen = -1;
    double qd = 0.0;
    for (int e = 0; e < view.size(); e++) {
        if (!d.has(enemy_rows[e], ZKS_QUEEN)) continue;
        const double dd = std::hypot(view.x[e] - centre.x, view.z[e] - centre.z);
        if (queen < 0 || dd < qd) queen = e, qd = dd;
    }
    if (queen >= 0) return {view.x[queen], view.z[queen]};
    const bool call = c.chickens.known && c.seconds >= QUEEN_CALL;
    const int burrow = c.chickens.anger <= HUNT_ANGER || call ? nearest(view.building, base) : -1;
    if (burrow >= 0 && (c.chickens.anger < HUNT_ANGER || call)) return {view.x[burrow], view.z[burrow]};
    std::vector<P2> mine;
    for (size_t i = 0; i < own.size(); i++)
        if (!d.has(own_rows[i], ZKS_MOBILE)) mine.push_back({(double)own[i]->x, (double)own[i]->z});
    const double reach2 = DEFEND_REACH * DEFEND_REACH;
    int best = -1;
    double bd = 0.0;
    for (int e = 0; e < view.size(); e++) {
        if (!view.threat[e]) continue;
        const double dd = std::hypot(view.x[e] - centre.x, view.z[e] - centre.z);
        if (best >= 0 && dd >= bd) continue;
        for (const P2& m : mine) {
            const double dx = view.x[e] - m.x, dz = view.z[e] - m.z;
            if (dx * dx + dz * dz <= reach2) {
                best = e, bd = dd;
                break;
            }
        }
    }
    if (best >= 0) return {view.x[best], view.z[best]};
    if (burrow >= 0) return {view.x[burrow], view.z[burrow]};
    return rally;
}

// Group orders (capi.h zk_group_rec).  The fighters' orders from army_out[first] on that are the same order, to the letter, and of a kind
// the policy groups (zkbot.policy.groups.GROUP_KINDS: move, fight, attack), go as one
// group order: from the one nearest their middle, the anchor, to the units of their roles within the smallest of the group radii
// (zks_data_desc.group_radii) that takes them all in, if that holds no other unit the sim could select, but for those already told
// this order: a move or fight the script would leave as it is (`_emit`: the same kind, within REORDER of it), which the group then
// tells again, to the letter.  If it holds another, or none is large enough, they're split in two across their wider spread and each
// half tried again; what's left goes alone.  So the sim does what the orders alone would, but that those already on their way there
// are told again (2026-11-08: a push of the whole army was some 20 orders alone, as most of it was on its way already): it applies a
// game's group orders before its unit orders, and any other order a fighter gets this step comes after its plan's.  Those it could
// select are a superset here, every own unit with a role, as it also asks that they be mobile and finished; and a member is within
// the radius by GROUP_MARGIN, anyone else outside it by as much, which float rounding can't undo.
void Seat::group_plans(const Ctx& c, size_t first) {
    constexpr double GROUP_MARGIN = 1.0;     // elmos
    const Data& d = *data;
    if (!c.groups || army_out.size() < first + 2) return;
    grouped.assign(army_out.size(), 0);
    anchor_of.assign(army_out.size(), -1);
    struct Mine { float x, z; int role; int64_t uid; };
    std::vector<Mine> mine;
    int64_t top = 0;
    for (const Rec* u : own) top = std::max<int64_t>(top, u->uid);
    for (size_t i = first; i < army_out.size(); i++) top = std::max<int64_t>(top, army_out[i].unit);
    mine_of.assign(size_t(top + 1), -1);
    member_of.assign(size_t(top + 1), 0);
    for (size_t i = 0; i < own.size(); i++) {
        int r = own_rows[i];
        int role = d.catalog_row[r] >= 0 ? d.role[r] : -1;
        if (role < 0 || role >= 32 || own[i]->uid < 0) continue;
        mine_of[size_t(own[i]->uid)] = (int32_t)mine.size();
        mine.push_back({own[i]->x, own[i]->z, role, own[i]->uid});
    }
    auto of = [&](size_t i) -> const Mine& { return mine[size_t(mine_of[size_t(army_out[i].unit)])]; };
    // The orders, those of one order together, in a fixed order.
    std::vector<size_t> at;
    for (size_t i = first; i < army_out.size(); i++)
        if (army_out[i].unit >= 0 && mine_of[size_t(army_out[i].unit)] >= 0 &&
            (army_out[i].kind == ZK_ORDER_MOVE || army_out[i].kind == ZK_ORDER_FIGHT || army_out[i].kind == ZK_ORDER_ATTACK))
            at.push_back(i);
    auto key = [&](size_t i) {
        const zk_order_rec& o = army_out[i];
        return std::make_tuple(o.kind, o.target, o.x, o.z);
    };
    std::stable_sort(at.begin(), at.end(), [&](size_t a, size_t b) { return key(a) < key(b); });
    int32_t stamp = 0;
    auto cover = [&](auto&& self, std::vector<size_t> m) -> void {
        if (m.size() < 2) return;
        uint32_t roles = 0;
        double sx = 0.0, sz = 0.0;
        for (size_t i : m) roles |= 1u << of(i).role, sx += of(i).x, sz += of(i).z;
        const double mx = sx / double(m.size()), mz = sz / double(m.size());
        size_t anchor = m[0];
        double best = INFINITY;
        for (size_t i : m) {
            double dd = std::hypot(double(of(i).x) - mx, double(of(i).z) - mz);
            if (dd < best) best = dd, anchor = i;
        }
        const double ax = of(anchor).x, az = of(anchor).z;
        double far = 0.0;
        for (size_t i : m) far = std::max(far, std::hypot(double(of(i).x) - ax, double(of(i).z) - az));
        double radius = -1.0;
        for (double r : d.group_radii)
            if (r >= far + GROUP_MARGIN) {
                radius = r;
                break;
            }
        bool clash = radius < 0.0;
        const zk_order_rec& o = army_out[anchor];
        auto told = [&](int64_t uid) {     // already on its way there, as `_emit` sees it
            return (o.kind == K_MOVE || o.kind == K_FIGHT) && size_t(uid) < ordered_kind.size() && ordered_kind[size_t(uid)] == o.kind &&
                   py_hypot(ordered_x[size_t(uid)] - o.x, ordered_z[size_t(uid)] - o.z) <= REORDER;
        };
        std::vector<int64_t> again;
        if (!clash) {
            ++stamp;
            for (size_t i : m) member_of[size_t(army_out[i].unit)] = stamp;
            for (const Mine& u : mine)
                if (((roles >> u.role) & 1u) && member_of[size_t(u.uid)] != stamp &&
                    std::hypot(double(u.x) - ax, double(u.z) - az) <= radius + GROUP_MARGIN) {
                    if (!told(u.uid)) {
                        clash = true;
                        break;
                    }
                    again.push_back(u.uid);
                }
        }
        if (!clash) {
            anchor_of[anchor] = (int32_t)groups.size();
            groups.push_back({o.unit, roles, float(radius), o.kind, o.x, o.z, o.target});
            for (size_t i : m) grouped[i] = 1;
            for (int64_t uid : again) ordered_x[size_t(uid)] = o.x, ordered_z[size_t(uid)] = o.z;
            return;
        }
        double lo_x = INFINITY, hi_x = -INFINITY, lo_z = INFINITY, hi_z = -INFINITY;
        for (size_t i : m) {
            lo_x = std::min(lo_x, double(of(i).x)), hi_x = std::max(hi_x, double(of(i).x));
            lo_z = std::min(lo_z, double(of(i).z)), hi_z = std::max(hi_z, double(of(i).z));
        }
        const bool by_x = hi_x - lo_x >= hi_z - lo_z;
        std::stable_sort(m.begin(), m.end(), [&](size_t a, size_t b) { return by_x ? of(a).x < of(b).x : of(a).z < of(b).z; });
        const size_t half = m.size() / 2;
        self(self, std::vector<size_t>(m.begin(), m.begin() + std::ptrdiff_t(half)));
        self(self, std::vector<size_t>(m.begin() + std::ptrdiff_t(half), m.end()));
    };
    for (size_t s = 0; s < at.size();) {
        size_t e = s + 1;
        while (e < at.size() && key(at[e]) == key(at[s])) e++;
        cover(cover, std::vector<size_t>(at.begin() + std::ptrdiff_t(s), at.begin() + std::ptrdiff_t(e)));
        s = e;
    }
}

// _cover.  The cover is a sum of whole costs, so any order of summing gives numpy's answer.
void Seat::cover(const double* px, const double* pz, int n, double reach, bool armies, std::vector<double>& out,
                 std::vector<uint8_t>& heavy) const {
    out.assign(n, 0.0);
    heavy.assign(n, 0);
    for (int p = 0; p < n; p++) {
        double sum = 0.0;
        bool h = false;
        for (int e = 0; e < view.size(); e++) {
            bool dfn = view.defence[e], thr = armies && view.threat[e], hr = view.heavy_riot[e];
            if (!dfn && !thr && !hr) continue;
            double dist = std::hypot(view.x[e] - px[p], view.z[e] - pz[p]);
            bool covered = (dfn && dist <= view.reach[e] + DEFENCE_MARGIN) || (thr && dist <= THREAT_REACH);
            if (covered) sum += view.cost[e] * (view.reach[e] > reach + OUTRANGE_MARGIN ? OUTRANGE_WEIGHT : 1.0);
            if (hr && dist <= HEAVY_RIOT_REACH) h = true;
        }
        out[p] = sum;
        heavy[p] = h;
    }
}

bool Seat::acceptable(int64_t target, double metal, double reach) {
    int i = view_of(target);
    if (i < 0 || !view.building[i]) return false;
    std::vector<double> cv;
    std::vector<uint8_t> hv;
    cover(&view.x[i], &view.z[i], 1, reach, true, cv, hv);
    return !hv[0] && cv[0] <= metal / style.commit;
}

int64_t Seat::choose_target(P2 origin, double metal, double reach, const std::vector<int64_t>& taken) {
    std::vector<int> cand;
    std::vector<double> cx, cz;
    for (int e = 0; e < view.size(); e++) {
        if (!view.building[e]) continue;
        if (std::find(taken.begin(), taken.end(), view.uid[e]) != taken.end()) continue;
        cand.push_back(e), cx.push_back(view.x[e]), cz.push_back(view.z[e]);
    }
    if (cand.empty()) return -1;
    std::vector<double> cv;
    std::vector<uint8_t> hv;
    cover(cx.data(), cz.data(), (int)cand.size(), reach, true, cv, hv);
    int best = -1;
    double bk = INFINITY;
    bool any = false;
    for (size_t k = 0; k < cand.size(); k++) {
        bool ok = !hv[k] && cv[k] <= metal / style.commit;
        any |= ok;
        double key = ok ? std::hypot(cx[k] - origin.x, cz[k] - origin.z) + (view.extractor[cand[k]] ? 0.0 : FAR) : INFINITY;
        if (best < 0 || key < bk) best = (int)k, bk = key;
    }
    if (!any) return -1;
    return view.uid[cand[best]];
}

// Towards where the army walks (`march`).
P2 Seat::probe(P2 centre, double metal, double reach) {
    P2 frm = centre;
    double dist = py_hypot(march.x - frm.x, march.z - frm.z);
    double start = dist < PROBE ? dist : PROBE;
    // np.arange(start, 0.0, -PROBE_STEP): its length and its fill
    double len_f = std::ceil((0.0 - start) / -PROBE_STEP);
    int len = len_f > 0 ? (int)len_f : 0;
    if (len <= 0) return frm;
    std::vector<double> steps(len);
    steps[0] = start;
    if (len > 1) steps[1] = start + -PROBE_STEP;
    double delta = len > 1 ? steps[1] - start : 0.0;
    for (int i = 2; i < len; i++) steps[i] = start + i * delta;
    std::vector<double> px(len), pz(len);
    for (int i = 0; i < len; i++) {
        P2 p = along(frm, march, steps[i]);
        px[i] = p.x, pz[i] = p.z;
    }
    std::vector<double> cv;
    std::vector<uint8_t> hv;
    cover(px.data(), pz.data(), len, reach, false, cv, hv);
    for (int i = 0; i < len; i++)
        if (!hv[i] && cv[i] <= metal / style.commit) return {px[i], pz[i]};
    return frm;
}

void Seat::raids(const Ctx& c, std::vector<Fighter>& f, const std::vector<int>& raider_ks, const std::vector<uint8_t>& retreat,
                 const std::vector<P2>& extractors, std::vector<Plan>& plans) {
    const double seconds = c.seconds;
    // alive: the raiders by uid
    std::vector<int> alive = raider_ks;
    std::sort(alive.begin(), alive.end(), [&](int a, int b) { return f[a].uid < f[b].uid; });
    for (int k : alive) raider_stamp[f[k].uid] = call, fighter_of[f[k].uid] = k;
    auto is_raider = [&](int64_t uid) { return uid >= 0 && uid < (int64_t)raider_stamp.size() && raider_stamp[uid] == call; };
    // _patrol_count
    int nr = (int)alive.size();
    int want = 0;
    if (!extractors.empty()) {
        want = (int)std::ceil(style.patrol_share * nr);
        if (seconds >= PATROL_MIN_FROM) want = std::max(want, PATROL_MIN);
        want = std::min(want, std::max(0, nr - FIRST_STRAIGHT));
    }
    for (size_t u = 0; u < patrol.size(); u++)
        if (patrol[u] && !is_raider((int64_t)u)) patrol[u] = 0;
    {
        int count = 0;
        for (size_t u = 0; u < patrol.size(); u++)
            if (patrol[u]) {
                if (count >= want) patrol[u] = 0;
                else count++;
            }
        for (int a = FIRST_STRAIGHT; a < nr; a++) {
            if (count >= want) break;
            int64_t u = f[alive[a]].uid;
            if (!patrol[u]) patrol[u] = 1, count++;
        }
    }
    for (size_t u = 0; u < squad_of.size(); u++)
        if (squad_of[u] >= 0 && (patrol[u] || !is_raider((int64_t)u))) squad_of[u] = -1;
    // _squad_with_room, with the counts kept as they change
    std::vector<int> counts(next_squad + 1, 0);
    for (size_t u = 0; u < squad_of.size(); u++)
        if (squad_of[u] >= 0) counts[squad_of[u]]++;
    for (int k : alive) {
        int64_t u = f[k].uid;
        if (patrol[u] || retreat[k] || squad_of[u] >= 0) continue;
        int sid = -1;
        for (int s = 0; s <= next_squad; s++)
            if (counts[s] > 0 && counts[s] < SQUAD_SIZE) {
                sid = s;
                break;
            }
        if (sid < 0) {
            sid = ++next_squad;
            counts.push_back(0);
        }
        squad_of[u] = sid;
        counts[sid]++;
    }
    int ns = next_squad + 1;
    if ((int)squad_live.size() < ns) {
        squad_live.resize(ns, 0);
        squad_target.resize(ns, -1);
        since_has.resize(ns, 0);
        since_t.resize(ns, 0.0);
        since_d.resize(ns, 0.0);
    }
    // members, in the order of each squad's lowest uid
    std::vector<std::vector<int64_t>> members(ns);
    std::vector<int> member_order;
    for (size_t u = 0; u < squad_of.size(); u++) {
        int s = squad_of[u];
        if (s < 0) continue;
        if (members[s].empty()) member_order.push_back(s);
        members[s].push_back((int64_t)u);
    }
    std::vector<uint8_t> in_members(ns, 0), in_centre(ns, 0);
    std::vector<P2> centre(ns);
    std::vector<double> metal(ns, 0.0), strength(ns, 0.0);
    for (int s : member_order) {
        in_members[s] = 1;
        PySum m;
        double st = 0.0, sx = 0.0, sz = 0.0;
        bool first = true;
        int active = 0;
        for (int64_t u : members[s]) {
            const Fighter& fu = f[fighter_of[u]];
            m.add(fu.cost);
            if (first || fu.reach > st) st = fu.reach;
            first = false;
            if (!retreat[fighter_of[u]]) sx += fu.x, sz += fu.z, active++;
        }
        metal[s] = m.value(), strength[s] = st;
        if (active) in_centre[s] = 1, centre[s] = {sx / active, sz / active};
    }
    // Targets
    given_up.erase(std::remove_if(given_up.begin(), given_up.end(), [&](auto& g) { return g.second <= seconds; }), given_up.end());
    std::vector<int64_t> target(ns, -1);
    std::vector<uint8_t> has_target(ns, 0);
    auto targeted = [&](int64_t uid) {
        for (int s = 0; s < ns; s++)
            if (has_target[s] && target[s] == uid) return true;
        return false;
    };
    for (int s = 0; s < ns; s++) {
        if (!in_centre[s]) continue;
        int64_t kept = squad_live[s] ? squad_target[s] : -1;
        if (kept < 0 || targeted(kept) || !acceptable(kept, metal[s], strength[s])) continue;
        // _patient
        int i = view_of(kept);
        bool near = py_hypot(view.x[i] - centre[s].x, view.z[i] - centre[s].z) <= NEAR_TARGET;
        double started = since_has[s] ? since_t[s] : seconds, distance = since_has[s] ? since_d[s] : 0.0;
        double allowed = GIVE_UP_SLACK * style.patience + distance / TRAVEL_SPEED + (near ? PATIENCE_AT_TARGET * style.patience : 0.0);
        if (seconds - started <= allowed) {
            target[s] = kept, has_target[s] = 1;
        } else {
            bool found = false;
            for (auto& g : given_up)
                if (g.first == kept) g.second = seconds + GIVE_UP_FOR, found = true;
            if (!found) given_up.push_back({kept, seconds + GIVE_UP_FOR});
        }
    }
    std::vector<int64_t> taken;
    for (int s = 0; s < ns; s++) {
        if (!in_centre[s] || has_target[s]) continue;
        taken.clear();
        for (int t = 0; t < ns; t++)
            if (has_target[t] && target[t] >= 0) taken.push_back(target[t]);
        for (auto& g : given_up) taken.push_back(g.first);
        int64_t t = choose_target(centre[s], metal[s], strength[s], taken);
        target[s] = t, has_target[s] = 1;
        int64_t before = squad_live[s] ? squad_target[s] : -1;
        if (t >= 0 && t != before) {
            int i = view_of(t);
            since_has[s] = 1, since_t[s] = seconds, since_d[s] = py_hypot(view.x[i] - centre[s].x, view.z[i] - centre[s].z);
        }
    }
    for (int s = 0; s < ns; s++)
        if (in_centre[s]) squad_live[s] = 1, squad_target[s] = target[s];
    // Probing squads close together become one
    std::vector<int> probing;
    for (int s = 0; s < ns; s++)
        if (in_centre[s] && squad_target[s] < 0) probing.push_back(s);
    for (size_t i = 0; i < probing.size(); i++) {
        int a = probing[i];
        for (size_t j = i + 1; j < probing.size(); j++) {
            int b = probing[j];
            if (in_members[a] && in_members[b] && members[a].size() + members[b].size() <= 2 * SQUAD_SIZE &&
                py_hypot(centre[a].x - centre[b].x, centre[a].z - centre[b].z) <= MERGE_REACH) {
                for (int64_t u : members[b]) squad_of[u] = a, members[a].push_back(u);
                members[b].clear();
                in_members[b] = 0;
                squad_live[b] = 0;
                in_centre[b] = 0;
                metal[a] += metal[b];
                strength[a] = std::max(strength[a], strength[b]);
            }
        }
    }
    for (int s : member_order) {
        if (!in_members[s] || !in_centre[s]) continue;
        P2 goal;
        int64_t t = squad_target[s];
        if (t >= 0) {
            int i = view_of(t);
            goal = {view.x[i], view.z[i]};
        } else {
            goal = probe(centre[s], metal[s], strength[s]);
        }
        for (int64_t u : members[s]) {
            int k = fighter_of[u];
            if (!retreat[k]) plans[k] = {true, K_FIGHT, goal.x, goal.z, -1};
        }
    }
    // _patrol_orders
    {
        int j = 0;
        for (size_t u = 0; u < patrol.size(); u++) {
            if (!patrol[u]) continue;
            int k = fighter_of[u];
            int jj = j++;
            if (retreat[k] || extractors.empty()) continue;
            P2 e = extractors[jj % extractors.size()];
            P2 point = along(e, enemy_base, std::min(PATROL_OFFSET, py_hypot(enemy_base.x - e.x, enemy_base.z - e.z)));
            int best = -1;
            double bd = 0.0;
            for (int t = 0; t < view.size(); t++) {
                if (!view.threat[t]) continue;
                double dd = std::hypot(view.x[t] - e.x, view.z[t] - e.z);
                if (best < 0 || dd < bd) best = t, bd = dd;
            }
            if (best >= 0 && bd <= PATROL_REACH) point = {view.x[best], view.z[best]};
            plans[k] = {true, K_FIGHT, point.x, point.z, -1};
        }
    }
    for (int k : alive)
        if (retreat[k]) plans[k] = {true, K_MOVE, rally.x, rally.z, -1};
}

// Aimer._aim
void Seat::aim(const Ctx& c) {
    const Data& d = *data;
    const double seconds = c.seconds;
    aimed.erase(std::remove_if(aimed.begin(), aimed.end(),
                               [&](const Aim& a) { return !(a.uid >= 0 && a.uid < (int64_t)stamp.size() && stamp[a.uid] == call); }),
                aimed.end());
    auto alive = [&](int64_t uid) { return view_of(uid) >= 0; };
    auto find = [&](int64_t uid) -> Aim* {
        for (auto& a : aimed)
            if (a.uid == uid) return &a;
        return nullptr;
    };
    auto is_seen = [&](int64_t uid) { return std::binary_search(seen.begin(), seen.end(), uid); };
    size_t ne = enemies.size();
    // valuable(): standing VALUABLE buildings, and fusions from the second
    std::vector<uint8_t> valuable(ne);
    {
        int fusions = 0;
        for (size_t e = 0; e < ne; e++) {
            bool built = !(enemies[e]->flags & zk::UF_BEING_BUILT);
            valuable[e] = d.has(enemy_rows[e], ZKS_VALUABLE) && built;
            fusions += d.special[X_FUSION] >= 0 && enemy_rows[e] == d.special[X_FUSION] && built;
        }
        if (fusions >= 2)
            for (size_t e = 0; e < ne; e++)
                if (enemy_rows[e] == d.special[X_FUSION] && !(enemies[e]->flags & zk::UF_BEING_BUILT)) valuable[e] = 1;
    }
    bool any_missile = false;
    for (size_t i = 0; i < own.size(); i++)
        any_missile |= !(own[i]->flags & zk::UF_BEING_BUILT) && d.has(own_rows[i], ZKS_MISSILE);
    if (any_missile) {
        std::vector<uint8_t> visible(ne);
        std::vector<double> worth(ne);
        for (size_t e = 0; e < ne; e++) {
            int r = enemy_rows[e];
            visible[e] = is_seen(enemies[e]->uid) && !d.has(r, ZKS_MOBILE) && !(enemies[e]->flags & zk::UF_BEING_BUILT) && r != d.unknown;
            worth[e] = (double)d.cost[r] * (valuable[e] ? 4.0 : 1.0);
        }
        for (size_t i = 0; i < own.size(); i++) {
            const Rec* m = own[i];
            if ((m->flags & zk::UF_BEING_BUILT) || !d.has(own_rows[i], ZKS_MISSILE)) continue;
            Aim* a = find(m->uid);
            if (a && alive(a->target) && seconds - a->when < FIRE_AGAIN) continue;    // not _due
            int best = -1;
            for (size_t e = 0; e < ne; e++) {
                if (!visible[e]) continue;
                float dd = hypotf(enemies[e]->x - m->x, enemies[e]->z - m->z);
                if (!(dd <= d.range[own_rows[i]])) continue;
                if (best < 0 || worth[e] > worth[best]) best = (int)e;
            }
            if (best < 0) continue;
            const Rec* t = enemies[best];
            if (a) a->target = t->uid, a->when = seconds;
            else aimed.push_back({m->uid, t->uid, seconds});
            log.push_back({seconds, 1, enemy_rows[best]});
            army_out.push_back(order(m->uid, K_ATTACK_UNIT, t->x, t->z, t->uid));
        }
    }
    if (d.special[X_NUKE] >= 0) {
        for (size_t i = 0; i < own.size(); i++) {
            const Rec* m = own[i];
            if ((m->flags & zk::UF_BEING_BUILT) || own_rows[i] != d.special[X_NUKE]) continue;
            if (m->ord_kind == K_MANUAL_FIRE && alive(m->ord_target)) continue;
            Aim* a = find(m->uid);
            if (a && alive(a->target) && seconds - a->when < FIRE_AGAIN) continue;
            // _nuke_target
            int best = -1;
            float best_value = 0.0f;
            for (size_t b = 0; b < ne; b++) {
                int r = enemy_rows[b];
                if (d.has(r, ZKS_MOBILE) || r == d.unknown || !is_seen(enemies[b]->uid)) continue;
                float value = np_sum<float>(
                    [&](int64_t e) {
                        float dd = hypotf(enemies[b]->x - enemies[e]->x, enemies[b]->z - enemies[e]->z);
                        return dd <= NUKE_VALUE_REACH ? d.cost[enemy_rows[e]] : 0.0f;
                    },
                    (int64_t)ne);
                if (best < 0 || value > best_value) best = (int)b, best_value = value;
            }
            if (best < 0) continue;
            const Rec* t = enemies[best];
            if (a) a->target = t->uid, a->when = seconds;
            else aimed.push_back({m->uid, t->uid, seconds});
            log.push_back({seconds, 2, enemy_rows[best]});
            army_out.push_back(order(m->uid, K_MANUAL_FIRE, t->x, t->z, t->uid));
        }
    }
}

}  // namespace zks
