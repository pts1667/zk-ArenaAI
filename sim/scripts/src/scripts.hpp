// Internals of libzkscripts: the shared data, a seat's state, and the scripts' entry points.
#pragma once

#include <cstdint>
#include <deque>
#include <memory>
#include <vector>

#include "numeric.hpp"
#include "zkscripts.h"
#include "zksim/rawframe.hpp"

namespace zks {

using Rec = zk::UnitRecord;

// Order kinds, as rawframe.py and the sim number them.
enum { K_MOVE = 1, K_FIGHT = 2, K_ATTACK_UNIT = 3, K_MANUAL_FIRE = 6, K_BUILD = 10, K_ASSIST = 11, K_RECLAIM = 13, K_PRODUCE = 14 };
// A transport's, as the sim numbers them: rawframe.py's are one less (policy_sim_env.py turns one into the other).
enum { K_LOAD = ZK_ORDER_LOAD, K_UNLOAD = ZK_ORDER_UNLOAD, K_DROP = ZK_ORDER_DROP };
// policy_script.py fighter states
enum { S_NEW = 0, S_WAITING = 1, S_SENT = 2, S_RETREATING = 3 };
// Roles the scripts tell apart, in ROLE_ORDER first (policy_mix.py)
enum { R_RAID, R_RIOT, R_ASSAULT, R_SKIRM, R_ARTY, R_AA, R_SNIPE, R_SCOUT, R_BUILDER, R_BOMB, R_COUNT };
// Special rows; X_FACTORY and X_SECOND are pending slots only
enum { X_CERBERUS, X_LUCIFER, X_SILO, X_NUKE, X_FUSION, X_PYLON, X_RADAR, X_FACTORY, X_SECOND };

// The standard army script's knobs (army.cpp): what a seat has when it's given none.
zks_style standard_style();

struct Ground {
    int rows = 0, cols = 0;
    double cell = 16.0, map_w = 0.0, map_h = 0.0;
    std::vector<float> spots;   // n x 3
    int n_spots = 0;
    std::vector<std::vector<uint8_t>> masks;
    std::vector<std::vector<float>> sites;
    std::vector<std::vector<int16_t>> labels;
    std::vector<float> height;  // rows x cols
    int largest[ZKS_MOVE_CLASSES] = {};

    int part_at(int c, double x, double z) const;
    bool joined(int c, P2 a, P2 b) const {
        int p = part_at(c, a.x, a.z);
        return p >= 0 && p == part_at(c, b.x, b.z);
    }
    bool fits(double x, double z, double half_x, double half_z, double height_dif, int mask) const;
};

struct Data {
    int rows = 0, unknown = 0;
    std::vector<int32_t> row_of_def;
    std::vector<uint32_t> bits;
    std::vector<float> cost, range, energy, pylon, needs_link, half_x, half_z, height_dif;
    std::vector<int8_t> move_class;
    std::vector<int32_t> catalog_row, role, footing, best_missile, generators, option_offsets, options, menu, row_of_catalog;
    std::vector<int32_t> plate_of;     // a factory row's plate row, -1 for none
    std::vector<int8_t> kind;           // R_* of a row's role, -1 for any other
    std::vector<int8_t> special_of;     // X_* of a row, -1 for none
    std::vector<uint8_t> air_plant;     // a factory whose fighters fly: the army script has its own mix for those
    std::vector<float> lift;            // rows x 3: a transport's speed with a load of each class, 0 for one it can't lift
    std::vector<int8_t> load_class;     // 1 light, 2 medium, 3 heavy, 0 for what can't be carried
    std::vector<int32_t> transport_of;  // a transport's index among the transports, -1 for any other row
    std::vector<float> fall_loss;       // n_transports x rows
    int n_transports = 0;
    int n_roles = 0;
    int role_ids[10] = {};
    int special[7] = {};
    int second_factories[3] = {};
    int caretaker = -1;
    double group_radii[ZKS_GROUP_RADII] = {};
    double cerberus_reach = 0.0, missile_reach = 0.0;
    float spot_half = 24.0f;
    int n_footings = 0;
    std::vector<std::unique_ptr<Ground>> maps;     // by pointer: seats keep theirs while maps are added

    bool has(int row, uint32_t bit) const { return (bits[row] & bit) != 0; }
    int row_of(int64_t def_id) const {
        int64_t n = (int64_t)row_of_def.size();
        int64_t i = (def_id < 0 || def_id >= n - 1) ? n - 1 : def_id;
        return row_of_def[i];
    }
    bool can_build(int builder, int row) const;
    // Whether a transport's row lifts a load's, and the share of its health the load would lose dropped from it in flight.
    bool lifts(int transport, int load) const {
        const int k = load_class[load];
        return transport_of[transport] >= 0 && k > 0 && lift[(size_t)transport * 3 + (k - 1)] > 0.0f;
    }
    double dropped(int transport, int load) const { return fall_loss[(size_t)transport_of[transport] * rows + load]; }
    int first_extractor_option(int builder) const;
    int menu_of(int factory, int r) const {
        int id = role_ids[r];
        return id < 0 ? -1 : menu[(size_t)factory * n_roles + id];
    }
};

// The random numbers of one economy: its own generator (xoshiro256**), or a tape of Python's draws.
struct Draws {
    uint64_t s[4] = {};
    bool has_spare = false;
    double spare = 0.0;
    std::vector<double> tape;
    std::vector<uint8_t> tape_kinds;
    size_t tape_at = 0;
    bool taped = false;
    int misses = 0;

    void seed(uint64_t seed);
    double own_uniform();
    double own_normal();
    double uniform();
    double normal();   // standard
    void end_step();
};

struct Fighter {
    int64_t uid;
    int kind, row;
    double x, z, hp, cost, reach;
    bool idle;
};

struct Plan {
    bool has = false;
    int kind = 0;
    double x = 0.0, z = 0.0;
    int64_t target = -1;
};

struct Event {
    double seconds;
    int what, row;
};

// The enemy as the army script sees it (policy_army.View).
struct View {
    std::vector<int64_t> uid;
    std::vector<double> x, z, cost, reach;
    std::vector<uint8_t> building, defence, threat, heavy_riot, slow_heavy, extractor;
    int size() const { return (int)uid.size(); }
};

// Who is in a game: each team's allyteam, whether the scripts play it, and its start.
struct Game {
    int n_teams = 0;
    int32_t ally_of[ZKS_MAX_TEAMS] = {};
    uint32_t players = 0;        // zks_reset's script_teams: a team without its bit (the chickens) has no base
    P2 starts[ZKS_MAX_TEAMS] = {};
    bool player(int t) const { return (players >> t) & 1u; }
    // A chicken game's chickens: the first team the scripts don't play, -1 for none.
    int chickens() const {
        for (int t = 0; t < n_teams; t++)
            if (!player(t)) return t;
        return -1;
    }
};

// Chicken defense as the players see it this step (zks_step_args.chickens): anger 0 and the queen far off without it.
struct Chickens {
    bool known = false;
    double anger = 0.0, until_queen = INFINITY;
    bool queen = false;     // alive now
};

// One seat's inputs for one step.
struct Ctx {
    const Rec* frames[ZKS_MAX_TEAMS];    // every team's frame of the game, by team, the seat's own among them
    int frame_n[ZKS_MAX_TEAMS];
    const float* glob;           // the seat's
    double reclaim;              // metal a second of the glob's income that's reclaim
    int64_t busy;
    double seconds;
    bool army_due, eco_due;
    double army_frames = 0.0;    // game frames from one look of the army script to the next
    bool groups;                 // zks_step_args.group_orders
    const zk::FeatureRecord* features = nullptr;   // what the seat sees to reclaim
    int feature_n = 0;
    Chickens chickens;           // in a chicken game
};

struct Job {
    bool has = false;
    zk_order_rec o{};
};

struct GridOf;
struct Pass;

// One team of one game.  Memory of own units is in arrays indexed by uid (a game's handles are small integers that only
// grow), one array a field.
struct Seat {
    const Data* data = nullptr;
    const Ground* ground = nullptr;
    int team = 0, ally = 0;
    Game game;
    bool active = false, economy = false, aimer = false;
    int flags = 0;     // this step's ZKS_SEAT_*
    zks_style style{};                  // the army script's knobs (standard_style unless the game gave its own)
    // The enemy teams, their starts nearest our base first (ties to the lower team).  `foe` is the first of them that still
    // has units this step, else the first, -1 with none; `enemy_base` is its start (the map's middle with none).  With one
    // enemy, they're it and its start.
    int foes[ZKS_MAX_TEAMS] = {};
    int n_foes = 0, foe = -1;
    P2 base{}, enemy_base{}, rally{};
    P2 march{};                         // where the army walks this step (Seat::fight)
    P2 waypoints[4];

    // By own uid.
    std::vector<int8_t> state;          // ArmyScript.state, S_NEW when absent
    std::vector<int8_t> ordered_kind;   // ArmyScript.ordered: 0 when absent
    std::vector<double> ordered_x, ordered_z;
    std::vector<int8_t> way;            // ArmyScript.way, -1 when absent
    std::vector<int32_t> squad_of;      // -1 when absent
    std::vector<int32_t> current;       // -1 when absent
    std::vector<double> still_x, still_z, still_since;  // where a fighter last moved off to, and when: NAN when absent (Seat::fight)
    std::vector<uint8_t> born;          // ArmyScript.seen: uids the mix has counted
    std::vector<uint8_t> patrol;
    std::vector<int64_t> helps;         // Economy.helps: INT64_MIN when absent
    std::vector<int64_t> reclaims;      // the feature it was last sent to reclaim: INT64_MIN when absent
    std::vector<int32_t> builds;        // the row it was last sent to build, -1 when absent: a frame doesn't say (World::order)
    std::vector<int32_t> lift_load;     // a transport's load, aboard or to fetch: -1 when absent (Seat::lifts)
    std::vector<uint8_t> lifted;        // seen carried, until it's seen down again
    std::vector<int64_t> drop_goal;     // what a unit brought to a target goes for: -1 when absent
    std::vector<int32_t> own_at;        // index into `own` this call, where `stamp` matches (Seat::lifts)
    std::vector<uint8_t> noted;         // Economy.finished, a bit a category
    std::vector<int32_t> stamp;         // the call that last saw this uid among its own
    std::vector<int32_t> fighter_of;    // index into the call's fighters, where the raider stamp matches
    std::vector<int32_t> raider_stamp;
    std::vector<int32_t> view_stamp;    // by enemy uid: index into the view where it matches
    std::vector<int32_t> view_index;
    int32_t call = 0;

    // Army: squads by id.
    std::vector<uint8_t> squad_live;    // in ArmyScript.squads
    std::vector<int64_t> squad_target;  // -1 for None
    std::vector<uint8_t> since_has;
    std::vector<double> since_t, since_d;
    int next_squad = 0;
    std::vector<std::pair<int64_t, double>> given_up;
    bool has_waiting_since = false;
    double waiting_since = 0.0;
    bool reached_start = false;
    // Transports: what the wing goes for (-1 for nothing), and since when its loaded ones have waited for the rest.
    int64_t lift_target = -1;
    double lift_waiting = NAN;
    // Mix
    std::vector<int32_t> made;
    std::vector<int32_t> made_air;      // the part of made that flies
    bool seen_air = false, seen_kitable = false, seen_heavy = false;
    // Aimer: own uid -> (target, when)
    struct Aim { int64_t uid, target; double when; };
    std::vector<Aim> aimed;
    std::vector<Event> log;

    // Economy
    bool floating = false, energy_low = false;
    int checks_eligible = 0, checks_stalled = 0;
    struct Count { double t; int ours, theirs; };
    std::deque<Count> counts;
    int nuke_drawn = -1;
    double pending[9] = {};     // by X_*
    int second = -2;            // the second factory's row, drawn when it's due: -1 for none to be had
    struct Going { int64_t uid; double gx, gz, since, nearest; };
    std::vector<Going> going;
    struct Avoid { int64_t ix, iz; double until; };
    std::vector<Avoid> avoid;
    int finished[7] = {};
    Draws draws;

    // This step's inputs, filtered, and outputs.  Enemies are every enemy team's own units, read from its frame, a team after
    // another; allies' units are in our frame, and are neither own nor enemies.
    std::vector<const Rec*> own, enemies, allies;
    std::vector<int> own_rows, enemy_rows, ally_rows;
    int units_of[ZKS_MAX_TEAMS] = {};   // each enemy's and ally's units this step
    std::vector<P2> bases;              // the enemy bases our half is measured against (Seat::ours)
    std::vector<std::pair<int, P2>> partners;   // the allied players with units, and their starts: they have the spots nearer them
    std::vector<int64_t> seen;   // sorted: enemies in our sight or radar
    View view;
    std::vector<zk_order_rec> army_out, eco_out, pre, post;
    std::vector<zk_group_rec> groups;   // the army script's group orders
    std::vector<uint8_t> grouped;       // by army_out index: sent in a group order, so not alone
    std::vector<int32_t> anchor_of;     // by army_out index: the group order it's the anchor of, -1 for none
    std::vector<int32_t> mine_of, member_of;    // group_plans' scratch, by uid

    void reset(const Data* d, const Ground* g, const Game& game, int team, P2 rally, bool economy, bool aimer, uint64_t seed,
               const zks_style* knobs);
    void grow(int64_t uid);
    void run(const Ctx& c);

    // army.cpp
    void army(const Ctx& c);
    void observe();
    void births_and_forget();
    void produce(const Ctx& c, double metal_income);
    int choose(int factory_row, double seconds, int constructors, double metal_income, bool queen_due, int transport) const;
    int transport_due(int factory_row) const;
    bool load_for_lift(size_t i) const;
    void lifts(const Ctx& c);
    bool landing(int move_class, P2 at, P2 from, P2& out) const;
    bool tell(const Rec* u, int kind, double x, double z, int64_t target);
    void make_view();
    void fight(const Ctx& c, std::vector<Fighter>& f, const std::vector<P2>& extractors);
    P2 guard_goal(const Ctx& c, P2 centre) const;
    void raids(const Ctx& c, std::vector<Fighter>& f, const std::vector<int>& raiders, const std::vector<uint8_t>& retreat,
               const std::vector<P2>& extractors, std::vector<Plan>& plans);
    int nearest(const std::vector<uint8_t>& mask, P2 p, double below = INFINITY) const;
    void group_plans(const Ctx& c, size_t first);
    void cover(const double* px, const double* pz, int n, double reach, bool armies, std::vector<double>& out,
               std::vector<uint8_t>& heavy) const;
    bool acceptable(int64_t target, double metal, double reach);
    int64_t choose_target(P2 origin, double metal, double reach, const std::vector<int64_t>& taken);
    P2 probe(P2 centre, double metal, double reach);
    void aim(const Ctx& c);
    int view_of(int64_t uid) const {
        return (uid >= 0 && uid < (int64_t)view_stamp.size() && view_stamp[uid] == call) ? view_index[uid] : -1;
    }

    // eco.cpp
    void eco(const Ctx& c);
    std::vector<uint8_t> stuck(const Ctx& c, const std::vector<uint8_t>& builders, P2 home);
    bool avoided(double x, double z) const;
    bool stalled(double seconds) const;
    int have(int x, double seconds) const;
    bool ours(P2 p) const;
    bool our_spot(float x, float z) const;
    P2 front() const;
    bool silo_site(P2& site) const;
    bool turret(int& row, P2& site) const;
    bool project(const Ctx& c, double income, const GridOf& grid, int& prow, P2& at);
    bool site(int row, P2 near, const std::vector<float>& taken, double spread, P2& place);
    Job build(int64_t uid, int row, P2 near, const std::vector<float>& taken, double spread, const GridOf* grid, int on,
              int by_class, P2 by_at, bool use_by);
    Job job(const Ctx& c, int i, const GridOf& grid, bool has_project, int prow, P2 pat, Pass& pass, double income);
    Job energy_job(const Ctx& c, int i, const GridOf& grid, Pass& pass, double income, const std::vector<int>& being, bool force);
    Job production_job(const Ctx& c, int i, Pass& pass, double income, int by_class, P2 by_at);
    int told_to_build(size_t j) const;      // the row an own unit was sent to BUILD and still is, -1 for none
    Job second_factory_job(const Ctx& c, int i, const GridOf& grid, Pass& pass, double income, int by_class, P2 by_at);
    Job radar_job(const Ctx& c, int i, Pass& pass, int by_class, P2 by_at);
    Job link_job(const Ctx& c, int i, const GridOf& grid, Pass& pass);
    Job spot_job(const Ctx& c, int i, Pass& pass);
    bool unsafe(P2 p, float army_reach) const;
    Job reclaim_job(const Ctx& c, int i, Pass& pass, double reach);
};

}  // namespace zks
