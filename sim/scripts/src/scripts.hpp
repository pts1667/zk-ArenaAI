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
    int largest[ZKS_MOVE_CLASSES] = {};

    int part_at(int c, double x, double z) const;
    bool joined(int c, P2 a, P2 b) const {
        int p = part_at(c, a.x, a.z);
        return p >= 0 && p == part_at(c, b.x, b.z);
    }
    bool fits(double x, double z, double radius, int mask) const;
};

struct Data {
    int rows = 0, unknown = 0;
    std::vector<int32_t> row_of_def;
    std::vector<uint32_t> bits;
    std::vector<float> cost, range, energy, pylon, needs_link, radius;
    std::vector<int8_t> move_class;
    std::vector<int32_t> catalog_row, role, footing, best_missile, generators, option_offsets, options, menu, row_of_catalog;
    std::vector<int8_t> kind;           // R_* of a row's role, -1 for any other
    std::vector<int8_t> special_of;     // X_* of a row, -1 for none
    std::vector<uint8_t> air_plant;     // a factory whose fighters fly: the army script has its own mix for those
    std::vector<double> footing_radius;
    int n_roles = 0;
    int role_ids[10] = {};
    int special[7] = {};
    int second_factories[3] = {};
    double group_radii[ZKS_GROUP_RADII] = {};
    double cerberus_reach = 0.0, missile_reach = 0.0;
    float extractor_radius = 24.0f;
    int n_footings = 0;
    std::vector<std::unique_ptr<Ground>> maps;     // by pointer: seats keep theirs while maps are added

    bool has(int row, uint32_t bit) const { return (bits[row] & bit) != 0; }
    int row_of(int64_t def_id) const {
        int64_t n = (int64_t)row_of_def.size();
        int64_t i = (def_id < 0 || def_id >= n - 1) ? n - 1 : def_id;
        return row_of_def[i];
    }
    bool can_build(int builder, int row) const;
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

// One seat's inputs for one step.
struct Ctx {
    const Rec* own_frame;        // the seat's frame
    int own_n;
    const Rec* enemy_frame;      // the other side's frame
    int enemy_n;
    const float* glob;           // the seat's
    double reclaim;              // metal a second of the glob's income that's reclaim
    int64_t busy;
    double seconds;
    bool army_due, eco_due;
    bool groups;                 // zks_step_args.group_orders
    const zk::FeatureRecord* features = nullptr;   // what the seat sees to reclaim
    int feature_n = 0;
};

struct Job {
    bool has = false;
    zk_order_rec o{};
};

struct GridOf;
struct Pass;

// One side of one game.  Memory of own units is in arrays indexed by uid (a game's handles are small integers that only
// grow), one array a field.
struct Seat {
    const Data* data = nullptr;
    const Ground* ground = nullptr;
    int side = 0;
    bool active = false, economy = false, aimer = false;
    int flags = 0;     // this step's ZKS_SEAT_*
    zks_style style{};                  // the army script's knobs (standard_style unless the game gave its own)
    P2 base{}, enemy_base{}, rally{};
    P2 waypoints[4];

    // By own uid.
    std::vector<int8_t> state;          // ArmyScript.state, S_NEW when absent
    std::vector<int8_t> ordered_kind;   // ArmyScript.ordered: 0 when absent
    std::vector<double> ordered_x, ordered_z;
    std::vector<int8_t> way;            // ArmyScript.way, -1 when absent
    std::vector<int32_t> squad_of;      // -1 when absent
    std::vector<int32_t> current;       // -1 when absent
    std::vector<uint8_t> born;          // ArmyScript.seen: uids the mix has counted
    std::vector<uint8_t> patrol;
    std::vector<int64_t> helps;         // Economy.helps: INT64_MIN when absent
    std::vector<int64_t> reclaims;      // the feature it was last sent to reclaim: INT64_MIN when absent
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

    // This step's inputs, filtered, and outputs.
    std::vector<const Rec*> own, enemies;
    std::vector<int> own_rows, enemy_rows;
    std::vector<int64_t> seen;   // sorted
    View view;
    std::vector<zk_order_rec> army_out, eco_out, pre, post;
    std::vector<zk_group_rec> groups;   // the army script's group orders
    std::vector<uint8_t> grouped;       // by army_out index: sent in a group order, so not alone
    std::vector<int32_t> anchor_of;     // by army_out index: the group order it's the anchor of, -1 for none
    std::vector<int32_t> mine_of, member_of;    // group_plans' scratch, by uid

    void reset(const Data* d, const Ground* g, int side, P2 base, P2 enemy, P2 rally, bool economy, bool aimer, uint64_t seed,
               const zks_style* knobs);
    void grow(int64_t uid);
    void run(const Ctx& c);

    // army.cpp
    void army(const Ctx& c);
    void observe();
    void births_and_forget();
    void produce(const Ctx& c, double metal_income);
    int choose(int factory_row, double seconds, int constructors, double metal_income) const;
    void make_view();
    void fight(const Ctx& c, std::vector<Fighter>& f, const std::vector<P2>& extractors);
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
    std::vector<uint8_t> stuck(const Ctx& c, const std::vector<uint8_t>& builders);
    bool avoided(double x, double z) const;
    bool stalled(double seconds) const;
    int have(int x, double seconds) const;
    bool ours(P2 p) const;
    P2 front() const;
    bool silo_site(P2& site) const;
    bool turret(int& row, P2& site) const;
    bool project(const Ctx& c, double income, const GridOf& grid, int& prow, P2& at);
    bool site(int mask, double radius, P2 near, const std::vector<float>& taken, double spread, P2& place);
    Job build(int64_t uid, int row, P2 near, const std::vector<float>& taken, double spread, const GridOf* grid, int on,
              int by_class, P2 by_at, bool use_by);
    Job job(const Ctx& c, int i, const GridOf& grid, bool has_project, int prow, P2 pat, Pass& pass, double income);
    Job energy_job(const Ctx& c, int i, const GridOf& grid, Pass& pass, double income, const std::vector<int>& being, bool force);
    Job factory_job(const Ctx& c, int i, Pass& pass, double income, int by_class, P2 by_at);
    Job second_factory_job(const Ctx& c, int i, const GridOf& grid, Pass& pass, double income, int by_class, P2 by_at);
    Job radar_job(const Ctx& c, int i, Pass& pass, int by_class, P2 by_at);
    Job link_job(const Ctx& c, int i, const GridOf& grid, Pass& pass);
    Job spot_job(const Ctx& c, int i, Pass& pass);
    bool unsafe(P2 p, float army_reach) const;
    Job reclaim_job(const Ctx& c, int i, Pass& pass, double reach);
};

}  // namespace zks
