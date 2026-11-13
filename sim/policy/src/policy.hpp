// libzkpolicy's insides: the type table, a map's bundle, a frame, and the numbers of featurize.py, featurize_kernels.py and
// actions.py.  Every function here gives what its Python gives, bit for bit (zkpolicy.h).
#pragma once

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "zksim/rawframe.hpp"

namespace zkp {

using zk::FeatureRecord;
using zk::SiteRecord;
using zk::UnitRecord;

// rawframe.py's kinds of order.
enum Kind {
    KEEP, MOVE, FIGHT, ATTACK_UNIT, ATTACK_GROUND, STOP, MANUAL_FIRE, JUMP, CLOAK_ON, CLOAK_OFF,
    BUILD, ASSIST, REPAIR, RECLAIM, PRODUCE, RESERVED_KIND, THROW, PUSH, PULL, PLACE_BEACON, WAIT_AT_BEACON,
    LOAD, UNLOAD, DROP, KIND_COUNT
};
constexpr int ORDER_SLOTS = 16;

// featurize.py's.
constexpr int INSTANCE_COLS = 21, ORDER_COLS = ORDER_SLOTS + 7, RAY_COLS = 24, SUMMARY_COLS = 3;
constexpr int RAY_CELLS = 12;
constexpr int REGION_DIM = 34, SITE_DIM = 9, FEATURE_DIM = 5, GLOBAL_DIM = zk::G_COUNT - 2;
constexpr int GATE_DIM = 15, MOVE_CLASSES = 6, REGION_STATIC = 27;
constexpr int CAP_CONTROLLABLE = 512, CAP_UNITS = 1280, CAP_HIDDEN = 256, CAP_SITES = 24, CAP_FEATURES = 512, CAP_SUMMARIES = 12;
constexpr int MAX_ALLY_TEAMS = 8, MAX_PLAYERS = 64;
constexpr int DECISION_FRAMES = 15;
enum TokenKind { TK_SEEN, TK_BLIP, TK_MEMORY, TK_SUMMARY, TK_REGION, TK_SITE, TK_GLOBAL, TK_FEATURE, TK_HIDDEN };
constexpr int ENEMY_POINTERS = 5;           // ATTACK_UNIT, ATTACK_GROUND, MANUAL_FIRE, ASSIST, LOAD
constexpr int FINE_OPTIONS = 9;
constexpr int PLACE_RINGS = 3;
constexpr int PLACE_OFFSETS = 1 + 8 * PLACE_RINGS;
constexpr float PASS_CELL = 16.0f;

// The table as the kernels read it (featurize._table_pack): one float matrix and one bool matrix, with these columns.
enum TCol {
    T_COST, T_SPEED, T_RANGE, T_DPS_GROUND, T_DPS_AIR, T_MANUAL_RANGE, T_MANUAL_RELOAD, T_JUMP_RANGE, T_THROW_RADIUS, T_THROW_REACH,
    T_LIFT_LIGHT, T_LIFT_MEDIUM, T_LIFT_HEAVY, T_LIFT_SIZE, T_LOAD_CLASS, T_LOAD_SIZE, T_COLS
};
enum BCol {
    B_MOBILE, B_AIR, B_ARMED, B_HIT_GROUND, B_HIT_AIR, B_BUILDER, B_FACTORY, B_COMBAT, B_MANUALFIRE, B_JUMP, B_CLOAK, B_SWITCHES,
    B_TELEPORTS, B_BEACON, B_MISSILE, B_LAUNCHER, B_ACTS_IN_PLACE, B_CONSTRUCTOR, B_PRODUCES, B_BUILDS, B_TRANSPORT, B_COLS
};

// tables.TypeTable, read from the flat file zkpolicy.py writes (table.cpp).
struct Table {
    int rows = 0, type_dim = 0, unit_dim = 0, producible_count = 0, unknown = 0, roles = 0;
    uint32_t choosable = 0;
    std::string hash;
    std::vector<float> stats;           // rows x type_dim
    std::vector<float> f;               // rows x T_COLS
    std::vector<uint8_t> b;             // rows x B_COLS
    std::vector<int8_t> move_class;     // rows, -1 for none
    std::vector<uint8_t> produces, builds;     // rows x producible_count
    std::vector<int64_t> producible;    // producible_count, sorted
    std::vector<int32_t> row_of_def;    // the last entry stands for -1 and anything unheard of
    std::vector<int32_t> def_of_row;    // rows, -1 for the unknown row
    std::vector<int32_t> role;          // rows: groups.roles()
    std::vector<std::string> names, role_names;

    float fv(int64_t r, int c) const { return f[r * T_COLS + c]; }
    bool bv(int64_t r, int c) const { return b[r * B_COLS + c] != 0; }
    // featurize_kernels._row: TypeTable.rows for one def id.
    int64_t row(int64_t def_id) const {
        const int64_t last = int64_t(row_of_def.size()) - 1;
        return (def_id < 0 || def_id >= last) ? row_of_def[last] : row_of_def[def_id];
    }
};

// bundle.Bundle: views of its arrays, which its owner keeps alive, and what featurize makes of it once (_bundle_pack).
struct Bundle {
    double map_w_d = 0, map_h_d = 0, cell_d = 0;        // as Python floats
    float map_w = 0, map_h = 0, cell = 0, map_max = 0;  // as the float32 numpy makes of them
    int label_h = 0, label_w = 0, pass_rows = 0, pass_cols = 0, regions = 0, gates = 0;
    const int16_t* label = nullptr;     // label_h x label_w
    const float* height = nullptr;      // label_h x label_w
    const uint8_t* passable = nullptr;  // pass_rows x pass_cols
    const float* centre = nullptr;      // regions x 2
    const float* walk = nullptr;        // MOVE_CLASSES x regions x regions
    std::vector<float> region_radius;   // regions
    std::vector<uint8_t> region_cells;  // regions
    std::vector<float> reg_x;           // regions x REGION_DIM, the static columns
    std::vector<float> gate_x;          // gates x GATE_DIM, normalised
};

struct Frame {
    const UnitRecord* units = nullptr;
    int64_t n_units = 0;
    const SiteRecord* sites = nullptr;
    int64_t n_sites = 0;
    const FeatureRecord* features = nullptr;
    int64_t n_features = 0;
    const float* glob = nullptr;
    const Bundle* bundle = nullptr;
};

struct Options {
    int commit_decisions = 6;
    bool privileged = false, economy = false;
    bool enabled[KIND_COUNT] = {};
};

// The fields of fz.Tensors that featurize lays out (zkpolicy.h's ZKP_F_*), in that order.
enum Field {
    F_UNIT_X, F_UNIT_NAME, F_REG_X, F_SITE_X, F_GLOB_X, F_TOK_KIND, F_KIND_MASK, F_PTR_ENEMY, F_PTR_REGION, F_PTR_JUMP, F_PTR_WAIT,
    F_PTR_PRODUCE, F_ROUTE_BIAS, F_GATE_X, F_UIDS, F_ENEMY_TOKENS, F_NORTH, F_REGION_RADIUS, F_OWN_BEACONS, F_BEACON_REGIONS,
    F_HIDDEN_X, F_HIDDEN_NAME, F_UNIT_ALLY, F_UNIT_PLAYER, F_HIDDEN_ALLY, F_HIDDEN_PLAYER, F_FEAT_X, F_FEAT_IDS, F_PLACE_TOKENS,
    F_SITE_POS, F_FEAT_POS, F_COUNT
};
constexpr int LAYOUT = 4;       // offset in bytes, then up to three dims (unused ones 0)

int64_t bound(const Table& t, const Frame& fr, bool privileged);
// 0, or -1 if `capacity` was too small (it isn't, at `bound`).
int featurize(const Table& t, const Frame& fr, const Options& opt, uint8_t* arena, int64_t capacity, int64_t* layout,
              int64_t* controllable);

// What actions.py reads of a frame's tensors (zkpolicy.h's zkp_tensors).
struct TensorsView {
    int64_t controllable, units, enemies, regions, beacons, places, sites, features, producible;
    const int32_t* unit_name;
    const int64_t* uids;
    const int64_t* enemy_tokens;
    const uint8_t* kind_mask;       // C x KIND_COUNT
    const uint8_t* ptr_enemy;       // ENEMY_POINTERS x C x E
    const uint8_t* ptr_region;      // C x R
    const uint8_t* ptr_jump;
    const uint8_t* ptr_wait;
    const uint8_t* ptr_produce;     // C x producible
    const float* north;
    const float* region_radius;
    const int64_t* own_beacons;
    const int32_t* beacon_regions;
    const int64_t* place_tokens;
    const float* site_pos;          // sites x 2
    const float* feat_pos;          // features x 2
    const int64_t* feat_ids;
    const float* unit_x;            // units: world positions (actions.world_positions)
    const float* unit_z;
};

}  // namespace zkp
