/* libzkpolicy: the policy's inference side in C++, bit for bit with the Python it ports (docs/learned-policy/02-policy.md,
 * "Deployment").  The type table (bots/zkbot/policy/tables.py), the featurizer (featurize.py, featurize_kernels.py), and the
 * actions (actions.py, groups.py).  Its ctypes binding is sim/py/zkpolicy.py, which says how Python calls it.
 *
 * The type table is read from a flat file, not from the catalog's JSON: zkpolicy.py writes it from tables.TypeTable
 * (`table_bytes`, and bots/zkbot/data/policy_table.bin for the bot), so the rules that make it stay in Python alone.
 * A map's bundle is views of bundle.Bundle's arrays, which must outlive it.  A frame is views of rawframe.Frame's records
 * (sim/include/zksim/rawframe.hpp's structs).
 *
 * Featurizing writes every array of fz.Tensors into a buffer the caller gives each frame, at least zkp_bound bytes, and says
 * where each is in its layout: ZKP_FIELDS rows of (byte offset, up to three dims, 0 for the unused).  zkp_featurize does many
 * frames on its own threads, made for the call and gone when it returns (so a process may fork after it).
 *
 * Bit for bit: float32 is computed in float32 where numpy and numba compute it so, float64 where they do, sums in their
 * order, and log1p, hypot, atan2, sin and cos are the C library's, as numpy's and numba's are on Linux x86-64 without AVX-512
 * (checked: tools/policy_parity.py).  Built without -ffast-math and with -ffp-contract=off (sim/CMakeLists.txt).
 */
#ifndef ZKPOLICY_H
#define ZKPOLICY_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ZKP_ABI_VERSION 2

int zkp_abi_version(void);

/* The type table, from its flat file's bytes.  NULL if they aren't one, with why in `error`. */
void* zkp_table_new(const void* bytes, int64_t size, char* error, int32_t error_size);
void zkp_table_free(void* table);
/* rows, type_dim, unit_dim, producible, unknown, roles, choosable, len(row_of_def) */
void zkp_table_info(const void* table, int64_t* out8);

typedef struct {
    double map_w, map_h, cell;
    int32_t label_h, label_w, pass_rows, pass_cols, regions, gates;
    const int16_t* label;           /* label_h x label_w */
    const float* height;            /* label_h x label_w */
    const uint8_t* passable;        /* pass_rows x pass_cols */
    const float* centre;            /* regions x 2 */
    const float* area;              /* regions */
    const float* region_height;     /* regions x 2 */
    const float* sea;               /* regions */
    const float* tags;              /* regions x 9 */
    const float* access;            /* regions x 6 */
    const float* shares;            /* regions x 2 */
    const float* spots;             /* regions x 2 */
    const float* gate_counts;       /* regions x 2 */
    const float* gate_x;            /* gates x 15 */
    const float* walk;              /* 6 x regions x regions */
} zkp_bundle_arrays;

void* zkp_bundle_new(const zkp_bundle_arrays* arrays);
void zkp_bundle_free(void* bundle);

typedef struct {
    const void* units;              /* UnitRecord */
    int64_t n_units;
    const void* sites;              /* SiteRecord */
    int64_t n_sites;
    const void* features;           /* FeatureRecord */
    int64_t n_features;
    const float* glob;              /* G_COUNT */
    const void* bundle;
} zkp_frame;

typedef struct {
    int32_t commit_decisions;       /* featurize's commit_decisions */
    int32_t privileged;             /* the critic's hidden units too */
    int32_t economy;
    uint8_t enabled[24];            /* by kind: enabled_kinds */
} zkp_options;

/* fz.Tensors' arrays in a layout, in this order (gate_a and gate_b are the bundle's own, and controllable comes apart). */
enum {
    ZKP_F_UNIT_X, ZKP_F_UNIT_NAME, ZKP_F_REG_X, ZKP_F_SITE_X, ZKP_F_GLOB_X, ZKP_F_TOK_KIND, ZKP_F_KIND_MASK, ZKP_F_PTR_ENEMY,
    ZKP_F_PTR_REGION, ZKP_F_PTR_JUMP, ZKP_F_PTR_WAIT, ZKP_F_PTR_PRODUCE, ZKP_F_ROUTE_BIAS, ZKP_F_GATE_X, ZKP_F_UIDS, ZKP_F_ENEMY_TOKENS,
    ZKP_F_NORTH, ZKP_F_REGION_RADIUS, ZKP_F_OWN_BEACONS, ZKP_F_BEACON_REGIONS, ZKP_F_HIDDEN_X, ZKP_F_HIDDEN_NAME, ZKP_F_UNIT_ALLY,
    ZKP_F_UNIT_PLAYER, ZKP_F_HIDDEN_ALLY, ZKP_F_HIDDEN_PLAYER, ZKP_F_FEAT_X, ZKP_F_FEAT_IDS, ZKP_F_PLACE_TOKENS, ZKP_F_SITE_POS,
    ZKP_F_FEAT_POS, ZKP_FIELDS
};

/* Bytes a frame's buffer needs, at most, for each of n frames. */
void zkp_bounds(const void* table, const zkp_frame* frames, int64_t n, int32_t privileged, int64_t* out);
/* n frames on `threads` threads (1 or less: this one).  layouts: n x ZKP_FIELDS x 4; controllable: n.  Returns how many
 * frames failed (a buffer too small, or a bad argument): 0. */
int64_t zkp_featurize(const void* table, const zkp_frame* frames, int64_t n, const zkp_options* options, uint8_t* const* buffers,
                      const int64_t* capacities, int64_t* layouts, int64_t* controllable, int32_t threads);

/* What decode and encode read of a frame's tensors, and the units' world positions (actions.world_positions). */
typedef struct {
    int64_t controllable, units, enemies, regions, beacons, places, sites, features, producible;
    const int32_t* unit_name;       /* the units' real names (fz.Tensors.true_name where a game dropped some): a throw's reach */
    const int64_t* uids;
    const int64_t* enemy_tokens;
    const uint8_t* kind_mask;
    const uint8_t* ptr_enemy;
    const uint8_t* ptr_region;
    const uint8_t* ptr_jump;
    const uint8_t* ptr_wait;
    const uint8_t* ptr_produce;
    const float* north;
    const float* region_radius;
    const int64_t* own_beacons;
    const int32_t* beacon_regions;
    const int64_t* place_tokens;
    const float* site_pos;
    const float* feat_pos;
    const int64_t* feat_ids;
    const float* unit_x;
    const float* unit_z;
} zkp_tensors;

typedef struct {
    int64_t uid, kind;
    double x, z;
    int64_t target;
    double radius;
    int64_t roles;
} zkp_order;

/* A decision is 9 arrays of `controllable` int64 one after another: kind, enemy, region, fine, produce, group, roles, place,
 * offset (actions.FIELDS).  decode writes at most `controllable` orders and returns how many, or -1 - i for the first unit i whose
 * choices point past the tensors (where actions.decode raises). */
int64_t zkp_decode(const void* table, const void* bundle, const zkp_tensors* t, const int64_t* decision, zkp_order* out);
/* Returns how many orders couldn't be put in the policy's terms. */
int64_t zkp_encode(const void* table, const void* bundle, const zkp_tensors* t, const zkp_order* orders, int64_t n, int64_t* decision);
/* fz.fine_point: x, z. */
void zkp_fine_point(const void* bundle, const zkp_tensors* t, int64_t region, int64_t option, double* out2);
/* actions.world_positions over unit_x's column `at` (fz._I). */
void zkp_world_positions(const float* unit_x, int64_t rows, int64_t unit_dim, int64_t at, double map_w, double map_h, float* x, float* z);
/* actions.to_slots: 10 arrays of `width` (anchor, then the decision's fields); order a permutation of the controllable units or
 * NULL; width the slots, or -1 for every acting unit.  Returns the width. */
int64_t zkp_to_slots(const int64_t* decision, int64_t controllable, const int64_t* order, int64_t width, int64_t* out);
/* Bound for zkp_to_slots' width with -1: how many units act. */
int64_t zkp_acting(const int64_t* decision, int64_t controllable);
/* actions.to_decision. */
void zkp_to_decision(const int64_t* slots, int64_t width, int64_t controllable, int64_t* decision);
/* groups.bucket and groups.expand (rows: each record's type-table row). */
int64_t zkp_group_bucket(double radius);
void zkp_group_expand(const void* table, int64_t anchor, int64_t roles_mask, double radius, const void* units, int64_t n,
                      const int64_t* rows, uint8_t* out);

#ifdef __cplusplus
}
#endif

#endif
