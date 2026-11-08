/* The scripted players of tools/policy_sim_env.py in C++: the economy script and the aimer (tools/policy_eco.py), the army
 * script (tools/policy_army.py) and its production (tools/policy_mix.py).  The Python modules are the specification.
 *
 * It reads only the raw-frame records (zksim/rawframe.hpp UnitRecord) and its own inputs, and writes zk_order_rec, never the
 * sim's World.  Static data (what each type is, each map's ground) is given once and shared read-only by every runner made from
 * it.  A runner holds the per-game state of many games and a thread pool of its own; a step computes every listed seat's orders
 * on that pool.  Each seat's work reads only its own state, so results don't depend on the thread count.  There are no mutable
 * globals: any number of runners can live in one process.
 */
#ifndef ZKSCRIPTS_H
#define ZKSCRIPTS_H

#include <stdint.h>

#include "zksim/capi.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ZKS_ABI_VERSION 1
#define ZKS_MOVE_CLASSES 6

/* Per type-table row bits (zks_data_desc.bits). */
enum {
    ZKS_MOBILE = 1 << 0,
    ZKS_AIR = 1 << 1,
    ZKS_ARMED = 1 << 2,          /* tables.armed */
    ZKS_BUILDER = 1 << 3,        /* tables.builder */
    ZKS_FACTORY = 1 << 4,
    ZKS_FIGHTS = 1 << 5,         /* ScriptData.fights: combat and mobile */
    ZKS_REGENERATES = 1 << 6,
    ZKS_EXTRACTOR = 1 << 7,
    ZKS_DEFENCE = 1 << 8,        /* ScriptData.defence */
    ZKS_KITABLE = 1 << 9,
    ZKS_HEAVY_RIOT = 1 << 10,
    ZKS_SLOW_HEAVY = 1 << 11,
    ZKS_MISSILE = 1 << 12,       /* EcoData.missile: the catalog's MISSILE flag */
    ZKS_COMMANDER = 1 << 13,
    ZKS_STATIC_ARMED = 1 << 14,  /* EcoData.armed: armed and not mobile */
    ZKS_VALUABLE = 1 << 15,      /* EcoData.valuable */
    ZKS_CONSTRUCTOR = 1 << 16,   /* counted by the mix as a constructor: role "builder" and mobile */
    ZKS_SINGULARITY = 1 << 17    /* energy of 100 or more: EcoData.finished["singularity"] */
};

/* What the type table and the scripts' data say, by row.  Row `rows - 1` is the unknown type.  Arrays have `rows` entries
 * unless said otherwise.  Copied: the caller's arrays can go once zks_data_new returns. */
typedef struct {
    int rows;
    int n_def;
    const int32_t* row_of_def;      /* n_def: tables.row_of_def, whose last entry stands for any def out of range */
    const uint32_t* bits;
    const float* cost;              /* tables.cost */
    const float* range;             /* tables.range */
    const int8_t* move_class;       /* tables.move_class, -1 for none */
    const int32_t* catalog_row;     /* the sim's catalog index, -1 if not in it */
    const int32_t* role;            /* index into the role names the caller keeps; ScriptData.role_of */
    const float* energy;            /* EcoData.energy */
    const float* pylon;             /* EcoData.pylon */
    const float* needs_link;        /* EcoData.needs_link */
    const float* radius;            /* EcoData.radius */
    const int32_t* footing;         /* index of the (low, high) water depths of its Footing, -1 for none */
    const double* footing_radius;   /* Footing.radius */
    const int32_t* best_missile;    /* ScriptData.best_missile, -1 for none */
    const int32_t* generators;      /* rows x 3: EcoData.generators, -1 for none */
    const int32_t* option_offsets;  /* rows + 1: EcoData.options of each row are options[offsets[r]:offsets[r + 1]], sorted */
    const int32_t* options;
    int n_roles;
    const int32_t* menu;            /* rows x n_roles: Mix.menu of a factory row, -1 where there's none */
    /* Role indices, -1 where the table has none: raid, riot, assault, skirm, arty, aa, snipe, scout, builder */
    int32_t role_ids[9];
    /* Special rows, -1 where absent: staticarty, turretantiheavy, staticmissilesilo, staticnuke, energyfusion, energypylon, staticradar */
    int32_t special[7];
    double cerberus_reach;          /* EcoData.reach["staticarty"] */
    double missile_reach;           /* EcoData.missile_reach */
    float extractor_radius;         /* the radius _taken gives metal spots */
    int n_footings;                 /* (low, high) pairs: each map has a mask and a site sample for each */
} zks_data_desc;

/* One map's ground, from policy_opening.Ground.  Copied. */
typedef struct {
    int rows, cols;                 /* of bundle.passable, at `cell` elmos */
    double cell, map_w, map_h;
    int n_spots;
    const float* spots;             /* n_spots x 3: bundle.spot_list */
    const uint8_t* const* masks;    /* n_footings, rows x cols: Ground.mask(low, high) */
    const int32_t* site_counts;     /* n_footings */
    const float* const* sites;      /* n_footings, site_counts x 2: Ground._sites(low, high) */
    const int16_t* const* labels;   /* ZKS_MOVE_CLASSES, rows x cols: Ground.labels */
    const int32_t* largest;         /* ZKS_MOVE_CLASSES: Ground._largest_part */
} zks_map_desc;

typedef struct zks_data zks_data;
typedef struct zks_runner zks_runner;

int zks_abi_version(void);
zks_data* zks_data_new(const zks_data_desc* desc);
/* Returns the map's index. */
int zks_data_add_map(zks_data* data, const zks_map_desc* map);
void zks_data_free(zks_data* data);

/* threads: of the runner's own pool, the calling thread included.  The data must outlive the runner. */
zks_runner* zks_runner_new(const zks_data* data, int games, int threads);
void zks_runner_free(zks_runner* runner);

/* A new game in this slot.  starts: x0, z0, x1, z1, each side's base, which is the other's enemy start.  rally: x0, z0, x1, z1.
 * economy: whether the economy script runs for both sides.  aimer: whether the army script aims missiles and the Trinity.
 * seeds: each side's economy's own random numbers. */
void zks_reset(zks_runner* runner, int game, int map, const double* starts, const double* rally, int economy, int aimer,
               uint64_t seed0, uint64_t seed1);

enum {
    ZKS_SEAT_ARMY = 1   /* the army script plays this seat */
};

typedef struct {
    int n;                          /* listed games */
    const int32_t* games;           /* n: slots */
    /* Indexed by slot: the batch's arrays (zksim.Batch.records, counts, globs) */
    const void* records;            /* slots x 2 x cap UnitRecord */
    int cap;
    const int32_t* counts;          /* slots x 2 */
    const float* globs;             /* slots x 2 x G_COUNT */
    const int32_t* decisions;       /* slots */
    /* Indexed by listed seat, 2 k + side */
    const int32_t* seat_flags;      /* ZKS_SEAT_* */
    const int32_t* busy;            /* the commander's uid while the opening has it, else -1 */
    int ticks, tick_frames, frames_per_second;  /* game seconds = decisions * ticks * tick_frames / frames_per_second */
    int script_every;               /* the army script runs when decisions % script_every == 0 */
    int eco_every;                  /* the economy script when decisions % eco_every == 0 */
    /* Orders from elsewhere, per listed seat (2 n + 1 offsets): `pre` goes before the script's (the policy's), `post` after
     * (the opening's).  The economy script's come last. */
    const zk_order_rec* pre;
    const int32_t* pre_offsets;
    const zk_order_rec* post;
    const int32_t* post_offsets;
} zks_step_args;

/* Computes every listed seat's orders, then writes them as zk_batch_step takes them, if they fit:
 *   out, out_offsets (n + 1): per listed game, side 0 then side 1, each pre, army script, post, economy script;
 *   army, army_offsets (2 n + 1): the army script's orders alone, per listed seat.
 * Returns the number of orders in `out`.  If they're more than out_cap, or the army's more than army_cap, nothing is written:
 * call zks_collect with room enough (zks_last_counts says how much). */
int zks_step(zks_runner* runner, const zks_step_args* args, zk_order_rec* out, int out_cap, int32_t* out_offsets,
             zk_order_rec* army, int army_cap, int32_t* army_offsets);
/* The last step's orders again, into buffers that fit. */
int zks_collect(zks_runner* runner, zk_order_rec* out, int out_cap, int32_t* out_offsets, zk_order_rec* army, int army_cap,
                int32_t* army_offsets);
/* counts[0]: orders in all, counts[1]: the army script's. */
void zks_last_counts(const zks_runner* runner, int32_t* counts);
/* The last step's orders by part: 4 per listed seat, pre, army script, post, economy script. */
void zks_seat_counts(const zks_runner* runner, int32_t* counts);

/* Measurements. */
/* made: n_roles, the army script's Mix.made.  Returns whether the side has an army script state. */
int zks_made(const zks_runner* runner, int game, int side, int32_t* made);
/* Economy.checks ["eligible", "stalled"] then the counts of Economy.finished for staticarty, turretantiheavy,
 * staticmissilesilo, staticnuke, energyfusion, singularity, missile.  9 values. */
void zks_eco_stats(const zks_runner* runner, int game, int side, int32_t* out);
/* Economy.log and Aimer.log: (seconds, what, row): what 0 "start <row>", 1 "missile at <row>", 2 "nuke at <row>".  Returns
 * how many there are; writes at most cap. */
int zks_log(const zks_runner* runner, int game, int side, double* seconds, int32_t* what, int32_t* row, int cap);

/* Random numbers for a parity check: the economy of this seat draws these, in order, at its next step, rather than its own.
 * kinds: 0 uniform in [0, 1), 1 standard normal.  A draw of the wrong kind, or past the end, comes from its own generator and
 * counts as a miss (zks_tape_misses).  The tape is dropped after the step; values left over count as misses too. */
void zks_tape(zks_runner* runner, int game, int side, const double* values, const uint8_t* kinds, int n);
int zks_tape_misses(const zks_runner* runner, int game, int side);

/* For tests: CPython's math.hypot as done here. */
double zks_py_hypot(double a, double b);

#ifdef __cplusplus
}
#endif
#endif
