/* The scripted players of tools/policy_sim_env.py in C++: the economy script and the aimer, the army script and its production.
 * They were ported from Python modules, which were the specification and are in git at 8e9bca7 (tools/policy_eco.py,
 * tools/policy_army.py, tools/policy_mix.py); policy_eco.py and policy_mix.py now hold only the tables built for them.
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

#define ZKS_ABI_VERSION 11
#define ZKS_MAX_TEAMS 8              /* ZK_MAX_TEAMS: the most teams a game has room for */
#define ZKS_MOVE_CLASSES 6
#define ZKS_ROLE_KNOBS 7             /* raid, riot, assault, skirm, arty, aa, snipe: the roles ROLE_ORDER of policy_mix.py has */
#define ZKS_GROUP_RADII 10           /* zkbot.policy.groups.GROUP_RADII */
#define ZKS_GROUP_MARK 100           /* in the army script's orders as recorded, after a group's anchor's: x the radius, target the roles */

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
    ZKS_SINGULARITY = 1 << 17,   /* energy of 100 or more: EcoData.finished["singularity"] */
    ZKS_QUEEN = 1 << 18          /* the chicken queen, in either form (rules.json's chickens: each difficulty's queen and queen_morph) */
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
    const float* half_x;            /* EcoData.half: half its footprint along x and along z, in elmos.  The game tests a build */
    const float* half_z;            /* against the footprint's rectangle (CGameHelper::TestUnitBuildSquare), not a circle */
    const float* height_dif;        /* Footing.height_dif: how far the ground under it may be from the ground at its middle, 0 for any */
    const int32_t* footing;         /* index of the (low, high) water depths of its Footing, -1 for none */
    const int32_t* best_missile;    /* ScriptData.best_missile, -1 for none */
    const int32_t* generators;      /* rows x 3: EcoData.generators, -1 for none */
    const int32_t* option_offsets;  /* rows + 1: EcoData.options of each row are options[offsets[r]:offsets[r + 1]], sorted */
    const int32_t* options;
    int n_roles;
    const int32_t* menu;            /* rows x n_roles: Mix.menu of a factory row, -1 where there's none */
    const int32_t* plate_of;        /* a factory row's plate row (the catalog's plate_parent, EcoData.plate_of), -1 for none */
    /* Role indices, -1 where the table has none: raid, riot, assault, skirm, arty, aa, snipe, scout, builder, bomb */
    int32_t role_ids[10];
    /* Special rows, -1 where absent: staticarty, turretantiheavy, staticmissilesilo, staticnuke, energyfusion, energypylon, staticradar */
    int32_t special[7];
    /* What the second factory may be, -1 where absent: factoryplane, factorygunship, striderhub */
    int32_t second_factories[3];
    int32_t caretaker;              /* staticcon's row, -1 where absent */
    /* The radii a group order may have, ascending (zkbot.policy.groups.GROUP_RADII).  Role bits are `role` indices. */
    double group_radii[ZKS_GROUP_RADII];
    double cerberus_reach;          /* EcoData.reach["staticarty"] */
    double missile_reach;           /* EcoData.missile_reach */
    float spot_half;                /* EcoData.spot_half: half the square _taken keeps clear at each metal spot */
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
    const float* height;            /* rows x cols: Ground.heights, the ground at each cell */
} zks_map_desc;

/* The army script's knobs for one side: what a variant changes (tools/policy_variants.py).  The standard script is every knob
 * at its standard value: each role_weight 1, commit 1, gather 1, patrol_share 0.25, patience 1, retreat_below 0.4. */
typedef struct {
    double role_weight[ZKS_ROLE_KNOBS];  /* multiplies that role's share of its factory's mix */
    double commit;          /* a squad goes at a target whose cover is at most its metal / commit: more than 1 is more cautious */
    double gather;          /* the army waits for the group metal times this before it goes: less than 1 goes sooner */
    double patrol_share;    /* of the raiders past the first few, the share that patrols our extractors */
    double patience;        /* scales how long a squad keeps at a target: getting there, and killing it once there */
    double retreat_below;   /* a regenerating fighter below this share of its health goes back to the rally until it's at 90% */
} zks_style;

typedef struct zks_data zks_data;
typedef struct zks_runner zks_runner;

int zks_abi_version(void);
zks_data* zks_data_new(const zks_data_desc* desc);
/* Returns the map's index. */
int zks_data_add_map(zks_data* data, const zks_map_desc* map);
void zks_data_free(zks_data* data);

/* threads: of the runner's own pool, the calling thread included.  teams: slots a game, 1 to ZKS_MAX_TEAMS, as the batch's
 * arrays have (zk_batch_new's teams); a seat is game * teams + team.  The data must outlive the runner. */
zks_runner* zks_runner_new(const zks_data* data, int games, int threads, int teams);
void zks_runner_free(zks_runner* runner);

/* A new game in this slot, of n_teams teams (at most the runner's teams; the slots past them stay empty).
 * ally_of_team: each team's allyteam, or NULL for each its own.  Allies' units are never own and never enemies; every other
 *   team's are enemies.
 * script_teams: bit t set if team t is a player the scripts may run.  A team without it (the chickens) has no seat and gets no
 *   orders, but is an enemy like any other; its start isn't taken for a base.  A game with such a team is a chicken game: the
 *   army script guards (army.cpp, "Chicken defense") and reads the game's zks_step_args.chickens.
 * starts: x, z per team, its base.  rally: x, z per team, its army's rally.  economy: whether the economy script runs for every
 *   player.  aimer: whether the army script aims missiles and the Trinity.  seeds: each team's economy's own random numbers.
 *   styles: each team's army script knobs (n_teams zks_style), or NULL for the standard script. */
void zks_reset(zks_runner* runner, int game, int map, int n_teams, const int32_t* ally_of_team, uint32_t script_teams,
               const double* starts, const double* rally, int economy, int aimer, const uint64_t* seeds, const zks_style* styles);

enum {
    ZKS_SEAT_ARMY = 1,          /* the army script plays this seat */
    ZKS_SEAT_ARMY_LABEL = 2,    /* the army script runs for this seat, but only advises: its orders go to `army` (the labels),
                                   not to `out`, and it gives no group orders.  For the policy's seat, to learn from (DAgger) */
    ZKS_SEAT_ECO_LABEL = 4      /* the economy script only advises: its orders are only in zks_eco_orders, not in `out` */
};

/* T below is the runner's teams. */
typedef struct {
    int n;                          /* listed games */
    const int32_t* games;           /* n: slots */
    /* Indexed by slot: the batch's arrays (zksim.Batch.records, counts, globs) */
    const void* records;            /* slots x T x cap UnitRecord */
    int cap;
    const int32_t* counts;          /* slots x T */
    const float* globs;             /* slots x T x G_COUNT */
    const float* stats;             /* slots x T x ZK_STATS (zksim.Batch.stats), for reclaim's share of income; NULL for none */
    const int32_t* decisions;       /* slots */
    /* Indexed by listed seat, T k + team */
    const int32_t* seat_flags;      /* ZKS_SEAT_* */
    const int32_t* busy;            /* the commander's uid while the opening has it, else -1 */
    int ticks, tick_frames, frames_per_second;  /* game seconds = decisions * ticks * tick_frames / frames_per_second */
    int script_every;               /* the army script runs when decisions % script_every == 0 */
    int eco_every;                  /* the economy script when decisions % eco_every == 0 */
    /* Orders from elsewhere, per listed seat (T n + 1 offsets): `pre` goes before the script's (the policy's), `post` after
     * (the opening's).  The economy script's come last.  They're passed on for any seat, one the scripts don't play too. */
    const zk_order_rec* pre;
    const int32_t* pre_offsets;
    const zk_order_rec* post;
    const int32_t* post_offsets;
    /* 1: the army script sends its fighters' plans as group orders (zk_group_rec) wherever a group order gives exactly the units
     * it means and no others; 0: every order goes to its unit. */
    int group_orders;
    /* What each team sees to reclaim, indexed by slot as records are (zksim.Batch.features, feature_counts): slots x T x
     * feature_cap FeatureRecord.  NULL for none: then the economy script doesn't reclaim. */
    const void* features;
    int feature_cap;
    const int32_t* feature_counts;  /* slots x T */
    /* Chicken defense as every player sees it (Zero-K shows the queen's countdown and the chickens' anger): slots x
     * ZK_CHICKEN_STATE, each slot's zk_batch_chicken_state.  Read only for a chicken game.  NULL, or a row whose ZK_CS_TEAM
     * isn't the game's team outside script_teams: the anger taken as 0 and the queen as far off. */
    const float* chickens;
} zks_step_args;

/* Computes every listed seat's orders, then writes them as zk_batch_step takes them, if they fit:
 *   out, out_offsets (n + 1): per listed game, team 0, then team 1, ..., each pre, army script, post, economy script, leaving
 *     out the army script's orders that go in a group order;
 *   army, army_offsets (T n + 1): the army script's orders alone, per listed seat, as the policy would give them: those that go
 *     alone, and each group order as its anchor's order followed by a record of kind ZKS_GROUP_MARK for the same unit, whose x is
 *     the radius and target the roles (the other units the group gives the order to have none);
 *   groups, group_offsets (n + 1): per listed game, team 0, then team 1, ..., the army script's group orders.
 * Returns the number of orders in `out`.  If they're more than out_cap, the army's more than army_cap or the groups more than
 * group_cap, nothing is written: call zks_collect with room enough (zks_last_counts says how much). */
int zks_step(zks_runner* runner, const zks_step_args* args, zk_order_rec* out, int out_cap, int32_t* out_offsets,
             zk_order_rec* army, int army_cap, int32_t* army_offsets, zk_group_rec* groups, int group_cap, int32_t* group_offsets);
/* The last step's orders again, into buffers that fit. */
int zks_collect(zks_runner* runner, zk_order_rec* out, int out_cap, int32_t* out_offsets, zk_order_rec* army, int army_cap,
                int32_t* army_offsets, zk_group_rec* groups, int group_cap, int32_t* group_offsets);
/* counts[0]: orders in `out`, counts[1]: the army script's, counts[2]: group orders. */
void zks_last_counts(const zks_runner* runner, int32_t* counts);
/* The last step's orders by part: 4 per listed seat, pre, army script, post, economy script. */
void zks_seat_counts(const zks_runner* runner, int32_t* counts);
/* The last step's economy-script orders alone, per listed seat (T n + 1 offsets), as they are in `out`.  Returns how many there
 * are; writes them only if they fit in cap. */
int zks_eco_orders(const zks_runner* runner, zk_order_rec* out, int cap, int32_t* offsets);

/* Measurements, by game and team. */
/* made: n_roles, the army script's Mix.made.  Returns whether the team has an army script state. */
int zks_made(const zks_runner* runner, int game, int team, int32_t* made);
/* Economy.checks ["eligible", "stalled"] then the counts of Economy.finished for staticarty, turretantiheavy,
 * staticmissilesilo, staticnuke, energyfusion, singularity, missile.  9 values. */
void zks_eco_stats(const zks_runner* runner, int game, int team, int32_t* out);
/* Economy.log and Aimer.log: (seconds, what, row): what 0 "start <row>", 1 "missile at <row>", 2 "nuke at <row>".  Returns
 * how many there are; writes at most cap. */
int zks_log(const zks_runner* runner, int game, int team, double* seconds, int32_t* what, int32_t* row, int cap);

/* Random numbers for a parity check: the economy of this seat draws these, in order, at its next step, rather than its own.
 * kinds: 0 uniform in [0, 1), 1 standard normal.  A draw of the wrong kind, or past the end, comes from its own generator and
 * counts as a miss (zks_tape_misses).  The tape is dropped after the step; values left over count as misses too. */
void zks_tape(zks_runner* runner, int game, int team, const double* values, const uint8_t* kinds, int n);
int zks_tape_misses(const zks_runner* runner, int game, int team);

/* For tests: CPython's math.hypot as done here. */
double zks_py_hypot(double a, double b);

#ifdef __cplusplus
}
#endif
#endif
