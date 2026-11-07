/* C interface of the simplified simulator.  sim/py/zksim.py mirrors these structs. */
#ifndef ZKSIM_CAPI_H
#define ZKSIM_CAPI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ZK_ABI_VERSION 15
#define ZK_MAX_MOUNTS 8
#define ZK_ARMOR_CLASSES 5
#define ZK_ARMOR_SHIELD 4
#define ZK_TICK_FRAMES 5

enum { ZK_WEAPON_INSTANT = 0, ZK_WEAPON_PROJECTILE = 1, ZK_WEAPON_GUIDED = 2, ZK_WEAPON_SHIELD = 3 };

enum {
    ZK_W_IMPACT_ONLY = 1,
    ZK_W_TURRET = 2,
    ZK_W_PARALYZER = 4,
    ZK_W_NO_SELF_DAMAGE = 8,
    ZK_W_NO_AUTO_TARGET = 16,
    ZK_W_NO_FRIENDLY_FIRE = 32,
    ZK_W_SLOW_ONLY = 64,
    ZK_W_AREA_SLOWS = 128,
    ZK_W_DISARM_ONLY = 256,   /* no health damage (disarm_defs.lua normalDamage) */
    ZK_W_MANUAL = 512,        /* fires only on a manual-fire order */
    ZK_W_STOCKPILE = 1024,    /* each shot uses one from the unit's stock */
    ZK_W_UNMODELLED = 2048,   /* what it's for isn't in the sim: capture, spawning, an area that pulls */
    ZK_W_UNLINKED = 4096,     /* a shield that shares charge with nothing (unit_shield_link.lua) */
    ZK_W_HIGH_ARC = 8192,     /* a cannon that takes the higher of its two paths */
    ZK_W_BEAM = 16384,        /* burns for some frames, so that a shield can stop part of it */
    ZK_W_OKP_STATIC_ONLY = 32768,   /* overkill prevention counts its shots only against known buildings */
    ZK_W_OKP_FLIGHT = 65536,        /* ... and takes the flight time for how long they take */
    ZK_W_OKP_TARGET_SPEED = 131072, /* ... or that, plus a margin for a target's speed */
    ZK_W_SCRIPT_BURST = 262144,     /* its salvo is the script firing shot by shot, so overkill prevention looks at each */
    ZK_W_PIERCES = 524288,          /* noExplode: flies on to the end of its range and hurts what it goes through, on every frame it's inside */
    ZK_W_ONE_HIT_EACH = 1048576,    /* ... but once per unit (weapon_single_hit.lua) */
    ZK_W_THROWS = 2097152           /* fired by a manual-fire order, it throws what's around the unit (unit_thrower.lua) */
};

/* Times are in frames and distances in elmos. */
typedef struct {
    float damage[ZK_ARMOR_CLASSES];
    float range, reload, salvo, salvo_delay, projectiles;
    float speed, accel, start_speed;
    float blast_radius, edge, height_mod;
    float spray, accuracy, moving_accuracy;
    float para_seconds, extra_para, slow_factor;
    /* Disarm: as stun, but only weapons stop (disarm_defs.lua, unit_boolean_disable.lua). */
    float disarm_mult, disarm_penalty, disarm_time;
    /* Stun and disarm can run this much past the weapon's own time, near the threshold. */
    float overstun_seconds;
    /* Slow level allowed above the cap: it lasts longer, not slows more (timeslow_defs.lua). */
    float overslow;
    float shield_cost;   /* charge each shot takes from the unit's own shield */
    float dyn_exp, dyn_range;   /* damage falling with distance, 0 exponent for none (DamageArray.cpp) */
    float climb_frames;         /* a starburst missile rises this long before it turns to the target */
    float shield_power, shield_regen, shield_radius;
    float shield_energy;        /* energy a frame while it charges */
    float arc;                  /* gravity over speed squared, for a cannon; 0 for what flies straight */
    float vel_threshold, vel_base, vel_scale, alpha_damage;
    /* Ground fire that keeps burning where the shot landed (weapon_area_damage.lua). */
    float area_dps, area_radius, area_plateau, area_range_fall, area_plateau_fall, area_time_fall, area_frames;
    /* Setting units on fire (unit_is_on_fire.lua). */
    float burn_frames, burn_random, burn_chance, burn_damage;
    /* Overkill prevention, as the unit's script asks for it (unit_overkill_prevention.lua).  0 damage for none. */
    float okp_damage, okp_frames, okp_full_frames, okp_hit_mod, okp_min_range, okp_fast_mult, okp_radar_mult;
    float radar_wobble_penalty, radar_dot_penalty;   /* target_priority_defs.lua: what firing at a radar dot costs a target's rating */
    float impulse;              /* what a hit pushes with, away from the shooter, or towards it if negative (weapon_impulse.lua) */
    int32_t kind, flags, shield_intercept, intercepted_by;
} zk_weapon_rec;

enum { ZK_U_MOBILE = 1, ZK_U_AIR = 2, ZK_U_ARMED = 4, ZK_U_FIREPROOF = 8, ZK_U_FACTORY = 16, ZK_U_PLOPPABLE = 32, ZK_U_COMMANDER = 64,
       ZK_U_CAN_CLOAK = 128,           /* its own cloak, which it can be told to use */
       ZK_U_STARTS_CLOAKED = 256,      /* wants to from the start */
       ZK_U_FIRES_CLOAKED = 512,       /* firing doesn't show it */
       ZK_U_AREA_CLOAK_ON = 1024,      /* its area cloak starts on */
       ZK_U_KAMIKAZE_BLIND = 4096,     /* as a bomb, needs no sight of what's near it to go off */
       ZK_U_FLOATS = 8192,             /* stands on the water rather than the sea bed */
       ZK_U_MISSILE = 16384,           /* a silo's: gone once it has fired (cruisemissile.lua) */
       ZK_U_NEUTRAL = 32768,           /* nothing shoots at it unless told to: a teleport beacon */
       ZK_U_LEAVES_GHOST = 2048 };     /* a building: once seen, radar shows just where it is */

typedef struct {
    float cost, hp, speed, accel, brake, turn_rate, slow_turn_speed;
    float footprint_radius, hit_radius, armored_mult, max_range;
    float idle_regen, idle_time;  /* health per frame, once this many frames have passed without damage */
    float los_radius, air_los_radius, radar_radius, los_height;
    float stockpile_frames;      /* time per stockpiled shot, 0 for none */
    float reload_move_penalty;   /* speed multiplier while the first weapon reloads, 1 for none */
    /* Construction: progress per second is build_speed / build_time, and each step costs its share of both resources. */
    float metal_cost, energy_cost, build_time, build_speed, build_range;
    float metal_make, energy_make;   /* per second once finished, as long as it lives */
    float extract_mult;          /* a finished one earns this times the income of the metal spot it stands on */
    float pylon_range;           /* two of these share an overdrive grid if closer than the sum of their ranges, 0 for neither */
    float reclaim_speed;         /* metal per second it takes from a wreck */
    /* What dying leaves, by the recent damage that killed it: a wreck for half its health or less, else a heap.  0 metal for nothing. */
    float wreck_metal, wreck_hp, heap_metal, heap_hp;
    float energy_upkeep;         /* per second, or it stops working */
    float stockpile_cost;        /* metal and energy for each stockpiled shot, paid as it's made */
    float close_frames;          /* shut for this long after a hit, makes nothing and takes armored_mult of damage.  0 for what doesn't */
    float cloak_cost, cloak_cost_moving;   /* energy per second */
    float decloak_distance;      /* an enemy this close shows it */
    float bestowed_decloak;      /* the same, when another unit's area cloak hides it.  0 for the same distance */
    float area_cloak_radius, area_cloak_upkeep;   /* its own area cloak, and the energy per second it adds to the upkeep */
    float perceived_range;       /* what the tactical AI thinks its reach is, when running from it */
    float heat_per_shot, heat_decay, heat_max_slow, heat_initial;   /* overheating, decay per frame; 0 per shot for none */
    float bait_cost;             /* its build time, for what won't shoot it while it's a cheap frame */
    float kamikaze_distance;     /* a bomb goes off within this of its target; 0 for anything else */
    float detonate_reach;        /* Blastwing: at top speed it goes off this far from what it wants to hit, and less the slower it goes */
    float drop_height, drop_gravity;   /* what it drops its bomb from, and how fast that falls */
    /* Jumping (unit_jumpjets.lua).  0 range for what doesn't.  speed: along the arc.  reload and delay: frames. */
    float jump_range, jump_speed, jump_height, jump_reload, jump_delay;
    float burrow_speed;          /* it digs in and cloaks below this speed; 0 for what doesn't */
    float mass;                  /* for falling, and how long it takes to teleport */
    float push_scale;            /* speed per unit of impulse, along the ground or through the air */
    float needed_link;           /* its weapons need a grid that makes this much energy (unit_mex_overdrive.lua); 0 for none */
    float throw_radius;          /* what its throwing weapon gathers (unit_thrower.lua) */
    float teleport_rate, beacon_frames;   /* Djinn: mass per frame it brings over, and frames to make a beacon */
    int32_t move_state, fire_state;   /* to start with: hold position 0, maneuver 1, roam 2; hold fire 0, return fire 1, fire at will 2 */
    int32_t no_chase;                 /* categories it doesn't go after on its own */
    int32_t bait_level;               /* unit_bait_prevention.lua: 0 shoots at anything */
    int32_t bait_target, bait_target_armor;   /* the level from which others won't shoot this, and won't while it's armoured.  0 for none */
    int32_t def_id;   /* the engine's, as raw frames carry it */
    int32_t wreck_feature, heap_feature;   /* the engine's feature def ids */
    int32_t armor_class, move_class, category, flags;
    int32_t explode_weapon, n_mounts;
    int32_t landing_weapon;      /* goes off where it lands from a jump, or -1 */
    int32_t drop_weapon;         /* dropped when it dies, or -1 */
    int32_t missile_pads;        /* a silo's, which its missiles stand on and wait for */
    int32_t beacon_type;         /* what a teleporter makes */
    int32_t mount_weapon[ZK_MAX_MOUNTS];
    int32_t mount_only[ZK_MAX_MOUNTS];
    int32_t mount_bad[ZK_MAX_MOUNTS];
} zk_type_rec;

/* unit_tactical_ai.lua's table for one unit type, after its defaults.  Distances in elmos, times in frames.  A negative stands for what it leaves out. */
enum { ZK_T_HANDLED = 1, ZK_T_IDLE_ONLY = 2, ZK_T_SKIRM_RADAR = 4, ZK_T_SKIRM_EVERYTHING = 8, ZK_T_KEEP_ORDER = 16, ZK_T_FLEE_EVERYTHING = 32,
       ZK_T_FLEE_RADAR = 64, ZK_T_FLEE_COMBAT = 128, ZK_T_IDLE_FLEE_COMBAT = 256 };
/* How a type regards each type of enemy: types x types of these. */
enum { ZK_R_SKIRM = 1, ZK_R_FLEE = 2, ZK_R_IDLE_FLEE = 4, ZK_R_HUG = 8, ZK_R_FIGHT_ONLY = 16, ZK_R_BONUS_RANGE = 32, ZK_R_AVOID_HEIGHT = 64 };

typedef struct {
    float search_range, idle_push_dist_sq, idle_chase_leeway, idle_commit_dist, idle_commit_mult, idle_enemy_mult;
    float flee_order_dis, flee_leeway, flee_distance, min_flee_range, flee_vel_prediction;
    float weapon_range, stopping_distance, velocity_prediction, skirm_order_dis, skirm_order_dis_min, hug_range, skirm_leeway;
    float skirm_range_override, reload_skirm_leeway, skirm_blocked_frames, vel_pred_chase, skirm_keep_leeway, skirm_only_near, my_speed;
    int32_t flags, weapon_num, blocked_on_fight;
} zk_tactics_rec;

/* ORDERS is a game's: fog, and units act on their own between orders (unit_ai.cpp).  SCRIPTED does only what it's told and sees everything,
   for measuring one mechanic at a time. */
enum { ZK_BEHAVIOUR_ADVANCE = 0, ZK_BEHAVIOUR_HOLD = 1, ZK_BEHAVIOUR_ORDERS = 2, ZK_BEHAVIOUR_SCRIPTED = 3 };
/* The same numbers as the raw frame's action kinds. */
enum { ZK_ORDER_NONE = 0, ZK_ORDER_MOVE = 1, ZK_ORDER_FIGHT = 2, ZK_ORDER_ATTACK = 3, ZK_ORDER_MANUAL_FIRE = 6,
       ZK_ORDER_JUMP = 7,       /* x, z: walks into range and goes when charged.  The order ends when it lands there. */
       ZK_ORDER_CLOAK_ON = 8, ZK_ORDER_CLOAK_OFF = 9,   /* Both a unit's own cloak and its area cloak.  Leave other orders alone. */
       ZK_ORDER_BUILD = 10,     /* target: the type to place at x, z.  The order ends when it's finished or lost. */
       ZK_ORDER_ASSIST = 11,    /* target: a unit.  A factory is helped with whatever it's building, until told otherwise. */
       ZK_ORDER_RECLAIM = 13,   /* target: a feature's id.  The order ends when it's all taken or gone. */
       ZK_ORDER_PRODUCE = 14,   /* target: the type a factory builds over and over, or -1 to stop.  Leaves other orders alone. */
       ZK_ORDER_MOVE_STATE = 15, ZK_ORDER_FIRE_STATE = 16,    /* target: the state.  Leave other orders alone. */
       ZK_ORDER_UNIT_AI = 17,    /* target: 0 turns off the tactical AI, as a player can, and 1 back on */
       ZK_ORDER_SELF_DESTRUCT = 18,    /* a bomb goes off now, unless it's stunned.  Nothing else does. */
       ZK_ORDER_PLACE_BEACON = 19,     /* x, z: a teleporter stops and makes its beacon there, replacing the one it has.  It deploys whenever it's standing with one. */
       ZK_ORDER_WAIT_AT_BEACON = 20,   /* target: a beacon of its side, or the teleporter it belongs to.  It walks there and waits its turn. */
       ZK_ORDER_ON_OFF = 21 };         /* target: 0 is off, 1 on.  What has both pulls when off and pushes when on.  Leaves other orders alone. */
/* ZK_ORDER_MANUAL_FIRE: a target, or for what throws, x, z with a target of -1. */

typedef struct {
    float x, z, vx, vz, heading, speed, hp, damage_taken, para, shield, disarm, build;
    float heat;
    float jump_charge;     /* 1 when it can jump */
    float altitude;        /* off the ground: jumping, thrown or pushed */
    int32_t alive, order, target, stock, cloaked;
    int32_t chasing;       /* the enemy its fight order or idleness has it attacking, or -1 */
    int32_t dodging;       /* the tactical AI has it stepping away */
    int32_t shots_fired;
    int32_t jumping;       /* 1 waiting to take off, 2 in the air, thrown as well */
    int32_t skidding;      /* pushed, and not walking until it stops */
    int32_t low_power;     /* its grid can't run its weapons */
    int32_t deployed;      /* a teleporter that's ready to bring things over */
    int32_t beacon;        /* a teleporter's, or -1 */
    int32_t pulling;       /* what has both is pulling */
} zk_unit_state;

int zk_abi_version(void);
int zk_sizeof_weapon(void);
int zk_sizeof_type(void);
int zk_sizeof_state(void);

void* zk_catalog_new(const zk_type_rec* types, int n_types, const zk_weapon_rec* weapons, int n_weapons,
                     const float* priority /* n_weapons x n_types */,
                     const zk_tactics_rec* tactics /* n_types */, const uint8_t* relations /* n_types x n_types */);
void zk_catalog_free(void* catalog);

/* height, slope, normal_x, normal_z: rows x cols.  speed: classes x rows x cols, the multiplier
   CMoveMath gives without a direction, 0 impassable.  slope_mod: one per class. */
void* zk_map_new(int cols, int rows, float cell, const float* height, const float* speed, int classes,
                 const float* slope, const float* normal_x, const float* normal_z, const float* slope_mod,
                 const short* segments);
// Per class, after zk_map_new.  Positions in elmos, as x, z pairs; next_edge is nodes x nodes.
void zk_map_set_routes(void* map, int move_class, int nodes, const float* centres, int edges, const int* edge_to,
                       const float* gates, const int* next_edge);
void zk_map_free(void* map);

void* zk_world_new(const void* catalog, const void* map, uint64_t seed, int behaviour);
void zk_world_free(void* world);
int zk_world_add(void* world, int type, int side, float x, float z, float heading, float hp_frac, float build_frac);
void zk_world_order(void* world, int unit, int kind, float x, float z, int target);
void zk_world_step(void* world, int ticks);
int zk_world_frame(const void* world);
int zk_world_count(const void* world);
/* Sets the frame number, which only the slow-update phase of units depends on. */
void zk_world_set_frame(void* world, int frame);
void zk_world_get(const void* world, int unit, zk_unit_state* out);
/* Weapons start reloaded, as scripted scenarios need; replays keep the random phase. */
void zk_world_ready_weapons(void* world);
/* Kills a unit at the next tick's end, with its death explosion. */
void zk_world_kill(void* world, int unit);
void zk_world_set_shield(void* world, int unit, float charge);
/* Health, which may be more than its type has: for scenarios that want to see what a target would take. */
void zk_world_set_hp(void* world, int unit, float hp);
/* Turns on resources and construction, which a world otherwise doesn't run, with both sides starting alike.
   spots: x, z, income for each of n_spots.  Incomes are per second. */
void zk_world_economy(void* world, int n_spots, const float* spots, float metal, float energy, float innate_metal, float innate_energy, float storage);
/* metal, energy, metal income, energy income, overdrive metal, overdrive energy */
void zk_world_resources(const void* world, int side, float* out);
/* A reclaimable feature that no unit left, such as a map's rock.  metal, hp: as a feature def has them; radius: how near a builder must be, past its range.
   Returns its id. */
int zk_world_add_feature(void* world, float x, float z, float metal, float hp, float radius, int feature_def);
/* What this side has in sight now, as the raw frame has them.  Returns the number of records. */
int zk_world_features(void* world, int side, void* records, int cap);

/* Replays many snapshots across threads.  Entities are flat: snapshot i owns
   offsets[i]..offsets[i+1].  tracked marks the units counted as losses.  horizons are in
   ticks, ascending.  out: n_snapshots x n_seeds x n_horizons x 2, losses of sides 0 and 1. */
void zk_replay(const void* catalog, const void* const* maps, const int* snapshot_map, int n_snapshots,
               const int* offsets, const int* type, const int* side, const float* x, const float* z,
               const float* hp_frac, const float* build_frac, const uint8_t* tracked, int n_seeds, uint64_t seed,
               const int* horizons, int n_horizons, int behaviour, int threads, float* out);

/* Raw frames.  zk_order_rec is one order; batches take them per game, in the order the games are listed. */
typedef struct {
    int32_t unit, kind;
    float x, z;
    int32_t target;
} zk_order_rec;

#define ZK_STATS 2 /* per side: tracked value, live units */

int zk_sizeof_unit_record(void);
int zk_sizeof_site_record(void);
int zk_sizeof_feature_record(void);
int zk_global_count(void);

/* One world's frame for one side, computed now.  Returns the number of records. */
int zk_world_view(void* world, int side, void* records, int cap, float* glob, int commander, const float* base);

/* Many games stepping across threads, each thread owning a fixed slice, so results don't depend on the thread count. */
void* zk_batch_new(const void* catalog, int games, int threads);
void zk_batch_free(void* batch);
/* Starts game `g` afresh.  commander: per side, or -1.  base: x, z per side. */
void zk_batch_reset(void* batch, int g, const void* map, uint64_t seed, int behaviour, int n, const int* type, const int* side,
                    const float* x, const float* z, const float* heading, const float* hp_frac, const float* build_frac,
                    const int* commander, const float* base);
void* zk_batch_world(void* batch, int g);
/* Applies orders, steps `ticks`, and writes both sides' frames.  For the i-th game listed:
     orders[order_offsets[i] .. order_offsets[i + 1]] are its orders;
     records[(i * 2 + side) * cap ...], counts[i * 2 + side], globs[(i * 2 + side) * G], stats[(i * 2 + side) * ZK_STATS],
     features[(i * 2 + side) * feature_cap ...], feature_counts[i * 2 + side]. */
void zk_batch_step(void* batch, const int* games, int n_games, const zk_order_rec* orders, const int* order_offsets, int ticks,
                   void* records, int cap, int* counts, float* globs, float* stats, void* features, int feature_cap, int* feature_counts);

#ifdef __cplusplus
}
#endif
#endif
