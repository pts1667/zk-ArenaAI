/* C interface of the simplified simulator.  sim/py/zksim.py mirrors these structs. */
#ifndef ZKSIM_CAPI_H
#define ZKSIM_CAPI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ZK_ABI_VERSION 4
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
    ZK_W_UNMODELLED = 2048    /* what it's for isn't in the sim: capture, spawning, pushing, throwing */
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
    float vel_threshold, vel_base, vel_scale, alpha_damage;
    /* Ground fire that keeps burning where the shot landed (weapon_area_damage.lua). */
    float area_dps, area_radius, area_plateau, area_range_fall, area_plateau_fall, area_time_fall, area_frames;
    /* Setting units on fire (unit_is_on_fire.lua). */
    float burn_frames, burn_random, burn_chance, burn_damage;
    int32_t kind, flags, shield_intercept, intercepted_by;
} zk_weapon_rec;

enum { ZK_U_MOBILE = 1, ZK_U_AIR = 2, ZK_U_ARMED = 4, ZK_U_FIREPROOF = 8 };

typedef struct {
    float cost, hp, speed, accel, brake, turn_rate, slow_turn_speed;
    float footprint_radius, hit_radius, armored_mult, auto_heal, max_range;
    float los_radius, air_los_radius, radar_radius, los_height;
    float stockpile_frames;      /* time per stockpiled shot, 0 for none */
    float reload_move_penalty;   /* speed multiplier while the first weapon reloads, 1 for none */
    int32_t def_id;   /* the engine's, as raw frames carry it */
    int32_t armor_class, move_class, category, flags;
    int32_t explode_weapon, n_mounts;
    int32_t mount_weapon[ZK_MAX_MOUNTS];
    int32_t mount_only[ZK_MAX_MOUNTS];
    int32_t mount_bad[ZK_MAX_MOUNTS];
} zk_type_rec;

enum { ZK_BEHAVIOUR_ADVANCE = 0, ZK_BEHAVIOUR_HOLD = 1, ZK_BEHAVIOUR_ORDERS = 2 };
/* The same numbers as the raw frame's action kinds. */
enum { ZK_ORDER_NONE = 0, ZK_ORDER_MOVE = 1, ZK_ORDER_FIGHT = 2, ZK_ORDER_ATTACK = 3, ZK_ORDER_MANUAL_FIRE = 6 };

typedef struct {
    float x, z, vx, vz, heading, speed, hp, damage_taken, para, shield, disarm;
    int32_t alive, order, target, stock;
} zk_unit_state;

int zk_abi_version(void);
int zk_sizeof_weapon(void);
int zk_sizeof_type(void);
int zk_sizeof_state(void);

void* zk_catalog_new(const zk_type_rec* types, int n_types, const zk_weapon_rec* weapons, int n_weapons,
                     const float* priority /* n_weapons x n_types */);
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
/* Sets the frame number, which only the slow-update phase of units depends on. */
void zk_world_set_frame(void* world, int frame);
void zk_world_get(const void* world, int unit, zk_unit_state* out);
/* Weapons start reloaded, as scripted scenarios need; replays keep the random phase. */
void zk_world_ready_weapons(void* world);
/* Kills a unit at the next tick's end, with its death explosion. */
void zk_world_kill(void* world, int unit);

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
     records[(i * 2 + side) * cap ...], counts[i * 2 + side], globs[(i * 2 + side) * G], stats[(i * 2 + side) * ZK_STATS]. */
void zk_batch_step(void* batch, const int* games, int n_games, const zk_order_rec* orders, const int* order_offsets, int ticks,
                   void* records, int cap, int* counts, float* globs, float* stats);

#ifdef __cplusplus
}
#endif
#endif
