/* C interface of the simplified simulator.  sim/py/zksim.py mirrors these structs. */
#ifndef ZKSIM_CAPI_H
#define ZKSIM_CAPI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ZK_ABI_VERSION 21
#define ZK_MAX_MOUNTS 8
#define ZK_MAX_TEAMS 8     /* players in a game, and so allyteams too */
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
    ZK_W_THROWS = 2097152,          /* fired by a manual-fire order, it throws what's around the unit (unit_thrower.lua) */
    ZK_W_BOMB = 4194304,            /* dropped from a plane: falls from it, let go where it will land on the target (BombDropper.cpp).
                                       One that does no damage only marks a bomber's target */
    ZK_W_DROPPED = 8388608          /* fired by the script a shot at a time straight down, when the unit's marking bomb would drop */
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
    float gravity;              /* a bomb's fall, elmos a frame squared */
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
       ZK_U_PLANE = 65536,             /* flies on wings: can't stop in the air, and turns in a circle (air.cpp) */
       ZK_U_LANDS = 131072,            /* lands when it has nothing to do: an aircraft from a plane factory (unit_air_plants.lua) */
       ZK_U_STRAFES = 262144,          /* a gunship that circles what it attacks (airStrafe) */
       ZK_U_DONT_COUNT = 524288,       /* doesn't keep its team in the game (game_over.lua; sim/data/rules.json) */
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
    /* Aircraft (air.cpp).  cruise_height: wantedHeight.  vertical_speed: up and down, elmos a frame.  A plane's turn_rate is how
       fast it comes round in a hard turn, slowing to turn_speed; both measured (tools/sim_engine.py air). */
    float cruise_height, vertical_speed, turn_speed;
    float strafe_radius, strafe_speed;    /* a strafing gunship circles at this distance and speed */
    /* Bombers (unit_bomber_command.lua, unit_refuel_pad_handler.lua): frames to rearm on a pad, and the energy a frame that takes. */
    float reammo_frames, reammo_energy;
    float pad_bp;                /* a pad's repair power for what lands on it */
    /* Air transports (unit_transport_load_unload.lua, unit_transport_speed.lua).  lift_*: its speed, as a share of its own, while it
       carries a load of that class, and 0 for a class it can't lift: all three for what isn't a transport.  lift_size: the widest
       load it takes, as load_size is measured (the engine's transportSize against xsize). */
    float lift_light, lift_medium, lift_heavy, lift_size;
    float load_size, load_height;   /* as a load: its footprint, and how far under its transport it hangs (its model's height) */
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
    int32_t ammo;                /* salvos before it goes back to rearm; 0 for no ammo */
    int32_t pad_count;           /* planes it can rearm at once: an airpad or plane factory's */
    int32_t plate_parent;        /* a factory plate's factory type, or -1: it builds only within its build range of a finished one
                                    of its allyteam (unit_factory_plate.lua) */
    int32_t load_class;          /* as a load: 0 can't be carried, 1 light, 2 medium, 3 heavy (unitdefs_post.lua, by its cost) */
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
       ZK_ORDER_ASSIST = 11,    /* target: a unit of its allyteam.  A factory is helped with whatever it's building, until told otherwise. */
       ZK_ORDER_RECLAIM = 13,   /* target: a feature's id.  The order ends when it's all taken or gone. */
       ZK_ORDER_PRODUCE = 14,   /* target: the type a factory builds over and over, or -1 to stop.  Leaves other orders alone. */
       ZK_ORDER_MOVE_STATE = 15, ZK_ORDER_FIRE_STATE = 16,    /* target: the state.  Leave other orders alone. */
       ZK_ORDER_UNIT_AI = 17,    /* target: 0 turns off the tactical AI, as a player can, and 1 back on */
       ZK_ORDER_SELF_DESTRUCT = 18,    /* a bomb goes off now, unless it's stunned.  Nothing else does. */
       ZK_ORDER_PLACE_BEACON = 19,     /* x, z: a teleporter stops and makes its beacon there, replacing the one it has.  It deploys whenever it's standing with one. */
       ZK_ORDER_WAIT_AT_BEACON = 20,   /* target: a beacon of its allyteam, or the teleporter it belongs to.  It walks there and waits its turn. */
       ZK_ORDER_ON_OFF = 21,           /* target: 0 is off, 1 on.  What has both pulls when off and pushes when on.  Leaves other orders alone. */
       /* Air transports (transport.cpp). */
       ZK_ORDER_LOAD = 22,             /* target: a unit it can lift, of any team.  It flies there, comes down and picks it up; one of its
                                          own allyteam stands still for it, an enemy must be nearly still, in sight and not cloaked. */
       ZK_ORDER_UNLOAD = 23,           /* x, z: where what it carries can stand.  It flies there, comes down and sets it down unhurt. */
       ZK_ORDER_DROP = 24 };           /* lets go of what it carries, now, from where it is: it falls, and is hurt as a fall hurts.
                                          Ready again 7 s later.  Leaves other orders alone. */
/* ZK_ORDER_MANUAL_FIRE: a target, or for what throws, x, z with a target of -1. */

typedef struct {
    float x, z, vx, vz, heading, speed, hp, damage_taken, para, shield, disarm, build;
    float heat;
    float jump_charge;     /* 1 when it can jump */
    float altitude;        /* off the ground: flying, jumping, thrown or pushed */
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
    int32_t ammo;          /* salvos left, for what has ammo */
    int32_t air_state;     /* aircraft: 0 flying, 1 landed, 2 taking off, 3 landing, 4 on a pad rearming */
    int32_t carrier;       /* the transport that carries it, or -1 */
    int32_t cargo;         /* what it carries, or -1 */
} zk_unit_state;

int zk_abi_version(void);
int zk_sizeof_weapon(void);
int zk_sizeof_type(void);
int zk_sizeof_state(void);

void* zk_catalog_new(const zk_type_rec* types, int n_types, const zk_weapon_rec* weapons, int n_weapons,
                     const float* priority /* n_weapons x n_types */,
                     const zk_tactics_rec* tactics /* n_types */, const uint8_t* relations /* n_types x n_types */);
void zk_catalog_free(void* catalog);

/* Game rules from Zero-K's gadgets rather than its units (tools/sim_rules.py, sim/data/rules.json), given to a catalog once, before
   any world of it is made.  A catalog without them has no reveal rule and no chickens. */

/* game_over.lua: with two alliances left, one worth more than value_ratio times the other has the other's units always visible,
   looked at every every_frames.  Value: the metal cost of finished units that keep a team in the game. */
void zk_catalog_reveal(void* catalog, float value_ratio, int every_frames);

/* Chicken defense (unit_spawner.lua, spawn_defs.lua): one difficulty's settings as the gadget ends up with them, the difficulty's own
   over the defaults.  Times in seconds, as the gadget's formulas have them; distances in elmos. */
#define ZK_CHICKEN_KINDS 32
#define ZK_CHICKEN_MINI_QUEENS 4
#define ZK_CHICKEN_NO_TARGET 4
enum { ZK_CHICKEN_OFFENSE = 0,    /* what waves are made of */
       ZK_CHICKEN_DEFENSE = 1,    /* turrets a burrow gets from the defense pool */
       ZK_CHICKEN_SUPPORT = 2,    /* sent alongside a wave, by chance */
       ZK_CHICKEN_SPECIAL = 3 };  /* a special power: an ambush or a sprouting spire, when the players have been slow to kill burrows */
typedef struct {
    int32_t type, role;
    float time, obsolete;         /* from when it comes, and after when no longer (seconds, by the chickens' tech time) */
    float squad_size, cost;       /* cost: a defender's, from the defense pool */
    /* A special power's (a negative for what the gadget leaves to its fallbacks): */
    float max_aggro, count, burrow_ratio, min_dist, max_dist, cooldown;
    int32_t target_human, tie_to_burrow;
} zk_chicken_kind;
typedef struct {
    int32_t level;                /* 1 Beginner .. 6 Suicidal, as the gadget numbers them */
    int32_t burrow, queen, queen_morph, mini_queen, test_building, test_building_queen;   /* types; queen_morph -1 for none */
    int32_t no_target[ZK_CHICKEN_NO_TARGET];   /* types chickens aren't sent at, -1 past the last */
    int32_t max_tries, max_tries_small, n_mini_queen_times, n_kinds;
    float spawn_square, spawn_square_increment, max_burrows, min_base_distance, max_base_distance, max_age, egg_decay_time, burrow_eggs;
    float player_malus, burrow_spawn_rate, chicken_spawn_rate, wave_ratio, base_wave_size, wave_size_mult, queen_spawn_mult;
    float defense_per_wave, defense_per_burrow_kill, grace_period, grace_penalty, grace_period_min, ramp_up_time;
    float queen_time, queen_health_mod, queen_morph_frames[2], mini_queen_time[ZK_CHICKEN_MINI_QUEENS];
    float burrow_queen_time, burrow_wave_size, burrow_respawn_chance, burrow_regress_time;
    float human_aggro_per_burrow, human_aggro_decay, human_aggro_min, human_aggro_max, human_aggro_wave_factor, human_aggro_wave_max;
    float human_aggro_defense_factor, human_aggro_tech_time_progress, human_aggro_tech_time_regress;
    float human_aggro_queen_time_factor, human_aggro_queen_time_min, human_aggro_queen_time_max;
    float tech_accel_per_player, tech_time_floor_factor, tech_time_max, time_spawn_bonus;
    zk_chicken_kind kinds[ZK_CHICKEN_KINDS];
} zk_chicken_rules;
/* difficulty: 0 Beginner .. 5 Suicidal. */
void zk_catalog_chickens(void* catalog, int difficulty, const zk_chicken_rules* rules);
int zk_sizeof_chicken_rules(void);
/* Per type, the egg it leaves when it dies (features/chicken_eggs.lua): a reclaimable feature, feature -1 for none. */
typedef struct {
    int32_t feature;
    float metal, hp, radius;
} zk_egg_rec;
void zk_catalog_eggs(void* catalog, const zk_egg_rec* eggs /* n_types */);

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
/* Teams (players: a purse, orders, a frame of their own) and their allyteams (who share sight and don't fight each other).
   ally_of_team: n_teams allyteams, 0 to ZK_MAX_TEAMS - 1, or NULL for each team its own.  Until set: two teams, each its own
   allyteam.  Call before any unit is added. */
void zk_world_set_teams(void* world, int n_teams, const int32_t* ally_of_team);
/* Returns the unit's handle, or -1 for a team the game hasn't. */
int zk_world_add(void* world, int type, int team, float x, float z, float heading, float hp_frac, float build_frac);
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
/* Turns on resources and construction, which a world otherwise doesn't run, with every team starting alike.
   spots: x, z, income for each of n_spots.  Incomes are per second.  Inside an allyteam of several teams, resources are shared as
   unit_mex_overdrive.lua shares them: extractors' income is pooled and split evenly among the teams with a unit alive, overdrive
   runs over the allyteam's grids and its metal goes by the energy each team put in, and what's over a team's storage goes to
   allies with room. */
void zk_world_economy(void* world, int n_spots, const float* spots, float metal, float energy, float innate_metal, float innate_energy, float storage);
/* metal, energy, metal income, energy income, overdrive metal, overdrive energy: the team's */
void zk_world_resources(const void* world, int team, float* out);
/* A reclaimable feature that no unit left, such as a map's rock.  metal, hp: as a feature def has them; radius: how near a builder must be, past its range.
   Returns its id. */
int zk_world_add_feature(void* world, float x, float z, float metal, float hp, float radius, int feature_def);
/* What this team's allyteam has in sight now, as the raw frame has them, by id.  With more than cap in sight, the cap with the most
   metal left (ties to the older).  Returns the number of records. */
int zk_world_features(void* world, int team, void* records, int cap);

/* The reveal rule is on in a world whose catalog has it, but for a chicken game's: Zero-K doesn't apply it there.  0 turns it off. */
void zk_world_set_reveal(void* world, int on);

/* Chicken defense.  Call after the teams are set and before the next step.  `team` has its own allyteam and was given no units;
   from then on the world spawns its units and orders them itself, and every other team is a player.  difficulty: 0 Beginner ..
   5 Suicidal, which the catalog must have rules for. */
void zk_world_chickens(void* world, int team, int difficulty);
#define ZK_CHICKEN_STATE 12
enum { ZK_CS_QUEEN = 0,           /* 0 not yet, 1 alive, 2 dead: the players have won */
       ZK_CS_UNTIL_QUEEN = 1,     /* seconds until she comes, as of now; 0 once she has */
       ZK_CS_BURROWS = 2, ZK_CS_BURROWS_KILLED = 3,
       ZK_CS_CHICKENS = 4,        /* the chicken team's live units, burrows aside */
       ZK_CS_WAVES = 5,
       ZK_CS_AGGRO = 6,           /* the gadget's humanAggro: up for each burrow killed, down for each wave */
       ZK_CS_TECH = 7,            /* its totalTechAccel: seconds the chickens' kinds come early by */
       ZK_CS_QUEEN_TIME = 8,      /* the game second she comes at, which burrows killed bring forward */
       ZK_CS_QUEEN_HP = 9,        /* her health, of all of it, while she lives */
       ZK_CS_DIFFICULTY = 10, ZK_CS_TEAM = 11 };
void zk_world_chicken_state(const void* world, float* out /* ZK_CHICKEN_STATE; all 0 and team -1 for no chickens */);

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

/* A group order: one order (kind, x, z, target), as zk_order_rec gives it, to the anchor and to every other unit of its team that
   is alive, mobile and finished, has a role in `roles` (bit r for role r, as zk_batch_set_roles or zk_world_set_roles gave them),
   and stands within `radius` of the anchor, in float: dx * dx + dz * dz <= radius * radius.  Nothing, if the anchor is dead.  A
   game's group orders go before its unit orders, each in turn, so a later one replaces an earlier one for a unit it selects, and a
   unit order replaces any group's. */
typedef struct {
    int32_t anchor;
    uint32_t roles;
    float radius;
    int32_t kind;
    float x, z;
    int32_t target;
} zk_group_rec;

/* Each catalog type's role, 0 to 31, or -1 for none: what group orders select by.  Until set, no type has one. */
void zk_world_set_roles(void* world, const int32_t* role_of_type, int n_types);
/* Returns how many units it selected. */
int zk_world_order_group(void* world, const zk_group_rec* group);

/* Per team: tracked value, live units, metal a second from reclaim (part of the frame's income), and the live units that keep a team
   in the game: finished, and not ZK_U_DONT_COUNT (game_over.lua).  The value is cost x build progress; a chicken, which costs nothing,
   counts its build time, which for every unit that has a cost is its metal cost. */
#define ZK_STATS 4

int zk_sizeof_unit_record(void);
int zk_sizeof_site_record(void);
int zk_sizeof_feature_record(void);
int zk_global_count(void);

/* One world's frame for one team, computed now: its own units, then its allies' (as in sight), then its allyteam's knowledge
   of everyone else.  Returns the number of records. */
int zk_world_view(void* world, int team, void* records, int cap, float* glob, int commander, const float* base);

/* Many games stepping across a batch's own threads, which live as long as it does.  Free threads take the costliest game
   left; a game is stepped whole by one thread with its own RNG, so results don't depend on the thread count or the schedule.
   Batches share only the read-only catalog and maps: several may step at once, from different threads.  One batch takes one
   call at a time.  teams: the slots each game has in the step's arrays, 1 to ZK_MAX_TEAMS; a game uses the first n_teams of
   them, and the rest read as empty frames (count 0, stats 0). */
void* zk_batch_new(const void* catalog, int games, int threads, int teams);
void zk_batch_free(void* batch);
/* From the next step on.  1 steps on the caller's thread alone. */
void zk_batch_set_threads(void* batch, int threads);
/* Starts game `g` afresh, with n_teams teams (no more than the batch's slots) in the allyteams ally_of_team gives, as
   zk_world_set_teams.  team: each unit's; a unit of a team outside the game's is left out.  commander: per team, or -1.
   base: x, z per team. */
void zk_batch_reset(void* batch, int g, const void* map, uint64_t seed, int behaviour, int n_teams, const int32_t* ally_of_team,
                    int n, const int* type, const int* team, const float* x, const float* z, const float* heading,
                    const float* hp_frac, const float* build_frac, const int* commander, const float* base);
void* zk_batch_world(void* batch, int g);
/* zk_world_set_roles for every game, now and after any reset. */
void zk_batch_set_roles(void* batch, const int32_t* role_of_type, int n_types);
/* zk_world_set_reveal for every game, now and after any reset. */
void zk_batch_set_reveal(void* batch, int on);
/* Chicken defense in game g (zk_world_chickens): after zk_batch_reset and before the game's next step. */
void zk_batch_chickens(void* batch, int g, int team, int difficulty);
void zk_batch_chicken_state(const void* batch, int g, float* out /* ZK_CHICKEN_STATE */);
/* Applies orders, steps `ticks`, and writes every team's frame.  For the i-th game listed and team t, with S the batch's slots:
     groups[group_offsets[i] .. group_offsets[i + 1]] are its group orders, which go first (groups may be NULL for none);
     orders[order_offsets[i] .. order_offsets[i + 1]] are its unit orders;
     records[(i * S + t) * cap ...], counts[i * S + t], globs[(i * S + t) * G], stats[(i * S + t) * ZK_STATS],
     features[(i * S + t) * feature_cap ...], feature_counts[i * S + t]. */
void zk_batch_step(void* batch, const int* games, int n_games, const zk_order_rec* orders, const int* order_offsets,
                   const zk_group_rec* groups, const int* group_offsets, int ticks,
                   void* records, int cap, int* counts, float* globs, float* stats, void* features, int feature_cap, int* feature_counts);
/* zk_batch_step in two halves: begin returns at once and the batch's threads (all `threads` of them) do the step; wait returns
   when it's over.  Until then the caller mustn't touch the arrays it passed, nor the batch but to wait; a step, a free or a change
   of threads waits by itself.  The results are those of zk_batch_step. */
void zk_batch_step_begin(void* batch, const int* games, int n_games, const zk_order_rec* orders, const int* order_offsets,
                         const zk_group_rec* groups, const int* group_offsets, int ticks,
                         void* records, int cap, int* counts, float* globs, float* stats, void* features, int feature_cap, int* feature_counts);
void zk_batch_step_wait(void* batch);

#ifdef __cplusplus
}
#endif
#endif
