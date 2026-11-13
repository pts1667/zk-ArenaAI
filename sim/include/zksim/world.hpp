// Struct-of-arrays state of one game, and the read-only catalog and map it shares with others.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "zksim/capi.h"
#include "zksim/rawframe.hpp"

namespace zk {

constexpr float MAX_SLOW = 0.5f;
constexpr int BURROW_FRAMES = 12;       // Sleep(400) in the scripts: what has idle_cloak digs in this long after stopping
constexpr float GRAVITY = 120.0f / 900.0f;   // elmos a frame squared: Game.gravity, which every map has here
constexpr float SEARCH_PAD = 1.0f;      // elmos added to a grid search's radius, so rounding can't leave out what the exact test takes
constexpr float WALK_LINE_STEP = 16.0f; // cmd_raw_move.lua's test spacing along a line a unit would walk (movement.cpp line_free)
constexpr int CLEAR_BLOCK = 4;          // cells a side of a block of Map::clearance

struct Catalog {
    std::vector<zk_type_rec> types;
    std::vector<zk_weapon_rec> weapons;
    std::vector<float> priority;
    std::vector<zk_tactics_rec> tactics;
    std::vector<uint8_t> relations;
    // Rules from the gadgets (capi.h zk_catalog_reveal, zk_catalog_chickens): none until given.
    float reveal_ratio = 0.0f;
    int reveal_every = 0;
    zk_chicken_rules chickens[6] = {};
    bool has_chickens[6] = {};
    std::vector<zk_egg_rec> eggs;          // per type, once given
    float target_priority(int weapon, int type) const { return priority[size_t(weapon) * types.size() + size_t(type)]; }
    uint8_t relation(int type, int enemy_type) const { return relations[size_t(type) * types.size() + size_t(enemy_type)]; }
};

// Whether a walking unit is getting anywhere, and what it's trying if it isn't.
struct WalkProgress {
    float anchor_x = 0.0f, anchor_z = 0.0f;  // where it was when the window began
    int ticks = 0;                           // into the window
    int attempts = 0;                        // windows without progress since it last changed segment
    int segment = -1;                        // where those attempts began
    int escape_ticks = 0;                    // left of aiming straight at its way out
};

// What one team has to spend.
struct Purse {
    // Income as the engine counts it: metal with its overdrive, energy after what overdrive drew.
    float metal = 0.0f, energy = 0.0f, metal_income = 0.0f, energy_income = 0.0f;
    float innate_metal = 0.0f, innate_energy = 0.0f, storage = 0.0f;
    float energy_made = 0.0f;       // per second, before overdrive
    float energy_spent = 0.0f;      // by builders, since the last overdrive update
    float overdrive_metal = 0.0f, overdrive_energy = 0.0f;      // per second, as of that update
    float overdrive_share = 1.0f;   // of its allyteam's overdrive metal, by the energy it put in at that update
    float reclaimed = 0.0f;     // metal, in the last tick
    float reclaim_income = 0.0f;    // per second, the part of metal_income that reclaim is
    bool free_factory = true;   // a team's first ploppable factory is free and instant (start_unit_setup.lua)
};

struct MetalSpot {
    float x, z, income;
};

// One class's way around a map: tools/sim_data.py works this out, so a run needs no pathfinding.
struct Routes {
    int nodes = 0;                      // segments
    std::vector<float> centre_x, centre_z;
    std::vector<int> edge_to;           // per gate between two segments
    std::vector<float> gate_x, gate_z;
    std::vector<int> next_edge;         // nodes x nodes: the gate to take from a segment towards a goal's, or -1
};

// What CMoveMath's directional speed reads of a cell, together so that it's one place in memory.  slope: 0 under water, where
// that speed doesn't use it, as on the level.
struct CellSlope {
    float slope, normal_x, normal_z;
};

struct Map {
    int cols = 0, rows = 0, classes = 0;
    float cell = 16.0f;
    std::vector<float> height;
    std::vector<float> speed;
    std::vector<float> slope_mod;
    std::vector<CellSlope> slopes;
    void set_slopes(const float* slope, const float* normal_x, const float* normal_z);
    // Per class and block of CLEAR_BLOCK x CLEAR_BLOCK cells: how many cells it is, at least, from any cell of the block to the
    // nearest the class can't stand on, going by the larger of the two axes (255 for that or more).  And from that, how many of
    // a walked line's tests can be passed over after one in the block: those so near are sure to be on ground it can stand on.
    int clear_cols = 0, clear_rows = 0;
    std::vector<uint8_t> clearance;
    int clear_skip[256];
    const uint8_t* clearance_grid(int cls) const { return &clearance[size_t(cls) * size_t(clear_rows) * size_t(clear_cols)]; }
    // Per class, a bit a cell: whether the class can stand there, which is its speed there being above 0.  A thirty-second of
    // the speeds' size, so that tests of many cells stay in the processor's cache.
    size_t stand_words = 0;
    std::vector<uint64_t> stand;
    const uint64_t* stand_bits(int cls) const { return &stand[size_t(cls) * stand_words]; }
    static bool bit(const uint64_t* bits, size_t i) { return (bits[i >> 6] >> (i & 63)) & 1u; }
    // Both, from the speeds.
    void set_standing();
    std::vector<short> nearest_segment;           // per class and cell: the segment there, or the nearest
    std::vector<Routes> routes;                   // per class
    void set_segments(const short* labels);
    int segment_at(int cls, float x, float z) const { return nearest_segment[size_t(cls) * size_t(rows) * size_t(cols) + size_t(cell_index(x, z))]; }
    float width() const { return float(cols) * cell; }
    float depth() const { return float(rows) * cell; }
    // Dividing by a power of two is exactly multiplying by its inverse, which is quicker; any other cell divides.
    float inv_cell = 1.0f / 16.0f;
    bool cell_pow2 = true;
    void set_cell(float c) {
        int e;
        cell = c;
        cell_pow2 = std::frexp(c, &e) == 0.5f;
        inv_cell = 1.0f / c;
    }
    void cell_of(float x, float z, int& cx, int& cz) const {
        if (cell_pow2) {
            cx = int(x * inv_cell);
            cz = int(z * inv_cell);
        } else {
            cx = int(x / cell);
            cz = int(z / cell);
        }
        cx = cx < 0 ? 0 : (cx >= cols ? cols - 1 : cx);
        cz = cz < 0 ? 0 : (cz >= rows ? rows - 1 : cz);
    }
    int cell_index(float x, float z) const {
        int cx, cz;
        cell_of(x, z, cx, cz);
        return cz * cols + cx;
    }
    float height_at(float x, float z) const { return height[size_t(cell_index(x, z))]; }
    const float* speed_grid(int cls) const { return &speed[size_t(cls) * size_t(rows) * size_t(cols)]; }
    float speed_at(int cls, float x, float z) const { return speed_grid(cls)[size_t(cell_index(x, z))]; }
    // CMoveMath's speed for a move in this direction: climbing slows a unit, descending doesn't.
    float speed_along(int cls, float x, float z, float dir_x, float dir_z) const { return speed_along_at(cls, size_t(cell_index(x, z)), dir_x, dir_z); }
    float speed_along_at(int cls, size_t i, float dir_x, float dir_z) const {
        const CellSlope& g = slopes[i];
        const float base = speed_grid(cls)[i], mod = slope_mod[size_t(cls)], s = g.slope;
        if (base <= 0.0f || mod <= 0.0f || s <= 0.0f) return base;
        const float climb = -(dir_x * g.normal_x + dir_z * g.normal_z);
        const float directed = climb > 0.0f ? s * climb : 0.0f;
        return base * (1.0f + s * mod) / (1.0f + directed * mod);
    }
};

// xoshiro256**: one per world, so a result doesn't depend on how worlds share threads.
struct Rng {
    uint64_t s[4];
    explicit Rng(uint64_t seed);
    uint64_t next();
    float uniform() { return float(next() >> 40) * (1.0f / float(1ull << 24)); }
    // Lua's math.random(m, n): an integer from m to n, both included.
    int between(int lo, int hi) { return hi <= lo ? lo : lo + int(double(next() >> 11) * (1.0 / 9007199254740992.0) * double(hi - lo + 1)); }
    // ... and math.random(): a double in [0, 1).
    double unit() { return double(next() >> 11) * (1.0 / 9007199254740992.0); }
};

// Chicken defense as unit_spawner.lua runs it (chickens.cpp): the gadget's `data` and the settings it adjusts for the players.
struct ChickenGame {
    const zk_chicken_rules* rules = nullptr;
    int team = -1, ally = -1, difficulty = -1, players = 0;
    double malus = 1.0, burrow_spawn_rate = 0.0, grace_period = 0.0, burrow_regress_time = 0.0, aggro_per_burrow = 0.0;
    double aggro_defense_factor = 0.0, defense_per_wave = 0.0;
    double queen_time = 0.0, base_queen_time = 0.0, queen_health_mod = 1.0;
    double wave_schedule = 0.0;              // frame: a wave comes on the first frame past it
    double time_of_last_spawn = 0.0;         // second the last burrow came
    double aggro = 0.0, aggro_delta = 0.0, tech = 0.0, defense_pool = 0.0, defense_quota = 0.0;
    int waves = 0, special_cooldown = 0, mini_queens_sent = 0;
    int queen = -1, morph_frame = -1;
    bool morphed = false, endgame = false, victory = false;
    std::vector<int> burrows;                // alive, in the order they came
    int burrows_killed = 0;
    struct Birth {
        int unit;
        double second;
    };
    std::vector<Birth> births;               // chickens of waves, which die of old age
    struct Egg {
        int wreck;
        double decay;
    };
    std::vector<Egg> eggs;                   // those that decay, with the second they do
    bool have_target = false;                // targetCache: where idle chickens are sent
    float target_x = 0.0f, target_z = 0.0f;
    size_t fallen_seen = 0;                  // deaths the gadget has heard of
    std::vector<int> pick;                   // ChooseTarget's working: a team's units
};

// A shot and what it does where it comes down.  One that flies there is a Projectile's; one that climbs first, is dropped or
// goes through what it hits is scheduled to land at its aim point when its flight is over.
struct Shot {
    int weapon, source, source_side, target;
    float x, z;
    bool guided, air;
    float power;   // of its damage, after falling with distance
    float origin_x, origin_z;
    bool aimed_at_blip;   // a guided one goes for where radar shows it
};

// A shot on its way: a shell, a bolt or a rocket (combat.cpp, "Projectiles").  Each tick it goes on along its path, and the first
// enemy in its way stops it.
struct Projectile {
    Shot shot;                // x, z: where it's aimed, which for one that homes is where its target is
    float x, z;               // where it is
    float dir_x, dir_z;       // the way it's going over the ground, a unit vector
    float speed;              // along its path, elmos a frame
    float flat;               // the share of that it makes over the ground: less for a shell's arc and for a rocket's rise
    float flown;              // over the ground, so far
    float aimed;              // over the ground, from where it left to where it's aimed
    float reach;              // how far it goes at most: to where it comes down, or to the end of its range
    float slope;              // of its path over the straight one, at each end: a shell's elevation, a rocket's rise
    float from_height;        // of where it left, over what it's aimed at
    float base;               // how high what it's aimed at is: a flier or a jumper, else 0
    float muzzle;             // the shooter's hit radius, which isn't part of how far its damage has faded
    int fuel;                 // frames a rocket has left
    int wait;                 // frames of the tick it was fired in that had gone by then
    bool homing;              // turns towards its target, as long as that lives
    bool fades;               // nothing happens at the end of its reach: a bolt out of range

    // Over what it's aimed at.  Past its aim point it's on its way down to the ground.
    float height() const { return (1.0f - flown / aimed) * (from_height + slope * flown); }
};

// Where along a piercing shot's line it meets a unit, and how many times over.
struct PierceHit {
    float along, times;
};

// What a unit does on its own, between orders.  unit_ai.cpp.
struct UnitAI {
    int move_state = 1, fire_state = 2;
    // An attack it gave itself, from a fight order or from being idle, and until when if it's the latter.
    int chasing = -1, chase_until = 0;
    int lost_timer = 4;                       // updates until a target that's out of radar is given up
    float line_x = 0.0f, line_z = 0.0f, line_end_x = 0.0f, line_end_z = 0.0f;   // what the leash goes from
    bool returning = false;                   // to the point on the line it left
    float return_x = 0.0f, return_z = 0.0f;
    float speed_limit = 0.0f;                 // following what's getting away; 0 for none
    int last_attacker = -1;
    // The tactical AI: a move that goes before whatever else it's doing.
    bool tactics_off = false;
    bool dodging = false;
    float dodge_x = 0.0f, dodge_z = 0.0f;
    // Swarming, the same AI's: jinking up to what's slow to hit it, then strafing (DoSwarmEnemy).
    float jink_side = 1.0f, strafe_side = 1.0f;       // the side it's going to: 1 or -1
    float jink_count = 0.0f;                  // updates since it changed sides, for what waits longer than one
    // Ward fire (unit_ward_fire.lua): where its weapons shoot while they've nothing to shoot at, until a frame, and whose shield it's for.
    float ward_x = 0.0f, ward_z = 0.0f;
    int ward_until = 0, ward_enemy = -1;
    int ready_frame = 0;                      // of the weapon its table goes by
    int aggressive_until = -1;                // as a target: someone's fighting it, so others may
    // An idle unit's place, which it goes back to.
    bool queue_was_empty = true, was_idle = false;      // no idle event for what just appears, as for CreateUnit
    bool idle_want_return = false, idle_aggression = false, has_idle_pos = false;
    bool queue_return = false, set_return = false, force_return = false, want_fight_return = false, has_fight_return = false;
    float idle_x = 0.0f, idle_z = 0.0f, fight_return_x = 0.0f, fight_return_z = 0.0f;
    float heat = 0.0f;
    int shots_fired = 0;
    // Where radar shows it, off by this much of RADAR_ERROR.
    bool err_started = false;
    float err_x = 0.0f, err_z = 0.0f;
};

// Jump: jump.cpp.  A unit that's busy goes where this says rather than walking.
struct JumpState {
    bool busy = false, flying = false;      // waiting to take off, then in the air
    bool stranded = false;                  // its order came while it was busy, and it won't walk for that one
    float delay = 0.0f, timeout = 0.0f;     // frames until it takes off, and of being disarmed until it won't
    float at = 0.0f, step = 0.0f;           // how far along the flight, and how much of it a frame is
    float from_x = 0.0f, from_z = 0.0f, to_x = 0.0f, to_z = 0.0f;
    float charge = 1.0f;
    float height = 0.0f;                    // the top of this flight's arc
    float sideways = 0.0f;                  // how far along the flight it starts across: a throw goes up first
    int thrown_by = -1;
};

// Pushed: abilities.cpp.  A ground unit goes where this says rather than walking, until it stops.
struct Skid {
    bool on = false;
    float vx = 0.0f, vz = 0.0f, vy = 0.0f, y = 0.0f;
};

// An aircraft: air.cpp.
enum AirPhase { AIR_FLYING = 0, AIR_LANDED = 1, AIR_TAKEOFF = 2, AIR_LANDING = 3, AIR_ON_PAD = 4 };
struct AirState {
    float height = 0.0f;         // above the ground
    int phase = AIR_FLYING;
    int takeoff_left = 0;        // frames of climbing before it moves off
    int ammo = 0;                // salvos left, for what has ammo
    int pad = -1;                // the pad it's making for or sits on, or -1
    float rearmed = 0.0f;        // frames of rearming done on the pad
    float want = -1.0f;          // the height a transport comes down to this tick, to load or unload; negative for its own
};

// A Djinn and its beacon.
struct Teleporter {
    int unit;
    int beacon = -1;
    bool deployed = false, making = false;
    float make_x = 0.0f, make_z = 0.0f, make_left = 0.0f;
    int still_since = -1;
    int carrying = -1;
    float carry_left = 0.0f;
};

// A missile and the silo whose pad it stands on.
struct SiloMissile {
    int unit, silo, pad;
};

// What overkill prevention thinks is on its way to a unit.
struct IncomingShot {
    int frame;
    float damage;
};

struct IncomingDamage {
    std::vector<IncomingShot> shots;
    int last_frame = -1;
    bool doomed = false, seen = false;
};

// A unit with a shield.
struct ShieldUnit {
    int unit;
    const zk_weapon_rec* weapon;
    std::vector<int> links;     // into the world's shields: those it starts transfers with
};

// Something reclaimable lying on the ground.  Its def's metal and how much of it is left are apart, as in the engine: the rate
// goes by the first, whatever's left.
struct Wreck {
    float x, z, radius;
    float metal;            // when whole
    float left = 1.0f;      // of that, and reclaimed from 1 to 0
    float hp;
    int feature_def;        // the engine's
    int heap_of;            // the unit type whose heap this turns into when destroyed, or -1 for nothing
};

// Ground fire left by a shot.
struct FireArea {
    const zk_weapon_rec* weapon;
    float x, z, damage_per_frame;
    int frames_left;
    int team;                 // whose shot lit it, or -1
};

// A blip's speed record, as blips.py keeps it: its top speed is the highest speed it has held for a whole window.
struct BlipTrack {
    float speeds[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    int frames[5] = {0, 0, 0, 0, 0};
    int head = 0, filled = 0;
    float top = 0.0f;
};

// What one allyteam knows about every unit in the game.
struct SideView {
    std::vector<uint8_t> in_los, in_radar, known, remembered;
    std::vector<int> seen_frame;
    int updated_frame = -1;
    std::vector<float> ghost_hp, ghost_build;
    std::vector<BlipTrack> tracks;
    int last_contact = 0;
    // The dead are looked at once, when they've just died (World::fallen from here on), and then only while remembered.
    size_t fallen_seen = 0;
    std::vector<int> memories;          // dead units it still remembers, in the order they died
};

struct World {
    const Catalog* cat;
    const Map* map;
    Rng rng;
    int behaviour;
    float cat_max_footprint = 0.0f;       // the widest footprint radius of any type
    int frame = 0;
    int tick_count = 0;
    uint32_t turn_salt = 0;                // from the seed: which ticks the tactical AI's turns fall on (unit_ai.cpp update_tactics)

    // Teams, who own units and pay for them, and the allyteams they're in, who share sight and don't fight each other.  Until
    // set_teams: two teams, each its own allyteam.
    int n_teams = 2, n_allies = 2;
    int ally_of[ZK_MAX_TEAMS] = {0, 1, 2, 3, 4, 5, 6, 7};
    int ally_size[ZK_MAX_TEAMS] = {1, 1, 0, 0, 0, 0, 0, 0};     // teams in each allyteam
    bool sharing = false;                 // some allyteam has several teams, whose resources are shared (economy.cpp)
    void set_teams(int n, const int32_t* ally_of_team);

    // Units.  side: the allyteam, whom it fights and whose sight it adds to.  team: who owns it.
    std::vector<int> type, side, team, order, order_target, nearest;
    std::vector<uint8_t> alive;
    // Handles only grow, and a long game's dead outnumber its living, so a tick goes through these rather than every handle: the
    // units alive at the grid's last rebuild and those added since, in handle order (some may have died since; loops look).
    std::vector<int> live;
    std::vector<int> fallen;             // every unit that has died, in the order it did
    size_t fallen_pruned = 0;            // of those, how many the lists by kind have been rid of
    std::vector<float> x, z, vx, vz, heading, speed, hp, build, para, disarm, slow, damage_taken;
    std::vector<UnitAI> ai;
    std::vector<IncomingDamage> incoming;
    std::vector<float> regen_wait, regen_seen_damage;    // frames until it regenerates, and damage_taken when that was last set
    std::vector<float> recent_damage;     // as the engine's, which falls by a tenth every 15 frames: what kind of remains it leaves
    std::vector<int> last_hit;            // frame, for what closes when hit
    std::vector<int> closing_since;       // frame it began to close, or as good as, had it been open
    std::vector<float> stock_progress;    // frames towards the next stockpiled shot
    std::vector<int> stock;
    std::vector<int> stun_attackers;      // disarm weapons that weighed this target this round
    std::vector<float> goal_x, goal_z, shield;
    std::vector<int> order_wait;          // frames until a new move order takes effect
    std::vector<int> order_lag;           // frames before the unit starts to brake for it
    std::vector<WalkProgress> progress;
    // The order log: what the controller said last, as a player remembers it.
    std::vector<int> mem_kind, mem_frame, mem_target;
    std::vector<float> mem_x, mem_z, last_damage;
    SideView views[ZK_MAX_TEAMS];         // per allyteam
    std::vector<int> radar_units;
    // Economy, only for worlds that turn it on.
    bool economy_on = false;
    Purse purses[ZK_MAX_TEAMS];           // per team
    std::vector<MetalSpot> spots;
    std::vector<int> work_on;             // the frame this builder or factory has started and is on, or -1
    std::vector<int> produces;            // the type this factory repeats, or -1
    std::vector<float> extracts;          // metal per second once finished: what it makes, and what its spot gives it
    std::vector<float> overdrive;         // extra metal per second, as of the last update
    std::vector<int> builders, earners, pylons;   // handles, so a tick needn't look through everyone
    std::vector<int> upkeepers;
    std::vector<int> grid_nodes, grid_parent;     // overdrive's working, kept so it needn't allocate
    std::vector<float> grid_capacity, grid_squares, grid_px, grid_pz, grid_range;
    std::vector<char> grid_maxed;
    std::vector<ShieldUnit> shields;
    std::vector<uint8_t> unpaid;          // its upkeep wasn't paid, so it's off
    // Cloak.
    std::vector<uint8_t> want_cloak, area_on, area_cloaked, cloaked;
    std::vector<int> cloak_blocked_until;  // frame: firing, damage or an enemy close by showed it
    std::vector<int> personal_cloakers, area_cloakers;
    std::vector<int> burrowers;
    std::vector<int> burrow_at;            // frame it can dig in from
    std::vector<JumpState> jump;
    std::vector<int> jumpers;
    std::vector<int> thrown;               // in the air, and not among the jumpers
    std::vector<Skid> skid;
    std::vector<int> throwers;
    std::vector<uint8_t> low_power, pulling;
    std::vector<Teleporter> teleporters;
    std::vector<int> wait_since;           // frame it was told to wait at a beacon
    std::vector<SiloMissile> silo_missiles;
    std::vector<int> droppers;             // what drops a bomb when it goes
    std::vector<AirState> air;
    std::vector<int> aircraft, pads;       // handles of what flies, and of what planes rearm at
    // Air transports (transport.cpp).  What's carried is out of play: stunned, as the engine has it, off the grid, so that nothing
    // finds it to shoot or push, and where its transport is.
    std::vector<int> carrier, cargo;       // the transport that carries it, and what it carries: -1 for none
    std::vector<int> drop_ready;           // frame from which a transport can drop again
    std::vector<int> held_for_lift;        // frame a transport of its allyteam, come to pick it up, last had it stand still
    std::vector<int> transports;
    // Wrecks, by id.  One with nothing left stays in the list, so an id stays good, and leaves the others.
    std::vector<Wreck> wrecks;
    std::vector<int> live_wrecks;
    std::vector<std::vector<int>> wreck_cells;   // ids per grid cell: wrecks don't move
    std::vector<int> sep_hits;             // ... and the places of the neighbours a unit overlaps
    // The live wrecks again, by grid cell (wreck_start, as cell_start), with their places alongside; remade when one comes or goes.
    std::vector<int> wreck_start, wreck_ids;
    std::vector<float> wreck_px, wreck_pz;
    bool wreck_index_stale = true;
    std::vector<float> burn_rate, burn_frames;
    std::vector<int8_t> burn_team;         // whose shot set it alight, or -1
    std::vector<FireArea> areas;
    // The damage each team's units have done to enemies' units and had done to them, in metal (team_stats, capi.h ZK_STATS).
    double metal_dealt[ZK_MAX_TEAMS] = {}, metal_taken[ZK_MAX_TEAMS] = {};
    std::vector<float> damage_mult;        // of what it's dealt: 1, but for a chicken queen, whose health the gadget stretches
    // The reveal rule (game_over.lua): on unless turned off, never in a chicken game.  revealed: the allyteams always in sight, by bit.
    bool reveal_on = true;
    uint32_t revealed = 0;
    // The alliances that have had a finished ZK_U_VITAL unit (check_vital), by bit.
    uint32_t had_vital = 0;
    ChickenGame chickens;
    void start_chickens(int team, int difficulty);
    void chicken_state(float* out) const;
    // Weapons, ZK_MAX_MOUNTS per unit.
    std::vector<float> cooldown, salvo_timer;
    std::vector<int> salvo_left, weapon_target;

    // Spatial index, rebuilt each tick: the live units by allyteam, then grid cell (row by row), then handle.  Allyteam s's units in
    // cell c are cell_units[cell_start[k] .. cell_start[k + 1]] for k = s * grid_cells() + c, so a row of cells is one run of them.
    int grid_cols = 0, grid_rows = 0;
    std::vector<int> cell_start, cell_units, unit_key;
    // Each unit's place in cell_units, or -1 for what isn't in the grid; and separate()'s working, by that place: where each
    // unit is, its footprint, and how much of an overlap with it a neighbour takes.
    std::vector<int> grid_slot;
    std::vector<float> sep_x, sep_z, sep_radius, sep_share;
    size_t grid_cells() const { return size_t(grid_cols) * size_t(grid_rows); }
    std::vector<uint8_t> scratch_seen;     // per unit or wreck, for a sweep that marks what's in sight
    // What a tick settles for every unit before it lets any of them act, so that it's nothing to a unit which were made before it:
    // where each means to walk, as (kind, x, z) by its place in `live`; who can fire, and how slowed; and who has died.
    struct MovePlan {
        int8_t kind;
        float x, z;
    };
    std::vector<MovePlan> move_plans;
    std::vector<uint8_t> fire_ok;
    std::vector<float> fire_rate;
    std::vector<int> dying;
    std::vector<int> scratch_eyes;         // an allyteam's finished units, for the same
    std::vector<int> scratch_found;        // the wrecks a sweep found in sight
    struct EmitLists {
        std::vector<int> own, allies, seen, blips, memory, hidden;
    } emit_lists;                          // emit_view's six kinds of record, in the order it writes them

    // Shots by the tick they land.
    std::vector<std::vector<Shot>> ring;
    std::vector<Projectile> projectiles;
    std::vector<PierceHit> pierce_hits;

    World(const Catalog* c, const Map* m, uint64_t seed, int behaviour_);
    // Its allyteam is its team's.  Units may come at any time, from a caller or from the world itself (a factory's, a beacon).
    int add(int type, int team, float x, float z, float heading, float hp_frac, float build_frac);
    void order_unit(int unit, int kind, float gx, float gz, int target);
    // Group orders (capi.h zk_group_rec): each catalog type's role, -1 for none.
    std::vector<int8_t> role_of_type;
    int order_group(const zk_group_rec& g);
    void ready_weapons();
    void set_economy(int n_spots, const float* spot_triples, float metal, float energy, float innate_metal, float innate_energy, float storage);
    int add_wreck(float px, float pz, float metal, float wreck_hp, float radius, int feature_def, int heap_of);
    int emit_features(int ally, FeatureRecord* out, int cap);
    void step(int ticks);
    int count() const { return int(type.size()); }
    void ensure_views();
    void update_view(int ally);
    // ally_frame, ally_count: a frame this call's world has just written for another team of the same allyteam, or none.  What
    // comes after an allyteam's own units is the same for each of its teams, and is copied from it.
    int emit_view(int team, UnitRecord* out, int cap, float* glob, int commander, const float* base, const UnitRecord* ally_frame = nullptr,
                  int ally_count = 0);
    void team_stats(int team, float* out) const;
    void rebuild_index() { rebuild_grid(); }
    const zk_type_rec& type_of(int u) const { return cat->types[size_t(type[size_t(u)])]; }
    // What's carried is stunned too, as in the engine (CUnit::SetStunned on loading): it does nothing until it's let go.
    bool stunned(int u) const { return para[size_t(u)] > 1.0f || carrier[size_t(u)] >= 0; }
    // How far off the ground what's carried hangs.
    float carried_altitude(int u) const {
        const float h = air[size_t(carrier[size_t(u)])].height - type_of(u).load_height;
        return h > 0.0f ? h : 0.0f;
    }
    bool disarmed(int u) const { return disarm[size_t(u)] > 1.0f; }
    bool power_off(int u) const { return economy_on && low_power[size_t(u)]; }
    const Teleporter* teleporter_of(int u) const;
    bool throw_ready(int u) const;
    float slow_now(int u) const { return slow[size_t(u)] < MAX_SLOW ? slow[size_t(u)] : MAX_SLOW; }
    bool active(int u) const { return build[size_t(u)] >= 1.0f && !stunned(u); }
    // Aircraft: how high it is, and whether it's up there, where only what fires at the air can reach it.
    float flight_height(int u) const { return air[size_t(u)].height; }
    bool airborne(int u) const;
    bool out_of_ammo(int u) const { return type_of(u).ammo > 0 && air[size_t(u)].ammo <= 0; }
    bool wreck_exists(int id) const { return id >= 0 && id < int(wrecks.size()) && wrecks[size_t(id)].left > 0.0f; }
    // energysolar.lua: it stops making at once, is armoured when shut, loses that when it starts to open, and makes again once open.
    bool shut_for_hit(int u) const;
    bool armoured_for_hit(int u) const;
    void hit_closes(int u);

    // The grid cells a square round a point touches.
    struct CellBox {
        int x0, x1, z0, z1;
    };
    CellBox cell_box(float cx, float cz, float radius) const;
    // Every live unit in the cells round a point, cell by cell (row by row), each cell's in handle order.
    template <class F>
    void for_near(float cx, float cz, float radius, F&& f) const;
    // The same, by place in cell_units.
    template <class F>
    void for_near_slots(float cx, float cz, float radius, F&& f) const;
    // The same, of one allyteam only: for_near with the others' left out, in the same order.
    template <class F>
    void for_side(int ally, float cx, float cz, float radius, F&& f) const;
    // Those of every other allyteam, an allyteam at a time in ascending order, each as for_side.
    template <class F>
    void for_enemies(int ally, float cx, float cz, float radius, F&& f) const;

private:
    template <class F>
    void side_in_box(int ally, const CellBox& b, F& f) const;
    void tick();
    void rebuild_grid();
    void drop_the_dead();
    void died(int u);
    void check_reveal();
    void check_vital();
    // chickens.cpp
    void run_chickens();
    void chicken_deaths();
    void chicken_frame(int f);
    void chicken_wave(double t);
    void chicken_second(double t, int f);
    void chicken_slow(double t);
    void burrow_killed(int u, double t);
    void lay_egg(int type, float px, float pz, double decay);
    void kill_all_chickens();
    void morph_queen(int f);
    void spawn_burrow();
    bool place_queen(float& px, float& pz);
    int spawn_queen(int f);
    void spawn_mini_queen();
    void spawn_chickens(int burrow, int n, int type, double t);
    void spawn_turret(int burrow, int type, double number, bool force, double t);
    void spawn_support(int burrow, int type, double t);
    void special_powers(double t);
    void spawn_anywhere(int type, int number, float min_dist, float max_dist, const float* target);
    int choose_kind(int role, bool tech, double t, int* other);
    bool choose_target(int unit, float& tx, float& tz);
    int nearest_enemy(int u) const;
    bool player_within(float px, float pz, float r) const;
    bool ground_blocked(int type, float px, float pz) const;
    bool can_place(int type, float px, float pz) const;
    int add_chicken(int type, float px, float pz);
    void find_nearest();
    void pick_targets();
    void fire_weapons();
    void fire_one(int u, int slot, int frame_offset);
    void fire_at_point(int u, int slot, int frame_offset);
    bool can_ward(int u, int slot) const;
    void lead_target(int u, const zk_weapon_rec& w, int t, float tx, float tz, float mult, float& lead_x, float& lead_z) const;
    void launch(const Shot& s, int u, const zk_weapon_rec& w, int frame_offset);
    void fly_projectiles();
    bool fly(Projectile& p);
    void land_shot(const Shot& s, float px, float pz, int struck, float height, bool fades = false);
    bool pay_for_shot(int u, const zk_weapon_rec& w);
    int manual_mount(int u) const;
    void resolve_shots();
    void apply_shot(const Shot& s);
    void note_attacker(int u, int source, int source_side);
    void explode(int weapon, float px, float pz, int source, int source_side, bool air, float power = 1.0f, float height = 0.0f, int struck = -1,
                 int source_team = -1);
    void pierce(const Shot& s, float px, float pz, float power);
    void damage_unit(int u, const zk_weapon_rec& w, float mult, int dealer);
    void count_damage(int victim, float amount, int dealer);
    float shield_takes(const zk_weapon_rec& w, const Shot& s, float px, float pz, float left, float& hit_x, float& hit_z);
    void update_shields();
    void recharge_shields();
    void find_shield_links();
    void share_shield_charge();
    void move_units();
    void track_progress(int u, bool walking);
    MovePlan plan_move(int u);
    void move_one(int u, const MovePlan& plan);
    bool wants_to_walk(int u, float& gx, float& gz);
    void separate();
    void finish_deaths();
    void leave_remains(int u);
    void hurt_wrecks(const zk_weapon_rec& w, float px, float pz, float power);
    void remove_wreck(int id);
    void index_wrecks();
    void reclaim_step(int u);
    bool pay_energy(int team, float amount);
    void pay_upkeep();
    void update_cloak();
    void consider_cloak(int u);
    void show_for_action(int u);
    bool enemy_within(int u, float reach) const;
    bool can_be_aimed_at(int s, int t) const;
    bool in_sight(int s, int t) const;
    bool identified(int s, int t) const;
    bool aim_is_off(int s, int t) const;
    void aim_error(int t, float& ex, float& ez) const;
    void update_radar_error();
    bool bait_blocks(int u, int t) const;
    bool doomed(int t) const;
    bool overkill_blocks(int u, const zk_weapon_rec& w, int t, int flight);
    float armor_mult(int t) const;
    bool armored_now(int t) const { return armor_mult(t) < 1.0f; }
    bool can_hit_now(int u, int t) const;
    bool valid_chase_target(int u, int t) const;
    int closest_target(int u, float cx, float cz, float radius) const;
    int nearest_seen_enemy(int u, float radius) const;
    bool target_lost(int u, int t);
    void chase(int u, int t, int until);
    bool chase_is_over(int u);
    void fight_step(int u);
    void idle_step(int u);
    void run_commands(int u);
    void drop_chase(int u);
    void set_dodge(int u, float tx, float tz);
    void unit_went_idle(int u);
    bool update_idle_aggression(int u, const zk_tactics_rec& b, int enemy, bool known, float default_range, float enemy_dist);
    bool skirm_enemy(int u, const zk_tactics_rec& b, int enemy, bool known, bool idle, bool fight_and_hold, bool hug);
    bool flee_enemy(int u, const zk_tactics_rec& b, int enemy, bool known, bool idle);
    bool tactics_against(int u, const zk_tactics_rec& own, int enemy, bool known, bool manual_attack, bool have_fight, bool hold, bool idle, bool particular,
                         bool always_jink);
    bool swarm_enemy(int u, const zk_tactics_rec& b, int enemy, bool idle);
    bool swarm_point(int u, const zk_tactics_rec& b, float px, float pz);
    void jink_towards(int u, const zk_tactics_rec& b, float px, float pz, float apart);
    void change_side(int u, const zk_tactics_rec& b, float& side_now, bool start_over);
    int tactics_row(int u) const;
    void run_tactics(int u);
    void run_ward_fire(int u);
    void update_tactics();
    void update_unit_ai();
    void update_kamikaze();
    void update_droppers();
    bool kamikaze_due(int u) const;
    void self_destruct(int u);
    void drop_bomb(int u);
    float jump_altitude(int u) const;
    bool can_land_at(int u, float gx, float gz) const;
    bool arc_is_clear(int u, float gx, float gz) const;
    void try_jump(int u);
    void step_jump(int u);
    void land(int u);
    void update_jumps();
    void update_burrows();
    void update_air();
    void fly_plane(int u);
    bool air_goal(int u, float& gx, float& gz);
    bool climb_or_land(int u, bool wants_to_go);
    int find_pad(int u) const;
    void used_ammo(int u);
    bool bomb_due(int u, const zk_weapon_rec& w, int t) const;
    void start_run(int u);
    void drop_shot(int u, int slot, int frame_offset);
    float bomb_fall_frames(int u, const zk_weapon_rec& w, int t) const;
    void update_transports();
    bool can_lift(int u, int t) const;
    float loaded_speed(int u) const;
    void pick_up(int u, int t);
    void drop_cargo(int u);
    void let_go(int u, bool falls);
    void push_from(int u, int source, float impulse);
    void step_skid(int u);
    void step_air_skid(int u, float& px, float& pz);
    void update_throwers();
    void throw_around(int u, float tx, float tz);
    bool someone_to_throw(int u) const;
    void update_teleporters();
    void teleport_waiter(Teleporter& t);
    int next_waiter(const Teleporter& t) const;
    void send_through(int u, const Teleporter& t);
    int free_pad(int silo) const;
    void launch_missile(int u);
    void destroy_missiles_of(int silo);
    void forget_commands(int u);
    bool walk_to_target(int u, int t, float& gx, float& gz);
    bool sees_point(int from, float tx, float tz) const;
    void run_economy();
    void collect_income();
    void share_overflow(const float* over, bool metal);
    void update_overdrive();
    void overdrive_ally(int s);
    void run_builder(int u);
    void run_factory(int u);
    bool plate_off(int u) const;
    void run_caretaker(int u);
    void advance_frame(int builder, int frame_unit);
    int place_frame(int t, int team, float px, float pz, float head);
    int place_frame_only(int t, int team, float px, float pz, float head);
    float spot_income(float px, float pz) const;
    float reach_of(int u, float radius) const { return type_of(u).build_range + radius; }
    bool within(int u, float px, float pz, float reach) const;
    void update_status();
    void burn_areas();
    float target_score(int u, int slot, int target);
    bool in_range(int u, const zk_weapon_rec& w, int target) const { return in_range_at(u, w, target, x[size_t(target)], z[size_t(target)]); }
    bool in_range_at(int u, const zk_weapon_rec& w, int target, float tx, float tz) const;
    bool can_reach(int u, const zk_weapon_rec& w, int target) const;
    bool faces(int u, const zk_weapon_rec& w, int target) const;
    void waypoint(int u, float gx, float gz, float& wx, float& wz) const;
    bool clear_sight(int from, int to) const;
};

inline World::CellBox World::cell_box(float cx, float cz, float radius) const {
    constexpr float CELL = 128.0f;
    const float lo_x = (cx - radius) / CELL, hi_x = (cx + radius) / CELL, lo_z = (cz - radius) / CELL, hi_z = (cz + radius) / CELL;
    // Wholly off the grid, or at no place (a shot led far past the map's edge: the dragon's slow kick), it's empty: no row, and
    // nothing read past the index.  A box that touches the grid is as it always was.
    if (!(hi_x > -1.0f && lo_x < float(grid_cols) && hi_z > -1.0f && lo_z < float(grid_rows))) return {0, -1, 0, -1};
    const int x0 = lo_x < 0.0f ? 0 : int(lo_x), z0 = lo_z < 0.0f ? 0 : int(lo_z);
    const int x1 = hi_x >= float(grid_cols) ? grid_cols - 1 : int(hi_x), z1 = hi_z >= float(grid_rows) ? grid_rows - 1 : int(hi_z);
    return {x0, x1, z0, z1};
}

template <class F>
void World::side_in_box(int s, const CellBox& b, F& f) const {
    const size_t base = size_t(s) * grid_cells();
    for (int gz = b.z0; gz <= b.z1; ++gz) {
        const size_t row = base + size_t(gz) * size_t(grid_cols);
        const int end = cell_start[row + size_t(b.x1) + 1];
        for (int i = cell_start[row + size_t(b.x0)]; i < end; ++i) f(cell_units[size_t(i)]);
    }
}

template <class F>
void World::for_side(int s, float cx, float cz, float radius, F&& f) const {
    side_in_box(s, cell_box(cx, cz, radius), f);
}

template <class F>
void World::for_enemies(int s, float cx, float cz, float radius, F&& f) const {
    const CellBox b = cell_box(cx, cz, radius);
    for (int e = 0; e < n_allies; ++e)
        if (e != s) side_in_box(e, b, f);
}

template <class F>
void World::for_near(float cx, float cz, float radius, F&& f) const {
    for_near_slots(cx, cz, radius, [&](int i) { f(cell_units[size_t(i)]); });
}

template <class F>
void World::for_near_slots(float cx, float cz, float radius, F&& f) const {
    const CellBox b = cell_box(cx, cz, radius);
    const size_t cells = grid_cells();
    for (int gz = b.z0; gz <= b.z1; ++gz) {
        const size_t row = size_t(gz) * size_t(grid_cols);
        const size_t first = row + size_t(b.x0), last = row + size_t(b.x1) + 1;
        // The allyteams with someone in this stretch of the row.  One alone is its run; otherwise each cell's runs go together by
        // handle: two as a merge, more by taking the lowest of their heads each time.
        size_t with[ZK_MAX_TEAMS];
        int k = 0;
        for (int s = 0; s < n_allies; ++s)
            if (cell_start[size_t(s) * cells + first] != cell_start[size_t(s) * cells + last]) with[k++] = size_t(s) * cells;
        if (k == 0) continue;
        if (k == 1) {
            for (int i = cell_start[with[0] + first], end = cell_start[with[0] + last]; i < end; ++i) f(i);
            continue;
        }
        if (k == 2) {
            const size_t one = with[0], other = with[1];
            for (size_t c = first; c < last; ++c) {
                int i = cell_start[one + c], j = cell_start[other + c];
                const int i_end = cell_start[one + c + 1], j_end = cell_start[other + c + 1];
                while (i < i_end && j < j_end) f(cell_units[size_t(i)] < cell_units[size_t(j)] ? i++ : j++);
                while (i < i_end) f(i++);
                while (j < j_end) f(j++);
            }
            continue;
        }
        for (size_t c = first; c < last; ++c) {
            int at[ZK_MAX_TEAMS], end[ZK_MAX_TEAMS];
            for (int r = 0; r < k; ++r) {
                at[r] = cell_start[with[r] + c];
                end[r] = cell_start[with[r] + c + 1];
            }
            for (;;) {
                int pick = -1;
                for (int r = 0; r < k; ++r)
                    if (at[r] < end[r] && (pick < 0 || cell_units[size_t(at[r])] < cell_units[size_t(at[pick])])) pick = r;
                if (pick < 0) break;
                f(at[pick]++);
            }
        }
    }
}

}  // namespace zk
