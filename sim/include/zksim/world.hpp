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

struct Catalog {
    std::vector<zk_type_rec> types;
    std::vector<zk_weapon_rec> weapons;
    std::vector<float> priority;
    std::vector<zk_tactics_rec> tactics;
    std::vector<uint8_t> relations;
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

// What one side has to spend.
struct Purse {
    // Income as the engine counts it: metal with its overdrive, energy after what overdrive drew.
    float metal = 0.0f, energy = 0.0f, metal_income = 0.0f, energy_income = 0.0f;
    float innate_metal = 0.0f, innate_energy = 0.0f, storage = 0.0f;
    float energy_made = 0.0f;       // per second, before overdrive
    float energy_spent = 0.0f;      // by builders, since the last overdrive update
    float overdrive_metal = 0.0f, overdrive_energy = 0.0f;      // per second, as of that update
    float reclaimed = 0.0f;     // metal, in the last tick
    bool free_factory = true;   // a side's first ploppable factory is free and instant (start_unit_setup.lua)
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

struct Map {
    int cols = 0, rows = 0, classes = 0;
    float cell = 16.0f;
    std::vector<float> height;
    std::vector<float> speed;
    std::vector<float> slope, normal_x, normal_z, slope_mod;
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
    int cell_index(float x, float z) const {
        int cx, cz;
        if (cell_pow2) {
            cx = int(x * inv_cell);
            cz = int(z * inv_cell);
        } else {
            cx = int(x / cell);
            cz = int(z / cell);
        }
        cx = cx < 0 ? 0 : (cx >= cols ? cols - 1 : cx);
        cz = cz < 0 ? 0 : (cz >= rows ? rows - 1 : cz);
        return cz * cols + cx;
    }
    float height_at(float x, float z) const { return height[size_t(cell_index(x, z))]; }
    const float* speed_grid(int cls) const { return &speed[size_t(cls) * size_t(rows) * size_t(cols)]; }
    float speed_at(int cls, float x, float z) const { return speed_grid(cls)[size_t(cell_index(x, z))]; }
    // CMoveMath's speed for a move in this direction: climbing slows a unit, descending doesn't.
    float speed_along(int cls, float x, float z, float dir_x, float dir_z) const { return speed_along_at(cls, size_t(cell_index(x, z)), dir_x, dir_z); }
    float speed_along_at(int cls, size_t i, float dir_x, float dir_z) const {
        const float base = speed_grid(cls)[i], mod = slope_mod[size_t(cls)], s = slope[i];
        if (base <= 0.0f || mod <= 0.0f || s <= 0.0f || height[i] < 0.0f) return base;
        const float climb = -(dir_x * normal_x[i] + dir_z * normal_z[i]);
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
};

// A shot in flight: damage is scheduled, there is no projectile.
struct Shot {
    int weapon, source, source_side, target;
    float x, z;
    bool guided, air;
    float power;   // of its damage, after falling with distance
    float origin_x, origin_z;
    bool aimed_at_blip;   // a guided one goes for where radar shows it
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
};

// A blip's speed record, as blips.py keeps it: its top speed is the highest speed it has held for a whole window.
struct BlipTrack {
    float speeds[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    int frames[5] = {0, 0, 0, 0, 0};
    int head = 0, filled = 0;
    float top = 0.0f;
};

// What one side knows about every unit in the game.
struct SideView {
    std::vector<uint8_t> in_los, in_radar, known, remembered;
    std::vector<int> seen_frame;
    int updated_frame = -1;
    std::vector<float> ghost_hp, ghost_build;
    std::vector<BlipTrack> tracks;
    int last_contact = 0;
};

struct World {
    const Catalog* cat;
    const Map* map;
    Rng rng;
    int behaviour;
    float cat_max_footprint = 0.0f;       // the widest footprint radius of any type
    int frame = 0;
    int tick_count = 0;

    // Units.
    std::vector<int> type, side, order, order_target, nearest;
    std::vector<uint8_t> alive;
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
    SideView views[2];
    std::vector<int> radar_units;
    // Economy, only for worlds that turn it on.
    bool economy_on = false;
    Purse purses[2];
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
    // Wrecks, by id.  One with nothing left stays in the list, so an id stays good, and leaves the others.
    std::vector<Wreck> wrecks;
    std::vector<int> live_wrecks;
    std::vector<std::vector<int>> wreck_cells;   // ids per grid cell: wrecks don't move
    // The live wrecks again, by grid cell (wreck_start, as cell_start), with their places alongside; remade when one comes or goes.
    std::vector<int> wreck_start, wreck_ids, wreck_slot;     // wreck_slot: by id, where a live one is in wreck_ids
    std::vector<float> wreck_px, wreck_pz;
    bool wreck_index_stale = true;
    std::vector<float> burn_rate, burn_frames;
    std::vector<FireArea> areas;
    // Weapons, ZK_MAX_MOUNTS per unit.
    std::vector<float> cooldown, salvo_timer;
    std::vector<int> salvo_left, weapon_target;

    // Spatial index, rebuilt each tick: the live units by side, then grid cell (row by row), then handle.  Side s's units in cell c
    // are cell_units[cell_start[k] .. cell_start[k + 1]] for k = s * grid_cells() + c, so a row of cells is one run of them.
    int grid_cols = 0, grid_rows = 0;
    std::vector<int> cell_start, cell_units, unit_key;
    size_t grid_cells() const { return size_t(grid_cols) * size_t(grid_rows); }
    std::vector<uint8_t> scratch_seen;     // per unit or wreck, for a sweep that marks what's in sight
    std::vector<int> scratch_eyes;         // a side's finished units, for the same
    struct EmitLists {
        std::vector<int> own, seen, blips, memory, hidden;
    } emit_lists;                          // emit_view's five kinds of record, in the order it writes them

    // Shots by the tick they land.
    std::vector<std::vector<Shot>> ring;
    std::vector<PierceHit> pierce_hits;

    World(const Catalog* c, const Map* m, uint64_t seed, int behaviour_);
    int add(int type, int side, float x, float z, float heading, float hp_frac, float build_frac);
    void order_unit(int unit, int kind, float gx, float gz, int target);
    void ready_weapons();
    void set_economy(int n_spots, const float* spot_triples, float metal, float energy, float innate_metal, float innate_energy, float storage);
    int add_wreck(float px, float pz, float metal, float wreck_hp, float radius, int feature_def, int heap_of);
    int emit_features(int side, FeatureRecord* out, int cap);
    void step(int ticks);
    int count() const { return int(type.size()); }
    void ensure_views();
    void update_view(int side);
    int emit_view(int side, UnitRecord* out, int cap, float* glob, int commander, const float* base);
    void side_stats(int side, float* out) const;
    void rebuild_index() { rebuild_grid(); }
    const zk_type_rec& type_of(int u) const { return cat->types[size_t(type[size_t(u)])]; }
    bool stunned(int u) const { return para[size_t(u)] > 1.0f; }
    bool disarmed(int u) const { return disarm[size_t(u)] > 1.0f; }
    bool power_off(int u) const { return economy_on && low_power[size_t(u)]; }
    const Teleporter* teleporter_of(int u) const;
    bool throw_ready(int u) const;
    float slow_now(int u) const { return slow[size_t(u)] < MAX_SLOW ? slow[size_t(u)] : MAX_SLOW; }
    bool active(int u) const { return build[size_t(u)] >= 1.0f && !stunned(u); }
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
    // The same, of one side only: for_near with the other side's left out, in the same order.
    template <class F>
    void for_side(int side, float cx, float cz, float radius, F&& f) const;

private:
    void tick();
    void rebuild_grid();
    void find_nearest();
    void pick_targets();
    void fire_weapons();
    void fire_one(int u, int slot, int frame_offset);
    bool pay_for_shot(int u, const zk_weapon_rec& w);
    int manual_mount(int u) const;
    void resolve_shots();
    void apply_shot(const Shot& s);
    void note_attacker(int u, int source, int source_side);
    void explode(int weapon, float px, float pz, int source, int source_side, bool air, float power = 1.0f, float height = 0.0f);
    void pierce(const Shot& s, float px, float pz, float power);
    void damage_unit(int u, const zk_weapon_rec& w, float mult);
    float shield_takes(const zk_weapon_rec& w, const Shot& s, float px, float pz, float left, float& hit_x, float& hit_z);
    void update_shields();
    void recharge_shields();
    void find_shield_links();
    void share_shield_charge();
    void move_units();
    void track_progress(int u, bool walking);
    void move_one(int u);
    bool wants_to_walk(int u, float& gx, float& gz);
    void separate();
    void finish_deaths();
    void leave_remains(int u);
    void hurt_wrecks(const zk_weapon_rec& w, float px, float pz, float power);
    void remove_wreck(int id);
    void index_wrecks();
    void reclaim_step(int u);
    bool pay_energy(int s, float amount);
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
    void set_dodge(int u, float tx, float tz);
    void unit_went_idle(int u);
    bool update_idle_aggression(int u, const zk_tactics_rec& b, int enemy, bool known, float default_range, float enemy_dist);
    bool skirm_enemy(int u, const zk_tactics_rec& b, int enemy, bool known, bool idle, bool fight_and_hold, bool hug);
    bool flee_enemy(int u, const zk_tactics_rec& b, int enemy, bool known, bool idle);
    bool tactics_against(int u, const zk_tactics_rec& b, int enemy, bool known, bool manual_attack, bool have_fight, bool hold, bool idle, bool particular);
    void run_tactics(int u);
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
    void update_overdrive();
    void overdrive_side(int s);
    void run_builder(int u);
    void run_factory(int u);
    void advance_frame(int builder, int frame_unit);
    int place_frame(int t, int s, float px, float pz, float head);
    int place_frame_only(int t, int s, float px, float pz, float head);
    float spot_income(float px, float pz) const;
    float reach_of(int u, float radius) const { return type_of(u).build_range + radius; }
    bool within(int u, float px, float pz, float reach) const;
    void update_status();
    void burn_areas();
    float target_score(int u, int slot, int target);
    bool in_range(int u, const zk_weapon_rec& w, int target) const;
    bool faces(int u, const zk_weapon_rec& w, int target) const;
    void waypoint(int u, float gx, float gz, float& wx, float& wz) const;
    bool clear_sight(int from, int to) const;
};

inline World::CellBox World::cell_box(float cx, float cz, float radius) const {
    constexpr float CELL = 128.0f;
    int x0 = int((cx - radius) / CELL), x1 = int((cx + radius) / CELL);
    int z0 = int((cz - radius) / CELL), z1 = int((cz + radius) / CELL);
    x0 = x0 < 0 ? 0 : x0;
    z0 = z0 < 0 ? 0 : z0;
    x1 = x1 >= grid_cols ? grid_cols - 1 : x1;
    z1 = z1 >= grid_rows ? grid_rows - 1 : z1;
    return {x0, x1, z0, z1};
}

template <class F>
void World::for_side(int s, float cx, float cz, float radius, F&& f) const {
    const CellBox b = cell_box(cx, cz, radius);
    const size_t base = size_t(s) * grid_cells();
    for (int gz = b.z0; gz <= b.z1; ++gz) {
        const size_t row = base + size_t(gz) * size_t(grid_cols);
        const int end = cell_start[row + size_t(b.x1) + 1];
        for (int i = cell_start[row + size_t(b.x0)]; i < end; ++i) f(cell_units[size_t(i)]);
    }
}

template <class F>
void World::for_near(float cx, float cz, float radius, F&& f) const {
    const CellBox b = cell_box(cx, cz, radius);
    const size_t other = grid_cells();
    for (int gz = b.z0; gz <= b.z1; ++gz) {
        const size_t row = size_t(gz) * size_t(grid_cols);
        const size_t first = row + size_t(b.x0), last = row + size_t(b.x1) + 1;
        // A row with one side in it is that side's run; otherwise each cell's two runs go together by handle.
        if (cell_start[other + first] == cell_start[other + last]) {
            for (int i = cell_start[first]; i < cell_start[last]; ++i) f(cell_units[size_t(i)]);
            continue;
        }
        if (cell_start[first] == cell_start[last]) {
            for (int i = cell_start[other + first]; i < cell_start[other + last]; ++i) f(cell_units[size_t(i)]);
            continue;
        }
        for (size_t c = first; c < last; ++c) {
            int i = cell_start[c], j = cell_start[other + c];
            const int i_end = cell_start[c + 1], j_end = cell_start[other + c + 1];
            while (i < i_end && j < j_end) f(cell_units[size_t(cell_units[size_t(i)] < cell_units[size_t(j)] ? i++ : j++)]);
            while (i < i_end) f(cell_units[size_t(i++)]);
            while (j < j_end) f(cell_units[size_t(j++)]);
        }
    }
}

}  // namespace zk
