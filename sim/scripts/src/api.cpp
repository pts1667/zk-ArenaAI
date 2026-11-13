// The C API of libzkscripts: the shared data, the runner with its thread pool, and a seat's plumbing.
#include <algorithm>
#include <atomic>
#include <climits>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>

#include "scripts.hpp"

namespace zks {

// Ground

int Ground::part_at(int c, double x, double z) const {
    if (c < 0 || c >= ZKS_MOVE_CLASSES || labels[c].empty()) return -1;
    int i = std::min(std::max((int)(z / cell), 0), rows - 1);
    int j = std::min(std::max((int)(x / cell), 0), cols - 1);
    int label = labels[c][(size_t)i * cols + j];
    if (label >= 0) return label;
    return largest[c];     // Ground.near gives a cell of the largest part, if there's one
}

// policy_opening.Ground.fits: whether the game would let a footprint of these half sizes be built here, but for what else
// stands there.  Its rectangle on the map, on ground or in water its data allows, and level: every cell under it within
// height_dif of the ground at its middle (the game's maxHeightDif; 0 for any).  One whose middle is under water isn't asked.
bool Ground::fits(double x, double z, double hx, double hz, double height_dif, int mask) const {
    if (x < hx || z < hz || x > map_w - hx || z > map_h - hz) return false;
    int r0 = (int)((z - hz) / cell), r1 = std::min((int)((z + hz) / cell) + 1, rows);
    int c0 = (int)((x - hx) / cell), c1 = std::min((int)((x + hx) / cell) + 1, cols);
    const std::vector<uint8_t>& m = masks[mask];
    for (int a = r0; a < r1; a++)
        for (int b = c0; b < c1; b++)
            if (!m[(size_t)a * cols + b]) return false;
    if (height_dif <= 0.0 || height.empty()) return true;
    float middle = height[(size_t)std::min((int)(z / cell), rows - 1) * cols + std::min((int)(x / cell), cols - 1)];
    if (middle < 0.0f) return true;
    for (int a = r0; a < r1; a++)
        for (int b = c0; b < c1; b++) {
            float off = height[(size_t)a * cols + b] - middle;
            if ((double)off > height_dif || (double)-off > height_dif) return false;
        }
    return true;
}

// Data

bool Data::can_build(int builder, int row) const {
    return std::binary_search(options.begin() + option_offsets[builder], options.begin() + option_offsets[builder + 1], row);
}

int Data::first_extractor_option(int builder) const {
    for (int k = option_offsets[builder]; k < option_offsets[builder + 1]; k++)
        if (has(options[k], ZKS_EXTRACTOR)) return options[k];
    return -1;
}

// Draws: xoshiro256**, seeded by splitmix64

namespace {
uint64_t splitmix(uint64_t& x) {
    uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
}  // namespace

void Draws::seed(uint64_t seed) {
    for (auto& v : s) v = splitmix(seed);
    has_spare = false;
    tape.clear();
    tape_kinds.clear();
    tape_at = 0;
    taped = false;
    misses = 0;
}

double Draws::own_uniform() {
    uint64_t r = rotl(s[1] * 5, 7) * 9;
    uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 45);
    return (double)(r >> 11) * 0x1.0p-53;
}

double Draws::own_normal() {
    if (has_spare) {
        has_spare = false;
        return spare;
    }
    double u, v, q;
    do {
        u = 2.0 * own_uniform() - 1.0;
        v = 2.0 * own_uniform() - 1.0;
        q = u * u + v * v;
    } while (q >= 1.0 || q == 0.0);
    double f = std::sqrt(-2.0 * std::log(q) / q);
    spare = v * f;
    has_spare = true;
    return u * f;
}

double Draws::uniform() {
    if (taped) {
        if (tape_at < tape.size() && tape_kinds[tape_at] == 0) return tape[tape_at++];
        misses++;
    }
    return own_uniform();
}

double Draws::normal() {
    if (taped) {
        if (tape_at < tape.size() && tape_kinds[tape_at] == 1) return tape[tape_at++];
        misses++;
    }
    return own_normal();
}

void Draws::end_step() {
    if (!taped) return;
    misses += (int)(tape.size() - tape_at);
    tape.clear();
    tape_kinds.clear();
    tape_at = 0;
    taped = false;
}

// Seat

void Seat::reset(const Data* d, const Ground* g, const Game& game_, int team_, P2 rally_, bool economy_, bool aimer_, uint64_t seed,
                 const zks_style* knobs) {
    data = d;
    ground = g;
    game = game_;
    team = team_;
    ally = game.ally_of[team];
    active = true;
    economy = economy_;
    aimer = aimer_;
    style = knobs ? *knobs : standard_style();
    base = game.starts[team], rally = rally_;
    n_foes = 0;
    for (int t = 0; t < game.n_teams; t++)
        if (game.ally_of[t] != ally) foes[n_foes++] = t;
    auto away = [&](int t) { return py_hypot(game.starts[t].x - base.x, game.starts[t].z - base.z); };
    std::stable_sort(foes, foes + n_foes, [&](int a, int b) { return away(a) < away(b); });
    foe = n_foes ? foes[0] : -1;
    enemy_base = foe >= 0 ? game.starts[foe] : P2{g->map_w * 0.5, g->map_h * 0.5};
    march = enemy_base;
    // _waypoints(rally, enemy_start), as numpy does it: the scouts' round, towards the nearest enemy's start
    const P2 enemy = enemy_base;
    double vx = enemy.x - rally_.x, vz = enemy.z - rally_.z;
    double sx = -vz * 0.3, sz = vx * 0.3;
    waypoints[0] = {rally_.x + vx * 0.3 + sx, rally_.z + vz * 0.3 + sz};
    waypoints[1] = {rally_.x + vx * 0.55 - sx, rally_.z + vz * 0.55 - sz};
    waypoints[2] = {rally_.x + vx * 0.8 + sx, rally_.z + vz * 0.8 + sz};
    waypoints[3] = {rally_.x + vx * 0.55, rally_.z + vz * 0.55};
    // One off the map is taken in to its edge: a scout sent past the edge stops there, never comes near enough to go on, and so
    // stays the whole game (an 80-minute game once ran on for 50 minutes with a scout plane its side's last unit).
    for (P2& w : waypoints) w = {std::min(std::max(w.x, 0.0), g->map_w), std::min(std::max(w.z, 0.0), g->map_h)};
    for (auto* v : {&state, &ordered_kind, &way}) v->clear();
    ordered_x.clear(), ordered_z.clear(), squad_of.clear(), current.clear(), born.clear(), patrol.clear(), helps.clear(), builds.clear();
    reclaims.clear(), still_x.clear(), still_z.clear(), still_since.clear();
    noted.clear(), stamp.clear(), fighter_of.clear(), raider_stamp.clear(), view_stamp.clear(), view_index.clear();
    lift_load.clear(), lifted.clear(), drop_goal.clear(), own_at.clear();
    lift_target = -1;
    lift_waiting = NAN;
    call = 0;
    squad_live.clear(), squad_target.clear(), since_has.clear(), since_t.clear(), since_d.clear();
    next_squad = 0;
    given_up.clear();
    has_waiting_since = false;
    waiting_since = 0.0;
    reached_start = false;
    made.assign(d->n_roles, 0);
    made_air.assign(d->n_roles, 0);
    seen_air = seen_kitable = seen_heavy = false;
    aimed.clear();
    log.clear();
    floating = energy_low = false;
    checks_eligible = checks_stalled = 0;
    counts.clear();
    nuke_drawn = -1;
    second = -2;
    for (double& p : pending) p = -1e9;
    going.clear();
    avoid.clear();
    for (int& f : finished) f = 0;
    draws.seed(seed);
}

void Seat::grow(int64_t uid) {
    size_t n = (size_t)uid + 1;
    if (state.size() >= n) return;
    state.resize(n, S_NEW);
    ordered_kind.resize(n, 0);
    ordered_x.resize(n, 0.0);
    ordered_z.resize(n, 0.0);
    way.resize(n, -1);
    squad_of.resize(n, -1);
    current.resize(n, -1);
    still_x.resize(n, 0.0);
    still_z.resize(n, 0.0);
    still_since.resize(n, NAN);
    born.resize(n, 0);
    patrol.resize(n, 0);
    helps.resize(n, INT64_MIN);
    reclaims.resize(n, INT64_MIN);
    builds.resize(n, -1);
    noted.resize(n, 0);
    lift_load.resize(n, -1);
    lifted.resize(n, 0);
    drop_goal.resize(n, -1);
    own_at.resize(n, 0);
    stamp.resize(n, -1);
    fighter_of.resize(n, 0);
    raider_stamp.resize(n, -1);
    view_stamp.resize(n, -1);
    view_index.resize(n, 0);
}

void Seat::run(const Ctx& c) {
    const Data& d = *data;
    own.clear(), enemies.clear(), allies.clear(), own_rows.clear(), enemy_rows.clear(), ally_rows.clear(), seen.clear();
    army_out.clear(), eco_out.clear();
    for (int& n : units_of) n = 0;
    int64_t most = -1;
    const Rec* frame = c.frames[team];
    for (int k = 0; k < c.frame_n[team]; k++) {
        const Rec* u = frame + k;
        if (u->player == team) {
            own.push_back(u);
            own_rows.push_back(d.row_of(u->def_id));
            most = std::max(most, u->uid);
        } else if (u->ally_team == ally) {
            allies.push_back(u);
            ally_rows.push_back(d.row_of(u->def_id));
            if (u->player >= 0 && u->player < ZKS_MAX_TEAMS) units_of[u->player]++;
        } else if (u->knowledge == zk::KNOW_LOS || u->knowledge == zk::KNOW_RADAR) {
            seen.push_back(u->uid);
        }
    }
    // What the scripts know of an enemy is all it has, as its own frame lists it.
    for (int t = 0; t < game.n_teams; t++) {
        if (game.ally_of[t] == ally) continue;
        for (int k = 0; k < c.frame_n[t]; k++) {
            const Rec* u = c.frames[t] + k;
            if (u->player != t) continue;
            enemies.push_back(u);
            enemy_rows.push_back(d.row_of(u->def_id));
            most = std::max(most, u->uid);
            units_of[t]++;
        }
    }
    // The enemy that counts this step is the nearest that's still in the game: so the army moves on once one is dead.
    foe = n_foes ? foes[0] : -1;
    for (int k = 0; k < n_foes; k++)
        if (units_of[foes[k]] > 0) {
            foe = foes[k];
            break;
        }
    if (foe >= 0) enemy_base = game.starts[foe];
    // Our half is what's nearer our base than any enemy's still in the game (all of theirs once none is); a team the scripts
    // don't play has no base, so with only those against us, all of it is.  Among allies, a spot is whoever's start is nearest.
    bases.clear();
    for (int k = 0; k < n_foes; k++)
        if (game.player(foes[k]) && units_of[foes[k]] > 0) bases.push_back(game.starts[foes[k]]);
    if (bases.empty())
        for (int k = 0; k < n_foes; k++)
            if (game.player(foes[k])) bases.push_back(game.starts[foes[k]]);
    partners.clear();
    for (int t = 0; t < game.n_teams; t++)
        if (t != team && game.ally_of[t] == ally && game.player(t) && units_of[t] > 0) partners.push_back({t, game.starts[t]});
    std::sort(seen.begin(), seen.end());
    if (most >= 0) grow(most);
    call++;
    for (size_t i = 0; i < own.size(); i++) stamp[own[i]->uid] = call, own_at[own[i]->uid] = (int32_t)i;
    if (c.army_due) army(c);
    if (c.eco_due) eco(c);
    draws.end_step();
}

// A pool of threads kept between steps.  Work is handed out an item at a time; the calling thread works too.

class Pool {
public:
    explicit Pool(int threads) {
        for (int t = 1; t < threads; t++) workers_.emplace_back([this] { loop(); });
    }
    ~Pool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
            generation_++;
        }
        wake_.notify_all();
        for (auto& w : workers_) w.join();
    }
    void run(int items, const std::function<void(int)>& fn) {
        if (workers_.empty() || items <= 1) {
            for (int i = 0; i < items; i++) fn(i);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            fn_ = &fn;
            items_ = items;
            next_.store(0);
            busy_ = (int)workers_.size();
            generation_++;
        }
        wake_.notify_all();
        work();
        std::unique_lock<std::mutex> lock(mutex_);
        done_.wait(lock, [this] { return busy_ == 0; });
        fn_ = nullptr;
    }

private:
    void work() {
        for (int i; (i = next_.fetch_add(1)) < items_;) (*fn_)(i);
    }
    void loop() {
        uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                wake_.wait(lock, [&] { return generation_ != seen; });
                seen = generation_;
                if (stop_) return;
            }
            work();
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (--busy_ == 0) done_.notify_one();
            }
        }
    }
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable wake_, done_;
    const std::function<void(int)>* fn_ = nullptr;
    std::atomic<int> next_{0};
    int items_ = 0, busy_ = 0;
    uint64_t generation_ = 0;
    bool stop_ = false;
};

}  // namespace zks

using namespace zks;

struct zks_data {
    Data d;
};

#ifdef ZK_MAX_TEAMS
static_assert(ZKS_MAX_TEAMS == ZK_MAX_TEAMS, "a game has room for as many teams as the sim's");
#endif

// Seats by game and team: teams * game + team.
struct zks_runner {
    const Data* data;
    int teams;
    std::vector<Seat> seats;
    Pool pool;
    std::vector<int32_t> listed;
    int total = 0, army_total = 0, group_total = 0;
    zks_runner(const Data* d, int games, int threads, int teams_)
        : data(d), teams(std::min(std::max(teams_, 1), ZKS_MAX_TEAMS)), seats((size_t)teams * games), pool(std::max(1, threads)) {}
    Seat& seat(int game, int team) { return seats[(size_t)teams * game + team]; }
    const Seat& seat(int game, int team) const { return seats[(size_t)teams * game + team]; }
};

extern "C" {

int zks_abi_version(void) { return ZKS_ABI_VERSION; }

double zks_py_hypot(double a, double b) { return py_hypot(a, b); }

zks_data* zks_data_new(const zks_data_desc* s) {
    auto* out = new zks_data;
    Data& d = out->d;
    int n = s->rows;
    d.rows = n;
    d.unknown = n - 1;
    d.row_of_def.assign(s->row_of_def, s->row_of_def + s->n_def);
    d.bits.assign(s->bits, s->bits + n);
    d.cost.assign(s->cost, s->cost + n);
    d.range.assign(s->range, s->range + n);
    d.move_class.assign(s->move_class, s->move_class + n);
    d.catalog_row.assign(s->catalog_row, s->catalog_row + n);
    d.role.assign(s->role, s->role + n);
    d.energy.assign(s->energy, s->energy + n);
    d.pylon.assign(s->pylon, s->pylon + n);
    d.needs_link.assign(s->needs_link, s->needs_link + n);
    d.half_x.assign(s->half_x, s->half_x + n);
    d.half_z.assign(s->half_z, s->half_z + n);
    d.height_dif.assign(s->height_dif, s->height_dif + n);
    d.footing.assign(s->footing, s->footing + n);
    d.best_missile.assign(s->best_missile, s->best_missile + n);
    d.generators.assign(s->generators, s->generators + 3 * n);
    d.option_offsets.assign(s->option_offsets, s->option_offsets + n + 1);
    d.options.assign(s->options, s->options + s->option_offsets[n]);
    d.n_roles = s->n_roles;
    d.menu.assign(s->menu, s->menu + (size_t)n * s->n_roles);
    d.plate_of.assign(s->plate_of, s->plate_of + n);
    d.lift.assign(s->lift, s->lift + 3 * (size_t)n);
    d.load_class.assign(s->load_class, s->load_class + n);
    d.n_transports = s->n_transports;
    d.fall_loss.assign(s->fall_loss, s->fall_loss + (size_t)s->n_transports * n);
    d.transport_of.assign(n, -1);
    for (int r = 0, k = 0; r < n; r++)
        if (d.has(r, ZKS_TRANSPORT)) d.transport_of[r] = k < d.n_transports ? k++ : -1;
    std::memcpy(d.role_ids, s->role_ids, sizeof d.role_ids);
    std::memcpy(d.special, s->special, sizeof d.special);
    std::memcpy(d.second_factories, s->second_factories, sizeof d.second_factories);
    d.caretaker = s->caretaker;
    std::memcpy(d.group_radii, s->group_radii, sizeof d.group_radii);
    d.cerberus_reach = s->cerberus_reach;
    d.missile_reach = s->missile_reach;
    d.spot_half = s->spot_half;
    d.n_footings = s->n_footings;
    d.kind.assign(n, -1);
    for (int r = 0; r < n; r++)
        for (int k = 0; k < R_COUNT; k++)
            if (d.role_ids[k] >= 0 && d.role[r] == d.role_ids[k]) d.kind[r] = (int8_t)k;
    // A plant whose fighting units fly (a constructor or scout of either kind doesn't decide it: a strider hub makes Athenas).
    d.air_plant.assign(n, 0);
    for (int r = 0; r < n; r++)
        for (int k = 0; k < d.n_roles; k++) {
            int u = d.menu[(size_t)r * d.n_roles + k];
            if (u >= 0 && k != d.role_ids[R_BUILDER] && k != d.role_ids[R_SCOUT] && d.has(u, ZKS_FIGHTS) && d.has(u, ZKS_AIR)) d.air_plant[r] = 1;
        }
    d.special_of.assign(n, -1);
    for (int k = 0; k < 7; k++)
        if (d.special[k] >= 0) d.special_of[d.special[k]] = (int8_t)k;
    int most = 0;
    for (int r = 0; r < n; r++) most = std::max(most, d.catalog_row[r] + 1);
    d.row_of_catalog.assign(most, -1);
    for (int r = 0; r < n; r++)
        if (d.catalog_row[r] >= 0 && d.row_of_catalog[d.catalog_row[r]] < 0) d.row_of_catalog[d.catalog_row[r]] = r;
    return out;
}

int zks_data_add_map(zks_data* data, const zks_map_desc* m) {
    Data& d = data->d;
    auto gp = std::make_unique<Ground>();
    Ground& g = *gp;
    g.rows = m->rows, g.cols = m->cols;
    g.cell = m->cell, g.map_w = m->map_w, g.map_h = m->map_h;
    g.n_spots = m->n_spots;
    g.spots.assign(m->spots, m->spots + 3 * m->n_spots);
    size_t cells = (size_t)m->rows * m->cols;
    for (int f = 0; f < d.n_footings; f++) {
        g.masks.emplace_back(m->masks[f], m->masks[f] + cells);
        g.sites.emplace_back(m->sites[f], m->sites[f] + 2 * (size_t)m->site_counts[f]);
    }
    for (int c = 0; c < ZKS_MOVE_CLASSES; c++) {
        g.labels.emplace_back(m->labels[c], m->labels[c] + cells);
        g.largest[c] = m->largest[c];
    }
    if (m->height) g.height.assign(m->height, m->height + cells);
    d.maps.push_back(std::move(gp));
    return (int)d.maps.size() - 1;
}

void zks_data_free(zks_data* data) { delete data; }

zks_runner* zks_runner_new(const zks_data* data, int games, int threads, int teams) {
    return new zks_runner(&data->d, games, threads, teams);
}

void zks_runner_free(zks_runner* runner) { delete runner; }

// A seat's pointers are to the shared data and map, and to the frames of its last step, which the next one fills afresh.
void zks_copy_game(zks_runner* r, int src, int dst) {
    if (src == dst) return;
    for (int t = 0; t < r->teams; ++t) r->seat(dst, t) = r->seat(src, t);
}

void zks_reset(zks_runner* r, int game, int map, int n_teams, const int32_t* ally_of_team, uint32_t script_teams, const double* starts,
               const double* rally, int economy, int aimer, const uint64_t* seeds, const zks_style* styles) {
    const Ground* g = r->data->maps[map].get();
    Game info;
    info.n_teams = std::min(std::max(n_teams, 0), r->teams);
    for (int t = 0; t < info.n_teams; t++) {
        info.ally_of[t] = ally_of_team ? ally_of_team[t] : t;
        info.starts[t] = {starts[2 * t], starts[2 * t + 1]};
        if ((script_teams >> t) & 1u) info.players |= 1u << t;
    }
    for (int t = 0; t < r->teams; t++) {
        Seat& seat = r->seat(game, t);
        if (info.player(t))
            seat.reset(r->data, g, info, t, {rally[2 * t], rally[2 * t + 1]}, economy != 0, aimer != 0, seeds ? seeds[t] : 0,
                       styles ? styles + t : nullptr);
        else
            seat = Seat();      // no seat: nothing runs for it, and nothing it measured is left from a game before
    }
}

int zks_step(zks_runner* r, const zks_step_args* a, zk_order_rec* out, int out_cap, int32_t* out_offsets, zk_order_rec* army,
             int army_cap, int32_t* army_offsets, zk_group_rec* groups, int group_cap, int32_t* group_offsets) {
    const int n = a->n, T = r->teams;
    r->listed.assign(a->games, a->games + n);
    const Rec* records = (const Rec*)a->records;
    for (int s = 0; s < T * n; s++) {
        Seat& seat = r->seat(a->games[s / T], s % T);
        seat.pre.assign(a->pre ? a->pre + a->pre_offsets[s] : nullptr, a->pre ? a->pre + a->pre_offsets[s + 1] : nullptr);
        seat.post.assign(a->post ? a->post + a->post_offsets[s] : nullptr, a->post ? a->post + a->post_offsets[s + 1] : nullptr);
        seat.flags = a->seat_flags[s];
    }
    r->pool.run(T * n, [&](int s) {
        const int g = a->games[s / T], team = s % T;
        const size_t at = (size_t)g * T + team;
        Seat& seat = r->seat(g, team);
        seat.army_out.clear(), seat.eco_out.clear(), seat.groups.clear(), seat.grouped.clear(), seat.anchor_of.clear();
        if (!seat.active) return;
        int d = a->decisions[g];
        Ctx c;
        for (int t = 0; t < seat.game.n_teams; t++) {
            c.frames[t] = records + ((size_t)g * T + t) * a->cap;
            c.frame_n[t] = a->counts[(size_t)g * T + t];
        }
        c.glob = a->globs + at * zk::G_COUNT;
        c.reclaim = a->stats ? a->stats[at * ZK_STATS + 2] : 0.0;
        c.busy = a->busy[s];
        c.seconds = (double)d * a->ticks * a->tick_frames / a->frames_per_second;
        c.army_due = (a->seat_flags[s] & (ZKS_SEAT_ARMY | ZKS_SEAT_ARMY_LABEL)) && d % a->script_every == 0;
        c.eco_due = seat.economy && d % a->eco_every == 0;
        c.army_frames = (double)a->script_every * a->ticks * a->tick_frames;
        c.groups = a->group_orders != 0 && !(a->seat_flags[s] & ZKS_SEAT_ARMY_LABEL);
        if (a->features) {
            c.features = (const zk::FeatureRecord*)a->features + at * a->feature_cap;
            c.feature_n = a->feature_counts[at];
        }
        const int chickens = seat.game.chickens();
        if (chickens >= 0 && a->chickens) {
            const float* cs = a->chickens + (size_t)g * ZK_CHICKEN_STATE;
            if ((int)cs[ZK_CS_TEAM] == chickens) {
                c.chickens.known = true;
                c.chickens.anger = cs[ZK_CS_AGGRO];
                c.chickens.queen = cs[ZK_CS_QUEEN] == 1.0f;
                c.chickens.until_queen = cs[ZK_CS_QUEEN] == 0.0f ? (double)cs[ZK_CS_UNTIL_QUEEN] : 0.0;
            }
        }
        if (!c.army_due && !c.eco_due) {
            seat.draws.end_step();
            return;
        }
        seat.run(c);
    });
    int total = 0, army_total = 0, group_total = 0;
    for (int s = 0; s < T * n; s++) {
        const Seat& seat = r->seat(a->games[s / T], s % T);
        int alone = 0;
        for (size_t i = 0; i < seat.army_out.size(); i++) alone += !(i < seat.grouped.size() && seat.grouped[i]);
        const bool army_runs = !(seat.flags & ZKS_SEAT_ARMY_LABEL), eco_runs = !(seat.flags & ZKS_SEAT_ECO_LABEL);
        total += (int)(seat.pre.size() + seat.post.size() + (eco_runs ? seat.eco_out.size() : 0)) + (army_runs ? alone : 0);
        army_total += alone + 2 * (int)seat.groups.size();
        group_total += (int)seat.groups.size();
    }
    r->total = total, r->army_total = army_total, r->group_total = group_total;
    if (total > out_cap || army_total > army_cap || group_total > group_cap) return total;
    return zks_collect(r, out, out_cap, out_offsets, army, army_cap, army_offsets, groups, group_cap, group_offsets);
}

int zks_collect(zks_runner* r, zk_order_rec* out, int out_cap, int32_t* out_offsets, zk_order_rec* army, int army_cap,
                int32_t* army_offsets, zk_group_rec* groups, int group_cap, int32_t* group_offsets) {
    if (r->total > out_cap || r->army_total > army_cap || r->group_total > group_cap) return -1;
    const int n = (int)r->listed.size(), T = r->teams;
    int o = 0, a = 0, g = 0;
    out_offsets[0] = 0;
    army_offsets[0] = 0;
    group_offsets[0] = 0;
    for (int k = 0; k < n; k++) {
        for (int team = 0; team < T; team++) {
            const Seat& seat = r->seat(r->listed[k], team);
            const bool army_runs = !(seat.flags & ZKS_SEAT_ARMY_LABEL), eco_runs = !(seat.flags & ZKS_SEAT_ECO_LABEL);
            for (const auto& order : seat.pre) out[o++] = order;
            if (army_runs)
                for (size_t i = 0; i < seat.army_out.size(); i++)
                    if (!(i < seat.grouped.size() && seat.grouped[i])) out[o++] = seat.army_out[i];
            for (const auto& order : seat.post) out[o++] = order;
            if (eco_runs)
                for (const auto& order : seat.eco_out) out[o++] = order;
            for (size_t i = 0; i < seat.army_out.size(); i++) {
                const bool in_group = i < seat.grouped.size() && seat.grouped[i];
                const int anchor = i < seat.anchor_of.size() ? seat.anchor_of[i] : -1;
                if (!in_group || anchor >= 0) army[a++] = seat.army_out[i];
                if (anchor >= 0) {
                    const zk_group_rec& g = seat.groups[size_t(anchor)];
                    zk_order_rec mark{};
                    mark.unit = g.anchor, mark.kind = ZKS_GROUP_MARK, mark.x = g.radius, mark.z = 0.0f, mark.target = (int32_t)g.roles;
                    army[a++] = mark;
                }
            }
            army_offsets[T * k + team + 1] = a;
            for (const auto& group : seat.groups) groups[g++] = group;
        }
        out_offsets[k + 1] = o;
        group_offsets[k + 1] = g;
    }
    return o;
}

void zks_last_counts(const zks_runner* r, int32_t* counts) {
    counts[0] = r->total;
    counts[1] = r->army_total;
    counts[2] = r->group_total;
}

void zks_seat_counts(const zks_runner* r, int32_t* counts) {
    const int T = r->teams;
    for (size_t k = 0; k < r->listed.size(); k++)
        for (int team = 0; team < T; team++) {
            const Seat& s = r->seat(r->listed[k], team);
            int32_t* c = counts + 4 * (T * k + team);
            c[0] = (int32_t)s.pre.size(), c[1] = (int32_t)s.army_out.size(), c[2] = (int32_t)s.post.size(), c[3] = (int32_t)s.eco_out.size();
        }
}

int zks_eco_orders(const zks_runner* r, zk_order_rec* out, int cap, int32_t* offsets) {
    const int T = r->teams;
    int total = 0;
    for (int32_t g : r->listed)
        for (int team = 0; team < T; team++) total += (int)r->seat(g, team).eco_out.size();
    if (total > cap) return total;
    int o = 0;
    offsets[0] = 0;
    for (size_t k = 0; k < r->listed.size(); k++)
        for (int team = 0; team < T; team++) {
            for (const auto& order : r->seat(r->listed[k], team).eco_out) out[o++] = order;
            offsets[T * k + team + 1] = o;
        }
    return total;
}

int zks_made(const zks_runner* r, int game, int team, int32_t* made) {
    const Seat& s = r->seat(game, team);
    for (int k = 0; k < r->data->n_roles; k++) made[k] = k < (int)s.made.size() ? s.made[k] : 0;
    return s.active;
}

void zks_eco_stats(const zks_runner* r, int game, int team, int32_t* out) {
    const Seat& s = r->seat(game, team);
    out[0] = s.checks_eligible;
    out[1] = s.checks_stalled;
    for (int k = 0; k < 7; k++) out[2 + k] = s.finished[k];
}

int zks_log(const zks_runner* r, int game, int team, double* seconds, int32_t* what, int32_t* row, int cap) {
    const Seat& s = r->seat(game, team);
    int n = (int)s.log.size();
    for (int k = 0; k < n && k < cap; k++) seconds[k] = s.log[k].seconds, what[k] = s.log[k].what, row[k] = s.log[k].row;
    return n;
}

void zks_tape(zks_runner* r, int game, int team, const double* values, const uint8_t* kinds, int n) {
    Draws& d = r->seat(game, team).draws;
    d.tape.assign(values, values + n);
    d.tape_kinds.assign(kinds, kinds + n);
    d.tape_at = 0;
    d.taped = true;
}

int zks_tape_misses(const zks_runner* r, int game, int team) { return r->seat(game, team).draws.misses; }

}  // extern "C"
