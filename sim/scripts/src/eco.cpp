// The economy script (tools/policy_eco.py Economy).  Each function follows the Python one of the same name.  Where numpy
// computes in float32 (an array of the frame's float32 against a Python float, which numpy casts to float32 first), so does
// this; where it computes in float64, so does this.
#include <algorithm>
#include <climits>
#include <cmath>
#include <numeric>

#include "scripts.hpp"

namespace zks {

namespace {

constexpr double INNATE_ENERGY = 2.0;
const double ENERGY_SHARE_X[4] = {0.0, 4.0, 8.0, 14.0}, ENERGY_SHARE_Y[4] = {1.25, 1.75, 2.5, 3.0};
constexpr double ENERGY_FIRST = 1.5;
constexpr float SMALL_ENERGY_BELOW = 10.0f;
constexpr double FUSION_FROM_S = 240.0, FUSION_FROM_M = 15.0;
constexpr double SINGULARITY_FROM_S = 720.0, SINGULARITY_FROM_M = 45.0;
constexpr int SMALL_PER_PASS = 8;
constexpr double LINK_SLACK = 0.85;
constexpr double BEHIND_BASE = 250.0;
constexpr double SITE_SPREAD = 120.0;
constexpr double FINISH_REACH = 1200.0;
constexpr float SAFE_FROM_ARMY = 600.0f;
constexpr float COMMANDER_REACH = 1500.0f;
constexpr float SAFE_MARGIN = 100.0f;
constexpr double OUR_HALF = 1.0;
constexpr double BIG_FROM_S = 960.0, BIG_FROM_M = 55.0;
constexpr double STALE_FOR = 120.0;
constexpr double STALE_SHARE = 0.1;
constexpr double STALE_ENDS = 30.0;
constexpr int MAX_TURRETS = 3;
constexpr double FRONT_BACK = 250.0;
constexpr double REACH_SHARE = 0.85;
constexpr double NUKE_FROM_S = 1080.0, NUKE_FROM_M = 60.0;
constexpr double NUKE_CHANCE = 0.3;
constexpr double PENDING_FOR = 60.0;
constexpr double STUCK_AFTER = 30.0;
constexpr double STUCK_NEARER = 50.0;
constexpr double AVOID_FOR = 180.0;
constexpr double FLOATING = 0.5;
constexpr double FLOATING_UNTIL = 0.15;
constexpr float FLOAT_ASSIST_REACH = 3000.0f;
constexpr double ENERGY_LOW = 0.3;
constexpr int FLOAT_EXPANDERS = 4;
// Metal that floats (the user, 2026-11-10): help a production line (a factory or plate) that fewer than HELPERS_PER_LINE are
// helping, else more production beside our first factory while income allows it: its plate, a line of its own for a fifth of a
// factory's metal, and once it has one, caretakers in reach of both, which help them (the sim's World::run_caretaker); else help
// the nearest line all the same.  Never a second factory like the first, nor a second plate.  A plate builds only within its
// build range of its factory (World::plate_off).
constexpr double PRODUCTION_PER_INCOME = 12.0;  // the factory, its plate and each caretaker (10 build power each): one for this much metal a second
constexpr int MOST_PRODUCTION = 8;
constexpr int HELPERS_PER_LINE = 2;
constexpr double PLATE_SPREAD = 120.0;
constexpr double PLATE_NEAR = 300.0;            // of the factory's centre; a plate's build range is 420
constexpr double CARETAKER_SPREAD = 100.0;
constexpr double CARETAKER_NEAR = 380.0;        // of the factory's centre and the plate's; a caretaker's build range is 500
constexpr double RADAR_FROM = 180.0;
constexpr int SITE_TRIES = 24;           // policy_opening.Ground.site
constexpr double NEVER = -1e9;           // pending.get(name, -1e9)
// The second factory (the user's rule, 2026-11-10): at 40 metal a second not counting reclaim, or at 20 minutes, which the games
// don't reach.  What it is is drawn once, when it's due: planes most often, then gunships, then striders (the user's play).  A
// hub runs only on a grid of enough energy, so with none it's planes.  No third: more production is a plate and caretakers.
constexpr double SECOND_FROM_S = 20.0 * 60.0, SECOND_FROM_M = 40.0;
const double SECOND_WEIGHTS[3] = {0.5, 0.3, 0.2};     // zks_data_desc.second_factories' order
// Reclaim (the user, 2026-11-09): a few constructors kept at it while there are wrecks worth taking, and an idle one takes what's
// near it when there's nothing better to do.  Only what's safe (no enemy army or defence near it) and on our side of the map, or
// a little past the middle, where the fighting leaves most.  A reclaimer goes home if an enemy army comes near or it's hurt, and
// a hurt constructor isn't sent again.
constexpr float RECLAIM_MIN = 30.0f;          // metal: less isn't worth the walk
constexpr int RECLAIMERS = 2;                 // at most this many kept at it
constexpr int RECLAIMER_PER = 3;              // one for each this many constructors (the commander isn't one)
constexpr double RECLAIM_HALF = 1.15;         // as near our base as this times the enemy's, or nearer
constexpr double RECLAIM_NEAR = 1200.0;       // an idle constructor's reach for it
constexpr double RECLAIM_WALK = 400.0;        // a wreck's worth is its metal / (distance + this)
constexpr float RECLAIM_FLEE = 500.0f;        // a reclaimer with an enemy army this near goes home
constexpr float RECLAIM_HURT = 0.5f;          // and one below this share of its health
constexpr float RECLAIM_FIT = 0.7f;           // a constructor below this isn't sent to reclaim

zk_order_rec order(int64_t uid, int kind, double x, double z, int64_t target) {
    zk_order_rec o;
    o.unit = (int32_t)uid;
    o.kind = kind;
    o.x = (float)x;
    o.z = (float)z;
    o.target = (int32_t)target;
    return o;
}

Job make(int64_t uid, int kind, double x, double z, int64_t target) {
    Job j;
    j.has = true;
    j.o = order(uid, kind, x, z, target);
    return j;
}

bool built(const Rec* u) { return (u->flags & zk::UF_BEING_BUILT) == 0; }

}  // namespace

// policy_eco.Grid: a side's pylon grids
struct GridOf {
    std::vector<double> x, z, reach;
    std::vector<int> label, row;
    int main = -1;
    double main_energy = 0.0;       // what the main grid makes

    bool links(double px, double pz, double r, int grid) const {
        for (size_t a = 0; a < x.size(); a++)
            if (label[a] == grid && std::hypot(x[a] - px, z[a] - pz) < LINK_SLACK * (reach[a] + r)) return true;
        return false;
    }
    int nearest(double px, double pz, int grid) const {
        int best = -1;
        double bd = 0.0;
        for (size_t a = 0; a < x.size(); a++) {
            if (label[a] != grid) continue;
            double dd = std::hypot(x[a] - px, z[a] - pz);
            if (best < 0 || dd < bd) best = (int)a, bd = dd;
        }
        return best;
    }
};

// policy_eco._Pass
struct Pass {
    double made = 0.0;
    int small = 0;
    std::vector<float> taken;       // n x 3
    int pylons = 0;
    bool radar = false, factory = false, second = false;
    int expanding = 0;
    std::vector<double> claimed;    // n x 2
    std::vector<int64_t> reclaimed; // features someone of ours is reclaiming or was sent to this pass
    int reclaimers = 0, reclaimers_wanted = 0;
    std::vector<int64_t> helping;   // what each ASSIST of this pass helps: every constructor helping is idle, so asked again
    int helpers(int64_t uid) const { return (int)std::count(helping.begin(), helping.end(), uid); }
};

bool Seat::avoided(double x, double z) const {
    int64_t ix = floordiv(x, 64.0), iz = floordiv(z, 64.0);
    for (auto& a : avoid)
        if (a.ix == ix && a.iz == iz) return true;
    return false;
}

std::vector<uint8_t> Seat::stuck(const Ctx& c, const std::vector<uint8_t>& builders) {
    const double seconds = c.seconds;
    std::vector<uint8_t> out(own.size(), 0);
    std::vector<int64_t> alive;
    for (size_t i = 0; i < own.size(); i++) {
        const Rec* u = own[i];
        if (!builders[i] || u->ord_kind != K_BUILD) continue;
        int64_t uid = u->uid;
        alive.push_back(uid);
        double gx = u->ord_x, gz = u->ord_z;
        double d = py_hypot((double)u->x - gx, (double)u->z - gz);
        Going* was = nullptr;
        for (auto& g : going)
            if (g.uid == uid) was = &g;
        if (!was || py_hypot(was->gx - gx, was->gz - gz) > 1.0 || d < was->nearest - STUCK_NEARER) {
            if (was) *was = {uid, gx, gz, seconds, d};
            else going.push_back({uid, gx, gz, seconds, d});
        } else if (seconds - was->since >= STUCK_AFTER && d > 300.0) {
            out[i] = 1;
            int64_t ix = floordiv(gx, 64.0), iz = floordiv(gz, 64.0);
            bool found = false;
            for (auto& a : avoid)
                if (a.ix == ix && a.iz == iz) a.until = seconds + AVOID_FOR, found = true;
            if (!found) avoid.push_back({ix, iz, seconds + AVOID_FOR});
            going.erase(going.begin() + (was - going.data()));
        }
    }
    going.erase(std::remove_if(going.begin(), going.end(),
                               [&](const Going& g) { return std::find(alive.begin(), alive.end(), g.uid) == alive.end(); }),
                going.end());
    avoid.erase(std::remove_if(avoid.begin(), avoid.end(), [&](const Avoid& a) { return a.until <= seconds; }), avoid.end());
    return out;
}

bool Seat::stalled(double seconds) const {
    size_t first = 0;
    while (first < counts.size() && !(counts[first].t >= seconds - STALE_FOR)) first++;
    if (first == counts.size() || seconds - counts[first].t < STALE_FOR - 10.0) return false;
    double t0 = counts[first].t;
    double s_ours = 0.0, s_theirs = 0.0, e_ours = 0.0, e_theirs = 0.0;
    int ns = 0, ne = 0;
    for (size_t k = first; k < counts.size(); k++) {
        const Count& q = counts[k];
        if (!(q.t >= seconds - STALE_FOR)) continue;
        if (q.t <= t0 + STALE_ENDS) s_ours += q.ours, s_theirs += q.theirs, ns++;
        if (q.t >= seconds - STALE_ENDS) e_ours += q.ours, e_theirs += q.theirs, ne++;
    }
    double allowed = std::max(2.0, STALE_SHARE * ground->n_spots);
    return std::fabs(e_ours / ne - s_ours / ns) <= allowed && std::fabs(e_theirs / ne - s_theirs / ns) <= allowed;
}

int Seat::have(int x, double seconds) const {
    int row = data->special[x];
    int n = 0;
    for (int r : own_rows) n += r == row;
    return (seconds - pending[x] < PENDING_FOR && n == 0) ? 1 : n;
}

bool Seat::ours(P2 p) const {
    return py_hypot(p.x - base.x, p.z - base.z) <= OUR_HALF * py_hypot(p.x - enemy_base.x, p.z - enemy_base.z);
}

P2 Seat::front() const {
    const Data& d = *data;
    float ex = (float)enemy_base.x, ez = (float)enemy_base.z;
    int best = -1;
    float bd = 0.0f;
    for (size_t i = 0; i < own.size(); i++) {
        if (!d.has(own_rows[i], ZKS_EXTRACTOR) || !built(own[i])) continue;
        float dd = hypotf(own[i]->x - ex, own[i]->z - ez);
        if (best < 0 || dd < bd) best = (int)i, bd = dd;
    }
    if (best < 0) return base;
    return {(double)own[best]->x, (double)own[best]->z};
}

bool Seat::silo_site(P2& site) const {
    const Data& d = *data;
    size_t ne = enemies.size();
    std::vector<uint8_t> valuable(ne);
    int fusions = 0;
    for (size_t e = 0; e < ne; e++) {
        valuable[e] = d.has(enemy_rows[e], ZKS_VALUABLE) && built(enemies[e]);
        fusions += d.special[X_FUSION] >= 0 && enemy_rows[e] == d.special[X_FUSION] && built(enemies[e]);
    }
    if (fusions >= 2)
        for (size_t e = 0; e < ne; e++)
            if (enemy_rows[e] == d.special[X_FUSION] && built(enemies[e])) valuable[e] = 1;
    std::vector<int> order_;
    for (size_t e = 0; e < ne; e++)
        if (valuable[e]) order_.push_back((int)e);
    std::stable_sort(order_.begin(), order_.end(), [&](int a, int b) { return -d.cost[enemy_rows[a]] < -d.cost[enemy_rows[b]]; });
    double reach = REACH_SHARE * d.missile_reach;
    for (int e : order_) {
        P2 target{(double)enemies[e]->x, (double)enemies[e]->z};
        double away = py_hypot(target.x - base.x, target.z - base.z);
        P2 s = along(target, base, away < reach ? away : reach);
        if (ours(s)) {
            site = s;
            return true;
        }
    }
    return false;
}

bool Seat::turret(int& row, P2& site) const {
    const Data& d = *data;
    P2 fr = front();
    if (d.special[X_CERBERUS] >= 0) {
        float fx = (float)fr.x, fz = (float)fr.z;
        int best = -1;
        float bd = 0.0f;
        for (size_t e = 0; e < enemies.size(); e++) {
            int r = enemy_rows[e];
            if (d.has(r, ZKS_MOBILE) || !built(enemies[e]) || r == d.unknown) continue;
            float dd = hypotf(enemies[e]->x - fx, enemies[e]->z - fz);
            if (best < 0 || dd < bd) best = (int)e, bd = dd;
        }
        if (best >= 0) {
            P2 target{(double)enemies[best]->x, (double)enemies[best]->z};
            P2 s = along(target, base, REACH_SHARE * d.cerberus_reach);
            bool behind_front = py_hypot(s.x - enemy_base.x, s.z - enemy_base.z) >=
                                py_hypot(fr.x - enemy_base.x, fr.z - enemy_base.z) - FRONT_BACK;
            if (behind_front && ours(s)) {
                row = d.special[X_CERBERUS], site = s;
                return true;
            }
        }
    }
    if (d.special[X_LUCIFER] >= 0) {
        row = d.special[X_LUCIFER], site = along(fr, base, FRONT_BACK);
        return true;
    }
    return false;
}

bool Seat::project(const Ctx& c, double income, const GridOf& grid, int& prow, P2& at) {
    (void)grid;
    const Data& d = *data;
    const double seconds = c.seconds;
    for (int x : {X_NUKE, X_SILO, X_CERBERUS, X_LUCIFER}) {
        int r = d.special[x];
        if (r < 0) continue;
        for (size_t i = 0; i < own.size(); i++)
            if (own_rows[i] == r && !built(own[i])) {
                prow = r, at = {(double)own[i]->x, (double)own[i]->z};
                return true;
            }
    }
    if (seconds >= NUKE_FROM_S && income > NUKE_FROM_M && d.special[X_NUKE] >= 0) {
        if (nuke_drawn < 0) nuke_drawn = draws.uniform() < NUKE_CHANCE;
        if (nuke_drawn && !have(X_NUKE, seconds)) {
            prow = d.special[X_NUKE], at = along(base, enemy_base, -BEHIND_BASE);
            return true;
        }
    }
    if (seconds < BIG_FROM_S || income < BIG_FROM_M) return false;
    bool st = stalled(seconds);
    checks_eligible++;
    checks_stalled += st;
    if (d.special[X_SILO] >= 0 && !have(X_SILO, seconds)) {
        P2 s;
        if (silo_site(s)) {
            prow = d.special[X_SILO], at = s;
            return true;
        }
    }
    int turrets = 0;
    for (int x : {X_CERBERUS, X_LUCIFER})
        if (d.special[x] >= 0) turrets += have(x, seconds);
    bool recent = seconds - pending[X_CERBERUS] < PENDING_FOR || seconds - pending[X_LUCIFER] < PENDING_FOR;
    if (st && turrets < MAX_TURRETS && !recent) return turret(prow, at);
    return false;
}

// policy_opening.Ground.site
bool Seat::site(int mask, double radius, P2 near, const std::vector<float>& taken, double spread, P2& place) {
    const Ground& g = *ground;
    size_t nt = taken.size() / 3;
    float rf = (float)radius;
    auto clear = [&](double px, double pz) {
        float fx = (float)px, fz = (float)pz;
        for (size_t t = 0; t < nt; t++)
            if (!(hypotf(taken[3 * t] - fx, taken[3 * t + 1] - fz) >= taken[3 * t + 2] + rf)) return false;
        return true;
    };
    for (int k = 0; k < SITE_TRIES; k++) {
        double px = near.x + (0.0 + spread * draws.normal());
        double pz = near.z + (0.0 + spread * draws.normal());
        if (g.fits(px, pz, radius, mask) && clear(px, pz)) {
            place = {px, pz};
            return true;
        }
    }
    const std::vector<float>& cells = g.sites[mask];
    size_t nc = cells.size() / 2;
    if (!nc) return false;
    float fx = (float)near.x, fz = (float)near.z;
    std::vector<float> dist(nc);
    for (size_t i = 0; i < nc; i++) dist[i] = hypotf(cells[2 * i] - fx, cells[2 * i + 1] - fz);
    // np.argsort's quicksort, then the first SITE_TRIES.  Ties are taken in index order here; numpy's order for them isn't
    // specified.
    std::vector<int> idx(nc);
    std::iota(idx.begin(), idx.end(), 0);
    size_t take = std::min(nc, (size_t)SITE_TRIES);
    std::partial_sort(idx.begin(), idx.begin() + take, idx.end(),
                      [&](int a, int b) { return dist[a] < dist[b] || (dist[a] == dist[b] && a < b); });
    for (size_t k = 0; k < take; k++) {
        double px = cells[2 * idx[k]], pz = cells[2 * idx[k] + 1];
        if (g.fits(px, pz, radius, mask) && clear(px, pz)) {
            place = {px, pz};
            return true;
        }
    }
    return false;
}

Job Seat::build(int64_t uid, int row, P2 near, const std::vector<float>& taken, double spread, const GridOf* grid, int on,
                int by_class, P2 by_at, bool use_by) {
    const Data& d = *data;
    int mask = d.footing[row];
    if (mask < 0) return {};
    for (int k = 0; k < 4; k++) {
        P2 place;
        if (!site(mask, d.footing_radius[row], near, taken, spread, place)) return {};
        if (avoided(place.x, place.z) || (use_by && by_class != -1 && !ground->joined(by_class, by_at, place))) continue;
        if (!grid || on < 0 || grid->links(place.x, place.z, (double)d.pylon[row], on))
            return make(uid, K_BUILD, place.x, place.z, d.catalog_row[row]);
    }
    return {};
}

Job Seat::energy_job(const Ctx& c, int i, const GridOf& grid, Pass& pass, double income, const std::vector<int>& being, bool force) {
    const Data& d = *data;
    const double seconds = c.seconds;
    int64_t uid = own[i]->uid;
    int row = own_rows[i];
    double wanted = interp(seconds / 60.0, ENERGY_SHARE_X, ENERGY_SHARE_Y, 4) * income;
    if (pass.made >= wanted && !force) return {};
    int small = d.generators[3 * row], big = d.generators[3 * row + 1], biggest = d.generators[3 * row + 2];
    bool big_being = false;
    for (int j : being) big_being |= d.energy[own_rows[j]] >= SMALL_ENERGY_BELOW;
    int choice = -1;
    if (!big_being) {
        if (biggest >= 0 && seconds >= SINGULARITY_FROM_S && income >= SINGULARITY_FROM_M) choice = biggest;
        else if (big >= 0 && seconds >= FUSION_FROM_S && income >= FUSION_FROM_M) choice = big;
    }
    if (choice >= 0) {
        int main = grid.main;
        P2 near = base;
        if (main >= 0) {
            int k = grid.nearest(base.x, base.z, main);
            near = {grid.x[k], grid.z[k]};
        }
        near = along(near, enemy_base, -BEHIND_BASE * 0.5);
        Job job = build(uid, choice, near, pass.taken, SITE_SPREAD, &grid, main, -1, {}, false);
        if (!job.has) job = build(uid, choice, near, pass.taken, SITE_SPREAD, nullptr, -1, -1, {}, false);
        if (job.has) {
            pass.made += (double)d.energy[choice];
            log.push_back({seconds, 0, choice});
        }
        return job;
    }
    if (small < 0 || pass.small >= SMALL_PER_PASS) return {};
    P2 spot = base;
    std::vector<int> ext, makers;
    for (size_t j = 0; j < own.size(); j++) {
        if (d.has(own_rows[j], ZKS_EXTRACTOR) && built(own[j])) ext.push_back((int)j);
        if (d.energy[own_rows[j]] > 0.0f) makers.push_back((int)j);
    }
    if (!ext.empty()) {
        int best = -1;
        double bk = 0.0;
        for (size_t e = 0; e < ext.size(); e++) {
            const Rec* x = own[ext[e]];
            int64_t near_energy = 0;
            for (int m : makers) near_energy += hypotf(own[m]->x - x->x, own[m]->z - x->z) < 200.0f;
            double key = (double)near_energy + draws.uniform() * 0.5;
            if (best < 0 || key < bk) best = (int)e, bk = key;
        }
        spot = {(double)own[ext[best]]->x, (double)own[ext[best]]->z};
    }
    Job job = build(uid, small, spot, pass.taken, (double)d.pylon[small] * 0.6, nullptr, -1, -1, {}, false);
    if (job.has) {
        pass.made += (double)d.energy[small];
        pass.small++;
    }
    return job;
}

int Seat::told_to_build(size_t j) const {
    const Rec* u = own[j];
    return u->ord_kind == K_BUILD ? builds[u->uid] : -1;
}

Job Seat::production_job(const Ctx& c, int i, Pass& pass, double income, int by_class, P2 by_at) {
    const Data& d = *data;
    const double seconds = c.seconds;
    if (!floating || pass.factory || seconds - pending[X_FACTORY] < PENDING_FOR) return {};
    int first = -1;
    for (size_t j = 0; j < own.size() && first < 0; j++)
        if (d.plate_of[own_rows[j]] >= 0 && built(own[j])) first = (int)j;
    if (first < 0) return {};
    // Its plate and the caretakers: finished, started, or that a constructor was sent to start (one building counts once).
    const int plate_row = d.plate_of[own_rows[first]];
    int plate = -1, done = 0, frames = 0, sent = 0, sent_plate = 0;
    for (size_t j = 0; j < own.size(); j++) {
        int r = own_rows[j], t = told_to_build(j);
        sent += t >= 0 && (t == plate_row || t == d.caretaker);
        sent_plate += t >= 0 && t == plate_row;
        if (r != plate_row && (d.caretaker < 0 || r != d.caretaker)) continue;
        if (built(own[j])) done++;
        else frames++;
        if (r == plate_row && (plate < 0 || built(own[j]))) plate = (int)j;
    }
    if (1 + done + std::max(frames, sent) >= std::min<int64_t>(MOST_PRODUCTION, 1 + (int64_t)(income / PRODUCTION_PER_INCOME))) return {};
    P2 at{own[first]->x, own[first]->z};
    int row;
    P2 near;
    double spread;
    if (plate < 0 && !sent_plate) {
        row = plate_row, near = at, spread = PLATE_SPREAD;
    } else if (plate >= 0 && built(own[plate]) && d.caretaker >= 0) {
        row = d.caretaker, near = {(at.x + own[plate]->x) * 0.5, (at.z + own[plate]->z) * 0.5}, spread = CARETAKER_SPREAD;
    } else {
        return {};      // its plate is on its way: caretakers go beside both
    }
    if (!d.can_build(own_rows[i], row)) return {};
    Job job = build(own[i]->uid, row, near, pass.taken, spread, nullptr, -1, by_class, by_at, true);
    if (!job.has) return {};
    const double from_factory = py_hypot((double)job.o.x - at.x, (double)job.o.z - at.z);
    if (row == plate_row ? from_factory > PLATE_NEAR
                         : from_factory > CARETAKER_NEAR ||
                               py_hypot((double)job.o.x - own[plate]->x, (double)job.o.z - own[plate]->z) > CARETAKER_NEAR)
        return {};
    pass.factory = true;
    pending[X_FACTORY] = seconds;
    log.push_back({seconds, 0, row});
    return job;
}

Job Seat::second_factory_job(const Ctx& c, int i, const GridOf& grid, Pass& pass, double income, int by_class, P2 by_at) {
    const Data& d = *data;
    const double seconds = c.seconds;
    if (pass.second || second == -1 || seconds - pending[X_SECOND] < PENDING_FOR) return {};
    if (seconds < SECOND_FROM_S && income - c.reclaim < SECOND_FROM_M) return {};
    // One we have, or one a constructor was sent to start: the walk there can take longer than PENDING_FOR.
    for (size_t j = 0; j < own.size(); j++)
        for (int k = 0; k < 3; k++)
            if (own_rows[j] == d.second_factories[k] || told_to_build(j) == d.second_factories[k]) return {};
    if (second == -2) {
        double total = 0.0;
        for (int k = 0; k < 3; k++) total += d.second_factories[k] >= 0 ? SECOND_WEIGHTS[k] : 0.0;
        second = -1;
        double pick = draws.uniform() * total;
        for (int k = 0; k < 3 && total > 0.0; k++) {
            if (d.second_factories[k] < 0) continue;
            second = d.second_factories[k];
            if ((pick -= SECOND_WEIGHTS[k]) < 0.0) break;
        }
        if (second < 0) return {};
    }
    int row = second;
    bool linked = d.needs_link[row] > 0.0f;
    if (linked && grid.main_energy < (double)d.needs_link[row] && d.second_factories[0] >= 0) row = second = d.second_factories[0], linked = false;
    if (!d.can_build(own_rows[i], row)) return {};
    P2 near = along(base, enemy_base, -BEHIND_BASE);
    if (linked) near = {grid.x[grid.nearest(base.x, base.z, grid.main)], grid.z[grid.nearest(base.x, base.z, grid.main)]};
    Job job = build(own[i]->uid, row, near, pass.taken, linked ? SITE_SPREAD : 300.0, linked ? &grid : nullptr, linked ? grid.main : -1,
                    by_class, by_at, true);
    if (job.has) {
        pass.second = true;
        pending[X_SECOND] = seconds;
        log.push_back({seconds, 0, row});
    }
    return job;
}

Job Seat::radar_job(const Ctx& c, int i, Pass& pass, int by_class, P2 by_at) {
    const Data& d = *data;
    const double seconds = c.seconds;
    int radar = d.special[X_RADAR];
    if (seconds < RADAR_FROM || radar < 0 || !d.can_build(own_rows[i], radar) || pass.radar) return {};
    int want = 1;
    if (d.special[X_SILO] >= 0)
        for (int r : own_rows)
            if (r == d.special[X_SILO]) {
                want = 2;
                break;
            }
    if (have(X_RADAR, seconds) >= want) return {};
    Job job = build(own[i]->uid, radar, along(front(), base, FRONT_BACK), pass.taken, SITE_SPREAD, nullptr, -1, by_class, by_at, true);
    if (job.has) {
        pass.radar = true;
        pending[X_RADAR] = seconds;
    }
    return job;
}

Job Seat::link_job(const Ctx& c, int i, const GridOf& grid, Pass& pass) {
    (void)c;
    const Data& d = *data;
    int pylon = d.special[X_PYLON];
    if (pylon < 0 || !d.can_build(own_rows[i], pylon) || pass.pylons) return {};
    int main = grid.main;
    if (main < 0) return {};
    int k = -1;
    double bd = 0.0;
    for (size_t a = 0; a < grid.x.size(); a++) {
        int r = grid.row[a];
        bool wants = d.has(r, ZKS_EXTRACTOR) || d.needs_link[r] > 0.0f;
        if (!wants || grid.label[a] == main) continue;
        double dd = std::hypot(grid.x[a] - base.x, grid.z[a] - base.z);
        if (k < 0 || dd < bd) k = (int)a, bd = dd;
    }
    if (k < 0) return {};
    int m = grid.nearest(grid.x[k], grid.z[k], main);
    double reach = (double)d.pylon[pylon];
    double gap = py_hypot(grid.x[m] - grid.x[k], grid.z[m] - grid.z[k]);
    double a = gap * 0.5, b = LINK_SLACK * (grid.reach[k] + reach);
    double step = b < a ? b : a;
    P2 point = along({grid.x[k], grid.z[k]}, {grid.x[m], grid.z[m]}, step);
    Job job = build(own[i]->uid, pylon, point, pass.taken, 40.0, nullptr, -1, -1, {}, false);
    if (job.has) pass.pylons++;
    return job;
}

Job Seat::spot_job(const Ctx& c, int i, Pass& pass) {
    (void)c;
    const Data& d = *data;
    const Ground& g = *ground;
    int row = own_rows[i];
    int ext_row = d.first_extractor_option(row);
    if (ext_row < 0 || !g.n_spots) return {};
    double x = own[i]->x, z = own[i]->z;
    std::vector<int> spots;
    float bx = (float)base.x, bz = (float)base.z, ebx = (float)enemy_base.x, ebz = (float)enemy_base.z;
    for (int s = 0; s < g.n_spots; s++) {
        if (d.has(row, ZKS_COMMANDER) && !(hypotf(g.spots[3 * s] - bx, g.spots[3 * s + 1] - bz) <= COMMANDER_REACH)) continue;
        spots.push_back(s);
    }
    if (spots.empty()) return {};
    // holders: float32 while nothing's been claimed this pass, float64 after (np.concatenate with a list of Python floats)
    std::vector<float> hf;
    for (size_t j = 0; j < own.size(); j++)
        if (d.has(own_rows[j], ZKS_EXTRACTOR)) hf.push_back(own[j]->x), hf.push_back(own[j]->z);
    for (size_t e = 0; e < enemies.size(); e++)
        if (d.has(enemy_rows[e], ZKS_EXTRACTOR)) hf.push_back(enemies[e]->x), hf.push_back(enemies[e]->z);
    bool wide = !pass.claimed.empty();
    std::vector<int> cand;
    for (int s : spots) {
        float sx = g.spots[3 * s], sz = g.spots[3 * s + 1];
        bool free_ = true;
        if (wide) {
            double best = INFINITY;
            for (size_t h = 0; h < hf.size() / 2; h++) best = std::min(best, std::hypot((double)sx - hf[2 * h], (double)sz - hf[2 * h + 1]));
            for (size_t h = 0; h < pass.claimed.size() / 2; h++)
                best = std::min(best, std::hypot((double)sx - pass.claimed[2 * h], (double)sz - pass.claimed[2 * h + 1]));
            free_ = best > 64.0;
        } else if (!hf.empty()) {
            float best = INFINITY;
            for (size_t h = 0; h < hf.size() / 2; h++) best = std::min(best, hypotf(sx - hf[2 * h], sz - hf[2 * h + 1]));
            free_ = best > 64.0f;
        }
        bool mine = hypotf(sx - bx, sz - bz) <= (float)OUR_HALF * hypotf(sx - ebx, sz - ebz);
        if (free_ && mine) cand.push_back(s);
    }
    if (cand.empty()) return {};
    float fx = (float)x, fz = (float)z;
    std::vector<float> dist(cand.size());
    for (size_t k = 0; k < cand.size(); k++) dist[k] = hypotf(g.spots[3 * cand[k]] - fx, g.spots[3 * cand[k] + 1] - fz);
    std::vector<int> idx(cand.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return dist[a] < dist[b]; });
    int c_ = d.move_class[row];
    for (size_t k = 0; k < idx.size() && k < 8; k++) {
        int s = cand[idx[k]];
        double sx = g.spots[3 * s], sz = g.spots[3 * s + 1];
        if (avoided(sx, sz)) continue;
        float fsx = (float)sx, fsz = (float)sz;
        bool blocked = false;
        for (size_t e = 0; e < enemies.size() && !blocked; e++) {
            int r = enemy_rows[e];
            bool defence = d.has(r, ZKS_STATIC_ARMED) && built(enemies[e]);
            bool army = d.has(r, ZKS_ARMED) && d.has(r, ZKS_MOBILE);
            if (!defence && !army) continue;
            float dd = hypotf(enemies[e]->x - fsx, enemies[e]->z - fsz);
            if ((defence && dd <= d.range[r] + SAFE_MARGIN) || (army && dd <= SAFE_FROM_ARMY)) blocked = true;
        }
        if (blocked) continue;
        if (c_ != -1 && !g.joined(c_, {x, z}, {sx, sz})) continue;
        pass.claimed.push_back(sx);
        pass.claimed.push_back(sz);
        return make(own[i]->uid, K_BUILD, sx, sz, d.catalog_row[ext_row]);
    }
    return {};
}

// An enemy defence in range of p (and a margin), or an enemy army within army_reach of it.
bool Seat::unsafe(P2 p, float army_reach) const {
    const Data& d = *data;
    float px = (float)p.x, pz = (float)p.z;
    for (size_t e = 0; e < enemies.size(); e++) {
        int r = enemy_rows[e];
        bool defence = d.has(r, ZKS_STATIC_ARMED) && built(enemies[e]);
        bool army = d.has(r, ZKS_ARMED) && d.has(r, ZKS_MOBILE);
        if (!defence && !army) continue;
        float dd = hypotf(enemies[e]->x - px, enemies[e]->z - pz);
        if ((defence && dd <= d.range[r] + SAFE_MARGIN) || (army && dd <= army_reach)) return true;
    }
    return false;
}

// The wreck most worth this constructor's walk, within reach of it: safe, on ground it can walk to, nearer our base than the
// enemy's (RECLAIM_HALF), and nobody of ours on it already.
Job Seat::reclaim_job(const Ctx& c, int i, Pass& pass, double reach) {
    const Data& d = *data;
    const Rec* u = own[i];
    int row = own_rows[i];
    if (!c.feature_n || u->hp_frac < RECLAIM_FIT || d.has(row, ZKS_COMMANDER)) return {};
    int cls = d.move_class[row];
    P2 at{(double)u->x, (double)u->z};
    int best = -1;
    double best_worth = 0.0;
    for (int k = 0; k < c.feature_n; k++) {
        const zk::FeatureRecord& f = c.features[k];
        if (f.metal < RECLAIM_MIN) continue;
        if (std::find(pass.reclaimed.begin(), pass.reclaimed.end(), f.feat_id) != pass.reclaimed.end()) continue;
        P2 p{(double)f.x, (double)f.z};
        double away = py_hypot(p.x - at.x, p.z - at.z);
        if (away > reach) continue;
        if (py_hypot(p.x - base.x, p.z - base.z) > RECLAIM_HALF * py_hypot(p.x - enemy_base.x, p.z - enemy_base.z)) continue;
        double worth = (double)f.metal / (away + RECLAIM_WALK);
        if (best >= 0 && worth <= best_worth) continue;
        if (avoided(p.x, p.z) || unsafe(p, SAFE_FROM_ARMY) || (cls != -1 && !ground->joined(cls, at, p))) continue;
        best = k, best_worth = worth;
    }
    if (best < 0) return {};
    pass.reclaimed.push_back(c.features[best].feat_id);
    return make(u->uid, K_RECLAIM, c.features[best].x, c.features[best].z, c.features[best].feat_id);
}

Job Seat::job(const Ctx& c, int i, const GridOf& grid, bool has_project, int prow, P2 pat, Pass& pass, double income) {
    const Data& d = *data;
    const double seconds = c.seconds;
    const Rec* u = own[i];
    int64_t uid = u->uid;
    double x = u->x, z = u->z;
    int row = own_rows[i];
    int by_class = d.move_class[row];
    P2 by_at{x, z};
    std::vector<int> being;
    for (size_t j = 0; j < own.size(); j++)
        if (!built(own[j]) && !d.has(own_rows[j], ZKS_MOBILE)) being.push_back((int)j);
    if (has_project) {
        for (int j : being)
            if (own_rows[j] == prow) return make(uid, K_ASSIST, 0.0, 0.0, own[j]->uid);
        if (d.can_build(row, prow)) {
            bool needs_grid = d.needs_link[prow] > 0.0f;
            Job job = build(uid, prow, pat, pass.taken, SITE_SPREAD, needs_grid ? &grid : nullptr, needs_grid ? grid.main : -1, by_class, by_at, true);
            if (!job.has && needs_grid) job = build(uid, prow, pat, pass.taken, SITE_SPREAD, nullptr, -1, by_class, by_at, true);
            int x_ = d.special_of[prow];
            if (job.has && seconds - pending[x_] >= PENDING_FOR) {
                pending[x_] = seconds;
                log.push_back({seconds, 0, prow});
                return job;
            }
        }
    }
    if (pass.reclaimers < pass.reclaimers_wanted) {
        Job j = reclaim_job(c, i, pass, INFINITY);
        if (j.has) {
            pass.reclaimers++;
            return j;
        }
    }
    {
        int best = -1;
        double bd = 0.0;
        float fx = u->x, fz = u->z;
        for (int j : being) {
            double dd = py_hypot((double)(own[j]->x - fx), (double)(own[j]->z - fz));
            if (dd <= FINISH_REACH && (best < 0 || dd < bd)) best = j, bd = dd;
        }
        if (best >= 0) return make(uid, K_ASSIST, 0.0, 0.0, own[best]->uid);
    }
    {
        Job j = second_factory_job(c, i, grid, pass, income, by_class, by_at);
        if (j.has) return j;
    }
    if (pass.made < ENERGY_FIRST * income) {
        Job j = energy_job(c, i, grid, pass, income, being, false);
        if (j.has) return j;
    }
    if (!floating || pass.expanding < FLOAT_EXPANDERS) {
        Job j = spot_job(c, i, pass);
        if (j.has) {
            pass.expanding++;
            return j;
        }
    }
    if (floating && energy_low) {
        Job j = energy_job(c, i, grid, pass, income, being, true);
        if (j.has) return j;
    }
    float fx = u->x, fz = u->z;
    if (floating) {
        int best = -1, open = -1;
        float bd = 0.0f, od = 0.0f;
        for (size_t k = 0; k < own.size(); k++) {
            if (!d.has(own_rows[k], ZKS_FACTORY)) continue;
            float dd = hypotf(own[k]->x - fx, own[k]->z - fz);
            if (best < 0 || dd < bd) best = (int)k, bd = dd;
            if (built(own[k]) && pass.helpers(own[k]->uid) < HELPERS_PER_LINE && (open < 0 || dd < od)) open = (int)k, od = dd;
        }
        if (open >= 0 && od <= FLOAT_ASSIST_REACH) return make(uid, K_ASSIST, 0.0, 0.0, own[open]->uid);
        Job j = production_job(c, i, pass, income, by_class, by_at);
        if (j.has) return j;
        if (best >= 0 && bd <= FLOAT_ASSIST_REACH) return make(uid, K_ASSIST, 0.0, 0.0, own[best]->uid);
    }
    {
        Job j = energy_job(c, i, grid, pass, income, being, false);
        if (j.has) return j;
    }
    {
        Job j = radar_job(c, i, pass, by_class, by_at);
        if (j.has) return j;
    }
    {
        Job j = link_job(c, i, grid, pass);
        if (j.has) return j;
    }
    {
        Job j = production_job(c, i, pass, income, by_class, by_at);
        if (j.has) return j;
    }
    {
        Job j = reclaim_job(c, i, pass, RECLAIM_NEAR);
        if (j.has) return j;
    }
    int best = -1;
    float bd = 0.0f;
    for (size_t k = 0; k < own.size(); k++) {
        if (!d.has(own_rows[k], ZKS_FACTORY) || !built(own[k])) continue;
        float dd = hypotf(own[k]->x - fx, own[k]->z - fz);
        if (best < 0 || dd < bd) best = (int)k, bd = dd;
    }
    if (best >= 0) return make(uid, K_ASSIST, 0.0, 0.0, own[best]->uid);
    return {};
}

void Seat::eco(const Ctx& c) {
    const Data& d = *data;
    const double seconds = c.seconds;
    size_t n = own.size();
    // _count
    {
        int ours_ = 0, theirs = 0;
        for (int r : own_rows) ours_ += d.has(r, ZKS_EXTRACTOR);
        for (int r : enemy_rows) theirs += d.has(r, ZKS_EXTRACTOR);
        counts.push_back({seconds, ours_, theirs});
        while (!counts.empty() && counts.front().t < seconds - STALE_FOR - 10.0) counts.pop_front();
    }
    // _note
    for (size_t i = 0; i < n; i++) {
        if (!built(own[i])) continue;
        int r = own_rows[i];
        uint8_t bits = 0;
        for (int x = X_CERBERUS; x <= X_FUSION; x++)
            if (d.special[x] >= 0 && r == d.special[x]) bits |= 1 << x;
        if (d.has(r, ZKS_SINGULARITY)) bits |= 1 << 5;
        if (d.has(r, ZKS_MISSILE)) bits |= 1 << 6;
        uint8_t fresh = bits & ~noted[own[i]->uid];
        noted[own[i]->uid] |= bits;
        for (int k = 0; k < 7; k++) finished[k] += (fresh >> k) & 1;
    }
    std::vector<uint8_t> builders(n);
    for (size_t i = 0; i < n; i++) {
        int r = own_rows[i];
        builders[i] = built(own[i]) && (d.has(r, ZKS_BUILDER) || d.has(r, ZKS_COMMANDER)) && d.has(r, ZKS_MOBILE);
    }
    // Reclaimers go home if an enemy army comes near or they're hurt.
    std::vector<uint8_t> fled(n, 0);
    {
        P2 home = along(base, enemy_base, -BEHIND_BASE * 0.5);
        for (size_t i = 0; i < n; i++) {
            const Rec* u = own[i];
            if (!builders[i] || u->ord_kind != K_RECLAIM) continue;
            if (u->hp_frac < RECLAIM_HURT || unsafe({(double)u->x, (double)u->z}, RECLAIM_FLEE)) {
                eco_out.push_back(order(u->uid, K_MOVE, home.x, home.z, -1));
                reclaims[u->uid] = INT64_MIN;
                fled[i] = 1;
            }
        }
    }
    std::vector<uint8_t> stuck_ = stuck(c, builders);
    std::vector<int> idle;
    std::vector<uint8_t> is_idle(n, 0);
    for (size_t i = 0; i < n; i++) {
        const Rec* u = own[i];
        if (builders[i] && ((u->flags & zk::UF_IDLE) || u->ord_kind == K_ASSIST || stuck_[i]) && u->uid != c.busy)
            idle.push_back((int)i), is_idle[i] = 1;
    }
    double metal = c.glob[zk::G_METAL], storage = c.glob[zk::G_METAL_STORAGE];
    double energy = c.glob[zk::G_ENERGY], energy_storage = c.glob[zk::G_ENERGY_STORAGE];
    double income = c.glob[zk::G_METAL_INCOME];
    floating = metal >= FLOATING * storage || (floating && metal > FLOATING_UNTIL * storage);
    energy_low = energy < ENERGY_LOW * energy_storage;
    if (idle.empty()) return;
    // Grid.of
    GridOf grid;
    {
        for (size_t i = 0; i < n; i++) {
            int r = own_rows[i];
            if (built(own[i]) && d.pylon[r] > 0.0f) {
                grid.x.push_back(own[i]->x);
                grid.z.push_back(own[i]->z);
                grid.reach.push_back(d.pylon[r]);
                grid.row.push_back(r);
            }
        }
        int m = (int)grid.x.size();
        std::vector<int> parent(m);
        std::iota(parent.begin(), parent.end(), 0);
        auto find = [&](int a) {
            while (parent[a] != a) {
                parent[a] = parent[parent[a]];
                a = parent[a];
            }
            return a;
        };
        for (int a = 0; a < m; a++)
            for (int b = a + 1; b < m; b++)
                if (std::hypot(grid.x[a] - grid.x[b], grid.z[a] - grid.z[b]) < grid.reach[a] + grid.reach[b]) {
                    int rb = find(b);
                    parent[find(a)] = rb;
                }
        grid.label.resize(m);
        for (int a = 0; a < m; a++) grid.label[a] = find(a);
        std::vector<double> makes(m, 0.0);
        std::vector<uint8_t> has(m, 0);
        for (int a = 0; a < m; a++) makes[grid.label[a]] += (double)d.energy[grid.row[a]], has[grid.label[a]] = 1;
        for (int g = 0; g < m; g++)
            if (has[g] && (grid.main < 0 || makes[g] > makes[grid.main])) grid.main = g;
        if (grid.main >= 0) grid.main_energy = makes[grid.main];
    }
    Pass pass;
    {
        std::vector<int> b;
        for (size_t i = 0; i < n; i++)
            if (built(own[i])) b.push_back(own_rows[i]);
        pass.made = INNATE_ENERGY + (double)np_sum<float>([&](int64_t k) { return d.energy[b[k]]; }, (int64_t)b.size());
    }
    // Who's reclaiming already, what, and how many should be.
    {
        int constructors = 0;
        for (size_t i = 0; i < n; i++) {
            if (!builders[i] || d.has(own_rows[i], ZKS_COMMANDER)) continue;
            constructors++;
            if (own[i]->ord_kind != K_RECLAIM || fled[i]) continue;
            pass.reclaimers++;
            if (reclaims[own[i]->uid] != INT64_MIN) pass.reclaimed.push_back(reclaims[own[i]->uid]);
        }
        pass.reclaimers_wanted = std::min(RECLAIMERS, constructors / RECLAIMER_PER);
    }
    // _taken
    for (size_t i = 0; i < n; i++) {
        int r = own_rows[i];
        if (d.has(r, ZKS_MOBILE)) continue;
        pass.taken.push_back(own[i]->x), pass.taken.push_back(own[i]->z), pass.taken.push_back(d.radius[r]);
    }
    const Ground& g = *ground;
    for (int s = 0; s < g.n_spots; s++)
        pass.taken.push_back(g.spots[3 * s]), pass.taken.push_back(g.spots[3 * s + 1]), pass.taken.push_back(d.extractor_radius);
    // _expanding
    if (g.n_spots) {
        for (size_t i = 0; i < n; i++) {
            if (!builders[i] || own[i]->ord_kind != K_BUILD || is_idle[i]) continue;
            float best = INFINITY;
            for (int s = 0; s < g.n_spots; s++)
                best = std::min(best, hypotf(own[i]->ord_x - g.spots[3 * s], own[i]->ord_z - g.spots[3 * s + 1]));
            pass.expanding += best < 64.0f;
        }
    }
    int prow = -1;
    P2 pat{};
    bool has_project = project(c, income, grid, prow, pat);
    for (int i : idle) {
        Job j = job(c, i, grid, has_project, prow, pat, pass, income);
        int64_t uid = own[i]->uid;
        if (j.has && j.o.kind == K_ASSIST) pass.helping.push_back(j.o.target);
        if (!j.has || (j.o.kind == K_ASSIST && own[i]->ord_kind == K_ASSIST && helps[uid] == j.o.target)) continue;
        helps[uid] = j.o.kind == K_ASSIST ? j.o.target : -1;
        reclaims[uid] = j.o.kind == K_RECLAIM ? j.o.target : INT64_MIN;
        eco_out.push_back(j.o);
        builds[uid] = j.o.kind == K_BUILD ? d.row_of_catalog[j.o.target] : -1;
        if (j.o.kind == K_BUILD) {
            int r = d.row_of_catalog[j.o.target];
            pass.taken.push_back(j.o.x), pass.taken.push_back(j.o.z), pass.taken.push_back(d.radius[r]);
        }
    }
}

}  // namespace zks
