// featurize.featurize and featurize_kernels, bit for bit: a raw frame, a map's bundle and the type table to every field of
// fz.Tensors.  Each function says which Python it is; the comments there say why it does what it does, and aren't repeated.
// float32 stays float32 as numpy and numba compute it, a sum in float64 goes in the order theirs went, and log1p, hypot and
// atan2 are glibc's, as theirs are (checked against numpy and numba on this machine: sim/policy/zkpolicy.h).
#include <algorithm>
#include <cmath>
#include <cstring>
#if defined(__SSE2__)
#include <emmintrin.h>
#endif

#include "npy.hpp"
#include "policy.hpp"

namespace zkp {

namespace {

constexpr float F_PI8 = float(M_PI / 8.0);
constexpr float F_PI4 = float(M_PI / 4.0);
constexpr float DPS_RADIUS2 = 360000.0f;       // RAY_DPS_RADIUS squared
constexpr float STILL = 0.5f;
constexpr float AT_ITS_TRANSPORT = 32.0f;
constexpr float LEAST_COST = 100.0f;
constexpr float DEAD_ZONE = 40.0f;
constexpr uint16_t INSTANCE_BITS[7] = {zk::UF_BEING_BUILT, zk::UF_CLOAKED, zk::UF_PARALYZED, zk::UF_IN_LOS, zk::UF_IDLE, zk::UF_TYPE_KNOWN,
                                       zk::UF_DAMAGED};
constexpr int8_t KIND_OF_KNOWLEDGE[4] = {TK_SEEN, TK_BLIP, TK_MEMORY, TK_HIDDEN};

// featurize_kernels' _RAY_DX, _RAY_DZ and _RAY_T: numpy's float64 sin and cos are glibc's, rounded to float32.
struct Rays {
    float dx[8], dz[8], t[RAY_CELLS];
    Rays() {
        for (int d = 0; d < 8; ++d) {
            double angle = double(d) * (M_PI / 4.0);
            dx[d] = float(std::sin(angle));
            dz[d] = float(std::cos(angle));
        }
        for (int k = 0; k < RAY_CELLS; ++k) t[k] = float(double(k + 1) * 32.0);
    }
};
const Rays RAYS;

// Bundle._cell and featurize_kernels._cell for one point.
inline void cell_of(float x, float z, float cell, int h, int w, int64_t& i, int64_t& j) {
    const float qi = z / cell, qj = x / cell;
    i = std::min<int64_t>(std::max<int64_t>(to_i32(qi), 0), h - 1);
    j = std::min<int64_t>(std::max<int64_t>(to_i32(qj), 0), w - 1);
}

// np.int64(np.int32(np.floor(f))) without a call to floorf, which baseline x86-64 has no instruction for.
inline int64_t floor_i32(float f) {
    if (!(f > -2147483648.0f && f < 2147483648.0f)) return std::numeric_limits<int32_t>::min();
    int32_t i = int32_t(f);
    if (float(i) > f) --i;      // a negative non-integer, so |f| < 2^23 and this can't wrap
    return i;
}

// The angles where a DPS sector changes, as featurize_kernels has them: floor((theta + pi/8) / (pi/4)) % 8, theta = atan2(dx, dz),
// so that sector k is the 45 degrees around k pi/4 from +z towards x, as ray k is.  With x on our side (theta > 0) sectors 0 to 4
// change at pi/8, 3pi/8, 5pi/8 and 7pi/8, and with it on the other, 0, 7, 6, 5 and 4 at -pi/8, -3pi/8, -5pi/8 and -7pi/8.  (Until
// FEATURE_VERSION 10 it was truncated: sector 0 was 90 degrees wide, 4 and 5 22.5.)
struct Boundaries {
    double c[4], s[4];      // of (2k + 1) pi / 8; the other side's are -(2k + 1) pi / 8, whose sine is -s[k]
    Boundaries() {
        for (int k = 0; k < 4; ++k) {
            c[k] = std::cos((2 * k + 1) * M_PI / 8.0);
            s[k] = std::sin((2 * k + 1) * M_PI / 8.0);
        }
    }
};
const Boundaries BOUNDS;

// A pair's DPS sector, as featurize_kernels.unit_block has it.  atan2f was most of a big frame's time, so which side of each
// boundary the direction is on is said by the sign of a cross product, r sin(theta - boundary), wherever that's more than 1e-4
// rad from it (atan2f is within 3e-7 rad, the float32 sums after it within 1e-6), and by atan2f near one: the same answer.
inline int64_t sector_of(float dx, float dz) {
    if (dx != 0.0f && dz != 0.0f && std::isfinite(dx) && std::isfinite(dz)) {
        const double x = dx, z = dz, tol = 1e-4 * (std::fabs(x) + std::fabs(z));
        bool sure = true;
        int past = 0;       // boundaries passed going from +z towards the direction
        for (int k = 0; k < 4; ++k) {
            const double cross = x > 0.0 ? x * BOUNDS.c[k] - z * BOUNDS.s[k] : x * BOUNDS.c[k] + z * BOUNDS.s[k];
            sure = sure && std::fabs(cross) > tol;
            past += x > 0.0 ? cross > 0.0 : cross < 0.0;
        }
        if (sure) return x > 0.0 ? past : (8 - past) % 8;
    }
    return py_mod(floor_i32((::atan2f(dx, dz) + F_PI8) / F_PI4), 8);
}

// The b, in order, with dx * dx + dz * dz <= DPS_RADIUS2 in float32 (dx = x[b] - xa): four at a time with SSE2, whose operations
// are the same IEEE ones, so as to scan a big frame's million pairs without a branch each.
inline int64_t within_radius(const float* xs, const float* zs, int64_t n, float xa, float za, int64_t* out) {
    int64_t m = 0, b = 0;
#if defined(__SSE2__)
    const __m128 ax = _mm_set1_ps(xa), az = _mm_set1_ps(za), r2 = _mm_set1_ps(DPS_RADIUS2);
    for (; b + 4 <= n; b += 4) {
        const __m128 dx = _mm_sub_ps(_mm_loadu_ps(xs + b), ax), dz = _mm_sub_ps(_mm_loadu_ps(zs + b), az);
        int mask = _mm_movemask_ps(_mm_cmple_ps(_mm_add_ps(_mm_mul_ps(dx, dx), _mm_mul_ps(dz, dz)), r2));
        while (mask) {
            out[m++] = b + __builtin_ctz(unsigned(mask));
            mask &= mask - 1;
        }
    }
#endif
    for (; b < n; ++b) {
        const float dx = xs[b] - xa, dz = zs[b] - za;
        if (dx * dx + dz * dz <= DPS_RADIUS2) out[m++] = b;
    }
    return m;
}

// hypotf(dx, dz) <= thr, without the hypotf where the squares say it's far beyond: hypotf is within an ulp, so past thr by a
// part in 1e5 can't round to within it.
inline bool within(float dx, float dz, float thr) {
    const double d2 = double(dx) * double(dx) + double(dz) * double(dz);
    const double t = thr;
    if (t >= 0.0 && d2 > t * t * (1.0 + 1e-5)) return false;
    return ::hypotf(dx, dz) <= thr;
}

struct Arena {
    uint8_t* base;
    int64_t capacity;
    int64_t* layout;
    int64_t used = 0;
    bool ok = true;

    template <class T>
    T* take(Field field, int64_t d0, int64_t d1 = -1, int64_t d2 = -1) {
        int64_t count = d0 * (d1 < 0 ? 1 : d1) * (d2 < 0 ? 1 : d2);
        int64_t at = (used + 63) / 64 * 64;
        int64_t bytes = count * int64_t(sizeof(T));
        int64_t* l = layout + field * LAYOUT;
        l[0] = at;
        l[1] = d0;
        l[2] = d1 < 0 ? 0 : d1;
        l[3] = d2 < 0 ? 0 : d2;
        if (!ok || at + bytes > capacity) {
            ok = false;
            static thread_local std::vector<uint8_t> sink;      // somewhere to write, that nobody reads
            sink.assign(size_t(std::max<int64_t>(bytes, 1)), 0);
            return reinterpret_cast<T*>(sink.data());
        }
        used = at + bytes;
        if (bytes) std::memset(base + at, 0, size_t(bytes));
        return reinterpret_cast<T*>(base + at);
    }
};

// What a frame's work keeps between its steps; one a thread, reused frame after frame.
struct Scratch {
    std::vector<int64_t> rows, goes, own, seen, cand, rest, order, picked, left, rank, uid, aim, aim_order, keys, members, targets;
    std::vector<int64_t> beacons, own_tokens, site_order, feat_order, hidden, close;
    std::vector<float> x, z, gather, near, neg_cost, neg_score, prod, dist;
    std::vector<int64_t> sorted_uid;
    std::vector<UnitRecord> units;
    std::vector<int64_t> kept_rows;
    std::vector<int32_t> region, rel_ally, rel_player;
    std::vector<uint8_t> locked, is_target, is_foe, is_beacon, is_own, taken, foes, with_beacon;
    std::vector<double> enemy_sum, own_sum, metal, sums;
    std::vector<float> summary_rows;
    std::vector<int64_t> summary_counts;
    std::vector<int32_t> summary_ally, summary_player;
};

thread_local Scratch scratch;

// featurize_kernels.select.  Returns the controllable count; s.rows, s.picked and s.left, army and foe, and whether any enemy is
// seen.
int64_t select(const Table& T, const UnitRecord* units, int64_t n, int64_t player, int64_t ally, Scratch& s, float army[2],
               float foe[2], bool& any_seen) {
    s.rows.resize(n);
    s.x.resize(n);
    s.z.resize(n);
    s.goes.clear();
    s.own.clear();
    s.seen.clear();
    s.cand.clear();
    for (int64_t i = 0; i < n; ++i) {
        const UnitRecord& u = units[i];
        const int64_t r = T.row(u.def_id);
        s.rows[i] = r;
        s.x[i] = u.x;
        s.z[i] = u.z;
        const bool mine = u.player == player;
        const bool fights = T.bv(r, B_COMBAT) && T.bv(r, B_MOBILE);
        const bool can = mine && (u.flags & zk::UF_BEING_BUILT) == 0 &&
                         (fights || T.bv(r, B_ACTS_IN_PLACE) || T.bv(r, B_CONSTRUCTOR) || T.bv(r, B_TRANSPORT)) && (u.flags & zk::UF_RESERVED) == 0;
        if (can) s.cand.push_back(i);
        if (can && T.bv(r, B_MOBILE)) s.goes.push_back(i);
        if (mine) s.own.push_back(i);
        if (u.ally_team != ally && (u.knowledge == zk::KNOW_LOS || u.knowledge == zk::KNOW_RADAR)) s.seen.push_back(i);
    }
    auto mean_of = [&](const std::vector<int64_t>& idx, const std::vector<float>& v) {
        s.gather.resize(idx.size());
        for (size_t k = 0; k < idx.size(); ++k) s.gather[k] = v[idx[k]];
        return np_mean(s.gather.data(), int64_t(idx.size()));
    };
    army[0] = army[1] = 0.0f;
    if (!s.goes.empty()) {
        army[0] = mean_of(s.goes, s.x);
        army[1] = mean_of(s.goes, s.z);
    } else if (!s.own.empty()) {
        army[0] = mean_of(s.own, s.x);
        army[1] = mean_of(s.own, s.z);
    }
    foe[0] = foe[1] = 0.0f;
    any_seen = !s.seen.empty();
    if (any_seen) {
        foe[0] = mean_of(s.seen, s.x);
        foe[1] = mean_of(s.seen, s.z);
    }

    // Most expensive first; among equals, nearest the army; uid.
    const int64_t nc = int64_t(s.cand.size());
    s.near.resize(nc);
    s.neg_cost.resize(nc);
    s.uid.resize(nc);
    for (int64_t k = 0; k < nc; ++k) {
        const int64_t i = s.cand[k];
        const float dx = s.x[i] - army[0], dz = s.z[i] - army[1];
        s.near[k] = ::hypotf(dx, dz);
        s.neg_cost[k] = -T.fv(s.rows[i], T_COST);
        s.uid[k] = units[i].uid;
    }
    lexorder(s.order, nc, s.neg_cost.data(), s.near.data(), s.uid.data());
    const int64_t c = std::min<int64_t>(nc, CAP_CONTROLLABLE);
    s.picked.resize(c);
    s.taken.assign(n, 0);
    for (int64_t k = 0; k < c; ++k) {
        s.picked[k] = s.cand[s.order[k]];
        s.taken[s.picked[k]] = 1;
    }

    // The rest: what can be aimed at first, in sight before on radar; within each, value over distance from our army; uid.
    s.rest.clear();
    for (int64_t i = 0; i < n; ++i)
        if (units[i].knowledge != zk::KNOW_HIDDEN && !s.taken[i]) s.rest.push_back(i);
    const int64_t m = int64_t(s.rest.size());
    s.rank.resize(m);
    s.neg_score.resize(m);
    s.uid.resize(m);
    for (int64_t k = 0; k < m; ++k) {
        const int64_t i = s.rest[k];
        const UnitRecord& u = units[i];
        const float dx = s.x[i] - army[0], dz = s.z[i] - army[1];
        const float d = ::hypotf(dx, dz);
        s.neg_score[k] = -(np_max(T.fv(s.rows[i], T_COST), LEAST_COST) / (d + 200.0f));
        const bool hostile = u.ally_team != ally;
        s.rank[k] = (hostile && u.knowledge == zk::KNOW_LOS) ? 0 : ((hostile && u.knowledge == zk::KNOW_RADAR) ? 1 : 2);
        s.uid[k] = u.uid;
    }
    lexorder(s.order, m, s.rank.data(), s.neg_score.data(), s.uid.data());
    const int64_t room = CAP_UNITS - c;
    s.left.clear();
    for (int64_t k = 0; k < m; ++k) {
        const int64_t i = s.rest[s.order[k]];
        if (k < room)
            s.picked.push_back(i);
        else
            s.left.push_back(i);
    }
    return c;
}

// featurize._summaries: into s.summary_rows (S x unit_dim), with their allies and players.
void summaries(const Table& T, const Bundle& B, const UnitRecord* units, int64_t player, int64_t ally, Scratch& s) {
    const int D = T.unit_dim, I = T.type_dim, S0 = T.unit_dim - SUMMARY_COLS;
    const int64_t nl = int64_t(s.left.size());
    s.keys.resize(nl);
    s.rel_ally.resize(nl);
    for (int64_t k = 0; k < nl; ++k) {
        const UnitRecord& u = units[s.left[k]];
        const bool mine = u.player == player;
        const int64_t rel = relative_id(u.ally_team, ally, MAX_ALLY_TEAMS);
        s.rel_ally[k] = int32_t(rel);
        s.keys[k] = mine ? -1 : rel * 4 + u.knowledge;
    }
    std::vector<int64_t> distinct(s.keys.begin(), s.keys.end());
    std::sort(distinct.begin(), distinct.end());
    distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
    s.summary_rows.clear();
    s.summary_counts.clear();
    s.summary_ally.clear();
    s.summary_player.clear();
    for (int64_t key : distinct) {
        int32_t first_ally = 0;
        bool first = true;
        for (int role = 0; role < 3; ++role) {
            s.members.clear();
            for (int64_t k = 0; k < nl; ++k) {
                if (s.keys[k] != key) continue;
                if (first) {
                    first_ally = s.rel_ally[k];
                    first = false;
                }
                const int64_t r = s.rows[s.left[k]];
                const int coarse = T.bv(r, B_BUILDER) ? 1 : ((T.bv(r, B_MOBILE) && T.bv(r, B_COMBAT)) ? 0 : 2);
                if (coarse == role) s.members.push_back(s.left[k]);
            }
            const int64_t nm = int64_t(s.members.size());
            if (!nm) continue;
            const size_t at = s.summary_rows.size();
            s.summary_rows.resize(at + D, 0.0f);
            float* row = s.summary_rows.data() + at;
            s.gather.resize(nm);
            for (int64_t k = 0; k < nm; ++k) s.gather[k] = units[s.members[k]].x;
            row[I + 0] = np_mean(s.gather.data(), nm) / B.map_w * 2.0f - 1.0f;
            for (int64_t k = 0; k < nm; ++k) s.gather[k] = units[s.members[k]].z;
            row[I + 1] = np_mean(s.gather.data(), nm) / B.map_h * 2.0f - 1.0f;
            s.prod.resize(nm);
            for (int64_t k = 0; k < nm; ++k) {
                float hp = units[s.members[k]].hp_frac;         // np.maximum(np.nan_to_num(hp, nan=1.0), 0.0)
                if (hp != hp)
                    hp = 1.0f;
                else if (std::isinf(hp))
                    hp = hp > 0 ? 3.4028234663852886e38f : -3.4028234663852886e38f;
                hp = np_max(hp, 0.0f);
                s.prod[k] = hp * T.fv(s.rows[s.members[k]], T_COST);
                s.gather[k] = T.fv(s.rows[s.members[k]], T_DPS_GROUND);
            }
            row[S0 + 0] = float(std::log1p(double(nm)) / 5.0);
            row[S0 + 1] = ::log1pf(np_sum(s.prod.data(), nm)) / 10.0f;
            row[S0 + 2] = ::log1pf(np_sum(s.gather.data(), nm)) / 8.0f;
            s.summary_counts.push_back(nm);
            s.summary_ally.push_back(first_ally);
            s.summary_player.push_back(key == -1 ? 0 : MAX_PLAYERS);
        }
    }
}

// featurize_kernels._instance_row.
void instance_row(const Table& T, const Bundle& B, const UnitRecord& u, int64_t r, float* out) {
    const int at = T.type_dim;
    std::memcpy(out, T.stats.data() + r * T.type_dim, sizeof(float) * size_t(T.type_dim));
    const float x = u.x, z = u.z;
    out[at + 0] = x / B.map_w * 2.0f - 1.0f;
    out[at + 1] = z / B.map_h * 2.0f - 1.0f;
    int64_t ci, cj;
    cell_of(x, z, B.cell, B.label_h, B.label_w, ci, cj);
    const float above = u.y - B.height[ci * B.label_w + cj];
    out[at + 2] = np_clip(np_sign(above) * np_max(std::fabs(above) - DEAD_ZONE, 0.0f) / 200.0f, -1.0f, 4.0f);
    const float top = np_max(T.fv(r, T_SPEED), 30.0f) / 30.0f;
    const float vx = np_clip(u.vx / top, -2.0f, 2.0f);
    const float vz = np_clip(u.vz / top, -2.0f, 2.0f);
    out[at + 3] = vx;
    out[at + 4] = vz;
    out[at + 5] = np_min(::hypotf(vx, vz), 2.0f);
    const float hp = u.hp_frac;
    const bool missing = hp != hp;
    out[at + 6] = missing ? 0.0f : np_clip(hp, 0.0f, 1.0f);
    out[at + 7] = missing ? 1.0f : 0.0f;
    out[at + 8] = np_clip(u.build_frac, 0.0f, 1.0f);
    const uint16_t flags = u.flags;
    for (int k = 0; k < 7; ++k) out[at + 9 + k] = (flags & INSTANCE_BITS[k]) != 0 ? 1.0f : 0.0f;
    out[at + 16] = u.blip_top_speed / 150.0f;
    out[at + 17] = u.knowledge == zk::KNOW_MEMORY ? ::log1pf(np_max(u.ord_age, 0.0f) / 30.0f) / 6.0f : 0.0f;
    out[at + 18] = (flags & zk::UF_JUMP_READY) != 0 ? 1.0f : 0.0f;
    out[at + 19] = (flags & zk::UF_CARRIED) != 0 ? 1.0f : 0.0f;
    out[at + 20] = (flags & zk::UF_LOADED) != 0 ? 1.0f : 0.0f;
}

// featurize_kernels._lifts.
inline bool lifts(const Table& T, int64_t me, int64_t load) {
    const int64_t k = int64_t(T.fv(load, T_LOAD_CLASS));
    return k > 0 && T.fv(me, T_LIFT_LIGHT + int(k) - 1) > 0.0f && T.fv(load, T_LOAD_SIZE) <= T.fv(me, T_LIFT_SIZE);
}

// featurize_kernels.unit_block, on s.units (the kept records) and s.kept_rows.
void unit_block(const Table& T, const Bundle& B, const UnitRecord* all, int64_t n_all, int64_t player, int64_t ally, float commit_frames,
                float* out, Scratch& s) {
    const UnitRecord* units = s.units.data();
    const int64_t n = int64_t(s.units.size());
    const int D = T.unit_dim;
    const int o_at = T.type_dim + INSTANCE_COLS;
    const int r_at = o_at + ORDER_COLS;
    s.locked.assign(n, 0);
    s.region.resize(n);
    s.rel_ally.resize(n);
    s.rel_player.resize(n);
    s.is_target.assign(n, 0);
    s.is_foe.assign(n, 0);
    s.is_beacon.assign(n, 0);
    s.is_own.assign(n, 0);

    // Whatever can be aimed at, kept or not, by uid.
    s.aim.clear();
    for (int64_t j = 0; j < n_all; ++j) {
        const UnitRecord& a = all[j];
        if (a.ally_team != ally && (a.knowledge == zk::KNOW_LOS || a.knowledge == zk::KNOW_RADAR)) s.aim.push_back(j);
    }
    const int64_t na = int64_t(s.aim.size());
    s.aim_order.resize(na);
    for (int64_t k = 0; k < na; ++k) s.aim_order[k] = k;
    std::sort(s.aim_order.begin(), s.aim_order.end(), [&](int64_t p, int64_t q) {
        const int64_t up = all[s.aim[p]].uid, uq = all[s.aim[q]].uid;
        return up != uq ? up < uq : p < q;
    });
    s.sorted_uid.resize(na);
    for (int64_t k = 0; k < na; ++k) s.sorted_uid[k] = all[s.aim[s.aim_order[k]]].uid;

    bool carries = false;
    for (int64_t i = 0; i < n; ++i) {
        const UnitRecord& u = units[i];
        carries = carries || (u.player == player && T.bv(s.kept_rows[i], B_TRANSPORT) && (u.flags & zk::UF_BEING_BUILT) == 0);
    }

    const int rows_p = B.pass_rows, cols_p = B.pass_cols;
    for (int64_t i = 0; i < n; ++i) {
        const UnitRecord& u = units[i];
        const int64_t r = s.kept_rows[i];
        float* o = out + i * D;
        instance_row(T, B, u, r, o);
        int64_t ci, cj;
        cell_of(u.x, u.z, B.cell, B.label_h, B.label_w, ci, cj);
        s.region[i] = B.label[ci * B.label_w + cj];
        s.rel_ally[i] = int32_t(relative_id(u.ally_team, ally, MAX_ALLY_TEAMS));
        s.rel_player[i] = int32_t(relative_id(u.player, player, MAX_PLAYERS));
        const uint16_t flags = u.flags;
        const bool own = u.player == player;
        const bool hostile = u.ally_team != ally;
        const bool built = (flags & zk::UF_BEING_BUILT) == 0;
        s.is_own[i] = own;
        s.is_foe[i] = hostile;
        const bool loads = carries && built && T.fv(r, T_LOAD_CLASS) > 0.0f && (flags & zk::UF_CARRIED) == 0;
        s.is_target[i] = (hostile && (u.knowledge == zk::KNOW_LOS || u.knowledge == zk::KNOW_RADAR)) ||
                         (own && (!built || T.bv(r, B_FACTORY) || loads));
        s.is_beacon[i] = own && T.bv(r, B_BEACON) && built;

        // Order memory.
        const int64_t kind = u.ord_kind;
        const bool has = kind >= 0 && own;
        if (has && o_at + kind < D) o[o_at + kind] = 1.0f;
        const float age = has ? u.ord_age : 0.0f;
        o[o_at + 16] = ::log1pf(np_max(age, 0.0f) / 15.0f) / 6.0f;
        const int64_t target = u.ord_target;
        const bool has_target = has && target >= 0 && kind != PRODUCE;
        bool known = false, alive = false;
        float goal_x = u.ord_x, goal_z = u.ord_z;
        if (has_target && na) {
            int64_t k = std::lower_bound(s.sorted_uid.begin(), s.sorted_uid.end(), target) - s.sorted_uid.begin();
            k = std::min<int64_t>(k, na - 1);
            if (s.sorted_uid[k] == target) {
                const UnitRecord& a = all[s.aim[s.aim_order[k]]];
                known = true;
                goal_x = a.x;
                goal_z = a.z;
                alive = a.knowledge == zk::KNOW_LOS;
            }
        }
        const bool goes = known || (has && kind != ATTACK_UNIT && kind != STOP);
        const float dx = goes ? goal_x - u.x : 0.0f;
        const float dz = goes ? goal_z - u.z : 0.0f;
        o[o_at + 17] = dx / B.map_w;
        o[o_at + 18] = dz / B.map_h;
        o[o_at + 19] = ::hypotf(dx, dz) / B.map_max;
        o[o_at + 20] = has_target ? 1.0f : 0.0f;
        o[o_at + 21] = (has_target && alive) ? 1.0f : 0.0f;
        const bool fresh = has && age < commit_frames;
        o[o_at + RESERVED_KIND] = (own && (flags & zk::UF_PULLING) != 0) ? 1.0f : 0.0f;
        const bool released = (flags & (zk::UF_DAMAGED | zk::UF_IDLE)) != 0 || (has_target && !alive);
        s.locked[i] = fresh && !released;
        o[o_at + 22] = s.locked[i] ? 1.0f : 0.0f;

        // Free distance in eight directions, by its own move class; air units see nothing in the way.
        const int64_t cls = T.bv(r, B_AIR) ? -1 : int64_t(T.move_class[r]);
        for (int d = 0; d < 8; ++d) {
            int first = RAY_CELLS;
            if (cls >= 0) {
                for (int k = 0; k < RAY_CELLS; ++k) {
                    const float pz = (u.z + RAYS.dz[d] * RAYS.t[k]) / PASS_CELL;
                    const float px = (u.x + RAYS.dx[d] * RAYS.t[k]) / PASS_CELL;
                    const int64_t pi = floor_i32(pz);
                    const int64_t pj = floor_i32(px);
                    const bool inside = pi >= 0 && pi < rows_p && pj >= 0 && pj < cols_p;
                    const uint8_t v = B.passable[std::min<int64_t>(std::max<int64_t>(pi, 0), rows_p - 1) * cols_p +
                                                 std::min<int64_t>(std::max<int64_t>(pj, 0), cols_p - 1)];
                    if (!(inside && ((v >> cls) & 1) != 0)) {
                        first = k;
                        break;
                    }
                }
            }
            o[r_at + d] = float(double(first) / double(RAY_CELLS));
        }
    }

    // Hostile and friendly DPS within RAY_DPS_RADIUS per 45-degree sector, summed in float64 in record order.
    s.enemy_sum.assign(size_t(n) * 8, 0.0);
    s.own_sum.assign(size_t(n) * 8, 0.0);
    s.x.resize(n);
    s.z.resize(n);
    s.dist.resize(n);      // each one's DPS, as float32
    for (int64_t b = 0; b < n; ++b) {
        s.x[b] = units[b].x;
        s.z[b] = units[b].z;
        s.dist[b] = T.fv(s.kept_rows[b], T_DPS_GROUND) + T.fv(s.kept_rows[b], T_DPS_AIR);
    }
    bool pairs = false;
    const float* xs = s.x.data();
    const float* zs = s.z.data();
    s.close.resize(n);
    for (int64_t a = 0; a < n; ++a) {
        const float xa = xs[a], za = zs[a];
        const int64_t m = within_radius(xs, zs, n, xa, za, s.close.data());
        double* es = s.enemy_sum.data() + a * 8;
        double* os = s.own_sum.data() + a * 8;
        const int16_t team = units[a].ally_team;
        for (int64_t k = 0; k < m; ++k) {
            const int64_t b = s.close[k];
            if (b == a) continue;
            pairs = true;
            const float dx = xs[b] - xa, dz = zs[b] - za;
            const int64_t sector = sector_of(dx, dz);
            const double dps = double(s.dist[b]);
            if (team == units[b].ally_team)
                os[sector] += dps;
            else
                es[sector] += dps;
        }
    }
    if (pairs) {
        for (int64_t a = 0; a < n; ++a) {
            float* o = out + a * D;
            for (int k = 0; k < 8; ++k) {
                o[r_at + 8 + k] = float(std::log1p(s.enemy_sum[a * 8 + k]) / 8.0);
                o[r_at + 16 + k] = float(std::log1p(s.own_sum[a * 8 + k]) / 8.0);
            }
        }
    }
    s.targets.clear();
    s.foes.clear();
    s.beacons.clear();
    s.own_tokens.clear();
    for (int64_t i = 0; i < n; ++i) {
        if (s.is_target[i]) {
            s.targets.push_back(i);
            s.foes.push_back(s.is_foe[i]);
        }
        if (s.is_beacon[i]) s.beacons.push_back(i);
        if (s.is_own[i]) s.own_tokens.push_back(i);
    }
}

// featurize_kernels.region_block, on featurize._regions' copy of the static columns.
void region_block(const Table& T, const Bundle& B, const UnitRecord* units, int64_t n, const FeatureRecord* features, int64_t nf,
                  int64_t ally, float* out, Scratch& s) {
    const int64_t R = B.regions;
    const int h = B.label_h, w = B.label_w;
    if (nf) {
        s.metal.assign(R, 0.0);
        for (int64_t k = 0; k < nf; ++k) {
            int64_t ci, cj;
            cell_of(features[k].x, features[k].z, B.cell, h, w, ci, cj);
            const int64_t at = B.label[ci * w + cj];
            if (at >= 0) s.metal[at] += double(features[k].metal);
        }
        for (int64_t k = 0; k < R; ++k) out[k * REGION_DIM + 33] = float(std::log1p(s.metal[k]) / 10.0);
    }
    if (!n) return;
    s.sums.assign(6 * R, 0.0);
    double* sums = s.sums.data();
    for (int64_t i = 0; i < n; ++i) {
        const UnitRecord& u = units[i];
        const int64_t row = s.rows[i];
        int64_t ci, cj;
        cell_of(u.x, u.z, B.cell, h, w, ci, cj);
        const int64_t at = py_index(B.label[ci * w + cj], R);
        float hp = u.hp_frac;
        if (hp != hp) hp = 1.0f;
        const double value = double(T.fv(row, T_COST) * np_clip(hp, 0.0f, 1.0f));
        const int know = u.knowledge;
        if (u.ally_team == ally) {
            sums[0 * R + at] += value;
            sums[3 * R + at] += double(T.fv(row, T_DPS_GROUND));
        } else if (know == zk::KNOW_LOS) {
            sums[1 * R + at] += value;
            sums[4 * R + at] += double(T.fv(row, T_DPS_GROUND));
        } else if (know == zk::KNOW_RADAR) {
            sums[2 * R + at] += value;
        } else if (know == zk::KNOW_MEMORY) {
            sums[5 * R + at] += 1.0;
        }
    }
    for (int64_t k = 0; k < R; ++k) {
        float* o = out + k * REGION_DIM;
        for (int j = 0; j < 3; ++j) o[27 + j] = float(std::log1p(sums[j * R + k]) / 10.0);
        o[30] = float(std::log1p(sums[3 * R + k]) / 8.0);
        o[31] = float(std::log1p(sums[4 * R + k]) / 8.0);
        o[32] = float(std::log1p(sums[5 * R + k]) / 3.0);
    }
}

// featurize_kernels.masks.
void masks(const Table& T, const Bundle& B, int64_t c, int64_t n_features, const bool* enabled, uint8_t* kind, uint8_t* ptr_enemy,
           uint8_t* ptr_region, uint8_t* ptr_jump, uint8_t* ptr_wait, uint8_t* ptr_produce, float* route, Scratch& s) {
    const UnitRecord* units = s.units.data();
    const int64_t n = int64_t(s.units.size());
    const int64_t R = B.regions, E = int64_t(s.targets.size()), Tn = T.producible_count;
    const int64_t* rows = s.kept_rows.data();
    for (int64_t i = 0; i < c; ++i) {
        const int64_t me = rows[i];
        uint8_t* pp = ptr_produce + i * Tn;
        for (int64_t k = 0; k < Tn; ++k) pp[k] = T.produces[me * Tn + k] || T.builds[me * Tn + k];
        const UnitRecord& u = units[i];
        if (Tn && u.ord_kind == PRODUCE && u.ord_target >= 0) {
            const int64_t made = T.row(u.ord_target);
            int64_t at = std::lower_bound(T.producible.begin(), T.producible.end(), made) - T.producible.begin();
            at = std::min<int64_t>(at, Tn - 1);
            if (T.producible[at] == made) pp[at] = 0;
        }
    }
    s.with_beacon.assign(R, 0);
    for (int64_t b : s.beacons)
        if (s.region[b] >= 0) s.with_beacon[s.region[b]] = 1;
    for (int64_t i = 0; i < c; ++i) {
        const UnitRecord& u = units[i];
        const int64_t me = rows[i];
        const uint16_t flags = u.flags;
        const bool free = !s.locked[i];
        const bool mobile = T.bv(me, B_MOBILE), armed = T.bv(me, B_ARMED), flies = T.bv(me, B_AIR);
        const int64_t cls = flies ? -1 : int64_t(T.move_class[me]);
        const int64_t here = s.region[i];
        uint8_t* pr = ptr_region + i * R;
        float* rb = route + i * R;
        if (flies && mobile) {
            const float speed = np_max(T.fv(me, T_SPEED), 1.0f);
            for (int64_t k = 0; k < R; ++k) {
                pr[k] = 1;
                rb[k] = ::hypotf(B.centre[k * 2] - u.x, B.centre[k * 2 + 1] - u.z) / speed;
            }
        } else if (cls >= 0 && here >= 0) {
            const float speed = np_max(T.fv(me, T_SPEED), 1.0f);
            const float* walk = B.walk + (cls * R + here) * R;
            for (int64_t k = 0; k < R; ++k) {
                const float w = walk[k];
                const bool finite = std::isfinite(w);
                pr[k] = finite;
                rb[k] = (finite ? w : 0.0f) / speed;
            }
        }
        if (here >= 0) pr[here] = 1;
        bool can_go = false;
        for (int64_t k = 0; k < R; ++k) can_go = can_go || pr[k];

        bool any0 = false, any1 = false, any2 = false, any3 = false, any4 = false;
        const bool lifts_now = T.bv(me, B_TRANSPORT) && (flags & zk::UF_LOADED) == 0;
        const bool hit_ground = T.bv(me, B_HIT_GROUND), hit_air = T.bv(me, B_HIT_AIR), manual = T.bv(me, B_MANUALFIRE);
        const bool missile = T.bv(me, B_MISSILE), launcher = T.bv(me, B_LAUNCHER), constructor = T.bv(me, B_CONSTRUCTOR);
        const float range = T.fv(me, T_RANGE), manual_range = T.fv(me, T_MANUAL_RANGE);
        const float far = range * 1.5f + 300.0f;
        uint8_t* e0 = ptr_enemy + (0 * c + i) * E;
        uint8_t* e1 = ptr_enemy + (1 * c + i) * E;
        uint8_t* e2 = ptr_enemy + (2 * c + i) * E;
        uint8_t* e3 = ptr_enemy + (3 * c + i) * E;
        uint8_t* e4 = ptr_enemy + (4 * c + i) * E;
        for (int64_t j = 0; j < E; ++j) {
            const int64_t tk = s.targets[j];
            const int64_t tr = rows[tk];
            const UnitRecord& tu = units[tk];
            const bool in_los = tu.knowledge == zk::KNOW_LOS;
            const float dx = u.x - tu.x, dz = u.z - tu.z;
            const bool air_target = T.bv(tr, B_AIR);
            const bool foe = s.foes[j];
            const bool hits = (hit_ground && !air_target) || (hit_air && air_target);
            bool p0 = hits && in_los && foe;
            const bool p1 = armed && foe && within(dx, dz, far);
            bool p2 = manual && in_los && foe && within(dx, dz, manual_range);
            if (missile) p0 = !air_target && foe && within(dx, dz, range);
            if (launcher) p2 = !air_target && foe && within(dx, dz, manual_range);
            const bool p3 = constructor && !foe && tk != i;
            const bool p4 = lifts_now && tk != i && lifts(T, me, tr) && (tu.flags & (zk::UF_BEING_BUILT | zk::UF_CARRIED)) == 0 &&
                            (!foe || (in_los && ::hypotf(tu.vx, tu.vz) <= STILL));
            e0[j] = p0;
            e1[j] = p1;
            e2[j] = p2;
            e3[j] = p3;
            e4[j] = p4;
            any0 = any0 || p0;
            any1 = any1 || p1;
            any2 = any2 || p2;
            any3 = any3 || p3;
            any4 = any4 || p4;
        }

        uint8_t* km = kind + i * KIND_COUNT;
        uint8_t* pj = ptr_jump + i * R;
        km[KEEP] = (flags & zk::UF_IDLE) == 0 || !mobile;
        km[MOVE] = free && mobile && can_go;
        km[FIGHT] = free && mobile && armed && can_go;
        km[ATTACK_UNIT] = free && any0;
        km[ATTACK_GROUND] = free && hit_ground && any1;
        km[STOP] = free && (mobile || armed);
        const bool fired = u.ord_kind == MANUAL_FIRE && u.ord_age < T.fv(me, T_MANUAL_RELOAD);
        km[MANUAL_FIRE] = free && any2 && !fired;
        const bool ready = cls >= 0 && here >= 0 && (flags & zk::UF_JUMP_READY) != 0;
        const bool charged = T.bv(me, B_JUMP) && ready;
        const bool throws = T.fv(me, T_THROW_RADIUS) > 0.0f && ready;
        bool can_land = false;
        if (charged || throws) {
            bool loaded = charged;
            if (!charged) {
                const float radius = T.fv(me, T_THROW_RADIUS);
                for (int64_t j = 0; j < n; ++j) {
                    const UnitRecord& o = units[j];
                    const int64_t ro = rows[j];
                    if (T.bv(ro, B_MOBILE) && !T.bv(ro, B_AIR) && o.knowledge == zk::KNOW_LOS && (o.flags & zk::UF_BEING_BUILT) == 0 &&
                        o.uid != u.uid && ::hypotf(o.x - u.x, o.z - u.z) <= radius) {
                        loaded = true;
                        break;
                    }
                }
            }
            if (loaded) {
                const float reach = charged ? T.fv(me, T_JUMP_RANGE) : T.fv(me, T_THROW_REACH);
                for (int64_t k = 0; k < R; ++k) {
                    const float away = ::hypotf(B.centre[k * 2] - u.x, B.centre[k * 2 + 1] - u.z);
                    const bool lands = ((B.region_cells[k] >> cls) & 1) != 0;
                    const bool near = away <= reach + B.region_radius[k];
                    pj[k] = ((charged ? (pr[k] || near) : near) && lands);
                    can_land = can_land || pj[k];
                }
            }
        }
        km[JUMP] = free && charged && can_land;
        km[THROW] = free && throws && can_land;
        const bool loaded = T.bv(me, B_TRANSPORT) && (flags & zk::UF_LOADED) != 0;
        bool can_set = false;
        if (loaded) {
            int64_t load_cls = -1;
            for (int64_t j = 0; j < n; ++j) {
                const UnitRecord& o = units[j];
                if ((o.flags & zk::UF_CARRIED) != 0 && std::fabs(o.x - u.x) <= AT_ITS_TRANSPORT && std::fabs(o.z - u.z) <= AT_ITS_TRANSPORT &&
                    !T.bv(rows[j], B_AIR)) {
                    load_cls = T.move_class[rows[j]];
                    break;
                }
            }
            for (int64_t k = 0; k < R; ++k) {
                pj[k] = load_cls >= 0 ? ((B.region_cells[k] >> load_cls) & 1) != 0 : B.region_cells[k] != 0;
                can_set = can_set || pj[k];
            }
        }
        km[LOAD] = free && any4;
        km[UNLOAD] = free && loaded && can_set;
        km[DROP] = loaded;
        const bool cloaked = (flags & zk::UF_CLOAKED) != 0;
        km[CLOAK_ON] = free && T.bv(me, B_CLOAK) && !cloaked;
        km[CLOAK_OFF] = free && T.bv(me, B_CLOAK) && cloaked;
        const bool pulling = (flags & zk::UF_PULLING) != 0;
        km[PUSH] = free && T.bv(me, B_SWITCHES) && pulling;
        km[PULL] = free && T.bv(me, B_SWITCHES) && !pulling;
        const bool teleports = T.bv(me, B_TELEPORTS);
        km[PLACE_BEACON] = free && teleports && mobile && can_go;
        bool waits = false;
        uint8_t* pw = ptr_wait + i * R;
        for (int64_t k = 0; k < R; ++k) {
            pw[k] = pr[k] && s.with_beacon[k] && mobile && !teleports;
            waits = waits || pw[k];
        }
        km[WAIT_AT_BEACON] = free && waits;
        bool makes = false;
        const uint8_t* pp = ptr_produce + i * Tn;
        for (int64_t k = 0; k < Tn; ++k) makes = makes || pp[k];
        km[PRODUCE] = free && T.bv(me, B_PRODUCES) && makes;
        km[BUILD] = free && T.bv(me, B_BUILDS);
        km[ASSIST] = free && any3;
        km[RECLAIM] = free && T.bv(me, B_CONSTRUCTOR) && n_features > 0;
        for (int k = 0; k < KIND_COUNT; ++k)
            if (!enabled[k]) km[k] = 0;
    }
}

}  // namespace

int64_t bound(const Table& t, const Frame& fr, bool privileged) {
    const int64_t n = fr.n_units, U = std::min<int64_t>(n, CAP_UNITS), NU = U + CAP_SUMMARIES;
    const int64_t C = std::min<int64_t>(n, CAP_CONTROLLABLE), E = U, H = privileged ? std::min<int64_t>(n, CAP_HIDDEN) : 0;
    const int64_t F = std::min<int64_t>(fr.n_features, CAP_FEATURES), S = std::min<int64_t>(fr.n_sites, CAP_SITES);
    const int64_t R = fr.bundle->regions, G = fr.bundle->gates, Tn = t.producible_count, D = t.unit_dim;
    const int64_t bytes = NU * D * 4 + NU * 4 + R * REGION_DIM * 4 + S * SITE_DIM * 4 + GLOBAL_DIM * 4 + (NU + R + S + F + 1) +
                          C * KIND_COUNT + ENEMY_POINTERS * C * E + 3 * C * R + C * Tn + C * R * 4 + G * GATE_DIM * 4 + NU * 8 + E * 8 +
                          8 + R * 4 + U * 8 + U * 4 + H * D * 4 + H * 4 + NU * 8 + H * 8 + F * FEATURE_DIM * 4 + F * 8 + U * 8 + S * 8 +
                          F * 8;
    return bytes + 64 * int64_t(F_COUNT);
}

int featurize(const Table& T, const Frame& fr, const Options& opt, uint8_t* arena_base, int64_t capacity, int64_t* layout,
              int64_t* controllable) {
    Scratch& s = scratch;
    const Bundle& B = *fr.bundle;
    const UnitRecord* all = fr.units;
    const int64_t n_all = fr.n_units;
    const float* g = fr.glob;
    const int64_t player = int64_t(g[zk::G_PLAYER]), ally = int64_t(g[zk::G_ALLY_TEAM]);
    Arena arena{arena_base, capacity, layout};
    const int D = T.unit_dim;

    float army[2], foe[2];
    bool seen = false;
    const int64_t c = select(T, all, n_all, player, ally, s, army, foe, seen);
    const int64_t u = int64_t(s.picked.size());
    s.units.resize(u);
    s.kept_rows.resize(u);
    for (int64_t k = 0; k < u; ++k) {
        s.units[k] = all[s.picked[k]];
        s.kept_rows[k] = s.rows[s.picked[k]];
    }

    // Summaries: one token per (whose, what's known, coarse role) for what the caps left out; the biggest CAP_SUMMARIES, in order.
    summaries(T, B, all, player, ally, s);
    const int64_t groups = int64_t(s.summary_counts.size());
    std::vector<int64_t> keep(groups);
    for (int64_t k = 0; k < groups; ++k) keep[k] = k;
    std::stable_sort(keep.begin(), keep.end(), [&](int64_t p, int64_t q) { return -s.summary_counts[p] < -s.summary_counts[q]; });
    keep.resize(std::min<int64_t>(groups, CAP_SUMMARIES));
    std::sort(keep.begin(), keep.end());
    const int64_t sn = int64_t(keep.size());

    // Unit tokens.
    float* unit_x = arena.take<float>(F_UNIT_X, u + sn, D);
    const float commit_frames = float(opt.commit_decisions * DECISION_FRAMES);
    unit_block(T, B, all, n_all, player, ally, commit_frames, unit_x, s);
    int32_t* unit_name = arena.take<int32_t>(F_UNIT_NAME, u + sn);
    int32_t* unit_ally = arena.take<int32_t>(F_UNIT_ALLY, u + sn);
    int32_t* unit_player = arena.take<int32_t>(F_UNIT_PLAYER, u + sn);
    int64_t* uids = arena.take<int64_t>(F_UIDS, u + sn);
    for (int64_t k = 0; k < u; ++k) {
        unit_name[k] = int32_t(s.kept_rows[k]);
        unit_ally[k] = s.rel_ally[k];
        unit_player[k] = s.rel_player[k];
        uids[k] = s.units[k].uid;
    }
    for (int64_t k = 0; k < sn; ++k) {
        std::memcpy(unit_x + (u + k) * D, s.summary_rows.data() + keep[k] * D, sizeof(float) * size_t(D));
        unit_name[u + k] = T.unknown;
        unit_ally[u + k] = s.summary_ally[keep[k]];
        unit_player[u + k] = s.summary_player[keep[k]];
        uids[u + k] = -1;
    }

    // Sites: nearest the base first.
    const float base_x = g[zk::G_BASE_X], base_z = g[zk::G_BASE_Z];
    const SiteRecord* sites = fr.sites;
    int64_t ns = fr.n_sites;
    s.site_order.resize(ns);
    for (int64_t k = 0; k < ns; ++k) s.site_order[k] = k;
    if (ns > CAP_SITES) {
        s.dist.resize(ns);
        std::vector<int32_t> ids(ns);
        for (int64_t k = 0; k < ns; ++k) {
            s.dist[k] = ::hypotf(sites[k].x - base_x, sites[k].z - base_z);
            ids[k] = sites[k].site_id;
        }
        lexorder2(s.site_order, ns, s.dist.data(), ids.data());
        ns = CAP_SITES;
        s.site_order.resize(ns);
    }
    float* site_x = arena.take<float>(F_SITE_X, ns, SITE_DIM);
    float* site_pos = arena.take<float>(F_SITE_POS, ns, 2);
    for (int64_t k = 0; k < ns; ++k) {
        const SiteRecord& st = sites[s.site_order[k]];
        float* o = site_x + k * SITE_DIM;
        o[0] = st.x / B.map_w * 2.0f - 1.0f;
        o[1] = st.z / B.map_h * 2.0f - 1.0f;
        o[2] = st.income / 5.0f;
        const int64_t status = opt.economy ? int64_t(st.status) : 0;
        if (3 + status < SITE_DIM) o[3 + status] = 1.0f;
        int64_t ci, cj;
        cell_of(st.x, st.z, B.cell, B.label_h, B.label_w, ci, cj);
        const int64_t region = py_index(B.label[ci * B.label_w + cj], B.regions);
        const float dx = B.centre[region * 2] - st.x, dz = B.centre[region * 2 + 1] - st.z;
        o[7] = ::hypotf(dx, dz) / 1000.0f;
        o[8] = ::hypotf(st.x - base_x, st.z - base_z) / B.map_max;
        site_pos[k * 2] = st.x;
        site_pos[k * 2 + 1] = st.z;
    }

    // Features: featurize._pick_features.
    const FeatureRecord* feats = fr.features;
    int64_t nf_all = fr.n_features;
    s.feat_order.resize(nf_all);
    for (int64_t k = 0; k < nf_all; ++k) s.feat_order[k] = k;
    if (nf_all > CAP_FEATURES) {
        std::vector<float> neg(nf_all);
        std::vector<int64_t> ids(nf_all);
        for (int64_t k = 0; k < nf_all; ++k) {
            const float score = feats[k].metal / (::hypotf(feats[k].x - army[0], feats[k].z - army[1]) + 200.0f);
            neg[k] = -score;
            ids[k] = feats[k].feat_id;
        }
        lexorder2(s.feat_order, nf_all, neg.data(), ids.data());
        s.feat_order.resize(CAP_FEATURES);
    }
    std::stable_sort(s.feat_order.begin(), s.feat_order.end(), [&](int64_t p, int64_t q) { return feats[p].feat_id < feats[q].feat_id; });
    const int64_t nf = int64_t(s.feat_order.size());
    float* feat_x = arena.take<float>(F_FEAT_X, nf, FEATURE_DIM);
    int64_t* feat_ids = arena.take<int64_t>(F_FEAT_IDS, nf);
    float* feat_pos = arena.take<float>(F_FEAT_POS, nf, 2);
    for (int64_t k = 0; k < nf; ++k) {
        const FeatureRecord& f = feats[s.feat_order[k]];
        float* o = feat_x + k * FEATURE_DIM;
        o[0] = f.x / B.map_w * 2.0f - 1.0f;
        o[1] = f.z / B.map_h * 2.0f - 1.0f;
        o[2] = ::log1pf(f.metal) / 8.0f;
        o[3] = ::hypotf(f.x - base_x, f.z - base_z) / B.map_max;
        o[4] = ::hypotf(f.x - army[0], f.z - army[1]) / B.map_max;
        feat_ids[k] = f.feat_id;
        feat_pos[k * 2] = f.x;
        feat_pos[k * 2 + 1] = f.z;
    }

    // Regions: the static columns, then what's on the map now.
    const int64_t R = B.regions;
    float* reg_x = arena.take<float>(F_REG_X, R, REGION_DIM);
    if (R) std::memcpy(reg_x, B.reg_x.data(), sizeof(float) * size_t(R * REGION_DIM));
    region_block(T, B, all, n_all, feats, nf_all, ally, reg_x, s);

    // Global: featurize._global.
    float* glob_x = arena.take<float>(F_GLOB_X, GLOBAL_DIM);
    glob_x[zk::G_FRAME] = g[zk::G_FRAME] / 144000.0f;     // fz.GAME_FRAMES: 80 minutes
    if (opt.economy) {
        for (int k : {zk::G_METAL, zk::G_METAL_INCOME, zk::G_METAL_STORAGE, zk::G_ENERGY, zk::G_ENERGY_INCOME, zk::G_ENERGY_STORAGE}) {
            const double v = double(g[k]);
            glob_x[k] = float(std::log1p(v > 0.0 ? v : 0.0) / 10.0);
        }
        glob_x[zk::G_UNFINISHED_STRUCTURES] = g[zk::G_UNFINISHED_STRUCTURES] / 8.0f;
        glob_x[zk::G_IDLE_BUILDERS] = g[zk::G_IDLE_BUILDERS] / 8.0f;
    }
    glob_x[zk::G_COMMANDER_HP] = g[zk::G_COMMANDER_HP];
    glob_x[zk::G_COMMANDER_UID] = g[zk::G_COMMANDER_UID] >= 0.0f ? 1.0f : 0.0f;
    glob_x[zk::G_BASE_X] = g[zk::G_BASE_X] / B.map_w * 2.0f - 1.0f;
    glob_x[zk::G_BASE_Z] = g[zk::G_BASE_Z] / B.map_h * 2.0f - 1.0f;
    glob_x[zk::G_MAP_W] = float(B.map_w_d / 16384.0);
    glob_x[zk::G_MAP_H] = float(B.map_h_d / 16384.0);
    {
        const double v = double(g[zk::G_FRAMES_SINCE_CONTACT]);
        glob_x[zk::G_FRAMES_SINCE_CONTACT] = float(std::log1p((v > 0.0 ? v : 0.0) / 30.0) / 6.0);
    }

    // Token kinds.
    const int64_t n_tok = u + sn + R + ns + nf + 1;
    int8_t* tok_kind = arena.take<int8_t>(F_TOK_KIND, n_tok);
    {
        int64_t at = 0;
        for (int64_t k = 0; k < u; ++k) tok_kind[at++] = KIND_OF_KNOWLEDGE[s.units[k].knowledge & 3];
        for (int64_t k = 0; k < sn; ++k) tok_kind[at++] = TK_SUMMARY;
        for (int64_t k = 0; k < R; ++k) tok_kind[at++] = TK_REGION;
        for (int64_t k = 0; k < ns; ++k) tok_kind[at++] = TK_SITE;
        for (int64_t k = 0; k < nf; ++k) tok_kind[at++] = TK_FEATURE;
        tok_kind[at] = TK_GLOBAL;
    }

    // Actions.
    const int64_t E = int64_t(s.targets.size());
    uint8_t* kind = arena.take<uint8_t>(F_KIND_MASK, c, KIND_COUNT);
    uint8_t* ptr_enemy = arena.take<uint8_t>(F_PTR_ENEMY, ENEMY_POINTERS, c, E);
    uint8_t* ptr_region = arena.take<uint8_t>(F_PTR_REGION, c, R);
    uint8_t* ptr_jump = arena.take<uint8_t>(F_PTR_JUMP, c, R);
    uint8_t* ptr_wait = arena.take<uint8_t>(F_PTR_WAIT, c, R);
    uint8_t* ptr_produce = arena.take<uint8_t>(F_PTR_PRODUCE, c, T.producible_count);
    float* route = arena.take<float>(F_ROUTE_BIAS, c, R);
    if (arena.ok) masks(T, B, c, nf, opt.enabled, kind, ptr_enemy, ptr_region, ptr_jump, ptr_wait, ptr_produce, route, s);

    // North: from our army towards the enemy.
    if (!seen) {
        foe[0] = float(B.map_w_d * 0.5);
        foe[1] = float(B.map_h_d * 0.5);
    }
    float* north = arena.take<float>(F_NORTH, 2);
    {
        const float nx = foe[0] - army[0], nz = foe[1] - army[1];
        const float norm = ::hypotf(nx, nz);
        if (double(norm) > 1.0) {
            north[0] = nx / norm;
            north[1] = nz / norm;
        } else {
            north[0] = 0.0f;
            north[1] = 1.0f;
        }
    }

    int64_t* enemy_tokens = arena.take<int64_t>(F_ENEMY_TOKENS, E);
    for (int64_t k = 0; k < E; ++k) enemy_tokens[k] = s.targets[k];
    const int64_t nb = int64_t(s.beacons.size());
    int64_t* own_beacons = arena.take<int64_t>(F_OWN_BEACONS, nb);
    int32_t* beacon_regions = arena.take<int32_t>(F_BEACON_REGIONS, nb);
    for (int64_t k = 0; k < nb; ++k) {
        own_beacons[k] = s.beacons[k];
        beacon_regions[k] = s.region[s.beacons[k]];
    }
    const int64_t np_ = int64_t(s.own_tokens.size());
    int64_t* place_tokens = arena.take<int64_t>(F_PLACE_TOKENS, np_);
    for (int64_t k = 0; k < np_; ++k) place_tokens[k] = s.own_tokens[k];
    float* gate_x = arena.take<float>(F_GATE_X, B.gates, GATE_DIM);
    if (B.gates) std::memcpy(gate_x, B.gate_x.data(), sizeof(float) * size_t(B.gates * GATE_DIM));
    float* region_radius = arena.take<float>(F_REGION_RADIUS, R);
    if (R) std::memcpy(region_radius, B.region_radius.data(), sizeof(float) * size_t(R));

    // Enemies the player can't see, for the critic alone: featurize._hidden.
    s.hidden.clear();
    if (opt.privileged) {
        for (int64_t i = 0; i < n_all; ++i)
            if (all[i].knowledge == zk::KNOW_HIDDEN) s.hidden.push_back(i);
        const int64_t nh = int64_t(s.hidden.size());
        if (nh > CAP_HIDDEN) {
            std::vector<float> d(nh);
            std::vector<int64_t> ids(nh), order;
            for (int64_t k = 0; k < nh; ++k) {
                const UnitRecord& h = all[s.hidden[k]];
                d[k] = ::hypotf(h.x - army[0], h.z - army[1]);
                ids[k] = h.uid;
            }
            lexorder2(order, nh, d.data(), ids.data());
            std::vector<int64_t> picked(CAP_HIDDEN);
            for (int k = 0; k < CAP_HIDDEN; ++k) picked[k] = s.hidden[order[k]];
            s.hidden.swap(picked);
        }
    }
    const int64_t nh = int64_t(s.hidden.size());
    float* hidden_x = arena.take<float>(F_HIDDEN_X, nh, D);
    int32_t* hidden_name = arena.take<int32_t>(F_HIDDEN_NAME, nh);
    int32_t* hidden_ally = arena.take<int32_t>(F_HIDDEN_ALLY, nh);
    int32_t* hidden_player = arena.take<int32_t>(F_HIDDEN_PLAYER, nh);
    for (int64_t k = 0; k < nh; ++k) {
        const UnitRecord& h = all[s.hidden[k]];
        const int64_t r = s.rows[s.hidden[k]];
        instance_row(T, B, h, r, hidden_x + k * D);
        hidden_name[k] = int32_t(r);
        hidden_ally[k] = int32_t(relative_id(h.ally_team, ally, MAX_ALLY_TEAMS));
        hidden_player[k] = int32_t(relative_id(h.player, player, MAX_PLAYERS));
    }
    *controllable = c;
    return arena.ok ? 0 : -1;
}

}  // namespace zkp
