// actions.py and groups.py, bit for bit: choices to orders (decode), orders to choices (encode, the cloning labels), the
// points they stand for (fz.fine_point, fine_points, place_points), actions and decisions (to_slots, to_decision), and group
// orders (bucket, expand).  Where Python computes in float32 (numpy scalars with Python floats, under NEP 50) this does too,
// and in float64 where it does.
#include "actions.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "npy.hpp"

namespace zkp {

namespace {

constexpr double GROUND_FINE_RADIUS = 200.0;
constexpr double THROW_SHORT = 0.98;
constexpr double PLACE_RING[PLACE_RINGS] = {64.0, 128.0, 256.0};

bool with_region(int64_t k) {
    return k == MOVE || k == FIGHT || k == JUMP || k == THROW || k == PLACE_BEACON || k == WAIT_AT_BEACON || k == UNLOAD;
}
bool with_fine(int64_t k) {
    return k == MOVE || k == FIGHT || k == JUMP || k == ATTACK_GROUND || k == THROW || k == PLACE_BEACON || k == UNLOAD;
}
bool reach_kind(int64_t k) { return k == JUMP || k == THROW || k == UNLOAD; }
int pointer_of(int64_t k) {
    switch (k) {
        case ATTACK_UNIT: return 0;
        case ATTACK_GROUND: return 1;
        case MANUAL_FIRE: return 2;
        case ASSIST: return 3;
        case LOAD: return 4;
        default: return -1;
    }
}

// The 8 compass points of actions.py's arrays: np.arange(8) * np.pi / 4.0, north towards the enemy, in float64.
void compass(const TensorsView& t, double out[8][2]) {
    const double n0 = t.north[0], n1 = t.north[1], e0 = n1, e1 = -n0;
    for (int k = 0; k < 8; ++k) {
        const double angle = double(k) * M_PI / 4.0;
        const double c = std::cos(angle), s = std::sin(angle);
        out[k][0] = c * n0 + s * e0;
        out[k][1] = c * n1 + s * e1;
    }
}

// actions._within_throw, in float64: the reach by t.unit_name, which the binding gives as the units' real names.
Point within_throw(const Table& T, const TensorsView& t, int64_t i, Point p) {
    const double reach = double(T.fv(t.unit_name[i], T_THROW_REACH)) * THROW_SHORT;
    const double ux = t.unit_x[i], uz = t.unit_z[i];
    const double dx = p.x - ux, dz = p.z - uz;
    const double away = std::hypot(dx, dz);
    if (away <= reach) return p;
    return {ux + dx * reach / away, uz + dz * reach / away};
}

// actions._beacon_in: the uid of the nearest beacon of ours in the region, or -1 if it has none (where Python raises).
int64_t beacon_in(const TensorsView& t, int64_t region, float x, float z) {
    int64_t best = -1;
    float best_d = 0.0f;
    for (int64_t k = 0; k < t.beacons; ++k) {
        if (t.beacon_regions[k] != region) continue;
        const int64_t b = t.own_beacons[k];
        const float d = ::hypotf(t.unit_x[b] - x, t.unit_z[b] - z);
        if (best >= 0 && best_d != best_d) break;
        if (best < 0 || d != d || d < best_d) {
            best = b;
            best_d = d;
        }
    }
    return best < 0 ? -1 : t.uids[best];
}

// The first legal option nearest (x, z), as actions._nearest: legal ones by distance, NaN first, ties by order.
int64_t nearest(const double* px, const double* pz, const uint8_t* legal, int64_t n, double x, double z) {
    int64_t best = 0;
    double best_d = std::numeric_limits<double>::infinity();
    bool have = false;
    for (int64_t k = 0; k < n; ++k) {
        const double d = legal[k] ? std::hypot(px[k] - x, pz[k] - z) : std::numeric_limits<double>::infinity();
        if (d != d) return k;
        if (!have || d < best_d) {
            best = k;
            best_d = d;
            have = true;
        }
    }
    return best;
}

int64_t producible_index(const Table& T, int64_t row) {
    const auto& made = T.producible;
    const int64_t p = std::lower_bound(made.begin(), made.end(), row) - made.begin();
    return (p < int64_t(made.size()) && made[p] == row) ? p : -1;
}

}  // namespace

Point fine_point(const Bundle& b, const TensorsView& t, int64_t region, int64_t option) {
    region = py_index(region, b.regions);
    const double cx = b.centre[region * 2], cz = b.centre[region * 2 + 1];
    if (option == 0) return {cx, cz};
    const double angle = double(option - 1) * M_PI / 4.0;
    const double c = std::cos(angle), s = std::sin(angle);
    const double n0 = t.north[0], n1 = t.north[1];
    const double r = t.region_radius[region];
    return {cx + r * (c * n0 + s * n1), cz + r * (c * n1 - s * n0)};
}

// actions.place_points[place, offset], given compass().
void place_point(const TensorsView& t, const double c[8][2], int64_t place, int64_t offset, double& x, double& z) {
    double bx, bz;
    if (place < t.places) {
        const int64_t token = t.place_tokens[place];
        bx = t.unit_x[token];
        bz = t.unit_z[token];
    } else {
        bx = t.site_pos[(place - t.places) * 2];
        bz = t.site_pos[(place - t.places) * 2 + 1];
    }
    double ox = 0.0, oz = 0.0;
    if (offset > 0) {
        const int64_t ring = (offset - 1) / 8, dir = (offset - 1) % 8;
        ox = PLACE_RING[ring] * c[dir][0];
        oz = PLACE_RING[ring] * c[dir][1];
    }
    x = bx + ox;
    z = bz + oz;
}

int64_t decode(const Table& T, const Bundle& B, const TensorsView& t, const int64_t* d, std::vector<Order>& out) {
    const int64_t C = t.controllable;
    const int64_t *kinds = d, *enemy = d + C, *region = d + 2 * C, *fine = d + 3 * C, *produce = d + 4 * C, *group = d + 5 * C,
                  *roles = d + 6 * C, *place = d + 7 * C, *offset = d + 8 * C;
    out.clear();
    for (int64_t i = 0; i < C; ++i) {
        const int64_t kind = kinds[i];
        if (kind == KEEP) continue;
        Order o{t.uids[i], kind, 0.0, 0.0, -1, 0.0, 0};
        if (kind == WAIT_AT_BEACON) {
            o.target = beacon_in(t, region[i], t.unit_x[i], t.unit_z[i]);
            if (o.target < 0) return -1 - i;
        } else if (with_region(kind)) {
            if (region[i] < -B.regions || region[i] >= B.regions) return -1 - i;
            Point p = fine_point(B, t, region[i], fine[i]);
            if (kind == THROW) p = within_throw(T, t, i, p);
            o.x = p.x;
            o.z = p.z;
        } else if (kind == BUILD) {
            if (place[i] < 0 || place[i] >= t.places + t.sites || offset[i] < 0 || offset[i] >= PLACE_OFFSETS || produce[i] < 0 ||
                produce[i] >= t.producible)
                return -1 - i;
            double c[8][2];
            compass(t, c);
            place_point(t, c, place[i], offset[i], o.x, o.z);
            o.target = T.producible[produce[i]];
        } else if (kind == PRODUCE) {
            if (produce[i] < 0 || produce[i] >= t.producible) return -1 - i;
            o.target = T.producible[produce[i]];
        } else if (kind == RECLAIM) {
            if (place[i] < 0 || place[i] >= t.features) return -1 - i;
            o.x = t.feat_pos[place[i] * 2];
            o.z = t.feat_pos[place[i] * 2 + 1];
            o.target = t.feat_ids[place[i]];
        } else if (kind == ATTACK_UNIT || kind == MANUAL_FIRE || kind == ASSIST || kind == LOAD) {
            if (enemy[i] < 0 || enemy[i] >= t.enemies) return -1 - i;
            const int64_t token = t.enemy_tokens[enemy[i]];
            o.x = t.unit_x[token];
            o.z = t.unit_z[token];
            o.target = t.uids[token];
        } else if (kind == ATTACK_GROUND) {
            if (enemy[i] < 0 || enemy[i] >= t.enemies) return -1 - i;
            const int64_t token = t.enemy_tokens[enemy[i]];
            o.x = t.unit_x[token];
            o.z = t.unit_z[token];
            const int64_t option = fine[i];
            if (option > 0) {
                const double angle = double(option - 1) * M_PI / 4.0;
                const double c = std::cos(angle), s = std::sin(angle);
                const double n0 = t.north[0], n1 = t.north[1];
                o.x += GROUND_FINE_RADIUS * (c * n0 + s * n1);
                o.z += GROUND_FINE_RADIUS * (c * n1 - s * n0);
            }
        }
        if (group_kind(kind) && group[i] >= 0) {
            if (group[i] >= GROUP_COUNT) return -1 - i;
            o.radius = GROUP_RADII[group[i]];
            o.roles = roles[i];
        }
        out.push_back(o);
    }
    return int64_t(out.size());
}

int64_t encode(const Table& T, const TensorsView& t, const Bundle& B, const Order* orders, int64_t n, int64_t* d) {
    const int64_t C = t.controllable, R = B.regions, E = t.enemies;
    int64_t *kind = d, *enemy = d + C, *region = d + 2 * C, *fine = d + 3 * C, *produce = d + 4 * C, *group = d + 5 * C,
            *roles = d + 6 * C, *place = d + 7 * C, *offset = d + 8 * C;
    for (int64_t i = 0; i < C; ++i) {
        kind[i] = KEEP;
        enemy[i] = region[i] = fine[i] = produce[i] = group[i] = place[i] = offset[i] = -1;
        roles[i] = 0;
    }
    // uid -> token, the last of equals (a dict built in order); token -> its place among the targets.
    std::vector<std::pair<int64_t, int64_t>> by_uid;
    by_uid.reserve(t.units);
    for (int64_t k = 0; k < t.units; ++k)
        if (t.uids[k] >= 0) by_uid.emplace_back(t.uids[k], k);
    std::stable_sort(by_uid.begin(), by_uid.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    auto token_of = [&](int64_t uid) -> int64_t {
        auto hi = std::upper_bound(by_uid.begin(), by_uid.end(), uid, [](int64_t v, const auto& e) { return v < e.first; });
        if (hi == by_uid.begin() || (hi - 1)->first != uid) return -1;
        return (hi - 1)->second;
    };
    std::vector<int64_t> enemy_of(t.units, -1);
    for (int64_t e = 0; e < E; ++e) enemy_of[t.enemy_tokens[e]] = e;

    // fine_points: R x FINE_OPTIONS, made when first wanted.
    std::vector<double> fx, fz;
    double cmp[8][2];
    compass(t, cmp);
    auto points = [&]() {
        if (!fx.empty() || !R) return;
        fx.resize(R * FINE_OPTIONS);
        fz.resize(R * FINE_OPTIONS);
        for (int64_t r = 0; r < R; ++r) {
            const double cx = B.centre[r * 2], cz = B.centre[r * 2 + 1], rr = t.region_radius[r];
            fx[r * FINE_OPTIONS] = cx;
            fz[r * FINE_OPTIONS] = cz;
            for (int k = 0; k < 8; ++k) {
                fx[r * FINE_OPTIONS + 1 + k] = cx + rr * cmp[k][0];
                fz[r * FINE_OPTIONS + 1 + k] = cz + rr * cmp[k][1];
            }
        }
    };
    std::vector<double> px, pz;
    auto places = [&]() {
        if (!px.empty()) return;
        const int64_t P = t.places + t.sites;
        px.resize(P * PLACE_OFFSETS);
        pz.resize(P * PLACE_OFFSETS);
        for (int64_t p = 0; p < P; ++p)
            for (int64_t k = 0; k < PLACE_OFFSETS; ++k) place_point(t, cmp, p, k, px[p * PLACE_OFFSETS + k], pz[p * PLACE_OFFSETS + k]);
    };
    std::vector<double> ox, oz;
    std::vector<uint8_t> legal;
    int64_t missed = 0;
    for (int64_t q = 0; q < n; ++q) {
        const Order& o = orders[q];
        const int64_t i = token_of(o.uid);
        const int64_t col = o.kind < 0 ? o.kind + KIND_COUNT : o.kind;
        if (!(0 <= i && i < C) || col < 0 || col >= KIND_COUNT || !t.kind_mask[i * KIND_COUNT + col]) {
            ++missed;
            continue;
        }
        const int64_t k = o.kind;
        // This order's choices, which replace whatever an earlier order to the unit chose (actions.encode).
        int64_t c_enemy = -1, c_region = -1, c_fine = -1, c_produce = -1, c_group = -1, c_roles = 0, c_place = -1, c_offset = -1;
        if (with_region(k)) {
            const uint8_t* mask = reach_kind(k) ? t.ptr_jump : (k == WAIT_AT_BEACON ? t.ptr_wait : t.ptr_region);
            double x = o.x, z = o.z;
            if (k == WAIT_AT_BEACON) {
                const int64_t beacon = token_of(o.target);
                if (beacon < 0) {
                    ++missed;
                    continue;
                }
                x = t.unit_x[beacon];
                z = t.unit_z[beacon];
            }
            points();
            const int64_t O = with_fine(k) ? FINE_OPTIONS : 1;
            ox.resize(R * O);
            oz.resize(R * O);
            legal.resize(R * O);
            for (int64_t r = 0; r < R; ++r)
                for (int64_t p = 0; p < O; ++p) {
                    ox[r * O + p] = fx[r * FINE_OPTIONS + p];
                    oz[r * O + p] = fz[r * FINE_OPTIONS + p];
                    legal[r * O + p] = mask[i * R + r];
                }
            const int64_t at = nearest(ox.data(), oz.data(), legal.data(), R * O, x, z);
            c_region = at / O;
            c_fine = with_fine(k) ? at % O : -1;
        } else if (pointer_of(k) >= 0) {
            const int pointer = pointer_of(k);
            const uint8_t* row = t.ptr_enemy + (pointer * C + i) * E;
            if (k == ATTACK_GROUND) {
                bool any = false;
                for (int64_t e = 0; e < E; ++e) any = any || row[e];
                if (!any) {
                    ++missed;
                    continue;
                }
                ox.resize(E * FINE_OPTIONS);
                oz.resize(E * FINE_OPTIONS);
                legal.resize(E * FINE_OPTIONS);
                for (int64_t e = 0; e < E; ++e) {
                    const int64_t token = t.enemy_tokens[e];
                    const double ax = t.unit_x[token], az = t.unit_z[token];
                    ox[e * FINE_OPTIONS] = ax;
                    oz[e * FINE_OPTIONS] = az;
                    for (int c = 0; c < 8; ++c) {
                        const double angle = double(c) * M_PI / 4.0;
                        const double cs = std::cos(angle), sn = std::sin(angle);
                        const double n0 = t.north[0], n1 = t.north[1];
                        ox[e * FINE_OPTIONS + 1 + c] = ax + GROUND_FINE_RADIUS * (cs * n0 + sn * n1);
                        oz[e * FINE_OPTIONS + 1 + c] = az + GROUND_FINE_RADIUS * (cs * n1 + sn * -n0);
                    }
                    for (int c = 0; c < FINE_OPTIONS; ++c) legal[e * FINE_OPTIONS + c] = row[e];
                }
                const int64_t at = nearest(ox.data(), oz.data(), legal.data(), E * FINE_OPTIONS, o.x, o.z);
                c_enemy = at / FINE_OPTIONS;
                c_fine = at % FINE_OPTIONS;
            } else {
                const int64_t token = token_of(o.target);
                const int64_t e = token < 0 ? -1 : enemy_of[token];
                if (e < 0 || !row[e]) {
                    ++missed;
                    continue;
                }
                c_enemy = e;
            }
        } else if (k == PRODUCE || k == BUILD) {
            const int64_t p = producible_index(T, o.target);
            if (p < 0 || !t.ptr_produce[i * t.producible + p]) {
                ++missed;
                continue;
            }
            c_produce = p;
            if (k == BUILD) {
                places();
                const int64_t P = int64_t(px.size());
                legal.assign(P, 1);
                const int64_t at = nearest(px.data(), pz.data(), legal.data(), P, o.x, o.z);
                c_place = at / PLACE_OFFSETS;
                c_offset = at % PLACE_OFFSETS;
            }
        } else if (k == RECLAIM) {
            int64_t f = -1;
            for (int64_t j = 0; j < t.features; ++j)
                if (t.feat_ids[j] == o.target) {
                    f = j;
                    break;
                }
            if (f < 0) {
                ++missed;
                continue;
            }
            c_place = f;
        }
        if (o.radius > 0.0 && group_kind(k)) {
            c_group = bucket(o.radius);
            c_roles = c_group >= 0 ? (o.roles & int64_t(T.choosable)) : 0;
        }
        kind[i] = k;
        enemy[i] = c_enemy;
        region[i] = c_region;
        fine[i] = c_fine;
        produce[i] = c_produce;
        group[i] = c_group;
        roles[i] = c_roles;
        place[i] = c_place;
        offset[i] = c_offset;
    }
    return missed;
}

int64_t bucket(double radius) {
    const double v = radius - 1e-3;
    if (v != v) return -1;      // np.searchsorted puts a NaN past everything
    int64_t i = 0;
    while (i < GROUP_COUNT && GROUP_RADII[i] < v) ++i;
    return i < GROUP_COUNT ? i : -1;
}

void expand(const Table& T, int64_t anchor, int64_t roles_mask, double radius, const UnitRecord* units, int64_t n, const int64_t* rows,
            uint8_t* out) {
    int64_t at = -1;
    for (int64_t k = 0; k < n; ++k)
        if (units[k].uid == anchor) {
            at = k;
            break;
        }
    if (at < 0) {
        for (int64_t k = 0; k < n; ++k) out[k] = 0;
        return;
    }
    const UnitRecord& a = units[at];
    const float r = float(radius);
    for (int64_t k = 0; k < n; ++k) {
        const UnitRecord& u = units[k];
        const int64_t row = rows[k];
        const bool moves = T.bv(row, B_MOBILE);
        const bool finished = (u.flags & zk::UF_BEING_BUILT) == 0;
        const bool in_mask = ((roles_mask >> T.role[row]) & 1) == 1;
        const float dx = u.x - a.x, dz = u.z - a.z;
        const bool near = dx * dx + dz * dz <= r * r;
        out[k] = u.uid == anchor || (moves && finished && in_mask && near);
    }
}

}  // namespace zkp
