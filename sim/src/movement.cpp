// Ground movement: acceleration, turning and terrain speed per frame; straight-line walking when the way is
// clear (Zero-K's cmd_raw_move.lua), otherwise the map's routes between segments; then sphere separation.
#include <algorithm>
#include <cmath>
#include <cstring>
#ifdef __SSE2__
#include <emmintrin.h>
#endif

#include "zksim/profile.hpp"
#include "zksim/world.hpp"

namespace zk {

namespace {
constexpr float PI = 3.14159265358979f;
constexpr float GOAL_RADIUS = 16.0f;
constexpr float LINE_STEP = WALK_LINE_STEP;
constexpr float SLOW_TURN_ANGLE = PI * 0.5f;  // unitdefs_post.lua: turnInPlaceAngleLimit 90
constexpr float SEPARATION_MARGIN = 64.0f;    // the largest footprint radius of a neighbour, near enough
constexpr float SEPARATION_SHARE = 0.5f;
// Two that overlap by this much or less are let be (the user, 2026-11-20: "It should be fine to proceed with a ~5 elmo
// tolerance"); past it they're pushed clear of each other, as before.  Pushed for any overlap at all, the units of a standing
// crowd shuffled for good: two in five of those separate looked at were moved every tick, nearly all by under an elmo.
constexpr float SEPARATION_SLACK = 5.0f;
// Units are pushed apart every second tick, a third of a second apart (the user, 2026-11-20, of "the separation every second
// or third tick": "Proceed"; the second is mine).  Every tick it was a fifth of a step's time in a long game with an army
// that moves.  What walks into another overlaps it for a tick longer before it's pushed.
constexpr int SEPARATION_EVERY = 2;
constexpr float MIN_TERRAIN_SPEED = 0.05f;    // a unit pushed onto a cliff can still walk off it
constexpr int ROUTE_LOOKAHEAD = 4;            // segments ahead a unit looks for a clear line, to cut corners
constexpr int STUCK_WINDOW_TICKS = 18;         // 3 s
constexpr float STUCK_SHARE = 0.2f;            // of the window's walk at top speed; less is no progress
constexpr int ESCAPE_TICKS = 18;               // 3 s, doubling with each attempt
constexpr int ESCAPE_MAX_DOUBLINGS = 3;
constexpr float AT_CENTRE = 64.0f;             // near enough: a turning circle can keep a unit from getting closer
constexpr float BACK_ON_LINE = 64.0f;          // MobileCAI.cpp: going back to a fight's line is over this close to it
constexpr float BEACON_WALK_TO = 150.0f;       // unit_teleporter.lua: BEACON_WAIT_RANGE_MOVE
constexpr float OVER_ITS_LOAD = 2.0f;          // a transport flies to right over what it loads, or where it unloads

float wrap(float a) {
    while (a > PI) a -= 2.0f * PI;
    while (a < -PI) a += 2.0f * PI;
    return a;
}

// The very same number, down to the sign of a zero.
bool same_bits(float a, float b) { return std::memcmp(&a, &b, sizeof a) == 0; }

// Whether every test point along the line, LINE_STEP apart, is on ground the class can stand on.  A point well clear of any
// other ground (Map::clearance) answers for the ones after it that are nearer than that ground is, which aren't looked at.
// How clear a point is has to be read before the next can be chosen, which the processor can't overlap as it does tests one
// after another: so it's asked only every few points, and a hop is taken only when it's a long one.
bool line_free(const Map& m, int cls, float x0, float z0, float x1, float z1) {
    constexpr int LONG_HOP = 12, PLAIN_RUN = 8;
    const float dx = x1 - x0, dz = z1 - z0, dist = std::hypot(dx, dz);
    const int steps = int(dist / LINE_STEP);
    const uint64_t* stands = m.stand_bits(cls);
    const uint8_t* clear = m.clearance_grid(cls);
    for (int i = 1; i <= steps;) {
        int cx, cz;
        {
            const float t = float(i) * LINE_STEP / dist;
            m.cell_of(x0 + dx * t, z0 + dz * t, cx, cz);
            if (!Map::bit(stands, size_t(cz) * size_t(m.cols) + size_t(cx))) return false;
        }
        const int hop = m.clear_skip[clear[size_t(cz / CLEAR_BLOCK) * size_t(m.clear_cols) + size_t(cx / CLEAR_BLOCK)]];
        if (hop >= LONG_HOP) {
            i += hop;
            continue;
        }
        const int until = std::min(steps, i + PLAIN_RUN);
        for (++i; i <= until; ++i) {
            const float t = float(i) * LINE_STEP / dist;
            if (!Map::bit(stands, size_t(m.cell_index(x0 + dx * t, z0 + dz * t)))) return false;
        }
    }
    return true;
}

// The places, from `from` up to `to`, of those a unit at (ux, uz) with this footprint overlaps, and isn't right on top of: in
// order, into `out`, which has room for them all.  Four at a time where the processor has it: the same sums, so the same answers.
// Eight at a time (AVX2) was tried on 2026-11-20 and made a step 7 to 13% slower: a stretch is 10 to 20 places long.
// The footprint is the caller's to shorten (SEPARATION_SLACK), so what it reaches with a small neighbour's may be nothing.
int overlapping(const float* sx, const float* sz, const float* radius, int from, int to, float ux, float uz, float footprint, int* out) {
    int n = 0, i = from;
#ifdef __SSE2__
    const __m128 vx = _mm_set1_ps(ux), vz = _mm_set1_ps(uz), vf = _mm_set1_ps(footprint), least = _mm_set1_ps(0.0001f);
    for (; i + 4 <= to; i += 4) {
        const __m128 dx = _mm_sub_ps(vx, _mm_loadu_ps(sx + i)), dz = _mm_sub_ps(vz, _mm_loadu_ps(sz + i));
        const __m128 d2 = _mm_add_ps(_mm_mul_ps(dx, dx), _mm_mul_ps(dz, dz));
        const __m128 reach = _mm_max_ps(_mm_add_ps(vf, _mm_loadu_ps(radius + i)), _mm_setzero_ps());
        int hits = _mm_movemask_ps(_mm_and_ps(_mm_cmplt_ps(d2, _mm_mul_ps(reach, reach)), _mm_cmpge_ps(d2, least)));
        for (; hits; hits &= hits - 1) out[n++] = i + __builtin_ctz(unsigned(hits));
    }
#endif
    for (; i < to; ++i) {
        const float dx = ux - sx[i], dz = uz - sz[i], d2 = dx * dx + dz * dz, reach = std::max(footprint + radius[i], 0.0f);
        if (d2 < reach * reach && d2 >= 0.0001f) out[n++] = i;
    }
    return n;
}
}  // namespace

void Map::set_slopes(const float* slope, const float* normal_x, const float* normal_z) {
    const size_t cells = size_t(cols) * size_t(rows);
    own.slopes.resize(cells);
    for (size_t i = 0; i < cells; ++i) own.slopes[i] = {height[i] < 0.0f ? 0.0f : slope[i], normal_x[i], normal_z[i]};
    slopes = own.slopes;
}

// The distance from each cell to the nearest the class can't stand on, by the larger of the two axes, in two sweeps; then the
// least of each block.  The map's edge is no such cell: a point past it is tested as the cell at the edge.
void Map::set_standing() {
    const size_t cells = size_t(cols) * size_t(rows);
    stand_words = (cells + 63) / 64;
    own.stand.assign(stand_words * size_t(classes), 0);
    stand = own.stand;
    for (int c = 0; c < classes; ++c) {
        const float* grid = speed_grid(c);
        uint64_t* bits = &own.stand[size_t(c) * stand_words];
        for (size_t i = 0; i < cells; ++i)
            if (grid[i] > 0.0f) bits[i >> 6] |= uint64_t(1) << (i & 63);
    }
    clear_cols = (cols + CLEAR_BLOCK - 1) / CLEAR_BLOCK;
    clear_rows = (rows + CLEAR_BLOCK - 1) / CLEAR_BLOCK;
    const size_t blocks = size_t(clear_cols) * size_t(clear_rows);
    own.clearance.assign(blocks * size_t(classes), 255);
    clearance = own.clearance;
    std::vector<int> far(cells);
    constexpr int FAR = 1 << 20;
    for (int c = 0; c < classes; ++c) {
        const float* grid = speed_grid(c);
        const auto at = [&](int cx, int cz) -> int& { return far[size_t(cz) * size_t(cols) + size_t(cx)]; };
        for (int cz = 0; cz < rows; ++cz)
            for (int cx = 0; cx < cols; ++cx) {
                int best = grid[size_t(cz) * size_t(cols) + size_t(cx)] > 0.0f ? FAR : 0;
                if (best > 0) {
                    if (cx > 0) best = std::min(best, at(cx - 1, cz) + 1);
                    if (cz > 0) {
                        best = std::min(best, at(cx, cz - 1) + 1);
                        if (cx > 0) best = std::min(best, at(cx - 1, cz - 1) + 1);
                        if (cx < cols - 1) best = std::min(best, at(cx + 1, cz - 1) + 1);
                    }
                }
                at(cx, cz) = best;
            }
        uint8_t* out = &own.clearance[size_t(c) * blocks];
        for (int cz = rows - 1; cz >= 0; --cz)
            for (int cx = cols - 1; cx >= 0; --cx) {
                int best = at(cx, cz);
                if (best > 0) {
                    if (cx < cols - 1) best = std::min(best, at(cx + 1, cz) + 1);
                    if (cz < rows - 1) {
                        best = std::min(best, at(cx, cz + 1) + 1);
                        if (cx < cols - 1) best = std::min(best, at(cx + 1, cz + 1) + 1);
                        if (cx > 0) best = std::min(best, at(cx - 1, cz + 1) + 1);
                    }
                    at(cx, cz) = best;
                }
                uint8_t& block = out[size_t(cz / CLEAR_BLOCK) * size_t(clear_cols) + size_t(cx / CLEAR_BLOCK)];
                block = uint8_t(std::min(int(block), best));
            }
    }
    // After a test in a cell n cells clear, the tests within (n - 1) cells' width of it are in cells no more than n - 1 away,
    // which the class can stand on: all those less than that far along the line, so one fewer than fit, for the rounding.
    for (int n = 0; n < 256; ++n) clear_skip[n] = std::max(1, int(float(n - 1) * cell / WALK_LINE_STEP));
}

void Map::set_segments(const short* labels) {
    const size_t cells = size_t(cols) * size_t(rows);
    own.nearest_segment.assign(labels, labels + cells * size_t(classes));
    nearest_segment = own.nearest_segment;
    routes.assign(size_t(classes), Routes{});
    for (auto* v : {&own.centre_x, &own.centre_z, &own.gate_x, &own.gate_z}) v->assign(size_t(classes), {});
    for (auto* v : {&own.edge_to, &own.next_edge}) v->assign(size_t(classes), {});
    for (int c = 0; c < classes; ++c) {
        short* fill = &own.nearest_segment[size_t(c) * cells];
        std::vector<int> frontier;
        for (size_t i = 0; i < cells; ++i)
            if (fill[i] >= 0) frontier.push_back(int(i));
        // Rings outward, so a goal on a cliff belongs to the segment beside it.
        for (size_t head = 0; head < frontier.size(); ++head) {
            const int i = frontier[head];
            const int cx = i % cols, cz = i / cols;
            const int around[4] = {cx > 0 ? i - 1 : -1, cx < cols - 1 ? i + 1 : -1, cz > 0 ? i - cols : -1, cz < rows - 1 ? i + cols : -1};
            for (int j : around)
                if (j >= 0 && fill[j] < 0) {
                    fill[j] = fill[i];
                    frontier.push_back(j);
                }
        }
    }
}

// A clear line to the goal first.  Otherwise along the route, as far as the unit can see; failing that to its
// segment's centre, from where the first gate should be in sight.  A unit that has stopped getting anywhere aims
// straight at the next gate or the goal in turn, sliding along whatever is in the way.
void World::waypoint(int u, float gx, float gz, float& wx, float& wz) const {
    const int cls = type_of(u).move_class;
    wx = gx;
    wz = gz;
    if (cls < 0) return;
    const float ux = x[size_t(u)], uz = z[size_t(u)];
    {
        ZK_PROF(WAYPOINT_LINE);
        if (line_free(*map, cls, ux, uz, gx, gz)) return;
    }
    ZK_PROF(WAYPOINT_ROUTE);
    const Routes& r = map->routes[size_t(cls)];
    const int goal = map->segment_at(cls, gx, gz), here = map->segment_at(cls, ux, uz);
    if (r.nodes == 0 || goal < 0 || here < 0) return;
    const int first_edge = here == goal ? -1 : r.next_edge[size_t(here) * size_t(r.nodes) + size_t(goal)];
    if (here != goal && first_edge < 0) return;         // no way there: as near as a straight line gets
    const WalkProgress& p = progress[size_t(u)];
    if (p.escape_ticks > 0) {
        if (here != goal && p.attempts % 2 == 1) {
            wx = r.gate_x[size_t(first_edge)];
            wz = r.gate_z[size_t(first_edge)];
        }
        return;
    }
    const float centre_x = r.centre_x[size_t(here)], centre_z = r.centre_z[size_t(here)];
    // At the centre, and what it should see isn't in sight: the splitting limit left this part hidden.
    const bool at_centre = std::hypot(centre_x - ux, centre_z - uz) < AT_CENTRE;
    if (here == goal) {
        if (!at_centre) {
            wx = centre_x;
            wz = centre_z;
        }
        return;
    }
    wx = centre_x;
    wz = centre_z;
    int node = here;
    for (int hop = 0; hop < ROUTE_LOOKAHEAD; ++hop) {
        const int e = r.next_edge[size_t(node) * size_t(r.nodes) + size_t(goal)];
        if (!line_free(*map, cls, ux, uz, r.gate_x[size_t(e)], r.gate_z[size_t(e)])) {
            if (hop == 0 && at_centre) {
                wx = r.gate_x[size_t(e)];
                wz = r.gate_z[size_t(e)];
            }
            return;
        }
        wx = r.gate_x[size_t(e)];
        wz = r.gate_z[size_t(e)];
        node = r.edge_to[size_t(e)];
        if (node == goal || !line_free(*map, cls, ux, uz, r.centre_x[size_t(node)], r.centre_z[size_t(node)])) return;
        wx = r.centre_x[size_t(node)];
        wz = r.centre_z[size_t(node)];
    }
}

bool World::wants_to_walk(int u, float& gx, float& gz) {
    const auto& d = type_of(u);
    if (!active(u)) return false;
    bool has_target = false;
    for (int m = 0; m < d.n_mounts; ++m) has_target = has_target || weapon_target[size_t(u) * ZK_MAX_MOUNTS + size_t(m)] >= 0;
    if (behaviour == ZK_BEHAVIOUR_ADVANCE) {
        // The gate's fixed behaviour: to the nearest enemy until something is in range.
        const int e = nearest[size_t(u)];
        if (!(d.flags & ZK_U_ARMED) || has_target || e < 0 || !alive[size_t(e)]) return false;
        gx = x[size_t(e)];
        gz = z[size_t(e)];
        return true;
    }
    if (behaviour != ZK_BEHAVIOUR_ORDERS && behaviour != ZK_BEHAVIOUR_SCRIPTED) return false;
    UnitAI& a = ai[size_t(u)];
    a.speed_limit = 0.0f;
    if (jump[size_t(u)].stranded) return false;
    // A transport of its own allyteam is about to pick it up: it waits (unit_transport_load_unload.lua).
    if (held_for_lift[size_t(u)] == frame) return false;
    if (a.dodging) {
        gx = a.dodge_x;
        gz = a.dodge_z;
        if (std::hypot(gx - x[size_t(u)], gz - z[size_t(u)]) > GOAL_RADIUS) return true;
        a.dodging = false;
    }
    if (a.chasing >= 0 && alive[size_t(a.chasing)]) return walk_to_target(u, a.chasing, gx, gz);
    const int o = order[size_t(u)];
    if (o == ZK_ORDER_ATTACK) {
        const int t = order_target[size_t(u)];
        return t >= 0 && alive[size_t(t)] && walk_to_target(u, t, gx, gz);
    }
    if (o == ZK_ORDER_MANUAL_FIRE) {
        const int t = order_target[size_t(u)];
        const float reach = cat->weapons[size_t(d.mount_weapon[manual_mount(u)])].range;
        if (t < 0 && d.throw_radius > 0.0f) {
            gx = goal_x[size_t(u)];
            gz = goal_z[size_t(u)];
            return std::hypot(gx - x[size_t(u)], gz - z[size_t(u)]) > reach;
        }
        if (t < 0 || !alive[size_t(t)] || std::hypot(x[size_t(t)] - x[size_t(u)], z[size_t(t)] - z[size_t(u)]) <= reach)
            return false;
        gx = x[size_t(t)];
        gz = z[size_t(t)];
        return true;
    }
    if (o == ZK_ORDER_JUMP) {
        // AttemptJump's Approach: into range.  There it stands until it's charged.
        gx = goal_x[size_t(u)];
        gz = goal_z[size_t(u)];
        return std::hypot(gx - x[size_t(u)], gz - z[size_t(u)]) >= d.jump_range;
    }
    if (o == ZK_ORDER_LOAD) {
        const int t = order_target[size_t(u)];
        if (t < 0 || !alive[size_t(t)] || carrier[size_t(t)] >= 0) {
            order[size_t(u)] = ZK_ORDER_NONE;
            return false;
        }
        gx = x[size_t(t)];
        gz = z[size_t(t)];
        return std::hypot(gx - x[size_t(u)], gz - z[size_t(u)]) > OVER_ITS_LOAD;
    }
    if (o == ZK_ORDER_UNLOAD) {
        if (cargo[size_t(u)] < 0) {
            order[size_t(u)] = ZK_ORDER_NONE;
            return false;
        }
        gx = goal_x[size_t(u)];
        gz = goal_z[size_t(u)];
        return std::hypot(gx - x[size_t(u)], gz - z[size_t(u)]) > OVER_ITS_LOAD;
    }
    if (o == ZK_ORDER_WAIT_AT_BEACON) {
        const int t = order_target[size_t(u)];
        if (t < 0 || !alive[size_t(t)]) {
            order[size_t(u)] = ZK_ORDER_NONE;
            return false;
        }
        gx = x[size_t(t)];
        gz = z[size_t(t)];
        return std::hypot(gx - x[size_t(u)], gz - z[size_t(u)]) > BEACON_WALK_TO;
    }
    if (o == ZK_ORDER_BUILD) {
        gx = goal_x[size_t(u)];
        gz = goal_z[size_t(u)];
        return !within(u, gx, gz, reach_of(u, cat->types[size_t(order_target[size_t(u)])].footprint_radius));
    }
    if (o == ZK_ORDER_RECLAIM) {
        const int id = order_target[size_t(u)];
        if (!wreck_exists(id)) return false;
        gx = wrecks[size_t(id)].x;
        gz = wrecks[size_t(id)].z;
        return !within(u, gx, gz, reach_of(u, wrecks[size_t(id)].radius));
    }
    if (o == ZK_ORDER_ASSIST) {
        const int t = order_target[size_t(u)];
        if (t < 0 || !alive[size_t(t)]) return false;
        gx = x[size_t(t)];
        gz = z[size_t(t)];
        return !within(u, gx, gz, reach_of(u, type_of(t).footprint_radius));
    }
    if (o == ZK_ORDER_FIGHT && a.returning) {
        if (std::hypot(a.return_x - x[size_t(u)], a.return_z - z[size_t(u)]) > BACK_ON_LINE) {
            gx = a.return_x;
            gz = a.return_z;
            return true;
        }
        a.returning = false;
    }
    if (o == ZK_ORDER_MOVE || o == ZK_ORDER_FIGHT) {
        gx = goal_x[size_t(u)];
        gz = goal_z[size_t(u)];
        if (std::hypot(gx - x[size_t(u)], gz - z[size_t(u)]) > GOAL_RADIUS) return true;
        order[size_t(u)] = ZK_ORDER_NONE;
    }
    return false;
}

void World::track_progress(int u, bool walking) {
    WalkProgress& p = progress[size_t(u)];
    const float px = x[size_t(u)], pz = z[size_t(u)];
    if (!walking) {
        p = WalkProgress{px, pz, 0, 0, -1, 0};
        return;
    }
    const int cls = type_of(u).move_class;
    const int here = cls >= 0 ? map->segment_at(cls, px, pz) : -1;
    if (here != p.segment) {
        // A new segment is real progress; wandering inside one isn't, however far it goes.
        p.segment = here;
        p.attempts = 0;
    }
    if (p.escape_ticks > 0) --p.escape_ticks;
    if (++p.ticks < STUCK_WINDOW_TICKS) return;
    const float expected = type_of(u).speed * float(STUCK_WINDOW_TICKS * ZK_TICK_FRAMES);
    if (std::hypot(px - p.anchor_x, pz - p.anchor_z) < STUCK_SHARE * expected) {
        p.escape_ticks = ESCAPE_TICKS << std::min(p.attempts, ESCAPE_MAX_DOUBLINGS);
        ++p.attempts;
    }
    p.anchor_x = px;
    p.anchor_z = pz;
    p.ticks = 0;
}

namespace {
enum : int8_t { PLAN_BUSY, PLAN_SKID, PLAN_STAND, PLAN_FACE, PLAN_WALK };
}

// What a unit means to do this tick, from where everyone is as the tick's moves begin: move_units asks all of them before it
// moves any.  Asked as each came to move, a unit made before its target walked on while the target, asked after, found it
// already in reach and stood.  PLAN_WALK: to x, z.  PLAN_FACE: standing, turned to what it shoots at, which is at x, z.
World::MovePlan World::plan_move(int u) {
    // Floating, it's held where it is (unit_impulsefloat_toggle.lua): it only goes up and down.
    if (jump[size_t(u)].busy || floating[size_t(u)]) return {PLAN_BUSY, 0.0f, 0.0f};
    const auto& d = type_of(u);
    if (skid[size_t(u)].on && !(d.flags & ZK_U_AIR)) return {PLAN_SKID, 0.0f, 0.0f};
    float gx = 0.0f, gz = 0.0f;
    {
        ZK_PROF(MOVE_WANT);
        if (wants_to_walk(u, gx, gz)) return {PLAN_WALK, gx, gz};
    }
    if (active(u)) {
        // Standing units turn to what they shoot at, which fixed forward weapons need.
        const int t = order[size_t(u)] == ZK_ORDER_MANUAL_FIRE ? order_target[size_t(u)] : weapon_target[size_t(u) * ZK_MAX_MOUNTS];
        if (t >= 0 && alive[size_t(t)]) return {PLAN_FACE, x[size_t(t)], z[size_t(t)]};
    }
    return {PLAN_STAND, 0.0f, 0.0f};
}

void World::move_one(int u, const MovePlan& plan) {
    if (plan.kind == PLAN_BUSY) {
        track_progress(u, false);
        return;
    }
    const auto& d = type_of(u);
    if (plan.kind == PLAN_SKID) {
        track_progress(u, false);
        step_skid(u);
        return;
    }
    const bool walking = plan.kind == PLAN_WALK;
    const float gx = plan.x, gz = plan.z;
    float wanted = heading[size_t(u)];
    track_progress(u, walking);
    if (walking) {
        float wx, wz;
        {
            ZK_PROF(MOVE_WAYPOINT);
            waypoint(u, gx, gz, wx, wz);
        }
        wanted = std::atan2(wx - x[size_t(u)], wz - z[size_t(u)]);
    } else if (plan.kind == PLAN_FACE) {
        wanted = std::atan2(gx - x[size_t(u)], gz - z[size_t(u)]);
    }
    const int cls = d.move_class;
    float h = heading[size_t(u)], v = speed[size_t(u)], px = x[size_t(u)], pz = z[size_t(u)];
    // Some units are slower until their main weapon has reloaded (cloakskirm.lua, cloaksnipe.lua).
    const float reload_mult = cooldown[size_t(u) * ZK_MAX_MOUNTS] > 0.0f ? d.reload_move_penalty : 1.0f;
    // A loaded transport is slower, by its load's class (unit_transport_speed.lua).
    const float slow_mult = (1.0f - slow_now(u)) * reload_mult * loaded_speed(u);
    const float speed_limit = ai[size_t(u)].speed_limit;
    int& wait = order_wait[size_t(u)];
    int& lag = order_lag[size_t(u)];
    ZK_PROF(MOVE_FRAMES);
    // A gunship goes up before it goes anywhere, and one that lands does when it has nothing to do (air.cpp).
    const bool flies = (d.flags & ZK_U_AIR) != 0;
    // Standing still on the ground and facing where it wants to, with numbers that adding nothing to leaves as they are (no
    // negative zero, a heading inside the circle): each frame below would leave it exactly so, and only its waits run down.
    const bool at_rest = !walking && !flies && same_bits(v, 0.0f) && same_bits(wanted, h) && h >= -PI && h <= PI && !same_bits(h, -0.0f) &&
                         !std::signbit(px) && !std::signbit(pz);
    if (at_rest) {
        wait = std::max(wait - ZK_TICK_FRAMES, 0);
        lag = std::max(lag - ZK_TICK_FRAMES, 0);
    }
    // The heading's sine and cosine, kept from one frame's step to the next frame's terrain and for as long as the heading is
    // the same, and the map cell it stands in, kept from the test of where it steps to.
    bool have_trig = cls >= 0 && !at_rest;
    float sin_h = have_trig ? std::sin(h) : 0.0f, cos_h = have_trig ? std::cos(h) : 0.0f;
    size_t here = cls >= 0 && !at_rest ? size_t(map->cell_index(px, pz)) : 0;
    const uint64_t* stands = cls >= 0 ? map->stand_bits(cls) : nullptr;
    const bool busy = walking || order[size_t(u)] != ZK_ORDER_NONE || ai[size_t(u)].chasing >= 0 || !(d.flags & ZK_U_LANDS);
    // A transport coming to its load, or to where it sets it down, flies straight there whichever way it faces, as the engine's
    // hovering aircraft do (HoverAirMoveType).  Turning as it goes, as the rest here do, it would circle what lies inside its turn.
    const bool straight = flies && (order[size_t(u)] == ZK_ORDER_LOAD || order[size_t(u)] == ZK_ORDER_UNLOAD);
    for (int f = at_rest ? ZK_TICK_FRAMES : 0; f < ZK_TICK_FRAMES; ++f) {
        if (flies && !climb_or_land(u, busy)) {
            v = 0.0f;
            continue;
        }
        const bool go = walking && (wait == 0 || lag > 0);
        if (wait > 0) --wait;
        if (lag > 0) --lag;
        const float terrain = cls >= 0 ? map->speed_along_at(cls, here, sin_h, cos_h) : 1.0f;
        float top = go ? d.speed * std::max(terrain, MIN_TERRAIN_SPEED) * slow_mult : 0.0f;
        if (speed_limit > 0.0f) top = std::min(top, speed_limit);
        // What flies comes to a stop at its goal, rather than getting there at full speed and coming back.
        if (cls < 0 && walking) top = std::min(top, std::sqrt(2.0f * d.brake * std::hypot(gx - px, gz - pz)));
        const float diff = (go || !walking) ? wrap(wanted - h) : 0.0f;
        const float limit = !straight && std::fabs(diff) > SLOW_TURN_ANGLE ? std::min(top, d.slow_turn_speed) : top;
        const float turned = straight ? wanted : wrap(h + std::clamp(diff, -d.turn_rate, d.turn_rate));
        v += std::clamp(limit - v, -d.brake, d.accel);
        if (!have_trig || !same_bits(turned, h)) {
            sin_h = std::sin(turned);
            cos_h = std::cos(turned);
            have_trig = true;
        }
        h = turned;
        const float nx = px + sin_h * v, nz = pz + cos_h * v;
        if (cls < 0) {
            px = nx;
            pz = nz;
            continue;
        }
        size_t to = size_t(map->cell_index(nx, nz));
        if (Map::bit(stands, to)) {
            px = nx;
            pz = nz;
            here = to;
        } else if (Map::bit(stands, to = size_t(map->cell_index(nx, pz)))) {
            px = nx;  // slide along what blocks the way
            here = to;
        } else if (Map::bit(stands, to = size_t(map->cell_index(px, nz)))) {
            pz = nz;
            here = to;
        } else {
            v = 0.0f;
        }
    }
    if (skid[size_t(u)].on) step_air_skid(u, px, pz);
    const float frames = float(ZK_TICK_FRAMES);
    px = std::clamp(px, 0.0f, map->width() - 1.0f);
    pz = std::clamp(pz, 0.0f, map->depth() - 1.0f);
    vx[size_t(u)] = (px - x[size_t(u)]) / frames;
    vz[size_t(u)] = (pz - z[size_t(u)]) / frames;
    x[size_t(u)] = px;
    z[size_t(u)] = pz;
    heading[size_t(u)] = h;
    speed[size_t(u)] = v;
    // What it carries is where it is, going as it goes.
    if (const int c = cargo[size_t(u)]; c >= 0) {
        x[size_t(c)] = px;
        z[size_t(c)] = pz;
        vx[size_t(c)] = vx[size_t(u)];
        vz[size_t(c)] = vz[size_t(u)];
        speed[size_t(c)] = v;
    }
}

void World::separate() {
    // What each unit of the grid is to a neighbour, in the grid's order, so that going through a unit's neighbours is going along
    // these: where it is now, its footprint, and the share of an overlap the neighbour takes (immobile things push fully, mobile
    // neighbours share it).  The dead, what flies and what's in the air mid-jump push no one: they're put nowhere near.
    const size_t slots = cell_units.size(), cells = grid_cells();
    sep_x.resize(slots);
    sep_z.resize(slots);
    sep_radius.resize(slots);
    sep_share.resize(slots);
    // Most units stand among units that stand, tick after tick, and a turn here leaves one as it is: it overlaps no one by
    // more than is let be, or is pushed nowhere it can stand.  What a turn does goes by where the unit is, its type, and the
    // units of the cells it looks through as they are when the turn comes.  So a unit left as it was has no turn until it
    // has moved or one of those cells has stirred: a unit of the cell that's anywhere or anything it wasn't, has come, or
    // has gone.  Each stir takes the next number, a cell keeps its last one's, and a unit that of the last before its turn.
    // The same numbers as giving every unit its turn, to the bit (2026-11-20).
    if (sep_seen.size() < size_t(count())) {
        sep_seen.resize(size_t(count()));
        sep_rest.resize(size_t(count()));
    }
    if (sep_stir.size() != cells) {
        sep_stir.assign(cells, 0);
        sep_last.clear();
        for (SepSeen& seen : sep_seen) seen.pass = -1;
        for (SepRest& rest : sep_rest) rest.since = -1;
    }
    ++sep_pass;
    for (size_t i = 0; i < slots; ++i) {
        const int o = cell_units[i];
        const auto& od = type_of(o);
        const bool pushes = alive[size_t(o)] && !(od.flags & ZK_U_AIR) && !jump[size_t(o)].flying;
        sep_x[i] = pushes ? x[size_t(o)] : INFINITY;
        sep_z[i] = z[size_t(o)];
        sep_radius[i] = od.footprint_radius;
        sep_share[i] = (od.flags & ZK_U_MOBILE) ? SEPARATION_SHARE : 1.0f;
        const int32_t cell = int32_t(size_t(unit_key[size_t(o)]) % cells);
        SepSeen& seen = sep_seen[size_t(o)];
        const bool was_here = seen.pass == sep_pass - 1;
        if (!was_here || seen.cell != cell || !same_bits(seen.x, sep_x[i]) || !same_bits(seen.z, sep_z[i]) || !same_bits(seen.radius, sep_radius[i]) ||
            !same_bits(seen.share, sep_share[i])) {
            if (was_here && seen.cell != cell) sep_stir[size_t(seen.cell)] = ++sep_stirs;       // gone from that one
            sep_stir[size_t(cell)] = ++sep_stirs;
            seen.x = sep_x[i];
            seen.z = sep_z[i];
            seen.radius = sep_radius[i];
            seen.share = sep_share[i];
            seen.cell = cell;
        }
        seen.pass = sep_pass;
    }
    // Those the last pass had and this one hasn't have gone from their cells: the dead, and what's been picked up.
    for (const int o : sep_last)
        if (sep_seen[size_t(o)].pass != sep_pass) sep_stir[size_t(sep_seen[size_t(o)].cell)] = ++sep_stirs;
    sep_last.assign(cell_units.begin(), cell_units.end());
    // Room for every place there is: a unit's stretches of the grid are apart, so it overlaps no more.
    sep_hits.resize(slots);
    for (size_t k = 0, n = live.size(); k < n; ++k) {
        const int u = live[k];
        if (!alive[size_t(u)]) continue;
        const auto& d = type_of(u);
        if (!(d.flags & ZK_U_MOBILE) || (d.flags & ZK_U_AIR) || jump[size_t(u)].flying || carrier[size_t(u)] >= 0) continue;
        // Those it overlaps by more than is let be, of all in the cells round it: each allyteam's in each row of cells is one
        // stretch of the grid.  The unit itself is no distance away, and goes with what's too near to tell a direction from.
        const float ux = x[size_t(u)], uz = z[size_t(u)];
        const CellBox b = cell_box(ux, uz, d.footprint_radius + SEPARATION_MARGIN);
        SepRest& rest = sep_rest[size_t(u)];
        if (rest.since >= 0 && rest.type == int32_t(type[size_t(u)]) && same_bits(ux, rest.x) && same_bits(uz, rest.z)) {
            bool stirred = false;
            for (int gz = b.z0; gz <= b.z1 && !stirred; ++gz)
                for (int gx = b.x0; gx <= b.x1 && !stirred; ++gx) stirred = sep_stir[size_t(gz) * size_t(grid_cols) + size_t(gx)] > rest.since;
            if (!stirred) continue;
        }
        // Left as it is, unless the turn ends with it moved.
        rest = {sep_stirs, ux, uz, int32_t(type[size_t(u)])};
        int hits = 0, hit_allies = 0;
        for (int gz = b.z0; gz <= b.z1; ++gz)
            for (int s = 0; s < n_allies; ++s) {
                const size_t row = size_t(s) * cells + size_t(gz) * size_t(grid_cols);
                const int from = cell_start[row + size_t(b.x0)], to = cell_start[row + size_t(b.x1) + 1];
                if (from == to) continue;
                const int found = overlapping(sep_x.data(), sep_z.data(), sep_radius.data(), from, to, ux, uz, d.footprint_radius - SEPARATION_SLACK,
                                              sep_hits.data() + hits);
                if (found) hit_allies |= 1 << s;
                hits += found;
            }
        if (!hits) continue;
        // Their pushes add up in the grid's order: by cell, row by row, and by handle in a cell, whoever's they are.  Found an
        // allyteam at a time, they're in that order unless two allyteams' are among them.
        if (hit_allies & (hit_allies - 1))
            std::sort(sep_hits.begin(), sep_hits.begin() + hits, [&](int a, int c) {
                const int ua = cell_units[size_t(a)], uc = cell_units[size_t(c)];
                const size_t cell_a = size_t(unit_key[size_t(ua)]) % cells, cell_c = size_t(unit_key[size_t(uc)]) % cells;
                return cell_a != cell_c ? cell_a < cell_c : ua < uc;
            });
        float push_x = 0.0f, push_z = 0.0f;
        for (int h = 0; h < hits; ++h) {
            const size_t i = size_t(sep_hits[size_t(h)]);
            const float dx = ux - sep_x[i], dz = uz - sep_z[i];
            const float d2 = dx * dx + dz * dz;
            const float reach = d.footprint_radius + sep_radius[i];
            const float dist = std::sqrt(d2);
            const float share = sep_share[i];
            push_x += dx / dist * (reach - dist) * share;
            push_z += dz / dist * (reach - dist) * share;
        }
        if (push_x == 0.0f && push_z == 0.0f) continue;
        const float nx = std::clamp(x[size_t(u)] + push_x, 0.0f, map->width() - 1.0f);
        const float nz = std::clamp(z[size_t(u)] + push_z, 0.0f, map->depth() - 1.0f);
        // Pushed by less than its place can hold: where it is.
        if (same_bits(nx, ux) && same_bits(nz, uz)) continue;
        if (d.move_class < 0 || Map::bit(map->stand_bits(d.move_class), size_t(map->cell_index(nx, nz)))) {
            x[size_t(u)] = nx;
            z[size_t(u)] = nz;
            rest.since = -1;
            // Later units are pushed from where this one is now: a stir in its cell.
            if (const int slot = grid_slot[size_t(u)]; slot >= 0) {
                sep_x[size_t(slot)] = nx;
                sep_z[size_t(slot)] = nz;
                sep_stir[size_t(unit_key[size_t(u)]) % cells] = ++sep_stirs;
                sep_seen[size_t(u)].x = nx;
                sep_seen[size_t(u)].z = nz;
            }
        }
    }
}

void World::move_units() {
    const size_t n = live.size();
    const auto walks = [&](int u) { return alive[size_t(u)] && (type_of(u).flags & ZK_U_MOBILE) && carrier[size_t(u)] < 0; };
    move_plans.resize(n);
    for (size_t k = 0; k < n; ++k)
        if (const int u = live[k]; walks(u) && !(type_of(u).flags & ZK_U_PLANE)) move_plans[k] = plan_move(u);
    for (size_t k = 0; k < n; ++k)
        if (const int u = live[k]; walks(u)) {
            if (type_of(u).flags & ZK_U_PLANE)
                fly_plane(u);
            else
                move_one(u, move_plans[k]);
        }
    if (tick_count % SEPARATION_EVERY == 0) {
        ZK_PROF(MOVE_SEPARATE);
        separate();
    }
}

}  // namespace zk
