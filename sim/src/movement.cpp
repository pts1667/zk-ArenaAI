// Ground movement: acceleration, turning and terrain speed per frame; straight-line walking when the way is
// clear (Zero-K's cmd_raw_move.lua), otherwise the map's routes between segments; then sphere separation.
#include <algorithm>
#include <cmath>

#include "zksim/profile.hpp"
#include "zksim/world.hpp"

namespace zk {

namespace {
constexpr float PI = 3.14159265358979f;
constexpr float GOAL_RADIUS = 16.0f;
constexpr float LINE_STEP = 16.0f;            // cmd_raw_move.lua's test spacing
constexpr float SLOW_TURN_ANGLE = PI * 0.5f;  // unitdefs_post.lua: turnInPlaceAngleLimit 90
constexpr float SEPARATION_MARGIN = 64.0f;    // the largest footprint radius of a neighbour, near enough
constexpr float SEPARATION_SHARE = 0.5f;
constexpr float MIN_TERRAIN_SPEED = 0.05f;    // a unit pushed onto a cliff can still walk off it
constexpr int ROUTE_LOOKAHEAD = 4;            // segments ahead a unit looks for a clear line, to cut corners
constexpr int STUCK_WINDOW_TICKS = 18;         // 3 s
constexpr float STUCK_SHARE = 0.2f;            // of the window's walk at top speed; less is no progress
constexpr int ESCAPE_TICKS = 18;               // 3 s, doubling with each attempt
constexpr int ESCAPE_MAX_DOUBLINGS = 3;
constexpr float AT_CENTRE = 64.0f;             // near enough: a turning circle can keep a unit from getting closer
constexpr float BACK_ON_LINE = 64.0f;          // MobileCAI.cpp: going back to a fight's line is over this close to it

float wrap(float a) {
    while (a > PI) a -= 2.0f * PI;
    while (a < -PI) a += 2.0f * PI;
    return a;
}

bool line_free(const Map& m, int cls, float x0, float z0, float x1, float z1) {
    const float dx = x1 - x0, dz = z1 - z0, dist = std::hypot(dx, dz);
    const int steps = int(dist / LINE_STEP);
    for (int i = 1; i <= steps; ++i) {
        const float t = float(i) * LINE_STEP / dist;
        if (m.speed_at(cls, x0 + dx * t, z0 + dz * t) <= 0.0f) return false;
    }
    return true;
}
}  // namespace

void Map::set_segments(const short* labels) {
    const size_t cells = size_t(cols) * size_t(rows);
    nearest_segment.assign(labels, labels + cells * size_t(classes));
    routes.assign(size_t(classes), Routes{});
    for (int c = 0; c < classes; ++c) {
        short* fill = &nearest_segment[size_t(c) * cells];
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

void World::move_one(int u) {
    if (jump[size_t(u)].busy) {
        track_progress(u, false);
        return;
    }
    const auto& d = type_of(u);
    float gx = 0.0f, gz = 0.0f;
    bool walking;
    {
        ZK_PROF(MOVE_WANT);
        walking = wants_to_walk(u, gx, gz);
    }
    float wanted = heading[size_t(u)];
    track_progress(u, walking);
    if (walking) {
        float wx, wz;
        {
            ZK_PROF(MOVE_WAYPOINT);
            waypoint(u, gx, gz, wx, wz);
        }
        wanted = std::atan2(wx - x[size_t(u)], wz - z[size_t(u)]);
    } else if (active(u)) {
        // Standing units turn to what they shoot at, which fixed forward weapons need.
        const int t = order[size_t(u)] == ZK_ORDER_MANUAL_FIRE ? order_target[size_t(u)] : weapon_target[size_t(u) * ZK_MAX_MOUNTS];
        if (t >= 0 && alive[size_t(t)]) wanted = std::atan2(x[size_t(t)] - x[size_t(u)], z[size_t(t)] - z[size_t(u)]);
    }
    const int cls = d.move_class;
    float h = heading[size_t(u)], v = speed[size_t(u)], px = x[size_t(u)], pz = z[size_t(u)];
    // Some units are slower until their main weapon has reloaded (cloakskirm.lua, cloaksnipe.lua).
    const float reload_mult = cooldown[size_t(u) * ZK_MAX_MOUNTS] > 0.0f ? d.reload_move_penalty : 1.0f;
    const float slow_mult = (1.0f - slow_now(u)) * reload_mult;
    const float speed_limit = ai[size_t(u)].speed_limit;
    int& wait = order_wait[size_t(u)];
    int& lag = order_lag[size_t(u)];
    ZK_PROF(MOVE_FRAMES);
    for (int f = 0; f < ZK_TICK_FRAMES; ++f) {
        const bool go = walking && (wait == 0 || lag > 0);
        if (wait > 0) --wait;
        if (lag > 0) --lag;
        const float terrain = cls >= 0 ? map->speed_along(cls, px, pz, std::sin(h), std::cos(h)) : 1.0f;
        float top = go ? d.speed * std::max(terrain, MIN_TERRAIN_SPEED) * slow_mult : 0.0f;
        if (speed_limit > 0.0f) top = std::min(top, speed_limit);
        // What flies comes to a stop at its goal, rather than getting there at full speed and coming back.
        if (cls < 0 && walking) top = std::min(top, std::sqrt(2.0f * d.brake * std::hypot(gx - px, gz - pz)));
        const float diff = (go || !walking) ? wrap(wanted - h) : 0.0f;
        const float limit = std::fabs(diff) > SLOW_TURN_ANGLE ? std::min(top, d.slow_turn_speed) : top;
        h = wrap(h + std::clamp(diff, -d.turn_rate, d.turn_rate));
        v += std::clamp(limit - v, -d.brake, d.accel);
        const float nx = px + std::sin(h) * v, nz = pz + std::cos(h) * v;
        if (cls < 0 || map->speed_at(cls, nx, nz) > 0.0f) {
            px = nx;
            pz = nz;
        } else if (map->speed_at(cls, nx, pz) > 0.0f) {
            px = nx;  // slide along what blocks the way
        } else if (map->speed_at(cls, px, nz) > 0.0f) {
            pz = nz;
        } else {
            v = 0.0f;
        }
    }
    const float frames = float(ZK_TICK_FRAMES);
    px = std::clamp(px, 0.0f, map->width() - 1.0f);
    pz = std::clamp(pz, 0.0f, map->depth() - 1.0f);
    vx[size_t(u)] = (px - x[size_t(u)]) / frames;
    vz[size_t(u)] = (pz - z[size_t(u)]) / frames;
    x[size_t(u)] = px;
    z[size_t(u)] = pz;
    heading[size_t(u)] = h;
    speed[size_t(u)] = v;
}

void World::separate() {
    const int n = count();
    for (int u = 0; u < n; ++u) {
        if (!alive[size_t(u)]) continue;
        const auto& d = type_of(u);
        if (!(d.flags & ZK_U_MOBILE) || (d.flags & ZK_U_AIR) || jump[size_t(u)].flying) continue;
        float push_x = 0.0f, push_z = 0.0f;
        for_near(x[size_t(u)], z[size_t(u)], d.footprint_radius + SEPARATION_MARGIN, [&](int o) {
            if (o == u || !alive[size_t(o)]) return;
            const auto& od = type_of(o);
            if ((od.flags & ZK_U_AIR) || jump[size_t(o)].flying) return;
            const float dx = x[size_t(u)] - x[size_t(o)], dz = z[size_t(u)] - z[size_t(o)];
            const float reach = d.footprint_radius + od.footprint_radius;
            const float d2 = dx * dx + dz * dz;
            if (d2 >= reach * reach || d2 < 0.0001f) return;
            const float dist = std::sqrt(d2);
            // Immobile things push fully, mobile neighbours share the overlap.
            const float share = (od.flags & ZK_U_MOBILE) ? SEPARATION_SHARE : 1.0f;
            push_x += dx / dist * (reach - dist) * share;
            push_z += dz / dist * (reach - dist) * share;
        });
        if (push_x == 0.0f && push_z == 0.0f) continue;
        const float nx = std::clamp(x[size_t(u)] + push_x, 0.0f, map->width() - 1.0f);
        const float nz = std::clamp(z[size_t(u)] + push_z, 0.0f, map->depth() - 1.0f);
        if (d.move_class < 0 || map->speed_at(d.move_class, nx, nz) > 0.0f) {
            x[size_t(u)] = nx;
            z[size_t(u)] = nz;
        }
    }
}

void World::move_units() {
    const int n = count();
    for (int u = 0; u < n; ++u)
        if (alive[size_t(u)] && (type_of(u).flags & ZK_U_MOBILE)) move_one(u);
    {
        ZK_PROF(MOVE_SEPARATE);
        separate();
    }
}

}  // namespace zk
