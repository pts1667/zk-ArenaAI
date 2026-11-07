// What a unit does without being told: a fight order's search and leash, an idle unit's search, hold position and fire states
// (MobileCAI.cpp), then what Zero-K's tactical AI adds to them: skirmishers keeping their distance, raiders running from what they
// can't fight, idle units going back to where they were (unit_tactical_ai.lua, whose tables tools/sim_data.py exports).
//
// Worlds with orders only.  It runs on the 15-frame update where the engine's runs per unit on the same period, and the gadget's every 20.
// A fight order's internal attack, and the one an idle unit gives itself, are a "chase" here rather than a command in a queue.
#include <algorithm>
#include <cmath>

#include "zksim/world.hpp"

namespace zk {

namespace {
constexpr int HOLD_POSITION = 0, MANEUVER = 1, ROAM = 2;
constexpr int HOLD_FIRE = 0, RETURN_FIRE = 1, FIRE_AT_WILL = 2;
constexpr int NO_ENEMY = -2, NEAREST_ENEMY = -1;     // GetUnitOrderState's first result
// gunshipbomb.lua.
constexpr float FRAMES_A_SECOND = 30.0f;
constexpr float DETONATE_SPEED_MIN = 10.0f, DETONATE_SPEED_MAX = 200.0f;    // elmos a second, less one frame's worth: what its range goes by
constexpr float DROP_PUSH = 0.5f;                     // up or down, to land on what it's told to attack
constexpr float DETONATE_LEAD_WITHIN = 600.0f, DETONATE_LEAD_SPEED = 2.8f, DETONATE_LEAD_FRAMES = 25.0f;

// MobileCAI.cpp, MoveType.cpp, CommandAI.cpp
constexpr float FIGHT_SEARCH_PER_STATE = 100.0f;     // times the move state squared, past the unit's range
constexpr float IDLE_SEARCH_PER_STATE = 200.0f;
constexpr float MANEUVER_LEASH = 500.0f;             // times the move state, past the unit's range
constexpr float OFF_THE_LINE = 96.0f;                // further than this and a fight's line starts again from the unit
constexpr float RETURNED = 64.0f;                    // a return to the line is over this close
constexpr int IDLE_ATTACK_FRAMES = 150;              // an idle unit's attack order times out
// unit_tactical_ai.lua and tactical_ai_defs.lua
constexpr float IDLE_SEARCH_RANGE = 500.0f;
constexpr float LEASH_AGGRESS_RANGE = 400.0f;
constexpr float LEASH_ENEMY_RANGE_LEEWAY = 20.0f;
constexpr float DEFAULT_ENEMY_RANGE = 250.0f;        // what an idle unit assumes of what it can't identify
constexpr int AGGRESSIVE_FRAMES = 80;                // a target someone's fighting stays worth fighting
constexpr float AVOID_HEIGHT_DIFF = 25.0f;
constexpr float SLOWER_ENEMY = 0.95f;
constexpr float SLOWER_ENEMY_PREDICTION = 0.8f;
// Unit.cpp: a radar dot wanders inside a ball of this size, moving a little towards a new point every 15 frames.
constexpr float RADAR_ERROR = 96.0f;
constexpr float ERROR_STEP = 15.0f / 256.0f;
constexpr int NANOFRAME_BAIT_COST[4] = {50, 60, 110, 240};   // bait_prevention_defs.lua, by bait level

struct Point {
    float x, z;
};

Point closest_on_line(Point a, Point b, Point p) {
    const float dx = b.x - a.x, dz = b.z - a.z, length = std::hypot(dx, dz);
    if (length < 1e-4f) return a;
    const float along = std::clamp((dx * (p.x - a.x) + dz * (p.z - a.z)) / length, 0.0f, length);
    return {a.x + dx / length * along, a.z + dz / length * along};
}

float dist(Point a, Point b) { return std::hypot(a.x - b.x, a.z - b.z); }

void random_ball(Rng& rng, float& ox, float& oz) {
    float oy;
    do {
        ox = rng.uniform() * 2.0f - 1.0f;
        oy = rng.uniform() * 2.0f - 1.0f;
        oz = rng.uniform() * 2.0f - 1.0f;
    } while (ox * ox + oy * oy + oz * oz > 1.0f);
}
}  // namespace

// ---------- What a side knows ----------

bool World::in_sight(int s, int t) const { return behaviour != ZK_BEHAVIOUR_ORDERS || views[size_t(s)].in_los[size_t(t)]; }

bool World::identified(int s, int t) const {
    if (behaviour != ZK_BEHAVIOUR_ORDERS) return true;
    const SideView& v = views[size_t(s)];
    return v.known[size_t(t)] || (v.remembered[size_t(t)] && (type_of(t).flags & ZK_U_LEAVES_GHOST));
}

// LOS or radar, and not cloaked.  Worlds without fog have only the cloak to go by.
bool World::can_be_aimed_at(int s, int t) const {
    if (cloaked[size_t(t)]) return false;
    if (behaviour != ZK_BEHAVIOUR_ORDERS) return true;
    const SideView& v = views[size_t(s)];
    return v.in_los[size_t(t)] || v.in_radar[size_t(t)];
}

// Unit.cpp GetErrorVector: none in sight, and none for a building that has been seen.
bool World::aim_is_off(int s, int t) const {
    if (behaviour != ZK_BEHAVIOUR_ORDERS) return false;
    const SideView& v = views[size_t(s)];
    return !v.in_los[size_t(t)] && !(v.remembered[size_t(t)] && (type_of(t).flags & ZK_U_LEAVES_GHOST));
}

void World::aim_error(int t, float& ex, float& ez) const {
    ex = ai[size_t(t)].err_x * RADAR_ERROR;
    ez = ai[size_t(t)].err_z * RADAR_ERROR;
}

void World::update_radar_error() {
    const int n = count();
    for (int u = 0; u < n; ++u) {
        if (!alive[size_t(u)]) continue;
        UnitAI& a = ai[size_t(u)];
        float nx, nz;
        random_ball(rng, nx, nz);
        if (!a.err_started) {
            a.err_started = true;
            a.err_x = nx;
            a.err_z = nz;
            continue;
        }
        if (nx * a.err_x + nz * a.err_z < 0.0f) {
            nx = -nx;
            nz = -nz;
        }
        a.err_x += (nx - a.err_x) * ERROR_STEP;
        a.err_z += (nz - a.err_z) * ERROR_STEP;
    }
}

// ---------- Overkill and bait ----------

// unit_bait_prevention.lua: weapons don't pick these for themselves.  An order or a fight's search still can.
bool World::bait_blocks(int u, int t) const {
    const int level = type_of(u).bait_level;
    if (level <= 0) return false;
    const int s = side[size_t(u)];
    if (!identified(s, t)) return level >= 2;
    const auto& td = type_of(t);
    if (td.bait_target > 0 && level >= td.bait_target) return true;
    if (td.bait_target_armor > 0 && level >= td.bait_target_armor) return !in_sight(s, t) || armored_now(t);
    return build[size_t(t)] < 1.0f && td.bait_cost * build[size_t(t)] <= float(NANOFRAME_BAIT_COST[std::min(level, 4) - 1]);
}

bool World::doomed(int t) const {
    const IncomingDamage& inc = incoming[size_t(t)];
    return inc.doomed && frame <= inc.last_frame;
}

// unit_overkill_prevention.lua, which a weapon's script asks just before firing.  True stops the shot and costs no reload.
// It keeps its own account of what's on its way, and it's the script's idea of the flight time, not the real one.
bool World::overkill_blocks(int u, const zk_weapon_rec& w, int t, int flight) {
    if (w.okp_damage <= 0.0f || (behaviour == ZK_BEHAVIOUR_ORDERS && ai[size_t(u)].fire_state != FIRE_AT_WILL)) return false;
    const auto& td = type_of(t);
    const float separation = std::hypot(x[size_t(t)] - x[size_t(u)], z[size_t(t)] - z[size_t(u)]);
    float timeout = w.okp_frames;
    if (w.flags & ZK_W_OKP_FLIGHT) {
        timeout = float(flight);
    } else if (w.okp_full_frames > 0.0f && w.range > 0.0f) {
        // constants.lua's OverkillPreventionCheck, of which N * distMult in a script is the simple case.
        float hit_mod = w.okp_hit_mod;
        if (w.flags & ZK_W_OKP_TARGET_SPEED) hit_mod += speed[size_t(t)] * 30.0f / w.range;
        timeout = std::floor(w.okp_full_frames * std::max(0.0f, std::max(w.okp_min_range, separation) / w.range - hit_mod));
    }
    constexpr float FAST_SPEED = 5.5f;
    if (w.okp_fast_mult > 0.0f && td.speed > FAST_SPEED) timeout *= w.okp_fast_mult;
    const int s = side[size_t(u)];
    const bool sight = in_sight(s, t), known = identified(s, t);
    const bool adds = !(w.flags & ZK_W_OKP_STATIC_ONLY) || (known && !(td.flags & ZK_U_MOBILE));
    float health = 0.0f;
    if (sight) {
        health = hp[size_t(t)] / armor_mult(t);
        if (shield[size_t(t)] > 0.0f) {
            health += shield[size_t(t)];
            for (int m = 0; m < td.n_mounts; ++m) {
                const auto& sw = cat->weapons[size_t(td.mount_weapon[m])];
                if (sw.kind == ZK_WEAPON_SHIELD) health += sw.shield_regen * timeout;
            }
        }
    } else {
        if (w.okp_radar_mult > 0.0f) timeout *= w.okp_radar_mult;
        if (known) {
            health = td.hp / td.armored_mult;
            for (int m = 0; m < td.n_mounts; ++m) health += cat->weapons[size_t(td.mount_weapon[m])].shield_power;
        }
    }
    IncomingDamage& inc = incoming[size_t(t)];
    if (!inc.seen) {
        if (!adds) return false;
        inc.seen = true;
    }
    if (known) {
        size_t kept = 0;
        for (const IncomingShot& shot : inc.shots)
            if (shot.frame >= frame) {
                inc.shots[kept++] = shot;
                health -= shot.damage;
            }
        inc.shots.resize(kept);
    }
    inc.doomed = known && health < 0.0f;
    if (inc.doomed) return true;
    if (adds) {
        // A tick on: its shot lands at the end of the tick it's due in, after that tick's weapons have looked.
        const int arrives = frame + int(timeout) + ZK_TICK_FRAMES;
        inc.shots.push_back({arrives, w.okp_damage});
        inc.last_frame = std::max(inc.last_frame, arrives);
    }
    return false;
}

// ---------- Commands ----------

bool World::can_hit_now(int u, int t) const {
    const auto& d = type_of(u);
    for (int m = 0; m < d.n_mounts; ++m) {
        const auto& w = cat->weapons[size_t(d.mount_weapon[m])];
        if (w.kind == ZK_WEAPON_SHIELD || (w.flags & ZK_W_MANUAL) || !(type_of(t).category & d.mount_only[m])) continue;
        if (in_range(u, w, t)) return true;
    }
    return false;
}

// MobileCAI::IsValidTarget, with no weapon given.
bool World::valid_chase_target(int u, int t) const {
    const auto& d = type_of(u);
    const auto& td = type_of(t);
    if (d.no_chase & td.category) return false;
    if (ai[size_t(u)].move_state == HOLD_POSITION) return can_hit_now(u, t);
    for (int m = 0; m < d.n_mounts; ++m) {
        const auto& w = cat->weapons[size_t(d.mount_weapon[m])];
        if (w.kind != ZK_WEAPON_SHIELD && !(w.flags & ZK_W_MANUAL) && (td.category & d.mount_only[m])) return true;
    }
    return false;
}

// GetClosestValidTarget, around a point that needn't be the unit's.
int World::closest_target(int u, float cx, float cz, float radius) const {
    float best = radius * radius;
    int found = -1;
    const int s = side[size_t(u)];
    for_near(cx, cz, radius, [&](int t) {
        if (side[size_t(t)] == s || !alive[size_t(t)]) return;
        const float dx = x[size_t(t)] - cx, dz = z[size_t(t)] - cz, d2 = dx * dx + dz * dz;
        if (d2 < best && can_be_aimed_at(s, t) && valid_chase_target(u, t)) {
            best = d2;
            found = t;
        }
    });
    return found;
}

int World::nearest_seen_enemy(int u, float radius) const {
    float best = radius * radius;
    int found = -1;
    const int s = side[size_t(u)];
    for_near(x[size_t(u)], z[size_t(u)], radius, [&](int t) {
        if (side[size_t(t)] == s || !alive[size_t(t)]) return;
        const float dx = x[size_t(t)] - x[size_t(u)], dz = z[size_t(t)] - z[size_t(u)], d2 = dx * dx + dz * dz;
        if (d2 < best && can_be_aimed_at(s, t)) {
            best = d2;
            found = t;
        }
    });
    return found;
}

// CommandAI.cpp: a target out of radar is given up after a few updates, unless it's something that doesn't move.
bool World::target_lost(int u, int t) {
    UnitAI& a = ai[size_t(u)];
    constexpr int TARGET_LOST_UPDATES = 4;
    if (!(type_of(t).flags & ZK_U_MOBILE) || can_be_aimed_at(side[size_t(u)], t)) {
        a.lost_timer = TARGET_LOST_UPDATES;
        return false;
    }
    return --a.lost_timer <= 0;
}

void World::chase(int u, int t, int until) {
    UnitAI& a = ai[size_t(u)];
    a.chasing = t;
    a.chase_until = until;
    a.lost_timer = 4;
}

// ExecuteAttack's checks for an attack it gave itself.
bool World::chase_is_over(int u) {
    UnitAI& a = ai[size_t(u)];
    const int t = a.chasing;
    if (!alive[size_t(t)] || (a.chase_until > 0 && frame >= a.chase_until) || target_lost(u, t)) return true;
    const float range = type_of(u).max_range;
    if (a.move_state < ROAM) {
        const Point at{x[size_t(u)], z[size_t(u)]}, end{a.line_end_x, a.line_end_z}, target{x[size_t(t)], z[size_t(t)]};
        const Point closest = closest_on_line({a.line_x, a.line_z}, end, at);
        if (dist(target, closest_on_line(closest, end, target)) > MANEUVER_LEASH * float(a.move_state) + range) return true;
    }
    return a.move_state == HOLD_POSITION && !can_hit_now(u, t);
}

// ExecuteFight: along the line, looking around the point on it nearest the unit.  What it finds it attacks, then comes back.
void World::fight_step(int u) {
    UnitAI& a = ai[size_t(u)];
    const auto& d = type_of(u);
    const Point at{x[size_t(u)], z[size_t(u)]}, end{a.line_end_x, a.line_end_z};
    if (!a.returning) {
        const Point start = closest_on_line({a.line_x, a.line_z}, end, at);
        const bool off = dist(at, start) > OFF_THE_LINE;
        a.line_x = off ? at.x : start.x;
        a.line_z = off ? at.z : start.z;
    }
    const Point start{a.line_x, a.line_z}, on_line = closest_on_line(start, end, at);
    if ((d.flags & ZK_U_ARMED) && a.fire_state >= FIRE_AT_WILL) {
        const float state = float(a.move_state);
        const int enemy = closest_target(u, on_line.x, on_line.z, d.max_range + FIGHT_SEARCH_PER_STATE * state * state);
        if (enemy >= 0) {
            a.returning = true;
            a.return_x = on_line.x;
            a.return_z = on_line.z;
            chase(u, enemy, 0);
            return;
        }
    }
}

// GenerateAttackCmd.
void World::idle_step(int u) {
    UnitAI& a = ai[size_t(u)];
    const auto& d = type_of(u);
    if (!(d.flags & ZK_U_ARMED) || a.fire_state < FIRE_AT_WILL) return;
    int enemy = -1;
    for (int m = 0; m < d.n_mounts && enemy < 0; ++m) {
        const int t = weapon_target[size_t(u) * ZK_MAX_MOUNTS + size_t(m)];
        if (t >= 0 && alive[size_t(t)] && can_be_aimed_at(side[size_t(u)], t) && valid_chase_target(u, t) && !bait_blocks(u, t)) enemy = t;
    }
    if (enemy < 0) {
        const float state = float(a.move_state);
        enemy = closest_target(u, x[size_t(u)], z[size_t(u)], d.max_range + IDLE_SEARCH_PER_STATE * state * state);
        if (enemy >= 0 && bait_blocks(u, enemy)) enemy = -1;
    }
    if (enemy < 0) return;
    a.line_x = a.line_end_x = x[size_t(u)];
    a.line_z = a.line_end_z = z[size_t(u)];
    chase(u, enemy, frame + IDLE_ATTACK_FRAMES);
}

void World::run_commands(int u) {
    UnitAI& a = ai[size_t(u)];
    if (a.chasing >= 0 && chase_is_over(u)) a.chasing = -1;
    int& o = order[size_t(u)];
    if (o == ZK_ORDER_ATTACK) {
        const int t = order_target[size_t(u)];
        if (t < 0 || !alive[size_t(t)] || target_lost(u, t)) o = ZK_ORDER_NONE;
    }
    if (o == ZK_ORDER_FIGHT && a.chasing < 0) fight_step(u);
    const bool empty = o == ZK_ORDER_NONE && a.chasing < 0 && !a.dodging;
    if (empty && !a.queue_was_empty) unit_went_idle(u);
    if (empty) idle_step(u);
    a.queue_was_empty = o == ZK_ORDER_NONE && a.chasing < 0 && !a.dodging;
}

// ---------- The tactical AI ----------

void World::set_dodge(int u, float tx, float tz) {
    UnitAI& a = ai[size_t(u)];
    a.dodging = true;
    a.dodge_x = std::clamp(tx, 0.0f, map->width() - 1.0f);
    a.dodge_z = std::clamp(tz, 0.0f, map->depth() - 1.0f);
}

// AddIdleUnit.
void World::unit_went_idle(int u) {
    UnitAI& a = ai[size_t(u)];
    a.was_idle = true;
    const auto& b = cat->tactics[size_t(type[size_t(u)])];
    if (!(b.flags & ZK_T_HANDLED)) return;
    if (a.idle_want_return && a.has_idle_pos) {
        a.queue_return = true;
        a.set_return = true;
        a.force_return = false;
        return;
    }
    a.idle_x = x[size_t(u)];
    a.idle_z = z[size_t(u)];
    a.has_idle_pos = true;
    a.want_fight_return = false;
    a.idle_want_return = false;
    const int s = side[size_t(u)];
    int near_enemy = nearest_seen_enemy(u, LEASH_AGGRESS_RANGE);
    if (near_enemy >= 0 && views[size_t(s)].known[size_t(near_enemy)] && !stunned(near_enemy)) {
        const float reach = type_of(near_enemy).max_range;
        const float apart = std::hypot(x[size_t(near_enemy)] - x[size_t(u)], z[size_t(near_enemy)] - z[size_t(u)]);
        if (reach > 0.0f && reach + LEASH_ENEMY_RANGE_LEEWAY < apart) near_enemy = -1;      // it can't shoot from there
    }
    a.idle_aggression = near_enemy >= 0;
    if (near_enemy >= 0) ai[size_t(near_enemy)].aggressive_until = frame + AGGRESSIVE_FRAMES;
}

// UpdateIdleAgressionState: from where the unit began, where it is and where the enemy is, it stops running and fights, or goes home.
// True if that made it take up the fight.
bool World::update_idle_aggression(int u, const zk_tactics_rec& b, int enemy, bool known, float default_range, float enemy_dist) {
    UnitAI& a = ai[size_t(u)];
    if (!a.has_idle_pos) return true;
    const float enemy_range = known ? type_of(enemy).max_range : default_range;
    if (!a.idle_aggression && frame <= ai[size_t(enemy)].aggressive_until) {
        a.idle_aggression = true;
        ai[size_t(enemy)].aggressive_until = frame + AGGRESSIVE_FRAMES;
        return true;
    }
    const Point home{a.idle_x, a.idle_z};
    const float my_home = dist(home, {x[size_t(u)], z[size_t(u)]}), enemy_home = dist(home, {x[size_t(enemy)], z[size_t(enemy)]});
    const bool pushing_me = enemy_home < my_home || enemy_home < enemy_range;
    a.want_fight_return = pushing_me;
    const uint8_t relation = cat->relation(type[size_t(u)], type[size_t(enemy)]);
    const bool ignore_close = known && (relation & ZK_R_SKIRM) && !(relation & ZK_R_HUG);
    if ((enemy_dist < enemy_range && !ignore_close) || b.idle_push_dist_sq < my_home * my_home) {
        if (pushing_me || enemy_dist * b.idle_enemy_mult + my_home * b.idle_commit_mult < b.idle_commit_dist) {
            a.idle_aggression = true;
            ai[size_t(enemy)].aggressive_until = frame + AGGRESSIVE_FRAMES;
        } else if (b.idle_chase_leeway < my_home) {
            a.queue_return = true;
            a.set_return = true;
            a.force_return = true;
        }
    }
    return false;
}

// DoSkirmEnemy: backs away from what's nearer than the unit's reach.  True if it has a move to keep, new or old.
bool World::skirm_enemy(int u, const zk_tactics_rec& b, int enemy, bool known, bool idle, bool fight_and_hold, bool hug) {
    UnitAI& a = ai[size_t(u)];
    const size_t iu = size_t(u), ie = size_t(enemy);
    const uint8_t relation = known ? cat->relation(type[iu], type[ie]) : uint8_t(0);
    const bool keep = (b.flags & ZK_T_KEEP_ORDER) != 0;
    const float height_diff = map->height_at(x[ie], z[ie]) - map->height_at(x[iu], z[iu]);
    if ((relation & ZK_R_AVOID_HEIGHT) && std::fabs(height_diff) > AVOID_HEIGHT_DIFF) return keep;
    const float ex = x[ie] - x[iu], ez = z[ie] - z[iu];
    float predict = 1.0f;
    if (speed[ie] < b.my_speed * SLOWER_ENEMY) predict = SLOWER_ENEMY_PREDICTION * speed[ie] / b.my_speed;
    predict *= b.velocity_prediction;
    const float dx = ex + vx[ie] * predict, dz = ez + vz[ie] * predict;
    const float e_dist = std::max(std::hypot(ex, ez), 0.01f);
    const float projection = (ex * dx + ez * dz) / (e_dist * e_dist);
    float predicted = e_dist;
    if (projection > 0.0f) {
        if (projection > 1.0f) {
            if (!(relation & ZK_R_BONUS_RANGE)) predicted *= b.vel_pred_chase >= 0.0f ? (projection - 1.0f) * b.vel_pred_chase + 1.0f : projection;
        } else {
            predicted *= projection;
        }
    } else {
        predicted = 0.0f;
    }
    if (idle) update_idle_aggression(u, b, enemy, known, DEFAULT_ENEMY_RANGE, predicted);
    const auto& d = type_of(u);
    float reach = b.weapon_range;
    if (b.weapon_num < d.n_mounts) {
        const auto& w = cat->weapons[size_t(d.mount_weapon[b.weapon_num])];
        const float dy = height_diff * w.height_mod;
        reach = std::sqrt(std::max(0.0f, w.range * w.range - dy * dy));
    }
    float skirm_range = hug ? b.hug_range : (b.skirm_range_override >= 0.0f ? b.skirm_range_override : reach) - b.skirm_leeway;
    const float reload_left = float(a.ready_frame - frame);
    if (b.reload_skirm_leeway > 0.0f && reload_left > 0.0f) skirm_range += reload_left * b.reload_skirm_leeway;
    const bool keep_order = keep || skirm_range + b.skirm_keep_leeway > predicted;
    if (hug || skirm_range > predicted) {
        if (b.skirm_only_near >= 0.0f && known && type_of(enemy).max_range + b.skirm_only_near < predicted) return keep_order;
        const bool blocks = !hug && !(relation & ZK_R_BONUS_RANGE) && (b.blocked_on_fight || !fight_and_hold) && b.skirm_blocked_frames >= 0.0f;
        // Negative is how long the weapon has been loaded: with a shot ready it doesn't back off from what's coming.
        if (blocks && b.skirm_blocked_frames < -reload_left) {
            if (a.dodging && !keep_order) a.dodging = false;
            return keep_order;
        }
        float wanted = std::min(b.skirm_order_dis, skirm_range - b.stopping_distance - predicted);
        if (b.skirm_order_dis_min >= 0.0f && b.skirm_order_dis_min > wanted) wanted = b.skirm_order_dis_min;
        set_dodge(u, x[iu] - wanted * ex / e_dist, z[iu] - wanted * ez / e_dist);
        return true;
    }
    if (a.dodging && !keep_order) {
        a.dodging = false;
        return true;
    }
    return keep_order;
}

// DoFleeEnemy.  True if it's running.
bool World::flee_enemy(int u, const zk_tactics_rec& b, int enemy, bool known, bool idle) {
    UnitAI& a = ai[size_t(u)];
    const size_t iu = size_t(u), ie = size_t(enemy);
    float enemy_range = b.min_flee_range;
    if (known) enemy_range = std::max(enemy_range, type_of(enemy).perceived_range);
    const float prediction = b.flee_vel_prediction >= 0.0f ? b.flee_vel_prediction : b.velocity_prediction;
    const float ex = x[ie] - x[iu], ez = z[ie] - z[iu];
    const float now = std::hypot(ex, ez), later = std::hypot(ex + vx[ie] * prediction, ez + vz[ie] * prediction);
    const float apart = std::max(std::min(now, later), 0.01f);        // velocity mustn't make it look further off
    if (idle && update_idle_aggression(u, b, enemy, known, b.min_flee_range, apart)) return false;
    if (enemy_range + b.flee_leeway > apart) {
        float step = b.flee_order_dis;
        if (apart + step > b.weapon_range - b.stopping_distance) step = enemy_range + b.flee_distance - apart;
        const float f = step / apart;
        set_dodge(u, x[iu] - ex * f, z[iu] - ez * f);
        return true;
    }
    if (a.dodging) a.dodging = false;
    return false;
}

// DoTacticalAI, less swarming.  True if the unit has a behaviour against this enemy.
bool World::tactics_against(int u, const zk_tactics_rec& b, int enemy, bool known, bool manual_attack, bool have_fight, bool hold, bool idle, bool particular) {
    UnitAI& a = ai[size_t(u)];
    const uint8_t relation = enemy >= 0 ? cat->relation(type[size_t(u)], type[size_t(enemy)]) : uint8_t(0);
    if (known && !have_fight && (relation & ZK_R_FIGHT_ONLY)) return false;
    if (enemy < 0) return false;
    if (idle && !a.idle_aggression && known &&
        (((b.flags & ZK_T_IDLE_FLEE_COMBAT) && (type_of(enemy).flags & ZK_U_ARMED)) || (relation & ZK_R_IDLE_FLEE))) {
        if (!flee_enemy(u, b, enemy, known, idle)) a.dodging = false;
        return true;
    }
    const bool type_skirm = known && (relation & (ZK_R_SKIRM | ZK_R_HUG));
    if (type_skirm || (!known && (b.flags & ZK_T_SKIRM_RADAR)) || (b.flags & ZK_T_SKIRM_EVERYTHING)) {
        const bool hug = particular && (relation & ZK_R_HUG);
        if (!skirm_enemy(u, b, enemy, known, idle, have_fight && hold, hug)) a.dodging = false;
        return true;
    }
    if (b.flags & ZK_T_FLEE_EVERYTHING) {
        if (!flee_enemy(u, b, enemy, known, idle)) a.dodging = false;
        return true;
    }
    if (manual_attack) return false;       // told to attack it, it doesn't run
    if ((known && ((relation & ZK_R_FLEE) || ((b.flags & ZK_T_FLEE_COMBAT) && (type_of(enemy).flags & ZK_U_ARMED)))) || (!known && (b.flags & ZK_T_FLEE_RADAR))) {
        if (!flee_enemy(u, b, enemy, known, idle)) a.dodging = false;
        return true;
    }
    return false;
}

// DoUnitUpdate, with GetUnitOrderState's cases in terms of what the sim keeps.
void World::run_tactics(int u) {
    const size_t i = size_t(u);
    const auto& b = cat->tactics[size_t(type[i])];
    if (!(b.flags & ZK_T_HANDLED)) return;
    UnitAI& a = ai[i];
    const int o = order[i], state = a.move_state;
    const bool hold = state == HOLD_POSITION, middle = state == MANEUVER, roam = state == ROAM;
    const bool ai_off = (b.flags & ZK_T_IDLE_ONLY) || a.tactics_off;
    const int s = side[i];
    if (!(middle && a.was_idle) && ai_off) {
        // Only this: an attack it gave itself gives way to a nearer enemy, or it would chase past good targets.
        a.idle_want_return = false;
        if (a.chasing >= 0) {
            const int nearer = nearest_seen_enemy(u, b.search_range);
            if (nearer >= 0 && nearer != a.chasing) a.chasing = -1;
        }
        return;
    }
    const bool attacks = o == ZK_ORDER_ATTACK;
    const bool has_cmd = o != ZK_ORDER_NONE || a.chasing >= 0;
    int enemy = NO_ENEMY, auto_target = -1;
    bool have_fight = false;
    if (!has_cmd) {
        enemy = !a.dodging && hold ? NO_ENEMY : NEAREST_ENEMY;
    } else if (a.chasing >= 0) {
        if (o == ZK_ORDER_FIGHT) {
            enemy = NEAREST_ENEMY;
            have_fight = true;
            auto_target = a.chasing;
        } else if (!hold) {
            enemy = NEAREST_ENEMY;
            auto_target = a.chasing;
        } else if (a.dodging) {
            enemy = NEAREST_ENEMY;
        }
    } else if (o == ZK_ORDER_FIGHT) {
        enemy = NEAREST_ENEMY;
        have_fight = true;
    } else if (attacks) {
        enemy = !hold ? order_target[i] : (a.dodging ? NEAREST_ENEMY : NO_ENEMY);
    }
    bool idle_attack = middle && ((!has_cmd && !a.dodging) || (auto_target >= 0 && !have_fight));
    if (a.was_idle && have_fight && !idle_attack && a.has_fight_return) {
        if (goal_x[i] == a.fight_return_x && goal_z[i] == a.fight_return_z && !hold)
            idle_attack = true;
        else
            a.has_fight_return = false;
    }
    a.idle_want_return = a.was_idle && ((a.idle_want_return && (enemy == NEAREST_ENEMY || a.dodging) && !have_fight) || idle_attack);
    bool found = false;
    if (enemy != NO_ENEMY) {
        const bool particular = enemy != NEAREST_ENEMY || auto_target >= 0;
        const bool nearest = enemy == NEAREST_ENEMY;
        if (nearest) enemy = nearest_seen_enemy(u, has_cmd || a.dodging ? b.search_range : IDLE_SEARCH_RANGE);
        // A stunned enemy tells it nothing (GetUnitVisibleInformation).
        const auto knows = [&](int e) { return !stunned(e) && (views[size_t(s)].known[size_t(e)] != 0); };
        const bool manual = attacks && a.chasing < 0;
        if (!ai_off) {
            found = enemy >= 0 && tactics_against(u, b, enemy, knows(enemy), manual, have_fight, hold, a.idle_want_return, particular);
            if (auto_target >= 0 && !found) found = tactics_against(u, b, auto_target, knows(auto_target), manual, have_fight, hold, a.idle_want_return, particular);
            if (!found) a.dodging = false;
        }
        // Always the nearest: what fight or idleness goes for next is by its own search.
        if (auto_target >= 0 && enemy >= 0 && auto_target != enemy) a.chasing = -1;
        if (enemy >= 0 && a.idle_want_return && !found) {
            const float apart = std::hypot(x[size_t(enemy)] - x[i], z[size_t(enemy)] - z[i]);
            update_idle_aggression(u, b, enemy, knows(enemy), DEFAULT_ENEMY_RANGE, apart);
        }
    } else {
        a.dodging = false;
    }
    const bool empty_now = order[i] == ZK_ORDER_NONE && a.chasing < 0;
    if (a.queue_return || (empty_now && !a.dodging && a.set_return && a.has_idle_pos)) {
        a.queue_return = false;
        a.idle_want_return = false;
        if (roam) {
            a.set_return = false;
        } else if ((found || !empty_now || hold) && !a.force_return) {
            a.idle_want_return = true;      // for the next time it's idle
        } else {
            // In its own right rather than as a player's order: nothing else looks at this but the unit.
            order[i] = a.want_fight_return ? ZK_ORDER_FIGHT : ZK_ORDER_MOVE;
            goal_x[i] = a.idle_x;
            goal_z[i] = a.idle_z;
            a.chasing = -1;
            a.dodging = false;
            a.returning = false;
            a.line_x = x[i];
            a.line_z = z[i];
            a.line_end_x = a.idle_x;
            a.line_end_z = a.idle_z;
            a.set_return = false;
            a.force_return = false;
            if (a.want_fight_return) {
                a.has_fight_return = true;
                a.fight_return_x = a.idle_x;
                a.fight_return_z = a.idle_z;
            }
        }
    }
}

void World::update_unit_ai() {
    update_radar_error();
    const int n = count();
    for (int u = 0; u < n; ++u) {
        if (!alive[size_t(u)] || !active(u) || !(type_of(u).flags & ZK_U_MOBILE)) continue;
        run_commands(u);
        run_tactics(u);
    }
}

// ---------- Bombs ----------

// CUnit::SlowUpdateKamikaze.  A bomb has no weapon, so MobileCAI never picks it a target: it goes where it's told.  It defaults to hold fire, and then
// goes off only near what it's told to attack.  On fire at will it also goes off for any enemy in reach, in sight unless the def says.  The engine works
// out whether that enemy is coming closer, but only hands it to AllowUnitKamikaze, which Zero-K doesn't use, and with no one asked the answer is yes: a
// game's Limpet went off for a Knight standing still and for one walking away.
// Fliers don't set it off: the engine measures in 3D, and they're mostly too high up to be within reach.
bool World::kamikaze_due(int u) const {
    const auto& d = type_of(u);
    const UnitAI& a = ai[size_t(u)];
    const float reach = d.kamikaze_distance;
    const float px = x[size_t(u)], pz = z[size_t(u)], up = jump_altitude(u);
    const auto within = [&](int t, float r) {
        const float dx = x[size_t(t)] - px, dz = z[size_t(t)] - pz, dy = jump_altitude(t) - up;
        return dx * dx + dz * dz + dy * dy < r * r;
    };
    if (order[size_t(u)] == ZK_ORDER_ATTACK) {
        const int t = order_target[size_t(u)];
        // gunshipbomb.lua: wanting to fire is going off (BlockShot), and AimFromWeapon has its range go with its speed.  Only a target it's been given
        // gets that far: it finds its own within the 10 elmos of the def's range, so a game's Blastwing on fire at will passed one at 120 without going off.
        const float aimed = std::clamp(speed[size_t(u)] * FRAMES_A_SECOND - FRAMES_A_SECOND, DETONATE_SPEED_MIN, DETONATE_SPEED_MAX) / DETONATE_SPEED_MAX * d.detonate_reach;
        if (t >= 0 && alive[size_t(t)] && within(t, std::max(reach, aimed))) return true;
    }
    if (a.fire_state < FIRE_AT_WILL) return false;
    bool due = false;
    const int own = side[size_t(u)];
    for_near(px, pz, reach, [&](int t) {
        if (due || !alive[size_t(t)] || side[size_t(t)] == own || (type_of(t).flags & ZK_U_AIR)) return;
        if (!within(t, reach)) return;
        due = (d.flags & ZK_U_KAMIKAZE_BLIND) || in_sight(own, t);
    });
    return due;
}

// recentDamage += 2 * maxHealth, in ForcedKillUnit: going off leaves a heap, not a wreck.  What it explodes with is its death explosion, the same in every bomb's def.
void World::self_destruct(int u) {
    recent_damage[size_t(u)] += 2.0f * type_of(u).hp;
    hp[size_t(u)] = 0.0f;
}

void World::update_kamikaze() {
    const int n = count();
    for (int u = 0; u < n; ++u) {
        if (alive[size_t(u)] && type_of(u).kamikaze_distance > 0.0f && active(u) && kamikaze_due(u)) self_destruct(u);
    }
}

// A weapon looks at its target each frame, and in the time between slow updates a Blastwing has gone 100 elmos.
void World::update_droppers() {
    for (int u : droppers)
        if (alive[size_t(u)] && active(u) && kamikaze_due(u)) self_destruct(u);
}

// gunshipbomb.lua ThrowBomb.  The bomb keeps the Blastwing's speed.  Told to attack and not stunned, it adds a little up or down to land on the target, which
// it does if that's within what that can do, so it lands early or late rather than off to the side.
void World::drop_bomb(int u) {
    const size_t i = size_t(u);
    const auto& d = type_of(u);
    const float g = d.drop_gravity, push = DROP_PUSH;
    const float root = std::sqrt(push * push + 2.0f * g * d.drop_height);
    // What's sitting on the ground has nowhere for it to fall.
    float frames = want_cloak[i] ? 1.0f : std::sqrt(2.0f * d.drop_height / g);
    if (!want_cloak[i] && !stunned(u) && order[i] == ZK_ORDER_ATTACK && order_target[i] >= 0 && alive[size_t(order_target[i])]) {
        const int t = order_target[i];
        const float across = std::hypot(x[size_t(t)] - x[i], z[size_t(t)] - z[i]);
        frames = std::clamp(across / std::max(0.1f, std::hypot(vx[i], vz[i])), (root - push) / g, (root + push) / g);
    }
    const float px = std::clamp(x[i] + vx[i] * frames, 0.0f, map->width() - 1.0f), pz = std::clamp(z[i] + vz[i] * frames, 0.0f, map->depth() - 1.0f);
    const int slots = int(ring.size());
    const int ahead = std::clamp(int(frames + float(ZK_TICK_FRAMES) * 0.5f) / ZK_TICK_FRAMES, 1, slots - 1);
    ring[size_t((tick_count + ahead) % slots)].push_back(Shot{d.drop_weapon, u, side[i], -1, px, pz, false, false, 1.0f, x[i], z[i], false});
}

void World::forget_commands(int u) {
    UnitAI& a = ai[size_t(u)];
    a.chasing = -1;
    a.returning = false;
    a.dodging = false;
    a.was_idle = false;
    a.queue_was_empty = false;
    a.idle_want_return = false;
    a.has_fight_return = false;
    a.lost_timer = 4;
    a.line_x = x[size_t(u)];
    a.line_z = z[size_t(u)];
    a.line_end_x = goal_x[size_t(u)];
    a.line_end_z = goal_z[size_t(u)];
}

// ---------- Walking ----------

// ExecuteObjectAttack: in range it stands, unless the target's getting away, when it follows just faster than that.
bool World::walk_to_target(int u, int t, float& gx, float& gz) {
    UnitAI& a = ai[size_t(u)];
    const size_t iu = size_t(u), it = size_t(t);
    if (can_hit_now(u, t)) {
        if (a.move_state == HOLD_POSITION) return false;
        constexpr float FOLLOW_FRAMES = 80.0f, FOLLOW_SPEED = 1.1f, MOVING = 0.05f;
        if (speed[it] > MOVING && (x[iu] - x[it]) * vx[it] + (z[iu] - z[it]) * vz[it] < 0.0f) {
            gx = x[iu] + vx[it] * FOLLOW_FRAMES;
            gz = z[iu] + vz[it] * FOLLOW_FRAMES;
            a.speed_limit = speed[it] * FOLLOW_SPEED;
            return true;
        }
        return false;
    }
    gx = x[it];
    gz = z[it];
    // gunshipbomb.lua: close to it, Blastwing makes for where it will be.
    if (type_of(u).detonate_reach > 0.0f && std::hypot(gx - x[iu], gz - z[iu]) < DETONATE_LEAD_WITHIN) {
        const float slower = speed[it] > DETONATE_LEAD_SPEED ? DETONATE_LEAD_SPEED / speed[it] : 1.0f;
        gx = std::clamp(gx + vx[it] * slower * DETONATE_LEAD_FRAMES, 0.0f, map->width() - 1.0f);
        gz = std::clamp(gz + vz[it] * slower * DETONATE_LEAD_FRAMES, 0.0f, map->depth() - 1.0f);
    }
    if (aim_is_off(side[iu], t)) {
        float ex, ez;
        aim_error(t, ex, ez);
        gx += ex;
        gz += ez;
    }
    return true;
}

}  // namespace zk
