// Targeting, firing, shots in flight, blasts, shields and deaths.
//
// Target order is Zero-K's (unit_target_priority.lua): a table per weapon and unit, plus status
// terms; lower is preferred and distance doesn't enter.
//
// Projectiles.  A shell, a bolt or a rocket is aimed where the engine would aim it (Weapon.cpp GetLeadVec: ahead of a target by
// what it moves in the flight time the weapon reckons with), flies there at its own speed, and goes off where it comes down or
// in the first enemy it meets, on whoever is there then.  Over the ground its path is a straight line, or a pursuit for a rocket
// that homes; its height over that line is a shell's arc or a rocket's rise, so that what's fired high goes over what stands in
// between and comes down on what's near where it lands.  What's still scheduled, to land at its aim point when its flight is
// over: what climbs before it turns (a starburst rocket), what's dropped, and what goes through what it hits.  The ground never
// stops a shot, as it doesn't stop a weapon from aiming.
#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>

#include "zksim/world.hpp"

namespace zk {

namespace {
constexpr float BAD_CATEGORY = 1000.0f;   // tried only if nothing better is in range (Weapon.cpp:720-776)
constexpr float DOOMED_PRIORITY = 60.0f;  // overkill prevention expects it to die of what's on its way
constexpr float UNIDENTIFIED_PRIORITY = 25.0f;   // a radar dot it can't tell the type of
constexpr float RADAR_PRIORITY = 1.5f;    // one it can
constexpr float STUN_PRIORITY = 2.0f;
// A disarmed target's status, which counts double as a stunned one's does (unit_target_priority.lua).
constexpr float DISARM_STATUS = 0.3f;
constexpr float DISARM_STATUS_ADD = 0.2f;
constexpr float DISARM_STATUS_FRAMES = 1500.0f;
constexpr float DISARM_DECLINE_FRAMES = 1200.0f;
constexpr int STUN_ATTACKERS_ALLOWED = 3;
constexpr float UNBUILT_PRIORITY = 15.0f;
constexpr float UNFINISHED_PRIORITY = 3.0f;
constexpr float ARMORED_PRIORITY = 2.0f;
constexpr float CURRENT_TARGET_BONUS = -0.3f;
constexpr float SCORE_OFFSET = 1000.0f;   // keeps every valid score positive
constexpr float STOPPED_SPEED = 0.05f;
constexpr float SEARCH_MARGIN = 80.0f;    // targets near the edge of a grid query
constexpr float BLAST_MARGIN = 80.0f;     // the largest hit radius, near enough
constexpr float PARA_DECLINE_SECONDS = 40.0f;
constexpr float FORWARD_COS = 0.96f;      // a default tolerance of 3000 heading units
constexpr float FRAMES_PER_SECOND = 30.0f;
constexpr float NANOFRAME_EXPLODES_ABOVE = 0.8f;
constexpr float RIGHT_ANGLE = 1.5707964f;
constexpr float HOMING_LEAD = 0.7f;       // MissileProjectile.cpp: of what its target moves in the time left, at its top speed
constexpr int BURNBLOW_FRAMES = 25;       // MissileLauncher.cpp: more fuel for a rocket that goes off when it runs out
constexpr float NEWTON_STABLE = 1.0f;     // Weapon.cpp GetAccuratePredictedImpactTime
constexpr float FRAME_ACCURATE = 1.0f;

// One axis of a point in the unit ball, as the engine's random vector.
float ball_axis(Rng& rng) {
    float ox, oy, oz;
    do {
        ox = rng.uniform() * 2.0f - 1.0f;
        oy = rng.uniform() * 2.0f - 1.0f;
        oz = rng.uniform() * 2.0f - 1.0f;
    } while (ox * ox + oy * oy + oz * oz > 1.0f);
    return ox;
}

// Stun and disarm work alike: 1 is the threshold, the level declines over 40 s, and a weapon can't take
// it past its own time, or by overstun past what's left if that is close to it.
float stun_level(float level, float added, float seconds, float overstun) {
    const float left = (level - 1.0f) * PARA_DECLINE_SECONDS;
    const float longest = std::max(seconds, std::min(seconds, left) + overstun);
    return std::max(level, std::min(1.0f + longest / PARA_DECLINE_SECONDS, level + added));
}

// Share of a shot's damage left at a distance (DamageArray.cpp).  Heat rays fade to nothing at range.
float range_falloff(const zk_weapon_rec& w, float dist) {
    if (w.dyn_exp <= 0.0f || w.dyn_range <= 0.0f) return 1.0f;
    return 1.0f - std::pow(std::min(w.dyn_range, dist) / w.dyn_range, w.dyn_exp);
}

bool harmless(const zk_weapon_rec& w) {
    for (float v : w.damage)
        if (v > 0.0f) return false;
    return true;
}

bool hurts_health(const zk_weapon_rec& w) { return !(w.flags & (ZK_W_PARALYZER | ZK_W_SLOW_ONLY | ZK_W_DISARM_ONLY)); }

int flight_frames(const zk_weapon_rec& w, float dist) {
    if (w.kind == ZK_WEAPON_INSTANT || w.speed <= 0.0f) return 0;
    float v = w.start_speed;
    int climb = 0;
    if (w.climb_frames > 0.0f) {
        // Up first, then a straight line from up there, already moving.
        const float top = std::min(w.speed, v + w.accel * w.climb_frames);
        const float height = (v + top) * 0.5f * w.climb_frames;
        climb = int(w.climb_frames);
        dist = std::hypot(dist, height);
        v = top;
    }
    if (w.accel <= 0.0f || v >= w.speed) return climb + int(std::ceil(dist / w.speed));
    // Missiles accelerate up to their top speed.
    const float t_acc = (w.speed - v) / w.accel;
    const float d_acc = v * t_acc + 0.5f * w.accel * t_acc * t_acc;
    if (dist <= d_acc) return climb + int(std::ceil((std::sqrt(v * v + 2.0f * w.accel * dist) - v) / w.accel));
    return climb + int(std::ceil(t_acc + (dist - d_acc) / w.speed));
}

// How far along its path, over the ground, a shot gets down to the ground, which is `under` below what it was aimed at.  Its height
// over that is (1 - d / aimed) (from + slope d): Projectile::height.  One that flies level never does.
float comes_down(float aimed, float from, float slope, float under) {
    if (slope > 0.0f) {
        const float b = slope * aimed - from;
        return (b + std::sqrt(b * b + 4.0f * slope * aimed * (from + under))) / (2.0f * slope);
    }
    if (from > 0.0f) return aimed * (1.0f + under / from);
    return std::numeric_limits<float>::infinity();
}

// How much longer than the straight line a path is that leaves it and comes back to it at this slope: a parabola's length.
float rise_length(float slope) {
    if (slope <= 0.0f) return 1.0f;
    return 0.5f * std::sqrt(1.0f + slope * slope) + std::asinh(slope) / (2.0f * slope);
}

// Whether it flies as a projectile: not what climbs first, a bomb, or what goes on through what it hits.
bool flies(const zk_weapon_rec& w) { return w.speed > 0.0f && w.climb_frames <= 0.0f && !(w.flags & (ZK_W_BOMB | ZK_W_PIERCES)); }
}  // namespace

float World::armor_mult(int u) const {
    const auto& d = type_of(u);
    // A solar's armour comes with being hit, not with standing still as it always does.
    if (d.close_frames > 0.0f) return armoured_for_hit(u) ? d.armored_mult : 1.0f;
    // Armored forms close when the unit stops.
    const bool closed = d.armored_mult < 1.0f && speed[size_t(u)] < STOPPED_SPEED;
    return closed ? d.armored_mult : 1.0f;
}

// Weapon.cpp TestRange, of a target taken to be at a place.
bool World::in_range_at(int u, const zk_weapon_rec& w, int t, float tx, float tz) const {
    const float dx = tx - x[size_t(u)], dz = tz - z[size_t(u)];
    // A jumper in the air may be out of reach of what's right under it: a game's Jack didn't get to use its spike.  So may what flies.
    const float dy = (map->height_at(x[size_t(t)], z[size_t(t)]) + jump_altitude(t) + flight_height(t) - map->height_at(x[size_t(u)], z[size_t(u)]) -
                      jump_altitude(u) - flight_height(u)) * w.height_mod;
    const float r2 = w.range * w.range - dy * dy;
    return r2 > 0.0f && dx * dx + dz * dz <= r2;
}

// Weapon.cpp TryTarget: a weapon's range is tested at where it would aim, ahead of what moves.  So a slow shot is fired at what's
// coming from further off than the weapon reaches, and isn't at what's leaving near the end of its reach.
bool World::can_reach(int u, const zk_weapon_rec& w, int t) const {
    if (w.kind == ZK_WEAPON_INSTANT || (w.flags & (ZK_W_BOMB | ZK_W_LEAD_ADDED))) return in_range(u, w, t);
    float lead_x, lead_z;
    lead_target(u, w, t, x[size_t(t)], z[size_t(t)], 1.0f, lead_x, lead_z);
    return in_range_at(u, w, t, x[size_t(t)] + lead_x, z[size_t(t)] + lead_z);
}

bool World::faces(int u, const zk_weapon_rec& w, int t) const {
    if (w.flags & ZK_W_TURRET) return true;
    const float dx = x[size_t(t)] - x[size_t(u)], dz = z[size_t(t)] - z[size_t(u)];
    const float dist = std::sqrt(dx * dx + dz * dz);
    if (dist < 1.0f) return true;
    return (std::sin(heading[size_t(u)]) * dx + std::cos(heading[size_t(u)]) * dz) / dist >= FORWARD_COS;
}

float World::target_score(int u, int slot, int t) {
    const auto& d = type_of(u);
    const auto& w = cat->weapons[size_t(d.mount_weapon[slot])];
    const auto& td = type_of(t);
    const int s = side[size_t(u)];
    // In range where it is (GameHelper.cpp GenerateWeaponTargets) and where the weapon would aim (Weapon.cpp AutoTarget).
    if ((td.flags & ZK_U_NEUTRAL) || !(td.category & d.mount_only[slot]) || !in_range(u, w, t) || !can_be_aimed_at(s, t) || bait_blocks(u, t)) return -1.0f;
    if (!can_reach(u, w, t)) return -1.0f;
    const bool known = identified(s, t);
    float p = known ? cat->target_priority(d.mount_weapon[slot], type[size_t(t)]) : UNIDENTIFIED_PRIORITY;
    if (td.category & d.mount_bad[slot]) p += BAD_CATEGORY;
    const float v = speed[size_t(t)];
    if (w.vel_threshold > 0.0f && v > w.vel_threshold) p += w.vel_base + v * w.vel_scale;
    if (!in_sight(s, t)) {
        // unit_target_priority.lua: a dot's rating says nothing of its state, and aim that wobbles is worse against what moves.
        const float wobble = w.radar_wobble_penalty > 0.0f && (!known || (td.flags & ZK_U_MOBILE)) ? w.radar_wobble_penalty : w.radar_dot_penalty;
        p += wobble + (known ? RADAR_PRIORITY : 0.0f);
        if (weapon_target[size_t(u) * ZK_MAX_MOUNTS + size_t(slot)] == t) p += CURRENT_TARGET_BONUS;
        return p + SCORE_OFFSET;
    }
    // 1 for stunned, less for a unit that has only lost its weapons.
    const float status = stunned(t) ? 1.0f
                         : disarmed(t) ? DISARM_STATUS + std::min(DISARM_STATUS_ADD, (disarm[size_t(t)] - 1.0f) * DISARM_DECLINE_FRAMES / DISARM_STATUS_FRAMES)
                                       : 0.0f;
    p += status * STUN_PRIORITY;
    if (w.disarm_penalty > 0.0f) {
        // A few may share a target; the rest look for one that can still shoot.
        p += w.disarm_penalty;
        if (build[size_t(t)] >= 1.0f && stun_attackers[size_t(t)] < STUN_ATTACKERS_ALLOWED) {
            ++stun_attackers[size_t(t)];
            p -= status * STUN_PRIORITY;
        } else if (status > w.disarm_time) {
            p += w.disarm_penalty;
        }
    }
    if (build[size_t(t)] < 1.0f) p += UNBUILT_PRIORITY * (1.0f - build[size_t(t)]) + UNFINISHED_PRIORITY;
    if (doomed(t)) p += DOOMED_PRIORITY;
    const float armor = armor_mult(t);
    if (armor < 1.0f) p += ARMORED_PRIORITY;
    const float health = hp[size_t(t)] / armor;
    if (w.alpha_damage > health)
        p += -0.2f + 0.2f * (w.alpha_damage - health) / w.alpha_damage;
    else
        p += hp[size_t(t)] / td.hp;
    if (weapon_target[size_t(u) * ZK_MAX_MOUNTS + size_t(slot)] == t) p += CURRENT_TARGET_BONUS;
    return p + SCORE_OFFSET;
}

void World::pick_targets() {
    // Only what's in the grid is counted, and the grid's units are all here.
    for (const int u : live) stun_attackers[size_t(u)] = 0;
    for (size_t k = 0, n = live.size(); k < n; ++k) {
        const int u = live[k];
        if (!alive[size_t(u)] || !active(u)) continue;
        const auto& d = type_of(u);
        for (int m = 0; m < d.n_mounts; ++m) {
            const auto& w = cat->weapons[size_t(d.mount_weapon[m])];
            int& current = weapon_target[size_t(u) * ZK_MAX_MOUNTS + size_t(m)];
            // The script holds its fire (turretheavy.lua), and throwing has no target to hold (update_throwers).  On means it pushes, and off that it pulls: one or the other is blocked.
            if (power_off(u) || (w.flags & ZK_W_THROWS) || (w.impulse != 0.0f && (w.impulse < 0.0f) != (pulling[size_t(u)] != 0))) {
                current = -1;
                continue;
            }
            // A manual weapon answers only to a manual-fire order, and the others don't take that order's
            // target (Weapon.cpp:1023).
            if (w.flags & ZK_W_MANUAL) {
                const int t = order[size_t(u)] == ZK_ORDER_MANUAL_FIRE ? order_target[size_t(u)] : -1;
                current = t >= 0 && alive[size_t(t)] && can_reach(u, w, t) && (type_of(t).category & d.mount_only[m]) ? t : -1;
                continue;
            }
            if (w.kind == ZK_WEAPON_SHIELD || (w.flags & ZK_W_NO_AUTO_TARGET)) {
                current = -1;
                continue;
            }
            // What the unit is attacking goes to its weapons, as long as they can reach where they'd aim at it (Weapon.cpp SlowUpdate).
            const UnitAI& a = ai[size_t(u)];
            const int forced = order[size_t(u)] == ZK_ORDER_ATTACK ? order_target[size_t(u)] : a.chasing;
            // A bomb goes only on what it's sent at, when it would land on it (fire_weapons).  One that does no damage marks a
            // bomber's target: it starts a run if the unit has one, and otherwise does nothing.
            if (w.flags & ZK_W_BOMB) {
                bool runs = !harmless(w);
                for (int k = 0; k < d.n_mounts; ++k) runs = runs || (cat->weapons[size_t(d.mount_weapon[k])].flags & ZK_W_DROPPED);
                current = runs && forced >= 0 && alive[size_t(forced)] && (type_of(forced).category & d.mount_only[m]) ? forced : -1;
                continue;
            }
            if (forced >= 0 && alive[size_t(forced)] && can_reach(u, w, forced) && (type_of(forced).category & d.mount_only[m]) && can_be_aimed_at(side[size_t(u)], forced)) {
                current = forced;
                continue;
            }
            if (behaviour == ZK_BEHAVIOUR_ORDERS && a.fire_state < 2) {
                // Nothing of its own choosing: on return fire, only what hit it last.
                const int back = a.fire_state == 1 ? a.last_attacker : -1;
                current = back >= 0 && alive[size_t(back)] && can_reach(u, w, back) && (type_of(back).category & d.mount_only[m]) && can_be_aimed_at(side[size_t(u)], back) ? back : -1;
                continue;
            }
            float best = 1e30f;
            int chosen = -1;
            for_enemies(side[size_t(u)], x[size_t(u)], z[size_t(u)], w.range + SEARCH_MARGIN, [&](int t) {
                const float score = target_score(u, m, t);
                if (score >= 0.0f && score < best) {
                    best = score;
                    chosen = t;
                }
            });
            current = chosen;
        }
    }
}

void World::fire_one(int u, int slot, int frame_offset) {
    const auto& d = type_of(u);
    const int weapon = d.mount_weapon[slot];
    const auto& w = cat->weapons[size_t(weapon)];
    const int t = weapon_target[size_t(u) * ZK_MAX_MOUNTS + size_t(slot)];
    if (t < 0 && can_ward(u, slot)) {
        // The rest of a salvo begun at the ground.
        fire_at_point(u, slot, frame_offset);
        return;
    }
    if (t < 0 || !alive[size_t(t)] || hp[size_t(t)] <= 0.0f) return;
    ++ai[size_t(u)].shots_fired;
    // Weapon.cpp: firing shows a unit, unless its def says otherwise.
    if (!(d.flags & ZK_U_FIRES_CLOAKED)) show_for_action(u);
    float tx = x[size_t(t)], tz = z[size_t(t)];
    // Off a radar dot's position by what's wandering at the moment (Unit.cpp).  A guided one follows the dot.
    const bool blip = aim_is_off(side[size_t(u)], t);
    if (blip) {
        float ex, ez;
        aim_error(t, ex, ez);
        tx += ex;
        tz += ez;
    }
    float dist = std::hypot(tx - x[size_t(u)], tz - z[size_t(u)]);
    int flight = flight_frames(w, dist);
    if (w.flags & ZK_W_BOMB) {
        // It lands where the plane's going, steered towards the target by up to an elmo a frame (BombDropper.cpp).
        const float fall = bomb_fall_frames(u, w, t);
        float ix = x[size_t(u)] + vx[size_t(u)] * fall, iz = z[size_t(u)] + vz[size_t(u)] * fall;
        const float cx = tx - ix, cz = tz - iz, off = std::hypot(cx, cz);
        if (off > 0.0f) {
            const float steer = std::min(off, fall);
            ix += cx / off * steer;
            iz += cz / off * steer;
        }
        tx = ix;
        tz = iz;
        dist = std::hypot(tx - x[size_t(u)], tz - z[size_t(u)]);
        flight = int(std::ceil(fall));
    }
    // Measured from muzzle to where it hits, not centre to centre (the battery's Scorcher duels).
    const float power = range_falloff(w, std::max(0.0f, dist - d.hit_radius - type_of(t).hit_radius));
    const float aim_error = speed[size_t(u)] > STOPPED_SPEED && w.moving_accuracy > 0.0f ? w.moving_accuracy : w.accuracy;
    const float spread = (w.spray + aim_error) * dist;
    const auto& td = type_of(t);
    const int slots = int(ring.size());
    // Ahead of the target, as the engine leads, and off by its error against what moves.
    float lead_x = 0.0f, lead_z = 0.0f;
    if (w.kind != ZK_WEAPON_INSTANT && !(w.flags & ZK_W_BOMB)) {
        // predictSpeedMod at no experience: anywhere from none of the target's speed to twice it, less as predictBoost nears 1.
        const float mult = w.predict_boost < 1.0f ? 1.0f + (1.0f - w.predict_boost) * (rng.uniform() * 2.0f - 1.0f) : 1.0f;
        lead_target(u, w, t, tx, tz, mult, lead_x, lead_z);
        if (w.move_error > 0.0f && speed[size_t(t)] > 0.0f) {
            float ex, ey, ez;
            do {
                ex = rng.uniform() * 2.0f - 1.0f;
                ey = rng.uniform() * 2.0f - 1.0f;
                ez = rng.uniform() * 2.0f - 1.0f;
            } while (ex * ex + ey * ey + ez * ez > 1.0f);
            const float off = w.move_error * FRAMES_PER_SECOND * speed[size_t(t)];
            lead_x += ex * off;
            lead_z += ez * off;
        }
    }
    for (int p = 0; p < int(w.projectiles); ++p) {
        float ax = tx + lead_x, az = tz + lead_z;
        if (spread > 0.0f && dist > 0.0f) {
            // Sideways only: a shot off along its line flies on until it meets something.  A shell's error in elevation, which
            // brings it down short or long, isn't taken: against a round target it would also send flat shots over its head.
            const float side_error = ball_axis(rng) * spread;
            ax += (tz - z[size_t(u)]) / dist * side_error;
            az -= (tx - x[size_t(u)]) / dist * side_error;
        }
        if (blip && (w.flags & ZK_W_IMPACT_ONLY)) {
            // What can only hit by touching never gets a moving dot right, however the wobble falls (the user's measure of a Lance).
            const float reach = td.hit_radius + 1.0f, off = std::hypot(ax - x[size_t(t)], az - z[size_t(t)]);
            if (off < reach) {
                const float nx = off > 0.01f ? (ax - x[size_t(t)]) / off : 1.0f, nz = off > 0.01f ? (az - z[size_t(t)]) / off : 0.0f;
                ax = x[size_t(t)] + nx * reach;
                az = z[size_t(t)] + nz * reach;
            }
        }
        Shot s{weapon, u, side[size_t(u)], t, ax, az, w.kind == ZK_WEAPON_GUIDED, airborne(t), power, x[size_t(u)], z[size_t(u)], blip};
        if (w.kind == ZK_WEAPON_INSTANT) {
            apply_shot(s);
        } else if (flies(w)) {
            launch(s, u, w, frame_offset);
        } else {
            // To the nearest tick, so landing times aren't biased late.
            const int ahead = std::clamp((frame_offset + flight + ZK_TICK_FRAMES / 2) / ZK_TICK_FRAMES, 1, slots - 1);
            ring[size_t((tick_count + ahead) % slots)].push_back(s);
        }
    }
}

// Ward fire (unit_ai.cpp run_ward_fire): whether a weapon with no target would fire at the place its unit has named.  One that
// picks its own targets and could pick the shield's unit, with the place in its reach and in front of it.
bool World::can_ward(int u, int slot) const {
    constexpr int NOT_FOR_WARDING = ZK_W_MANUAL | ZK_W_NO_AUTO_TARGET | ZK_W_BOMB | ZK_W_DROPPED | ZK_W_THROWS | ZK_W_STOCKPILE | ZK_W_UNMODELLED;
    const size_t i = size_t(u);
    const UnitAI& a = ai[i];
    if (frame >= a.ward_until || a.ward_enemy < 0 || !alive[size_t(a.ward_enemy)]) return false;
    const auto& d = type_of(u);
    const auto& w = cat->weapons[size_t(d.mount_weapon[slot])];
    if (w.kind == ZK_WEAPON_SHIELD || (w.flags & NOT_FOR_WARDING) || !(type_of(a.ward_enemy).category & d.mount_only[slot])) return false;
    const float dx = a.ward_x - x[i], dz = a.ward_z - z[i], dist = std::hypot(dx, dz);
    if (dist > w.range) return false;
    return (w.flags & ZK_W_TURRET) || dist < 1.0f || (std::sin(heading[i]) * dx + std::cos(heading[i]) * dz) / dist >= FORWARD_COS;
}

// A shot at the place its unit is ward firing at.  It goes off there, or in the first shield or enemy in its way.
void World::fire_at_point(int u, int slot, int frame_offset) {
    const auto& d = type_of(u);
    const int weapon = d.mount_weapon[slot];
    const auto& w = cat->weapons[size_t(weapon)];
    const size_t i = size_t(u);
    const float px = ai[i].ward_x, pz = ai[i].ward_z;
    ++ai[i].shots_fired;
    if (!(d.flags & ZK_U_FIRES_CLOAKED)) show_for_action(u);
    const float dist = std::hypot(px - x[i], pz - z[i]);
    const float power = range_falloff(w, std::max(0.0f, dist - d.hit_radius));
    const float aim_error = speed[i] > STOPPED_SPEED && w.moving_accuracy > 0.0f ? w.moving_accuracy : w.accuracy;
    const float spread = (w.spray + aim_error) * dist;
    for (int p = 0; p < int(w.projectiles); ++p) {
        float ax = px, az = pz;
        if (spread > 0.0f && dist > 0.0f) {
            const float side_error = ball_axis(rng) * spread;
            ax += (pz - z[i]) / dist * side_error;
            az -= (px - x[i]) / dist * side_error;
        }
        const Shot s{weapon, u, side[i], -1, ax, az, false, false, power, x[i], z[i], false};
        if (w.kind == ZK_WEAPON_INSTANT) {
            apply_shot(s);
        } else if (flies(w)) {
            launch(s, u, w, frame_offset);
        } else {
            const int slots = int(ring.size());
            const int ahead = std::clamp((frame_offset + flight_frames(w, dist) + ZK_TICK_FRAMES / 2) / ZK_TICK_FRAMES, 1, slots - 1);
            ring[size_t((tick_count + ahead) % slots)].push_back(s);
        }
    }
}

// Weapon.cpp GetLeadVec: how far ahead of its target a weapon aims.  The target's velocity times the flight time the weapon
// reckons with: distance over its speed, which a weapon with accurate leading refines as the engine does, by the flight of a
// shell under the weapon's gravity, or the map's when it has none of its own, rocket or not.  mult: of the target's speed, what
// the weapon takes it for.
void World::lead_target(int u, const zk_weapon_rec& w, int t, float tx, float tz, float mult, float& lead_x, float& lead_z) const {
    lead_x = lead_z = 0.0f;
    const float tvx = vx[size_t(t)], tvz = vz[size_t(t)];
    if (w.speed <= 0.0f || w.lead_limit == 0.0f || (tvx == 0.0f && tvz == 0.0f)) return;
    const float ox = x[size_t(u)], oz = z[size_t(u)];
    float predict = std::hypot(tx - ox, tz - oz) / w.speed;
    if (w.accurate_leading > 0 && w.gravity > 0.0f) {
        // GetAccuratePredictedImpactTime: Newton's method on the flight time to where the target will be by then, a few steps of it.
        const float gg = w.gravity * w.gravity, ps2 = w.speed * w.speed, arc = (w.flags & ZK_W_HIGH_ARC) ? 1.0f : -1.0f;
        float dx = tx + tvx * mult * predict - ox, dz = tz + tvz * mult * predict - oz, delta = predict;
        for (int i = 0; i < w.accurate_leading && delta >= 1.0f; ++i) {
            const float far = (dx * dx + dz * dz) * gg, can = ps2 * ps2;
            if (far >= can) break;         // out of a shell's reach at that speed: left at the estimate it has
            float t1 = std::sqrt((ps2 + arc * std::sqrt(can - far)) / (0.5f * gg));
            const float slope = (t1 - predict) / delta;
            if (std::fabs(slope + 1e-4f) < NEWTON_STABLE) t1 = predict - (t1 - predict) / (slope - 1.0f);
            if (std::fabs(t1 - predict) < FRAME_ACCURATE) {
                predict = t1;
                break;
            }
            delta = t1 - predict;
            predict = t1;
            dx = tx + tvx * mult * predict - ox;
            dz = tz + tvz * mult * predict - oz;
        }
    }
    lead_x = tvx * predict * mult;
    lead_z = tvz * predict * mult;
    if (w.lead_limit > 0.0f) {
        const float length = std::hypot(lead_x, lead_z);
        if (length > w.lead_limit) {
            lead_x *= w.lead_limit / (length + 0.01f);
            lead_z *= w.lead_limit / (length + 0.01f);
        }
    }
}

// A shot leaves for its aim point (Cannon.cpp, MissileLauncher.cpp, LaserCannon.cpp FireImpl).  With no target, that's a place
// on the ground, or as far over it as its unit's ward fire aims.
void World::launch(const Shot& s, int u, const zk_weapon_rec& w, int frame_offset) {
    const int t = s.target;
    Projectile p{};
    p.shot = s;
    p.x = x[size_t(u)];
    p.z = z[size_t(u)];
    const float dx = s.x - p.x, dz = s.z - p.z;
    p.aimed = std::hypot(dx, dz);
    if (p.aimed < 1.0f) {
        land_shot(s, s.x, s.z, -1, 0.0f);
        return;
    }
    p.dir_x = dx / p.aimed;
    p.dir_z = dz / p.aimed;
    p.base = t >= 0 ? jump_altitude(t) + flight_height(t) : 0.0f;
    p.from_height = jump_altitude(u) + flight_height(u) - p.base;
    p.muzzle = type_of(u).hit_radius;
    p.wait = frame_offset;
    p.fuel = INT_MAX;
    p.speed = w.speed;
    p.flat = 1.0f;
    const bool rocket = (w.flags & ZK_W_ROCKET) != 0, burnblow = (w.flags & ZK_W_BURNBLOW) != 0;
    if (w.arc > 0.0f) {
        // Cannon.cpp's two solutions: the elevation that reaches the aim point at the shell's speed.  One that can't goes at 45 degrees.
        const float low = 0.5f * std::asin(std::min(1.0f, w.arc * p.aimed));
        const float elevation = (w.flags & ZK_W_HIGH_ARC) ? RIGHT_ANGLE - low : low;
        p.slope = std::tan(elevation);
        p.flat = std::cos(elevation);
    } else if (rocket) {
        p.homing = w.kind == ZK_WEAPON_GUIDED && t >= 0;
        p.slope = w.rise;
        p.flat = 1.0f / rise_length(w.rise);
        if (w.start_speed > 0.0f) p.speed = w.start_speed;
        p.fuel = w.flight_frames > 0.0f ? int(w.flight_frames) : int(std::ceil(std::max(p.aimed, w.range) / w.speed)) + (burnblow ? BURNBLOW_FRAMES : 0);
    }
    // The ground is as far under its aim point as the target's middle is over the ground.
    const float under = t >= 0 ? p.base + type_of(t).hit_radius : cat->tactics[size_t(type[size_t(u)])].ward_height;
    // Aimed at the ground itself, that's where it comes down, however level it flies.
    const float down = t < 0 && under <= 0.0f ? p.aimed : comes_down(p.aimed, p.from_height, p.slope, under);
    if (p.homing) {
        p.reach = std::numeric_limits<float>::infinity();
    } else if (burnblow && !rocket) {
        p.reach = p.aimed;
    } else if (w.arc > 0.0f || rocket) {
        p.reach = down;
    } else {
        // A bolt goes on to the end of its range and is gone (LaserCannon.cpp).
        const float range = std::max(p.aimed, w.range);
        p.fades = range < down;
        p.reach = std::min(range, down);
    }
    projectiles.push_back(p);
}

// A tick of one's flight.  False when it's over.
bool World::fly(Projectile& p) {
    Shot& s = p.shot;
    const auto& w = cat->weapons[size_t(s.weapon)];
    const int frames = ZK_TICK_FRAMES - p.wait, t = s.target;
    p.wait = 0;
    const float x0 = p.x, z0 = p.z, y0 = p.height(), flown0 = p.flown;
    const bool rocket = (w.flags & ZK_W_ROCKET) != 0;
    bool spent = false;
    int flew = 0;
    for (; flew < frames && !spent && p.flown < p.reach; ++flew) {
        if (p.speed < w.speed) p.speed = std::min(w.speed, p.speed + w.accel);
        if (p.homing && t >= 0 && alive[size_t(t)]) {
            // MissileProjectile::Update: it turns, by so much a frame, to where its target will be when it has come most of the way.
            const float back = float(frames - 1 - flew);
            float tx = x[size_t(t)] - vx[size_t(t)] * back, tz = z[size_t(t)] - vz[size_t(t)] * back;
            if (s.aimed_at_blip) {
                float ex, ez;
                aim_error(t, ex, ez);
                tx += ex;
                tz += ez;
            }
            s.x = tx;
            s.z = tz;
            const float left = std::hypot(tx - p.x, tz - p.z);
            p.aimed = std::max(p.flown + left, 1.0f);
            const float ahead = (left + 0.1f) / w.speed * HOMING_LEAD;
            float wx = tx + vx[size_t(t)] * ahead - p.x, wz = tz + vz[size_t(t)] * ahead - p.z;
            const float want = std::hypot(wx, wz);
            if (want > 0.0f) {
                wx /= want;
                wz /= want;
                float turn_x = wx - p.dir_x, turn_z = wz - p.dir_z;
                if (turn_x * turn_x + turn_z * turn_z < w.turn_rate * w.turn_rate) {
                    p.dir_x = wx;
                    p.dir_z = wz;
                } else {
                    const float along = turn_x * p.dir_x + turn_z * p.dir_z;
                    turn_x -= p.dir_x * along;
                    turn_z -= p.dir_z * along;
                    const float across = std::hypot(turn_x, turn_z);
                    if (across > 0.0f) {
                        const float nx = p.dir_x + turn_x / across * w.turn_rate, nz = p.dir_z + turn_z / across * w.turn_rate;
                        const float length = std::hypot(nx, nz);
                        p.dir_x = nx / length;
                        p.dir_z = nz / length;
                    }
                }
            }
        } else if (p.homing) {
            // Its target's gone: on to where that was, as any rocket.
            p.homing = false;
            p.aimed = std::max(p.aimed, p.flown + 1.0f);
            p.reach = comes_down(p.aimed, p.from_height, p.slope, p.base + (t >= 0 ? type_of(t).hit_radius : 0.0f));
        }
        const float step = p.speed * p.flat;
        p.x += p.dir_x * step;
        p.z += p.dir_z * step;
        p.flown += step;
        if (rocket) spent = --p.fuel <= 0;
    }
    const bool down = p.flown >= p.reach;
    if (down) {
        const float over = p.flown - p.reach;
        p.x -= p.dir_x * over;
        p.z -= p.dir_z * over;
        p.flown = p.reach;
    }
    const float x1 = p.x, z1 = p.z, y1 = p.height();
    // The first enemy its path goes through, as each of them moved over the same frames.  Not while it's over all their heads.
    int struck = -1;
    float struck_at = 2.0f;
    if (s.air || std::min(y0, y1) + p.base <= BLAST_MARGIN) {
        const float n = float(flew);
        for_enemies(s.source_side, (x0 + x1) * 0.5f, (z0 + z1) * 0.5f, std::hypot(x1 - x0, z1 - z0) * 0.5f + SEARCH_MARGIN + BLAST_MARGIN, [&](int e) {
            const size_t i = size_t(e);
            if (!alive[i] || e == s.source || airborne(e) != s.air) return;
            const float r = type_of(e).hit_radius, up = jump_altitude(e) + flight_height(e) - p.base;
            const float ax = x0 - (x[i] - vx[i] * n), ay = y0 - up, az = z0 - (z[i] - vz[i] * n);
            const float dx = x1 - x[i] - ax, dy = y1 - up - ay, dz = z1 - z[i] - az;
            const float c = ax * ax + ay * ay + az * az - r * r;
            float at = 0.0f;
            if (c > 0.0f) {
                const float a = dx * dx + dy * dy + dz * dz, b = ax * dx + ay * dy + az * dz;
                if (a <= 0.0f || b >= 0.0f) return;
                const float disc = b * b - a * c;
                if (disc < 0.0f) return;
                at = (-b - std::sqrt(disc)) / a;
                if (at > 1.0f) return;
            }
            if (at < struck_at) {
                struck_at = at;
                struck = e;
            }
        });
    }
    if (struck >= 0) {
        const float flown = flown0 + (p.flown - flown0) * struck_at;
        s.power = range_falloff(w, std::max(0.0f, flown - p.muzzle));
        land_shot(s, x0 + (x1 - x0) * struck_at, z0 + (z1 - z0) * struck_at, struck, jump_altitude(struck));
        return false;
    }
    if (!down && !spent) return true;
    s.power = range_falloff(w, std::max(0.0f, p.flown - p.muzzle));
    // On the ground, or in the air where its time was up.  A bolt at the end of its range is gone, if no shield was in its way.
    land_shot(s, p.x, p.z, -1, std::max(0.0f, p.height() + p.base), down && p.fades);
    return false;
}

void World::fly_projectiles() {
    size_t kept = 0;
    for (size_t i = 0; i < projectiles.size(); ++i) {
        Projectile p = projectiles[i];
        if (fly(p)) projectiles[kept++] = p;
    }
    projectiles.resize(kept);
}

bool World::pay_for_shot(int u, const zk_weapon_rec& w) {
    if ((w.flags & ZK_W_STOCKPILE) && stock[size_t(u)] < 1) return false;
    if (shield[size_t(u)] < w.shield_cost) return false;
    if (w.flags & ZK_W_STOCKPILE) --stock[size_t(u)];
    shield[size_t(u)] -= w.shield_cost;
    return true;
}

int World::manual_mount(int u) const {
    const auto& d = type_of(u);
    for (int m = 0; m < d.n_mounts; ++m)
        if (cat->weapons[size_t(d.mount_weapon[m])].flags & ZK_W_MANUAL) return m;
    return -1;
}

void World::fire_weapons() {
    // Within a tick both sides shoot together: who can fire, and how slowed, is as the tick's shooting begins, whoever is stunned,
    // disarmed or slowed by a shot of it, and a unit killed by one still fires.  Else whoever was made first would shoot first.
    // 1: active.  2: able to fire.  Disarm stops the reload as well (unit_attributes.lua reloadMult), and stun doesn't.
    // What flies fires only in the air, and nothing new once its ammo is spent: a salvo it has begun it finishes.
    fire_ok.resize(size_t(count()));
    fire_rate.resize(size_t(count()));
    for (const int u : live) {
        if (!alive[size_t(u)]) continue;
        const auto& d = type_of(u);
        const bool is_active = active(u);
        fire_ok[size_t(u)] = uint8_t(is_active) | uint8_t((is_active && !disarmed(u) && !power_off(u) && (!(d.flags & ZK_U_AIR) || airborne(u))) << 1);
        fire_rate[size_t(u)] = disarmed(u) ? 0.0f : 1.0f - slow_now(u);
    }
    for (size_t k = 0, n = live.size(); k < n; ++k) {
        const int u = live[k];
        if (!alive[size_t(u)]) continue;
        const auto& d = type_of(u);
        const bool can_fire = (fire_ok[size_t(u)] & 2) != 0;
        UnitAI& a = ai[size_t(u)];
        // turretriot.lua: it overheats, and it cools unless stunned or unfinished.
        if (d.heat_per_shot > 0.0f && (fire_ok[size_t(u)] & 1)) a.heat = std::max(0.0f, a.heat - d.heat_decay * float(ZK_TICK_FRAMES));
        const float heat_rate = 1.0f - a.heat * d.heat_max_slow;
        const float rate = fire_rate[size_t(u)] * heat_rate;
        const int tactics_mount = cat->tactics[size_t(type[size_t(u)])].weapon_num;
        for (int m = 0; m < d.n_mounts && alive[size_t(u)]; ++m) {
            const auto& w = cat->weapons[size_t(d.mount_weapon[m])];
            if (w.kind == ZK_WEAPON_SHIELD) continue;
            const size_t i = size_t(u) * ZK_MAX_MOUNTS + size_t(m);
            for (int f = 0; f < ZK_TICK_FRAMES && alive[size_t(u)]; ++f) {
                if (cooldown[i] > 0.0f) cooldown[i] -= rate;
                if (!can_fire) continue;
                if (salvo_left[i] > 0) {
                    salvo_timer[i] -= 1.0f;
                    if (salvo_timer[i] <= 0.0f) {
                        const int t = weapon_target[i];
                        const bool blocked = (w.flags & ZK_W_SCRIPT_BURST) && t >= 0 && alive[size_t(t)] &&
                                             overkill_blocks(u, w, t, flight_frames(w, std::hypot(x[size_t(t)] - x[size_t(u)], z[size_t(t)] - z[size_t(u)])));
                        if (!blocked && pay_for_shot(u, w)) {
                            if (w.flags & ZK_W_DROPPED)
                                drop_shot(u, m, f);
                            else
                                fire_one(u, m, f);
                        }
                        --salvo_left[i];
                        salvo_timer[i] = w.salvo_delay;
                    }
                } else if (cooldown[i] <= 0.0f && !out_of_ammo(u) && weapon_target[i] >= 0 && alive[size_t(weapon_target[i])]) {
                    if (w.flags & ZK_W_BOMB) {
                        if (!bomb_due(u, w, weapon_target[i])) continue;
                    } else if (!can_reach(u, w, weapon_target[i])) {
                        weapon_target[i] = -1;
                        continue;
                    }
                    if ((w.flags & ZK_W_BOMB) || faces(u, w, weapon_target[i])) {
                        const int t = weapon_target[i];
                        const float apart = std::hypot(x[size_t(t)] - x[size_t(u)], z[size_t(t)] - z[size_t(u)]);
                        // Costs no reload, and it looks again when it next picks a target.
                        if (overkill_blocks(u, w, t, flight_frames(w, apart))) {
                            weapon_target[i] = -1;
                            if (a.chasing == t && order[size_t(u)] == ZK_ORDER_NONE) a.chasing = -1;
                            continue;
                        }
                        if (!pay_for_shot(u, w)) continue;
                        const bool marks = (w.flags & ZK_W_BOMB) && harmless(w);
                        if (marks)
                            start_run(u);
                        else
                            fire_one(u, m, f);
                        cooldown[i] = w.reload;
                        if (m == tactics_mount) a.ready_frame = frame + f + int(w.reload);
                        if (d.heat_per_shot > 0.0f) a.heat = std::min(1.0f, a.heat + d.heat_per_shot);
                        salvo_left[i] = marks ? 0 : int(w.salvo) - 1;
                        salvo_timer[i] = w.salvo_delay;
                        if (d.ammo > 0) used_ammo(u);
                        // Firing finishes the order (CommandAI.cpp WeaponFired).
                        if ((w.flags & ZK_W_MANUAL) && order[size_t(u)] == ZK_ORDER_MANUAL_FIRE) order[size_t(u)] = ZK_ORDER_NONE;
                        if (d.flags & ZK_U_MISSILE) launch_missile(u);
                    }
                } else if (cooldown[i] <= 0.0f && weapon_target[i] < 0 && !out_of_ammo(u) && can_ward(u, m)) {
                    // Ward fire: with no target of its own, at the place its unit has named (unit_ai.cpp run_ward_fire).
                    if (!pay_for_shot(u, w)) continue;
                    fire_at_point(u, m, f);
                    cooldown[i] = w.reload;
                    if (m == tactics_mount) a.ready_frame = frame + f + int(w.reload);
                    if (d.heat_per_shot > 0.0f) a.heat = std::min(1.0f, a.heat + d.heat_per_shot);
                    salvo_left[i] = int(w.salvo) - 1;
                    salvo_timer[i] = w.salvo_delay;
                    if (d.ammo > 0) used_ammo(u);
                }
            }
        }
    }
}

// What a team's units do to enemies' units, for the reward (capi.h ZK_STATS): the health a hit is about to take off, as metal.
// dealer: the team whose unit did it, or -1.
void World::count_damage(int victim, float amount, int dealer) {
    const size_t i = size_t(victim);
    const float removed = std::min(amount, hp[i]);
    if (dealer < 0 || !(removed > 0.0f)) return;
    const int owner = team[i];
    if (ally_of[dealer] == ally_of[owner]) return;
    const auto& d = type_of(victim);
    const double metal = double(removed) * (d.cost > 0.0f ? d.cost : d.build_time) / d.hp;
    metal_dealt[dealer] += metal;
    metal_taken[owner] += metal;
}

void World::damage_unit(int u, const zk_weapon_rec& w, float mult, int dealer) {
    const auto& d = type_of(u);
    const float amount = w.damage[d.armor_class] * mult * armor_mult(u) * damage_mult[size_t(u)];
    // energysolar.lua's HitByWeapon: anything that does damage by the book, except a weapon that can't hurt friends.  Each hit starts it over.
    if (d.close_frames > 0.0f && amount > 0.0f && build[size_t(u)] >= 1.0f && !(w.flags & ZK_W_NO_FRIENDLY_FIRE)) hit_closes(u);
    // unit_decloak_damaged.lua: any damage shows a unit, pure status effects included.
    if (amount > 0.0f) show_for_action(u);
    // Stun, disarm and slow all go by current health, so hurt units fall to them sooner.
    const float seconds = std::max(w.para_seconds, 1.0f);
    const auto add_slow = [&](float added) {
        const float cap = MAX_SLOW + w.overslow;
        slow[size_t(u)] = std::max(slow[size_t(u)], std::min(cap, slow[size_t(u)] + added / std::max(hp[size_t(u)], 1.0f)));
    };
    if (w.disarm_mult > 0.0f)
        disarm[size_t(u)] = stun_level(disarm[size_t(u)], amount * w.disarm_mult / std::max(hp[size_t(u)], 1.0f), seconds, w.overstun_seconds);
    if (w.flags & ZK_W_DISARM_ONLY) return;
    if (w.flags & ZK_W_PARALYZER) {
        para[size_t(u)] = stun_level(para[size_t(u)], amount / std::max(hp[size_t(u)], 1.0f), seconds, w.overstun_seconds);
        return;
    }
    if (w.flags & ZK_W_SLOW_ONLY) {
        add_slow(amount * w.slow_factor);
        return;
    }
    count_damage(u, amount, dealer);
    hp[size_t(u)] -= amount;
    damage_taken[size_t(u)] += amount;
    recent_damage[size_t(u)] += amount;
    if (w.burn_frames > 0.0f && !(d.flags & ZK_U_FIREPROOF) && rng.uniform() <= w.burn_chance) {
        const float frames = w.burn_frames * (rng.uniform() * w.burn_random + 1.0f - w.burn_random);
        // A longer fire replaces a shorter one, as in unit_is_on_fire.lua.
        if (frames * w.burn_damage > burn_frames[size_t(u)] * burn_rate[size_t(u)]) {
            burn_frames[size_t(u)] = frames;
            burn_rate[size_t(u)] = w.burn_damage;
            burn_team[size_t(u)] = int8_t(dealer);
        }
    }
    if (w.extra_para > 0.0f)
        para[size_t(u)] = stun_level(para[size_t(u)], w.extra_para * mult / std::max(hp[size_t(u)], 1.0f), seconds, w.overstun_seconds);
    if (w.slow_factor > 0.0f) add_slow(amount * w.slow_factor);
}

// height: where it goes off, for a jumper being hit in the air.  Everything else is on the ground, as far as a flat sim goes.
// struck: the unit a projectile went off in, which gets all of it; without one, a weapon that only hurts what it hits takes whoever's there.
// source_team: whose it is when there's no unit to say (a death's blast).
void World::explode(int weapon, float px, float pz, int source, int source_side, bool air, float power, float height, int struck, int source_team) {
    const auto& w = cat->weapons[size_t(weapon)];
    const int dealer = source >= 0 ? team[size_t(source)] : source_team;
    const float radius = std::max(w.blast_radius, 1.0f);
    const float edge = w.edge;
    const bool impact_only = (w.flags & ZK_W_IMPACT_ONLY) != 0;
    int direct = struck;
    float best = 1e30f;
    for_near(px, pz, radius + BLAST_MARGIN, [&](int u) {
        if (!alive[size_t(u)]) return;
        if (airborne(u) != air) return;  // no height: a shot hits its own layer, and what's landed is on the ground's
        if (u == source && (w.flags & ZK_W_NO_SELF_DAMAGE)) return;
        if ((w.flags & ZK_W_NO_FRIENDLY_FIRE) && side[size_t(u)] == source_side) return;
        const float dx = x[size_t(u)] - px, dz = z[size_t(u)] - pz;
        const float dy = jump_altitude(u) - height;
        const float dist = std::sqrt(dx * dx + dz * dz + dy * dy);
        const float surface = u == struck ? 0.0f : std::max(0.0f, dist - type_of(u).hit_radius);
        if (struck < 0 && surface <= 0.0f && dist < best) {
            best = dist;
            direct = u;
        }
        if (impact_only || surface >= radius) return;
        // Engine falloff (GameHelper.cpp:146-150).
        const float rim = surface * edge;
        const float mod = (edge >= 1.0f || surface < 1.0f) ? 1.0f : (radius + 0.001f - surface) / (radius + 0.001f - rim);
        damage_unit(u, w, mod * power, dealer);
        if (w.impulse != 0.0f && source >= 0) push_from(u, source, w.impulse);
        note_attacker(u, source, source_side);
    });
    if (impact_only && direct >= 0) {
        damage_unit(direct, w, power, dealer);
        if (w.impulse != 0.0f && source >= 0) push_from(direct, source, w.impulse);
        note_attacker(direct, source, source_side);
    }
    if (!air && !impact_only && hurts_health(w)) hurt_wrecks(w, px, pz, power);
    if (w.area_dps > 0.0f) areas.push_back({&w, px, pz, w.area_dps / FRAMES_PER_SECOND, int(w.area_frames), dealer});
}

void World::burn_areas() {
    // One step per tick where the gadget has two-frame steps: the same damage, on a coarser clock.
    const float frames = float(ZK_TICK_FRAMES);
    size_t kept = 0;
    for (size_t i = 0; i < areas.size(); ++i) {
        FireArea area = areas[i];
        const auto& w = *area.weapon;
        for_near(area.x, area.z, w.area_radius, [&](int u) {
            if (!alive[size_t(u)] || airborne(u)) return;
            const float dist = std::hypot(x[size_t(u)] - area.x, z[size_t(u)] - area.z);
            if (dist > w.area_radius) return;
            float amount = area.damage_per_frame;
            if (w.area_range_fall != 0.0f && dist > w.area_plateau)
                amount -= amount * w.area_range_fall * (dist - w.area_plateau) / (w.area_radius - w.area_plateau);
            if (w.area_plateau_fall != 0.0f && dist < w.area_plateau)
                amount -= amount * w.area_plateau_fall * (w.area_plateau - dist) / w.area_plateau;
            amount *= frames * damage_mult[size_t(u)];
            if (w.flags & ZK_W_AREA_SLOWS) {
                slow[size_t(u)] = std::max(slow[size_t(u)], std::min(MAX_SLOW, slow[size_t(u)] + amount / std::max(hp[size_t(u)], 1.0f)));
            } else {
                count_damage(u, amount, area.team);
                hp[size_t(u)] -= amount;
                damage_taken[size_t(u)] += amount;
                recent_damage[size_t(u)] += amount;
            }
        });
        area.damage_per_frame -= w.area_dps / FRAMES_PER_SECOND * w.area_time_fall * frames / w.area_frames;
        area.frames_left -= ZK_TICK_FRAMES;
        if (area.frames_left > 0) areas[kept++] = area;
    }
    areas.resize(kept);
}

// Unit.cpp: for return fire.
void World::note_attacker(int u, int source, int source_side) {
    if (source < 0 || source_side == side[size_t(u)]) return;
    ai[size_t(u)].last_attacker = source;
}

void World::apply_shot(const Shot& s) {
    float px = s.x, pz = s.z;
    // A guided missile follows its target, or the dot radar shows for it, and hits unless it died first.
    if (s.guided && s.target >= 0 && alive[size_t(s.target)]) {
        px = x[size_t(s.target)];
        pz = z[size_t(s.target)];
        if (s.aimed_at_blip) {
            float ex, ez;
            aim_error(s.target, ex, ez);
            px += ex;
            pz += ez;
        }
    }
    land_shot(s, px, pz, -1, 0.0f);
}

// What a shot does where its flight ends: at the first shield on the way there, or else there, unless it fades.
void World::land_shot(const Shot& s, float px, float pz, int struck, float height, bool fades) {
    const auto& w = cat->weapons[size_t(s.weapon)];
    float left = 1.0f;
    while (left > 0.0f) {
        float hit_x, hit_z;
        const float taken = shield_takes(w, s, px, pz, left, hit_x, hit_z);
        if (taken <= 0.0f) break;
        // It explodes at the surface.  There's no unit for an impact-only weapon to have hit (GameHelper.cpp).
        if (!(w.flags & ZK_W_IMPACT_ONLY)) explode(s.weapon, hit_x, hit_z, s.source, s.source_side, s.air, s.power * taken, s.target >= 0 ? jump_altitude(s.target) : 0.0f);
        left -= taken;
    }
    if (left <= 0.0f || fades) return;
    if (w.flags & ZK_W_PIERCES) pierce(s, px, pz, s.power * left);
    else explode(s.weapon, px, pz, s.source, s.source_side, s.air, s.power * left, height, struck);
}

// A noExplode shot (WeaponProjectile.cpp) goes off in every unit it's inside on a frame, and flies on to the end of its range; what it
// misses, it doesn't hurt.  A step of `speed` elmos meets a chord of a unit 1 + chord/speed times on average, which is what a game
// measures (a Pyro's flame: 2 on a Glaive, 3 on a tank), and an average has no favoured distances.  It goes through friends, as a
// shooter here doesn't hold fire for them the way a real one does: that's not in the sim for any weapon.
void World::pierce(const Shot& s, float px, float pz, float power) {
    const auto& w = cat->weapons[size_t(s.weapon)];
    const float ox = s.origin_x, oz = s.origin_z, aimed = std::hypot(px - ox, pz - oz);
    if (aimed < 0.01f) {
        explode(s.weapon, px, pz, s.source, s.source_side, s.air, power);
        return;
    }
    const float ux = (px - ox) / aimed, uz = (pz - oz) / aimed, reach = std::max(w.range, aimed);
    // Up to a jumper it was aimed at, and on.
    const float rise = s.target >= 0 && jump[size_t(s.target)].flying ? jump_altitude(s.target) / aimed : 0.0f;
    std::vector<PierceHit>& hits = pierce_hits;      // nothing below pierces again
    hits.clear();
    for_enemies(s.source_side, ox + ux * reach * 0.5f, oz + uz * reach * 0.5f, reach * 0.5f + BLAST_MARGIN, [&](int u) {
        if (!alive[size_t(u)] || u == s.source) return;
        if (airborne(u) != s.air) return;
        const float r = type_of(u).hit_radius, dx = x[size_t(u)] - ox, dz = z[size_t(u)] - oz;
        const float along = dx * ux + dz * uz;
        if (along < -r || along > reach + r) return;
        const float at = std::clamp(along, 0.0f, reach), off = dx * uz - dz * ux, dy = jump_altitude(u) - rise * at;
        const float left = r * r - off * off - dy * dy;
        if (left <= 0.0f) return;
        hits.push_back({at, (w.flags & ZK_W_ONE_HIT_EACH) || w.speed <= 0.0f ? 1.0f : 1.0f + 2.0f * std::sqrt(left) / w.speed});
    });
    for (const PierceHit& h : hits) explode(s.weapon, ox + ux * h.along, oz + uz * h.along, s.source, s.source_side, s.air, power * h.times, rise * h.along);
}

void World::resolve_shots() {
    std::vector<Shot>& due = ring[size_t(tick_count % int(ring.size()))];
    for (const Shot& s : due) apply_shot(s);
    due.clear();
}

void World::finish_deaths() {
    // Who has died is settled first: what a death's blast kills goes next tick, whichever of the two was made first.
    dying.clear();
    for (const int u : live)
        if (alive[size_t(u)] && hp[size_t(u)] <= 0.0f) dying.push_back(u);
    for (const int u : dying) {
        died(u);
        const auto& d = type_of(u);
        // unit_nanoframe_death.lua: a frame explodes only if it's nearly done.
        // No one to return fire at, but a side for no friendly fire.  Before its remains, as in ForcedKillUnit, so a blast doesn't eat its own heap.
        if (d.explode_weapon >= 0 && build[size_t(u)] > NANOFRAME_EXPLODES_ABOVE && !(d.flags & ZK_U_AIR))
            explode(d.explode_weapon, x[size_t(u)], z[size_t(u)], -1, side[size_t(u)], false, 1.0f, 0.0f, -1, team[size_t(u)]);
        if (d.drop_weapon >= 0 && build[size_t(u)] >= 1.0f) drop_bomb(u);
        leave_remains(u);
        if (d.missile_pads > 0) destroy_missiles_of(u);
    }
}

}  // namespace zk
