// Targeting, firing, scheduled damage, blasts, shields and deaths.
//
// Target order is Zero-K's (unit_target_priority.lua): a table per weapon and unit, plus status
// terms; lower is preferred and distance doesn't enter.  There are no projectiles: a shot's damage
// is scheduled for its flight time and lands where it was aimed.
#include <algorithm>
#include <cmath>

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
}  // namespace

float World::armor_mult(int u) const {
    const auto& d = type_of(u);
    // A solar's armour comes with being hit, not with standing still as it always does.
    if (d.close_frames > 0.0f) return armoured_for_hit(u) ? d.armored_mult : 1.0f;
    // Armored forms close when the unit stops.
    const bool closed = d.armored_mult < 1.0f && speed[size_t(u)] < STOPPED_SPEED;
    return closed ? d.armored_mult : 1.0f;
}

bool World::in_range(int u, const zk_weapon_rec& w, int t) const {
    const float dx = x[size_t(t)] - x[size_t(u)], dz = z[size_t(t)] - z[size_t(u)];
    // A jumper in the air may be out of reach of what's right under it: a game's Jack didn't get to use its spike.
    const float dy = (map->height_at(x[size_t(t)], z[size_t(t)]) + jump_altitude(t) - map->height_at(x[size_t(u)], z[size_t(u)]) - jump_altitude(u)) * w.height_mod;
    const float r2 = w.range * w.range - dy * dy;
    return r2 > 0.0f && dx * dx + dz * dz <= r2;
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
    if ((td.flags & ZK_U_NEUTRAL) || !(td.category & d.mount_only[slot]) || !in_range(u, w, t) || !can_be_aimed_at(s, t) || bait_blocks(u, t)) return -1.0f;
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
    const int n = count();
    std::fill(stun_attackers.begin(), stun_attackers.end(), 0);
    for (int u = 0; u < n; ++u) {
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
                current = t >= 0 && alive[size_t(t)] && in_range(u, w, t) && (type_of(t).category & d.mount_only[m]) ? t : -1;
                continue;
            }
            if (w.kind == ZK_WEAPON_SHIELD || (w.flags & ZK_W_NO_AUTO_TARGET)) {
                current = -1;
                continue;
            }
            // What the unit is attacking goes to its weapons, as long as they can reach it (Weapon.cpp SlowUpdate).
            const UnitAI& a = ai[size_t(u)];
            const int forced = order[size_t(u)] == ZK_ORDER_ATTACK ? order_target[size_t(u)] : a.chasing;
            if (forced >= 0 && alive[size_t(forced)] && in_range(u, w, forced) && (type_of(forced).category & d.mount_only[m]) && can_be_aimed_at(side[size_t(u)], forced)) {
                current = forced;
                continue;
            }
            if (behaviour == ZK_BEHAVIOUR_ORDERS && a.fire_state < 2) {
                // Nothing of its own choosing: on return fire, only what hit it last.
                const int back = a.fire_state == 1 ? a.last_attacker : -1;
                current = back >= 0 && alive[size_t(back)] && in_range(u, w, back) && (type_of(back).category & d.mount_only[m]) && can_be_aimed_at(side[size_t(u)], back) ? back : -1;
                continue;
            }
            float best = 1e30f;
            int chosen = -1;
            for_side(1 - side[size_t(u)], x[size_t(u)], z[size_t(u)], w.range + SEARCH_MARGIN, [&](int t) {
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
    const float dist = std::hypot(tx - x[size_t(u)], tz - z[size_t(u)]);
    const int flight = flight_frames(w, dist);
    // Measured from muzzle to where it hits, not centre to centre (the battery's Scorcher duels).
    const float power = range_falloff(w, std::max(0.0f, dist - d.hit_radius - type_of(t).hit_radius));
    const float aim_error = speed[size_t(u)] > STOPPED_SPEED && w.moving_accuracy > 0.0f ? w.moving_accuracy : w.accuracy;
    const float spread = (w.spray + aim_error) * dist;
    const auto& td = type_of(t);
    const int slots = int(ring.size());
    for (int p = 0; p < int(w.projectiles); ++p) {
        // Weapons lead with the target's velocity, then add their random error.
        float ax = tx, az = tz;
        if (w.kind == ZK_WEAPON_PROJECTILE) {
            ax += vx[size_t(t)] * float(flight);
            az += vz[size_t(t)] * float(flight);
        }
        if (spread > 0.0f && dist > 0.0f) {
            // Sideways only: a shot off along its line flies on until it meets something.
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
        Shot s{weapon, u, side[size_t(u)], t, ax, az, w.kind == ZK_WEAPON_GUIDED, (td.flags & ZK_U_AIR) != 0, power, x[size_t(u)], z[size_t(u)], blip};
        if (w.kind == ZK_WEAPON_INSTANT) {
            apply_shot(s);
        } else {
            // To the nearest tick, so landing times aren't biased late.
            const int ahead = std::clamp((frame_offset + flight + ZK_TICK_FRAMES / 2) / ZK_TICK_FRAMES, 1, slots - 1);
            ring[size_t((tick_count + ahead) % slots)].push_back(s);
        }
    }
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
    const int n = count();
    for (int u = 0; u < n; ++u) {
        // A unit killed earlier this tick still fires: within a tick both sides shoot together.
        if (!alive[size_t(u)]) continue;
        const auto& d = type_of(u);
        // Disarm stops the reload as well (unit_attributes.lua reloadMult), and stun doesn't.
        const bool can_fire = active(u) && !disarmed(u) && !power_off(u);
        UnitAI& a = ai[size_t(u)];
        // turretriot.lua: it overheats, and it cools unless stunned or unfinished.
        if (d.heat_per_shot > 0.0f && active(u)) a.heat = std::max(0.0f, a.heat - d.heat_decay * float(ZK_TICK_FRAMES));
        const float heat_rate = 1.0f - a.heat * d.heat_max_slow;
        const float rate = disarmed(u) ? 0.0f : (1.0f - slow_now(u)) * heat_rate;
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
                        if (!blocked && pay_for_shot(u, w)) fire_one(u, m, f);
                        --salvo_left[i];
                        salvo_timer[i] = w.salvo_delay;
                    }
                } else if (cooldown[i] <= 0.0f && weapon_target[i] >= 0 && alive[size_t(weapon_target[i])]) {
                    if (!in_range(u, w, weapon_target[i])) {
                        weapon_target[i] = -1;
                    } else if (faces(u, w, weapon_target[i])) {
                        const int t = weapon_target[i];
                        const float apart = std::hypot(x[size_t(t)] - x[size_t(u)], z[size_t(t)] - z[size_t(u)]);
                        // Costs no reload, and it looks again when it next picks a target.
                        if (overkill_blocks(u, w, t, flight_frames(w, apart))) {
                            weapon_target[i] = -1;
                            if (a.chasing == t && order[size_t(u)] == ZK_ORDER_NONE) a.chasing = -1;
                            continue;
                        }
                        if (!pay_for_shot(u, w)) continue;
                        fire_one(u, m, f);
                        cooldown[i] = w.reload;
                        if (m == tactics_mount) a.ready_frame = frame + f + int(w.reload);
                        if (d.heat_per_shot > 0.0f) a.heat = std::min(1.0f, a.heat + d.heat_per_shot);
                        salvo_left[i] = int(w.salvo) - 1;
                        salvo_timer[i] = w.salvo_delay;
                        // Firing finishes the order (CommandAI.cpp WeaponFired).
                        if ((w.flags & ZK_W_MANUAL) && order[size_t(u)] == ZK_ORDER_MANUAL_FIRE) order[size_t(u)] = ZK_ORDER_NONE;
                        if (d.flags & ZK_U_MISSILE) launch_missile(u);
                    }
                }
            }
        }
    }
}

void World::damage_unit(int u, const zk_weapon_rec& w, float mult) {
    const auto& d = type_of(u);
    const float amount = w.damage[d.armor_class] * mult * armor_mult(u);
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
    hp[size_t(u)] -= amount;
    damage_taken[size_t(u)] += amount;
    recent_damage[size_t(u)] += amount;
    if (w.burn_frames > 0.0f && !(d.flags & ZK_U_FIREPROOF) && rng.uniform() <= w.burn_chance) {
        const float frames = w.burn_frames * (rng.uniform() * w.burn_random + 1.0f - w.burn_random);
        // A longer fire replaces a shorter one, as in unit_is_on_fire.lua.
        if (frames * w.burn_damage > burn_frames[size_t(u)] * burn_rate[size_t(u)]) {
            burn_frames[size_t(u)] = frames;
            burn_rate[size_t(u)] = w.burn_damage;
        }
    }
    if (w.extra_para > 0.0f)
        para[size_t(u)] = stun_level(para[size_t(u)], w.extra_para * mult / std::max(hp[size_t(u)], 1.0f), seconds, w.overstun_seconds);
    if (w.slow_factor > 0.0f) add_slow(amount * w.slow_factor);
}

// height: where it goes off, for a jumper being hit in the air.  Everything else is on the ground, as far as a flat sim goes.
void World::explode(int weapon, float px, float pz, int source, int source_side, bool air, float power, float height) {
    const auto& w = cat->weapons[size_t(weapon)];
    const float radius = std::max(w.blast_radius, 1.0f);
    const float edge = w.edge;
    const bool impact_only = (w.flags & ZK_W_IMPACT_ONLY) != 0;
    int direct = -1;
    float best = 1e30f;
    for_near(px, pz, radius + BLAST_MARGIN, [&](int u) {
        if (!alive[size_t(u)]) return;
        if (((type_of(u).flags & ZK_U_AIR) != 0) != air) return;  // no height: a shot hits its own layer
        if (u == source && (w.flags & ZK_W_NO_SELF_DAMAGE)) return;
        if ((w.flags & ZK_W_NO_FRIENDLY_FIRE) && side[size_t(u)] == source_side) return;
        const float dx = x[size_t(u)] - px, dz = z[size_t(u)] - pz;
        const float dy = jump_altitude(u) - height;
        const float dist = std::sqrt(dx * dx + dz * dz + dy * dy);
        const float surface = std::max(0.0f, dist - type_of(u).hit_radius);
        if (surface <= 0.0f && dist < best) {
            best = dist;
            direct = u;
        }
        if (impact_only || surface >= radius) return;
        // Engine falloff (GameHelper.cpp:146-150).
        const float rim = surface * edge;
        const float mod = (edge >= 1.0f || surface < 1.0f) ? 1.0f : (radius + 0.001f - surface) / (radius + 0.001f - rim);
        damage_unit(u, w, mod * power);
        if (w.impulse != 0.0f && source >= 0) push_from(u, source, w.impulse);
        note_attacker(u, source, source_side);
    });
    if (impact_only && direct >= 0) {
        damage_unit(direct, w, power);
        if (w.impulse != 0.0f && source >= 0) push_from(direct, source, w.impulse);
        note_attacker(direct, source, source_side);
    }
    if (!air && !impact_only && hurts_health(w)) hurt_wrecks(w, px, pz, power);
    if (w.area_dps > 0.0f) areas.push_back({&w, px, pz, w.area_dps / FRAMES_PER_SECOND, int(w.area_frames)});
}

void World::burn_areas() {
    // One step per tick where the gadget has two-frame steps: the same damage, on a coarser clock.
    const float frames = float(ZK_TICK_FRAMES);
    size_t kept = 0;
    for (size_t i = 0; i < areas.size(); ++i) {
        FireArea area = areas[i];
        const auto& w = *area.weapon;
        for_near(area.x, area.z, w.area_radius, [&](int u) {
            if (!alive[size_t(u)] || (type_of(u).flags & ZK_U_AIR)) return;
            const float dist = std::hypot(x[size_t(u)] - area.x, z[size_t(u)] - area.z);
            if (dist > w.area_radius) return;
            float amount = area.damage_per_frame;
            if (w.area_range_fall != 0.0f && dist > w.area_plateau)
                amount -= amount * w.area_range_fall * (dist - w.area_plateau) / (w.area_radius - w.area_plateau);
            if (w.area_plateau_fall != 0.0f && dist < w.area_plateau)
                amount -= amount * w.area_plateau_fall * (w.area_plateau - dist) / w.area_plateau;
            amount *= frames;
            if (w.flags & ZK_W_AREA_SLOWS) {
                slow[size_t(u)] = std::max(slow[size_t(u)], std::min(MAX_SLOW, slow[size_t(u)] + amount / std::max(hp[size_t(u)], 1.0f)));
            } else {
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
    const auto& w = cat->weapons[size_t(s.weapon)];
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
    float left = 1.0f;
    while (left > 0.0f) {
        float hit_x, hit_z;
        const float taken = shield_takes(w, s, px, pz, left, hit_x, hit_z);
        if (taken <= 0.0f) break;
        // It explodes at the surface.  There's no unit for an impact-only weapon to have hit (GameHelper.cpp).
        if (!(w.flags & ZK_W_IMPACT_ONLY)) explode(s.weapon, hit_x, hit_z, s.source, s.source_side, s.air, s.power * taken, s.target >= 0 ? jump_altitude(s.target) : 0.0f);
        left -= taken;
    }
    if (left <= 0.0f) return;
    if (w.flags & ZK_W_PIERCES) pierce(s, px, pz, s.power * left);
    else explode(s.weapon, px, pz, s.source, s.source_side, s.air, s.power * left);
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
    for_side(1 - s.source_side, ox + ux * reach * 0.5f, oz + uz * reach * 0.5f, reach * 0.5f + BLAST_MARGIN, [&](int u) {
        if (!alive[size_t(u)] || u == s.source) return;
        if (((type_of(u).flags & ZK_U_AIR) != 0) != s.air) return;
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
    const int n = count();
    for (int u = 0; u < n; ++u) {
        if (!alive[size_t(u)] || hp[size_t(u)] > 0.0f) continue;
        alive[size_t(u)] = 0;
        const auto& d = type_of(u);
        // unit_nanoframe_death.lua: a frame explodes only if it's nearly done.  Chains are caught later in this loop or next tick.
        // No one to return fire at, but a side for no friendly fire.  Before its remains, as in ForcedKillUnit, so a blast doesn't eat its own heap.
        if (d.explode_weapon >= 0 && build[size_t(u)] > NANOFRAME_EXPLODES_ABOVE && !(d.flags & ZK_U_AIR))
            explode(d.explode_weapon, x[size_t(u)], z[size_t(u)], -1, side[size_t(u)], false);
        if (d.drop_weapon >= 0 && build[size_t(u)] >= 1.0f) drop_bomb(u);
        leave_remains(u);
        if (d.missile_pads > 0) destroy_missiles_of(u);
    }
}

}  // namespace zk
