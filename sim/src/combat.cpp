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
constexpr float DOOMED_PRIORITY = 60.0f;  // pending shots already kill it
constexpr float STUN_PRIORITY = 2.0f;
constexpr float UNBUILT_PRIORITY = 15.0f;
constexpr float UNFINISHED_PRIORITY = 3.0f;
constexpr float ARMORED_PRIORITY = 2.0f;
constexpr float CURRENT_TARGET_BONUS = -0.3f;
constexpr float SCORE_OFFSET = 1000.0f;   // keeps every valid score positive
constexpr float STOPPED_SPEED = 0.05f;
constexpr float SEARCH_MARGIN = 80.0f;    // targets near the edge of a grid query
constexpr float BLAST_MARGIN = 80.0f;     // the largest hit radius, near enough
constexpr float SHIELD_SEARCH = 450.0f;   // the largest shield radius, near enough
constexpr float MAX_SLOW = 0.5f;
constexpr float PARA_DECLINE_SECONDS = 40.0f;
constexpr float FORWARD_COS = 0.96f;      // a default tolerance of 3000 heading units
constexpr float FRAMES_PER_SECOND = 30.0f;

float armor_of(const World& w, int u) {
    const auto& d = w.type_of(u);
    // Armored forms close when the unit stops.
    const bool closed = d.armored_mult < 1.0f && w.speed[size_t(u)] < STOPPED_SPEED;
    return closed ? d.armored_mult : 1.0f;
}

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

int flight_frames(const zk_weapon_rec& w, float dist) {
    if (w.kind == ZK_WEAPON_INSTANT || w.speed <= 0.0f) return 0;
    const float v = w.start_speed;
    if (w.accel <= 0.0f || v >= w.speed) return int(std::ceil(dist / w.speed));
    // Missiles accelerate up to their top speed.
    const float t_acc = (w.speed - v) / w.accel;
    const float d_acc = v * t_acc + 0.5f * w.accel * t_acc * t_acc;
    if (dist <= d_acc) return int(std::ceil((std::sqrt(v * v + 2.0f * w.accel * dist) - v) / w.accel));
    return int(std::ceil(t_acc + (dist - d_acc) / w.speed));
}
}  // namespace

bool World::in_range(int u, const zk_weapon_rec& w, int t) const {
    const float dx = x[size_t(t)] - x[size_t(u)], dz = z[size_t(t)] - z[size_t(u)];
    const float dy = (map->height_at(x[size_t(t)], z[size_t(t)]) - map->height_at(x[size_t(u)], z[size_t(u)])) * w.height_mod;
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

float World::target_score(int u, int slot, int t) const {
    const auto& d = type_of(u);
    const auto& w = cat->weapons[size_t(d.mount_weapon[slot])];
    const auto& td = type_of(t);
    if (!(td.category & d.mount_only[slot]) || !in_range(u, w, t)) return -1.0f;
    float p = cat->target_priority(d.mount_weapon[slot], type[size_t(t)]);
    if (td.category & d.mount_bad[slot]) p += BAD_CATEGORY;
    const float v = speed[size_t(t)];
    if (w.vel_threshold > 0.0f && v > w.vel_threshold) p += w.vel_base + v * w.vel_scale;
    if (stunned(t)) p += STUN_PRIORITY;
    if (build[size_t(t)] < 1.0f) p += UNBUILT_PRIORITY * (1.0f - build[size_t(t)]) + UNFINISHED_PRIORITY;
    if (hp[size_t(t)] <= pending[size_t(t)]) p += DOOMED_PRIORITY;
    const float armor = armor_of(*this, t);
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
    for (int u = 0; u < n; ++u) {
        if (!alive[size_t(u)] || !active(u)) continue;
        const auto& d = type_of(u);
        for (int m = 0; m < d.n_mounts; ++m) {
            const auto& w = cat->weapons[size_t(d.mount_weapon[m])];
            int& current = weapon_target[size_t(u) * ZK_MAX_MOUNTS + size_t(m)];
            if (w.kind == ZK_WEAPON_SHIELD || (w.flags & ZK_W_NO_AUTO_TARGET)) {
                current = -1;
                continue;
            }
            const int forced = order[size_t(u)] == ZK_ORDER_ATTACK ? order_target[size_t(u)] : -1;
            if (forced >= 0 && alive[size_t(forced)] && in_range(u, w, forced) && (type_of(forced).category & d.mount_only[m])) {
                current = forced;
                continue;
            }
            float best = 1e30f;
            int chosen = -1;
            for_near(x[size_t(u)], z[size_t(u)], w.range + SEARCH_MARGIN, [&](int t) {
                if (side[size_t(t)] == side[size_t(u)]) return;
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
    const float tx = x[size_t(t)], tz = z[size_t(t)];
    const float dist = std::hypot(tx - x[size_t(u)], tz - z[size_t(u)]);
    const int flight = flight_frames(w, dist);
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
        Shot s{weapon, u, side[size_t(u)], t, ax, az, 0.0f, w.kind == ZK_WEAPON_GUIDED, (td.flags & ZK_U_AIR) != 0};
        if (w.kind == ZK_WEAPON_INSTANT) {
            apply_shot(s);
        } else {
            s.pending = w.damage[td.armor_class] * armor_of(*this, t);
            pending[size_t(t)] += s.pending;
            // To the nearest tick, so landing times aren't biased late.
            const int ahead = std::clamp((frame_offset + flight + ZK_TICK_FRAMES / 2) / ZK_TICK_FRAMES, 1, slots - 1);
            ring[size_t((tick_count + ahead) % slots)].push_back(s);
        }
    }
}

void World::fire_weapons() {
    const int n = count();
    for (int u = 0; u < n; ++u) {
        // A unit killed earlier this tick still fires: within a tick both sides shoot together.
        if (!alive[size_t(u)]) continue;
        const auto& d = type_of(u);
        const bool can_fire = active(u);
        const float rate = 1.0f - slow[size_t(u)];
        for (int m = 0; m < d.n_mounts; ++m) {
            const auto& w = cat->weapons[size_t(d.mount_weapon[m])];
            if (w.kind == ZK_WEAPON_SHIELD) continue;
            const size_t i = size_t(u) * ZK_MAX_MOUNTS + size_t(m);
            for (int f = 0; f < ZK_TICK_FRAMES; ++f) {
                if (cooldown[i] > 0.0f) cooldown[i] -= rate;
                if (!can_fire) continue;
                if (salvo_left[i] > 0) {
                    salvo_timer[i] -= 1.0f;
                    if (salvo_timer[i] <= 0.0f) {
                        fire_one(u, m, f);
                        --salvo_left[i];
                        salvo_timer[i] = w.salvo_delay;
                    }
                } else if (cooldown[i] <= 0.0f && weapon_target[i] >= 0 && alive[size_t(weapon_target[i])]) {
                    if (!in_range(u, w, weapon_target[i])) {
                        weapon_target[i] = -1;
                    } else if (faces(u, w, weapon_target[i])) {
                        fire_one(u, m, f);
                        cooldown[i] = w.reload;
                        salvo_left[i] = int(w.salvo) - 1;
                        salvo_timer[i] = w.salvo_delay;
                    }
                }
            }
        }
    }
}

bool World::absorbed_by_shield(const zk_weapon_rec& w, float px, float pz, const Shot& s) {
    if (!w.intercepted_by) return false;
    const float cost = w.damage[ZK_ARMOR_SHIELD];
    bool done = false;
    for_near(px, pz, SHIELD_SEARCH, [&](int u) {
        if (done || side[size_t(u)] == s.source_side || shield[size_t(u)] < cost || !active(u)) return;
        const auto& d = type_of(u);
        for (int m = 0; m < d.n_mounts; ++m) {
            const auto& sw = cat->weapons[size_t(d.mount_weapon[m])];
            if (sw.kind != ZK_WEAPON_SHIELD || !(sw.shield_intercept & w.intercepted_by)) continue;
            const float r2 = sw.shield_radius * sw.shield_radius;
            const float dx = x[size_t(u)] - px, dz = z[size_t(u)] - pz;
            if (dx * dx + dz * dz > r2) continue;
            // A shooter inside the bubble fires past it.
            const float sx = x[size_t(u)] - x[size_t(s.source)], sz = z[size_t(u)] - z[size_t(s.source)];
            if (sx * sx + sz * sz <= r2) continue;
            shield[size_t(u)] -= cost;
            done = true;
            return;
        }
    });
    return done;
}

void World::damage_unit(int u, const zk_weapon_rec& w, float mult) {
    const auto& d = type_of(u);
    const float amount = w.damage[d.armor_class] * mult * armor_of(*this, u);
    if (w.flags & ZK_W_PARALYZER) {
        // Fraction of max HP, capped by the weapon's stun time (paralyzeOnMaxHealth).
        const float cap = 1.0f + w.para_seconds / PARA_DECLINE_SECONDS;
        para[size_t(u)] = std::max(para[size_t(u)], std::min(cap, para[size_t(u)] + amount / d.hp));
        return;
    }
    if (w.flags & ZK_W_SLOW_ONLY) {
        slow[size_t(u)] = std::min(MAX_SLOW, slow[size_t(u)] + amount * w.slow_factor / d.hp);
        return;
    }
    hp[size_t(u)] -= amount;
    damage_taken[size_t(u)] += amount;
    if (w.burn_frames > 0.0f && !(d.flags & ZK_U_FIREPROOF) && rng.uniform() <= w.burn_chance) {
        const float frames = w.burn_frames * (rng.uniform() * w.burn_random + 1.0f - w.burn_random);
        // A longer fire replaces a shorter one, as in unit_is_on_fire.lua.
        if (frames * w.burn_damage > burn_frames[size_t(u)] * burn_rate[size_t(u)]) {
            burn_frames[size_t(u)] = frames;
            burn_rate[size_t(u)] = w.burn_damage;
        }
    }
    if (w.extra_para > 0.0f) {
        const float cap = 1.0f + std::max(w.para_seconds, 1.0f) / PARA_DECLINE_SECONDS;
        para[size_t(u)] = std::max(para[size_t(u)], std::min(cap, para[size_t(u)] + w.extra_para * mult / d.hp));
    }
    if (w.slow_factor > 0.0f) slow[size_t(u)] = std::min(MAX_SLOW, slow[size_t(u)] + amount * w.slow_factor / d.hp);
}

void World::explode(int weapon, float px, float pz, int source, int source_side, bool air) {
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
        const float dist = std::sqrt(dx * dx + dz * dz);
        const float surface = std::max(0.0f, dist - type_of(u).hit_radius);
        if (surface <= 0.0f && dist < best) {
            best = dist;
            direct = u;
        }
        if (impact_only || surface >= radius) return;
        // Engine falloff (GameHelper.cpp:146-150).
        const float rim = surface * edge;
        const float mod = (edge >= 1.0f || surface < 1.0f) ? 1.0f : (radius + 0.001f - surface) / (radius + 0.001f - rim);
        damage_unit(u, w, mod);
    });
    if (impact_only && direct >= 0) damage_unit(direct, w, 1.0f);
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
                slow[size_t(u)] = std::min(MAX_SLOW, slow[size_t(u)] + amount / type_of(u).hp);
            } else {
                hp[size_t(u)] -= amount;
                damage_taken[size_t(u)] += amount;
            }
        });
        area.damage_per_frame -= w.area_dps / FRAMES_PER_SECOND * w.area_time_fall * frames / w.area_frames;
        area.frames_left -= ZK_TICK_FRAMES;
        if (area.frames_left > 0) areas[kept++] = area;
    }
    areas.resize(kept);
}

void World::apply_shot(const Shot& s) {
    const auto& w = cat->weapons[size_t(s.weapon)];
    float px = s.x, pz = s.z;
    // A guided missile follows its target, and hits it unless it died first.
    if (s.guided && s.target >= 0 && alive[size_t(s.target)]) {
        px = x[size_t(s.target)];
        pz = z[size_t(s.target)];
    }
    if (absorbed_by_shield(w, px, pz, s)) return;
    explode(s.weapon, px, pz, s.source, s.source_side, s.air);
}

void World::resolve_shots() {
    std::vector<Shot>& due = ring[size_t(tick_count % int(ring.size()))];
    for (const Shot& s : due) {
        apply_shot(s);
        pending[size_t(s.target)] -= s.pending;
    }
    due.clear();
}

void World::finish_deaths() {
    const int n = count();
    for (int u = 0; u < n; ++u) {
        if (!alive[size_t(u)] || hp[size_t(u)] > 0.0f) continue;
        alive[size_t(u)] = 0;
        const auto& d = type_of(u);
        // Nanoframes don't explode.  Chains are caught later in this loop or next tick.
        if (d.explode_weapon >= 0 && build[size_t(u)] >= 1.0f && !(d.flags & ZK_U_AIR))
            explode(d.explode_weapon, x[size_t(u)], z[size_t(u)], -1, -1, false);
    }
}

}  // namespace zk
