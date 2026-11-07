// What a few units can do to other units: push and pull them, throw them, bring them over through a beacon, and what a silo holds.
//
// Pushing: weapon_impulse.lua, then the engine's skid (GroundMoveType.cpp UpdateSkid).  A pushed ground unit skids and doesn't walk until it has stopped, and friction takes more
// than a hit gives anything heavy: a Glaive at a Newton is thrown back at speed where a Knight stays where it is, held from walking any nearer.  A unit facing the way it's pushed
// doesn't skid, and goes by as much as each hit and keeps none of it.  A hit that lifts it enough takes it off the ground, where nothing slows it and each further hit adds on:
// the gun stands higher than what it shoots, so pulling lifts where pushing presses down, which is why a Newton set to pull flings a Glaive about and hurts it.  A push of less
// than a tenth of an elmo a frame does nothing, as in CanApplyImpulse.  The fall is unit_fall_damage.lua's.  Left out: units hitting each other or a wall on the way, which do
// damage in a game, and the slope a skid runs down, which on a hill counts for more than the push.  A beam can miss what's skidding fast between ticks, so a Glaive isn't thrown
// quite as far as in a game: 150 elmos past the Newton's reach against 115 here.
//
// Throwing: unit_thrower.lua.  What's around the thrower goes up, and if the thrower can still do it, across by the same offset, so that a group keeps its shape.
//
// Teleport: unit_teleporter.lua.  Whoever's been waiting longest at a beacon comes over to the Djinn, one at a time, for a time that goes by its mass.
//
// Missiles: unit_missilesilo.lua and cruisemissile.lua.  A silo builds on a free pad, and firing takes the missile away and frees its pad.
#include <algorithm>
#include <cmath>

#include "zksim/world.hpp"

namespace zk {

namespace {
constexpr float PI = 3.14159265358979f;
constexpr float SMALLEST_PUSH = 0.1f;       // CanApplyImpulse: impulse.SqLength() <= 0.01
constexpr float STARTS_FLYING = 0.2f;       // StartFlying: speed off the ground
constexpr float SKID_FRICTION = 0.35f;      // UpdateSkid's speedReduction, per frame
constexpr float SKID_DRAG = 0.01f;          // groundFrictionCoefficient, of its speed per frame
constexpr float LANDING_COUNTS = 1.0f;      // minCollisionSpeed: a landing slower than this is nothing
constexpr float BOUNCE = 0.3f, FRICTION = 0.8f;     // unit_fall_damage.lua: elasticity, friction
constexpr float TANGENT_DAMAGE = 0.5f;
constexpr float SAFE_SPEED = 3.0f;          // velocityDamageThreshold
constexpr float DAMAGE_PER_SPEED = 0.6f;    // of its mass: velocityDamageScale
constexpr float IMPULSE_LIFT = 1.0f / 20.0f;   // AddGadgetImpulse: up, per unit of impulse over mass
constexpr float PUSH_LOW = 0.4f, PUSH_SPREAD = 1.2f;   // a hit's impulse is random, between 0.4 and 1.6 of the weapon's
constexpr float GUN_STANDS_HIGHER = 25.0f;  // the gun's middle over its target's: what gives a hit its tilt
constexpr float FACING_THE_PUSH = 0.95f;    // StartSkidding: a hit along its own facing is no skid
constexpr float THROWN_FASTEST = 20.0f;     // SPEED_MAX
constexpr float UP_FRAMES = 130.0f;         // of a throw, once it's going across: as measured, where MIN_FLY_TIME says 125
constexpr float ACROSS_AFTER = 18.0f;       // "Sideways time" and the frames it takes to get going
constexpr float THROW_APEX = 317.0f;        // as measured, on flat ground over 500 elmos
constexpr float THROW_OVERSHOOT = 1.02f;    // of the offset, as measured
constexpr float BEACON_RADIUS = 200.0f;     // BEACON_TELEPORT_RADIUS
constexpr float EXIT_GAP = 40.0f;           // past its own footprint, from the Djinn
constexpr int EXIT_TRIES = 8;
constexpr int DEPLOY_FRAMES = 30;           // amphtele.lua: DEPLOY_SPEED and its turns, about a second
constexpr float STILL_SPEED = 0.05f;
constexpr int TELEPORT_CHECK_TICKS = 3;     // the gadget looks every 16 frames
}  // namespace

// ---------- Pushing ----------

void World::push_from(int u, int source, float impulse) {
    const size_t i = size_t(u);
    const auto& d = type_of(u);
    if (!(d.flags & ZK_U_MOBILE) || build[i] < 1.0f || d.push_scale <= 0.0f || jump[i].flying) return;
    const float dx = x[i] - x[size_t(source)], dz = z[i] - z[size_t(source)], dis = std::hypot(dx, dz);
    if (dis <= 0.0f) return;
    const float magnitude = impulse * (PUSH_LOW + rng.uniform() * PUSH_SPREAD);
    const float across = magnitude * d.push_scale / dis;
    const float pvx = dx * across, pvz = dz * across;
    const bool flies = (d.flags & ZK_U_AIR) != 0;
    const float lift = flies ? 0.0f : std::fabs(magnitude) * IMPULSE_LIFT / d.mass - GUN_STANDS_HIGHER * across;
    if (std::hypot(std::hypot(pvx, pvz), lift) <= SMALLEST_PUSH) return;
    Skid& s = skid[i];
    if (!flies && s.y <= 0.0f && lift <= STARTS_FLYING) {
        const float hit = std::hypot(pvx, pvz);
        if (!s.on && hit > 0.0f && (pvx * std::sin(heading[i]) + pvz * std::cos(heading[i])) / hit > FACING_THE_PUSH) {
            x[i] = std::clamp(x[i] + pvx, 0.0f, map->width() - 1.0f);
            z[i] = std::clamp(z[i] + pvz, 0.0f, map->depth() - 1.0f);
            return;
        }
        if (hit <= SKID_FRICTION) {
            s.on = true;
            return;
        }
    }
    if (!s.on) {
        // From however it was going, if it's on the ground: that's what it carries on with.
        s = Skid{};
        s.on = true;
        if (!flies) {
            s.vx = vx[i];
            s.vz = vz[i];
        }
    }
    s.vx += pvx;
    s.vz += pvz;
    if (s.y > 0.0f || lift > STARTS_FLYING) s.vy += lift;
}

// A ground unit, this tick: in the air there's only gravity, and on the ground friction.
void World::step_skid(int u) {
    const size_t i = size_t(u);
    const auto& d = type_of(u);
    Skid& s = skid[i];
    float px = x[i], pz = z[i];
    for (int f = 0; f < ZK_TICK_FRAMES && s.on; ++f) {
        if (s.y > 0.0f || s.vy > STARTS_FLYING) {
            s.vy -= GRAVITY;
            s.y += s.vy;
            px += s.vx;
            pz += s.vz;
            if (s.y > 0.0f) continue;
            const float impact = -s.vy, along = std::hypot(s.vx, s.vz);
            s.y = 0.0f;
            s.vy = 0.0f;
            if (impact <= LANDING_COUNTS) continue;
            const float hit = std::hypot(impact, along * TANGENT_DAMAGE) - SAFE_SPEED;
            if (hit > 0.0f) {
                const float amount = hit * d.mass * DAMAGE_PER_SPEED * armor_mult(u);
                hp[i] -= amount;
                damage_taken[i] += amount;
                recent_damage[i] += amount;
            }
            s.vx *= FRICTION;
            s.vz *= FRICTION;
            s.vy = impact * BOUNCE;
            continue;
        }
        s.vy = 0.0f;
        const float along = std::hypot(s.vx, s.vz), left = along - SKID_FRICTION - along * SKID_DRAG;
        if (left <= 0.0f) {
            s = Skid{};
            break;
        }
        s.vx *= left / along;
        s.vz *= left / along;
        px += s.vx;
        pz += s.vz;
    }
    const float frames = float(ZK_TICK_FRAMES);
    px = std::clamp(px, 0.0f, map->width() - 1.0f);
    pz = std::clamp(pz, 0.0f, map->depth() - 1.0f);
    vx[i] = (px - x[i]) / frames;
    vz[i] = (pz - z[i]) / frames;
    speed[i] = std::hypot(vx[i], vz[i]);
    x[i] = px;
    z[i] = pz;
}

// What flies goes on doing what it was doing, and works its way back at its own rate of braking.
void World::step_air_skid(int u, float& px, float& pz) {
    Skid& s = skid[size_t(u)];
    const float brake = type_of(u).brake;
    for (int f = 0; f < ZK_TICK_FRAMES && s.on; ++f) {
        px += s.vx;
        pz += s.vz;
        const float along = std::hypot(s.vx, s.vz), left = along - brake;
        if (left <= 0.0f) {
            s = Skid{};
            break;
        }
        s.vx *= left / along;
        s.vz *= left / along;
    }
}

// ---------- Throwing ----------

bool World::throw_ready(int u) const {
    if (type_of(u).throw_radius <= 0.0f) return false;
    const int m = manual_mount(u);
    return m >= 0 && cooldown[size_t(u) * ZK_MAX_MOUNTS + size_t(m)] <= 0.0f;
}

// CheckOverlobPrevention: it won't fire with nothing around it that its side can see.
bool World::someone_to_throw(int u) const {
    bool found = false;
    const size_t i = size_t(u);
    const float radius = type_of(u).throw_radius;
    for_near(x[i], z[i], radius, [&](int o) {
        const size_t k = size_t(o);
        if (found || o == u || !alive[k] || !(type_of(o).flags & ZK_U_MOBILE) || (type_of(o).flags & ZK_U_AIR) || jump[k].flying) return;
        if (std::hypot(x[k] - x[i], z[k] - z[i]) <= radius && speed[k] <= THROWN_FASTEST && (side[k] == side[i] || in_sight(side[i], o))) found = true;
    });
    return found;
}

void World::throw_around(int u, float tx, float tz) {
    const size_t i = size_t(u);
    const float radius = type_of(u).throw_radius, ox = tx - x[i], oz = tz - z[i];
    const float total = UP_FRAMES + ACROSS_AFTER;
    for_near(x[i], z[i], radius, [&](int o) {
        const size_t k = size_t(o);
        if (o == u || !alive[k] || !(type_of(o).flags & ZK_U_MOBILE) || (type_of(o).flags & ZK_U_AIR) || jump[k].flying) return;
        if (std::hypot(x[k] - x[i], z[k] - z[i]) > radius || speed[k] > THROWN_FASTEST) return;
        JumpState& j = jump[k];
        j.busy = j.flying = true;
        j.delay = j.timeout = 0.0f;
        j.at = 0.0f;
        j.step = 1.0f / total;
        j.from_x = x[k];
        j.from_z = z[k];
        j.to_x = std::clamp(x[k] + ox * THROW_OVERSHOOT, 0.0f, map->width() - 1.0f);
        j.to_z = std::clamp(z[k] + oz * THROW_OVERSHOOT, 0.0f, map->depth() - 1.0f);
        j.height = THROW_APEX;
        j.sideways = ACROSS_AFTER / total;
        j.thrown_by = u;
        skid[k] = Skid{};
        speed[k] = vx[k] = vz[k] = 0.0f;
        if (type_of(o).jump_range <= 0.0f) thrown.push_back(o);
    });
}

void World::update_throwers() {
    for (int u : throwers) {
        const size_t i = size_t(u);
        if (!alive[i] || order[i] != ZK_ORDER_MANUAL_FIRE || !active(u) || disarmed(u) || jump[i].flying || skid[i].y > 0.0f) continue;
        const int m = manual_mount(u);
        if (m < 0 || cooldown[i * ZK_MAX_MOUNTS + size_t(m)] > 0.0f) continue;
        const int t = order_target[i];
        if (t >= 0 && !alive[size_t(t)]) {
            order[i] = ZK_ORDER_NONE;
            continue;
        }
        const float tx = t >= 0 ? x[size_t(t)] : goal_x[i], tz = t >= 0 ? z[size_t(t)] : goal_z[i];
        const auto& d = type_of(u);
        const auto& w = cat->weapons[size_t(d.mount_weapon[m])];
        if (std::hypot(tx - x[i], tz - z[i]) > w.range || !someone_to_throw(u)) continue;
        throw_around(u, tx, tz);
        for (int k = 0; k < d.n_mounts; ++k)
            if (cat->weapons[size_t(d.mount_weapon[k])].flags & ZK_W_THROWS) cooldown[i * ZK_MAX_MOUNTS + size_t(k)] = w.reload;
        if (!(d.flags & ZK_U_FIRES_CLOAKED)) show_for_action(u);
        ++ai[i].shots_fired;
        order[i] = ZK_ORDER_NONE;
    }
}

// ---------- Teleporting ----------

const Teleporter* World::teleporter_of(int u) const {
    for (const Teleporter& t : teleporters)
        if (t.unit == u) return &t;
    return nullptr;
}

// Who has waited longest at this beacon.
int World::next_waiter(const Teleporter& t) const {
    int best = -1;
    const size_t b = size_t(t.beacon);
    for_near(x[b], z[b], BEACON_RADIUS, [&](int o) {
        const size_t k = size_t(o);
        if (!alive[k] || order[k] != ZK_ORDER_WAIT_AT_BEACON || order_target[k] != t.beacon || jump[k].busy) return;
        if (std::hypot(x[k] - x[b], z[k] - z[b]) > BEACON_RADIUS) return;
        if (best < 0 || wait_since[k] < wait_since[size_t(best)]) best = o;
    });
    return best;
}

// GetTeleTargetPosRandomNoBuildTest: out from the Djinn, on ground it can walk on.  It has no next order to face here, so it goes any way.
void World::send_through(int u, const Teleporter& t) {
    const size_t i = size_t(u), via = size_t(t.unit);
    const auto& d = type_of(u);
    const float gap = d.footprint_radius + EXIT_GAP;
    float angle = rng.uniform() * 2.0f * PI;
    const float turn = rng.uniform() > 0.5f ? PI / 4.0f : -PI / 4.0f;
    float px = x[via], pz = z[via];
    for (int k = 0; k < EXIT_TRIES; ++k) {
        const float tx = std::clamp(x[via] + std::sin(angle) * gap, 0.0f, map->width() - 1.0f);
        const float tz = std::clamp(z[via] + std::cos(angle) * gap, 0.0f, map->depth() - 1.0f);
        if (d.move_class < 0 || map->speed_at(d.move_class, tx, tz) > 0.0f) {
            px = tx;
            pz = tz;
            break;
        }
        angle += turn;
    }
    x[i] = px;
    z[i] = pz;
    vx[i] = vz[i] = speed[i] = 0.0f;
    skid[i] = Skid{};
    order[i] = ZK_ORDER_NONE;
    wait_since[i] = -1;
}

void World::teleport_waiter(Teleporter& t) {
    const size_t i = size_t(t.unit), b = size_t(t.beacon);
    const bool can_work = active(t.unit) && !disarmed(t.unit) && active(t.beacon) && !disarmed(t.beacon);
    if (t.carrying >= 0) {
        const size_t k = size_t(t.carrying);
        // Walked off, told something else or dead: that's the end of that one.
        if (!alive[k] || order[k] != ZK_ORDER_WAIT_AT_BEACON || order_target[k] != t.beacon || std::hypot(x[k] - x[b], z[k] - z[b]) > BEACON_RADIUS) {
            t.carrying = -1;
        } else if (can_work) {
            t.carry_left -= float(ZK_TICK_FRAMES);
            if (t.carry_left <= 0.0f) {
                send_through(t.carrying, t);
                t.carrying = -1;
            }
        }
    }
    if (t.carrying < 0 && can_work && tick_count % TELEPORT_CHECK_TICKS == 0) {
        t.carrying = next_waiter(t);
        if (t.carrying >= 0) t.carry_left = std::floor(type_of(t.carrying).mass / type_of(int(i)).teleport_rate);
    }
}

void World::update_teleporters() {
    // By index, as making a beacon adds a unit, though nothing that's a teleporter.
    for (size_t k = 0; k < teleporters.size(); ++k) {
        Teleporter& t = teleporters[k];
        const size_t i = size_t(t.unit);
        if (!alive[i]) {
            // Its beacon goes with it.
            if (t.beacon >= 0 && alive[size_t(t.beacon)]) hp[size_t(t.beacon)] = 0.0f;
            t.beacon = t.carrying = -1;
            continue;
        }
        if (t.beacon >= 0 && !alive[size_t(t.beacon)]) {
            t.beacon = t.carrying = -1;
            t.deployed = false;
        }
        const auto& d = type_of(t.unit);
        if (order[i] == ZK_ORDER_PLACE_BEACON) {
            // Told to do the same again, it carries on with what it's making.
            if (!t.making || t.make_x != goal_x[i] || t.make_z != goal_z[i]) {
                t.making = true;
                t.make_x = goal_x[i];
                t.make_z = goal_z[i];
                t.make_left = d.beacon_frames;
                t.deployed = false;
                t.carrying = -1;
            }
            if (active(t.unit) && !disarmed(t.unit)) t.make_left -= float(ZK_TICK_FRAMES) * (1.0f - slow_now(t.unit));
            if (t.make_left <= 0.0f) {
                if (t.beacon >= 0) hp[size_t(t.beacon)] = 0.0f;
                const int side_of = side[i], beacon_type = d.beacon_type;
                const float bx = std::clamp(t.make_x, 0.0f, map->width() - 1.0f), bz = std::clamp(t.make_z, 0.0f, map->depth() - 1.0f);
                const int made = add(beacon_type, side_of, bx, bz, 0.0f, 1.0f, 1.0f);
                std::fill_n(cooldown.begin() + std::ptrdiff_t(size_t(made) * ZK_MAX_MOUNTS), ZK_MAX_MOUNTS, 0.0f);
                ai[size_t(made)].queue_was_empty = true;
                t.beacon = made;
                t.making = false;
                order[i] = ZK_ORDER_NONE;
            }
            continue;
        }
        t.making = false;
        // amphtele.lua: walking undeploys it, and standing still with a beacon deploys it again.
        if (speed[i] > STILL_SPEED || order[i] != ZK_ORDER_NONE) {
            t.deployed = false;
            t.carrying = -1;
            t.still_since = -1;
            continue;
        }
        if (t.beacon < 0) continue;
        if (t.still_since < 0) t.still_since = frame;
        if (!t.deployed && frame - t.still_since >= DEPLOY_FRAMES) t.deployed = true;
        if (t.deployed) teleport_waiter(t);
    }
}

// ---------- Missiles ----------

// The first pad it has nothing on, or -1.
int World::free_pad(int silo) const {
    uint32_t used = 0;
    for (const SiloMissile& m : silo_missiles)
        if (m.silo == silo && alive[size_t(m.unit)]) used |= 1u << m.pad;
    for (int pad = 0; pad < type_of(silo).missile_pads; ++pad)
        if (!(used & (1u << pad))) return pad;
    return -1;
}

// cruisemissile.lua's RemoveMissile: it goes off the map when it fires, and leaves nothing behind.
void World::launch_missile(int u) {
    alive[size_t(u)] = 0;
    silo_missiles.erase(std::remove_if(silo_missiles.begin(), silo_missiles.end(), [u](const SiloMissile& m) { return m.unit == u; }), silo_missiles.end());
}

void World::destroy_missiles_of(int silo) {
    for (const SiloMissile& m : silo_missiles)
        if (m.silo == silo && alive[size_t(m.unit)]) hp[size_t(m.unit)] = 0.0f;
}

}  // namespace zk
