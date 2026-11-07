// Shields: what they stop, what charging them costs, and how they share charge.
//
// CheckShieldCollisions and TraceRayShields for what's stopped: a shot stops where its path first enters a hostile sphere that can
// pay for it, and explodes there, so its splash still reaches whatever's near.  The spheres are where they are when it lands.
// weapon_shield_merge.lua's DrainShieldAndCheckProjectilePenetrate for what it costs: the listed amount, whatever the range, and a
// shield with no more than that doesn't stop it.  A beam is charged frame by frame, so a shield that runs out stops part of it.
// unit_shield_charge.lua for charging and unit_shield_link.lua for sharing, both without their priorities.
#include <algorithm>
#include <cmath>

#include "zksim/world.hpp"

namespace zk {

namespace {
constexpr float FRAMES_PER_SECOND = 30.0f;
constexpr float RIGHT_ANGLE = 1.5707964f;
constexpr int ARC_SAMPLES = 8;              // along the part of a path that's over a sphere: it's the same when a path is level
constexpr float SURFACE_MARGIN = 1.001f;    // of the radius squared, so that a level shot is in at the point it was worked out to enter
constexpr int LINK_STEP_FRAMES = 2;
constexpr int LINK_UPDATE_TICKS = 30 / ZK_TICK_FRAMES;
constexpr float LINK_FLOW = 0.01f;          // of the difference in charge: RECHARGE_KOEF

// shieldRank.  A shield doesn't start a transfer with a lower rank.
int link_rank(float radius) { return radius > 400.0f ? 3 : radius > 200.0f ? 2 : 1; }

struct ShotPath {
    float x, y, z;              // where it starts
    float dx, dy, dz;           // to where it's going
    float peak;                 // how far a ballistic shot is over the straight line, half way
    bool from_above;            // nothing but where it lands counts: fired from or at the air, or climbing first

    float height(float t) const { return y + dy * t + 4.0f * peak * t * (1.0f - t); }
};

// Cannon.cpp's two solutions, as a parabola.  A shot that can't reach is treated as one at 45 degrees.
float arc_peak(const zk_weapon_rec& w, float distance) {
    if (w.arc <= 0.0f) return 0.0f;
    const float low = 0.5f * std::asin(std::min(1.0f, w.arc * distance));
    const float elevation = (w.flags & ZK_W_HIGH_ARC) ? RIGHT_ANGLE - low : low;
    return distance * std::tan(elevation) / 4.0f;
}

// How far along its path a shot enters this sphere, or less than 0 if it doesn't.  Starting inside doesn't count: exteriorShield.
float entry(const ShotPath& p, float cx, float cy, float cz, float radius) {
    const float fx = p.x - cx, fy = p.y - cy, fz = p.z - cz;
    const float r2 = radius * radius;
    if (fx * fx + fy * fy + fz * fz <= r2) return -1.0f;
    if (p.from_above) {
        const float lx = fx + p.dx, lz = fz + p.dz;
        return lx * lx + lz * lz <= r2 ? 1.0f : -1.0f;
    }
    const float a = p.dx * p.dx + p.dz * p.dz;
    if (a <= 0.0f) return -1.0f;
    const float b = fx * p.dx + fz * p.dz;
    const float disc = b * b - a * (fx * fx + fz * fz - r2);
    if (disc < 0.0f) return -1.0f;
    const float root = std::sqrt(disc);
    const float first = std::max(0.0f, (-b - root) / a), last = std::min(1.0f, (-b + root) / a);
    if (first > last) return -1.0f;
    for (int k = 0; k <= ARC_SAMPLES; ++k) {
        const float t = first + (last - first) * float(k) / float(ARC_SAMPLES);
        const float sx = fx + p.dx * t, sy = p.height(t) - cy, sz = fz + p.dz * t;
        if (sx * sx + sy * sy + sz * sz <= r2 * SURFACE_MARGIN) return t;
    }
    return -1.0f;
}
}  // namespace

float World::shield_takes(const zk_weapon_rec& w, const Shot& s, float px, float pz, float left, float& hit_x, float& hit_z) {
    if (shields.empty() || !w.intercepted_by) return 0.0f;
    const float dx = px - s.origin_x, dz = pz - s.origin_z;
    const float oy = map->height_at(s.origin_x, s.origin_z);
    const bool from_above = s.air || (type_of(s.source).flags & ZK_U_AIR) || w.climb_frames > 0.0f;
    const ShotPath path{s.origin_x, oy, s.origin_z, dx, map->height_at(px, pz) - oy, dz, arc_peak(w, std::hypot(dx, dz)), from_above};
    const float whole = w.damage[ZK_ARMOR_SHIELD];
    const bool beam = (w.flags & ZK_W_BEAM) != 0;
    int stopper = -1;
    float nearest = 2.0f;
    for (const ShieldUnit& sh : shields) {
        const int u = sh.unit;
        if (!alive[size_t(u)] || side[size_t(u)] == s.source_side || !active(u)) continue;
        if (shield[size_t(u)] <= (beam ? 0.0f : whole) || !(sh.weapon->shield_intercept & w.intercepted_by)) continue;
        const float t = entry(path, x[size_t(u)], map->height_at(x[size_t(u)], z[size_t(u)]), z[size_t(u)], sh.weapon->shield_radius);
        if (t >= 0.0f && t < nearest) {
            nearest = t;
            stopper = u;
        }
    }
    if (stopper < 0) return 0.0f;
    const float taken = beam && whole > 0.0f ? std::min(left, shield[size_t(stopper)] / whole) : left;
    shield[size_t(stopper)] = std::max(0.0f, shield[size_t(stopper)] - whole * taken);
    hit_x = s.origin_x + dx * nearest;
    hit_z = s.origin_z + dz * nearest;
    return taken;
}

void World::recharge_shields() {
    const float frames = float(ZK_TICK_FRAMES);
    for (const ShieldUnit& sh : shields) {
        const int u = sh.unit;
        if (!alive[size_t(u)] || !active(u)) continue;
        const auto& w = *sh.weapon;
        const float rate = 1.0f - slow_now(u);      // unit_attributes.lua shieldRegen
        const float whole = w.shield_regen * rate * frames;
        const float added = std::min(whole, w.shield_power - shield[size_t(u)]);
        if (added <= 0.0f) continue;
        // All of a step or none of it, and only what fills it is paid for.
        if (economy_on && w.shield_energy > 0.0f && !pay_energy(side[size_t(u)], w.shield_energy * rate * frames * added / whole)) continue;
        shield[size_t(u)] += added;
    }
}

void World::find_shield_links() {
    for (ShieldUnit& sh : shields) {
        sh.links.clear();
        if (!alive[size_t(sh.unit)] || !active(sh.unit) || (sh.weapon->flags & ZK_W_UNLINKED)) continue;
        const float own_y = map->height_at(x[size_t(sh.unit)], z[size_t(sh.unit)]);
        for (int k = 0; k < int(shields.size()); ++k) {
            const ShieldUnit& other = shields[size_t(k)];
            if (other.unit == sh.unit || side[size_t(other.unit)] != side[size_t(sh.unit)] || !alive[size_t(other.unit)] || !active(other.unit)) continue;
            if ((other.weapon->flags & ZK_W_UNLINKED) || link_rank(sh.weapon->shield_radius) > link_rank(other.weapon->shield_radius)) continue;
            const float reach = sh.weapon->shield_radius + other.weapon->shield_radius;
            const float gx = x[size_t(other.unit)] - x[size_t(sh.unit)], gz = z[size_t(other.unit)] - z[size_t(sh.unit)];
            const float gy = map->height_at(x[size_t(other.unit)], z[size_t(other.unit)]) - own_y;
            if (gx * gx + gy * gy + gz * gz < reach * reach) sh.links.push_back(k);
        }
    }
}

void World::share_shield_charge() {
    for (const ShieldUnit& sh : shields) {
        if (sh.links.empty() || !alive[size_t(sh.unit)] || !active(sh.unit)) continue;
        // One more choice than it has links: the gadget means to penalise a unit with few of them.
        const size_t pick = size_t(rng.uniform() * float(sh.links.size() + 1));
        if (pick >= sh.links.size()) continue;
        const ShieldUnit& other = shields[size_t(sh.links[pick])];
        if (!alive[size_t(other.unit)] || !active(other.unit)) continue;
        const bool gives = shield[size_t(sh.unit)] > shield[size_t(other.unit)];
        const ShieldUnit& high = gives ? sh : other;
        const ShieldUnit& low = gives ? other : sh;
        float& high_charge = shield[size_t(high.unit)];
        float& low_charge = shield[size_t(low.unit)];
        // Room is left for the one getting it to charge itself.
        const float room = low.weapon->shield_power - low.weapon->shield_regen * FRAMES_PER_SECOND - low_charge;
        const float flow = std::min({LINK_FLOW * (high_charge - low_charge), high_charge, room});
        if (flow <= 0.0f) continue;
        const float moved = flow * (1.0f - slow_now(high.unit));    // totalReloadSpeedChange
        high_charge -= moved;
        low_charge += moved;
    }
}

void World::update_shields() {
    if (shields.empty()) return;
    recharge_shields();
    if (shields.size() < 2) return;
    if (tick_count % LINK_UPDATE_TICKS == 0) find_shield_links();
    for (int f = 0; f < ZK_TICK_FRAMES; ++f)
        if ((frame + f) % LINK_STEP_FRAMES == 0) share_shield_charge();
}

}  // namespace zk
