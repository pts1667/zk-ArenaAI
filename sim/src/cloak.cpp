// Cloak: who is hidden, what it costs and what shows them.
//
// unit_decloak_damaged.lua for a unit's own cloak.  unit_cloak_shield.lua for an area cloak, simply: its radius is there at once rather
// than growing, and what it covers is hidden at once rather than charging up in turn.  The cloaker's upkeep pays for it all.
#include <algorithm>

#include "zksim/world.hpp"

namespace zk {

namespace {
constexpr float MOVING_SPEED = 0.4472136f;       // elmos a frame: the root of 0.2, CLOAK_MOVE_THRESHOLD
constexpr int OWN_CLOAK_SHOWN_FRAMES = 90;       // after firing, damage or building: PERSONAL_DECLOAK_TIME
constexpr int LENT_CLOAK_SHOWN_FRAMES = 240;     // DEFAULT_DECLOAK_TIME
constexpr int OWN_CLOAK_NEAR_FRAMES = 45;        // after an enemy came close: PERSONAL_PROXIMITY_DECLOAK_TIME
constexpr int LENT_CLOAK_NEAR_FRAMES = 90;       // DEFAULT_PROXIMITY_DECLOAK_TIME
constexpr float UPDATES_PER_SECOND = 2.0f;       // a slow update, twice a second
}  // namespace

void World::show_for_action(int u) {
    const size_t i = size_t(u);
    if (!want_cloak[i] && !area_cloaked[i]) return;
    cloaked[i] = 0;
    const int shown = (type_of(u).flags & ZK_U_CAN_CLOAK) ? OWN_CLOAK_SHOWN_FRAMES : LENT_CLOAK_SHOWN_FRAMES;
    cloak_blocked_until[i] = std::max(cloak_blocked_until[i], frame + shown);
}

bool World::enemy_within(int u, float reach) const {
    bool found = false;
    for_near(x[size_t(u)], z[size_t(u)], reach, [&](int e) {
        if (found || side[size_t(e)] == side[size_t(u)] || !alive[size_t(e)] || build[size_t(e)] < 1.0f) return;
        const float dx = x[size_t(e)] - x[size_t(u)], dz = z[size_t(e)] - z[size_t(u)];
        found = dx * dx + dz * dz <= reach * reach;
    });
    return found;
}

void World::consider_cloak(int u) {
    const size_t i = size_t(u);
    if (!alive[i]) return;
    if (!want_cloak[i] && !area_cloaked[i]) {
        cloaked[i] = 0;
        return;
    }
    const auto& d = type_of(u);
    const bool own = (d.flags & ZK_U_CAN_CLOAK) != 0;
    const float reach = area_cloaked[i] && !own && d.bestowed_decloak > 0.0f ? d.bestowed_decloak : d.decloak_distance;
    if (enemy_within(u, reach)) {
        cloak_blocked_until[i] = std::max(cloak_blocked_until[i], frame + (own ? OWN_CLOAK_NEAR_FRAMES : LENT_CLOAK_NEAR_FRAMES));
        cloaked[i] = 0;
        return;
    }
    // Stunned, disarmed or on fire, it can't go back to hiding.
    if (frame < cloak_blocked_until[i] || !active(u) || disarmed(u) || burn_frames[i] > 0.0f) {
        cloaked[i] = 0;
        return;
    }
    // Under an area cloak it costs nothing of its own.  In a world with no economy there's nothing to pay with, and it's free.
    const float per_second = speed[i] > MOVING_SPEED ? d.cloak_cost_moving : d.cloak_cost;
    cloaked[i] = area_cloaked[i] || !economy_on || pay_energy(side[i], per_second / UPDATES_PER_SECOND);
}

void World::update_cloak() {
    update_burrows();
    if (area_cloakers.empty()) {
        for (int u : personal_cloakers) consider_cloak(u);
        return;
    }
    std::fill(area_cloaked.begin(), area_cloaked.end(), uint8_t(0));
    for (int c : area_cloakers) {
        const size_t i = size_t(c);
        if (!alive[i] || !area_on[i] || !active(c) || disarmed(c) || unpaid[i]) continue;
        const float r = type_of(c).area_cloak_radius;
        for_near(x[i], z[i], r, [&](int u) {
            if (side[size_t(u)] != side[i] || !alive[size_t(u)]) return;
            const float dx = x[size_t(u)] - x[i], dz = z[size_t(u)] - z[i];
            if (dx * dx + dz * dz <= r * r) area_cloaked[size_t(u)] = 1;
        });
    }
    // Anyone may be under one.
    for (int u = 0; u < count(); ++u) consider_cloak(u);
}

}  // namespace zk
