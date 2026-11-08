// Resources and construction: income, overdrive through pylon grids, upkeep, build steps that stall without resources, assisting,
// factories that repeat a type, and a side's free first factory.  Reclaim is in wrecks.cpp.  Not here yet: repair (01-simulator.md 8.9).
#include <algorithm>
#include <cmath>

#include "zksim/profile.hpp"
#include "zksim/world.hpp"

namespace zk {

namespace {
constexpr float TICK_SECONDS = float(ZK_TICK_FRAMES) / 30.0f;
constexpr float SPOT_REACH = 64.0f;          // an extractor stands on a spot, near enough
constexpr float EXIT_GAP = 8.0f;             // between a factory's footprint and a new unit's
constexpr float PI = 3.14159265358979f;
constexpr int EXIT_TRIES = 8;
constexpr float PAD_OFFSET = 1.0f / 3.0f;   // of a silo's footprint: where its four pads are, 16 elmos out in a game
constexpr int OVERDRIVE_TICKS = 30 / ZK_TICK_FRAMES;      // TEAM_SLOWUPDATE_RATE
constexpr float OVERDRIVE_PER_ROOT = 0.25f;  // energyToExtraM: an extractor earns this times the root of its energy, of its income
constexpr int UPKEEP_TICKS = 30 / ZK_TICK_FRAMES;         // unit_energy_drain.lua's PERIOD
constexpr int SOLAR_SHUTTING_FRAMES = 19;    // energysolar.lua: 75 degrees at 120 a second
constexpr int SOLAR_OPENING_FRAMES = 38;     // ... and at 60 a second
}  // namespace

void World::set_economy(int n_spots, const float* spot_triples, float metal, float energy, float innate_metal, float innate_energy, float storage) {
    economy_on = true;
    spots.resize(size_t(n_spots));
    for (int i = 0; i < n_spots; ++i) spots[size_t(i)] = MetalSpot{spot_triples[3 * i], spot_triples[3 * i + 1], spot_triples[3 * i + 2]};
    // Whatever already stands was added before there were spots to stand on.
    for (size_t u = 0; u < extracts.size(); ++u) {
        const auto& d = type_of(int(u));
        extracts[u] = d.metal_make + (d.extract_mult > 0.0f ? d.extract_mult * spot_income(x[u], z[u]) : 0.0f);
    }
    for (auto& p : purses) {
        p = Purse{};
        p.metal = metal;
        p.energy = energy;
        p.innate_metal = p.metal_income = innate_metal;
        p.innate_energy = p.energy_income = innate_energy;
        p.storage = storage;
    }
}

float World::spot_income(float px, float pz) const {
    float best = SPOT_REACH * SPOT_REACH, income = 0.0f;
    for (const auto& s : spots) {
        const float d = (s.x - px) * (s.x - px) + (s.z - pz) * (s.z - pz);
        if (d <= best) {
            best = d;
            income = s.income;
        }
    }
    return income;
}

bool World::within(int u, float px, float pz, float reach) const {
    const float dx = px - x[size_t(u)], dz = pz - z[size_t(u)];
    return dx * dx + dz * dz <= reach * reach;
}

int World::place_frame(int t, int s, float px, float pz, float head) {
    const int fresh = place_frame_only(t, s, px, pz, head);
    // A factory sends what it makes on its way, so it goes idle in the ordinary way, and from there it has a place to return to.
    if (fresh >= 0) ai[size_t(fresh)].queue_was_empty = false;
    return fresh;
}

int World::place_frame_only(int t, int s, float px, float pz, float head) {
    const int f = add(t, s, px, pz, head, 0.0f, 0.0f);
    // Whatever add() drew for a unit caught mid-fight: this one is new, and its weapons are ready.
    std::fill_n(cooldown.begin() + std::ptrdiff_t(size_t(f) * ZK_MAX_MOUNTS), ZK_MAX_MOUNTS, 0.0f);
    return f;
}

bool World::shut_for_hit(int u) const {
    const float wait = type_of(u).close_frames;
    return wait > 0.0f && float(frame - last_hit[size_t(u)]) < wait + float(SOLAR_OPENING_FRAMES);
}

// Armoured once shut, and from the moment it starts to open again.
bool World::armoured_for_hit(int u) const {
    return frame - closing_since[size_t(u)] >= SOLAR_SHUTTING_FRAMES && float(frame - last_hit[size_t(u)]) < type_of(u).close_frames;
}

// A hit while it's shutting or shut only restarts the wait.  One while it's opening turns it back from as far as it had got, at twice the speed.
void World::hit_closes(int u) {
    const size_t i = size_t(u);
    const int opened = frame - last_hit[i] - int(type_of(u).close_frames);
    if (opened >= SOLAR_OPENING_FRAMES) closing_since[i] = frame;
    else if (opened >= 0) closing_since[i] = frame + opened / 2 - SOLAR_SHUTTING_FRAMES;
    last_hit[i] = frame;
}

bool World::pay_energy(int s, float amount) {
    Purse& p = purses[s];
    if (p.energy < amount) return false;
    p.energy -= amount;
    p.energy_spent += amount;
    return true;
}

// unit_energy_drain.lua, without its priorities: first come, first served.  A radar left unpaid goes off until it's paid again.
void World::pay_upkeep() {
    for (int u : upkeepers) {
        const size_t i = size_t(u);
        if (!alive[i] || !active(u) || disarmed(u)) continue;
        const auto& d = type_of(u);
        unpaid[i] = !pay_energy(side[i], d.energy_upkeep + (area_on[i] ? d.area_cloak_upkeep : 0.0f));
    }
}

void World::collect_income() {
    float metal[2] = {purses[0].innate_metal, purses[1].innate_metal};
    float energy[2] = {purses[0].innate_energy, purses[1].innate_energy};
    for (int u : earners) {
        const size_t i = size_t(u);
        if (!alive[i] || !active(u) || shut_for_hit(u)) continue;
        const auto& d = type_of(u);
        energy[side[i]] += d.energy_make;
        if (d.extract_mult <= 0.0f || !disarmed(u)) metal[side[i]] += extracts[i] + overdrive[i];
    }
    for (int s = 0; s < 2; ++s) {
        Purse& p = purses[s];
        p.energy_made = energy[s];
        // As the engine counts it, reclaim is income.  This tick's comes after this, so it shows from the next.
        p.metal_income = metal[s] + p.reclaimed / TICK_SECONDS;
        p.reclaimed = 0.0f;
        p.energy_income = energy[s] - p.overdrive_energy;
        p.metal = std::min(p.storage, p.metal + p.metal_income * TICK_SECONDS);
        p.energy = std::clamp(p.energy + p.energy_income * TICK_SECONDS, 0.0f, p.storage);
    }
}

void World::update_overdrive() {
    pylons.erase(std::remove_if(pylons.begin(), pylons.end(), [this](int u) { return !alive[size_t(u)]; }), pylons.end());
    overdrive_side(0);
    overdrive_side(1);
}

// unit_mex_overdrive.lua, for an ally team of one player.  What's spare goes to overdrive in proportion to how full storage is.
// Each extractor's share goes by its income squared, over all grids; a grid can't take more than its own generators make, and
// what it can't take goes to the other grids.  What no grid can take stays in storage.
void World::overdrive_side(int s) {
    Purse& p = purses[s];
    const float full = p.energy > 0.0f ? std::min(1.0f, p.energy / std::max(p.storage, p.energy)) : 0.0f;
    float energy = std::max(0.0f, p.energy_made - p.energy_spent) * full;
    p.energy_spent = 0.0f;

    // Grids are worked out afresh: stun, death and new buildings all change them, and there are few pylons.
    std::vector<int> nodes;
    for (int u : pylons) {
        if (side[size_t(u)] != s) continue;
        overdrive[size_t(u)] = 0.0f;
        if (active(u) && !disarmed(u)) nodes.push_back(u);
        else low_power[size_t(u)] = type_of(u).needed_link > 0.0f;     // on no grid
    }
    const size_t n = nodes.size();
    std::vector<int> parent(n);
    for (size_t a = 0; a < n; ++a) parent[a] = int(a);
    const auto find = [&parent](int a) {
        while (parent[size_t(a)] != a) a = parent[size_t(a)] = parent[size_t(parent[size_t(a)])];
        return a;
    };
    for (size_t a = 0; a < n; ++a) {
        for (size_t b = a + 1; b < n; ++b) {
            const size_t ua = size_t(nodes[a]), ub = size_t(nodes[b]);
            const float reach = type_of(nodes[a]).pylon_range + type_of(nodes[b]).pylon_range;
            const float dx = x[ua] - x[ub], dz = z[ua] - z[ub];
            if (dx * dx + dz * dz < reach * reach) parent[size_t(find(int(a)))] = find(int(b));
        }
    }

    // Per grid, by root.
    std::vector<float> capacity(n, 0.0f), squares(n, 0.0f);
    std::vector<char> maxed(n, 0);
    float all_squares = 0.0f;
    for (size_t a = 0; a < n; ++a) {
        const auto& d = type_of(nodes[a]);
        const size_t g = size_t(find(int(a)));
        if (!shut_for_hit(nodes[a])) capacity[g] += d.energy_make;
        if (d.extract_mult > 0.0f) {
            const float m = extracts[size_t(nodes[a])];
            squares[g] += m * m;
            all_squares += m * m;
        }
    }
    // unit_mex_overdrive.lua: a weapon that needs a grid has to be on one that makes enough.  It goes by what the grid makes, spent or not.
    for (size_t a = 0; a < n; ++a) {
        const float needs = type_of(nodes[a]).needed_link;
        if (needs > 0.0f) low_power[size_t(nodes[a])] = needs > capacity[size_t(find(int(a)))];
    }
    for (bool again = true; again;) {
        again = false;
        for (size_t g = 0; g < n && !again; ++g) {
            if (maxed[g] || squares[g] <= 0.0f || energy * squares[g] / all_squares <= capacity[g]) continue;
            maxed[g] = 1;
            energy -= capacity[g];
            all_squares -= squares[g];
            again = true;
        }
    }

    p.overdrive_metal = p.overdrive_energy = 0.0f;
    for (size_t a = 0; a < n; ++a) {
        const int u = nodes[a];
        if (type_of(u).extract_mult <= 0.0f) continue;
        const size_t g = size_t(find(int(a)));
        const float m = extracts[size_t(u)];
        float drawn = 0.0f;
        if (maxed[g]) drawn = capacity[g] * m * m / squares[g];
        else if (all_squares > 0.0f) drawn = energy * m * m / all_squares;
        overdrive[size_t(u)] = m * OVERDRIVE_PER_ROOT * std::sqrt(drawn);
        p.overdrive_metal += overdrive[size_t(u)];
        p.overdrive_energy += drawn;
    }
}

// A step as far as the side can pay for.  First come, first served, in handle order, as the engine's builders are.
void World::advance_frame(int b, int f) {
    const auto& d = type_of(f);
    float step = std::min(type_of(b).build_speed * TICK_SECONDS / std::max(d.build_time, 1.0f), 1.0f - build[size_t(f)]);
    Purse& p = purses[side[size_t(b)]];
    if (d.metal_cost * step > p.metal) step = p.metal / d.metal_cost;
    if (d.energy_cost * step > p.energy) step = p.energy / d.energy_cost;
    p.metal = std::max(0.0f, p.metal - d.metal_cost * step);
    p.energy = std::max(0.0f, p.energy - d.energy_cost * step);
    p.energy_spent += d.energy_cost * step;
    hp[size_t(f)] = std::min(d.hp, hp[size_t(f)] + d.hp * step);
    build[size_t(f)] = build[size_t(f)] + step >= 0.99999f ? 1.0f : build[size_t(f)] + step;
}

void World::run_builder(int u) {
    const size_t i = size_t(u);
    if (order[i] == ZK_ORDER_RECLAIM) {
        reclaim_step(u);
    } else if (order[i] == ZK_ORDER_BUILD) {
        const int t = order_target[i];
        const auto& d = cat->types[size_t(t)];
        if (work_on[i] < 0) {
            if (!within(u, goal_x[i], goal_z[i], reach_of(u, d.footprint_radius))) return;
            Purse& p = purses[side[i]];
            const float centre_x = map->width() * 0.5f, centre_z = map->depth() * 0.5f;
            const float head = std::atan2(centre_x - goal_x[i], centre_z - goal_z[i]);
            const int f = place_frame(t, side[i], goal_x[i], goal_z[i], head);
            if ((d.flags & ZK_U_PLOPPABLE) && p.free_factory) {
                p.free_factory = false;
                build[size_t(f)] = 1.0f;
                hp[size_t(f)] = d.hp;
                order[i] = ZK_ORDER_NONE;
                return;
            }
            work_on[i] = f;
        }
        const int f = work_on[i];
        if (alive[size_t(f)] && within(u, x[size_t(f)], z[size_t(f)], reach_of(u, d.footprint_radius))) {
            advance_frame(u, f);
            show_for_action(u);
        }
        if (!alive[size_t(f)] || build[size_t(f)] >= 1.0f) {
            order[i] = ZK_ORDER_NONE;
            work_on[i] = -1;
        }
    } else if (order[i] == ZK_ORDER_ASSIST) {
        const int t = order_target[i];
        // Help for a finished factory goes to what it's making; one still being built is helped like any other building.
        const bool helps_factory = t >= 0 && alive[size_t(t)] && (type_of(t).flags & ZK_U_FACTORY) && build[size_t(t)] >= 1.0f;
        const int f = helps_factory ? work_on[size_t(t)] : t;
        if (f < 0 || !alive[size_t(f)] || build[size_t(f)] >= 1.0f) {
            if (!helps_factory) order[i] = ZK_ORDER_NONE;
            return;
        }
        if (within(u, x[size_t(t)], z[size_t(t)], reach_of(u, type_of(t).footprint_radius))) {
            advance_frame(u, f);
            show_for_action(u);
        }
    }
}

void World::run_factory(int u) {
    const size_t i = size_t(u);
    if (produces[i] < 0) return;
    int f = work_on[i];
    if (f >= 0 && (!alive[size_t(f)] || build[size_t(f)] >= 1.0f)) f = work_on[i] = -1;
    if (f < 0) {
        const auto& d = cat->types[size_t(produces[i])];
        // A silo builds on a pad, and with none free it can't start (unit_missilesilo.lua).
        if (type_of(u).missile_pads > 0) {
            const int pad = free_pad(u);
            if (pad < 0) return;
            const float across = type_of(u).footprint_radius * PAD_OFFSET;
            f = work_on[i] = place_frame(produces[i], side[i], x[i] + (pad % 2 ? across : -across), z[i] + (pad / 2 ? across : -across), heading[i]);
            silo_missiles.push_back({f, u, pad});
            advance_frame(u, f);
            return;
        }
        const float gap = type_of(u).footprint_radius + d.footprint_radius + EXIT_GAP;
        // Out of the front, or the nearest side a new unit could walk away from.
        float px = x[i], pz = z[i];
        for (int k = 0; k < EXIT_TRIES; ++k) {
            const float step = float((k + 1) / 2) * PI / 4.0f;
            const float h = heading[i] + (k % 2 ? step : -step);
            const float tx = x[i] + std::sin(h) * gap, tz = z[i] + std::cos(h) * gap;
            if (d.move_class < 0 || map->speed_at(d.move_class, tx, tz) > 0.0f) {
                px = tx;
                pz = tz;
                break;
            }
        }
        f = work_on[i] = place_frame(produces[i], side[i], px, pz, heading[i]);
    }
    advance_frame(u, f);
}

void World::run_economy() {
    collect_income();
    if (tick_count % OVERDRIVE_TICKS == 0) update_overdrive();
    if (tick_count % UPKEEP_TICKS == 0) pay_upkeep();
    // By handle, not by kind: a list can grow while it's walked, since placing a frame adds to the world.
    for (size_t k = 0, n = builders.size(); k < n; ++k) {
        const int u = builders[k];
        if (!active(u) || !alive[size_t(u)] || power_off(u)) continue;
        if (type_of(u).flags & ZK_U_FACTORY) run_factory(u);
        else run_builder(u);
    }
}

}  // namespace zk
