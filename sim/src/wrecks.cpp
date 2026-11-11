// What dying leaves behind, what's left of it after explosions, and reclaiming it.
//
// featuredefs_post.lua makes a wreck 40% of a unit's metal and a heap 20%, with health as the unit's and a reclaim time equal to its
// metal, so that a builder takes metal at its build speed whatever it's taking it from.
#include <algorithm>
#include <cmath>

#include "zksim/profile.hpp"
#include "zksim/world.hpp"

namespace zk {

namespace {
constexpr float GRID_CELL = 128.0f;          // as the unit grid
constexpr float WRECK_ABOVE = 0.5f;          // of health, in recent damage: the scripts' usual line between a wreck and a heap
constexpr float NANOFRAME_LEAVES_ABOVE = 0.05f;      // unit_nanoframe_death.lua
constexpr float TICK_SECONDS = float(ZK_TICK_FRAMES) / 30.0f;
}  // namespace

int World::add_wreck(float px, float pz, float metal, float wreck_hp, float radius, int feature_def, int heap_of) {
    if (wreck_cells.empty()) wreck_cells.resize(size_t(grid_cols) * size_t(grid_rows));
    const int id = int(wrecks.size());
    wrecks.push_back(Wreck{px, pz, radius, metal, 1.0f, wreck_hp, feature_def, heap_of});
    live_wrecks.push_back(id);
    wreck_index_stale = true;
    const int gx = std::clamp(int(px / GRID_CELL), 0, grid_cols - 1), gz = std::clamp(int(pz / GRID_CELL), 0, grid_rows - 1);
    wreck_cells[size_t(gz) * size_t(grid_cols) + size_t(gx)].push_back(id);
    return id;
}

void World::remove_wreck(int id) {
    Wreck& wreck = wrecks[size_t(id)];
    wreck.left = 0.0f;
    wreck_index_stale = true;
    const int gx = std::clamp(int(wreck.x / GRID_CELL), 0, grid_cols - 1), gz = std::clamp(int(wreck.z / GRID_CELL), 0, grid_rows - 1);
    auto& cell = wreck_cells[size_t(gz) * size_t(grid_cols) + size_t(gx)];
    cell.erase(std::find(cell.begin(), cell.end(), id));
    live_wrecks.erase(std::find(live_wrecks.begin(), live_wrecks.end(), id));
}

// Scripts' Killed give level 1 up to half its health in recent damage and level 2 above it, with a few exceptions that aren't followed.
// A frame killed part way leaves part of a heap, and nothing if it had hardly begun.
void World::leave_remains(int u) {
    const size_t i = size_t(u);
    const auto& d = type_of(u);
    if (build[i] >= 1.0f) {
        if (recent_damage[i] <= WRECK_ABOVE * d.hp && d.wreck_metal > 0.0f)
            add_wreck(x[i], z[i], d.wreck_metal, d.wreck_hp, d.footprint_radius, d.wreck_feature, type[i]);
        else if (d.heap_metal > 0.0f)
            add_wreck(x[i], z[i], d.heap_metal, d.heap_hp, d.footprint_radius, d.heap_feature, -1);
    } else if (build[i] > NANOFRAME_LEAVES_ABOVE && d.heap_metal > 0.0f) {
        Wreck& heap = wrecks[size_t(add_wreck(x[i], z[i], d.heap_metal, d.heap_hp * build[i], d.footprint_radius, d.heap_feature, -1))];
        heap.left = build[i];
    }
}

// CFeature::DoDamage: features have no armour class, so the default damage.  A wreck that's destroyed leaves a fresh heap, and a heap nothing.
void World::hurt_wrecks(const zk_weapon_rec& w, float px, float pz, float power) {
    if (live_wrecks.empty()) return;
    const float radius = std::max(w.blast_radius, 1.0f);
    const int x0 = std::max(0, int((px - radius) / GRID_CELL) - 1), x1 = std::min(grid_cols - 1, int((px + radius) / GRID_CELL) + 1);
    const int z0 = std::max(0, int((pz - radius) / GRID_CELL) - 1), z1 = std::min(grid_rows - 1, int((pz + radius) / GRID_CELL) + 1);
    for (int gz = z0; gz <= z1; ++gz)
        for (int gx = x0; gx <= x1; ++gx) {
            const auto& cell = wreck_cells[size_t(gz) * size_t(grid_cols) + size_t(gx)];
            // Backwards, since this can take the one it's looking at out of the cell.
            for (size_t k = cell.size(); k-- > 0;) {
                const int id = cell[k];
                Wreck& wreck = wrecks[size_t(id)];
                const float surface = std::max(0.0f, std::hypot(wreck.x - px, wreck.z - pz) - wreck.radius);
                if (surface >= radius) continue;
                const float rim = surface * w.edge;
                const float mod = (w.edge >= 1.0f || surface < 1.0f) ? 1.0f : (radius + 0.001f - surface) / (radius + 0.001f - rim);
                wreck.hp -= w.damage[0] * mod * power;
                if (wreck.hp > 0.0f) continue;
                if (wreck.heap_of < 0 || cat->types[size_t(wreck.heap_of)].heap_metal <= 0.0f) {
                    remove_wreck(id);
                } else {
                    const auto& d = cat->types[size_t(wreck.heap_of)];
                    // CFeature::CreateWreck: what's been reclaimed of a wreck is gone from its heap too.
                    wreck.metal = d.heap_metal;
                    wreck.hp = d.heap_hp;
                    wreck.feature_def = d.heap_feature;
                    wreck.heap_of = -1;
                }
            }
        }
}

// CFeature::AddBuildPower with reclaimMethod 0: gradual, and several builders add up.  It's paid as income, and what storage can't hold is lost with the rest of it.
void World::reclaim_step(int u) {
    const size_t i = size_t(u);
    const int id = order_target[i];
    if (!wreck_exists(id)) {
        order[i] = ZK_ORDER_NONE;
        return;
    }
    Wreck& wreck = wrecks[size_t(id)];
    if (!within(u, wreck.x, wreck.z, reach_of(u, wreck.radius))) return;
    show_for_action(u);
    const float taken = std::min(wreck.left, type_of(u).reclaim_speed * TICK_SECONDS / wreck.metal);
    purses[team[i]].reclaimed += wreck.metal * taken;
    wreck.left -= taken;
    if (wreck.left <= 0.0f) {
        remove_wreck(id);
        order[i] = ZK_ORDER_NONE;
    }
}

}  // namespace zk
