// What featurize makes of a map's bundle alone (featurize._bundle_pack): the regions' static columns, the gates' normalised
// columns, each region's half equivalent radius and the passable bits at its centre.
#include <algorithm>
#include <cmath>

#include "bundle.hpp"
#include "npy.hpp"

namespace zkp {

void pack_bundle(Bundle& b, const BundleArrays& a) {
    b.map_w_d = a.map_w;
    b.map_h_d = a.map_h;
    b.cell_d = a.cell;
    b.map_w = float(a.map_w);
    b.map_h = float(a.map_h);
    b.cell = float(a.cell);
    b.map_max = float(a.map_w > a.map_h ? a.map_w : a.map_h);
    b.label_h = a.label_h;
    b.label_w = a.label_w;
    b.pass_rows = a.pass_rows;
    b.pass_cols = a.pass_cols;
    b.regions = a.regions;
    b.gates = a.gates;
    b.label = a.label;
    b.height = a.height;
    b.passable = a.passable;
    b.centre = a.centre;
    b.walk = a.walk;
    const int64_t R = a.regions, G = a.gates;
    b.region_cells.resize(R);
    b.region_radius.resize(R);
    b.reg_x.assign(R * REGION_DIM, 0.0f);
    for (int64_t k = 0; k < R; ++k) {
        const float cx = a.centre[k * 2], cz = a.centre[k * 2 + 1];
        const int64_t i = std::min<int64_t>(std::max<int64_t>(to_i64(cz / PASS_CELL), 0), a.pass_rows - 1);
        const int64_t j = std::min<int64_t>(std::max<int64_t>(to_i64(cx / PASS_CELL), 0), a.pass_cols - 1);
        b.region_cells[k] = a.passable[i * a.pass_cols + j];
        b.region_radius[k] = std::sqrt(a.area[k] / float(M_PI)) * 0.5f;
        float* o = b.reg_x.data() + k * REGION_DIM;
        o[0] = cx / b.map_w * 2.0f - 1.0f;
        o[1] = cz / b.map_h * 2.0f - 1.0f;
        o[2] = ::log1pf(a.area[k]) / 16.0f;
        o[3] = a.region_height[k * 2] / 300.0f;
        o[4] = a.region_height[k * 2 + 1] / 300.0f;
        o[5] = a.sea[k];
        for (int t = 0; t < 9; ++t) o[6 + t] = a.tags[k * 9 + t];
        for (int t = 0; t < 6; ++t) o[15 + t] = a.access[k * 6 + t];
        o[21] = a.shares[k * 2];
        o[22] = a.shares[k * 2 + 1];
        o[23] = a.spots[k * 2] / 5.0f;
        o[24] = a.spots[k * 2 + 1] / 10.0f;
        o[25] = a.gate_counts[k * 2] / 4.0f;
        o[26] = a.gate_counts[k * 2 + 1] / 4.0f;
    }
    b.gate_x.resize(G * GATE_DIM);
    for (int64_t k = 0; k < G; ++k) {
        const float* g = a.gate_x + k * GATE_DIM;
        float* o = b.gate_x.data() + k * GATE_DIM;
        for (int c = 0; c < 6; ++c) o[c] = ::log1pf(g[c]) / 8.0f;
        for (int c = 6; c < 12; ++c) o[c] = g[c] / 2000.0f;
        for (int c = 12; c < GATE_DIM; ++c) o[c] = g[c];
    }
}

}  // namespace zkp
