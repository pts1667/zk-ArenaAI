// zkpolicy.h's C functions over the C++ of table.cpp, bundle.cpp, featurize.cpp and actions.cpp.
#include <algorithm>
#include <atomic>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "../zkpolicy.h"
#include "actions.hpp"
#include "bundle.hpp"
#include "npy.hpp"
#include "policy.hpp"
#include "table.hpp"

using namespace zkp;

static_assert(sizeof(zkp_bundle_arrays) == sizeof(BundleArrays), "zkp_bundle_arrays and BundleArrays disagree");
static_assert(sizeof(zkp_order) == sizeof(Order), "zkp_order and Order disagree");
static_assert(int(ZKP_FIELDS) == int(F_COUNT), "the fields disagree");
static_assert(sizeof(zkp_tensors) == sizeof(TensorsView), "zkp_tensors and TensorsView disagree");

namespace {

Frame frame_of(const zkp_frame& f) {
    Frame out;
    out.units = static_cast<const UnitRecord*>(f.units);
    out.n_units = f.n_units;
    out.sites = static_cast<const SiteRecord*>(f.sites);
    out.n_sites = f.n_sites;
    out.features = static_cast<const FeatureRecord*>(f.features);
    out.n_features = f.n_features;
    out.glob = f.glob;
    out.bundle = static_cast<const Bundle*>(f.bundle);
    return out;
}

const TensorsView& view(const zkp_tensors* t) { return *reinterpret_cast<const TensorsView*>(t); }

}  // namespace

extern "C" {

int zkp_abi_version(void) { return ZKP_ABI_VERSION; }

void* zkp_table_new(const void* bytes, int64_t size, char* error, int32_t error_size) {
    auto* t = new Table();
    std::string why;
    if (!load_table(*t, bytes, size, why)) {
        delete t;
        if (error && error_size > 0) {
            std::strncpy(error, why.c_str(), size_t(error_size) - 1);
            error[error_size - 1] = '\0';
        }
        return nullptr;
    }
    return t;
}

void zkp_table_free(void* table) { delete static_cast<Table*>(table); }

void zkp_table_info(const void* table, int64_t* out) {
    const Table& t = *static_cast<const Table*>(table);
    out[0] = t.rows;
    out[1] = t.type_dim;
    out[2] = t.unit_dim;
    out[3] = t.producible_count;
    out[4] = t.unknown;
    out[5] = t.roles;
    out[6] = t.choosable;
    out[7] = int64_t(t.row_of_def.size());
}

void* zkp_bundle_new(const zkp_bundle_arrays* arrays) {
    auto* b = new Bundle();
    BundleArrays a;
    std::memcpy(&a, arrays, sizeof a);
    pack_bundle(*b, a);
    return b;
}

void zkp_bundle_free(void* bundle) { delete static_cast<Bundle*>(bundle); }

void zkp_bounds(const void* table, const zkp_frame* frames, int64_t n, int32_t privileged, int64_t* out) {
    const Table& t = *static_cast<const Table*>(table);
    for (int64_t k = 0; k < n; ++k) out[k] = bound(t, frame_of(frames[k]), privileged != 0);
}

int64_t zkp_featurize(const void* table, const zkp_frame* frames, int64_t n, const zkp_options* options, uint8_t* const* buffers,
                      const int64_t* capacities, int64_t* layouts, int64_t* controllable, int32_t threads) {
    const Table& t = *static_cast<const Table*>(table);
    Options opt;
    opt.commit_decisions = options->commit_decisions;
    opt.privileged = options->privileged != 0;
    opt.economy = options->economy != 0;
    for (int k = 0; k < KIND_COUNT; ++k) opt.enabled[k] = options->enabled[k] != 0;
    std::atomic<int64_t> next{0}, failed{0};
    auto work = [&]() {
        for (int64_t k; (k = next.fetch_add(1)) < n;) {
            const Frame f = frame_of(frames[k]);
            int64_t* layout = layouts + k * F_COUNT * LAYOUT;
            if (!f.bundle || !f.glob || featurize(t, f, opt, buffers[k], capacities[k], layout, controllable + k) != 0) failed.fetch_add(1);
        }
    };
    const int64_t workers = std::min<int64_t>(std::max<int32_t>(threads, 1), n);
    if (workers <= 1) {
        work();
    } else {
        std::vector<std::thread> pool;
        pool.reserve(workers);
        for (int64_t w = 0; w < workers; ++w) pool.emplace_back(work);
        for (auto& th : pool) th.join();
    }
    return failed.load();
}

int64_t zkp_decode(const void* table, const void* bundle, const zkp_tensors* t, const int64_t* decision, zkp_order* out) {
    std::vector<Order> orders;
    const int64_t got = decode(*static_cast<const Table*>(table), *static_cast<const Bundle*>(bundle), view(t), decision, orders);
    if (got < 0) return got;
    if (got) std::memcpy(out, orders.data(), sizeof(Order) * size_t(got));
    return got;
}

int64_t zkp_encode(const void* table, const void* bundle, const zkp_tensors* t, const zkp_order* orders, int64_t n, int64_t* decision) {
    return encode(*static_cast<const Table*>(table), view(t), *static_cast<const Bundle*>(bundle), reinterpret_cast<const Order*>(orders), n,
                  decision);
}

void zkp_fine_point(const void* bundle, const zkp_tensors* t, int64_t region, int64_t option, double* out2) {
    const Point p = fine_point(*static_cast<const Bundle*>(bundle), view(t), region, option);
    out2[0] = p.x;
    out2[1] = p.z;
}

void zkp_world_positions(const float* unit_x, int64_t rows, int64_t unit_dim, int64_t at, double map_w, double map_h, float* x, float* z) {
    const float w = float(map_w), h = float(map_h);
    for (int64_t k = 0; k < rows; ++k) {
        x[k] = (unit_x[k * unit_dim + at] + 1.0f) * 0.5f * w;
        z[k] = (unit_x[k * unit_dim + at + 1] + 1.0f) * 0.5f * h;
    }
}

int64_t zkp_acting(const int64_t* decision, int64_t controllable) {
    int64_t n = 0;
    for (int64_t i = 0; i < controllable; ++i) n += decision[i] != KEEP;
    return n;
}

int64_t zkp_to_slots(const int64_t* d, int64_t c, const int64_t* order, int64_t width, int64_t* out) {
    std::vector<int64_t> acting;
    for (int64_t i = 0; i < c; ++i)
        if (d[i] != KEEP) acting.push_back(i);
    if (order) {
        std::vector<int64_t> rank(c, 0);
        for (int64_t k = 0; k < c; ++k) rank[order[k]] = k;
        std::stable_sort(acting.begin(), acting.end(), [&](int64_t a, int64_t b) { return rank[a] < rank[b]; });
    }
    if (width < 0) width = int64_t(acting.size());
    if (int64_t(acting.size()) > width) acting.resize(width);
    static const int64_t UNSET[1 + DECISION_FIELDS] = {-1, KEEP, -1, -1, -1, -1, -1, 0, -1, -1};
    for (int f = 0; f <= DECISION_FIELDS; ++f)
        for (int64_t k = 0; k < width; ++k) out[f * width + k] = UNSET[f];
    for (size_t k = 0; k < acting.size(); ++k) {
        out[k] = acting[k];
        for (int f = 0; f < DECISION_FIELDS; ++f) out[(f + 1) * width + int64_t(k)] = d[f * c + acting[k]];
    }
    return width;
}

void zkp_to_decision(const int64_t* s, int64_t width, int64_t c, int64_t* d) {
    static const int64_t UNSET[DECISION_FIELDS] = {KEEP, -1, -1, -1, -1, -1, 0, -1, -1};
    for (int f = 0; f < DECISION_FIELDS; ++f)
        for (int64_t i = 0; i < c; ++i) d[f * c + i] = UNSET[f];
    for (int64_t k = 0; k < width; ++k) {
        const int64_t a = s[k];
        if (a < 0) continue;
        for (int f = 0; f < DECISION_FIELDS; ++f) d[f * c + a] = s[(f + 1) * width + k];
    }
}

int64_t zkp_group_bucket(double radius) { return bucket(radius); }

void zkp_group_expand(const void* table, int64_t anchor, int64_t roles_mask, double radius, const void* units, int64_t n,
                      const int64_t* rows, uint8_t* out) {
    expand(*static_cast<const Table*>(table), anchor, roles_mask, radius, static_cast<const UnitRecord*>(units), n, rows, out);
}

}  // extern "C"
