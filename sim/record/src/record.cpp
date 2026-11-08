// libzkrecord: zkrecord.h says what it writes.
#include "zkrecord.h"

#include <zstd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include "zksim/rawframe.hpp"

namespace {

using zk::FeatureRecord;
using zk::SiteRecord;
using zk::UnitRecord;

constexpr char MAGIC[8] = {'Z', 'K', 'R', 'E', 'C', '0', '0', '1'};
constexpr int64_t KEY_LIMIT = int64_t(1) << 24;     // uids and feature ids are handles: past this one isn't matched

enum Clamp { C_X, C_Z, C_Y, C_ORD_X, C_ORD_Z, C_BUILD, C_ORD_AGE, C_HALF, C_FEATURE_XZ };
static_assert(C_FEATURE_XZ + 1 == ZKR_CLAMPS, "zkr_clamps");

// The narrow records, field by field in the order they're stored.  sim/py/zkrecord.py has the same lists.
#define ZKR_UNIT_FIELDS(X)                                                                                                       \
    X(int32_t, uid) X(int32_t, ord_target) X(int16_t, def_id) X(uint16_t, x) X(uint16_t, z) X(uint16_t, y) X(uint16_t, vx)      \
    X(uint16_t, vz) X(uint16_t, hp_frac) X(uint16_t, build_frac) X(uint16_t, experience) X(uint16_t, blip_top_speed)              \
    X(uint16_t, ord_age) X(uint16_t, ord_x) X(uint16_t, ord_z) X(uint16_t, flags) X(uint8_t, knowledge) X(int8_t, ord_kind)      \
    X(int8_t, player) X(int8_t, ally_team)
#define ZKR_FEATURE_FIELDS(X) X(int32_t, feat_id) X(uint16_t, x) X(uint16_t, z) X(uint32_t, metal) X(int16_t, def_id)
#define ZKR_ORDER_FIELDS(X) X(int32_t, unit) X(int32_t, kind) X(float, x) X(float, z) X(int32_t, target)

#define ZKR_DECLARE(type, name) type name;
struct NarrowUnit {
    ZKR_UNIT_FIELDS(ZKR_DECLARE)
};
struct NarrowFeature {
    ZKR_FEATURE_FIELDS(ZKR_DECLARE)
};
static_assert(sizeof(NarrowUnit) == 40, "a narrow unit is 40 bytes");

template <class T>
T wrap_sub(T a, T b) {
    using U = std::make_unsigned_t<T>;
    return T(U(U(a) - U(b)));
}

template <class T>
void residual(T& out, const T& cur, const T& prev);
#define ZKR_SUB(type, name) out.name = wrap_sub(cur.name, prev.name);
template <>
void residual(NarrowUnit& out, const NarrowUnit& cur, const NarrowUnit& prev) {
    ZKR_UNIT_FIELDS(ZKR_SUB)
}
template <>
void residual(NarrowFeature& out, const NarrowFeature& cur, const NarrowFeature& prev) {
    ZKR_FEATURE_FIELDS(ZKR_SUB)
}

// Round to nearest even, as numpy's float32 -> float16.  NaN stays NaN.
uint16_t to_half(float f, int64_t* clamps) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u, a = x & 0x7fffffffu;
    if (a >= 0x7f800000u) return uint16_t(sign | 0x7c00u | (a > 0x7f800000u ? 0x200u | ((a >> 13) & 0x3ffu) : 0u));
    if (a >= 0x477ff000u) {     // 65520 and up round to infinity
        ++clamps[C_HALF];
        return uint16_t(sign | 0x7c00u);
    }
    if (a < 0x38800000u) {      // below the smallest normal half: a multiple of 2^-24
        float v;
        std::memcpy(&v, &a, 4);
        return uint16_t(sign | uint32_t(std::lrintf(v * 16777216.0f)));
    }
    const uint32_t e = a - 0x38000000u;
    return uint16_t(sign | ((e + 0xfffu + ((e >> 13) & 1u)) >> 13));
}

// v in [lo, lo + span] as 16 bits; clamped and counted outside it (by more than rounding), and NaN as lo.
uint16_t fraction(float v, double lo, double span, int64_t* clamps, int which) {
    double t = (double(v) - lo) / span;
    if (!(t >= 0.0)) {
        if (!(t > -1e-6)) ++clamps[which];
        t = 0.0;
    } else if (t > 1.0) {
        if (t > 1.0 + 1e-6) ++clamps[which];
        t = 1.0;
    }
    return uint16_t(std::lrint(t * 65535.0));
}

// A pool of threads kept between calls, as libzkscripts' (sim/scripts/src/api.cpp).  The calling thread works too.
class Pool {
public:
    explicit Pool(int threads) {
        for (int t = 1; t < threads; t++) workers_.emplace_back([this] { loop(); });
    }
    ~Pool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
            generation_++;
        }
        wake_.notify_all();
        for (auto& w : workers_) w.join();
    }
    void run(int items, const std::function<void(int)>& fn) {
        if (workers_.empty() || items <= 1) {
            for (int i = 0; i < items; i++) fn(i);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            fn_ = &fn;
            items_ = items;
            next_.store(0);
            busy_ = (int)workers_.size();
            generation_++;
        }
        wake_.notify_all();
        work();
        std::unique_lock<std::mutex> lock(mutex_);
        done_.wait(lock, [this] { return busy_ == 0; });
        fn_ = nullptr;
    }

private:
    void work() {
        for (int i; (i = next_.fetch_add(1)) < items_;) (*fn_)(i);
    }
    void loop() {
        uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                wake_.wait(lock, [&] { return generation_ != seen; });
                seen = generation_;
                if (stop_) return;
            }
            work();
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (--busy_ == 0) done_.notify_one();
            }
        }
    }
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable wake_, done_;
    const std::function<void(int)>* fn_ = nullptr;
    std::atomic<int> next_{0};
    int items_ = 0, busy_ = 0;
    uint64_t generation_ = 0;
    bool stop_ = false;
};

// Where each key (uid, feature id) was in the last frame: a table by key, with stamps so that clearing is free.
class KeyIndex {
public:
    void clear() {
        if (++generation_ == 0) {
            std::fill(stamp_.begin(), stamp_.end(), 0u);
            generation_ = 1;
        }
    }
    void put(int64_t key, int at) {
        if (key < 0 || key >= KEY_LIMIT) return;
        const size_t k = size_t(key);
        if (k >= at_.size()) {
            const size_t size = std::max(k + 1, at_.size() * 2);
            at_.resize(size);
            stamp_.resize(size, 0u);
        }
        if (stamp_[k] == generation_) return;       // a key twice in a frame: the first is the one matched
        stamp_[k] = generation_;
        at_[k] = at;
    }
    int get(int64_t key) const {
        if (key < 0 || size_t(key) >= at_.size()) return -1;
        return stamp_[size_t(key)] == generation_ ? at_[size_t(key)] : -1;
    }

private:
    std::vector<int> at_;
    std::vector<uint32_t> stamp_;
    uint32_t generation_ = 1;
};

// A side's run: what its last frame was, for the residuals, and what the open chunk holds.
struct Side {
    std::vector<NarrowUnit> last_units;
    std::vector<NarrowFeature> last_features;
    KeyIndex unit_index, feature_index;
    uint32_t last_glob[ZKR_GLOBALS] = {};
    // The open chunk.
    std::vector<int32_t> unit_counts, feature_counts, order_counts;
    std::vector<uint32_t> globs;            // by frame, ZKR_GLOBALS each
    std::vector<NarrowUnit> units;
    std::vector<int32_t> unit_back;
    std::vector<NarrowFeature> features;
    std::vector<int32_t> feature_back;
    std::vector<zk_order_rec> orders;

    void clear_chunk() {
        unit_counts.clear();
        feature_counts.clear();
        order_counts.clear();
        globs.clear();
        units.clear();
        unit_back.clear();
        features.clear();
        feature_back.clear();
        orders.clear();
    }
    void clear() {
        clear_chunk();
        last_units.clear();
        last_features.clear();
        unit_index.clear();
        feature_index.clear();
        std::memset(last_glob, 0, sizeof(last_glob));
    }
};

struct Slot {
    bool on = false;
    float map_w = 1.0f, map_h = 1.0f, y_lo = 0.0f, y_hi = 1.0f;
    std::vector<SiteRecord> sites;
    Side sides[2];
    int frames = 0, chunks = 0;
    std::vector<uint8_t> body;              // the compressed chunks so far, each with its length in front
    std::vector<uint8_t> raw;               // a chunk before compressing
    int64_t clamps[ZKR_CLAMPS] = {};
};

struct CCtxFree {
    void operator()(ZSTD_CCtx* c) const { ZSTD_freeCCtx(c); }
};

ZSTD_CCtx* thread_cctx() {
    thread_local std::unique_ptr<ZSTD_CCtx, CCtxFree> cctx(ZSTD_createCCtx());
    return cctx.get();
}

void put_planes(std::vector<uint8_t>& out, const void* base, size_t stride, size_t size, size_t n) {
    const auto* p = static_cast<const uint8_t*>(base);
    const size_t at = out.size();
    out.resize(at + size * n);
    uint8_t* o = out.data() + at;
    for (size_t b = 0; b < size; b++)
        for (size_t i = 0; i < n; i++) *o++ = p[i * stride + b];
}

template <class T>
void put_raw(std::vector<uint8_t>& out, const std::vector<T>& v) {
    const size_t at = out.size();
    out.resize(at + v.size() * sizeof(T));
    if (!v.empty()) std::memcpy(out.data() + at, v.data(), v.size() * sizeof(T));
}

template <class T>
void put_value(std::vector<uint8_t>& out, T v) {
    const size_t at = out.size();
    out.resize(at + sizeof(T));
    std::memcpy(out.data() + at, &v, sizeof(T));
}

// A file to write: its front, then the compressed chunks.
struct Pending {
    std::string path;
    std::vector<uint8_t> head, body;
};

int write_file(const Pending& p) {
    const std::string part = p.path + ".part";
    int err = 0;
    if (FILE* f = std::fopen(part.c_str(), "wb")) {
        if (std::fwrite(p.head.data(), 1, p.head.size(), f) != p.head.size() ||
            (!p.body.empty() && std::fwrite(p.body.data(), 1, p.body.size(), f) != p.body.size()))
            err = errno ? errno : EIO;
        if (std::fclose(f) != 0 && !err) err = errno ? errno : EIO;
        if (!err && std::rename(part.c_str(), p.path.c_str()) != 0) err = errno ? errno : EIO;
        if (err) std::remove(part.c_str());
    } else {
        err = errno ? errno : EIO;
    }
    return err;
}

class Recorder {
public:
    Recorder(int slots, int threads, int level, int chunk_frames)
        : slots_(size_t(slots)), pool_(std::max(1, threads)), level_(level), chunk_frames_(std::max(1, chunk_frames)),
          writer_([this] { write_loop(); }) {}
    ~Recorder() {
        {
            std::lock_guard<std::mutex> lock(write_mutex_);
            stop_ = true;
        }
        write_wake_.notify_all();
        writer_.join();         // after writing what's queued
    }

    int wait() {
        std::unique_lock<std::mutex> lock(write_mutex_);
        write_idle_.wait(lock, [this] { return queue_.empty() && !writing_; });
        const int err = write_error_;
        write_error_ = 0;
        return err;
    }

    void begin(int s, float map_w, float map_h, float y_lo, float y_hi, const void* sites, int n_sites) {
        Slot& slot = slots_[size_t(s)];
        slot.on = true;
        slot.map_w = map_w;
        slot.map_h = map_h;
        slot.y_lo = y_lo;
        slot.y_hi = y_hi > y_lo ? y_hi : y_lo + 1.0f;
        const auto* p = static_cast<const SiteRecord*>(sites);
        slot.sites.assign(p, p + std::max(0, n_sites));
        for (Side& side : slot.sides) side.clear();
        slot.frames = slot.chunks = 0;
        slot.body.clear();
    }

    void frames(const zkr_frames_args& a) {
        pool_.run(a.n, [&](int k) {
            const int s = a.games[k];
            Slot& slot = slots_[size_t(s)];
            if (!slot.on) return;
            for (int side = 0; side < 2; side++) {
                const size_t seat = size_t(s) * 2 + size_t(side);
                const auto* units = static_cast<const UnitRecord*>(a.records) + seat * size_t(a.cap);
                const auto* features = static_cast<const FeatureRecord*>(a.features) + seat * size_t(a.feature_cap);
                add_frame(slot, slot.sides[side], units, a.counts[seat], a.globs + seat * ZKR_GLOBALS, features, a.feature_counts[seat]);
            }
            slot.frames++;
        });
    }

    void orders(int n, const int32_t* games, const zk_order_rec* orders, const int32_t* offsets) {
        pool_.run(n, [&](int k) {
            const int s = games[k];
            Slot& slot = slots_[size_t(s)];
            if (!slot.on) return;
            for (int side = 0; side < 2; side++) {
                Side& sd = slot.sides[side];
                const size_t seat = size_t(s) * 2 + size_t(side);
                const int a = offsets[seat], b = offsets[seat + 1];
                // Orders for a frame not taken (or taken twice) would put the two out of step: the count follows the frames.
                if (sd.order_counts.size() >= sd.unit_counts.size()) continue;
                sd.order_counts.push_back(b - a);
                sd.orders.insert(sd.orders.end(), orders + a, orders + b);
            }
            if (int(slot.sides[0].unit_counts.size()) >= chunk_frames_) close_chunk(slot);
        });
    }

    int finish(int s, const char* path, const char* meta, int meta_len) {
        Slot& slot = slots_[size_t(s)];
        if (!slot.on) return -1;
        if (!slot.sides[0].unit_counts.empty()) close_chunk(slot);
        std::vector<uint8_t> head;
        head.insert(head.end(), MAGIC, MAGIC + 8);
        put_value<uint32_t>(head, uint32_t(std::max(0, meta_len)));
        head.insert(head.end(), meta, meta + std::max(0, meta_len));
        for (float v : {slot.map_w, slot.map_h, slot.y_lo, slot.y_hi}) put_value(head, v);
        for (int32_t v : {int32_t(slot.frames), int32_t(slot.chunks), int32_t(chunk_frames_), int32_t(slot.sites.size())}) put_value(head, v);
        std::vector<uint8_t> sites;
        put_raw(sites, slot.sites);
        compress_into(head, sites);
        {
            std::lock_guard<std::mutex> lock(write_mutex_);
            queue_.push_back(Pending{path, std::move(head), std::move(slot.body)});
        }
        write_wake_.notify_one();
        for (int c = 0; c < ZKR_CLAMPS; c++) clamps_[c] += slot.clamps[c];
        std::fill(slot.clamps, slot.clamps + ZKR_CLAMPS, 0);
        slot.on = false;
        for (Side& side : slot.sides) side.clear();
        slot.body = std::vector<uint8_t>();
        slot.frames = slot.chunks = 0;
        return 0;
    }

    int frames_held(int s) const { return slots_[size_t(s)].frames; }

    void clamps(int64_t* out) const {
        for (int c = 0; c < ZKR_CLAMPS; c++) {
            out[c] = clamps_[c];
            for (const Slot& slot : slots_) out[c] += slot.clamps[c];
        }
    }

private:
    NarrowUnit narrow(const Slot& slot, const UnitRecord& r, int64_t* clamps) const {
        NarrowUnit q;
        q.uid = int32_t(r.uid);
        q.ord_target = int32_t(r.ord_target);
        q.def_id = int16_t(r.def_id);
        q.x = fraction(r.x, 0.0, slot.map_w, clamps, C_X);
        q.z = fraction(r.z, 0.0, slot.map_h, clamps, C_Z);
        q.y = fraction(r.y, slot.y_lo, double(slot.y_hi) - slot.y_lo, clamps, C_Y);
        q.vx = to_half(r.vx, clamps);
        q.vz = to_half(r.vz, clamps);
        q.hp_frac = to_half(r.hp_frac, clamps);
        q.build_frac = fraction(r.build_frac, 0.0, 1.0, clamps, C_BUILD);
        q.experience = to_half(r.experience, clamps);
        q.blip_top_speed = to_half(r.blip_top_speed, clamps);
        float age = std::nearbyint(r.ord_age);
        if (!(age >= 0.0f && age <= 65535.0f) || age != r.ord_age) {
            ++clamps[C_ORD_AGE];
            age = age > 65535.0f ? 65535.0f : (age >= 0.0f ? age : 0.0f);
        }
        q.ord_age = uint16_t(age);
        q.ord_x = fraction(r.ord_x, -0.25 * slot.map_w, 1.5 * slot.map_w, clamps, C_ORD_X);
        q.ord_z = fraction(r.ord_z, -0.25 * slot.map_h, 1.5 * slot.map_h, clamps, C_ORD_Z);
        q.flags = r.flags;
        q.knowledge = r.knowledge;
        q.ord_kind = r.ord_kind;
        q.player = int8_t(r.player);
        q.ally_team = int8_t(r.ally_team);
        return q;
    }

    NarrowFeature narrow(const Slot& slot, const FeatureRecord& r, int64_t* clamps) const {
        NarrowFeature q;
        q.feat_id = int32_t(r.feat_id);
        q.x = fraction(r.x, 0.0, slot.map_w, clamps, C_FEATURE_XZ);
        q.z = fraction(r.z, 0.0, slot.map_h, clamps, C_FEATURE_XZ);
        std::memcpy(&q.metal, &r.metal, 4);
        q.def_id = int16_t(r.def_id);
        return q;
    }

    // Records as residuals against the side's last frame, keyed by `key`; then this frame becomes the last.
    template <class Narrow, class Raw, class Key>
    void add_records(Slot& slot, std::vector<Narrow>& last, KeyIndex& index, std::vector<Narrow>& out, std::vector<int32_t>& back,
                     const Raw* raw, int n, Key key) {
        std::vector<Narrow> now(size_t(std::max(0, n)));
        const int before = int(last.size());
        for (int i = 0; i < n; i++) {
            now[size_t(i)] = narrow(slot, raw[i], slot.clamps);
            const int j = index.get(key(raw[i]));
            Narrow r = now[size_t(i)];
            if (j >= 0) residual(r, now[size_t(i)], last[size_t(j)]);
            out.push_back(r);
            back.push_back(j >= 0 ? before - j + i : 0);
        }
        index.clear();
        for (int i = 0; i < n; i++) index.put(key(raw[i]), i);
        last.swap(now);
    }

    void add_frame(Slot& slot, Side& sd, const UnitRecord* units, int n_units, const float* glob, const FeatureRecord* features,
                   int n_features) {
        sd.unit_counts.push_back(n_units);
        sd.feature_counts.push_back(n_features);
        for (int c = 0; c < ZKR_GLOBALS; c++) {
            uint32_t bits;
            std::memcpy(&bits, glob + c, 4);
            sd.globs.push_back(bits - sd.last_glob[c]);
            sd.last_glob[c] = bits;
        }
        add_records(slot, sd.last_units, sd.unit_index, sd.units, sd.unit_back, units, n_units, [](const UnitRecord& r) { return r.uid; });
        add_records(slot, sd.last_features, sd.feature_index, sd.features, sd.feature_back, features, n_features,
                    [](const FeatureRecord& r) { return r.feat_id; });
    }

    void close_chunk(Slot& slot) {
        std::vector<uint8_t>& raw = slot.raw;
        raw.clear();
        Side& first = slot.sides[0];
        const int frames = int(first.unit_counts.size());
        put_value<int32_t>(raw, frames);
        for (Side& sd : slot.sides) {
            sd.order_counts.resize(sd.unit_counts.size(), 0);       // frames whose orders never came: none
            put_raw(raw, sd.unit_counts);
            put_raw(raw, sd.feature_counts);
            put_raw(raw, sd.order_counts);
            // Globals: a column each.
            for (int c = 0; c < ZKR_GLOBALS; c++) put_planes(raw, sd.globs.data() + c, ZKR_GLOBALS * 4, 4, size_t(frames));
            put_planes(raw, sd.unit_back.data(), 4, 4, sd.unit_back.size());
#define ZKR_PLANES_UNIT(type, name) \
    put_planes(raw, reinterpret_cast<const uint8_t*>(sd.units.data()) + offsetof(NarrowUnit, name), sizeof(NarrowUnit), sizeof(type), sd.units.size());
            ZKR_UNIT_FIELDS(ZKR_PLANES_UNIT)
            put_planes(raw, sd.feature_back.data(), 4, 4, sd.feature_back.size());
#define ZKR_PLANES_FEATURE(type, name)                                                                                    \
    put_planes(raw, reinterpret_cast<const uint8_t*>(sd.features.data()) + offsetof(NarrowFeature, name), sizeof(NarrowFeature), \
               sizeof(type), sd.features.size());
            ZKR_FEATURE_FIELDS(ZKR_PLANES_FEATURE)
#define ZKR_PLANES_ORDER(type, name) \
    put_planes(raw, reinterpret_cast<const uint8_t*>(sd.orders.data()) + offsetof(zk_order_rec, name), sizeof(zk_order_rec), sizeof(type), sd.orders.size());
            ZKR_ORDER_FIELDS(ZKR_PLANES_ORDER)
            sd.clear_chunk();
        }
        compress_into(slot.body, raw);
        slot.chunks++;
    }

    void compress_into(std::vector<uint8_t>& out, const std::vector<uint8_t>& raw) const {
        const size_t bound = ZSTD_compressBound(raw.size());
        const size_t at = out.size();
        out.resize(at + 4 + bound);
        const size_t n = ZSTD_compressCCtx(thread_cctx(), out.data() + at + 4, bound, raw.data(), raw.size(), level_);
        if (ZSTD_isError(n)) {
            std::fprintf(stderr, "zkrecord: %s\n", ZSTD_getErrorName(n));
            std::abort();
        }
        const uint32_t len = uint32_t(n);
        std::memcpy(out.data() + at, &len, 4);
        out.resize(at + 4 + n);
    }

    // The writer thread: files are written in the order finished, off the caller's thread (the disk can be slow: WSL's /mnt).
    void write_loop() {
        for (;;) {
            Pending p;
            {
                std::unique_lock<std::mutex> lock(write_mutex_);
                write_wake_.wait(lock, [this] { return stop_ || !queue_.empty(); });
                if (queue_.empty()) return;
                p = std::move(queue_.front());
                queue_.pop_front();
                writing_ = true;
            }
            const int err = write_file(p);
            {
                std::lock_guard<std::mutex> lock(write_mutex_);
                writing_ = false;
                if (err && !write_error_) write_error_ = err;
                if (queue_.empty()) write_idle_.notify_all();
            }
        }
    }

    std::vector<Slot> slots_;
    Pool pool_;
    int level_, chunk_frames_;
    int64_t clamps_[ZKR_CLAMPS] = {};
    std::mutex write_mutex_;
    std::condition_variable write_wake_, write_idle_;
    std::deque<Pending> queue_;
    bool writing_ = false, stop_ = false;
    int write_error_ = 0;
    std::thread writer_;        // last: it starts once the rest is made
};

}  // namespace

extern "C" {

int zkr_abi_version(void) { return ZKR_ABI_VERSION; }

void* zkr_new(int slots, int threads, int level, int chunk_frames) { return new Recorder(slots, threads, level, chunk_frames); }

void zkr_free(void* rec) { delete static_cast<Recorder*>(rec); }

void zkr_begin(void* rec, int slot, float map_w, float map_h, float y_lo, float y_hi, const void* sites, int n_sites) {
    static_cast<Recorder*>(rec)->begin(slot, map_w, map_h, y_lo, y_hi, sites, n_sites);
}

void zkr_frames(void* rec, const zkr_frames_args* args) { static_cast<Recorder*>(rec)->frames(*args); }

void zkr_orders(void* rec, int n, const int32_t* games, const zk_order_rec* orders, const int32_t* offsets) {
    static_cast<Recorder*>(rec)->orders(n, games, orders, offsets);
}

int zkr_finish(void* rec, int slot, const char* path, const char* meta, int meta_len) {
    return static_cast<Recorder*>(rec)->finish(slot, path, meta, meta_len);
}

int zkr_wait(void* rec) { return static_cast<Recorder*>(rec)->wait(); }

int zkr_frames_held(void* rec, int slot) { return static_cast<Recorder*>(rec)->frames_held(slot); }

void zkr_clamps(void* rec, int64_t* out) { static_cast<Recorder*>(rec)->clamps(out); }

int64_t zkr_frame_size(const void* src, int64_t len) {
    const unsigned long long n = ZSTD_getFrameContentSize(src, size_t(len));
    return n == ZSTD_CONTENTSIZE_UNKNOWN || n == ZSTD_CONTENTSIZE_ERROR ? -1 : int64_t(n);
}

int64_t zkr_decompress(const void* src, int64_t len, void* dst, int64_t cap) {
    const size_t n = ZSTD_decompress(dst, size_t(cap), src, size_t(len));
    return ZSTD_isError(n) ? -1 : int64_t(n);
}
}
