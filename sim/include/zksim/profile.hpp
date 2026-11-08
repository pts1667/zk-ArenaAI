// Timers per tick phase, compiled in only with -DZKSIM_PROFILE=ON.  A timer adds to the counters its thread has been pointed
// at: a batch points each of its workers at its own (zk_batch_prof_read sums them), and anything else at the thread's own
// (zk_prof_read), so batches stepping at once never share counters.
#pragma once

#ifdef ZKSIM_PROFILE
#include <chrono>

namespace zk::prof {

// Dots show nesting: a child's time is also in its parent's.
enum Phase {
    GRID, CLOAK, NEAREST, TARGETS, FIRE, SHOTS, BURN, DEATHS, STATUS, ECONOMY,
    MOVE, MOVE_WANT, MOVE_WAYPOINT, WAYPOINT_LINE, WAYPOINT_ROUTE, MOVE_FRAMES, MOVE_SEPARATE, FOG, UNIT_AI,
    KAMIKAZE, ORDERS, EMIT, FEATURES, STATS, FOG_SWEEP, FOG_GONE, AI_ERROR, AI_COMMANDS, AI_TACTICS,
    FEATURES_INDEX, FEATURES_SWEEP, MOVE_ABILITIES,
    PHASE_COUNT
};

inline constexpr const char* NAMES[PHASE_COUNT] = {
    "grid", "cloak", "nearest", "targets", "fire", "shots", "burn", "deaths", "status", "economy",
    "move", "move.wants_to_walk", "move.waypoint", "move.waypoint.goal_line", "move.waypoint.route", "move.frames", "move.separate", "fog", "unit_ai",
    "kamikaze", "orders", "emit_view", "emit_features", "side_stats", "fog.sweep", "fog.gone", "unit_ai.radar_error", "unit_ai.commands",
    "unit_ai.tactics", "emit_features.index", "emit_features.sweep", "move.abilities",
};

struct Counters {
    long long nanos[PHASE_COUNT] = {};
    long long calls[PHASE_COUNT] = {};
    void add_to(long long* n, long long* c) const {
        for (int i = 0; i < PHASE_COUNT; ++i) {
            n[i] += nanos[i];
            c[i] += calls[i];
        }
    }
    void clear() { *this = Counters{}; }
};

inline thread_local Counters own;
inline thread_local Counters* target = nullptr;     // or own

inline Counters& current() { return target ? *target : own; }

// Points this thread's timers at other counters for as long as it lives.
class Redirect {
public:
    explicit Redirect(Counters* to) : before(target) { target = to; }
    ~Redirect() { target = before; }
    Redirect(const Redirect&) = delete;
    Redirect& operator=(const Redirect&) = delete;

private:
    Counters* before;
};

class Scope {
public:
    explicit Scope(Phase p) : phase(p), into(current()), start(std::chrono::steady_clock::now()) {}
    ~Scope() {
        into.nanos[phase] += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
        ++into.calls[phase];
    }

private:
    Phase phase;
    Counters& into;
    std::chrono::steady_clock::time_point start;
};

}  // namespace zk::prof

#define ZK_PROF_JOIN2(a, b) a##b
#define ZK_PROF_JOIN(a, b) ZK_PROF_JOIN2(a, b)
#define ZK_PROF(phase) ::zk::prof::Scope ZK_PROF_JOIN(zk_prof_, __LINE__)(::zk::prof::phase)
#else
#define ZK_PROF(phase) ((void)0)
#endif
