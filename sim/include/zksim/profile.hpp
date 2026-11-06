// Timers per tick phase, compiled in only with -DZKSIM_PROFILE=ON.  Counters belong to the calling thread, so
// profile single-threaded runs (tools/sim_profile.py); zk_replay's workers would keep their own.
#pragma once

#ifdef ZKSIM_PROFILE
#include <chrono>

namespace zk::prof {

// Dots show nesting: a child's time is also in its parent's.
enum Phase {
    GRID, NEAREST, TARGETS, FIRE, SHOTS, BURN, DEATHS, STATUS,
    MOVE, MOVE_WANT, MOVE_WAYPOINT, WAYPOINT_LINE, WAYPOINT_ROUTE, MOVE_FRAMES, MOVE_SEPARATE, FOG,
    PHASE_COUNT
};

inline constexpr const char* NAMES[PHASE_COUNT] = {
    "grid", "nearest", "targets", "fire", "shots", "burn", "deaths", "status",
    "move", "move.wants_to_walk", "move.waypoint", "move.waypoint.goal_line", "move.waypoint.route", "move.frames", "move.separate", "fog",
};

inline thread_local long long nanos[PHASE_COUNT];
inline thread_local long long calls[PHASE_COUNT];

class Scope {
public:
    explicit Scope(Phase p) : phase(p), start(std::chrono::steady_clock::now()) {}
    ~Scope() {
        nanos[phase] += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
        ++calls[phase];
    }

private:
    Phase phase;
    std::chrono::steady_clock::time_point start;
};

}  // namespace zk::prof

#define ZK_PROF_JOIN2(a, b) a##b
#define ZK_PROF_JOIN(a, b) ZK_PROF_JOIN2(a, b)
#define ZK_PROF(phase) ::zk::prof::Scope ZK_PROF_JOIN(zk_prof_, __LINE__)(::zk::prof::phase)
#else
#define ZK_PROF(phase) ((void)0)
#endif
