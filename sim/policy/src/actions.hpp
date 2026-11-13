#pragma once

#include <vector>

#include "policy.hpp"

namespace zkp {

// actions.Order: target a uid, or for PRODUCE and BUILD a type-table row; a radius makes it a group order.
struct Order {
    int64_t uid, kind;
    double x, z;
    int64_t target;
    double radius;
    int64_t roles;
};

// A point, as Python's floats.
struct Point {
    double x, z;
};

constexpr int GROUP_COUNT = 10;
constexpr double GROUP_RADII[GROUP_COUNT] = {60.0, 90.0, 135.0, 200.0, 300.0, 450.0, 675.0, 1000.0, 1500.0, 2250.0};
inline bool group_kind(int64_t k) { return k == MOVE || k == FIGHT || k == ATTACK_UNIT; }

// A decision is 9 arrays of `controllable` int64, in this order: actions.FIELDS.
constexpr int DECISION_FIELDS = 9;

Point fine_point(const Bundle& b, const TensorsView& t, int64_t region, int64_t option);
// The orders, or -1 - i for the first unit i whose choices point past what the tensors hold (where Python raises).
int64_t decode(const Table& T, const Bundle& B, const TensorsView& t, const int64_t* decision, std::vector<Order>& out);
// Writes the decision; returns how many orders couldn't be said.
int64_t encode(const Table& T, const TensorsView& t, const Bundle& B, const Order* orders, int64_t n, int64_t* decision);
int64_t bucket(double radius);
void expand(const Table& T, int64_t anchor, int64_t roles_mask, double radius, const UnitRecord* units, int64_t n, const int64_t* rows,
            uint8_t* out);

}  // namespace zkp
