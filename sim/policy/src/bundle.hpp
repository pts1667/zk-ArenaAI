#pragma once

#include "policy.hpp"

namespace zkp {

// bundle.Bundle's arrays as zkpolicy.h's zkp_bundle_arrays gives them: C order, these types.
struct BundleArrays {
    double map_w, map_h, cell;
    int32_t label_h, label_w, pass_rows, pass_cols, regions, gates;
    const int16_t* label;           // label_h x label_w
    const float* height;            // label_h x label_w
    const uint8_t* passable;        // pass_rows x pass_cols
    const float* centre;            // regions x 2
    const float* area;              // regions
    const float* region_height;     // regions x 2
    const float* sea;               // regions
    const float* tags;              // regions x 9
    const float* access;            // regions x 6
    const float* shares;            // regions x 2
    const float* spots;             // regions x 2
    const float* gate_counts;       // regions x 2
    const float* gate_x;            // gates x GATE_DIM
    const float* walk;              // MOVE_CLASSES x regions x regions
};

void pack_bundle(Bundle& b, const BundleArrays& a);

}  // namespace zkp
