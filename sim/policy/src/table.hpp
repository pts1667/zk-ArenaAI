// The type table's flat file (zkpolicy.h, "The type table"), little-endian: this header, then each array 8-byte aligned in
// this order: stats f32 [rows x type_dim], f f32 [rows x T_COLS], b u8 [rows x B_COLS], move_class i8 [rows], produces u8
// [rows x producible], builds u8 [rows x producible], producible i64, row_of_def i32 [defs], def_of_row i32 [rows], role i32
// [rows], then the names and the role names, each ended by a NUL.
#pragma once

#include <cstdint>
#include <string>

#include "policy.hpp"

namespace zkp {

constexpr char TABLE_MAGIC[9] = "ZKPTABLE";
constexpr uint32_t TABLE_VERSION = 1;

struct TableHeader {
    char magic[8];
    uint32_t version, rows, type_dim, f_cols, b_cols, producible, defs, roles, names_bytes, role_names_bytes, unknown, choosable;
    char hash[16];
};
static_assert(sizeof(TableHeader) == 72, "TableHeader: sim/py/zkpolicy.py writes 72 bytes");

bool load_table(Table& t, const void* bytes, int64_t size, std::string& error);

}  // namespace zkp
