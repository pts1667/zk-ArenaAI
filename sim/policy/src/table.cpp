// The type table from its flat file (zkpolicy.h): tables.TypeTable as the kernels read it, written by sim/py/zkpolicy.py.
#include <cstring>

#include "policy.hpp"
#include "table.hpp"

namespace zkp {

namespace {

struct Reader {
    const uint8_t* at;
    const uint8_t* end;
    bool ok = true;

    template <class T>
    void array(std::vector<T>& out, int64_t n) {
        const uint8_t* start = at;
        int64_t bytes = n * int64_t(sizeof(T));
        if (!ok || n < 0 || end - start < bytes) {
            ok = false;
            return;
        }
        out.resize(n);
        if (bytes) std::memcpy(out.data(), start, bytes);
        at = start + ((bytes + 7) / 8) * 8;     // each array starts 8-byte aligned
        if (at > end) at = end;
    }

    void strings(std::vector<std::string>& out, int64_t bytes, int64_t count) {
        std::vector<char> block;
        array(block, bytes);
        if (!ok) return;
        out.clear();
        size_t from = 0;
        for (size_t i = 0; i < block.size(); ++i) {
            if (block[i] == '\0') {
                out.emplace_back(block.data() + from, i - from);
                from = i + 1;
            }
        }
        if (int64_t(out.size()) != count) ok = false;
    }
};

}  // namespace

bool load_table(Table& t, const void* bytes, int64_t size, std::string& error) {
    TableHeader h;
    if (size < int64_t(sizeof h)) {
        error = "shorter than its header";
        return false;
    }
    std::memcpy(&h, bytes, sizeof h);
    if (std::memcmp(h.magic, TABLE_MAGIC, 8) != 0) {
        error = "not a policy table";
        return false;
    }
    if (h.version != TABLE_VERSION) {
        error = "table version " + std::to_string(h.version) + ", this library reads " + std::to_string(TABLE_VERSION);
        return false;
    }
    if (h.f_cols != T_COLS || h.b_cols != B_COLS || h.rows < 1 || h.unknown != h.rows - 1 || h.defs < 1) {
        error = "the table's columns aren't this library's";
        return false;
    }
    t.rows = int(h.rows);
    t.type_dim = int(h.type_dim);
    t.unit_dim = t.type_dim + INSTANCE_COLS + ORDER_COLS + RAY_COLS + SUMMARY_COLS;
    t.producible_count = int(h.producible);
    t.unknown = int(h.unknown);
    t.roles = int(h.roles);
    t.choosable = h.choosable;
    t.hash.assign(h.hash, strnlen(h.hash, sizeof h.hash));
    Reader r{static_cast<const uint8_t*>(bytes) + sizeof h, static_cast<const uint8_t*>(bytes) + size};
    const int64_t rows = h.rows, T = h.producible;
    r.array(t.stats, rows * h.type_dim);
    r.array(t.f, rows * T_COLS);
    r.array(t.b, rows * B_COLS);
    r.array(t.move_class, rows);
    r.array(t.produces, rows * T);
    r.array(t.builds, rows * T);
    r.array(t.producible, T);
    r.array(t.row_of_def, h.defs);
    r.array(t.def_of_row, rows);
    r.array(t.role, rows);
    r.strings(t.names, h.names_bytes, rows - 1);
    r.strings(t.role_names, h.role_names_bytes, h.roles);
    if (!r.ok) {
        error = "the table is cut short";
        return false;
    }
    for (int32_t v : t.row_of_def) {
        if (v < 0 || v >= t.rows) {
            error = "row_of_def points past the table";
            return false;
        }
    }
    return true;
}

}  // namespace zkp
