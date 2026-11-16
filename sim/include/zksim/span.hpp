// A run of values something else keeps: a vector, or a file mapped into memory that processes share.
#pragma once

#include <cstddef>
#include <vector>

namespace zk {

template <class T>
struct Span {
    const T* p = nullptr;
    size_t n = 0;
    Span() = default;
    Span(const T* p_, size_t n_) : p(p_), n(n_) {}
    Span(const std::vector<T>& v) : p(v.data()), n(v.size()) {}  // NOLINT: a vector's values, while it keeps them
    const T& operator[](size_t i) const { return p[i]; }
    const T* data() const { return p; }
    size_t size() const { return n; }
    bool empty() const { return n == 0; }
    const T* begin() const { return p; }
    const T* end() const { return p + n; }
};

}  // namespace zk
