// numpy's and numba's arithmetic, where it isn't C's: scalar max, min, clip and sign as numpy has them, float to int as x86
// converts it, a float32 sum as numpy's pairwise one, Python's modulo and negative indices, and argsort's order.  The build
// has no -ffast-math and no contraction (sim/CMakeLists.txt), so float32 stays float32 and nothing is fused or reordered.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace zkp {

// featurize_kernels' _max, _min, _clip and _sign: numpy's own scalar rules, a NaN passed on.
inline float np_max(float a, float b) { return (a >= b || a != a) ? a : b; }
inline float np_min(float a, float b) { return (a <= b || a != a) ? a : b; }
inline float np_clip(float v, float lo, float hi) { return np_min(np_max(v, lo), hi); }
inline float np_sign(float a) {
    if (a > 0.0f) return 1.0f;
    if (a < 0.0f) return -1.0f;
    if (a == 0.0f) return 0.0f;
    return a;
}

// Float to int as numba's fptosi and numpy's astype give it on x86 (cvttss2si): towards zero, and the type's least value for
// a NaN or anything out of its range, where C++ would leave it undefined.
inline int32_t to_i32(double v) {
    if (!(v > -2147483649.0 && v < 2147483648.0)) return std::numeric_limits<int32_t>::min();
    return int32_t(v);
}
inline int64_t to_i64(double v) {
    if (!(v >= -9223372036854775808.0 && v < 9223372036854775808.0)) return std::numeric_limits<int64_t>::min();
    return int64_t(v);
}

// Python's modulo, the divisor's sign.
inline int64_t py_mod(int64_t a, int64_t b) {
    int64_t r = a % b;
    return (r != 0 && ((r < 0) != (b < 0))) ? r + b : r;
}

// Python's index: a negative one counts from the end.
inline int64_t py_index(int64_t i, int64_t n) { return i < 0 ? i + n : i; }

// numpy's pairwise sum of float32 (loops_utils.h.src), as featurize_kernels._pairwise_block and _mean have it.
inline float pairwise_block(const float* a, int64_t n) {
    if (n < 8) {
        float res = -0.0f;
        for (int64_t i = 0; i < n; ++i) res += a[i];
        return res;
    }
    float r0 = a[0], r1 = a[1], r2 = a[2], r3 = a[3], r4 = a[4], r5 = a[5], r6 = a[6], r7 = a[7];
    int64_t i = 8;
    for (; i < n - (n % 8); i += 8) {
        r0 += a[i];
        r1 += a[i + 1];
        r2 += a[i + 2];
        r3 += a[i + 3];
        r4 += a[i + 4];
        r5 += a[i + 5];
        r6 += a[i + 6];
        r7 += a[i + 7];
    }
    float res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
    for (; i < n; ++i) res += a[i];
    return res;
}

inline float pairwise(const float* a, int64_t n) {
    if (n <= 128) return pairwise_block(a, n);
    int64_t half = n / 2;
    half -= half % 8;
    return pairwise(a, half) + pairwise(a + half, n - half);
}

// np.add.reduce of float32: the identity, then the pairwise sum.
inline float np_sum(const float* a, int64_t n) { return 0.0f + pairwise(a, n); }
inline float np_mean(const float* a, int64_t n) { return np_sum(a, n) / float(n); }

// numpy's sort order of floats: a NaN after everything.
inline bool f_less(float a, float b) { return a < b || (b != b && a == a); }

template <class K>
inline int key_cmp(const K& a, const K& b) {
    if constexpr (std::is_floating_point_v<K>) {
        if (f_less(a, b)) return -1;
        if (f_less(b, a)) return 1;
        return 0;
    } else {
        return a < b ? -1 : (b < a ? 1 : 0);
    }
}

// np.lexsort((k3, k2, k1)) and the like: positions 0..n-1 by k1, then k2, then k3, ties in order.
template <class A, class B, class C>
void lexorder(std::vector<int64_t>& out, int64_t n, const A* k1, const B* k2, const C* k3) {
    out.resize(n);
    for (int64_t i = 0; i < n; ++i) out[i] = i;
    std::stable_sort(out.begin(), out.end(), [&](int64_t p, int64_t q) {
        int c = key_cmp(k1[p], k1[q]);
        if (c) return c < 0;
        c = key_cmp(k2[p], k2[q]);
        if (c) return c < 0;
        return key_cmp(k3[p], k3[q]) < 0;
    });
}

template <class A, class B>
void lexorder2(std::vector<int64_t>& out, int64_t n, const A* k1, const B* k2) {
    out.resize(n);
    for (int64_t i = 0; i < n; ++i) out[i] = i;
    std::stable_sort(out.begin(), out.end(), [&](int64_t p, int64_t q) {
        int c = key_cmp(k1[p], k1[q]);
        if (c) return c < 0;
        return key_cmp(k2[p], k2[q]) < 0;
    });
}

// np.argmin: the first least, or the first NaN.
template <class F>
int64_t np_argmin(const F* a, int64_t n) {
    int64_t best = 0;
    for (int64_t i = 0; i < n; ++i) {
        if (a[i] != a[i]) return i;
        if (a[i] < a[best]) best = i;
    }
    return best;
}

// featurize.relative_id for one id.
inline int64_t relative_id(int64_t v, int64_t own, int64_t count) {
    int64_t out = v < own ? v + 1 : v;
    if (v == own) out = 0;
    if (v < 0 || out >= count) out = count;
    return out;
}

}  // namespace zkp
