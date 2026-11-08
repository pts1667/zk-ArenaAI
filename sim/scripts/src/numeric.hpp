// The arithmetic of the Python scripts, done the same way, so that orders come out the same to the bit where they can.
//
//   py_hypot   CPython 3.12's math.hypot (vector_norm in Modules/mathmodule.c), which is not libm's hypot
//   std::hypot numpy's np.hypot on float64 arrays (libm)
//   hypotf     numpy's np.hypot on float32 arrays (libm)
//   pairwise   numpy's sum of a contiguous 1-d array (pairwise summation, loops_utils.h.src)
//   interp     numpy's np.interp
//   along      the scripts' _along, with math.hypot
#pragma once

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <utility>

namespace zks {

inline double py_hypot(double a, double b) {
    double v[2] = {std::fabs(a), std::fabs(b)};
    if (std::isnan(v[0]) || std::isnan(v[1])) {
        if (std::isinf(v[0]) || std::isinf(v[1])) return INFINITY;
        return NAN;
    }
    double mx = v[0] > v[1] ? v[0] : v[1];
    if (std::isinf(mx)) return mx;
    if (mx == 0.0) return mx;
    int max_e;
    std::frexp(mx, &max_e);
    double extra = 1.0;
    if (max_e < -1023) {       // subnormals: as CPython, scale them up first
        v[0] /= DBL_MIN;
        v[1] /= DBL_MIN;
        mx /= DBL_MIN;
        std::frexp(mx, &max_e);
        extra = DBL_MIN;
    }
    double scale = std::ldexp(1.0, -max_e);
    double csum = 1.0, frac1 = 0.0, frac2 = 0.0;
    for (int i = 0; i < 2; i++) {
        double x = v[i] * scale;
        double hi = x * x;
        double lo = std::fma(x, x, -hi);
        double s = csum + hi;
        double s_lo = (csum - s) + hi;
        csum = s;
        frac1 += lo;
        frac2 += s_lo;
    }
    double h = std::sqrt(csum - 1.0 + (frac1 + frac2));
    double hi = -h * h;
    double lo = std::fma(-h, h, -hi);
    double s = csum + hi;
    double s_lo = (csum - s) + hi;
    csum = s;
    frac1 += lo;
    frac2 += s_lo;
    double x = csum - 1.0 + (frac1 + frac2);
    h += x / (2.0 * h);
    return extra * (h / scale);
}

// CPython 3.12's built-in sum() of floats from the start 0: the first is added to 0 exactly, the rest by Neumaier's compensated
// summation (Python/bltinmodule.c builtin_sum_impl).  Policy_mix's shares are summed so, and ties between roles hang on it.
struct PySum {
    double f = 0.0, c = 0.0;
    bool any = false;
    void add(double x) {
        if (!any) {
            f = 0.0 + x;
            any = true;
            return;
        }
        double t = f + x;
        if (std::fabs(f) >= std::fabs(x)) c += (f - t) + x;
        else c += (x - t) + f;
        f = t;
    }
    double value() const { return (c != 0.0 && std::isfinite(c)) ? f + c : f; }
};

// numpy's pairwise summation of n values a[0], a[stride], ... (FLOAT_pairwise_sum / DOUBLE_pairwise_sum).
template <class T, class Get>
T pairwise(Get get, int64_t lo, int64_t n) {
    if (n < 8) {
        T res = -0.0;
        for (int64_t i = 0; i < n; i++) res += get(lo + i);
        return res;
    }
    if (n <= 128) {
        T r[8];
        for (int j = 0; j < 8; j++) r[j] = get(lo + j);
        int64_t i = 8;
        for (; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; j++) r[j] += get(lo + i + j);
        T res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; i++) res += get(lo + i);
        return res;
    }
    int64_t n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise<T>(get, lo, n2) + pairwise<T>(get, lo + n2, n - n2);
}

// np.sum of a 1-d array: the identity, then the pairwise sum.
template <class T, class Get>
T np_sum(Get get, int64_t n) {
    T zero = 0;
    return zero + pairwise<T>(get, 0, n);
}

inline double interp(double x, const double* xs, const double* ys, int n) {
    if (std::isnan(x)) return x;
    if (x < xs[0]) return ys[0];
    if (x > xs[n - 1]) return ys[n - 1];
    int j = 0;
    while (j + 1 < n && xs[j + 1] <= x) j++;
    if (j == n - 1) return ys[j];
    if (xs[j] == x) return ys[j];
    double slope = (ys[j + 1] - ys[j]) / (xs[j + 1] - xs[j]);
    double r = slope * (x - xs[j]) + ys[j];
    if (std::isnan(r)) {
        r = slope * (x - xs[j + 1]) + ys[j + 1];
        if (std::isnan(r) && ys[j] == ys[j + 1]) r = ys[j];
    }
    return r;
}

struct P2 {
    double x, z;
};

// policy_army._along / policy_eco._along
inline P2 along(P2 frm, P2 to, double dist) {
    double d = py_hypot(to.x - frm.x, to.z - frm.z);
    if (d < 1e-6) return frm;
    return {frm.x + (to.x - frm.x) / d * dist, frm.z + (to.z - frm.z) / d * dist};
}

// Python's float floor division by a positive power of two, then int(): exact.
inline int64_t floordiv(double a, double b) { return (int64_t)std::floor(a / b); }

}  // namespace zks
