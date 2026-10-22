/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SV_FE_LS_TWO_LANE_H
#define SV_FE_LS_TWO_LANE_H

//--------------------------------------------------------------------
// Two-lane double vectors (GCC/Clang vector extension: NEON on aarch64,
// SSE2 on x86-64) for kernels that run two independent sums side by side.
//
// Multiplies and adds of these vectors are lane-wise IEEE operations: each
// lane gets exactly the rounding of the scalar operation, they are never
// contracted into fused multiply-adds (the build uses -ffp-contract=off) and
// nothing here reduces across lanes.  A kernel that keeps, per lane, the
// operation sequence of a scalar sum therefore gives the scalar bits.
//--------------------------------------------------------------------

#if defined(__GNUC__) || defined(__clang__)
#define FSILS_HAVE_TWO_LANE 1

#include <cstring>

namespace fe_fsi_linear_solver::two_lane {

typedef double v2d __attribute__((vector_size(16)));
typedef long long v2i __attribute__((vector_size(16)));

inline v2d load2(const double* p) noexcept
{
  v2d r;
  std::memcpy(&r, p, sizeof(r));
  return r;
}

inline void store2(double* p, const v2d v) noexcept
{
  std::memcpy(p, &v, sizeof(v));
}

inline v2d splat(const double x) noexcept
{
  return v2d{x, x};
}

/// {a[0], b[0]}
inline v2d first_lanes(const v2d a, const v2d b) noexcept
{
#if defined(__clang__)
  return __builtin_shufflevector(a, b, 0, 2);
#else
  return __builtin_shuffle(a, b, v2i{0, 2});
#endif
}

/// {a[1], b[1]}
inline v2d second_lanes(const v2d a, const v2d b) noexcept
{
#if defined(__clang__)
  return __builtin_shufflevector(a, b, 1, 3);
#else
  return __builtin_shuffle(a, b, v2i{1, 3});
#endif
}

} // namespace fe_fsi_linear_solver::two_lane

#endif // __GNUC__ || __clang__

#endif // SV_FE_LS_TWO_LANE_H
