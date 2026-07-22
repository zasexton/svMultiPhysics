/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_MATH_DENSETRANSFORMFALLBACK_H
#define SVMP_FE_MATH_DENSETRANSFORMFALLBACK_H

#include "Core/Types.h"

#include <cstddef>

namespace svmp {
namespace FE {
namespace math {

inline void dense_transform_batched_row_major(
    const Real* SVMP_RESTRICT matrix,
    std::size_t rows,
    std::size_t cols,
    const Real* SVMP_RESTRICT input,
    std::size_t input_row_stride,
    Real* SVMP_RESTRICT output,
    std::size_t output_row_stride,
    std::size_t rhs_count)
{
    if (rows == 0u || cols == 0u || rhs_count == 0u) {
        return;
    }

    for (std::size_t row = 0; row < rows; ++row) {
        const Real* matrix_row = matrix + row * cols;
        Real* output_row = output + row * output_row_stride;
        for (std::size_t rhs = 0; rhs < rhs_count; ++rhs) {
            Real value{0.0};
            for (std::size_t column = 0; column < cols; ++column) {
                value += matrix_row[column] *
                         input[column * input_row_stride + rhs];
            }
            output_row[rhs] = value;
        }
    }
}

} // namespace math
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_MATH_DENSETRANSFORMFALLBACK_H
