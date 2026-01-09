/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ASSEMBLY_VECTORIZED_ASSEMBLER_H
#define SVMP_FE_ASSEMBLY_VECTORIZED_ASSEMBLER_H

/**
 * @file VectorizedAssembler.h
 * @brief Decorator placeholder for SIMD/batched traversal
 *
 * This wrapper is intentionally lightweight: it enables composition and a
 * stable opt-in surface while the underlying kernels/assemblers evolve to
 * exploit SIMD-friendly batching.
 */

#include "Assembly/DecoratorAssembler.h"

#include <string>

namespace svmp {
namespace FE {
namespace assembly {

class VectorizedAssembler final : public DecoratorAssembler {
public:
    explicit VectorizedAssembler(std::unique_ptr<Assembler> base, int batch_size = 32)
        : DecoratorAssembler(std::move(base)), batch_size_(batch_size)
    {
    }

    void setBatchSize(int batch_size) noexcept { batch_size_ = batch_size; }
    [[nodiscard]] int batchSize() const noexcept { return batch_size_; }

    [[nodiscard]] std::string name() const override
    {
        return "Vectorized(" + base().name() + ")";
    }

private:
    int batch_size_{32};
};

} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ASSEMBLY_VECTORIZED_ASSEMBLER_H

