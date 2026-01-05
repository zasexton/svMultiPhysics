/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_BACKENDS_BLOCK_VECTOR_H
#define SVMP_FE_BACKENDS_BLOCK_VECTOR_H

#include "Backends/Interfaces/GenericVector.h"
#include "Core/FEException.h"

#include <algorithm>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace backends {

/**
 * @brief A vector composed of sub-vectors (fields).
 *
 * This class provides a backend-agnostic block structure for multi-field
 * problems while still presenting the `GenericVector` interface when a
 * monolithic view is required.
 *
 * @note `localSpan()` is only supported when there is exactly one block.
 */
class BlockVector final : public GenericVector {
public:
    explicit BlockVector(std::vector<std::unique_ptr<GenericVector>> blocks);

    [[nodiscard]] BackendKind backendKind() const noexcept override { return backend_kind_; }
    [[nodiscard]] GlobalIndex size() const noexcept override { return global_size_; }

    void zero() override;
    void set(Real value) override;
    void add(Real value) override;
    void scale(Real alpha) override;

    [[nodiscard]] Real dot(const GenericVector& other) const override;
    [[nodiscard]] Real norm() const override;

    void updateGhosts() override;

    [[nodiscard]] std::unique_ptr<assembly::GlobalSystemView> createAssemblyView() override;

    [[nodiscard]] std::span<Real> localSpan() override;
    [[nodiscard]] std::span<const Real> localSpan() const override;

    [[nodiscard]] std::size_t numBlocks() const noexcept { return blocks_.size(); }
    [[nodiscard]] GenericVector& block(std::size_t i);
    [[nodiscard]] const GenericVector& block(std::size_t i) const;
    [[nodiscard]] std::span<const GlobalIndex> blockOffsets() const noexcept { return offsets_; }

    [[nodiscard]] std::pair<std::size_t, GlobalIndex> locate(GlobalIndex global_index) const;

    BackendKind backend_kind_{BackendKind::Eigen};
    std::vector<std::unique_ptr<GenericVector>> blocks_{};
    std::vector<GlobalIndex> offsets_{}; // size numBlocks()+1
    GlobalIndex global_size_{0};
};

} // namespace backends
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BACKENDS_BLOCK_VECTOR_H
