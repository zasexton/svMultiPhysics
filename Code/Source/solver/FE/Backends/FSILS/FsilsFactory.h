/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_BACKENDS_FSILS_FACTORY_H
#define SVMP_FE_BACKENDS_FSILS_FACTORY_H

#include "Backends/Interfaces/BackendFactory.h"

#include "Backends/FSILS/FsilsShared.h"

namespace svmp {
namespace FE {
namespace backends {

class FsilsFactory final : public BackendFactory {
public:
    using BackendFactory::createMatrix;
    using BackendFactory::createVector;

    explicit FsilsFactory(int dof_per_node = 1,
                          std::shared_ptr<const DofPermutation> dof_permutation = {})
        : dof_per_node_(dof_per_node)
        , dof_permutation_(std::move(dof_permutation))
    {
    }

    [[nodiscard]] BackendKind backendKind() const noexcept override { return BackendKind::FSILS; }

    [[nodiscard]] std::unique_ptr<GenericMatrix>
    createMatrix(const sparsity::SparsityPattern& sparsity) const override;

    [[nodiscard]] std::unique_ptr<GenericMatrix>
    createMatrix(const sparsity::DistributedSparsityPattern& sparsity) const override;

    [[nodiscard]] std::unique_ptr<GenericVector> createVector(GlobalIndex size) const override;

    [[nodiscard]] std::unique_ptr<GenericVector>
    createVector(GlobalIndex local_size, GlobalIndex global_size) const override;

    [[nodiscard]] std::unique_ptr<LinearSolver>
    createLinearSolver(const SolverOptions& options) const override;

private:
    int dof_per_node_{1};
    std::shared_ptr<const DofPermutation> dof_permutation_{};
    mutable std::shared_ptr<const FsilsShared> cached_shared_{};
};

} // namespace backends
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BACKENDS_FSILS_FACTORY_H
