/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_BACKENDS_TRILINOS_FACTORY_H
#define SVMP_FE_BACKENDS_TRILINOS_FACTORY_H

#include "Backends/Interfaces/BackendFactory.h"

#if defined(FE_HAS_TRILINOS)

#include <vector>

namespace svmp {
namespace FE {
namespace sparsity {
class DistributedSparsityPattern;
} // namespace sparsity

namespace backends {

class TrilinosFactory final : public BackendFactory {
public:
    using BackendFactory::createMatrix;
    using BackendFactory::createVector;

    [[nodiscard]] BackendKind backendKind() const noexcept override { return BackendKind::Trilinos; }

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
    struct CachedGhosting {
        GlobalIndex global_size{0};
        GlobalIndex owned_first{0};
        GlobalIndex local_owned_size{0};
        std::vector<GlobalIndex> ghost_global_indices{};
        bool valid{false};
    };

    mutable CachedGhosting cached_{};
};

} // namespace backends
} // namespace FE
} // namespace svmp

#endif // FE_HAS_TRILINOS

#endif // SVMP_FE_BACKENDS_TRILINOS_FACTORY_H
