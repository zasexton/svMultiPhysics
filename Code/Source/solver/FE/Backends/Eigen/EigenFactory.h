/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_BACKENDS_EIGEN_FACTORY_H
#define SVMP_FE_BACKENDS_EIGEN_FACTORY_H

#include "Backends/Interfaces/BackendFactory.h"

namespace svmp {
namespace FE {
namespace backends {

#if defined(FE_HAS_EIGEN)

class EigenFactory final : public BackendFactory {
public:
    using BackendFactory::createMatrix;
    using BackendFactory::createVector;

    [[nodiscard]] BackendKind backendKind() const noexcept override { return BackendKind::Eigen; }

    [[nodiscard]] std::unique_ptr<GenericMatrix>
    createMatrix(const sparsity::SparsityPattern& sparsity) const override;

    [[nodiscard]] std::unique_ptr<GenericVector> createVector(GlobalIndex size) const override;

    [[nodiscard]] std::unique_ptr<LinearSolver>
    createLinearSolver(const SolverOptions& options) const override;
};

#endif // FE_HAS_EIGEN

} // namespace backends
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BACKENDS_EIGEN_FACTORY_H
