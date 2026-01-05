/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Backends/FSILS/FsilsFactory.h"

#include "Backends/FSILS/FsilsLinearSolver.h"
#include "Backends/FSILS/FsilsMatrix.h"
#include "Backends/FSILS/FsilsVector.h"

#include "Core/FEException.h"
#include "Sparsity/DistributedSparsityPattern.h"

namespace svmp {
namespace FE {
namespace backends {

std::unique_ptr<GenericMatrix>
FsilsFactory::createMatrix(const sparsity::SparsityPattern& sparsity) const
{
    auto mat = std::make_unique<FsilsMatrix>(sparsity, dof_per_node_);
    cached_shared_ = mat->shared();
    return mat;
}

std::unique_ptr<GenericMatrix>
FsilsFactory::createMatrix(const sparsity::DistributedSparsityPattern& sparsity) const
{
    auto mat = std::make_unique<FsilsMatrix>(sparsity, dof_per_node_);
    cached_shared_ = mat->shared();
    return mat;
}

std::unique_ptr<GenericVector> FsilsFactory::createVector(GlobalIndex size) const
{
    if (cached_shared_) {
        FE_THROW_IF(size != cached_shared_->global_dofs, InvalidArgumentException,
                    "FsilsFactory::createVector: size mismatch with last FSILS matrix");
        return std::make_unique<FsilsVector>(cached_shared_);
    }
    return std::make_unique<FsilsVector>(size);
}

std::unique_ptr<GenericVector>
FsilsFactory::createVector(GlobalIndex local_size, GlobalIndex global_size) const
{
    if (!cached_shared_) {
        // Fall back to a local-only vector when no FSILS layout is known.
        // For distributed FSILS runs, createMatrix(DistributedSparsityPattern) must be called first.
        return std::make_unique<FsilsVector>(global_size);
    }

    FE_THROW_IF(global_size != cached_shared_->global_dofs, InvalidArgumentException,
                "FsilsFactory::createVector: global_size mismatch with last FSILS matrix");
    FE_THROW_IF(local_size != static_cast<GlobalIndex>(cached_shared_->dof) *
                                 static_cast<GlobalIndex>(cached_shared_->lhs.nNo),
                InvalidArgumentException,
                "FsilsFactory::createVector: local_size must equal dof * lhs.nNo for FSILS overlap vectors");
    return std::make_unique<FsilsVector>(cached_shared_);
}

std::unique_ptr<LinearSolver>
FsilsFactory::createLinearSolver(const SolverOptions& options) const
{
    return std::make_unique<FsilsLinearSolver>(options);
}

} // namespace backends
} // namespace FE
} // namespace svmp
