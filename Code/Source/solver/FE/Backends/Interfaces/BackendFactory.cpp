/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Backends/Interfaces/BackendFactory.h"

#include "Core/FEException.h"

#include "Backends/Interfaces/BackendKind.h"

#if defined(FE_HAS_EIGEN) && FE_HAS_EIGEN
#include "Backends/Eigen/EigenFactory.h"
#endif

#include "Backends/FSILS/FsilsFactory.h"

#if defined(FE_HAS_PETSC) && FE_HAS_PETSC
#include "Backends/PETSc/PetscFactory.h"
#endif

#if defined(FE_HAS_TRILINOS) && FE_HAS_TRILINOS
#include "Backends/Trilinos/TrilinosFactory.h"
#endif

namespace svmp {
namespace FE {
namespace backends {

std::unique_ptr<GenericMatrix>
BackendFactory::createMatrix(const sparsity::DistributedSparsityPattern&) const
{
    FE_THROW(NotImplementedException, "BackendFactory: distributed matrix creation not supported by this backend");
}

std::unique_ptr<GenericVector>
BackendFactory::createVector(GlobalIndex, GlobalIndex global_size) const
{
    return createVector(global_size);
}

std::unique_ptr<BackendFactory> BackendFactory::create(BackendKind kind)
{
    return create(kind, CreateOptions{});
}

std::unique_ptr<BackendFactory> BackendFactory::create(std::string_view name)
{
    return create(name, CreateOptions{});
}

std::unique_ptr<BackendFactory> BackendFactory::create(BackendKind kind, const CreateOptions& options)
{
    switch (kind) {
        case BackendKind::Eigen:
        #if defined(FE_HAS_EIGEN) && FE_HAS_EIGEN
            return std::make_unique<EigenFactory>();
        #else
            FE_THROW(NotImplementedException, "BackendFactory: Eigen backend not compiled (enable FE_ENABLE_EIGEN)");
        #endif

        case BackendKind::FSILS:
            FE_THROW_IF(options.dof_per_node <= 0, InvalidArgumentException,
                        "BackendFactory: fsils dof_per_node must be > 0");
            return std::make_unique<FsilsFactory>(options.dof_per_node);

        case BackendKind::PETSc:
        #if defined(FE_HAS_PETSC) && FE_HAS_PETSC
            return std::make_unique<PetscFactory>();
        #else
            FE_THROW(NotImplementedException, "BackendFactory: PETSc backend not compiled (enable FE_ENABLE_PETSC)");
        #endif

        case BackendKind::Trilinos:
        #if defined(FE_HAS_TRILINOS) && FE_HAS_TRILINOS
            return std::make_unique<TrilinosFactory>();
        #else
            FE_THROW(NotImplementedException, "BackendFactory: Trilinos backend not compiled (enable FE_ENABLE_TRILINOS)");
        #endif
        default:
            FE_THROW(InvalidArgumentException, "BackendFactory: unknown backend kind");
    }
}

std::unique_ptr<BackendFactory> BackendFactory::create(std::string_view name, const CreateOptions& options)
{
    const auto kind = backendKindFromString(name);
    FE_THROW_IF(!kind.has_value(), InvalidArgumentException,
                "BackendFactory: unknown backend name '" + std::string(name) + "'");
    return create(*kind, options);
}

} // namespace backends
} // namespace FE
} // namespace svmp
