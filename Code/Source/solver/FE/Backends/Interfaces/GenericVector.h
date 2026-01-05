/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_BACKENDS_GENERIC_VECTOR_H
#define SVMP_FE_BACKENDS_GENERIC_VECTOR_H

#include "Backends/Interfaces/BackendKind.h"
#include "Core/Types.h"

#include "Assembly/GlobalSystemView.h"

#include <memory>
#include <span>

namespace svmp {
namespace FE {
namespace backends {

class GenericVector {
public:
    virtual ~GenericVector() = default;

    [[nodiscard]] virtual BackendKind backendKind() const noexcept = 0;
    [[nodiscard]] virtual GlobalIndex size() const noexcept = 0;

    virtual void zero() = 0;
    virtual void set(Real value) = 0;
    virtual void add(Real value) = 0;
    virtual void scale(Real alpha) = 0;

    [[nodiscard]] virtual Real dot(const GenericVector& other) const = 0;
    [[nodiscard]] virtual Real norm() const = 0;

    virtual void updateGhosts() = 0;

    [[nodiscard]] virtual std::unique_ptr<assembly::GlobalSystemView> createAssemblyView() = 0;

    [[nodiscard]] virtual std::span<Real> localSpan() = 0;
    [[nodiscard]] virtual std::span<const Real> localSpan() const = 0;
};

} // namespace backends
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BACKENDS_GENERIC_VECTOR_H
