/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_OPERATORBACKENDS_H
#define SVMP_FE_SYSTEMS_OPERATORBACKENDS_H

#include "Core/Types.h"
#include "Systems/OperatorRegistry.h"
#include "Systems/SystemState.h"

#include <memory>
#include <string>

namespace svmp {
namespace FE {

namespace assembly {
struct MatrixFreeOptions;
class IMatrixFreeKernel;
class MatrixFreeOperator;
class FunctionalKernel;
} // namespace assembly

namespace systems {

class FESystem;

/**
 * @brief Optional operator backends owned by Systems
 *
 * This is a thin wiring layer that lets Systems expose additional FE
 * infrastructure (matrix-free operators and scalar functionals) while keeping
 * the primary "assembled matrix/vector" path unchanged.
 *
 * Phase 1 scope:
 * - Single-field systems only
 * - Mesh iteration provided by `assembly::IMeshAccess` (typically MeshAccess)
 */
class OperatorBackends {
public:
    OperatorBackends();
    ~OperatorBackends();

    OperatorBackends(OperatorBackends&&) noexcept;
    OperatorBackends& operator=(OperatorBackends&&) noexcept;

    OperatorBackends(const OperatorBackends&) = delete;
    OperatorBackends& operator=(const OperatorBackends&) = delete;

    void clear();
    void invalidateCache();

    // ---------------------------------------------------------------------
    // Matrix-free backends
    // ---------------------------------------------------------------------

    void registerMatrixFree(OperatorTag tag,
                            std::shared_ptr<assembly::IMatrixFreeKernel> kernel);

    void registerMatrixFree(OperatorTag tag,
                            std::shared_ptr<assembly::IMatrixFreeKernel> kernel,
                            const assembly::MatrixFreeOptions& options);

    [[nodiscard]] bool hasMatrixFree(const OperatorTag& tag) const noexcept;

    [[nodiscard]] std::shared_ptr<assembly::MatrixFreeOperator>
    matrixFreeOperator(const FESystem& system, const OperatorTag& tag) const;

    // ---------------------------------------------------------------------
    // Scalar functionals (QoIs)
    // ---------------------------------------------------------------------

    void registerFunctional(std::string tag,
                            std::shared_ptr<assembly::FunctionalKernel> kernel);

    [[nodiscard]] Real evaluateFunctional(const FESystem& system,
                                          const std::string& tag,
                                          const SystemStateView& state) const;

    [[nodiscard]] Real evaluateBoundaryFunctional(const FESystem& system,
                                                  const std::string& tag,
                                                  int boundary_marker,
                                                  const SystemStateView& state) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_OPERATORBACKENDS_H
