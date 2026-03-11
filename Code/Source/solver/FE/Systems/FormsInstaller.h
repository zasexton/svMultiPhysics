/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_FORMSINSTALLER_H
#define SVMP_FE_SYSTEMS_FORMSINSTALLER_H

/**
 * @file FormsInstaller.h
 * @brief Public API for installing weak-form formulations into an FESystem.
 *
 * Physics modules should use installFormulation() as the single entry point.
 * It auto-selects single-field or multi-field (mixed) assembly paths based on
 * the number of TestFunction spaces in the residual expression.
 *
 * For internal/low-level installer functions, see FormsInstallerDetail.h.
 */

#include "Core/Types.h"

#include "Forms/BoundaryConditions.h"
#include "Systems/OperatorRegistry.h"

#include <initializer_list>
#include <memory>
#include <span>
#include <vector>

namespace svmp {
namespace FE {

namespace assembly {
class AssemblyKernel;
}

namespace forms {
class FormExpr;
}

namespace systems {

class FESystem;

struct FormInstallOptions {
    forms::ADMode ad_mode{forms::ADMode::Forward};
    forms::SymbolicOptions compiler_options{};

    // Coupled residual installation controls (used internally by installFormulation).
    bool coupled_residual_install_residual_kernels{true};
    bool coupled_residual_install_jacobian_blocks{true};

    // When true, the residual vector for each test field is produced by a single
    // Jacobian block kernel (output=Both) instead of a separate vector-only kernel.
    // This reduces assembly passes during Newton when assembling matrix+vector together.
    bool coupled_residual_from_jacobian_block{false};
};

using KernelPtr = std::shared_ptr<assembly::AssemblyKernel>;

void installStrongDirichlet(
    FESystem& system,
    std::span<const forms::bc::StrongDirichlet> bcs);

struct CoupledResidualKernels {
    std::vector<KernelPtr> residual;                      // one per test field
    std::vector<std::vector<KernelPtr>> jacobian_blocks;  // [test][trial]
};

/**
 * @brief Unified formulation installer — auto-selects single or multi-field path
 *
 * This is the preferred entry point for physics modules. The caller provides a
 * single residual FormExpr and the field IDs for the unknowns. The function
 * inspects the expression to determine the number of test-function spaces and
 * automatically routes to the appropriate internal path:
 *
 *   - 1 TestFunction space → single-field path
 *   - 2+ TestFunction spaces → mixed multi-field path
 *
 * Unknown fields should be created with FormExpr::stateField(). The function
 * handles lowering StateField nodes to TrialFunction/DiscreteField internally.
 * Expressions with TrialFunction (no StateField nodes) are also supported for
 * single-field formulations.
 *
 * For multi-field formulations, the coupled assembly strategy is set automatically:
 * diagonal Jacobian blocks produce both matrix and residual vector (Both mode),
 * off-diagonal blocks produce matrix only.
 *
 * @param system   FESystem to install kernels into
 * @param op       Operator tag (e.g., "equations")
 * @param fields   FieldId for each unknown (test = trial, standard Galerkin)
 * @param residual Full residual expression (may reference multiple TestFunctions)
 * @param options  Compiler and JIT options (coupled_residual_* flags are overridden)
 */
CoupledResidualKernels installFormulation(
    FESystem& system,
    const OperatorTag& op,
    std::span<const FieldId> fields,
    const forms::FormExpr& residual,
    const FormInstallOptions& options = {});

CoupledResidualKernels installFormulation(
    FESystem& system,
    const OperatorTag& op,
    std::initializer_list<FieldId> fields,
    const forms::FormExpr& residual,
    const FormInstallOptions& options = {});

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_FORMSINSTALLER_H
