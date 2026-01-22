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
 * @brief Helpers to lower FE/Forms weak forms into FE/Systems kernels.
 */

#include "Core/Types.h"

#include "Forms/BlockForm.h"
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
struct WeakForm;
}

namespace systems {

class FESystem;

struct FormInstallOptions {
    forms::ADMode ad_mode{forms::ADMode::Forward};
    forms::SymbolicOptions compiler_options{};

    // Coupled residual installation controls (used by installCoupledResidual()).
    bool coupled_residual_install_residual_kernels{true};
    bool coupled_residual_install_jacobian_blocks{true};

    // When true, the residual vector for each test field is produced by a single
    // Jacobian block kernel (output=Both) instead of a separate vector-only kernel.
    // This reduces assembly passes during Newton when assembling matrix+vector together.
    // Requires:
    // - coupled_residual_install_residual_kernels == false
    // - coupled_residual_install_jacobian_blocks == true
    bool coupled_residual_from_jacobian_block{false};
};

using KernelPtr = std::shared_ptr<assembly::AssemblyKernel>;

KernelPtr installResidualForm(
    FESystem& system,
    const OperatorTag& op,
    FieldId test_field,
    FieldId trial_field,
    const forms::FormExpr& residual_form,
    const FormInstallOptions& options = {});

void installStrongDirichlet(
    FESystem& system,
    std::span<const forms::bc::StrongDirichlet> bcs);

KernelPtr installResidualForm(
    FESystem& system,
    const OperatorTag& op,
    FieldId test_field,
    FieldId trial_field,
    const forms::FormExpr& residual_form,
    std::span<const forms::bc::StrongDirichlet> bcs,
    const FormInstallOptions& options = {});

KernelPtr installWeakForm(
    FESystem& system,
    const OperatorTag& op,
    FieldId test_field,
    FieldId trial_field,
    const forms::WeakForm& form,
    const FormInstallOptions& options = {});

std::vector<KernelPtr> installWeakForm(
    FESystem& system,
    std::initializer_list<OperatorTag> ops,
    FieldId test_field,
    FieldId trial_field,
    const forms::WeakForm& form,
    const FormInstallOptions& options = {});

std::vector<std::vector<KernelPtr>> installResidualBlocks(
    FESystem& system,
    const OperatorTag& op,
    std::span<const FieldId> test_fields,
    std::span<const FieldId> trial_fields,
    const forms::BlockBilinearForm& blocks,
    const FormInstallOptions& options = {});

std::vector<std::vector<KernelPtr>> installResidualBlocks(
    FESystem& system,
    const OperatorTag& op,
    std::initializer_list<FieldId> test_fields,
    std::initializer_list<FieldId> trial_fields,
    const forms::BlockBilinearForm& blocks,
    const FormInstallOptions& options = {});

struct CoupledResidualKernels {
    std::vector<KernelPtr> residual;                      // one per test field
    std::vector<std::vector<KernelPtr>> jacobian_blocks;  // [test][trial]
};

/**
 * @brief Install a coupled multi-field residual without double-counting the RHS
 *
 * The input is a block vector of residual forms (one per test field) that may
 * reference multiple state fields via `forms::FormExpr::stateField(FieldId, ...)`.
 *
 * This helper registers:
 * - vector-only residual kernels (one per test field, trial field paired by index),
 * - matrix-only Jacobian kernels for every (test,trial) block.
 */
CoupledResidualKernels installCoupledResidual(
    FESystem& system,
    const OperatorTag& op,
    std::span<const FieldId> test_fields,
    std::span<const FieldId> trial_fields,
    const forms::BlockLinearForm& residual_blocks,
    const FormInstallOptions& options = {});

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_FORMSINSTALLER_H
