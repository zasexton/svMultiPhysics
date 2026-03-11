/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_FORMSINSTALLERDETAIL_H
#define SVMP_FE_SYSTEMS_FORMSINSTALLERDETAIL_H

/**
 * @file FormsInstallerDetail.h
 * @brief Internal FormsInstaller APIs — not part of the public interface.
 *
 * These functions expose low-level control over form lowering and kernel
 * installation. They are used internally and by unit tests that verify
 * specific lowering behaviour.
 *
 * Physics modules should use installFormulation() from FormsInstaller.h
 * instead, which auto-selects the correct installation path.
 */

#include "Systems/FormsInstaller.h"

#include "Forms/BlockForm.h"
#include "Forms/MixedFormIR.h"

namespace svmp {
namespace FE {

namespace forms {
struct WeakForm;
}

namespace systems {

KernelPtr installResidualForm(
    FESystem& system,
    const OperatorTag& op,
    FieldId test_field,
    FieldId trial_field,
    const forms::FormExpr& residual_form,
    const FormInstallOptions& options = {});

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

CoupledResidualKernels installCoupledResidual(
    FESystem& system,
    const OperatorTag& op,
    std::span<const FieldId> test_fields,
    std::span<const FieldId> trial_fields,
    const forms::BlockLinearForm& residual_blocks,
    const FormInstallOptions& options = {});

CoupledResidualKernels installCoupledResidualMixed(
    FESystem& system,
    const OperatorTag& op,
    std::span<const FieldId> test_fields,
    std::span<const FieldId> trial_fields,
    const forms::FormExpr& mixed_residual,
    const FormInstallOptions& options = {});

std::vector<std::vector<KernelPtr>>
installMixedFormIR(
    FESystem& system,
    const OperatorTag& op,
    std::span<const FieldId> test_fields,
    std::span<const FieldId> trial_fields,
    const forms::MixedFormIR& mir,
    const FormInstallOptions& options = {});

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_FORMSINSTALLERDETAIL_H
