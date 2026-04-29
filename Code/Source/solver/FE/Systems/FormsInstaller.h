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
 */

#include "Core/Types.h"

#include "Analysis/FormAnalysisBridge.h"
#include "Forms/BoundaryConditions.h"
#include "Forms/MixedFormIR.h"
#include "Systems/MixedKernelPlan.h"
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

    /// Additional trial fields that contribute tangent columns but do not own
    /// residual rows in this formulation. This is intended for monolithic
    /// couplings such as ALE fluid residuals differentiated with respect to a
    /// solved mesh-displacement field.
    std::vector<FieldId> extra_trial_fields{};
};

using KernelPtr = std::shared_ptr<assembly::AssemblyKernel>;

void installStrongDirichlet(
    FESystem& system,
    std::span<const forms::bc::StrongDirichlet> bcs);

struct CoupledResidualKernels {
    std::vector<KernelPtr> residual;                      // one per test field
    std::vector<std::vector<KernelPtr>> jacobian_blocks;  // [test][trial]
    std::shared_ptr<const MixedKernelPlan> mixed_plan{};
};

struct CoupledResidualMetadata {
    CoupledResidualKernels kernels{};
    analysis::FormContributionAnalysisMetadata analysis{};
};

/**
 * @brief Unified formulation installer — auto-selects single or multi-field path
 *
 * This is the canonical public residual entry point for physics modules. The
 * caller provides a single residual FormExpr and the field IDs for the
 * unknowns. The function
 * inspects the expression to determine the number of test-function spaces and
 * automatically routes to the appropriate internal path:
 *
 *   - 1 TestFunction space → single-field path
 *   - 2+ TestFunction spaces → mixed multi-field path
 *
 * Unknown fields should be created with the field-bound StateField() helper
 * from Vocabulary.h (or equivalently, FormExpr::stateField()). The function
 * handles lowering StateField nodes to TrialFunction/DiscreteField internally.
 * Expressions with unbound TrialFunction (no StateField nodes) are also
 * supported for single-field formulations.
 *
 * For multi-field formulations, the installer builds an explicit MixedKernelPlan
 * and selects either a semantic monolithic cell kernel or exact per-block
 * kernels for the active domains.
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

/**
 * @brief Install a formulation and return public bridge metadata for that install.
 *
 * This wrapper captures the formulation and contribution ranges created by one
 * successful installFormulation() call and adapts them through
 * analysis::buildFormAnalysisMetadata(). Callers should supply a stable
 * contribution name, diagnostic origin, and owning system name in
 * metadata_options when they need setup-time dependency diagnostics.
 */
CoupledResidualMetadata installFormulationWithMetadata(
    FESystem& system,
    const OperatorTag& op,
    std::span<const FieldId> fields,
    const forms::FormExpr& residual,
    const FormInstallOptions& options,
    const analysis::FormAnalysisBridgeOptions& metadata_options = {});

CoupledResidualMetadata installFormulationWithMetadata(
    FESystem& system,
    const OperatorTag& op,
    std::initializer_list<FieldId> fields,
    const forms::FormExpr& residual,
    const FormInstallOptions& options = {},
    const analysis::FormAnalysisBridgeOptions& metadata_options = {});

// ============================================================================
// Mixed-form installation (stable lowering contract)
// ============================================================================

/**
 * @brief Install a pre-compiled MixedFormIR into an FESystem
 *
 * Stable lowering from MixedFormIR to the block-based operator registry for
 * expert/lower-level installation flows. Each active block is installed as an
 * independent operator term through the existing registration model
 * (addCellKernel, addBoundaryKernel, addInteriorFaceKernel,
 * addInterfaceFaceKernel).
 *
 * The block layout produced by this function is identical to what would result
 * from manual block decomposition and per-block installation.
 *
 * Public residual authoring should still enter through installFormulation().
 * For linear forms, use installMixedLinear() which maps the synthetic trial
 * column that compileMixed() creates for the 1-column linear layout.
 *
 * @param system       FESystem to install kernels into
 * @param op           Operator tag
 * @param test_fields  FieldId for each test field (size must match mir.numTestFields())
 * @param trial_fields FieldId for each trial field (size must match mir.numTrialFields())
 * @param mir          Pre-compiled mixed form IR
 * @param options      Compiler and JIT options
 * @return NxM kernel matrix (nullptr for zero blocks)
 */
std::vector<std::vector<KernelPtr>>
installMixedFormIR(
    FESystem& system,
    const OperatorTag& op,
    std::span<const FieldId> test_fields,
    std::span<const FieldId> trial_fields,
    const forms::MixedFormIR& mir,
    const FormInstallOptions& options = {});

/**
 * @brief Compile and install a mixed bilinear form expression
 *
 * Takes a mixed bilinear FormExpr (multiple test/trial spaces), compiles it
 * via FormCompiler::compileMixed(), and installs via installMixedFormIR().
 *
 * This is the bilinear analog of installFormulation() for mixed expressions.
 *
 * @param system       FESystem to install kernels into
 * @param op           Operator tag
 * @param test_fields  FieldId for each test field
 * @param trial_fields FieldId for each trial field
 * @param bilinear     Mixed bilinear form expression
 * @param options      Compiler and JIT options
 * @return NxM kernel matrix (nullptr for zero blocks)
 */
std::vector<std::vector<KernelPtr>>
installMixedBilinear(
    FESystem& system,
    const OperatorTag& op,
    std::span<const FieldId> test_fields,
    std::span<const FieldId> trial_fields,
    const forms::FormExpr& bilinear,
    const FormInstallOptions& options = {});

/**
 * @brief Compile and install a mixed linear form expression
 *
 * Takes a mixed linear FormExpr (multiple test spaces, no trial), compiles it
 * via FormCompiler::compileMixed(), and installs via installMixedFormIR().
 * compileMixed() produces a 1-column MixedFormIR with a synthetic trial
 * column; this function maps it to placeholder trial FieldIds internally.
 *
 * @param system      FESystem to install kernels into
 * @param op          Operator tag
 * @param test_fields FieldId for each test field
 * @param linear      Mixed linear form expression
 * @param options     Compiler and JIT options
 * @return One kernel per test field (nullptr for zero blocks)
 */
std::vector<KernelPtr>
installMixedLinear(
    FESystem& system,
    const OperatorTag& op,
    std::span<const FieldId> test_fields,
    const forms::FormExpr& linear,
    const FormInstallOptions& options = {});

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_FORMSINSTALLER_H
