/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ASSEMBLY_DECORATOR_ASSEMBLER_H
#define SVMP_FE_ASSEMBLY_DECORATOR_ASSEMBLER_H

/**
 * @file DecoratorAssembler.h
 * @brief Base class for composable assembler wrappers
 *
 * DecoratorAssembler forwards the full `assembly::Assembler` interface to an
 * underlying base assembler, enabling orthogonal features (caching/scheduling/
 * vectorization/etc.) to compose without re-implementing `StandardAssembler`.
 */

#include "Assembly/Assembler.h"

#include <memory>
#include <utility>

namespace svmp {
namespace FE {
namespace assembly {

class DecoratorAssembler : public Assembler {
public:
    explicit DecoratorAssembler(std::unique_ptr<Assembler> base)
        : base_(std::move(base))
    {
        FE_THROW_IF(base_ == nullptr, FEException, "DecoratorAssembler: base assembler is null");
    }

    ~DecoratorAssembler() override = default;

    DecoratorAssembler(DecoratorAssembler&&) noexcept = default;
    DecoratorAssembler& operator=(DecoratorAssembler&&) noexcept = default;

    DecoratorAssembler(const DecoratorAssembler&) = delete;
    DecoratorAssembler& operator=(const DecoratorAssembler&) = delete;

    // Bring base-class overloads into scope (avoid overload hiding)
    using Assembler::assembleMatrix;
    using Assembler::assembleBoth;
    using Assembler::assembleBoundaryFaces;

protected:
    [[nodiscard]] Assembler& base() noexcept { return *base_; }
    [[nodiscard]] const Assembler& base() const noexcept { return *base_; }

    [[nodiscard]] std::unique_ptr<Assembler> releaseBase() noexcept { return std::move(base_); }

    void setBase(std::unique_ptr<Assembler> next)
    {
        FE_THROW_IF(next == nullptr, FEException, "DecoratorAssembler::setBase: base assembler is null");
        base_ = std::move(next);
    }

public:
    // =========================================================================
    // Configuration
    // =========================================================================

    void setDofMap(const dofs::DofMap& dof_map) override { base_->setDofMap(dof_map); }

    void setRowDofMap(const dofs::DofMap& dof_map, GlobalIndex row_offset = 0) override
    {
        base_->setRowDofMap(dof_map, row_offset);
    }

    void setColDofMap(const dofs::DofMap& dof_map, GlobalIndex col_offset = 0) override
    {
        base_->setColDofMap(dof_map, col_offset);
    }

    void setDofHandler(const dofs::DofHandler& dof_handler) override { base_->setDofHandler(dof_handler); }

    void setConstraints(const constraints::AffineConstraints* constraints) override
    {
        base_->setConstraints(constraints);
    }

    void setSparsityPattern(const sparsity::SparsityPattern* sparsity) override
    {
        base_->setSparsityPattern(sparsity);
    }

    void setOptions(const AssemblyOptions& options) override { base_->setOptions(options); }

    void setCurrentSolution(std::span<const Real> solution) override { base_->setCurrentSolution(solution); }

    void setCurrentSolutionView(const GlobalSystemView* solution_view) override
    {
        base_->setCurrentSolutionView(solution_view);
    }

    void setFieldSolutionAccess(std::span<const FieldSolutionAccess> fields) override
    {
        base_->setFieldSolutionAccess(fields);
    }

    void setPreviousSolution(std::span<const Real> solution) override { base_->setPreviousSolution(solution); }

    void setPreviousSolution2(std::span<const Real> solution) override { base_->setPreviousSolution2(solution); }

    void setPreviousSolutionK(int k, std::span<const Real> solution) override
    {
        base_->setPreviousSolutionK(k, solution);
    }

    void setTimeIntegrationContext(const TimeIntegrationContext* ctx) override
    {
        base_->setTimeIntegrationContext(ctx);
    }

    void setTime(Real time) override { base_->setTime(time); }
    void setTimeStep(Real dt) override { base_->setTimeStep(dt); }

    void setRealParameterGetter(
        const std::function<std::optional<Real>(std::string_view)>* get_real_param) noexcept override
    {
        base_->setRealParameterGetter(get_real_param);
    }

    void setParameterGetter(
        const std::function<std::optional<params::Value>(std::string_view)>* get_param) noexcept override
    {
        base_->setParameterGetter(get_param);
    }

    void setUserData(const void* user_data) noexcept override { base_->setUserData(user_data); }

    void setJITConstants(std::span<const Real> constants) noexcept override { base_->setJITConstants(constants); }

    void setCoupledValues(std::span<const Real> integrals,
                          std::span<const Real> aux_state) noexcept override
    {
        base_->setCoupledValues(integrals, aux_state);
    }

    void setMaterialStateProvider(IMaterialStateProvider* provider) noexcept override
    {
        base_->setMaterialStateProvider(provider);
    }

    [[nodiscard]] const AssemblyOptions& getOptions() const noexcept override { return base_->getOptions(); }

    // =========================================================================
    // Assembly operations
    // =========================================================================

    [[nodiscard]] AssemblyResult assembleMatrix(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        AssemblyKernel& kernel,
        GlobalSystemView& matrix_view) override
    {
        return base_->assembleMatrix(mesh, test_space, trial_space, kernel, matrix_view);
    }

    [[nodiscard]] AssemblyResult assembleVector(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& space,
        AssemblyKernel& kernel,
        GlobalSystemView& vector_view) override
    {
        return base_->assembleVector(mesh, space, kernel, vector_view);
    }

    [[nodiscard]] AssemblyResult assembleBoth(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        AssemblyKernel& kernel,
        GlobalSystemView& matrix_view,
        GlobalSystemView& vector_view) override
    {
        return base_->assembleBoth(mesh, test_space, trial_space, kernel, matrix_view, vector_view);
    }

    [[nodiscard]] AssemblyResult assembleBoundaryFaces(
        const IMeshAccess& mesh,
        int boundary_marker,
        const spaces::FunctionSpace& space,
        AssemblyKernel& kernel,
        GlobalSystemView* matrix_view,
        GlobalSystemView* vector_view) override
    {
        return base_->assembleBoundaryFaces(mesh, boundary_marker, space, kernel, matrix_view, vector_view);
    }

    [[nodiscard]] AssemblyResult assembleInteriorFaces(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        AssemblyKernel& kernel,
        GlobalSystemView& matrix_view,
        GlobalSystemView* vector_view) override
    {
        return base_->assembleInteriorFaces(mesh, test_space, trial_space, kernel, matrix_view, vector_view);
    }

    // =========================================================================
    // Lifecycle
    // =========================================================================

    void initialize() override { base_->initialize(); }
    void finalize(GlobalSystemView* matrix_view, GlobalSystemView* vector_view) override
    {
        base_->finalize(matrix_view, vector_view);
    }
    void reset() override { base_->reset(); }

    // =========================================================================
    // Query
    // =========================================================================

    [[nodiscard]] std::string name() const override { return base_->name(); }
    [[nodiscard]] bool isConfigured() const noexcept override { return base_->isConfigured(); }

    [[nodiscard]] bool supportsRectangular() const noexcept override { return base_->supportsRectangular(); }
    [[nodiscard]] bool supportsDG() const noexcept override { return base_->supportsDG(); }
    [[nodiscard]] bool supportsFullContext() const noexcept override { return base_->supportsFullContext(); }
    [[nodiscard]] bool supportsSolution() const noexcept override { return base_->supportsSolution(); }
    [[nodiscard]] bool supportsSolutionHistory() const noexcept override { return base_->supportsSolutionHistory(); }
    [[nodiscard]] bool supportsTimeIntegrationContext() const noexcept override { return base_->supportsTimeIntegrationContext(); }
    [[nodiscard]] bool supportsDofOffsets() const noexcept override { return base_->supportsDofOffsets(); }
    [[nodiscard]] bool supportsFieldRequirements() const noexcept override { return base_->supportsFieldRequirements(); }
    [[nodiscard]] bool supportsMaterialState() const noexcept override { return base_->supportsMaterialState(); }
    [[nodiscard]] bool isThreadSafe() const noexcept override { return base_->isThreadSafe(); }

private:
    std::unique_ptr<Assembler> base_;
};

} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ASSEMBLY_DECORATOR_ASSEMBLER_H
