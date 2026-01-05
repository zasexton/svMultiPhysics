/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Assembly/SymbolicAssembler.h"

#include "Assembly/StandardAssembler.h"

#include <stdexcept>
#include <utility>

namespace svmp {
namespace FE {
namespace assembly {

struct SymbolicAssembler::Impl {
    forms::SymbolicOptions sym_options{};
    forms::FormCompiler compiler{};
    AssemblyOptions options{};
    std::unique_ptr<StandardAssembler> standard_assembler{std::make_unique<StandardAssembler>()};
};

SymbolicAssembler::SymbolicAssembler()
    : impl_(std::make_unique<Impl>())
{
}

SymbolicAssembler::SymbolicAssembler(const forms::SymbolicOptions& options)
    : impl_(std::make_unique<Impl>())
{
    impl_->sym_options = options;
    impl_->compiler.setOptions(options);
}

SymbolicAssembler::~SymbolicAssembler() = default;
SymbolicAssembler::SymbolicAssembler(SymbolicAssembler&& other) noexcept = default;
SymbolicAssembler& SymbolicAssembler::operator=(SymbolicAssembler&& other) noexcept = default;

void SymbolicAssembler::setDofMap(const dofs::DofMap& dof_map)
{
    impl_->standard_assembler->setDofMap(dof_map);
}

void SymbolicAssembler::setDofHandler(const dofs::DofHandler& dof_handler)
{
    impl_->standard_assembler->setDofHandler(dof_handler);
}

void SymbolicAssembler::setConstraints(const constraints::AffineConstraints* constraints)
{
    impl_->standard_assembler->setConstraints(constraints);
}

void SymbolicAssembler::setSparsityPattern(const sparsity::SparsityPattern* sparsity)
{
    impl_->standard_assembler->setSparsityPattern(sparsity);
}

void SymbolicAssembler::setOptions(const AssemblyOptions& options)
{
    impl_->options = options;
    impl_->standard_assembler->setOptions(options);
}

void SymbolicAssembler::setCurrentSolution(std::span<const Real> solution)
{
    impl_->standard_assembler->setCurrentSolution(solution);
}

void SymbolicAssembler::setPreviousSolution(std::span<const Real> solution)
{
    impl_->standard_assembler->setPreviousSolution(solution);
}

void SymbolicAssembler::setPreviousSolution2(std::span<const Real> solution)
{
    impl_->standard_assembler->setPreviousSolution2(solution);
}

void SymbolicAssembler::setPreviousSolutionK(int k, std::span<const Real> solution)
{
    impl_->standard_assembler->setPreviousSolutionK(k, solution);
}

void SymbolicAssembler::setTimeIntegrationContext(const TimeIntegrationContext* ctx)
{
    impl_->standard_assembler->setTimeIntegrationContext(ctx);
}

const AssemblyOptions& SymbolicAssembler::getOptions() const noexcept
{
    return impl_->options;
}

bool SymbolicAssembler::isConfigured() const noexcept
{
    return impl_->standard_assembler->isConfigured();
}

void SymbolicAssembler::initialize()
{
    if (!isConfigured()) {
        throw std::runtime_error("SymbolicAssembler::initialize: not configured");
    }
    impl_->standard_assembler->initialize();
}

void SymbolicAssembler::finalize(GlobalSystemView* matrix_view, GlobalSystemView* vector_view)
{
    impl_->standard_assembler->finalize(matrix_view, vector_view);
}

void SymbolicAssembler::reset()
{
    impl_->standard_assembler->reset();
}

AssemblyResult SymbolicAssembler::assembleMatrix(const IMeshAccess& mesh,
                                                 const spaces::FunctionSpace& test_space,
                                                 const spaces::FunctionSpace& trial_space,
                                                 AssemblyKernel& kernel,
                                                 GlobalSystemView& matrix_view)
{
    return impl_->standard_assembler->assembleMatrix(mesh, test_space, trial_space, kernel, matrix_view);
}

AssemblyResult SymbolicAssembler::assembleVector(const IMeshAccess& mesh,
                                                 const spaces::FunctionSpace& space,
                                                 AssemblyKernel& kernel,
                                                 GlobalSystemView& vector_view)
{
    return impl_->standard_assembler->assembleVector(mesh, space, kernel, vector_view);
}

AssemblyResult SymbolicAssembler::assembleBoth(const IMeshAccess& mesh,
                                               const spaces::FunctionSpace& test_space,
                                               const spaces::FunctionSpace& trial_space,
                                               AssemblyKernel& kernel,
                                               GlobalSystemView& matrix_view,
                                               GlobalSystemView& vector_view)
{
    return impl_->standard_assembler->assembleBoth(mesh, test_space, trial_space, kernel, matrix_view, vector_view);
}

AssemblyResult SymbolicAssembler::assembleBoundaryFaces(const IMeshAccess& mesh,
                                                        int boundary_marker,
                                                        const spaces::FunctionSpace& space,
                                                        AssemblyKernel& kernel,
                                                        GlobalSystemView* matrix_view,
                                                        GlobalSystemView* vector_view)
{
    return impl_->standard_assembler->assembleBoundaryFaces(mesh, boundary_marker, space, kernel, matrix_view, vector_view);
}

AssemblyResult SymbolicAssembler::assembleInteriorFaces(const IMeshAccess& mesh,
                                                        const spaces::FunctionSpace& test_space,
                                                        const spaces::FunctionSpace& trial_space,
                                                        AssemblyKernel& kernel,
                                                        GlobalSystemView& matrix_view,
                                                        GlobalSystemView* vector_view)
{
    return impl_->standard_assembler->assembleInteriorFaces(mesh, test_space, trial_space, kernel, matrix_view, vector_view);
}

AssemblyResult SymbolicAssembler::assembleForm(const forms::FormExpr& bilinear_form,
                                               const IMeshAccess& mesh,
                                               const spaces::FunctionSpace& test_space,
                                               const spaces::FunctionSpace& trial_space,
                                               GlobalSystemView& matrix_view)
{
    auto ir = impl_->compiler.compileBilinear(bilinear_form);
    forms::FormKernel kernel(std::move(ir));
    return assembleMatrix(mesh, test_space, trial_space, kernel, matrix_view);
}

AssemblyResult SymbolicAssembler::assembleLinearForm(const forms::FormExpr& linear_form,
                                                     const IMeshAccess& mesh,
                                                     const spaces::FunctionSpace& space,
                                                     GlobalSystemView& vector_view)
{
    auto ir = impl_->compiler.compileLinear(linear_form);
    forms::FormKernel kernel(std::move(ir));
    return assembleVector(mesh, space, kernel, vector_view);
}

AssemblyResult SymbolicAssembler::assembleResidualAndJacobian(const forms::FormExpr& residual_form,
                                                              const IMeshAccess& mesh,
                                                              const spaces::FunctionSpace& space,
                                                              std::span<const Real> solution,
                                                              GlobalSystemView& jacobian_view,
                                                              GlobalSystemView& residual_view)
{
    setCurrentSolution(solution);
    auto ir = impl_->compiler.compileResidual(residual_form);
    forms::NonlinearFormKernel kernel(std::move(ir), impl_->sym_options.ad_mode);
    return assembleBoth(mesh, space, space, kernel, jacobian_view, residual_view);
}

void SymbolicAssembler::setSymbolicOptions(forms::SymbolicOptions options)
{
    impl_->sym_options = std::move(options);
    impl_->compiler.setOptions(impl_->sym_options);
}

const forms::SymbolicOptions& SymbolicAssembler::getSymbolicOptions() const noexcept
{
    return impl_->sym_options;
}

std::unique_ptr<AssemblyKernel> SymbolicAssembler::precompileBilinear(const forms::FormExpr& bilinear_form)
{
    auto ir = impl_->compiler.compileBilinear(bilinear_form);
    return std::make_unique<forms::FormKernel>(std::move(ir));
}

std::unique_ptr<Assembler> createSymbolicAssembler()
{
    return std::make_unique<SymbolicAssembler>();
}

std::unique_ptr<Assembler> createSymbolicAssembler(const forms::SymbolicOptions& options)
{
    return std::make_unique<SymbolicAssembler>(options);
}

} // namespace assembly
} // namespace FE
} // namespace svmp
