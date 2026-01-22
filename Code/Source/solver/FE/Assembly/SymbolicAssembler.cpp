/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Assembly/SymbolicAssembler.h"

#include "Assembly/StandardAssembler.h"
#include "Forms/JIT/JITKernelWrapper.h"

#include <utility>

namespace svmp {
namespace FE {
namespace assembly {

struct SymbolicAssembler::Impl {
    forms::SymbolicOptions sym_options{};
    forms::FormCompiler compiler{};
};

SymbolicAssembler::SymbolicAssembler()
    : DecoratorAssembler(createStandardAssembler()),
      impl_(std::make_unique<Impl>())
{
}

SymbolicAssembler::SymbolicAssembler(const forms::SymbolicOptions& options)
    : DecoratorAssembler(createStandardAssembler()),
      impl_(std::make_unique<Impl>())
{
    impl_->sym_options = options;
    impl_->compiler.setOptions(options);
}

SymbolicAssembler::SymbolicAssembler(std::unique_ptr<Assembler> base)
    : DecoratorAssembler(std::move(base)),
      impl_(std::make_unique<Impl>())
{
}

SymbolicAssembler::SymbolicAssembler(std::unique_ptr<Assembler> base, const forms::SymbolicOptions& options)
    : DecoratorAssembler(std::move(base)),
      impl_(std::make_unique<Impl>())
{
    impl_->sym_options = options;
    impl_->compiler.setOptions(options);
}

SymbolicAssembler::~SymbolicAssembler() = default;
SymbolicAssembler::SymbolicAssembler(SymbolicAssembler&& other) noexcept = default;
SymbolicAssembler& SymbolicAssembler::operator=(SymbolicAssembler&& other) noexcept = default;

AssemblyResult SymbolicAssembler::assembleForm(const forms::FormExpr& bilinear_form,
                                               const IMeshAccess& mesh,
                                               const spaces::FunctionSpace& test_space,
                                               const spaces::FunctionSpace& trial_space,
                                               GlobalSystemView& matrix_view)
{
    auto ir = impl_->compiler.compileBilinear(bilinear_form);
    std::shared_ptr<AssemblyKernel> kernel = std::make_shared<forms::FormKernel>(std::move(ir));
    if (impl_->sym_options.jit.enable) {
        kernel = std::make_shared<forms::jit::JITKernelWrapper>(kernel, impl_->sym_options.jit);
    }
    return assembleMatrix(mesh, test_space, trial_space, *kernel, matrix_view);
}

AssemblyResult SymbolicAssembler::assembleLinearForm(const forms::FormExpr& linear_form,
                                                     const IMeshAccess& mesh,
                                                     const spaces::FunctionSpace& space,
                                                     GlobalSystemView& vector_view)
{
    auto ir = impl_->compiler.compileLinear(linear_form);
    std::shared_ptr<AssemblyKernel> kernel = std::make_shared<forms::FormKernel>(std::move(ir));
    if (impl_->sym_options.jit.enable) {
        kernel = std::make_shared<forms::jit::JITKernelWrapper>(kernel, impl_->sym_options.jit);
    }
    return assembleVector(mesh, space, *kernel, vector_view);
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
    std::shared_ptr<AssemblyKernel> kernel =
        std::make_shared<forms::NonlinearFormKernel>(std::move(ir), impl_->sym_options.ad_mode);
    if (impl_->sym_options.jit.enable) {
        kernel = std::make_shared<forms::jit::JITKernelWrapper>(kernel, impl_->sym_options.jit);
    }
    return assembleBoth(mesh, space, space, *kernel, jacobian_view, residual_view);
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
    if (!impl_->sym_options.jit.enable) {
        return std::make_unique<forms::FormKernel>(std::move(ir));
    }
    auto fallback = std::make_shared<forms::FormKernel>(std::move(ir));
    return std::make_unique<forms::jit::JITKernelWrapper>(std::move(fallback), impl_->sym_options.jit);
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
