/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef SVMP_FE_ASSEMBLY_SYMBOLIC_ASSEMBLER_H
#define SVMP_FE_ASSEMBLY_SYMBOLIC_ASSEMBLER_H

/**
 * @file SymbolicAssembler.h
 * @brief Assembler backend that assembles FE/Forms weak forms via FE/Assembly
 *
 * The symbolic form vocabulary and compiler live in `FE/Forms`. This assembler
 * provides a convenience layer that:
 * - compiles `forms::FormExpr` → `forms::FormIR`,
 * - instantiates the appropriate `forms::*Kernel` adapter,
 * - delegates execution to `StandardAssembler`.
 */

#include "Assembly/DecoratorAssembler.h"
#include "Forms/Forms.h"

#include <memory>
#include <span>

namespace svmp {
namespace FE {
namespace assembly {

/**
 * @brief Assembler using FE/Forms symbolic expressions
 */
class SymbolicAssembler : public DecoratorAssembler {
public:
    SymbolicAssembler();
    explicit SymbolicAssembler(const forms::SymbolicOptions& options);
    explicit SymbolicAssembler(std::unique_ptr<Assembler> base);
    SymbolicAssembler(std::unique_ptr<Assembler> base, const forms::SymbolicOptions& options);
    ~SymbolicAssembler() override;

    SymbolicAssembler(SymbolicAssembler&& other) noexcept;
    SymbolicAssembler& operator=(SymbolicAssembler&& other) noexcept;

    SymbolicAssembler(const SymbolicAssembler&) = delete;
    SymbolicAssembler& operator=(const SymbolicAssembler&) = delete;

    [[nodiscard]] std::string name() const override { return "Symbolic(" + base().name() + ")"; }

    // ---- Forms assembly convenience ----
    AssemblyResult assembleForm(const forms::FormExpr& bilinear_form,
                                const IMeshAccess& mesh,
                                const spaces::FunctionSpace& test_space,
                                const spaces::FunctionSpace& trial_space,
                                GlobalSystemView& matrix_view);

    AssemblyResult assembleLinearForm(const forms::FormExpr& linear_form,
                                      const IMeshAccess& mesh,
                                      const spaces::FunctionSpace& space,
                                      GlobalSystemView& vector_view);

    AssemblyResult assembleResidualAndJacobian(const forms::FormExpr& residual_form,
                                               const IMeshAccess& mesh,
                                               const spaces::FunctionSpace& space,
                                               std::span<const Real> solution,
                                               GlobalSystemView& jacobian_view,
                                               GlobalSystemView& residual_view);

    // ---- Symbolic options ----
    void setSymbolicOptions(forms::SymbolicOptions options);
    [[nodiscard]] const forms::SymbolicOptions& getSymbolicOptions() const noexcept;

    // ---- Precompilation ----
    [[nodiscard]] std::unique_ptr<AssemblyKernel> precompileBilinear(const forms::FormExpr& bilinear_form);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::unique_ptr<Assembler> createSymbolicAssembler();
std::unique_ptr<Assembler> createSymbolicAssembler(const forms::SymbolicOptions& options);

} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ASSEMBLY_SYMBOLIC_ASSEMBLER_H
