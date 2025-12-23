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
 * @brief Symbolic assembly with automatic differentiation and JIT compilation
 *
 * SymbolicAssembler provides optimized assembly through:
 *
 * 1. FORM INTERMEDIATE REPRESENTATION (FormIR):
 *    Variational forms are expressed symbolically and compiled to efficient code.
 *    - Bilinear forms: a(u,v) = integral grad(u) . grad(v) dx
 *    - Linear forms: L(v) = integral f * v dx
 *    - Supports mixed forms, boundary integrals, DG terms
 *
 * 2. AUTOMATIC DIFFERENTIATION (AD):
 *    Jacobians for nonlinear problems computed automatically.
 *    - Forward mode AD for few inputs, many outputs
 *    - Reverse mode AD for many inputs, few outputs
 *    - Supports Sacado-style AD types
 *
 * 3. JUST-IN-TIME (JIT) COMPILATION:
 *    Runtime code generation for maximum performance.
 *    - Generate optimized quadrature kernels
 *    - Specialize for element type and polynomial degree
 *    - Cache compiled kernels for reuse
 *
 * 4. EXPRESSION TEMPLATES:
 *    Zero-overhead abstractions for form expressions.
 *    - Lazy evaluation of form expressions
 *    - Compile-time expression optimization
 *    - Seamless integration with numerical assembly
 *
 * Design Philosophy:
 * - Write forms once in high-level notation
 * - Framework generates optimal assembly code
 * - Automatic Jacobian computation for Newton methods
 * - JIT compilation removes interpretation overhead
 *
 * Module Boundaries:
 * - SymbolicAssembler OWNS: form representation, AD integration, JIT hooks
 * - SymbolicAssembler does NOT OWN: physics definitions (comes from user forms)
 *
 * @see Assembler for base assembly interface
 * @see AssemblyKernel for computed kernel interface
 */

#include "Core/Types.h"
#include "Assembler.h"
#include "AssemblyKernel.h"
#include "AssemblyContext.h"
#include "GlobalSystemView.h"

#include <vector>
#include <memory>
#include <functional>
#include <string>
#include <unordered_map>
#include <variant>
#include <optional>

namespace svmp {
namespace FE {

// Forward declarations
namespace dofs {
    class DofMap;
    class DofHandler;
}

namespace spaces {
    class FunctionSpace;
}

namespace sparsity {
    class SparsityPattern;
}

namespace constraints {
    class AffineConstraints;
    class ConstraintDistributor;
}

namespace assembly {

// ============================================================================
// Form Expression Types
// ============================================================================

/**
 * @brief Type of form expression
 */
enum class FormExprType {
    // Test/Trial functions
    TestFunction,       ///< v
    TrialFunction,      ///< u
    Coefficient,        ///< c (known function)

    // Derivatives
    Gradient,           ///< grad(u)
    Divergence,         ///< div(u)
    Curl,               ///< curl(u)
    Hessian,            ///< H(u)

    // Operations
    InnerProduct,       ///< a . b
    OuterProduct,       ///< a tensor b
    Multiply,           ///< a * b
    Add,                ///< a + b
    Subtract,           ///< a - b
    Negate,             ///< -a

    // Integration
    CellIntegral,       ///< integral_K dx
    BoundaryIntegral,   ///< integral_Gamma ds
    InteriorFaceIntegral,  ///< integral_F ds (DG)

    // Special
    Constant,           ///< scalar constant
    Identity,           ///< identity matrix
    Normal,             ///< outward normal n
    Jump,               ///< [[u]] (DG jump)
    Average             ///< {u} (DG average)
};

/**
 * @brief AD mode for Jacobian computation
 */
enum class ADMode {
    None,      ///< No automatic differentiation
    Forward,   ///< Forward mode (Sacado::Fad)
    Reverse,   ///< Reverse mode (Sacado::Rad)
    Taylor     ///< Taylor series mode
};

/**
 * @brief JIT compilation options
 */
struct JITOptions {
    /**
     * @brief Enable JIT compilation
     */
    bool enable{false};

    /**
     * @brief Optimization level (0-3)
     */
    int optimization_level{2};

    /**
     * @brief Cache compiled kernels
     */
    bool cache_kernels{true};

    /**
     * @brief Generate SIMD code
     */
    bool vectorize{true};

    /**
     * @brief Kernel cache directory (empty = in-memory only)
     */
    std::string cache_directory;
};

/**
 * @brief Options for symbolic assembly
 */
struct SymbolicOptions {
    /**
     * @brief AD mode for Jacobian computation
     */
    ADMode ad_mode{ADMode::None};

    /**
     * @brief JIT compilation options
     */
    JITOptions jit;

    /**
     * @brief Simplify expressions before compilation
     */
    bool simplify_expressions{true};

    /**
     * @brief Exploit sparsity in Jacobian
     */
    bool exploit_sparsity{true};

    /**
     * @brief Enable expression caching
     */
    bool cache_expressions{true};

    /**
     * @brief Verbose output for debugging
     */
    bool verbose{false};
};

// ============================================================================
// Form Expression Nodes
// ============================================================================

// Forward declaration
class FormExpr;

/**
 * @brief Base class for form expression nodes
 *
 * Forms an expression tree representing variational forms.
 */
class FormExprNode {
public:
    virtual ~FormExprNode() = default;

    /**
     * @brief Get expression type
     */
    [[nodiscard]] virtual FormExprType type() const noexcept = 0;

    /**
     * @brief Get string representation
     */
    [[nodiscard]] virtual std::string toString() const = 0;

    /**
     * @brief Check if expression contains test function
     */
    [[nodiscard]] virtual bool hasTest() const = 0;

    /**
     * @brief Check if expression contains trial function
     */
    [[nodiscard]] virtual bool hasTrial() const = 0;

    /**
     * @brief Get children nodes (if any)
     */
    [[nodiscard]] virtual std::vector<const FormExprNode*> children() const {
        return {};
    }
};

/**
 * @brief Handle class for form expressions with value semantics
 */
class FormExpr {
public:
    FormExpr();
    explicit FormExpr(std::shared_ptr<FormExprNode> node);

    // =========================================================================
    // Factory Methods for Basic Expressions
    // =========================================================================

    /**
     * @brief Test function v
     */
    static FormExpr testFunction(const std::string& name = "v");

    /**
     * @brief Trial function u
     */
    static FormExpr trialFunction(const std::string& name = "u");

    /**
     * @brief Coefficient function (known)
     */
    static FormExpr coefficient(const std::string& name,
                                std::function<Real(Real, Real, Real)> func = nullptr);

    /**
     * @brief Scalar constant
     */
    static FormExpr constant(Real value);

    /**
     * @brief Identity matrix
     */
    static FormExpr identity(int dim);

    /**
     * @brief Outward normal (for boundary integrals)
     */
    static FormExpr normal();

    // =========================================================================
    // Derivative Operators
    // =========================================================================

    /**
     * @brief Gradient: grad(expr)
     */
    [[nodiscard]] FormExpr grad() const;

    /**
     * @brief Divergence: div(expr)
     */
    [[nodiscard]] FormExpr div() const;

    /**
     * @brief Curl: curl(expr)
     */
    [[nodiscard]] FormExpr curl() const;

    /**
     * @brief Hessian: H(expr)
     */
    [[nodiscard]] FormExpr hessian() const;

    // =========================================================================
    // DG Operators
    // =========================================================================

    /**
     * @brief Jump across face: [[u]]
     */
    [[nodiscard]] FormExpr jump() const;

    /**
     * @brief Average across face: {u}
     */
    [[nodiscard]] FormExpr avg() const;

    // =========================================================================
    // Arithmetic Operators
    // =========================================================================

    FormExpr operator-() const;
    FormExpr operator+(const FormExpr& rhs) const;
    FormExpr operator-(const FormExpr& rhs) const;
    FormExpr operator*(const FormExpr& rhs) const;
    FormExpr operator*(Real scalar) const;

    /**
     * @brief Inner product: a . b
     */
    [[nodiscard]] FormExpr inner(const FormExpr& rhs) const;

    /**
     * @brief Outer product: a tensor b
     */
    [[nodiscard]] FormExpr outer(const FormExpr& rhs) const;

    // =========================================================================
    // Integration
    // =========================================================================

    /**
     * @brief Cell integral: integral_K expr dx
     */
    [[nodiscard]] FormExpr dx() const;

    /**
     * @brief Boundary integral: integral_Gamma expr ds
     */
    [[nodiscard]] FormExpr ds(int boundary_marker = -1) const;

    /**
     * @brief Interior face integral: integral_F expr dS (DG)
     */
    [[nodiscard]] FormExpr dS() const;

    // =========================================================================
    // Query
    // =========================================================================

    [[nodiscard]] bool isValid() const noexcept { return node_ != nullptr; }
    [[nodiscard]] const FormExprNode* node() const { return node_.get(); }
    [[nodiscard]] std::string toString() const;
    [[nodiscard]] bool hasTest() const;
    [[nodiscard]] bool hasTrial() const;
    [[nodiscard]] bool isBilinear() const { return hasTest() && hasTrial(); }
    [[nodiscard]] bool isLinear() const { return hasTest() && !hasTrial(); }

private:
    std::shared_ptr<FormExprNode> node_;
};

// Free function operators
inline FormExpr operator*(Real scalar, const FormExpr& expr) {
    return expr * scalar;
}

// Convenience functions
inline FormExpr grad(const FormExpr& expr) { return expr.grad(); }
inline FormExpr div(const FormExpr& expr) { return expr.div(); }
inline FormExpr curl(const FormExpr& expr) { return expr.curl(); }
inline FormExpr inner(const FormExpr& a, const FormExpr& b) { return a.inner(b); }
inline FormExpr jump(const FormExpr& expr) { return expr.jump(); }
inline FormExpr avg(const FormExpr& expr) { return expr.avg(); }

// ============================================================================
// Form Intermediate Representation (FormIR)
// ============================================================================

/**
 * @brief Compiled intermediate representation of a variational form
 *
 * FormIR is the result of compiling a FormExpr into an efficient representation
 * suitable for numerical evaluation.
 */
class FormIR {
public:
    FormIR();
    ~FormIR();

    FormIR(FormIR&& other) noexcept;
    FormIR& operator=(FormIR&& other) noexcept;

    // Non-copyable
    FormIR(const FormIR&) = delete;
    FormIR& operator=(const FormIR&) = delete;

    // =========================================================================
    // Query
    // =========================================================================

    /**
     * @brief Check if IR is compiled and ready
     */
    [[nodiscard]] bool isCompiled() const noexcept;

    /**
     * @brief Get required data flags for assembly
     */
    [[nodiscard]] RequiredData getRequiredData() const noexcept;

    /**
     * @brief Get human-readable IR representation
     */
    [[nodiscard]] std::string dump() const;

    /**
     * @brief Check if form is bilinear (matrix assembly)
     */
    [[nodiscard]] bool isBilinear() const noexcept;

    /**
     * @brief Check if form is linear (vector assembly)
     */
    [[nodiscard]] bool isLinear() const noexcept;

    /**
     * @brief Check if form has cell terms
     */
    [[nodiscard]] bool hasCellTerms() const noexcept;

    /**
     * @brief Check if form has boundary terms
     */
    [[nodiscard]] bool hasBoundaryTerms() const noexcept;

    /**
     * @brief Check if form has interior face terms (DG)
     */
    [[nodiscard]] bool hasFaceTerms() const noexcept;

private:
    friend class FormCompiler;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ============================================================================
// Form Compiler
// ============================================================================

/**
 * @brief Compiles symbolic forms to FormIR
 */
class FormCompiler {
public:
    FormCompiler();
    explicit FormCompiler(const SymbolicOptions& options);
    ~FormCompiler();

    /**
     * @brief Compile a form expression to IR
     */
    [[nodiscard]] FormIR compile(const FormExpr& form);

    /**
     * @brief Compile bilinear form a(u,v)
     */
    [[nodiscard]] FormIR compileBilinear(const FormExpr& form);

    /**
     * @brief Compile linear form L(v)
     */
    [[nodiscard]] FormIR compileLinear(const FormExpr& form);

    /**
     * @brief Compile nonlinear form F(u;v) with AD-based Jacobian
     */
    [[nodiscard]] std::pair<FormIR, FormIR> compileNonlinear(
        const FormExpr& residual,
        ADMode ad_mode);

    /**
     * @brief Set compilation options
     */
    void setOptions(const SymbolicOptions& options);

    /**
     * @brief Get compilation statistics
     */
    struct CompileStats {
        double compile_seconds{0.0};
        std::size_t num_cell_terms{0};
        std::size_t num_boundary_terms{0};
        std::size_t num_face_terms{0};
        bool simplified{false};
        bool jit_compiled{false};
    };
    [[nodiscard]] const CompileStats& getLastStats() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ============================================================================
// Symbolic Assembly Kernel
// ============================================================================

/**
 * @brief Assembly kernel generated from symbolic form
 */
class SymbolicKernel : public AssemblyKernel {
public:
    explicit SymbolicKernel(FormIR form_ir);
    ~SymbolicKernel() override;

    SymbolicKernel(SymbolicKernel&& other) noexcept;
    SymbolicKernel& operator=(SymbolicKernel&& other) noexcept;

    // Non-copyable
    SymbolicKernel(const SymbolicKernel&) = delete;
    SymbolicKernel& operator=(const SymbolicKernel&) = delete;

    // =========================================================================
    // AssemblyKernel Interface
    // =========================================================================

    [[nodiscard]] RequiredData getRequiredData() const noexcept override;
    [[nodiscard]] bool hasCell() const noexcept override;
    [[nodiscard]] bool hasBoundaryFace() const noexcept override;
    [[nodiscard]] bool hasInteriorFace() const noexcept override;

    void computeCell(const AssemblyContext& ctx, KernelOutput& output) override;

    void computeBoundaryFace(
        const AssemblyContext& ctx,
        int boundary_marker,
        KernelOutput& output) override;

    void computeInteriorFace(
        const AssemblyContext& ctx_minus,
        const AssemblyContext& ctx_plus,
        KernelOutput& output_minus,
        KernelOutput& output_plus,
        KernelOutput& coupling_mp,
        KernelOutput& coupling_pm) override;

private:
    FormIR form_ir_;
};

// ============================================================================
// SymbolicAssembler
// ============================================================================

/**
 * @brief Assembler using symbolic form expressions
 *
 * SymbolicAssembler provides a high-level interface for assembly based on
 * variational form expressions. Forms are compiled to efficient code and
 * evaluated during assembly.
 *
 * Usage:
 * @code
 *   SymbolicAssembler assembler;
 *   assembler.setDofMap(dof_map);
 *
 *   // Define bilinear form: a(u,v) = integral grad(u) . grad(v) dx
 *   auto u = FormExpr::trialFunction("u");
 *   auto v = FormExpr::testFunction("v");
 *   auto a = inner(grad(u), grad(v)).dx();
 *
 *   // Assemble
 *   assembler.assembleForm(a, mesh, test_space, trial_space, matrix_view);
 *
 *   // For nonlinear problems with automatic Jacobian:
 *   SymbolicOptions opts{.ad_mode = ADMode::Forward};
 *   assembler.setOptions(opts);
 *
 *   auto residual = ...;  // Define nonlinear residual
 *   assembler.assembleResidualAndJacobian(residual, mesh, space, u_h,
 *                                          matrix_view, vector_view);
 * @endcode
 */
class SymbolicAssembler : public Assembler {
public:
    // =========================================================================
    // Construction
    // =========================================================================

    SymbolicAssembler();
    explicit SymbolicAssembler(const SymbolicOptions& options);
    ~SymbolicAssembler() override;

    SymbolicAssembler(SymbolicAssembler&& other) noexcept;
    SymbolicAssembler& operator=(SymbolicAssembler&& other) noexcept;

    // Non-copyable
    SymbolicAssembler(const SymbolicAssembler&) = delete;
    SymbolicAssembler& operator=(const SymbolicAssembler&) = delete;

    // =========================================================================
    // Configuration (Assembler interface)
    // =========================================================================

    void setDofMap(const dofs::DofMap& dof_map) override;
    void setDofHandler(const dofs::DofHandler& dof_handler) override;
    void setConstraints(const constraints::AffineConstraints* constraints) override;
    void setSparsityPattern(const sparsity::SparsityPattern* sparsity) override;
    void setOptions(const AssemblyOptions& options) override;

    [[nodiscard]] const AssemblyOptions& getOptions() const noexcept override;
    [[nodiscard]] bool isConfigured() const noexcept override;
    [[nodiscard]] std::string name() const override { return "SymbolicAssembler"; }

    // =========================================================================
    // Symbolic-Specific Configuration
    // =========================================================================

    /**
     * @brief Set symbolic assembly options
     */
    void setSymbolicOptions(const SymbolicOptions& options);

    /**
     * @brief Get current symbolic options
     */
    [[nodiscard]] const SymbolicOptions& getSymbolicOptions() const noexcept;

    // =========================================================================
    // Lifecycle
    // =========================================================================

    void initialize() override;
    void finalize(GlobalSystemView* matrix_view, GlobalSystemView* vector_view) override;
    void reset() override;

    // =========================================================================
    // Standard Assembly (Assembler interface)
    // =========================================================================

    AssemblyResult assembleMatrix(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        AssemblyKernel& kernel,
        GlobalSystemView& matrix_view) override;

    AssemblyResult assembleVector(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& space,
        AssemblyKernel& kernel,
        GlobalSystemView& vector_view) override;

    AssemblyResult assembleBoth(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        AssemblyKernel& kernel,
        GlobalSystemView& matrix_view,
        GlobalSystemView& vector_view) override;

    AssemblyResult assembleBoundaryFaces(
        const IMeshAccess& mesh,
        int boundary_marker,
        const spaces::FunctionSpace& space,
        AssemblyKernel& kernel,
        GlobalSystemView* matrix_view,
        GlobalSystemView* vector_view) override;

    AssemblyResult assembleInteriorFaces(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        AssemblyKernel& kernel,
        GlobalSystemView& matrix_view,
        GlobalSystemView* vector_view) override;

    // =========================================================================
    // Symbolic Form Assembly
    // =========================================================================

    /**
     * @brief Compile and assemble a bilinear form
     */
    AssemblyResult assembleForm(
        const FormExpr& form,
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        GlobalSystemView& matrix_view);

    /**
     * @brief Compile and assemble a linear form
     */
    AssemblyResult assembleLinearForm(
        const FormExpr& form,
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& space,
        GlobalSystemView& vector_view);

    /**
     * @brief Assemble nonlinear residual and Jacobian using AD
     *
     * @param residual_form Residual expression F(u;v)
     * @param mesh Mesh
     * @param space Function space
     * @param solution Current solution coefficients
     * @param jacobian_view Output: Jacobian matrix
     * @param residual_view Output: Residual vector
     * @return Assembly statistics
     */
    AssemblyResult assembleResidualAndJacobian(
        const FormExpr& residual_form,
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& space,
        std::span<const Real> solution,
        GlobalSystemView& jacobian_view,
        GlobalSystemView& residual_view);

    // =========================================================================
    // Precompilation
    // =========================================================================

    /**
     * @brief Precompile a form for repeated assembly
     *
     * @param form Form expression
     * @return Compiled kernel (reusable)
     */
    [[nodiscard]] std::unique_ptr<SymbolicKernel> precompile(const FormExpr& form);

    /**
     * @brief Clear compiled kernel cache
     */
    void clearCache();

private:
    // =========================================================================
    // Internal
    // =========================================================================

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * @brief Create symbolic assembler with default options
 */
std::unique_ptr<Assembler> createSymbolicAssembler();

/**
 * @brief Create symbolic assembler with specified options
 */
std::unique_ptr<Assembler> createSymbolicAssembler(const SymbolicOptions& options);

} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ASSEMBLY_SYMBOLIC_ASSEMBLER_H
