/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "SymbolicAssembler.h"
#include "StandardAssembler.h"
#include "Dofs/DofMap.h"
#include "Dofs/DofHandler.h"
#include "Constraints/AffineConstraints.h"
#include "Constraints/ConstraintDistributor.h"
#include "Sparsity/SparsityPattern.h"
#include "Spaces/FunctionSpace.h"
#include "Elements/Element.h"

#include <chrono>
#include <sstream>
#include <stdexcept>

namespace svmp {
namespace FE {
namespace assembly {

// ============================================================================
// Concrete Expression Node Implementations
// ============================================================================

namespace {

class TestFunctionNode : public FormExprNode {
public:
    explicit TestFunctionNode(std::string name) : name_(std::move(name)) {}

    [[nodiscard]] FormExprType type() const noexcept override {
        return FormExprType::TestFunction;
    }

    [[nodiscard]] std::string toString() const override { return name_; }
    [[nodiscard]] bool hasTest() const override { return true; }
    [[nodiscard]] bool hasTrial() const override { return false; }

private:
    std::string name_;
};

class TrialFunctionNode : public FormExprNode {
public:
    explicit TrialFunctionNode(std::string name) : name_(std::move(name)) {}

    [[nodiscard]] FormExprType type() const noexcept override {
        return FormExprType::TrialFunction;
    }

    [[nodiscard]] std::string toString() const override { return name_; }
    [[nodiscard]] bool hasTest() const override { return false; }
    [[nodiscard]] bool hasTrial() const override { return true; }

private:
    std::string name_;
};

class CoefficientNode : public FormExprNode {
public:
    CoefficientNode(std::string name, std::function<Real(Real, Real, Real)> func)
        : name_(std::move(name))
        , func_(std::move(func))
    {
    }

    [[nodiscard]] FormExprType type() const noexcept override {
        return FormExprType::Coefficient;
    }

    [[nodiscard]] std::string toString() const override { return name_; }
    [[nodiscard]] bool hasTest() const override { return false; }
    [[nodiscard]] bool hasTrial() const override { return false; }

private:
    std::string name_;
    std::function<Real(Real, Real, Real)> func_;
};

class ConstantNode : public FormExprNode {
public:
    explicit ConstantNode(Real value) : value_(value) {}

    [[nodiscard]] FormExprType type() const noexcept override {
        return FormExprType::Constant;
    }

    [[nodiscard]] std::string toString() const override {
        return std::to_string(value_);
    }

    [[nodiscard]] bool hasTest() const override { return false; }
    [[nodiscard]] bool hasTrial() const override { return false; }

    [[nodiscard]] Real value() const { return value_; }

private:
    Real value_;
};

class GradientNode : public FormExprNode {
public:
    explicit GradientNode(std::shared_ptr<FormExprNode> arg) : arg_(std::move(arg)) {}

    [[nodiscard]] FormExprType type() const noexcept override {
        return FormExprType::Gradient;
    }

    [[nodiscard]] std::string toString() const override {
        return "grad(" + arg_->toString() + ")";
    }

    [[nodiscard]] bool hasTest() const override { return arg_->hasTest(); }
    [[nodiscard]] bool hasTrial() const override { return arg_->hasTrial(); }

    [[nodiscard]] std::vector<const FormExprNode*> children() const override {
        return {arg_.get()};
    }

private:
    std::shared_ptr<FormExprNode> arg_;
};

class InnerProductNode : public FormExprNode {
public:
    InnerProductNode(std::shared_ptr<FormExprNode> lhs, std::shared_ptr<FormExprNode> rhs)
        : lhs_(std::move(lhs)), rhs_(std::move(rhs)) {}

    [[nodiscard]] FormExprType type() const noexcept override {
        return FormExprType::InnerProduct;
    }

    [[nodiscard]] std::string toString() const override {
        return "(" + lhs_->toString() + " . " + rhs_->toString() + ")";
    }

    [[nodiscard]] bool hasTest() const override {
        return lhs_->hasTest() || rhs_->hasTest();
    }

    [[nodiscard]] bool hasTrial() const override {
        return lhs_->hasTrial() || rhs_->hasTrial();
    }

    [[nodiscard]] std::vector<const FormExprNode*> children() const override {
        return {lhs_.get(), rhs_.get()};
    }

private:
    std::shared_ptr<FormExprNode> lhs_, rhs_;
};

class MultiplyNode : public FormExprNode {
public:
    MultiplyNode(std::shared_ptr<FormExprNode> lhs, std::shared_ptr<FormExprNode> rhs)
        : lhs_(std::move(lhs)), rhs_(std::move(rhs)) {}

    [[nodiscard]] FormExprType type() const noexcept override {
        return FormExprType::Multiply;
    }

    [[nodiscard]] std::string toString() const override {
        return "(" + lhs_->toString() + " * " + rhs_->toString() + ")";
    }

    [[nodiscard]] bool hasTest() const override {
        return lhs_->hasTest() || rhs_->hasTest();
    }

    [[nodiscard]] bool hasTrial() const override {
        return lhs_->hasTrial() || rhs_->hasTrial();
    }

    [[nodiscard]] std::vector<const FormExprNode*> children() const override {
        return {lhs_.get(), rhs_.get()};
    }

private:
    std::shared_ptr<FormExprNode> lhs_, rhs_;
};

class AddNode : public FormExprNode {
public:
    AddNode(std::shared_ptr<FormExprNode> lhs, std::shared_ptr<FormExprNode> rhs)
        : lhs_(std::move(lhs)), rhs_(std::move(rhs)) {}

    [[nodiscard]] FormExprType type() const noexcept override {
        return FormExprType::Add;
    }

    [[nodiscard]] std::string toString() const override {
        return "(" + lhs_->toString() + " + " + rhs_->toString() + ")";
    }

    [[nodiscard]] bool hasTest() const override {
        return lhs_->hasTest() || rhs_->hasTest();
    }

    [[nodiscard]] bool hasTrial() const override {
        return lhs_->hasTrial() || rhs_->hasTrial();
    }

    [[nodiscard]] std::vector<const FormExprNode*> children() const override {
        return {lhs_.get(), rhs_.get()};
    }

private:
    std::shared_ptr<FormExprNode> lhs_, rhs_;
};

class CellIntegralNode : public FormExprNode {
public:
    explicit CellIntegralNode(std::shared_ptr<FormExprNode> integrand)
        : integrand_(std::move(integrand)) {}

    [[nodiscard]] FormExprType type() const noexcept override {
        return FormExprType::CellIntegral;
    }

    [[nodiscard]] std::string toString() const override {
        return "integral_K(" + integrand_->toString() + ") dx";
    }

    [[nodiscard]] bool hasTest() const override { return integrand_->hasTest(); }
    [[nodiscard]] bool hasTrial() const override { return integrand_->hasTrial(); }

    [[nodiscard]] std::vector<const FormExprNode*> children() const override {
        return {integrand_.get()};
    }

private:
    std::shared_ptr<FormExprNode> integrand_;
};

class BoundaryIntegralNode : public FormExprNode {
public:
    BoundaryIntegralNode(std::shared_ptr<FormExprNode> integrand, int marker)
        : integrand_(std::move(integrand)), marker_(marker) {}

    [[nodiscard]] FormExprType type() const noexcept override {
        return FormExprType::BoundaryIntegral;
    }

    [[nodiscard]] std::string toString() const override {
        std::string marker_str = marker_ >= 0 ? std::to_string(marker_) : "all";
        return "integral_Gamma_" + marker_str + "(" + integrand_->toString() + ") ds";
    }

    [[nodiscard]] bool hasTest() const override { return integrand_->hasTest(); }
    [[nodiscard]] bool hasTrial() const override { return integrand_->hasTrial(); }

    [[nodiscard]] std::vector<const FormExprNode*> children() const override {
        return {integrand_.get()};
    }

private:
    std::shared_ptr<FormExprNode> integrand_;
    int marker_;
};

} // anonymous namespace

// ============================================================================
// FormExpr Implementation
// ============================================================================

FormExpr::FormExpr() = default;

FormExpr::FormExpr(std::shared_ptr<FormExprNode> node)
    : node_(std::move(node))
{
}

FormExpr FormExpr::testFunction(const std::string& name)
{
    return FormExpr(std::make_shared<TestFunctionNode>(name));
}

FormExpr FormExpr::trialFunction(const std::string& name)
{
    return FormExpr(std::make_shared<TrialFunctionNode>(name));
}

FormExpr FormExpr::coefficient(const std::string& name,
                               std::function<Real(Real, Real, Real)> func)
{
    return FormExpr(std::make_shared<CoefficientNode>(name, std::move(func)));
}

FormExpr FormExpr::constant(Real value)
{
    return FormExpr(std::make_shared<ConstantNode>(value));
}

FormExpr FormExpr::identity(int /*dim*/)
{
    return FormExpr(std::make_shared<ConstantNode>(1.0));  // Simplified
}

FormExpr FormExpr::normal()
{
    return FormExpr(std::make_shared<TestFunctionNode>("n"));  // Simplified
}

FormExpr FormExpr::grad() const
{
    return FormExpr(std::make_shared<GradientNode>(node_));
}

FormExpr FormExpr::div() const
{
    // Simplified: divergence as special gradient
    return FormExpr(std::make_shared<GradientNode>(node_));
}

FormExpr FormExpr::curl() const
{
    return FormExpr(std::make_shared<GradientNode>(node_));
}

FormExpr FormExpr::hessian() const
{
    return grad().grad();
}

FormExpr FormExpr::jump() const
{
    return *this;  // Simplified for non-DG
}

FormExpr FormExpr::avg() const
{
    return *this;  // Simplified for non-DG
}

FormExpr FormExpr::operator-() const
{
    return FormExpr::constant(-1.0) * (*this);
}

FormExpr FormExpr::operator+(const FormExpr& rhs) const
{
    return FormExpr(std::make_shared<AddNode>(node_, rhs.node_));
}

FormExpr FormExpr::operator-(const FormExpr& rhs) const
{
    return *this + (-rhs);
}

FormExpr FormExpr::operator*(const FormExpr& rhs) const
{
    return FormExpr(std::make_shared<MultiplyNode>(node_, rhs.node_));
}

FormExpr FormExpr::operator*(Real scalar) const
{
    return FormExpr::constant(scalar) * (*this);
}

FormExpr FormExpr::inner(const FormExpr& rhs) const
{
    return FormExpr(std::make_shared<InnerProductNode>(node_, rhs.node_));
}

FormExpr FormExpr::outer(const FormExpr& rhs) const
{
    return *this * rhs;  // Simplified
}

FormExpr FormExpr::dx() const
{
    return FormExpr(std::make_shared<CellIntegralNode>(node_));
}

FormExpr FormExpr::ds(int boundary_marker) const
{
    return FormExpr(std::make_shared<BoundaryIntegralNode>(node_, boundary_marker));
}

FormExpr FormExpr::dS() const
{
    return FormExpr(std::make_shared<BoundaryIntegralNode>(node_, -1));
}

std::string FormExpr::toString() const
{
    return node_ ? node_->toString() : "<empty>";
}

bool FormExpr::hasTest() const
{
    return node_ && node_->hasTest();
}

bool FormExpr::hasTrial() const
{
    return node_ && node_->hasTrial();
}

// ============================================================================
// FormIR Implementation
// ============================================================================

struct FormIR::Impl {
    bool compiled{false};
    RequiredData required_data{RequiredData::Standard};
    bool is_bilinear{false};
    bool is_linear{false};
    bool has_cell_terms{false};
    bool has_boundary_terms{false};
    bool has_face_terms{false};
    std::string ir_dump;
};

FormIR::FormIR() : impl_(std::make_unique<Impl>()) {}
FormIR::~FormIR() = default;
FormIR::FormIR(FormIR&& other) noexcept = default;
FormIR& FormIR::operator=(FormIR&& other) noexcept = default;

bool FormIR::isCompiled() const noexcept { return impl_->compiled; }
RequiredData FormIR::getRequiredData() const noexcept { return impl_->required_data; }
std::string FormIR::dump() const { return impl_->ir_dump; }
bool FormIR::isBilinear() const noexcept { return impl_->is_bilinear; }
bool FormIR::isLinear() const noexcept { return impl_->is_linear; }
bool FormIR::hasCellTerms() const noexcept { return impl_->has_cell_terms; }
bool FormIR::hasBoundaryTerms() const noexcept { return impl_->has_boundary_terms; }
bool FormIR::hasFaceTerms() const noexcept { return impl_->has_face_terms; }

// ============================================================================
// FormCompiler Implementation
// ============================================================================

struct FormCompiler::Impl {
    SymbolicOptions options;
    CompileStats last_stats;
};

FormCompiler::FormCompiler() : impl_(std::make_unique<Impl>()) {}

FormCompiler::FormCompiler(const SymbolicOptions& options)
    : impl_(std::make_unique<Impl>())
{
    impl_->options = options;
}

FormCompiler::~FormCompiler() = default;

FormIR FormCompiler::compile(const FormExpr& form)
{
    auto start = std::chrono::steady_clock::now();

    FormIR ir;

    if (!form.isValid()) {
        return ir;
    }

    // Analyze form type
    ir.impl_->is_bilinear = form.isBilinear();
    ir.impl_->is_linear = form.isLinear();

    // Analyze required data based on form structure
    RequiredData required = RequiredData::Standard | RequiredData::IntegrationWeights;

    // Check for gradient terms
    std::function<void(const FormExprNode*)> analyzeNode = [&](const FormExprNode* node) {
        if (!node) return;

        if (node->type() == FormExprType::Gradient) {
            required = required | RequiredData::PhysicalGradients;
        }
        if (node->type() == FormExprType::CellIntegral) {
            ir.impl_->has_cell_terms = true;
        }
        if (node->type() == FormExprType::BoundaryIntegral) {
            ir.impl_->has_boundary_terms = true;
            required = required | RequiredData::Normals;
        }
        if (node->type() == FormExprType::InteriorFaceIntegral) {
            ir.impl_->has_face_terms = true;
        }

        for (auto* child : node->children()) {
            analyzeNode(child);
        }
    };

    analyzeNode(form.node());
    ir.impl_->required_data = required;

    // Generate IR dump
    std::ostringstream oss;
    oss << "FormIR:\n";
    oss << "  Expression: " << form.toString() << "\n";
    oss << "  Bilinear: " << (ir.impl_->is_bilinear ? "yes" : "no") << "\n";
    oss << "  Linear: " << (ir.impl_->is_linear ? "yes" : "no") << "\n";
    oss << "  Cell terms: " << (ir.impl_->has_cell_terms ? "yes" : "no") << "\n";
    oss << "  Boundary terms: " << (ir.impl_->has_boundary_terms ? "yes" : "no") << "\n";
    ir.impl_->ir_dump = oss.str();

    ir.impl_->compiled = true;

    auto end = std::chrono::steady_clock::now();
    impl_->last_stats.compile_seconds = std::chrono::duration<double>(end - start).count();
    impl_->last_stats.num_cell_terms = ir.impl_->has_cell_terms ? 1 : 0;
    impl_->last_stats.num_boundary_terms = ir.impl_->has_boundary_terms ? 1 : 0;

    return ir;
}

FormIR FormCompiler::compileBilinear(const FormExpr& form)
{
    if (!form.isBilinear()) {
        throw std::invalid_argument("FormCompiler::compileBilinear: form is not bilinear");
    }
    return compile(form);
}

FormIR FormCompiler::compileLinear(const FormExpr& form)
{
    if (!form.isLinear()) {
        throw std::invalid_argument("FormCompiler::compileLinear: form is not linear");
    }
    return compile(form);
}

std::pair<FormIR, FormIR> FormCompiler::compileNonlinear(
    const FormExpr& residual,
    ADMode /*ad_mode*/)
{
    // Simplified: compile residual and return placeholder for Jacobian
    FormIR residual_ir = compile(residual);
    FormIR jacobian_ir;

    // In full implementation, would use AD to derive Jacobian
    jacobian_ir.impl_->compiled = true;
    jacobian_ir.impl_->is_bilinear = true;
    jacobian_ir.impl_->required_data = residual_ir.getRequiredData();
    jacobian_ir.impl_->has_cell_terms = residual_ir.hasCellTerms();

    return {std::move(residual_ir), std::move(jacobian_ir)};
}

void FormCompiler::setOptions(const SymbolicOptions& options)
{
    impl_->options = options;
}

const FormCompiler::CompileStats& FormCompiler::getLastStats() const noexcept
{
    return impl_->last_stats;
}

// ============================================================================
// SymbolicKernel Implementation
// ============================================================================

SymbolicKernel::SymbolicKernel(FormIR form_ir)
    : form_ir_(std::move(form_ir))
{
}

SymbolicKernel::~SymbolicKernel() = default;

SymbolicKernel::SymbolicKernel(SymbolicKernel&& other) noexcept = default;
SymbolicKernel& SymbolicKernel::operator=(SymbolicKernel&& other) noexcept = default;

RequiredData SymbolicKernel::getRequiredData() const noexcept
{
    return form_ir_.getRequiredData();
}

bool SymbolicKernel::hasCell() const noexcept
{
    return form_ir_.hasCellTerms();
}

bool SymbolicKernel::hasBoundaryFace() const noexcept
{
    return form_ir_.hasBoundaryTerms();
}

bool SymbolicKernel::hasInteriorFace() const noexcept
{
    return form_ir_.hasFaceTerms();
}

void SymbolicKernel::computeCell(const AssemblyContext& ctx, KernelOutput& output)
{
    // Simplified implementation for Laplacian-type forms
    // Real implementation would interpret the FormIR

    const LocalIndex num_dofs = ctx.numTestDofs();
    const LocalIndex num_qpts = ctx.numQuadraturePoints();

    output.reserve(num_dofs, num_dofs, form_ir_.isBilinear(), form_ir_.isLinear());

    if (form_ir_.isBilinear()) {
        // Assume grad-grad form
        for (LocalIndex i = 0; i < num_dofs; ++i) {
            for (LocalIndex j = 0; j < num_dofs; ++j) {
                Real sum = 0.0;
                for (LocalIndex q = 0; q < num_qpts; ++q) {
                    auto grad_i = ctx.physicalGradient(i, q);
                    auto grad_j = ctx.physicalGradient(j, q);
                    Real jxw = ctx.integrationWeight(q);

                    Real dot = 0.0;
                    for (int d = 0; d < ctx.dimension(); ++d) {
                        dot += grad_i[static_cast<std::size_t>(d)] *
                               grad_j[static_cast<std::size_t>(d)];
                    }
                    sum += dot * jxw;
                }
                output.local_matrix[static_cast<std::size_t>(i * num_dofs + j)] = sum;
            }
        }
        output.has_matrix = true;
    }

    if (form_ir_.isLinear()) {
        for (LocalIndex i = 0; i < num_dofs; ++i) {
            Real sum = 0.0;
            for (LocalIndex q = 0; q < num_qpts; ++q) {
                Real phi = ctx.basisValue(i, q);
                Real jxw = ctx.integrationWeight(q);
                sum += phi * jxw;  // Simplified: constant source
            }
            output.local_vector[static_cast<std::size_t>(i)] = sum;
        }
        output.has_vector = true;
    }
}

void SymbolicKernel::computeBoundaryFace(
    const AssemblyContext& ctx,
    int /*boundary_marker*/,
    KernelOutput& output)
{
    const LocalIndex num_dofs = ctx.numTestDofs();
    output.reserve(num_dofs, num_dofs, false, false);
    output.has_matrix = false;
    output.has_vector = false;
    // Placeholder - real implementation would handle boundary integrals
}

void SymbolicKernel::computeInteriorFace(
    const AssemblyContext& /*ctx_minus*/,
    const AssemblyContext& /*ctx_plus*/,
    KernelOutput& /*output_minus*/,
    KernelOutput& /*output_plus*/,
    KernelOutput& /*coupling_mp*/,
    KernelOutput& /*coupling_pm*/)
{
    // Placeholder - real implementation would handle DG face terms
}

// ============================================================================
// SymbolicAssembler Implementation
// ============================================================================

struct SymbolicAssembler::Impl {
    SymbolicOptions sym_options;
    AssemblyOptions options;
    FormCompiler compiler;
    std::unique_ptr<StandardAssembler> standard_assembler;

    const dofs::DofMap* dof_map{nullptr};
    const dofs::DofHandler* dof_handler{nullptr};
    const constraints::AffineConstraints* constraints{nullptr};
    const sparsity::SparsityPattern* sparsity{nullptr};
    std::unique_ptr<constraints::ConstraintDistributor> constraint_distributor;

    bool initialized{false};

    // Cache for compiled forms
    std::unordered_map<std::string, std::unique_ptr<SymbolicKernel>> kernel_cache;

    Impl() : standard_assembler(std::make_unique<StandardAssembler>()) {}
};

SymbolicAssembler::SymbolicAssembler()
    : impl_(std::make_unique<Impl>())
{
}

SymbolicAssembler::SymbolicAssembler(const SymbolicOptions& options)
    : impl_(std::make_unique<Impl>())
{
    impl_->sym_options = options;
    impl_->compiler.setOptions(options);
}

SymbolicAssembler::~SymbolicAssembler() = default;

SymbolicAssembler::SymbolicAssembler(SymbolicAssembler&& other) noexcept = default;
SymbolicAssembler& SymbolicAssembler::operator=(SymbolicAssembler&& other) noexcept = default;

// Configuration
void SymbolicAssembler::setDofMap(const dofs::DofMap& dof_map)
{
    impl_->dof_map = &dof_map;
    impl_->standard_assembler->setDofMap(dof_map);
}

void SymbolicAssembler::setDofHandler(const dofs::DofHandler& dof_handler)
{
    impl_->dof_handler = &dof_handler;
    impl_->dof_map = &dof_handler.getDofMap();
    impl_->standard_assembler->setDofHandler(dof_handler);
}

void SymbolicAssembler::setConstraints(const constraints::AffineConstraints* constraints)
{
    impl_->constraints = constraints;
    impl_->standard_assembler->setConstraints(constraints);

    if (constraints && constraints->isClosed()) {
        impl_->constraint_distributor =
            std::make_unique<constraints::ConstraintDistributor>(*constraints);
    } else {
        impl_->constraint_distributor.reset();
    }
}

void SymbolicAssembler::setSparsityPattern(const sparsity::SparsityPattern* sparsity)
{
    impl_->sparsity = sparsity;
    impl_->standard_assembler->setSparsityPattern(sparsity);
}

void SymbolicAssembler::setOptions(const AssemblyOptions& options)
{
    impl_->options = options;
    impl_->standard_assembler->setOptions(options);
}

const AssemblyOptions& SymbolicAssembler::getOptions() const noexcept
{
    return impl_->options;
}

bool SymbolicAssembler::isConfigured() const noexcept
{
    return impl_->dof_map != nullptr;
}

void SymbolicAssembler::setSymbolicOptions(const SymbolicOptions& options)
{
    impl_->sym_options = options;
    impl_->compiler.setOptions(options);
}

const SymbolicOptions& SymbolicAssembler::getSymbolicOptions() const noexcept
{
    return impl_->sym_options;
}

// Lifecycle
void SymbolicAssembler::initialize()
{
    if (!isConfigured()) {
        throw std::runtime_error("SymbolicAssembler::initialize: not configured");
    }
    impl_->standard_assembler->initialize();
    impl_->initialized = true;
}

void SymbolicAssembler::finalize(GlobalSystemView* matrix_view, GlobalSystemView* vector_view)
{
    impl_->standard_assembler->finalize(matrix_view, vector_view);
}

void SymbolicAssembler::reset()
{
    impl_->standard_assembler->reset();
    impl_->kernel_cache.clear();
    impl_->initialized = false;
}

// Standard assembly (delegate to StandardAssembler)
AssemblyResult SymbolicAssembler::assembleMatrix(
    const IMeshAccess& mesh,
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    AssemblyKernel& kernel,
    GlobalSystemView& matrix_view)
{
    return impl_->standard_assembler->assembleMatrix(mesh, test_space, trial_space,
                                                     kernel, matrix_view);
}

AssemblyResult SymbolicAssembler::assembleVector(
    const IMeshAccess& mesh,
    const spaces::FunctionSpace& space,
    AssemblyKernel& kernel,
    GlobalSystemView& vector_view)
{
    return impl_->standard_assembler->assembleVector(mesh, space, kernel, vector_view);
}

AssemblyResult SymbolicAssembler::assembleBoth(
    const IMeshAccess& mesh,
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    AssemblyKernel& kernel,
    GlobalSystemView& matrix_view,
    GlobalSystemView& vector_view)
{
    return impl_->standard_assembler->assembleBoth(mesh, test_space, trial_space,
                                                   kernel, matrix_view, vector_view);
}

AssemblyResult SymbolicAssembler::assembleBoundaryFaces(
    const IMeshAccess& mesh,
    int boundary_marker,
    const spaces::FunctionSpace& space,
    AssemblyKernel& kernel,
    GlobalSystemView* matrix_view,
    GlobalSystemView* vector_view)
{
    return impl_->standard_assembler->assembleBoundaryFaces(mesh, boundary_marker, space,
                                                            kernel, matrix_view, vector_view);
}

AssemblyResult SymbolicAssembler::assembleInteriorFaces(
    const IMeshAccess& mesh,
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    AssemblyKernel& kernel,
    GlobalSystemView& matrix_view,
    GlobalSystemView* vector_view)
{
    return impl_->standard_assembler->assembleInteriorFaces(mesh, test_space, trial_space,
                                                            kernel, matrix_view, vector_view);
}

// Symbolic form assembly
AssemblyResult SymbolicAssembler::assembleForm(
    const FormExpr& form,
    const IMeshAccess& mesh,
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    GlobalSystemView& matrix_view)
{
    // Compile form
    FormIR ir = impl_->compiler.compile(form);

    // Create kernel
    SymbolicKernel kernel(std::move(ir));

    // Assemble
    return assembleMatrix(mesh, test_space, trial_space, kernel, matrix_view);
}

AssemblyResult SymbolicAssembler::assembleLinearForm(
    const FormExpr& form,
    const IMeshAccess& mesh,
    const spaces::FunctionSpace& space,
    GlobalSystemView& vector_view)
{
    FormIR ir = impl_->compiler.compileLinear(form);
    SymbolicKernel kernel(std::move(ir));
    return assembleVector(mesh, space, kernel, vector_view);
}

AssemblyResult SymbolicAssembler::assembleResidualAndJacobian(
    const FormExpr& residual_form,
    const IMeshAccess& mesh,
    const spaces::FunctionSpace& space,
    std::span<const Real> /*solution*/,
    GlobalSystemView& jacobian_view,
    GlobalSystemView& residual_view)
{
    auto [residual_ir, jacobian_ir] = impl_->compiler.compileNonlinear(
        residual_form, impl_->sym_options.ad_mode);

    SymbolicKernel jacobian_kernel(std::move(jacobian_ir));
    SymbolicKernel residual_kernel(std::move(residual_ir));

    // Assemble both
    return assembleBoth(mesh, space, space, jacobian_kernel,
                        jacobian_view, residual_view);
}

std::unique_ptr<SymbolicKernel> SymbolicAssembler::precompile(const FormExpr& form)
{
    FormIR ir = impl_->compiler.compile(form);
    return std::make_unique<SymbolicKernel>(std::move(ir));
}

void SymbolicAssembler::clearCache()
{
    impl_->kernel_cache.clear();
}

// Factory functions
std::unique_ptr<Assembler> createSymbolicAssembler()
{
    return std::make_unique<SymbolicAssembler>();
}

std::unique_ptr<Assembler> createSymbolicAssembler(const SymbolicOptions& options)
{
    return std::make_unique<SymbolicAssembler>(options);
}

} // namespace assembly
} // namespace FE
} // namespace svmp
