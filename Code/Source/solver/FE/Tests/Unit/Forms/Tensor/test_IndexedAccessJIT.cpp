/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Assembly/StandardAssembler.h"
#include "Assembly/TimeIntegrationContext.h"
#include "Core/FEException.h"
#include "Forms/ConstitutiveModel.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/Index.h"
#include "Forms/JIT/InlinableConstitutiveModel.h"
#include "Forms/JIT/JITKernelWrapper.h"
#include "Spaces/HCurlSpace.h"
#include "Spaces/H1Space.h"
#include "Spaces/HDivSpace.h"
#include "Spaces/L2Space.h"
#include "Spaces/ProductSpace.h"
#include "Systems/MaterialStateProvider.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace test {

#ifndef SVMP_FE_ENABLE_LLVM_JIT
#define SVMP_FE_ENABLE_LLVM_JIT 0
#endif

#if SVMP_FE_ENABLE_LLVM_JIT
namespace {

[[nodiscard]] dofs::DofMap makeSingleCellDofMap(GlobalIndex n_dofs)
{
    dofs::DofMap dof_map(1, n_dofs, n_dofs);
    std::vector<GlobalIndex> cell_dofs(static_cast<std::size_t>(n_dofs));
    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        cell_dofs[static_cast<std::size_t>(i)] = i;
    }
    dof_map.setCellDofs(0, cell_dofs);
    dof_map.setNumDofs(n_dofs);
    dof_map.setNumLocalDofs(n_dofs);
    dof_map.finalize();
    return dof_map;
}

void expectDenseNear(const assembly::DenseMatrixView& A,
                     const assembly::DenseMatrixView& B,
                     double tol)
{
    ASSERT_EQ(A.numRows(), B.numRows());
    ASSERT_EQ(A.numCols(), B.numCols());
    for (GlobalIndex i = 0; i < A.numRows(); ++i) {
        for (GlobalIndex j = 0; j < A.numCols(); ++j) {
            SCOPED_TRACE(::testing::Message() << "i=" << i << " j=" << j);
            EXPECT_NEAR(A.getMatrixEntry(i, j), B.getMatrixEntry(i, j), tol);
        }
    }
}

void expectVectorNear(const assembly::DenseVectorView& a,
                      const assembly::DenseVectorView& b,
                      double tol)
{
    ASSERT_EQ(a.numRows(), b.numRows());
    for (GlobalIndex i = 0; i < a.numRows(); ++i) {
        SCOPED_TRACE(::testing::Message() << "i=" << i);
        EXPECT_NEAR(a.getVectorEntry(i), b.getVectorEntry(i), tol);
    }
}

class InlinableStatefulWriteModel final : public ConstitutiveModel, public forms::InlinableConstitutiveModel {
public:
    [[nodiscard]] Value<Real> evaluate(const Value<Real>& /*input*/, int /*dim*/) const override
    {
        throw std::logic_error("InlinableStatefulWriteModel: evaluate() should not be called after inlining");
    }

    [[nodiscard]] Value<Dual> evaluate(const Value<Dual>& /*input*/,
                                       int /*dim*/,
                                       DualWorkspace& /*ws*/) const override
    {
        throw std::logic_error("InlinableStatefulWriteModel: evaluate() should not be called after inlining (dual)");
    }

    [[nodiscard]] StateSpec stateSpec() const noexcept override
    {
        StateSpec s;
        s.bytes_per_qpt = sizeof(Real);
        s.alignment = alignof(Real);
        return s;
    }

    [[nodiscard]] std::vector<params::Spec> parameterSpecs() const override
    {
        return {params::Spec{.key = "k", .type = params::ValueType::Real, .required = true}};
    }

    [[nodiscard]] const forms::InlinableConstitutiveModel* inlinable() const noexcept override { return this; }

    [[nodiscard]] std::uint64_t kindId() const noexcept override
    {
        return forms::InlinableConstitutiveModel::fnv1a64("InlinableStatefulWriteModel");
    }

    [[nodiscard]] MaterialStateAccess stateAccess() const noexcept override { return MaterialStateAccess::ReadWrite; }

    [[nodiscard]] forms::InlinedConstitutiveExpansion inlineExpand(
        std::span<const FormExpr> inputs,
        const forms::InlinableConstitutiveContext& ctx) const override
    {
        FE_THROW_IF(inputs.size() != 1u, InvalidArgumentException,
                    "InlinableStatefulWriteModel: expected exactly 1 input");

        const std::uint32_t state_off = ctx.state_base_offset_bytes;
        const auto state_value = inputs[0] * FormExpr::parameter("k");

        forms::InlinedConstitutiveExpansion out;
        out.state_updates.push_back(forms::MaterialStateUpdateOp{.offset_bytes = state_off, .value = state_value});
        out.outputs.push_back(FormExpr::materialStateWorkRef(state_off));
        return out;
    }
};

} // namespace

TEST(IndexedAccessJIT, CellMatrixContractionMatchesInterpreter)
{
    SingleTetraMeshAccess mesh;
    auto base = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    spaces::ProductSpace space(base, 3);
    auto dof_map = makeSingleCellDofMap(static_cast<GlobalIndex>(space.dofs_per_element()));

    SymbolicOptions sym_opts;
    sym_opts.jit.enable = true;

    FormCompiler compiler(sym_opts);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const Index i("i");
    const Index j("j");
    const auto form = (grad(u)(i, j) * grad(v)(i, j)).dx();

    auto ir_interp = compiler.compileBilinear(form);
    auto ir_jit = compiler.compileBilinear(form);

    auto interp_kernel = std::make_shared<FormKernel>(std::move(ir_interp));

    auto jit_fallback = std::make_shared<FormKernel>(std::move(ir_jit));
    forms::JITOptions jit_opts;
    jit_opts.enable = true;
    jit_opts.optimization_level = 2;
    jit_opts.vectorize = true;
    forms::jit::JITKernelWrapper jit_kernel(jit_fallback, jit_opts);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseMatrixView A_interp(static_cast<GlobalIndex>(space.dofs_per_element()));
    A_interp.zero();
    (void)assembler.assembleMatrix(mesh, space, space, *interp_kernel, A_interp);

    assembly::DenseMatrixView A_jit(static_cast<GlobalIndex>(space.dofs_per_element()));
    A_jit.zero();
    (void)assembler.assembleMatrix(mesh, space, space, jit_kernel, A_jit);

    expectDenseNear(A_jit, A_interp, 1e-12);
}

TEST(IndexedAccessJIT, BoundaryVectorContractionMatchesInterpreter)
{
    SingleTetraOneBoundaryFaceMeshAccess mesh(/*boundary_marker=*/2);
    auto base = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    spaces::ProductSpace space(base, 3);
    auto dof_map = makeSingleCellDofMap(static_cast<GlobalIndex>(space.dofs_per_element()));

    SymbolicOptions sym_opts;
    sym_opts.jit.enable = true;

    FormCompiler compiler(sym_opts);
    const auto v = FormExpr::testFunction(space, "v");

    const auto g = FormExpr::asVector({FormExpr::constant(1.0), FormExpr::constant(2.0), FormExpr::constant(3.0)});
    const Index i("i");
    const auto form = (g(i) * v(i)).ds(2);

    auto ir_interp = compiler.compileLinear(form);
    auto ir_jit = compiler.compileLinear(form);
    auto interp_kernel = std::make_shared<FormKernel>(std::move(ir_interp));

    auto jit_fallback = std::make_shared<FormKernel>(std::move(ir_jit));
    forms::JITOptions jit_opts;
    jit_opts.enable = true;
    jit_opts.optimization_level = 2;
    jit_opts.vectorize = true;
    forms::jit::JITKernelWrapper jit_kernel(jit_fallback, jit_opts);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseVectorView b_interp(static_cast<GlobalIndex>(space.dofs_per_element()));
    b_interp.zero();
    (void)assembler.assembleBoundaryFaces(mesh, /*boundary_marker=*/2, space, *interp_kernel, nullptr, &b_interp);

    assembly::DenseVectorView b_jit(static_cast<GlobalIndex>(space.dofs_per_element()));
    b_jit.zero();
    (void)assembler.assembleBoundaryFaces(mesh, /*boundary_marker=*/2, space, jit_kernel, nullptr, &b_jit);

    expectVectorNear(b_jit, b_interp, 1e-12);
}

TEST(IndexedAccessJIT, InteriorFaceVectorContractionMatchesInterpreter)
{
    TwoTetraSharedFaceMeshAccess mesh;
    auto dof_map = createTwoTetraDG_DofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    SymbolicOptions sym_opts;
    sym_opts.jit.enable = true;

    FormCompiler compiler(sym_opts);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const Index i("i");
    const auto form = (FormExpr::constant(2.5) * jump(grad(u))(i) * jump(grad(v))(i)).dS();

    auto ir_interp = compiler.compileBilinear(form);
    auto ir_jit = compiler.compileBilinear(form);

    auto interp_kernel = std::make_shared<FormKernel>(std::move(ir_interp));

    auto jit_fallback = std::make_shared<FormKernel>(std::move(ir_jit));
    forms::JITOptions jit_opts;
    jit_opts.enable = true;
    jit_opts.optimization_level = 2;
    jit_opts.vectorize = true;
    forms::jit::JITKernelWrapper jit_kernel(jit_fallback, jit_opts);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseMatrixView A_interp(8);
    A_interp.zero();
    (void)assembler.assembleInteriorFaces(mesh, space, space, *interp_kernel, A_interp, nullptr);

    assembly::DenseMatrixView A_jit(8);
    A_jit.zero();
    (void)assembler.assembleInteriorFaces(mesh, space, space, jit_kernel, A_jit, nullptr);

    expectDenseNear(A_jit, A_interp, 1e-12);
}

TEST(IndexedAccessJIT, TimeDerivativeWeightedFormMatchesInterpreterAndScalesByDtCoeff0)
{
    SingleTetraMeshAccess mesh;
    auto base = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    spaces::ProductSpace space(base, 3);
    auto dof_map = makeSingleCellDofMap(static_cast<GlobalIndex>(space.dofs_per_element()));

    SymbolicOptions sym_opts;
    sym_opts.jit.enable = true;

    FormCompiler compiler(sym_opts);
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const Index i("i");
    const auto mass_form = (u(i) * v(i)).dx();
    const auto dt_form = (u.dt(1)(i) * v(i)).dx();

    auto ir_mass = compiler.compileBilinear(mass_form);
    FormKernel mass_kernel(std::move(ir_mass));

    auto ir_interp = compiler.compileBilinear(dt_form);
    auto ir_jit = compiler.compileBilinear(dt_form);

    auto interp_kernel = std::make_shared<FormKernel>(std::move(ir_interp));
    auto jit_fallback = std::make_shared<FormKernel>(std::move(ir_jit));
    forms::JITOptions jit_opts;
    jit_opts.enable = true;
    jit_opts.optimization_level = 2;
    jit_opts.vectorize = true;
    forms::jit::JITKernelWrapper jit_kernel(jit_fallback, jit_opts);

    assembly::TimeIntegrationContext tctx;
    tctx.integrator_name = "unit_test";
    tctx.dt1 = assembly::TimeDerivativeStencil{.order = 1, .a = {2.0}};

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.setTimeIntegrationContext(&tctx);

    assembly::DenseMatrixView M(static_cast<GlobalIndex>(space.dofs_per_element()));
    M.zero();
    (void)assembler.assembleMatrix(mesh, space, space, mass_kernel, M);

    assembly::DenseMatrixView A_interp(static_cast<GlobalIndex>(space.dofs_per_element()));
    A_interp.zero();
    (void)assembler.assembleMatrix(mesh, space, space, *interp_kernel, A_interp);

    assembly::DenseMatrixView A_jit(static_cast<GlobalIndex>(space.dofs_per_element()));
    A_jit.zero();
    (void)assembler.assembleMatrix(mesh, space, space, jit_kernel, A_jit);

    expectDenseNear(A_jit, A_interp, 1e-12);

    for (GlobalIndex r = 0; r < M.numRows(); ++r) {
        for (GlobalIndex c = 0; c < M.numCols(); ++c) {
            EXPECT_NEAR(A_interp.getMatrixEntry(r, c), 2.0 * M.getMatrixEntry(r, c), 1e-12);
        }
    }
}

TEST(IndexedAccessJIT, ParameterRefAndCoupledSlotLoadsMatchInterpreter)
{
    SingleTetraOneBoundaryFaceMeshAccess mesh(/*boundary_marker=*/2);
    auto base = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    spaces::ProductSpace space(base, 3);
    auto dof_map = makeSingleCellDofMap(static_cast<GlobalIndex>(space.dofs_per_element()));

    SymbolicOptions sym_opts;
    sym_opts.jit.enable = true;

    FormCompiler compiler(sym_opts);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto g = FormExpr::asVector({FormExpr::constant(1.0), FormExpr::constant(2.0), FormExpr::constant(3.0)});

    const Index i("i");
    const Index j("j");

    const auto base_form = (grad(u)(i, j) * grad(v)(i, j)).dx();
    const auto param_form = (FormExpr::parameterRef(0) * grad(u)(i, j) * grad(v)(i, j)).dx();

    const auto coupled_weight = FormExpr::boundaryIntegralRef(0) + FormExpr::auxiliaryStateRef(0);
    const auto coupled_form = (coupled_weight * g(i) * v(i)).ds(2);

    const Real k = 2.5;
    const Real coupled_scale = 1.25;
    const std::vector<Real> jit_constants = {k};
    const std::vector<Real> coupled_integrals = {1.0};
    const std::vector<Real> coupled_aux = {coupled_scale - coupled_integrals[0]};

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.setJITConstants(jit_constants);
    assembler.setCoupledValues(coupled_integrals, coupled_aux);

    // ParameterRef: check scaling against a constant-free baseline.
    {
        auto ir_base = compiler.compileBilinear(base_form);
        FormKernel base_kernel(std::move(ir_base));

        auto ir_interp = compiler.compileBilinear(param_form);
        auto ir_jit = compiler.compileBilinear(param_form);

        auto interp_kernel = std::make_shared<FormKernel>(std::move(ir_interp));
        auto jit_fallback = std::make_shared<FormKernel>(std::move(ir_jit));
        forms::JITOptions jit_opts;
        jit_opts.enable = true;
        jit_opts.optimization_level = 2;
        jit_opts.vectorize = true;
        forms::jit::JITKernelWrapper jit_kernel(jit_fallback, jit_opts);

        assembly::DenseMatrixView A_base(static_cast<GlobalIndex>(space.dofs_per_element()));
        A_base.zero();
        (void)assembler.assembleMatrix(mesh, space, space, base_kernel, A_base);

        assembly::DenseMatrixView A_interp(static_cast<GlobalIndex>(space.dofs_per_element()));
        A_interp.zero();
        (void)assembler.assembleMatrix(mesh, space, space, *interp_kernel, A_interp);

        assembly::DenseMatrixView A_jit(static_cast<GlobalIndex>(space.dofs_per_element()));
        A_jit.zero();
        (void)assembler.assembleMatrix(mesh, space, space, jit_kernel, A_jit);

        expectDenseNear(A_jit, A_interp, 1e-12);
        for (GlobalIndex r = 0; r < A_base.numRows(); ++r) {
            for (GlobalIndex c = 0; c < A_base.numCols(); ++c) {
                EXPECT_NEAR(A_interp.getMatrixEntry(r, c), k * A_base.getMatrixEntry(r, c), 1e-12);
            }
        }
    }

    // Coupled slots: boundaryIntegralRef + auxiliaryStateRef.
    {
        auto ir_base = compiler.compileLinear((g(i) * v(i)).ds(2));
        FormKernel base_kernel(std::move(ir_base));

        auto ir_interp = compiler.compileLinear(coupled_form);
        auto ir_jit = compiler.compileLinear(coupled_form);

        auto interp_kernel = std::make_shared<FormKernel>(std::move(ir_interp));
        auto jit_fallback = std::make_shared<FormKernel>(std::move(ir_jit));
        forms::JITOptions jit_opts;
        jit_opts.enable = true;
        jit_opts.optimization_level = 2;
        jit_opts.vectorize = true;
        forms::jit::JITKernelWrapper jit_kernel(jit_fallback, jit_opts);

        assembly::DenseVectorView b_base(static_cast<GlobalIndex>(space.dofs_per_element()));
        b_base.zero();
        (void)assembler.assembleBoundaryFaces(mesh, /*boundary_marker=*/2, space, base_kernel, nullptr, &b_base);

        assembly::DenseVectorView b_interp(static_cast<GlobalIndex>(space.dofs_per_element()));
        b_interp.zero();
        (void)assembler.assembleBoundaryFaces(mesh, /*boundary_marker=*/2, space, *interp_kernel, nullptr, &b_interp);

        assembly::DenseVectorView b_jit(static_cast<GlobalIndex>(space.dofs_per_element()));
        b_jit.zero();
        (void)assembler.assembleBoundaryFaces(mesh, /*boundary_marker=*/2, space, jit_kernel, nullptr, &b_jit);

        expectVectorNear(b_jit, b_interp, 1e-12);
        for (GlobalIndex d = 0; d < b_base.numRows(); ++d) {
            EXPECT_NEAR(b_interp.getVectorEntry(d), coupled_scale * b_base.getVectorEntry(d), 1e-12);
        }
    }
}

TEST(IndexedAccessJIT, InlinedMaterialStateLoadsStoresMatchInterpreter)
{
    SingleTetraMeshAccess mesh;
    auto base = std::make_shared<spaces::L2Space>(ElementType::Tetra4, 0);
    spaces::ProductSpace space(base, 3);
    auto dof_map = makeSingleCellDofMap(static_cast<GlobalIndex>(space.dofs_per_element()));

    auto model = std::make_shared<InlinableStatefulWriteModel>();

    SymbolicOptions sym_opts;
    sym_opts.jit.enable = true;
    FormCompiler compiler(sym_opts);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto K = FormExpr::constitutive(model, FormExpr::constant(2.0));
    const Index i("i");
    const auto form = (K * u(i) * v(i)).dx();

    auto ir_interp = compiler.compileBilinear(form);
    auto ir_jit = compiler.compileBilinear(form);

    auto interp_kernel = std::make_shared<FormKernel>(std::move(ir_interp));
    auto jit_fallback = std::make_shared<FormKernel>(std::move(ir_jit));

    for (auto* k : {interp_kernel.get(), jit_fallback.get()}) {
        k->resolveInlinableConstitutives();
        k->resolveParameterSlots([](std::string_view key) -> std::optional<std::uint32_t> {
            if (key == "k") return 0u;
            return std::nullopt;
        });
    }

    forms::JITOptions jit_opts;
    jit_opts.enable = true;
    jit_opts.optimization_level = 2;
    jit_opts.vectorize = true;
    forms::jit::JITKernelWrapper jit_kernel(jit_fallback, jit_opts);

    systems::MaterialStateProvider provider_interp(/*num_cells=*/mesh.numCells());
    provider_interp.addKernel(*interp_kernel, interp_kernel->materialStateSpec(), /*max_qpts=*/64);
    systems::MaterialStateProvider provider_jit(/*num_cells=*/mesh.numCells());
    provider_jit.addKernel(jit_kernel, jit_kernel.materialStateSpec(), /*max_qpts=*/64);

    const std::vector<Real> constants = {3.0};

    assembly::StandardAssembler assembler_interp;
    assembler_interp.setDofMap(dof_map);
    assembler_interp.setJITConstants(constants);
    assembler_interp.setMaterialStateProvider(&provider_interp);

    assembly::DenseMatrixView A_interp(static_cast<GlobalIndex>(space.dofs_per_element()));
    A_interp.zero();
    (void)assembler_interp.assembleMatrix(mesh, space, space, *interp_kernel, A_interp);

    assembly::StandardAssembler assembler_jit;
    assembler_jit.setDofMap(dof_map);
    assembler_jit.setJITConstants(constants);
    assembler_jit.setMaterialStateProvider(&provider_jit);

    assembly::DenseMatrixView A_jit(static_cast<GlobalIndex>(space.dofs_per_element()));
    A_jit.zero();
    (void)assembler_jit.assembleMatrix(mesh, space, space, jit_kernel, A_jit);

    expectDenseNear(A_jit, A_interp, 1e-12);

    // input=2, k=3 => state=6. For a unit tetra: V=1/6 => entry=6*V=1 on each diagonal block.
    for (GlobalIndex r = 0; r < A_jit.numRows(); ++r) {
        for (GlobalIndex c = 0; c < A_jit.numCols(); ++c) {
            const Real expected = (r == c) ? 1.0 : 0.0;
            EXPECT_NEAR(A_jit.getMatrixEntry(r, c), expected, 1e-12);
        }
    }

    const auto state = provider_jit.getCellState(jit_kernel, /*cell_id=*/0, /*num_qpts=*/1);
    ASSERT_TRUE(static_cast<bool>(state));
    ASSERT_NE(state.data_work, nullptr);
    Real stored{};
    std::memcpy(&stored, state.data_work, sizeof(Real));
    EXPECT_NEAR(stored, 6.0, 1e-12);
}

TEST(IndexedAccessJIT, HCurlCellValueContractionMatchesInterpreter)
{
    SingleTetraMeshAccess mesh;
    spaces::HCurlSpace space(ElementType::Tetra4, /*order=*/0);
    auto dof_map = makeSingleCellDofMap(static_cast<GlobalIndex>(space.dofs_per_element()));

    SymbolicOptions sym_opts;
    sym_opts.jit.enable = true;

    FormCompiler compiler(sym_opts);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const Index i("i");
    const auto form = (u(i) * v(i)).dx();

    auto ir_interp = compiler.compileBilinear(form);
    auto ir_jit = compiler.compileBilinear(form);

    auto interp_kernel = std::make_shared<FormKernel>(std::move(ir_interp));

    auto jit_fallback = std::make_shared<FormKernel>(std::move(ir_jit));
    forms::JITOptions jit_opts;
    jit_opts.enable = true;
    jit_opts.optimization_level = 2;
    jit_opts.vectorize = true;
    forms::jit::JITKernelWrapper jit_kernel(jit_fallback, jit_opts);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseMatrixView A_interp(static_cast<GlobalIndex>(space.dofs_per_element()));
    A_interp.zero();
    (void)assembler.assembleMatrix(mesh, space, space, *interp_kernel, A_interp);

    assembly::DenseMatrixView A_jit(static_cast<GlobalIndex>(space.dofs_per_element()));
    A_jit.zero();
    (void)assembler.assembleMatrix(mesh, space, space, jit_kernel, A_jit);

    expectDenseNear(A_jit, A_interp, 1e-12);
}

TEST(IndexedAccessJIT, HCurlCellCurlContractionMatchesInterpreter)
{
    SingleTetraMeshAccess mesh;
    spaces::HCurlSpace space(ElementType::Tetra4, /*order=*/0);
    auto dof_map = makeSingleCellDofMap(static_cast<GlobalIndex>(space.dofs_per_element()));

    SymbolicOptions sym_opts;
    sym_opts.jit.enable = true;

    FormCompiler compiler(sym_opts);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const Index i("i");
    const auto form = (curl(u)(i) * curl(v)(i)).dx();

    auto ir_interp = compiler.compileBilinear(form);
    auto ir_jit = compiler.compileBilinear(form);

    auto interp_kernel = std::make_shared<FormKernel>(std::move(ir_interp));

    auto jit_fallback = std::make_shared<FormKernel>(std::move(ir_jit));
    forms::JITOptions jit_opts;
    jit_opts.enable = true;
    jit_opts.optimization_level = 2;
    jit_opts.vectorize = true;
    forms::jit::JITKernelWrapper jit_kernel(jit_fallback, jit_opts);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseMatrixView A_interp(static_cast<GlobalIndex>(space.dofs_per_element()));
    A_interp.zero();
    (void)assembler.assembleMatrix(mesh, space, space, *interp_kernel, A_interp);

    assembly::DenseMatrixView A_jit(static_cast<GlobalIndex>(space.dofs_per_element()));
    A_jit.zero();
    (void)assembler.assembleMatrix(mesh, space, space, jit_kernel, A_jit);

    expectDenseNear(A_jit, A_interp, 1e-12);
}

TEST(IndexedAccessJIT, HDivCellValueContractionMatchesInterpreter)
{
    SingleTetraMeshAccess mesh;
    spaces::HDivSpace space(ElementType::Tetra4, /*order=*/0);
    auto dof_map = makeSingleCellDofMap(static_cast<GlobalIndex>(space.dofs_per_element()));

    SymbolicOptions sym_opts;
    sym_opts.jit.enable = true;

    FormCompiler compiler(sym_opts);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const Index i("i");
    const auto form = (u(i) * v(i)).dx();

    auto ir_interp = compiler.compileBilinear(form);
    auto ir_jit = compiler.compileBilinear(form);

    auto interp_kernel = std::make_shared<FormKernel>(std::move(ir_interp));

    auto jit_fallback = std::make_shared<FormKernel>(std::move(ir_jit));
    forms::JITOptions jit_opts;
    jit_opts.enable = true;
    jit_opts.optimization_level = 2;
    jit_opts.vectorize = true;
    forms::jit::JITKernelWrapper jit_kernel(jit_fallback, jit_opts);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseMatrixView A_interp(static_cast<GlobalIndex>(space.dofs_per_element()));
    A_interp.zero();
    (void)assembler.assembleMatrix(mesh, space, space, *interp_kernel, A_interp);

    assembly::DenseMatrixView A_jit(static_cast<GlobalIndex>(space.dofs_per_element()));
    A_jit.zero();
    (void)assembler.assembleMatrix(mesh, space, space, jit_kernel, A_jit);

    expectDenseNear(A_jit, A_interp, 1e-12);
}

TEST(IndexedAccessJIT, HDivCellDivMatchesInterpreter)
{
    SingleTetraMeshAccess mesh;
    spaces::HDivSpace space(ElementType::Tetra4, /*order=*/0);
    auto dof_map = makeSingleCellDofMap(static_cast<GlobalIndex>(space.dofs_per_element()));

    SymbolicOptions sym_opts;
    sym_opts.jit.enable = true;

    FormCompiler compiler(sym_opts);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto form = (div(u) * div(v)).dx();

    auto ir_interp = compiler.compileBilinear(form);
    auto ir_jit = compiler.compileBilinear(form);

    auto interp_kernel = std::make_shared<FormKernel>(std::move(ir_interp));

    auto jit_fallback = std::make_shared<FormKernel>(std::move(ir_jit));
    forms::JITOptions jit_opts;
    jit_opts.enable = true;
    jit_opts.optimization_level = 2;
    jit_opts.vectorize = true;
    forms::jit::JITKernelWrapper jit_kernel(jit_fallback, jit_opts);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseMatrixView A_interp(static_cast<GlobalIndex>(space.dofs_per_element()));
    A_interp.zero();
    (void)assembler.assembleMatrix(mesh, space, space, *interp_kernel, A_interp);

    assembly::DenseMatrixView A_jit(static_cast<GlobalIndex>(space.dofs_per_element()));
    A_jit.zero();
    (void)assembler.assembleMatrix(mesh, space, space, jit_kernel, A_jit);

    expectDenseNear(A_jit, A_interp, 1e-12);
}
#endif

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp
