/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_FormVocabulary.cpp
 * @brief Unit tests for expanded FE/Forms vocabulary operators/terminals
 */

#include <gtest/gtest.h>

#include "Assembly/AssemblyKernel.h"
#include "Assembly/GlobalSystemView.h"
#include "Assembly/StandardAssembler.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/Vocabulary.h"
#include "Spaces/H1Space.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace test {

namespace {

Real singleTetraVolume()
{
    return 1.0 / 6.0;
}

Real singleTetraP1BasisIntegral()
{
    return singleTetraVolume() / 4.0;
}

class SingleTetraDomainMeshAccess final : public assembly::IMeshAccess {
public:
    explicit SingleTetraDomainMeshAccess(int domain_id)
        : domain_id_(domain_id)
    {
        nodes_ = {
            {0.0, 0.0, 0.0},  // 0
            {1.0, 0.0, 0.0},  // 1
            {0.0, 1.0, 0.0},  // 2
            {0.0, 0.0, 1.0}   // 3
        };
        cell_ = {0, 1, 2, 3};
    }

    [[nodiscard]] GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex /*cell_id*/) const override { return true; }

    [[nodiscard]] ElementType getCellType(GlobalIndex /*cell_id*/) const override { return ElementType::Tetra4; }

    [[nodiscard]] int getCellDomainId(GlobalIndex /*cell_id*/) const override { return domain_id_; }

    void getCellNodes(GlobalIndex /*cell_id*/, std::vector<GlobalIndex>& nodes) const override
    {
        nodes.assign(cell_.begin(), cell_.end());
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override
    {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(GlobalIndex /*cell_id*/, std::vector<std::array<Real, 3>>& coords) const override
    {
        coords.resize(cell_.size());
        for (std::size_t i = 0; i < cell_.size(); ++i) {
            coords[i] = nodes_.at(static_cast<std::size_t>(cell_[i]));
        }
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex /*face_id*/, GlobalIndex /*cell_id*/) const override
    {
        return 0;
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex /*face_id*/) const override { return -1; }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex /*face_id*/) const override
    {
        return {0, 0};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override { callback(0); }
    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override { callback(0); }

    void forEachBoundaryFace(int /*marker*/,
                             std::function<void(GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

private:
    int domain_id_{0};
    std::vector<std::array<Real, 3>> nodes_{};
    std::array<GlobalIndex, 4> cell_{};
};

assembly::DenseVectorView assembleCellLinear(const FormExpr& scalar_expr,
                                             dofs::DofMap& dof_map,
                                             const assembly::IMeshAccess& mesh,
                                             const spaces::FunctionSpace& space)
{
    FormCompiler compiler;
    const auto v = FormExpr::testFunction(space, "v");
    const auto form = (scalar_expr * v).dx();
    auto ir = compiler.compileLinear(form);
    FormKernel kernel(std::move(ir));

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseVectorView vec(static_cast<GlobalIndex>(dof_map.getNumDofs()));
    vec.zero();
    (void)assembler.assembleVector(mesh, space, kernel, vec);
    return vec;
}

} // namespace

TEST(FormVocabularyTest, ScalarOpsIntegrateConstantsOverCell)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    struct Case {
        const char* name;
        FormExpr expr;
        Real expected_value;
    };

    const auto beta = FormExpr::coefficient("beta",
                                            [](Real, Real, Real) { return std::array<Real, 3>{1.0, 2.0, 3.0}; });
    const auto K = FormExpr::coefficient(
        "K",
        [](Real, Real, Real) {
            return std::array<std::array<Real, 3>, 3>{
                std::array<Real, 3>{2.0, 0.0, 0.0},
                std::array<Real, 3>{0.0, 3.0, 0.0},
                std::array<Real, 3>{0.0, 0.0, 4.0},
            };
        });

    const std::vector<Case> cases = {
        {"divide", FormExpr::constant(6.0) / FormExpr::constant(2.0), 3.0},
        {"pow", pow(FormExpr::constant(2.0), FormExpr::constant(3.0)), 8.0},
        {"sqrt", sqrt(FormExpr::constant(4.0)), 2.0},
        {"exp", exp(FormExpr::constant(1.0)), std::exp(1.0)},
        {"log", log(FormExpr::constant(2.5)), std::log(2.5)},
        {"abs", abs(FormExpr::constant(-2.0)), 2.0},
        {"sign", sign(FormExpr::constant(-2.0)), -1.0},
        {"min", min(FormExpr::constant(2.0), FormExpr::constant(3.0)), 2.0},
        {"max", max(FormExpr::constant(2.0), FormExpr::constant(3.0)), 3.0},
        {"gt", gt(FormExpr::constant(2.0), FormExpr::constant(1.0)), 1.0},
        {"le", le(FormExpr::constant(2.0), FormExpr::constant(1.0)), 0.0},
        {"conditional", conditional(gt(FormExpr::constant(2.0), FormExpr::constant(1.0)),
                                    FormExpr::constant(10.0),
                                    FormExpr::constant(20.0)),
         10.0},
        {"heaviside-", heaviside(FormExpr::constant(-0.1)), 0.0},
        {"heaviside+", heaviside(FormExpr::constant(0.1)), 1.0},
        {"indicator", indicator(gt(FormExpr::constant(2.0), FormExpr::constant(1.0))), 1.0},
        {"clamp", clamp(FormExpr::constant(5.0), FormExpr::constant(0.0), FormExpr::constant(3.0)), 3.0},

        // Geometry/Jacobian-derived matrix ops on the affine unit tetra: J == I
        {"trace(J)", trace(J()), 3.0},
        {"det(J)", det(J()), 1.0},
        {"norm(J)", norm(J()), std::sqrt(3.0)},
        {"norm(dev(J))", norm(dev(J())), 0.0},
        {"component(inv(J),0,0)", component(inv(J()), 0, 0), 1.0},
        {"component(cofactor(J),0,0)", component(cofactor(J()), 0, 0), 1.0},

        // Vector ops
        {"norm(beta)", norm(beta),
         std::sqrt(14.0)},
        {"inner(normalize(beta),normalize(beta))",
         inner(normalize(beta), normalize(beta)),
         1.0},
        {"component(cross(beta,ez),0)",
         component(cross(beta,
                         FormExpr::coefficient("ez",
                                               [](Real, Real, Real) { return std::array<Real, 3>{0.0, 0.0, 1.0}; })),
                   0),
         2.0},

        // Tensor contraction / matrix-vector multiply (UFL-like)
        {"component(K*beta,0)", component(K * beta, 0), 2.0},
        {"component(beta*K,0)", component(beta * K, 0), 2.0},
        {"trace(K*I)", trace(K * FormExpr::identity(3)), 9.0},
    };

    const Real expected_integral_scale = singleTetraP1BasisIntegral();

    for (const auto& c : cases) {
        SCOPED_TRACE(::testing::Message() << c.name);
        auto vec = assembleCellLinear(c.expr, dof_map, mesh, space);
        const Real expected_entry = c.expected_value * expected_integral_scale;
        for (GlobalIndex i = 0; i < 4; ++i) {
            EXPECT_NEAR(vec.getVectorEntry(i), expected_entry, 5e-12);
        }
    }
}

TEST(FormVocabularyTest, EntityMeasureTerminalsWorkOnCellAndFace)
{
    {
        SingleTetraMeshAccess mesh;
        auto dof_map = createSingleTetraDofMap();
        spaces::H1Space space(ElementType::Tetra4, 1);

        // Cell volume and diameter for the unit tetra.
        const Real V = singleTetraVolume();
        const Real h_expected = std::sqrt(2.0);

        {
            auto vec = assembleCellLinear(vol(), dof_map, mesh, space);
            const Real expected_entry = V * (V / 4.0);
            for (GlobalIndex i = 0; i < 4; ++i) {
                EXPECT_NEAR(vec.getVectorEntry(i), expected_entry, 5e-12);
            }
        }
        {
            auto vec = assembleCellLinear(h(), dof_map, mesh, space);
            const Real expected_entry = h_expected * (V / 4.0);
            for (GlobalIndex i = 0; i < 4; ++i) {
                EXPECT_NEAR(vec.getVectorEntry(i), expected_entry, 5e-12);
            }
        }
    }

    {
        SingleTetraOneBoundaryFaceMeshAccess mesh(/*boundary_marker=*/2);
        auto dof_map = createSingleTetraDofMap();
        spaces::H1Space space(ElementType::Tetra4, 1);

        FormCompiler compiler;
        const auto v = FormExpr::testFunction(space, "v");
        const auto form = (area() * v).ds(2);
        auto ir = compiler.compileLinear(form);
        FormKernel kernel(std::move(ir));

        assembly::StandardAssembler assembler;
        assembler.setDofMap(dof_map);

        assembly::DenseVectorView vec(4);
        vec.zero();
        (void)assembler.assembleBoundaryFaces(mesh, 2, space, kernel, nullptr, &vec);

        const Real A = 0.5;                 // area(face {0,1,2})
        const Real expected = A * (A / 3);  // (area terminal) * ∫phi ds = A*(A/3)

        EXPECT_NEAR(vec.getVectorEntry(0), expected, 5e-12);
        EXPECT_NEAR(vec.getVectorEntry(1), expected, 5e-12);
        EXPECT_NEAR(vec.getVectorEntry(2), expected, 5e-12);
        EXPECT_NEAR(vec.getVectorEntry(3), 0.0, 5e-12);
    }
}

TEST(FormVocabularyTest, CellDomainIdAndRegionIndicatorWork)
{
    constexpr int domain_id_value = 7;
    SingleTetraDomainMeshAccess mesh(domain_id_value);
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    const Real expected_integral_scale = singleTetraP1BasisIntegral();

    {
        auto vec = assembleCellLinear(domainId(), dof_map, mesh, space);
        const Real expected_entry = static_cast<Real>(domain_id_value) * expected_integral_scale;
        for (GlobalIndex i = 0; i < 4; ++i) {
            EXPECT_NEAR(vec.getVectorEntry(i), expected_entry, 5e-12);
        }
    }

    {
        auto vec = assembleCellLinear(regionIndicator(domain_id_value), dof_map, mesh, space);
        const Real expected_entry = 1.0 * expected_integral_scale;
        for (GlobalIndex i = 0; i < 4; ++i) {
            EXPECT_NEAR(vec.getVectorEntry(i), expected_entry, 5e-12);
        }
    }

    {
        auto vec = assembleCellLinear(regionIndicator(domain_id_value + 1), dof_map, mesh, space);
        for (GlobalIndex i = 0; i < 4; ++i) {
            EXPECT_NEAR(vec.getVectorEntry(i), 0.0, 5e-12);
        }
    }
}

TEST(FormVocabularyTest, AsVectorAndAsTensorConstructorsWork)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    const Real expected_integral_scale = singleTetraP1BasisIntegral();

    {
        const auto a = as_vector({FormExpr::constant(1.0),
                                  FormExpr::constant(2.0),
                                  FormExpr::constant(3.0)});
        auto vec = assembleCellLinear(component(a, 0), dof_map, mesh, space);
        const Real expected_entry = 1.0 * expected_integral_scale;
        for (GlobalIndex i = 0; i < 4; ++i) {
            EXPECT_NEAR(vec.getVectorEntry(i), expected_entry, 5e-12);
        }
    }

    {
        const auto A = as_tensor({{FormExpr::constant(1.0), FormExpr::constant(2.0)},
                                  {FormExpr::constant(3.0), FormExpr::constant(4.0)}});

        auto vec01 = assembleCellLinear(component(A, 0, 1), dof_map, mesh, space);
        const Real expected01 = 2.0 * expected_integral_scale;
        for (GlobalIndex i = 0; i < 4; ++i) {
            EXPECT_NEAR(vec01.getVectorEntry(i), expected01, 5e-12);
        }

        auto vec_tr = assembleCellLinear(trace(A), dof_map, mesh, space);
        const Real expected_tr = 5.0 * expected_integral_scale;
        for (GlobalIndex i = 0; i < 4; ++i) {
            EXPECT_NEAR(vec_tr.getVectorEntry(i), expected_tr, 5e-12);
        }
	    }

	    {
	        const auto b = as_vector({FormExpr::constant(1.0),
	                                  FormExpr::constant(2.0),
	                                  FormExpr::constant(3.0),
	                                  FormExpr::constant(4.0)});
	        auto vec3 = assembleCellLinear(component(b, 3), dof_map, mesh, space);
	        const Real expected3 = 4.0 * expected_integral_scale;
	        for (GlobalIndex i = 0; i < 4; ++i) {
	            EXPECT_NEAR(vec3.getVectorEntry(i), expected3, 5e-12);
	        }
	    }

	    EXPECT_THROW(as_tensor({{FormExpr::constant(1.0)},
	                            {FormExpr::constant(2.0), FormExpr::constant(3.0)}}),
	                 std::invalid_argument);

    EXPECT_THROW(as_tensor({{}}), std::invalid_argument);
}

TEST(FormVocabularyTest, RequiredDataInferenceIncludesGeometryAndMeasures)
{
    FormCompiler compiler;

    spaces::H1Space space(ElementType::Tetra4, 1);
    const auto v = FormExpr::testFunction(space, "v");
    const auto u = FormExpr::trialFunction(space, "u");

    const auto ir_x = compiler.compileLinear((component(x(), 0) * v).dx());
    EXPECT_TRUE(assembly::hasFlag(ir_x.requiredData(), assembly::RequiredData::PhysicalPoints));

    const auto ir_X = compiler.compileLinear((component(X(), 0) * v).dx());
    EXPECT_TRUE(assembly::hasFlag(ir_X.requiredData(), assembly::RequiredData::QuadraturePoints));

    const auto ir_J = compiler.compileLinear((trace(J()) * v).dx());
    EXPECT_TRUE(assembly::hasFlag(ir_J.requiredData(), assembly::RequiredData::Jacobians));

    const auto ir_detJ = compiler.compileLinear((detJ() * v).dx());
    EXPECT_TRUE(assembly::hasFlag(ir_detJ.requiredData(), assembly::RequiredData::JacobianDets));

    const auto ir_Jinv = compiler.compileLinear((component(Jinv(), 0, 0) * v).dx());
    EXPECT_TRUE(assembly::hasFlag(ir_Jinv.requiredData(), assembly::RequiredData::InverseJacobians));

    const auto ir_h = compiler.compileLinear((h() * v).dx());
    EXPECT_TRUE(assembly::hasFlag(ir_h.requiredData(), assembly::RequiredData::EntityMeasures));

    const auto ir_dg = compiler.compileBilinear((u.plus() * v.minus()).dS());
    EXPECT_TRUE(assembly::hasFlag(ir_dg.requiredData(), assembly::RequiredData::NeighborData));
}

TEST(FormVocabularyTest, DGRestrictionsAssembleCrossBlockMass)
{
    TwoTetraSharedFaceMeshAccess mesh;
    auto dof_map = createTwoTetraDG_DofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    // Coupling term: ∫_F v(-) * u(+) dS
    const auto form = (v.minus() * u.plus()).dS();
    auto ir = compiler.compileBilinear(form);
    FormKernel kernel(std::move(ir));

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseMatrixView mat(8);
    mat.zero();

    const auto result = assembler.assembleInteriorFaces(mesh, space, space, kernel, mat, nullptr);
    EXPECT_EQ(result.interior_faces_assembled, 1);

    const Real area_face = std::sqrt(3.0) / 2.0;
    const Real mdiag = area_face / 6.0;
    const Real moff = area_face / 12.0;

    auto faceIndexMinus = [](GlobalIndex dof) -> int {
        if (dof == 1) return 0;
        if (dof == 2) return 1;
        if (dof == 3) return 2;
        return -1;
    };
    auto faceIndexPlus = [](GlobalIndex dof) -> int {
        if (dof == 4) return 0;
        if (dof == 5) return 1;
        if (dof == 6) return 2;
        return -1;
    };

    for (GlobalIndex i = 0; i < 8; ++i) {
        for (GlobalIndex j = 0; j < 8; ++j) {
            const int fi = faceIndexMinus(i);
            const int fj = faceIndexPlus(j);
            const Real expected = (fi >= 0 && fj >= 0) ? ((fi == fj) ? mdiag : moff) : 0.0;
            SCOPED_TRACE(::testing::Message() << "i=" << i << ", j=" << j);
            EXPECT_NEAR(mat.getMatrixEntry(i, j), expected, 5e-11);
        }
    }
}

TEST(FormVocabularyTest, DGRestrictionsHandlePermutedPlusFaceOrdering)
{
    TwoTetraSharedFacePermutedPlusMeshAccess mesh;
    auto dof_map = createTwoTetraDG_DofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    // Coupling term: ∫_F v(-) * u(+) dS
    const auto form = (v.minus() * u.plus()).dS();
    auto ir = compiler.compileBilinear(form);
    FormKernel kernel(std::move(ir));

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseMatrixView mat(8);
    mat.zero();

    const auto result = assembler.assembleInteriorFaces(mesh, space, space, kernel, mat, nullptr);
    EXPECT_EQ(result.interior_faces_assembled, 1);

    const Real area_face = std::sqrt(3.0) / 2.0;
    const Real mdiag = area_face / 6.0;
    const Real moff = area_face / 12.0;

    // Minus face ordering (cell 0 face 2): global nodes {1,2,3} in that order.
    // Plus face ordering  (cell 1 face 0): global nodes {2,3,1} in that order.
    const std::array<int, 3> perm_plus_to_minus = {1, 2, 0};

    auto minusFaceIndex = [](GlobalIndex dof) -> int {
        if (dof == 1) return 0;
        if (dof == 2) return 1;
        if (dof == 3) return 2;
        return -1;
    };
    auto plusFaceIndex = [](GlobalIndex dof) -> int {
        if (dof == 4) return 0;
        if (dof == 5) return 1;
        if (dof == 6) return 2;
        return -1;
    };

    for (GlobalIndex i = 0; i < 8; ++i) {
        for (GlobalIndex j = 0; j < 8; ++j) {
            const int fi = minusFaceIndex(i);
            const int fj = plusFaceIndex(j);
            Real expected = 0.0;
            if (fi >= 0 && fj >= 0) {
                const bool match = (fi == perm_plus_to_minus[static_cast<std::size_t>(fj)]);
                expected = match ? mdiag : moff;
            }
            SCOPED_TRACE(::testing::Message() << "i=" << i << ", j=" << j);
            EXPECT_NEAR(mat.getMatrixEntry(i, j), expected, 5e-11);
        }
    }
}

TEST(NonlinearFormKernelTest, JacobianMatchesFiniteDifferencesForNonlinearScalarOps)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto residual = ((sqrt(u * u + FormExpr::constant(1.0)) +
                            exp(u) +
                            log(u * u + FormExpr::constant(1.0)) +
                            (u / FormExpr::constant(2.0)) +
                            pow(u, FormExpr::constant(3.0))) *
                           v)
                              .dx();

    auto ir = compiler.compileResidual(residual);
    NonlinearFormKernel kernel(std::move(ir), ADMode::Forward);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    std::vector<Real> U = {0.1, -0.2, 0.3, -0.1};
    assembler.setCurrentSolution(U);

    assembly::DenseMatrixView J(4);
    assembly::DenseVectorView R(4);
    J.zero();
    R.zero();
    (void)assembler.assembleBoth(mesh, space, space, kernel, J, R);

    std::array<Real, 4> R0{};
    for (GlobalIndex i = 0; i < 4; ++i) {
        R0[static_cast<std::size_t>(i)] = R.getVectorEntry(i);
    }

    const Real eps = 1e-7;
    for (GlobalIndex j = 0; j < 4; ++j) {
        auto U_plus = U;
        U_plus[static_cast<std::size_t>(j)] += eps;
        assembler.setCurrentSolution(U_plus);

        assembly::DenseVectorView Rp(4);
        Rp.zero();
        (void)assembler.assembleVector(mesh, space, kernel, Rp);

        for (GlobalIndex i = 0; i < 4; ++i) {
            const Real fd = (Rp.getVectorEntry(i) - R0[static_cast<std::size_t>(i)]) / eps;
            EXPECT_NEAR(J.getMatrixEntry(i, j), fd, 5e-6);
        }
    }
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp
