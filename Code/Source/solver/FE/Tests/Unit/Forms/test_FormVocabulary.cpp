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
#include "Assembly/TimeIntegrationContext.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/JIT/JITKernelWrapper.h"
#include "Forms/Vocabulary.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"
#include "Tests/Unit/Forms/JITTestHelpers.h"

#include <array>
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

dofs::DofMap createSingleTetraDenseDofMap(LocalIndex n_dofs)
{
    dofs::DofMap dof_map(1, n_dofs, n_dofs);
    std::vector<GlobalIndex> cell_dofs(static_cast<std::size_t>(n_dofs));
    for (LocalIndex i = 0; i < n_dofs; ++i) {
        cell_dofs[static_cast<std::size_t>(i)] = i;
    }
    dof_map.setCellDofs(0, cell_dofs);
    dof_map.setNumDofs(n_dofs);
    dof_map.setNumLocalDofs(n_dofs);
    dof_map.finalize();
    return dof_map;
}

dofs::DofMap createTwoTetraDGDenseDofMap(LocalIndex n_dofs_per_cell)
{
    dofs::DofMap dof_map(2, 2 * n_dofs_per_cell, n_dofs_per_cell);
    for (GlobalIndex cell = 0; cell < 2; ++cell) {
        std::vector<GlobalIndex> cell_dofs(static_cast<std::size_t>(n_dofs_per_cell));
        for (LocalIndex i = 0; i < n_dofs_per_cell; ++i) {
            cell_dofs[static_cast<std::size_t>(i)] =
                static_cast<GlobalIndex>(cell * n_dofs_per_cell + i);
        }
        dof_map.setCellDofs(cell, cell_dofs);
    }
    dof_map.setNumDofs(static_cast<GlobalIndex>(2 * n_dofs_per_cell));
    dof_map.setNumLocalDofs(static_cast<GlobalIndex>(2 * n_dofs_per_cell));
    dof_map.finalize();
    return dof_map;
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
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 1; }
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
                             std::function<void(GlobalIndex, GlobalIndex)> callback) const override
    {
        callback(0, 0);
    }

    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

private:
    int domain_id_{0};
    std::vector<std::array<Real, 3>> nodes_{};
    std::array<GlobalIndex, 4> cell_{};
};

class MovingSingleTetraMeshAccess final : public assembly::IMeshAccess {
public:
    MovingSingleTetraMeshAccess()
    {
        reference_nodes_ = {
            {0.0, 0.0, 0.0},
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0}
        };
        current_nodes_.resize(reference_nodes_.size());
        for (std::size_t i = 0; i < reference_nodes_.size(); ++i) {
            current_nodes_[i] = {
                reference_nodes_[i][0] + translation_[0],
                reference_nodes_[i][1] + translation_[1],
                reference_nodes_[i][2] + translation_[2],
            };
        }
        cell_ = {0, 1, 2, 3};
    }

    [[nodiscard]] GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }
    [[nodiscard]] bool isOwnedCell(GlobalIndex /*cell_id*/) const override { return true; }
    [[nodiscard]] ElementType getCellType(GlobalIndex /*cell_id*/) const override { return ElementType::Tetra4; }

    void getCellNodes(GlobalIndex /*cell_id*/, std::vector<GlobalIndex>& nodes) const override
    {
        nodes.assign(cell_.begin(), cell_.end());
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override
    {
        return current_nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(GlobalIndex /*cell_id*/,
                            std::vector<std::array<Real, 3>>& coords) const override
    {
        coords = current_nodes_;
    }

    [[nodiscard]] bool supportsCoordinateFrame(assembly::CoordinateFrame frame) const override
    {
        return frame == assembly::CoordinateFrame::Active ||
               frame == assembly::CoordinateFrame::Reference ||
               frame == assembly::CoordinateFrame::Current;
    }

    void getCellCoordinates(GlobalIndex /*cell_id*/,
                            assembly::CoordinateFrame frame,
                            std::vector<std::array<Real, 3>>& coords) const override
    {
        switch (frame) {
            case assembly::CoordinateFrame::Active:
            case assembly::CoordinateFrame::Current:
                coords = current_nodes_;
                return;
            case assembly::CoordinateFrame::Reference:
                coords = reference_nodes_;
                return;
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

    [[nodiscard]] const std::array<Real, 3>& translation() const noexcept { return translation_; }

    void perturbCurrentNodeComponent(GlobalIndex node_id, int component, Real delta)
    {
        current_nodes_.at(static_cast<std::size_t>(node_id)).at(static_cast<std::size_t>(component)) += delta;
    }

private:
    std::array<Real, 3> translation_{0.25, 0.5, -0.75};
    std::vector<std::array<Real, 3>> reference_nodes_{};
    std::vector<std::array<Real, 3>> current_nodes_{};
    std::array<GlobalIndex, 4> cell_{};
};

class FrameChangedSingleTetraMeshAccess final : public assembly::IMeshAccess {
public:
    explicit FrameChangedSingleTetraMeshAccess(std::array<Real, 3> scale,
                                               int boundary_marker = -1)
        : scale_(scale),
          boundary_marker_(boundary_marker)
    {
        reference_nodes_ = {
            {0.0, 0.0, 0.0},
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0}
        };
        current_nodes_ = {
            {0.0, 0.0, 0.0},
            {scale_[0], 0.0, 0.0},
            {0.0, scale_[1], 0.0},
            {0.0, 0.0, scale_[2]}
        };
        cell_ = {0, 1, 2, 3};
    }

    [[nodiscard]] GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override
    {
        return boundary_marker_ >= 0 ? 1 : 0;
    }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }
    [[nodiscard]] bool isOwnedCell(GlobalIndex /*cell_id*/) const override { return true; }
    [[nodiscard]] ElementType getCellType(GlobalIndex /*cell_id*/) const override { return ElementType::Tetra4; }

    void getCellNodes(GlobalIndex /*cell_id*/, std::vector<GlobalIndex>& nodes) const override
    {
        nodes.assign(cell_.begin(), cell_.end());
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override
    {
        return current_nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(GlobalIndex /*cell_id*/,
                            std::vector<std::array<Real, 3>>& coords) const override
    {
        coords = current_nodes_;
    }

    [[nodiscard]] bool supportsCoordinateFrame(assembly::CoordinateFrame frame) const override
    {
        return frame == assembly::CoordinateFrame::Active ||
               frame == assembly::CoordinateFrame::Reference ||
               frame == assembly::CoordinateFrame::Current;
    }

    void getCellCoordinates(GlobalIndex /*cell_id*/,
                            assembly::CoordinateFrame frame,
                            std::vector<std::array<Real, 3>>& coords) const override
    {
        switch (frame) {
            case assembly::CoordinateFrame::Active:
            case assembly::CoordinateFrame::Current:
                coords = current_nodes_;
                return;
            case assembly::CoordinateFrame::Reference:
                coords = reference_nodes_;
                return;
        }
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex /*face_id*/, GlobalIndex /*cell_id*/) const override
    {
        return 0;
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex /*face_id*/) const override { return boundary_marker_; }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex /*face_id*/) const override
    {
        return {0, 0};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override { callback(0); }
    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override { callback(0); }

    void forEachBoundaryFace(int marker,
                             std::function<void(GlobalIndex, GlobalIndex)> callback) const override
    {
        if (boundary_marker_ >= 0 && (marker < 0 || marker == boundary_marker_)) {
            callback(0, 0);
        }
    }

    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

private:
    std::array<Real, 3> scale_{};
    int boundary_marker_{-1};
    std::vector<std::array<Real, 3>> reference_nodes_{};
    std::vector<std::array<Real, 3>> current_nodes_{};
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

assembly::DenseVectorView assembleMovingCellLinearWithKernel(const FormExpr& scalar_expr,
                                                             dofs::DofMap& dof_map,
                                                             const assembly::IMeshAccess& mesh,
                                                             const spaces::FunctionSpace& space,
                                                             assembly::AssemblyKernel& kernel)
{
    (void)scalar_expr;

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseVectorView vec(static_cast<GlobalIndex>(dof_map.getNumDofs()));
    vec.zero();
    (void)assembler.assembleVector(mesh, space, kernel, vec);
    return vec;
}

assembly::DenseVectorView assembleMovingCellLinearWithMeshFields(dofs::DofMap& scalar_dof_map,
                                                                 const assembly::IMeshAccess& mesh,
                                                                 const spaces::FunctionSpace& scalar_space,
                                                                 assembly::AssemblyKernel& kernel,
                                                                 const spaces::FunctionSpace& vector_space,
                                                                 const dofs::DofMap& vector_dof_map,
                                                                 const std::vector<Real>& current_solution)
{
    constexpr FieldId kDisplacement = 701;
    constexpr FieldId kVelocity = 702;
    constexpr FieldId kAcceleration = 703;

    std::array<assembly::FieldSolutionAccess, 3> field_access = {{
        assembly::FieldSolutionAccess{
            .field = kDisplacement,
            .space = &vector_space,
            .dof_map = &vector_dof_map,
            .dof_offset = 0,
        },
        assembly::FieldSolutionAccess{
            .field = kVelocity,
            .space = &vector_space,
            .dof_map = &vector_dof_map,
            .dof_offset = vector_dof_map.getNumDofs(),
        },
        assembly::FieldSolutionAccess{
            .field = kAcceleration,
            .space = &vector_space,
            .dof_map = &vector_dof_map,
            .dof_offset = 2 * vector_dof_map.getNumDofs(),
        },
    }};

    assembly::StandardAssembler assembler;
    assembler.setDofMap(scalar_dof_map);
    assembler.setFieldSolutionAccess(field_access);
    assembler.setMeshMotionFieldAccess(assembly::MeshMotionFieldAccess{
        .mesh_displacement = kDisplacement,
        .mesh_velocity = kVelocity,
        .mesh_acceleration = kAcceleration,
    });
    assembler.setCurrentSolution(current_solution);

    assembly::DenseVectorView vec(static_cast<GlobalIndex>(scalar_dof_map.getNumDofs()));
    vec.zero();
    (void)assembler.assembleVector(mesh, scalar_space, kernel, vec);
    return vec;
}

assembly::DenseVectorView assembleMovingCellLinearWithTimeLevelMeshFields(
    dofs::DofMap& scalar_dof_map,
    const assembly::IMeshAccess& mesh,
    const spaces::FunctionSpace& scalar_space,
    assembly::AssemblyKernel& kernel,
    const spaces::FunctionSpace& vector_space,
    const dofs::DofMap& vector_dof_map,
    const std::vector<Real>& current_solution)
{
    constexpr FieldId kDisplacement = 711;
    constexpr FieldId kVelocity = 712;
    constexpr FieldId kAcceleration = 713;
    constexpr FieldId kPreviousCoordinates = 714;
    constexpr FieldId kPreviousVelocity = 715;
    constexpr FieldId kPredictedVelocity = 716;

    const auto n = vector_dof_map.getNumDofs();
    std::array<assembly::FieldSolutionAccess, 6> field_access = {{
        assembly::FieldSolutionAccess{
            .field = kDisplacement,
            .space = &vector_space,
            .dof_map = &vector_dof_map,
            .dof_offset = 0,
        },
        assembly::FieldSolutionAccess{
            .field = kVelocity,
            .space = &vector_space,
            .dof_map = &vector_dof_map,
            .dof_offset = n,
        },
        assembly::FieldSolutionAccess{
            .field = kAcceleration,
            .space = &vector_space,
            .dof_map = &vector_dof_map,
            .dof_offset = 2 * n,
        },
        assembly::FieldSolutionAccess{
            .field = kPreviousCoordinates,
            .space = &vector_space,
            .dof_map = &vector_dof_map,
            .dof_offset = 3 * n,
        },
        assembly::FieldSolutionAccess{
            .field = kPreviousVelocity,
            .space = &vector_space,
            .dof_map = &vector_dof_map,
            .dof_offset = 4 * n,
        },
        assembly::FieldSolutionAccess{
            .field = kPredictedVelocity,
            .space = &vector_space,
            .dof_map = &vector_dof_map,
            .dof_offset = 5 * n,
        },
    }};

    assembly::StandardAssembler assembler;
    assembler.setDofMap(scalar_dof_map);
    assembler.setFieldSolutionAccess(field_access);
    assembler.setMeshMotionFieldAccess(assembly::MeshMotionFieldAccess{
        .mesh_displacement = kDisplacement,
        .mesh_velocity = kVelocity,
        .mesh_acceleration = kAcceleration,
        .previous_coordinates = kPreviousCoordinates,
        .previous_mesh_velocity = kPreviousVelocity,
        .predicted_mesh_velocity = kPredictedVelocity,
    });
    assembler.setCurrentSolution(current_solution);

    assembly::DenseVectorView vec(static_cast<GlobalIndex>(scalar_dof_map.getNumDofs()));
    vec.zero();
    (void)assembler.assembleVector(mesh, scalar_space, kernel, vec);
    return vec;
}

assembly::DenseVectorView assembleBoundaryLinearWithKernel(dofs::DofMap& dof_map,
                                                           const assembly::IMeshAccess& mesh,
                                                           int marker,
                                                           const spaces::FunctionSpace& space,
                                                           assembly::AssemblyKernel& kernel)
{
    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseVectorView vec(static_cast<GlobalIndex>(dof_map.getNumDofs()));
    vec.zero();
    (void)assembler.assembleBoundaryFaces(mesh, marker, space, kernel, nullptr, &vec);
    return vec;
}

void expectDenseNear(const assembly::DenseMatrixView& a,
                     const assembly::DenseMatrixView& b,
                     Real tol)
{
    ASSERT_EQ(a.numRows(), b.numRows());
    ASSERT_EQ(a.numCols(), b.numCols());
    for (GlobalIndex i = 0; i < a.numRows(); ++i) {
        for (GlobalIndex j = 0; j < a.numCols(); ++j) {
            SCOPED_TRACE(::testing::Message() << "row=" << i << ", col=" << j);
            EXPECT_NEAR(a.getMatrixEntry(i, j), b.getMatrixEntry(i, j), tol);
        }
    }
}

void expectDenseNear(const assembly::DenseVectorView& a,
                     const assembly::DenseVectorView& b,
                     Real tol)
{
    ASSERT_EQ(a.numRows(), b.numRows());
    for (GlobalIndex i = 0; i < a.numRows(); ++i) {
        SCOPED_TRACE(::testing::Message() << "row=" << i);
        EXPECT_NEAR(a.getVectorEntry(i), b.getVectorEntry(i), tol);
    }
}

void configureGeometrySensitivityAssembler(assembly::StandardAssembler& assembler,
                                           const dofs::DofMap& dof_map,
                                           const spaces::FunctionSpace& vector_space,
                                           FieldId mesh_field,
                                           const std::vector<Real>& current_solution,
                                           bool bind_mesh_velocity = false,
                                           const assembly::TimeIntegrationContext* time_ctx = nullptr,
                                           Real time_step = 0.0)
{
    const std::array<assembly::FieldSolutionAccess, 1> field_access = {{
        assembly::FieldSolutionAccess{
            .field = mesh_field,
            .space = &vector_space,
            .dof_map = &dof_map,
            .dof_offset = 0,
        },
    }};

    assembler.setDofMap(dof_map);
    assembler.setFieldSolutionAccess(field_access);
    assembler.setMeshMotionFieldAccess(assembly::MeshMotionFieldAccess{
        .mesh_displacement = mesh_field,
        .mesh_velocity = bind_mesh_velocity ? mesh_field : INVALID_FIELD_ID,
    });
    assembler.setCurrentSolution(current_solution);
    assembler.setTimeIntegrationContext(time_ctx);
    assembler.setTimeStep(time_step);
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
        const auto z = zeroVector(3);
        auto vec2 = assembleCellLinear(component(z, 2), dof_map, mesh, space);
        for (GlobalIndex i = 0; i < 4; ++i) {
            EXPECT_NEAR(vec2.getVectorEntry(i), 0.0, 5e-12);
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
    EXPECT_THROW(zeroVector(0), std::invalid_argument);
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

    const auto require_linear_flag = [&](const FormExpr& scalar_expr, assembly::RequiredData flag) {
        auto ir = compiler.compileLinear((scalar_expr * v).dx());
        EXPECT_TRUE(assembly::hasFlag(ir.requiredData(), flag));
    };
    const auto require_linear_flags = [&](const FormExpr& scalar_expr,
                                          assembly::RequiredData value_flag,
                                          assembly::RequiredData gradient_flag) {
        auto ir = compiler.compileLinear((scalar_expr * v).dx());
        EXPECT_TRUE(assembly::hasFlag(ir.requiredData(), value_flag));
        EXPECT_TRUE(assembly::hasFlag(ir.requiredData(), gradient_flag));
    };

    require_linear_flag(component(meshDisplacement(), 0), assembly::RequiredData::MeshDisplacement);
    require_linear_flag(component(meshVelocity(), 0), assembly::RequiredData::MeshVelocity);
    require_linear_flag(component(meshAcceleration(), 0), assembly::RequiredData::MeshAcceleration);
    require_linear_flags(component(grad(meshDisplacement()), 0, 0),
                         assembly::RequiredData::MeshDisplacement,
                         assembly::RequiredData::MeshDisplacementGradient);
    require_linear_flags(div(meshVelocity()),
                         assembly::RequiredData::MeshVelocity,
                         assembly::RequiredData::MeshVelocityGradient);
    require_linear_flags(component(grad(meshAcceleration()), 0, 0),
                         assembly::RequiredData::MeshAcceleration,
                         assembly::RequiredData::MeshAccelerationGradient);
    require_linear_flag(component(previousCoordinate(), 0), assembly::RequiredData::PreviousPhysicalPoints);
    require_linear_flag(component(previousMeshVelocity(), 0), assembly::RequiredData::PreviousMeshVelocity);
    require_linear_flag(component(predictedMeshVelocity(), 0), assembly::RequiredData::PredictedMeshVelocity);
    require_linear_flags(component(grad(previousMeshVelocity()), 0, 0),
                         assembly::RequiredData::PreviousMeshVelocity,
                         assembly::RequiredData::PreviousMeshVelocityGradient);
    require_linear_flags(component(grad(predictedMeshVelocity()), 0, 0),
                         assembly::RequiredData::PredictedMeshVelocity,
                         assembly::RequiredData::PredictedMeshVelocityGradient);
    require_linear_flag(component(currentCoordinate(), 0), assembly::RequiredData::CurrentPhysicalPoints);
    require_linear_flag(component(referenceCoordinatePhysical(), 0), assembly::RequiredData::ReferencePhysicalPoints);
    require_linear_flag(component(currentJacobian(), 0, 0), assembly::RequiredData::CurrentJacobians);
    require_linear_flag(component(referenceJacobian(), 0, 0), assembly::RequiredData::ReferenceJacobians);
    require_linear_flag(currentJacobianDeterminant(), assembly::RequiredData::CurrentJacobians);
    require_linear_flag(referenceJacobianDeterminant(), assembly::RequiredData::ReferenceJacobians);
    require_linear_flag(component(currentNormal(), 0), assembly::RequiredData::CurrentNormals);
    require_linear_flag(component(referenceNormal(), 0), assembly::RequiredData::ReferenceNormals);
    require_linear_flag(currentMeasure(), assembly::RequiredData::CurrentMeasures);
    require_linear_flag(referenceMeasure(), assembly::RequiredData::ReferenceMeasures);
    require_linear_flag(component(surfaceJacobian(), 0, 0), assembly::RequiredData::SurfaceJacobians);

    const auto ir_nanson = compiler.compileLinear((component(nanson(), 0) * v).dx());
    EXPECT_TRUE(assembly::hasFlag(ir_nanson.requiredData(), assembly::RequiredData::CurrentJacobians));
    EXPECT_TRUE(assembly::hasFlag(ir_nanson.requiredData(), assembly::RequiredData::ReferenceJacobians));
    EXPECT_TRUE(assembly::hasFlag(ir_nanson.requiredData(), assembly::RequiredData::ReferenceMeasures));
    EXPECT_TRUE(assembly::hasFlag(ir_nanson.requiredData(), assembly::RequiredData::ReferenceNormals));
    EXPECT_FALSE(assembly::hasFlag(ir_nanson.requiredData(), assembly::RequiredData::ConfigurationTransforms));

    require_linear_flag(component(pushforward(referenceSurfaceVector(),
                                             GeometryConfiguration::Reference,
                                             GeometryConfiguration::Current), 0),
                        assembly::RequiredData::ConfigurationTransforms);
    require_linear_flag(component(pullback(currentSurfaceVector(),
                                           GeometryConfiguration::Current,
                                           GeometryConfiguration::Reference), 0),
                        assembly::RequiredData::ConfigurationTransforms);
}

TEST(FormVocabularyTest, StaticReferenceFormsDoNotRequestMovingDomainData)
{
    FormCompiler compiler;
    spaces::H1Space space(ElementType::Tetra4, 1);
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto ir = compiler.compileBilinear(inner(grad(u), grad(v)).dx());
    EXPECT_TRUE(assembly::hasFlag(ir.requiredData(), assembly::RequiredData::PhysicalGradients));
    EXPECT_FALSE(assembly::hasFlag(ir.requiredData(), assembly::RequiredData::MeshDisplacement));
    EXPECT_FALSE(assembly::hasFlag(ir.requiredData(), assembly::RequiredData::MeshVelocity));
    EXPECT_FALSE(assembly::hasFlag(ir.requiredData(), assembly::RequiredData::MeshAcceleration));
    EXPECT_FALSE(assembly::hasFlag(ir.requiredData(), assembly::RequiredData::CurrentPhysicalPoints));
    EXPECT_FALSE(assembly::hasFlag(ir.requiredData(), assembly::RequiredData::ReferencePhysicalPoints));
    EXPECT_FALSE(assembly::hasFlag(ir.requiredData(), assembly::RequiredData::CurrentJacobians));
    EXPECT_FALSE(assembly::hasFlag(ir.requiredData(), assembly::RequiredData::ReferenceJacobians));
    EXPECT_FALSE(assembly::hasFlag(ir.requiredData(), assembly::RequiredData::CurrentMeasures));
    EXPECT_FALSE(assembly::hasFlag(ir.requiredData(), assembly::RequiredData::ReferenceMeasures));
    EXPECT_FALSE(assembly::hasFlag(ir.requiredData(), assembly::RequiredData::ConfigurationTransforms));
}

TEST(FormVocabularyTest, MovingDomainTimeLevelTerminalsMatchInterpreterAndJIT)
{
    requireLLVMJITOrSkip();

    MovingSingleTetraMeshAccess mesh;
    auto scalar_dof_map = createSingleTetraDofMap();
    spaces::H1Space scalar_space(ElementType::Tetra4, 1);

    auto base_space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    spaces::ProductSpace vector_space(base_space, /*components=*/3);
    auto vector_dof_map =
        createSingleTetraDenseDofMap(static_cast<LocalIndex>(vector_space.dofs_per_element()));

    const auto n_field_dofs = static_cast<std::size_t>(vector_dof_map.getNumDofs());
    std::vector<Real> current_solution(6u * n_field_dofs, 0.0);
    std::fill(current_solution.begin() + static_cast<std::ptrdiff_t>(3u * n_field_dofs),
              current_solution.begin() + static_cast<std::ptrdiff_t>(4u * n_field_dofs),
              0.75);
    std::fill(current_solution.begin() + static_cast<std::ptrdiff_t>(4u * n_field_dofs),
              current_solution.begin() + static_cast<std::ptrdiff_t>(5u * n_field_dofs),
              0.125);
    std::fill(current_solution.begin() + static_cast<std::ptrdiff_t>(5u * n_field_dofs),
              current_solution.end(),
              0.25);

    FormCompiler compiler;
    const auto v = FormExpr::testFunction(scalar_space, "v");
    const auto scalar = component(previousCoordinate(), 0) +
                        component(previousMeshVelocity(), 1) +
                        component(predictedMeshVelocity(), 2);
    const auto form = (scalar * v).dx();

    auto interp_kernel = std::make_shared<FormKernel>(compiler.compileLinear(form));
    auto jit_fallback = std::make_shared<FormKernel>(compiler.compileLinear(form));
    jit::JITKernelWrapper jit_kernel(jit_fallback, makeUnitTestJITOptions());
    jit_kernel.ensureCompiled();
    ASSERT_TRUE(jit_kernel.isJITReady());

    auto interp = assembleMovingCellLinearWithTimeLevelMeshFields(scalar_dof_map,
                                                                  mesh,
                                                                  scalar_space,
                                                                  *interp_kernel,
                                                                  vector_space,
                                                                  vector_dof_map,
                                                                  current_solution);
    auto jit = assembleMovingCellLinearWithTimeLevelMeshFields(scalar_dof_map,
                                                               mesh,
                                                               scalar_space,
                                                               jit_kernel,
                                                               vector_space,
                                                               vector_dof_map,
                                                               current_solution);

    const Real expected_entry = (0.75 + 0.125 + 0.25) * singleTetraP1BasisIntegral();
    for (GlobalIndex i = 0; i < 4; ++i) {
        EXPECT_NEAR(interp.getVectorEntry(i), expected_entry, 5e-12);
        EXPECT_NEAR(jit.getVectorEntry(i), interp.getVectorEntry(i), 5e-12);
    }
}

TEST(FormVocabularyTest, MovingDomainCoordinateTerminalsUseExplicitFrames)
{
    MovingSingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    const auto dx0 = component(currentCoordinate() - referenceCoordinatePhysical(), 0);
    auto vec = assembleCellLinear(dx0, dof_map, mesh, space);

    const Real expected_entry = mesh.translation()[0] * singleTetraP1BasisIntegral();
    for (GlobalIndex i = 0; i < 4; ++i) {
        EXPECT_NEAR(vec.getVectorEntry(i), expected_entry, 5e-12);
    }

    const auto legacy_x0 = component(x() - currentCoordinate(), 0);
    vec = assembleCellLinear(legacy_x0, dof_map, mesh, space);
    for (GlobalIndex i = 0; i < 4; ++i) {
        EXPECT_NEAR(vec.getVectorEntry(i), Real(0.0), 5e-12);
    }
}

TEST(FormVocabularyTest, MovingDomainCoordinateTerminalsMatchInterpreterAndJIT)
{
    requireLLVMJITOrSkip();

    MovingSingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    FormCompiler compiler;
    const auto v = FormExpr::testFunction(space, "v");
    const auto scalar = component(currentCoordinate() - referenceCoordinatePhysical(), 0);
    const auto form = (scalar * v).dx();

    auto interp_kernel = std::make_shared<FormKernel>(compiler.compileLinear(form));
    auto jit_fallback = std::make_shared<FormKernel>(compiler.compileLinear(form));
    jit::JITKernelWrapper jit_kernel(jit_fallback, makeUnitTestJITOptions());
    jit_kernel.ensureCompiled();
    ASSERT_TRUE(jit_kernel.isJITReady());

    auto interp = assembleMovingCellLinearWithKernel(scalar, dof_map, mesh, space, *interp_kernel);
    auto jit = assembleMovingCellLinearWithKernel(scalar, dof_map, mesh, space, jit_kernel);

    for (GlobalIndex i = 0; i < 4; ++i) {
        EXPECT_NEAR(jit.getVectorEntry(i), interp.getVectorEntry(i), 5e-12);
    }
}

TEST(FormVocabularyTest, MovingDomainVolumeTerminalsMatchInterpreterAndJIT)
{
    requireLLVMJITOrSkip();

    MovingSingleTetraMeshAccess mesh;
    auto scalar_dof_map = createSingleTetraDofMap();
    spaces::H1Space scalar_space(ElementType::Tetra4, 1);

    auto base_space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    spaces::ProductSpace vector_space(base_space, /*components=*/3);
    auto vector_dof_map =
        createSingleTetraDenseDofMap(static_cast<LocalIndex>(vector_space.dofs_per_element()));

    const auto n_field_dofs = static_cast<std::size_t>(vector_dof_map.getNumDofs());
    std::vector<Real> current_solution(3u * n_field_dofs, 0.0);
    std::fill(current_solution.begin(), current_solution.begin() + static_cast<std::ptrdiff_t>(n_field_dofs), 0.125);
    std::fill(current_solution.begin() + static_cast<std::ptrdiff_t>(n_field_dofs),
              current_solution.begin() + static_cast<std::ptrdiff_t>(2u * n_field_dofs),
              0.25);
    std::fill(current_solution.begin() + static_cast<std::ptrdiff_t>(2u * n_field_dofs),
              current_solution.end(),
              0.5);

    FormCompiler compiler;
    const auto v = FormExpr::testFunction(scalar_space, "v");
    const auto scalar =
        component(meshDisplacement(), 0) +
        component(meshVelocity(), 1) +
        component(meshAcceleration(), 2) +
        component(currentCoordinate() - referenceCoordinatePhysical(), 0) +
        trace(currentJacobian() - referenceJacobian()) +
        currentJacobianDeterminant() +
        referenceJacobianDeterminant() +
        currentMeasure() +
        referenceMeasure() +
        component(surfaceJacobian(), 0, 0);
    const auto form = (scalar * v).dx();

    auto interp_kernel = std::make_shared<FormKernel>(compiler.compileLinear(form));
    auto jit_fallback = std::make_shared<FormKernel>(compiler.compileLinear(form));
    jit::JITKernelWrapper jit_kernel(jit_fallback, makeUnitTestJITOptions());
    jit_kernel.ensureCompiled();
    ASSERT_TRUE(jit_kernel.isJITReady());

    auto interp = assembleMovingCellLinearWithMeshFields(scalar_dof_map,
                                                         mesh,
                                                         scalar_space,
                                                         *interp_kernel,
                                                         vector_space,
                                                         vector_dof_map,
                                                         current_solution);
    auto jit = assembleMovingCellLinearWithMeshFields(scalar_dof_map,
                                                      mesh,
                                                      scalar_space,
                                                      jit_kernel,
                                                      vector_space,
                                                      vector_dof_map,
                                                      current_solution);

    for (GlobalIndex i = 0; i < 4; ++i) {
        EXPECT_NEAR(jit.getVectorEntry(i), interp.getVectorEntry(i), 5e-12);
    }
}

TEST(FormVocabularyTest, FrameExplicitJacobianDeterminantsAreSignedAndDistinctFromMeasures)
{
    FrameChangedSingleTetraMeshAccess mesh(/*scale=*/{-2.0, 1.0, 1.0});
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    auto det_vec = assembleCellLinear(currentJacobianDeterminant(), dof_map, mesh, space);
    auto measure_vec = assembleCellLinear(currentMeasure(), dof_map, mesh, space);

    // Active integration uses the positive current volume. The determinant
    // terminal remains signed and is therefore negative for the reflected cell.
    const Real active_basis_integral = (2.0 * singleTetraVolume()) / 4.0;
    const Real expected_det_entry = -2.0 * active_basis_integral;
    const Real expected_measure_entry = 2.0 * active_basis_integral;

    for (GlobalIndex i = 0; i < 4; ++i) {
        EXPECT_NEAR(det_vec.getVectorEntry(i), expected_det_entry, 5e-12);
        EXPECT_NEAR(measure_vec.getVectorEntry(i), expected_measure_entry, 5e-12);
    }
}

TEST(FormVocabularyTest, NansonSurfaceVectorMatchesCurrentSurfaceVectorOnScaledFace)
{
    requireLLVMJITOrSkip();

    constexpr int marker = 2;
    FrameChangedSingleTetraMeshAccess mesh(/*scale=*/{2.0, 3.0, 4.0}, marker);
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    FormCompiler compiler;
    const auto v = FormExpr::testFunction(space, "v");
    const auto scalar = component(nanson() - currentSurfaceVector(), 2);
    const auto form = (scalar * v).ds(marker);

    auto interp_kernel = std::make_shared<FormKernel>(compiler.compileLinear(form));
    auto jit_fallback = std::make_shared<FormKernel>(compiler.compileLinear(form));
    jit::JITKernelWrapper jit_kernel(jit_fallback, makeUnitTestJITOptions());
    jit_kernel.ensureCompiled();
    ASSERT_TRUE(jit_kernel.isJITReady());

    auto interp = assembleBoundaryLinearWithKernel(dof_map, mesh, marker, space, *interp_kernel);
    auto jit = assembleBoundaryLinearWithKernel(dof_map, mesh, marker, space, jit_kernel);

    for (GlobalIndex i = 0; i < 4; ++i) {
        EXPECT_NEAR(interp.getVectorEntry(i), 0.0, 5e-12);
        EXPECT_NEAR(jit.getVectorEntry(i), interp.getVectorEntry(i), 5e-12);
    }
}

TEST(FormVocabularyTest, GenericFrameTransformMarkersFailClosedWhenEvaluated)
{
    FrameChangedSingleTetraMeshAccess mesh(/*scale=*/{2.0, 3.0, 4.0});
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    const auto vector_value = as_vector({FormExpr::constant(1.0),
                                         FormExpr::constant(0.0),
                                         FormExpr::constant(0.0)});
    const auto scalar = component(pushforward(vector_value,
                                              GeometryConfiguration::Reference,
                                              GeometryConfiguration::Current), 0);

    EXPECT_THROW((void)assembleCellLinear(scalar, dof_map, mesh, space), std::exception);
}

TEST(FormVocabularyTest, MovingDomainFaceNormalsUseExplicitTerminals)
{
    SingleTetraOneBoundaryFaceMeshAccess mesh(/*boundary_marker=*/2);
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    FormCompiler compiler;
    const auto v = FormExpr::testFunction(space, "v");
    const auto scalar = component(currentNormal() - referenceNormal(), 0) +
                        component(currentNormal() - referenceNormal(), 1) +
                        component(currentNormal() - referenceNormal(), 2);
    auto ir = compiler.compileLinear((scalar * v).ds(2));
    FormKernel kernel(std::move(ir));

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseVectorView vec(4);
    vec.zero();
    (void)assembler.assembleBoundaryFaces(mesh, 2, space, kernel, nullptr, &vec);

    for (GlobalIndex i = 0; i < 4; ++i) {
        EXPECT_NEAR(vec.getVectorEntry(i), Real(0.0), 5e-12);
    }
}

TEST(FormVocabularyTest, MovingDomainBoundaryTerminalsMatchInterpreterAndJIT)
{
    requireLLVMJITOrSkip();

    constexpr int marker = 2;
    SingleTetraOneBoundaryFaceMeshAccess mesh(marker);
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    FormCompiler compiler;
    const auto v = FormExpr::testFunction(space, "v");
    const auto scalar = component(currentNormal(), 0) +
                        component(referenceNormal(), 1) +
                        component(surfaceJacobian(), 0, 0);
    const auto form = (scalar * v).ds(marker);

    auto interp_kernel = std::make_shared<FormKernel>(compiler.compileLinear(form));
    auto jit_fallback = std::make_shared<FormKernel>(compiler.compileLinear(form));
    jit::JITKernelWrapper jit_kernel(jit_fallback, makeUnitTestJITOptions());
    jit_kernel.ensureCompiled();
    ASSERT_TRUE(jit_kernel.isJITReady());

    auto interp = assembleBoundaryLinearWithKernel(dof_map, mesh, marker, space, *interp_kernel);
    auto jit = assembleBoundaryLinearWithKernel(dof_map, mesh, marker, space, jit_kernel);

    for (GlobalIndex i = 0; i < 4; ++i) {
        EXPECT_NEAR(jit.getVectorEntry(i), interp.getVectorEntry(i), 5e-12);
    }
}

TEST(FormVocabularyTest, MovingDomainGeometryTerminalsAreADConstants)
{
    MovingSingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto scalar = component(currentCoordinate() - referenceCoordinatePhysical(), 0);
    auto ir = compiler.compileResidual((scalar * u * v).dx());
    NonlinearFormKernel kernel(std::move(ir), ADMode::Forward);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.setCurrentSolution(std::vector<Real>{0.1, -0.2, 0.3, 0.4});

    assembly::DenseMatrixView J(4);
    assembly::DenseVectorView R(4);
    J.zero();
    R.zero();
    (void)assembler.assembleBoth(mesh, space, space, kernel, J, R);

    const Real scale = mesh.translation()[0];
    const Real V = singleTetraVolume();
    for (GlobalIndex r = 0; r < 4; ++r) {
        for (GlobalIndex c = 0; c < 4; ++c) {
            const Real mass = (r == c) ? V / 10.0 : V / 20.0;
            EXPECT_NEAR(J.getMatrixEntry(r, c), scale * mass, 5e-12);
        }
    }
}

TEST(FormVocabularyTest, OptInCurrentGeometrySensitivitySeedsMeshMotionJacobian)
{
    MovingSingleTetraMeshAccess mesh;
    auto base_space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    spaces::ProductSpace vector_space(base_space, /*components=*/3);
    auto dof_map =
        createSingleTetraDenseDofMap(static_cast<LocalIndex>(vector_space.dofs_per_element()));

    const auto u = FormExpr::trialFunction(vector_space, "mesh_displacement");
    const auto v = FormExpr::testFunction(vector_space, "v");
    const auto residual =
        (inner(currentCoordinate(), v) +
         FormExpr::constant(0.0) * inner(u, v)).dx();

    auto assemble_matrix = [&](SymbolicOptions options) {
        FormCompiler compiler(options);
        auto ir = compiler.compileResidual(residual);
        NonlinearFormKernel kernel(std::move(ir), ADMode::Forward);

        assembly::StandardAssembler assembler;
        assembler.setDofMap(dof_map);
        if (options.geometry_sensitivity.mode ==
            GeometrySensitivityMode::MeshMotionUnknowns) {
            const std::array<assembly::FieldSolutionAccess, 1> field_access = {{
                assembly::FieldSolutionAccess{
                    .field = options.geometry_sensitivity.mesh_motion_field,
                    .space = &vector_space,
                    .dof_map = &dof_map,
                    .dof_offset = 0,
                },
            }};
            assembler.setFieldSolutionAccess(field_access);
        }
        assembler.setCurrentSolution(
            std::vector<Real>(static_cast<std::size_t>(dof_map.getNumDofs()), 0.0));

        assembly::DenseMatrixView J(dof_map.getNumDofs());
        assembly::DenseVectorView R(dof_map.getNumDofs());
        J.zero();
        R.zero();
        (void)assembler.assembleBoth(mesh, vector_space, vector_space, kernel, J, R);
        return J;
    };

    const auto default_J = assemble_matrix(SymbolicOptions{});
    for (GlobalIndex r = 0; r < dof_map.getNumDofs(); ++r) {
        for (GlobalIndex c = 0; c < dof_map.getNumDofs(); ++c) {
            EXPECT_NEAR(default_J.getMatrixEntry(r, c), 0.0, 1e-14);
        }
    }

    SymbolicOptions monolithic_options;
    monolithic_options.geometry_sensitivity.mode =
        GeometrySensitivityMode::MeshMotionUnknowns;
    monolithic_options.geometry_sensitivity.mesh_motion_field = 17;

    const auto monolithic_J = assemble_matrix(monolithic_options);
    Real max_abs = 0.0;
    for (GlobalIndex r = 0; r < dof_map.getNumDofs(); ++r) {
        for (GlobalIndex c = 0; c < dof_map.getNumDofs(); ++c) {
            max_abs = std::max(max_abs, std::abs(monolithic_J.getMatrixEntry(r, c)));
        }
    }
    EXPECT_GT(max_abs, 1e-4);
}

TEST(FormVocabularyTest, CurrentGeometrySensitivityJacobianMatchesFiniteDifference)
{
    MovingSingleTetraMeshAccess mesh;
    auto base_space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    spaces::ProductSpace vector_space(base_space, /*components=*/3);
    auto dof_map =
        createSingleTetraDenseDofMap(static_cast<LocalIndex>(vector_space.dofs_per_element()));

    constexpr FieldId kMeshMotionField = 17;

    const auto u = FormExpr::trialFunction(vector_space, "mesh_displacement");
    const auto v = FormExpr::testFunction(vector_space, "v");
    const auto residual =
        (inner(currentCoordinate(), v) +
         FormExpr::constant(0.0) * inner(u, v)).dx();

    SymbolicOptions options;
    options.geometry_sensitivity.mode = GeometrySensitivityMode::MeshMotionUnknowns;
    options.geometry_sensitivity.mesh_motion_field = kMeshMotionField;

    FormCompiler compiler(options);
    auto ir = compiler.compileResidual(residual);
    NonlinearFormKernel kernel(std::move(ir), ADMode::Forward);

    const std::array<assembly::FieldSolutionAccess, 1> field_access = {{
        assembly::FieldSolutionAccess{
            .field = kMeshMotionField,
            .space = &vector_space,
            .dof_map = &dof_map,
            .dof_offset = 0,
        },
    }};
    const std::vector<Real> zero_solution(static_cast<std::size_t>(dof_map.getNumDofs()), 0.0);

    auto assemble = [&](const MovingSingleTetraMeshAccess& mesh_access,
                        assembly::DenseMatrixView* J,
                        assembly::DenseVectorView& R) {
        assembly::StandardAssembler assembler;
        assembler.setDofMap(dof_map);
        assembler.setFieldSolutionAccess(field_access);
        assembler.setCurrentSolution(zero_solution);
        R.zero();
        if (J != nullptr) {
            J->zero();
            (void)assembler.assembleBoth(mesh_access, vector_space, vector_space, kernel, *J, R);
        } else {
            (void)assembler.assembleVector(mesh_access, vector_space, kernel, R);
        }
    };

    assembly::DenseMatrixView J(dof_map.getNumDofs());
    assembly::DenseVectorView R(dof_map.getNumDofs());
    assemble(mesh, &J, R);

    const Real eps = 1.0e-7;
    constexpr GlobalIndex nodes_per_component = 4;
    for (GlobalIndex col = 0; col < dof_map.getNumDofs(); ++col) {
        const auto node = static_cast<GlobalIndex>(col % nodes_per_component);
        const int component = static_cast<int>(col / nodes_per_component);

        auto mesh_plus = mesh;
        mesh_plus.perturbCurrentNodeComponent(node, component, eps);
        assembly::DenseVectorView R_plus(dof_map.getNumDofs());
        assemble(mesh_plus, nullptr, R_plus);

        auto mesh_minus = mesh;
        mesh_minus.perturbCurrentNodeComponent(node, component, -eps);
        assembly::DenseVectorView R_minus(dof_map.getNumDofs());
        assemble(mesh_minus, nullptr, R_minus);

        for (GlobalIndex row = 0; row < dof_map.getNumDofs(); ++row) {
            const Real fd = (R_plus.getVectorEntry(row) - R_minus.getVectorEntry(row)) /
                            (Real(2.0) * eps);
            SCOPED_TRACE(::testing::Message() << "row=" << row << ", col=" << col);
            EXPECT_NEAR(J.getMatrixEntry(row, col), fd, 2.0e-6);
        }
    }
}

TEST(FormVocabularyTest, SymbolicCurrentGeometryTangentMatchesAD)
{
    MovingSingleTetraMeshAccess mesh;
    auto base_space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    spaces::ProductSpace vector_space(base_space, /*components=*/3);
    auto dof_map =
        createSingleTetraDenseDofMap(static_cast<LocalIndex>(vector_space.dofs_per_element()));

    constexpr FieldId kMeshMotionField = 17;

    const auto u = FormExpr::trialFunction(vector_space, "mesh_displacement");
    const auto v = FormExpr::testFunction(vector_space, "v");
    const auto residual =
        (inner(currentCoordinate(), v) +
         currentMeasure() * inner(u, v) +
         trace(currentJacobian()) * v.component(0)).dx();

    SymbolicOptions options;
    options.geometry_sensitivity.mode = GeometrySensitivityMode::MeshMotionUnknowns;
    options.geometry_sensitivity.mesh_motion_field = kMeshMotionField;

    FormCompiler compiler_ad(options);
    auto ir_ad = compiler_ad.compileResidual(residual);
    NonlinearFormKernel ad_kernel(std::move(ir_ad), ADMode::Forward);

    FormCompiler compiler_sym(options);
    auto ir_sym = compiler_sym.compileResidual(residual);
    SymbolicNonlinearFormKernel sym_kernel(std::move(ir_sym), NonlinearKernelOutput::Both);
    sym_kernel.resolveInlinableConstitutives();

    std::vector<Real> solution(static_cast<std::size_t>(dof_map.getNumDofs()), 0.0);
    for (std::size_t i = 0; i < solution.size(); ++i) {
        const Real sign = (i % 2u == 0u) ? Real(1.0) : Real(-1.0);
        solution[i] = sign * (Real(0.03) + Real(0.004) * static_cast<Real>(i));
    }

    auto assemble = [&](assembly::AssemblyKernel& kernel,
                        assembly::DenseMatrixView& J,
                        assembly::DenseVectorView& R) {
        assembly::StandardAssembler assembler;
        configureGeometrySensitivityAssembler(
            assembler, dof_map, vector_space, kMeshMotionField, solution);
        J.zero();
        R.zero();
        (void)assembler.assembleBoth(mesh, vector_space, vector_space, kernel, J, R);
    };

    assembly::DenseMatrixView J_ad(dof_map.getNumDofs());
    assembly::DenseVectorView R_ad(dof_map.getNumDofs());
    assembly::DenseMatrixView J_sym(dof_map.getNumDofs());
    assembly::DenseVectorView R_sym(dof_map.getNumDofs());

    assemble(ad_kernel, J_ad, R_ad);
    assemble(sym_kernel, J_sym, R_sym);

    expectDenseNear(R_sym, R_ad, 1.0e-12);
    expectDenseNear(J_sym, J_ad, 2.0e-11);
}

TEST(FormVocabularyTest, SymbolicMeshVelocityGeometryTangentMatchesAD)
{
    MovingSingleTetraMeshAccess mesh;
    auto base_space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    spaces::ProductSpace vector_space(base_space, /*components=*/3);
    auto dof_map =
        createSingleTetraDenseDofMap(static_cast<LocalIndex>(vector_space.dofs_per_element()));

    constexpr FieldId kMeshMotionField = 17;
    constexpr Real dt = 0.2;

    assembly::TimeIntegrationContext time_ctx;
    time_ctx.integrator_name = "unit_backward_euler";
    assembly::TimeDerivativeStencil dt1;
    dt1.order = 1;
    dt1.a = {Real(1.0) / dt, -Real(1.0) / dt};
    time_ctx.dt1 = dt1;

    const auto u = FormExpr::trialFunction(vector_space, "mesh_displacement");
    const auto v = FormExpr::testFunction(vector_space, "v");
    const auto residual =
        (inner(meshVelocity(), v) +
         trace(grad(meshVelocity())) * v.component(0) +
         FormExpr::constant(0.0) * inner(u, v)).dx();

    SymbolicOptions options;
    options.geometry_sensitivity.mode = GeometrySensitivityMode::MeshMotionUnknowns;
    options.geometry_sensitivity.mesh_motion_field = kMeshMotionField;

    FormCompiler compiler_ad(options);
    auto ir_ad = compiler_ad.compileResidual(residual);
    NonlinearFormKernel ad_kernel(std::move(ir_ad), ADMode::Forward);

    FormCompiler compiler_sym(options);
    auto ir_sym = compiler_sym.compileResidual(residual);
    SymbolicNonlinearFormKernel sym_kernel(std::move(ir_sym), NonlinearKernelOutput::Both);
    sym_kernel.resolveInlinableConstitutives();

    std::vector<Real> solution(static_cast<std::size_t>(dof_map.getNumDofs()), 0.0);
    for (std::size_t i = 0; i < solution.size(); ++i) {
        solution[i] = Real(0.02) + Real(0.003) * static_cast<Real>(i);
    }
    std::vector<Real> previous_solution(solution.size(), Real(0.0));

    auto assemble = [&](assembly::AssemblyKernel& kernel,
                        assembly::DenseMatrixView& J,
                        assembly::DenseVectorView& R) {
        assembly::StandardAssembler assembler;
        configureGeometrySensitivityAssembler(
            assembler,
            dof_map,
            vector_space,
            kMeshMotionField,
            solution,
            /*bind_mesh_velocity=*/true,
            &time_ctx,
            dt);
        assembler.setPreviousSolution(previous_solution);
        J.zero();
        R.zero();
        (void)assembler.assembleBoth(mesh, vector_space, vector_space, kernel, J, R);
    };

    assembly::DenseMatrixView J_ad(dof_map.getNumDofs());
    assembly::DenseVectorView R_ad(dof_map.getNumDofs());
    assembly::DenseMatrixView J_sym(dof_map.getNumDofs());
    assembly::DenseVectorView R_sym(dof_map.getNumDofs());

    assemble(ad_kernel, J_ad, R_ad);
    assemble(sym_kernel, J_sym, R_sym);

    expectDenseNear(R_sym, R_ad, 1.0e-12);
    expectDenseNear(J_sym, J_ad, 2.0e-11);
}

TEST(FormVocabularyTest, SymbolicBoundaryGeometryTangentMatchesAD)
{
    constexpr int marker = 2;
    FrameChangedSingleTetraMeshAccess mesh({Real(1.15), Real(0.85), Real(1.25)}, marker);
    auto base_space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    spaces::ProductSpace vector_space(base_space, /*components=*/3);
    auto dof_map =
        createSingleTetraDenseDofMap(static_cast<LocalIndex>(vector_space.dofs_per_element()));

    constexpr FieldId kMeshMotionField = 17;

    const auto u = FormExpr::trialFunction(vector_space, "mesh_displacement");
    const auto v = FormExpr::testFunction(vector_space, "v");
    const auto residual =
        (inner(currentNormal(), v) +
         currentMeasure() * inner(u, v) +
         component(surfaceJacobian(), 0, 0) * v.component(0)).ds(marker);

    SymbolicOptions options;
    options.geometry_sensitivity.mode = GeometrySensitivityMode::MeshMotionUnknowns;
    options.geometry_sensitivity.mesh_motion_field = kMeshMotionField;

    FormCompiler compiler_ad(options);
    auto ir_ad = compiler_ad.compileResidual(residual);
    NonlinearFormKernel ad_kernel(std::move(ir_ad), ADMode::Forward);

    FormCompiler compiler_sym(options);
    auto ir_sym = compiler_sym.compileResidual(residual);
    SymbolicNonlinearFormKernel sym_kernel(std::move(ir_sym), NonlinearKernelOutput::Both);
    sym_kernel.resolveInlinableConstitutives();

    std::vector<Real> solution(static_cast<std::size_t>(dof_map.getNumDofs()), 0.0);
    for (std::size_t i = 0; i < solution.size(); ++i) {
        const Real sign = (i % 2u == 0u) ? Real(1.0) : Real(-1.0);
        solution[i] = sign * (Real(0.015) + Real(0.002) * static_cast<Real>(i));
    }

    auto assemble = [&](assembly::AssemblyKernel& kernel,
                        assembly::DenseMatrixView& J,
                        assembly::DenseVectorView& R) {
        assembly::StandardAssembler assembler;
        configureGeometrySensitivityAssembler(
            assembler, dof_map, vector_space, kMeshMotionField, solution);
        J.zero();
        R.zero();
        (void)assembler.assembleBoundaryFaces(mesh, marker, vector_space, kernel, &J, &R);
    };

    assembly::DenseMatrixView J_ad(dof_map.getNumDofs());
    assembly::DenseVectorView R_ad(dof_map.getNumDofs());
    assembly::DenseMatrixView J_sym(dof_map.getNumDofs());
    assembly::DenseVectorView R_sym(dof_map.getNumDofs());

    assemble(ad_kernel, J_ad, R_ad);
    assemble(sym_kernel, J_sym, R_sym);

    expectDenseNear(R_sym, R_ad, 1.0e-12);
    expectDenseNear(J_sym, J_ad, 2.0e-10);
}

TEST(FormVocabularyTest, SymbolicInteriorGeometryTangentMatchesAD)
{
    TwoTetraSharedFaceMeshAccess mesh;
    auto base_space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    spaces::ProductSpace vector_space(base_space, /*components=*/3);
    auto dof_map =
        createTwoTetraDGDenseDofMap(static_cast<LocalIndex>(vector_space.dofs_per_element()));

    constexpr FieldId kMeshMotionField = 17;

    const auto u = FormExpr::trialFunction(vector_space, "mesh_displacement");
    const auto v = FormExpr::testFunction(vector_space, "v");
    const auto residual =
        (inner(currentNormal().minus(), v.minus()) +
         inner(currentNormal().plus(), v.plus()) +
         component(surfaceJacobian().minus(), 0, 0) * v.minus().component(0) +
         component(surfaceJacobian().plus(), 0, 0) * v.plus().component(0) +
         FormExpr::constant(0.0) * inner(u.minus(), v.minus())).dS();

    SymbolicOptions options;
    options.geometry_sensitivity.mode = GeometrySensitivityMode::MeshMotionUnknowns;
    options.geometry_sensitivity.mesh_motion_field = kMeshMotionField;

    FormCompiler compiler_ad(options);
    auto ir_ad = compiler_ad.compileResidual(residual);
    NonlinearFormKernel ad_kernel(std::move(ir_ad), ADMode::Forward);

    FormCompiler compiler_sym(options);
    auto ir_sym = compiler_sym.compileResidual(residual);
    SymbolicNonlinearFormKernel sym_kernel(std::move(ir_sym), NonlinearKernelOutput::Both);
    sym_kernel.resolveInlinableConstitutives();

    std::vector<Real> solution(static_cast<std::size_t>(dof_map.getNumDofs()), 0.0);
    for (std::size_t i = 0; i < solution.size(); ++i) {
        const Real sign = (i % 2u == 0u) ? Real(1.0) : Real(-1.0);
        solution[i] = sign * (Real(0.01) + Real(0.0015) * static_cast<Real>(i));
    }

    auto assemble = [&](assembly::AssemblyKernel& kernel,
                        assembly::DenseMatrixView& J,
                        assembly::DenseVectorView& R) {
        assembly::StandardAssembler assembler;
        configureGeometrySensitivityAssembler(
            assembler, dof_map, vector_space, kMeshMotionField, solution);
        J.zero();
        R.zero();
        (void)assembler.assembleInteriorFaces(mesh, vector_space, vector_space, kernel, J, &R);
    };

    assembly::DenseMatrixView J_ad(dof_map.getNumDofs());
    assembly::DenseVectorView R_ad(dof_map.getNumDofs());
    assembly::DenseMatrixView J_sym(dof_map.getNumDofs());
    assembly::DenseVectorView R_sym(dof_map.getNumDofs());

    assemble(ad_kernel, J_ad, R_ad);
    assemble(sym_kernel, J_sym, R_sym);

    expectDenseNear(R_sym, R_ad, 1.0e-12);
    expectDenseNear(J_sym, J_ad, 3.0e-10);
}

TEST(FormVocabularyTest, SymbolicCurrentGeometryTangentJITMatchesInterpreter)
{
    requireLLVMJITOrSkip();

    MovingSingleTetraMeshAccess mesh;
    auto base_space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    spaces::ProductSpace vector_space(base_space, /*components=*/3);
    auto dof_map =
        createSingleTetraDenseDofMap(static_cast<LocalIndex>(vector_space.dofs_per_element()));

    constexpr FieldId kMeshMotionField = 17;

    const auto u = FormExpr::trialFunction(vector_space, "mesh_displacement");
    const auto v = FormExpr::testFunction(vector_space, "v");
    const auto residual =
        (inner(currentCoordinate(), v) +
         currentMeasure() * inner(u, v) +
         trace(currentJacobian()) * v.component(0)).dx();

    SymbolicOptions options;
    options.geometry_sensitivity.mode = GeometrySensitivityMode::MeshMotionUnknowns;
    options.geometry_sensitivity.mesh_motion_field = kMeshMotionField;

    FormCompiler compiler_interp(options);
    auto ir_interp = compiler_interp.compileResidual(residual);
    SymbolicNonlinearFormKernel interp_kernel(std::move(ir_interp), NonlinearKernelOutput::Both);
    interp_kernel.resolveInlinableConstitutives();

    FormCompiler compiler_jit(options);
    auto ir_jit = compiler_jit.compileResidual(residual);
    auto jit_fallback =
        std::make_shared<SymbolicNonlinearFormKernel>(std::move(ir_jit), NonlinearKernelOutput::Both);
    jit_fallback->resolveInlinableConstitutives();
    jit::JITKernelWrapper jit_kernel(jit_fallback, makeUnitTestJITOptions());
    jit_kernel.ensureCompiled();
    ASSERT_TRUE(jit_kernel.isJITReady());
    EXPECT_TRUE(jit_kernel.hasCompiledTangentDispatch(IntegralDomain::Cell));

    std::vector<Real> solution(static_cast<std::size_t>(dof_map.getNumDofs()), 0.0);
    for (std::size_t i = 0; i < solution.size(); ++i) {
        solution[i] = Real(0.01) + Real(0.005) * static_cast<Real>(i);
    }

    auto assemble = [&](assembly::AssemblyKernel& kernel,
                        assembly::DenseMatrixView& J,
                        assembly::DenseVectorView& R) {
        assembly::StandardAssembler assembler;
        configureGeometrySensitivityAssembler(
            assembler, dof_map, vector_space, kMeshMotionField, solution);
        J.zero();
        R.zero();
        (void)assembler.assembleBoth(mesh, vector_space, vector_space, kernel, J, R);
    };

    assembly::DenseMatrixView J_interp(dof_map.getNumDofs());
    assembly::DenseVectorView R_interp(dof_map.getNumDofs());
    assembly::DenseMatrixView J_jit(dof_map.getNumDofs());
    assembly::DenseVectorView R_jit(dof_map.getNumDofs());

    assemble(interp_kernel, J_interp, R_interp);
    assemble(jit_kernel, J_jit, R_jit);

    expectDenseNear(R_jit, R_interp, 1.0e-12);
    expectDenseNear(J_jit, J_interp, 2.0e-11);
}

TEST(FormVocabularyTest, SymbolicBoundaryGeometryTangentJITMatchesInterpreter)
{
    requireLLVMJITOrSkip();

    constexpr int marker = 2;
    FrameChangedSingleTetraMeshAccess mesh({Real(1.15), Real(0.85), Real(1.25)}, marker);
    auto base_space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    spaces::ProductSpace vector_space(base_space, /*components=*/3);
    auto dof_map =
        createSingleTetraDenseDofMap(static_cast<LocalIndex>(vector_space.dofs_per_element()));

    constexpr FieldId kMeshMotionField = 17;

    const auto u = FormExpr::trialFunction(vector_space, "mesh_displacement");
    const auto v = FormExpr::testFunction(vector_space, "v");
    const auto residual =
        (inner(currentNormal(), v) +
         currentMeasure() * inner(u, v) +
         component(surfaceJacobian(), 0, 0) * v.component(0)).ds(marker);

    SymbolicOptions options;
    options.geometry_sensitivity.mode = GeometrySensitivityMode::MeshMotionUnknowns;
    options.geometry_sensitivity.mesh_motion_field = kMeshMotionField;

    FormCompiler compiler_interp(options);
    auto ir_interp = compiler_interp.compileResidual(residual);
    SymbolicNonlinearFormKernel interp_kernel(std::move(ir_interp), NonlinearKernelOutput::Both);
    interp_kernel.resolveInlinableConstitutives();

    FormCompiler compiler_jit(options);
    auto ir_jit = compiler_jit.compileResidual(residual);
    auto jit_fallback =
        std::make_shared<SymbolicNonlinearFormKernel>(std::move(ir_jit), NonlinearKernelOutput::Both);
    jit_fallback->resolveInlinableConstitutives();
    jit::JITKernelWrapper jit_kernel(jit_fallback, makeUnitTestJITOptions());
    jit_kernel.ensureCompiled();
    ASSERT_TRUE(jit_kernel.isJITReady());
    EXPECT_TRUE(jit_kernel.hasCompiledTangentDispatch(IntegralDomain::Boundary, marker));

    std::vector<Real> solution(static_cast<std::size_t>(dof_map.getNumDofs()), 0.0);
    for (std::size_t i = 0; i < solution.size(); ++i) {
        const Real sign = (i % 2u == 0u) ? Real(1.0) : Real(-1.0);
        solution[i] = sign * (Real(0.012) + Real(0.003) * static_cast<Real>(i));
    }

    auto assemble = [&](assembly::AssemblyKernel& kernel,
                        assembly::DenseMatrixView& J,
                        assembly::DenseVectorView& R) {
        assembly::StandardAssembler assembler;
        configureGeometrySensitivityAssembler(
            assembler, dof_map, vector_space, kMeshMotionField, solution);
        J.zero();
        R.zero();
        (void)assembler.assembleBoundaryFaces(mesh, marker, vector_space, kernel, &J, &R);
    };

    assembly::DenseMatrixView J_interp(dof_map.getNumDofs());
    assembly::DenseVectorView R_interp(dof_map.getNumDofs());
    assembly::DenseMatrixView J_jit(dof_map.getNumDofs());
    assembly::DenseVectorView R_jit(dof_map.getNumDofs());

    assemble(interp_kernel, J_interp, R_interp);
    assemble(jit_kernel, J_jit, R_jit);

    expectDenseNear(R_jit, R_interp, 1.0e-12);
    expectDenseNear(J_jit, J_interp, 2.0e-10);
}

TEST(FormVocabularyTest, SymbolicInteriorGeometryTangentJITMatchesInterpreter)
{
    requireLLVMJITOrSkip();

    TwoTetraSharedFaceMeshAccess mesh;
    auto base_space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    spaces::ProductSpace vector_space(base_space, /*components=*/3);
    auto dof_map =
        createTwoTetraDGDenseDofMap(static_cast<LocalIndex>(vector_space.dofs_per_element()));

    constexpr FieldId kMeshMotionField = 17;

    const auto u = FormExpr::trialFunction(vector_space, "mesh_displacement");
    const auto v = FormExpr::testFunction(vector_space, "v");
    const auto residual =
        (inner(currentNormal().minus(), v.minus()) +
         inner(currentNormal().plus(), v.plus()) +
         currentMeasure().minus() * inner(u.minus(), v.minus()) +
         currentMeasure().plus() * inner(u.plus(), v.plus()) +
         component(surfaceJacobian().minus(), 0, 0) * v.minus().component(0) +
         component(surfaceJacobian().plus(), 0, 0) * v.plus().component(0)).dS();

    SymbolicOptions options;
    options.geometry_sensitivity.mode = GeometrySensitivityMode::MeshMotionUnknowns;
    options.geometry_sensitivity.mesh_motion_field = kMeshMotionField;

    FormCompiler compiler_interp(options);
    auto ir_interp = compiler_interp.compileResidual(residual);
    SymbolicNonlinearFormKernel interp_kernel(std::move(ir_interp), NonlinearKernelOutput::Both);
    interp_kernel.resolveInlinableConstitutives();

    FormCompiler compiler_jit(options);
    auto ir_jit = compiler_jit.compileResidual(residual);
    auto jit_fallback =
        std::make_shared<SymbolicNonlinearFormKernel>(std::move(ir_jit), NonlinearKernelOutput::Both);
    jit_fallback->resolveInlinableConstitutives();
    jit::JITKernelWrapper jit_kernel(jit_fallback, makeUnitTestJITOptions());
    jit_kernel.ensureCompiled();
    ASSERT_TRUE(jit_kernel.isJITReady());
    EXPECT_TRUE(jit_kernel.hasCompiledTangentDispatch(IntegralDomain::InteriorFace));

    std::vector<Real> solution(static_cast<std::size_t>(dof_map.getNumDofs()), 0.0);
    for (std::size_t i = 0; i < solution.size(); ++i) {
        const Real sign = (i % 2u == 0u) ? Real(1.0) : Real(-1.0);
        solution[i] = sign * (Real(0.008) + Real(0.002) * static_cast<Real>(i));
    }

    auto assemble = [&](assembly::AssemblyKernel& kernel,
                        assembly::DenseMatrixView& J,
                        assembly::DenseVectorView& R) {
        assembly::StandardAssembler assembler;
        configureGeometrySensitivityAssembler(
            assembler, dof_map, vector_space, kMeshMotionField, solution);
        J.zero();
        R.zero();
        (void)assembler.assembleInteriorFaces(mesh, vector_space, vector_space, kernel, J, &R);
    };

    assembly::DenseMatrixView J_interp(dof_map.getNumDofs());
    assembly::DenseVectorView R_interp(dof_map.getNumDofs());
    assembly::DenseMatrixView J_jit(dof_map.getNumDofs());
    assembly::DenseVectorView R_jit(dof_map.getNumDofs());

    assemble(interp_kernel, J_interp, R_interp);
    assemble(jit_kernel, J_jit, R_jit);

    expectDenseNear(R_jit, R_interp, 1.0e-12);
    expectDenseNear(J_jit, J_interp, 3.0e-10);
}

TEST(FormVocabularyTest, SymbolicInterfaceGeometryTangentJITCompilesTangentDispatch)
{
    requireLLVMJITOrSkip();

    constexpr int marker = 11;
    auto base_space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    spaces::ProductSpace vector_space(base_space, /*components=*/3);
    constexpr FieldId kMeshMotionField = 17;

    const auto u = FormExpr::trialFunction(vector_space, "mesh_displacement");
    const auto v = FormExpr::testFunction(vector_space, "v");
    const auto residual =
        (inner(currentNormal().minus(), v.minus()) +
         inner(currentNormal().plus(), v.plus()) +
         currentMeasure().minus() * inner(u.minus(), v.minus()) +
         currentMeasure().plus() * inner(u.plus(), v.plus()) +
         component(surfaceJacobian().minus(), 0, 0) * v.minus().component(0) +
         component(surfaceJacobian().plus(), 0, 0) * v.plus().component(0)).dI(marker);

    SymbolicOptions options;
    options.geometry_sensitivity.mode = GeometrySensitivityMode::MeshMotionUnknowns;
    options.geometry_sensitivity.mesh_motion_field = kMeshMotionField;

    FormCompiler compiler(options);
    auto ir = compiler.compileResidual(residual);
    auto jit_fallback =
        std::make_shared<SymbolicNonlinearFormKernel>(std::move(ir), NonlinearKernelOutput::Both);
    jit_fallback->resolveInlinableConstitutives();
    ASSERT_TRUE(jit_fallback->hasInterfaceFace());
    ASSERT_TRUE(jit_fallback->tangentIR().hasInterfaceFaceTerms());

    jit::JITKernelWrapper jit_kernel(jit_fallback, makeUnitTestJITOptions());
    jit_kernel.ensureCompiled();
    ASSERT_TRUE(jit_kernel.isJITReady());
    EXPECT_TRUE(jit_kernel.hasCompiledTangentDispatch(IntegralDomain::InterfaceFace, marker));
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
