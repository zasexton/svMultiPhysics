/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_FormKernel_DG.cpp
 * @brief Unit tests for FE/Forms interior-facet (dS) assembly
 */

#include <gtest/gtest.h>

#include "Assembly/CutIntegrationContext.h"
#include "Assembly/GlobalSystemView.h"
#include "Assembly/StandardAssembler.h"
#include "Forms/CutCellForms.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/Vocabulary.h"
#include "Spaces/H1Space.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <array>
#include <cmath>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace test {

namespace {

class ThreeTetraChainMeshAccess final : public assembly::IMeshAccess {
public:
    ThreeTetraChainMeshAccess()
    {
        nodes_ = {
            {0.0, 0.0, 0.0},
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0},
            {1.0, 1.0, 1.0},
            {1.0, 0.0, 2.0}
        };
        cells_ = {
            {0, 1, 2, 3},
            {1, 2, 3, 4},
            {2, 3, 4, 5}
        };
    }

    [[nodiscard]] GlobalIndex numCells() const override { return 3; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 3; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 2; }
    [[nodiscard]] int dimension() const override { return 3; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex) const override { return true; }

    [[nodiscard]] ElementType getCellType(GlobalIndex) const override {
        return ElementType::Tetra4;
    }

    void getCellNodes(GlobalIndex cell_id, std::vector<GlobalIndex>& nodes) const override {
        const auto& cell = cells_.at(static_cast<std::size_t>(cell_id));
        nodes.assign(cell.begin(), cell.end());
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(GlobalIndex cell_id,
                            std::vector<std::array<Real, 3>>& coords) const override
    {
        const auto& cell = cells_.at(static_cast<std::size_t>(cell_id));
        coords.resize(cell.size());
        for (std::size_t i = 0; i < cell.size(); ++i) {
            coords[i] = nodes_.at(static_cast<std::size_t>(cell[i]));
        }
    }

    [[nodiscard]] bool supportsCoordinateFrame(assembly::CoordinateFrame frame) const override
    {
        return frame == assembly::CoordinateFrame::Active ||
               frame == assembly::CoordinateFrame::Reference ||
               frame == assembly::CoordinateFrame::Current;
    }

    void getCellCoordinates(GlobalIndex cell_id,
                            assembly::CoordinateFrame,
                            std::vector<std::array<Real, 3>>& coords) const override
    {
        getCellCoordinates(cell_id, coords);
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex face_id,
                                               GlobalIndex cell_id) const override
    {
        if (face_id == 0 && cell_id == 0) return 2;
        if (face_id == 0 && cell_id == 1) return 0;
        if (face_id == 1 && cell_id == 1) return 2;
        if (face_id == 1 && cell_id == 2) return 0;
        return 0;
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex) const override { return -1; }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex>
    getInteriorFaceCells(GlobalIndex face_id) const override {
        if (face_id == 0) return {0, 1};
        if (face_id == 1) return {1, 2};
        return {0, 0};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override {
        callback(0);
        callback(1);
        callback(2);
    }

    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override {
        forEachCell(std::move(callback));
    }

    void forEachBoundaryFace(int,
                             std::function<void(GlobalIndex, GlobalIndex)>) const override
    {
    }

    void forEachInteriorFace(
        std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> callback) const override
    {
        callback(0, 0, 1);
        callback(1, 1, 2);
    }

private:
    std::vector<std::array<Real, 3>> nodes_;
    std::vector<std::array<GlobalIndex, 4>> cells_;
};

[[nodiscard]] dofs::DofMap createThreeTetraDG_DofMap()
{
    dofs::DofMap dof_map(3, 12, 4);
    dof_map.setCellDofs(0, std::vector<GlobalIndex>{0, 1, 2, 3});
    dof_map.setCellDofs(1, std::vector<GlobalIndex>{4, 5, 6, 7});
    dof_map.setCellDofs(2, std::vector<GlobalIndex>{8, 9, 10, 11});
    dof_map.setNumDofs(12);
    dof_map.setNumLocalDofs(12);
    dof_map.finalize();
    return dof_map;
}

[[nodiscard]] dofs::DofMap createThreeTetraDG_P2_DofMap()
{
    dofs::DofMap dof_map(3, 30, 10);
    dof_map.setCellDofs(
        0, std::vector<GlobalIndex>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    dof_map.setCellDofs(
        1, std::vector<GlobalIndex>{10, 11, 12, 13, 14, 15, 16, 17, 18, 19});
    dof_map.setCellDofs(
        2, std::vector<GlobalIndex>{20, 21, 22, 23, 24, 25, 26, 27, 28, 29});
    dof_map.setNumDofs(30);
    dof_map.setNumLocalDofs(30);
    dof_map.finalize();
    return dof_map;
}

} // namespace

TEST(FormKernelDGTest, PenaltyJumpJumpProducesExpectedBlocks)
{
    TwoTetraSharedFaceMeshAccess mesh;
    auto dof_map = createTwoTetraDG_DofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    const Real eta = 2.5;

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto form = (FormExpr::constant(eta) * inner(jump(u), jump(v))).dS();

    auto ir = compiler.compileBilinear(form);
    FormKernel kernel(std::move(ir));

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseMatrixView mat(8);
    mat.zero();

    const auto result = assembler.assembleInteriorFaces(mesh, space, space, kernel, mat, nullptr);
    EXPECT_EQ(result.interior_faces_assembled, 1);

    const Real area = std::sqrt(3.0) / 2.0;
    const Real mdiag = eta * (area / 6.0);
    const Real moff = eta * (area / 12.0);

    auto expected_entry = [&](GlobalIndex i, GlobalIndex j) -> Real {
        // Map global DOFs to (side, local index on shared face)
        // Minus cell DOFs: 0,1,2,3 (face DOFs are 1,2,3 -> map to 0,1,2)
        // Plus cell DOFs: 4,5,6,7 (face DOFs are 4,5,6 -> map to 0,1,2)
        auto faceIndex = [](GlobalIndex dof) -> int {
            if (dof == 1) return 0;
            if (dof == 2) return 1;
            if (dof == 3) return 2;
            if (dof == 4) return 0;
            if (dof == 5) return 1;
            if (dof == 6) return 2;
            return -1;
        };
        auto isMinus = [](GlobalIndex dof) -> bool { return dof < 4; };

        const int fi = faceIndex(i);
        const int fj = faceIndex(j);
        if (fi < 0 || fj < 0) return 0.0;

        const bool ii_minus = isMinus(i);
        const bool jj_minus = isMinus(j);
        const bool same_side = (ii_minus == jj_minus);

        const Real base = (fi == fj) ? mdiag : moff;
        return same_side ? base : -base;
    };

    for (GlobalIndex i = 0; i < 8; ++i) {
        for (GlobalIndex j = 0; j < 8; ++j) {
            SCOPED_TRACE(::testing::Message() << "i=" << i << ", j=" << j);
            EXPECT_NEAR(mat.getMatrixEntry(i, j), expected_entry(i, j), 5e-11);
        }
    }
}

TEST(FormKernelDGTest, MarkedInteriorFacesUseCutFacetSetHandle)
{
    TwoTetraSharedFaceMeshAccess mesh;
    auto dof_map = createTwoTetraDG_DofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    constexpr int active_marker = 12;
    constexpr int inactive_marker = 13;
    const Real eta = 2.5;

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto form = (FormExpr::constant(eta) * inner(jump(u), jump(v))).dS(active_marker);

    auto ir = compiler.compileBilinear(form);
    FormKernel kernel(std::move(ir));

    assembly::CutIntegrationContext cut_context;
    assembly::CutFacetSetHandle active_handle;
    active_handle.marker = active_marker;
    active_handle.name = "active-cut-adjacent-facets";
    active_handle.facets = {0};
    cut_context.addFacetSetHandle(std::move(active_handle));

    assembly::CutFacetSetHandle inactive_handle;
    inactive_handle.marker = inactive_marker;
    inactive_handle.name = "inactive-cut-adjacent-facets";
    inactive_handle.facets = {0};
    cut_context.addFacetSetHandle(std::move(inactive_handle));

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.setCutIntegrationContext(&cut_context);

    assembly::DenseMatrixView active_mat(8);
    active_mat.zero();
    const auto active_result = assembler.assembleInteriorFaces(mesh, space, space, kernel,
                                                               active_mat, nullptr,
                                                               active_marker);
    EXPECT_EQ(active_result.interior_faces_assembled, 1);

    const Real area = std::sqrt(3.0) / 2.0;
    EXPECT_NEAR(active_mat.getMatrixEntry(1, 1), eta * area / 6.0, 5e-11);
    EXPECT_NEAR(active_mat.getMatrixEntry(1, 4), -eta * area / 6.0, 5e-11);

    assembly::DenseMatrixView inactive_mat(8);
    inactive_mat.zero();
    const auto inactive_result = assembler.assembleInteriorFaces(mesh, space, space, kernel,
                                                                 inactive_mat, nullptr,
                                                                 inactive_marker);
    EXPECT_EQ(inactive_result.interior_faces_assembled, 1);
    for (GlobalIndex i = 0; i < 8; ++i) {
        for (GlobalIndex j = 0; j < 8; ++j) {
            SCOPED_TRACE(::testing::Message() << "i=" << i << ", j=" << j);
            EXPECT_DOUBLE_EQ(inactive_mat.getMatrixEntry(i, j), 0.0);
        }
    }

    assembly::DenseMatrixView missing_mat(8);
    missing_mat.zero();
    EXPECT_THROW((void)assembler.assembleInteriorFaces(mesh, space, space, kernel,
                                                       missing_mat, nullptr,
                                                       /*interior_facet_marker=*/99),
                 FEException);
}

TEST(FormKernelDGTest, MarkedCutAdjacentInteriorFacesRequireCutContext)
{
    TwoTetraSharedFaceMeshAccess mesh;
    auto dof_map = createTwoTetraDG_DofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    constexpr int marker = 12;

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto form = cutAdjacentFacetIntegral(
        cutStabilizationScale() * inner(jump(u), jump(v)), marker);

    auto ir = compiler.compileBilinear(form);
    FormKernel kernel(std::move(ir));

    assembly::StandardAssembler assembler_without_context;
    assembler_without_context.setDofMap(dof_map);
    assembly::DenseMatrixView no_context_mat(8);
    no_context_mat.zero();
    try {
        (void)assembler_without_context.assembleInteriorFaces(
            mesh, space, space, kernel, no_context_mat, nullptr, marker);
        FAIL() << "Expected marked cut-adjacent assembly to require a cut context";
    } catch (const FEException& ex) {
        const std::string message = ex.what();
        EXPECT_NE(message.find("requires a cut integration context"), std::string::npos);
    }

    assembly::CutIntegrationContext empty_context;
    assembly::StandardAssembler assembler_without_facet_set;
    assembler_without_facet_set.setDofMap(dof_map);
    assembler_without_facet_set.setCutIntegrationContext(&empty_context);
    assembly::DenseMatrixView missing_set_mat(8);
    missing_set_mat.zero();
    try {
        (void)assembler_without_facet_set.assembleInteriorFaces(
            mesh, space, space, kernel, missing_set_mat, nullptr, marker);
        FAIL() << "Expected marked cut-adjacent assembly to require a facet set";
    } catch (const FEException& ex) {
        const std::string message = ex.what();
        EXPECT_NE(message.find("missing cut-adjacent facet set"), std::string::npos);
    }
}

TEST(FormKernelDGTest, MarkedInteriorFaceUsesFacetBoundCutStabilizationScale)
{
    TwoTetraSharedFaceMeshAccess mesh;
    auto dof_map = createTwoTetraDG_DofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    constexpr int marker = 18;
    constexpr Real eta = 3.0;

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto form = (cutStabilizationScale() * inner(jump(u), jump(v))).dS(marker);

    auto ir = compiler.compileBilinear(form);
    FormKernel kernel(std::move(ir));

    assembly::CutIntegrationContext cut_context;
    assembly::CutFacetSetHandle handle;
    handle.marker = marker;
    handle.name = "facet-bound-cut-scale";
    handle.facets = {0};
    assembly::CutFacetSetFacetMetadata metadata;
    metadata.facet = 0;
    metadata.first_cell = 0;
    metadata.second_cell = 1;
    metadata.stabilization_scale = eta;
    handle.facet_metadata.push_back(metadata);
    cut_context.addFacetSetHandle(std::move(handle));

    const auto* stored_handle = cut_context.facetSetHandleForMarker(marker);
    ASSERT_NE(stored_handle, nullptr);
    EXPECT_NEAR(stored_handle->stabilizationScaleForFacet(0), eta, 1.0e-12);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.setCutIntegrationContext(&cut_context);

    assembly::DenseMatrixView mat(8);
    mat.zero();

    const auto result = assembler.assembleInteriorFaces(mesh, space, space, kernel,
                                                        mat, nullptr, marker);
    EXPECT_EQ(result.interior_faces_assembled, 1);

    const Real area = std::sqrt(3.0) / 2.0;
    EXPECT_NEAR(mat.getMatrixEntry(1, 1), eta * area / 6.0, 5e-11);
    EXPECT_NEAR(mat.getMatrixEntry(1, 4), -eta * area / 6.0, 5e-11);
}

TEST(FormKernelDGTest, UnscaledMarkedCutAdjacentFacetIntegralSkipsNonCutInteriorFaces)
{
    ThreeTetraChainMeshAccess mesh;
    auto dof_map = createThreeTetraDG_DofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    constexpr int marker = 22;

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto form = (FormExpr::constant(1.0) * inner(jump(u), jump(v))).dS(marker);

    auto ir = compiler.compileBilinear(form);
    FormKernel kernel(std::move(ir));

    assembly::CutIntegrationContext cut_context;
    assembly::CutFacetSetHandle handle;
    handle.marker = marker;
    handle.name = "single-cut-adjacent-facet";
    handle.facets = {0};
    assembly::CutFacetSetFacetMetadata metadata;
    metadata.facet = 0;
    metadata.first_cell = 0;
    metadata.second_cell = 1;
    metadata.stabilization_scale = 1.0;
    handle.facet_metadata.push_back(metadata);
    cut_context.addFacetSetHandle(std::move(handle));
    const auto* stored_handle = cut_context.facetSetHandleForMarker(marker);
    ASSERT_NE(stored_handle, nullptr);
    EXPECT_TRUE(stored_handle->containsFacet(0));
    EXPECT_FALSE(stored_handle->containsFacet(1));

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.setCutIntegrationContext(&cut_context);

    assembly::DenseMatrixView mat(12);
    mat.zero();

    const auto result = assembler.assembleInteriorFaces(mesh, space, space, kernel,
                                                        mat, nullptr, marker);
    EXPECT_EQ(result.interior_faces_assembled, 1);

    const Real area = std::sqrt(3.0) / 2.0;
    EXPECT_NEAR(mat.getMatrixEntry(1, 1), area / 6.0, 5e-11);
    EXPECT_NEAR(mat.getMatrixEntry(1, 4), -area / 6.0, 5e-11);

    const std::array<GlobalIndex, 4> far_field_cell_dofs{{8, 9, 10, 11}};
    for (GlobalIndex i = 0; i < 12; ++i) {
        for (const GlobalIndex j : far_field_cell_dofs) {
            SCOPED_TRACE(::testing::Message() << "row=" << i << ", far_field_col=" << j);
            EXPECT_DOUBLE_EQ(mat.getMatrixEntry(i, j), 0.0);
        }
    }
    for (const GlobalIndex i : far_field_cell_dofs) {
        for (GlobalIndex j = 0; j < 12; ++j) {
            SCOPED_TRACE(::testing::Message() << "far_field_row=" << i << ", col=" << j);
            EXPECT_DOUBLE_EQ(mat.getMatrixEntry(i, j), 0.0);
        }
    }
}

TEST(FormKernelDGTest, HighOrderMarkedCutAdjacentFacetIntegralSkipsFarFieldCell)
{
    ThreeTetraChainMeshAccess mesh;
    auto dof_map = createThreeTetraDG_P2_DofMap();
    spaces::H1Space space(ElementType::Tetra4, 2);

    constexpr int marker = 23;

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto form = (FormExpr::constant(1.0) * inner(jump(u), jump(v))).dS(marker);

    auto ir = compiler.compileBilinear(form);
    FormKernel kernel(std::move(ir));

    assembly::CutIntegrationContext cut_context;
    assembly::CutFacetSetHandle handle;
    handle.marker = marker;
    handle.name = "p2-single-cut-adjacent-facet";
    handle.facets = {0};
    assembly::CutFacetSetFacetMetadata metadata;
    metadata.facet = 0;
    metadata.first_cell = 0;
    metadata.second_cell = 1;
    metadata.stabilization_scale = 1.0;
    handle.facet_metadata.push_back(metadata);
    cut_context.addFacetSetHandle(std::move(handle));

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.setCutIntegrationContext(&cut_context);

    assembly::DenseMatrixView mat(30);
    mat.zero();

    const auto result = assembler.assembleInteriorFaces(mesh, space, space, kernel,
                                                        mat, nullptr, marker);
    EXPECT_EQ(result.interior_faces_assembled, 1);

    Real active_block_norm = 0.0;
    for (GlobalIndex i = 0; i < 20; ++i) {
        for (GlobalIndex j = 0; j < 20; ++j) {
            active_block_norm += std::abs(mat.getMatrixEntry(i, j));
        }
    }
    EXPECT_GT(active_block_norm, 0.0);

    for (GlobalIndex i = 0; i < 30; ++i) {
        for (GlobalIndex j = 20; j < 30; ++j) {
            SCOPED_TRACE(::testing::Message() << "row=" << i
                                              << ", far_field_col=" << j);
            EXPECT_DOUBLE_EQ(mat.getMatrixEntry(i, j), 0.0);
        }
    }
    for (GlobalIndex i = 20; i < 30; ++i) {
        for (GlobalIndex j = 0; j < 30; ++j) {
            SCOPED_TRACE(::testing::Message() << "far_field_row=" << i
                                              << ", col=" << j);
            EXPECT_DOUBLE_EQ(mat.getMatrixEntry(i, j), 0.0);
        }
    }
}

TEST(FormKernelDGTest, CutAdjacentGradientPenaltyScalesWithTraceHeight)
{
    constexpr int marker = 24;

    const auto assemble_entry = [](Real scale, int h_power) {
        TwoTetraSharedFaceMeshAccess mesh(scale);
        auto dof_map = createTwoTetraDG_DofMap();
        spaces::H1Space space(ElementType::Tetra4, 1);

        FormCompiler compiler;
        const auto u = FormExpr::trialFunction(space, "u");
        const auto v = FormExpr::testFunction(space, "v");
        auto h_factor = avg(hNormal());
        for (int power = 1; power < h_power; ++power) {
            h_factor = h_factor * avg(hNormal());
        }
        const auto form =
            (h_factor * inner(cutAdjacentFacetGradientJump(u),
                              cutAdjacentFacetGradientJump(v))).dS(marker);

        auto ir = compiler.compileBilinear(form);
        FormKernel kernel(std::move(ir));

        assembly::CutIntegrationContext cut_context;
        assembly::CutFacetSetHandle handle;
        handle.marker = marker;
        handle.name = "height-scaled-cut-adjacent-facet";
        handle.facets = {0};
        cut_context.addFacetSetHandle(std::move(handle));

        assembly::StandardAssembler assembler;
        assembler.setDofMap(dof_map);
        assembler.setCutIntegrationContext(&cut_context);

        assembly::DenseMatrixView mat(8);
        mat.zero();
        const auto result = assembler.assembleInteriorFaces(mesh, space, space, kernel,
                                                            mat, nullptr, marker);
        EXPECT_EQ(result.interior_faces_assembled, 1);
        return mat.getMatrixEntry(1, 1);
    };

    const Real velocity_base = assemble_entry(/*scale=*/1.0, /*h_power=*/1);
    const Real velocity_scaled = assemble_entry(/*scale=*/2.0, /*h_power=*/1);
    ASSERT_GT(std::abs(velocity_base), 1.0e-12);
    EXPECT_NEAR(velocity_scaled / velocity_base, 2.0, 1.0e-12);

    const Real pressure_base = assemble_entry(/*scale=*/1.0, /*h_power=*/3);
    const Real pressure_scaled = assemble_entry(/*scale=*/2.0, /*h_power=*/3);
    ASSERT_GT(std::abs(pressure_base), 1.0e-12);
    EXPECT_NEAR(pressure_scaled / pressure_base, 8.0, 1.0e-12);
}

TEST(FormKernelDGTest, CutStabilizationScaleUsesAdjacentCutCellMetadata)
{
    TwoTetraSharedFaceMeshAccess mesh;
    auto dof_map = createTwoTetraDG_DofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto form = (cutStabilizationScale() * inner(jump(u), jump(v))).dS();

    auto ir = compiler.compileBilinear(form);
    FormKernel kernel(std::move(ir));

    assembly::CutIntegrationContext cut_context;
    assembly::CutCellAssemblyMetadata metadata;
    metadata.parent_entity = 0;
    metadata.volume_fraction = 0.25;
    metadata.side = geometry::CutIntegrationSide::Negative;

    geometry::CutQuadratureRule rule;
    rule.kind = geometry::CutQuadratureKind::Volume;
    rule.side = geometry::CutIntegrationSide::Negative;
    rule.volume_fraction = metadata.volume_fraction;
    rule.provenance.parent_entity = metadata.parent_entity;
    cut_context.addVolumeRule(metadata, rule);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.setCutIntegrationContext(&cut_context);

    assembly::DenseMatrixView mat(8);
    mat.zero();

    const auto result = assembler.assembleInteriorFaces(mesh, space, space, kernel, mat, nullptr);
    EXPECT_EQ(result.interior_faces_assembled, 1);

    const Real eta = 4.0;
    const Real area = std::sqrt(3.0) / 2.0;
    const Real mdiag = eta * (area / 6.0);
    const Real moff = eta * (area / 12.0);

    auto expected_entry = [&](GlobalIndex i, GlobalIndex j) -> Real {
        auto faceIndex = [](GlobalIndex dof) -> int {
            if (dof == 1) return 0;
            if (dof == 2) return 1;
            if (dof == 3) return 2;
            if (dof == 4) return 0;
            if (dof == 5) return 1;
            if (dof == 6) return 2;
            return -1;
        };
        auto isMinus = [](GlobalIndex dof) -> bool { return dof < 4; };

        const int fi = faceIndex(i);
        const int fj = faceIndex(j);
        if (fi < 0 || fj < 0) return 0.0;

        const bool same_side = (isMinus(i) == isMinus(j));
        const Real base = (fi == fj) ? mdiag : moff;
        return same_side ? base : -base;
    };

    for (GlobalIndex i = 0; i < 8; ++i) {
        for (GlobalIndex j = 0; j < 8; ++j) {
            SCOPED_TRACE(::testing::Message() << "i=" << i << ", j=" << j);
            EXPECT_NEAR(mat.getMatrixEntry(i, j), expected_entry(i, j), 5e-11);
        }
    }
}

TEST(FormKernelDGTest, CutStabilizationScaleIsZeroAwayFromCutCells)
{
    TwoTetraSharedFaceMeshAccess mesh;
    auto dof_map = createTwoTetraDG_DofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto form = (cutStabilizationScale() * inner(jump(u), jump(v))).dS();

    auto ir = compiler.compileBilinear(form);
    FormKernel kernel(std::move(ir));

    assembly::CutIntegrationContext cut_context;
    assembly::CutCellAssemblyMetadata metadata;
    metadata.parent_entity = 99;
    metadata.volume_fraction = 0.25;
    metadata.side = geometry::CutIntegrationSide::Negative;

    geometry::CutQuadratureRule rule;
    rule.kind = geometry::CutQuadratureKind::Volume;
    rule.side = geometry::CutIntegrationSide::Negative;
    rule.volume_fraction = metadata.volume_fraction;
    rule.provenance.parent_entity = metadata.parent_entity;
    cut_context.addVolumeRule(metadata, rule);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.setCutIntegrationContext(&cut_context);

    assembly::DenseMatrixView mat(8);
    mat.zero();

    const auto result = assembler.assembleInteriorFaces(mesh, space, space, kernel, mat, nullptr);
    EXPECT_EQ(result.interior_faces_assembled, 1);

    for (GlobalIndex i = 0; i < 8; ++i) {
        for (GlobalIndex j = 0; j < 8; ++j) {
            SCOPED_TRACE(::testing::Message() << "i=" << i << ", j=" << j);
            EXPECT_DOUBLE_EQ(mat.getMatrixEntry(i, j), 0.0);
        }
    }
}

TEST(FormKernelDGTest, PenaltyJumpJumpHandlesPermutedPlusFaceOrdering)
{
    TwoTetraSharedFacePermutedPlusMeshAccess mesh;
    auto dof_map = createTwoTetraDG_DofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    const Real eta = 2.5;

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto form = (FormExpr::constant(eta) * inner(jump(u), jump(v))).dS();

    auto ir = compiler.compileBilinear(form);
    FormKernel kernel(std::move(ir));

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseMatrixView mat(8);
    mat.zero();

    const auto result = assembler.assembleInteriorFaces(mesh, space, space, kernel, mat, nullptr);
    EXPECT_EQ(result.interior_faces_assembled, 1);

    const Real area = std::sqrt(3.0) / 2.0;
    const Real mdiag = eta * (area / 6.0);
    const Real moff = eta * (area / 12.0);

    // Minus face ordering (cell 0 face 2): global nodes {1,2,3} in that order.
    // Plus face ordering  (cell 1 face 0): global nodes {2,3,1} in that order.
    // perm_plus_to_minus[j] gives which minus vertex index matches plus vertex j.
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
    auto isMinus = [](GlobalIndex dof) -> bool { return dof < 4; };

    auto expected_entry = [&](GlobalIndex i, GlobalIndex j) -> Real {
        const bool i_minus = isMinus(i);
        const bool j_minus = isMinus(j);

        const int fi = i_minus ? minusFaceIndex(i) : plusFaceIndex(i);
        const int fj = j_minus ? minusFaceIndex(j) : plusFaceIndex(j);
        if (fi < 0 || fj < 0) return 0.0;

        const bool same_side = (i_minus == j_minus);
        if (same_side) {
            const Real base = (fi == fj) ? mdiag : moff;
            return base;
        }

        // Cross terms: diag/off-diag determined by matching physical vertices.
        const bool i_is_minus_j_is_plus = i_minus && !j_minus;
        bool match = false;
        if (i_is_minus_j_is_plus) {
            match = (fi == perm_plus_to_minus[static_cast<std::size_t>(fj)]);
        } else {
            // i is plus, j is minus
            match = (perm_plus_to_minus[static_cast<std::size_t>(fi)] == fj);
        }

        const Real base = match ? mdiag : moff;
        return -base;
    };

    for (GlobalIndex i = 0; i < 8; ++i) {
        for (GlobalIndex j = 0; j < 8; ++j) {
            SCOPED_TRACE(::testing::Message() << "i=" << i << ", j=" << j);
            EXPECT_NEAR(mat.getMatrixEntry(i, j), expected_entry(i, j), 5e-11);
        }
    }
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp
