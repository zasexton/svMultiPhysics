#include "LevelSet/LevelSetRestart.h"

#include "Assembly/Assembler.h"
#include "Dofs/DofHandler.h"
#include "Dofs/EntityDofMap.h"
#include "Spaces/SpaceFactory.h"
#include "Systems/FESystem.h"
#include "Systems/SystemSetup.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace FE = svmp::FE;
namespace level_set = svmp::FE::level_set;

class SingleTetraMeshAccess final : public FE::assembly::IMeshAccess {
public:
    SingleTetraMeshAccess()
    {
        nodes_ = {
            std::array<FE::Real, 3>{0.0, 0.0, 0.0},
            std::array<FE::Real, 3>{1.0, 0.0, 0.0},
            std::array<FE::Real, 3>{0.0, 1.0, 0.0},
            std::array<FE::Real, 3>{0.0, 0.0, 1.0},
        };
        cell_ = {0, 1, 2, 3};
    }

    [[nodiscard]] FE::GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] FE::GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }
    [[nodiscard]] bool revisionTrackingAvailable() const override { return true; }
    [[nodiscard]] std::uint64_t geometryRevision() const override { return 7; }
    [[nodiscard]] std::uint64_t topologyRevision() const override { return 11; }
    [[nodiscard]] std::uint64_t ownershipRevision() const override { return 13; }
    [[nodiscard]] bool isOwnedCell(FE::GlobalIndex /*cell_id*/) const override { return true; }

    [[nodiscard]] FE::ElementType getCellType(FE::GlobalIndex /*cell_id*/) const override
    {
        return FE::ElementType::Tetra4;
    }

    void getCellNodes(FE::GlobalIndex /*cell_id*/,
                      std::vector<FE::GlobalIndex>& nodes) const override
    {
        nodes.assign(cell_.begin(), cell_.end());
    }

    [[nodiscard]] std::array<FE::Real, 3> getNodeCoordinates(
        FE::GlobalIndex node_id) const override
    {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(
        FE::GlobalIndex /*cell_id*/,
        std::vector<std::array<FE::Real, 3>>& coords) const override
    {
        coords = nodes_;
    }

    [[nodiscard]] FE::LocalIndex getLocalFaceIndex(
        FE::GlobalIndex /*face_id*/,
        FE::GlobalIndex /*cell_id*/) const override
    {
        return 0;
    }

    [[nodiscard]] int getBoundaryFaceMarker(FE::GlobalIndex /*face_id*/) const override
    {
        return -1;
    }

    [[nodiscard]] std::pair<FE::GlobalIndex, FE::GlobalIndex>
    getInteriorFaceCells(FE::GlobalIndex /*face_id*/) const override
    {
        return {0, 0};
    }

    void forEachCell(std::function<void(FE::GlobalIndex)> callback) const override
    {
        callback(0);
    }

    void forEachOwnedCell(std::function<void(FE::GlobalIndex)> callback) const override
    {
        callback(0);
    }

    void forEachBoundaryFace(
        int /*marker*/,
        std::function<void(FE::GlobalIndex, FE::GlobalIndex)> /*callback*/) const override
    {
    }

    void forEachInteriorFace(
        std::function<void(FE::GlobalIndex, FE::GlobalIndex, FE::GlobalIndex)>
            /*callback*/) const override
    {
    }

private:
    std::vector<std::array<FE::Real, 3>> nodes_{};
    std::array<FE::GlobalIndex, 4> cell_{};
};

[[nodiscard]] FE::systems::SetupInputs makeSingleTetraSetupInputs()
{
    FE::dofs::MeshTopologyInfo topo;
    topo.n_cells = 1;
    topo.n_vertices = 4;
    topo.n_edges = 0;
    topo.n_faces = 0;
    topo.dim = 3;

    topo.cell2vertex_offsets = {0, 4};
    topo.cell2vertex_data = {0, 1, 2, 3};
    topo.vertex_gids = {0, 1, 2, 3};
    topo.cell_gids = {0};
    topo.cell_owner_ranks = {0};

    FE::systems::SetupInputs inputs;
    inputs.topology_override = std::move(topo);
    return inputs;
}

void setFieldValue(std::vector<FE::Real>& solution,
                   const FE::systems::FESystem& system,
                   FE::FieldId field,
                   FE::GlobalIndex vertex,
                   FE::Real value)
{
    const auto& field_dofs = system.fieldDofHandler(field);
    const auto* entity_map = field_dofs.getEntityDofMap();
    if (entity_map == nullptr) {
        throw std::runtime_error("setFieldValue: field has no entity DOF map");
    }
    const auto dofs = entity_map->getVertexDofs(vertex);
    if (dofs.size() != 1u) {
        throw std::runtime_error("setFieldValue: expected one vertex DOF");
    }
    const auto offset = system.fieldDofOffset(field);
    solution[static_cast<std::size_t>(offset + dofs.front())] = value;
}

} // namespace

TEST(LevelSetRestart, CapturesFieldAndGeneratedInterfaceRecords)
{
    auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        const auto x = mesh->getNodeCoordinates(vertex);
        setFieldValue(solution, system, phi, vertex,
                      x[0] + x[1] + x[2] - FE::Real{0.5});
    }

    level_set::LevelSetTransportOptions transport_options{};
    transport_options.level_set.field_name = "phi";
    transport_options.level_set.source = level_set::LevelSetFieldSource::PrescribedData;
    transport_options.level_set.auto_register_field = false;

    const auto field_record =
        level_set::captureLevelSetFieldRestartRecord(system, transport_options, 6u);
    EXPECT_EQ(field_record.field_name, "phi");
    EXPECT_EQ(field_record.field_id, phi);
    EXPECT_EQ(field_record.source, level_set::LevelSetFieldSource::PrescribedData);
    EXPECT_FALSE(field_record.auto_register_field);
    EXPECT_EQ(field_record.components, 1);
    EXPECT_GT(field_record.dof_count, 0);
    EXPECT_EQ(field_record.value_revision, 6u);

    level_set::LevelSetGeneratedInterfaceOptions interface_options{};
    interface_options.level_set_field_name = "phi";
    interface_options.domain_id = "water-air";
    interface_options.isovalue = 0.0;
    interface_options.tolerance = 1.0e-12;
    interface_options.quadrature_order = 1;
    interface_options.interface_quadrature_order = 0;
    interface_options.volume_quadrature_order = 1;
    interface_options.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::LinearCorner;
    interface_options.implicit_cut_quadrature_backend =
        level_set::ImplicitCutQuadratureBackend::LinearCorner;
    interface_options.implicit_cut_fallback_policy =
        level_set::ImplicitCutFallbackPolicy::LinearCorner;
    interface_options.geometry_tangent_policy =
        level_set::GeometryTangentPolicy::RefreshedFrozenQuadrature;
    interface_options.implicit_cut_root_tolerance = 2.0e-10;
    interface_options.implicit_cut_root_coordinate_tolerance = 3.0e-12;
    interface_options.implicit_cut_root_max_iterations = 52;
    interface_options.implicit_cut_max_subdivision_depth = 18;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto built = lifecycle.build(system, interface_options, solution);
    ASSERT_TRUE(built.success) << built.diagnostic;

    const auto interface_record =
        level_set::captureLevelSetGeneratedInterfaceRestartRecord(
            system, interface_options, built);
    EXPECT_EQ(interface_record.level_set_field_name, "phi");
    EXPECT_EQ(interface_record.level_set_field_id, phi);
    EXPECT_EQ(interface_record.domain_id, "water-air");
    EXPECT_EQ(interface_record.interface_marker, built.interface_marker);
    EXPECT_EQ(interface_record.interface_quadrature_order, 0);
    EXPECT_EQ(interface_record.volume_quadrature_order, 1);
    EXPECT_EQ(interface_record.geometry_mode,
              level_set::GeneratedInterfaceGeometryMode::LinearCorner);
    EXPECT_EQ(interface_record.implicit_cut_quadrature_backend,
              level_set::ImplicitCutQuadratureBackend::LinearCorner);
    EXPECT_EQ(interface_record.implicit_cut_fallback_policy,
              level_set::ImplicitCutFallbackPolicy::LinearCorner);
    EXPECT_EQ(interface_record.geometry_tangent_policy,
              level_set::GeometryTangentPolicy::RefreshedFrozenQuadrature);
    EXPECT_DOUBLE_EQ(interface_record.implicit_cut_root_tolerance, 2.0e-10);
    EXPECT_DOUBLE_EQ(interface_record.implicit_cut_root_coordinate_tolerance,
                     3.0e-12);
    EXPECT_EQ(interface_record.implicit_cut_root_max_iterations, 52);
    EXPECT_EQ(interface_record.implicit_cut_max_subdivision_depth, 18);
    EXPECT_EQ(interface_record.value_revision, built.value_revision);
    EXPECT_EQ(interface_record.mesh_geometry_revision, 7u);
    EXPECT_EQ(interface_record.mesh_topology_revision, 11u);
    EXPECT_EQ(interface_record.ownership_revision, 13u);
    EXPECT_EQ(interface_record.summary.active_fragment_count,
              built.summary.active_fragment_count);

    std::string diagnostic;
    EXPECT_TRUE(level_set::levelSetGeneratedInterfaceRestartRecordMatches(
        system, interface_record, &diagnostic)) << diagnostic;
    EXPECT_TRUE(diagnostic.empty());

    auto mismatch = interface_record;
    ++mismatch.mesh_geometry_revision;
    EXPECT_FALSE(level_set::levelSetGeneratedInterfaceRestartRecordMatches(
        system, mismatch, &diagnostic));
    EXPECT_NE(diagnostic.find("geometry revision"), std::string::npos);

    auto linear_differentiated_tangent = interface_record;
    linear_differentiated_tangent.geometry_tangent_policy =
        level_set::GeometryTangentPolicy::DifferentiatedQuadrature;
    EXPECT_TRUE(level_set::levelSetGeneratedInterfaceRestartRecordMatches(
        system, linear_differentiated_tangent, &diagnostic)) << diagnostic;

    auto unsupported_tangent = linear_differentiated_tangent;
    unsupported_tangent.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
    unsupported_tangent.implicit_cut_quadrature_backend =
        level_set::ImplicitCutQuadratureBackend::HighOrderSubcell;
    unsupported_tangent.geometry_tangent_policy =
        level_set::GeometryTangentPolicy::DifferentiatedQuadrature;
    EXPECT_FALSE(level_set::levelSetGeneratedInterfaceRestartRecordMatches(
        system, unsupported_tangent, &diagnostic));
    EXPECT_NE(diagnostic.find("differentiated geometry tangent policy"),
              std::string::npos);

    const auto restored_options =
        level_set::optionsFromLevelSetGeneratedInterfaceRestartRecord(interface_record);
    EXPECT_EQ(restored_options.requested_interface_marker,
              built.interface_marker);
    EXPECT_EQ(restored_options.domain_id, interface_options.domain_id);
    EXPECT_EQ(restored_options.interface_quadrature_order, 0);
    EXPECT_EQ(restored_options.volume_quadrature_order, 1);
    EXPECT_EQ(restored_options.geometry_mode,
              interface_options.geometry_mode);
    EXPECT_EQ(restored_options.implicit_cut_quadrature_backend,
              interface_options.implicit_cut_quadrature_backend);
    EXPECT_EQ(restored_options.implicit_cut_fallback_policy,
              interface_options.implicit_cut_fallback_policy);
    EXPECT_EQ(restored_options.geometry_tangent_policy,
              interface_options.geometry_tangent_policy);
    EXPECT_DOUBLE_EQ(restored_options.implicit_cut_root_tolerance,
                     interface_options.implicit_cut_root_tolerance);
    EXPECT_DOUBLE_EQ(restored_options.implicit_cut_root_coordinate_tolerance,
                     interface_options.implicit_cut_root_coordinate_tolerance);
    EXPECT_EQ(restored_options.implicit_cut_root_max_iterations,
              interface_options.implicit_cut_root_max_iterations);
    EXPECT_EQ(restored_options.implicit_cut_max_subdivision_depth,
              interface_options.implicit_cut_max_subdivision_depth);

    level_set::LevelSetGeneratedInterfaceLifecycle restored_lifecycle;
    restored_lifecycle.restoreValueRevision(interface_record.value_revision);
    const auto rebuilt = restored_lifecycle.build(
        system, restored_options, solution);
    ASSERT_TRUE(rebuilt.success) << rebuilt.diagnostic;
    EXPECT_EQ(rebuilt.interface_marker, built.interface_marker);
    EXPECT_EQ(rebuilt.value_revision, interface_record.value_revision + 1u);
}
