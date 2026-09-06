#include "LevelSet/LevelSetCurvatureProjection.h"

#include "Dofs/EntityDofMap.h"
#include "Interfaces/FreeSurfaceGeometrySnapshot.h"
#include "Interfaces/LevelSetInterfaceBuilder.h"
#include "Mesh/Mesh.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Topology/CellShape.h"
#include "Spaces/H1Space.h"
#include "Systems/FESystem.h"

#include <gtest/gtest.h>

#if FE_HAS_MPI
#include <mpi.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace FE = svmp::FE;
namespace ls = FE::level_set;
namespace interfaces = FE::interfaces;

struct BindingFixture {
    std::shared_ptr<svmp::MeshBase> base;
    std::shared_ptr<svmp::Mesh> mesh;
    std::unique_ptr<FE::systems::FESystem> system;
    FE::FieldId field{FE::INVALID_FIELD_ID};
    std::array<FE::Real, 3> values{{0.25, -1.0, 1.0}};
    std::vector<FE::Real> solution;
    interfaces::LevelSetInterfaceSource source;
    interfaces::FreeSurfaceDiscreteFunctionalParameters functional;
    ls::LevelSetCurvatureProjectionOptions options;
    std::shared_ptr<const interfaces::FreeSurfaceGeometrySnapshot> snapshot;

    explicit BindingFixture(
        bool evaluator = false,
        bool permuted = false,
        FE::systems::FieldSourceKind source_kind =
            FE::systems::FieldSourceKind::Unknown,
        bool distributed = false,
        int field_order = 1)
    {
        int rank = 0;
        int size = 1;
#if FE_HAS_MPI
        const MPI_Comm communicator =
            distributed ? MPI_COMM_WORLD : MPI_COMM_SELF;
        MPI_Comm_rank(communicator, &rank);
        MPI_Comm_size(communicator, &size);
#else
        (void)distributed;
#endif
        base = std::make_shared<svmp::MeshBase>();
        svmp::CellShape shape{};
        shape.family = svmp::CellFamily::Triangle;
        shape.num_corners = 3;
        shape.order = 1;
        base->build_from_arrays(
            2, {0.0, 0.0, 1.0, 0.0, 0.0, 1.0},
            {0, 3}, {0, 1, 2}, {shape});
        base->finalize();
        const auto first = static_cast<svmp::gid_t>(3 * rank);
        base->set_vertex_gids(permuted
            ? std::vector<svmp::gid_t>{first + 2, first, first + 1}
            : std::vector<svmp::gid_t>{first, first + 1, first + 2});
        base->set_cell_gids({static_cast<svmp::gid_t>(100 + rank)});
#if FE_HAS_MPI
        mesh = svmp::create_mesh(base, svmp::MeshComm(communicator));
#else
        mesh = svmp::create_mesh(base);
#endif
        system = std::make_unique<FE::systems::FESystem>(mesh);
        const auto space = std::make_shared<FE::spaces::H1Space>(
            FE::ElementType::Triangle3, field_order);
        (void)system->addField(FE::systems::FieldSpec{
            .name = "unused", .space = space});
        field = system->addField(FE::systems::FieldSpec{
            .name = "phi", .space = space, .source_kind = source_kind});
        FE::systems::SetupOptions setup;
        setup.dof_options.global_numbering =
            FE::dofs::GlobalNumberingMode::GlobalIds;
        if (permuted && size == 1) {
            setup.dof_options.numbering =
                FE::dofs::DofNumberingStrategy::CuthillMcKee;
        }
        system->setup(setup);
        solution.assign(static_cast<std::size_t>(system->fieldMap().totalDofs()),
                        std::numeric_limits<FE::Real>::quiet_NaN());
        const auto& dofs = system->fieldDofHandler(field);
        const auto* map = dofs.getEntityDofMap();
        const auto offset = system->fieldDofOffset(field);
        if (source_kind != FE::systems::FieldSourceKind::PrescribedData) {
            for (FE::GlobalIndex vertex = 0; vertex < 3; ++vertex) {
                const auto dof = map->getVertexDofs(vertex).front();
                solution.at(static_cast<std::size_t>(offset + dof)) =
                    values[static_cast<std::size_t>(vertex)];
            }
        }
        source = evaluator
            ? interfaces::LevelSetInterfaceSource::fromEvaluator(
                  "borrowed-triangle", 19u, 23u)
            : interfaces::LevelSetInterfaceSource::fromField(
                  field, dofs.getDofStateRevision(), 23u);
        functional.surface_tension = 2.5;
        options.recovery_mode =
            ls::LevelSetCurvatureRecoveryMode::KinematicAreaGradient;
        options.kinematic_area_gradient_filter_coefficient = 0.0;

        const auto& access = system->meshAccess();
        interfaces::CutInterfaceDomainRequest request;
        request.source = source;
        request.generated_domain_id = "borrowed-triangle-domain";
        request.interface_marker = 31;
        request.mesh_geometry_revision = access.geometryRevision();
        request.mesh_topology_revision = access.topologyRevision();
        request.ownership_revision = access.ownershipRevision();
        request.implicit_geometry_mode = "LinearCorner";
        request.implicit_quadrature_backend = "LinearCorner";
        interfaces::LevelSetCellCutInput input;
        input.parent_cell = 0;
        input.element_type = FE::ElementType::Triangle3;
        input.node_coordinates = {
            {{0.0, 0.0, 0.0}}, {{1.0, 0.0, 0.0}}, {{0.0, 1.0, 0.0}}};
        input.level_set_values.assign(values.begin(), values.end());
        auto cut = interfaces::cutLinearLevelSetCell2D(request, input);
        if (!cut.supported || cut.fragments.size() != 1u ||
            cut.volume_regions.size() != 2u) {
            throw std::runtime_error("strict triangle cut setup failed");
        }
        interfaces::LevelSetInterfaceDomain domain(request);
        for (auto& fragment : cut.fragments) {
            fragment.parent_cell_global_id = access.getCellGlobalId(0);
            fragment.owner_rank = access.getCellOwnerRank(0);
            domain.addFragment(std::move(fragment));
        }
        for (auto& region : cut.volume_regions) {
            region.parent_cell_global_id = access.getCellGlobalId(0);
            region.owner_rank = access.getCellOwnerRank(0);
            domain.addVolumeRegion(std::move(region));
        }
        interfaces::FreeSurfaceGeometryScalarEvaluator scalar;
        scalar.value = [v = values](FE::GlobalIndex,
            const std::array<FE::Real, 3>& xi,
            const FE::geometry::CutQuadratureProvenance&) {
            return v[0] * (1.0 - xi[0] - xi[1]) +
                   v[1] * xi[0] + v[2] * xi[1];
        };
        scalar.reference_gradient = [v = values](FE::GlobalIndex,
            const std::array<FE::Real, 3>&,
            const FE::geometry::CutQuadratureProvenance&) {
            return std::array<FE::Real, 3>{{v[1] - v[0], v[2] - v[0], 0.0}};
        };
        interfaces::FreeSurfaceGeometrySnapshotPolicy policy;
        policy.require_complete_exterior_boundary_partition = false;
        interfaces::FreeSurfaceGeometryOwnershipCollective collective;
        collective.rank = rank;
        collective.size = size;
#if FE_HAS_MPI
        const auto gather = [communicator, size](
            std::span<const std::uint64_t> local) {
            const auto count = static_cast<int>(local.size());
            std::vector<int> counts(static_cast<std::size_t>(size));
            MPI_Allgather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT,
                          communicator);
            std::vector<int> offsets(static_cast<std::size_t>(size));
            int total = 0;
            for (int peer = 0; peer < size; ++peer) {
                offsets[static_cast<std::size_t>(peer)] = total;
                total += counts[static_cast<std::size_t>(peer)];
            }
            std::vector<std::uint64_t> result(static_cast<std::size_t>(total));
            MPI_Allgatherv(local.data(), count, MPI_UINT64_T, result.data(),
                           counts.data(), offsets.data(), MPI_UINT64_T,
                           communicator);
            return result;
        };
        collective.all_gather_owned_rule_identity_values = gather;
        collective.all_gather_revision_values = gather;
#endif
        snapshot = interfaces::buildFreeSurfaceGeometrySnapshot(
            std::move(domain), {}, {}, access, policy, scalar,
            request.generated_domain_id, collective);
        const auto state = interfaces::evaluateFreeSurfaceDiscreteFunctional(
            *snapshot, functional);
        if (!(state.owned_liquid_volume > 0.0) ||
            !(state.owned_liquid_gas_area > 0.0)) {
            throw std::runtime_error("strict triangle scalar setup failed");
        }
    }

    ls::LevelSetCurvatureProjectionResult project(
        std::vector<FE::Real>& curvature,
        ls::LevelSetCurvatureProjectionWorkspace* workspace = nullptr)
    {
        const ls::LevelSetAuthoritativeDerivativeBinding binding{
            *snapshot, functional, source};
        if (source.kind == interfaces::CutInterfaceSourceKind::Evaluator) {
            return ls::projectLevelSetMeanCurvatureToVertices(
                system->meshAccess(), values, binding, options, curvature,
                workspace);
        }
        return ls::projectLevelSetMeanCurvatureToVertices(
            *system, field, solution, binding, options, curvature, workspace);
    }
};

void expectUnavailable(const ls::LevelSetCurvatureProjectionResult& result,
                       const std::vector<FE::Real>& curvature,
                       const std::string& reason)
{
    EXPECT_FALSE(result.success);
    EXPECT_NE(result.diagnostic.find(reason), std::string::npos)
        << result.diagnostic;
    EXPECT_TRUE(result.kinematic_area_gradient_total_energy_derivative.empty());
    EXPECT_TRUE(result.kinematic_area_gradient_liquid_volume_derivative.empty());
    EXPECT_FALSE(result.kinematic_area_gradient_derivatives_global_dof_order);
    EXPECT_EQ(result.free_surface_snapshot_revision_key, 0u);
    EXPECT_EQ(result.source_value_revision, 0u);
    EXPECT_TRUE(curvature.empty());
}

TEST(LevelSetAuthoritativeDerivativeBinding, StrictInputsRemainUnavailable)
{
    for (const bool evaluator : {false, true}) {
        BindingFixture fixture(evaluator);
        for (const bool negative : {false, true}) {
            fixture.functional.liquid_side = negative
                ? FE::geometry::CutIntegrationSide::Negative
                : FE::geometry::CutIntegrationSide::Positive;
            fixture.options.kinematic_area_gradient_negative_liquid_side = negative;
            std::vector<FE::Real> curvature{99.0};
            ls::LevelSetCurvatureProjectionWorkspace workspace;
            expectUnavailable(fixture.project(curvature, &workspace), curvature,
                              "source_branch_unverified");
        }
    }
}

TEST(LevelSetAuthoritativeDerivativeBinding, RejectsDeclaredSourceAndEpochChanges)
{
    for (const bool evaluator : {false, true}) {
        BindingFixture fixture(evaluator);
        const auto original = fixture.source;
        for (int change = 0; change < 3; ++change) {
            fixture.source = original;
            const char* reason = "source_identity_mismatch";
            if (change == 0) {
                if (evaluator) fixture.source.evaluator_id = "different-evaluator";
                else ++fixture.source.field_id;
            } else if (change == 1) {
                ++fixture.source.layout_revision;
                reason = "source_layout_mismatch";
            } else {
                ++fixture.source.value_revision;
                reason = "source_value_mismatch";
            }
            std::vector<FE::Real> curvature{99.0};
            expectUnavailable(fixture.project(curvature), curvature, reason);
        }
    }
}

TEST(LevelSetAuthoritativeDerivativeBinding, RejectsMeshAndSolutionExtentChanges)
{
    BindingFixture fixture;
    std::vector<FE::Real> curvature{99.0};
    fixture.solution.pop_back();
    expectUnavailable(fixture.project(curvature), curvature,
                      "producer_solution_extent_mismatch");
    fixture.solution.push_back(1.0);
    fixture.solution.push_back(1.0);
    expectUnavailable(fixture.project(curvature), curvature,
                      "producer_solution_extent_mismatch");
    fixture.solution.pop_back();
    fixture.base->set_reference_geometry_dof_coords(0, {{0.125, 0.0, 0.0}});
    expectUnavailable(fixture.project(curvature), curvature,
                      "source_mesh_revision_mismatch");
}

TEST(LevelSetAuthoritativeDerivativeBinding, RejectsVisibleVertexExtentAndValues)
{
    BindingFixture fixture(true);
    const ls::LevelSetAuthoritativeDerivativeBinding binding{
        *fixture.snapshot, fixture.functional, fixture.source};
    std::vector<FE::Real> curvature{99.0};
    const auto short_values = std::span<const FE::Real>(fixture.values).first(2);
    expectUnavailable(ls::projectLevelSetMeanCurvatureToVertices(
        fixture.system->meshAccess(), short_values, binding, fixture.options,
        curvature), curvature, "producer_vertex_extent_mismatch");
    fixture.values[1] = std::numeric_limits<FE::Real>::infinity();
    expectUnavailable(fixture.project(curvature), curvature,
                      "producer_vertex_nonfinite: vertex 1");
}

TEST(LevelSetAuthoritativeDerivativeBinding, SamplesRequiredFieldThroughPermutedDofs)
{
    BindingFixture fixture(false, true);
    const auto* map = fixture.system->fieldDofHandler(fixture.field).getEntityDofMap();
    EXPECT_EQ(map->getVertexDofs(0).front(), 2);
    EXPECT_EQ(map->getVertexDofs(1).front(), 1);
    EXPECT_EQ(map->getVertexDofs(2).front(), 0);
    ASSERT_TRUE(std::isnan(fixture.solution.front()));
    std::vector<FE::Real> curvature{99.0};
    expectUnavailable(fixture.project(curvature), curvature,
                      "source_branch_unverified");
    const auto offset = fixture.system->fieldDofOffset(fixture.field);
    fixture.solution.at(static_cast<std::size_t>(offset)) =
        std::numeric_limits<FE::Real>::quiet_NaN();
    expectUnavailable(fixture.project(curvature), curvature,
                      "producer_vertex_nonfinite: vertex 2");
}

TEST(LevelSetAuthoritativeDerivativeBinding, RejectsPrescribedCoefficientSource)
{
    BindingFixture fixture(false, false,
        FE::systems::FieldSourceKind::PrescribedData);
    fixture.system->setPrescribedFieldCoefficients(fixture.field, fixture.values);
    std::vector<FE::Real> curvature{99.0};
    expectUnavailable(fixture.project(curvature), curvature,
                      "prescribed_field_source_unsupported");
}

TEST(LevelSetAuthoritativeDerivativeBinding, RejectsNonP1ScalarLayout)
{
    BindingFixture fixture(false, false,
        FE::systems::FieldSourceKind::Unknown, false, 2);
    std::vector<FE::Real> curvature{99.0};
    expectUnavailable(fixture.project(curvature), curvature,
                      "scalar_field_layout_unsupported");
}

TEST(LevelSetAuthoritativeDerivativeBinding, RequiresTheOverloadsDeclaredSourceKind)
{
    BindingFixture field_fixture;
    BindingFixture evaluator_fixture(true);
    const ls::LevelSetAuthoritativeDerivativeBinding field_binding{
        *field_fixture.snapshot, field_fixture.functional, field_fixture.source};
    const ls::LevelSetAuthoritativeDerivativeBinding evaluator_binding{
        *evaluator_fixture.snapshot, evaluator_fixture.functional,
        evaluator_fixture.source};
    std::vector<FE::Real> curvature{99.0};
    expectUnavailable(ls::projectLevelSetMeanCurvatureToVertices(
        field_fixture.system->meshAccess(), field_fixture.values, field_binding,
        field_fixture.options, curvature), curvature, "source_kind_mismatch");
    expectUnavailable(ls::projectLevelSetMeanCurvatureToVertices(
        *evaluator_fixture.system, evaluator_fixture.field,
        evaluator_fixture.solution, evaluator_binding,
        evaluator_fixture.options, curvature), curvature, "source_kind_mismatch");
}

TEST(LevelSetAuthoritativeDerivativeBinding, RejectsParameterDisagreement)
{
    BindingFixture fixture;
    const auto original = fixture.options;
    std::vector<FE::Real> curvature{99.0};
    fixture.options.isovalue = 0.125;
    expectUnavailable(fixture.project(curvature), curvature, "isovalue_mismatch");
    fixture.options = original;
    fixture.options.kinematic_area_gradient_negative_liquid_side = false;
    expectUnavailable(fixture.project(curvature), curvature, "liquid_side_mismatch");
    fixture.options = original;
    fixture.options.kinematic_area_gradient_young_walls = {{7, 1.0}};
    expectUnavailable(fixture.project(curvature), curvature, "young_wall_parameter_mismatch");
    fixture.functional.young_wall_coefficients = {{7, 1.125}};
    expectUnavailable(fixture.project(curvature), curvature, "young_wall_parameter_mismatch");
    fixture.options = original;
    fixture.functional.young_wall_coefficients.clear();
    fixture.functional.surface_tension = -1.0;
    expectUnavailable(fixture.project(curvature), curvature, "functional_parameter_invalid");
    fixture.functional.surface_tension = 2.5;
    fixture.functional.dynamic_contact_coefficients = {{7, 1.0}};
    expectUnavailable(fixture.project(curvature), curvature, "functional_terms_unsupported");
}

TEST(LevelSetAuthoritativeDerivativeBinding, LegacyStrictDerivativeOrderingIsPreserved)
{
    BindingFixture fixture(false, true);
    std::vector<FE::Real> vertex_curvature;
    const auto vertex_result = ls::projectLevelSetMeanCurvatureToVertices(
        fixture.system->meshAccess(), fixture.values, fixture.options,
        vertex_curvature);
    ASSERT_TRUE(vertex_result.success) << vertex_result.diagnostic;
    std::vector<FE::Real> field_curvature;
    const auto field_result = ls::projectLevelSetMeanCurvatureToVertices(
        *fixture.system, fixture.field, fixture.values,
        std::span<const ls::LevelSetCurvatureProjectionSample>{},
        fixture.options, field_curvature);
    ASSERT_TRUE(field_result.success) << field_result.diagnostic;
    ASSERT_TRUE(field_result.kinematic_area_gradient_derivatives_global_dof_order);
    ASSERT_EQ(field_result.kinematic_area_gradient_total_energy_derivative.size(), 3u);
    ASSERT_EQ(field_result.kinematic_area_gradient_liquid_volume_derivative.size(), 3u);
    const std::array<std::size_t, 3> dofs{{2u, 1u, 0u}};
    for (std::size_t vertex = 0; vertex < 3u; ++vertex) {
        EXPECT_DOUBLE_EQ(field_result.kinematic_area_gradient_total_energy_derivative[dofs[vertex]],
                         vertex_result.kinematic_area_gradient_total_energy_derivative[vertex]);
        EXPECT_DOUBLE_EQ(field_result.kinematic_area_gradient_liquid_volume_derivative[dofs[vertex]],
                         vertex_result.kinematic_area_gradient_liquid_volume_derivative[vertex]);
    }
}

#if FE_HAS_MPI
TEST(LevelSetAuthoritativeDerivativeBindingMPI, AgreesOnRankLocalPreflightAndSamplingFailure)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    ASSERT_EQ(size, 2);
    BindingFixture fixture(false, true, FE::systems::FieldSourceKind::Unknown, true);
    std::vector<FE::Real> curvature{99.0};
    if (rank == 1) ++fixture.source.value_revision;
    auto result = fixture.project(curvature);
    expectUnavailable(result, curvature, "source_value_mismatch");
    EXPECT_NE(result.diagnostic.find("rank 1"), std::string::npos);
    if (rank == 1) --fixture.source.value_revision;
    if (rank == 1) {
        const auto dof = fixture.system->fieldDofHandler(fixture.field)
            .getEntityDofMap()->getVertexDofs(1).front();
        fixture.solution.at(static_cast<std::size_t>(
            fixture.system->fieldDofOffset(fixture.field) + dof)) =
                std::numeric_limits<FE::Real>::quiet_NaN();
    }
    result = fixture.project(curvature);
    expectUnavailable(result, curvature, "producer_vertex_nonfinite: vertex 1");
    EXPECT_NE(result.diagnostic.find("rank 1"), std::string::npos);
}
#endif
} // namespace

int main(int argc, char** argv)
{
#if FE_HAS_MPI
    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::string rank_output;
    if (size > 1) {
        for (int argument = 1; argument < argc; ++argument) {
            const std::string value = argv[argument];
            if (value.rfind("--gtest_output=xml:", 0) == 0) {
                rank_output = value + ".rank" + std::to_string(rank);
                argv[argument] = rank_output.data();
                break;
            }
        }
    }
#endif
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
#if FE_HAS_MPI
    MPI_Finalize();
#endif
    return result;
}
