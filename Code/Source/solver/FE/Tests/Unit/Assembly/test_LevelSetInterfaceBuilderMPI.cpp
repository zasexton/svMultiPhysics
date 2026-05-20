#include "Interfaces/LevelSetInterfaceBuilder.h"
#include "LevelSet/LevelSetInterfaceLifecycle.h"
#include "Spaces/SpaceFactory.h"
#include "Systems/FESystem.h"
#include "Systems/SystemSetup.h"

#include <gtest/gtest.h>

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

using namespace svmp::FE;
using namespace svmp::FE::interfaces;

namespace {

CutInterfaceDomainRequest make_mpi_request()
{
    CutInterfaceDomainRequest request;
    request.source = LevelSetInterfaceSource::fromField(/*field_id=*/5,
                                                        /*layout_revision=*/1,
                                                        /*value_revision=*/1);
    request.interface_marker = 70;
    request.isovalue = 0.0;
    request.tolerance = 1.0e-12;
    return request;
}

class SingleQuad9MeshAccess final : public assembly::IMeshAccess {
public:
    [[nodiscard]] GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numVertices() const override { return 4; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 2; }
    [[nodiscard]] bool revisionTrackingAvailable() const override { return true; }
    [[nodiscard]] std::uint64_t geometryRevision() const override { return 7; }
    [[nodiscard]] std::uint64_t topologyRevision() const override { return 11; }
    [[nodiscard]] std::uint64_t ownershipRevision() const override { return 13; }
    [[nodiscard]] std::uint64_t fieldLayoutRevision() const override { return 17; }
    [[nodiscard]] bool isOwnedCell(GlobalIndex /*cell_id*/) const override { return true; }

    [[nodiscard]] ElementType getCellType(GlobalIndex /*cell_id*/) const override
    {
        return ElementType::Quad9;
    }

    void getCellNodes(GlobalIndex /*cell_id*/,
                      std::vector<GlobalIndex>& nodes) const override
    {
        nodes = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(
        GlobalIndex node_id) const override
    {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(
        GlobalIndex /*cell_id*/,
        std::vector<std::array<Real, 3>>& coords) const override
    {
        coords.assign(nodes_.begin(), nodes_.end());
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(
        GlobalIndex /*face_id*/,
        GlobalIndex /*cell_id*/) const override
    {
        return 0;
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex /*face_id*/) const override
    {
        return -1;
    }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex>
    getInteriorFaceCells(GlobalIndex /*face_id*/) const override
    {
        return {0, 0};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override
    {
        callback(0);
    }

    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override
    {
        callback(0);
    }

    void forEachBoundaryFace(
        int /*marker*/,
        std::function<void(GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

    void forEachInteriorFace(
        std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)>
            /*callback*/) const override
    {
    }

private:
    std::array<std::array<Real, 3>, 9> nodes_{{
        {{-1.0, -1.0, 0.0}},
        {{1.0, -1.0, 0.0}},
        {{1.0, 1.0, 0.0}},
        {{-1.0, 1.0, 0.0}},
        {{0.0, -1.0, 0.0}},
        {{1.0, 0.0, 0.0}},
        {{0.0, 1.0, 0.0}},
        {{-1.0, 0.0, 0.0}},
        {{0.0, 0.0, 0.0}},
    }};
};

[[nodiscard]] systems::SetupInputs make_single_quad_setup_inputs()
{
    dofs::MeshTopologyInfo topo;
    topo.n_cells = 1;
    topo.n_vertices = 4;
    topo.n_edges = 0;
    topo.n_faces = 0;
    topo.dim = 2;

    topo.cell2vertex_offsets = {0, 4};
    topo.cell2vertex_data = {0, 1, 2, 3};
    topo.vertex_gids = {0, 1, 2, 3};
    topo.cell_gids = {0};
    topo.cell_owner_ranks = {0};

    systems::SetupInputs inputs;
    inputs.topology_override = std::move(topo);
    return inputs;
}

void mix_hash(std::uint64_t& hash, std::uint64_t value) noexcept
{
    hash ^= value;
    hash *= 1099511628211ull;
}

void mix_real_hash(std::uint64_t& hash, Real value) noexcept
{
    std::uint64_t bits = 0u;
    static_assert(sizeof(value) <= sizeof(bits));
    std::memcpy(&bits, &value, sizeof(value));
    mix_hash(hash, bits);
}

void mix_string_hash(std::uint64_t& hash, const std::string& value) noexcept
{
    for (const char c : value) {
        mix_hash(hash, static_cast<std::uint64_t>(
                           static_cast<unsigned char>(c)));
    }
    mix_hash(hash, 0xffu);
}

[[nodiscard]] std::uint64_t domain_rule_hash(
    const LevelSetInterfaceDomain& domain) noexcept
{
    std::uint64_t hash = 1469598103934665603ull;
    const auto mix_rule = [&hash](const geometry::CutQuadratureRule& rule) {
        mix_hash(hash, static_cast<std::uint64_t>(rule.kind));
        mix_hash(hash, static_cast<std::uint64_t>(rule.side));
        mix_real_hash(hash, rule.measure);
        mix_string_hash(hash, rule.provenance.cut_topology_id);
        mix_string_hash(hash, rule.provenance.implicit_quadrature_backend);
        mix_hash(hash,
                 static_cast<std::uint64_t>(
                     rule.provenance.achieved_quadrature_order));
        mix_hash(hash, static_cast<std::uint64_t>(rule.points.size()));
        for (const auto& point : rule.points) {
            for (const Real x : point.point) {
                mix_real_hash(hash, x);
            }
            for (const Real n : point.normal) {
                mix_real_hash(hash, n);
            }
            mix_real_hash(hash, point.weight);
        }
    };
    for (const auto& rule : domain.interfaceQuadratureRules()) {
        mix_rule(rule);
    }
    for (const auto& rule : domain.volumeQuadratureRules()) {
        mix_rule(rule);
    }
    return hash;
}

void expect_all_ranks_same_real(Real local, Real tolerance, MPI_Comm comm)
{
    double local_value = static_cast<double>(local);
    double min_value = 0.0;
    double max_value = 0.0;
    MPI_Allreduce(&local_value, &min_value, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&local_value, &max_value, 1, MPI_DOUBLE, MPI_MAX, comm);
    EXPECT_NEAR(max_value, min_value, static_cast<double>(tolerance));
}

void expect_all_ranks_same_count(std::size_t local, MPI_Comm comm)
{
    const auto local_value = static_cast<unsigned long long>(local);
    unsigned long long min_value = 0;
    unsigned long long max_value = 0;
    MPI_Allreduce(&local_value,
                  &min_value,
                  1,
                  MPI_UNSIGNED_LONG_LONG,
                  MPI_MIN,
                  comm);
    MPI_Allreduce(&local_value,
                  &max_value,
                  1,
                  MPI_UNSIGNED_LONG_LONG,
                  MPI_MAX,
                  comm);
    EXPECT_EQ(max_value, min_value);
}

} // namespace

TEST(LevelSetInterfaceBuilderMPI, OwnedFragmentsReduceToConsistentGlobalMeasure)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    ASSERT_GE(size, 2);

    LevelSetInterfaceDomain domain(make_mpi_request());
    if (rank == 0) {
        appendLinearLevelSetCellCut2D(
            domain,
            LevelSetCellCutInput{.parent_cell = 0,
                                 .element_type = ElementType::Triangle3,
                                 .node_coordinates = {{{0.0, 0.0, 0.0}},
                                                      {{1.0, 0.0, 0.0}},
                                                      {{0.0, 1.0, 0.0}}},
                                 .level_set_values = {-0.25, 0.75, -0.25}});
    } else if (rank == 1) {
        appendLinearLevelSetCellCut2D(
            domain,
            LevelSetCellCutInput{.parent_cell = 1,
                                 .element_type = ElementType::Quad4,
                                 .node_coordinates = {{{0.0, 0.0, 0.0}},
                                                      {{1.0, 0.0, 0.0}},
                                                      {{1.0, 1.0, 0.0}},
                                                      {{0.0, 1.0, 0.0}}},
                                 .level_set_values = {-0.5, 0.5, 0.5, -0.5}});
    }

    const auto local_summary = domain.summary();
    const auto local_fragments =
        static_cast<unsigned long long>(local_summary.active_fragment_count);
    unsigned long long global_fragments = 0;
    MPI_Allreduce(&local_fragments,
                  &global_fragments,
                  1,
                  MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM,
                  MPI_COMM_WORLD);

    const double local_measure = static_cast<double>(local_summary.measure);
    double global_measure = 0.0;
    MPI_Allreduce(&local_measure,
                  &global_measure,
                  1,
                  MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD);

    const auto local_owned_cells =
        local_summary.active_fragment_count == 0u ? 0ull : 1ull;
    unsigned long long global_owned_cells = 0;
    MPI_Allreduce(&local_owned_cells,
                  &global_owned_cells,
                  1,
                  MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM,
                  MPI_COMM_WORLD);

    EXPECT_EQ(global_fragments, 2ull);
    EXPECT_EQ(global_owned_cells, 2ull);
    EXPECT_NEAR(global_measure, 1.75, 1.0e-14);
}

TEST(LevelSetInterfaceBuilderMPI, HighOrderImplicitRulesAreDeterministicAcrossRanks)
{
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    ASSERT_GE(size, 2);

    constexpr int interface_marker = 95;
    constexpr Real radius = 0.5;
    const auto mesh = std::make_shared<SingleQuad9MeshAccess>();
    auto scalar_space =
        spaces::Space(spaces::SpaceType::H1, mesh, /*order=*/2, /*components=*/1);

    systems::FESystem system(mesh);
    const auto phi = system.addField(systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, make_single_quad_setup_inputs()));

    std::vector<Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    const auto& field_dofs = system.fieldDofHandler(phi);
    const auto cell_dofs = field_dofs.getCellDofs(0);
    ASSERT_GE(cell_dofs.size(), 9u);
    const auto offset = system.fieldDofOffset(phi);
    for (std::size_t i = 0; i < 9u; ++i) {
        const auto x = mesh->getNodeCoordinates(static_cast<GlobalIndex>(i));
        solution[static_cast<std::size_t>(offset + cell_dofs[i])] =
            x[0] * x[0] + x[1] * x[1] - radius * radius;
    }

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = interface_marker;
    options.domain_id = "water-air";
    options.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
    options.implicit_cut_quadrature_backend =
        level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle;
    options.implicit_cut_max_subdivision_depth = 5;
    options.interface_quadrature_order = 2;
    options.volume_quadrature_order = 2;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto result = lifecycle.build(system, options, solution);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_GT(result.summary.active_fragment_count, 1u);
    expect_all_ranks_same_count(result.summary.active_fragment_count, MPI_COMM_WORLD);
    expect_all_ranks_same_count(result.summary.active_volume_region_count, MPI_COMM_WORLD);
    expect_all_ranks_same_count(result.summary.quadrature_point_count, MPI_COMM_WORLD);
    expect_all_ranks_same_real(result.summary.measure, 1.0e-14, MPI_COMM_WORLD);
    expect_all_ranks_same_real(result.summary.negative_volume_measure,
                               1.0e-14,
                               MPI_COMM_WORLD);
    expect_all_ranks_same_real(result.summary.positive_volume_measure,
                               1.0e-14,
                               MPI_COMM_WORLD);

    const auto local_hash =
        static_cast<unsigned long long>(domain_rule_hash(result.domain));
    unsigned long long min_hash = 0;
    unsigned long long max_hash = 0;
    MPI_Allreduce(&local_hash,
                  &min_hash,
                  1,
                  MPI_UNSIGNED_LONG_LONG,
                  MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local_hash,
                  &max_hash,
                  1,
                  MPI_UNSIGNED_LONG_LONG,
                  MPI_MAX,
                  MPI_COMM_WORLD);
    EXPECT_EQ(max_hash, min_hash);
}
