#include "LevelSet/LevelSetTransport.h"

#include "Assembly/Assembler.h"
#include "Dofs/EntityDofMap.h"
#include "Forms/FormExpr.h"
#include "Spaces/SpaceFactory.h"
#include "Systems/FESystem.h"
#include "Systems/SystemSetup.h"
#include "Systems/TimeIntegrator.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

namespace FE = svmp::FE;
namespace level_set = svmp::FE::level_set;

using FE::forms::FormExpr;
using FE::forms::FormExprNode;
using FE::forms::FormExprType;

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

[[nodiscard]] std::shared_ptr<FE::spaces::FunctionSpace> scalarSpace(
    const std::shared_ptr<const FE::assembly::IMeshAccess>& mesh)
{
    return FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);
}

[[nodiscard]] std::shared_ptr<FE::spaces::FunctionSpace> vectorSpace(
    const std::shared_ptr<const FE::assembly::IMeshAccess>& mesh)
{
    return FE::spaces::VectorSpace(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/3);
}

bool containsExprType(const FormExprNode* node, FormExprType target)
{
    if (node == nullptr) {
        return false;
    }
    if (node->type() == target) {
        return true;
    }
    for (const auto* child : node->children()) {
        if (containsExprType(child, target)) {
            return true;
        }
    }
    return false;
}

bool formulationRecordsContain(const FE::systems::FESystem& system,
                               FormExprType target)
{
    for (const auto& record : system.formulationRecords()) {
        if (containsExprType(record.residual_expr.get(), target)) {
            return true;
        }
        for (const auto& [block, expr] : record.block_residual_exprs) {
            (void)block;
            if (containsExprType(expr.get(), target)) {
                return true;
            }
        }
    }
    return false;
}

void addScalarAndVelocityFields(FE::systems::FESystem& system,
                                const std::shared_ptr<const FE::spaces::FunctionSpace>& scalar_space,
                                const std::shared_ptr<const FE::spaces::FunctionSpace>& velocity_space,
                                FE::systems::FieldSourceKind velocity_source =
                                    FE::systems::FieldSourceKind::PrescribedData)
{
    system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    system.addField(FE::systems::FieldSpec{
        .name = "advecting_velocity",
        .space = velocity_space,
        .components = velocity_space->value_dimension(),
        .source_kind = velocity_source,
    });
}

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

std::vector<FE::Real> constantVectorTetraCoefficients(FE::Real x,
                                                      FE::Real y,
                                                      FE::Real z)
{
    std::vector<FE::Real> coefficients(12u, 0.0);
    for (std::size_t node = 0; node < 4u; ++node) {
        coefficients[node] = x;
        coefficients[4u + node] = y;
        coefficients[8u + node] = z;
    }
    return coefficients;
}

void setFieldComponentValue(std::vector<FE::Real>& solution,
                            const FE::systems::FESystem& system,
                            FE::FieldId field,
                            FE::GlobalIndex vertex,
                            int component,
                            FE::Real value)
{
    const auto& handler = system.fieldDofHandler(field);
    const auto offset = system.fieldDofOffset(field);
    const auto* entity_map = handler.getEntityDofMap();
    if (entity_map == nullptr) {
        throw std::runtime_error("setFieldComponentValue: field has no entity DOF map");
    }
    const auto dofs = entity_map->getVertexDofs(vertex);
    if (component < 0 || static_cast<std::size_t>(component) >= dofs.size()) {
        throw std::runtime_error("setFieldComponentValue: component is out of range");
    }
    const auto index = static_cast<std::size_t>(
        dofs[static_cast<std::size_t>(component)] + offset);
    if (index >= solution.size()) {
        throw std::runtime_error("setFieldComponentValue: DOF index is out of range");
    }
    solution[index] = value;
}

void expectOperatorJacobianMatchesCentralFD(FE::systems::FESystem& system,
                                            const FE::systems::SystemStateView& base_state,
                                            FE::Real eps,
                                            FE::Real rtol,
                                            FE::Real atol)
{
    const auto n = system.dofHandler().getNumDofs();
    ASSERT_GT(n, 0);

    const std::vector<FE::Real> base_u(base_state.u.begin(), base_state.u.end());
    ASSERT_EQ(static_cast<FE::GlobalIndex>(base_u.size()), n);

    FE::assembly::DenseMatrixView jacobian(n);
    {
        FE::systems::AssemblyRequest request;
        request.op = "level_set";
        request.want_matrix = true;
        const auto result = system.assemble(request, base_state, &jacobian, nullptr);
        ASSERT_TRUE(result.success) << result.error_message;
    }

    for (FE::GlobalIndex column = 0; column < n; ++column) {
        std::vector<FE::Real> u_plus = base_u;
        std::vector<FE::Real> u_minus = base_u;
        u_plus[static_cast<std::size_t>(column)] += eps;
        u_minus[static_cast<std::size_t>(column)] -= eps;

        FE::systems::SystemStateView state_plus = base_state;
        FE::systems::SystemStateView state_minus = base_state;
        state_plus.u = std::span<const FE::Real>(u_plus);
        state_minus.u = std::span<const FE::Real>(u_minus);

        FE::assembly::DenseVectorView r_plus(n);
        FE::assembly::DenseVectorView r_minus(n);
        {
            FE::systems::AssemblyRequest request;
            request.op = "level_set";
            request.want_vector = true;
            const auto result = system.assemble(request, state_plus, nullptr, &r_plus);
            ASSERT_TRUE(result.success) << result.error_message;
        }
        {
            FE::systems::AssemblyRequest request;
            request.op = "level_set";
            request.want_vector = true;
            const auto result = system.assemble(request, state_minus, nullptr, &r_minus);
            ASSERT_TRUE(result.success) << result.error_message;
        }

        for (FE::GlobalIndex row = 0; row < n; ++row) {
            const FE::Real finite_difference = (r_plus[row] - r_minus[row]) / (2.0 * eps);
            const FE::Real assembled = jacobian(row, column);
            const FE::Real tolerance =
                atol + rtol * std::max<FE::Real>(1.0, std::abs(finite_difference));
            EXPECT_NEAR(assembled, finite_difference, tolerance)
                << "Mismatch at (row=" << row << ", column=" << column << ")";
        }
    }
}

} // namespace

TEST(LevelSetTransport, ValidatesFieldOptions)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);
    auto velocity_space = vectorSpace(mesh);

    FE::systems::FESystem scalar_system(mesh);
    scalar_system.addField(FE::systems::FieldSpec{
        .name = "level_set",
        .space = phi_space,
        .components = 1,
    });
    scalar_system.addField(FE::systems::FieldSpec{
        .name = "Velocity",
        .space = velocity_space,
        .components = velocity_space->value_dimension(),
    });
    EXPECT_NO_THROW(
        (void)level_set::installLevelSetTransport(scalar_system, phi_space, {}));

    FE::systems::FESystem vector_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(vector_system, velocity_space, {}),
        std::invalid_argument);

    level_set::LevelSetTransportOptions options{};
    options.level_set.field_name.clear();
    FE::systems::FESystem empty_name_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(empty_name_system, phi_space, options),
        std::invalid_argument);

    options.level_set.field_name = "phi";
    options.velocity.field_name.clear();
    FE::systems::FESystem empty_velocity_name_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(
            empty_velocity_name_system,
            phi_space,
            options),
        std::invalid_argument);
}

TEST(LevelSetTransport, AutoRegistersConfiguredFields)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);
    auto velocity_space = vectorSpace(mesh);

    FE::systems::FESystem system(mesh);

    level_set::LevelSetTransportOptions options{};
    options.level_set.field_name = "phi";
    options.level_set.source = level_set::LevelSetFieldSource::PrescribedData;
    options.velocity.field_name = "advecting_velocity";
    options.velocity.source = level_set::LevelSetVelocitySource::PrescribedData;
    options.velocity.auto_register_field = true;
    options.velocity.space = velocity_space;

    const auto kernels = level_set::installLevelSetTransport(system, phi_space, options);

    const auto phi = system.findFieldByName("phi");
    const auto velocity = system.findFieldByName("advecting_velocity");
    ASSERT_NE(phi, FE::INVALID_FIELD_ID);
    ASSERT_NE(velocity, FE::INVALID_FIELD_ID);
    EXPECT_EQ(system.fieldRecord(phi).source_kind, FE::systems::FieldSourceKind::Unknown);
    EXPECT_EQ(system.fieldRecord(velocity).source_kind,
              FE::systems::FieldSourceKind::PrescribedData);
    EXPECT_TRUE(system.hasOperator("level_set"));
    EXPECT_FALSE(kernels.residual.empty());
}

TEST(LevelSetTransport, InstallsOnConfiguredOperatorTag)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);
    auto velocity_space = vectorSpace(mesh);

    FE::systems::FESystem system(mesh);

    level_set::LevelSetTransportOptions options{};
    options.operator_tag = "equations";
    options.level_set.field_name = "phi";
    options.level_set.source = level_set::LevelSetFieldSource::PrescribedData;
    options.velocity.field_name = "Velocity";
    options.velocity.source = level_set::LevelSetVelocitySource::CoupledField;
    options.velocity.auto_register_field = true;
    options.velocity.space = velocity_space;

    const auto kernels = level_set::installLevelSetTransport(system, phi_space, options);

    EXPECT_TRUE(system.hasOperator("equations"));
    EXPECT_FALSE(system.hasOperator("level_set"));
    EXPECT_FALSE(kernels.residual.empty());
}

TEST(LevelSetTransport, AutoRegistersCoupledVelocityAsUnknownWhenRequested)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);
    auto velocity_space = vectorSpace(mesh);

    FE::systems::FESystem system(mesh);

    level_set::LevelSetTransportOptions options{};
    options.level_set.field_name = "phi";
    options.level_set.source = level_set::LevelSetFieldSource::PrescribedData;
    options.velocity.field_name = "Velocity";
    options.velocity.source = level_set::LevelSetVelocitySource::CoupledField;
    options.velocity.auto_register_field = true;
    options.velocity.space = velocity_space;

    const auto kernels = level_set::installLevelSetTransport(system, phi_space, options);

    const auto phi = system.findFieldByName("phi");
    const auto velocity = system.findFieldByName("Velocity");
    ASSERT_NE(phi, FE::INVALID_FIELD_ID);
    ASSERT_NE(velocity, FE::INVALID_FIELD_ID);
    EXPECT_EQ(system.fieldRecord(phi).source_kind, FE::systems::FieldSourceKind::Unknown);
    EXPECT_EQ(system.fieldRecord(velocity).source_kind,
              FE::systems::FieldSourceKind::Unknown);
    EXPECT_TRUE(system.hasOperator("level_set"));
    EXPECT_FALSE(kernels.residual.empty());
}

TEST(LevelSetTransport, InstallsResidualFormStructure)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);
    auto velocity_space = vectorSpace(mesh);

    FE::systems::FESystem system(mesh);
    addScalarAndVelocityFields(system, phi_space, velocity_space);

    level_set::LevelSetTransportOptions options{};
    options.level_set.field_name = "phi";
    options.level_set.auto_register_field = false;
    options.velocity.field_name = "advecting_velocity";
    options.velocity.source = level_set::LevelSetVelocitySource::PrescribedData;

    (void)level_set::installLevelSetTransport(system, phi_space, options);

    EXPECT_TRUE(system.hasOperator("level_set"));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CellIntegral));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::TimeDerivative));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::Gradient));
    EXPECT_FALSE(formulationRecordsContain(system, FormExprType::CellDiameter));
}

TEST(LevelSetTransport, SUPGAddsCellDiameterStabilization)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);
    auto velocity_space = vectorSpace(mesh);

    FE::systems::FESystem system(mesh);
    addScalarAndVelocityFields(system, phi_space, velocity_space);

    level_set::LevelSetTransportOptions options{};
    options.level_set.field_name = "phi";
    options.level_set.auto_register_field = false;
    options.velocity.field_name = "advecting_velocity";
    options.velocity.source = level_set::LevelSetVelocitySource::PrescribedData;
    options.supg.enabled = true;

    (void)level_set::installLevelSetTransport(system, phi_space, options);

    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CellDiameter));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::TimeDerivative));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::Gradient));
}

TEST(LevelSetTransport, ValidatesSUPGOptions)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);

    level_set::LevelSetTransportOptions options{};
    options.supg.enabled = true;

    options.supg.tau_scale = 0.0;
    FE::systems::FESystem tau_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(tau_system, phi_space, options),
        std::invalid_argument);

    options.supg.tau_scale = 0.5;
    options.supg.velocity_epsilon = 0.0;
    FE::systems::FESystem epsilon_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(epsilon_system, phi_space, options),
        std::invalid_argument);
}

TEST(LevelSetTransport, ValidatesReinitializationOptions)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);
    auto velocity_space = vectorSpace(mesh);

    level_set::LevelSetTransportOptions options{};
    options.reinitialization.enabled = true;

    options.reinitialization.cadence_steps = 0;
    FE::systems::FESystem cadence_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(cadence_system, phi_space, options),
        std::invalid_argument);

    options.reinitialization.cadence_steps = 1;
    options.reinitialization.max_iterations = 0;
    FE::systems::FESystem iterations_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(iterations_system, phi_space, options),
        std::invalid_argument);

    options.reinitialization.max_iterations = 10;
    options.reinitialization.pseudo_time_step_scale = 0.0;
    FE::systems::FESystem pseudo_time_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(pseudo_time_system, phi_space, options),
        std::invalid_argument);

    options.reinitialization.pseudo_time_step_scale = 0.3;
    options.reinitialization.interface_band_width = 0.0;
    FE::systems::FESystem band_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(band_system, phi_space, options),
        std::invalid_argument);

    options.reinitialization.interface_band_width = 3.0;
    options.reinitialization.signed_distance_tolerance = 0.0;
    FE::systems::FESystem tolerance_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(tolerance_system, phi_space, options),
        std::invalid_argument);

    options.reinitialization.signed_distance_tolerance = 1.0e-6;
    FE::systems::FESystem valid_system(mesh);
    valid_system.addField(FE::systems::FieldSpec{
        .name = "Velocity",
        .space = velocity_space,
        .components = velocity_space->value_dimension(),
    });
    EXPECT_NO_THROW(
        (void)level_set::installLevelSetTransport(valid_system, phi_space, options));
}

TEST(LevelSetTransport, ValidatesVolumeCorrectionOptions)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);
    auto velocity_space = vectorSpace(mesh);

    level_set::LevelSetTransportOptions options{};
    options.volume_correction.enabled = true;

    options.volume_correction.cadence_steps = 0;
    FE::systems::FESystem cadence_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(cadence_system, phi_space, options),
        std::invalid_argument);

    options.volume_correction.cadence_steps = 1;
    options.volume_correction.volume_tolerance = 0.0;
    FE::systems::FESystem tolerance_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(tolerance_system, phi_space, options),
        std::invalid_argument);

    options.volume_correction.volume_tolerance = 1.0e-10;
    options.volume_correction.max_iterations = 0;
    FE::systems::FESystem iterations_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(iterations_system, phi_space, options),
        std::invalid_argument);

    options.volume_correction.max_iterations = 50;
    options.volume_correction.use_initial_negative_volume_as_target = false;
    options.volume_correction.target_negative_volume = -1.0;
    FE::systems::FESystem target_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(target_system, phi_space, options),
        std::invalid_argument);

    options.volume_correction.target_negative_volume = 0.125;
    FE::systems::FESystem valid_system(mesh);
    valid_system.addField(FE::systems::FieldSpec{
        .name = "Velocity",
        .space = velocity_space,
        .components = velocity_space->value_dimension(),
    });
    EXPECT_NO_THROW(
        (void)level_set::installLevelSetTransport(valid_system, phi_space, options));
}

TEST(LevelSetTransport, InflowBoundaryAddsUpwindPenalty)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);
    auto velocity_space = vectorSpace(mesh);

    FE::systems::FESystem system(mesh);
    addScalarAndVelocityFields(system, phi_space, velocity_space);

    level_set::LevelSetTransportOptions options{};
    options.level_set.field_name = "phi";
    options.level_set.auto_register_field = false;
    options.velocity.field_name = "advecting_velocity";
    options.velocity.source = level_set::LevelSetVelocitySource::PrescribedData;
    options.boundaries.inflow.push_back(level_set::LevelSetInflowBoundary{
        .boundary_marker = 4,
        .value = FE::Real{1.25},
        .penalty_scale = 2.0,
    });

    (void)level_set::installLevelSetTransport(system, phi_space, options);

    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::BoundaryIntegral));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::Normal));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::AbsoluteValue));
}

TEST(LevelSetTransport, OutflowBoundaryIsNatural)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);
    auto velocity_space = vectorSpace(mesh);

    FE::systems::FESystem system(mesh);
    addScalarAndVelocityFields(system, phi_space, velocity_space);

    level_set::LevelSetTransportOptions options{};
    options.level_set.field_name = "phi";
    options.level_set.auto_register_field = false;
    options.velocity.field_name = "advecting_velocity";
    options.velocity.source = level_set::LevelSetVelocitySource::PrescribedData;
    options.boundaries.outflow.push_back(
        level_set::LevelSetOutflowBoundary{.boundary_marker = 5});

    (void)level_set::installLevelSetTransport(system, phi_space, options);

    EXPECT_FALSE(formulationRecordsContain(system, FormExprType::BoundaryIntegral));
}

TEST(LevelSetTransport, ValidatesBoundaryOptions)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);

    level_set::LevelSetTransportOptions options{};
    options.boundaries.inflow.push_back(level_set::LevelSetInflowBoundary{});
    FE::systems::FESystem missing_marker_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(
            missing_marker_system,
            phi_space,
            options),
        std::invalid_argument);

    options.boundaries.inflow.clear();
    options.boundaries.inflow.push_back(level_set::LevelSetInflowBoundary{
        .boundary_marker = 4,
        .penalty_scale = 0.0,
    });
    FE::systems::FESystem penalty_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(penalty_system, phi_space, options),
        std::invalid_argument);

    options.boundaries.inflow.clear();
    options.boundaries.inflow.push_back(
        level_set::LevelSetInflowBoundary{.boundary_marker = 4});
    options.boundaries.outflow.push_back(
        level_set::LevelSetOutflowBoundary{.boundary_marker = 4});
    FE::systems::FESystem duplicate_marker_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(
            duplicate_marker_system,
            phi_space,
            options),
        std::invalid_argument);
}

TEST(LevelSetTransport, PrescribedVelocityJacobianMatchesFiniteDifference)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);
    auto velocity_space = vectorSpace(mesh);

    FE::systems::FESystem system(mesh);
    addScalarAndVelocityFields(system, phi_space, velocity_space);

    level_set::LevelSetTransportOptions options{};
    options.level_set.field_name = "phi";
    options.level_set.auto_register_field = false;
    options.velocity.field_name = "advecting_velocity";
    options.velocity.source = level_set::LevelSetVelocitySource::PrescribedData;

    (void)level_set::installLevelSetTransport(system, phi_space, options);
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    const auto phi = system.findFieldByName("phi");
    const auto velocity = system.findFieldByName("advecting_velocity");
    ASSERT_NE(phi, FE::INVALID_FIELD_ID);
    ASSERT_NE(velocity, FE::INVALID_FIELD_ID);
    system.setPrescribedFieldCoefficients(
        velocity,
        constantVectorTetraCoefficients(0.70, -0.15, 0.25));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    std::vector<FE::Real> previous_solution = solution;
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        const auto x = static_cast<FE::Real>(vertex);
        setFieldComponentValue(
            solution,
            system,
            phi,
            vertex,
            0,
            FE::Real(0.20) + FE::Real(0.035) * x);
        setFieldComponentValue(
            previous_solution,
            system,
            phi,
            vertex,
            0,
            FE::Real(0.18) + FE::Real(0.025) * x);
    }

    FE::systems::SystemStateView state;
    state.dt = 0.1;
    state.u = std::span<const FE::Real>(solution);
    state.u_prev = std::span<const FE::Real>(previous_solution);
    const FE::systems::BackwardDifferenceIntegrator integrator;
    const auto time_context = integrator.buildContext(/*max_time_derivative_order=*/1, state);
    state.time_integration = &time_context;

    expectOperatorJacobianMatchesCentralFD(
        system,
        state,
        1.0e-6,
        2.0e-5,
        1.0e-8);
}

TEST(LevelSetTransport, CoupledVelocityJacobianMatchesFiniteDifference)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);
    auto velocity_space = vectorSpace(mesh);

    FE::systems::FESystem system(mesh);
    addScalarAndVelocityFields(
        system,
        phi_space,
        velocity_space,
        FE::systems::FieldSourceKind::Unknown);

    level_set::LevelSetTransportOptions options{};
    options.level_set.field_name = "phi";
    options.level_set.auto_register_field = false;
    options.velocity.field_name = "advecting_velocity";
    options.velocity.source = level_set::LevelSetVelocitySource::CoupledField;

    (void)level_set::installLevelSetTransport(system, phi_space, options);
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    const auto phi = system.findFieldByName("phi");
    const auto velocity = system.findFieldByName("advecting_velocity");
    ASSERT_NE(phi, FE::INVALID_FIELD_ID);
    ASSERT_NE(velocity, FE::INVALID_FIELD_ID);

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    std::vector<FE::Real> previous_solution = solution;
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        const auto x = static_cast<FE::Real>(vertex);
        setFieldComponentValue(
            solution,
            system,
            phi,
            vertex,
            0,
            FE::Real(0.15) + FE::Real(0.04) * x);
        setFieldComponentValue(
            previous_solution,
            system,
            phi,
            vertex,
            0,
            FE::Real(0.12) + FE::Real(0.03) * x);
        setFieldComponentValue(
            solution,
            system,
            velocity,
            vertex,
            0,
            FE::Real(0.40) + FE::Real(0.015) * x);
        setFieldComponentValue(
            solution,
            system,
            velocity,
            vertex,
            1,
            FE::Real(-0.20) + FE::Real(0.010) * x);
        setFieldComponentValue(
            solution,
            system,
            velocity,
            vertex,
            2,
            FE::Real(0.30) - FE::Real(0.005) * x);
    }

    FE::systems::SystemStateView state;
    state.dt = 0.1;
    state.u = std::span<const FE::Real>(solution);
    state.u_prev = std::span<const FE::Real>(previous_solution);
    const FE::systems::BackwardDifferenceIntegrator integrator;
    const auto time_context = integrator.buildContext(/*max_time_derivative_order=*/1, state);
    state.time_integration = &time_context;

    expectOperatorJacobianMatchesCentralFD(
        system,
        state,
        1.0e-6,
        5.0e-5,
        1.0e-8);
}
