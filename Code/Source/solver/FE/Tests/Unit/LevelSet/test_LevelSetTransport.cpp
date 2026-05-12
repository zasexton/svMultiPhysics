#include "LevelSet/LevelSetTransport.h"

#include "Assembly/Assembler.h"
#include "Forms/FormExpr.h"
#include "Spaces/SpaceFactory.h"
#include "Systems/FESystem.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <functional>
#include <memory>
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
