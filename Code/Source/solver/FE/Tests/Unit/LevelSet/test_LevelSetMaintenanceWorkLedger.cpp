/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "LevelSet/Maintenance/LevelSetMaintenanceWorkLedger.h"

#include "Core/LogRecord.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace level_set = svmp::FE::level_set;

namespace {

using level_set::LevelSetAuthoritativeFunctionalValue;
using level_set::LevelSetMaintenanceDeclaredStage;
using level_set::LevelSetMaintenanceWorkLedger;
using level_set::LevelSetMaintenanceWorkStatus;
using level_set::LevelSetMaintenanceWorkTransaction;
using Substage = level_set::MaintenanceSubstage;

/// One interface with surface energy 3, Young-wall energy -0.25 and the
/// volume-constraint potential that closes the total.
LevelSetAuthoritativeFunctionalValue functionalValue(
    double total_potential,
    std::uint64_t snapshot_revision,
    int interface_marker = 407)
{
    LevelSetAuthoritativeFunctionalValue value;
    value.interface_marker = interface_marker;
    value.snapshot_revision = snapshot_revision;
    value.mesh_topology_revision = 19u;
    value.cut_topology_revision = 23u;
    value.liquid_volume = 1.25;
    value.liquid_gas_area = 2.5;
    value.wetted_wall_area = 0.75;
    value.contact_measure = 0.5;
    value.surface_energy = 3.0;
    value.young_wall_energy = -0.25;
    value.volume_constraint_potential = total_potential - 2.75;
    value.total_potential = total_potential;
    return value;
}

/// As functionalValue, and the owner of the active-volume energies.
LevelSetAuthoritativeFunctionalValue ownerValue(
    double total_potential,
    std::uint64_t snapshot_revision,
    double kinetic_energy,
    double gravitational_energy,
    int interface_marker = 407)
{
    auto value =
        functionalValue(total_potential, snapshot_revision, interface_marker);
    value.kinetic_energy = kinetic_energy;
    value.gravitational_energy = gravitational_energy;
    value.gravitational_potential_power = -0.5;
    value.modeled_stored_energy = kinetic_energy + gravitational_energy +
                                  value.surface_energy +
                                  value.young_wall_energy;
    return value;
}

LevelSetMaintenanceWorkTransaction transaction(
    std::uint64_t transaction_id,
    LevelSetMaintenanceDeclaredStage stage =
        LevelSetMaintenanceDeclaredStage::AcceptedEndpointPostStep)
{
    LevelSetMaintenanceWorkTransaction result;
    result.transaction_id = transaction_id;
    result.step = 8u;
    result.attempt = 2u;
    result.time = 0.4;
    result.dt = 0.05;
    result.declared_stage = stage;
    result.extension_map_revision = 313u;
    return result;
}

} // namespace

TEST(LevelSetMaintenanceWorkLedger, CommitPublishesTrialRowsWithTheirWork)
{
    LevelSetMaintenanceWorkLedger ledger;
    const auto before = ledger.mark();
    ledger.beginTransaction(transaction(101u));
    EXPECT_TRUE(ledger.transactionActive());
    ledger.stageRow(Substage::Reinitialization, 1001u, 1002u,
                    {functionalValue(2.0, 501u)},
                    {functionalValue(2.125, 502u)});
    ASSERT_EQ(ledger.trialRows().size(), 1u);
    EXPECT_EQ(ledger.trialRows().front().status,
              LevelSetMaintenanceWorkStatus::Trial);
    EXPECT_DOUBLE_EQ(ledger.trialRows().front().numerical_work, 0.125);
    EXPECT_DOUBLE_EQ(ledger.trialRows().front().accepted_numerical_work, 0.0);

    ledger.commitTransaction();
    EXPECT_FALSE(ledger.transactionActive());
    EXPECT_TRUE(ledger.trialRows().empty());
    ASSERT_EQ(ledger.acceptedRows().size(), 1u);
    const auto& row = ledger.acceptedRows().front();
    EXPECT_EQ(row.status, LevelSetMaintenanceWorkStatus::Accepted);
    EXPECT_EQ(row.step, 8u);
    EXPECT_EQ(row.attempt, 2u);
    EXPECT_EQ(row.extension_map_revision_before, std::optional<std::uint64_t>{313u});
    EXPECT_EQ(row.extension_map_revision_after, std::optional<std::uint64_t>{313u});
    EXPECT_DOUBLE_EQ(row.accepted_numerical_work, 0.125);
    ASSERT_EQ(ledger.acceptedAttempts().size(), 1u);
    EXPECT_EQ(ledger.acceptedAttempts().front().row_count, 1u);
    EXPECT_DOUBLE_EQ(ledger.acceptedAttempts().front().accepted_numerical_work,
                     0.125);

    const auto after = ledger.mark();
    EXPECT_EQ(after.accepted_rows, before.accepted_rows + 1u);
    EXPECT_EQ(after.accepted_attempts, before.accepted_attempts + 1u);
    EXPECT_EQ(after.rejected_rows, before.rejected_rows);
    EXPECT_EQ(after.rejected_attempts, before.rejected_attempts);
}

TEST(LevelSetMaintenanceWorkLedger, RejectionPublishesZeroAcceptedWork)
{
    LevelSetMaintenanceWorkLedger ledger;
    ledger.beginTransaction(transaction(102u));
    ledger.stageRow(Substage::GeometryReconciliation, 2001u, 2002u,
                    {functionalValue(3.0, 601u)},
                    {functionalValue(3.25, 602u)});
    ledger.rejectTransaction();
    EXPECT_TRUE(ledger.acceptedRows().empty());
    ASSERT_EQ(ledger.rejectedRows().size(), 1u);
    EXPECT_EQ(ledger.rejectedRows().front().status,
              LevelSetMaintenanceWorkStatus::Rejected);
    EXPECT_DOUBLE_EQ(ledger.rejectedRows().front().numerical_work, 0.25);
    EXPECT_DOUBLE_EQ(ledger.rejectedRows().front().accepted_numerical_work, 0.0);
    ASSERT_EQ(ledger.rejectedAttempts().size(), 1u);
    EXPECT_DOUBLE_EQ(ledger.rejectedAttempts().front().numerical_work, 0.25);
    EXPECT_DOUBLE_EQ(ledger.rejectedAttempts().front().accepted_numerical_work,
                     0.0);

    // Identifiers increase; a second active transaction is refused.
    EXPECT_THROW(ledger.beginTransaction(transaction(102u)),
                 std::invalid_argument);
    ledger.beginTransaction(transaction(103u));
    EXPECT_THROW(ledger.beginTransaction(transaction(104u)), std::logic_error);
    ledger.rejectTransaction();
    EXPECT_THROW(ledger.commitTransaction(), std::logic_error);
}

TEST(LevelSetMaintenanceWorkLedger, RowsMustChainAndValuesMustCloseTheirIdentities)
{
    LevelSetMaintenanceWorkLedger ledger;
    ledger.beginTransaction(transaction(105u));
    ledger.stageRow(Substage::Reinitialization, 3001u, 3002u,
                    {functionalValue(3.0, 701u)},
                    {functionalValue(3.5, 702u)});
    // A row that does not start where the previous one ended.
    EXPECT_THROW(ledger.stageRow(Substage::GlobalCorrection, 3005u, 3006u,
                                 {functionalValue(3.5, 702u)},
                                 {functionalValue(3.25, 703u)}),
                 std::invalid_argument);
    EXPECT_THROW(ledger.stageRow(Substage::GlobalCorrection, 3002u, 3003u,
                                 {functionalValue(3.4, 702u)},
                                 {functionalValue(3.25, 703u)}),
                 std::invalid_argument);
    EXPECT_EQ(ledger.trialRows().size(), 1u);

    auto negative_measure = functionalValue(3.25, 703u);
    negative_measure.liquid_volume = -0.25;
    EXPECT_THROW(ledger.stageRow(Substage::GlobalCorrection, 3002u, 3003u,
                                 {functionalValue(3.5, 702u)},
                                 {negative_measure}),
                 std::invalid_argument);
    auto open_potential = functionalValue(3.25, 703u);
    open_potential.total_potential += 0.125;
    EXPECT_THROW(ledger.stageRow(Substage::GlobalCorrection, 3002u, 3003u,
                                 {functionalValue(3.5, 702u)},
                                 {open_potential}),
                 std::invalid_argument);
    auto no_cut_topology = functionalValue(3.25, 703u);
    no_cut_topology.cut_topology_revision = 0u;
    EXPECT_THROW(ledger.stageRow(Substage::GlobalCorrection, 3002u, 3003u,
                                 {functionalValue(3.5, 702u)},
                                 {no_cut_topology}),
                 std::invalid_argument);
    EXPECT_THROW(ledger.stageRow(static_cast<Substage>(255u), 3002u, 3003u,
                                 {functionalValue(3.5, 702u)},
                                 {functionalValue(3.25, 703u)}),
                 std::invalid_argument);
    EXPECT_EQ(ledger.trialRows().size(), 1u);

    ledger.stageRow(Substage::GlobalCorrection, 3002u, 3003u,
                    {functionalValue(3.5, 702u)},
                    {functionalValue(3.25, 703u)});
    ledger.commitTransaction();
    ASSERT_EQ(ledger.acceptedRows().size(), 2u);
    EXPECT_EQ(ledger.acceptedRows()[0].after, ledger.acceptedRows()[1].before);
    EXPECT_DOUBLE_EQ(ledger.acceptedAttempts().front().numerical_work, 0.25);
}

TEST(LevelSetMaintenanceWorkLedger, ModeledEnergyCountsOneActiveVolumeOwnerAcrossInterfaces)
{
    LevelSetMaintenanceWorkLedger ledger;
    ledger.beginTransaction(transaction(106u));
    // The owner's kinetic and gravitational energies plus the surface and
    // wall energies of both interfaces: 1 + 2 + 2 * 2.75 before, 1.5 + 1.8 +
    // 2 * 2.75 after.
    ledger.stageRow(Substage::Reinitialization, 4001u, 4002u,
                    {ownerValue(4.0, 801u, 1.0, 2.0, 407),
                     functionalValue(5.0, 811u, 409)},
                    {functionalValue(5.0, 812u, 409),
                     ownerValue(4.0, 802u, 1.5, 1.8, 407)});
    ASSERT_EQ(ledger.trialRows().size(), 1u);
    const auto& row = ledger.trialRows().front();
    // Values are kept in marker order.
    ASSERT_EQ(row.after.size(), 2u);
    EXPECT_EQ(row.after[0].interface_marker, 407);
    EXPECT_EQ(row.after[1].interface_marker, 409);
    EXPECT_DOUBLE_EQ(row.numerical_work, 0.0);
    ASSERT_TRUE(row.modeled_energy_numerical_work.has_value());
    EXPECT_NEAR(*row.modeled_energy_numerical_work, 0.3, 1.0e-15);

    // Two owners in one functional set.
    EXPECT_THROW(ledger.stageRow(Substage::GlobalCorrection, 4002u, 4003u,
                                 {ownerValue(4.0, 802u, 1.5, 1.8, 407),
                                  functionalValue(5.0, 812u, 409)},
                                 {ownerValue(4.0, 803u, 1.5, 1.8, 407),
                                  ownerValue(5.0, 813u, 0.5, 0.5, 409)}),
                 std::invalid_argument);
    ledger.rejectTransaction();
}

TEST(LevelSetMaintenanceWorkLedger, AccountKeepsPhysicalTransportApartFromNumericalMaintenance)
{
    LevelSetMaintenanceWorkLedger ledger;
    ledger.beginTransaction(transaction(
        107u, LevelSetMaintenanceDeclaredStage::ProspectiveAcceptedEndpoint));
    const auto start = ownerValue(4.0, 901u, 1.0, 2.0);
    const auto transported = ownerValue(4.0, 902u, 1.4, 1.8);
    const auto corrected = ownerValue(4.5, 903u, 1.4, 1.8);
    ledger.stageRow(Substage::Transport, 5001u, 5002u, {start}, {transported});
    ledger.stageRow(Substage::GlobalCorrection, 5002u, 5003u, {transported},
                    {corrected});
    ledger.commitTransaction();

    const auto account = level_set::aggregateLevelSetMaintenanceAcceptedStepEnergy(
        ledger.acceptedAttempts(), ledger.acceptedRows());
    ASSERT_TRUE(account.has_value());
    EXPECT_EQ(account->transaction_count, 1u);
    EXPECT_EQ(account->row_count, 2u);
    const auto& breakdown = account->modeled_energy_breakdown;
    EXPECT_EQ(breakdown.transport.row_count, 1u);
    EXPECT_EQ(breakdown.global_correction.row_count, 1u);
    EXPECT_EQ(breakdown.numerical_maintenance_total.row_count, 1u);
    ASSERT_TRUE(breakdown.transport.accepted_modeled_energy_change.has_value());
    EXPECT_NEAR(*breakdown.transport.accepted_modeled_energy_change, 0.2,
                1.0e-15);
    ASSERT_TRUE(
        breakdown.numerical_maintenance_total.accepted_modeled_energy_change
            .has_value());
    EXPECT_DOUBLE_EQ(
        *breakdown.numerical_maintenance_total.accepted_modeled_energy_change,
        0.0);
    ASSERT_TRUE(account->maintenance_start.has_value());
    ASSERT_TRUE(account->post_transport.has_value());
    ASSERT_TRUE(account->maintenance_end.has_value());
    EXPECT_EQ(account->post_transport->algebraic_state_revision, 5002u);
    EXPECT_EQ(account->maintenance_end->algebraic_state_revision, 5003u);
    EXPECT_DOUBLE_EQ(account->post_transport->liquid_gas_surface_energy, 3.0);
    EXPECT_DOUBLE_EQ(account->post_transport->solid_liquid_wall_energy, -0.25);
    ASSERT_TRUE(account->physical_transport_endpoint_residual.has_value());
    EXPECT_NEAR(*account->physical_transport_endpoint_residual, 0.0, 1.0e-15);

    // No published transaction, no account; rows without an attempt fail.
    EXPECT_FALSE(level_set::aggregateLevelSetMaintenanceAcceptedStepEnergy({}, {})
                     .has_value());
    EXPECT_THROW((void)level_set::aggregateLevelSetMaintenanceAcceptedStepEnergy(
                     {}, ledger.acceptedRows()),
                 std::invalid_argument);
    // A rejected attempt is not part of an accepted-step account.
    LevelSetMaintenanceWorkLedger rejected;
    rejected.beginTransaction(transaction(108u));
    rejected.rejectTransaction();
    EXPECT_THROW((void)level_set::aggregateLevelSetMaintenanceAcceptedStepEnergy(
                     rejected.rejectedAttempts(), {}),
                 std::invalid_argument);
}

TEST(LevelSetMaintenanceWorkLedger, StageObserverRowEvaluatesTheAfterStateAndAdvances)
{
    LevelSetMaintenanceWorkLedger ledger;
    ledger.beginTransaction(transaction(109u));
    std::vector<std::string> calls;
    level_set::LevelSetMaintenanceWorkRowSource source;
    source.functional_values = [&](std::span<const svmp::FE::Real> state) {
        calls.push_back("functionals");
        return std::vector<LevelSetAuthoritativeFunctionalValue>{
            functionalValue(state.front(),
                            static_cast<std::uint64_t>(state.back()))};
    };
    source.algebraic_state_revision = [&](std::span<const svmp::FE::Real> state) {
        calls.push_back("revision");
        return static_cast<std::uint64_t>(state.back()) + 1000u;
    };
    source.extension_map_revision = [&]() -> std::optional<std::uint64_t> {
        calls.push_back("extension");
        return 314u;
    };
    std::vector<LevelSetAuthoritativeFunctionalValue> current{
        functionalValue(2.0, 11u)};
    const std::vector<svmp::FE::Real> before{2.0, 11.0};
    const std::vector<svmp::FE::Real> after{2.5, 12.0};
    level_set::stageLevelSetMaintenanceWorkRow(
        ledger, source, current, Substage::GeometryReconciliation, before,
        after);
    EXPECT_EQ(calls, (std::vector<std::string>{"functionals", "revision",
                                               "revision", "extension"}));
    ASSERT_EQ(current.size(), 1u);
    EXPECT_EQ(current.front(), functionalValue(2.5, 12u));
    ASSERT_EQ(ledger.trialRows().size(), 1u);
    const auto& row = ledger.trialRows().front();
    EXPECT_EQ(row.substage, Substage::GeometryReconciliation);
    EXPECT_EQ(row.algebraic_state_revision_before, 1011u);
    EXPECT_EQ(row.algebraic_state_revision_after, 1012u);
    EXPECT_EQ(row.extension_map_revision_before, std::optional<std::uint64_t>{313u});
    EXPECT_EQ(row.extension_map_revision_after, std::optional<std::uint64_t>{314u});
    EXPECT_DOUBLE_EQ(row.numerical_work, 0.5);

    // A rejected row leaves the current values in place.
    std::vector<LevelSetAuthoritativeFunctionalValue> stale{
        functionalValue(9.0, 99u)};
    EXPECT_THROW(level_set::stageLevelSetMaintenanceWorkRow(
                     ledger, source, stale, Substage::GlobalCorrection, after,
                     before),
                 std::invalid_argument);
    EXPECT_EQ(stale.front(), functionalValue(9.0, 99u));

    level_set::LevelSetMaintenanceWorkRowSource incomplete;
    EXPECT_THROW(level_set::stageLevelSetMaintenanceWorkRow(
                     ledger, incomplete, current, Substage::GlobalCorrection,
                     after, after),
                 std::invalid_argument);
    ledger.rejectTransaction();
}

TEST(LevelSetMaintenanceWorkLedger, PublicationLogsNameOnlyTheNewOutcome)
{
    LevelSetMaintenanceWorkLedger ledger;
    ledger.beginTransaction(transaction(110u));
    ledger.stageRow(Substage::Reinitialization, 6001u, 6002u,
                    {functionalValue(2.0, 31u)}, {functionalValue(2.25, 32u)});
    ledger.commitTransaction();
    const auto mark = ledger.mark();
    ledger.beginTransaction(transaction(111u));
    ledger.stageRow(Substage::Reinitialization, 6002u, 6003u,
                    {functionalValue(2.25, 32u)}, {functionalValue(2.5, 33u)});
    ledger.commitTransaction();

    testing::internal::CaptureStdout();
    level_set::logLevelSetMaintenanceWorkPublishedSince(
        ledger, mark, LevelSetMaintenanceWorkStatus::Accepted);
    level_set::logLevelSetMaintenanceWorkPublishedSince(
        ledger, mark, LevelSetMaintenanceWorkStatus::Rejected);
    const auto output = testing::internal::GetCapturedStdout();
    if (output.empty()) {
        GTEST_SKIP() << "This rank does not log.";
    }
    EXPECT_EQ(output.find("transaction_id=110"), std::string::npos);
    // The log records of map entries R2.143 and R2.144.
    const auto attempt = output.find(
        "LevelSetMaintenanceWorkLedger: work attempt "
        "diagnostic=level_set_maintenance_work_attempt transaction_id=111 "
        "status=accepted");
    const auto row = output.find(
        "LevelSetMaintenanceWorkLedger: work row "
        "diagnostic=level_set_maintenance_work transaction_id=111 "
        "status=accepted substage=reinitialization");
    ASSERT_NE(attempt, std::string::npos);
    ASSERT_NE(row, std::string::npos);
    EXPECT_LT(attempt, row);
    EXPECT_THROW(level_set::logLevelSetMaintenanceWorkPublishedSince(
                     ledger, mark, LevelSetMaintenanceWorkStatus::Trial),
                 std::invalid_argument);
}

// A row of two interfaces is one record and one child line per interface
// (map entry R2.144): every key once per line, the row's identifying keys
// on each child.  No gate deck has two interfaces, so the text is checked
// here.
TEST(LevelSetMaintenanceWorkLedger, RowRecordHasOneChildLinePerInterface)
{
    level_set::LevelSetMaintenanceWorkRow row;
    row.transaction_id = 121u;
    row.status = LevelSetMaintenanceWorkStatus::Accepted;
    row.substage = Substage::Reinitialization;
    row.step = 8u;
    row.attempt = 2u;
    row.time = 0.4;
    row.dt = 0.05;
    row.algebraic_state_revision_before = 7001u;
    row.algebraic_state_revision_after = 7002u;
    row.snapshot_set_revision_before = 5001u;
    row.snapshot_set_revision_after = 5002u;
    row.mesh_topology_set_revision_before = 19u;
    row.mesh_topology_set_revision_after = 19u;
    row.cut_topology_set_revision_before = 6001u;
    row.cut_topology_set_revision_after = 6002u;
    row.extension_map_revision_before = 313u;
    row.extension_map_revision_after = 313u;
    row.declared_stage = LevelSetMaintenanceDeclaredStage::AcceptedEndpointPostStep;
    row.before = {functionalValue(2.0, 41u, 407), ownerValue(1.5, 43u, 0.5, 0.25, 409)};
    row.after = {functionalValue(2.125, 42u, 407), ownerValue(1.75, 44u, 0.75, 0.25, 409)};
    row.numerical_work = 0.375;
    row.accepted_numerical_work = 0.375;

    const auto lines = level_set::levelSetMaintenanceWorkRowRecord(row).lines();
    const std::string head = "LevelSetMaintenanceWorkLedger: work row ";
    const std::string row_keys = "transaction_id=121 substage=reinitialization step=8 attempt=2 ";
    const std::vector<std::string> expected{
        head + "diagnostic=level_set_maintenance_work transaction_id=121 status=accepted "
               "substage=reinitialization step=8 attempt=2 time=0.4 dt=0.05 "
               "declared_stage=accepted_endpoint_post_step algebraic_state_revision_before=7001 "
               "algebraic_state_revision_after=7002 snapshot_set_revision_before=5001 "
               "snapshot_set_revision_after=5002 mesh_topology_set_revision_before=19 "
               "mesh_topology_set_revision_after=19 cut_topology_set_revision_before=6001 "
               "cut_topology_set_revision_after=6002 extension_map_revision_before=313 "
               "extension_map_revision_after=313 authoritative_functional_count=2 "
               "numerical_work_sign=potential_after_minus_before numerical_work=0.375 "
               "accepted_numerical_work=0.375 modeled_energy_work_available=false "
               "modeled_energy_numerical_work=0 accepted_modeled_energy_numerical_work=0 "
               "modeled_energy_classification=numerical_maintenance_change "
               "legacy_account_scope=maintenance_surface_wall_volume_potential "
               "modeled_account_scope=kinetic_gravitational_surface_young_wall "
               "complete_time_discrete_energy_identity=false",
        head + "diagnostic=level_set_maintenance_work_interface " + row_keys +
            "interface_marker=407 before_snapshot_revision=41 after_snapshot_revision=42 "
            "before_mesh_topology_revision=19 after_mesh_topology_revision=19 "
            "before_cut_topology_revision=23 after_cut_topology_revision=23 "
            "before_liquid_volume=1.25 after_liquid_volume=1.25 before_liquid_gas_area=2.5 "
            "after_liquid_gas_area=2.5 before_wetted_wall_area=0.75 after_wetted_wall_area=0.75 "
            "before_contact_measure=0.5 after_contact_measure=0.5 before_surface_energy=3 "
            "after_surface_energy=3 before_young_wall_energy=-0.25 after_young_wall_energy=-0.25 "
            "before_volume_constraint_potential=-0.75 after_volume_constraint_potential=-0.625 "
            "before_total_potential=2 after_total_potential=2.125",
        head + "diagnostic=level_set_maintenance_work_interface " + row_keys +
            "interface_marker=409 before_snapshot_revision=43 after_snapshot_revision=44 "
            "before_mesh_topology_revision=19 after_mesh_topology_revision=19 "
            "before_cut_topology_revision=23 after_cut_topology_revision=23 "
            "before_liquid_volume=1.25 after_liquid_volume=1.25 before_liquid_gas_area=2.5 "
            "after_liquid_gas_area=2.5 before_wetted_wall_area=0.75 after_wetted_wall_area=0.75 "
            "before_contact_measure=0.5 after_contact_measure=0.5 before_surface_energy=3 "
            "after_surface_energy=3 before_young_wall_energy=-0.25 after_young_wall_energy=-0.25 "
            "before_volume_constraint_potential=-1.25 after_volume_constraint_potential=-1 "
            "before_total_potential=1.5 after_total_potential=1.75 before_kinetic_energy=0.5 "
            "after_kinetic_energy=0.75 before_gravitational_energy=0.25 after_gravitational_energy=0.25 "
            "before_gravitational_potential_power=-0.5 after_gravitational_potential_power=-0.5 "
            "before_modeled_stored_energy=3.5 after_modeled_stored_energy=3.75",
    };
    EXPECT_EQ(lines, expected);
    EXPECT_TRUE(level_set::levelSetMaintenanceWorkRowRecord(row).isRootOnly());
}
