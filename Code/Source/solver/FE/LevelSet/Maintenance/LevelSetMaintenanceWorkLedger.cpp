/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "LevelSet/Maintenance/LevelSetMaintenanceWorkLedger.h"

#include "Core/LogRecord.h"
#include "Core/RankLog.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace svmp {
namespace FE {
namespace level_set {

void mixMaintenanceRevision(
    std::uint64_t& revision,
    std::uint64_t value) noexcept
{
  constexpr std::uint64_t prime = 1099511628211ull;
  for (std::size_t byte = 0u; byte < sizeof(value); ++byte) {
    revision ^= (value >> (byte * 8u)) & 0xffu;
    revision *= prime;
  }
}

namespace {

enum class MaintenanceRevisionKind : std::uint8_t {
  Snapshot,
  MeshTopology,
  CutTopology
};

std::uint64_t maintenanceRevisionSet(
    const std::vector<LevelSetAuthoritativeFunctionalValue>& values,
    MaintenanceRevisionKind kind) noexcept
{
  if (values.empty()) {
    return 0u;
  }
  std::uint64_t revision = 1469598103934665603ull;
  for (const auto& value : values) {
    mixMaintenanceRevision(
        revision, static_cast<std::uint64_t>(value.interface_marker));
    switch (kind) {
      case MaintenanceRevisionKind::Snapshot:
        mixMaintenanceRevision(revision, value.snapshot_revision);
        break;
      case MaintenanceRevisionKind::MeshTopology:
        mixMaintenanceRevision(revision, value.mesh_topology_revision);
        break;
      case MaintenanceRevisionKind::CutTopology:
        mixMaintenanceRevision(revision, value.cut_topology_revision);
        break;
    }
  }
  return revision == 0u ? 1u : revision;
}

void validateMaintenanceFunctionalValues(
    const std::vector<LevelSetAuthoritativeFunctionalValue>& values)
{
  int previous_marker = -1;
  std::size_t active_volume_owner_count = 0u;
  for (const auto& value : values) {
    const std::array<double, 8> scalars{
        value.liquid_volume,
        value.liquid_gas_area,
        value.wetted_wall_area,
        value.contact_measure,
        value.surface_energy,
        value.young_wall_energy,
        value.volume_constraint_potential,
        value.total_potential};
    if (value.interface_marker < 0 ||
        value.interface_marker <= previous_marker ||
        value.snapshot_revision == 0u ||
        value.cut_topology_revision == 0u ||
        value.liquid_volume < 0.0 ||
        value.liquid_gas_area < 0.0 ||
        value.wetted_wall_area < 0.0 ||
        value.contact_measure < 0.0 ||
        value.surface_energy < 0.0 ||
        !std::all_of(scalars.begin(), scalars.end(), [](double scalar) {
          return std::isfinite(scalar);
        })) {
      throw std::invalid_argument(
        "Level-set maintenance work values require unique ordered markers, "
        "complete snapshot and cut-topology revisions, "
        "nonnegative measures and surface energy, and finite scalars.");
    }
    const double expected_total_potential =
        value.surface_energy + value.young_wall_energy +
        value.volume_constraint_potential;
    const double potential_scale = std::max(
        {1.0,
         std::abs(value.total_potential),
         std::abs(expected_total_potential)});
    if (std::abs(
            value.total_potential - expected_total_potential) >
        64.0 * std::numeric_limits<double>::epsilon() *
            potential_scale) {
      throw std::invalid_argument(
          "Level-set maintenance total potential must equal the surface, "
          "Young-wall, and volume-constraint contributions.");
    }
    const bool has_active_energy =
        value.kinetic_energy.has_value();
    if (value.gravitational_energy.has_value() != has_active_energy ||
        value.gravitational_potential_power.has_value() !=
            has_active_energy ||
        value.modeled_stored_energy.has_value() != has_active_energy) {
      throw std::invalid_argument(
          "Level-set maintenance active-volume energy channels must be supplied together.");
    }
    if (has_active_energy) {
      ++active_volume_owner_count;
      const double modeled_sum =
          *value.kinetic_energy +
          *value.gravitational_energy +
          value.surface_energy +
          value.young_wall_energy;
      const double scale = std::max(
          {1.0,
           std::abs(modeled_sum),
           std::abs(*value.modeled_stored_energy)});
      if (!std::isfinite(*value.kinetic_energy) ||
          *value.kinetic_energy < 0.0 ||
          !std::isfinite(*value.gravitational_energy) ||
          !std::isfinite(
              *value.gravitational_potential_power) ||
          !std::isfinite(*value.modeled_stored_energy) ||
          std::abs(
              *value.modeled_stored_energy - modeled_sum) >
              64.0 * std::numeric_limits<double>::epsilon() *
                  scale) {
        throw std::invalid_argument(
            "Level-set maintenance modeled stored-energy channels are invalid or inconsistent.");
      }
    }
    if (value.surface_wall_potential_power.has_value() !=
        value.volume_constraint_potential_power.has_value()) {
      throw std::invalid_argument(
          "Level-set maintenance endpoint functional-power channels must be supplied together.");
    }
    if ((value.surface_wall_potential_power.has_value() &&
         (!std::isfinite(*value.surface_wall_potential_power) ||
          !std::isfinite(
              *value.volume_constraint_potential_power))) ||
        (value.bulk_viscous_dissipation_rate.has_value() &&
         (!std::isfinite(
              *value.bulk_viscous_dissipation_rate) ||
          *value.bulk_viscous_dissipation_rate < 0.0)) ||
        (value.external_pressure_power.has_value() &&
         !std::isfinite(*value.external_pressure_power))) {
      throw std::invalid_argument(
          "Level-set maintenance endpoint power or dissipation channels are invalid.");
    }
    previous_marker = value.interface_marker;
  }
  if (active_volume_owner_count > 1u) {
    throw std::invalid_argument(
        "Level-set maintenance modeled energy requires at most one "
        "active-volume owner across all interface functionals.");
  }
}

std::optional<double> maintenanceModeledStoredEnergy(
    const std::vector<LevelSetAuthoritativeFunctionalValue>& values)
{
  const LevelSetAuthoritativeFunctionalValue* active_volume_owner =
      nullptr;
  double surface_and_wall_energy = 0.0;
  for (const auto& value : values) {
    surface_and_wall_energy +=
        value.surface_energy + value.young_wall_energy;
    if (!value.modeled_stored_energy.has_value()) {
      continue;
    }
    if (active_volume_owner != nullptr) {
      throw std::invalid_argument(
          "Level-set maintenance modeled energy has multiple "
          "active-volume owners.");
    }
    active_volume_owner = &value;
  }
  if (active_volume_owner == nullptr) {
    return std::nullopt;
  }
  const double total =
      *active_volume_owner->kinetic_energy +
      *active_volume_owner->gravitational_energy +
      surface_and_wall_energy;
  if (!std::isfinite(total)) {
    throw std::invalid_argument(
        "Level-set maintenance total modeled stored energy is not finite.");
  }
  return total;
}

LevelSetMaintenanceWorkAttempt makeMaintenanceWorkAttempt(
    const LevelSetMaintenanceWorkTransaction& transaction,
    LevelSetMaintenanceWorkStatus status,
    const std::vector<LevelSetMaintenanceWorkRow>& rows) noexcept
{
  struct ModeledEnergyAccumulator {
    std::size_t row_count{0u};
    bool complete{true};
    double change{0.0};
  };
  std::array<ModeledEnergyAccumulator, 5> substage_accumulators{};
  const auto substage_index =
      [](LevelSetMaintenanceWorkSubstage substage) noexcept {
        switch (substage) {
          case LevelSetMaintenanceWorkSubstage::Transport:
            return std::size_t{0u};
          case LevelSetMaintenanceWorkSubstage::Limiting:
            return std::size_t{1u};
          case LevelSetMaintenanceWorkSubstage::Reinitialization:
            return std::size_t{2u};
          case LevelSetMaintenanceWorkSubstage::GeometryReconciliation:
            return std::size_t{3u};
          case LevelSetMaintenanceWorkSubstage::GlobalCorrection:
            return std::size_t{4u};
        }
        return std::size_t{5u};
  };
  double numerical_work = 0.0;
  std::optional<double> modeled_energy_numerical_work =
      rows.empty() ? std::nullopt : std::optional<double>{0.0};
  for (const auto& row : rows) {
    numerical_work += row.numerical_work;
    const auto index = substage_index(row.substage);
    if (index < substage_accumulators.size()) {
      auto& accumulator = substage_accumulators[index];
      ++accumulator.row_count;
      if (!row.modeled_energy_numerical_work.has_value()) {
        accumulator.complete = false;
      } else {
        accumulator.change +=
            *row.modeled_energy_numerical_work;
      }
    }
    if (!row.modeled_energy_numerical_work.has_value()) {
      modeled_energy_numerical_work.reset();
    } else if (modeled_energy_numerical_work.has_value()) {
      *modeled_energy_numerical_work +=
          *row.modeled_energy_numerical_work;
    }
  }
  const auto publish_substage =
      [status](const ModeledEnergyAccumulator& accumulator) {
        LevelSetMaintenanceModeledEnergySubstage result;
        result.row_count = accumulator.row_count;
        if (accumulator.row_count == 0u || !accumulator.complete) {
          return result;
        }
        result.modeled_energy_change = accumulator.change;
        result.accepted_modeled_energy_change =
            status == LevelSetMaintenanceWorkStatus::Accepted
                ? accumulator.change
                : 0.0;
        return result;
      };
  ModeledEnergyAccumulator numerical_maintenance;
  for (std::size_t index = 1u;
       index < substage_accumulators.size();
       ++index) {
    const auto& accumulator = substage_accumulators[index];
    numerical_maintenance.row_count += accumulator.row_count;
    numerical_maintenance.complete =
        numerical_maintenance.complete && accumulator.complete;
    numerical_maintenance.change += accumulator.change;
  }
  const LevelSetMaintenanceModeledEnergyBreakdown breakdown{
      .transport = publish_substage(substage_accumulators[0]),
      .limiting = publish_substage(substage_accumulators[1]),
      .reinitialization =
          publish_substage(substage_accumulators[2]),
      .geometry_reconciliation =
          publish_substage(substage_accumulators[3]),
      .global_correction =
          publish_substage(substage_accumulators[4]),
      .numerical_maintenance_total =
          publish_substage(numerical_maintenance),
  };
  return LevelSetMaintenanceWorkAttempt{
      .transaction_id = transaction.transaction_id,
      .status = status,
      .step = transaction.step,
      .attempt = transaction.attempt,
      .time = transaction.time,
      .dt = transaction.dt,
      .declared_stage = transaction.declared_stage,
      .extension_map_revision = transaction.extension_map_revision,
      .row_count = rows.size(),
      .numerical_work = numerical_work,
      .accepted_numerical_work =
          status == LevelSetMaintenanceWorkStatus::Accepted
              ? numerical_work
              : 0.0,
      .modeled_energy_numerical_work =
          modeled_energy_numerical_work,
      .accepted_modeled_energy_numerical_work =
          modeled_energy_numerical_work.has_value()
              ? std::optional<double>{
                    status ==
                            LevelSetMaintenanceWorkStatus::Accepted
                        ? *modeled_energy_numerical_work
                        : 0.0}
              : std::nullopt,
      .modeled_energy_breakdown = breakdown,
  };
}

const char* levelSetMaintenanceWorkStatusName(
    LevelSetMaintenanceWorkStatus status) noexcept
{
  using Status = LevelSetMaintenanceWorkStatus;
  switch (status) {
    case Status::Trial:
      return "trial";
    case Status::Accepted:
      return "accepted";
    case Status::Rejected:
      return "rejected";
  }
  return "unknown";
}

const char* levelSetMaintenanceWorkSubstageName(
    LevelSetMaintenanceWorkSubstage substage) noexcept
{
  using Substage = LevelSetMaintenanceWorkSubstage;
  switch (substage) {
    case Substage::Transport:
      return "transport";
    case Substage::Limiting:
      return "limiting";
    case Substage::Reinitialization:
      return "reinitialization";
    case Substage::GeometryReconciliation:
      return "geometry_reconciliation";
    case Substage::GlobalCorrection:
      return "global_correction";
  }
  return "unknown";
}

const char* levelSetMaintenanceDeclaredStageName(
    LevelSetMaintenanceDeclaredStage stage) noexcept
{
  using Stage = LevelSetMaintenanceDeclaredStage;
  switch (stage) {
    case Stage::ProspectiveAcceptedEndpoint:
      return "prospective_accepted_endpoint";
    case Stage::AcceptedEndpointPostStep:
      return "accepted_endpoint_post_step";
  }
  return "unknown";
}

} // namespace

std::optional<LevelSetMaintenanceAcceptedStepEnergyAccount>
aggregateLevelSetMaintenanceAcceptedStepEnergy(
    std::span<const LevelSetMaintenanceWorkAttempt> attempts,
    std::span<const LevelSetMaintenanceWorkRow> rows)
{
  if (attempts.empty()) {
    if (!rows.empty()) {
      throw std::invalid_argument(
          "Accepted-step maintenance rows require an accepted attempt.");
    }
    return std::nullopt;
  }

  struct Accumulator {
    std::size_t row_count{0u};
    bool complete{true};
    double change{0.0};
  };
  const auto checked_row_count_sum =
      [](std::size_t left, std::size_t right) {
        if (right >
            std::numeric_limits<std::size_t>::max() - left) {
          throw std::overflow_error(
              "Accepted-step maintenance row count overflowed.");
        }
        return left + right;
      };
  const auto validate_substage =
      [](const LevelSetMaintenanceModeledEnergySubstage& substage) {
        const bool diagnostic_available =
            substage.modeled_energy_change.has_value();
        const bool accepted_available =
            substage.accepted_modeled_energy_change.has_value();
        if (substage.row_count == 0u) {
          if (diagnostic_available || accepted_available) {
            throw std::invalid_argument(
                "A zero-row accepted-step maintenance substage cannot publish modeled-energy values.");
          }
          return;
        }
        if (diagnostic_available != accepted_available) {
          throw std::invalid_argument(
              "An accepted-step maintenance substage must publish matching diagnostic and accepted modeled-energy availability.");
        }
        if (diagnostic_available &&
            (!std::isfinite(*substage.modeled_energy_change) ||
             !std::isfinite(
                 *substage.accepted_modeled_energy_change) ||
             *substage.modeled_energy_change !=
                 *substage.accepted_modeled_energy_change)) {
          throw std::invalid_argument(
              "An accepted-step maintenance substage has inconsistent modeled-energy values.");
        }
      };
  const auto merge_substage =
      [&](Accumulator& accumulator,
          const LevelSetMaintenanceModeledEnergySubstage& substage) {
        validate_substage(substage);
        accumulator.row_count = checked_row_count_sum(
            accumulator.row_count, substage.row_count);
        if (substage.row_count == 0u) {
          return;
        }
        if (!substage.accepted_modeled_energy_change.has_value()) {
          accumulator.complete = false;
          return;
        }
        accumulator.change +=
            *substage.accepted_modeled_energy_change;
        if (!std::isfinite(accumulator.change)) {
          throw std::overflow_error(
              "Accepted-step maintenance modeled-energy sum is not finite.");
        }
      };
  const auto publish_substage =
      [](const Accumulator& accumulator) {
        LevelSetMaintenanceModeledEnergySubstage result;
        result.row_count = accumulator.row_count;
        if (accumulator.row_count != 0u && accumulator.complete) {
          result.modeled_energy_change = accumulator.change;
          result.accepted_modeled_energy_change =
              accumulator.change;
        }
        return result;
      };
  const auto require_attempt_breakdown =
      [&](const LevelSetMaintenanceWorkAttempt& attempt) {
        const auto& breakdown = attempt.modeled_energy_breakdown;
        validate_substage(breakdown.transport);
        validate_substage(breakdown.limiting);
        validate_substage(breakdown.reinitialization);
        validate_substage(breakdown.geometry_reconciliation);
        validate_substage(breakdown.global_correction);
        validate_substage(breakdown.numerical_maintenance_total);

        std::size_t numerical_row_count = 0u;
        numerical_row_count = checked_row_count_sum(
            numerical_row_count, breakdown.limiting.row_count);
        numerical_row_count = checked_row_count_sum(
            numerical_row_count,
            breakdown.reinitialization.row_count);
        numerical_row_count = checked_row_count_sum(
            numerical_row_count,
            breakdown.geometry_reconciliation.row_count);
        numerical_row_count = checked_row_count_sum(
            numerical_row_count,
            breakdown.global_correction.row_count);
        if (numerical_row_count !=
            breakdown.numerical_maintenance_total.row_count) {
          throw std::invalid_argument(
              "An accepted-step maintenance attempt has inconsistent numerical-maintenance row coverage.");
        }
        const std::array<
            const LevelSetMaintenanceModeledEnergySubstage*,
            4>
            numerical_substages{{
                &breakdown.limiting,
                &breakdown.reinitialization,
                &breakdown.geometry_reconciliation,
                &breakdown.global_correction,
            }};
        bool numerical_complete = numerical_row_count != 0u;
        double numerical_change = 0.0;
        for (const auto* substage : numerical_substages) {
          if (substage->row_count == 0u) {
            continue;
          }
          numerical_complete =
              numerical_complete &&
              substage->modeled_energy_change.has_value();
          if (substage->modeled_energy_change.has_value()) {
            numerical_change += *substage->modeled_energy_change;
          }
        }
        const bool numerical_total_available =
            breakdown.numerical_maintenance_total
                .modeled_energy_change.has_value();
        if (numerical_complete != numerical_total_available ||
            (numerical_complete &&
             numerical_change !=
                 *breakdown.numerical_maintenance_total
                      .modeled_energy_change)) {
          throw std::invalid_argument(
              "An accepted-step maintenance attempt has an inconsistent numerical-maintenance modeled-energy total.");
        }

        const auto covered_row_count = checked_row_count_sum(
            breakdown.transport.row_count, numerical_row_count);
        if (covered_row_count != attempt.row_count) {
          throw std::invalid_argument(
              "An accepted-step maintenance attempt has inconsistent total row coverage.");
        }
        const bool all_rows_complete =
            attempt.row_count != 0u &&
            (breakdown.transport.row_count == 0u ||
             breakdown.transport.modeled_energy_change.has_value()) &&
            (numerical_row_count == 0u || numerical_complete);
        const bool aggregate_available =
            attempt.modeled_energy_numerical_work.has_value();
        const bool accepted_aggregate_available =
            attempt.accepted_modeled_energy_numerical_work.has_value();
        if (aggregate_available != accepted_aggregate_available ||
            aggregate_available != all_rows_complete) {
          throw std::invalid_argument(
              "An accepted-step maintenance attempt has inconsistent aggregate modeled-energy availability.");
        }
        if (aggregate_available) {
          if (!std::isfinite(
                  *attempt.modeled_energy_numerical_work) ||
              !std::isfinite(
                  *attempt
                       .accepted_modeled_energy_numerical_work) ||
              *attempt.accepted_modeled_energy_numerical_work !=
                  *attempt.modeled_energy_numerical_work) {
            throw std::invalid_argument(
                "An accepted-step maintenance attempt has inconsistent aggregate modeled-energy values.");
          }
        }
      };

  const auto& first = attempts.front();
  if (first.status != LevelSetMaintenanceWorkStatus::Accepted ||
      first.transaction_id == 0u || first.step == 0u ||
      first.attempt == 0u || !std::isfinite(first.time) ||
      !std::isfinite(first.dt) || first.dt < 0.0 ||
      (first.extension_map_revision.has_value() &&
       *first.extension_map_revision == 0u)) {
    throw std::invalid_argument(
        "Accepted-step maintenance aggregation requires complete accepted-attempt metadata.");
  }
  LevelSetMaintenanceAcceptedStepEnergyAccount account{
      .step = first.step,
      .attempt = first.attempt,
      .time = first.time,
      .dt = first.dt,
      .transaction_count = attempts.size(),
  };
  Accumulator transport;
  Accumulator limiting;
  Accumulator reinitialization;
  Accumulator geometry_reconciliation;
  Accumulator global_correction;
  Accumulator numerical_maintenance_total;
  std::uint64_t previous_transaction_id{0u};
  for (const auto& attempt : attempts) {
    if (attempt.status != LevelSetMaintenanceWorkStatus::Accepted ||
        attempt.transaction_id == 0u ||
        attempt.transaction_id <= previous_transaction_id ||
        attempt.step != account.step ||
        attempt.attempt != account.attempt ||
        attempt.time != account.time || attempt.dt != account.dt ||
        (attempt.extension_map_revision.has_value() &&
         *attempt.extension_map_revision == 0u) ||
        !std::isfinite(attempt.numerical_work) ||
        !std::isfinite(attempt.accepted_numerical_work) ||
        attempt.numerical_work != attempt.accepted_numerical_work) {
      throw std::invalid_argument(
          "Accepted-step maintenance attempts must form one metadata-consistent accepted transaction sequence.");
    }
    require_attempt_breakdown(attempt);
    previous_transaction_id = attempt.transaction_id;
    account.row_count = checked_row_count_sum(
        account.row_count, attempt.row_count);
    const auto& breakdown = attempt.modeled_energy_breakdown;
    merge_substage(transport, breakdown.transport);
    merge_substage(limiting, breakdown.limiting);
    merge_substage(
        reinitialization, breakdown.reinitialization);
    merge_substage(
        geometry_reconciliation,
        breakdown.geometry_reconciliation);
    merge_substage(
        global_correction, breakdown.global_correction);
    merge_substage(
        numerical_maintenance_total,
        breakdown.numerical_maintenance_total);
  }
  account.modeled_energy_breakdown =
      LevelSetMaintenanceModeledEnergyBreakdown{
          .transport = publish_substage(transport),
          .limiting = publish_substage(limiting),
          .reinitialization =
              publish_substage(reinitialization),
          .geometry_reconciliation =
              publish_substage(geometry_reconciliation),
          .global_correction =
              publish_substage(global_correction),
          .numerical_maintenance_total =
              publish_substage(numerical_maintenance_total),
      };
  if (rows.size() != account.row_count) {
    throw std::invalid_argument(
        "Accepted-step maintenance row history does not match the published attempt coverage.");
  }
  if (rows.empty()) {
    bool poststep_stage_seen = false;
    const auto extension_map_revision =
        attempts.front().extension_map_revision;
    for (const auto& attempt : attempts) {
      const bool prospective =
          attempt.declared_stage ==
          LevelSetMaintenanceDeclaredStage::
              ProspectiveAcceptedEndpoint;
      const bool poststep =
          attempt.declared_stage ==
          LevelSetMaintenanceDeclaredStage::
              AcceptedEndpointPostStep;
      if ((!prospective && !poststep) ||
          (prospective && poststep_stage_seen) ||
          attempt.extension_map_revision !=
              extension_map_revision) {
        throw std::invalid_argument(
            "Zero-row accepted-step maintenance transactions have invalid stage ordering or inconsistent extension provenance.");
      }
      poststep_stage_seen = poststep_stage_seen || poststep;
    }
    return account;
  }

  const auto make_endpoint =
      [&](const LevelSetMaintenanceWorkRow& row, bool use_after) {
        const auto& values = use_after ? row.after : row.before;
        validateMaintenanceFunctionalValues(values);
        LevelSetMaintenanceAcceptedStepEnergyAccount::Endpoint
            endpoint{
                .algebraic_state_revision =
                    use_after
                        ? row.algebraic_state_revision_after
                        : row.algebraic_state_revision_before,
                .snapshot_set_revision =
                    use_after ? row.snapshot_set_revision_after
                              : row.snapshot_set_revision_before,
                .mesh_topology_set_revision =
                    use_after
                        ? row.mesh_topology_set_revision_after
                        : row.mesh_topology_set_revision_before,
                .cut_topology_set_revision =
                    use_after ? row.cut_topology_set_revision_after
                              : row.cut_topology_set_revision_before,
                .extension_map_revision =
                    use_after ? row.extension_map_revision_after
                              : row.extension_map_revision_before,
            };
        const LevelSetAuthoritativeFunctionalValue*
            active_volume_owner = nullptr;
        const LevelSetAuthoritativeFunctionalValue*
            bulk_dissipation_owner = nullptr;
        bool surface_power_complete = true;
        bool external_pressure_power_complete = true;
        double surface_wall_power = 0.0;
        double volume_constraint_power = 0.0;
        double external_pressure_power = 0.0;
        for (const auto& value : values) {
          endpoint.liquid_gas_surface_energy +=
              value.surface_energy;
          endpoint.solid_liquid_wall_energy +=
              value.young_wall_energy;
          if (value.kinetic_energy.has_value()) {
            if (active_volume_owner != nullptr) {
              throw std::invalid_argument(
                  "An accepted-step maintenance endpoint has more than one active-volume energy owner.");
            }
            active_volume_owner = &value;
          }
          if (!value.surface_wall_potential_power.has_value()) {
            surface_power_complete = false;
          } else {
            surface_wall_power +=
                *value.surface_wall_potential_power;
            volume_constraint_power +=
                *value.volume_constraint_potential_power;
          }
          if (!value.external_pressure_power.has_value()) {
            external_pressure_power_complete = false;
          } else {
            external_pressure_power +=
                *value.external_pressure_power;
          }
          if (value.bulk_viscous_dissipation_rate.has_value()) {
            if (bulk_dissipation_owner != nullptr) {
              throw std::invalid_argument(
                  "An accepted-step maintenance endpoint has more than one bulk-dissipation owner.");
            }
            bulk_dissipation_owner = &value;
          }
        }
        if (!std::isfinite(
                endpoint.liquid_gas_surface_energy) ||
            !std::isfinite(
                endpoint.solid_liquid_wall_energy) ||
            !std::isfinite(surface_wall_power) ||
            !std::isfinite(volume_constraint_power) ||
            !std::isfinite(external_pressure_power)) {
          throw std::overflow_error(
              "Accepted-step maintenance endpoint energy sums are not finite.");
        }
        if (active_volume_owner != nullptr) {
          endpoint.kinetic_energy =
              active_volume_owner->kinetic_energy;
          endpoint.gravitational_energy =
              active_volume_owner->gravitational_energy;
          endpoint.gravitational_potential_power =
              active_volume_owner->gravitational_potential_power;
        }
        if (surface_power_complete) {
          endpoint.surface_wall_potential_power =
              surface_wall_power;
          endpoint.volume_constraint_potential_power =
              volume_constraint_power;
        }
        if (bulk_dissipation_owner != nullptr) {
          endpoint.bulk_viscous_dissipation_rate =
              bulk_dissipation_owner
                  ->bulk_viscous_dissipation_rate;
        }
        if (external_pressure_power_complete) {
          endpoint.external_pressure_power =
              external_pressure_power;
        }
        endpoint.modeled_stored_energy =
            maintenanceModeledStoredEnergy(values);
        return endpoint;
      };
  const auto substage_index =
      [](LevelSetMaintenanceWorkSubstage substage) {
        switch (substage) {
          case LevelSetMaintenanceWorkSubstage::Transport:
            return std::size_t{0u};
          case LevelSetMaintenanceWorkSubstage::Limiting:
            return std::size_t{1u};
          case LevelSetMaintenanceWorkSubstage::Reinitialization:
            return std::size_t{2u};
          case LevelSetMaintenanceWorkSubstage::GeometryReconciliation:
            return std::size_t{3u};
          case LevelSetMaintenanceWorkSubstage::GlobalCorrection:
            return std::size_t{4u};
        }
        return std::size_t{5u};
      };
  const LevelSetMaintenanceWorkRow* previous_row = nullptr;
  bool nontransport_seen = false;
  bool poststep_stage_seen = false;
  bool expected_extension_initialized = false;
  std::optional<std::uint64_t> expected_extension_revision;
  std::size_t row_index = 0u;
  for (const auto& attempt : attempts) {
    const bool prospective =
        attempt.declared_stage ==
        LevelSetMaintenanceDeclaredStage::
            ProspectiveAcceptedEndpoint;
    const bool poststep =
        attempt.declared_stage ==
        LevelSetMaintenanceDeclaredStage::
            AcceptedEndpointPostStep;
    if ((!prospective && !poststep) ||
        (prospective && poststep_stage_seen)) {
      throw std::invalid_argument(
          "Accepted-step maintenance transaction stages are invalid or out of order.");
    }
    poststep_stage_seen = poststep_stage_seen || poststep;
    if (expected_extension_initialized &&
        attempt.extension_map_revision !=
            expected_extension_revision) {
      throw std::invalid_argument(
          "Accepted-step maintenance transactions do not form one extension-map revision chain.");
    }
    expected_extension_initialized = true;
    expected_extension_revision =
        attempt.extension_map_revision;

    std::array<Accumulator, 5> row_substages{};
    for (std::size_t transaction_row = 0u;
         transaction_row < attempt.row_count;
         ++transaction_row, ++row_index) {
      const auto& row = rows[row_index];
      const auto index = substage_index(row.substage);
      if (index >= row_substages.size() ||
          row.before.empty() || row.after.empty() ||
          row.status != LevelSetMaintenanceWorkStatus::Accepted ||
          row.transaction_id != attempt.transaction_id ||
          row.step != account.step ||
          row.attempt != account.attempt ||
          row.time != account.time || row.dt != account.dt ||
          row.declared_stage != attempt.declared_stage ||
          !std::isfinite(row.numerical_work) ||
          !std::isfinite(row.accepted_numerical_work) ||
          row.numerical_work != row.accepted_numerical_work ||
          row.modeled_energy_numerical_work.has_value() !=
              row.accepted_modeled_energy_numerical_work
                  .has_value() ||
          (row.modeled_energy_numerical_work.has_value() &&
           (!std::isfinite(
                *row.modeled_energy_numerical_work) ||
            !std::isfinite(
                *row.accepted_modeled_energy_numerical_work) ||
            *row.modeled_energy_numerical_work !=
                *row.accepted_modeled_energy_numerical_work))) {
        throw std::invalid_argument(
            "Accepted-step maintenance row metadata or accepted work is inconsistent with its transaction.");
      }
      if (transaction_row == 0u &&
          row.extension_map_revision_before !=
              attempt.extension_map_revision) {
        throw std::invalid_argument(
            "An accepted-step maintenance transaction starts from the wrong extension-map revision.");
      }
      if (previous_row != nullptr &&
          (previous_row->algebraic_state_revision_after !=
                   row.algebraic_state_revision_before ||
           previous_row->snapshot_set_revision_after !=
                   row.snapshot_set_revision_before ||
           previous_row->mesh_topology_set_revision_after !=
                   row.mesh_topology_set_revision_before ||
           previous_row->cut_topology_set_revision_after !=
                   row.cut_topology_set_revision_before ||
           previous_row->extension_map_revision_after !=
                   row.extension_map_revision_before ||
           previous_row->after != row.before)) {
        throw std::invalid_argument(
            "Accepted-step maintenance rows do not form one exact algebraic, functional, geometry, topology, and extension chain.");
      }
      if (previous_row == nullptr) {
        account.maintenance_start = make_endpoint(row, false);
        account.post_transport = account.maintenance_start;
      }
      if (row.substage ==
          LevelSetMaintenanceWorkSubstage::Transport) {
        if (nontransport_seen) {
          throw std::invalid_argument(
              "Accepted-step physical Transport rows must precede every numerical-maintenance row.");
        }
        account.post_transport = make_endpoint(row, true);
      } else {
        nontransport_seen = true;
      }
      auto& row_accumulator = row_substages[index];
      ++row_accumulator.row_count;
      if (!row.accepted_modeled_energy_numerical_work
               .has_value()) {
        row_accumulator.complete = false;
      } else {
        row_accumulator.change +=
            *row.accepted_modeled_energy_numerical_work;
        if (!std::isfinite(row_accumulator.change)) {
          throw std::overflow_error(
              "Accepted-step maintenance row energy sum is not finite.");
        }
      }
      expected_extension_revision =
          row.extension_map_revision_after;
      previous_row = &row;
    }

    Accumulator row_numerical_maintenance;
    for (std::size_t index = 1u;
         index < row_substages.size();
         ++index) {
      const auto& row_substage = row_substages[index];
      row_numerical_maintenance.row_count =
          checked_row_count_sum(
              row_numerical_maintenance.row_count,
              row_substage.row_count);
      row_numerical_maintenance.complete =
          row_numerical_maintenance.complete &&
          row_substage.complete;
      row_numerical_maintenance.change +=
          row_substage.change;
    }
    const LevelSetMaintenanceModeledEnergyBreakdown
        row_breakdown{
            .transport = publish_substage(row_substages[0]),
            .limiting = publish_substage(row_substages[1]),
            .reinitialization =
                publish_substage(row_substages[2]),
            .geometry_reconciliation =
                publish_substage(row_substages[3]),
            .global_correction =
                publish_substage(row_substages[4]),
            .numerical_maintenance_total =
                publish_substage(row_numerical_maintenance),
        };
    if (row_breakdown != attempt.modeled_energy_breakdown) {
      throw std::invalid_argument(
          "Accepted-step maintenance rows do not reproduce their published attempt breakdown.");
    }
  }
  if (row_index != rows.size() || previous_row == nullptr) {
    throw std::logic_error(
        "Accepted-step maintenance row traversal is incomplete.");
  }
  account.maintenance_end = make_endpoint(*previous_row, true);
  const auto endpoint_residual =
      [](const LevelSetMaintenanceAcceptedStepEnergyAccount::Endpoint&
             before,
          const LevelSetMaintenanceAcceptedStepEnergyAccount::Endpoint&
              after,
          const LevelSetMaintenanceModeledEnergySubstage& work,
          bool physical_transport) -> std::optional<double> {
        if (work.row_count == 0u ||
            !work.accepted_modeled_energy_change.has_value()) {
          return std::nullopt;
        }
        if (!before.modeled_stored_energy.has_value() ||
            !after.modeled_stored_energy.has_value()) {
          throw std::invalid_argument(
              "An accepted-step maintenance row sum is available without both modeled-energy endpoints.");
        }
        const double endpoint_change =
            *after.modeled_stored_energy -
            *before.modeled_stored_energy;
        const double residual =
            endpoint_change -
            *work.accepted_modeled_energy_change;
        const double scale = std::max(
            {1.0,
             std::abs(endpoint_change),
             std::abs(
                 *work.accepted_modeled_energy_change)});
        if (!std::isfinite(residual) ||
            std::abs(residual) >
                64.0 * std::numeric_limits<double>::epsilon() *
                    scale) {
          if (physical_transport) {
            throw std::invalid_argument(
                "Accepted-step physical Transport work does not telescope to its modeled-energy endpoints.");
          }
          throw std::invalid_argument(
              "Accepted-step numerical-maintenance work does not telescope to its modeled-energy endpoints.");
        }
        return residual;
      };
  account.physical_transport_endpoint_residual =
      endpoint_residual(
          *account.maintenance_start,
          *account.post_transport,
          account.modeled_energy_breakdown.transport,
          true);
  account.numerical_maintenance_endpoint_residual =
      endpoint_residual(
          *account.post_transport,
          *account.maintenance_end,
          account.modeled_energy_breakdown
              .numerical_maintenance_total,
          false);
  return account;
}

void LevelSetMaintenanceWorkLedger::beginTransaction(
    LevelSetMaintenanceWorkTransaction transaction)
{
  if (active_transaction_.has_value()) {
    throw std::logic_error(
        "A level-set maintenance work transaction is already active.");
  }
  if (!trial_rows_.empty()) {
    throw std::logic_error(
        "A level-set maintenance work transaction has unpublished trial rows.");
  }
  if (transaction.transaction_id == 0u || transaction.attempt == 0u ||
      !std::isfinite(transaction.time) ||
      !std::isfinite(transaction.dt) || transaction.dt < 0.0 ||
      (transaction.extension_map_revision.has_value() &&
       *transaction.extension_map_revision == 0u)) {
    throw std::invalid_argument(
        "Level-set maintenance work transaction metadata is incomplete.");
  }
  if (transaction.transaction_id <= last_transaction_id_) {
    throw std::invalid_argument(
        "Level-set maintenance work transaction identifiers must increase.");
  }
  // Reserve both possible publication destinations before making the
  // transaction active. Commit and rejection can then publish their outcome
  // without allocating after geometry has been committed or rolled back.
  accepted_attempts_.reserve(accepted_attempts_.size() + 1u);
  rejected_attempts_.reserve(rejected_attempts_.size() + 1u);
  last_transaction_id_ = transaction.transaction_id;
  active_transaction_ = std::move(transaction);
}

void LevelSetMaintenanceWorkLedger::stageRow(
    LevelSetMaintenanceWorkSubstage substage,
    std::uint64_t algebraic_state_revision_before,
    std::uint64_t algebraic_state_revision_after,
    std::vector<LevelSetAuthoritativeFunctionalValue> before,
    std::vector<LevelSetAuthoritativeFunctionalValue> after,
    std::optional<std::uint64_t> extension_map_revision_after)
{
  if (!active_transaction_.has_value()) {
    throw std::logic_error(
        "Level-set maintenance work rows require an active transaction.");
  }
  const bool recognized_substage =
      substage == LevelSetMaintenanceWorkSubstage::Transport ||
      substage == LevelSetMaintenanceWorkSubstage::Limiting ||
      substage == LevelSetMaintenanceWorkSubstage::Reinitialization ||
      substage ==
          LevelSetMaintenanceWorkSubstage::GeometryReconciliation ||
      substage == LevelSetMaintenanceWorkSubstage::GlobalCorrection;
  if (!recognized_substage) {
    throw std::invalid_argument(
        "Level-set maintenance work rows require a recognized substage.");
  }
  if (algebraic_state_revision_before == 0u ||
      algebraic_state_revision_after == 0u) {
    throw std::invalid_argument(
        "Level-set maintenance work rows require complete algebraic state revisions.");
  }
  if (extension_map_revision_after.has_value() &&
      *extension_map_revision_after == 0u) {
    throw std::invalid_argument(
        "Level-set maintenance work rows require a nonzero extension-map revision when present.");
  }
  const auto by_marker = [](const auto& left, const auto& right) {
    return left.interface_marker < right.interface_marker;
  };
  std::sort(before.begin(), before.end(), by_marker);
  std::sort(after.begin(), after.end(), by_marker);
  validateMaintenanceFunctionalValues(before);
  validateMaintenanceFunctionalValues(after);
  if (before.size() != after.size()) {
    throw std::invalid_argument(
        "Level-set maintenance work rows require matching functional coverage.");
  }

  double numerical_work = 0.0;
  for (std::size_t index = 0u; index < before.size(); ++index) {
    if (before[index].interface_marker != after[index].interface_marker) {
      throw std::invalid_argument(
          "Level-set maintenance work rows require matching functional markers.");
    }
    if (before[index].modeled_stored_energy.has_value() !=
            after[index].modeled_stored_energy.has_value() ||
        before[index].surface_wall_potential_power.has_value() !=
            after[index].surface_wall_potential_power.has_value() ||
        before[index].bulk_viscous_dissipation_rate.has_value() !=
            after[index].bulk_viscous_dissipation_rate.has_value() ||
        before[index].external_pressure_power.has_value() !=
            after[index].external_pressure_power.has_value()) {
      throw std::invalid_argument(
          "Level-set maintenance modeled-energy, endpoint-power, dissipation, and exterior-work ownership cannot change within a row.");
    }
    numerical_work +=
        after[index].total_potential - before[index].total_potential;
  }
  if (!std::isfinite(numerical_work)) {
    throw std::invalid_argument(
        "Level-set maintenance numerical work is not finite.");
  }
  const auto modeled_energy_before =
      maintenanceModeledStoredEnergy(before);
  const auto modeled_energy_after =
      maintenanceModeledStoredEnergy(after);
  if (modeled_energy_before.has_value() !=
      modeled_energy_after.has_value()) {
    throw std::invalid_argument(
        "Level-set maintenance active-volume ownership cannot change "
        "within a row.");
  }
  std::optional<double> modeled_energy_numerical_work;
  if (modeled_energy_before.has_value()) {
    modeled_energy_numerical_work =
        *modeled_energy_after - *modeled_energy_before;
    if (!std::isfinite(*modeled_energy_numerical_work)) {
      throw std::invalid_argument(
          "Level-set maintenance modeled-energy numerical work is not finite.");
    }
  }

  const auto& transaction = *active_transaction_;
  const auto extension_map_revision_before =
      trial_rows_.empty()
          ? transaction.extension_map_revision
          : trial_rows_.back().extension_map_revision_after;
  if (!trial_rows_.empty()) {
    const auto& previous = trial_rows_.back();
    if (previous.algebraic_state_revision_after !=
            algebraic_state_revision_before ||
        previous.after != before) {
      throw std::invalid_argument(
          "Level-set maintenance work rows must form one continuous algebraic and functional state chain.");
    }
  }
  const auto resolved_extension_map_revision_after =
      extension_map_revision_after.has_value()
          ? extension_map_revision_after
          : extension_map_revision_before;

  // Preflight both terminal row destinations while candidate geometry is
  // still rollback-capable. Neither commitTransaction nor rejectTransaction
  // then needs to allocate after the physical outcome is known.
  const auto terminal_row_count =
      accepted_rows_.size() + trial_rows_.size() + 1u;
  accepted_rows_.reserve(terminal_row_count);
  rejected_rows_.reserve(
      rejected_rows_.size() + trial_rows_.size() + 1u);
  trial_rows_.reserve(trial_rows_.size() + 1u);
  trial_rows_.push_back(LevelSetMaintenanceWorkRow{
      .transaction_id = transaction.transaction_id,
      .status = LevelSetMaintenanceWorkStatus::Trial,
      .substage = substage,
      .step = transaction.step,
      .attempt = transaction.attempt,
      .time = transaction.time,
      .dt = transaction.dt,
      .algebraic_state_revision_before =
          algebraic_state_revision_before,
      .algebraic_state_revision_after =
          algebraic_state_revision_after,
      .snapshot_set_revision_before =
          maintenanceRevisionSet(
              before, MaintenanceRevisionKind::Snapshot),
      .snapshot_set_revision_after =
          maintenanceRevisionSet(
              after, MaintenanceRevisionKind::Snapshot),
      .mesh_topology_set_revision_before =
          maintenanceRevisionSet(
              before, MaintenanceRevisionKind::MeshTopology),
      .mesh_topology_set_revision_after =
          maintenanceRevisionSet(
              after, MaintenanceRevisionKind::MeshTopology),
      .cut_topology_set_revision_before =
          maintenanceRevisionSet(
              before, MaintenanceRevisionKind::CutTopology),
      .cut_topology_set_revision_after =
          maintenanceRevisionSet(
              after, MaintenanceRevisionKind::CutTopology),
      .extension_map_revision_before =
          extension_map_revision_before,
      .extension_map_revision_after =
          resolved_extension_map_revision_after,
      .declared_stage = transaction.declared_stage,
      .before = std::move(before),
      .after = std::move(after),
      .numerical_work = numerical_work,
      // Trial and rejected rows never contribute to the accepted-state
      // account. This channel is a maintenance term only, not a complete
      // time-discrete energy identity.
      .accepted_numerical_work = 0.0,
      .modeled_energy_numerical_work =
          modeled_energy_numerical_work,
      .accepted_modeled_energy_numerical_work =
          modeled_energy_numerical_work.has_value()
              ? std::optional<double>{0.0}
              : std::nullopt,
  });
}

void LevelSetMaintenanceWorkLedger::commitTransaction()
{
  if (!active_transaction_.has_value()) {
    throw std::logic_error(
        "No level-set maintenance work transaction is active.");
  }
  if (accepted_attempts_.capacity() < accepted_attempts_.size() + 1u ||
      accepted_rows_.capacity() <
          accepted_rows_.size() + trial_rows_.size()) {
    throw std::logic_error(
        "Level-set maintenance work commit was not allocation-preflighted.");
  }
  const auto attempt = makeMaintenanceWorkAttempt(
      *active_transaction_,
      LevelSetMaintenanceWorkStatus::Accepted,
      trial_rows_);
  for (auto& row : trial_rows_) {
    row.status = LevelSetMaintenanceWorkStatus::Accepted;
    row.accepted_numerical_work = row.numerical_work;
    row.accepted_modeled_energy_numerical_work =
        row.modeled_energy_numerical_work;
    accepted_rows_.push_back(std::move(row));
  }
  accepted_attempts_.push_back(attempt);
  trial_rows_.clear();
  active_transaction_.reset();
}

void LevelSetMaintenanceWorkLedger::rejectTransaction()
{
  if (!active_transaction_.has_value()) {
    throw std::logic_error(
        "No level-set maintenance work transaction is active.");
  }
  if (rejected_attempts_.capacity() < rejected_attempts_.size() + 1u ||
      rejected_rows_.capacity() <
          rejected_rows_.size() + trial_rows_.size()) {
    throw std::logic_error(
        "Level-set maintenance work rejection was not allocation-preflighted.");
  }
  const auto attempt = makeMaintenanceWorkAttempt(
      *active_transaction_,
      LevelSetMaintenanceWorkStatus::Rejected,
      trial_rows_);
  for (auto& row : trial_rows_) {
    row.status = LevelSetMaintenanceWorkStatus::Rejected;
    row.accepted_numerical_work = 0.0;
    row.accepted_modeled_energy_numerical_work =
        row.modeled_energy_numerical_work.has_value()
            ? std::optional<double>{0.0}
            : std::nullopt;
    rejected_rows_.push_back(std::move(row));
  }
  rejected_attempts_.push_back(attempt);
  trial_rows_.clear();
  active_transaction_.reset();
}

bool LevelSetMaintenanceWorkLedger::transactionActive() const noexcept
{
  return active_transaction_.has_value();
}

const LevelSetMaintenanceWorkTransaction*
LevelSetMaintenanceWorkLedger::activeTransaction() const noexcept
{
  return active_transaction_.has_value()
      ? &*active_transaction_
      : nullptr;
}

const std::vector<LevelSetMaintenanceWorkRow>&
LevelSetMaintenanceWorkLedger::trialRows() const noexcept
{
  return trial_rows_;
}

const std::vector<LevelSetMaintenanceWorkRow>&
LevelSetMaintenanceWorkLedger::acceptedRows() const noexcept
{
  return accepted_rows_;
}

const std::vector<LevelSetMaintenanceWorkRow>&
LevelSetMaintenanceWorkLedger::rejectedRows() const noexcept
{
  return rejected_rows_;
}

const std::vector<LevelSetMaintenanceWorkAttempt>&
LevelSetMaintenanceWorkLedger::acceptedAttempts() const noexcept
{
  return accepted_attempts_;
}

const std::vector<LevelSetMaintenanceWorkAttempt>&
LevelSetMaintenanceWorkLedger::rejectedAttempts() const noexcept
{
  return rejected_attempts_;
}

LevelSetMaintenanceWorkMark
LevelSetMaintenanceWorkLedger::mark() const noexcept
{
  return LevelSetMaintenanceWorkMark{
      .accepted_rows = accepted_rows_.size(),
      .rejected_rows = rejected_rows_.size(),
      .accepted_attempts = accepted_attempts_.size(),
      .rejected_attempts = rejected_attempts_.size(),
  };
}

LogRecord levelSetMaintenanceWorkRowRecord(const LevelSetMaintenanceWorkRow& row)
{
  LogRecord log("LevelSetMaintenanceWorkLedger", "work row");
  log.diagnostic("level_set_maintenance_work")
      .rootOnly()
      .value("transaction_id", row.transaction_id)
      .text("status", levelSetMaintenanceWorkStatusName(row.status))
      .text("substage", levelSetMaintenanceWorkSubstageName(row.substage))
      .value("step", row.step)
      .value("attempt", row.attempt)
      .value("time", row.time)
      .value("dt", row.dt)
      .text("declared_stage", levelSetMaintenanceDeclaredStageName(row.declared_stage))
      .value("algebraic_state_revision_before", row.algebraic_state_revision_before)
      .value("algebraic_state_revision_after", row.algebraic_state_revision_after)
      .value("snapshot_set_revision_before", row.snapshot_set_revision_before)
      .value("snapshot_set_revision_after", row.snapshot_set_revision_after)
      .value("mesh_topology_set_revision_before", row.mesh_topology_set_revision_before)
      .value("mesh_topology_set_revision_after", row.mesh_topology_set_revision_after)
      .value("cut_topology_set_revision_before", row.cut_topology_set_revision_before)
      .value("cut_topology_set_revision_after", row.cut_topology_set_revision_after)
      .value("extension_map_revision_before", row.extension_map_revision_before.value_or(0u))
      .value("extension_map_revision_after", row.extension_map_revision_after.value_or(0u))
      .value("authoritative_functional_count", row.before.size())
      .text("numerical_work_sign", "potential_after_minus_before")
      .value("numerical_work", row.numerical_work)
      .value("accepted_numerical_work", row.accepted_numerical_work)
      .boolean("modeled_energy_work_available", row.modeled_energy_numerical_work.has_value())
      .value("modeled_energy_numerical_work", row.modeled_energy_numerical_work.value_or(0.0))
      .value("accepted_modeled_energy_numerical_work", row.accepted_modeled_energy_numerical_work.value_or(0.0))
      .text("modeled_energy_classification",
            row.substage == LevelSetMaintenanceWorkSubstage::Transport ? "physical_transport_change"
                                                                       : "numerical_maintenance_change")
      .text("legacy_account_scope", "maintenance_surface_wall_volume_potential")
      .text("modeled_account_scope", "kinetic_gravitational_surface_young_wall")
      .boolean("complete_time_discrete_energy_identity", false);
  // One child line per interface, with the keys that identify the row.
  for (std::size_t index = 0u; index < row.before.size(); ++index) {
    const auto& before = row.before[index];
    const auto& after = row.after[index];
    auto& interface_line =
        log.child("level_set_maintenance_work_interface", {"transaction_id", "substage", "step", "attempt"});
    interface_line.value("interface_marker", before.interface_marker)
        .value("before_snapshot_revision", before.snapshot_revision)
        .value("after_snapshot_revision", after.snapshot_revision)
        .value("before_mesh_topology_revision", before.mesh_topology_revision)
        .value("after_mesh_topology_revision", after.mesh_topology_revision)
        .value("before_cut_topology_revision", before.cut_topology_revision)
        .value("after_cut_topology_revision", after.cut_topology_revision)
        .value("before_liquid_volume", before.liquid_volume)
        .value("after_liquid_volume", after.liquid_volume)
        .value("before_liquid_gas_area", before.liquid_gas_area)
        .value("after_liquid_gas_area", after.liquid_gas_area)
        .value("before_wetted_wall_area", before.wetted_wall_area)
        .value("after_wetted_wall_area", after.wetted_wall_area)
        .value("before_contact_measure", before.contact_measure)
        .value("after_contact_measure", after.contact_measure)
        .value("before_surface_energy", before.surface_energy)
        .value("after_surface_energy", after.surface_energy)
        .value("before_young_wall_energy", before.young_wall_energy)
        .value("after_young_wall_energy", after.young_wall_energy)
        .value("before_volume_constraint_potential", before.volume_constraint_potential)
        .value("after_volume_constraint_potential", after.volume_constraint_potential)
        .value("before_total_potential", before.total_potential)
        .value("after_total_potential", after.total_potential);
    if (before.modeled_stored_energy.has_value() && after.modeled_stored_energy.has_value()) {
      interface_line.value("before_kinetic_energy", *before.kinetic_energy)
          .value("after_kinetic_energy", *after.kinetic_energy)
          .value("before_gravitational_energy", *before.gravitational_energy)
          .value("after_gravitational_energy", *after.gravitational_energy)
          .value("before_gravitational_potential_power", *before.gravitational_potential_power)
          .value("after_gravitational_potential_power", *after.gravitational_potential_power)
          .value("before_modeled_stored_energy", *before.modeled_stored_energy)
          .value("after_modeled_stored_energy", *after.modeled_stored_energy);
    }
    if (before.surface_wall_potential_power.has_value() && after.surface_wall_potential_power.has_value()) {
      interface_line.value("before_surface_wall_potential_power", *before.surface_wall_potential_power)
          .value("after_surface_wall_potential_power", *after.surface_wall_potential_power)
          .value("before_volume_constraint_potential_power", *before.volume_constraint_potential_power)
          .value("after_volume_constraint_potential_power", *after.volume_constraint_potential_power);
    }
    if (before.bulk_viscous_dissipation_rate.has_value() && after.bulk_viscous_dissipation_rate.has_value()) {
      interface_line.value("before_bulk_viscous_dissipation_rate", *before.bulk_viscous_dissipation_rate)
          .value("after_bulk_viscous_dissipation_rate", *after.bulk_viscous_dissipation_rate);
    }
    if (before.external_pressure_power.has_value() && after.external_pressure_power.has_value()) {
      interface_line.value("before_external_pressure_power", *before.external_pressure_power)
          .value("after_external_pressure_power", *after.external_pressure_power);
    }
  }
  return log;
}

void logLevelSetMaintenanceWorkRows(
    std::span<const LevelSetMaintenanceWorkRow> rows)
{
  // Map entry R2.144; rank 0 prints it.
  for (const auto& row : rows) {
    logRecord(levelSetMaintenanceWorkRowRecord(row));
  }
}

void logLevelSetMaintenanceWorkAttempts(
    std::span<const
        LevelSetMaintenanceWorkAttempt> attempts)
{
  // Map entry R2.143; rank 0 prints it.
  for (const auto& attempt : attempts) {
    LogRecord log("LevelSetMaintenanceWorkLedger", "work attempt");
    const auto append_substage =
        [&log](
            std::string_view name,
            const LevelSetMaintenanceModeledEnergySubstage& substage) {
          const std::string prefix(name);
          log.value(prefix + "_row_count", substage.row_count)
              .boolean(prefix + "_modeled_energy_available", substage.modeled_energy_change.has_value())
              .value(prefix + "_modeled_energy_change", substage.modeled_energy_change.value_or(0.0))
              .value("accepted_" + prefix + "_modeled_energy_change",
                     substage.accepted_modeled_energy_change.value_or(0.0));
        };
    log.diagnostic("level_set_maintenance_work_attempt")
        .rootOnly()
        .value("transaction_id", attempt.transaction_id)
        .text("status", levelSetMaintenanceWorkStatusName(attempt.status))
        .value("step", attempt.step)
        .value("attempt", attempt.attempt)
        .value("time", attempt.time)
        .value("dt", attempt.dt)
        .text("declared_stage", levelSetMaintenanceDeclaredStageName(attempt.declared_stage))
        .value("extension_map_revision", attempt.extension_map_revision.value_or(0u))
        .value("row_count", attempt.row_count)
        .value("numerical_work", attempt.numerical_work)
        .value("accepted_numerical_work", attempt.accepted_numerical_work)
        .boolean("modeled_energy_work_available", attempt.modeled_energy_numerical_work.has_value())
        .value("modeled_energy_numerical_work", attempt.modeled_energy_numerical_work.value_or(0.0))
        .value("accepted_modeled_energy_numerical_work", attempt.accepted_modeled_energy_numerical_work.value_or(0.0))
        .text("accepted_account_publication",
              attempt.status == LevelSetMaintenanceWorkStatus::Accepted ? "accepted_rows" : "zero_for_rejection")
        .boolean("physical_transport_excluded_from_numerical_maintenance", true);
    const auto& breakdown = attempt.modeled_energy_breakdown;
    append_substage("transport", breakdown.transport);
    append_substage("limiting", breakdown.limiting);
    append_substage(
        "reinitialization", breakdown.reinitialization);
    append_substage(
        "geometry_reconciliation",
        breakdown.geometry_reconciliation);
    append_substage(
        "global_correction", breakdown.global_correction);
    append_substage(
        "numerical_maintenance_total",
        breakdown.numerical_maintenance_total);
    log.boolean("complete_time_discrete_energy_identity", false);
    logRecord(log);
  }
}

void logLevelSetMaintenanceAcceptedStepEnergyAccount(
    const LevelSetMaintenanceAcceptedStepEnergyAccount& account)
{
  // Map entry R2.135; rank 0 prints it.
  LogRecord log("LevelSetMaintenanceWorkLedger", "accepted-step energy account");
  const auto append_substage =
      [&log](
          std::string_view name,
          const LevelSetMaintenanceModeledEnergySubstage& substage) {
        const std::string prefix(name);
        log.value(prefix + "_row_count", substage.row_count)
            .boolean(prefix + "_modeled_energy_available", substage.accepted_modeled_energy_change.has_value())
            .value("accepted_" + prefix + "_modeled_energy_change",
                   substage.accepted_modeled_energy_change.value_or(0.0));
      };
  const auto append_endpoint =
      [&log](
          std::string_view name,
          const std::optional<LevelSetMaintenanceAcceptedStepEnergyAccount::
                  Endpoint>& endpoint) {
        const std::string prefix(name);
        log.boolean(prefix + "_available", endpoint.has_value());
        if (!endpoint.has_value()) {
          return;
        }
        log.value(prefix + "_algebraic_state_revision", endpoint->algebraic_state_revision)
            .value(prefix + "_snapshot_set_revision", endpoint->snapshot_set_revision)
            .value(prefix + "_mesh_topology_set_revision", endpoint->mesh_topology_set_revision)
            .value(prefix + "_cut_topology_set_revision", endpoint->cut_topology_set_revision)
            .value(prefix + "_extension_map_revision", endpoint->extension_map_revision.value_or(0u))
            .boolean(prefix + "_modeled_energy_available", endpoint->modeled_stored_energy.has_value())
            .boolean(prefix + "_gravitational_potential_power_available",
                     endpoint->gravitational_potential_power.has_value())
            .boolean(prefix + "_surface_wall_potential_power_available",
                     endpoint->surface_wall_potential_power.has_value())
            .boolean(prefix + "_bulk_viscous_dissipation_available",
                     endpoint->bulk_viscous_dissipation_rate.has_value())
            .boolean(prefix + "_external_pressure_power_available", endpoint->external_pressure_power.has_value())
            .value(prefix + "_kinetic_energy", endpoint->kinetic_energy.value_or(0.0))
            .value(prefix + "_gravitational_energy", endpoint->gravitational_energy.value_or(0.0))
            .value(prefix + "_gravitational_potential_power", endpoint->gravitational_potential_power.value_or(0.0))
            .value(prefix + "_liquid_gas_surface_energy", endpoint->liquid_gas_surface_energy)
            .value(prefix + "_solid_liquid_wall_energy", endpoint->solid_liquid_wall_energy)
            .value(prefix + "_surface_wall_potential_power", endpoint->surface_wall_potential_power.value_or(0.0))
            .value(prefix + "_volume_constraint_potential_power",
                   endpoint->volume_constraint_potential_power.value_or(0.0))
            .value(prefix + "_bulk_viscous_dissipation_rate", endpoint->bulk_viscous_dissipation_rate.value_or(0.0))
            .value(prefix + "_external_pressure_power", endpoint->external_pressure_power.value_or(0.0))
            .value(prefix + "_modeled_stored_energy", endpoint->modeled_stored_energy.value_or(0.0));
      };
  log.diagnostic("level_set_maintenance_accepted_step_energy")
      .rootOnly()
      .value("step", account.step)
      .value("attempt", account.attempt)
      .value("time", account.time)
      .value("dt", account.dt)
      .value("transaction_count", account.transaction_count)
      .value("row_count", account.row_count)
      .boolean("physical_transport_excluded_from_numerical_maintenance", true);
  const auto& breakdown = account.modeled_energy_breakdown;
  append_substage("transport", breakdown.transport);
  append_substage("limiting", breakdown.limiting);
  append_substage(
      "reinitialization", breakdown.reinitialization);
  append_substage(
      "geometry_reconciliation",
      breakdown.geometry_reconciliation);
  append_substage(
      "global_correction", breakdown.global_correction);
  append_substage(
      "numerical_maintenance_total",
      breakdown.numerical_maintenance_total);
  append_endpoint(
      "maintenance_start", account.maintenance_start);
  append_endpoint("post_transport", account.post_transport);
  append_endpoint("maintenance_end", account.maintenance_end);
  log.boolean("physical_transport_endpoint_residual_available",
              account.physical_transport_endpoint_residual.has_value())
      .value("physical_transport_endpoint_residual", account.physical_transport_endpoint_residual.value_or(0.0))
      .boolean("numerical_maintenance_endpoint_residual_available",
               account.numerical_maintenance_endpoint_residual.has_value())
      .value("numerical_maintenance_endpoint_residual", account.numerical_maintenance_endpoint_residual.value_or(0.0))
      .boolean("complete_time_discrete_energy_identity", false);
  logRecord(log);
}

void logLevelSetMaintenanceWorkPublishedSince(
    const LevelSetMaintenanceWorkLedger& ledger,
    const LevelSetMaintenanceWorkMark& mark,
    LevelSetMaintenanceWorkStatus status)
{
  const bool accepted =
      status == LevelSetMaintenanceWorkStatus::Accepted;
  if (!accepted && status != LevelSetMaintenanceWorkStatus::Rejected) {
    throw std::invalid_argument(
        "Level-set maintenance work publication logs name an accepted or rejected history.");
  }
  const auto& attempts =
      accepted ? ledger.acceptedAttempts() : ledger.rejectedAttempts();
  const auto& rows =
      accepted ? ledger.acceptedRows() : ledger.rejectedRows();
  const auto attempt_begin =
      accepted ? mark.accepted_attempts : mark.rejected_attempts;
  const auto row_begin = accepted ? mark.accepted_rows : mark.rejected_rows;
  if (attempts.size() > attempt_begin) {
    logLevelSetMaintenanceWorkAttempts(
        std::span<const LevelSetMaintenanceWorkAttempt>(attempts)
            .subspan(attempt_begin));
  }
  if (rows.size() > row_begin) {
    logLevelSetMaintenanceWorkRows(
        std::span<const LevelSetMaintenanceWorkRow>(rows).subspan(
            row_begin));
  }
}

void stageLevelSetMaintenanceWorkRow(
    LevelSetMaintenanceWorkLedger& ledger,
    const LevelSetMaintenanceWorkRowSource& source,
    std::vector<LevelSetAuthoritativeFunctionalValue>& current,
    MaintenanceSubstage substage,
    std::span<const Real> before_candidate,
    std::span<const Real> after_candidate)
{
  if (!source.functional_values || !source.algebraic_state_revision ||
      !source.extension_map_revision) {
    throw std::invalid_argument(
        "Level-set maintenance work rows require functional, algebraic-revision and extension-revision sources.");
  }
  auto after_values = source.functional_values(after_candidate);
  const auto revision_before =
      source.algebraic_state_revision(before_candidate);
  const auto revision_after =
      source.algebraic_state_revision(after_candidate);
  const auto extension_map_revision_after =
      source.extension_map_revision();
  ledger.stageRow(
      substage,
      revision_before,
      revision_after,
      current,
      after_values,
      extension_map_revision_after);
  current = std::move(after_values);
}

} // namespace level_set
} // namespace FE
} // namespace svmp
