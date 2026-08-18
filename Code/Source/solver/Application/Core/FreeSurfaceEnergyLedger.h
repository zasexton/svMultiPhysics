/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace application::core {

enum class FreeSurfaceEnergyAttemptStatus : std::uint8_t {
  Trial,
  Accepted,
  Rejected
};

enum class FreeSurfaceEnergyRejectionReason : std::uint8_t {
  None,
  NonlinearSolveFailure,
  StepControllerRejection,
  PreacceptRejection,
  TopologyChange,
  MaintenanceRollback,
  PublicationFailure
};

enum class FreeSurfaceGasEnergyApplicability : std::uint8_t {
  Unspecified,
  Active,
  NotApplicable
};

enum class FreeSurfaceEnergyChannelApplicability : std::uint8_t {
  Unspecified,
  Produced,
  NotApplicable
};

/**
 * Time-integration identity carried by an energy attempt.
 *
 * GeneralizedAlpha is an explicit provenance value, not a supported ledger
 * scheme. The fixed-topology ledger currently stages only BackwardEuler.
 */
enum class FreeSurfaceEnergyTemporalScheme : std::uint8_t {
  Unspecified,
  BackwardEuler,
  GeneralizedAlpha
};

/**
 * Exactly one named production owner and applicability decision for one
 * ledger channel.
 *
 * A NotApplicable channel remains explicit and must carry an exact-zero
 * value. Produced channels may also evaluate to zero.
 */
struct FreeSurfaceEnergyChannelSource {
  FreeSurfaceEnergyChannelApplicability applicability{
      FreeSurfaceEnergyChannelApplicability::Unspecified};
  std::string owner{};

  bool operator==(const FreeSurfaceEnergyChannelSource&) const = default;
};

struct FreeSurfaceStoredEnergy {
  double kinetic{std::numeric_limits<double>::quiet_NaN()};
  double gravitational{std::numeric_limits<double>::quiet_NaN()};
  double liquid_gas_surface{std::numeric_limits<double>::quiet_NaN()};
  double solid_liquid_wall{std::numeric_limits<double>::quiet_NaN()};
  FreeSurfaceGasEnergyApplicability gas_applicability{
      FreeSurfaceGasEnergyApplicability::Unspecified};
  double gas_or_compressibility{
      std::numeric_limits<double>::quiet_NaN()};
};

struct FreeSurfacePhysicalDissipationRate {
  double bulk_viscous{std::numeric_limits<double>::quiet_NaN()};
  double navier_slip{std::numeric_limits<double>::quiet_NaN()};
  double line_friction{std::numeric_limits<double>::quiet_NaN()};
};

struct FreeSurfaceExternalWork {
  double pressure{std::numeric_limits<double>::quiet_NaN()};
  double body_force{std::numeric_limits<double>::quiet_NaN()};
  double imposed_traction{std::numeric_limits<double>::quiet_NaN()};
  double open_boundary_flux{std::numeric_limits<double>::quiet_NaN()};
};

struct FreeSurfaceNumericalWork {
  double time_discretization{
      std::numeric_limits<double>::quiet_NaN()};
  /**
   * K^n on the endpoint retained domain minus K^n on the preceding accepted
   * retained domain.
   */
  double kinetic_domain_transport{
      std::numeric_limits<double>::quiet_NaN()};
  /**
   * Gravitational-energy change minus dt times endpoint gravitational
   * potential power.
   */
  double gravitational_transport_coupling{
      std::numeric_limits<double>::quiet_NaN()};
  double convection{std::numeric_limits<double>::quiet_NaN()};
  double pressure_continuity{
      std::numeric_limits<double>::quiet_NaN()};
  double surface_transport_coupling{
      std::numeric_limits<double>::quiet_NaN()};
  double weak_boundary{std::numeric_limits<double>::quiet_NaN()};
  double vms_pspg{std::numeric_limits<double>::quiet_NaN()};
  double cut_stabilization{std::numeric_limits<double>::quiet_NaN()};
  double ghost_penalty{std::numeric_limits<double>::quiet_NaN()};
  double aggregation{std::numeric_limits<double>::quiet_NaN()};
  double extension{std::numeric_limits<double>::quiet_NaN()};
  double pruning{std::numeric_limits<double>::quiet_NaN()};
  double limiting{std::numeric_limits<double>::quiet_NaN()};
  double redistancing{std::numeric_limits<double>::quiet_NaN()};
  double local_reconciliation{std::numeric_limits<double>::quiet_NaN()};
  double global_correction{std::numeric_limits<double>::quiet_NaN()};
};

struct FreeSurfaceStoredEnergySources {
  FreeSurfaceEnergyChannelSource kinetic{};
  FreeSurfaceEnergyChannelSource gravitational{};
  FreeSurfaceEnergyChannelSource liquid_gas_surface{};
  FreeSurfaceEnergyChannelSource solid_liquid_wall{};
  FreeSurfaceEnergyChannelSource gas_or_compressibility{};

  bool operator==(const FreeSurfaceStoredEnergySources&) const = default;
};

struct FreeSurfacePhysicalDissipationSources {
  FreeSurfaceEnergyChannelSource bulk_viscous{};
  FreeSurfaceEnergyChannelSource navier_slip{};
  FreeSurfaceEnergyChannelSource line_friction{};

  bool operator==(
      const FreeSurfacePhysicalDissipationSources&) const = default;
};

struct FreeSurfaceExternalWorkSources {
  FreeSurfaceEnergyChannelSource pressure{};
  FreeSurfaceEnergyChannelSource body_force{};
  FreeSurfaceEnergyChannelSource imposed_traction{};
  FreeSurfaceEnergyChannelSource open_boundary_flux{};

  bool operator==(const FreeSurfaceExternalWorkSources&) const = default;
};

struct FreeSurfaceNumericalWorkSources {
  FreeSurfaceEnergyChannelSource time_discretization{};
  FreeSurfaceEnergyChannelSource kinetic_domain_transport{};
  FreeSurfaceEnergyChannelSource gravitational_transport_coupling{};
  FreeSurfaceEnergyChannelSource convection{};
  FreeSurfaceEnergyChannelSource pressure_continuity{};
  FreeSurfaceEnergyChannelSource surface_transport_coupling{};
  FreeSurfaceEnergyChannelSource weak_boundary{};
  FreeSurfaceEnergyChannelSource vms_pspg{};
  FreeSurfaceEnergyChannelSource cut_stabilization{};
  FreeSurfaceEnergyChannelSource ghost_penalty{};
  FreeSurfaceEnergyChannelSource aggregation{};
  FreeSurfaceEnergyChannelSource extension{};
  FreeSurfaceEnergyChannelSource pruning{};
  FreeSurfaceEnergyChannelSource limiting{};
  FreeSurfaceEnergyChannelSource redistancing{};
  FreeSurfaceEnergyChannelSource local_reconciliation{};
  FreeSurfaceEnergyChannelSource global_correction{};

  bool operator==(const FreeSurfaceNumericalWorkSources&) const = default;
};

struct FreeSurfaceEnergyChannelSources {
  FreeSurfaceStoredEnergySources stored{};
  FreeSurfacePhysicalDissipationSources dissipation{};
  FreeSurfaceExternalWorkSources external{};
  FreeSurfaceNumericalWorkSources numerical{};

  bool operator==(const FreeSurfaceEnergyChannelSources&) const = default;
};

struct FreeSurfaceEnergyAttemptMetadata {
  std::uint64_t transaction_id{0};
  std::uint64_t step{0};
  std::uint64_t attempt{0};
  double time_before{std::numeric_limits<double>::quiet_NaN()};
  double time_after{std::numeric_limits<double>::quiet_NaN()};
  double dt{std::numeric_limits<double>::quiet_NaN()};
  FreeSurfaceEnergyTemporalScheme temporal_scheme{
      FreeSurfaceEnergyTemporalScheme::Unspecified};
  // May remain unavailable only for an unstaged rejected attempt.
  double physical_evaluation_time{
      std::numeric_limits<double>::quiet_NaN()};
  // One is the accepted endpoint; other stages are not supported here.
  double physical_evaluation_stage_fraction{
      std::numeric_limits<double>::quiet_NaN()};
  std::uint64_t algebraic_state_revision_before{0};
  // May remain zero only for an unstaged rejected attempt.
  std::uint64_t
      physical_endpoint_algebraic_state_revision{0};
  // May remain zero only for an unstaged rejected attempt.
  std::uint64_t algebraic_state_revision_after{0};
  std::uint64_t snapshot_set_revision_before{0};
  // May remain zero only for an unstaged rejected attempt.
  std::uint64_t physical_endpoint_snapshot_set_revision{0};
  // May remain zero only for an unstaged rejected attempt.
  std::uint64_t snapshot_set_revision_after{0};
  // These are nonzero fingerprints of the complete declaration set. Raw
  // per-mesh topology epochs, including the valid initial epoch zero, are
  // folded into the fingerprint rather than stored here as sentinels.
  std::uint64_t mesh_topology_set_revision_before{0};
  // May remain zero only for an unstaged rejected attempt.
  std::uint64_t
      physical_endpoint_mesh_topology_set_revision{0};
  // May remain zero only for an unstaged rejected attempt.
  std::uint64_t mesh_topology_set_revision_after{0};
  std::uint64_t cut_topology_set_revision_before{0};
  // May remain zero only for an unstaged rejected attempt.
  std::uint64_t
      physical_endpoint_cut_topology_set_revision{0};
  // May remain zero only for an unstaged rejected attempt.
  std::uint64_t cut_topology_set_revision_after{0};
  // A staged balance requires all three extension revisions to be either
  // present and nonzero or absent.
  std::optional<std::uint64_t> extension_map_revision_before{};
  std::optional<std::uint64_t>
      physical_endpoint_extension_map_revision{};
  std::optional<std::uint64_t> extension_map_revision_after{};
};

struct FreeSurfaceEnergyAttempt {
  FreeSurfaceEnergyAttemptStatus status{
      FreeSurfaceEnergyAttemptStatus::Trial};
  FreeSurfaceEnergyRejectionReason rejection_reason{
      FreeSurfaceEnergyRejectionReason::None};
  FreeSurfaceEnergyAttemptMetadata metadata{};
  bool balance_staged{false};
  FreeSurfaceStoredEnergy before{};
  FreeSurfaceStoredEnergy physical_endpoint_before_maintenance{};
  FreeSurfaceStoredEnergy after{};
  FreeSurfacePhysicalDissipationRate dissipation_rate{};
  FreeSurfaceExternalWork external_work{};
  FreeSurfaceNumericalWork numerical_work{};
  FreeSurfaceEnergyChannelSources channel_sources{};
  double stored_energy_before{0.0};
  double stored_energy_physical_endpoint_before_maintenance{0.0};
  double stored_energy_after{0.0};
  double physical_stored_energy_change{0.0};
  double maintenance_stored_energy_change{0.0};
  double stored_energy_change{0.0};
  double integrated_physical_dissipation{0.0};
  double total_external_work{0.0};
  double total_numerical_work{0.0};
  double trial_balance_residual{0.0};
  double accepted_stored_energy_change{0.0};
  double accepted_physical_stored_energy_change{0.0};
  double accepted_maintenance_stored_energy_change{0.0};
  double accepted_integrated_physical_dissipation{0.0};
  double accepted_external_work{0.0};
  double accepted_numerical_work{0.0};
  double accepted_balance_residual{0.0};
};

/**
 * Transactional low-level ledger for a fixed-topology backward-Euler
 * free-surface energy balance.
 *
 * Dissipation fields are endpoint rates and are multiplied by dt. External
 * and numerical-work fields are already step-integrated, with positive
 * values denoting work added to the modeled stored energy. Negative numerical
 * work therefore denotes a numerical energy loss. Every channel must be
 * supplied explicitly, including exact zeros. A gas/compressibility channel
 * must be declared Active or NotApplicable; the latter requires an
 * exact-zero value. Every channel also carries exactly one nonempty owner
 * name and a Produced or NotApplicable decision. NotApplicable channels
 * require exact-zero values, while a Produced channel may legitimately
 * evaluate to zero.
 * Every attempt distinguishes the preceding post-maintenance accepted
 * endpoint, the physical pre-maintenance endpoint, and the resulting
 * post-maintenance accepted endpoint. Physical rates and work are evaluated
 * at the middle state. A staged record must identify that physical state as
 * the backward-Euler accepted endpoint at time_after and stage fraction one;
 * generalized-alpha and nonendpoint data fail closed. Maintenance numerical
 * work accounts for the middle to final state change. Consecutive accepted
 * records form one exact provenance chain from the final state of one attempt
 * to the initial state of the next: time, algebraic state, snapshot, topology,
 * extension map, and stored-energy values must all agree.
 * After acceptance, the next record starts the next step at attempt one.
 * After rejection, only the next attempt on the identical accepted starting
 * endpoint is admissible; its trial duration may change under step control.
 * Channel ownership and applicability are immutable within one ledger
 * history.
 * An attempt rejected before all balance channels are available is recorded
 * with balance_staged=false. Its unavailable diagnostic balance values remain
 * NaN rather than being presented as physical zeros, while every accepted
 * contribution is exact zero.
 *
 * This class validates and publishes records. It does not assemble any
 * channel and does not establish a discrete energy theorem.
 */
class FreeSurfaceEnergyLedger {
public:
  void beginAttempt(FreeSurfaceEnergyAttemptMetadata metadata);

  void stageBalance(
      FreeSurfaceStoredEnergy before,
      FreeSurfaceStoredEnergy physical_endpoint_before_maintenance,
      FreeSurfaceStoredEnergy after,
      FreeSurfacePhysicalDissipationRate dissipation_rate,
      FreeSurfaceExternalWork external_work,
      FreeSurfaceNumericalWork numerical_work,
      FreeSurfaceEnergyChannelSources channel_sources);

  void commitAttempt();
  void rejectAttempt(FreeSurfaceEnergyRejectionReason reason);
  /** Record a rejection that occurs before a complete balance can be staged. */
  void rejectUnstagedAttempt(
      FreeSurfaceEnergyRejectionReason reason);

  [[nodiscard]] bool attemptActive() const noexcept;
  [[nodiscard]] const FreeSurfaceEnergyAttemptMetadata*
  activeAttempt() const noexcept;
  [[nodiscard]] const FreeSurfaceEnergyAttempt*
  trialBalance() const noexcept;
  [[nodiscard]] const std::vector<FreeSurfaceEnergyAttempt>&
  acceptedAttempts() const noexcept;
  [[nodiscard]] const std::vector<FreeSurfaceEnergyAttempt>&
  rejectedAttempts() const noexcept;

private:
  std::optional<FreeSurfaceEnergyAttemptMetadata> active_attempt_{};
  std::optional<FreeSurfaceEnergyAttempt> trial_balance_{};
  std::optional<FreeSurfaceEnergyAttemptStatus>
      last_published_status_{};
  std::optional<FreeSurfaceEnergyAttemptMetadata>
      last_published_metadata_{};
  std::uint64_t last_transaction_id_{0};
  std::vector<FreeSurfaceEnergyAttempt> accepted_attempts_{};
  std::vector<FreeSurfaceEnergyAttempt> rejected_attempts_{};
};

} // namespace application::core
