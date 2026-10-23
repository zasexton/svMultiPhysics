/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_ENERGYBALANCELEDGER_H
#define SVMP_FE_SYSTEMS_ENERGYBALANCELEDGER_H

/**
 * @file EnergyBalanceLedger.h
 * @brief Transactional ledger of a step energy balance with named channels
 *        and declared identities.
 *
 * A formulation declares its channels as data (EnergyBalanceCatalogue): the
 * stored energies, the dissipation rates, the external works and the
 * numerical works, each with a name, in the order in which they are summed.
 * A derived channel may declare its identity (EnergyChannelIdentity), and the
 * ledger evaluates it from the staged stored energies and named inputs.  The
 * ledger checks every staged record, sums each kind in declaration order and
 * keeps the accepted and rejected history:
 *
 *   residual = (E_after - E_before) + dt * sum(rates) - sum(external work)
 *              - sum(numerical work).
 *
 * It does not assemble any channel and does not establish a discrete energy
 * theorem.
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

/// What a channel measures.  The ledger keeps one list of channels per kind.
enum class EnergyChannelKind : std::uint8_t {
    Stored,          ///< an energy at the step's endpoints
    DissipationRate, ///< a nonnegative rate at the physical endpoint (times dt)
    ExternalWork,    ///< step work added from outside the modeled system
    NumericalWork    ///< step work added by the discretization (negative = loss)
};

inline constexpr std::array<EnergyChannelKind, 4> kEnergyChannelKinds{
    EnergyChannelKind::Stored,
    EnergyChannelKind::DissipationRate,
    EnergyChannelKind::ExternalWork,
    EnergyChannelKind::NumericalWork};

/// "stored", "dissipation", "external" or "numerical".
[[nodiscard]] std::string_view energyChannelKindPrefix(EnergyChannelKind kind) noexcept;

/// Position of @p kind in kEnergyChannelKinds.
[[nodiscard]] constexpr std::size_t energyChannelKindIndex(EnergyChannelKind kind) noexcept
{
    return static_cast<std::size_t>(kind);
}

enum class EnergyChannelApplicability : std::uint8_t {
    Unspecified,
    Produced,
    NotApplicable
};

/// "produced", "not_applicable", "unspecified" (or "invalid").
[[nodiscard]] const char* energyChannelApplicabilityName(
    EnergyChannelApplicability applicability) noexcept;

/**
 * @brief Exactly one named production owner and applicability decision for
 *        one channel.
 *
 * A NotApplicable channel remains explicit and must carry an exact-zero
 * value; a Produced channel may also evaluate to zero.
 */
struct EnergyChannelSource {
    EnergyChannelApplicability applicability{EnergyChannelApplicability::Unspecified};
    std::string owner{};

    bool operator==(const EnergyChannelSource&) const = default;
};

/// Appends " <label>_applicability=<name> <label>_owner=<quoted owner>".
void appendEnergyChannelSource(std::ostream& stream,
                               std::string_view label,
                               const EnergyChannelSource& source);

/**
 * @brief How the ledger evaluates a derived (non-stored) channel.
 *
 * With dt the attempt's duration, E_c(x) the staged stored energy c at
 * endpoint x and P the named input:
 * - Supplied: the caller stages the value;
 * - StoredTransport: (sum_c E_c(physical endpoint) - sum_c E_c(before))
 *   - dt * P, over the listed stored channels in their listed order;
 * - DomainTransport: P - E_c(before) for the one listed stored channel;
 * - StepWork: dt * P.
 */
struct EnergyChannelIdentity {
    enum class Rule : std::uint8_t { Supplied, StoredTransport, DomainTransport, StepWork };

    Rule rule{Rule::Supplied};
    std::vector<std::string> stored{};
    std::string input{};

    bool operator==(const EnergyChannelIdentity&) const = default;
};

struct EnergyChannelDeclaration {
    std::string name{};
    /// The staged value must not be negative (dissipation rates always).
    bool nonnegative{false};
    EnergyChannelIdentity identity{};

    bool operator==(const EnergyChannelDeclaration&) const = default;
};

/**
 * @brief The declared channels of one energy balance.
 *
 * @p subject names the balance in messages ("free-surface energy").  Names
 * are unique within a kind; only non-stored channels declare identities, and
 * an identity names declared stored channels and one input.
 */
class EnergyBalanceCatalogue {
public:
    EnergyBalanceCatalogue() = default;
    EnergyBalanceCatalogue(std::string subject,
                           std::array<std::vector<EnergyChannelDeclaration>, 4> channels);

    [[nodiscard]] const std::string& subject() const noexcept { return subject_; }
    [[nodiscard]] std::span<const EnergyChannelDeclaration> channels(
        EnergyChannelKind kind) const noexcept;
    [[nodiscard]] std::size_t count(EnergyChannelKind kind) const noexcept;
    /// Position of channel @p name within @p kind, if declared.
    [[nodiscard]] std::optional<std::size_t> find(EnergyChannelKind kind,
                                                  std::string_view name) const noexcept;
    /// Position of channel @p name within @p kind; throws if not declared.
    [[nodiscard]] std::size_t position(EnergyChannelKind kind, std::string_view name) const;
    /// "<kind prefix>.<name>", for example "stored.kinetic".
    [[nodiscard]] std::string qualifiedName(EnergyChannelKind kind, std::size_t position) const;

    bool operator==(const EnergyBalanceCatalogue&) const = default;

private:
    std::string subject_{};
    std::array<std::vector<EnergyChannelDeclaration>, 4> channels_{};
};

/// Sum of @p values in their order (the first value, then each added).
[[nodiscard]] double sumInOrder(std::span<const double> values) noexcept;

/// sum(after) - before[0] - before[1] - ..., evaluated left to right.
[[nodiscard]] double energyChange(std::span<const double> after,
                                  std::span<const double> before) noexcept;

/// @p energy_change - dt * @p power: the transport-coupling work of a stored
/// energy whose endpoint power is declared.
[[nodiscard]] double transportCouplingWork(double energy_change, double dt, double power) noexcept;

/**
 * @brief The pairing R(x)[x] of one declared work channel's residual with the
 *        converged state x.
 *
 * A produced channel names the operator whose residual was assembled; an
 * inapplicable one has no operator and an exact-zero pairing.
 */
struct ResidualWorkPairing {
    std::string channel{};
    bool produced{false};
    std::string owner_component{};
    std::string operator_tag{};
    double residual_state_pairing{0.0};
};

/**
 * @brief The step works -dt * R(x)[x] of @p pairings, one per name of
 *        @p channels and in that order (exact zero for an inapplicable row).
 *
 * Every declared channel appears exactly once, with a named owner and a
 * finite pairing.  @p subject names the works in messages.
 */
[[nodiscard]] std::vector<double> evaluateResidualStepWork(
    std::span<const ResidualWorkPairing> pairings,
    std::span<const std::string> channels,
    double dt,
    std::string_view subject);

enum class EnergyBalanceAttemptStatus : std::uint8_t { Trial, Accepted, Rejected };

enum class EnergyBalanceRejectionReason : std::uint8_t {
    None,
    NonlinearSolveFailure,
    StepControllerRejection,
    PreacceptRejection,
    TopologyChange,
    MaintenanceRollback,
    PublicationFailure
};

/// "nonlinear_solve_failure", ..., "none" (or "invalid").
[[nodiscard]] const char* energyBalanceRejectionReasonName(
    EnergyBalanceRejectionReason reason) noexcept;

/**
 * Time-integration identity carried by an attempt.
 *
 * GeneralizedAlpha is an explicit provenance value, not a supported ledger
 * scheme: the ledger stages only BackwardEuler endpoint balances.
 */
enum class EnergyBalanceTemporalScheme : std::uint8_t {
    Unspecified,
    BackwardEuler,
    GeneralizedAlpha
};

struct EnergyBalanceAttemptMetadata {
    std::uint64_t transaction_id{0};
    std::uint64_t step{0};
    std::uint64_t attempt{0};
    double time_before{std::numeric_limits<double>::quiet_NaN()};
    double time_after{std::numeric_limits<double>::quiet_NaN()};
    double dt{std::numeric_limits<double>::quiet_NaN()};
    EnergyBalanceTemporalScheme temporal_scheme{EnergyBalanceTemporalScheme::Unspecified};
    // May remain unavailable only for an unstaged rejected attempt.
    double physical_evaluation_time{std::numeric_limits<double>::quiet_NaN()};
    // One is the accepted endpoint; other stages are not supported here.
    double physical_evaluation_stage_fraction{std::numeric_limits<double>::quiet_NaN()};
    std::uint64_t algebraic_state_revision_before{0};
    // May remain zero only for an unstaged rejected attempt.
    std::uint64_t physical_endpoint_algebraic_state_revision{0};
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
    std::uint64_t physical_endpoint_mesh_topology_set_revision{0};
    // May remain zero only for an unstaged rejected attempt.
    std::uint64_t mesh_topology_set_revision_after{0};
    std::uint64_t cut_topology_set_revision_before{0};
    // May remain zero only for an unstaged rejected attempt.
    std::uint64_t physical_endpoint_cut_topology_set_revision{0};
    // May remain zero only for an unstaged rejected attempt.
    std::uint64_t cut_topology_set_revision_after{0};
    // A staged balance requires all three extension revisions to be either
    // present and nonzero or absent.
    std::optional<std::uint64_t> extension_map_revision_before{};
    std::optional<std::uint64_t> physical_endpoint_extension_map_revision{};
    std::optional<std::uint64_t> extension_map_revision_after{};
};

/**
 * @brief The staged values of one balance, one per declared channel of each
 *        kind in declaration order.
 *
 * Stored channels have a value at the preceding post-maintenance accepted
 * endpoint (before), at the physical endpoint before maintenance and at the
 * resulting post-maintenance endpoint (after).  Rates are endpoint rates;
 * works are step-integrated, positive when they add to the modeled stored
 * energy.  A declared identity channel is staged as NaN and evaluated by the
 * ledger.
 */
struct EnergyBalanceValues {
    std::vector<double> stored_before{};
    std::vector<double> stored_physical_endpoint{};
    std::vector<double> stored_after{};
    std::vector<double> dissipation_rate{};
    std::vector<double> external_work{};
    std::vector<double> numerical_work{};

    /// The rate or work values of @p kind (not Stored).
    [[nodiscard]] std::vector<double>& values(EnergyChannelKind kind);
    [[nodiscard]] const std::vector<double>& values(EnergyChannelKind kind) const;
};

/// One source per declared channel of each kind, indexed by
/// energyChannelKindIndex.
using EnergyBalanceSources = std::array<std::vector<EnergyChannelSource>, 4>;

/// A named input of the declared identities.
struct EnergyBalanceInput {
    std::string name{};
    double value{std::numeric_limits<double>::quiet_NaN()};
};

struct EnergyBalanceAttempt {
    EnergyBalanceAttemptStatus status{EnergyBalanceAttemptStatus::Trial};
    EnergyBalanceRejectionReason rejection_reason{EnergyBalanceRejectionReason::None};
    EnergyBalanceAttemptMetadata metadata{};
    bool balance_staged{false};
    EnergyBalanceValues values{};
    EnergyBalanceSources sources{};
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
 * @brief Transactional ledger of a fixed-topology backward-Euler energy
 *        balance over the channels of a catalogue.
 *
 * Every channel must be supplied explicitly (or evaluated from its declared
 * identity), including exact zeros, with exactly one nonempty owner name and
 * a Produced or NotApplicable decision; NotApplicable channels require
 * exact-zero values.  Every attempt distinguishes the preceding
 * post-maintenance accepted endpoint, the physical pre-maintenance endpoint
 * and the resulting post-maintenance accepted endpoint.  Physical rates and
 * works are evaluated at the middle state; a staged record must identify it
 * as the backward-Euler accepted endpoint at time_after and stage fraction
 * one (generalized-alpha and non-endpoint data fail closed).  Consecutive
 * accepted records form one exact provenance chain from the final state of
 * one attempt to the initial state of the next: time, algebraic state,
 * snapshot, topology, extension map and stored values must all agree.
 * After acceptance the next record starts the next step at attempt one;
 * after rejection only the next attempt on the identical accepted starting
 * endpoint is admissible (its duration may change under step control).
 * Channel ownership and applicability are immutable within one history.  An
 * attempt rejected before all channels are available is recorded with
 * balance_staged=false: its balance values remain NaN rather than physical
 * zeros, while every accepted contribution is exact zero.
 */
class EnergyBalanceLedger {
public:
    explicit EnergyBalanceLedger(EnergyBalanceCatalogue catalogue);

    [[nodiscard]] const EnergyBalanceCatalogue& catalogue() const noexcept { return catalogue_; }

    void beginAttempt(EnergyBalanceAttemptMetadata metadata);

    /// Stages the one complete balance of the active attempt; evaluates the
    /// declared identities from @p inputs.
    void stageBalance(EnergyBalanceValues values,
                      EnergyBalanceSources sources,
                      std::span<const EnergyBalanceInput> inputs = {});

    void commitAttempt();
    void rejectAttempt(EnergyBalanceRejectionReason reason);
    /// Records a rejection that occurs before a complete balance can be staged.
    void rejectUnstagedAttempt(EnergyBalanceRejectionReason reason);

    [[nodiscard]] bool attemptActive() const noexcept;
    [[nodiscard]] const EnergyBalanceAttemptMetadata* activeAttempt() const noexcept;
    [[nodiscard]] const EnergyBalanceAttempt* trialBalance() const noexcept;
    [[nodiscard]] const std::vector<EnergyBalanceAttempt>& acceptedAttempts() const noexcept;
    [[nodiscard]] const std::vector<EnergyBalanceAttempt>& rejectedAttempts() const noexcept;

private:
    EnergyBalanceCatalogue catalogue_;
    std::optional<EnergyBalanceAttemptMetadata> active_attempt_{};
    std::optional<EnergyBalanceAttempt> trial_balance_{};
    std::optional<EnergyBalanceAttemptStatus> last_published_status_{};
    std::optional<EnergyBalanceAttemptMetadata> last_published_metadata_{};
    std::uint64_t last_transaction_id_{0};
    std::vector<EnergyBalanceAttempt> accepted_attempts_{};
    std::vector<EnergyBalanceAttempt> rejected_attempts_{};
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_ENERGYBALANCELEDGER_H
