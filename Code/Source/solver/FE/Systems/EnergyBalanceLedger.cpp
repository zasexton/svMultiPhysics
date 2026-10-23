/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/EnergyBalanceLedger.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace svmp {
namespace FE {
namespace systems {

// The transaction rules, provenance chain and balance are those of the
// Navier-Stokes free-surface energy ledger
// (Physics/.../FreeSurface/FreeSurfaceEnergyLedger.cpp), whose hard-coded
// channel fields became the declared channels of a catalogue.  Each kind is
// summed in declaration order, as the fields were.

namespace {

static_assert(std::is_nothrow_move_constructible_v<EnergyBalanceAttempt>);
static_assert(std::is_nothrow_copy_constructible_v<EnergyBalanceAttemptMetadata>);
static_assert(std::is_nothrow_copy_assignable_v<EnergyBalanceAttemptMetadata>);

bool finite(double value) noexcept
{
    return std::isfinite(value);
}

/// The subject with its first letter in upper case, to start a sentence.
std::string sentence(const std::string& subject)
{
    std::string text = subject;
    if (!text.empty() && text.front() >= 'a' && text.front() <= 'z') {
        text.front() = static_cast<char>(text.front() - 'a' + 'A');
    }
    return text;
}

void requireFiniteChannel(const std::string& subject, double value, const std::string& channel)
{
    if (!finite(value)) {
        throw std::invalid_argument(sentence(subject) + " channel '" + channel +
                                    "' must be finite.");
    }
}

void requireChannelValue(const std::string& subject,
                         double value,
                         bool nonnegative,
                         const std::string& channel)
{
    requireFiniteChannel(subject, value, channel);
    if (nonnegative && value < 0.0) {
        throw std::invalid_argument(sentence(subject) + " channel '" + channel +
                                    "' must be nonnegative.");
    }
}

void validateChannelSource(const std::string& subject,
                           double value,
                           const EnergyChannelSource& source,
                           const std::string& channel)
{
    const bool named_owner = source.owner.find_first_not_of(" \t\r\n") != std::string::npos;
    const bool produced = source.applicability == EnergyChannelApplicability::Produced;
    const bool inapplicable = source.applicability == EnergyChannelApplicability::NotApplicable;
    if ((!produced && !inapplicable) || !named_owner) {
        throw std::invalid_argument(
            sentence(subject) + " channel '" + channel +
            "' requires one named owner and an explicit applicability decision.");
    }
    if (inapplicable && value != 0.0) {
        throw std::invalid_argument("Inapplicable " + subject + " channel '" + channel +
                                    "' must be exactly zero.");
    }
}

template <typename Value>
void reserveForOneMore(std::vector<Value>& values, const std::string& subject)
{
    if (values.size() < values.capacity()) {
        return;
    }
    const auto maximum = values.max_size();
    if (values.size() == maximum) {
        throw std::length_error(sentence(subject) + " history reached its maximum size.");
    }
    const auto doubled = values.capacity() > maximum / 2u
                             ? maximum
                             : std::max<std::size_t>(1u, values.capacity() * 2u);
    values.reserve(std::max(values.size() + 1u, doubled));
}

bool sameValues(const std::vector<double>& left, const std::vector<double>& right) noexcept
{
    if (left.size() != right.size()) {
        return false;
    }
    for (std::size_t i = 0; i < left.size(); ++i) {
        if (!(left[i] == right[i])) {
            return false;
        }
    }
    return true;
}

void validateAttemptEnvelope(const std::string& subject,
                             const EnergyBalanceAttemptMetadata& metadata)
{
    const bool extension_revisions_valid =
        (!metadata.extension_map_revision_before.has_value() ||
         *metadata.extension_map_revision_before != 0u) &&
        (!metadata.physical_endpoint_extension_map_revision.has_value() ||
         *metadata.physical_endpoint_extension_map_revision != 0u) &&
        (!metadata.extension_map_revision_after.has_value() ||
         *metadata.extension_map_revision_after != 0u);
    const double time_span = metadata.time_after - metadata.time_before;
    const double time_scale = std::max({1.0,
                                        std::abs(metadata.time_before),
                                        std::abs(metadata.time_after),
                                        std::abs(metadata.dt)});
    const double time_tolerance = 64.0 * std::numeric_limits<double>::epsilon() * time_scale;
    if (metadata.transaction_id == 0u || metadata.step == 0u || metadata.attempt == 0u ||
        !finite(metadata.time_before) || !finite(metadata.time_after) || !finite(metadata.dt) ||
        metadata.dt <= 0.0 || metadata.time_after <= metadata.time_before ||
        std::abs(time_span - metadata.dt) > time_tolerance ||
        metadata.algebraic_state_revision_before == 0u ||
        metadata.snapshot_set_revision_before == 0u ||
        metadata.mesh_topology_set_revision_before == 0u ||
        metadata.cut_topology_set_revision_before == 0u || !extension_revisions_valid) {
        throw std::invalid_argument(
            sentence(subject) +
            "-attempt envelope is incomplete or is not one positive endpoint interval.");
    }
}

bool endpointMetadataComplete(const EnergyBalanceAttemptMetadata& metadata) noexcept
{
    const bool extension_coverage_complete =
        metadata.extension_map_revision_before.has_value() ==
            metadata.physical_endpoint_extension_map_revision.has_value() &&
        metadata.extension_map_revision_before.has_value() ==
            metadata.extension_map_revision_after.has_value();
    return extension_coverage_complete &&
           metadata.physical_endpoint_algebraic_state_revision != 0u &&
           metadata.algebraic_state_revision_after != 0u &&
           metadata.physical_endpoint_snapshot_set_revision != 0u &&
           metadata.snapshot_set_revision_after != 0u &&
           metadata.physical_endpoint_mesh_topology_set_revision != 0u &&
           metadata.mesh_topology_set_revision_after != 0u &&
           metadata.physical_endpoint_cut_topology_set_revision != 0u &&
           metadata.cut_topology_set_revision_after != 0u;
}

bool topologyChanged(const EnergyBalanceAttemptMetadata& metadata) noexcept
{
    return metadata.mesh_topology_set_revision_before !=
               metadata.physical_endpoint_mesh_topology_set_revision ||
           metadata.physical_endpoint_mesh_topology_set_revision !=
               metadata.mesh_topology_set_revision_after ||
           metadata.cut_topology_set_revision_before !=
               metadata.physical_endpoint_cut_topology_set_revision ||
           metadata.physical_endpoint_cut_topology_set_revision !=
               metadata.cut_topology_set_revision_after;
}

void requireStageableBackwardEulerEndpoint(const std::string& subject,
                                           const EnergyBalanceAttemptMetadata& metadata)
{
    if (!endpointMetadataComplete(metadata)) {
        throw std::invalid_argument(
            "A staged " + subject +
            " balance requires complete physical and post-maintenance endpoint provenance.");
    }
    if (topologyChanged(metadata)) {
        throw std::invalid_argument("A " + subject +
                                    " balance cannot be staged across a mesh or cut topology change.");
    }
    if (metadata.temporal_scheme != EnergyBalanceTemporalScheme::BackwardEuler ||
        !finite(metadata.physical_evaluation_time) ||
        !finite(metadata.physical_evaluation_stage_fraction) ||
        metadata.physical_evaluation_time != metadata.time_after ||
        metadata.physical_evaluation_stage_fraction != 1.0) {
        throw std::invalid_argument(
            "A staged " + subject +
            " balance requires explicit backward-Euler physical evaluation at the accepted endpoint.");
    }
}

void validateRejectionReason(const std::string& subject,
                             const EnergyBalanceAttemptMetadata& metadata,
                             EnergyBalanceRejectionReason reason)
{
    const bool recognized_reason =
        reason == EnergyBalanceRejectionReason::NonlinearSolveFailure ||
        reason == EnergyBalanceRejectionReason::StepControllerRejection ||
        reason == EnergyBalanceRejectionReason::PreacceptRejection ||
        reason == EnergyBalanceRejectionReason::TopologyChange ||
        reason == EnergyBalanceRejectionReason::MaintenanceRollback ||
        reason == EnergyBalanceRejectionReason::PublicationFailure;
    if (!recognized_reason) {
        throw std::invalid_argument("A rejected " + subject +
                                    " attempt requires an explicit reason.");
    }
    const bool complete_endpoint = endpointMetadataComplete(metadata);
    const bool topology_changed = complete_endpoint && topologyChanged(metadata);
    if ((reason == EnergyBalanceRejectionReason::TopologyChange &&
         (!complete_endpoint || !topology_changed)) ||
        (reason != EnergyBalanceRejectionReason::TopologyChange && topology_changed)) {
        throw std::invalid_argument(
            "The topology-change rejection reason must agree with the recorded mesh and cut topology revisions.");
    }
}

const EnergyBalanceInput* findInput(std::span<const EnergyBalanceInput> inputs,
                                    std::string_view name) noexcept
{
    for (const auto& input : inputs) {
        if (input.name == name) {
            return &input;
        }
    }
    return nullptr;
}

/// The staged values with unavailable (NaN) entries, one per channel.
EnergyBalanceValues unavailableValues(const EnergyBalanceCatalogue& catalogue)
{
    const double unavailable = std::numeric_limits<double>::quiet_NaN();
    EnergyBalanceValues values;
    const auto stored = catalogue.count(EnergyChannelKind::Stored);
    values.stored_before.assign(stored, unavailable);
    values.stored_physical_endpoint.assign(stored, unavailable);
    values.stored_after.assign(stored, unavailable);
    for (const auto kind : kEnergyChannelKinds) {
        if (kind != EnergyChannelKind::Stored) {
            values.values(kind).assign(catalogue.count(kind), unavailable);
        }
    }
    return values;
}

} // namespace

std::string_view energyChannelKindPrefix(EnergyChannelKind kind) noexcept
{
    switch (kind) {
        case EnergyChannelKind::Stored:
            return "stored";
        case EnergyChannelKind::DissipationRate:
            return "dissipation";
        case EnergyChannelKind::ExternalWork:
            return "external";
        case EnergyChannelKind::NumericalWork:
            return "numerical";
    }
    return "invalid";
}

const char* energyChannelApplicabilityName(EnergyChannelApplicability applicability) noexcept
{
    switch (applicability) {
        case EnergyChannelApplicability::Produced:
            return "produced";
        case EnergyChannelApplicability::NotApplicable:
            return "not_applicable";
        case EnergyChannelApplicability::Unspecified:
            return "unspecified";
    }
    return "invalid";
}

void appendEnergyChannelSource(std::ostream& stream,
                               std::string_view label,
                               const EnergyChannelSource& source)
{
    stream << ' ' << label << "_applicability="
           << energyChannelApplicabilityName(source.applicability) << ' ' << label
           << "_owner=" << std::quoted(source.owner);
}

const char* energyBalanceRejectionReasonName(EnergyBalanceRejectionReason reason) noexcept
{
    switch (reason) {
        case EnergyBalanceRejectionReason::NonlinearSolveFailure:
            return "nonlinear_solve_failure";
        case EnergyBalanceRejectionReason::StepControllerRejection:
            return "step_controller_rejection";
        case EnergyBalanceRejectionReason::PreacceptRejection:
            return "preaccept_rejection";
        case EnergyBalanceRejectionReason::TopologyChange:
            return "topology_change";
        case EnergyBalanceRejectionReason::MaintenanceRollback:
            return "maintenance_rollback";
        case EnergyBalanceRejectionReason::PublicationFailure:
            return "publication_failure";
        case EnergyBalanceRejectionReason::None:
            return "none";
    }
    return "invalid";
}

double sumInOrder(std::span<const double> values) noexcept
{
    if (values.empty()) {
        return 0.0;
    }
    double sum = values.front();
    for (std::size_t i = 1; i < values.size(); ++i) {
        sum += values[i];
    }
    return sum;
}

double energyChange(std::span<const double> after, std::span<const double> before) noexcept
{
    double change = sumInOrder(after);
    for (const double value : before) {
        change -= value;
    }
    return change;
}

double transportCouplingWork(double energy_change, double dt, double power) noexcept
{
    return energy_change - dt * power;
}

// The residual-work rows of the free-surface energy account
// (Physics/.../FreeSurface/FreeSurfaceEnergyAccount.cpp), with the seven
// channel names declared by the caller; checks in the same order.
std::vector<double> evaluateResidualStepWork(std::span<const ResidualWorkPairing> pairings,
                                             std::span<const std::string> channels,
                                             double dt,
                                             std::string_view subject)
{
    const std::string name(subject);
    if (!finite(dt) || !(dt > 0.0)) {
        throw std::invalid_argument(sentence(name) + " requires a finite positive duration.");
    }
    if (pairings.size() != channels.size()) {
        throw std::invalid_argument(sentence(name) + " requires exactly " +
                                    std::to_string(channels.size()) +
                                    " connected channel rows.");
    }
    std::vector<std::optional<double>> works(channels.size());
    for (const auto& pairing : pairings) {
        if (pairing.owner_component.find_first_not_of(" \t\r\n") == std::string::npos) {
            throw std::invalid_argument(sentence(name) + " requires one named channel owner.");
        }
        if (!finite(pairing.residual_state_pairing)) {
            throw std::invalid_argument(sentence(name) +
                                        " requires a finite residual/state pairing.");
        }
        if (pairing.produced && pairing.operator_tag.empty()) {
            throw std::invalid_argument("A produced " + name +
                                        " channel requires an operator tag.");
        }
        if (!pairing.produced &&
            (!pairing.operator_tag.empty() || pairing.residual_state_pairing != 0.0)) {
            throw std::invalid_argument(
                "An inapplicable " + name +
                " channel requires an empty operator and exact-zero pairing.");
        }
        const double signed_work = pairing.produced ? -dt * pairing.residual_state_pairing : 0.0;
        if (!finite(signed_work)) {
            throw std::overflow_error(sentence(name) + " step work is not finite.");
        }
        const auto declared = std::find(channels.begin(), channels.end(), pairing.channel);
        if (declared == channels.end()) {
            throw std::invalid_argument(sentence(name) + " contains an unknown connected channel '" +
                                        pairing.channel + "'.");
        }
        auto& destination = works[static_cast<std::size_t>(declared - channels.begin())];
        if (destination.has_value()) {
            throw std::invalid_argument(sentence(name) + " contains a duplicate channel '" +
                                        pairing.channel + "'.");
        }
        destination = signed_work;
    }
    std::vector<double> values;
    values.reserve(works.size());
    for (const auto& work : works) {
        if (!work.has_value()) {
            throw std::invalid_argument(sentence(name) + " is missing a connected channel.");
        }
        values.push_back(*work);
    }
    return values;
}

// ---------------------------------------------------------------------------
// EnergyBalanceCatalogue
// ---------------------------------------------------------------------------

EnergyBalanceCatalogue::EnergyBalanceCatalogue(
    std::string subject,
    std::array<std::vector<EnergyChannelDeclaration>, 4> channels)
    : subject_(std::move(subject)), channels_(std::move(channels))
{
    if (subject_.find_first_not_of(" \t\r\n") == std::string::npos) {
        throw std::invalid_argument("An energy balance catalogue requires a subject.");
    }
    for (const auto kind : kEnergyChannelKinds) {
        const auto& declared = channels_[energyChannelKindIndex(kind)];
        for (std::size_t i = 0; i < declared.size(); ++i) {
            const auto& channel = declared[i];
            const auto qualified = std::string(energyChannelKindPrefix(kind)) + "." + channel.name;
            if (channel.name.empty() ||
                channel.name.find_first_of(" \t\r\n") != std::string::npos) {
                throw std::invalid_argument(sentence(subject_) +
                                            " channels require nonempty names without spaces.");
            }
            for (std::size_t j = 0; j < i; ++j) {
                if (declared[j].name == channel.name) {
                    throw std::invalid_argument(sentence(subject_) + " channel '" + qualified +
                                                "' is declared twice.");
                }
            }
            const auto& identity = channel.identity;
            using Rule = EnergyChannelIdentity::Rule;
            if (identity.rule == Rule::Supplied) {
                if (!identity.stored.empty() || !identity.input.empty()) {
                    throw std::invalid_argument(sentence(subject_) + " channel '" + qualified +
                                                "' is supplied and declares no identity terms.");
                }
                continue;
            }
            const bool stored_count_valid =
                (identity.rule == Rule::StoredTransport && !identity.stored.empty()) ||
                (identity.rule == Rule::DomainTransport && identity.stored.size() == 1u) ||
                (identity.rule == Rule::StepWork && identity.stored.empty());
            if (kind == EnergyChannelKind::Stored || identity.input.empty() ||
                !stored_count_valid) {
                throw std::invalid_argument(sentence(subject_) + " channel '" + qualified +
                                            "' declares an incomplete identity.");
            }
        }
    }
    // Identities name declared stored channels (checked once all are known).
    const auto& stored = channels_[energyChannelKindIndex(EnergyChannelKind::Stored)];
    for (const auto kind : kEnergyChannelKinds) {
        for (const auto& channel : channels_[energyChannelKindIndex(kind)]) {
            for (const auto& name : channel.identity.stored) {
                const bool declared =
                    std::any_of(stored.begin(), stored.end(), [&](const auto& candidate) {
                        return candidate.name == name;
                    });
                if (!declared) {
                    throw std::invalid_argument(
                        sentence(subject_) + " channel '" +
                        std::string(energyChannelKindPrefix(kind)) + "." + channel.name +
                        "' names the undeclared stored channel '" + name + "'.");
                }
            }
        }
    }
}

std::span<const EnergyChannelDeclaration> EnergyBalanceCatalogue::channels(
    EnergyChannelKind kind) const noexcept
{
    return channels_[energyChannelKindIndex(kind)];
}

std::size_t EnergyBalanceCatalogue::count(EnergyChannelKind kind) const noexcept
{
    return channels_[energyChannelKindIndex(kind)].size();
}

std::optional<std::size_t> EnergyBalanceCatalogue::find(EnergyChannelKind kind,
                                                        std::string_view name) const noexcept
{
    const auto& declared = channels_[energyChannelKindIndex(kind)];
    for (std::size_t i = 0; i < declared.size(); ++i) {
        if (declared[i].name == name) {
            return i;
        }
    }
    return std::nullopt;
}

std::size_t EnergyBalanceCatalogue::position(EnergyChannelKind kind, std::string_view name) const
{
    const auto found = find(kind, name);
    if (!found.has_value()) {
        throw std::out_of_range(sentence(subject_) + " channel '" +
                                std::string(energyChannelKindPrefix(kind)) + "." +
                                std::string(name) + "' is not declared.");
    }
    return *found;
}

std::string EnergyBalanceCatalogue::qualifiedName(EnergyChannelKind kind,
                                                  std::size_t position) const
{
    return std::string(energyChannelKindPrefix(kind)) + "." +
           channels_[energyChannelKindIndex(kind)].at(position).name;
}

// ---------------------------------------------------------------------------
// EnergyBalanceValues
// ---------------------------------------------------------------------------

namespace {

template <typename Values>
auto& valuesOfKind(Values& values, EnergyChannelKind kind)
{
    switch (kind) {
        case EnergyChannelKind::DissipationRate:
            return values.dissipation_rate;
        case EnergyChannelKind::ExternalWork:
            return values.external_work;
        case EnergyChannelKind::NumericalWork:
            return values.numerical_work;
        case EnergyChannelKind::Stored:
            break;
    }
    throw std::invalid_argument(
        "Stored energy channels have values at three endpoints, not one.");
}

} // namespace

std::vector<double>& EnergyBalanceValues::values(EnergyChannelKind kind)
{
    return valuesOfKind(*this, kind);
}

const std::vector<double>& EnergyBalanceValues::values(EnergyChannelKind kind) const
{
    return valuesOfKind(*this, kind);
}

// ---------------------------------------------------------------------------
// EnergyBalanceLedger
// ---------------------------------------------------------------------------

EnergyBalanceLedger::EnergyBalanceLedger(EnergyBalanceCatalogue catalogue)
    : catalogue_(std::move(catalogue))
{
    if (catalogue_.subject().empty()) {
        throw std::invalid_argument("An energy balance ledger requires a declared catalogue.");
    }
}

void EnergyBalanceLedger::beginAttempt(EnergyBalanceAttemptMetadata metadata)
{
    const auto& subject = catalogue_.subject();
    if (active_attempt_.has_value()) {
        throw std::logic_error("A " + subject + " attempt is already active.");
    }
    if (trial_balance_.has_value()) {
        throw std::logic_error("A " + subject + " trial balance remains unpublished.");
    }
    validateAttemptEnvelope(subject, metadata);
    if (metadata.transaction_id <= last_transaction_id_) {
        throw std::invalid_argument(sentence(subject) +
                                    " transaction identifiers must increase.");
    }
    if (last_published_status_.has_value() != last_published_metadata_.has_value()) {
        throw std::logic_error(sentence(subject) +
                               " published-attempt sequencing state is incomplete.");
    }
    if (!last_published_metadata_.has_value()) {
        if (metadata.attempt != 1u) {
            throw std::invalid_argument("The first published " + subject +
                                        " attempt for a step must use attempt one.");
        }
    } else {
        const auto& previous = *last_published_metadata_;
        if (*last_published_status_ == EnergyBalanceAttemptStatus::Accepted) {
            if (previous.step == std::numeric_limits<std::uint64_t>::max() ||
                metadata.step != previous.step + 1u || metadata.attempt != 1u) {
                throw std::invalid_argument(
                    "A " + subject +
                    " attempt after acceptance must start the next step at attempt one.");
            }
        } else if (*last_published_status_ == EnergyBalanceAttemptStatus::Rejected) {
            const bool retry_valid =
                previous.attempt != std::numeric_limits<std::uint64_t>::max() &&
                metadata.step == previous.step && metadata.attempt == previous.attempt + 1u &&
                metadata.time_before == previous.time_before &&
                metadata.algebraic_state_revision_before ==
                    previous.algebraic_state_revision_before &&
                metadata.snapshot_set_revision_before == previous.snapshot_set_revision_before &&
                metadata.mesh_topology_set_revision_before ==
                    previous.mesh_topology_set_revision_before &&
                metadata.cut_topology_set_revision_before ==
                    previous.cut_topology_set_revision_before &&
                metadata.extension_map_revision_before == previous.extension_map_revision_before;
            if (!retry_valid) {
                throw std::invalid_argument(
                    "A rejected " + subject +
                    " attempt must be followed by the next attempt on the same accepted starting endpoint.");
            }
        } else {
            throw std::logic_error("A trial " + subject +
                                   " attempt cannot be a published sequencing predecessor.");
        }
    }
    if (!accepted_attempts_.empty()) {
        const auto& previous = accepted_attempts_.back().metadata;
        const bool endpoint_chain_valid =
            previous.step != std::numeric_limits<std::uint64_t>::max() &&
            metadata.step == previous.step + 1u && metadata.time_before == previous.time_after &&
            metadata.algebraic_state_revision_before == previous.algebraic_state_revision_after &&
            metadata.snapshot_set_revision_before == previous.snapshot_set_revision_after &&
            metadata.mesh_topology_set_revision_before ==
                previous.mesh_topology_set_revision_after &&
            metadata.cut_topology_set_revision_before ==
                previous.cut_topology_set_revision_after &&
            metadata.extension_map_revision_before == previous.extension_map_revision_after;
        if (!endpoint_chain_valid) {
            throw std::invalid_argument(
                "A " + subject +
                " attempt must continue the latest accepted step, time, algebraic state, geometry, topology, and extension endpoint.");
        }
    }

    last_transaction_id_ = metadata.transaction_id;
    active_attempt_ = std::move(metadata);
}

void EnergyBalanceLedger::stageBalance(EnergyBalanceValues values,
                                       EnergyBalanceSources sources,
                                       std::span<const EnergyBalanceInput> inputs)
{
    const auto& subject = catalogue_.subject();
    if (!active_attempt_.has_value()) {
        throw std::logic_error("A " + subject + " balance requires an active attempt.");
    }
    if (trial_balance_.has_value()) {
        throw std::logic_error("A " + subject + " attempt accepts exactly one complete balance.");
    }
    requireStageableBackwardEulerEndpoint(subject, *active_attempt_);

    const auto stored_count = catalogue_.count(EnergyChannelKind::Stored);
    bool sizes_match = values.stored_before.size() == stored_count &&
                       values.stored_physical_endpoint.size() == stored_count &&
                       values.stored_after.size() == stored_count;
    for (const auto kind : kEnergyChannelKinds) {
        sizes_match = sizes_match &&
                      sources[energyChannelKindIndex(kind)].size() == catalogue_.count(kind) &&
                      (kind == EnergyChannelKind::Stored ||
                       values.values(kind).size() == catalogue_.count(kind));
    }
    if (!sizes_match) {
        throw std::invalid_argument("A " + subject +
                                    " balance requires one value and one source per declared channel.");
    }

    const auto stored_channels = catalogue_.channels(EnergyChannelKind::Stored);
    for (const auto* endpoint :
         {&values.stored_before, &values.stored_physical_endpoint, &values.stored_after}) {
        for (std::size_t i = 0; i < stored_count; ++i) {
            requireChannelValue(subject,
                                (*endpoint)[i],
                                stored_channels[i].nonnegative,
                                catalogue_.qualifiedName(EnergyChannelKind::Stored, i));
        }
    }

    // Declared identities, from the validated stored energies and the inputs.
    const double dt = active_attempt_->dt;
    for (const auto kind : kEnergyChannelKinds) {
        if (kind == EnergyChannelKind::Stored) {
            continue;
        }
        const auto declared = catalogue_.channels(kind);
        auto& slots = values.values(kind);
        for (std::size_t i = 0; i < declared.size(); ++i) {
            const auto& identity = declared[i].identity;
            using Rule = EnergyChannelIdentity::Rule;
            if (identity.rule == Rule::Supplied) {
                continue;
            }
            const auto qualified = catalogue_.qualifiedName(kind, i);
            if (!std::isnan(slots[i])) {
                throw std::invalid_argument(
                    sentence(subject) + " channel '" + qualified +
                    "' is a declared identity; the ledger evaluates it.");
            }
            const auto* input = findInput(inputs, identity.input);
            if (input == nullptr) {
                throw std::invalid_argument(sentence(subject) + " channel '" + qualified +
                                            "' requires the input '" + identity.input + "'.");
            }
            std::vector<double> after;
            std::vector<double> before;
            after.reserve(identity.stored.size());
            before.reserve(identity.stored.size());
            for (const auto& name : identity.stored) {
                const auto position = catalogue_.position(EnergyChannelKind::Stored, name);
                after.push_back(values.stored_physical_endpoint[position]);
                before.push_back(values.stored_before[position]);
            }
            switch (identity.rule) {
                case Rule::StoredTransport:
                    slots[i] = transportCouplingWork(energyChange(after, before), dt, input->value);
                    break;
                case Rule::DomainTransport:
                    slots[i] = input->value - before.front();
                    break;
                case Rule::StepWork:
                    slots[i] = dt * input->value;
                    break;
                case Rule::Supplied:
                    break;
            }
        }
    }

    for (const auto kind : kEnergyChannelKinds) {
        if (kind == EnergyChannelKind::Stored) {
            continue;
        }
        const auto declared = catalogue_.channels(kind);
        const auto& slots = values.values(kind);
        for (std::size_t i = 0; i < declared.size(); ++i) {
            requireChannelValue(subject,
                                slots[i],
                                kind == EnergyChannelKind::DissipationRate ||
                                    declared[i].nonnegative,
                                catalogue_.qualifiedName(kind, i));
        }
    }
    for (const auto kind : kEnergyChannelKinds) {
        const auto& kind_sources = sources[energyChannelKindIndex(kind)];
        for (std::size_t i = 0; i < kind_sources.size(); ++i) {
            const auto qualified = catalogue_.qualifiedName(kind, i);
            if (kind == EnergyChannelKind::Stored) {
                validateChannelSource(subject, values.stored_before[i], kind_sources[i], qualified);
                validateChannelSource(
                    subject, values.stored_physical_endpoint[i], kind_sources[i], qualified);
                validateChannelSource(subject, values.stored_after[i], kind_sources[i], qualified);
            } else {
                validateChannelSource(subject, values.values(kind)[i], kind_sources[i], qualified);
            }
        }
    }

    const EnergyBalanceAttempt* latest_staged_attempt = nullptr;
    if (!accepted_attempts_.empty()) {
        latest_staged_attempt = &accepted_attempts_.back();
    }
    const auto latest_staged_rejection =
        std::find_if(rejected_attempts_.rbegin(), rejected_attempts_.rend(), [](const auto& attempt) {
            return attempt.balance_staged;
        });
    if (latest_staged_rejection != rejected_attempts_.rend() &&
        (latest_staged_attempt == nullptr ||
         latest_staged_rejection->metadata.transaction_id >
             latest_staged_attempt->metadata.transaction_id)) {
        latest_staged_attempt = &*latest_staged_rejection;
    }
    if (latest_staged_attempt != nullptr && latest_staged_attempt->sources != sources) {
        throw std::invalid_argument(
            sentence(subject) +
            " channel ownership and applicability cannot change within one ledger history.");
    }
    if (last_published_status_ == EnergyBalanceAttemptStatus::Rejected &&
        latest_staged_rejection != rejected_attempts_.rend() &&
        latest_staged_rejection->metadata.step == active_attempt_->step &&
        !sameValues(values.stored_before, latest_staged_rejection->values.stored_before)) {
        throw std::invalid_argument(
            "A retried " + subject +
            " attempt must preserve the latest staged rejected starting energy.");
    }
    if (!accepted_attempts_.empty() &&
        !sameValues(values.stored_before, accepted_attempts_.back().values.stored_after)) {
        throw std::invalid_argument(
            sentence(subject) +
            " stored values before an attempt must match the latest accepted endpoint.");
    }

    const double energy_before = sumInOrder(values.stored_before);
    const double physical_endpoint_energy = sumInOrder(values.stored_physical_endpoint);
    const double energy_after = sumInOrder(values.stored_after);
    const double physical_energy_change = physical_endpoint_energy - energy_before;
    const double maintenance_energy_change = energy_after - physical_endpoint_energy;
    const double energy_change = energy_after - energy_before;
    const double integrated_dissipation =
        active_attempt_->dt * sumInOrder(values.dissipation_rate);
    const double total_external_work = sumInOrder(values.external_work);
    const double total_numerical_work = sumInOrder(values.numerical_work);
    const double residual =
        energy_change + integrated_dissipation - total_external_work - total_numerical_work;
    const std::array<double, 10> totals{energy_before,
                                        physical_endpoint_energy,
                                        energy_after,
                                        physical_energy_change,
                                        maintenance_energy_change,
                                        energy_change,
                                        integrated_dissipation,
                                        total_external_work,
                                        total_numerical_work,
                                        residual};
    if (!std::all_of(totals.begin(), totals.end(), finite)) {
        throw std::invalid_argument("The assembled " + subject + " balance is not finite.");
    }

    trial_balance_ = EnergyBalanceAttempt{
        .status = EnergyBalanceAttemptStatus::Trial,
        .rejection_reason = EnergyBalanceRejectionReason::None,
        .metadata = *active_attempt_,
        .balance_staged = true,
        .values = std::move(values),
        .sources = std::move(sources),
        .stored_energy_before = energy_before,
        .stored_energy_physical_endpoint_before_maintenance = physical_endpoint_energy,
        .stored_energy_after = energy_after,
        .physical_stored_energy_change = physical_energy_change,
        .maintenance_stored_energy_change = maintenance_energy_change,
        .stored_energy_change = energy_change,
        .integrated_physical_dissipation = integrated_dissipation,
        .total_external_work = total_external_work,
        .total_numerical_work = total_numerical_work,
        .trial_balance_residual = residual,
        .accepted_stored_energy_change = 0.0,
        .accepted_physical_stored_energy_change = 0.0,
        .accepted_maintenance_stored_energy_change = 0.0,
        .accepted_integrated_physical_dissipation = 0.0,
        .accepted_external_work = 0.0,
        .accepted_numerical_work = 0.0,
        .accepted_balance_residual = 0.0,
    };
}

void EnergyBalanceLedger::commitAttempt()
{
    const auto& subject = catalogue_.subject();
    if (!active_attempt_.has_value() || !trial_balance_.has_value()) {
        throw std::logic_error("A complete staged " + subject +
                               " balance is required before commit.");
    }
    if (topologyChanged(*active_attempt_)) {
        throw std::logic_error("A fixed-topology " + subject +
                               " balance cannot be accepted across a topology change.");
    }
    reserveForOneMore(accepted_attempts_, subject);

    trial_balance_->status = EnergyBalanceAttemptStatus::Accepted;
    trial_balance_->accepted_stored_energy_change = trial_balance_->stored_energy_change;
    trial_balance_->accepted_physical_stored_energy_change =
        trial_balance_->physical_stored_energy_change;
    trial_balance_->accepted_maintenance_stored_energy_change =
        trial_balance_->maintenance_stored_energy_change;
    trial_balance_->accepted_integrated_physical_dissipation =
        trial_balance_->integrated_physical_dissipation;
    trial_balance_->accepted_external_work = trial_balance_->total_external_work;
    trial_balance_->accepted_numerical_work = trial_balance_->total_numerical_work;
    trial_balance_->accepted_balance_residual = trial_balance_->trial_balance_residual;
    accepted_attempts_.push_back(std::move(*trial_balance_));
    last_published_status_ = EnergyBalanceAttemptStatus::Accepted;
    last_published_metadata_ = accepted_attempts_.back().metadata;
    trial_balance_.reset();
    active_attempt_.reset();
}

void EnergyBalanceLedger::rejectAttempt(EnergyBalanceRejectionReason reason)
{
    const auto& subject = catalogue_.subject();
    if (!active_attempt_.has_value() || !trial_balance_.has_value()) {
        throw std::logic_error("A complete staged " + subject +
                               " balance is required before rejection.");
    }
    validateRejectionReason(subject, *active_attempt_, reason);
    reserveForOneMore(rejected_attempts_, subject);

    trial_balance_->status = EnergyBalanceAttemptStatus::Rejected;
    trial_balance_->rejection_reason = reason;
    trial_balance_->accepted_stored_energy_change = 0.0;
    trial_balance_->accepted_physical_stored_energy_change = 0.0;
    trial_balance_->accepted_maintenance_stored_energy_change = 0.0;
    trial_balance_->accepted_integrated_physical_dissipation = 0.0;
    trial_balance_->accepted_external_work = 0.0;
    trial_balance_->accepted_numerical_work = 0.0;
    trial_balance_->accepted_balance_residual = 0.0;
    rejected_attempts_.push_back(std::move(*trial_balance_));
    last_published_status_ = EnergyBalanceAttemptStatus::Rejected;
    last_published_metadata_ = rejected_attempts_.back().metadata;
    trial_balance_.reset();
    active_attempt_.reset();
}

void EnergyBalanceLedger::rejectUnstagedAttempt(EnergyBalanceRejectionReason reason)
{
    const auto& subject = catalogue_.subject();
    if (!active_attempt_.has_value()) {
        throw std::logic_error("An unstaged " + subject + " rejection requires an active attempt.");
    }
    if (trial_balance_.has_value()) {
        throw std::logic_error("A staged " + subject +
                               " balance must use the staged rejection path.");
    }
    validateRejectionReason(subject, *active_attempt_, reason);
    reserveForOneMore(rejected_attempts_, subject);

    const double unavailable = std::numeric_limits<double>::quiet_NaN();
    EnergyBalanceSources unspecified;
    for (const auto kind : kEnergyChannelKinds) {
        unspecified[energyChannelKindIndex(kind)].resize(catalogue_.count(kind));
    }
    rejected_attempts_.push_back(EnergyBalanceAttempt{
        .status = EnergyBalanceAttemptStatus::Rejected,
        .rejection_reason = reason,
        .metadata = *active_attempt_,
        .balance_staged = false,
        .values = unavailableValues(catalogue_),
        .sources = std::move(unspecified),
        .stored_energy_before = unavailable,
        .stored_energy_physical_endpoint_before_maintenance = unavailable,
        .stored_energy_after = unavailable,
        .physical_stored_energy_change = unavailable,
        .maintenance_stored_energy_change = unavailable,
        .stored_energy_change = unavailable,
        .integrated_physical_dissipation = unavailable,
        .total_external_work = unavailable,
        .total_numerical_work = unavailable,
        .trial_balance_residual = unavailable,
        .accepted_stored_energy_change = 0.0,
        .accepted_physical_stored_energy_change = 0.0,
        .accepted_maintenance_stored_energy_change = 0.0,
        .accepted_integrated_physical_dissipation = 0.0,
        .accepted_external_work = 0.0,
        .accepted_numerical_work = 0.0,
        .accepted_balance_residual = 0.0,
    });
    last_published_status_ = EnergyBalanceAttemptStatus::Rejected;
    last_published_metadata_ = rejected_attempts_.back().metadata;
    active_attempt_.reset();
}

bool EnergyBalanceLedger::attemptActive() const noexcept
{
    return active_attempt_.has_value();
}

const EnergyBalanceAttemptMetadata* EnergyBalanceLedger::activeAttempt() const noexcept
{
    return active_attempt_.has_value() ? &*active_attempt_ : nullptr;
}

const EnergyBalanceAttempt* EnergyBalanceLedger::trialBalance() const noexcept
{
    return trial_balance_.has_value() ? &*trial_balance_ : nullptr;
}

const std::vector<EnergyBalanceAttempt>& EnergyBalanceLedger::acceptedAttempts() const noexcept
{
    return accepted_attempts_;
}

const std::vector<EnergyBalanceAttempt>& EnergyBalanceLedger::rejectedAttempts() const noexcept
{
    return rejected_attempts_;
}

} // namespace systems
} // namespace FE
} // namespace svmp
