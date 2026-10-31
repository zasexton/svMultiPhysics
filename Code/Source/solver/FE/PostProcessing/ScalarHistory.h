/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_POSTPROCESSING_SCALAR_HISTORY_H
#define SVMP_FE_POSTPROCESSING_SCALAR_HISTORY_H

/**
 * @file ScalarHistory.h
 * @brief The scalar history of a run: one row of recorded scalars per accepted step
 *
 * A formulation records the scalars worth a time series on its FE system
 * (an attachment), in declaration order:
 *
 *   record(system, Q);                          a named quantity
 *   record(system, X_id);                       a scalar global field
 *   record(system, Q_in, "outlet.Q");           an auxiliary input
 *   recordAuxiliaryState(system, rcr, "X");     a state of a single-entity
 *   recordAuxiliaryOutput(system, rcr, "P_out");  auxiliary instance (Global,
 *                                               Boundary or Region scope)
 *
 * Recording changes no operator, kernel, constraint or input of the system;
 * nothing is recorded by default, and a system that records nothing has no
 * history.  ScalarHistoryWriter, a step participant, writes the history:
 * a header line "step,time,<column>,..." and, after each accepted step, the
 * line "<step>,<time>,<value>,...", each number the shortest decimal that
 * reads back to the same double (std::to_chars).
 *
 * The values are the same on every rank and at every rank count for the
 * same state: a quantity is evaluated with exact sums
 * (evaluateFunctionalForms), a global field is read on its owner, rank 0,
 * and auxiliary inputs and single-entity states and outputs are replicated
 * on every rank.  Rank 0 writes the file, so for the same states it is the
 * same file at any rank count.
 */

#include "Core/Types.h"
#include "Forms/FormExpr.h"
#include "Systems/SystemCollectives.h"
#include "TimeStepping/StepParticipants.h"

#include <cstdint>
#include <fstream>
#include <span>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {
class AuxiliaryInputHandle;
class AuxiliaryInstanceHandle;
class FESystem;
class Quantity;
struct SystemStateView;
} // namespace systems

namespace post {

/// What one column of a scalar history reads.
enum class ScalarHistorySource : std::uint8_t {
    Quantity,        ///< a quantity's measured 0-form, summed exactly
    GlobalField,     ///< the value of a scalar global field
    AuxiliaryInput,  ///< the current value of a registered auxiliary input
    AuxiliaryState,  ///< a committed state of a single-entity auxiliary instance
    AuxiliaryOutput  ///< an output of a single-entity auxiliary instance at its committed state
};

/// One column of a scalar history.
struct ScalarHistoryColumn {
    std::string name{};                       ///< the header of the column
    ScalarHistorySource source{ScalarHistorySource::Quantity};
    forms::FormExpr form{};                   ///< Quantity: the measured 0-form
    FieldId field{INVALID_FIELD_ID};          ///< GlobalField
    std::string target{};                     ///< the input's registry name, or the instance
    std::string value{};                      ///< AuxiliaryState/AuxiliaryOutput: the state or output
};

/// The columns a system records (its attachment), in declaration order.
struct ScalarHistoryDeclarations {
    std::vector<ScalarHistoryColumn> columns{};
};

/**
 * Records @p quantity under its name.  Throws InvalidArgumentException for
 * an invalid or unnamed quantity and for a name that is already a column
 * or holds a comma, a quote or a line break.
 */
void record(systems::FESystem& system, const systems::Quantity& quantity);

/// Records the scalar global field @p field under its name (the checks of record(quantity)).
void record(systems::FESystem& system, FieldId field);

/// Records the auxiliary input @p input under @p name.
void record(systems::FESystem& system, const systems::AuxiliaryInputHandle& input, std::string name);

/// Records state @p state of the single-entity instance @p instance as "<instance>.<state>".
void recordAuxiliaryState(systems::FESystem& system,
                          const systems::AuxiliaryInstanceHandle& instance,
                          const std::string& state);

/// Records output @p output of the single-entity instance @p instance as "<instance>.<output>".
void recordAuxiliaryOutput(systems::FESystem& system,
                           const systems::AuxiliaryInstanceHandle& instance,
                           const std::string& output);

/// The columns @p system records (empty when none).
[[nodiscard]] std::span<const ScalarHistoryColumn> scalarHistoryColumns(const systems::FESystem& system) noexcept;

/// "step,time,<column>,...\n".
[[nodiscard]] std::string scalarHistoryHeader(const systems::FESystem& system);

/**
 * @brief The value of every column of @p system at @p state, in column order
 *
 * Collective over @p comm (every rank evaluates the quantities); the result
 * is the same on every rank.  The system must be set up.
 */
[[nodiscard]] std::vector<Real> scalarHistoryValues(const systems::FESystem& system,
                                                    const systems::SystemStateView& state,
                                                    const systems::CollectiveCommunicator& comm);

/// "<step>,<time>,<value>,...\n", each number the shortest decimal that reads back to it.
[[nodiscard]] std::string scalarHistoryRow(std::uint64_t step, Real time, std::span<const Real> values);

/**
 * @brief Writes the scalar history of a system, one row per accepted step
 *
 * The constructor (collective) truncates @p path on rank 0 of @p comm and
 * writes the header; onStepAccepted evaluates the columns at the accepted
 * state of the time history (collective) and appends its row on rank 0.
 */
class ScalarHistoryWriter final : public timestepping::IStepParticipant {
public:
    ScalarHistoryWriter(const systems::FESystem& system,
                        std::string path,
                        systems::CollectiveCommunicator comm);

    /// Accepted.
    [[nodiscard]] timestepping::StepHook hooks() const noexcept override;

    void onStepAccepted(timestepping::TimeHistory& history) override;

    /// Appends the row of @p step at @p state (collective).
    void write(std::uint64_t step, const systems::SystemStateView& state);

    [[nodiscard]] const std::string& path() const noexcept { return path_; }

private:
    const systems::FESystem* system_{nullptr};
    std::string path_{};
    systems::CollectiveCommunicator comm_{};
    std::ofstream out_{};
};

} // namespace post
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_POSTPROCESSING_SCALAR_HISTORY_H
