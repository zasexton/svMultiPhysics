/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_ACCEPTED_STAGE_RECORDS_H
#define SVMP_FE_SYSTEMS_ACCEPTED_STAGE_RECORDS_H

/**
 * @file AcceptedStageRecords.h
 * @brief Generic accepted-stage records of an FE system: declarations that
 *        freeze, the pending group of the current attempt, and the accepted
 *        history, kept as an FE-system attachment (FESystem::attachment).
 *
 * A record family is one AcceptedStageRecordStore type.  Its owner (a
 * physics declaration, a measurement or diagnostic service) validates every
 * declaration and record with its own rules and messages, and uses the
 * handle below for the storage and the attempt transaction:
 *
 * - declarations are made before the first stage and then frozen;
 * - a stage replaces the pending group of the attempt;
 * - a rejected attempt discards the pending group;
 * - an accepted step publishes the pending group to the history on every
 *   rank of the active communicator, or on none (the history is restored
 *   when any rank fails);
 * - records without a pending stage are appended directly.
 *
 * Collective helpers coordinate a local failure across the system's active
 * communicator (record_support in MeshBoundaryRecordSupport.h).  Collective
 * functions must be called on every rank in the same order.
 */

#include "Core/FEException.h"
#include "Systems/FESystem.h"
#include "Systems/MeshBoundaryRecordSupport.h"

#include <algorithm>
#include <cstddef>
#include <exception>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

/**
 * @brief Storage of one accepted-stage record family.
 *
 * @tparam DeclarationT setup-time declaration of one record owner
 * @tparam RecordT accepted (or pending) record of one declaration
 * @tparam TagT distinguishes two families with the same types (the
 *         FE-system attachment is keyed by the store type)
 */
template <class DeclarationT, class RecordT, class TagT = void>
struct AcceptedStageRecordStore {
    using Declaration = DeclarationT;
    using Record = RecordT;
    using Tag = TagT;

    std::vector<Declaration> declarations{};
    std::vector<Record> pending{};
    std::vector<Record> history{};
    bool declarations_frozen{false};
    bool transaction_active{false};
};

/**
 * @brief Run one local phase of a collective record operation.
 *
 * Runs @p local and agrees on its failure across the active communicator
 * (coordinateMeshBoundaryLocalFailure with @p phase).  When any rank
 * failed, @p on_failure runs on every rank, then the failing ranks rethrow
 * their exception and the others throw InvalidArgumentException with
 * @p remote_failure_message.
 */
template <class Local, class OnFailure>
void runCollectiveRecordPhase(
    const record_support::MeshBoundaryCollectiveContext& collective,
    const char* phase,
    const char* remote_failure_message,
    Local&& local,
    OnFailure&& on_failure)
{
    std::exception_ptr local_exception;
    try {
        std::forward<Local>(local)();
    } catch (...) {
        local_exception = std::current_exception();
    }
    if (record_support::coordinateMeshBoundaryLocalFailure(
            collective, local_exception != nullptr, phase)) {
        std::forward<OnFailure>(on_failure)();
        if (local_exception != nullptr) {
            std::rethrow_exception(local_exception);
        }
        throw InvalidArgumentException(remote_failure_message);
    }
}

/// runCollectiveRecordPhase without a failure action.
template <class Local>
void runCollectiveRecordPhase(
    const record_support::MeshBoundaryCollectiveContext& collective,
    const char* phase,
    const char* remote_failure_message,
    Local&& local)
{
    runCollectiveRecordPhase(
        collective,
        phase,
        remote_failure_message,
        std::forward<Local>(local),
        [] {});
}

/**
 * @brief The transaction protocol on an accepted-stage record store.
 *
 * A lightweight handle on a store (normally an FE-system attachment, see
 * acceptedStageRecords()).  It holds no state of its own.
 */
template <class Store>
class AcceptedStageRecords {
public:
    using Declaration = typename Store::Declaration;
    using Record = typename Store::Record;

    explicit AcceptedStageRecords(Store& store) noexcept : store_(&store) {}

    [[nodiscard]] Store& store() const noexcept { return *store_; }

    // ---- Declarations ----

    [[nodiscard]] std::span<const Declaration> declarations() const noexcept
    {
        return store_->declarations;
    }
    [[nodiscard]] bool declarationsFrozen() const noexcept
    {
        return store_->declarations_frozen;
    }
    /// Append a declaration the owner has validated.
    void declare(Declaration declaration)
    {
        store_->declarations.push_back(std::move(declaration));
    }
    void freezeDeclarations() noexcept { store_->declarations_frozen = true; }

    // ---- The attempt transaction ----

    [[nodiscard]] bool transactionActive() const noexcept
    {
        return store_->transaction_active;
    }
    void beginTransaction() noexcept { store_->transaction_active = true; }
    void endTransaction() noexcept { store_->transaction_active = false; }

    [[nodiscard]] std::span<const Record> pending() const noexcept
    {
        return store_->pending;
    }
    /// The pending group, for owners that bind more evidence into it.
    [[nodiscard]] std::span<Record> mutablePending() noexcept
    {
        return store_->pending;
    }
    /// Replace the pending group of the attempt.
    void stagePending(std::vector<Record> group) noexcept
    {
        static_assert(std::is_nothrow_move_assignable_v<std::vector<Record>>);
        store_->pending = std::move(group);
    }
    /// Discard the pending group of a rejected attempt.
    void discardPending() noexcept
    {
        store_->pending.clear();
        store_->transaction_active = false;
    }
    /// Drop the pending group without ending a transaction (an identical
    /// replay of a published step).
    void clearPending() noexcept { store_->pending.clear(); }

    /**
     * @brief Publish the pending group to the history on every rank, or on
     *        none.
     *
     * The records are moved to the end of the history while the transaction
     * is active.  When any rank of @p collective failed, every rank moves
     * its records back, restores the history, ends the transaction and
     * rethrows (or throws InvalidArgumentException with
     * @p remote_failure_message).  Otherwise the pending group is cleared
     * and the transaction ends.
     *
     * @return the history index of the first published record
     */
    std::size_t publishPendingCollectively(
        const record_support::MeshBoundaryCollectiveContext& collective,
        const char* phase,
        const char* remote_failure_message)
    {
        static_assert(std::is_nothrow_move_constructible_v<Record>);
        static_assert(std::is_nothrow_move_assignable_v<Record>);
        auto& pending_records = store_->pending;
        auto& history_records = store_->history;
        const auto history_size = history_records.size();
        std::size_t appended_count = 0u;
        std::exception_ptr local_exception;
        store_->transaction_active = true;
        try {
            for (auto& record : pending_records) {
                history_records.push_back(std::move(record));
                ++appended_count;
            }
        } catch (...) {
            local_exception = std::current_exception();
        }
        if (record_support::coordinateMeshBoundaryLocalFailure(
                collective, local_exception != nullptr, phase)) {
            for (std::size_t index = 0u; index < appended_count; ++index) {
                pending_records[index] =
                    std::move(history_records[history_size + index]);
            }
            history_records.resize(history_size);
            store_->transaction_active = false;
            if (local_exception != nullptr) {
                std::rethrow_exception(local_exception);
            }
            throw InvalidArgumentException(remote_failure_message);
        }
        pending_records.clear();
        store_->transaction_active = false;
        return history_size;
    }

    // ---- The accepted history ----

    [[nodiscard]] std::span<const Record> history() const noexcept
    {
        return store_->history;
    }
    /// Reserve room for exactly @p additional more records.
    void reserveHistory(std::size_t additional)
    {
        store_->history.reserve(store_->history.size() + additional);
    }
    /// Reserve room for @p additional more records, growing geometrically.
    void growHistoryFor(std::size_t additional)
    {
        auto& history_records = store_->history;
        const auto required = history_records.size() + additional;
        if (required > history_records.capacity()) {
            history_records.reserve(
                std::max(required, 2u * history_records.capacity()));
        }
    }
    /// Append one accepted record the owner has validated.
    void appendHistory(Record record)
    {
        store_->history.push_back(std::move(record));
    }

private:
    Store* store_;
};

/**
 * @brief Read-only view of an accepted-stage record store; empty when the
 *        store does not exist.
 */
template <class Store>
class ConstAcceptedStageRecords {
public:
    using Declaration = typename Store::Declaration;
    using Record = typename Store::Record;

    explicit ConstAcceptedStageRecords(const Store* store) noexcept
        : store_(store)
    {
    }

    [[nodiscard]] bool exists() const noexcept { return store_ != nullptr; }
    [[nodiscard]] std::span<const Declaration> declarations() const noexcept
    {
        return store_ == nullptr ? std::span<const Declaration>{}
                                 : std::span<const Declaration>(
                                       store_->declarations);
    }
    [[nodiscard]] bool declarationsFrozen() const noexcept
    {
        return store_ != nullptr && store_->declarations_frozen;
    }
    [[nodiscard]] bool transactionActive() const noexcept
    {
        return store_ != nullptr && store_->transaction_active;
    }
    [[nodiscard]] std::span<const Record> pending() const noexcept
    {
        return store_ == nullptr ? std::span<const Record>{}
                                 : std::span<const Record>(store_->pending);
    }
    [[nodiscard]] std::span<const Record> history() const noexcept
    {
        return store_ == nullptr ? std::span<const Record>{}
                                 : std::span<const Record>(store_->history);
    }

private:
    const Store* store_;
};

/// The record store of type @p Store attached to @p system (created on
/// first use).
template <class Store>
[[nodiscard]] AcceptedStageRecords<Store> acceptedStageRecords(
    FESystem& system)
{
    return AcceptedStageRecords<Store>(system.template attachment<Store>());
}

/// The record store of type @p Store attached to @p system, read-only.
template <class Store>
[[nodiscard]] ConstAcceptedStageRecords<Store> acceptedStageRecords(
    const FESystem& system) noexcept
{
    return ConstAcceptedStageRecords<Store>(
        system.template findAttachment<Store>());
}

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_ACCEPTED_STAGE_RECORDS_H
