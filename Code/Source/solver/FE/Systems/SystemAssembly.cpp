/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/SystemAssembly.h"

#include "Systems/FESystem.h"
#include "Auxiliary/AuxiliaryStateManager.h"
#include "Auxiliary/AuxiliaryInputRegistry.h"
#include "Auxiliary/AuxiliaryOperatorRegistry.h"
#include "Systems/GlobalKernel.h"
#include "Systems/SystemsExceptions.h"

#include "Assembly/Assembler.h"
#include "Assembly/GlobalSystemView.h"
#include "Assembly/AssemblyKernel.h"
#include "Assembly/TimeIntegrationContext.h"

#include "Forms/FormKernels.h"
#include "Forms/JIT/JITKernelWrapper.h"
#include "Forms/JIT/ExternalCalls.h"

#include "Backends/Interfaces/GenericVector.h"
#include "Backends/FSILS/FsilsMatrix.h"
#include "Core/KernelTrace.h"
#include "Core/Logger.h"
#include "Core/Alignment.h"
#include "Core/AlignedAllocator.h"
#include "Core/FEConfig.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <type_traits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if FE_HAS_MPI
#  include <mpi.h>
#endif

namespace svmp {
namespace FE {
namespace systems {

namespace {

#if FE_HAS_MPI
constexpr std::uint64_t kBorderedConsistencyHashSeed = 1469598103934665603ULL;

[[nodiscard]] std::uint64_t mixBorderedConsistencyHash(std::uint64_t hash,
                                                       std::uint64_t value) noexcept
{
    hash ^= value + 0x9e3779b97f4a7c15ULL + (hash << 6U) + (hash >> 2U);
    return hash;
}

template <class T>
void hashBorderedConsistencyBytes(std::uint64_t& hash, const T& value)
{
    static_assert(std::is_trivially_copyable_v<T>);
    const auto* bytes = reinterpret_cast<const unsigned char*>(&value);
    for (std::size_t i = 0; i < sizeof(T); ++i) {
        hash = mixBorderedConsistencyHash(hash, static_cast<std::uint64_t>(bytes[i]));
    }
}

template <class T>
void hashBorderedConsistencySpan(std::uint64_t& hash, std::span<const T> values)
{
    hashBorderedConsistencyBytes(hash, values.size());
    for (const auto& value : values) {
        hashBorderedConsistencyBytes(hash, value);
    }
}

void hashBorderedConsistencyString(std::uint64_t& hash, const std::string& value)
{
    hashBorderedConsistencyBytes(hash, value.size());
    for (const char ch : value) {
        hashBorderedConsistencyBytes(hash, ch);
    }
}

[[nodiscard]] bool borderedCouplingShapeConsistentAcrossRanks(
    const FESystem::BorderedCouplingData& bc,
    bool sync_matrix_terms,
    bool sync_vector_terms,
    MPI_Comm comm)
{
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized) {
        return true;
    }

    int world_size = 1;
    MPI_Comm_size(comm, &world_size);
    if (world_size <= 1) {
        return true;
    }

    std::uint64_t local_hash = kBorderedConsistencyHashSeed;
    hashBorderedConsistencyBytes(local_hash, bc.active);
    hashBorderedConsistencyBytes(local_hash, sync_matrix_terms);
    hashBorderedConsistencyBytes(local_hash, sync_vector_terms);
    hashBorderedConsistencyBytes(local_hash, bc.globally_reduced);
    hashBorderedConsistencyBytes(local_hash, bc.aux_self_terms_replicated);
    hashBorderedConsistencyBytes(local_hash, bc.n_aux);
    hashBorderedConsistencyBytes(local_hash, bc.n_field_dofs);
    hashBorderedConsistencyBytes(local_hash, bc.D.size());
    hashBorderedConsistencyBytes(local_hash, bc.g.size());
    hashBorderedConsistencyBytes(local_hash, bc.B.size());
    hashBorderedConsistencyBytes(local_hash, bc.Ct.size());
    hashBorderedConsistencyBytes(local_hash, bc.dF_dxdot.size());
    hashBorderedConsistencySpan(local_hash, std::span<const int>(bc.aux_row_owner_ranks));
    hashBorderedConsistencySpan(local_hash, std::span<const char>(bc.aux_row_owner_routed));
    hashBorderedConsistencyBytes(local_hash, bc.aux_variable_kinds.size());
    for (const auto kind : bc.aux_variable_kinds) {
        hashBorderedConsistencyBytes(local_hash, static_cast<std::uint8_t>(kind));
    }
    hashBorderedConsistencyBytes(local_hash, bc.aux_blocks.size());
    for (const auto& block : bc.aux_blocks) {
        hashBorderedConsistencyString(local_hash, block.name);
        hashBorderedConsistencyBytes(local_hash, block.dim);
    }

    int local_count = 0;
    local_count += 1;
    local_count += 5;
    local_count += static_cast<int>(bc.aux_row_owner_ranks.size());
    local_count += static_cast<int>(bc.aux_row_owner_routed.size());
    local_count += 2 * static_cast<int>(bc.aux_variable_kinds.size());
    local_count += 2 * static_cast<int>(bc.aux_blocks.size());

    int min_count = 0;
    int max_count = 0;
    MPI_Allreduce(&local_count, &min_count, 1, MPI_INT, MPI_MIN, comm);
    MPI_Allreduce(&local_count, &max_count, 1, MPI_INT, MPI_MAX, comm);

    std::uint64_t min_hash = 0ULL;
    std::uint64_t max_hash = 0ULL;
    MPI_Allreduce(&local_hash, &min_hash, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, comm);
    MPI_Allreduce(&local_hash, &max_hash, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, comm);

    return min_count == max_count && min_hash == max_hash;
}

[[nodiscard]] bool replicatedBorderedCouplingConsistentAcrossRanks(
    const FESystem::BorderedCouplingData& bc,
    bool sync_matrix_terms,
    bool sync_vector_terms,
    MPI_Comm comm)
{
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized) {
        return true;
    }

    int world_size = 1;
    MPI_Comm_size(comm, &world_size);
    if (world_size <= 1) {
        return true;
    }

    std::uint64_t local_hash = kBorderedConsistencyHashSeed;
    hashBorderedConsistencyBytes(local_hash, bc.active);
    hashBorderedConsistencyBytes(local_hash, bc.globally_reduced);
    hashBorderedConsistencyBytes(local_hash, bc.aux_self_terms_replicated);
    hashBorderedConsistencyBytes(local_hash, bc.n_aux);
    hashBorderedConsistencyBytes(local_hash, bc.n_field_dofs);
    hashBorderedConsistencySpan(local_hash, std::span<const int>(bc.aux_row_owner_ranks));
    hashBorderedConsistencySpan(local_hash, std::span<const char>(bc.aux_row_owner_routed));
    hashBorderedConsistencySpan(local_hash, std::span<const int>(bc.aux_row_global_contributor_counts));

    hashBorderedConsistencyBytes(local_hash, bc.aux_variable_kinds.size());
    for (const auto kind : bc.aux_variable_kinds) {
        hashBorderedConsistencyBytes(local_hash, static_cast<std::uint8_t>(kind));
    }

    hashBorderedConsistencyBytes(local_hash, bc.aux_blocks.size());
    for (const auto& block : bc.aux_blocks) {
        hashBorderedConsistencyString(local_hash, block.name);
        hashBorderedConsistencyBytes(local_hash, block.dim);
    }

    int local_count = 0;
    local_count += 5;
    local_count += static_cast<int>(bc.aux_row_owner_ranks.size());
    local_count += static_cast<int>(bc.aux_row_owner_routed.size());
    local_count += static_cast<int>(bc.aux_row_global_contributor_counts.size());
    local_count += 2 * static_cast<int>(bc.aux_variable_kinds.size());
    local_count += 2 * static_cast<int>(bc.aux_blocks.size());

    if (sync_matrix_terms) {
        hashBorderedConsistencySpan(local_hash, std::span<const Real>(bc.B));
        hashBorderedConsistencySpan(local_hash, std::span<const Real>(bc.Ct));
        local_count += static_cast<int>(bc.B.size() + bc.Ct.size());
        if (bc.aux_self_terms_replicated || !bc.D.empty() || !bc.dF_dxdot.empty()) {
            hashBorderedConsistencySpan(local_hash, std::span<const Real>(bc.D));
            hashBorderedConsistencySpan(local_hash, std::span<const Real>(bc.dF_dxdot));
            local_count += static_cast<int>(bc.D.size() + bc.dF_dxdot.size());
        }
    }

    if (sync_vector_terms) {
        hashBorderedConsistencySpan(local_hash, std::span<const Real>(bc.g));
        local_count += static_cast<int>(bc.g.size());
    }

    int min_count = 0;
    int max_count = 0;
    MPI_Allreduce(&local_count, &min_count, 1, MPI_INT, MPI_MIN, comm);
    MPI_Allreduce(&local_count, &max_count, 1, MPI_INT, MPI_MAX, comm);

    std::uint64_t min_hash = 0ULL;
    std::uint64_t max_hash = 0ULL;
    MPI_Allreduce(&local_hash, &min_hash, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, comm);
    MPI_Allreduce(&local_hash, &max_hash, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, comm);

    return min_count == max_count && min_hash == max_hash;
}
#endif

std::string summarizeSparsePairsForTrace(std::span<const std::pair<GlobalIndex, Real>> entries,
                                         std::size_t max_items = 8)
{
    long double sq_norm = 0.0L;
    Real max_abs = Real(0.0);
    GlobalIndex max_dof = INVALID_GLOBAL_INDEX;
    for (const auto& [dof, value] : entries) {
        const Real abs_value = std::abs(value);
        sq_norm += static_cast<long double>(value) * static_cast<long double>(value);
        if (abs_value > max_abs) {
            max_abs = abs_value;
            max_dof = dof;
        }
    }

    std::ostringstream oss;
    oss << "nnz=" << entries.size()
        << " l2=" << std::sqrt(static_cast<double>(sq_norm))
        << " max_abs=" << max_abs
        << " max_dof=" << max_dof;

    const auto n_show = std::min(max_items, entries.size());
    if (n_show > 0) {
        oss << " entries=[";
        for (std::size_t i = 0; i < n_show; ++i) {
            if (i > 0) {
                oss << ", ";
            }
            oss << entries[i].first << ":" << entries[i].second;
        }
        if (entries.size() > n_show) {
            oss << ", ...";
        }
        oss << "]";
    }

    return oss.str();
}

[[nodiscard]] bool oopTraceEnabled() noexcept
{
    return core::kernelTraceEnabled(core::KernelTraceChannel::Assembly);
}

void traceLog(const std::string& msg)
{
    core::kernelTraceLog(core::KernelTraceChannel::Assembly, msg);
}

void mergeAssemblyResult(assembly::AssemblyResult& total, const assembly::AssemblyResult& part)
{
    if (!part.success && total.success) {
        total.success = false;
        total.error_message = part.error_message;
    } else if (!part.success && !part.error_message.empty()) {
        if (!total.error_message.empty()) {
            total.error_message += "\n";
        }
        total.error_message += part.error_message;
    }

    total.elements_assembled += part.elements_assembled;
    total.boundary_faces_assembled += part.boundary_faces_assembled;
    total.interior_faces_assembled += part.interior_faces_assembled;
    total.interface_faces_assembled += part.interface_faces_assembled;
    total.elapsed_time_seconds += part.elapsed_time_seconds;
    total.matrix_entries_inserted += part.matrix_entries_inserted;
    total.vector_entries_inserted += part.vector_entries_inserted;
}

#if FE_HAS_MPI
MPI_Datatype mpiRealType()
{
    if (sizeof(Real) == sizeof(double)) {
        return MPI_DOUBLE;
    }
    if (sizeof(Real) == sizeof(float)) {
        return MPI_FLOAT;
    }
    return MPI_LONG_DOUBLE;
}

MPI_Datatype mpiGlobalIndexType()
{
    if (sizeof(GlobalIndex) == sizeof(std::int64_t)) {
        return MPI_INT64_T;
    }
    if (sizeof(GlobalIndex) == sizeof(long long)) {
        return MPI_LONG_LONG;
    }
    if (sizeof(GlobalIndex) == sizeof(long)) {
        return MPI_LONG;
    }
    return MPI_LONG_LONG;
}

std::vector<std::pair<GlobalIndex, Real>> allreduceSumSparsePairs(std::vector<std::pair<GlobalIndex, Real>> local,
                                                                  MPI_Comm comm)
{
    int comm_size = 1;
    MPI_Comm_size(comm, &comm_size);
    if (comm_size <= 1) {
        return local;
    }

    const int local_n = static_cast<int>(local.size());
    std::vector<int> counts(static_cast<std::size_t>(comm_size), 0);
    MPI_Allgather(&local_n, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);

    std::vector<int> displs(static_cast<std::size_t>(comm_size), 0);
    int total_n = 0;
    for (int r = 0; r < comm_size; ++r) {
        displs[static_cast<std::size_t>(r)] = total_n;
        total_n += counts[static_cast<std::size_t>(r)];
    }

    std::vector<GlobalIndex> idx_local(static_cast<std::size_t>(local_n), GlobalIndex(0));
    std::vector<Real> val_local(static_cast<std::size_t>(local_n), Real(0.0));
    for (int i = 0; i < local_n; ++i) {
        idx_local[static_cast<std::size_t>(i)] = local[static_cast<std::size_t>(i)].first;
        val_local[static_cast<std::size_t>(i)] = local[static_cast<std::size_t>(i)].second;
    }

    std::vector<GlobalIndex> idx_all(static_cast<std::size_t>(total_n), GlobalIndex(0));
    std::vector<Real> val_all(static_cast<std::size_t>(total_n), Real(0.0));
    MPI_Allgatherv(idx_local.data(),
                   local_n,
                   mpiGlobalIndexType(),
                   idx_all.data(),
                   counts.data(),
                   displs.data(),
                   mpiGlobalIndexType(),
                   comm);
    MPI_Allgatherv(val_local.data(),
                   local_n,
                   mpiRealType(),
                   val_all.data(),
                   counts.data(),
                   displs.data(),
                   mpiRealType(),
                   comm);

    std::vector<std::pair<GlobalIndex, Real>> merged;
    merged.reserve(static_cast<std::size_t>(total_n));
    for (int i = 0; i < total_n; ++i) {
        merged.emplace_back(idx_all[static_cast<std::size_t>(i)], val_all[static_cast<std::size_t>(i)]);
    }

    std::sort(merged.begin(), merged.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    std::vector<std::pair<GlobalIndex, Real>> out;
    out.reserve(merged.size());
    for (const auto& kv : merged) {
        if (out.empty() || kv.first != out.back().first) {
            out.push_back(kv);
        } else {
            out.back().second += kv.second;
        }
    }
    return out;
}

void allreduceDenseEntries(std::vector<Real>& data, MPI_Comm comm)
{
    if (data.empty()) {
        return;
    }

    std::vector<Real> global(data.size(), Real(0.0));
    MPI_Allreduce(data.data(),
                  global.data(),
                  static_cast<int>(data.size()),
                  mpiRealType(),
                  MPI_SUM,
                  comm);
    data.swap(global);
}

void allreduceIntEntries(std::vector<int>& data, MPI_Comm comm)
{
    if (data.empty()) {
        return;
    }

    std::vector<int> global(data.size(), 0);
    MPI_Allreduce(data.data(),
                  global.data(),
                  static_cast<int>(data.size()),
                  MPI_INT,
                  MPI_SUM,
                  comm);
    data.swap(global);
}
#endif

template <class OwnedPredicate>
std::vector<std::pair<GlobalIndex, Real>>
filterSparsePairsToOwned(std::span<const std::pair<GlobalIndex, Real>> pairs,
                         OwnedPredicate&& is_owned)
{
    std::vector<std::pair<GlobalIndex, Real>> owned_pairs;
    owned_pairs.reserve(pairs.size());
    for (const auto& kv : pairs) {
        if (is_owned(kv.first)) {
            owned_pairs.push_back(kv);
        }
    }
    return owned_pairs;
}

void synchronizeBorderedCouplingForReplicatedSolve(FESystem::BorderedCouplingData& bc,
                                                   bool sync_matrix_terms,
                                                   bool sync_vector_terms,
                                                   MPI_Comm comm)
{
    if (!bc.active) {
        return;
    }

#if FE_HAS_MPI
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized) {
        if (!borderedCouplingShapeConsistentAcrossRanks(
                bc, sync_matrix_terms, sync_vector_terms, comm)) {
            FE_THROW(FEException,
                     "assembleOperator: bordered coupling structure differs across MPI ranks "
                     "before replicated dense reduction.");
        }
        if (sync_matrix_terms) {
            allreduceDenseEntries(bc.B, comm);
            allreduceDenseEntries(bc.Ct, comm);
            if (!bc.aux_self_terms_replicated) {
                allreduceDenseEntries(bc.D, comm);
                allreduceDenseEntries(bc.dF_dxdot, comm);
            }
        }
        if (sync_vector_terms) {
            if (!bc.aux_self_terms_replicated) {
                allreduceDenseEntries(bc.g, comm);
            }
        }
        bc.aux_row_global_contributor_counts =
            bc.aux_row_local_contribution_flags;
        allreduceIntEntries(bc.aux_row_global_contributor_counts, comm);

        bc.globally_reduced = true;
        if (!replicatedBorderedCouplingConsistentAcrossRanks(
                bc, sync_matrix_terms, sync_vector_terms, comm)) {
            FE_THROW(FEException,
                     "assembleOperator: replicated bordered coupling data differs across MPI "
                     "ranks after synchronization.");
        }
        return;
    }
#endif

    bc.globally_reduced = true;
    bc.aux_row_global_contributor_counts =
        bc.aux_row_local_contribution_flags;
}

class SparseVectorAccumulatorView final : public assembly::GlobalSystemView {
public:
    explicit SparseVectorAccumulatorView(GlobalIndex size)
        : size_(size)
    {
    }

    // Matrix operations: not supported for this view.
    void addMatrixEntries(std::span<const GlobalIndex> /*dofs*/,
                          std::span<const Real> /*local_matrix*/,
                          assembly::AddMode /*mode*/ = assembly::AddMode::Add) override
    {
    }

    void addMatrixEntries(std::span<const GlobalIndex> /*row_dofs*/,
                          std::span<const GlobalIndex> /*col_dofs*/,
                          std::span<const Real> /*local_matrix*/,
                          assembly::AddMode /*mode*/ = assembly::AddMode::Add) override
    {
    }

    void addMatrixEntry(GlobalIndex /*row*/,
                        GlobalIndex /*col*/,
                        Real /*value*/,
                        assembly::AddMode /*mode*/ = assembly::AddMode::Add) override
    {
    }

    void setDiagonal(std::span<const GlobalIndex> /*dofs*/,
                     std::span<const Real> /*values*/) override
    {
    }

    void setDiagonal(GlobalIndex /*dof*/, Real /*value*/) override
    {
    }

    void zeroRows(std::span<const GlobalIndex> /*rows*/,
                  bool /*set_diagonal*/ = true) override
    {
    }

    // Vector operations
    void addVectorEntries(std::span<const GlobalIndex> dofs,
                          std::span<const Real> local_vector,
                          assembly::AddMode mode = assembly::AddMode::Add) override
    {
        FE_THROW_IF(dofs.size() != local_vector.size(), InvalidArgumentException,
                    "SparseVectorAccumulatorView::addVectorEntries: size mismatch");
        for (std::size_t i = 0; i < dofs.size(); ++i) {
            addVectorEntry(dofs[i], local_vector[i], mode);
        }
    }

    void addVectorEntry(GlobalIndex dof,
                        Real value,
                        assembly::AddMode mode = assembly::AddMode::Add) override
    {
        if (dof < 0 || dof >= size_) {
            return;
        }
        switch (mode) {
            case assembly::AddMode::Add:
                values_[dof] += value;
                break;
            case assembly::AddMode::Insert:
                values_[dof] = value;
                break;
            case assembly::AddMode::Max: {
                auto& v = values_[dof];
                v = std::max(v, value);
                break;
            }
            case assembly::AddMode::Min: {
                auto& v = values_[dof];
                v = std::min(v, value);
                break;
            }
        }
    }

    void setVectorEntries(std::span<const GlobalIndex> dofs,
                          std::span<const Real> values) override
    {
        FE_THROW_IF(dofs.size() != values.size(), InvalidArgumentException,
                    "SparseVectorAccumulatorView::setVectorEntries: size mismatch");
        for (std::size_t i = 0; i < dofs.size(); ++i) {
            const auto dof = dofs[i];
            if (dof < 0 || dof >= size_) {
                continue;
            }
            values_[dof] = values[i];
        }
    }

    void zeroVectorEntries(std::span<const GlobalIndex> dofs) override
    {
        for (const auto dof : dofs) {
            if (dof < 0 || dof >= size_) continue;
            values_.erase(dof);
        }
    }

    [[nodiscard]] Real getVectorEntry(GlobalIndex dof) const override
    {
        if (dof < 0 || dof >= size_) {
            return 0.0;
        }
        auto it = values_.find(dof);
        if (it == values_.end()) {
            return 0.0;
        }
        return it->second;
    }

    void beginAssemblyPhase() override { phase_ = assembly::AssemblyPhase::Building; }
    void endAssemblyPhase() override { phase_ = assembly::AssemblyPhase::Flushing; }
    void finalizeAssembly() override { phase_ = assembly::AssemblyPhase::Finalized; }
    [[nodiscard]] assembly::AssemblyPhase getPhase() const noexcept override { return phase_; }

    [[nodiscard]] bool hasMatrix() const noexcept override { return false; }
    [[nodiscard]] bool hasVector() const noexcept override { return true; }
    [[nodiscard]] GlobalIndex numRows() const noexcept override { return size_; }
    [[nodiscard]] GlobalIndex numCols() const noexcept override { return 1; }
    [[nodiscard]] std::string backendName() const override { return "SparseVectorAccumulator"; }

    void zero() override { values_.clear(); }

    [[nodiscard]] const std::unordered_map<GlobalIndex, Real>& values() const noexcept { return values_; }

    [[nodiscard]] std::vector<std::pair<GlobalIndex, Real>> entriesSorted(Real abs_tol = 0.0) const
    {
        std::vector<std::pair<GlobalIndex, Real>> out;
        out.reserve(values_.size());
        for (const auto& kv : values_) {
            if (std::abs(kv.second) <= abs_tol) continue;
            out.emplace_back(kv.first, kv.second);
        }
        std::sort(out.begin(), out.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        return out;
    }

private:
    GlobalIndex size_{0};
    std::unordered_map<GlobalIndex, Real> values_{};
    assembly::AssemblyPhase phase_{assembly::AssemblyPhase::NotStarted};
};

} // namespace

assembly::AssemblyResult assembleOperator(
    FESystem& system,
    const AssemblyRequest& request,
    const SystemStateView& state,
    assembly::GlobalSystemView* matrix_out,
    assembly::GlobalSystemView* vector_out)
{
    system.requireSetup();
    system.last_rank_one_updates_.clear();
    system.last_reduced_field_updates_.clear();

    FE_THROW_IF(request.op.empty(), InvalidArgumentException, "assembleOperator: empty operator tag");
    FE_THROW_IF(!system.operator_registry_.has(request.op), InvalidArgumentException,
                "assembleOperator: unknown operator '" + request.op + "'");

    FE_THROW_IF(request.want_matrix && matrix_out == nullptr, InvalidArgumentException,
                "assembleOperator: want_matrix but matrix_out is null");
    FE_THROW_IF(request.want_vector && vector_out == nullptr, InvalidArgumentException,
                "assembleOperator: want_vector but vector_out is null");
    FE_THROW_IF(!request.want_matrix && !request.want_vector, InvalidArgumentException,
                "assembleOperator: nothing requested (want_matrix=false and want_vector=false)");

    FE_CHECK_NOT_NULL(system.assembler_.get(), "FESystem::assembler");

    // Partitioned auxiliary blocks must be stepped from committed state
    // before assembly so their outputs match the current FE iterate. This
    // mirrors the legacy coupled-boundary timing, while monolithic blocks
    // remain part of the assembled global system and are not stepped here.
    bool has_partitioned_auxiliary = false;
    for (const auto& entry : system.deployed_aux_entries_) {
        if (entry.spec.solve_mode == AuxiliarySolveMode::Partitioned) {
            has_partitioned_auxiliary = true;
            break;
        }
    }

    auto same_partitioned_auxiliary_step = [&](Real time, Real dt) {
        if (!system.partitioned_auxiliary_advance_valid_) {
            return false;
        }
        const auto nearly_equal = [](Real a, Real b) {
            const auto scale = std::max<Real>(
                Real(1.0), std::max(std::abs(a), std::abs(b)));
            return std::abs(a - b) <= Real(1e-12) * scale;
        };
        return nearly_equal(system.partitioned_auxiliary_advance_time_, time) &&
               nearly_equal(system.partitioned_auxiliary_advance_dt_, dt);
    };

    if (has_partitioned_auxiliary && !same_partitioned_auxiliary_step(state.time, state.dt)) {
        // Advance partitioned auxiliary blocks exactly once for the current
        // nonlinear solve state. The first assembly of a time step is often a
        // Newton iteration, so gating on request.is_nonlinear_iteration would
        // leave the outlet state frozen at its committed value.
        if (oopTraceEnabled()) {
            traceLog("assembleOperator: advanceAuxiliaryState() begin");
        }
        const auto t0 = std::chrono::steady_clock::now();
        system.advanceAuxiliaryState(state, request.is_nonlinear_iteration);
        const auto t1 = std::chrono::steady_clock::now();
        if (oopTraceEnabled()) {
            std::ostringstream oss;
            oss << "assembleOperator: advanceAuxiliaryState() done time="
                << std::chrono::duration<double>(t1 - t0).count();
            traceLog(oss.str());
        }
    }

    // Always refresh inputs and re-evaluate outputs for every assembly
    // pass (including Newton iterations), so P_out = X + Rp*Q uses the
    // latest flow rate Q computed from the current velocity iterate.
    system.prepareAuxiliaryForAssembly(state, request.is_nonlinear_iteration);

    if (request.zero_outputs) {
        if (request.want_matrix) {
            matrix_out->zero();
        }
        if (request.want_vector && vector_out != matrix_out) {
            vector_out->zero();
        } else if (request.want_vector && vector_out == matrix_out) {
            vector_out->zero();
        }
    }

    if (request.want_matrix) {
        auto it = system.sparsity_by_op_.find(request.op);
        if (it != system.sparsity_by_op_.end() && it->second) {
            system.assembler_->setSparsityPattern(it->second.get());
        }
    } else {
        system.assembler_->setSparsityPattern(nullptr);
    }

    auto& assembler = *system.assembler_;
    assembler.setCurrentSolution(state.u);
    std::unique_ptr<assembly::GlobalSystemView> current_solution_view;
    if (state.u_vector != nullptr) {
        // `createAssemblyView()` is non-const for historical reasons; treat this as read-only use.
        auto* vec = const_cast<backends::GenericVector*>(state.u_vector);
        current_solution_view = vec->createAssemblyView();
    }
    assembler.setCurrentSolutionView(current_solution_view.get());
    {
        std::vector<assembly::FieldSolutionAccess> access;
        access.reserve(system.field_registry_.size());
        for (const auto& rec : system.field_registry_.records()) {
            FE_CHECK_NOT_NULL(rec.space.get(), "assembleOperator: field space");
            const auto idx = static_cast<std::size_t>(rec.id);
            FE_THROW_IF(idx >= system.field_dof_handlers_.size(), InvalidStateException,
                        "assembleOperator: invalid field DOF handler index for field '" + rec.name + "'");
            FE_THROW_IF(idx >= system.field_dof_offsets_.size(), InvalidStateException,
                        "assembleOperator: invalid field DOF offset index for field '" + rec.name + "'");
            access.push_back(assembly::FieldSolutionAccess{
                rec.id,
                rec.space.get(),
                &system.field_dof_handlers_[idx].getDofMap(),
                system.field_dof_offsets_[idx],
            });
        }
        assembler.setFieldSolutionAccess(access);
    }
    assembler.setTimeIntegrationContext(state.time_integration);

    int required_history = 0;
    if (state.time_integration != nullptr) {
        if (state.time_integration->dt1) {
            required_history = std::max(required_history, state.time_integration->dt1->requiredHistoryStates());
        }
        if (state.time_integration->dt2) {
            required_history = std::max(required_history, state.time_integration->dt2->requiredHistoryStates());
        }
        for (const auto& s : state.time_integration->dt_extra) {
            if (s) {
                required_history = std::max(required_history, s->requiredHistoryStates());
            }
        }
    }

    std::unique_ptr<assembly::GlobalSystemView> prev_solution_view;
    std::unique_ptr<assembly::GlobalSystemView> prev2_solution_view;

    if (required_history > 0) {
        if (!state.u_history.empty()) {
            FE_THROW_IF(static_cast<int>(state.u_history.size()) < required_history, InvalidArgumentException,
                        "assembleOperator: insufficient solution history (need " + std::to_string(required_history) +
                            ", have " + std::to_string(state.u_history.size()) + ")");
            for (int k = 1; k <= required_history; ++k) {
                assembler.setPreviousSolutionK(k, state.u_history[static_cast<std::size_t>(k - 1)]);
            }
        } else {
            // Backward-compatible path (supports up to 2 history states).
            FE_THROW_IF(required_history > 2, InvalidArgumentException,
                        "assembleOperator: time integration requires more than 2 history states, but state.u_history was not provided");
            assembler.setPreviousSolution(state.u_prev);
            assembler.setPreviousSolution2(state.u_prev2);
        }

        // Provide global-indexed views for history states when available (needed by some backends).
        if (required_history >= 1) {
            if (state.u_prev_vector != nullptr) {
                auto* vec = const_cast<backends::GenericVector*>(state.u_prev_vector);
                prev_solution_view = vec->createAssemblyView();
                assembler.setPreviousSolutionView(prev_solution_view.get());
            } else {
                assembler.setPreviousSolutionView(nullptr);
            }
        }
        if (required_history >= 2) {
            if (state.u_prev2_vector != nullptr) {
                auto* vec = const_cast<backends::GenericVector*>(state.u_prev2_vector);
                prev2_solution_view = vec->createAssemblyView();
                assembler.setPreviousSolution2View(prev2_solution_view.get());
            } else {
                assembler.setPreviousSolution2View(nullptr);
            }
        }
    } else {
        // No dt(...) required by the active time-integration context.
        assembler.setPreviousSolution({});
        assembler.setPreviousSolution2({});
        assembler.setPreviousSolutionView(nullptr);
        assembler.setPreviousSolution2View(nullptr);
    }

    // Optional parameter validation + defaults.
    system.parameter_registry_.validate(state);
    std::function<std::optional<Real>(std::string_view)> get_real_param_wrapped{};
    std::function<std::optional<params::Value>(std::string_view)> get_param_wrapped{};

    const bool have_param_contracts = !system.parameter_registry_.specs().empty();
    if (have_param_contracts) {
        get_real_param_wrapped = system.parameter_registry_.makeRealGetter(state);
        get_param_wrapped = system.parameter_registry_.makeParamGetter(state);
    }

    assembler.setTime(static_cast<Real>(state.time));
    assembler.setTimeStep(static_cast<Real>(state.dt));
    assembler.setRealParameterGetter(have_param_contracts
                                         ? &get_real_param_wrapped
                                         : (state.getRealParam ? &state.getRealParam : nullptr));
    assembler.setParameterGetter(have_param_contracts
                                     ? &get_param_wrapped
                                     : (state.getParam ? &state.getParam : nullptr));
    forms::jit::external::ExternalCallTableV1 jit_table;
    jit_table.context = state.user_data;
    assembler.setUserData(&jit_table);

    // JIT-friendly constant slots (Real-valued parameters resolved to stable indices).
    std::vector<Real, AlignedAllocator<Real, kFEPreferredAlignmentBytes>> jit_constants;
    if (have_param_contracts && system.parameter_registry_.slotCount() > 0u) {
        const auto slots = system.parameter_registry_.evaluateRealSlots(state);
        jit_constants.assign(slots.begin(), slots.end());
        assembler.setJITConstants(jit_constants);
    } else {
        assembler.setJITConstants({});
    }

    assembler.setCoupledValues({}, {});

    // Inject generalized auxiliary values (neutral path).
    // These populate the auxiliary_inputs/auxiliary_state/auxiliary_outputs fields
    // in AssemblyContext, used by AuxiliaryInputRef/AuxiliaryOutputRef terminals.
    {
        auto* input_reg = system.auxiliaryInputRegistryIfPresent();
        auto* aux_mgr = system.auxiliaryStateManagerIfPresent();

        std::span<const Real> aux_inputs;
        if (input_reg && input_reg->totalSize() > 0) {
            aux_inputs = input_reg->all();
        }

        std::span<const Real> aux_state_flat;
        if (aux_mgr && aux_mgr->blockCount() > 0) {
            aux_state_flat = system.auxiliaryStateValues();
        }

        auto aux_outputs = system.auxiliaryOutputValues();
        assembler.setAuxiliaryOutputBindings(system.auxiliaryOutputBindings());

        if (!aux_inputs.empty() || !aux_state_flat.empty() || !aux_outputs.empty()) {
            if (oopTraceEnabled() && !aux_outputs.empty()) {
                std::ostringstream oss;
                oss << "assembleOperator: auxiliary outputs=[";
                for (std::size_t i = 0; i < aux_outputs.size(); ++i) {
                    if (i != 0) {
                        oss << ", ";
                    }
                    oss << aux_outputs[i];
                }
                oss << "]";
                traceLog(oss.str());
            }
            assembler.setAuxiliaryValues(aux_inputs, aux_state_flat, aux_outputs);
        }
    }

    const auto& mesh = system.meshAccess();

    assembler.setSuppressConstraintInhomogeneity(request.suppress_constraint_inhomogeneity);

    const auto& def = system.operator_registry_.get(request.op);
    auto plan_it = system.assembly_plan_by_op_.find(request.op);
    FE_THROW_IF(plan_it == system.assembly_plan_by_op_.end(), InvalidStateException,
                "assembleOperator: missing assembly plan for operator '" + request.op + "'");
    const auto& plan = plan_it->second;

    assembly::AssemblyResult total;

#ifdef SVMP_FE_ASSEMBLY_TIMING
    auto AO_TP = []() {
        return std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    };
#else
    auto AO_TP = []() -> double { return 0.0; };
#endif
    double ao_setup_end = AO_TP();
    double ao_cell_time = 0.0, ao_boundary_time = 0.0, ao_other_time = 0.0;
    double ao0;

    // Cell terms — fused multi-term assembly
    ao0 = AO_TP();
    {
        std::vector<assembly::FusedCellTerm> fused_terms;
        fused_terms.reserve(plan.cell_terms.size());

        for (const auto& term : plan.cell_terms) {
            const bool want_matrix = request.want_matrix && term.matrix_capable;
            const bool want_vector = request.want_vector && term.vector_capable;
            if (!want_matrix && !want_vector) {
                continue;
            }

            if (oopTraceEnabled()) {
                const auto& test_field = system.field_registry_.get(term.test_field);
                const auto& trial_field = system.field_registry_.get(term.trial_field);
                std::ostringstream oss;
                oss << "assembleOperator: op='" << request.op << "' cell term test='" << test_field.name
                    << "' trial='" << trial_field.name << "' want_matrix=" << (want_matrix ? 1 : 0)
                    << " want_vector=" << (want_vector ? 1 : 0);
                traceLog(oss.str());
            }

            assembly::FusedCellTerm ft;
            ft.test_space = term.test_space;
            ft.trial_space = term.trial_space;
            ft.kernel = term.kernel;
            ft.row_dof_map = term.row_dof_map;
            ft.col_dof_map = term.col_dof_map;
            ft.row_dof_offset = term.row_dof_offset;
            ft.col_dof_offset = term.col_dof_offset;
            ft.matrix_view = want_matrix ? matrix_out : nullptr;
            ft.vector_view = want_vector ? vector_out : nullptr;
            ft.assemble_matrix = want_matrix;
            ft.assemble_vector = want_vector;
            fused_terms.push_back(ft);
        }

        if (!fused_terms.empty()) {
            const auto fused_t0 = std::chrono::steady_clock::now();
            auto r = assembler.assembleCellsFused(mesh, fused_terms);
            mergeAssemblyResult(total, r);

            if (oopTraceEnabled()) {
                const auto fused_t1 = std::chrono::steady_clock::now();
                std::ostringstream oss;
                oss << "assembleOperator: op='" << request.op << "' fused cell terms ("
                    << fused_terms.size() << " terms) time="
                    << std::chrono::duration<double>(fused_t1 - fused_t0).count();
                traceLog(oss.str());
            }
        }
    }
    ao_cell_time += AO_TP() - ao0;

    // Boundary terms
    ao0 = AO_TP();
    if (request.assemble_boundary_terms) {
        for (const auto& term : plan.boundary_terms) {
            FE_CHECK_NOT_NULL(term.kernel, "assembleOperator: boundary term kernel");
            const bool want_matrix = request.want_matrix && term.matrix_capable;
            const bool want_vector = request.want_vector && term.vector_capable;
            if (!want_matrix && !want_vector) {
                continue;
            }

            assembler.setRowDofMap(*term.row_dof_map, term.row_dof_offset);
            assembler.setColDofMap(*term.col_dof_map, term.col_dof_offset);

            if (oopTraceEnabled()) {
                const auto& test_field = system.field_registry_.get(term.test_field);
                const auto& trial_field = system.field_registry_.get(term.trial_field);
                std::ostringstream oss;
                oss << "assembleOperator: op='" << request.op << "' boundary term marker=" << term.marker
                    << " test='" << test_field.name << "' trial='" << trial_field.name << "' want_matrix="
                    << (want_matrix ? 1 : 0) << " want_vector=" << (want_vector ? 1 : 0);
                traceLog(oss.str());
            }
            const auto term_t0 = std::chrono::steady_clock::now();

            auto r = assembler.assembleBoundaryFaces(
                mesh, term.marker, *term.test_space, *term.trial_space, *term.kernel,
                want_matrix ? matrix_out : nullptr,
                want_vector ? vector_out : nullptr);
            mergeAssemblyResult(total, r);

            if (oopTraceEnabled()) {
                const auto& test_field = system.field_registry_.get(term.test_field);
                const auto& trial_field = system.field_registry_.get(term.trial_field);
                const auto term_t1 = std::chrono::steady_clock::now();
                std::ostringstream oss;
                oss << "assembleOperator: op='" << request.op << "' boundary term done marker=" << term.marker
                    << " test='" << test_field.name << "' trial='" << trial_field.name << "' time="
                    << std::chrono::duration<double>(term_t1 - term_t0).count();
                traceLog(oss.str());
            }
        }
    }
    ao_boundary_time += AO_TP() - ao0;

    ao0 = AO_TP();
    // Interior face terms (DG)
    if (request.assemble_interior_face_terms) {
        for (const auto& term : plan.interior_terms) {
            FE_CHECK_NOT_NULL(term.kernel, "assembleOperator: interior-face term kernel");
            const bool want_matrix = request.want_matrix && term.matrix_capable;
            const bool want_vector = request.want_vector && term.vector_capable;
            if (!want_matrix && !want_vector) {
                continue;
            }

            assembler.setRowDofMap(*term.row_dof_map, term.row_dof_offset);
            assembler.setColDofMap(*term.col_dof_map, term.col_dof_offset);

            if (oopTraceEnabled()) {
                const auto& test_field = system.field_registry_.get(term.test_field);
                const auto& trial_field = system.field_registry_.get(term.trial_field);
                std::ostringstream oss;
                oss << "assembleOperator: op='" << request.op << "' interior-face term test='" << test_field.name
                    << "' trial='" << trial_field.name << "' want_matrix=" << (want_matrix ? 1 : 0)
                    << " want_vector=" << (want_vector ? 1 : 0);
                traceLog(oss.str());
            }
            const auto term_t0 = std::chrono::steady_clock::now();

            if (want_matrix) {
                auto r = assembler.assembleInteriorFaces(
                    mesh, *term.test_space, *term.trial_space, *term.kernel, *matrix_out,
                    want_vector ? vector_out : nullptr);
                mergeAssemblyResult(total, r);
            } else {
                assembly::DenseVectorView dummy_matrix(0);
                auto r = assembler.assembleInteriorFaces(
                    mesh, *term.test_space, *term.trial_space, *term.kernel, dummy_matrix,
                    vector_out);
                mergeAssemblyResult(total, r);
            }

            if (oopTraceEnabled()) {
                const auto& test_field = system.field_registry_.get(term.test_field);
                const auto& trial_field = system.field_registry_.get(term.trial_field);
                const auto term_t1 = std::chrono::steady_clock::now();
                std::ostringstream oss;
                oss << "assembleOperator: op='" << request.op << "' interior-face term done test='"
                    << test_field.name << "' trial='" << trial_field.name << "' time="
                    << std::chrono::duration<double>(term_t1 - term_t0).count();
                traceLog(oss.str());
            }
        }
    }

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    // Interface face terms (InterfaceMesh subset)
    if (request.assemble_interface_face_terms) {
        for (const auto& term : plan.interface_terms) {
            FE_CHECK_NOT_NULL(term.kernel, "assembleOperator: interface-face term kernel");
            const bool want_matrix = request.want_matrix && term.matrix_capable;
            const bool want_vector = request.want_vector && term.vector_capable;
            if (!want_matrix && !want_vector) {
                continue;
            }

            const auto& test_field = system.field_registry_.get(term.test_field);
            const auto& trial_field = system.field_registry_.get(term.trial_field);
            const auto row_scope =
                (test_field.scope == FieldScope::InterfaceFace)
                    ? assembly::DofEntityScope::InterfaceFace
                    : assembly::DofEntityScope::Cell;
            const auto col_scope =
                (trial_field.scope == FieldScope::InterfaceFace)
                    ? assembly::DofEntityScope::InterfaceFace
                    : assembly::DofEntityScope::Cell;

            assembler.setRowDofMap(*term.row_dof_map, term.row_dof_offset, row_scope);
            assembler.setColDofMap(*term.col_dof_map, term.col_dof_offset, col_scope);

            auto assemble_on_marker = [&](int marker) {
                auto it = system.interface_meshes_.find(marker);
                FE_THROW_IF(it == system.interface_meshes_.end() || !it->second, InvalidArgumentException,
                            "assembleOperator: missing InterfaceMesh for interface marker " + std::to_string(marker));
                const auto& iface_mesh = *it->second;

                if (want_matrix) {
                    auto r = assembler.assembleInterfaceFaces(
                        mesh, iface_mesh, marker, *term.test_space, *term.trial_space, *term.kernel, *matrix_out,
                        want_vector ? vector_out : nullptr);
                    mergeAssemblyResult(total, r);
                } else {
                    assembly::DenseVectorView dummy_matrix(0);
                    auto r = assembler.assembleInterfaceFaces(
                        mesh, iface_mesh, marker, *term.test_space, *term.trial_space, *term.kernel, dummy_matrix,
                        vector_out);
                    mergeAssemblyResult(total, r);
                }
            };

            if (term.marker < 0) {
                FE_THROW_IF(system.interface_meshes_.empty(), InvalidArgumentException,
                            "assembleOperator: interface-face term requested for all interface markers, but no InterfaceMesh was registered");
                for (const auto& kv : system.interface_meshes_) {
                    if (!kv.second) {
                        continue;
                    }
                    assemble_on_marker(kv.first);
                }
            } else {
                assemble_on_marker(term.marker);
            }
        }
    }
#endif

    // Global (non-element-local) terms (e.g., contact)
    if (request.assemble_global_terms) {
        for (auto* kernel : plan.global_terms) {
            FE_CHECK_NOT_NULL(kernel, "assembleOperator: global term kernel");
            auto r = kernel->assemble(system, request, state,
                                      request.want_matrix ? matrix_out : nullptr,
                                      request.want_vector ? vector_out : nullptr);
            mergeAssemblyResult(total, r);
        }
    }

    ao_other_time += AO_TP() - ao0;

    assembler.finalize(request.want_matrix ? matrix_out : nullptr,
                       request.want_vector ? vector_out : nullptr);

#ifdef SVMP_FE_ASSEMBLY_TIMING
    {
        int rank = 0;
        int size = 1;
        const double ao_total = ao_cell_time + ao_boundary_time + ao_other_time;
        std::array<double, 4> local_times{
            ao_total,
            ao_cell_time,
            ao_boundary_time,
            ao_other_time
        };
        std::array<double, 4> min_times = local_times;
        std::array<double, 4> max_times = local_times;
        std::array<double, 4> sum_times = local_times;
#if FE_HAS_MPI
        int mpi_init = 0;
        MPI_Initialized(&mpi_init);
        if (mpi_init) {
            const auto comm = system.dofHandler().mpiComm();
            MPI_Comm_rank(comm, &rank);
            MPI_Comm_size(comm, &size);
            MPI_Allreduce(local_times.data(), min_times.data(),
                          static_cast<int>(local_times.size()),
                          MPI_DOUBLE, MPI_MIN, comm);
            MPI_Allreduce(local_times.data(), max_times.data(),
                          static_cast<int>(local_times.size()),
                          MPI_DOUBLE, MPI_MAX, comm);
            MPI_Allreduce(local_times.data(), sum_times.data(),
                          static_cast<int>(local_times.size()),
                          MPI_DOUBLE, MPI_SUM, comm);
        }
#endif
        if (rank == 0) {
            if (ao_total > 1e-7) {
                std::fprintf(stderr,
                    "  === assembleOperator TIMING (rank 0, op='%s') ===\n"
                    "    Total:              %9.6f s\n"
                    "    Cell terms:         %9.6f s  (%5.1f%%)\n"
                    "    Boundary terms:     %9.6f s  (%5.1f%%)\n"
                    "    Other (DG+global):  %9.6f s  (%5.1f%%)\n",
                    request.op.c_str(),
                    ao_total,
                    ao_cell_time,     100.0 * ao_cell_time     / ao_total,
                    ao_boundary_time, 100.0 * ao_boundary_time / ao_total,
                    ao_other_time,    100.0 * ao_other_time    / ao_total);
                if (size > 1) {
                    std::fprintf(stderr,
                        "    MPI ranks:          %d\n"
                        "    Rank min/mean/max total:    %9.6f / %9.6f / %9.6f s\n"
                        "    Rank min/mean/max cell:     %9.6f / %9.6f / %9.6f s\n"
                        "    Rank min/mean/max boundary: %9.6f / %9.6f / %9.6f s\n"
                        "    Rank min/mean/max other:    %9.6f / %9.6f / %9.6f s\n",
                        size,
                        min_times[0], sum_times[0] / static_cast<double>(size), max_times[0],
                        min_times[1], sum_times[1] / static_cast<double>(size), max_times[1],
                        min_times[2], sum_times[2] / static_cast<double>(size), max_times[2],
                        min_times[3], sum_times[3] / static_cast<double>(size), max_times[3]);
                }
                std::fprintf(stderr,
                    "  ================================================\n");
            }
        }
    }
#endif

    const bool has_partitioned_output_coupling =
        request.want_matrix &&
        std::any_of(system.deployed_aux_entries_.begin(),
                    system.deployed_aux_entries_.end(),
                    [&](const auto& entry) {
                        if (!entry.materialized ||
                            entry.spec.solve_mode != AuxiliarySolveMode::Partitioned) {
                            return false;
                        }
                        if (!entry.deriv_provider || entry.output_ids.empty()) {
                            return false;
                        }
                        return !system.consumersOfEntry_(entry).empty();
                    });

    if (has_partitioned_output_coupling &&
        system.auxiliaryStateManagerIfPresent() &&
        !system.auxiliaryOperatorRegistryIfPresent()) {
        auto& aux_registry = system.auxiliaryOperatorRegistry();
        if (!aux_registry.isLayoutFinalized()) {
            aux_registry.finalizeLayout();
        }
    }

    // Monolithic auxiliary assembly: inject auxiliary residual/Jacobian
    // contributions.  For entries within PDE DOF range, forward to the
    // FSILS matrix/vector.  For entries involving auxiliary DOFs (outside
    // FSILS bounds), capture into bordered coupling storage for post-solve
    // static condensation. Partitioned FE-coupled outputs also reuse this
    // path to emit exact field-side reduced updates when their outputs are
    // consumed by PDE forms.
    if (system.auxiliaryStateManagerIfPresent() &&
        system.auxiliaryOperatorRegistryIfPresent()) {
        auto* aux_registry = system.auxiliaryOperatorRegistryIfPresent();
        const bool has_monolithic_deployments =
            std::any_of(system.deployed_aux_entries_.begin(),
                        system.deployed_aux_entries_.end(),
                        [](const auto& entry) {
                            return entry.spec.solve_mode == AuxiliarySolveMode::Monolithic;
                        });
        const bool has_auxiliary_operators = !aux_registry->operatorNames().empty();
        if (!has_monolithic_deployments &&
            !has_auxiliary_operators &&
            !has_partitioned_output_coupling) {
            return total;
        }

        std::size_t n_field_dofs = 0;
        if (system.dofHandler().getEntityDofMap()) {
            n_field_dofs = static_cast<std::size_t>(
                system.dofHandler().getNumDofs());
        } else if (!state.u.empty()) {
            n_field_dofs = state.u.size();
        }

        int n_aux = 0;
        if (aux_registry->isLayoutFinalized()) {
            const auto mixed = aux_registry->composeMixedLayout(n_field_dofs);
            n_aux = static_cast<int>(mixed.n_aux_unknowns);
        }

        if (n_aux > 0) {
            auto& bc = system.borderedCoupling();
            const auto mixed_for_bordered =
                aux_registry->composeMixedLayout(n_field_dofs);

            auto layoutBlockFor = [&](std::string_view name)
                -> const AuxiliaryBlockUnknownLayout* {
                for (const auto& block : mixed_for_bordered.aux_layout.blocks) {
                    if (block.name == name) {
                        return &block;
                    }
                }
                return nullptr;
            };

            auto populate_bordered_metadata = [&]() {
                bc.aux_blocks.clear();
                bc.aux_variable_kinds.clear();
                bc.aux_row_owner_ranks.assign(static_cast<std::size_t>(bc.n_aux), -1);
                bc.aux_row_owner_routed.assign(static_cast<std::size_t>(bc.n_aux), char{0});
                bc.aux_row_local_contribution_flags.assign(
                    static_cast<std::size_t>(bc.n_aux), 0);
                bc.aux_row_global_contributor_counts.assign(
                    static_cast<std::size_t>(bc.n_aux), 0);
                bc.aux_self_terms_replicated = true;

                std::size_t aux_row_offset = 0;
                for (const auto& entry : system.deployed_aux_entries_) {
                    if (entry.spec.solve_mode != AuxiliarySolveMode::Monolithic ||
                        entry.lower_to_direct_only ||
                        entry.local_condensed) {
                        continue;
                    }

                    if (entry.spec.scope == AuxiliaryStateScope::Node ||
                        entry.spec.scope == AuxiliaryStateScope::Region) {
                        bc.aux_self_terms_replicated = false;
                    }

                    const AuxiliaryBlockStorage* blk_ptr = nullptr;
                    int block_dim = entry.spec.size;
                    if (auto* mgr = system.auxiliaryStateManagerIfPresent();
                        mgr && mgr->hasBlock(entry.instance_name)) {
                        blk_ptr = &mgr->getBlock(entry.instance_name);
                        block_dim = static_cast<int>(blk_ptr->storageSize());
                    }
                    bc.aux_blocks.push_back({entry.instance_name, block_dim});

                    const auto* layout_block = layoutBlockFor(entry.instance_name);
                    const auto block_rows = static_cast<std::size_t>(
                        std::max(block_dim, 0));
                    const bool owner_routed =
                        layout_block != nullptr &&
                        (layout_block->row_ownership ==
                             backends::MixedRowOwnershipPolicy::BackendDofOwner ||
                         layout_block->row_ownership ==
                             backends::MixedRowOwnershipPolicy::RegionOwner);

                    for (std::size_t local_row = 0; local_row < block_rows; ++local_row) {
                        const auto global_row = aux_row_offset + local_row;
                        if (global_row >= bc.aux_row_owner_ranks.size()) {
                            break;
                        }
                        int owner = -1;
                        if (layout_block != nullptr) {
                            if (local_row < layout_block->row_owner_ranks.size()) {
                                owner = layout_block->row_owner_ranks[local_row];
                            } else if (layout_block->row_ownership ==
                                           backends::MixedRowOwnershipPolicy::SingleOwner) {
                                owner = layout_block->single_owner_rank;
                            }
                        }
                        bc.aux_row_owner_ranks[global_row] = owner;
                        bc.aux_row_owner_routed[global_row] =
                            owner_routed ? char{1} : char{0};
                    }
                    aux_row_offset += block_rows;

                    const auto& meta = entry.model->structuralMetadata();
                    const auto& kinds = meta.variable_kinds;
                    const int stride = entry.spec.size;
                    std::size_t entity_count = entry.explicit_entity_count;
                    auto ordering = entry.spec.ordering;
                    if (blk_ptr != nullptr) {
                        entity_count = blk_ptr->entityCount();
                        ordering = blk_ptr->ordering();
                    } else if (entity_count == 0) {
                        entity_count = 1;
                    }
                    auto kind_at = [&](int component) {
                        const auto idx = static_cast<std::size_t>(component);
                        return (idx < kinds.size())
                            ? kinds[idx]
                            : AuxiliaryVariableKind::Differential;
                    };
                    if (ordering == AuxiliaryEntityOrdering::ByComponentThenEntity) {
                        for (int c = 0; c < stride; ++c) {
                            for (std::size_t e = 0; e < entity_count; ++e) {
                                (void)e;
                                bc.aux_variable_kinds.push_back(kind_at(c));
                            }
                        }
                    } else {
                        for (std::size_t e = 0; e < entity_count; ++e) {
                            (void)e;
                            for (int c = 0; c < stride; ++c) {
                                bc.aux_variable_kinds.push_back(kind_at(c));
                            }
                        }
                    }
                }
            };

            // Only reset bordered blocks when assembling the Jacobian.
            // Residual-only assembly must not clear the Jacobian blocks
            // (D, B, Ct) that were populated during the J+r assembly.
            if (request.want_matrix) {
                bc.resize(n_aux, n_field_dofs);
                populate_bordered_metadata();
            } else if (!bc.active) {
                // First call (residual-only before any J+r): initialize.
                bc.resize(n_aux, n_field_dofs);
                populate_bordered_metadata();
            }
            // Always re-zero g (auxiliary residual) since it changes each assembly.
            std::fill(bc.g.begin(), bc.g.end(), 0.0);

            // Create a wrapper view that routes aux-DOF entries to bordered
            // storage and PDE-range entries to the real FSILS views.
            struct BorderedView final : public assembly::GlobalSystemView {
                assembly::GlobalSystemView* inner_mat;
                assembly::GlobalSystemView* inner_vec;
                FESystem::BorderedCouplingData* bc;
                GlobalIndex nf; // n_field_dofs
                int rank{0};

                [[nodiscard]] bool ownsAuxRow(std::size_t aux_row) const noexcept
                {
                    if (bc == nullptr ||
                        aux_row >= bc->aux_row_owner_routed.size() ||
                        bc->aux_row_owner_routed[aux_row] == char{0}) {
                        return true;
                    }
                    return aux_row < bc->aux_row_owner_ranks.size() &&
                           bc->aux_row_owner_ranks[aux_row] == rank;
                }

                void markAuxRowContribution(std::size_t aux_row) const noexcept
                {
                    if (bc == nullptr ||
                        aux_row >= bc->aux_row_owner_routed.size() ||
                        bc->aux_row_owner_routed[aux_row] == char{0} ||
                        aux_row >= bc->aux_row_local_contribution_flags.size()) {
                        return;
                    }
                    bc->aux_row_local_contribution_flags[aux_row] = 1;
                }

                void addMatrixEntries(std::span<const GlobalIndex> row_dofs,
                    std::span<const GlobalIndex> col_dofs,
                    std::span<const Real> vals, assembly::AddMode mode) override
                {
                    const auto nr = row_dofs.size();
                    const auto nc = col_dofs.size();
                    for (std::size_t i = 0; i < nr; ++i) {
                        const auto r = row_dofs[i];
                        for (std::size_t j = 0; j < nc; ++j) {
                            const auto c = col_dofs[j];
                            const auto v = vals[i * nc + j];
                            if (std::abs(v) < 1e-30) continue;

                            if (r < nf && c < nf) {
                                // PDE×PDE → forward to FSILS
                                if (inner_mat) inner_mat->addMatrixEntry(r, c, v, mode);
                            } else if (r >= nf && c >= nf) {
                                // Aux×Aux → D block
                                const auto ai = static_cast<std::size_t>(r - nf);
                                const auto aj = static_cast<std::size_t>(c - nf);
                                const auto na = static_cast<std::size_t>(bc->n_aux);
                                if (ai < na && aj < na && ownsAuxRow(ai)) {
                                    bc->D[ai * na + aj] += v;
                                    markAuxRowContribution(ai);
                                }
                            } else if (r < nf && c >= nf) {
                                // PDE×Aux → B block (col-major: B[r + nf*aux_col])
                                const auto aj = static_cast<std::size_t>(c - nf);
                                if (aj < static_cast<std::size_t>(bc->n_aux))
                                    bc->B[static_cast<std::size_t>(r) + bc->n_field_dofs * aj] += v;
                            } else {
                                // Aux×PDE → C^T block (row-major: Ct[aux_row * nf + col])
                                const auto ai = static_cast<std::size_t>(r - nf);
                                if (ai < static_cast<std::size_t>(bc->n_aux) &&
                                    ownsAuxRow(ai)) {
                                    bc->Ct[ai * bc->n_field_dofs + static_cast<std::size_t>(c)] += v;
                                    markAuxRowContribution(ai);
                                }
                            }
                        }
                    }
                }
                void addMatrixEntries(std::span<const GlobalIndex> dofs,
                    std::span<const Real> vals, assembly::AddMode mode) override
                { addMatrixEntries(dofs, dofs, vals, mode); }
                void addMatrixEntry(GlobalIndex r, GlobalIndex c, Real v, assembly::AddMode mode) override {
                    std::array<GlobalIndex,1> rd{r}, cd{c};
                    std::array<Real,1> vv{v};
                    addMatrixEntries(rd, cd, vv, mode);
                }
                void addVectorEntries(std::span<const GlobalIndex> dofs,
                    std::span<const Real> vals, assembly::AddMode mode) override
                {
                    for (std::size_t i = 0; i < dofs.size(); ++i) {
                        if (dofs[i] < nf) {
                            if (inner_vec) inner_vec->addVectorEntry(dofs[i], vals[i], mode);
                        } else {
                            const auto ai = static_cast<std::size_t>(dofs[i] - nf);
                            if (ai < static_cast<std::size_t>(bc->n_aux) &&
                                ownsAuxRow(ai)) {
                                bc->g[ai] += vals[i];
                                if (std::abs(vals[i]) > Real(1e-30)) {
                                    markAuxRowContribution(ai);
                                }
                            }
                        }
                    }
                }
                void addVectorEntry(GlobalIndex d, Real v, assembly::AddMode mode) override {
                    std::array<GlobalIndex,1> dd{d};
                    std::array<Real,1> vv{v};
                    addVectorEntries(dd, vv, mode);
                }
                // Pass-through for unused methods.
                void setDiagonal(std::span<const GlobalIndex> d, std::span<const Real> v) override { if (inner_mat) inner_mat->setDiagonal(d, v); }
                void setDiagonal(GlobalIndex d, Real v) override { if (inner_mat) inner_mat->setDiagonal(d, v); }
                void zeroRows(std::span<const GlobalIndex> d, bool b) override { if (inner_mat) inner_mat->zeroRows(d, b); }
                void setVectorEntries(std::span<const GlobalIndex> d, std::span<const Real> v) override { if (inner_vec) inner_vec->setVectorEntries(d, v); }
                void zeroVectorEntries(std::span<const GlobalIndex> d) override { if (inner_vec) inner_vec->zeroVectorEntries(d); }
                [[nodiscard]] Real getVectorEntry(GlobalIndex d) const override { return inner_vec ? inner_vec->getVectorEntry(d) : 0.0; }
                void beginAssemblyPhase() override {}
                void endAssemblyPhase() override {}
                void finalizeAssembly() override {}
                [[nodiscard]] assembly::AssemblyPhase getPhase() const noexcept override { return assembly::AssemblyPhase::Building; }
                [[nodiscard]] bool hasMatrix() const noexcept override { return inner_mat != nullptr; }
                [[nodiscard]] bool hasVector() const noexcept override { return inner_vec != nullptr; }
                [[nodiscard]] GlobalIndex numRows() const noexcept override { return nf + bc->n_aux; }
                [[nodiscard]] GlobalIndex numCols() const noexcept override { return nf + bc->n_aux; }
                [[nodiscard]] std::string backendName() const override { return "BorderedView"; }
                void zero() override {}
            };

            BorderedView bview;
            bview.inner_mat = (request.want_matrix ? matrix_out : nullptr);
            bview.inner_vec = (request.want_vector ? vector_out : nullptr);
            bview.bc = &bc;
            bview.nf = static_cast<GlobalIndex>(n_field_dofs);
            bview.rank = 0;
#if FE_HAS_MPI
            {
                int mpi_initialized = 0;
                MPI_Initialized(&mpi_initialized);
                if (mpi_initialized) {
                    MPI_Comm_rank(system.dofHandler().mpiComm(), &bview.rank);
                }
            }
#endif

            system.assembleMixedAuxiliaryIntoGlobal(
                state, &bview, &bview,
                request.want_matrix, request.want_vector,
                n_field_dofs, request.is_nonlinear_iteration);

            synchronizeBorderedCouplingForReplicatedSolve(
                bc, request.want_matrix, request.want_vector, system.dofHandler().mpiComm());
        } else {
            if (request.want_matrix) {
                system.borderedCoupling().clear();
            }
            system.assembleMixedAuxiliaryIntoGlobal(
                state, matrix_out, vector_out,
                request.want_matrix, request.want_vector,
                n_field_dofs, request.is_nonlinear_iteration);
        }
    }

    return total;
}

} // namespace systems
} // namespace FE
} // namespace svmp
