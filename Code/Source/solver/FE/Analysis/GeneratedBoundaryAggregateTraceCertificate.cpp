/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/GeneratedBoundaryAggregateTraceCertificate.h"

#include "Assembly/Assembler.h"
#include "Assembly/BackgroundEntityMeasures.h"
#include "Assembly/CutIntegrationContext.h"
#include "Basis/LagrangeBasis.h"
#include "Constraints/AffineConstraints.h"
#include "Constraints/SmallCutAggregationConstraint.h"
#include "Core/FEException.h"
#include "Dofs/DofHandler.h"
#include "Dofs/DofMap.h"
#include "Dofs/EntityDofMap.h"
#include "Geometry/CutQuadratureMapping.h"
#include "Interfaces/FreeSurfaceGeometrySnapshot.h"
#include "Spaces/FunctionSpace.h"
#include "Spaces/ProductSpace.h"
#include "Systems/FESystem.h"
#include "Systems/FieldRegistry.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#if FE_HAS_MPI
#include <mpi.h>
#endif

namespace svmp::FE::analysis {
namespace {

constexpr std::size_t kHardMaximumReducedDimension = 128u;
constexpr std::size_t kMaximumActiveCells = 8192u;
constexpr std::size_t kMaximumPatches = 4096u;
constexpr std::size_t kMaximumVolumeRules = 16384u;
constexpr std::size_t kMaximumRetainedRulesPerCell = 64u;
constexpr std::size_t kMaximumBoundaryRules = 8192u;
constexpr std::size_t kMaximumQuadraturePointsPerRule = 4096u;
constexpr std::size_t kMaximumLocalQuadraturePoints = 1024u * 1024u;
constexpr std::size_t kMaximumConstraintEntries =
    kHardMaximumReducedDimension;
constexpr std::size_t kMaximumPatchRawDofs = 8192u;
constexpr std::size_t kMaximumDenseModeledBytes = 8u * 1024u * 1024u;
constexpr std::size_t kMaximumLocalWords = 4u * 1024u * 1024u;
constexpr std::size_t kMaximumGatheredWords = 16u * 1024u * 1024u;
// The canonical rank-exchange payload is unchanged by the exact quotient
// proof.  Version its certificate digest independently so a proof-schema
// change does not masquerade as an incompatible wire encoding.
constexpr std::uint64_t kWireVersion = 1u;
constexpr std::uint64_t kCertificateDigestVersion = 2u;

[[noreturn]] void reject(std::string_view diagnostic)
{
    throw std::runtime_error(
        "GeneratedBoundaryAggregateTraceCertificate: " +
        std::string(diagnostic));
}

[[nodiscard]] bool finitePositive(Real value) noexcept
{
    return std::isfinite(value) && value > Real{0.0};
}

[[nodiscard]] bool sameRealBits(Real left, Real right) noexcept
{
    return std::bit_cast<std::uint64_t>(left) ==
           std::bit_cast<std::uint64_t>(right);
}

[[nodiscard]] Real outwardRoundNonnegative(long double value,
                                           std::string_view label)
{
    if (!(value >= 0.0L) || !std::isfinite(value) ||
        value >
            static_cast<long double>(
                std::numeric_limits<Real>::max())) {
        reject(std::string(label) + " is outside the finite Real range");
    }
    Real rounded = static_cast<Real>(value);
    if (static_cast<long double>(rounded) < value) {
        rounded = std::nextafter(
            rounded, std::numeric_limits<Real>::infinity());
    }
    if (!std::isfinite(rounded) || rounded < Real{0.0}) {
        reject(std::string(label) + " could not be outward rounded");
    }
    return rounded;
}

[[nodiscard]] Real outwardAddNonnegative(Real left,
                                        Real right,
                                        std::string_view label)
{
    if (!std::isfinite(left) || left < Real{0.0} ||
        !std::isfinite(right) || right < Real{0.0}) {
        reject(std::string(label) + " has an invalid nonnegative addend");
    }
    if (right == Real{0.0}) {
        return left;
    }
    Real rounded = left + right;
    if (!std::isfinite(rounded)) {
        reject(std::string(label) + " overflows its finite range");
    }
    // One unconditional successor makes the result an upper bound whether
    // the implementation rounded the addition down, exactly, or up.
    rounded = std::nextafter(
        rounded, std::numeric_limits<Real>::infinity());
    if (!std::isfinite(rounded) ||
        rounded < left ||
        rounded < right) {
        reject(std::string(label) + " overflows its finite range");
    }
    return rounded;
}

[[nodiscard]] std::size_t checkedSquare(std::size_t value,
                                        std::string_view label)
{
    if (value != 0u &&
        value > std::numeric_limits<std::size_t>::max() / value) {
        reject(std::string(label) + " dimension overflows");
    }
    return value * value;
}

struct CollectiveContext {
    int rank{0};
    int size{1};
#if FE_HAS_MPI
    MPI_Comm communicator{MPI_COMM_NULL};
    bool active{false};
#endif
};

[[nodiscard]] CollectiveContext collectiveContext(
    const systems::FESystem& system)
{
    CollectiveContext result;
#if FE_HAS_MPI
    int initialized = 0;
    int finalized = 0;
    MPI_Initialized(&initialized);
    if (initialized != 0) {
        MPI_Finalized(&finalized);
        if (finalized != 0) {
            reject("MPI is already finalized");
        }
        result.communicator = system.dofHandler().mpiComm();
        if (result.communicator == MPI_COMM_NULL) {
            reject("field communicator is null");
        }
        MPI_Comm_rank(result.communicator, &result.rank);
        MPI_Comm_size(result.communicator, &result.size);
        result.active = result.size > 1;
    }
#else
    (void)system;
#endif
    return result;
}

void coordinateFailure(const CollectiveContext& collective,
                       const std::exception_ptr& local_exception,
                       std::string_view phase)
{
    bool any_failed = local_exception != nullptr;
#if FE_HAS_MPI
    if (collective.active) {
        const int local_ok = local_exception == nullptr ? 1 : 0;
        int all_ok = 0;
        MPI_Allreduce(&local_ok,
                      &all_ok,
                      1,
                      MPI_INT,
                      MPI_MIN,
                      collective.communicator);
        any_failed = all_ok == 0;
    }
#else
    (void)collective;
#endif
    if (!any_failed) {
        return;
    }
    if (local_exception != nullptr) {
        std::rethrow_exception(local_exception);
    }
    reject(
        "collective phase '" + std::string(phase) +
        "' failed on another communicator rank");
}

void requireCollectiveAgreement(const CollectiveContext& collective,
                                std::uint64_t signature,
                                std::string_view label)
{
#if FE_HAS_MPI
    if (collective.active) {
        std::uint64_t minimum = 0u;
        std::uint64_t maximum = 0u;
        MPI_Allreduce(&signature,
                      &minimum,
                      1,
                      MPI_UINT64_T,
                      MPI_MIN,
                      collective.communicator);
        MPI_Allreduce(&signature,
                      &maximum,
                      1,
                      MPI_UINT64_T,
                      MPI_MAX,
                      collective.communicator);
        if (minimum != maximum) {
            reject(std::string(label) +
                   " differs across communicator ranks");
        }
    }
#else
    (void)collective;
    (void)signature;
    (void)label;
#endif
}

void digestMix(std::uint64_t& digest, std::uint64_t value) noexcept
{
    digest ^= value;
    digest *= UINT64_C(1099511628211);
}

class WireWriter {
public:
    void pushUnsigned(std::uint64_t value)
    {
        requireRoom(1u);
        words_.push_back(std::bit_cast<std::int64_t>(value));
    }

    void pushSigned(std::int64_t value)
    {
        requireRoom(1u);
        words_.push_back(value);
    }

    void pushSize(std::size_t value)
    {
        if (value >
            static_cast<std::size_t>(
                std::numeric_limits<std::int64_t>::max())) {
            reject("wire size exceeds the signed 64-bit range");
        }
        pushSigned(static_cast<std::int64_t>(value));
    }

    void pushReal(Real value)
    {
        if (!std::isfinite(value)) {
            reject("wire payload contains a nonfinite scalar");
        }
        pushUnsigned(std::bit_cast<std::uint64_t>(value));
    }

    [[nodiscard]] const std::vector<std::int64_t>& words() const noexcept
    {
        return words_;
    }

private:
    void requireRoom(std::size_t additional) const
    {
        if (additional > kMaximumLocalWords - words_.size()) {
            reject("local wire payload exceeds the hard word cap");
        }
    }

    std::vector<std::int64_t> words_{};
};

class WireReader {
public:
    explicit WireReader(std::span<const std::int64_t> words)
        : words_(words)
    {
    }

    [[nodiscard]] std::uint64_t unsignedValue()
    {
        return std::bit_cast<std::uint64_t>(signedValue());
    }

    [[nodiscard]] std::int64_t signedValue()
    {
        if (cursor_ >= words_.size()) {
            reject("wire payload ended before the declared records");
        }
        return words_[cursor_++];
    }

    [[nodiscard]] std::size_t sizeValue(std::size_t maximum,
                                        std::string_view label)
    {
        const auto value = signedValue();
        if (value < 0 ||
            static_cast<std::uint64_t>(value) >
                static_cast<std::uint64_t>(maximum)) {
            reject(std::string(label) + " exceeds its hard cap");
        }
        return static_cast<std::size_t>(value);
    }

    [[nodiscard]] Real realValue()
    {
        const Real value =
            std::bit_cast<Real>(unsignedValue());
        if (!std::isfinite(value)) {
            reject("decoded wire scalar is nonfinite");
        }
        return value;
    }

    void requireEnd() const
    {
        if (cursor_ != words_.size()) {
            reject("wire payload has trailing words");
        }
    }

private:
    std::span<const std::int64_t> words_{};
    std::size_t cursor_{0u};
};

struct GatheredWords {
    std::vector<std::int64_t> words{};
    std::vector<int> counts{};
    std::vector<int> displacements{};
};

[[nodiscard]] GatheredWords allGatherWords(
    const CollectiveContext& collective,
    std::span<const std::int64_t> local_words)
{
    const bool local_overflow =
        local_words.size() > kMaximumLocalWords ||
        local_words.size() >
            static_cast<std::size_t>(
                std::numeric_limits<int>::max());
#if FE_HAS_MPI
    if (collective.active) {
        const int local_flag = local_overflow ? 1 : 0;
        int global_flag = 0;
        MPI_Allreduce(
            &local_flag,
            &global_flag,
            1,
            MPI_INT,
            MPI_MAX,
            collective.communicator);
        if (global_flag != 0) {
            reject("local collective payload exceeds its hard cap");
        }
    }
#endif
    if (local_overflow) {
        reject("local collective payload exceeds its hard cap");
    }

    GatheredWords result;
    if (collective.size == 1) {
        result.words.assign(local_words.begin(), local_words.end());
        result.counts.push_back(static_cast<int>(local_words.size()));
        result.displacements.push_back(0);
        return result;
    }

#if FE_HAS_MPI
    const int local_count = static_cast<int>(local_words.size());
    std::exception_ptr allocation_exception;
    try {
        result.counts.assign(
            static_cast<std::size_t>(collective.size), 0);
        result.displacements.assign(
            static_cast<std::size_t>(collective.size), 0);
    } catch (...) {
        allocation_exception = std::current_exception();
    }
    coordinateFailure(
        collective, allocation_exception, "gather count allocation");

    MPI_Allgather(&local_count,
                  1,
                  MPI_INT,
                  result.counts.data(),
                  1,
                  MPI_INT,
                  collective.communicator);

    allocation_exception = nullptr;
    try {
        std::size_t total = 0u;
        for (int rank = 0; rank < collective.size; ++rank) {
            const int count =
                result.counts[static_cast<std::size_t>(rank)];
            if (count < 0 ||
                static_cast<std::size_t>(count) >
                    kMaximumGatheredWords - total ||
                total >
                    static_cast<std::size_t>(
                        std::numeric_limits<int>::max())) {
                reject("gathered collective payload exceeds its hard cap");
            }
            result.displacements[static_cast<std::size_t>(rank)] =
                static_cast<int>(total);
            total += static_cast<std::size_t>(count);
        }
        if (total > kMaximumGatheredWords ||
            total >
                static_cast<std::size_t>(
                    std::numeric_limits<int>::max())) {
            reject("gathered collective payload exceeds its hard cap");
        }
        result.words.resize(total);
    } catch (...) {
        allocation_exception = std::current_exception();
    }
    coordinateFailure(
        collective, allocation_exception, "gather receive allocation");

    MPI_Allgatherv(local_words.data(),
                   local_count,
                   MPI_INT64_T,
                   result.words.data(),
                   result.counts.data(),
                   result.displacements.data(),
                   MPI_INT64_T,
                   collective.communicator);
    return result;
#else
    reject("parallel collective requested without MPI support");
#endif
}

[[nodiscard]] bool sameConstraintRevision(
    const constraints::ConstraintRevisionSnapshot& left,
    const constraints::ConstraintRevisionSnapshot& right) noexcept
{
    return left.valid == right.valid &&
           left.geometry == right.geometry &&
           left.reference_rebase == right.reference_rebase &&
           left.topology == right.topology &&
           left.ownership == right.ownership &&
           left.numbering == right.numbering &&
           left.mesh_field_layout == right.mesh_field_layout &&
           left.mesh_field_values == right.mesh_field_values &&
           left.labels == right.labels &&
           left.active_configuration == right.active_configuration &&
           left.fe_space == right.fe_space &&
           left.fe_dof_layout == right.fe_dof_layout &&
           left.fe_constraint_layout == right.fe_constraint_layout &&
           left.fe_block_layout == right.fe_block_layout &&
           left.time_epoch == right.time_epoch;
}

struct DofDescriptor {
    GlobalIndex dof{INVALID_GLOBAL_INDEX};
    std::size_t component{0u};
    std::array<Real, 3> coordinate{{0.0, 0.0, 0.0}};
};

struct CellContribution {
    GlobalIndex cell_gid{INVALID_GLOBAL_INDEX};
    constraints::SmallCutAggregationActiveCellKind kind{
        constraints::SmallCutAggregationActiveCellKind::Cut};
    std::vector<std::uint64_t> stable_rule_ids{};
    std::vector<DofDescriptor> dofs{};
    std::vector<Real> denominator{};
    Real retained_measure{0.0};
};

struct BoundaryContribution {
    std::uint64_t stable_rule_id{0u};
    GlobalIndex parent_cell_gid{INVALID_GLOBAL_INDEX};
    GlobalIndex parent_face_gid{INVALID_GLOBAL_INDEX};
    std::vector<GlobalIndex> raw_dofs{};
    std::vector<Real> numerator{};
    Real physical_measure{0.0};
};

struct TangentRow {
    GlobalIndex dof{INVALID_GLOBAL_INDEX};
    bool constrained{false};
    Real inhomogeneity{0.0};
    std::vector<constraints::ConstraintEntry> entries{};
};

struct CanonicalData {
    std::map<GlobalIndex, CellContribution> cells{};
    std::map<std::uint64_t, BoundaryContribution> boundaries{};
    std::map<GlobalIndex, TangentRow> rows{};
    std::map<GlobalIndex, DofDescriptor> descriptors{};
};

void encodeDescriptor(WireWriter& writer,
                      const DofDescriptor& descriptor)
{
    writer.pushSigned(descriptor.dof);
    writer.pushSize(descriptor.component);
    for (const Real coordinate : descriptor.coordinate) {
        writer.pushReal(coordinate);
    }
}

[[nodiscard]] DofDescriptor decodeDescriptor(WireReader& reader,
                                             std::size_t dimension)
{
    DofDescriptor result;
    result.dof = reader.signedValue();
    result.component =
        reader.sizeValue(dimension - 1u, "DOF component");
    for (Real& coordinate : result.coordinate) {
        coordinate = reader.realValue();
    }
    return result;
}

void encodePayload(WireWriter& writer,
                   std::span<const CellContribution> cells,
                   std::span<const BoundaryContribution> boundaries,
                   std::span<const TangentRow> rows)
{
    writer.pushUnsigned(kWireVersion);
    writer.pushSize(cells.size());
    for (const auto& cell : cells) {
        writer.pushSigned(cell.cell_gid);
        writer.pushUnsigned(
            static_cast<std::uint64_t>(cell.kind));
        writer.pushReal(cell.retained_measure);
        writer.pushSize(cell.stable_rule_ids.size());
        for (const auto stable_id : cell.stable_rule_ids) {
            writer.pushUnsigned(stable_id);
        }
        writer.pushSize(cell.dofs.size());
        for (const auto& descriptor : cell.dofs) {
            encodeDescriptor(writer, descriptor);
        }
        writer.pushSize(cell.denominator.size());
        for (const Real value : cell.denominator) {
            writer.pushReal(value);
        }
    }

    writer.pushSize(boundaries.size());
    for (const auto& boundary : boundaries) {
        writer.pushUnsigned(boundary.stable_rule_id);
        writer.pushSigned(boundary.parent_cell_gid);
        writer.pushSigned(boundary.parent_face_gid);
        writer.pushReal(boundary.physical_measure);
        writer.pushSize(boundary.raw_dofs.size());
        for (const auto dof : boundary.raw_dofs) {
            writer.pushSigned(dof);
        }
        writer.pushSize(boundary.numerator.size());
        for (const Real value : boundary.numerator) {
            writer.pushReal(value);
        }
    }

    writer.pushSize(rows.size());
    for (const auto& row : rows) {
        writer.pushSigned(row.dof);
        writer.pushUnsigned(row.constrained ? 1u : 0u);
        writer.pushReal(row.inhomogeneity);
        writer.pushSize(row.entries.size());
        for (const auto& entry : row.entries) {
            writer.pushSigned(entry.master_dof);
            writer.pushReal(entry.weight);
        }
    }
}

[[nodiscard]] bool sameDescriptor(const DofDescriptor& left,
                                  const DofDescriptor& right) noexcept
{
    if (left.dof != right.dof ||
        left.component != right.component) {
        return false;
    }
    for (std::size_t component = 0u; component < 3u; ++component) {
        if (!sameRealBits(
                left.coordinate[component],
                right.coordinate[component])) {
            return false;
        }
    }
    return true;
}

void mergeDescriptor(CanonicalData& data,
                     const DofDescriptor& descriptor)
{
    const auto [found, inserted] =
        data.descriptors.emplace(descriptor.dof, descriptor);
    if (!inserted && !sameDescriptor(found->second, descriptor)) {
        reject("one raw DOF has inconsistent component or coordinate data");
    }
}

void decodePayload(std::span<const std::int64_t> words,
                   std::size_t dimension,
                   CanonicalData& data)
{
    WireReader reader(words);
    if (reader.unsignedValue() != kWireVersion) {
        reject("wire payload uses an unsupported format version");
    }

    const auto cell_count =
        reader.sizeValue(kMaximumActiveCells, "cell record count");
    for (std::size_t record = 0u; record < cell_count; ++record) {
        if (data.cells.size() >= kMaximumActiveCells) {
            reject("canonical retained-cell ledger exceeds its global cap");
        }
        CellContribution cell;
        cell.cell_gid = reader.signedValue();
        const auto kind = reader.unsignedValue();
        if (kind >
            static_cast<std::uint64_t>(
                constraints::SmallCutAggregationActiveCellKind::Cut)) {
            reject("cell record has an invalid active-cell kind");
        }
        cell.kind =
            static_cast<constraints::SmallCutAggregationActiveCellKind>(
                kind);
        cell.retained_measure = reader.realValue();
        const auto stable_count =
            reader.sizeValue(
                kMaximumRetainedRulesPerCell,
                "retained rule count");
        cell.stable_rule_ids.reserve(stable_count);
        for (std::size_t index = 0u; index < stable_count; ++index) {
            cell.stable_rule_ids.push_back(reader.unsignedValue());
        }
        const auto dof_count =
            reader.sizeValue(
                12u, "affine simplex cell DOF count");
        if (dof_count != dimension * (dimension + 1u)) {
            reject("cell record is not an affine P1 velocity cell");
        }
        cell.dofs.reserve(dof_count);
        for (std::size_t index = 0u; index < dof_count; ++index) {
            cell.dofs.push_back(decodeDescriptor(reader, dimension));
        }
        const auto matrix_count =
            reader.sizeValue(
                checkedSquare(dof_count, "cell matrix"),
                "cell matrix entry count");
        if (matrix_count != checkedSquare(dof_count, "cell matrix")) {
            reject("cell denominator has an incompatible dimension");
        }
        cell.denominator.reserve(matrix_count);
        for (std::size_t index = 0u; index < matrix_count; ++index) {
            cell.denominator.push_back(reader.realValue());
        }
        for (std::size_t row = 0u; row < dof_count; ++row) {
            for (std::size_t column = row;
                 column < dof_count;
                 ++column) {
                if (!sameRealBits(
                        cell.denominator[
                            row * dof_count + column],
                        cell.denominator[
                            column * dof_count + row])) {
                    reject("cell denominator is not exactly symmetric");
                }
            }
        }
        if (cell.cell_gid < 0 ||
            !finitePositive(cell.retained_measure) ||
            cell.stable_rule_ids.empty() ||
            !std::is_sorted(
                cell.stable_rule_ids.begin(),
                cell.stable_rule_ids.end()) ||
            std::adjacent_find(
                cell.stable_rule_ids.begin(),
                cell.stable_rule_ids.end()) !=
                cell.stable_rule_ids.end()) {
            reject("cell record has invalid canonical metadata");
        }
        for (const auto stable_id : cell.stable_rule_ids) {
            if (stable_id == 0u) {
                reject("cell record contains an unavailable rule identity");
            }
        }
        for (const auto& descriptor : cell.dofs) {
            if (descriptor.dof < 0) {
                reject("cell record contains an invalid raw DOF");
            }
            mergeDescriptor(data, descriptor);
        }
        if (!data.cells.emplace(cell.cell_gid, std::move(cell)).second) {
            reject("one retained cell has more than one provider record");
        }
    }

    const auto boundary_count =
        reader.sizeValue(
            kMaximumBoundaryRules, "boundary record count");
    for (std::size_t record = 0u; record < boundary_count; ++record) {
        if (data.boundaries.size() >= kMaximumBoundaryRules) {
            reject(
                "canonical generated-boundary ledger exceeds its global cap");
        }
        BoundaryContribution boundary;
        boundary.stable_rule_id = reader.unsignedValue();
        boundary.parent_cell_gid = reader.signedValue();
        boundary.parent_face_gid = reader.signedValue();
        boundary.physical_measure = reader.realValue();
        const auto dof_count =
            reader.sizeValue(
                12u, "boundary cell DOF count");
        if (dof_count != dimension * (dimension + 1u)) {
            reject("boundary record is not an affine P1 velocity cell");
        }
        boundary.raw_dofs.reserve(dof_count);
        for (std::size_t index = 0u; index < dof_count; ++index) {
            const auto dof = reader.signedValue();
            if (dof < 0) {
                reject("boundary record contains an invalid raw DOF");
            }
            boundary.raw_dofs.push_back(dof);
        }
        const auto matrix_count =
            reader.sizeValue(
                checkedSquare(dof_count, "boundary matrix"),
                "boundary matrix entry count");
        if (matrix_count !=
            checkedSquare(dof_count, "boundary matrix")) {
            reject("boundary numerator has an incompatible dimension");
        }
        boundary.numerator.reserve(matrix_count);
        for (std::size_t index = 0u; index < matrix_count; ++index) {
            boundary.numerator.push_back(reader.realValue());
        }
        for (std::size_t row = 0u; row < dof_count; ++row) {
            for (std::size_t column = row;
                 column < dof_count;
                 ++column) {
                if (!sameRealBits(
                        boundary.numerator[
                            row * dof_count + column],
                        boundary.numerator[
                            column * dof_count + row])) {
                    reject("boundary numerator is not exactly symmetric");
                }
            }
        }
        if (boundary.stable_rule_id == 0u ||
            boundary.parent_cell_gid < 0 ||
            boundary.parent_face_gid < 0 ||
            !finitePositive(boundary.physical_measure)) {
            reject("boundary record has invalid provenance or measure");
        }
        if (!data.boundaries
                 .emplace(
                     boundary.stable_rule_id,
                     std::move(boundary))
                 .second) {
            reject("one generated-boundary rule has more than one owner record");
        }
    }

    const auto row_count =
        reader.sizeValue(
            12u * kMaximumActiveCells, "tangent row count");
    for (std::size_t record = 0u; record < row_count; ++record) {
        if (data.rows.size() >= 12u * kMaximumActiveCells) {
            reject("canonical tangent-row ledger exceeds its global cap");
        }
        TangentRow row;
        row.dof = reader.signedValue();
        const auto constrained = reader.unsignedValue();
        if (constrained > 1u) {
            reject("tangent row has an invalid constrained flag");
        }
        row.constrained = constrained != 0u;
        row.inhomogeneity = reader.realValue();
        const auto entry_count =
            reader.sizeValue(
                kMaximumConstraintEntries,
                "tangent row entry count");
        row.entries.reserve(entry_count);
        for (std::size_t index = 0u; index < entry_count; ++index) {
            row.entries.push_back(constraints::ConstraintEntry{
                .master_dof = reader.signedValue(),
                .weight = reader.realValue()});
        }
        if (row.dof < 0) {
            reject("tangent row has an invalid raw DOF");
        }
        if (!row.constrained) {
            if (row.entries.size() != 1u ||
                row.entries.front().master_dof != row.dof ||
                !sameRealBits(
                    row.entries.front().weight, Real{1.0}) ||
                !sameRealBits(row.inhomogeneity, Real{0.0})) {
                reject("unconstrained tangent row is not canonical identity");
            }
        }
        GlobalIndex previous = INVALID_GLOBAL_INDEX;
        for (const auto& entry : row.entries) {
            if (entry.master_dof < 0 ||
                !std::isfinite(entry.weight) ||
                entry.weight == Real{0.0} ||
                entry.master_dof <= previous) {
                reject("tangent row entries are invalid or not strictly sorted");
            }
            previous = entry.master_dof;
        }
        if (!data.rows.emplace(row.dof, std::move(row)).second) {
            reject("one raw DOF has more than one canonical owner row");
        }
    }
    reader.requireEnd();
}

struct SelectedState {
    std::shared_ptr<
        const constraints::SmallCutAggregationProlongationReport>
        report{};
    std::shared_ptr<
        const interfaces::FreeSurfaceGeometrySnapshot>
        snapshot{};
    const assembly::CutIntegrationContext* cut_context{nullptr};
    std::size_t dimension{0u};
};

[[nodiscard]] SelectedState selectAndValidateState(
    const systems::FESystem& system,
    const GeneratedBoundaryAggregateTraceCertificationOptions& options,
    const CollectiveContext& collective)
{
    if (options.field == INVALID_FIELD_ID) {
        reject("field is invalid");
    }
    if (options.physical_boundary_marker < 0 ||
        options.volume_interface_marker < 0 ||
        options.generated_active_boundary_marker < 0) {
        reject("physical and generated markers must be nonnegative");
    }
    if (!finitePositive(options.dynamic_viscosity)) {
        reject("dynamic viscosity must be finite and positive");
    }
    if (options.maximum_reduced_dimension == 0u ||
        options.maximum_reduced_dimension >
            kHardMaximumReducedDimension) {
        reject("requested reduced dimension is outside the hard range");
    }

    SelectedState selected;
    const auto& mesh = system.meshAccess();
    const int mesh_dimension = mesh.dimension();
    if (mesh_dimension != 2 && mesh_dimension != 3) {
        reject("only two- and three-dimensional meshes are supported");
    }
    selected.dimension = static_cast<std::size_t>(mesh_dimension);

    const auto& field = system.fieldRecord(options.field);
    if (!field.space ||
        field.space->space_type() != spaces::SpaceType::Product ||
        field.space->continuity() != Continuity::C0 ||
        field.space->polynomial_order() != 1 ||
        field.space->value_dimension() != mesh_dimension ||
        field.components != mesh_dimension ||
        field.space->element().basis().is_vector_valued() ||
        field.space->element().basis().size() !=
            selected.dimension + 1u) {
        reject(
            "field must be an affine P1 Product H1 velocity with one "
            "component per spatial dimension");
    }
    const auto element_type = field.space->element_type();
    if ((mesh_dimension == 2 &&
         element_type != ElementType::Triangle3) ||
        (mesh_dimension == 3 &&
         element_type != ElementType::Tetra4)) {
        reject("field must use Triangle3 or Tetra4 affine simplices");
    }
#if FE_HAS_MPI
    if (collective.active) {
        const auto field_communicator =
            system.fieldDofHandler(options.field).mpiComm();
        if (field_communicator == MPI_COMM_NULL) {
            reject("field communicator is null");
        }
        int comparison = MPI_UNEQUAL;
        MPI_Comm_compare(
            collective.communicator,
            field_communicator,
            &comparison);
        if (comparison != MPI_IDENT &&
            comparison != MPI_CONGRUENT) {
            reject(
                "field and system communicators are not congruent");
        }
    }
#endif

    const auto& closed_constraints = system.constraints();
    if (!closed_constraints.isClosed()) {
        reject("affine constraints are not closed");
    }

    const auto reports =
        system.finalizedSmallCutAggregationProlongations();
    for (const auto& candidate : reports) {
        if (candidate &&
            candidate->field == options.field &&
            candidate->interface_marker ==
                options.volume_interface_marker) {
            if (selected.report) {
                reject(
                    "more than one aggregation report matches the field "
                    "and interface marker");
            }
            selected.report = candidate;
        }
    }
    if (!selected.report) {
        reject(
            "no finalized aggregation report matches the field and "
            "interface marker");
    }
    const auto& report = *selected.report;
    if (!report.trace_bound_eligible) {
        reject("aggregation report is not trace-bound eligible");
    }
    if (report.active_side ==
        geometry::CutIntegrationSide::Interface) {
        reject("aggregation report has an invalid active side");
    }
    if (report.revision.communicator_size != collective.size ||
        report.revision.local_rank != collective.rank) {
        reject("aggregation report communicator metadata is stale");
    }
    if (!sameConstraintRevision(
            report.revision.constraint,
            system.constraintRevisionSnapshot()) ||
        report.revision.affine_constraint_layout_revision !=
            closed_constraints.constraintLayoutRevision()) {
        reject("aggregation report constraint revisions are stale");
    }
    if (report.active_cells.empty() ||
        report.active_cells.size() > kMaximumActiveCells ||
        report.patches.size() > kMaximumPatches) {
        reject("aggregation report exceeds the supported cell or patch scope");
    }

    GlobalIndex previous_cell = INVALID_GLOBAL_INDEX;
    std::set<GlobalIndex> active_cell_ids;
    std::set<GlobalIndex> active_raw_dofs;
    for (const auto& cell : report.active_cells) {
        if (cell.cell_gid < 0 ||
            cell.cell_gid <= previous_cell ||
            cell.owner_rank < 0 ||
            cell.owner_rank >= collective.size ||
            cell.retained_measure_provider_rank < 0 ||
            cell.retained_measure_provider_rank >= collective.size ||
            cell.active_feature_id < 0 ||
            !finitePositive(cell.retained_physical_volume) ||
            cell.retained_rule_stable_ids.empty() ||
            cell.retained_rule_stable_ids.size() >
                kMaximumRetainedRulesPerCell ||
            cell.field_dofs.empty() ||
            cell.field_dofs.size() !=
                selected.dimension *
                    (selected.dimension + 1u) ||
            !std::is_sorted(
                cell.retained_rule_stable_ids.begin(),
                cell.retained_rule_stable_ids.end()) ||
            std::adjacent_find(
                cell.retained_rule_stable_ids.begin(),
                cell.retained_rule_stable_ids.end()) !=
                cell.retained_rule_stable_ids.end() ||
            !std::is_sorted(
                cell.field_dofs.begin(),
                cell.field_dofs.end()) ||
            std::adjacent_find(
                cell.field_dofs.begin(),
                cell.field_dofs.end()) !=
                cell.field_dofs.end()) {
            reject("aggregation active-cell metadata is not canonical");
        }
        for (const auto stable_id :
             cell.retained_rule_stable_ids) {
            if (stable_id == 0u) {
                reject(
                    "aggregation active cell has an unavailable retained "
                    "rule identity");
            }
        }
        for (const auto dof : cell.field_dofs) {
            if (dof < 0) {
                reject("aggregation active cell has an invalid field DOF");
            }
        }
        previous_cell = cell.cell_gid;
        active_cell_ids.insert(cell.cell_gid);
        active_raw_dofs.insert(
            cell.field_dofs.begin(), cell.field_dofs.end());
    }

    for (const auto& patch : report.patches) {
        if (patch.kind ==
            constraints::SmallCutAggregationPatchKind::Rootless) {
            reject(
                "rootless aggregate support cannot certify a physical "
                "trace bound");
        }
        if (patch.active_feature_ids.size() != 1u ||
            patch.active_feature_ids.front() < 0 ||
            patch.member_cell_gids.empty() ||
            patch.member_cell_gids.size() >
                report.active_cells.size() ||
            patch.support_cell_gids.empty() ||
            patch.support_cell_gids.size() >
                report.active_cells.size() ||
            !std::is_sorted(
                patch.active_feature_ids.begin(),
                patch.active_feature_ids.end()) ||
            !std::is_sorted(
                patch.member_cell_gids.begin(),
                patch.member_cell_gids.end()) ||
            std::adjacent_find(
                patch.member_cell_gids.begin(),
                patch.member_cell_gids.end()) !=
                patch.member_cell_gids.end() ||
            !std::is_sorted(
                patch.support_cell_gids.begin(),
                patch.support_cell_gids.end()) ||
            std::adjacent_find(
                patch.support_cell_gids.begin(),
                patch.support_cell_gids.end()) !=
                patch.support_cell_gids.end()) {
            reject(
                "aggregation patch is empty, noncanonical, or spans "
                "multiple active features");
        }
        for (const auto cell_gid : patch.support_cell_gids) {
            if (!active_cell_ids.contains(cell_gid)) {
                reject("aggregation patch references an unknown support cell");
            }
        }
        for (const auto cell_gid : patch.member_cell_gids) {
            if (!active_cell_ids.contains(cell_gid) ||
                !std::binary_search(
                    patch.support_cell_gids.begin(),
                    patch.support_cell_gids.end(),
                    cell_gid)) {
                reject(
                    "aggregation patch member is unknown or absent from "
                    "its support");
            }
        }
        if (!std::is_sorted(
                patch.slave_dofs.begin(),
                patch.slave_dofs.end()) ||
            std::adjacent_find(
                patch.slave_dofs.begin(),
                patch.slave_dofs.end()) !=
                patch.slave_dofs.end()) {
            reject("aggregation patch slave DOFs are not canonical");
        }
        for (const auto dof : patch.slave_dofs) {
            if (dof < 0 || !active_raw_dofs.contains(dof)) {
                reject("aggregation patch references an unknown slave DOF");
            }
        }
        if (patch.kind ==
                constraints::SmallCutAggregationPatchKind::Rooted &&
            (patch.root_cell_gid < 0 ||
             !std::binary_search(
                 patch.support_cell_gids.begin(),
                 patch.support_cell_gids.end(),
                 patch.root_cell_gid))) {
            reject("rooted aggregation patch lacks its declared root");
        }
    }

    if (report.rows.size() > active_raw_dofs.size()) {
        reject("aggregation candidate-row ledger exceeds active support");
    }
    GlobalIndex previous_candidate = INVALID_GLOBAL_INDEX;
    for (const auto& row : report.rows) {
        if (row.slave_dof < 0 ||
            row.slave_dof <= previous_candidate ||
            !active_raw_dofs.contains(row.slave_dof)) {
            reject(
                "aggregation candidate rows are noncanonical or outside "
                "active support");
        }
        previous_candidate = row.slave_dof;
    }

    selected.cut_context = system.cutIntegrationContext();
    if (selected.cut_context == nullptr) {
        reject("cut integration context is unavailable");
    }
    const auto& cut = *selected.cut_context;
    cut.assertAllFreeSurfaceGeometrySnapshotsCurrent(mesh);
    if (cut.contentRevision() !=
        report.revision.cut_context_content_revision) {
        reject("aggregation report cut-context revision is stale");
    }
    if (!cut.hasGeneratedVolumeMarker(
            options.volume_interface_marker) ||
        !cut.hasGeneratedActiveBoundaryMarker(
            options.generated_active_boundary_marker)) {
        reject(
            "requested markers are not the generated volume and active "
            "boundary domains");
    }
    if (!report.revision.has_free_surface_snapshot_revision ||
        !report.revision.has_source_value_revision) {
        reject(
            "aggregation report lacks authoritative free-surface "
            "snapshot revisions");
    }
    if (!cut.hasFreeSurfaceGeometrySnapshotForMarker(
            options.volume_interface_marker) ||
        !cut.hasFreeSurfaceGeometrySnapshotForMarker(
            options.generated_active_boundary_marker)) {
        reject("generated markers are not bound to a free-surface snapshot");
    }
    const auto volume_snapshot_revision =
        cut.freeSurfaceGeometrySnapshotRevisionForMarker(
            options.volume_interface_marker);
    const auto boundary_snapshot_revision =
        cut.freeSurfaceGeometrySnapshotRevisionForMarker(
            options.generated_active_boundary_marker);
    if (volume_snapshot_revision != boundary_snapshot_revision ||
        volume_snapshot_revision !=
            report.revision.free_surface_snapshot_revision) {
        reject("generated markers and aggregation report use different snapshots");
    }

    for (const auto& snapshot :
         cut.freeSurfaceGeometrySnapshots()) {
        if (snapshot &&
            snapshot->revision().snapshot_revision_key ==
                volume_snapshot_revision) {
            if (selected.snapshot) {
                reject("free-surface snapshot revision is ambiguous");
            }
            selected.snapshot = snapshot;
        }
    }
    if (!selected.snapshot) {
        reject("authoritative free-surface snapshot is unavailable");
    }
    const auto& snapshot_revision =
        selected.snapshot->revision();
    if (!snapshot_revision.complete() ||
        snapshot_revision.interface_marker !=
            options.volume_interface_marker ||
        snapshot_revision.source_value_revision !=
            report.revision.source_value_revision) {
        reject("free-surface snapshot does not match the aggregation report");
    }

    std::size_t active_domain_matches = 0u;
    for (const auto& domain :
         selected.snapshot->activeBoundaryDomains()) {
        if (domain.marker() !=
            options.generated_active_boundary_marker) {
            continue;
        }
        ++active_domain_matches;
        const auto& request = domain.request();
        if (request.boundary_marker !=
                options.physical_boundary_marker ||
            request.interface_marker !=
                options.volume_interface_marker ||
            request.side != report.active_side ||
            request.source_value_revision !=
                report.revision.source_value_revision ||
            request.frame !=
                geometry::CutGeometryFrame::Reference) {
            reject(
                "generated active-boundary request does not match the "
                "aggregation report");
        }
    }
    if (active_domain_matches != 1u) {
        reject("generated active-boundary domain is missing or ambiguous");
    }
    return selected;
}

[[nodiscard]] std::uint64_t requestSignature(
    const SelectedState& selected,
    const GeneratedBoundaryAggregateTraceCertificationOptions& options)
{
    const auto& report = *selected.report;
    std::uint64_t digest = UINT64_C(1469598103934665603);
    digestMix(digest, static_cast<std::uint64_t>(options.field));
    digestMix(
        digest,
        static_cast<std::uint64_t>(
            options.physical_boundary_marker));
    digestMix(
        digest,
        static_cast<std::uint64_t>(
            options.volume_interface_marker));
    digestMix(
        digest,
        static_cast<std::uint64_t>(
            options.generated_active_boundary_marker));
    digestMix(
        digest,
        std::bit_cast<std::uint64_t>(
            options.dynamic_viscosity));
    digestMix(digest, options.maximum_reduced_dimension);
    digestMix(
        digest,
        static_cast<std::uint64_t>(report.active_side));
    digestMix(digest, report.canonical_content_digest);
    digestMix(digest, kHardMaximumReducedDimension);
    digestMix(digest, kMaximumActiveCells);
    digestMix(digest, kMaximumPatches);
    digestMix(digest, kMaximumVolumeRules);
    digestMix(digest, kMaximumRetainedRulesPerCell);
    digestMix(digest, kMaximumBoundaryRules);
    digestMix(digest, kMaximumQuadraturePointsPerRule);
    digestMix(digest, kMaximumLocalQuadraturePoints);
    digestMix(digest, kMaximumConstraintEntries);
    digestMix(digest, kMaximumPatchRawDofs);
    digestMix(digest, kMaximumDenseModeledBytes);
    digestMix(digest, kMaximumLocalWords);
    digestMix(digest, kMaximumGatheredWords);
    return digest;
}

[[nodiscard]] std::unordered_map<GlobalIndex, GlobalIndex>
localCellByGlobalId(
    const assembly::IMeshAccess& mesh,
    const std::set<GlobalIndex>& requested_gids)
{
    if (mesh.parallelSize() > 1 &&
        !mesh.globalEntityIdsAvailable()) {
        reject("distributed mesh lacks globally unique cell identities");
    }
    std::unordered_map<GlobalIndex, GlobalIndex> result;
    result.reserve(requested_gids.size());
    for (GlobalIndex local_cell = 0;
         local_cell < mesh.numCells();
         ++local_cell) {
        const auto gid = mesh.getCellGlobalId(local_cell);
        if (gid < 0) {
            reject("local mesh has an invalid global cell identity");
        }
        if (!requested_gids.contains(gid)) {
            continue;
        }
        if (!result.emplace(gid, local_cell).second) {
            reject("local mesh has a duplicate requested cell identity");
        }
    }
    return result;
}

[[nodiscard]] std::vector<DofDescriptor> cellDofDescriptors(
    const systems::FESystem& system,
    FieldId field,
    GlobalIndex local_cell,
    std::size_t dimension)
{
    const auto& mesh = system.meshAccess();
    const auto type = mesh.getCellType(local_cell);
    if ((dimension == 2u && type != ElementType::Triangle3) ||
        (dimension == 3u && type != ElementType::Tetra4) ||
        mesh.getCellGeometryOrder(local_cell) != 1) {
        reject("certificate encountered a non-affine simplex cell");
    }

    std::vector<GlobalIndex> nodes;
    mesh.getCellNodes(local_cell, nodes);
    if (nodes.size() != dimension + 1u) {
        reject("affine simplex has an unexpected node count");
    }
    const auto& field_handler =
        system.fieldDofHandler(field);
    const auto* entity_map =
        field_handler.getEntityDofMap();
    if (entity_map == nullptr) {
        reject("field DOF handler has no entity map");
    }
    const auto offset = system.fieldDofOffset(field);

    std::vector<DofDescriptor> result;
    result.reserve(nodes.size() * dimension);
    for (std::size_t node = 0u; node < nodes.size(); ++node) {
        const auto vertex_dofs =
            entity_map->getVertexDofs(nodes[node]);
        if (vertex_dofs.size() != dimension) {
            reject(
                "P1 velocity vertex does not carry exactly one DOF per "
                "component");
        }
        const auto coordinate =
            mesh.getNodeCoordinates(nodes[node]);
        for (const auto value : coordinate) {
            if (!std::isfinite(value)) {
                reject("P1 velocity node has a nonfinite coordinate");
            }
        }
        for (std::size_t component = 0u;
             component < dimension;
             ++component) {
            const GlobalIndex field_dof =
                vertex_dofs[component];
            if (field_dof < 0 ||
                offset >
                    std::numeric_limits<GlobalIndex>::max() -
                        field_dof) {
                reject("field DOF plus system offset overflows");
            }
            result.push_back(DofDescriptor{
                .dof = offset + field_dof,
                .component = component,
                .coordinate = coordinate});
        }
    }

    std::vector<GlobalIndex> cell_dofs;
    for (const auto dof :
         field_handler.getCellDofs(local_cell)) {
        if (dof < 0 ||
            offset >
                std::numeric_limits<GlobalIndex>::max() - dof) {
            reject("cell field DOF plus system offset overflows");
        }
        cell_dofs.push_back(offset + dof);
    }
    std::vector<GlobalIndex> descriptor_dofs;
    descriptor_dofs.reserve(result.size());
    for (const auto& descriptor : result) {
        descriptor_dofs.push_back(descriptor.dof);
    }
    std::sort(cell_dofs.begin(), cell_dofs.end());
    std::sort(descriptor_dofs.begin(), descriptor_dofs.end());
    if (cell_dofs != descriptor_dofs ||
        std::adjacent_find(
            descriptor_dofs.begin(),
            descriptor_dofs.end()) != descriptor_dofs.end()) {
        reject(
            "cell field-DOF support disagrees with its P1 vertex/component "
            "map");
    }
    return result;
}

[[nodiscard]] std::array<Real, 3> physicalGradient(
    const geometry::MappedCutQuadraturePoint& point,
    const basis::Gradient& reference_gradient,
    std::size_t dimension)
{
    std::array<Real, 3> result{{0.0, 0.0, 0.0}};
    for (std::size_t physical = 0u;
         physical < dimension;
         ++physical) {
        for (std::size_t reference = 0u;
             reference < dimension;
             ++reference) {
            result[physical] +=
                point.inverse_jacobian[reference][physical] *
                reference_gradient[reference];
        }
        if (!std::isfinite(result[physical])) {
            reject("mapped P1 gradient is nonfinite");
        }
    }
    return result;
}

using Tensor = std::array<std::array<Real, 3>, 3>;

[[nodiscard]] Tensor symmetricGradientBasis(
    const std::array<Real, 3>& gradient,
    std::size_t component,
    std::size_t dimension) noexcept
{
    Tensor result{};
    for (std::size_t row = 0u; row < dimension; ++row) {
        for (std::size_t column = 0u;
             column < dimension;
             ++column) {
            result[row][column] =
                Real{0.5} *
                ((row == component ? gradient[column] : Real{0.0}) +
                 (column == component ? gradient[row] : Real{0.0}));
        }
    }
    return result;
}

[[nodiscard]] Real tensorDot(const Tensor& left,
                             const Tensor& right,
                             std::size_t dimension) noexcept
{
    Real result = 0.0;
    for (std::size_t row = 0u; row < dimension; ++row) {
        for (std::size_t column = 0u;
             column < dimension;
             ++column) {
            result +=
                left[row][column] * right[row][column];
        }
    }
    return result;
}

[[nodiscard]] std::array<Real, 3> strainNormalBasis(
    const Tensor& strain,
    const std::array<Real, 3>& normal,
    std::size_t dimension) noexcept
{
    std::array<Real, 3> result{{0.0, 0.0, 0.0}};
    for (std::size_t row = 0u; row < dimension; ++row) {
        for (std::size_t column = 0u;
             column < dimension;
             ++column) {
            result[row] +=
                strain[row][column] * normal[column];
        }
    }
    return result;
}

[[nodiscard]] Real vectorDot(
    const std::array<Real, 3>& left,
    const std::array<Real, 3>& right,
    std::size_t dimension) noexcept
{
    Real result = 0.0;
    for (std::size_t component = 0u;
         component < dimension;
         ++component) {
        result += left[component] * right[component];
    }
    return result;
}

[[nodiscard]] std::vector<Real> assembleCellDenominator(
    const assembly::IMeshAccess& mesh,
    GlobalIndex local_cell,
    std::span<const geometry::CutQuadratureRule* const> rules,
    std::span<const DofDescriptor> dofs,
    std::size_t dimension,
    long double& measure)
{
    if (rules.empty() ||
        dofs.size() != dimension * (dimension + 1u)) {
        reject("retained cell has an invalid rule or P1 DOF count");
    }
    basis::LagrangeBasis p1_basis(
        mesh.getCellType(local_cell), 1);
    const std::size_t n = dofs.size();
    std::vector<long double> accumulated(
        checkedSquare(n, "cell denominator"), 0.0L);

    for (const auto* rule : rules) {
        if (rule == nullptr ||
            rule->kind != geometry::CutQuadratureKind::Volume ||
            rule->frame != geometry::CutGeometryFrame::Reference ||
            rule->provenance.parent_entity != local_cell) {
            reject("retained cell rule has incompatible geometry metadata");
        }
        const auto mapped =
            geometry::mapCutQuadratureRuleToPhysical(mesh, *rule);
        measure +=
            static_cast<long double>(mapped.physical_measure);
        for (const auto& point : mapped.points) {
            const math::Vector<Real, 3> xi{
                point.reference_point[0],
                point.reference_point[1],
                point.reference_point[2]};
            std::vector<basis::Gradient> reference_gradients;
            p1_basis.evaluate_gradients(xi, reference_gradients);
            if (reference_gradients.size() != dimension + 1u) {
                reject("P1 basis returned an unexpected gradient count");
            }
            std::vector<Tensor> strains;
            strains.reserve(n);
            for (std::size_t node = 0u;
                 node < reference_gradients.size();
                 ++node) {
                const auto gradient =
                    physicalGradient(
                        point,
                        reference_gradients[node],
                        dimension);
                for (std::size_t component = 0u;
                     component < dimension;
                     ++component) {
                    strains.push_back(
                        symmetricGradientBasis(
                            gradient, component, dimension));
                }
            }
            for (std::size_t row = 0u; row < n; ++row) {
                for (std::size_t column = row;
                     column < n;
                     ++column) {
                    accumulated[row * n + column] +=
                        static_cast<long double>(
                            point.physical_weight) *
                        static_cast<long double>(
                            tensorDot(
                                strains[row],
                                strains[column],
                                dimension));
                }
            }
        }
    }

    std::vector<Real> result(
        checkedSquare(n, "cell denominator"), Real{0.0});
    for (std::size_t row = 0u; row < n; ++row) {
        for (std::size_t column = row;
             column < n;
             ++column) {
            const Real value =
                static_cast<Real>(
                    accumulated[row * n + column]);
            if (!std::isfinite(value)) {
                reject("cell denominator entry is nonfinite");
            }
            result[row * n + column] = value;
            result[column * n + row] = value;
        }
    }
    return result;
}

[[nodiscard]] std::vector<Real> assembleBoundaryNumerator(
    const assembly::IMeshAccess& mesh,
    const geometry::CutQuadratureRule& rule,
    std::span<const DofDescriptor> dofs,
    std::size_t dimension,
    Real h_normal,
    long double& measure)
{
    if (rule.kind != geometry::CutQuadratureKind::Interface ||
        rule.frame != geometry::CutGeometryFrame::Reference ||
        dofs.size() != dimension * (dimension + 1u) ||
        !finitePositive(h_normal)) {
        reject("generated-boundary rule has incompatible trace metadata");
    }
    const auto local_cell =
        static_cast<GlobalIndex>(
            rule.provenance.parent_entity);
    basis::LagrangeBasis p1_basis(
        mesh.getCellType(local_cell), 1);
    const auto mapped =
        geometry::mapCutQuadratureRuleToPhysical(mesh, rule);
    measure =
        static_cast<long double>(mapped.physical_measure);
    const std::size_t n = dofs.size();
    std::vector<long double> accumulated(
        checkedSquare(n, "boundary numerator"), 0.0L);

    for (const auto& point : mapped.points) {
        Real normal_norm_squared = 0.0;
        for (std::size_t component = 0u;
             component < dimension;
             ++component) {
            normal_norm_squared +=
                point.normal[component] *
                point.normal[component];
        }
        if (!std::isfinite(normal_norm_squared) ||
            std::abs(normal_norm_squared - Real{1.0}) >
                Real{4096.0} *
                std::numeric_limits<Real>::epsilon()) {
            reject("generated-boundary normal is not finite and unit length");
        }

        const math::Vector<Real, 3> xi{
            point.reference_point[0],
            point.reference_point[1],
            point.reference_point[2]};
        std::vector<basis::Gradient> reference_gradients;
        p1_basis.evaluate_gradients(xi, reference_gradients);
        if (reference_gradients.size() != dimension + 1u) {
            reject("P1 basis returned an unexpected gradient count");
        }
        std::vector<std::array<Real, 3>> tractions;
        tractions.reserve(n);
        for (std::size_t node = 0u;
             node < reference_gradients.size();
             ++node) {
            const auto gradient =
                physicalGradient(
                    point,
                    reference_gradients[node],
                    dimension);
            for (std::size_t component = 0u;
                 component < dimension;
                 ++component) {
                const auto strain =
                    symmetricGradientBasis(
                        gradient, component, dimension);
                tractions.push_back(
                    strainNormalBasis(
                        strain,
                        point.normal,
                        dimension));
            }
        }
        const long double scale =
            static_cast<long double>(point.physical_weight) *
            static_cast<long double>(
                Real{2.0} * h_normal);
        for (std::size_t row = 0u; row < n; ++row) {
            for (std::size_t column = row;
                 column < n;
                 ++column) {
                accumulated[row * n + column] +=
                    scale *
                    static_cast<long double>(
                        vectorDot(
                            tractions[row],
                            tractions[column],
                            dimension));
            }
        }
    }

    std::vector<Real> result(
        checkedSquare(n, "boundary numerator"), Real{0.0});
    for (std::size_t row = 0u; row < n; ++row) {
        for (std::size_t column = row;
             column < n;
             ++column) {
            const Real value =
                static_cast<Real>(
                    accumulated[row * n + column]);
            if (!std::isfinite(value)) {
                reject("boundary numerator entry is nonfinite");
            }
            result[row * n + column] = value;
            result[column * n + row] = value;
        }
    }
    return result;
}

[[nodiscard]] bool measuresAgree(Real left, Real right) noexcept
{
    if (!std::isfinite(left) || !std::isfinite(right)) {
        return false;
    }
    const long double scale =
        std::max(
            std::abs(static_cast<long double>(left)),
            std::abs(static_cast<long double>(right)));
    const long double tolerance =
        4096.0L *
            static_cast<long double>(
                std::numeric_limits<Real>::epsilon()) *
            scale +
        4.0L *
            static_cast<long double>(
                std::numeric_limits<Real>::denorm_min());
    return std::abs(
               static_cast<long double>(left) -
               static_cast<long double>(right)) <=
           tolerance;
}

struct LocalData {
    std::vector<CellContribution> cells{};
    std::vector<BoundaryContribution> boundaries{};
    std::vector<TangentRow> rows{};
};

[[nodiscard]] LocalData buildLocalData(
    const systems::FESystem& system,
    const SelectedState& selected,
    const GeneratedBoundaryAggregateTraceCertificationOptions& options,
    const CollectiveContext& collective)
{
    const auto& report = *selected.report;
    const auto& mesh = system.meshAccess();
    if (mesh.parallelRank() != collective.rank ||
        mesh.parallelSize() != collective.size) {
        reject("mesh and field communicator metadata disagree");
    }

    std::map<GlobalIndex,
             const constraints::SmallCutAggregationProlongationCell*>
        report_cells;
    std::set<GlobalIndex> report_cell_gids;
    std::set<GlobalIndex> raw_dofs;
    for (const auto& cell : report.active_cells) {
        report_cells.emplace(cell.cell_gid, &cell);
        report_cell_gids.insert(cell.cell_gid);
        raw_dofs.insert(
            cell.field_dofs.begin(), cell.field_dofs.end());
    }
    const auto local_by_gid =
        localCellByGlobalId(mesh, report_cell_gids);

    const auto volume_rule_indices =
        selected.cut_context
            ->generatedVolumeRuleIndexSpanForMarkerAndSide(
                options.volume_interface_marker,
                report.active_side);
    if (volume_rule_indices.size() > kMaximumVolumeRules) {
        reject("local retained-volume rule count exceeds its hard cap");
    }
    const auto& volume_rule_storage =
        selected.cut_context->volumeRules();
    std::size_t local_quadrature_points = 0u;
    std::map<GlobalIndex,
             std::map<std::uint64_t,
                      const geometry::CutQuadratureRule*>>
        provider_rules;
    for (const auto rule_index : volume_rule_indices) {
        if (rule_index >= volume_rule_storage.size()) {
            reject("retained-volume rule index is out of range");
        }
        const auto* rule =
            &volume_rule_storage[rule_index];
        if (rule->kind != geometry::CutQuadratureKind::Volume ||
            rule->side != report.active_side ||
            rule->frame != geometry::CutGeometryFrame::Reference ||
            rule->provenance.marker !=
                options.volume_interface_marker ||
            rule->points.empty() ||
            rule->points.size() >
                kMaximumQuadraturePointsPerRule ||
            rule->provenance.parent_entity_global_id < 0 ||
            rule->provenance.cut_topology_revision == 0u ||
            rule->provenance.free_surface_snapshot_revision_key !=
                report.revision.free_surface_snapshot_revision ||
            rule->provenance.source_value_revision !=
                report.revision.source_value_revision) {
            reject("retained volume rule has stale or incompatible provenance");
        }
        if (rule->points.size() >
            kMaximumLocalQuadraturePoints -
                local_quadrature_points) {
            reject(
                "local retained-volume quadrature count exceeds its hard "
                "cap");
        }
        local_quadrature_points += rule->points.size();
        const auto found =
            report_cells.find(
                rule->provenance.parent_entity_global_id);
        if (found == report_cells.end()) {
            reject("retained volume rule is outside aggregation active support");
        }
        const auto& cell = *found->second;
        if (rule->provenance.owner_rank !=
            cell.owner_rank) {
            reject(
                "retained volume rule owner disagrees with its report "
                "cell");
        }
        if (!std::binary_search(
                cell.retained_rule_stable_ids.begin(),
                cell.retained_rule_stable_ids.end(),
                rule->provenance.cut_topology_revision)) {
            reject("retained volume rule identity is absent from its report cell");
        }
        if (cell.retained_measure_provider_rank !=
            collective.rank) {
            continue;
        }
        const auto local =
            static_cast<GlobalIndex>(
                rule->provenance.parent_entity);
        const auto local_found =
            local_by_gid.find(cell.cell_gid);
        if (local < 0 ||
            local_found == local_by_gid.end() ||
            local_found->second != local ||
            mesh.getCellOwnerRank(local) !=
                cell.owner_rank) {
            reject("retained-volume provider cannot resolve its physical cell");
        }
        if (!provider_rules[cell.cell_gid]
                 .emplace(
                     rule->provenance.cut_topology_revision,
                     rule)
                 .second) {
            reject("retained-volume provider has a duplicate stable rule");
        }
    }

    LocalData result;
    result.cells.reserve(report.active_cells.size());
    for (const auto& cell : report.active_cells) {
        if (cell.retained_measure_provider_rank !=
            collective.rank) {
            continue;
        }
        const auto local_found =
            local_by_gid.find(cell.cell_gid);
        if (local_found == local_by_gid.end()) {
            reject("retained-volume provider lacks its declared support cell");
        }
        const auto rules_found =
            provider_rules.find(cell.cell_gid);
        if (rules_found == provider_rules.end() ||
            rules_found->second.size() !=
                cell.retained_rule_stable_ids.size()) {
            reject("retained-volume provider lacks an exact rule set");
        }
        std::vector<const geometry::CutQuadratureRule*> ordered_rules;
        ordered_rules.reserve(
            cell.retained_rule_stable_ids.size());
        for (const auto stable_id :
             cell.retained_rule_stable_ids) {
            const auto rule =
                rules_found->second.find(stable_id);
            if (rule == rules_found->second.end()) {
                reject("retained-volume provider lacks a declared stable rule");
            }
            ordered_rules.push_back(rule->second);
        }

        CellContribution contribution;
        contribution.cell_gid = cell.cell_gid;
        contribution.kind = cell.kind;
        contribution.stable_rule_ids =
            cell.retained_rule_stable_ids;
        contribution.dofs =
            cellDofDescriptors(
                system,
                options.field,
                local_found->second,
                selected.dimension);
        std::vector<GlobalIndex> sorted_dofs;
        sorted_dofs.reserve(contribution.dofs.size());
        for (const auto& descriptor : contribution.dofs) {
            sorted_dofs.push_back(descriptor.dof);
        }
        std::sort(sorted_dofs.begin(), sorted_dofs.end());
        if (sorted_dofs != cell.field_dofs) {
            reject(
                "retained-volume provider P1 DOFs disagree with the "
                "canonical report cell");
        }
        long double retained_measure = 0.0L;
        contribution.denominator =
            assembleCellDenominator(
                mesh,
                local_found->second,
                ordered_rules,
                contribution.dofs,
                selected.dimension,
                retained_measure);
        contribution.retained_measure =
            static_cast<Real>(retained_measure);
        if (!finitePositive(contribution.retained_measure) ||
            !measuresAgree(
                contribution.retained_measure,
                cell.retained_physical_volume)) {
            reject(
                "retained-volume provider measure disagrees with the "
                "aggregation report");
        }
        result.cells.push_back(std::move(contribution));
    }

    const auto boundary_rule_indices =
        selected.cut_context
            ->generatedInterfaceRuleIndexSpanForMarker(
            options.generated_active_boundary_marker);
    if (boundary_rule_indices.size() > kMaximumBoundaryRules) {
        reject("local generated-boundary rule count exceeds its hard cap");
    }
    const auto& boundary_rule_storage =
        selected.cut_context->interfaceRules();
    local_quadrature_points = 0u;
    std::set<std::uint64_t> local_boundary_ids;
    std::map<
        std::uint64_t,
        std::pair<GlobalIndex, GlobalIndex>>
        local_boundary_provenance;
    std::map<
        std::uint64_t,
        std::pair<GlobalIndex, GlobalIndex>>
        expected_local_boundary_provenance;
    for (const auto& domain :
         selected.snapshot->activeBoundaryDomains()) {
        if (domain.marker() !=
            options.generated_active_boundary_marker) {
            continue;
        }
        for (const auto& fragment : domain.fragments()) {
            if (fragment.owner_rank != collective.rank) {
                continue;
            }
            if (expected_local_boundary_provenance.size() >=
                kMaximumBoundaryRules) {
                reject(
                    "owned snapshot boundary fragments exceed their hard "
                    "cap");
            }
            if (!fragment.active() ||
                fragment.stable_id == 0u ||
                fragment.parent_cell_global_id < 0 ||
                fragment.parent_face_global_id < 0 ||
                fragment.side != report.active_side ||
                !expected_local_boundary_provenance
                     .emplace(
                         fragment.stable_id,
                         std::pair{
                             fragment.parent_cell_global_id,
                             fragment.parent_face_global_id})
                     .second) {
                reject(
                    "authoritative snapshot has invalid or duplicate "
                    "owned boundary fragments");
            }
        }
    }
    result.boundaries.reserve(boundary_rule_indices.size());
    for (const auto rule_index : boundary_rule_indices) {
        if (rule_index >= boundary_rule_storage.size()) {
            reject("generated-boundary rule index is out of range");
        }
        const auto* rule =
            &boundary_rule_storage[rule_index];
        if (rule->kind != geometry::CutQuadratureKind::Interface ||
            rule->side != report.active_side ||
            rule->frame != geometry::CutGeometryFrame::Reference ||
            rule->provenance.marker !=
                options.generated_active_boundary_marker ||
            rule->points.empty() ||
            rule->points.size() >
                kMaximumQuadraturePointsPerRule ||
            rule->provenance.parent_entity_global_id < 0 ||
            rule->provenance.parent_boundary_entity_global_id < 0 ||
            rule->provenance.owner_rank < 0 ||
            rule->provenance.owner_rank >= collective.size ||
            rule->provenance.cut_topology_revision == 0u ||
            rule->provenance.free_surface_snapshot_revision_key !=
                report.revision.free_surface_snapshot_revision ||
            rule->provenance.source_value_revision !=
                report.revision.source_value_revision) {
            reject(
                "generated-boundary rule has stale or incompatible "
                "provenance");
        }
        if (rule->points.size() >
            kMaximumLocalQuadraturePoints -
                local_quadrature_points) {
            reject(
                "local generated-boundary quadrature count exceeds its "
                "hard cap");
        }
        local_quadrature_points += rule->points.size();
        const auto report_cell =
            report_cells.find(
                rule->provenance.parent_entity_global_id);
        if (report_cell == report_cells.end()) {
            reject("generated-boundary parent cell is outside active support");
        }
        if (rule->provenance.owner_rank !=
            collective.rank) {
            continue;
        }
        if (!local_boundary_ids.insert(
                 rule->provenance.cut_topology_revision)
                 .second) {
            reject("boundary owner has a duplicate stable rule identity");
        }
        if (!local_boundary_provenance
                 .emplace(
                     rule->provenance.cut_topology_revision,
                     std::pair{
                         rule->provenance.parent_entity_global_id,
                         rule->provenance
                             .parent_boundary_entity_global_id})
                 .second) {
            reject("boundary owner has duplicate provenance");
        }
        const auto local_cell =
            static_cast<GlobalIndex>(
                rule->provenance.parent_entity);
        const auto local_face =
            static_cast<GlobalIndex>(
                rule->provenance.parent_boundary_entity);
        const auto local_found =
            local_by_gid.find(
                rule->provenance.parent_entity_global_id);
        if (local_cell < 0 ||
            local_face < 0 ||
            local_found == local_by_gid.end() ||
            local_found->second != local_cell ||
            mesh.getBoundaryFaceOwnerRank(
                local_face, local_cell) !=
                collective.rank ||
            mesh.getBoundaryFaceGlobalId(local_face) !=
                rule->provenance
                    .parent_boundary_entity_global_id) {
            reject("boundary owner cannot resolve its declared physical entities");
        }

        const auto dofs =
            cellDofDescriptors(
                system,
                options.field,
                local_cell,
                selected.dimension);
        std::vector<GlobalIndex> sorted_dofs;
        sorted_dofs.reserve(dofs.size());
        for (const auto& descriptor : dofs) {
            sorted_dofs.push_back(descriptor.dof);
        }
        std::sort(sorted_dofs.begin(), sorted_dofs.end());
        if (sorted_dofs !=
            report_cell->second->field_dofs) {
            reject(
                "boundary-owner P1 DOFs disagree with the canonical "
                "report cell");
        }
        const auto background =
            assembly::computeBackgroundEntityMeasures(
                mesh,
                local_cell,
                local_face,
                1,
                1);

        BoundaryContribution contribution;
        contribution.stable_rule_id =
            rule->provenance.cut_topology_revision;
        contribution.parent_cell_gid =
            rule->provenance.parent_entity_global_id;
        contribution.parent_face_gid =
            rule->provenance.parent_boundary_entity_global_id;
        contribution.raw_dofs.reserve(dofs.size());
        for (const auto& descriptor : dofs) {
            contribution.raw_dofs.push_back(descriptor.dof);
        }
        long double physical_measure = 0.0L;
        contribution.numerator =
            assembleBoundaryNumerator(
                mesh,
                *rule,
                dofs,
                selected.dimension,
                background.h_normal,
                physical_measure);
        contribution.physical_measure =
            static_cast<Real>(physical_measure);
        if (!finitePositive(
                contribution.physical_measure)) {
            reject("generated-boundary physical measure is invalid");
        }
        result.boundaries.push_back(std::move(contribution));
    }
    if (local_boundary_provenance !=
        expected_local_boundary_provenance) {
        reject(
            "cut context boundary rules do not exactly match the "
            "owner-filtered authoritative snapshot");
    }

    const auto& global_map =
        system.dofHandler().getDofMap();
    const auto& partition =
        system.dofHandler().getPartition();
    const auto& closed = system.constraints();
    result.rows.reserve(raw_dofs.size());
    for (const auto dof : raw_dofs) {
        const int owner = global_map.getDofOwner(dof);
        if (owner < 0 || owner >= collective.size) {
            reject("canonical owner of an active raw DOF is unavailable");
        }
        if (owner != collective.rank) {
            continue;
        }
        if (!partition.isOwned(dof) ||
            !partition.isRelevant(dof)) {
            reject("raw-DOF owner cannot materialize its tangent row");
        }
        TangentRow row;
        row.dof = dof;
        const auto constraint = closed.getConstraint(dof);
        if (!constraint.has_value()) {
            row.entries.push_back(
                constraints::ConstraintEntry{
                    .master_dof = dof,
                    .weight = Real{1.0}});
        } else {
            row.constrained = true;
            row.inhomogeneity =
                constraint->inhomogeneity;
            if (!std::isfinite(row.inhomogeneity) ||
                constraint->entries.size() >
                    kMaximumConstraintEntries) {
                reject(
                    "closed tangent row has invalid affine data or "
                    "exceeds its entry cap");
            }
            row.entries.assign(
                constraint->entries.begin(),
                constraint->entries.end());
            GlobalIndex previous = INVALID_GLOBAL_INDEX;
            for (const auto& entry : row.entries) {
                if (entry.master_dof < 0 ||
                    entry.master_dof <= previous ||
                    !std::isfinite(entry.weight) ||
                    entry.weight == Real{0.0} ||
                    closed.isConstrained(entry.master_dof)) {
                    reject(
                        "closed tangent row is nonterminal or "
                        "noncanonical");
                }
                previous = entry.master_dof;
            }
        }
        result.rows.push_back(std::move(row));
    }
    return result;
}

[[nodiscard]] constraints::SmallCutAggregationFinalRowKind
rowKind(const TangentRow& row) noexcept
{
    if (!row.constrained) {
        return constraints::SmallCutAggregationFinalRowKind::Identity;
    }
    if (!row.entries.empty()) {
        return constraints::SmallCutAggregationFinalRowKind::MasterBearing;
    }
    return row.inhomogeneity == Real{0.0}
               ? constraints::SmallCutAggregationFinalRowKind::
                     HomogeneousPin
               : constraints::SmallCutAggregationFinalRowKind::
                     FixedValue;
}

void validateCanonicalData(
    const CanonicalData& data,
    const SelectedState& selected)
{
    const auto& report = *selected.report;
    std::set<GlobalIndex> expected_rows;
    if (data.cells.size() != report.active_cells.size()) {
        reject("canonical retained-cell provider ledger is incomplete");
    }
    for (const auto& report_cell : report.active_cells) {
        const auto found =
            data.cells.find(report_cell.cell_gid);
        if (found == data.cells.end()) {
            reject("canonical retained-cell provider ledger is incomplete");
        }
        const auto& cell = found->second;
        if (cell.kind != report_cell.kind ||
            cell.stable_rule_ids !=
                report_cell.retained_rule_stable_ids ||
            !measuresAgree(
                cell.retained_measure,
                report_cell.retained_physical_volume)) {
            reject("canonical retained-cell record disagrees with its report");
        }
        std::vector<GlobalIndex> sorted_dofs;
        sorted_dofs.reserve(cell.dofs.size());
        for (const auto& descriptor : cell.dofs) {
            sorted_dofs.push_back(descriptor.dof);
        }
        std::sort(sorted_dofs.begin(), sorted_dofs.end());
        if (sorted_dofs != report_cell.field_dofs) {
            reject("canonical retained-cell DOFs disagree with its report");
        }
        expected_rows.insert(
            report_cell.field_dofs.begin(),
            report_cell.field_dofs.end());
    }
    if (data.rows.size() != expected_rows.size()) {
        reject("canonical live tangent-row ledger is incomplete");
    }
    for (const auto dof : expected_rows) {
        if (!data.rows.contains(dof) ||
            !data.descriptors.contains(dof)) {
            reject(
                "active raw DOF lacks a tangent row or geometric "
                "descriptor");
        }
    }

    for (const auto& [dof, row] : data.rows) {
        if (!expected_rows.contains(dof)) {
            reject("canonical tangent ledger contains an unexpected raw DOF");
        }
        for (const auto& entry : row.entries) {
            if (!expected_rows.contains(entry.master_dof)) {
                reject(
                    "closed tangent row has an external terminal master");
            }
            const auto master =
                data.rows.find(entry.master_dof);
            if (master == data.rows.end() ||
                master->second.constrained ||
                master->second.entries.size() != 1u ||
                master->second.entries.front().master_dof !=
                    entry.master_dof ||
                !sameRealBits(
                    master->second.entries.front().weight,
                    Real{1.0})) {
                reject(
                    "closed tangent row master is not canonical identity");
            }
        }
    }

    for (const auto& candidate : report.rows) {
        const auto found =
            data.rows.find(candidate.slave_dof);
        if (found == data.rows.end()) {
            reject("aggregation candidate row lacks a live tangent row");
        }
        const auto& live = found->second;
        if (candidate.final_kind != rowKind(live) ||
            !sameRealBits(
                candidate.final_inhomogeneity,
                live.inhomogeneity) ||
            candidate.final_entries.size() !=
                live.entries.size()) {
            reject(
                "aggregation candidate row disagrees with its live "
                "closed tangent");
        }
        for (std::size_t index = 0u;
             index < live.entries.size();
             ++index) {
            if (candidate.final_entries[index].master_dof !=
                    live.entries[index].master_dof ||
                !sameRealBits(
                    candidate.final_entries[index].weight,
                    live.entries[index].weight)) {
                reject(
                    "aggregation candidate row entries disagree with "
                    "the live closed tangent");
            }
        }
    }

    if (data.boundaries.size() > kMaximumBoundaryRules) {
        reject("canonical generated-boundary rule count exceeds its cap");
    }
    for (const auto& [stable_id, boundary] :
         data.boundaries) {
        (void)stable_id;
        const auto cell =
            data.cells.find(boundary.parent_cell_gid);
        if (cell == data.cells.end() ||
            boundary.raw_dofs.size() !=
                cell->second.dofs.size() ||
            boundary.numerator.size() !=
                checkedSquare(
                    boundary.raw_dofs.size(),
                    "boundary numerator")) {
            reject("boundary record lacks its canonical parent cell");
        }
        std::set<GlobalIndex> unique;
        for (std::size_t index = 0u;
             index < boundary.raw_dofs.size();
             ++index) {
            const auto dof = boundary.raw_dofs[index];
            if (dof != cell->second.dofs[index].dof ||
                !unique.insert(dof).second) {
                reject(
                    "boundary raw-DOF order disagrees with its parent "
                    "cell");
            }
        }
    }
}

struct WorkPatch {
    std::size_t canonical_patch_index{
        std::numeric_limits<std::size_t>::max()};
    bool synthetic{false};
    GlobalIndex root_cell_gid{INVALID_GLOBAL_INDEX};
    std::vector<GlobalIndex> support_cell_gids{};
    std::vector<std::uint64_t> boundary_rule_ids{};
};

[[nodiscard]] std::vector<WorkPatch> assignBoundaryRulesToPatches(
    const CanonicalData& data,
    const SelectedState& selected)
{
    const auto& report = *selected.report;
    std::map<std::size_t, std::vector<std::uint64_t>>
        assigned_existing;
    std::map<GlobalIndex, std::vector<std::uint64_t>>
        assigned_synthetic;

    for (const auto& [stable_id, boundary] :
         data.boundaries) {
        std::vector<std::size_t> member_candidates;
        std::vector<std::size_t> support_candidates;
        for (std::size_t index = 0u;
             index < report.patches.size();
             ++index) {
            const auto& patch = report.patches[index];
            if (std::binary_search(
                    patch.member_cell_gids.begin(),
                    patch.member_cell_gids.end(),
                    boundary.parent_cell_gid)) {
                member_candidates.push_back(index);
            } else if (std::binary_search(
                           patch.support_cell_gids.begin(),
                           patch.support_cell_gids.end(),
                           boundary.parent_cell_gid)) {
                support_candidates.push_back(index);
            }
        }
        const auto& candidates =
            member_candidates.empty()
                ? support_candidates
                : member_candidates;
        if (!candidates.empty()) {
            const auto index = candidates.front();
            if (!assigned_existing.contains(index) &&
                assigned_existing.size() +
                        assigned_synthetic.size() >=
                    kMaximumPatches) {
                reject(
                    "certified patch count exceeds its hard cap");
            }
            assigned_existing[index].push_back(stable_id);
            continue;
        }

        const auto cell =
            data.cells.find(boundary.parent_cell_gid);
        if (cell == data.cells.end()) {
            reject("boundary rule has no retained parent cell");
        }
        if (cell->second.kind !=
            constraints::SmallCutAggregationActiveCellKind::
                FullActive) {
            reject(
                "cut generated-boundary parent is not covered by an "
                "aggregation patch");
        }
        if (!assigned_synthetic.contains(
                boundary.parent_cell_gid) &&
            assigned_existing.size() +
                    assigned_synthetic.size() >=
                kMaximumPatches) {
            reject(
                "certified patch count exceeds its hard cap");
        }
        assigned_synthetic[boundary.parent_cell_gid]
            .push_back(stable_id);
    }

    std::vector<WorkPatch> result;
    result.reserve(
        assigned_existing.size() +
        assigned_synthetic.size());
    for (auto& [index, boundary_ids] :
         assigned_existing) {
        const auto& patch = report.patches[index];
        for (const auto stable_id : boundary_ids) {
            const auto boundary =
                data.boundaries.find(stable_id);
            if (boundary == data.boundaries.end() ||
                !std::binary_search(
                    patch.support_cell_gids.begin(),
                    patch.support_cell_gids.end(),
                    boundary->second.parent_cell_gid)) {
                reject(
                    "assigned aggregate patch does not include the "
                    "boundary parent in its denominator support");
            }
        }
        result.push_back(WorkPatch{
            .canonical_patch_index = index,
            .synthetic = false,
            .root_cell_gid = patch.root_cell_gid,
            .support_cell_gids = patch.support_cell_gids,
            .boundary_rule_ids = std::move(boundary_ids)});
    }
    for (auto& [cell_gid, boundary_ids] :
         assigned_synthetic) {
        result.push_back(WorkPatch{
            .canonical_patch_index =
                std::numeric_limits<std::size_t>::max(),
            .synthetic = true,
            .root_cell_gid = cell_gid,
            .support_cell_gids = {cell_gid},
            .boundary_rule_ids = std::move(boundary_ids)});
    }

    std::size_t assigned_count = 0u;
    std::set<std::uint64_t> assigned_ids;
    for (const auto& patch : result) {
        if (patch.support_cell_gids.empty() ||
            patch.boundary_rule_ids.empty()) {
            reject("certificate patch has empty support or numerator");
        }
        for (const auto stable_id :
             patch.boundary_rule_ids) {
            ++assigned_count;
            if (!assigned_ids.insert(stable_id).second) {
                reject("generated-boundary rule was assigned more than once");
            }
        }
    }
    if (assigned_count != data.boundaries.size()) {
        reject("generated-boundary rule assignment is incomplete");
    }
    return result;
}

[[nodiscard]] std::vector<Real> symmetrizedRealMatrix(
    std::span<const long double> wide,
    std::size_t dimension,
    std::string_view label)
{
    if (wide.size() !=
        checkedSquare(dimension, label)) {
        reject(std::string(label) + " has an incompatible wide size");
    }
    std::vector<Real> result(
        wide.size(), Real{0.0});
    for (std::size_t row = 0u;
         row < dimension;
         ++row) {
        for (std::size_t column = row;
             column < dimension;
             ++column) {
            const long double upper =
                wide[row * dimension + column];
            const long double lower =
                wide[column * dimension + row];
            const long double scale =
                std::max(
                    {1.0L,
                     std::abs(upper),
                     std::abs(lower)});
            const long double tolerance =
                8192.0L *
                static_cast<long double>(
                    std::numeric_limits<Real>::epsilon()) *
                static_cast<long double>(
                    std::max<std::size_t>(dimension, 1u)) *
                scale;
            if (!std::isfinite(upper) ||
                !std::isfinite(lower) ||
                std::abs(upper - lower) > tolerance) {
                reject(
                    std::string(label) +
                    " loses symmetry during tangent transformation");
            }
            const long double value = upper;
            const Real cast = static_cast<Real>(value);
            if (!std::isfinite(cast)) {
                reject(std::string(label) + " contains a nonfinite entry");
            }
            result[row * dimension + column] = cast;
            result[column * dimension + row] = cast;
        }
    }
    return result;
}

void accumulateReducedMatrix(
    std::span<const Real> raw_matrix,
    std::span<const GlobalIndex> raw_dofs,
    const std::map<GlobalIndex, TangentRow>& rows,
    const std::map<GlobalIndex, std::size_t>& terminal_index,
    std::span<long double> reduced)
{
    const std::size_t raw_dimension = raw_dofs.size();
    const std::size_t terminal_dimension =
        terminal_index.size();
    if (raw_matrix.size() !=
            checkedSquare(raw_dimension, "raw patch matrix") ||
        reduced.size() !=
            checkedSquare(
                terminal_dimension, "reduced patch matrix")) {
        reject("raw or reduced patch matrix has an incompatible dimension");
    }
    for (std::size_t raw_row = 0u;
         raw_row < raw_dimension;
         ++raw_row) {
        const auto row =
            rows.find(raw_dofs[raw_row]);
        if (row == rows.end()) {
            reject("raw patch row is absent from the tangent ledger");
        }
        for (std::size_t raw_column = 0u;
             raw_column < raw_dimension;
             ++raw_column) {
            const Real matrix_value =
                raw_matrix[
                    raw_row * raw_dimension + raw_column];
            if (matrix_value == Real{0.0}) {
                continue;
            }
            const auto column =
                rows.find(raw_dofs[raw_column]);
            if (column == rows.end()) {
                reject(
                    "raw patch column is absent from the tangent ledger");
            }
            for (const auto& row_entry :
                 row->second.entries) {
                const auto reduced_row =
                    terminal_index.find(
                        row_entry.master_dof);
                if (reduced_row ==
                    terminal_index.end()) {
                    reject("patch tangent row uses an unknown terminal master");
                }
                for (const auto& column_entry :
                     column->second.entries) {
                    const auto reduced_column =
                        terminal_index.find(
                            column_entry.master_dof);
                    if (reduced_column ==
                        terminal_index.end()) {
                        reject(
                            "patch tangent column uses an unknown "
                            "terminal master");
                    }
                    reduced[
                        reduced_row->second *
                            terminal_dimension +
                        reduced_column->second] +=
                        static_cast<long double>(
                            matrix_value) *
                        static_cast<long double>(
                            row_entry.weight) *
                        static_cast<long double>(
                            column_entry.weight);
                }
            }
        }
    }
}

[[nodiscard]] std::size_t rigidParameterCount(
    std::size_t dimension)
{
    if (dimension == 2u) {
        return 3u;
    }
    if (dimension == 3u) {
        return 6u;
    }
    reject("rigid modes require dimension two or three");
}

[[nodiscard]] std::vector<Real> rigidEvaluation(
    const DofDescriptor& descriptor,
    std::size_t dimension,
    const std::array<Real, 3>& center,
    Real scale)
{
    if (descriptor.component >= dimension ||
        !finitePositive(scale)) {
        reject("rigid-mode descriptor or coordinate scale is invalid");
    }
    const std::size_t count =
        rigidParameterCount(dimension);
    std::vector<Real> result(count, Real{0.0});
    result[descriptor.component] = Real{1.0};
    const Real x =
        (descriptor.coordinate[0] - center[0]) / scale;
    const Real y =
        (descriptor.coordinate[1] - center[1]) / scale;
    const Real z =
        (descriptor.coordinate[2] - center[2]) / scale;
    if (dimension == 2u) {
        result[2] =
            descriptor.component == 0u
                ? -y
                : x;
    } else {
        switch (descriptor.component) {
        case 0u:
            result[3] = Real{0.0};
            result[4] = z;
            result[5] = -y;
            break;
        case 1u:
            result[3] = -z;
            result[4] = Real{0.0};
            result[5] = x;
            break;
        case 2u:
            result[3] = y;
            result[4] = -x;
            result[5] = Real{0.0};
            break;
        default:
            reject("rigid-mode component is invalid");
        }
    }
    return result;
}

struct SmallNullspace {
    std::size_t rank{0u};
    std::size_t nullity{0u};
    Real reproduction_tolerance{0.0};
    Real maximum_reproduction_residual{0.0};
    // Row-major original_columns x nullity.
    std::vector<Real> basis{};
};

[[nodiscard]] SmallNullspace smallFullPivotNullspace(
    std::span<const long double> matrix,
    std::size_t rows,
    std::size_t columns)
{
    if (columns == 0u || columns > 6u ||
        matrix.size() != rows * columns) {
        reject("rigid-mode mismatch matrix has an invalid shape");
    }
    std::vector<long double> work(
        matrix.begin(), matrix.end());
    std::vector<std::size_t> permutation(columns);
    for (std::size_t column = 0u;
         column < columns;
         ++column) {
        permutation[column] = column;
    }
    long double maximum = 0.0L;
    for (const auto value : work) {
        if (!std::isfinite(value)) {
            reject("rigid-mode mismatch matrix is nonfinite");
        }
        maximum =
            std::max(maximum, std::abs(value));
    }
    const long double tolerance =
        4096.0L *
        static_cast<long double>(
            std::numeric_limits<Real>::epsilon()) *
        static_cast<long double>(
            std::max({rows, columns, std::size_t{1u}})) *
        std::max(maximum, 1.0L);

    std::size_t rank = 0u;
    while (rank < rows && rank < columns) {
        std::size_t pivot_row = rows;
        std::size_t pivot_column = columns;
        long double pivot_magnitude = 0.0L;
        for (std::size_t row = rank; row < rows; ++row) {
            for (std::size_t column = rank;
                 column < columns;
                 ++column) {
                const long double magnitude =
                    std::abs(work[row * columns + column]);
                if (magnitude > pivot_magnitude) {
                    pivot_magnitude = magnitude;
                    pivot_row = row;
                    pivot_column = column;
                }
            }
        }
        // Only an exact zero is structural. A nonzero pivot, however small,
        // is retained so a weak finite direction can never be quotiented out.
        if (pivot_magnitude == 0.0L) {
            break;
        }
        if (pivot_row != rank) {
            for (std::size_t column = 0u;
                 column < columns;
                 ++column) {
                std::swap(
                    work[rank * columns + column],
                    work[pivot_row * columns + column]);
            }
        }
        if (pivot_column != rank) {
            for (std::size_t row = 0u; row < rows; ++row) {
                std::swap(
                    work[row * columns + rank],
                    work[row * columns + pivot_column]);
            }
            std::swap(
                permutation[rank],
                permutation[pivot_column]);
        }
        const long double pivot =
            work[rank * columns + rank];
        for (std::size_t column = 0u;
             column < columns;
             ++column) {
            work[rank * columns + column] /= pivot;
        }
        for (std::size_t row = 0u; row < rows; ++row) {
            if (row == rank) {
                continue;
            }
            const long double factor =
                work[row * columns + rank];
            if (factor == 0.0L) {
                continue;
            }
            for (std::size_t column = 0u;
                 column < columns;
                 ++column) {
                work[row * columns + column] -=
                    factor *
                    work[rank * columns + column];
            }
        }
        ++rank;
    }

    SmallNullspace result;
    result.rank = rank;
    result.nullity = columns - rank;
    result.reproduction_tolerance =
        outwardRoundNonnegative(
            tolerance,
            "rigid-mode reproduction tolerance");
    result.basis.assign(
        columns * result.nullity, Real{0.0});
    for (std::size_t free_mode = 0u;
         free_mode < result.nullity;
         ++free_mode) {
        const std::size_t free_column =
            rank + free_mode;
        std::vector<long double> permuted(
            columns, 0.0L);
        permuted[free_column] = 1.0L;
        for (std::size_t pivot = 0u;
             pivot < rank;
             ++pivot) {
            permuted[pivot] =
                -work[pivot * columns + free_column];
        }
        for (std::size_t column = 0u;
             column < columns;
             ++column) {
            const Real value =
                static_cast<Real>(permuted[column]);
            if (!std::isfinite(value)) {
                reject("rigid-mode nullspace basis is nonfinite");
            }
            result.basis[
                permutation[column] * result.nullity +
                free_mode] = value;
        }
    }
    long double maximum_residual = 0.0L;
    for (std::size_t row = 0u; row < rows; ++row) {
        for (std::size_t mode = 0u;
             mode < result.nullity;
             ++mode) {
            long double residual = 0.0L;
            for (std::size_t column = 0u;
                 column < columns;
                 ++column) {
                residual +=
                    matrix[row * columns + column] *
                    static_cast<long double>(
                        result.basis[
                            column * result.nullity +
                            mode]);
            }
            maximum_residual =
                std::max(
                    maximum_residual,
                    std::abs(residual));
        }
    }
    result.maximum_reproduction_residual =
        outwardRoundNonnegative(
            maximum_residual,
            "maximum rigid-mode reproduction residual");
    return result;
}

struct Binary64DyadicFactor {
    bool negative{false};
    std::uint64_t significand{0u};
    int exponent{0};
};

[[nodiscard]] Binary64DyadicFactor decodeBinary64Factor(Real value)
{
    static_assert(
        sizeof(Real) == sizeof(std::uint64_t) &&
            std::numeric_limits<Real>::is_iec559 &&
            std::numeric_limits<Real>::digits == 53,
        "exact trace certification requires IEEE binary64 Real");
    constexpr std::uint64_t kFractionMask =
        (UINT64_C(1) << 52u) - UINT64_C(1);
    const auto bits = std::bit_cast<std::uint64_t>(value);
    const auto exponent_bits =
        (bits >> 52u) & UINT64_C(0x7ff);
    const auto fraction = bits & kFractionMask;
    if (exponent_bits == UINT64_C(0x7ff)) {
        reject("exact binary64 accumulator received a nonfinite value");
    }
    Binary64DyadicFactor result;
    result.negative = (bits >> 63u) != 0u;
    if (exponent_bits == 0u) {
        result.significand = fraction;
        result.exponent = -1074;
    } else {
        result.significand =
            (UINT64_C(1) << 52u) | fraction;
        result.exponent =
            static_cast<int>(exponent_bits) - 1023 - 52;
    }
    return result;
}

struct UnsignedProduct128 {
    std::uint64_t low{0u};
    std::uint64_t high{0u};
};

[[nodiscard]] UnsignedProduct128 multiplyBinary64Significands(
    std::uint64_t left,
    std::uint64_t right) noexcept
{
    constexpr std::uint64_t kLowWordMask =
        UINT64_C(0xffffffff);
    const std::uint64_t left_low =
        left & kLowWordMask;
    const std::uint64_t left_high =
        left >> 32u;
    const std::uint64_t right_low =
        right & kLowWordMask;
    const std::uint64_t right_high =
        right >> 32u;

    const std::uint64_t low_low =
        left_low * right_low;
    const std::uint64_t low_high =
        left_low * right_high;
    const std::uint64_t high_low =
        left_high * right_low;
    const std::uint64_t high_high =
        left_high * right_high;

    UnsignedProduct128 result;
    result.low = low_low;
    const auto add_shifted_low =
        [&](std::uint64_t partial) {
            const std::uint64_t prior =
                result.low;
            result.low += partial << 32u;
            return static_cast<std::uint64_t>(
                result.low < prior);
        };
    const std::uint64_t first_carry =
        add_shifted_low(low_high);
    const std::uint64_t second_carry =
        add_shifted_low(high_low);
    result.high =
        high_high +
        (low_high >> 32u) +
        (high_low >> 32u) +
        first_carry +
        second_carry;
    return result;
}

/**
 * Exact signed sum of the capped products used by one certification row.
 *
 * Products use a common 2^-2148 unit. Sixty-seven 64-bit limbs cover the
 * full product exponent range plus the carry bits needed by the 128-entry
 * tangent cap and the six rigid parameters.
 */
class ExactBinary64ProductAccumulator {
public:
    void addProduct(Real left,
                    Real right,
                    bool negate = false)
    {
        const auto left_factor =
            decodeBinary64Factor(left);
        const auto right_factor =
            decodeBinary64Factor(right);
        if (left_factor.significand == 0u ||
            right_factor.significand == 0u) {
            return;
        }
        const auto product =
            multiplyBinary64Significands(
                left_factor.significand,
                right_factor.significand);
        const int product_exponent =
            left_factor.exponent +
            right_factor.exponent;
        constexpr int kMinimumProductExponent = -2148;
        if (product_exponent <
            kMinimumProductExponent) {
            reject("exact binary64 product exponent is out of range");
        }
        const auto bit_offset =
            static_cast<std::size_t>(
                product_exponent -
                kMinimumProductExponent);
        auto& magnitude =
            (left_factor.negative ^
             right_factor.negative ^
             negate)
                ? negative_
                : positive_;
        addShifted(
            magnitude,
            product.low,
            product.high,
            bit_offset);
    }

    [[nodiscard]] bool isZero() const noexcept
    {
        return positive_ == negative_;
    }

private:
    static constexpr std::size_t kLimbCount = 67u;
    using Magnitude =
        std::array<std::uint64_t, kLimbCount>;

    static void addWord(Magnitude& magnitude,
                        std::size_t index,
                        std::uint64_t word)
    {
        while (word != 0u) {
            if (index >= magnitude.size()) {
                reject(
                    "exact binary64 accumulator exceeded its fixed "
                    "range");
            }
            const auto prior = magnitude[index];
            magnitude[index] += word;
            word =
                magnitude[index] < prior
                    ? UINT64_C(1)
                    : UINT64_C(0);
            ++index;
        }
    }

    static void addShifted(Magnitude& magnitude,
                           std::uint64_t low,
                           std::uint64_t high,
                           std::size_t bit_offset)
    {
        const std::size_t limb = bit_offset / 64u;
        const unsigned shift =
            static_cast<unsigned>(bit_offset % 64u);
        if (shift == 0u) {
            addWord(magnitude, limb, low);
            addWord(magnitude, limb + 1u, high);
            return;
        }
        addWord(magnitude, limb, low << shift);
        addWord(
            magnitude,
            limb + 1u,
            (high << shift) |
                (low >> (64u - shift)));
        addWord(
            magnitude,
            limb + 2u,
            high >> (64u - shift));
    }

    Magnitude positive_{};
    Magnitude negative_{};
};

[[nodiscard]] std::uint64_t modularPower(
    std::uint64_t base,
    std::uint64_t exponent,
    std::uint64_t modulus)
{
    std::uint64_t result = 1u;
    while (exponent != 0u) {
        if ((exponent & UINT64_C(1)) != 0u) {
            result = (result * base) % modulus;
        }
        base = (base * base) % modulus;
        exponent >>= 1u;
    }
    return result;
}

[[nodiscard]] std::uint64_t binary64IntegerResidue(
    Real value,
    std::uint64_t modulus)
{
    const auto factor = decodeBinary64Factor(value);
    if (factor.significand == 0u) {
        return 0u;
    }
    const int scaled_exponent =
        factor.exponent + 1074;
    if (scaled_exponent < 0) {
        reject("binary64 exact-rank exponent is out of range");
    }
    std::uint64_t residue =
        (factor.significand % modulus) *
        modularPower(
            UINT64_C(2),
            static_cast<std::uint64_t>(
                scaled_exponent),
            modulus) %
        modulus;
    if (factor.negative && residue != 0u) {
        residue = modulus - residue;
    }
    return residue;
}

/**
 * Prove full column rank over the binary64 dyadics.
 *
 * Scaling by 2^1074 maps every finite binary64 entry to an integer. Full rank
 * modulo the fixed prime implies full rank over those integers. A nonzero
 * exact minor divisible by the prime can only cause conservative rejection.
 */
[[nodiscard]] std::optional<std::vector<std::size_t>>
provenIndependentRowCoordinates(
    std::span<const Real> matrix,
    std::size_t rows,
    std::size_t columns)
{
    if (columns == 0u) {
        return std::vector<std::size_t>{};
    }
    if (matrix.size() != rows * columns) {
        reject("exact-rank matrix has an invalid shape");
    }
    if (columns > rows) {
        return std::nullopt;
    }
    constexpr std::uint64_t kPrime =
        UINT64_C(4294967291);
    std::vector<std::uint64_t> work(
        matrix.size(), 0u);
    std::vector<std::size_t> row_coordinates(rows, 0u);
    for (std::size_t row = 0u; row < rows; ++row) {
        row_coordinates[row] = row;
    }
    for (std::size_t index = 0u;
         index < matrix.size();
         ++index) {
        work[index] =
            binary64IntegerResidue(
                matrix[index], kPrime);
    }
    std::size_t rank = 0u;
    for (std::size_t column = 0u;
         column < columns && rank < rows;
         ++column) {
        std::size_t pivot = rank;
        while (pivot < rows &&
               work[pivot * columns + column] == 0u) {
            ++pivot;
        }
        if (pivot == rows) {
            continue;
        }
        if (pivot != rank) {
            for (std::size_t entry = 0u;
                 entry < columns;
                 ++entry) {
                std::swap(
                    work[rank * columns + entry],
                    work[pivot * columns + entry]);
            }
            std::swap(
                row_coordinates[rank],
                row_coordinates[pivot]);
        }
        const auto inverse =
            modularPower(
                work[rank * columns + column],
                kPrime - UINT64_C(2),
                kPrime);
        for (std::size_t row = rank + 1u;
             row < rows;
             ++row) {
            const auto entry =
                work[row * columns + column];
            if (entry == 0u) {
                continue;
            }
            const auto factor =
                (entry * inverse) % kPrime;
            for (std::size_t trailing = column;
                 trailing < columns;
                 ++trailing) {
                const auto reduction =
                    (factor *
                     work[rank * columns + trailing]) %
                    kPrime;
                auto& target =
                    work[row * columns + trailing];
                target =
                    target >= reduction
                        ? target - reduction
                        : kPrime - (reduction - target);
            }
        }
        ++rank;
    }
    if (rank != columns) {
        return std::nullopt;
    }
    row_coordinates.resize(columns);
    return row_coordinates;
}

[[nodiscard]] bool hasProvenFullColumnRank(
    std::span<const Real> matrix,
    std::size_t rows,
    std::size_t columns)
{
    return provenIndependentRowCoordinates(
               matrix, rows, columns)
        .has_value();
}

[[nodiscard]] bool hasExactNullspaceAction(
    std::span<const Real> matrix,
    std::size_t dimension,
    std::span<const Real> nullspace,
    std::size_t nullity)
{
    if (matrix.size() !=
            checkedSquare(dimension, "nullspace action matrix") ||
        nullspace.size() != dimension * nullity) {
        reject("exact nullspace-action inputs have incompatible sizes");
    }
    for (std::size_t row = 0u; row < dimension; ++row) {
        for (std::size_t mode = 0u;
             mode < nullity;
             ++mode) {
            ExactBinary64ProductAccumulator action;
            for (std::size_t column = 0u;
                 column < dimension;
                 ++column) {
                action.addProduct(
                    matrix[row * dimension + column],
                    nullspace[
                        column * nullity + mode]);
            }
            if (!action.isZero()) {
                return false;
            }
        }
    }
    return true;
}

[[nodiscard]] bool hasExactRigidReproduction(
    const std::set<GlobalIndex>& raw_dofs,
    const std::map<GlobalIndex, std::size_t>& terminal_index,
    const CanonicalData& data,
    std::size_t dimension,
    const std::array<Real, 3>& center,
    Real coordinate_scale,
    std::span<const Real> terminal_rigid,
    std::span<const Real> rigid_parameter_basis,
    std::span<const Real> explicit_nullspace,
    std::size_t rigid_count,
    std::size_t nullity)
{
    const std::size_t terminal_count =
        terminal_index.size();
    if (terminal_rigid.size() !=
            terminal_count * rigid_count ||
        rigid_parameter_basis.size() !=
            rigid_count * nullity ||
        explicit_nullspace.size() !=
            terminal_count * nullity) {
        reject("exact rigid-reproduction inputs have incompatible sizes");
    }
    for (std::size_t terminal = 0u;
         terminal < terminal_count;
         ++terminal) {
        for (std::size_t mode = 0u;
             mode < nullity;
             ++mode) {
            ExactBinary64ProductAccumulator residual;
            for (std::size_t parameter = 0u;
                 parameter < rigid_count;
                 ++parameter) {
                residual.addProduct(
                    terminal_rigid[
                        terminal * rigid_count +
                        parameter],
                    rigid_parameter_basis[
                        parameter * nullity +
                        mode]);
            }
            residual.addProduct(
                explicit_nullspace[
                    terminal * nullity + mode],
                Real{1.0},
                true);
            if (!residual.isZero()) {
                return false;
            }
        }
    }

    for (const auto dof : raw_dofs) {
        const auto descriptor =
            data.descriptors.find(dof);
        const auto row = data.rows.find(dof);
        if (descriptor == data.descriptors.end() ||
            row == data.rows.end()) {
            reject("exact rigid reproduction lacks canonical DOF data");
        }
        const auto raw_rigid =
            rigidEvaluation(
                descriptor->second,
                dimension,
                center,
                coordinate_scale);
        for (std::size_t mode = 0u;
             mode < nullity;
             ++mode) {
            ExactBinary64ProductAccumulator residual;
            for (const auto& entry : row->second.entries) {
                const auto terminal =
                    terminal_index.find(
                        entry.master_dof);
                if (terminal ==
                    terminal_index.end()) {
                    reject(
                        "exact rigid reproduction references an unknown "
                        "terminal master");
                }
                residual.addProduct(
                    entry.weight,
                    explicit_nullspace[
                        terminal->second * nullity +
                        mode]);
            }
            for (std::size_t parameter = 0u;
                 parameter < rigid_count;
                 ++parameter) {
                residual.addProduct(
                    raw_rigid[parameter],
                    rigid_parameter_basis[
                        parameter * nullity +
                        mode],
                    true);
            }
            if (!residual.isZero()) {
                return false;
            }
        }
    }
    return true;
}

struct PatchPencil {
    std::vector<Real> numerator{};
    std::vector<Real> denominator{};
    std::vector<Real> explicit_nullspace{};
    std::size_t raw_dof_count{0u};
    std::size_t terminal_dimension{0u};
    std::size_t rigid_candidate_count{0u};
    std::size_t rigid_constraint_rank{0u};
    std::size_t structural_rigid_mode_count{0u};
    GeneratedBoundaryRigidModeQuotientStatus
        rigid_quotient_status{
            GeneratedBoundaryRigidModeQuotientStatus::
                NotApplicable};
    Real rigid_reproduction_tolerance{0.0};
    Real maximum_rigid_reproduction_residual{0.0};
    Real retained_measure{0.0};
    Real boundary_measure{0.0};
};

[[nodiscard]] PatchPencil buildPatchPencil(
    const WorkPatch& patch,
    const CanonicalData& data,
    std::size_t dimension,
    std::size_t maximum_reduced_dimension)
{
    std::set<GlobalIndex> raw_dofs;
    Real retained_measure = 0.0;
    for (const auto cell_gid :
         patch.support_cell_gids) {
        const auto cell = data.cells.find(cell_gid);
        if (cell == data.cells.end()) {
            reject("certificate patch references an unknown support cell");
        }
        retained_measure =
            outwardAddNonnegative(
                retained_measure,
                cell->second.retained_measure,
                "patch retained measure");
        for (const auto& descriptor :
             cell->second.dofs) {
            if (!raw_dofs.contains(descriptor.dof) &&
                raw_dofs.size() >=
                    kMaximumPatchRawDofs) {
                reject(
                    "patch raw-DOF support exceeds its hard cap");
            }
            raw_dofs.insert(descriptor.dof);
        }
    }

    std::set<GlobalIndex> terminal_dofs;
    const std::size_t terminal_limit =
        std::min(
            maximum_reduced_dimension,
            kHardMaximumReducedDimension);
    for (const auto dof : raw_dofs) {
        const auto row = data.rows.find(dof);
        if (row == data.rows.end()) {
            reject("certificate patch lacks a live tangent row");
        }
        for (const auto& entry : row->second.entries) {
            if (!raw_dofs.contains(entry.master_dof)) {
                reject(
                    "patch support omits a cell carrying a terminal "
                    "master");
            }
            if (!terminal_dofs.contains(
                    entry.master_dof) &&
                terminal_dofs.size() >=
                    terminal_limit) {
                reject(
                    "patch terminal tangent dimension exceeds its hard "
                    "cap");
            }
            terminal_dofs.insert(entry.master_dof);
        }
    }
    if (terminal_dofs.size() >
            maximum_reduced_dimension ||
        terminal_dofs.size() >
            kHardMaximumReducedDimension) {
        reject("patch terminal tangent dimension exceeds its hard cap");
    }
    const auto dense_entries =
        checkedSquare(
            terminal_dofs.size(), "patch reduced matrix");
    if (terminal_dofs.size() != 0u &&
        (dense_entries >
             std::numeric_limits<std::size_t>::max() / 256u ||
         dense_entries * 256u >
             kMaximumDenseModeledBytes)) {
        reject("patch dense-memory model exceeds its hard cap");
    }

    std::map<GlobalIndex, std::size_t> terminal_index;
    std::size_t next_terminal = 0u;
    for (const auto dof : terminal_dofs) {
        terminal_index.emplace(dof, next_terminal++);
    }
    std::vector<long double> denominator(
        dense_entries, 0.0L);
    for (const auto cell_gid :
         patch.support_cell_gids) {
        const auto& cell =
            data.cells.at(cell_gid);
        std::vector<GlobalIndex> cell_dofs;
        cell_dofs.reserve(cell.dofs.size());
        for (const auto& descriptor : cell.dofs) {
            cell_dofs.push_back(descriptor.dof);
        }
        accumulateReducedMatrix(
            cell.denominator,
            cell_dofs,
            data.rows,
            terminal_index,
            denominator);
    }

    std::vector<long double> numerator(
        dense_entries, 0.0L);
    Real boundary_measure = 0.0;
    for (const auto stable_id :
         patch.boundary_rule_ids) {
        const auto& boundary =
            data.boundaries.at(stable_id);
        boundary_measure =
            outwardAddNonnegative(
                boundary_measure,
                boundary.physical_measure,
                "patch boundary measure");
        accumulateReducedMatrix(
            boundary.numerator,
            boundary.raw_dofs,
            data.rows,
            terminal_index,
            numerator);
    }

    PatchPencil result;
    result.raw_dof_count = raw_dofs.size();
    result.terminal_dimension = terminal_dofs.size();
    result.retained_measure = retained_measure;
    result.boundary_measure = boundary_measure;
    result.denominator =
        symmetrizedRealMatrix(
            denominator,
            terminal_dofs.size(),
            "patch denominator");
    result.numerator =
        symmetrizedRealMatrix(
            numerator,
            terminal_dofs.size(),
            "patch numerator");

    if (terminal_dofs.empty()) {
        return result;
    }

    std::array<Real, 3> center{{
        std::numeric_limits<Real>::infinity(),
        std::numeric_limits<Real>::infinity(),
        std::numeric_limits<Real>::infinity()}};
    std::vector<std::array<Real, 3>> unique_coordinates;
    for (const auto dof : raw_dofs) {
        const auto descriptor =
            data.descriptors.find(dof);
        if (descriptor == data.descriptors.end()) {
            reject("patch raw DOF lacks a geometric descriptor");
        }
        unique_coordinates.push_back(
            descriptor->second.coordinate);
    }
    std::sort(
        unique_coordinates.begin(),
        unique_coordinates.end());
    unique_coordinates.erase(
        std::unique(
            unique_coordinates.begin(),
            unique_coordinates.end(),
            [](const auto& left, const auto& right) {
                return sameRealBits(left[0], right[0]) &&
                       sameRealBits(left[1], right[1]) &&
                       sameRealBits(left[2], right[2]);
            }),
        unique_coordinates.end());
    if (unique_coordinates.size() < dimension + 1u) {
        reject("patch support does not span an affine simplex");
    }
    center = unique_coordinates.front();
    Real maximum_coordinate_delta = 0.0;
    for (const auto& coordinate : unique_coordinates) {
        for (std::size_t component = 0u;
             component < dimension;
             ++component) {
            const Real delta =
                coordinate[component] - center[component];
            if (!std::isfinite(delta)) {
                reject("patch coordinate difference is nonfinite");
            }
            maximum_coordinate_delta =
                std::max(
                    maximum_coordinate_delta,
                    std::abs(delta));
        }
    }
    if (!finitePositive(maximum_coordinate_delta)) {
        reject("patch coordinate diameter is invalid");
    }
    int coordinate_exponent = 0;
    (void)std::frexp(
        maximum_coordinate_delta,
        &coordinate_exponent);
    const Real coordinate_scale =
        std::scalbn(
            Real{1.0},
            coordinate_exponent - 1);
    if (!finitePositive(coordinate_scale)) {
        reject("patch dyadic coordinate scale is invalid");
    }

    const std::size_t rigid_count =
        rigidParameterCount(dimension);
    std::vector<Real> terminal_rigid(
        terminal_dofs.size() * rigid_count,
        Real{0.0});
    for (const auto& [dof, index] : terminal_index) {
        const auto descriptor =
            data.descriptors.find(dof);
        if (descriptor == data.descriptors.end()) {
            reject("terminal master lacks a geometric descriptor");
        }
        const auto values =
            rigidEvaluation(
                descriptor->second,
                dimension,
                center,
                coordinate_scale);
        for (std::size_t mode = 0u;
             mode < rigid_count;
             ++mode) {
            terminal_rigid[
                index * rigid_count + mode] =
                values[mode];
        }
    }

    std::vector<long double> mismatch(
        raw_dofs.size() * rigid_count,
        0.0L);
    std::size_t raw_index = 0u;
    for (const auto dof : raw_dofs) {
        const auto& descriptor =
            data.descriptors.at(dof);
        const auto raw_rigid =
            rigidEvaluation(
                descriptor,
                dimension,
                center,
                coordinate_scale);
        const auto& row = data.rows.at(dof);
        for (std::size_t mode = 0u;
             mode < rigid_count;
             ++mode) {
            long double value =
                -static_cast<long double>(
                    raw_rigid[mode]);
            for (const auto& entry : row.entries) {
                const auto terminal =
                    terminal_index.find(
                        entry.master_dof);
                if (terminal ==
                    terminal_index.end()) {
                    reject(
                        "rigid-mode tangent uses an unknown terminal "
                        "master");
                }
                value +=
                    static_cast<long double>(
                        entry.weight) *
                    static_cast<long double>(
                        terminal_rigid[
                            terminal->second *
                                rigid_count +
                            mode]);
            }
            mismatch[
                raw_index * rigid_count + mode] =
                value;
        }
        ++raw_index;
    }
    const auto rigid_nullspace =
        smallFullPivotNullspace(
            mismatch,
            raw_dofs.size(),
            rigid_count);
    result.rigid_constraint_rank =
        rigid_nullspace.rank;
    result.rigid_candidate_count =
        rigid_nullspace.nullity;
    result.rigid_quotient_status =
        rigid_nullspace.nullity == 0u
            ? GeneratedBoundaryRigidModeQuotientStatus::
                  NoCandidate
            : GeneratedBoundaryRigidModeQuotientStatus::
                  ReproductionNotExact;
    result.rigid_reproduction_tolerance =
        rigid_nullspace.reproduction_tolerance;
    result.maximum_rigid_reproduction_residual =
        rigid_nullspace.maximum_reproduction_residual;
    result.explicit_nullspace.assign(
        terminal_dofs.size() *
            rigid_nullspace.nullity,
        Real{0.0});
    for (std::size_t terminal = 0u;
         terminal < terminal_dofs.size();
         ++terminal) {
        for (std::size_t mode = 0u;
             mode < rigid_nullspace.nullity;
             ++mode) {
            long double value = 0.0L;
            for (std::size_t parameter = 0u;
                 parameter < rigid_count;
                 ++parameter) {
                value +=
                    static_cast<long double>(
                        terminal_rigid[
                            terminal * rigid_count +
                            parameter]) *
                    static_cast<long double>(
                        rigid_nullspace.basis[
                            parameter *
                                rigid_nullspace.nullity +
                            mode]);
            }
            const Real cast =
                static_cast<Real>(value);
            if (!std::isfinite(cast)) {
                reject("explicit rigid nullspace contains a nonfinite value");
            }
            result.explicit_nullspace[
                terminal * rigid_nullspace.nullity +
                mode] = cast;
        }
    }
    if (rigid_nullspace.nullity == 0u) {
        return result;
    }
    if (!hasExactRigidReproduction(
            raw_dofs,
            terminal_index,
            data,
            dimension,
            center,
            coordinate_scale,
            terminal_rigid,
            rigid_nullspace.basis,
            result.explicit_nullspace,
            rigid_count,
            rigid_nullspace.nullity)) {
        result.explicit_nullspace.clear();
        return result;
    }
    if (!hasProvenFullColumnRank(
            result.explicit_nullspace,
            result.terminal_dimension,
            rigid_nullspace.nullity)) {
        result.explicit_nullspace.clear();
        result.rigid_quotient_status =
            GeneratedBoundaryRigidModeQuotientStatus::
                CandidateRankNotProven;
        return result;
    }
    if (!hasExactNullspaceAction(
            result.denominator,
            result.terminal_dimension,
            result.explicit_nullspace,
            rigid_nullspace.nullity) ||
        !hasExactNullspaceAction(
            result.numerator,
            result.terminal_dimension,
            result.explicit_nullspace,
            rigid_nullspace.nullity)) {
        result.explicit_nullspace.clear();
        result.rigid_quotient_status =
            GeneratedBoundaryRigidModeQuotientStatus::
                NonzeroPencilAction;
        return result;
    }
    result.structural_rigid_mode_count =
        rigid_nullspace.nullity;
    result.rigid_quotient_status =
        GeneratedBoundaryRigidModeQuotientStatus::
            Applied;
    return result;
}

[[nodiscard]] GeneratedBoundaryAggregateTracePatchCertificate
certifyPatch(
    const WorkPatch& patch,
    const CanonicalData& data,
    std::size_t dimension,
    std::size_t maximum_reduced_dimension)
{
    const auto pencil =
        buildPatchPencil(
            patch,
            data,
            dimension,
            maximum_reduced_dimension);
    GeneratedBoundaryAggregateTracePatchCertificate result;
    result.canonical_patch_index =
        patch.canonical_patch_index;
    result.synthetic_full_active_patch =
        patch.synthetic;
    result.root_cell_gid =
        patch.root_cell_gid;
    result.support_cell_gids =
        patch.support_cell_gids;
    result.boundary_rule_stable_ids =
        patch.boundary_rule_ids;
    result.raw_support_dof_count =
        pencil.raw_dof_count;
    result.terminal_tangent_dof_count =
        pencil.terminal_dimension;
    result.rigid_mode_candidate_count =
        pencil.rigid_candidate_count;
    result.structural_rigid_mode_count =
        pencil.structural_rigid_mode_count;
    result.rigid_mode_constraint_rank =
        pencil.rigid_constraint_rank;
    result.rigid_mode_quotient_status =
        pencil.rigid_quotient_status;
    result.rigid_mode_reproduction_tolerance =
        pencil.rigid_reproduction_tolerance;
    result.maximum_rigid_mode_reproduction_residual =
        pencil.maximum_rigid_reproduction_residual;
    result.retained_support_physical_volume =
        pencil.retained_measure;
    result.generated_boundary_physical_measure =
        pencil.boundary_measure;
    if (pencil.terminal_dimension == 0u) {
        result.generalized_bound.dimension = 0u;
        result.generalized_bound.denominator_converged = true;
        result.generalized_bound.quotient_converged = true;
        return result;
    }
    const auto expected_positive_rank =
        pencil.terminal_dimension -
        pencil.structural_rigid_mode_count;
    const auto proven_anchors =
        provenIndependentRowCoordinates(
            pencil.explicit_nullspace,
            pencil.terminal_dimension,
            pencil.structural_rigid_mode_count);
    if (!proven_anchors.has_value() ||
        proven_anchors->size() !=
            pencil.structural_rigid_mode_count) {
        reject(
            "patch pencil quotient coordinate anchors lack an exact "
            "rank proof");
    }
    const auto& anchors = *proven_anchors;
    std::vector<bool> eliminated(
        pencil.terminal_dimension, false);
    for (const auto anchor : anchors) {
        if (anchor >= eliminated.size() ||
            eliminated[anchor]) {
            reject(
                "patch pencil quotient has an invalid coordinate anchor");
        }
        eliminated[anchor] = true;
    }
    std::vector<std::size_t> retained;
    retained.reserve(expected_positive_rank);
    for (std::size_t coordinate = 0u;
         coordinate < eliminated.size();
         ++coordinate) {
        if (!eliminated[coordinate]) {
            retained.push_back(coordinate);
        }
    }
    if (retained.size() !=
        expected_positive_rank) {
        reject(
            "patch pencil quotient retained-coordinate count is "
            "inconsistent");
    }
    std::vector<Real> retained_numerator(
        expected_positive_rank *
            expected_positive_rank,
        Real{0.0});
    std::vector<Real> retained_denominator(
        expected_positive_rank *
            expected_positive_rank,
        Real{0.0});
    for (std::size_t row = 0u;
         row < retained.size();
         ++row) {
        for (std::size_t column = 0u;
             column < retained.size();
             ++column) {
            const auto retained_entry =
                row * retained.size() +
                column;
            const auto source_entry =
                retained[row] *
                    pencil.terminal_dimension +
                retained[column];
            retained_numerator[retained_entry] =
                pencil.numerator[source_entry];
            retained_denominator[retained_entry] =
                pencil.denominator[
                    source_entry];
        }
    }

    math::DenseExactDyadicSpdGeneralizedUpperBound
        exact_bound;
    if (expected_positive_rank == 0u) {
        exact_bound.applied = true;
        exact_bound.denominator_positive_definite_proven = true;
        exact_bound.numerator_positive_semidefinite_proven = true;
        exact_bound.upper_inequality_proven = true;
    } else {
        exact_bound =
            math::dense_exact_dyadic_spd_generalized_upper_bound(
                retained_numerator,
                retained_denominator,
                expected_positive_rank,
                "generated-boundary aggregate viscous trace exact "
                "coordinate quotient");
    }
    if (!exact_bound.applied ||
        !exact_bound.denominator_positive_definite_proven ||
        !exact_bound.numerator_positive_semidefinite_proven ||
        !exact_bound.upper_inequality_proven ||
        exact_bound.dimension != expected_positive_rank ||
        exact_bound.denominator_rank != expected_positive_rank ||
        !std::isfinite(
            exact_bound.directly_proven_upper_bound) ||
        exact_bound.directly_proven_upper_bound < Real{0.0}) {
        reject(
            "patch exact dyadic quotient proof is incomplete");
    }

    math::DensePsdGeneralizedEigenvalueBound bound;
    bool floating_diagnostics_available = false;
    if (expected_positive_rank != 0u) {
        try {
            bound =
                math::dense_psd_generalized_eigenvalue_bound(
                    retained_numerator,
                    retained_denominator,
                    expected_positive_rank,
                    "generated-boundary aggregate viscous trace "
                    "coordinate quotient diagnostics");
            floating_diagnostics_available =
                bound.positive_rank == expected_positive_rank &&
                bound.nullity == 0u &&
                bound.denominator_converged &&
                bound.quotient_converged;
        } catch (const FEException&) {
            // The exact dyadic proof above is authoritative.  The floating
            // eigensolver is retained only for optional spectral diagnostics.
        }
    }
    if (!floating_diagnostics_available) {
        bound = {};
        bound.denominator_scale =
            math::dense_matrix_max_abs(retained_denominator);
        bound.numerator_scale =
            math::dense_matrix_max_abs(retained_numerator);
        bound.positive_rank = expected_positive_rank;
        bound.largest_quotient_eigenvalue =
            exact_bound.directly_proven_upper_bound;
        bound.conservative_upper_bound =
            exact_bound.directly_proven_upper_bound;
    }
    bound.conservative_upper_bound =
        std::max({
            bound.conservative_upper_bound,
            bound.largest_quotient_eigenvalue,
            exact_bound.directly_proven_upper_bound});
    bound.dimension = pencil.terminal_dimension;
    bound.positive_rank = expected_positive_rank;
    bound.nullity = pencil.structural_rigid_mode_count;
    bound.exact_dyadic = exact_bound;
    bound.explicit_nullspace.applied =
        pencil.structural_rigid_mode_count != 0u;
    bound.explicit_nullspace.supplied_nullity =
        pencil.structural_rigid_mode_count;
    bound.explicit_nullspace.reduced_dimension =
        expected_positive_rank;
    bound.explicit_nullspace.original_denominator_scale =
        math::dense_matrix_max_abs(pencil.denominator);
    bound.explicit_nullspace.original_numerator_scale =
        math::dense_matrix_max_abs(pencil.numerator);
    bound.explicit_nullspace.exact_binary64_actions_proven = true;
    bound.explicit_nullspace.exact_binary64_anchor_rank_proven = true;
    bound.explicit_nullspace.eliminated_coordinates = anchors;
    result.generalized_bound = std::move(bound);

    if (!std::isfinite(
            result.generalized_bound
                .conservative_upper_bound) ||
        result.generalized_bound
                .conservative_upper_bound <
            Real{0.0}) {
        reject("patch generalized bound is invalid");
    }
    return result;
}

[[nodiscard]] std::uint64_t certificateDigest(
    const GeneratedBoundaryAggregateTraceCertificate& certificate)
{
    const auto mix_real = [](std::uint64_t& digest, Real value) {
        digestMix(
            digest,
            std::bit_cast<std::uint64_t>(value));
    };
    const auto mix_explicit =
        [&](std::uint64_t& digest,
            const math::DenseExplicitNullspaceDiagnostics& value) {
            digestMix(digest, UINT64_C(0x4558504c49434954));
            digestMix(digest, value.applied ? 1u : 0u);
            digestMix(digest, value.supplied_nullity);
            digestMix(digest, value.reduced_dimension);
            mix_real(digest, value.original_denominator_scale);
            mix_real(digest, value.original_numerator_scale);
            mix_real(digest, value.basis_rank_tolerance);
            mix_real(digest, value.smallest_selected_row_residual);
            mix_real(digest, value.denominator_action_tolerance);
            mix_real(digest, value.maximum_denominator_action);
            mix_real(digest, value.numerator_action_tolerance);
            mix_real(digest, value.maximum_numerator_action);
            digestMix(
                digest,
                value.exact_binary64_actions_proven ? 1u : 0u);
            digestMix(
                digest,
                value.exact_binary64_anchor_rank_proven ? 1u : 0u);
            digestMix(
                digest,
                value.eliminated_coordinates.size());
            for (const auto coordinate :
                 value.eliminated_coordinates) {
                digestMix(digest, coordinate);
            }
        };
    const auto mix_exact =
        [&](std::uint64_t& digest,
            const math::DenseExactDyadicSpdGeneralizedUpperBound& value) {
            digestMix(digest, UINT64_C(0x4558445941444943));
            digestMix(digest, value.applied ? 1u : 0u);
            digestMix(
                digest,
                value.denominator_positive_definite_proven ? 1u : 0u);
            digestMix(
                digest,
                value.numerator_positive_semidefinite_proven ? 1u : 0u);
            digestMix(
                digest,
                value.upper_inequality_proven ? 1u : 0u);
            digestMix(digest, value.dimension);
            digestMix(digest, value.denominator_rank);
            digestMix(digest, value.numerator_rank);
            digestMix(
                digest,
                value.failing_lower_bound_proven ? 1u : 0u);
            mix_real(
                digest,
                value.largest_failing_lower_bound);
            mix_real(
                digest,
                value.directly_proven_upper_bound);
            digestMix(digest, value.psd_oracle_calls);
            digestMix(digest, value.binary64_search_steps);
            digestMix(digest, value.exact_update_count);
            digestMix(digest, value.maximum_integer_bits);
        };
    const auto mix_bound =
        [&](std::uint64_t& digest,
            const math::DensePsdGeneralizedEigenvalueBound& value) {
            digestMix(digest, UINT64_C(0x505344424f554e44));
            digestMix(digest, value.dimension);
            digestMix(digest, value.positive_rank);
            digestMix(digest, value.nullity);
            mix_real(digest, value.denominator_scale);
            mix_real(digest, value.numerator_scale);
            mix_real(
                digest,
                value.denominator_eigenvalue_tolerance);
            mix_real(
                digest,
                value.nullspace_compatibility_tolerance);
            mix_real(
                digest,
                value.maximum_nullspace_residual);
            mix_real(
                digest,
                value.smallest_positive_denominator_eigenvalue);
            mix_real(
                digest,
                value.largest_denominator_eigenvalue);
            mix_real(
                digest,
                value.smallest_quotient_eigenvalue);
            mix_real(
                digest,
                value.largest_quotient_eigenvalue);
            mix_real(
                digest,
                value.conservative_upper_bound);
            mix_real(
                digest,
                value.quotient_maximum_off_diagonal);
            mix_real(digest, value.quotient_tolerance);
            digestMix(digest, value.denominator_sweeps);
            digestMix(digest, value.quotient_sweeps);
            digestMix(
                digest,
                value.denominator_converged ? 1u : 0u);
            digestMix(
                digest,
                value.quotient_converged ? 1u : 0u);
            mix_explicit(digest, value.explicit_nullspace);
            mix_exact(digest, value.exact_dyadic);
        };

    std::uint64_t digest = UINT64_C(1469598103934665603);
    digestMix(digest, UINT64_C(0x5452414345434552));
    digestMix(digest, kCertificateDigestVersion);
    digestMix(
        digest,
        static_cast<std::uint64_t>(
            certificate.field));
    digestMix(
        digest,
        static_cast<std::uint64_t>(
            certificate.physical_boundary_marker));
    digestMix(
        digest,
        static_cast<std::uint64_t>(
            certificate.volume_interface_marker));
    digestMix(
        digest,
        static_cast<std::uint64_t>(
            certificate.generated_active_boundary_marker));
    digestMix(
        digest,
        static_cast<std::uint64_t>(
            certificate.communicator_size));
    mix_real(digest, certificate.dynamic_viscosity);
    digestMix(
        digest,
        certificate.aggregation_content_digest);
    digestMix(digest, certificate.active_cell_count);
    digestMix(
        digest,
        certificate.generated_boundary_rule_count);
    digestMix(digest, certificate.certified_patch_count);
    digestMix(digest, certificate.maximum_support_overlap);
    digestMix(
        digest,
        certificate.maximum_terminal_tangent_dimension);
    mix_real(
        digest,
        certificate.retained_active_physical_volume);
    mix_real(
        digest,
        certificate.generated_boundary_physical_measure);
    mix_real(
        digest,
        certificate.maximum_patch_conservative_upper_bound);
    mix_real(
        digest,
        certificate.global_conservative_upper_bound);
    digestMix(
        digest,
        certificate.patches.size());
    for (const auto& patch : certificate.patches) {
        digestMix(digest, UINT64_C(0x5041544348424547));
        if (patch.synthetic_full_active_patch) {
            digestMix(
                digest,
                std::numeric_limits<std::uint64_t>::max());
        } else {
            digestMix(
                digest,
                static_cast<std::uint64_t>(
                    patch.canonical_patch_index));
        }
        digestMix(
            digest,
            patch.synthetic_full_active_patch ? 1u : 0u);
        digestMix(
            digest,
            static_cast<std::uint64_t>(
                patch.root_cell_gid));
        digestMix(
            digest,
            patch.support_cell_gids.size());
        for (const auto cell_gid :
             patch.support_cell_gids) {
            digestMix(
                digest,
                static_cast<std::uint64_t>(cell_gid));
        }
        digestMix(
            digest,
            patch.boundary_rule_stable_ids.size());
        for (const auto stable_id :
             patch.boundary_rule_stable_ids) {
            digestMix(digest, stable_id);
        }
        digestMix(digest, patch.raw_support_dof_count);
        digestMix(
            digest,
            patch.terminal_tangent_dof_count);
        digestMix(
            digest,
            patch.rigid_mode_candidate_count);
        digestMix(
            digest,
            patch.structural_rigid_mode_count);
        digestMix(
            digest,
            patch.rigid_mode_constraint_rank);
        digestMix(
            digest,
            static_cast<std::uint64_t>(
                patch.rigid_mode_quotient_status));
        mix_real(
            digest,
            patch.rigid_mode_reproduction_tolerance);
        mix_real(
            digest,
            patch.maximum_rigid_mode_reproduction_residual);
        digestMix(
            digest,
            patch.maximum_cell_support_overlap);
        mix_real(
            digest,
            patch.retained_support_physical_volume);
        mix_real(
            digest,
            patch.generated_boundary_physical_measure);
        mix_bound(digest, patch.generalized_bound);
        digestMix(digest, UINT64_C(0x5041544348454e44));
    }
    digestMix(digest, UINT64_C(0x5452414345454e44));
    return digest == 0u ? UINT64_C(1) : digest;
}

} // namespace

GeneratedBoundaryAggregateTraceCertificate
certifyGeneratedBoundaryAggregateTrace(
    const systems::FESystem& system,
    const GeneratedBoundaryAggregateTraceCertificationOptions& options)
{
    const auto collective =
        collectiveContext(system);

    SelectedState selected;
    std::exception_ptr local_exception;
    try {
        selected =
            selectAndValidateState(
                system, options, collective);
    } catch (...) {
        local_exception = std::current_exception();
    }
    coordinateFailure(
        collective,
        local_exception,
        "state preflight");
    requireCollectiveAgreement(
        collective,
        requestSignature(selected, options),
        "certification request signature");

    LocalData local_data;
    local_exception = nullptr;
    try {
        local_data =
            buildLocalData(
                system,
                selected,
                options,
                collective);
    } catch (...) {
        local_exception = std::current_exception();
    }
    coordinateFailure(
        collective,
        local_exception,
        "provider and owner extraction");

    WireWriter writer;
    local_exception = nullptr;
    try {
        encodePayload(
            writer,
            local_data.cells,
            local_data.boundaries,
            local_data.rows);
    } catch (...) {
        local_exception = std::current_exception();
    }
    coordinateFailure(
        collective,
        local_exception,
        "canonical payload serialization");

    GatheredWords gathered;
    local_exception = nullptr;
    try {
        gathered =
            allGatherWords(
                collective, writer.words());
    } catch (...) {
        local_exception = std::current_exception();
    }
    coordinateFailure(
        collective,
        local_exception,
        "canonical payload gather");

    CanonicalData canonical;
    local_exception = nullptr;
    try {
        for (int rank = 0; rank < collective.size; ++rank) {
            const auto index =
                static_cast<std::size_t>(rank);
            const auto displacement =
                static_cast<std::size_t>(
                    gathered.displacements[index]);
            const auto count =
                static_cast<std::size_t>(
                    gathered.counts[index]);
            if (displacement > gathered.words.size() ||
                count >
                    gathered.words.size() - displacement) {
                reject("gathered rank payload range is invalid");
            }
            decodePayload(
                std::span<const std::int64_t>(
                    gathered.words.data() + displacement,
                    count),
                selected.dimension,
                canonical);
        }
        validateCanonicalData(
            canonical, selected);
    } catch (...) {
        local_exception = std::current_exception();
    }
    coordinateFailure(
        collective,
        local_exception,
        "canonical payload validation");

    GeneratedBoundaryAggregateTraceCertificate result;
    local_exception = nullptr;
    try {
        const auto work_patches =
            assignBoundaryRulesToPatches(
                canonical, selected);
        result.field = options.field;
        result.physical_boundary_marker =
            options.physical_boundary_marker;
        result.volume_interface_marker =
            options.volume_interface_marker;
        result.generated_active_boundary_marker =
            options.generated_active_boundary_marker;
        result.dynamic_viscosity =
            options.dynamic_viscosity;
        result.communicator_size =
            collective.size;
        result.aggregation_content_digest =
            selected.report->canonical_content_digest;
        result.cut_context_content_revision =
            selected.report->revision
                .cut_context_content_revision;
        result.free_surface_snapshot_revision =
            selected.report->revision
                .free_surface_snapshot_revision;
        result.source_value_revision =
            selected.report->revision
                .source_value_revision;
        result.affine_constraint_layout_revision =
            selected.report->revision
                .affine_constraint_layout_revision;
        result.active_cell_count =
            selected.report->active_cells.size();
        result.generated_boundary_rule_count =
            canonical.boundaries.size();
        result.certified_patch_count =
            work_patches.size();
        result.patches.reserve(work_patches.size());

        Real active_volume = 0.0;
        for (const auto& cell :
             selected.report->active_cells) {
            active_volume =
                outwardAddNonnegative(
                    active_volume,
                    cell.retained_physical_volume,
                    "retained active volume");
        }
        result.retained_active_physical_volume =
            active_volume;

        Real boundary_measure = 0.0;
        for (const auto& [stable_id, boundary] :
             canonical.boundaries) {
            (void)stable_id;
            boundary_measure =
                outwardAddNonnegative(
                    boundary_measure,
                    boundary.physical_measure,
                    "generated boundary measure");
        }
        result.generated_boundary_physical_measure =
            boundary_measure;

        for (const auto& patch : work_patches) {
            result.patches.push_back(
                certifyPatch(
                    patch,
                    canonical,
                    selected.dimension,
                    options.maximum_reduced_dimension));
            result.maximum_terminal_tangent_dimension =
                std::max(
                    result.maximum_terminal_tangent_dimension,
                    result.patches.back()
                        .terminal_tangent_dof_count);
            result.maximum_patch_conservative_upper_bound =
                std::max(
                    result.maximum_patch_conservative_upper_bound,
                    result.patches.back()
                        .generalized_bound
                        .conservative_upper_bound);
        }

        std::map<GlobalIndex, std::size_t> overlap;
        std::map<GlobalIndex, Real> weighted_bound;
        for (const auto& patch : result.patches) {
            for (const auto cell_gid :
                 patch.support_cell_gids) {
                ++overlap[cell_gid];
                auto& sum = weighted_bound[cell_gid];
                sum =
                    outwardAddNonnegative(
                        sum,
                        patch.generalized_bound
                            .conservative_upper_bound,
                        "cell overlap-weighted trace bound");
            }
        }
        Real global_bound = 0.0;
        for (const auto& [cell_gid, value] :
             weighted_bound) {
            (void)cell_gid;
            global_bound =
                std::max(global_bound, value);
        }
        result.global_conservative_upper_bound =
            global_bound;
        for (auto& patch : result.patches) {
            for (const auto cell_gid :
                 patch.support_cell_gids) {
                const auto count =
                    overlap.at(cell_gid);
                patch.maximum_cell_support_overlap =
                    std::max(
                        patch.maximum_cell_support_overlap,
                        count);
                result.maximum_support_overlap =
                    std::max(
                        result.maximum_support_overlap,
                        count);
            }
        }
        result.canonical_certificate_digest =
            certificateDigest(result);
    } catch (...) {
        local_exception = std::current_exception();
    }
    coordinateFailure(
        collective,
        local_exception,
        "patch pencil certification");
    requireCollectiveAgreement(
        collective,
        result.canonical_certificate_digest,
        "canonical certificate digest");
    return result;
}

} // namespace svmp::FE::analysis
