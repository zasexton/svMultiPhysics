/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Backends/MUMPS/MumpsDistributedSolver.h"

#include "Core/FEException.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <sstream>
#include <type_traits>

#if defined(FE_HAS_MUMPS) && FE_HAS_MUMPS
#include <dmumps_c.h>
#endif

namespace svmp {
namespace FE {
namespace backends {

bool mumpsAvailable() noexcept
{
#if defined(FE_HAS_MUMPS) && FE_HAS_MUMPS
    return true;
#else
    return false;
#endif
}

#if defined(FE_HAS_MUMPS) && FE_HAS_MUMPS

static_assert(std::is_same_v<Real, double>, "MumpsDistributedSolver uses double precision MUMPS");

namespace {

constexpr int kRoot = 0;
constexpr int kMaxMemoryRetries = 4;

[[nodiscard]] double wallSeconds() noexcept
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

[[nodiscard]] bool mpiUsable() noexcept
{
    int initialized = 0;
    int finalized = 0;
    MPI_Initialized(&initialized);
    MPI_Finalized(&finalized);
    return initialized != 0 && finalized == 0;
}

// INFOG(i) / ICNTL(i) / INFO(i) with the 1-based indices of the MUMPS guide.
[[nodiscard]] MUMPS_INT& icntl(DMUMPS_STRUC_C& id, int i) { return id.icntl[i - 1]; }
[[nodiscard]] MUMPS_INT infog(const DMUMPS_STRUC_C& id, int i) { return id.infog[i - 1]; }
[[nodiscard]] MUMPS_INT info(const DMUMPS_STRUC_C& id, int i) { return id.info[i - 1]; }

// Sizes that MUMPS reports as negative numbers of millions.
[[nodiscard]] std::int64_t millionsIfNegative(MUMPS_INT value) noexcept
{
    return value < 0 ? -static_cast<std::int64_t>(value) * 1000000 : static_cast<std::int64_t>(value);
}

} // namespace

struct MumpsDistributedSolver::Impl {
    MPI_Comm comm{MPI_COMM_NULL};
    int rank{0};
    int size{1};
    Symmetry symmetry{Symmetry::Unsymmetric};
    Ordering ordering{Ordering::Metis};
    DMUMPS_STRUC_C id{};
    bool initialized{false};
    bool analyzed{false};
    bool factorized{false};
    std::int64_t n{0};
    std::vector<MUMPS_INT> irn{};
    std::vector<MUMPS_INT> jcn{};
    std::vector<double> a{};
    std::vector<double> rhs{};
    Statistics stats{};
    std::string error{};

    void call(int job)
    {
        id.job = job;
        dmumps_c(&id);
    }

    [[nodiscard]] std::string describe(const char* phase) const
    {
        std::ostringstream oss;
        oss << "MUMPS " << phase << " failed: INFOG(1)=" << infog(id, 1) << " INFOG(2)=" << infog(id, 2);
        return oss.str();
    }
};

MumpsDistributedSolver::MumpsDistributedSolver(MPI_Comm comm, Symmetry symmetry, Ordering ordering)
    : impl_(std::make_unique<Impl>())
{
    auto& s = *impl_;
    FE_THROW_IF(!mpiUsable(), FEException, "MumpsDistributedSolver: MPI is not initialized");
    MPI_Comm_dup(comm, &s.comm);
    MPI_Comm_rank(s.comm, &s.rank);
    MPI_Comm_size(s.comm, &s.size);
    s.symmetry = symmetry;
    s.ordering = ordering;
    s.id.comm_fortran = static_cast<MUMPS_INT>(MPI_Comm_c2f(s.comm));
    s.id.par = 1;  // the host takes part in the factorization
    s.id.sym = static_cast<MUMPS_INT>(symmetry);
    s.call(-1);
    FE_THROW_IF(infog(s.id, 1) < 0, FEException, s.describe("initialization"));
    s.initialized = true;
    icntl(s.id, 1) = 6;   // error messages
    icntl(s.id, 2) = -1;  // no diagnostic messages
    icntl(s.id, 3) = -1;  // no global information
    icntl(s.id, 4) = 1;   // errors only
    icntl(s.id, 5) = 0;   // assembled format
    icntl(s.id, 18) = 3;  // distributed matrix input
    icntl(s.id, 20) = 0;  // dense centralized right-hand side
    icntl(s.id, 21) = 0;  // centralized solution
    icntl(s.id, 28) = 1;  // sequential analysis
    icntl(s.id, 7) = static_cast<MUMPS_INT>(ordering);
    s.stats.memory_relaxation_percent = icntl(s.id, 14);
}

MumpsDistributedSolver::~MumpsDistributedSolver()
{
    if (!impl_) {
        return;
    }
    auto& s = *impl_;
    // Termination is collective; instances must be destroyed on every rank of
    // the communicator together (after MPI_Finalize the memory is left to the OS).
    if (s.initialized && mpiUsable()) {
        s.call(-2);
    }
    if (s.comm != MPI_COMM_NULL && mpiUsable()) {
        MPI_Comm_free(&s.comm);
    }
}

bool MumpsDistributedSolver::factorize(GlobalIndex n,
                                       std::span<const GlobalIndex> rows,
                                       std::span<const GlobalIndex> cols,
                                       std::span<const Real> values)
{
    auto& s = *impl_;
    FE_THROW_IF(rows.size() != cols.size() || rows.size() != values.size(), InvalidArgumentException,
                "MumpsDistributedSolver::factorize: triplet sizes differ");
    FE_THROW_IF(n <= 0 || n > static_cast<GlobalIndex>(std::numeric_limits<MUMPS_INT>::max()),
                InvalidArgumentException, "MumpsDistributedSolver::factorize: size out of the MUMPS_INT range");
    s.error.clear();

    // Pattern of this rank (1-based) and the collective change decision.
    std::vector<MUMPS_INT> irn(rows.size());
    std::vector<MUMPS_INT> jcn(cols.size());
    int local_bad = 0;
    for (std::size_t k = 0; k < rows.size(); ++k) {
        if (rows[k] < 0 || rows[k] >= n || cols[k] < 0 || cols[k] >= n) {
            local_bad = 1;
            break;
        }
        irn[k] = static_cast<MUMPS_INT>(rows[k] + 1);
        jcn[k] = static_cast<MUMPS_INT>(cols[k] + 1);
    }
    int local_changed = (!s.analyzed || n != s.n || irn != s.irn || jcn != s.jcn) ? 1 : 0;
    int flags[2] = {local_bad, local_changed};
    int global_flags[2] = {0, 0};
    MPI_Allreduce(flags, global_flags, 2, MPI_INT, MPI_MAX, s.comm);
    if (global_flags[0] != 0) {
        s.error = "MumpsDistributedSolver::factorize: entry index out of range";
        s.factorized = false;
        return false;
    }
    long long local_entries = static_cast<long long>(rows.size());
    long long total_entries = 0;
    MPI_Allreduce(&local_entries, &total_entries, 1, MPI_LONG_LONG, MPI_SUM, s.comm);

    s.a.assign(values.begin(), values.end());
    s.id.a_loc = s.a.empty() ? nullptr : s.a.data();
    if (global_flags[1] != 0) {
        s.irn = std::move(irn);
        s.jcn = std::move(jcn);
        s.n = n;
        s.id.n = static_cast<MUMPS_INT>(n);
        s.id.nnz_loc = static_cast<MUMPS_INT8>(s.irn.size());
        s.id.irn_loc = s.irn.empty() ? nullptr : s.irn.data();
        s.id.jcn_loc = s.jcn.empty() ? nullptr : s.jcn.data();
        const double t0 = wallSeconds();
        s.call(1);
        s.stats.analyze_seconds += wallSeconds() - t0;
        ++s.stats.analyses;
        if (infog(s.id, 1) < 0) {
            s.error = s.describe("analysis");
            s.analyzed = false;
            s.factorized = false;
            return false;
        }
        s.analyzed = true;
    }

    const double t0 = wallSeconds();
    for (int attempt = 0;; ++attempt) {
        s.call(2);
        const int status = infog(s.id, 1);
        // Workspace too small (-8, -9, -14, -15, -17, -20): relax and retry.
        const bool workspace = status == -8 || status == -9 || status == -14 || status == -15 ||
                               status == -17 || status == -20;
        if (status >= 0 || !workspace || attempt >= kMaxMemoryRetries) {
            break;
        }
        icntl(s.id, 14) = std::max<MUMPS_INT>(2 * icntl(s.id, 14), 40);
    }
    s.stats.factor_seconds += wallSeconds() - t0;
    s.stats.memory_relaxation_percent = icntl(s.id, 14);
    if (infog(s.id, 1) < 0) {
        s.error = s.describe("factorization");
        s.analyzed = false;  // analyze again next time
        s.factorized = false;
        return false;
    }
    ++s.stats.factorizations;
    s.factorized = true;
    s.stats.n = n;
    s.stats.entries = total_entries;
    s.stats.factor_entries = millionsIfNegative(infog(s.id, 29));
    s.stats.peak_memory_mb_max_rank = static_cast<double>(infog(s.id, 21));
    s.stats.peak_memory_mb_total = static_cast<double>(infog(s.id, 22));
    return true;
}

bool MumpsDistributedSolver::solveReplicated(std::span<const Real> rhs, std::vector<Real>& solution)
{
    auto& s = *impl_;
    s.error.clear();
    int local_ok = s.factorized ? 1 : 0;
    if (s.rank == kRoot && static_cast<std::int64_t>(rhs.size()) != s.n) {
        local_ok = 0;
    }
    int ok = 0;
    MPI_Allreduce(&local_ok, &ok, 1, MPI_INT, MPI_MIN, s.comm);
    if (ok == 0) {
        s.error = "MumpsDistributedSolver::solveReplicated: no factorization or wrong right-hand side size";
        return false;
    }
    if (s.rank == kRoot) {
        s.rhs.assign(rhs.begin(), rhs.end());
        s.id.rhs = s.rhs.data();
        s.id.nrhs = 1;
        s.id.lrhs = static_cast<MUMPS_INT>(s.n);
    }
    const double t0 = wallSeconds();
    s.call(3);
    if (infog(s.id, 1) < 0) {
        s.stats.solve_seconds += wallSeconds() - t0;
        s.error = s.describe("solve");
        return false;
    }
    solution.resize(static_cast<std::size_t>(s.n));
    if (s.rank == kRoot) {
        std::copy(s.rhs.begin(), s.rhs.end(), solution.begin());
    }
    MPI_Bcast(solution.data(), static_cast<int>(s.n), MPI_DOUBLE, kRoot, s.comm);
    s.stats.solve_seconds += wallSeconds() - t0;
    ++s.stats.solves;
    return true;
}

const MumpsDistributedSolver::Statistics& MumpsDistributedSolver::statistics() const noexcept
{
    return impl_->stats;
}

const std::string& MumpsDistributedSolver::lastError() const noexcept
{
    return impl_->error;
}

std::size_t MumpsDistributedSolver::localFactorBytes() const noexcept
{
    if (!impl_->factorized) {
        return 0u;
    }
    // INFO(16): memory (MB) of this rank's factorization.
    const auto mb = info(impl_->id, 16);
    return mb > 0 ? static_cast<std::size_t>(mb) * 1000000u : 0u;
}

bool MumpsDistributedSolver::factorized() const noexcept
{
    return impl_->factorized;
}

#else  // !FE_HAS_MUMPS

struct MumpsDistributedSolver::Impl {
    Statistics stats{};
    std::string error{"FE was built without FE_ENABLE_MUMPS"};
};

#if defined(FE_HAS_MPI) && FE_HAS_MPI
MumpsDistributedSolver::MumpsDistributedSolver(MPI_Comm, Symmetry, Ordering)
    : impl_(std::make_unique<Impl>())
{
    FE_THROW(NotImplementedException, "MumpsDistributedSolver: FE was built without FE_ENABLE_MUMPS");
}
#endif

MumpsDistributedSolver::~MumpsDistributedSolver() = default;

bool MumpsDistributedSolver::factorize(GlobalIndex, std::span<const GlobalIndex>, std::span<const GlobalIndex>,
                                       std::span<const Real>)
{
    return false;
}

bool MumpsDistributedSolver::solveReplicated(std::span<const Real>, std::vector<Real>&)
{
    return false;
}

const MumpsDistributedSolver::Statistics& MumpsDistributedSolver::statistics() const noexcept
{
    return impl_->stats;
}

const std::string& MumpsDistributedSolver::lastError() const noexcept
{
    return impl_->error;
}

std::size_t MumpsDistributedSolver::localFactorBytes() const noexcept
{
    return 0u;
}

bool MumpsDistributedSolver::factorized() const noexcept
{
    return false;
}

#endif

} // namespace backends
} // namespace FE
} // namespace svmp
