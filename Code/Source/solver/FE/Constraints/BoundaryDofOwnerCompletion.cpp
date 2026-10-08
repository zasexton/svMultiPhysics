/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Constraints/BoundaryDofOwnerCompletion.h"

#include "Core/FEException.h"
#include "Core/HaloDiagnostics.h"
#include "Core/Logger.h"
#include "Systems/FESystem.h"
#include "Systems/SystemsExceptions.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>

#if FE_HAS_MPI
#  include <mpi.h>
#endif

namespace svmp {
namespace FE {
namespace constraints {

namespace {

#if FE_HAS_MPI
// Reports, once per constraint (context) and process, how many owned DOFs had
// to come from other ranks, and adds the communicator total to the run
// summary.  Collective.
void reportCompletion(MPI_Comm comm, int rank, int size, std::string_view context,
                      std::size_t local_added)
{
    long long mine = static_cast<long long>(local_added);
    long long total = 0;
    MPI_Allreduce(&mine, &total, 1, MPI_LONG_LONG, MPI_SUM, comm);
    diagnostics::recordBoundaryDofOwnerCompletion(static_cast<std::uint64_t>(total));
    if (total == 0) {
        return;
    }
    std::vector<long long> per_rank(static_cast<std::size_t>(rank == 0 ? size : 0), 0);
    MPI_Gather(&mine, 1, MPI_LONG_LONG, per_rank.data(), 1, MPI_LONG_LONG, 0, comm);
    if (rank != 0) {
        return;
    }
    static std::mutex mutex;
    static std::set<std::string> reported;
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (!reported.insert(std::string(context)).second) {
            return;
        }
    }
    std::ostringstream oss;
    oss << "BoundaryDofOwnerCompletion: diagnostic=boundary_dof_owner_completion "
        << context << " added_dofs_total=" << total << " added_dofs_per_rank=";
    for (int r = 0; r < size; ++r) {
        oss << (r > 0 ? "," : "") << per_rank[static_cast<std::size_t>(r)];
    }
    oss << " (owned boundary DOFs whose owner holds none of their marker faces as an"
           " owned cell; first occurrence for this constraint)";
    FE_LOG_WARNING(oss.str());
}
#endif

} // namespace

std::size_t completeOwnedBoundaryDofs(const systems::FESystem& system,
                                      std::vector<GlobalIndex>& dofs,
                                      std::vector<Real>& payload,
                                      std::size_t stride,
                                      std::string_view context)
{
    FE_THROW_IF(payload.size() != dofs.size() * stride, InvalidArgumentException,
                "completeOwnedBoundaryDofs: payload size does not match DOFs x stride");
#if FE_HAS_MPI
    static_assert(std::is_same_v<Real, double>,
                  "completeOwnedBoundaryDofs sends Real payload as MPI_DOUBLE");
    int initialized = 0;
    int finalized = 0;
    MPI_Initialized(&initialized);
    MPI_Finalized(&finalized);
    if (initialized == 0 || finalized != 0) {
        return 0;
    }
    const MPI_Comm comm = system.dofHandler().mpiComm();
    if (comm == MPI_COMM_NULL) {
        return 0;
    }
    int size = 1;
    int rank = 0;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
    if (size <= 1) {
        return 0;
    }

    const auto& owned = system.dofHandler().getPartition().locallyOwned();
    const auto& dof_map = system.dofHandler().getDofMap();

    // Collected DOFs this rank does not own, grouped by owner.
    std::vector<std::vector<std::int64_t>> send_dofs(static_cast<std::size_t>(size));
    std::vector<std::vector<double>> send_payload(static_cast<std::size_t>(size));
    for (std::size_t i = 0; i < dofs.size(); ++i) {
        const auto dof = dofs[i];
        if (owned.contains(dof)) {
            continue;
        }
        const int owner = dof_map.getDofOwner(dof);
        FE_THROW_IF(owner < 0 || owner >= size || owner == rank, systems::InvalidStateException,
                    "completeOwnedBoundaryDofs: boundary DOF " + std::to_string(dof) +
                        " has no valid remote owner (owner=" + std::to_string(owner) + ")");
        send_dofs[static_cast<std::size_t>(owner)].push_back(static_cast<std::int64_t>(dof));
        auto& out = send_payload[static_cast<std::size_t>(owner)];
        out.insert(out.end(),
                   payload.begin() + static_cast<std::ptrdiff_t>(i * stride),
                   payload.begin() + static_cast<std::ptrdiff_t>((i + 1) * stride));
    }

    std::vector<int> send_counts(static_cast<std::size_t>(size), 0);
    for (int r = 0; r < size; ++r) {
        const auto n = send_dofs[static_cast<std::size_t>(r)].size();
        FE_THROW_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max() /
                                                 static_cast<int>(std::max<std::size_t>(stride, 1))),
                    systems::InvalidStateException,
                    "completeOwnedBoundaryDofs: message too large");
        send_counts[static_cast<std::size_t>(r)] = static_cast<int>(n);
    }
    std::vector<int> recv_counts(static_cast<std::size_t>(size), 0);
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm);

    std::vector<int> send_displs(static_cast<std::size_t>(size), 0);
    std::vector<int> recv_displs(static_cast<std::size_t>(size), 0);
    for (int r = 1; r < size; ++r) {
        send_displs[static_cast<std::size_t>(r)] =
            send_displs[static_cast<std::size_t>(r - 1)] + send_counts[static_cast<std::size_t>(r - 1)];
        recv_displs[static_cast<std::size_t>(r)] =
            recv_displs[static_cast<std::size_t>(r - 1)] + recv_counts[static_cast<std::size_t>(r - 1)];
    }
    const int total_send = send_displs.back() + send_counts.back();
    const int total_recv = recv_displs.back() + recv_counts.back();

    std::vector<std::int64_t> flat_send_dofs;
    std::vector<double> flat_send_payload;
    flat_send_dofs.reserve(static_cast<std::size_t>(total_send));
    flat_send_payload.reserve(static_cast<std::size_t>(total_send) * stride);
    for (int r = 0; r < size; ++r) {
        const auto& d = send_dofs[static_cast<std::size_t>(r)];
        const auto& p = send_payload[static_cast<std::size_t>(r)];
        flat_send_dofs.insert(flat_send_dofs.end(), d.begin(), d.end());
        flat_send_payload.insert(flat_send_payload.end(), p.begin(), p.end());
    }
    std::vector<std::int64_t> recv_dofs(static_cast<std::size_t>(total_recv));
    MPI_Alltoallv(flat_send_dofs.data(), send_counts.data(), send_displs.data(), MPI_INT64_T,
                  recv_dofs.data(), recv_counts.data(), recv_displs.data(), MPI_INT64_T, comm);

    std::vector<double> recv_payload(static_cast<std::size_t>(total_recv) * stride);
    if (stride > 0) {
        std::vector<int> send_counts_p(send_counts);
        std::vector<int> recv_counts_p(recv_counts);
        std::vector<int> send_displs_p(send_displs);
        std::vector<int> recv_displs_p(recv_displs);
        const int s = static_cast<int>(stride);
        for (int r = 0; r < size; ++r) {
            send_counts_p[static_cast<std::size_t>(r)] *= s;
            recv_counts_p[static_cast<std::size_t>(r)] *= s;
            send_displs_p[static_cast<std::size_t>(r)] *= s;
            recv_displs_p[static_cast<std::size_t>(r)] *= s;
        }
        MPI_Alltoallv(flat_send_payload.data(), send_counts_p.data(), send_displs_p.data(), MPI_DOUBLE,
                      recv_payload.data(), recv_counts_p.data(), recv_displs_p.data(), MPI_DOUBLE, comm);
    }

    // Received entries arrive grouped by ascending source rank, so the first
    // occurrence of a DOF is the lowest source rank's.
    std::map<GlobalIndex, std::size_t> added; // dof -> index into recv arrays
    for (std::size_t k = 0; k < recv_dofs.size(); ++k) {
        const auto dof = static_cast<GlobalIndex>(recv_dofs[k]);
        FE_THROW_IF(!owned.contains(dof), systems::InvalidStateException,
                    "completeOwnedBoundaryDofs: received boundary DOF " + std::to_string(dof) +
                        " that this rank does not own");
        if (std::binary_search(dofs.begin(), dofs.end(), dof)) {
            continue;
        }
        added.emplace(dof, k);
    }
    if (!context.empty()) {
        reportCompletion(comm, rank, size, context, added.size());
    }
    if (added.empty()) {
        return 0;
    }

    std::vector<GlobalIndex> merged_dofs;
    std::vector<Real> merged_payload;
    merged_dofs.reserve(dofs.size() + added.size());
    merged_payload.reserve((dofs.size() + added.size()) * stride);
    std::size_t i = 0;
    auto it = added.begin();
    while (i < dofs.size() || it != added.end()) {
        if (it == added.end() || (i < dofs.size() && dofs[i] < it->first)) {
            merged_dofs.push_back(dofs[i]);
            merged_payload.insert(merged_payload.end(),
                                  payload.begin() + static_cast<std::ptrdiff_t>(i * stride),
                                  payload.begin() + static_cast<std::ptrdiff_t>((i + 1) * stride));
            ++i;
        } else {
            merged_dofs.push_back(it->first);
            merged_payload.insert(merged_payload.end(),
                                  recv_payload.begin() + static_cast<std::ptrdiff_t>(it->second * stride),
                                  recv_payload.begin() + static_cast<std::ptrdiff_t>((it->second + 1) * stride));
            ++it;
        }
    }
    const std::size_t n_added = added.size();
    dofs = std::move(merged_dofs);
    payload = std::move(merged_payload);
    return n_added;
#else
    (void)system;
    (void)dofs;
    (void)payload;
    (void)stride;
    (void)context;
    return 0;
#endif
}

} // namespace constraints
} // namespace FE
} // namespace svmp
