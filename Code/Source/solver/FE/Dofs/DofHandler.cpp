/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "DofHandler.h"
#include "EntityDofMap.h"
#include "GhostDofManager.h"
#include "DofNumbering.h"
#include "MeshTopologyBuilder.h"
#include "Elements/ReferenceElement.h"
#include "Constraints/AffineConstraints.h"
#include "Spaces/FunctionSpace.h"
#include "Spaces/AdaptiveSpace.h"
#include "Spaces/OrientationManager.h"
#include "Spaces/MortarSpace.h"
#include "Spaces/SpaceInterpolation.h"
#include "Spaces/TraceSpace.h"
#include "Basis/LagrangeBasis.h"
#include "Basis/VectorBasis.h"

// Mesh convenience overloads require the Mesh library to be linked, not just headers present.
// Standalone FE builds inside the svMultiPhysics repo still see Mesh headers, so __has_include
// alone is insufficient. Prefer an explicit compile definition when provided.
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#  include "Mesh/Mesh.h"
#  include "Mesh/Core/InterfaceMesh.h"
#  include "Mesh/Observer/ScopedSubscription.h"
#  define DOFHANDLER_HAS_MESH 1
#else
#  define DOFHANDLER_HAS_MESH 0
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <map>

namespace svmp {
namespace FE {
namespace dofs {

#if FE_HAS_MPI
namespace {

inline void fe_mpi_check(int rc, const char* op) {
    if (rc == MPI_SUCCESS) return;
    char err[MPI_MAX_ERROR_STRING];
    int len = 0;
    MPI_Error_string(rc, err, &len);
    throw FEException(std::string(op) + " failed: " + std::string(err, static_cast<std::size_t>(len)));
}

inline std::uint64_t mix_u64(std::uint64_t h) {
    // MurmurHash3 finalizer mix for 64-bit values.
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return h;
}

constexpr std::uint64_t kHashSalt = static_cast<std::uint64_t>(0x9e3779b97f4a7c15ULL);

static gid_t checked_nonneg_mul(gid_t a, gid_t b, const char* ctx) {
    if (a < 0 || b < 0) {
        throw FEException(std::string(ctx) + ": negative value in multiplication");
    }
    using ugid_t = std::make_unsigned_t<gid_t>;
    const ugid_t ua = static_cast<ugid_t>(a);
    const ugid_t ub = static_cast<ugid_t>(b);
    const ugid_t umax = static_cast<ugid_t>(std::numeric_limits<gid_t>::max());
    if (ua != 0 && ub > umax / ua) {
        throw FEException(std::string(ctx) + ": multiplication overflow");
    }
    return static_cast<gid_t>(ua * ub);
}

static gid_t checked_nonneg_add(gid_t a, gid_t b, const char* ctx) {
    if (a < 0 || b < 0) {
        throw FEException(std::string(ctx) + ": negative value in addition");
    }
    using ugid_t = std::make_unsigned_t<gid_t>;
    const ugid_t ua = static_cast<ugid_t>(a);
    const ugid_t ub = static_cast<ugid_t>(b);
    const ugid_t umax = static_cast<ugid_t>(std::numeric_limits<gid_t>::max());
    if (ua > umax - ub) {
        throw FEException(std::string(ctx) + ": addition overflow");
    }
    return static_cast<gid_t>(ua + ub);
}

struct RankSignature {
    std::uint64_t min_cell_gid{std::numeric_limits<std::uint64_t>::max()};
    std::uint64_t hash{0};
};

static void mpi_allgather_fixed_bytes_no_collectives(MPI_Comm comm,
                                                    int my_rank,
                                                    int world_size,
                                                    const void* send_buf,
                                                    int send_bytes,
                                                    void* recv_buf,
                                                    int tag_base) {
    if (send_bytes < 0) {
        throw FEException("mpi_allgather_fixed_bytes_no_collectives: negative send_bytes");
    }
    const auto bytes = static_cast<std::size_t>(send_bytes);
    auto* out = static_cast<unsigned char*>(recv_buf);
    std::memcpy(out + static_cast<std::size_t>(my_rank) * bytes, send_buf, bytes);

    if (world_size <= 1) {
        return;
    }

    const int left = (my_rank - 1 + world_size) % world_size;
    const int right = (my_rank + 1) % world_size;

    for (int step = 0; step < world_size - 1; ++step) {
        const int send_idx = (my_rank - step + world_size) % world_size;
        const int recv_idx = (my_rank - step - 1 + world_size) % world_size;
        auto* send_ptr = out + static_cast<std::size_t>(send_idx) * bytes;
        auto* recv_ptr = out + static_cast<std::size_t>(recv_idx) * bytes;

        fe_mpi_check(MPI_Sendrecv(send_ptr,
                                  send_bytes,
                                  MPI_BYTE,
                                  right,
                                  tag_base + 0,
                                  recv_ptr,
                                  send_bytes,
                                  MPI_BYTE,
                                  left,
                                  tag_base + 0,
                                  comm,
                                  MPI_STATUS_IGNORE),
                     "MPI_Sendrecv in mpi_allgather_fixed_bytes_no_collectives");
    }
}

static std::vector<int> compute_stable_rank_order(MPI_Comm comm,
                                                  int my_rank,
                                                  int world_size,
                                                  std::span<const gid_t> cell_gids,
                                                  std::span<const int> cell_owner_ranks,
                                                  bool no_global_collectives,
                                                  int tag_base) {
    RankSignature local{};
    if (cell_gids.size() != cell_owner_ranks.size()) {
        // Cannot build a stable signature; fall back to MPI rank order.
        std::vector<int> rank_to_order(static_cast<std::size_t>(world_size), 0);
        for (int r = 0; r < world_size; ++r) {
            rank_to_order[static_cast<std::size_t>(r)] = r;
        }
        return rank_to_order;
    }

    for (std::size_t i = 0; i < cell_gids.size(); ++i) {
        if (cell_owner_ranks[i] != my_rank) continue;
        const gid_t gid = cell_gids[i];
        if (gid < 0) continue;
        const auto ugid = static_cast<std::uint64_t>(gid);
        local.min_cell_gid = std::min(local.min_cell_gid, ugid);
        local.hash ^= mix_u64(ugid + kHashSalt + (local.hash << 6) + (local.hash >> 2));
    }

    std::vector<RankSignature> all(static_cast<std::size_t>(world_size));
    if (!no_global_collectives) {
        fe_mpi_check(MPI_Allgather(&local, sizeof(RankSignature), MPI_BYTE,
                                   all.data(), sizeof(RankSignature), MPI_BYTE,
                                   comm),
                     "MPI_Allgather (RankSignature) in compute_stable_rank_order");
    } else {
        mpi_allgather_fixed_bytes_no_collectives(comm,
                                                 my_rank,
                                                 world_size,
                                                 &local,
                                                 static_cast<int>(sizeof(RankSignature)),
                                                 all.data(),
                                                 tag_base);
    }

    std::vector<int> ranks(static_cast<std::size_t>(world_size), 0);
    std::iota(ranks.begin(), ranks.end(), 0);
    std::sort(ranks.begin(), ranks.end(), [&](int a, int b) {
        const auto& sa = all[static_cast<std::size_t>(a)];
        const auto& sb = all[static_cast<std::size_t>(b)];
        if (sa.min_cell_gid != sb.min_cell_gid) return sa.min_cell_gid < sb.min_cell_gid;
        if (sa.hash != sb.hash) return sa.hash < sb.hash;
        return a < b;
    });

    std::vector<int> rank_to_order(static_cast<std::size_t>(world_size), 0);
    for (int ord = 0; ord < world_size; ++ord) {
        rank_to_order[static_cast<std::size_t>(ranks[static_cast<std::size_t>(ord)])] = ord;
    }
    return rank_to_order;
}

static void mpi_exscan_sum_and_bcast_total_no_collectives(MPI_Comm comm,
                                                         int my_rank,
                                                         int world_size,
                                                         gid_t local_value,
                                                         gid_t& out_exclusive_prefix,
                                                         gid_t& out_global_sum,
                                                         int tag_base) {
    out_exclusive_prefix = 0;
    out_global_sum = local_value;

    if (world_size <= 1) {
        return;
    }

    // Forward pass: compute exclusive prefix sums in MPI-rank order.
    gid_t running = local_value;
    if (my_rank == 0) {
        // Prefix for rank 0 is 0. Send running sum to rank 1.
        fe_mpi_check(MPI_Send(&running, 1, MPI_INT64_T, 1, tag_base + 0, comm),
                     "MPI_Send in mpi_exscan_sum_and_bcast_total_no_collectives (prefix)");
    } else {
        gid_t incoming = 0;
        fe_mpi_check(MPI_Recv(&incoming, 1, MPI_INT64_T, my_rank - 1, tag_base + 0, comm, MPI_STATUS_IGNORE),
                     "MPI_Recv in mpi_exscan_sum_and_bcast_total_no_collectives (prefix)");
        out_exclusive_prefix = incoming;
        running = incoming + local_value;
        if (my_rank < world_size - 1) {
            fe_mpi_check(MPI_Send(&running, 1, MPI_INT64_T, my_rank + 1, tag_base + 0, comm),
                         "MPI_Send in mpi_exscan_sum_and_bcast_total_no_collectives (prefix)");
        }
        if (my_rank == world_size - 1) {
            // Last rank sends global sum to rank 0.
            fe_mpi_check(MPI_Send(&running, 1, MPI_INT64_T, 0, tag_base + 1, comm),
                         "MPI_Send in mpi_exscan_sum_and_bcast_total_no_collectives (total)");
        }
    }

    // Rank 0 receives global sum from last rank.
    if (my_rank == 0) {
        gid_t total = 0;
        fe_mpi_check(MPI_Recv(&total, 1, MPI_INT64_T, world_size - 1, tag_base + 1, comm, MPI_STATUS_IGNORE),
                     "MPI_Recv in mpi_exscan_sum_and_bcast_total_no_collectives (total)");
        out_global_sum = total;
        // Broadcast total forward.
        fe_mpi_check(MPI_Send(&out_global_sum, 1, MPI_INT64_T, 1, tag_base + 2, comm),
                     "MPI_Send in mpi_exscan_sum_and_bcast_total_no_collectives (bcast)");
    } else {
        // Receive broadcast total from previous and forward.
        fe_mpi_check(MPI_Recv(&out_global_sum, 1, MPI_INT64_T, my_rank - 1, tag_base + 2, comm, MPI_STATUS_IGNORE),
                     "MPI_Recv in mpi_exscan_sum_and_bcast_total_no_collectives (bcast)");
        if (my_rank < world_size - 1) {
            fe_mpi_check(MPI_Send(&out_global_sum, 1, MPI_INT64_T, my_rank + 1, tag_base + 2, comm),
                         "MPI_Send in mpi_exscan_sum_and_bcast_total_no_collectives (bcast)");
        }
    }
}

static void mpi_exscan_sum_and_bcast_total_no_collectives_ordered(MPI_Comm comm,
                                                                  int my_rank,
                                                                  int world_size,
                                                                  std::span<const int> rank_to_order,
                                                                  gid_t local_value,
                                                                  gid_t& out_exclusive_prefix,
                                                                  gid_t& out_global_sum,
                                                                  int tag_base) {
    out_exclusive_prefix = 0;
    out_global_sum = local_value;

    if (world_size <= 1) {
        return;
    }
    if (rank_to_order.size() != static_cast<std::size_t>(world_size)) {
        throw FEException("mpi_exscan_sum_and_bcast_total_no_collectives_ordered: rank_to_order size mismatch");
    }

    std::vector<int> order_to_rank(static_cast<std::size_t>(world_size), -1);
    for (int r = 0; r < world_size; ++r) {
        const int ord = rank_to_order[static_cast<std::size_t>(r)];
        if (ord < 0 || ord >= world_size) {
            throw FEException("mpi_exscan_sum_and_bcast_total_no_collectives_ordered: invalid rank_to_order entry");
        }
        if (order_to_rank[static_cast<std::size_t>(ord)] != -1) {
            throw FEException("mpi_exscan_sum_and_bcast_total_no_collectives_ordered: rank_to_order is not a permutation");
        }
        order_to_rank[static_cast<std::size_t>(ord)] = r;
    }

    const int my_order = rank_to_order[static_cast<std::size_t>(my_rank)];
    const int root_rank = order_to_rank[0];
    const int last_rank = order_to_rank[static_cast<std::size_t>(world_size - 1)];
    if (root_rank < 0 || last_rank < 0) {
        throw FEException("mpi_exscan_sum_and_bcast_total_no_collectives_ordered: missing root/last rank");
    }

    const int prev_rank = (my_order > 0) ? order_to_rank[static_cast<std::size_t>(my_order - 1)] : MPI_PROC_NULL;
    const int next_rank =
        (my_order + 1 < world_size) ? order_to_rank[static_cast<std::size_t>(my_order + 1)] : MPI_PROC_NULL;

    // Forward pass: compute exclusive prefix sums along stable order.
    gid_t running = local_value;
    if (my_order == 0) {
        if (next_rank != MPI_PROC_NULL) {
            fe_mpi_check(MPI_Send(&running, 1, MPI_INT64_T, next_rank, tag_base + 0, comm),
                         "MPI_Send in mpi_exscan_sum_and_bcast_total_no_collectives_ordered (prefix)");
        }
    } else {
        gid_t incoming = 0;
        fe_mpi_check(MPI_Recv(&incoming, 1, MPI_INT64_T, prev_rank, tag_base + 0, comm, MPI_STATUS_IGNORE),
                     "MPI_Recv in mpi_exscan_sum_and_bcast_total_no_collectives_ordered (prefix)");
        out_exclusive_prefix = incoming;
        running = incoming + local_value;
        if (next_rank != MPI_PROC_NULL) {
            fe_mpi_check(MPI_Send(&running, 1, MPI_INT64_T, next_rank, tag_base + 0, comm),
                         "MPI_Send in mpi_exscan_sum_and_bcast_total_no_collectives_ordered (prefix)");
        }
        if (my_order == world_size - 1) {
            fe_mpi_check(MPI_Send(&running, 1, MPI_INT64_T, root_rank, tag_base + 1, comm),
                         "MPI_Send in mpi_exscan_sum_and_bcast_total_no_collectives_ordered (total)");
        }
    }

    // Root receives global sum from last rank and broadcasts along stable order.
    if (my_order == 0) {
        gid_t total = 0;
        fe_mpi_check(MPI_Recv(&total, 1, MPI_INT64_T, last_rank, tag_base + 1, comm, MPI_STATUS_IGNORE),
                     "MPI_Recv in mpi_exscan_sum_and_bcast_total_no_collectives_ordered (total)");
        out_global_sum = total;
        if (next_rank != MPI_PROC_NULL) {
            fe_mpi_check(MPI_Send(&out_global_sum, 1, MPI_INT64_T, next_rank, tag_base + 2, comm),
                         "MPI_Send in mpi_exscan_sum_and_bcast_total_no_collectives_ordered (bcast)");
        }
    } else {
        fe_mpi_check(MPI_Recv(&out_global_sum, 1, MPI_INT64_T, prev_rank, tag_base + 2, comm, MPI_STATUS_IGNORE),
                     "MPI_Recv in mpi_exscan_sum_and_bcast_total_no_collectives_ordered (bcast)");
        if (next_rank != MPI_PROC_NULL) {
            fe_mpi_check(MPI_Send(&out_global_sum, 1, MPI_INT64_T, next_rank, tag_base + 2, comm),
                         "MPI_Send in mpi_exscan_sum_and_bcast_total_no_collectives_ordered (bcast)");
        }
    }
}

static gid_t mpi_allreduce_max_no_collectives(MPI_Comm comm,
                                              int my_rank,
                                              int world_size,
                                              gid_t local_value,
                                              int tag_base) {
    gid_t out_global_max = local_value;
    if (world_size <= 1) {
        return out_global_max;
    }

    // Forward pass: running max in MPI-rank order.
    gid_t running = local_value;
    if (my_rank == 0) {
        fe_mpi_check(MPI_Send(&running, 1, MPI_INT64_T, 1, tag_base + 0, comm),
                     "MPI_Send in mpi_allreduce_max_no_collectives (fwd)");
    } else {
        gid_t incoming = 0;
        fe_mpi_check(MPI_Recv(&incoming, 1, MPI_INT64_T, my_rank - 1, tag_base + 0, comm, MPI_STATUS_IGNORE),
                     "MPI_Recv in mpi_allreduce_max_no_collectives (fwd)");
        running = std::max(incoming, local_value);
        if (my_rank < world_size - 1) {
            fe_mpi_check(MPI_Send(&running, 1, MPI_INT64_T, my_rank + 1, tag_base + 0, comm),
                         "MPI_Send in mpi_allreduce_max_no_collectives (fwd)");
        }
        if (my_rank == world_size - 1) {
            fe_mpi_check(MPI_Send(&running, 1, MPI_INT64_T, 0, tag_base + 1, comm),
                         "MPI_Send in mpi_allreduce_max_no_collectives (back)");
        }
    }

    // Rank 0 receives global max and broadcasts forward.
    if (my_rank == 0) {
        gid_t total = 0;
        fe_mpi_check(MPI_Recv(&total, 1, MPI_INT64_T, world_size - 1, tag_base + 1, comm, MPI_STATUS_IGNORE),
                     "MPI_Recv in mpi_allreduce_max_no_collectives (back)");
        out_global_max = total;
        fe_mpi_check(MPI_Send(&out_global_max, 1, MPI_INT64_T, 1, tag_base + 2, comm),
                     "MPI_Send in mpi_allreduce_max_no_collectives (bcast)");
    } else {
        fe_mpi_check(MPI_Recv(&out_global_max, 1, MPI_INT64_T, my_rank - 1, tag_base + 2, comm, MPI_STATUS_IGNORE),
                     "MPI_Recv in mpi_allreduce_max_no_collectives (bcast)");
        if (my_rank < world_size - 1) {
            fe_mpi_check(MPI_Send(&out_global_max, 1, MPI_INT64_T, my_rank + 1, tag_base + 2, comm),
                         "MPI_Send in mpi_allreduce_max_no_collectives (bcast)");
        }
    }

    return out_global_max;
}

static gid_t mpi_allreduce_min_no_collectives(MPI_Comm comm,
                                              int my_rank,
                                              int world_size,
                                              gid_t local_value,
                                              int tag_base) {
    gid_t out_global_min = local_value;
    if (world_size <= 1) {
        return out_global_min;
    }

    // Forward pass: running min in MPI-rank order.
    gid_t running = local_value;
    if (my_rank == 0) {
        fe_mpi_check(MPI_Send(&running, 1, MPI_INT64_T, 1, tag_base + 0, comm),
                     "MPI_Send in mpi_allreduce_min_no_collectives (fwd)");
    } else {
        gid_t incoming = 0;
        fe_mpi_check(MPI_Recv(&incoming, 1, MPI_INT64_T, my_rank - 1, tag_base + 0, comm, MPI_STATUS_IGNORE),
                     "MPI_Recv in mpi_allreduce_min_no_collectives (fwd)");
        running = std::min(incoming, local_value);
        if (my_rank < world_size - 1) {
            fe_mpi_check(MPI_Send(&running, 1, MPI_INT64_T, my_rank + 1, tag_base + 0, comm),
                         "MPI_Send in mpi_allreduce_min_no_collectives (fwd)");
        }
        if (my_rank == world_size - 1) {
            fe_mpi_check(MPI_Send(&running, 1, MPI_INT64_T, 0, tag_base + 1, comm),
                         "MPI_Send in mpi_allreduce_min_no_collectives (back)");
        }
    }

    // Rank 0 receives global min and broadcasts forward.
    if (my_rank == 0) {
        gid_t total = 0;
        fe_mpi_check(MPI_Recv(&total, 1, MPI_INT64_T, world_size - 1, tag_base + 1, comm, MPI_STATUS_IGNORE),
                     "MPI_Recv in mpi_allreduce_min_no_collectives (back)");
        out_global_min = total;
        fe_mpi_check(MPI_Send(&out_global_min, 1, MPI_INT64_T, 1, tag_base + 2, comm),
                     "MPI_Send in mpi_allreduce_min_no_collectives (bcast)");
    } else {
        fe_mpi_check(MPI_Recv(&out_global_min, 1, MPI_INT64_T, my_rank - 1, tag_base + 2, comm, MPI_STATUS_IGNORE),
                     "MPI_Recv in mpi_allreduce_min_no_collectives (bcast)");
        if (my_rank < world_size - 1) {
            fe_mpi_check(MPI_Send(&out_global_min, 1, MPI_INT64_T, my_rank + 1, tag_base + 2, comm),
                         "MPI_Send in mpi_allreduce_min_no_collectives (bcast)");
        }
    }

    return out_global_min;
}

struct EdgeKey {
    gid_t a{0};
    gid_t b{0};

    bool operator==(const EdgeKey& other) const noexcept { return a == other.a && b == other.b; }
    bool operator<(const EdgeKey& other) const noexcept {
        return (a < other.a) || (a == other.a && b < other.b);
    }
};

struct EdgeKeyHash {
    std::size_t operator()(const EdgeKey& k) const noexcept {
        std::uint64_t h = mix_u64(static_cast<std::uint64_t>(k.a));
        h ^= mix_u64(static_cast<std::uint64_t>(k.b) + kHashSalt);
        return static_cast<std::size_t>(h);
    }
};

struct FaceKey {
    std::array<gid_t, 4> gids{};
    std::uint8_t n{0};

    bool operator==(const FaceKey& other) const noexcept { return n == other.n && gids == other.gids; }
    bool operator<(const FaceKey& other) const noexcept {
        if (n != other.n) return n < other.n;
        for (std::size_t i = 0; i < 4u; ++i) {
            if (gids[i] != other.gids[i]) return gids[i] < other.gids[i];
        }
        return false;
    }
};

struct FaceKeyHash {
    std::size_t operator()(const FaceKey& k) const noexcept {
        std::uint64_t seed = mix_u64(static_cast<std::uint64_t>(k.n));
        for (std::size_t i = 0; i < static_cast<std::size_t>(k.n); ++i) {
            seed ^= mix_u64(static_cast<std::uint64_t>(k.gids[i]) + kHashSalt + (seed << 6) + (seed >> 2));
        }
        return static_cast<std::size_t>(seed);
    }
};

static void mpi_allreduce_max_u8_no_collectives(MPI_Comm comm,
                                                int my_rank,
                                                int world_size,
                                                std::uint8_t* inout,
                                                int count,
                                                int tag_base) {
    if (count < 0) {
        throw FEException("mpi_allreduce_max_u8_no_collectives: negative count");
    }
    if (world_size <= 1 || count == 0) {
        return;
    }

    std::vector<std::uint8_t> recv(static_cast<std::size_t>(count), 0u);
    const int prev = (my_rank - 1 + world_size) % world_size;
    const int next = (my_rank + 1) % world_size;

    for (int step = 0; step < world_size - 1; ++step) {
        fe_mpi_check(MPI_Sendrecv(inout,
                                  count,
                                  MPI_UNSIGNED_CHAR,
                                  next,
                                  tag_base + 0,
                                  recv.data(),
                                  count,
                                  MPI_UNSIGNED_CHAR,
                                  prev,
                                  tag_base + 0,
                                  comm,
                                  MPI_STATUS_IGNORE),
                     "MPI_Sendrecv in mpi_allreduce_max_u8_no_collectives");
        for (int i = 0; i < count; ++i) {
            inout[i] = std::max(inout[i], recv[static_cast<std::size_t>(i)]);
        }
    }
}

struct HllSketch {
    static constexpr int p = 12;
    static constexpr std::size_t m = (1u << p);

    std::array<std::uint8_t, m> reg{};

    void add_hash(std::uint64_t h) noexcept {
        const std::uint32_t idx = static_cast<std::uint32_t>(h >> (64u - p));
        const std::uint64_t w = h << p;
        const std::uint8_t rho = (w == 0u)
                                     ? static_cast<std::uint8_t>(64u - p + 1u)
                                     : static_cast<std::uint8_t>(static_cast<unsigned>(__builtin_clzll(w)) + 1u);
        reg[static_cast<std::size_t>(idx)] = std::max(reg[static_cast<std::size_t>(idx)], rho);
    }

    double estimate() const {
        const double mm = static_cast<double>(m);
        const double alpha = (m == 16u) ? 0.673
                             : (m == 32u) ? 0.697
                             : (m == 64u) ? 0.709
                                          : (0.7213 / (1.0 + 1.079 / mm));

        double inv_sum = 0.0;
        std::size_t zeros = 0;
        for (std::size_t i = 0; i < m; ++i) {
            const auto r = reg[i];
            if (r == 0u) {
                ++zeros;
            }
            inv_sum += std::ldexp(1.0, -static_cast<int>(r));
        }

        double e = alpha * mm * mm / inv_sum;

        // Small-range correction (linear counting).
        if (e <= 2.5 * mm && zeros > 0u) {
            e = mm * std::log(mm / static_cast<double>(zeros));
        }

        return e;
    }
};

template <typename Key, typename KeyHash>
static double hll_estimate_global_unique(MPI_Comm comm,
                                        int my_rank,
                                        int world_size,
                                        std::span<const Key> local_keys,
                                        bool no_global_collectives,
                                        int tag_base) {
    HllSketch local;
    for (const auto& key : local_keys) {
        std::uint64_t h = static_cast<std::uint64_t>(KeyHash{}(key));
        h = mix_u64(h + kHashSalt);
        local.add_hash(h);
    }

    HllSketch global = local;
    if (!no_global_collectives) {
        fe_mpi_check(MPI_Allreduce(local.reg.data(),
                                   global.reg.data(),
                                   static_cast<int>(HllSketch::m),
                                   MPI_UNSIGNED_CHAR,
                                   MPI_MAX,
                                   comm),
                     "MPI_Allreduce (HLL max) in hll_estimate_global_unique");
    } else {
        mpi_allreduce_max_u8_no_collectives(comm,
                                            my_rank,
                                            world_size,
                                            global.reg.data(),
                                            static_cast<int>(HllSketch::m),
                                            tag_base);
    }

    return global.estimate();
}

#if 0
// Legacy Alltoallv-based rendezvous numbering path. Kept for reference only;
// all production paths use neighbor-only point-to-point schedules.

template <typename Key>
struct KeyRequest {
    Key key{};
    std::int32_t reply_rank{0};
    std::int32_t touch_rank{0};
    gid_t cell_gid_candidate{-1};
    std::int32_t cell_owner_candidate{-1};
};

template <typename Key>
struct KeyResponse {
    Key key{};
    gid_t global_id{-1};
    std::int32_t owner_rank{-1};
};

struct KeyAggregate {
    gid_t global_id{-1};
    int min_rank{std::numeric_limits<int>::max()};
    int max_rank{-1};
    std::uint64_t best_hash_score{0};
    int best_hash_rank{-1};
    gid_t best_cell_gid{std::numeric_limits<gid_t>::max()};
    int best_cell_owner{-1};
    int fixed_owner{-1};
    bool fixed_owner_set{false};
};

template <typename Key, typename KeyHash, typename KeyLess, typename HomeRankFunc>
static void assign_contiguous_ids_and_owners_mpi(
    MPI_Comm comm,
    int my_rank,
    int world_size,
    OwnershipStrategy ownership,
    std::span<const Key> local_keys,
    std::span<const int> local_touch_ranks,
    std::span<const gid_t> cell_gid_candidate,
    std::span<const int> cell_owner_candidate,
    HomeRankFunc home_rank,
    bool force_owner_from_candidate,
    std::vector<gid_t>& out_global_ids,
    std::vector<int>& out_owner_ranks,
    gid_t& out_global_count)
{
    if (local_keys.size() != cell_gid_candidate.size() || local_keys.size() != cell_owner_candidate.size() ||
        local_keys.size() != local_touch_ranks.size()) {
        throw FEException("assign_contiguous_ids_and_owners_mpi: candidate arrays size mismatch");
    }

    if (world_size <= 1) {
        std::unordered_map<Key, std::size_t, KeyHash> local_index;
        local_index.reserve(local_keys.size());
        for (std::size_t i = 0; i < local_keys.size(); ++i) {
            local_index.emplace(local_keys[i], i);
        }

        std::vector<Key> keys(local_keys.begin(), local_keys.end());
        std::sort(keys.begin(), keys.end(), KeyLess{});

        out_global_ids.assign(local_keys.size(), gid_t{-1});
        out_owner_ranks.assign(local_keys.size(), -1);

        for (std::size_t k = 0; k < keys.size(); ++k) {
            const auto it = local_index.find(keys[k]);
            if (it == local_index.end()) continue;
            out_global_ids[it->second] = static_cast<gid_t>(k);

            int owner = my_rank;
            if (force_owner_from_candidate) {
                owner = cell_owner_candidate[it->second];
            } else if (ownership == OwnershipStrategy::CellOwner) {
                owner = (cell_owner_candidate[it->second] >= 0) ? cell_owner_candidate[it->second] : my_rank;
            } else if (ownership == OwnershipStrategy::VertexGID) {
                owner = my_rank;
            }
            out_owner_ranks[it->second] = owner;
        }

        for (std::size_t i = 0; i < local_keys.size(); ++i) {
            if (out_global_ids[i] < 0 || out_owner_ranks[i] < 0) {
                throw FEException("assign_contiguous_ids_and_owners_mpi: missing assignment in serial path");
            }
        }

        out_global_count = static_cast<gid_t>(keys.size());
        return;
    }

    std::vector<std::vector<KeyRequest<Key>>> send_by_rank(static_cast<std::size_t>(world_size));
    for (std::size_t i = 0; i < local_keys.size(); ++i) {
        const int dest = home_rank(local_keys[i], world_size);
        if (dest < 0 || dest >= world_size) {
            throw FEException("assign_contiguous_ids_and_owners_mpi: computed home rank out of range");
        }
        const int touch_rank = local_touch_ranks[i];
        if (touch_rank < -1 || touch_rank >= world_size) {
            throw FEException("assign_contiguous_ids_and_owners_mpi: touch rank out of range");
        }
        send_by_rank[static_cast<std::size_t>(dest)].push_back(
            KeyRequest<Key>{local_keys[i], static_cast<std::int32_t>(my_rank), static_cast<std::int32_t>(touch_rank),
                            cell_gid_candidate[i],
                            static_cast<std::int32_t>(cell_owner_candidate[i])});
    }

    std::vector<int> send_counts(static_cast<std::size_t>(world_size), 0);
    std::vector<int> send_displs(static_cast<std::size_t>(world_size) + 1u, 0);
    for (int r = 0; r < world_size; ++r) {
        const auto bytes =
            static_cast<std::size_t>(send_by_rank[static_cast<std::size_t>(r)].size()) * sizeof(KeyRequest<Key>);
        if (bytes > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw FEException("assign_contiguous_ids_and_owners_mpi: send buffer too large for MPI int counts");
        }
        send_counts[static_cast<std::size_t>(r)] = static_cast<int>(bytes);
        send_displs[static_cast<std::size_t>(r) + 1u] = send_displs[static_cast<std::size_t>(r)] + send_counts[static_cast<std::size_t>(r)];
    }

    std::vector<char> send_buffer(static_cast<std::size_t>(send_displs[static_cast<std::size_t>(world_size)]));
    for (int r = 0; r < world_size; ++r) {
        const auto& payload = send_by_rank[static_cast<std::size_t>(r)];
        if (payload.empty()) continue;
        std::memcpy(send_buffer.data() + send_displs[static_cast<std::size_t>(r)],
                    payload.data(),
                    payload.size() * sizeof(KeyRequest<Key>));
    }

    std::vector<int> recv_counts(static_cast<std::size_t>(world_size), 0);
    fe_mpi_check(MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                              recv_counts.data(), 1, MPI_INT,
                              comm),
                 "MPI_Alltoall (counts) in assign_contiguous_ids_and_owners_mpi");

    std::vector<int> recv_displs(static_cast<std::size_t>(world_size) + 1u, 0);
    for (int r = 0; r < world_size; ++r) {
        recv_displs[static_cast<std::size_t>(r) + 1u] = recv_displs[static_cast<std::size_t>(r)] + recv_counts[static_cast<std::size_t>(r)];
    }

    std::vector<char> recv_buffer(static_cast<std::size_t>(recv_displs[static_cast<std::size_t>(world_size)]));
    fe_mpi_check(MPI_Alltoallv(send_buffer.data(), send_counts.data(), send_displs.data(), MPI_BYTE,
                               recv_buffer.data(), recv_counts.data(), recv_displs.data(), MPI_BYTE,
                               comm),
                 "MPI_Alltoallv (data) in assign_contiguous_ids_and_owners_mpi");

    const auto n_reqs = recv_buffer.size() / sizeof(KeyRequest<Key>);
    if (n_reqs * sizeof(KeyRequest<Key>) != recv_buffer.size()) {
        throw FEException("assign_contiguous_ids_and_owners_mpi: received buffer size misaligned");
    }
    const auto* reqs = reinterpret_cast<const KeyRequest<Key>*>(recv_buffer.data());

    std::unordered_map<Key, KeyAggregate, KeyHash> aggregates;
    aggregates.reserve(n_reqs);

    for (std::size_t i = 0; i < n_reqs; ++i) {
        const auto& req = reqs[i];
        auto& agg = aggregates[req.key];
        const int touch_rank = static_cast<int>(req.touch_rank);
        if (touch_rank >= 0) {
            agg.min_rank = std::min(agg.min_rank, touch_rank);
            agg.max_rank = std::max(agg.max_rank, touch_rank);
            if (ownership == OwnershipStrategy::VertexGID && !force_owner_from_candidate) {
                const auto key_hash = mix_u64(static_cast<std::uint64_t>(KeyHash{}(req.key)));
                const auto rank_hash = mix_u64(static_cast<std::uint64_t>(touch_rank) + kHashSalt);
                const auto score = mix_u64(key_hash ^ rank_hash);
                if (agg.best_hash_rank < 0 || score > agg.best_hash_score ||
                    (score == agg.best_hash_score && touch_rank < agg.best_hash_rank)) {
                    agg.best_hash_score = score;
                    agg.best_hash_rank = touch_rank;
                }
            }
        }

        if (force_owner_from_candidate) {
            const int owner = static_cast<int>(req.cell_owner_candidate);
            if (!agg.fixed_owner_set) {
                agg.fixed_owner = owner;
                agg.fixed_owner_set = true;
            } else if (agg.fixed_owner != owner) {
                throw FEException("assign_contiguous_ids_and_owners_mpi: inconsistent fixed owner across ranks for the same key");
            }
        } else if (ownership == OwnershipStrategy::CellOwner) {
            const auto cgid = req.cell_gid_candidate;
            const int owner = static_cast<int>(req.cell_owner_candidate);
            if (cgid >= 0 && owner >= 0) {
                if (cgid < agg.best_cell_gid || (cgid == agg.best_cell_gid && owner < agg.best_cell_owner)) {
                    agg.best_cell_gid = cgid;
                    agg.best_cell_owner = owner;
                }
            }
        }
    }

    std::vector<Key> unique_keys;
    unique_keys.reserve(aggregates.size());
    for (const auto& [k, v] : aggregates) {
        (void)v;
        unique_keys.push_back(k);
    }
    std::sort(unique_keys.begin(), unique_keys.end(), KeyLess{});

    const gid_t local_unique = static_cast<gid_t>(unique_keys.size());
    gid_t gid_offset = 0;
    fe_mpi_check(MPI_Exscan(&local_unique, &gid_offset, 1, MPI_INT64_T, MPI_SUM, comm),
                 "MPI_Exscan (offset) in assign_contiguous_ids_and_owners_mpi");
    if (my_rank == 0) {
        gid_offset = 0;
    }

    for (gid_t i = 0; i < local_unique; ++i) {
        auto& agg = aggregates[unique_keys[static_cast<std::size_t>(i)]];
        agg.global_id = gid_offset + i;

        if (force_owner_from_candidate) {
            agg.max_rank = agg.min_rank; // suppress unused rank logic
        }

        if (force_owner_from_candidate) {
            // Owner is fixed by candidate consistency (used for cells).
            // Nothing to do here.
        } else {
            if (agg.min_rank == std::numeric_limits<int>::max()) {
                throw FEException("assign_contiguous_ids_and_owners_mpi: missing touching ranks for key while assigning ownership");
            }
            int owner = my_rank;
            switch (ownership) {
                case OwnershipStrategy::LowestRank:
                    owner = agg.min_rank;
                    break;
                case OwnershipStrategy::HighestRank:
                    owner = agg.max_rank;
                    break;
                case OwnershipStrategy::VertexGID:
                    owner = (agg.best_hash_rank >= 0) ? agg.best_hash_rank : agg.min_rank;
                    break;
                case OwnershipStrategy::CellOwner:
                    owner = (agg.best_cell_owner >= 0) ? agg.best_cell_owner : agg.min_rank;
                    break;
            }
            agg.fixed_owner = owner;
            agg.fixed_owner_set = true;
        }
    }

    // Build response buffers: one response per received request (return to source rank).
    std::vector<std::vector<KeyResponse<Key>>> responses_by_rank(static_cast<std::size_t>(world_size));
    for (std::size_t i = 0; i < n_reqs; ++i) {
        const auto& req = reqs[i];
        const int dest = static_cast<int>(req.reply_rank);
        if (dest < 0 || dest >= world_size) {
            throw FEException("assign_contiguous_ids_and_owners_mpi: invalid source rank in request");
        }
        const auto it = aggregates.find(req.key);
        if (it == aggregates.end() || it->second.global_id < 0 || !it->second.fixed_owner_set) {
            throw FEException("assign_contiguous_ids_and_owners_mpi: missing aggregate entry while building response");
        }
        responses_by_rank[static_cast<std::size_t>(dest)].push_back(
            KeyResponse<Key>{req.key, it->second.global_id, static_cast<std::int32_t>(it->second.fixed_owner)});
    }

    std::vector<int> resp_send_counts(static_cast<std::size_t>(world_size), 0);
    std::vector<int> resp_send_displs(static_cast<std::size_t>(world_size) + 1u, 0);
    for (int r = 0; r < world_size; ++r) {
        const auto bytes =
            static_cast<std::size_t>(responses_by_rank[static_cast<std::size_t>(r)].size()) * sizeof(KeyResponse<Key>);
        if (bytes > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw FEException("assign_contiguous_ids_and_owners_mpi: response buffer too large for MPI int counts");
        }
        resp_send_counts[static_cast<std::size_t>(r)] = static_cast<int>(bytes);
        resp_send_displs[static_cast<std::size_t>(r) + 1u] =
            resp_send_displs[static_cast<std::size_t>(r)] + resp_send_counts[static_cast<std::size_t>(r)];
    }

    std::vector<char> resp_send_buffer(static_cast<std::size_t>(resp_send_displs[static_cast<std::size_t>(world_size)]));
    for (int r = 0; r < world_size; ++r) {
        const auto& payload = responses_by_rank[static_cast<std::size_t>(r)];
        if (payload.empty()) continue;
        std::memcpy(resp_send_buffer.data() + resp_send_displs[static_cast<std::size_t>(r)],
                    payload.data(),
                    payload.size() * sizeof(KeyResponse<Key>));
    }

    std::vector<int> resp_recv_counts(static_cast<std::size_t>(world_size), 0);
    fe_mpi_check(MPI_Alltoall(resp_send_counts.data(), 1, MPI_INT,
                              resp_recv_counts.data(), 1, MPI_INT,
                              comm),
                 "MPI_Alltoall (resp counts) in assign_contiguous_ids_and_owners_mpi");

    std::vector<int> resp_recv_displs(static_cast<std::size_t>(world_size) + 1u, 0);
    for (int r = 0; r < world_size; ++r) {
        resp_recv_displs[static_cast<std::size_t>(r) + 1u] =
            resp_recv_displs[static_cast<std::size_t>(r)] + resp_recv_counts[static_cast<std::size_t>(r)];
    }

    std::vector<char> resp_recv_buffer(static_cast<std::size_t>(resp_recv_displs[static_cast<std::size_t>(world_size)]));
    fe_mpi_check(MPI_Alltoallv(resp_send_buffer.data(), resp_send_counts.data(), resp_send_displs.data(), MPI_BYTE,
                               resp_recv_buffer.data(), resp_recv_counts.data(), resp_recv_displs.data(), MPI_BYTE,
                               comm),
                 "MPI_Alltoallv (resp data) in assign_contiguous_ids_and_owners_mpi");

    const auto n_resps = resp_recv_buffer.size() / sizeof(KeyResponse<Key>);
    if (n_resps * sizeof(KeyResponse<Key>) != resp_recv_buffer.size()) {
        throw FEException("assign_contiguous_ids_and_owners_mpi: response buffer size misaligned");
    }

    std::unordered_map<Key, std::size_t, KeyHash> local_index;
    local_index.reserve(local_keys.size());
    for (std::size_t i = 0; i < local_keys.size(); ++i) {
        local_index.emplace(local_keys[i], i);
    }

    out_global_ids.assign(local_keys.size(), gid_t{-1});
    out_owner_ranks.assign(local_keys.size(), -1);

    const auto* resps = reinterpret_cast<const KeyResponse<Key>*>(resp_recv_buffer.data());
    for (std::size_t i = 0; i < n_resps; ++i) {
        const auto& resp = resps[i];
        const auto it = local_index.find(resp.key);
        if (it == local_index.end()) {
            continue; // Response for a key we don't have locally (should not happen, but ignore safely)
        }
        out_global_ids[it->second] = resp.global_id;
        out_owner_ranks[it->second] = static_cast<int>(resp.owner_rank);
    }

    for (std::size_t i = 0; i < local_keys.size(); ++i) {
        if (out_global_ids[i] < 0 || out_owner_ranks[i] < 0) {
            throw FEException("assign_contiguous_ids_and_owners_mpi: missing assignment for a local key");
        }
    }

    fe_mpi_check(MPI_Allreduce(&local_unique, &out_global_count, 1, MPI_INT64_T, MPI_SUM, comm),
                 "MPI_Allreduce (global count) in assign_contiguous_ids_and_owners_mpi");
}

#endif // 0

	template <typename T>
	static void mpi_neighbor_exchange_bytes(
	    MPI_Comm comm,
	    std::span<const int> neighbors,
	    std::span<const std::vector<T>> send_lists,
	    std::vector<std::vector<T>>& recv_lists,
	    int tag_counts,
	    int tag_data)
	{
    static_assert(std::is_trivially_copyable_v<T>, "mpi_neighbor_exchange_bytes requires trivially copyable T");
    if (neighbors.size() != send_lists.size()) {
        throw FEException("mpi_neighbor_exchange_bytes: neighbors/send_lists size mismatch");
    }

    recv_lists.assign(neighbors.size(), {});

    std::vector<int> send_counts(neighbors.size(), 0);
    for (std::size_t i = 0; i < neighbors.size(); ++i) {
        const auto bytes = send_lists[i].size() * sizeof(T);
        if (bytes > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw FEException("mpi_neighbor_exchange_bytes: send buffer too large for MPI int counts");
        }
        send_counts[i] = static_cast<int>(bytes);
    }

    std::vector<int> recv_counts(neighbors.size(), 0);
    std::vector<MPI_Request> count_reqs;
    count_reqs.reserve(neighbors.size() * 2u);
    for (std::size_t i = 0; i < neighbors.size(); ++i) {
        MPI_Request req{};
        fe_mpi_check(MPI_Isend(&send_counts[i], 1, MPI_INT, neighbors[i], tag_counts, comm, &req),
                     "MPI_Isend (counts) in mpi_neighbor_exchange_bytes");
        count_reqs.push_back(req);
        fe_mpi_check(MPI_Irecv(&recv_counts[i], 1, MPI_INT, neighbors[i], tag_counts, comm, &req),
                     "MPI_Irecv (counts) in mpi_neighbor_exchange_bytes");
        count_reqs.push_back(req);
    }
    fe_mpi_check(MPI_Waitall(static_cast<int>(count_reqs.size()), count_reqs.data(), MPI_STATUSES_IGNORE),
                 "MPI_Waitall (counts) in mpi_neighbor_exchange_bytes");

    for (std::size_t i = 0; i < neighbors.size(); ++i) {
        const int nbytes = recv_counts[i];
        if (nbytes < 0) {
            throw FEException("mpi_neighbor_exchange_bytes: negative receive byte count");
        }
        if (nbytes == 0) {
            recv_lists[i].clear();
            continue;
        }
        if ((static_cast<std::size_t>(nbytes) % sizeof(T)) != 0u) {
            throw FEException("mpi_neighbor_exchange_bytes: receive buffer size not divisible by element size");
        }
        recv_lists[i].resize(static_cast<std::size_t>(nbytes) / sizeof(T));
    }

    std::vector<MPI_Request> data_reqs;
    data_reqs.reserve(neighbors.size() * 2u);
    for (std::size_t i = 0; i < neighbors.size(); ++i) {
        MPI_Request req{};
        const auto& send = send_lists[i];
        fe_mpi_check(MPI_Isend(send.empty() ? nullptr : static_cast<const void*>(send.data()),
                               send_counts[i],
                               MPI_BYTE,
                               neighbors[i],
                               tag_data,
                               comm,
                               &req),
                     "MPI_Isend (data) in mpi_neighbor_exchange_bytes");
        data_reqs.push_back(req);

        auto& recv = recv_lists[i];
        fe_mpi_check(MPI_Irecv(recv.empty() ? nullptr : static_cast<void*>(recv.data()),
                               recv_counts[i],
                               MPI_BYTE,
                               neighbors[i],
                               tag_data,
                               comm,
                               &req),
                     "MPI_Irecv (data) in mpi_neighbor_exchange_bytes");
        data_reqs.push_back(req);
    }
	    fe_mpi_check(MPI_Waitall(static_cast<int>(data_reqs.size()), data_reqs.data(), MPI_STATUSES_IGNORE),
	                 "MPI_Waitall (data) in mpi_neighbor_exchange_bytes");
	}

	template <typename T>
	static void mpi_neighbor_exchange_broadcast_bytes(
	    MPI_Comm comm,
	    std::span<const int> neighbors,
	    std::span<const T> send_data,
	    std::vector<std::vector<T>>& recv_lists,
	    int tag_counts,
	    int tag_data)
	{
	    static_assert(std::is_trivially_copyable_v<T>, "mpi_neighbor_exchange_broadcast_bytes requires trivially copyable T");
	    recv_lists.assign(neighbors.size(), {});

	    const std::size_t send_bytes_sz = send_data.size() * sizeof(T);
	    if (send_bytes_sz > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
	        throw FEException("mpi_neighbor_exchange_broadcast_bytes: send buffer too large for MPI int counts");
	    }
	    const int send_bytes = static_cast<int>(send_bytes_sz);

	    std::vector<int> recv_counts(neighbors.size(), 0);
	    std::vector<MPI_Request> count_reqs;
	    count_reqs.reserve(neighbors.size() * 2u);
	    for (std::size_t i = 0; i < neighbors.size(); ++i) {
	        MPI_Request req{};
	        fe_mpi_check(MPI_Isend(&send_bytes, 1, MPI_INT, neighbors[i], tag_counts, comm, &req),
	                     "MPI_Isend (counts) in mpi_neighbor_exchange_broadcast_bytes");
	        count_reqs.push_back(req);
	        fe_mpi_check(MPI_Irecv(&recv_counts[i], 1, MPI_INT, neighbors[i], tag_counts, comm, &req),
	                     "MPI_Irecv (counts) in mpi_neighbor_exchange_broadcast_bytes");
	        count_reqs.push_back(req);
	    }
	    fe_mpi_check(MPI_Waitall(static_cast<int>(count_reqs.size()), count_reqs.data(), MPI_STATUSES_IGNORE),
	                 "MPI_Waitall (counts) in mpi_neighbor_exchange_broadcast_bytes");

	    for (std::size_t i = 0; i < neighbors.size(); ++i) {
	        const int nbytes = recv_counts[i];
	        if (nbytes < 0) {
	            throw FEException("mpi_neighbor_exchange_broadcast_bytes: negative receive byte count");
	        }
	        if (nbytes == 0) {
	            recv_lists[i].clear();
	            continue;
	        }
	        if ((static_cast<std::size_t>(nbytes) % sizeof(T)) != 0u) {
	            throw FEException("mpi_neighbor_exchange_broadcast_bytes: receive buffer size not divisible by element size");
	        }
	        recv_lists[i].resize(static_cast<std::size_t>(nbytes) / sizeof(T));
	    }

	    std::vector<MPI_Request> data_reqs;
	    data_reqs.reserve(neighbors.size() * 2u);
	    for (std::size_t i = 0; i < neighbors.size(); ++i) {
	        MPI_Request req{};
	        fe_mpi_check(MPI_Isend(send_data.empty() ? nullptr : static_cast<const void*>(send_data.data()),
	                               send_bytes,
	                               MPI_BYTE,
	                               neighbors[i],
	                               tag_data,
	                               comm,
	                               &req),
	                     "MPI_Isend (data) in mpi_neighbor_exchange_broadcast_bytes");
	        data_reqs.push_back(req);

	        auto& recv = recv_lists[i];
	        fe_mpi_check(MPI_Irecv(recv.empty() ? nullptr : static_cast<void*>(recv.data()),
	                               recv_counts[i],
	                               MPI_BYTE,
	                               neighbors[i],
	                               tag_data,
	                               comm,
	                               &req),
	                     "MPI_Irecv (data) in mpi_neighbor_exchange_broadcast_bytes");
	        data_reqs.push_back(req);
	    }
	    fe_mpi_check(MPI_Waitall(static_cast<int>(data_reqs.size()), data_reqs.data(), MPI_STATUSES_IGNORE),
	                 "MPI_Waitall (data) in mpi_neighbor_exchange_broadcast_bytes");
	}

	template <typename Key, typename KeyHash>
	static std::uint64_t hrw_score(const Key& key, int rank) {
	    const std::uint64_t base = mix_u64(static_cast<std::uint64_t>(KeyHash{}(key)));
		    const std::uint64_t r = mix_u64(static_cast<std::uint64_t>(rank) + kHashSalt);
	    return mix_u64(base ^ r);
	}

	template <typename Key>
	struct KeyValuePair {
	    Key key{};
	    gid_t value{-1};
	};

template <typename Key, typename KeyHash, typename KeyLess>
static void assign_global_ordinals_with_neighbors(
    MPI_Comm comm,
    int my_rank,
    int world_size,
    std::span<const int> neighbors,
    std::span<const Key> local_keys,
    std::span<const int> local_owner_ranks,
    std::span<const int> rank_to_order,
    bool no_global_collectives,
    std::vector<gid_t>& out_global_ids,
    gid_t& out_global_count,
    int tag_base)
{
    static_assert(std::is_trivially_copyable_v<Key>, "assign_global_ordinals_with_neighbors requires trivially copyable Key");
    static_assert(std::is_trivially_copyable_v<KeyValuePair<Key>>, "assign_global_ordinals_with_neighbors requires trivially copyable KeyValuePair");
    if (local_keys.size() != local_owner_ranks.size()) {
        throw FEException("assign_global_ordinals_with_neighbors: key/owner size mismatch");
    }

    // Owned keys (unique, sorted).
    std::vector<Key> owned_keys;
    owned_keys.reserve(local_keys.size());
    for (std::size_t i = 0; i < local_keys.size(); ++i) {
        if (local_owner_ranks[i] == my_rank) {
            owned_keys.push_back(local_keys[i]);
        }
    }
    std::sort(owned_keys.begin(), owned_keys.end(), KeyLess{});
    owned_keys.erase(std::unique(owned_keys.begin(), owned_keys.end()), owned_keys.end());

    const gid_t local_owned = static_cast<gid_t>(owned_keys.size());
    gid_t gid_offset = 0;
    if (!no_global_collectives) {
        if (rank_to_order.empty()) {
            fe_mpi_check(MPI_Exscan(&local_owned, &gid_offset, 1, MPI_INT64_T, MPI_SUM, comm),
                         "MPI_Exscan (offset) in assign_global_ordinals_with_neighbors");
            if (my_rank == 0) {
                gid_offset = 0;
            }
        } else {
            if (rank_to_order.size() != static_cast<std::size_t>(world_size)) {
                throw FEException("assign_global_ordinals_with_neighbors: rank_to_order size mismatch");
            }
            std::vector<gid_t> owned_counts(static_cast<std::size_t>(world_size), 0);
            fe_mpi_check(MPI_Allgather(&local_owned, 1, MPI_INT64_T,
                                       owned_counts.data(), 1, MPI_INT64_T,
                                       comm),
                         "MPI_Allgather (owned counts) in assign_global_ordinals_with_neighbors");

            std::vector<int> order_to_rank(static_cast<std::size_t>(world_size), -1);
            for (int r = 0; r < world_size; ++r) {
                const int ord = rank_to_order[static_cast<std::size_t>(r)];
                if (ord < 0 || ord >= world_size) {
                    throw FEException("assign_global_ordinals_with_neighbors: invalid rank_to_order entry");
                }
                if (order_to_rank[static_cast<std::size_t>(ord)] != -1) {
                    throw FEException("assign_global_ordinals_with_neighbors: rank_to_order is not a permutation");
                }
                order_to_rank[static_cast<std::size_t>(ord)] = r;
            }

            const int my_order = rank_to_order[static_cast<std::size_t>(my_rank)];
            gid_offset = 0;
            for (int ord = 0; ord < my_order; ++ord) {
                const int rr = order_to_rank[static_cast<std::size_t>(ord)];
                if (rr < 0) {
                    throw FEException("assign_global_ordinals_with_neighbors: missing order_to_rank entry");
                }
                gid_offset += owned_counts[static_cast<std::size_t>(rr)];
            }
        }
    } else {
        gid_t global_sum = 0;
        if (rank_to_order.empty()) {
            mpi_exscan_sum_and_bcast_total_no_collectives(comm,
                                                         my_rank,
                                                         world_size,
                                                         local_owned,
                                                         gid_offset,
                                                         global_sum,
                                                         tag_base + 50);
        } else {
            mpi_exscan_sum_and_bcast_total_no_collectives_ordered(comm,
                                                                  my_rank,
                                                                  world_size,
                                                                  rank_to_order,
                                                                  local_owned,
                                                                  gid_offset,
                                                                  global_sum,
                                                                  tag_base + 50);
        }
        out_global_count = global_sum;
    }

    std::unordered_map<Key, gid_t, KeyHash> owned_map;
    owned_map.reserve(owned_keys.size());
    for (gid_t i = 0; i < local_owned; ++i) {
        owned_map.emplace(owned_keys[static_cast<std::size_t>(i)], gid_offset + i);
    }

    if (!no_global_collectives) {
        fe_mpi_check(MPI_Allreduce(&local_owned, &out_global_count, 1, MPI_INT64_T, MPI_SUM, comm),
                     "MPI_Allreduce (global count) in assign_global_ordinals_with_neighbors");
    }

    // Local key -> indices (normally unique, but allow duplicates).
    std::unordered_map<Key, std::vector<std::size_t>, KeyHash> local_key_to_indices;
    local_key_to_indices.reserve(local_keys.size());
    for (std::size_t i = 0; i < local_keys.size(); ++i) {
        local_key_to_indices[local_keys[i]].push_back(i);
    }

    out_global_ids.assign(local_keys.size(), gid_t{-1});
    for (std::size_t i = 0; i < local_keys.size(); ++i) {
        if (local_owner_ranks[i] != my_rank) continue;
        const auto it = owned_map.find(local_keys[i]);
        if (it == owned_map.end()) {
            throw FEException("assign_global_ordinals_with_neighbors: missing owned key in owned_map");
        }
        out_global_ids[i] = it->second;
    }

    if (neighbors.empty()) {
        for (std::size_t i = 0; i < out_global_ids.size(); ++i) {
            if (out_global_ids[i] < 0) {
                throw FEException("assign_global_ordinals_with_neighbors: missing assignment with empty neighbor list");
            }
        }
        return;
    }

    std::unordered_map<int, std::size_t> neighbor_to_index;
    neighbor_to_index.reserve(neighbors.size());
    for (std::size_t i = 0; i < neighbors.size(); ++i) {
        neighbor_to_index.emplace(neighbors[i], i);
    }

    // Requests: keys owned by neighbor that we need locally.
    std::vector<std::vector<Key>> requests(neighbors.size());
    for (std::size_t i = 0; i < local_keys.size(); ++i) {
        const int owner = local_owner_ranks[i];
        if (owner == my_rank) continue;
        const auto it = neighbor_to_index.find(owner);
        if (it == neighbor_to_index.end()) {
            throw FEException("assign_global_ordinals_with_neighbors: owner rank not present in neighbor list");
        }
        requests[it->second].push_back(local_keys[i]);
    }
    for (auto& list : requests) {
        std::sort(list.begin(), list.end(), KeyLess{});
        list.erase(std::unique(list.begin(), list.end()), list.end());
    }

    std::vector<std::vector<Key>> incoming_requests;
    mpi_neighbor_exchange_bytes(comm,
                               neighbors,
                               std::span<const std::vector<Key>>(requests.data(), requests.size()),
                               incoming_requests,
                               tag_base + 0,
                               tag_base + 1);

    // Build responses for keys requested from us.
    std::vector<std::vector<KeyValuePair<Key>>> responses(neighbors.size());
    for (std::size_t i = 0; i < neighbors.size(); ++i) {
        const auto& req = incoming_requests[i];
        auto& resp = responses[i];
        resp.reserve(req.size());
        for (const auto& key : req) {
            const auto it = owned_map.find(key);
            if (it == owned_map.end()) {
                throw FEException("assign_global_ordinals_with_neighbors: received request for key not owned by this rank");
            }
            resp.push_back(KeyValuePair<Key>{key, it->second});
        }
    }

    std::vector<std::vector<KeyValuePair<Key>>> incoming_responses;
    mpi_neighbor_exchange_bytes(comm,
                               neighbors,
                               std::span<const std::vector<KeyValuePair<Key>>>(responses.data(), responses.size()),
                               incoming_responses,
                               tag_base + 2,
                               tag_base + 3);

    // Apply received values.
    for (std::size_t i = 0; i < neighbors.size(); ++i) {
        const auto& resp = incoming_responses[i];
        for (const auto& kv : resp) {
            const auto it = local_key_to_indices.find(kv.key);
            if (it == local_key_to_indices.end()) {
                continue;
            }
            for (auto idx : it->second) {
                out_global_ids[idx] = kv.value;
            }
        }
    }

	    for (std::size_t i = 0; i < out_global_ids.size(); ++i) {
	        if (out_global_ids[i] < 0) {
	            throw FEException("assign_global_ordinals_with_neighbors: missing assignment for a local key");
	        }
		    }
		}

        // -----------------------------------------------------------------------------
        // Dense, process-count independent global ordinals via deterministic bucket/DHT.
        //
        // Defines a dense 0..N-1 ordinal set for the union of entity keys across ranks.
        // The global ordering is deterministic and independent of communicator size.
        //
        // Two deterministic orderings are supported (auto-selected by a deterministic
        // global unique-key cardinality estimate via HyperLogLog):
        //   - KeyOrdered: lexicographic key order (monotone bucketization in key),
        //   - HashBucketThenKey: stable hash-bucket order, then key order within bucket
        //     (better load-balance for large key counts).
        //
        // Ranks are assigned contiguous bucket ranges in (stable) rank-order so that
        // prefix offsets can be computed with an exscan.
        // -----------------------------------------------------------------------------

#ifndef SVMP_FE_DOF_DENSE_BUCKETS_LOG2
#define SVMP_FE_DOF_DENSE_BUCKETS_LOG2 16
#endif

        constexpr std::uint32_t kDenseBucketsLog2 = static_cast<std::uint32_t>(SVMP_FE_DOF_DENSE_BUCKETS_LOG2);
        static_assert(kDenseBucketsLog2 > 0 && kDenseBucketsLog2 < 31, "SVMP_FE_DOF_DENSE_BUCKETS_LOG2 must be in (0,31)");
        constexpr std::uint32_t kDenseBucketCount = (1u << kDenseBucketsLog2);

        static std::uint64_t ceil_div_u64(std::uint64_t a, std::uint64_t b) {
            return (b == 0u) ? 0u : (a + b - 1u) / b;
        }

        template <typename Key, typename KeyHash>
        static std::uint32_t dense_bucket_id_hash(const Key& key) noexcept {
            std::uint64_t h = static_cast<std::uint64_t>(KeyHash{}(key));
            h = mix_u64(h + kHashSalt);
            return static_cast<std::uint32_t>(h >> (64u - kDenseBucketsLog2));
        }

        enum class DenseOrdinalOrdering : std::uint8_t {
            HashBucketThenKey,  ///< Deterministic hash-bucket order, then key order within bucket.
            KeyOrdered          ///< Deterministic lexicographic key order (bucketization monotone in key).
        };

#ifndef SVMP_FE_DOF_DENSE_KEYORDER_MAX_UNIQUE
#define SVMP_FE_DOF_DENSE_KEYORDER_MAX_UNIQUE 50000
#endif

        constexpr double kDenseKeyOrderMaxUnique = static_cast<double>(SVMP_FE_DOF_DENSE_KEYORDER_MAX_UNIQUE);

        static std::uint64_t dense_key_primary_u64(gid_t key) noexcept { return static_cast<std::uint64_t>(key); }

        static std::uint64_t dense_key_primary_u64(const EdgeKey& key) noexcept { return static_cast<std::uint64_t>(key.a); }

        static std::uint64_t dense_key_primary_u64(const FaceKey& key) noexcept {
            // FaceKey ordering is (n, gids[0], gids[1], ...). Build a monotone scalar proxy:
            // - reserve top 8 bits for n,
            // - use the top 56 bits of gids[0] (shift-right 7) for coarse bucketing.
            const std::uint64_t n = static_cast<std::uint64_t>(key.n);
            const std::uint64_t g0 = static_cast<std::uint64_t>(key.gids[0]);
            return (n << 56u) | (g0 >> 7u);
        }

        template <typename Key, typename KeyHash>
        static DenseOrdinalOrdering choose_dense_ordinal_order(
            MPI_Comm comm,
            int my_rank,
            int world_size,
            std::span<const Key> local_keys,
            bool no_global_collectives,
            int tag_base) {
#if defined(SVMP_FE_DOF_DENSE_FORCE_KEYORDER)
            (void)comm;
            (void)my_rank;
            (void)world_size;
            (void)local_keys;
            (void)no_global_collectives;
            (void)tag_base;
            return DenseOrdinalOrdering::KeyOrdered;
#elif defined(SVMP_FE_DOF_DENSE_FORCE_HASHORDER)
            (void)comm;
            (void)my_rank;
            (void)world_size;
            (void)local_keys;
            (void)no_global_collectives;
            (void)tag_base;
            return DenseOrdinalOrdering::HashBucketThenKey;
#else
            const double est_unique = hll_estimate_global_unique<Key, KeyHash>(comm,
                                                                              my_rank,
                                                                              world_size,
                                                                              local_keys,
                                                                              no_global_collectives,
                                                                              tag_base);
            return (est_unique <= kDenseKeyOrderMaxUnique) ? DenseOrdinalOrdering::KeyOrdered
                                                           : DenseOrdinalOrdering::HashBucketThenKey;
#endif
        }

        template <typename Key>
        struct DenseKeyOrderedBucket {
            gid_t min_primary{0};
            gid_t max_primary{0};

            std::uint32_t operator()(const Key& key) const noexcept {
                if (max_primary <= min_primary) {
                    return 0u;
                }
                const std::uint64_t primary = dense_key_primary_u64(key);
                const std::uint64_t minp = static_cast<std::uint64_t>(min_primary);
                const std::uint64_t maxp = static_cast<std::uint64_t>(max_primary);
                std::uint32_t out = 0u;
#if defined(__SIZEOF_INT128__)
                __extension__ typedef unsigned __int128 uint128_t;
                const uint128_t range = static_cast<uint128_t>(maxp - minp) + 1u;
                const uint128_t off = static_cast<uint128_t>(primary - minp);
                const uint128_t b = (off * static_cast<uint128_t>(kDenseBucketCount)) / range;
                out = static_cast<std::uint32_t>(b);
#else
                const long double range = static_cast<long double>(maxp - minp) + 1.0L;
                const long double off = static_cast<long double>(primary - minp);
                const long double b = (off * static_cast<long double>(kDenseBucketCount)) / range;
                out = static_cast<std::uint32_t>(b);
#endif
                return (out < kDenseBucketCount) ? out : (kDenseBucketCount - 1u);
            }
        };

        template <typename Key, typename KeyHash>
        struct DenseHashBucket {
            std::uint32_t operator()(const Key& key) const noexcept {
                return dense_bucket_id_hash<Key, KeyHash>(key);
            }
        };

        template <typename Key, typename KeyHash, typename KeyLess, typename BucketId>
        static void assign_dense_global_ordinals_dht(
            MPI_Comm comm,
            int my_rank,
            int world_size,
            std::span<const int> rank_to_order,
            std::span<const Key> local_keys,
            const BucketId& bucket_id,
            bool no_global_collectives,
            std::vector<gid_t>& out_global_ids,
            gid_t& out_global_count,
            int tag_base)
        {
            static_assert(std::is_trivially_copyable_v<Key>, "assign_dense_global_ordinals_dht requires trivially copyable Key");
            if (!rank_to_order.empty() && rank_to_order.size() != static_cast<std::size_t>(world_size)) {
                throw FEException("assign_dense_global_ordinals_dht: rank_to_order size mismatch");
            }
            if (world_size <= 0) {
                throw FEException("assign_dense_global_ordinals_dht: invalid world_size");
            }

            std::vector<int> order_to_rank(static_cast<std::size_t>(world_size), -1);
            if (rank_to_order.empty()) {
                for (int r = 0; r < world_size; ++r) {
                    order_to_rank[static_cast<std::size_t>(r)] = r;
                }
            } else {
                for (int r = 0; r < world_size; ++r) {
                    const int ord = rank_to_order[static_cast<std::size_t>(r)];
                    if (ord < 0 || ord >= world_size) {
                        throw FEException("assign_dense_global_ordinals_dht: invalid rank_to_order entry");
                    }
                    if (order_to_rank[static_cast<std::size_t>(ord)] != -1) {
                        throw FEException("assign_dense_global_ordinals_dht: rank_to_order is not a permutation");
                    }
                    order_to_rank[static_cast<std::size_t>(ord)] = r;
                }
            }

            const int my_order = rank_to_order.empty() ? my_rank : rank_to_order[static_cast<std::size_t>(my_rank)];
            if (my_order < 0 || my_order >= world_size) {
                throw FEException("assign_dense_global_ordinals_dht: invalid my_order");
            }

            // Build requests: local key -> owning bucket-rank.
            std::vector<std::vector<Key>> send_keys(static_cast<std::size_t>(world_size));
            std::vector<std::vector<std::size_t>> send_indices(static_cast<std::size_t>(world_size));
            for (std::size_t i = 0; i < local_keys.size(); ++i) {
                const auto b = bucket_id(local_keys[i]);
                const int owner_ord = static_cast<int>((static_cast<std::uint64_t>(b) * static_cast<std::uint64_t>(world_size)) /
                                                       static_cast<std::uint64_t>(kDenseBucketCount));
                if (owner_ord < 0 || owner_ord >= world_size) {
                    throw FEException("assign_dense_global_ordinals_dht: computed owner order out of range");
                }
                const int owner_rank = order_to_rank[static_cast<std::size_t>(owner_ord)];
                if (owner_rank < 0 || owner_rank >= world_size) {
                    throw FEException("assign_dense_global_ordinals_dht: computed owner rank out of range");
                }
                send_keys[static_cast<std::size_t>(owner_rank)].push_back(local_keys[i]);
                send_indices[static_cast<std::size_t>(owner_rank)].push_back(i);
            }

            std::vector<int> send_counts_bytes(static_cast<std::size_t>(world_size), 0);
            std::vector<int> send_displs_bytes(static_cast<std::size_t>(world_size) + 1u, 0);
            for (int r = 0; r < world_size; ++r) {
                const auto bytes = send_keys[static_cast<std::size_t>(r)].size() * sizeof(Key);
                if (bytes > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                    throw FEException("assign_dense_global_ordinals_dht: send buffer too large for MPI int counts");
                }
                send_counts_bytes[static_cast<std::size_t>(r)] = static_cast<int>(bytes);
                send_displs_bytes[static_cast<std::size_t>(r) + 1u] =
                    send_displs_bytes[static_cast<std::size_t>(r)] + send_counts_bytes[static_cast<std::size_t>(r)];
            }

            const std::size_t total_send_bytes = static_cast<std::size_t>(send_displs_bytes[static_cast<std::size_t>(world_size)]);
            if (total_send_bytes % sizeof(Key) != 0u) {
                throw FEException("assign_dense_global_ordinals_dht: send buffer size misaligned");
            }
            std::vector<Key> send_flat(total_send_bytes / sizeof(Key));
            for (int r = 0; r < world_size; ++r) {
                auto& list = send_keys[static_cast<std::size_t>(r)];
                const std::size_t elem_disp = static_cast<std::size_t>(send_displs_bytes[static_cast<std::size_t>(r)]) / sizeof(Key);
                std::copy(list.begin(), list.end(), send_flat.begin() + static_cast<std::ptrdiff_t>(elem_disp));
            }

            std::vector<int> recv_counts_bytes(static_cast<std::size_t>(world_size), 0);
            if (!no_global_collectives) {
                fe_mpi_check(MPI_Alltoall(send_counts_bytes.data(), 1, MPI_INT,
                                          recv_counts_bytes.data(), 1, MPI_INT,
                                          comm),
                             "MPI_Alltoall (req counts) in assign_dense_global_ordinals_dht");
            } else {
                for (int r = 0; r < world_size; ++r) {
                    if (r == my_rank) {
                        recv_counts_bytes[static_cast<std::size_t>(r)] = send_counts_bytes[static_cast<std::size_t>(r)];
                        continue;
                    }
                    fe_mpi_check(MPI_Sendrecv(&send_counts_bytes[static_cast<std::size_t>(r)],
                                              1,
                                              MPI_INT,
                                              r,
                                              tag_base + 0,
                                              &recv_counts_bytes[static_cast<std::size_t>(r)],
                                              1,
                                              MPI_INT,
                                              r,
                                              tag_base + 0,
                                              comm,
                                              MPI_STATUS_IGNORE),
                                 "MPI_Sendrecv (req counts) in assign_dense_global_ordinals_dht");
                }
            }

            std::vector<int> recv_displs_bytes(static_cast<std::size_t>(world_size) + 1u, 0);
            for (int r = 0; r < world_size; ++r) {
                const int bytes = recv_counts_bytes[static_cast<std::size_t>(r)];
                if (bytes < 0) {
                    throw FEException("assign_dense_global_ordinals_dht: negative receive byte count");
                }
                if ((static_cast<std::size_t>(bytes) % sizeof(Key)) != 0u) {
                    throw FEException("assign_dense_global_ordinals_dht: receive buffer size misaligned");
                }
                recv_displs_bytes[static_cast<std::size_t>(r) + 1u] =
                    recv_displs_bytes[static_cast<std::size_t>(r)] + bytes;
            }

            const std::size_t total_recv_bytes = static_cast<std::size_t>(recv_displs_bytes[static_cast<std::size_t>(world_size)]);
            if (total_recv_bytes % sizeof(Key) != 0u) {
                throw FEException("assign_dense_global_ordinals_dht: receive buffer size misaligned");
            }
            std::vector<Key> recv_flat(total_recv_bytes / sizeof(Key));

            if (!no_global_collectives) {
                fe_mpi_check(MPI_Alltoallv(send_flat.empty() ? nullptr : static_cast<const void*>(send_flat.data()),
                                           send_counts_bytes.data(),
                                           send_displs_bytes.data(),
                                           MPI_BYTE,
                                           recv_flat.empty() ? nullptr : static_cast<void*>(recv_flat.data()),
                                           recv_counts_bytes.data(),
                                           recv_displs_bytes.data(),
                                           MPI_BYTE,
                                           comm),
                             "MPI_Alltoallv (req data) in assign_dense_global_ordinals_dht");
            } else {
                std::vector<MPI_Request> reqs;
                reqs.reserve(static_cast<std::size_t>(world_size) * 2u);

                // Self-copy (rank -> rank) to avoid self-send/recv.
                const int self_send_bytes = send_counts_bytes[static_cast<std::size_t>(my_rank)];
                const int self_recv_bytes = recv_counts_bytes[static_cast<std::size_t>(my_rank)];
                if (self_send_bytes != self_recv_bytes) {
                    throw FEException("assign_dense_global_ordinals_dht: self send/recv byte mismatch");
                }
                if (self_send_bytes > 0) {
                    const auto* src = reinterpret_cast<const unsigned char*>(send_flat.data()) +
                                      send_displs_bytes[static_cast<std::size_t>(my_rank)];
                    auto* dst = reinterpret_cast<unsigned char*>(recv_flat.data()) +
                                recv_displs_bytes[static_cast<std::size_t>(my_rank)];
                    std::memcpy(dst, src, static_cast<std::size_t>(self_send_bytes));
                }

                for (int r = 0; r < world_size; ++r) {
                    if (r == my_rank) continue;
                    MPI_Request req{};
                    const int nbytes = recv_counts_bytes[static_cast<std::size_t>(r)];
                    if (nbytes > 0) {
                        auto* dst = reinterpret_cast<unsigned char*>(recv_flat.data()) +
                                    recv_displs_bytes[static_cast<std::size_t>(r)];
                        fe_mpi_check(MPI_Irecv(dst, nbytes, MPI_BYTE, r, tag_base + 1, comm, &req),
                                     "MPI_Irecv (req data) in assign_dense_global_ordinals_dht");
                        reqs.push_back(req);
                    }
                }
                for (int r = 0; r < world_size; ++r) {
                    if (r == my_rank) continue;
                    MPI_Request req{};
                    const int nbytes = send_counts_bytes[static_cast<std::size_t>(r)];
                    if (nbytes > 0) {
                        const auto* src = reinterpret_cast<const unsigned char*>(send_flat.data()) +
                                          send_displs_bytes[static_cast<std::size_t>(r)];
                        fe_mpi_check(MPI_Isend(src, nbytes, MPI_BYTE, r, tag_base + 1, comm, &req),
                                     "MPI_Isend (req data) in assign_dense_global_ordinals_dht");
                        reqs.push_back(req);
                    }
                }

                if (!reqs.empty()) {
                    fe_mpi_check(MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE),
                                 "MPI_Waitall (req data) in assign_dense_global_ordinals_dht");
                }
            }

            // Compute owned bucket range for this rank in bucket order.
            const std::uint64_t begin_bucket =
                ceil_div_u64(static_cast<std::uint64_t>(my_order) * static_cast<std::uint64_t>(kDenseBucketCount),
                             static_cast<std::uint64_t>(world_size));
            const std::uint64_t end_bucket =
                ceil_div_u64(static_cast<std::uint64_t>(my_order + 1) * static_cast<std::uint64_t>(kDenseBucketCount),
                             static_cast<std::uint64_t>(world_size));

            struct BucketedKey {
                std::uint32_t bucket{0};
                Key key{};
            };

            std::vector<BucketedKey> bucketed;
            bucketed.reserve(recv_flat.size());
            for (const auto& key : recv_flat) {
                const auto b = bucket_id(key);
                if (b < begin_bucket || b >= end_bucket) {
                    throw FEException("assign_dense_global_ordinals_dht: received key outside owned bucket range");
                }
                bucketed.push_back(BucketedKey{b, key});
            }

            std::sort(bucketed.begin(), bucketed.end(), [](const BucketedKey& a, const BucketedKey& b) {
                if (a.bucket != b.bucket) return a.bucket < b.bucket;
                return KeyLess{}(a.key, b.key);
            });
            bucketed.erase(std::unique(bucketed.begin(),
                                       bucketed.end(),
                                       [](const BucketedKey& a, const BucketedKey& b) {
                                           return a.bucket == b.bucket && a.key == b.key;
                                       }),
                           bucketed.end());

            const gid_t local_unique = static_cast<gid_t>(bucketed.size());
            gid_t gid_offset = 0;
            if (!no_global_collectives) {
                if (rank_to_order.empty()) {
                    fe_mpi_check(MPI_Exscan(&local_unique, &gid_offset, 1, MPI_INT64_T, MPI_SUM, comm),
                                 "MPI_Exscan (offset) in assign_dense_global_ordinals_dht");
                    if (my_rank == 0) {
                        gid_offset = 0;
                    }
                } else {
                    std::vector<gid_t> counts(static_cast<std::size_t>(world_size), 0);
                    fe_mpi_check(MPI_Allgather(&local_unique, 1, MPI_INT64_T,
                                               counts.data(), 1, MPI_INT64_T,
                                               comm),
                                 "MPI_Allgather (owned counts) in assign_dense_global_ordinals_dht");
                    gid_offset = 0;
                    for (int ord = 0; ord < my_order; ++ord) {
                        const int rr = order_to_rank[static_cast<std::size_t>(ord)];
                        if (rr < 0) {
                            throw FEException("assign_dense_global_ordinals_dht: missing order_to_rank entry");
                        }
                        gid_offset += counts[static_cast<std::size_t>(rr)];
                    }
                }
                fe_mpi_check(MPI_Allreduce(&local_unique, &out_global_count, 1, MPI_INT64_T, MPI_SUM, comm),
                             "MPI_Allreduce (global count) in assign_dense_global_ordinals_dht");
            } else {
                gid_t global_sum = 0;
                if (rank_to_order.empty()) {
                    mpi_exscan_sum_and_bcast_total_no_collectives(comm,
                                                                 my_rank,
                                                                 world_size,
                                                                 local_unique,
                                                                 gid_offset,
                                                                 global_sum,
                                                                 tag_base + 20);
                } else {
                    mpi_exscan_sum_and_bcast_total_no_collectives_ordered(comm,
                                                                          my_rank,
                                                                          world_size,
                                                                          rank_to_order,
                                                                          local_unique,
                                                                          gid_offset,
                                                                          global_sum,
                                                                          tag_base + 20);
                }
                out_global_count = global_sum;
            }

            std::unordered_map<Key, gid_t, KeyHash> id_map;
            id_map.reserve(bucketed.size());
            for (gid_t i = 0; i < static_cast<gid_t>(bucketed.size()); ++i) {
                id_map.emplace(bucketed[static_cast<std::size_t>(i)].key, gid_offset + i);
            }

            // Build response IDs for each requester rank, matching the order of keys received from that rank.
            std::vector<int> resp_send_counts_bytes(static_cast<std::size_t>(world_size), 0);
            std::vector<int> resp_send_displs_bytes(static_cast<std::size_t>(world_size) + 1u, 0);
            for (int r = 0; r < world_size; ++r) {
                const int nkeys_bytes = recv_counts_bytes[static_cast<std::size_t>(r)];
                const auto nkeys = static_cast<std::size_t>(nkeys_bytes) / sizeof(Key);
                const std::size_t bytes = nkeys * sizeof(gid_t);
                if (bytes > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                    throw FEException("assign_dense_global_ordinals_dht: response buffer too large for MPI int counts");
                }
                resp_send_counts_bytes[static_cast<std::size_t>(r)] = static_cast<int>(bytes);
                resp_send_displs_bytes[static_cast<std::size_t>(r) + 1u] =
                    resp_send_displs_bytes[static_cast<std::size_t>(r)] + resp_send_counts_bytes[static_cast<std::size_t>(r)];
            }

            const std::size_t resp_send_bytes =
                static_cast<std::size_t>(resp_send_displs_bytes[static_cast<std::size_t>(world_size)]);
            if (resp_send_bytes % sizeof(gid_t) != 0u) {
                throw FEException("assign_dense_global_ordinals_dht: response send buffer size misaligned");
            }
            std::vector<gid_t> resp_send_flat(resp_send_bytes / sizeof(gid_t));

            for (int r = 0; r < world_size; ++r) {
                const auto keys_offset = static_cast<std::size_t>(recv_displs_bytes[static_cast<std::size_t>(r)]) / sizeof(Key);
                const auto nkeys = static_cast<std::size_t>(recv_counts_bytes[static_cast<std::size_t>(r)]) / sizeof(Key);
                const auto ids_offset = static_cast<std::size_t>(resp_send_displs_bytes[static_cast<std::size_t>(r)]) / sizeof(gid_t);
                for (std::size_t i = 0; i < nkeys; ++i) {
                    const Key& key = recv_flat[keys_offset + i];
                    const auto it = id_map.find(key);
                    if (it == id_map.end()) {
                        throw FEException("assign_dense_global_ordinals_dht: missing ID for received key");
                    }
                    resp_send_flat[ids_offset + i] = it->second;
                }
            }

            // Receive buffer: IDs for keys we requested (in the same send order per destination rank).
            std::vector<int> resp_recv_counts_bytes(static_cast<std::size_t>(world_size), 0);
            std::vector<int> resp_recv_displs_bytes(static_cast<std::size_t>(world_size) + 1u, 0);
            for (int r = 0; r < world_size; ++r) {
                const auto nkeys = static_cast<std::size_t>(send_counts_bytes[static_cast<std::size_t>(r)]) / sizeof(Key);
                const std::size_t bytes = nkeys * sizeof(gid_t);
                if (bytes > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                    throw FEException("assign_dense_global_ordinals_dht: response receive buffer too large for MPI int counts");
                }
                resp_recv_counts_bytes[static_cast<std::size_t>(r)] = static_cast<int>(bytes);
                resp_recv_displs_bytes[static_cast<std::size_t>(r) + 1u] =
                    resp_recv_displs_bytes[static_cast<std::size_t>(r)] + resp_recv_counts_bytes[static_cast<std::size_t>(r)];
            }

            const std::size_t resp_recv_bytes =
                static_cast<std::size_t>(resp_recv_displs_bytes[static_cast<std::size_t>(world_size)]);
            if (resp_recv_bytes % sizeof(gid_t) != 0u) {
                throw FEException("assign_dense_global_ordinals_dht: response recv buffer size misaligned");
            }
            std::vector<gid_t> resp_recv_flat(resp_recv_bytes / sizeof(gid_t));

            if (!no_global_collectives) {
                fe_mpi_check(MPI_Alltoallv(resp_send_flat.empty() ? nullptr : static_cast<const void*>(resp_send_flat.data()),
                                           resp_send_counts_bytes.data(),
                                           resp_send_displs_bytes.data(),
                                           MPI_BYTE,
                                           resp_recv_flat.empty() ? nullptr : static_cast<void*>(resp_recv_flat.data()),
                                           resp_recv_counts_bytes.data(),
                                           resp_recv_displs_bytes.data(),
                                           MPI_BYTE,
                                           comm),
                             "MPI_Alltoallv (resp data) in assign_dense_global_ordinals_dht");
            } else {
                std::vector<MPI_Request> reqs;
                reqs.reserve(static_cast<std::size_t>(world_size) * 2u);

                const int self_send_bytes2 = resp_send_counts_bytes[static_cast<std::size_t>(my_rank)];
                const int self_recv_bytes2 = resp_recv_counts_bytes[static_cast<std::size_t>(my_rank)];
                if (self_send_bytes2 != self_recv_bytes2) {
                    throw FEException("assign_dense_global_ordinals_dht: self response send/recv byte mismatch");
                }
                if (self_send_bytes2 > 0) {
                    const auto* src = reinterpret_cast<const unsigned char*>(resp_send_flat.data()) +
                                      resp_send_displs_bytes[static_cast<std::size_t>(my_rank)];
                    auto* dst = reinterpret_cast<unsigned char*>(resp_recv_flat.data()) +
                                resp_recv_displs_bytes[static_cast<std::size_t>(my_rank)];
                    std::memcpy(dst, src, static_cast<std::size_t>(self_send_bytes2));
                }

                for (int r = 0; r < world_size; ++r) {
                    if (r == my_rank) continue;
                    MPI_Request req{};
                    const int nbytes = resp_recv_counts_bytes[static_cast<std::size_t>(r)];
                    if (nbytes > 0) {
                        auto* dst = reinterpret_cast<unsigned char*>(resp_recv_flat.data()) +
                                    resp_recv_displs_bytes[static_cast<std::size_t>(r)];
                        fe_mpi_check(MPI_Irecv(dst, nbytes, MPI_BYTE, r, tag_base + 3, comm, &req),
                                     "MPI_Irecv (resp data) in assign_dense_global_ordinals_dht");
                        reqs.push_back(req);
                    }
                }
                for (int r = 0; r < world_size; ++r) {
                    if (r == my_rank) continue;
                    MPI_Request req{};
                    const int nbytes = resp_send_counts_bytes[static_cast<std::size_t>(r)];
                    if (nbytes > 0) {
                        const auto* src = reinterpret_cast<const unsigned char*>(resp_send_flat.data()) +
                                          resp_send_displs_bytes[static_cast<std::size_t>(r)];
                        fe_mpi_check(MPI_Isend(src, nbytes, MPI_BYTE, r, tag_base + 3, comm, &req),
                                     "MPI_Isend (resp data) in assign_dense_global_ordinals_dht");
                        reqs.push_back(req);
                    }
                }

                if (!reqs.empty()) {
                    fe_mpi_check(MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE),
                                 "MPI_Waitall (resp data) in assign_dense_global_ordinals_dht");
                }
            }

            out_global_ids.assign(local_keys.size(), gid_t{-1});
            for (int r = 0; r < world_size; ++r) {
                const auto ids_offset = static_cast<std::size_t>(resp_recv_displs_bytes[static_cast<std::size_t>(r)]) / sizeof(gid_t);
                const auto nids = static_cast<std::size_t>(resp_recv_counts_bytes[static_cast<std::size_t>(r)]) / sizeof(gid_t);
                const auto& idx_list = send_indices[static_cast<std::size_t>(r)];
                if (idx_list.size() != nids) {
                    throw FEException("assign_dense_global_ordinals_dht: response size mismatch for destination rank");
                }
                for (std::size_t i = 0; i < nids; ++i) {
                    out_global_ids[idx_list[i]] = resp_recv_flat[ids_offset + i];
                }
            }

            for (std::size_t i = 0; i < out_global_ids.size(); ++i) {
                if (out_global_ids[i] < 0) {
                    throw FEException("assign_dense_global_ordinals_dht: missing assignment for a local key");
                }
            }
        }

        template <typename Key, typename KeyHash, typename KeyLess, typename BucketId>
        static void assign_dense_global_ordinals_compact(
            MPI_Comm comm,
            int my_rank,
            int world_size,
            std::span<const int> rank_to_order,
            std::span<const Key> local_keys,
            const BucketId& bucket_id,
            bool no_global_collectives,
            std::vector<gid_t>& out_global_ids,
            gid_t& out_global_count,
            int tag_base)
        {
#if MPI_VERSION >= 3
            if (!no_global_collectives) {
                MPI_Comm node_comm = MPI_COMM_NULL;
                fe_mpi_check(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, my_rank, MPI_INFO_NULL, &node_comm),
                             "MPI_Comm_split_type (shared) in assign_dense_global_ordinals_compact");

                int node_rank = 0;
                int node_size = 1;
                fe_mpi_check(MPI_Comm_rank(node_comm, &node_rank),
                             "MPI_Comm_rank (node_comm) in assign_dense_global_ordinals_compact");
                fe_mpi_check(MPI_Comm_size(node_comm, &node_size),
                             "MPI_Comm_size (node_comm) in assign_dense_global_ordinals_compact");

                const int color = (node_rank == 0) ? 0 : MPI_UNDEFINED;
                MPI_Comm leader_comm = MPI_COMM_NULL;
                fe_mpi_check(MPI_Comm_split(comm, color, my_rank, &leader_comm),
                             "MPI_Comm_split (leader_comm) in assign_dense_global_ordinals_compact");

                // Gather local keys to node root (rank 0 in node_comm).
                const std::size_t local_bytes_sz = local_keys.size() * sizeof(Key);
                if (local_bytes_sz > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                    throw FEException("assign_dense_global_ordinals_compact: local key buffer too large for MPI int counts");
                }
                const int local_bytes = static_cast<int>(local_bytes_sz);

                std::vector<int> recv_counts_bytes;
                if (node_rank == 0) {
                    recv_counts_bytes.assign(static_cast<std::size_t>(node_size), 0);
                }

                fe_mpi_check(MPI_Gather(&local_bytes,
                                        1,
                                        MPI_INT,
                                        (node_rank == 0) ? recv_counts_bytes.data() : nullptr,
                                        1,
                                        MPI_INT,
                                        0,
                                        node_comm),
                             "MPI_Gather (node key counts) in assign_dense_global_ordinals_compact");

                std::vector<int> recv_displs_bytes;
                std::vector<Key> gathered;
                if (node_rank == 0) {
                    recv_displs_bytes.assign(static_cast<std::size_t>(node_size) + 1u, 0);
                    for (int r = 0; r < node_size; ++r) {
                        const int nbytes = recv_counts_bytes[static_cast<std::size_t>(r)];
                        if (nbytes < 0) {
                            throw FEException("assign_dense_global_ordinals_compact: negative node receive byte count");
                        }
                        if ((static_cast<std::size_t>(nbytes) % sizeof(Key)) != 0u) {
                            throw FEException("assign_dense_global_ordinals_compact: node gather buffer misaligned");
                        }
                        recv_displs_bytes[static_cast<std::size_t>(r) + 1u] =
                            recv_displs_bytes[static_cast<std::size_t>(r)] + nbytes;
                    }
                    const std::size_t total_bytes =
                        static_cast<std::size_t>(recv_displs_bytes[static_cast<std::size_t>(node_size)]);
                    gathered.resize(total_bytes / sizeof(Key));
                }

                fe_mpi_check(MPI_Gatherv(local_keys.empty() ? nullptr : static_cast<const void*>(local_keys.data()),
                                         local_bytes,
                                         MPI_BYTE,
                                         (node_rank == 0 && !gathered.empty()) ? static_cast<void*>(gathered.data()) : nullptr,
                                         (node_rank == 0) ? recv_counts_bytes.data() : nullptr,
                                         (node_rank == 0) ? recv_displs_bytes.data() : nullptr,
                                         MPI_BYTE,
                                         0,
                                         node_comm),
                             "MPI_Gatherv (node key data) in assign_dense_global_ordinals_compact");

                std::vector<gid_t> local_ids(local_keys.size(), gid_t{-1});
                gid_t global_count = 0;

                if (node_rank == 0) {
                    // Unique keys on this node (cheap intra-node compaction).
                    std::vector<Key> unique_node_keys = gathered;
                    std::sort(unique_node_keys.begin(), unique_node_keys.end(), KeyLess{});
                    unique_node_keys.erase(std::unique(unique_node_keys.begin(), unique_node_keys.end()), unique_node_keys.end());

                    // Node roots collectively compute dense ordinals for their (node-unique) key sets.
                    std::vector<gid_t> unique_node_ids;
                    gid_t n_global = 0;
                    if (leader_comm == MPI_COMM_NULL) {
                        throw FEException("assign_dense_global_ordinals_compact: leader_comm is null on node root");
                    }
                    int leader_rank = 0;
                    int leader_size = 1;
                    fe_mpi_check(MPI_Comm_rank(leader_comm, &leader_rank),
                                 "MPI_Comm_rank (leader_comm) in assign_dense_global_ordinals_compact");
                    fe_mpi_check(MPI_Comm_size(leader_comm, &leader_size),
                                 "MPI_Comm_size (leader_comm) in assign_dense_global_ordinals_compact");

                    assign_dense_global_ordinals_dht<Key, KeyHash, KeyLess>(leader_comm,
                                                                           leader_rank,
                                                                           leader_size,
                                                                           /*rank_to_order=*/{},
                                                                           unique_node_keys,
                                                                           bucket_id,
                                                                           /*no_global_collectives=*/false,
                                                                           unique_node_ids,
                                                                           n_global,
                                                                           tag_base);
                    global_count = n_global;

                    std::unordered_map<Key, gid_t, KeyHash> id_map;
                    id_map.reserve(unique_node_keys.size());
                    for (std::size_t i = 0; i < unique_node_keys.size(); ++i) {
                        id_map.emplace(unique_node_keys[i], unique_node_ids[i]);
                    }

                    // Prepare per-node-rank scatter buffer matching the original key order per rank.
                    std::vector<int> send_counts(static_cast<std::size_t>(node_size), 0);
                    std::vector<int> send_displs(static_cast<std::size_t>(node_size) + 1u, 0);
                    for (int r = 0; r < node_size; ++r) {
                        const int nbytes = recv_counts_bytes[static_cast<std::size_t>(r)];
                        const auto nkeys = static_cast<std::size_t>(nbytes) / sizeof(Key);
                        if (nkeys > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                            throw FEException("assign_dense_global_ordinals_compact: node scatter count too large for MPI int counts");
                        }
                        send_counts[static_cast<std::size_t>(r)] = static_cast<int>(nkeys);
                        send_displs[static_cast<std::size_t>(r) + 1u] =
                            send_displs[static_cast<std::size_t>(r)] + send_counts[static_cast<std::size_t>(r)];
                    }

                    const std::size_t total_send = static_cast<std::size_t>(send_displs[static_cast<std::size_t>(node_size)]);
                    std::vector<gid_t> send_ids(total_send, gid_t{-1});
                    for (int r = 0; r < node_size; ++r) {
                        const auto key_off = static_cast<std::size_t>(recv_displs_bytes[static_cast<std::size_t>(r)]) / sizeof(Key);
                        const auto nkeys = static_cast<std::size_t>(recv_counts_bytes[static_cast<std::size_t>(r)]) / sizeof(Key);
                        const auto id_off = static_cast<std::size_t>(send_displs[static_cast<std::size_t>(r)]);
                        for (std::size_t i = 0; i < nkeys; ++i) {
                            const Key& key = gathered[key_off + i];
                            const auto it = id_map.find(key);
                            if (it == id_map.end()) {
                                throw FEException("assign_dense_global_ordinals_compact: missing ID while building node scatter");
                            }
                            send_ids[id_off + i] = it->second;
                        }
                    }

                    fe_mpi_check(MPI_Scatterv(send_ids.empty() ? nullptr : send_ids.data(),
                                              send_counts.data(),
                                              send_displs.data(),
                                              MPI_INT64_T,
                                              local_ids.empty() ? nullptr : local_ids.data(),
                                              static_cast<int>(local_ids.size()),
                                              MPI_INT64_T,
                                              0,
                                              node_comm),
                                 "MPI_Scatterv (node ids) in assign_dense_global_ordinals_compact");
                } else {
                    fe_mpi_check(MPI_Scatterv(nullptr,
                                              nullptr,
                                              nullptr,
                                              MPI_INT64_T,
                                              local_ids.empty() ? nullptr : local_ids.data(),
                                              static_cast<int>(local_ids.size()),
                                              MPI_INT64_T,
                                              0,
                                              node_comm),
                                 "MPI_Scatterv (node ids) in assign_dense_global_ordinals_compact (non-root)");
                }

                fe_mpi_check(MPI_Bcast(&global_count, 1, MPI_INT64_T, 0, node_comm),
                             "MPI_Bcast (global count) in assign_dense_global_ordinals_compact");

                // Clean up temporary communicators before returning (safe even if MPI_Finalize is called later).
                if (leader_comm != MPI_COMM_NULL && leader_comm != MPI_COMM_WORLD && leader_comm != MPI_COMM_SELF) {
                    MPI_Comm_free(&leader_comm);
                    leader_comm = MPI_COMM_NULL;
                }
                if (node_comm != MPI_COMM_NULL && node_comm != MPI_COMM_WORLD && node_comm != MPI_COMM_SELF) {
                    MPI_Comm_free(&node_comm);
                    node_comm = MPI_COMM_NULL;
                }

                out_global_ids = std::move(local_ids);
                out_global_count = global_count;
                return;
            }
#endif

            assign_dense_global_ordinals_dht<Key, KeyHash, KeyLess>(comm,
                                                                   my_rank,
                                                                   world_size,
                                                                   rank_to_order,
                                                                   local_keys,
                                                                   bucket_id,
                                                                   no_global_collectives,
                                                                   out_global_ids,
                                                                   out_global_count,
                                                                   tag_base);
        }

        template <typename Key, typename KeyHash, typename KeyLess>
        static void assign_dense_global_ordinals_compact_auto(
            MPI_Comm comm,
            int my_rank,
            int world_size,
            std::span<const int> rank_to_order,
            std::span<const Key> local_keys,
            bool no_global_collectives,
            std::vector<gid_t>& out_global_ids,
            gid_t& out_global_count,
            int tag_base)
        {
            const DenseOrdinalOrdering ordering = choose_dense_ordinal_order<Key, KeyHash>(comm,
                                                                                           my_rank,
                                                                                           world_size,
                                                                                           local_keys,
                                                                                           no_global_collectives,
                                                                                           tag_base + 900);

            if (ordering == DenseOrdinalOrdering::KeyOrdered) {
                gid_t local_min = std::numeric_limits<gid_t>::max();
                gid_t local_max = std::numeric_limits<gid_t>::lowest();
                for (const auto& key : local_keys) {
                    const gid_t p = static_cast<gid_t>(dense_key_primary_u64(key));
                    local_min = std::min(local_min, p);
                    local_max = std::max(local_max, p);
                }

                gid_t global_min = local_min;
                gid_t global_max = local_max;
                if (!no_global_collectives) {
                    fe_mpi_check(MPI_Allreduce(&local_min, &global_min, 1, MPI_INT64_T, MPI_MIN, comm),
                                 "MPI_Allreduce (min primary) in assign_dense_global_ordinals_compact_auto");
                    fe_mpi_check(MPI_Allreduce(&local_max, &global_max, 1, MPI_INT64_T, MPI_MAX, comm),
                                 "MPI_Allreduce (max primary) in assign_dense_global_ordinals_compact_auto");
                } else {
                    global_min = mpi_allreduce_min_no_collectives(comm, my_rank, world_size, local_min, tag_base + 920);
                    global_max = mpi_allreduce_max_no_collectives(comm, my_rank, world_size, local_max, tag_base + 930);
                }

                DenseKeyOrderedBucket<Key> bucket_id;
                bucket_id.min_primary = global_min;
                bucket_id.max_primary = global_max;

                assign_dense_global_ordinals_compact<Key, KeyHash, KeyLess>(comm,
                                                                            my_rank,
                                                                            world_size,
                                                                            rank_to_order,
                                                                            local_keys,
                                                                            bucket_id,
                                                                            no_global_collectives,
                                                                            out_global_ids,
                                                                            out_global_count,
                                                                            tag_base);
                return;
            }

            const DenseHashBucket<Key, KeyHash> bucket_id;
            assign_dense_global_ordinals_compact<Key, KeyHash, KeyLess>(comm,
                                                                        my_rank,
                                                                        world_size,
                                                                        rank_to_order,
                                                                        local_keys,
                                                                        bucket_id,
                                                                        no_global_collectives,
                                                                        out_global_ids,
                                                                        out_global_count,
                                                                        tag_base);
        }

			template <typename Key>
			struct TouchCandidate {
			    Key key{};
		    gid_t cell_gid_candidate{-1};
	    std::int32_t cell_owner_candidate{-1};
	};

	template <typename Key, typename KeyHash>
	static void assign_owners_with_neighbors(
	    MPI_Comm comm,
	    int my_rank,
	    int world_size,
	    std::span<const int> neighbors,
	    OwnershipStrategy ownership,
	    std::span<const Key> local_keys,
	    std::span<const int> local_touch,
	    std::span<const gid_t> cell_gid_candidate,
	    std::span<const int> cell_owner_candidate,
	    std::span<const int> rank_to_order,
	    std::vector<int>& out_owner_ranks,
	    int tag_base)
	{
	    static_assert(std::is_trivially_copyable_v<Key>, "assign_owners_with_neighbors requires trivially copyable Key");
	    static_assert(std::is_trivially_copyable_v<TouchCandidate<Key>>, "assign_owners_with_neighbors requires trivially copyable TouchCandidate");
	    if (local_touch.size() != local_keys.size() ||
	        cell_gid_candidate.size() != local_keys.size() ||
	        cell_owner_candidate.size() != local_keys.size()) {
	        throw FEException("assign_owners_with_neighbors: candidate arrays size mismatch");
	    }
	    if (!rank_to_order.empty() && rank_to_order.size() != static_cast<std::size_t>(world_size)) {
	        throw FEException("assign_owners_with_neighbors: rank_to_order size mismatch");
	    }
	    for (auto r : neighbors) {
	        if (r < 0 || r >= world_size) {
	            throw FEException("assign_owners_with_neighbors: neighbor rank out of range");
	        }
	    }

	    std::unordered_map<Key, std::size_t, KeyHash> local_index;
	    local_index.reserve(local_keys.size());
	    for (std::size_t i = 0; i < local_keys.size(); ++i) {
	        local_index.emplace(local_keys[i], i);
	    }

	    out_owner_ranks.assign(local_keys.size(), -1);
	    std::vector<int> min_touch_key(local_keys.size(), std::numeric_limits<int>::max());
	    std::vector<int> min_touch_rank(local_keys.size(), -1);
	    std::vector<int> max_touch_key(local_keys.size(), std::numeric_limits<int>::min());
	    std::vector<int> max_touch_rank(local_keys.size(), -1);
	    std::vector<gid_t> best_cell_gid(local_keys.size(), std::numeric_limits<gid_t>::max());
	    std::vector<int> best_cell_owner_key(local_keys.size(), std::numeric_limits<int>::max());
	    std::vector<int> best_cell_owner_rank(local_keys.size(), -1);
	    std::vector<std::uint64_t> best_hrw_score(local_keys.size(), 0);
	    std::vector<int> best_hrw_key(local_keys.size(), std::numeric_limits<int>::max());
	    std::vector<int> best_hrw_rank(local_keys.size(), -1);
	    std::vector<std::uint8_t> has_touch(local_keys.size(), 0);

	    auto rank_key = [&](int r) -> int {
	        if (rank_to_order.empty()) return r;
	        if (r < 0 || r >= world_size) return std::numeric_limits<int>::max();
	        return rank_to_order[static_cast<std::size_t>(r)];
	    };

	    auto consider_touch = [&](std::size_t idx, int touch_rank, gid_t cand_gid, int cand_owner) {
	        has_touch[idx] = 1;
	        const int tkey = rank_key(touch_rank);
	        if (tkey < min_touch_key[idx] || (tkey == min_touch_key[idx] && touch_rank < min_touch_rank[idx])) {
	            min_touch_key[idx] = tkey;
	            min_touch_rank[idx] = touch_rank;
	        }
	        if (tkey > max_touch_key[idx] || (tkey == max_touch_key[idx] && touch_rank > max_touch_rank[idx])) {
	            max_touch_key[idx] = tkey;
	            max_touch_rank[idx] = touch_rank;
	        }

	        const int ckey = rank_key(cand_owner);
	        if (cand_gid < best_cell_gid[idx] ||
	            (cand_gid == best_cell_gid[idx] && ckey < best_cell_owner_key[idx])) {
	            best_cell_gid[idx] = cand_gid;
	            best_cell_owner_key[idx] = ckey;
	            best_cell_owner_rank[idx] = cand_owner;
	        }

	        const auto score = hrw_score<Key, KeyHash>(local_keys[idx], tkey);
	        if (best_hrw_rank[idx] < 0 || score > best_hrw_score[idx] ||
	            (score == best_hrw_score[idx] && (tkey < best_hrw_key[idx] ||
	                                              (tkey == best_hrw_key[idx] && touch_rank < best_hrw_rank[idx])))) {
	            best_hrw_score[idx] = score;
	            best_hrw_key[idx] = tkey;
	            best_hrw_rank[idx] = touch_rank;
	        }
	    };

	    // Seed with local touches.
	    for (std::size_t i = 0; i < local_keys.size(); ++i) {
	        if (local_touch[i] == my_rank) {
	            consider_touch(i, my_rank, cell_gid_candidate[i], cell_owner_candidate[i]);
	        }
	    }

	    // Exchange touched keys (and candidates) with neighbors.
	    std::vector<TouchCandidate<Key>> touched;
	    touched.reserve(local_keys.size());
	    for (std::size_t i = 0; i < local_keys.size(); ++i) {
	        if (local_touch[i] != my_rank) continue;
	        touched.push_back(TouchCandidate<Key>{local_keys[i],
	                                              cell_gid_candidate[i],
	                                              static_cast<std::int32_t>(cell_owner_candidate[i])});
	    }

	    std::vector<std::vector<TouchCandidate<Key>>> incoming;
	    mpi_neighbor_exchange_broadcast_bytes(comm,
	                                          neighbors,
	                                          std::span<const TouchCandidate<Key>>(touched.data(), touched.size()),
	                                          incoming,
	                                          tag_base + 0,
	                                          tag_base + 1);

	    for (std::size_t n = 0; n < neighbors.size(); ++n) {
	        const int touch_rank = neighbors[n];
	        for (const auto& rec : incoming[n]) {
	            const auto it = local_index.find(rec.key);
	            if (it == local_index.end()) continue;
	            consider_touch(it->second, touch_rank, rec.cell_gid_candidate, static_cast<int>(rec.cell_owner_candidate));
	        }
	    }

	    for (std::size_t i = 0; i < local_keys.size(); ++i) {
	        if (has_touch[i] == 0) {
	            throw FEException("assign_owners_with_neighbors: local key has no touching ranks (missing neighbor list?)");
	        }

	        int owner = -1;
	        switch (ownership) {
	            case OwnershipStrategy::LowestRank:
	                owner = min_touch_rank[i];
	                break;
	            case OwnershipStrategy::HighestRank:
	                owner = max_touch_rank[i];
	                break;
	            case OwnershipStrategy::CellOwner:
	                owner = best_cell_owner_rank[i];
	                break;
	            case OwnershipStrategy::VertexGID:
	                owner = best_hrw_rank[i];
	                break;
	            default:
	                owner = min_touch_rank[i];
	                break;
	        }

	        if (owner < 0 || owner >= world_size) {
	            throw FEException("assign_owners_with_neighbors: computed owner out of range");
	        }
	        out_owner_ranks[i] = owner;
	    }
	}

	template <typename Key>
	struct KeyAssignment {
	    Key key{};
	    gid_t global_id{-1};
	    std::int32_t owner_rank{-1};
	};

	struct ValidationDigest {
	    std::uint64_t hash{0};
	    std::int64_t count{0};
	};

	template <typename Key, typename KeyHash, typename KeyLess>
		static void assign_contiguous_ids_and_owners_with_neighbors(
		    MPI_Comm comm,
		    int my_rank,
		    int world_size,
		    std::span<const int> neighbors,
		    OwnershipStrategy ownership,
		    std::span<const Key> local_keys,
		    std::span<const int> local_touch,
		    std::span<const gid_t> cell_gid_candidate,
		    std::span<const int> cell_owner_candidate,
		    std::span<const int> rank_to_order,
		    bool no_global_collectives,
		    std::vector<gid_t>& out_global_ids,
		    std::vector<int>& out_owner_ranks,
		    gid_t& out_global_count,
		    int tag_base)
		{
	    static_assert(std::is_trivially_copyable_v<Key>, "assign_contiguous_ids_and_owners_with_neighbors requires trivially copyable Key");
	    static_assert(std::is_trivially_copyable_v<TouchCandidate<Key>>, "assign_contiguous_ids_and_owners_with_neighbors requires trivially copyable TouchCandidate");
		    if (local_touch.size() != local_keys.size() ||
		        cell_gid_candidate.size() != local_keys.size() ||
		        cell_owner_candidate.size() != local_keys.size()) {
		        throw FEException("assign_contiguous_ids_and_owners_with_neighbors: candidate arrays size mismatch");
		    }
		    if (!rank_to_order.empty() && rank_to_order.size() != static_cast<std::size_t>(world_size)) {
		        throw FEException("assign_contiguous_ids_and_owners_with_neighbors: rank_to_order size mismatch");
		    }
		    for (auto r : neighbors) {
		        if (r < 0 || r >= world_size) {
		            throw FEException("assign_contiguous_ids_and_owners_with_neighbors: neighbor rank out of range");
		        }
		    }

	    std::unordered_map<Key, std::size_t, KeyHash> local_index;
	    local_index.reserve(local_keys.size());
	    for (std::size_t i = 0; i < local_keys.size(); ++i) {
	        local_index.emplace(local_keys[i], i);
	    }

		    out_owner_ranks.assign(local_keys.size(), -1);
		    std::vector<int> min_touch_key(local_keys.size(), std::numeric_limits<int>::max());
		    std::vector<int> min_touch_rank(local_keys.size(), -1);
		    std::vector<int> max_touch_key(local_keys.size(), std::numeric_limits<int>::min());
		    std::vector<int> max_touch_rank(local_keys.size(), -1);
		    std::vector<gid_t> best_cell_gid(local_keys.size(), std::numeric_limits<gid_t>::max());
		    std::vector<int> best_cell_owner_key(local_keys.size(), std::numeric_limits<int>::max());
		    std::vector<int> best_cell_owner_rank(local_keys.size(), -1);
		    std::vector<std::uint64_t> best_hrw_score(local_keys.size(), 0);
		    std::vector<int> best_hrw_key(local_keys.size(), std::numeric_limits<int>::max());
		    std::vector<int> best_hrw_rank(local_keys.size(), -1);
		    std::vector<std::uint8_t> has_touch(local_keys.size(), 0);

		    auto rank_key = [&](int r) -> int {
		        if (rank_to_order.empty()) return r;
		        if (r < 0 || r >= world_size) return std::numeric_limits<int>::max();
		        return rank_to_order[static_cast<std::size_t>(r)];
		    };

		    auto consider_touch = [&](std::size_t idx, int touch_rank, gid_t cand_gid, int cand_owner) {
		        has_touch[idx] = 1;
		        const int tkey = rank_key(touch_rank);
		        if (tkey < min_touch_key[idx] || (tkey == min_touch_key[idx] && touch_rank < min_touch_rank[idx])) {
		            min_touch_key[idx] = tkey;
		            min_touch_rank[idx] = touch_rank;
		        }
		        if (tkey > max_touch_key[idx] || (tkey == max_touch_key[idx] && touch_rank > max_touch_rank[idx])) {
		            max_touch_key[idx] = tkey;
		            max_touch_rank[idx] = touch_rank;
		        }

		        const int ckey = rank_key(cand_owner);
		        if (cand_gid < best_cell_gid[idx] ||
		            (cand_gid == best_cell_gid[idx] && ckey < best_cell_owner_key[idx])) {
		            best_cell_gid[idx] = cand_gid;
		            best_cell_owner_key[idx] = ckey;
		            best_cell_owner_rank[idx] = cand_owner;
		        }

		        const auto score = hrw_score<Key, KeyHash>(local_keys[idx], tkey);
		        if (best_hrw_rank[idx] < 0 || score > best_hrw_score[idx] ||
		            (score == best_hrw_score[idx] && (tkey < best_hrw_key[idx] ||
		                                              (tkey == best_hrw_key[idx] && touch_rank < best_hrw_rank[idx])))) {
		            best_hrw_score[idx] = score;
		            best_hrw_key[idx] = tkey;
		            best_hrw_rank[idx] = touch_rank;
		        }
		    };

	    // Seed with local touches.
	    for (std::size_t i = 0; i < local_keys.size(); ++i) {
	        if (local_touch[i] == my_rank) {
	            consider_touch(i, my_rank, cell_gid_candidate[i], cell_owner_candidate[i]);
	        }
	    }

	    // Exchange touched keys (and candidates) with neighbors.
	    std::vector<TouchCandidate<Key>> touched;
	    touched.reserve(local_keys.size());
	    for (std::size_t i = 0; i < local_keys.size(); ++i) {
	        if (local_touch[i] != my_rank) continue;
	        touched.push_back(TouchCandidate<Key>{local_keys[i], cell_gid_candidate[i], static_cast<std::int32_t>(cell_owner_candidate[i])});
	    }

	    std::vector<std::vector<TouchCandidate<Key>>> incoming;
	    mpi_neighbor_exchange_broadcast_bytes(comm, neighbors, std::span<const TouchCandidate<Key>>(touched.data(), touched.size()),
	                                          incoming, tag_base + 0, tag_base + 1);

	    for (std::size_t n = 0; n < neighbors.size(); ++n) {
	        const int touch_rank = neighbors[n];
	        for (const auto& rec : incoming[n]) {
	            const auto it = local_index.find(rec.key);
	            if (it == local_index.end()) continue;
	            consider_touch(it->second, touch_rank, rec.cell_gid_candidate, static_cast<int>(rec.cell_owner_candidate));
	        }
	    }

	    for (std::size_t i = 0; i < local_keys.size(); ++i) {
	        if (has_touch[i] == 0) {
	            throw FEException("assign_contiguous_ids_and_owners_with_neighbors: local key has no touching ranks (missing neighbor list?)");
	        }

		        int owner = -1;
		        switch (ownership) {
		            case OwnershipStrategy::LowestRank:
		                owner = min_touch_rank[i];
		                break;
		            case OwnershipStrategy::HighestRank:
		                owner = max_touch_rank[i];
		                break;
		            case OwnershipStrategy::CellOwner:
		                owner = best_cell_owner_rank[i];
		                break;
		            case OwnershipStrategy::VertexGID:
		                owner = best_hrw_rank[i];
		                break;
		            default:
		                owner = min_touch_rank[i];
		                break;
		        }

	        if (owner < 0 || owner >= world_size) {
	            throw FEException("assign_contiguous_ids_and_owners_with_neighbors: computed owner out of range");
	        }
	        out_owner_ranks[i] = owner;
		    }

		    assign_global_ordinals_with_neighbors<Key, KeyHash, KeyLess>(
		        comm,
		        my_rank,
		        world_size,
		        neighbors,
		        local_keys,
		        out_owner_ranks,
		        rank_to_order,
		        no_global_collectives,
		        out_global_ids,
		        out_global_count,
		        tag_base + 2);
		}

	template <typename Key, typename KeyHash, typename KeyLess>
	static void validate_entity_assignments_with_neighbors(
	    MPI_Comm comm,
	    int my_rank,
	    std::span<const int> neighbors,
	    std::span<const Key> local_keys,
	    std::span<const int> local_touch,
	    std::span<const gid_t> local_global_ids,
	    std::span<const int> local_owner_ranks,
	    int tag_base,
	    const char* label)
	{
	    static_assert(std::is_trivially_copyable_v<Key>, "validate_entity_assignments_with_neighbors requires trivially copyable Key");
	    static_assert(std::is_trivially_copyable_v<KeyAssignment<Key>>, "validate_entity_assignments_with_neighbors requires trivially copyable KeyAssignment");
	    if (local_keys.size() != local_touch.size() ||
	        local_keys.size() != local_global_ids.size() ||
	        local_keys.size() != local_owner_ranks.size()) {
	        throw FEException("validate_entity_assignments_with_neighbors: local array size mismatch");
	    }
	    if (neighbors.empty()) return;

	    std::unordered_map<Key, std::size_t, KeyHash> local_index;
	    local_index.reserve(local_keys.size());
	    for (std::size_t i = 0; i < local_keys.size(); ++i) {
	        local_index.emplace(local_keys[i], i);
	    }

	    // Exchange touched key sets (neighbor-only) to establish shared subsets.
	    std::vector<Key> touched_keys;
	    touched_keys.reserve(local_keys.size());
	    for (std::size_t i = 0; i < local_keys.size(); ++i) {
	        if (local_touch[i] == my_rank) {
	            touched_keys.push_back(local_keys[i]);
	        }
	    }
	    std::sort(touched_keys.begin(), touched_keys.end(), KeyLess{});
	    touched_keys.erase(std::unique(touched_keys.begin(), touched_keys.end()), touched_keys.end());

	    std::vector<std::vector<Key>> neighbor_touched;
	    mpi_neighbor_exchange_broadcast_bytes(comm,
	                                          neighbors,
	                                          std::span<const Key>(touched_keys.data(), touched_keys.size()),
	                                          neighbor_touched,
	                                          tag_base + 0,
	                                          tag_base + 1);

	    // Compute per-neighbor digests over shared keys.
	    std::vector<std::vector<ValidationDigest>> digests(neighbors.size());
	    for (std::size_t n = 0; n < neighbors.size(); ++n) {
	        std::uint64_t h = 0;
	        std::int64_t cnt = 0;
	        for (const auto& key : neighbor_touched[n]) {
	            const auto it = local_index.find(key);
	            if (it == local_index.end()) continue;
	            const auto idx = it->second;
	            if (local_touch[idx] != my_rank) continue; // validate only among touching ranks
	            const std::uint64_t kh = mix_u64(static_cast<std::uint64_t>(KeyHash{}(key)));
	            const std::uint64_t gh = mix_u64(static_cast<std::uint64_t>(local_global_ids[idx]));
	            const std::uint64_t oh = mix_u64(static_cast<std::uint64_t>(local_owner_ranks[idx]));
	            h ^= mix_u64(kh ^ gh ^ oh);
	            ++cnt;
	        }
	        digests[n].push_back(ValidationDigest{h, cnt});
	    }

	    std::vector<std::vector<ValidationDigest>> incoming_digests;
	    mpi_neighbor_exchange_bytes(comm,
	                               neighbors,
	                               std::span<const std::vector<ValidationDigest>>(digests.data(), digests.size()),
	                               incoming_digests,
	                               tag_base + 2,
	                               tag_base + 3);

	    // For any mismatch, exchange full assignment lists for that neighbor and compare.
	    std::vector<std::vector<KeyAssignment<Key>>> assignments(neighbors.size());
	    for (std::size_t n = 0; n < neighbors.size(); ++n) {
	        if (incoming_digests[n].size() != 1u) {
	            throw FEException("validate_entity_assignments_with_neighbors: invalid digest receive size");
	        }
	        const auto remote = incoming_digests[n][0];
	        const auto local = digests[n][0];
	        if (remote.count == local.count && remote.hash == local.hash) {
	            continue;
	        }

	        auto& list = assignments[n];
	        list.reserve(static_cast<std::size_t>(std::max<std::int64_t>(0, local.count)));
	        for (const auto& key : neighbor_touched[n]) {
	            const auto it = local_index.find(key);
	            if (it == local_index.end()) continue;
	            const auto idx = it->second;
	            if (local_touch[idx] != my_rank) continue;
	            list.push_back(KeyAssignment<Key>{key, local_global_ids[idx], static_cast<std::int32_t>(local_owner_ranks[idx])});
	        }
	        std::sort(list.begin(), list.end(), [&](const auto& a, const auto& b) { return KeyLess{}(a.key, b.key); });
	    }

	    bool need_full = false;
	    for (const auto& l : assignments) {
	        if (!l.empty()) {
	            need_full = true;
	            break;
	        }
	    }
	    if (!need_full) {
	        return;
	    }

	    std::vector<std::vector<KeyAssignment<Key>>> incoming_assignments;
	    mpi_neighbor_exchange_bytes(comm,
	                               neighbors,
	                               std::span<const std::vector<KeyAssignment<Key>>>(assignments.data(), assignments.size()),
	                               incoming_assignments,
	                               tag_base + 4,
	                               tag_base + 5);

		for (std::size_t n = 0; n < neighbors.size(); ++n) {
	        if (assignments[n].empty()) continue; // no mismatch detected for this neighbor

	        for (const auto& rec : incoming_assignments[n]) {
	            const auto it = local_index.find(rec.key);
	            if (it == local_index.end()) continue;
	            const auto idx = it->second;
	            if (local_touch[idx] != my_rank) continue;
	            if (local_global_ids[idx] != rec.global_id || local_owner_ranks[idx] != static_cast<int>(rec.owner_rank)) {
	                throw FEException(std::string("DofHandler::validate_parallel: mismatch in ") + label +
	                                  " assignments between ranks " + std::to_string(my_rank) + " and " +
	                                  std::to_string(neighbors[n]));
	            }
	        }
	    }
	}

	template <typename Key>
	struct OrdinalKey {
	    Key key{};
	    std::int32_t ordinal{0};

	    bool operator==(const OrdinalKey& other) const noexcept {
	        return key == other.key && ordinal == other.ordinal;
	    }

	    bool operator<(const OrdinalKey& other) const noexcept {
	        if (key < other.key) return true;
	        if (other.key < key) return false;
	        return ordinal < other.ordinal;
	    }
	};

	template <typename Key, typename KeyHash>
	struct OrdinalKeyHash {
	    std::size_t operator()(const OrdinalKey<Key>& k) const noexcept {
	        const std::size_t base = KeyHash{}(k.key);
	        const std::uint64_t mixed =
	            mix_u64(static_cast<std::uint64_t>(base) ^
	                    (static_cast<std::uint64_t>(static_cast<std::uint32_t>(k.ordinal)) + kHashSalt));
	        return static_cast<std::size_t>(mixed);
	    }
	};

	struct CellPrivateKey {
	    gid_t cell_gid{gid_t{-1}};
	    std::int32_t ordinal{0};

	    bool operator==(const CellPrivateKey& other) const noexcept {
	        return cell_gid == other.cell_gid && ordinal == other.ordinal;
	    }

	    bool operator<(const CellPrivateKey& other) const noexcept {
	        if (cell_gid != other.cell_gid) return cell_gid < other.cell_gid;
	        return ordinal < other.ordinal;
	    }
	};

	struct CellPrivateKeyHash {
	    std::size_t operator()(const CellPrivateKey& k) const noexcept {
	        std::uint64_t h = mix_u64(static_cast<std::uint64_t>(k.cell_gid));
	        h ^= mix_u64(static_cast<std::uint64_t>(static_cast<std::uint32_t>(k.ordinal)) + kHashSalt);
	        return static_cast<std::size_t>(h);
	    }
	};

	template <typename Key>
	struct CountRecord {
	    Key key{};
	    std::int32_t count{0};
	};

	template <typename Key, typename Hash>
	static void reduce_min_counts_with_neighbors(
	    MPI_Comm comm,
	    std::span<const int> neighbors,
	    std::span<const Key> local_keys,
	    std::span<const LocalIndex> local_counts,
	    std::vector<LocalIndex>& out_counts,
	    int tag_base) {
	    static_assert(std::is_trivially_copyable_v<Key>,
	                  "reduce_min_counts_with_neighbors requires trivially copyable keys");
	    static_assert(std::is_trivially_copyable_v<CountRecord<Key>>,
	                  "reduce_min_counts_with_neighbors requires trivially copyable records");

	    if (local_keys.size() != local_counts.size()) {
	        throw FEException("reduce_min_counts_with_neighbors: key/count size mismatch");
	    }

	    out_counts.assign(local_counts.begin(), local_counts.end());
	    if (neighbors.empty() || local_keys.empty()) {
	        return;
	    }

	    std::unordered_map<Key, std::size_t, Hash> local_index;
	    local_index.reserve(local_keys.size());
	    for (std::size_t i = 0; i < local_keys.size(); ++i) {
	        local_index.emplace(local_keys[i], i);
	    }

	    std::vector<CountRecord<Key>> send_records;
	    send_records.reserve(local_keys.size());
	    for (std::size_t i = 0; i < local_keys.size(); ++i) {
	        send_records.push_back(CountRecord<Key>{
	            local_keys[i],
	            static_cast<std::int32_t>(std::max<LocalIndex>(0, local_counts[i]))});
	    }

	    for (std::size_t n = 0; n < neighbors.size(); ++n) {
	        const int nbr = neighbors[n];
	        int send_n = static_cast<int>(send_records.size());
	        int recv_n = 0;
	        fe_mpi_check(MPI_Sendrecv(&send_n,
	                                  1,
	                                  MPI_INT,
	                                  nbr,
	                                  tag_base,
	                                  &recv_n,
	                                  1,
	                                  MPI_INT,
	                                  nbr,
	                                  tag_base,
	                                  comm,
	                                  MPI_STATUS_IGNORE),
	                     "MPI_Sendrecv count in reduce_min_counts_with_neighbors");

	        std::vector<CountRecord<Key>> recv_records(static_cast<std::size_t>(std::max(0, recv_n)));
	        const int send_bytes = static_cast<int>(send_records.size() * sizeof(CountRecord<Key>));
	        const int recv_bytes = static_cast<int>(recv_records.size() * sizeof(CountRecord<Key>));
	        fe_mpi_check(MPI_Sendrecv(send_records.empty() ? nullptr : send_records.data(),
	                                  send_bytes,
	                                  MPI_BYTE,
	                                  nbr,
	                                  tag_base + 1,
	                                  recv_records.empty() ? nullptr : recv_records.data(),
	                                  recv_bytes,
	                                  MPI_BYTE,
	                                  nbr,
	                                  tag_base + 1,
	                                  comm,
	                                  MPI_STATUS_IGNORE),
	                     "MPI_Sendrecv payload in reduce_min_counts_with_neighbors");

	        for (const auto& rec : recv_records) {
	            const auto it = local_index.find(rec.key);
	            if (it == local_index.end()) {
	                continue;
	            }
	            out_counts[it->second] =
	                std::min(out_counts[it->second],
	                         static_cast<LocalIndex>(std::max<std::int32_t>(0, rec.count)));
	        }
	    }
	}

} // namespace
#endif // FE_HAS_MPI

namespace {

ElementType infer_element_type_from_cell(int dim, std::size_t n_verts) {
    if (dim == 1 && n_verts == 2) return ElementType::Line2;
    if (dim == 2 && n_verts == 3) return ElementType::Triangle3;
    if (dim == 2 && n_verts == 4) return ElementType::Quad4;
    if (dim == 3 && n_verts == 4) return ElementType::Tetra4;
    if (dim == 3 && n_verts == 8) return ElementType::Hex8;
    if (dim == 3 && n_verts == 6) return ElementType::Wedge6;
    if (dim == 3 && n_verts == 5) return ElementType::Pyramid5;
    return ElementType::Unknown;
}

LocalIndex checked_local_index_from_size(std::size_t value, const char* context) {
    const auto max_local = static_cast<std::size_t>(std::numeric_limits<LocalIndex>::max());
    if (value > max_local) {
        throw FEException(std::string(context) + ": local index overflow");
    }
    return static_cast<LocalIndex>(value);
}

LocalIndex simplex_interior_dofs(int order, int simplex_dim) {
    if (order <= simplex_dim) {
        return 0;
    }

    std::size_t numer = 1u;
    for (int i = 1; i <= simplex_dim; ++i) {
        numer *= static_cast<std::size_t>(order - i);
    }

    std::size_t denom = 1u;
    for (int i = 2; i <= simplex_dim; ++i) {
        denom *= static_cast<std::size_t>(i);
    }

    return checked_local_index_from_size(numer / denom, "simplex_interior_dofs");
}

LocalIndex lagrange_total_dofs(ElementType element_type, int order) {
    const basis::LagrangeBasis basis_fn(element_type, order);
    return checked_local_index_from_size(basis_fn.size(), "lagrange_total_dofs");
}

std::pair<int, int> count_reference_faces_by_vertices(ElementType cell_type) {
    const auto ref = elements::ReferenceElement::create(cell_type);
    int n_tri_faces = 0;
    int n_quad_faces = 0;
    for (std::size_t f = 0; f < ref.num_faces(); ++f) {
        const auto nverts = ref.face_nodes(f).size();
        if (nverts == 3u) {
            ++n_tri_faces;
        } else if (nverts == 4u) {
            ++n_quad_faces;
        }
    }
    return {n_tri_faces, n_quad_faces};
}

bool is_scalar_face_interior_node(ElementType face_type,
                                  const math::Vector<Real, 3>& xi) {
    constexpr Real tol = Real(1e-12);

    switch (face_type) {
        case ElementType::Triangle3:
            return xi[0] > tol && xi[1] > tol && (xi[0] + xi[1]) < (Real(1) - tol);
        case ElementType::Quad4:
            return std::abs(xi[0]) < (Real(1) - tol) && std::abs(xi[1]) < (Real(1) - tol);
        default:
            return false;
    }
}

struct FaceAffine2 {
    Real a00{Real(1)};
    Real a01{Real(0)};
    Real a10{Real(0)};
    Real a11{Real(1)};
    Real b0{Real(0)};
    Real b1{Real(0)};

    [[nodiscard]] math::Vector<Real, 3> apply(const math::Vector<Real, 3>& p) const noexcept {
        math::Vector<Real, 3> out = p;
        out[0] = a00 * p[0] + a01 * p[1] + b0;
        out[1] = a10 * p[0] + a11 * p[1] + b1;
        return out;
    }
};

std::vector<math::Vector<Real, 3>> canonical_face_vertices(ElementType face_type) {
    using Vec3 = math::Vector<Real, 3>;
    switch (face_type) {
        case ElementType::Triangle3:
            return {Vec3{Real(0), Real(0), Real(0)},
                    Vec3{Real(1), Real(0), Real(0)},
                    Vec3{Real(0), Real(1), Real(0)}};
        case ElementType::Quad4:
            return {Vec3{Real(-1), Real(-1), Real(0)},
                    Vec3{Real(1),  Real(-1), Real(0)},
                    Vec3{Real(1),  Real(1),  Real(0)},
                    Vec3{Real(-1), Real(1),  Real(0)}};
        default:
            return {};
    }
}

FaceAffine2 compute_face_affine_from_vertex_map(
    const std::vector<math::Vector<Real, 3>>& verts,
    const std::vector<int>& local_to_global) {

    FE_CHECK_ARG(verts.size() == local_to_global.size(),
                 "compute_face_affine_from_vertex_map: vertex map size mismatch");
    FE_CHECK_ARG(verts.size() == 3u || verts.size() == 4u,
                 "compute_face_affine_from_vertex_map: supported for triangle/quad faces only");

    const auto S0 = verts[0];
    const auto S1 = verts[1];
    const auto S2 = verts[2];
    const auto T0 = verts[static_cast<std::size_t>(local_to_global[0])];
    const auto T1 = verts[static_cast<std::size_t>(local_to_global[1])];
    const auto T2 = verts[static_cast<std::size_t>(local_to_global[2])];

    const Real b00 = S1[0] - S0[0];
    const Real b01 = S2[0] - S0[0];
    const Real b10 = S1[1] - S0[1];
    const Real b11 = S2[1] - S0[1];

    const Real c00 = T1[0] - T0[0];
    const Real c01 = T2[0] - T0[0];
    const Real c10 = T1[1] - T0[1];
    const Real c11 = T2[1] - T0[1];

    const Real detB = b00 * b11 - b01 * b10;
    FE_CHECK_ARG(std::abs(detB) > Real(0),
                 "compute_face_affine_from_vertex_map: degenerate face map");
    const Real inv_detB = Real(1) / detB;

    FaceAffine2 map;
    map.a00 = ( c00 * b11 - c01 * b10) * inv_detB;
    map.a01 = (-c00 * b01 + c01 * b00) * inv_detB;
    map.a10 = ( c10 * b11 - c11 * b10) * inv_detB;
    map.a11 = (-c10 * b01 + c11 * b00) * inv_detB;
    map.b0 = T0[0] - (map.a00 * S0[0] + map.a01 * S0[1]);
    map.b1 = T0[1] - (map.a10 * S0[0] + map.a11 * S0[1]);
    return map;
}

std::vector<int> compute_scalar_face_interior_local_to_global(
    ElementType face_type,
    int poly_order,
    const std::vector<int>& vertex_perm) {

    FE_CHECK_ARG(poly_order >= 0, "compute_scalar_face_interior_local_to_global: negative poly_order");

    basis::LagrangeBasis face_basis(face_type, poly_order);
    const auto& nodes = face_basis.nodes();
    const std::size_t n_nodes = nodes.size();

    std::vector<int> interior_full_indices;
    interior_full_indices.reserve(n_nodes);
    std::vector<int> interior_slot_of_full(n_nodes, -1);
    for (std::size_t i = 0; i < n_nodes; ++i) {
        if (!is_scalar_face_interior_node(face_type, nodes[i])) {
            continue;
        }
        interior_slot_of_full[i] = static_cast<int>(interior_full_indices.size());
        interior_full_indices.push_back(static_cast<int>(i));
    }

    if (interior_full_indices.empty()) {
        return {};
    }

    const auto verts = canonical_face_vertices(face_type);
    FE_CHECK_ARG(!verts.empty(), "compute_scalar_face_interior_local_to_global: unsupported face type");
    FE_CHECK_ARG(vertex_perm.size() == verts.size(),
                 "compute_scalar_face_interior_local_to_global: vertex permutation size mismatch");

    const auto local_to_global = spaces::OrientationManager::invert_permutation(vertex_perm);
    const auto map = compute_face_affine_from_vertex_map(verts, local_to_global);

    std::vector<int> full_perm(n_nodes, -1); // global full node -> local full node
    std::vector<bool> used(n_nodes, false);
    for (std::size_t i_local = 0; i_local < n_nodes; ++i_local) {
        const auto x_global = map.apply(nodes[i_local]);
        int matched = -1;
        for (std::size_t j = 0; j < n_nodes; ++j) {
            if (used[j]) {
                continue;
            }
            if (nodes[j].approx_equal(x_global, Real(1e-12))) {
                matched = static_cast<int>(j);
                break;
            }
        }
        FE_CHECK_ARG(matched >= 0,
                     "compute_scalar_face_interior_local_to_global: face node match failed");
        used[static_cast<std::size_t>(matched)] = true;
        full_perm[static_cast<std::size_t>(matched)] = static_cast<int>(i_local);
    }

    std::vector<int> local_to_global_interior(interior_full_indices.size(), -1);
    for (std::size_t global_int = 0; global_int < interior_full_indices.size(); ++global_int) {
        const int global_full = interior_full_indices[global_int];
        const int local_full = full_perm[static_cast<std::size_t>(global_full)];
        FE_CHECK_ARG(local_full >= 0,
                     "compute_scalar_face_interior_local_to_global: full permutation incomplete");
        const int local_int = interior_slot_of_full[static_cast<std::size_t>(local_full)];
        FE_CHECK_ARG(local_int >= 0,
                     "compute_scalar_face_interior_local_to_global: boundary node mapped into interior set");
        local_to_global_interior[static_cast<std::size_t>(local_int)] =
            static_cast<int>(global_int);
    }

    for (const int idx : local_to_global_interior) {
        FE_CHECK_ARG(idx >= 0,
                     "compute_scalar_face_interior_local_to_global: interior permutation incomplete");
    }
    return local_to_global_interior;
}

struct LocalEdgeKey {
    MeshIndex a{0};
    MeshIndex b{0};
    bool operator==(const LocalEdgeKey& other) const noexcept { return a == other.a && b == other.b; }
};

struct LocalEdgeKeyHash {
    std::size_t operator()(const LocalEdgeKey& k) const noexcept {
        const std::size_t h1 = std::hash<MeshIndex>{}(k.a);
        const std::size_t h2 = std::hash<MeshIndex>{}(k.b);
        return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
    }
};

struct LocalFaceKey {
    std::array<MeshIndex, 4> verts{};
    std::uint8_t n{0};
    bool operator==(const LocalFaceKey& other) const noexcept { return n == other.n && verts == other.verts; }
};

struct LocalFaceKeyHash {
    std::size_t operator()(const LocalFaceKey& k) const noexcept {
        std::size_t seed = std::hash<std::uint8_t>{}(k.n);
        for (std::size_t i = 0; i < static_cast<std::size_t>(k.n); ++i) {
            const std::size_t h = std::hash<MeshIndex>{}(k.verts[i]);
            seed ^= h + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

static std::vector<MeshIndex> canonical_cycle_vertices(std::span<const MeshIndex> verts,
                                                         std::span<const gid_t> vertex_gids) {
    if (verts.empty()) {
        return {};
    }

    auto vertex_key = [&](MeshIndex v) -> gid_t {
        if (!vertex_gids.empty() && v >= 0 && static_cast<std::size_t>(v) < vertex_gids.size()) {
            return vertex_gids[static_cast<std::size_t>(v)];
        }
        return static_cast<gid_t>(v);
    };

    const std::size_t n = verts.size();

    std::size_t start = 0;
    gid_t best = vertex_key(verts[0]);
    MeshIndex best_vid = verts[0];
    for (std::size_t i = 1; i < n; ++i) {
        const auto v = verts[i];
        const auto k = vertex_key(v);
        if (k < best || (k == best && v < best_vid)) {
            best = k;
            best_vid = v;
            start = i;
        }
    }

    const std::size_t next = (start + 1u) % n;
    const std::size_t prev = (start + n - 1u) % n;
    const gid_t k_next = vertex_key(verts[next]);
    const gid_t k_prev = vertex_key(verts[prev]);
    const bool forward = (k_next < k_prev) || (k_next == k_prev && verts[next] < verts[prev]);

    std::vector<MeshIndex> out;
    out.reserve(n);
    if (forward) {
        for (std::size_t i = 0; i < n; ++i) {
            out.push_back(verts[(start + i) % n]);
        }
    } else {
        for (std::size_t i = 0; i < n; ++i) {
            out.push_back(verts[(start + n - i) % n]);
        }
    }
    return out;
}

static void derive_edge_connectivity(MeshTopologyInfo& topology) {
    if (topology.n_cells <= 0 || topology.cell2vertex_offsets.empty() || topology.cell2vertex_data.empty()) {
        throw FEException("derive_edge_connectivity: missing cell2vertex connectivity");
    }

    const auto n_cells = static_cast<std::size_t>(topology.n_cells);
    topology.cell2edge_offsets.assign(n_cells + 1u, MeshOffset{0});
    topology.cell2edge_data.clear();
    topology.edge2vertex_data.clear();

    std::unordered_map<LocalEdgeKey, MeshIndex, LocalEdgeKeyHash> edge_ids;
    edge_ids.reserve(static_cast<std::size_t>(topology.n_cells) * 4u);
    std::vector<std::array<MeshIndex, 2>> edges;

    for (GlobalIndex c = 0; c < topology.n_cells; ++c) {
        const auto cell_verts = topology.getCellVertices(c);
        const auto base_type = infer_element_type_from_cell(topology.dim, cell_verts.size());
        if (base_type == ElementType::Unknown) {
            throw FEException("derive_edge_connectivity: unsupported cell type for edge derivation");
        }
        const auto ref = elements::ReferenceElement::create(base_type);

        topology.cell2edge_offsets[static_cast<std::size_t>(c)] = static_cast<MeshOffset>(topology.cell2edge_data.size());

        for (std::size_t le = 0; le < ref.num_edges(); ++le) {
            const auto& en = ref.edge_nodes(le);
            if (en.size() != 2u) {
                continue;
            }
            const auto lv0 = static_cast<std::size_t>(en[0]);
            const auto lv1 = static_cast<std::size_t>(en[1]);
            if (lv0 >= cell_verts.size() || lv1 >= cell_verts.size()) {
                throw FEException("derive_edge_connectivity: reference edge node out of range");
            }
            const MeshIndex gv0 = cell_verts[lv0];
            const MeshIndex gv1 = cell_verts[lv1];
            if (gv0 < 0 || gv1 < 0) {
                throw FEException("derive_edge_connectivity: negative vertex index");
            }

            LocalEdgeKey key{std::min(gv0, gv1), std::max(gv0, gv1)};
            auto it = edge_ids.find(key);
            MeshIndex edge_id = -1;
            if (it == edge_ids.end()) {
                edge_id = static_cast<MeshIndex>(edges.size());
                edge_ids.emplace(key, edge_id);
                edges.push_back({key.a, key.b});
            } else {
                edge_id = it->second;
            }
            topology.cell2edge_data.push_back(edge_id);
        }

        topology.cell2edge_offsets[static_cast<std::size_t>(c) + 1u] =
            static_cast<MeshOffset>(topology.cell2edge_data.size());
    }

    topology.n_edges = static_cast<GlobalIndex>(edges.size());
    topology.edge2vertex_data.resize(static_cast<std::size_t>(topology.n_edges) * 2u);
    for (GlobalIndex e = 0; e < topology.n_edges; ++e) {
        const auto& ev = edges[static_cast<std::size_t>(e)];
        topology.edge2vertex_data[static_cast<std::size_t>(2 * e + 0)] = ev[0];
        topology.edge2vertex_data[static_cast<std::size_t>(2 * e + 1)] = ev[1];
    }
}

static void derive_face_connectivity(MeshTopologyInfo& topology) {
    if (topology.n_cells <= 0 || topology.cell2vertex_offsets.empty() || topology.cell2vertex_data.empty()) {
        throw FEException("derive_face_connectivity: missing cell2vertex connectivity");
    }

    const auto n_cells = static_cast<std::size_t>(topology.n_cells);
    topology.cell2face_offsets.assign(n_cells + 1u, MeshOffset{0});
    topology.cell2face_data.clear();
    topology.face2vertex_offsets.clear();
    topology.face2vertex_data.clear();

    std::unordered_map<LocalFaceKey, MeshIndex, LocalFaceKeyHash> face_ids;
    face_ids.reserve(static_cast<std::size_t>(topology.n_cells) * 6u);
    std::vector<std::vector<MeshIndex>> face_vertices;

    for (GlobalIndex c = 0; c < topology.n_cells; ++c) {
        const auto cell_verts = topology.getCellVertices(c);
        const auto base_type = infer_element_type_from_cell(topology.dim, cell_verts.size());
        if (base_type == ElementType::Unknown) {
            throw FEException("derive_face_connectivity: unsupported cell type for face derivation");
        }
        const auto ref = elements::ReferenceElement::create(base_type);

        topology.cell2face_offsets[static_cast<std::size_t>(c)] = static_cast<MeshOffset>(topology.cell2face_data.size());

        for (std::size_t lf = 0; lf < ref.num_faces(); ++lf) {
            const auto& fn = ref.face_nodes(lf);
            if (fn.size() != 3u && fn.size() != 4u) {
                continue;
            }
            std::vector<MeshIndex> fv;
            fv.reserve(fn.size());
            for (auto ln : fn) {
                const auto lv = static_cast<std::size_t>(ln);
                if (lv >= cell_verts.size()) {
                    throw FEException("derive_face_connectivity: reference face node out of range");
                }
                const MeshIndex gv = cell_verts[lv];
                if (gv < 0) {
                    throw FEException("derive_face_connectivity: negative face vertex index");
                }
                fv.push_back(gv);
            }

            LocalFaceKey key{};
            const std::size_t n = fv.size();
            key.n = static_cast<std::uint8_t>(n);
            for (std::size_t i = 0; i < n; ++i) {
                key.verts[i] = fv[i];
            }
            if (n == 3u) {
                std::sort(key.verts.begin(), key.verts.begin() + 3u);
            } else if (n == 4u) {
                std::sort(key.verts.begin(), key.verts.begin() + 4u);
            } else {
                throw FEException("derive_face_connectivity: unsupported face arity while sorting face key");
            }
            for (std::size_t i = n; i < key.verts.size(); ++i) {
                key.verts[i] = MeshIndex{0};
            }

            auto it = face_ids.find(key);
            MeshIndex face_id = -1;
            if (it == face_ids.end()) {
                face_id = static_cast<MeshIndex>(face_vertices.size());
                face_ids.emplace(key, face_id);
                face_vertices.push_back(canonical_cycle_vertices(fv, topology.vertex_gids));
            } else {
                face_id = it->second;
            }
            topology.cell2face_data.push_back(face_id);
        }

        topology.cell2face_offsets[static_cast<std::size_t>(c) + 1u] =
            static_cast<MeshOffset>(topology.cell2face_data.size());
    }

    topology.n_faces = static_cast<GlobalIndex>(face_vertices.size());
    topology.face2vertex_offsets.reserve(static_cast<std::size_t>(topology.n_faces) + 1u);
    topology.face2vertex_offsets.push_back(0);
    for (const auto& fv : face_vertices) {
        topology.face2vertex_data.insert(topology.face2vertex_data.end(), fv.begin(), fv.end());
        topology.face2vertex_offsets.push_back(static_cast<MeshOffset>(topology.face2vertex_data.size()));
    }
}

} // namespace

// =============================================================================
// DofLayoutInfo factories
// =============================================================================

DofLayoutInfo DofLayoutInfo::Lagrange(int order, int dim, int num_verts_per_cell, int num_components) {
    DofLayoutInfo info;
    info.is_continuous = true;
    info.num_components = std::max(1, num_components);
    FE_CHECK_ARG(order >= 1, "DofLayoutInfo::Lagrange requires polynomial order >= 1");

    const auto cell_type = infer_element_type_from_cell(dim, static_cast<std::size_t>(num_verts_per_cell));
    FE_CHECK_ARG(cell_type != ElementType::Unknown,
                 "DofLayoutInfo::Lagrange: unsupported cell topology");

    info.dofs_per_vertex = 1;
    info.dofs_per_edge = 0;
    info.dofs_per_face = 0;
    info.dofs_per_tri_face = 0;
    info.dofs_per_quad_face = 0;
    info.dofs_per_cell = 0;
    info.total_dofs_per_element = lagrange_total_dofs(cell_type, order);

    if (dim == 1) {
        info.dofs_per_cell = checked_local_index_from_size(static_cast<std::size_t>(order - 1),
                                                           "DofLayoutInfo::Lagrange line interior");
    } else if (dim == 2) {
        info.dofs_per_edge = checked_local_index_from_size(static_cast<std::size_t>(order - 1),
                                                           "DofLayoutInfo::Lagrange edge interior");
        if (cell_type == ElementType::Triangle3) {
            info.dofs_per_cell = simplex_interior_dofs(order, /*simplex_dim=*/2);
        } else if (cell_type == ElementType::Quad4) {
            info.dofs_per_cell = checked_local_index_from_size(
                static_cast<std::size_t>(order - 1) * static_cast<std::size_t>(order - 1),
                "DofLayoutInfo::Lagrange quad interior");
        } else {
            throw FEException("DofLayoutInfo::Lagrange: unsupported 2D element topology");
        }
    } else if (dim == 3) {
        info.dofs_per_edge = checked_local_index_from_size(static_cast<std::size_t>(order - 1),
                                                           "DofLayoutInfo::Lagrange edge interior");
        switch (cell_type) {
            case ElementType::Tetra4:
                info.dofs_per_face = simplex_interior_dofs(order, /*simplex_dim=*/2);
                info.dofs_per_cell = simplex_interior_dofs(order, /*simplex_dim=*/3);
                break;
            case ElementType::Hex8:
                info.dofs_per_face = checked_local_index_from_size(
                    static_cast<std::size_t>(order - 1) * static_cast<std::size_t>(order - 1),
                    "DofLayoutInfo::Lagrange hex face interior");
                info.dofs_per_cell = checked_local_index_from_size(
                    static_cast<std::size_t>(order - 1) *
                    static_cast<std::size_t>(order - 1) *
                    static_cast<std::size_t>(order - 1),
                    "DofLayoutInfo::Lagrange hex cell interior");
                break;
            case ElementType::Wedge6:
            case ElementType::Pyramid5: {
                const auto [n_tri_faces, n_quad_faces] = count_reference_faces_by_vertices(cell_type);
                const auto ref = elements::ReferenceElement::create(cell_type);

                info.dofs_per_tri_face = simplex_interior_dofs(order, /*simplex_dim=*/2);
                info.dofs_per_quad_face = checked_local_index_from_size(
                    static_cast<std::size_t>(order - 1) * static_cast<std::size_t>(order - 1),
                    "DofLayoutInfo::Lagrange mixed quad face interior");

                const auto face_total = static_cast<std::size_t>(n_tri_faces) *
                                            static_cast<std::size_t>(info.dofs_per_tri_face) +
                                        static_cast<std::size_t>(n_quad_faces) *
                                            static_cast<std::size_t>(info.dofs_per_quad_face);
                const auto edge_total = ref.num_edges() * static_cast<std::size_t>(info.dofs_per_edge);
                const auto vertex_total = static_cast<std::size_t>(num_verts_per_cell) *
                                          static_cast<std::size_t>(info.dofs_per_vertex);
                const auto total = static_cast<std::size_t>(info.total_dofs_per_element);
                FE_THROW_IF(total < vertex_total + edge_total + face_total, FEException,
                            "DofLayoutInfo::Lagrange: mixed-face wedge/pyramid layout exceeds total basis size");

                info.dofs_per_cell = checked_local_index_from_size(
                    total - vertex_total - edge_total - face_total,
                    "DofLayoutInfo::Lagrange mixed wedge/pyramid cell interior");
                break;
            }
            default:
                throw FEException("DofLayoutInfo::Lagrange: unsupported 3D element topology");
        }
    } else {
        throw FEException("DofLayoutInfo::Lagrange: unsupported dimension");
    }

    // total_dofs_per_element describes the full field (all components).
    if (info.num_components > 1 && info.total_dofs_per_element > 0) {
        info.total_dofs_per_element = static_cast<LocalIndex>(
            static_cast<GlobalIndex>(info.total_dofs_per_element) *
            static_cast<GlobalIndex>(info.num_components));
    }

    // DofHandler can currently permute scalar face-interior DOFs canonically for
    // tetrahedral and hexahedral complete Lagrange spaces.
    info.tensor_face_dof_layout =
        (dim == 3 &&
         ((cell_type == ElementType::Tetra4 && info.dofs_per_face > 1) ||
          (cell_type == ElementType::Hex8 && info.dofs_per_face > 1) ||
          (cell_type == ElementType::Wedge6 &&
           (info.dofs_per_tri_face > 1 || info.dofs_per_quad_face > 1)) ||
          (cell_type == ElementType::Pyramid5 &&
           (info.dofs_per_tri_face > 1 || info.dofs_per_quad_face > 1))));

    return info;
}

DofLayoutInfo DofLayoutInfo::DG(int order, int num_verts_per_cell, int num_components) {
    DofLayoutInfo info;
    info.is_continuous = false;
    info.num_components = std::max(1, num_components);
    info.tensor_face_dof_layout = false;

    // For DG, all DOFs are cell-interior (no sharing)
    info.dofs_per_vertex = 0;
    info.dofs_per_edge = 0;
    info.dofs_per_face = 0;
    info.dofs_per_tri_face = 0;
    info.dofs_per_quad_face = 0;
    FE_CHECK_ARG(order >= 0, "DofLayoutInfo::DG requires polynomial order >= 0");

    ElementType cell_type = ElementType::Unknown;
    switch (num_verts_per_cell) {
        case 2: cell_type = ElementType::Line2; break;
        case 3: cell_type = ElementType::Triangle3; break;
        case 4:
            // DG layout does not receive dimension, so only unambiguous complete-family
            // aliases are handled here. Callers using 3D tetrahedra should construct the
            // total from the space directly rather than relying on this convenience path.
            cell_type = ElementType::Quad4;
            break;
        case 5: cell_type = ElementType::Pyramid5; break;
        case 6: cell_type = ElementType::Wedge6; break;
        case 8: cell_type = ElementType::Hex8; break;
        default:
            throw FEException("DofLayoutInfo::DG: unsupported cell topology");
    }

    info.dofs_per_cell = lagrange_total_dofs(cell_type, order);
    info.total_dofs_per_element = static_cast<LocalIndex>(
        static_cast<GlobalIndex>(info.dofs_per_cell) * static_cast<GlobalIndex>(info.num_components));

    return info;
}

enum class VariableLocalDofKind : std::uint8_t {
    Vertex,
    Edge,
    Face,
    Cell
};

struct VariableLocalDofTag {
    VariableLocalDofKind kind{VariableLocalDofKind::Cell};
    int local_entity_id{-1};
    LocalIndex ordinal{0};
};

struct VariableCellLayout {
    ElementType cell_type{ElementType::Unknown};
    int polynomial_order{0};
    FieldType field_type{FieldType::Scalar};
    Continuity continuity{Continuity::Custom};
    BasisType basis_type{BasisType::Custom};
    LocalIndex scalar_dofs_per_element{0};
    LocalIndex cell_interior_dofs{0};
    std::vector<LocalIndex> vertex_counts;
    std::vector<LocalIndex> edge_counts;
    std::vector<LocalIndex> face_counts;
    std::vector<VariableLocalDofTag> dof_tags;
};

static bool variable_order_scalar_basis_supported(BasisType basis_type) noexcept {
    switch (basis_type) {
        case BasisType::Lagrange:
        case BasisType::Hierarchical:
        case BasisType::Bernstein:
        case BasisType::Spectral:
            return true;
        default:
            return false;
    }
}

static VariableCellLayout build_variable_cell_layout(const spaces::FunctionSpace& space,
                                                     const MeshTopologyView& topology,
                                                     GlobalIndex cell_id) {
    const auto cell_verts = topology.getCellVertices(cell_id);
    const ElementType cell_type =
        infer_element_type_from_cell(topology.dim, cell_verts.size());
    FE_CHECK_ARG(cell_type != ElementType::Unknown,
                 "DofHandler::distributeVariableOrderDofs: unsupported cell topology");

    const auto& elem = space.getElement(cell_type, cell_id);
    const auto& cell_basis = elem.basis();

    VariableCellLayout layout;
    layout.cell_type = cell_type;
    layout.polynomial_order = space.polynomial_order(cell_id);
    layout.field_type = space.field_type();
    layout.continuity = space.continuity();
    layout.basis_type = cell_basis.basis_type();

    const auto ref = elements::ReferenceElement::create(cell_type);
    layout.vertex_counts.assign(cell_verts.size(), 0);
    layout.edge_counts.assign(ref.num_edges(), 0);
    layout.face_counts.assign(topology.dim >= 3 ? ref.num_faces() : 0u, 0);

    if (layout.continuity == Continuity::C0 || layout.continuity == Continuity::C1) {
        FE_CHECK_ARG(variable_order_scalar_basis_supported(layout.basis_type),
                     "DofHandler::distributeVariableOrderDofs: mixed-order scalar numbering currently supports only Lagrange-like scalar bases");

        const auto scalar_layout = DofLayoutInfo::Lagrange(layout.polynomial_order,
                                                           topology.dim,
                                                           static_cast<int>(cell_verts.size()));

        std::fill(layout.vertex_counts.begin(), layout.vertex_counts.end(), scalar_layout.dofs_per_vertex);
        std::fill(layout.edge_counts.begin(), layout.edge_counts.end(), scalar_layout.dofs_per_edge);
        if (topology.dim >= 3) {
            for (std::size_t lf = 0; lf < ref.num_faces(); ++lf) {
                layout.face_counts[lf] =
                    scalar_layout.face_dofs_for_vertex_count(ref.face_nodes(lf).size());
            }
        }
        layout.cell_interior_dofs = scalar_layout.dofs_per_cell;

        layout.dof_tags.reserve(static_cast<std::size_t>(elem.num_dofs()));
        for (std::size_t lv = 0; lv < layout.vertex_counts.size(); ++lv) {
            for (LocalIndex d = 0; d < layout.vertex_counts[lv]; ++d) {
                layout.dof_tags.push_back(
                    VariableLocalDofTag{VariableLocalDofKind::Vertex, static_cast<int>(lv), d});
            }
        }
        for (std::size_t le = 0; le < layout.edge_counts.size(); ++le) {
            for (LocalIndex d = 0; d < layout.edge_counts[le]; ++d) {
                layout.dof_tags.push_back(
                    VariableLocalDofTag{VariableLocalDofKind::Edge, static_cast<int>(le), d});
            }
        }
        for (std::size_t lf = 0; lf < layout.face_counts.size(); ++lf) {
            for (LocalIndex d = 0; d < layout.face_counts[lf]; ++d) {
                layout.dof_tags.push_back(
                    VariableLocalDofTag{VariableLocalDofKind::Face, static_cast<int>(lf), d});
            }
        }
        for (LocalIndex d = 0; d < layout.cell_interior_dofs; ++d) {
            layout.dof_tags.push_back(VariableLocalDofTag{VariableLocalDofKind::Cell, -1, d});
        }

        layout.scalar_dofs_per_element =
            static_cast<LocalIndex>(layout.dof_tags.size());
        FE_CHECK_ARG(layout.scalar_dofs_per_element == static_cast<LocalIndex>(elem.num_dofs()),
                     "DofHandler::distributeVariableOrderDofs: scalar cell layout does not match element DOF count");
        return layout;
    }

    if (layout.continuity == Continuity::H_curl || layout.continuity == Continuity::H_div) {
        FE_CHECK_ARG(cell_basis.is_vector_valued(),
                     "DofHandler::distributeVariableOrderDofs: vector continuity requires a vector-valued basis");
        const auto* vb = dynamic_cast<const basis::VectorBasisFunction*>(&cell_basis);
        FE_CHECK_ARG(vb != nullptr,
                     "DofHandler::distributeVariableOrderDofs: vector basis is not a VectorBasisFunction");

        std::vector<LocalIndex> vertex_ord(layout.vertex_counts.size(), 0);
        std::vector<LocalIndex> edge_ord(layout.edge_counts.size(), 0);
        std::vector<LocalIndex> face_ord(layout.face_counts.size(), 0);
        LocalIndex cell_ord = 0;

        const auto associations = vb->dof_associations();
        layout.dof_tags.reserve(associations.size());
        for (const auto& assoc : associations) {
            switch (assoc.entity_type) {
                case basis::DofEntity::Vertex:
                    FE_CHECK_ARG(assoc.entity_id >= 0 &&
                                     static_cast<std::size_t>(assoc.entity_id) < layout.vertex_counts.size(),
                                 "DofHandler::distributeVariableOrderDofs: vector vertex association out of range");
                    layout.dof_tags.push_back(VariableLocalDofTag{
                        VariableLocalDofKind::Vertex,
                        assoc.entity_id,
                        vertex_ord[static_cast<std::size_t>(assoc.entity_id)]++});
                    layout.vertex_counts[static_cast<std::size_t>(assoc.entity_id)] += 1;
                    break;
                case basis::DofEntity::Edge:
                    FE_CHECK_ARG(assoc.entity_id >= 0 &&
                                     static_cast<std::size_t>(assoc.entity_id) < layout.edge_counts.size(),
                                 "DofHandler::distributeVariableOrderDofs: vector edge association out of range");
                    layout.dof_tags.push_back(VariableLocalDofTag{
                        VariableLocalDofKind::Edge,
                        assoc.entity_id,
                        edge_ord[static_cast<std::size_t>(assoc.entity_id)]++});
                    layout.edge_counts[static_cast<std::size_t>(assoc.entity_id)] += 1;
                    break;
                case basis::DofEntity::Face:
                    if (topology.dim >= 3) {
                        FE_CHECK_ARG(assoc.entity_id >= 0 &&
                                         static_cast<std::size_t>(assoc.entity_id) < layout.face_counts.size(),
                                     "DofHandler::distributeVariableOrderDofs: vector face association out of range");
                        layout.dof_tags.push_back(VariableLocalDofTag{
                            VariableLocalDofKind::Face,
                            assoc.entity_id,
                            face_ord[static_cast<std::size_t>(assoc.entity_id)]++});
                        layout.face_counts[static_cast<std::size_t>(assoc.entity_id)] += 1;
                    } else {
                        layout.dof_tags.push_back(
                            VariableLocalDofTag{VariableLocalDofKind::Cell, -1, cell_ord++});
                        layout.cell_interior_dofs += 1;
                    }
                    break;
                case basis::DofEntity::Interior:
                default:
                    layout.dof_tags.push_back(
                        VariableLocalDofTag{VariableLocalDofKind::Cell, -1, cell_ord++});
                    layout.cell_interior_dofs += 1;
                    break;
            }
        }

        layout.scalar_dofs_per_element =
            static_cast<LocalIndex>(layout.dof_tags.size());
        FE_CHECK_ARG(layout.scalar_dofs_per_element == static_cast<LocalIndex>(elem.num_dofs()),
                     "DofHandler::distributeVariableOrderDofs: vector cell layout does not match element DOF count");
        return layout;
    }

    layout.scalar_dofs_per_element =
        static_cast<LocalIndex>(elem.num_dofs() / std::max(1, space.value_dimension()));
    if (space.value_dimension() <= 1 ||
        static_cast<std::size_t>(layout.scalar_dofs_per_element) != elem.num_dofs()) {
        layout.scalar_dofs_per_element = static_cast<LocalIndex>(elem.num_dofs());
    }
    layout.cell_interior_dofs = layout.scalar_dofs_per_element;
    layout.dof_tags.reserve(static_cast<std::size_t>(layout.scalar_dofs_per_element));
    for (LocalIndex d = 0; d < layout.scalar_dofs_per_element; ++d) {
        layout.dof_tags.push_back(VariableLocalDofTag{VariableLocalDofKind::Cell, -1, d});
    }
    return layout;
}

static void reduce_variable_entity_count(LocalIndex count,
                                         GlobalIndex& shared_count) noexcept {
    if (shared_count < 0) {
        shared_count = count;
    } else {
        shared_count = std::min(shared_count, static_cast<GlobalIndex>(count));
    }
}

// =============================================================================
// Mesh Association (used by Mesh convenience overloads)
// =============================================================================

#if DOFHANDLER_HAS_MESH
struct DofHandler::MeshCacheState : MeshObserver {
    struct PointerSnapshot {
        const void* cell2vertex_offsets{nullptr};
        std::size_t cell2vertex_offsets_size{0};
        const void* cell2vertex{nullptr};
        std::size_t cell2vertex_size{0};
        const void* vertex_gids{nullptr};
        std::size_t vertex_gids_size{0};
        const void* vertex_coords{nullptr};
        std::size_t vertex_coords_size{0};
        const void* cell_gids{nullptr};
        std::size_t cell_gids_size{0};
        const void* edge2vertex{nullptr};
        std::size_t edge2vertex_size{0}; // number of edges (pairs)
        const void* edge_gids{nullptr};
        std::size_t edge_gids_size{0};
        const void* face2vertex_offsets{nullptr};
        std::size_t face2vertex_offsets_size{0};
        const void* face2vertex{nullptr};
        std::size_t face2vertex_size{0};
        const void* face_gids{nullptr};
        std::size_t face_gids_size{0};

        bool operator==(const PointerSnapshot& other) const noexcept {
            return cell2vertex_offsets == other.cell2vertex_offsets &&
                   cell2vertex_offsets_size == other.cell2vertex_offsets_size &&
                   cell2vertex == other.cell2vertex &&
                   cell2vertex_size == other.cell2vertex_size &&
                   vertex_gids == other.vertex_gids &&
                   vertex_gids_size == other.vertex_gids_size &&
                   vertex_coords == other.vertex_coords &&
                   vertex_coords_size == other.vertex_coords_size &&
                   cell_gids == other.cell_gids &&
                   cell_gids_size == other.cell_gids_size &&
                   edge2vertex == other.edge2vertex &&
                   edge2vertex_size == other.edge2vertex_size &&
                   edge_gids == other.edge_gids &&
                   edge_gids_size == other.edge_gids_size &&
                   face2vertex_offsets == other.face2vertex_offsets &&
                   face2vertex_offsets_size == other.face2vertex_offsets_size &&
                   face2vertex == other.face2vertex &&
                   face2vertex_size == other.face2vertex_size &&
                   face_gids == other.face_gids &&
                   face_gids_size == other.face_gids_size;
        }
    };

    struct Signature {
        DofLayoutInfo layout{};
        DofNumberingStrategy numbering{DofNumberingStrategy::Sequential};
        bool enable_spatial_locality_ordering{true};
        SpatialCurveType spatial_curve{SpatialCurveType::Morton};
        OwnershipStrategy ownership{OwnershipStrategy::LowestRank};
        GlobalNumberingMode global_numbering{GlobalNumberingMode::OwnerContiguous};
        TopologyCompletion topology_completion{TopologyCompletion::DeriveMissing};
        bool use_canonical_ordering{true};
        bool validate_parallel{false};
        bool reproducible_across_communicators{false};
        bool no_global_collectives{false};
        int my_rank{0};
        int world_size{1};
#if FE_HAS_MPI
        MPI_Comm mpi_comm{MPI_COMM_WORLD};
#endif

        static bool layout_equal(const DofLayoutInfo& a, const DofLayoutInfo& b) noexcept {
            return a.dofs_per_vertex == b.dofs_per_vertex &&
                   a.dofs_per_edge == b.dofs_per_edge &&
                   a.dofs_per_face == b.dofs_per_face &&
                   a.dofs_per_tri_face == b.dofs_per_tri_face &&
                   a.dofs_per_quad_face == b.dofs_per_quad_face &&
                   a.dofs_per_cell == b.dofs_per_cell &&
                   a.num_components == b.num_components &&
                   a.is_continuous == b.is_continuous &&
                   a.tensor_face_dof_layout == b.tensor_face_dof_layout &&
                   a.total_dofs_per_element == b.total_dofs_per_element;
        }

        bool operator==(const Signature& other) const noexcept {
            return layout_equal(layout, other.layout) &&
                   numbering == other.numbering &&
                   enable_spatial_locality_ordering == other.enable_spatial_locality_ordering &&
                   spatial_curve == other.spatial_curve &&
                   ownership == other.ownership &&
                   global_numbering == other.global_numbering &&
                   topology_completion == other.topology_completion &&
                   use_canonical_ordering == other.use_canonical_ordering &&
                   validate_parallel == other.validate_parallel &&
                   reproducible_across_communicators == other.reproducible_across_communicators &&
                   no_global_collectives == other.no_global_collectives &&
                   my_rank == other.my_rank &&
                   world_size == other.world_size
#if FE_HAS_MPI
                   && mpi_comm == other.mpi_comm
#endif
                ;
        }
    };

	    ScopedSubscription subscription{};
	    const void* mesh_identity{nullptr};
	    std::uint64_t relevant_revision{0};
	    std::uint64_t last_seen_revision{0};
	    bool has_snapshot{false};
    PointerSnapshot pointers{};
    Signature signature{};
    std::uint64_t handler_revision{0};

    ~MeshCacheState() override { detach(); }

    void on_mesh_event(MeshEvent event) override {
        switch (event) {
            case MeshEvent::TopologyChanged:
            case MeshEvent::PartitionChanged:
            case MeshEvent::AdaptivityApplied:
                ++relevant_revision;
                break;
            default:
                break;
        }
	    }
	
	    void attach(MeshEventBus& new_bus) {
	        if (subscription.bus() == &new_bus && subscription.is_active()) return;
	        subscription = ScopedSubscription(&new_bus, this);
	    }
	
	    void detach() {
	        subscription.unsubscribe();
	    }

    static PointerSnapshot capturePointers(const MeshBase& mesh) {
        PointerSnapshot snap;
        snap.cell2vertex_offsets = mesh.cell2vertex_offsets().data();
        snap.cell2vertex_offsets_size = mesh.cell2vertex_offsets().size();
        snap.cell2vertex = mesh.cell2vertex().data();
        snap.cell2vertex_size = mesh.cell2vertex().size();
        snap.vertex_gids = mesh.vertex_gids().data();
        snap.vertex_gids_size = mesh.vertex_gids().size();
        snap.vertex_coords = mesh.X_ref().data();
        snap.vertex_coords_size = mesh.X_ref().size();
        snap.cell_gids = mesh.cell_gids().data();
        snap.cell_gids_size = mesh.cell_gids().size();
        snap.edge2vertex = mesh.edge2vertex().data();
        snap.edge2vertex_size = mesh.edge2vertex().size();
        snap.edge_gids = mesh.edge_gids().data();
        snap.edge_gids_size = mesh.edge_gids().size();
        snap.face2vertex_offsets = mesh.face2vertex_offsets().data();
        snap.face2vertex_offsets_size = mesh.face2vertex_offsets().size();
        snap.face2vertex = mesh.face2vertex().data();
        snap.face2vertex_size = mesh.face2vertex().size();
        snap.face_gids = mesh.face_gids().data();
        snap.face_gids_size = mesh.face_gids().size();
        return snap;
    }

    static Signature makeSignature(const DofLayoutInfo& layout, const DofDistributionOptions& options) {
        Signature sig;
        sig.layout = layout;
        sig.numbering = options.numbering;
        sig.enable_spatial_locality_ordering = options.enable_spatial_locality_ordering;
        sig.spatial_curve = options.spatial_curve;
        sig.ownership = options.ownership;
        sig.global_numbering = options.global_numbering;
        sig.topology_completion = options.topology_completion;
        sig.use_canonical_ordering = options.use_canonical_ordering;
        sig.validate_parallel = options.validate_parallel;
        sig.reproducible_across_communicators = options.reproducible_across_communicators;
        sig.no_global_collectives = options.no_global_collectives;
        sig.my_rank = options.my_rank;
        sig.world_size = options.world_size;
#if FE_HAS_MPI
        sig.mpi_comm = options.mpi_comm;
#endif
        return sig;
    }

    [[nodiscard]] bool canSkip(const void* mesh_id,
                               const PointerSnapshot& ptrs,
                               const Signature& sig,
                               std::uint64_t dof_state_revision) const noexcept {
        return has_snapshot &&
               mesh_identity == mesh_id &&
               pointers == ptrs &&
               signature == sig &&
               last_seen_revision == relevant_revision &&
               handler_revision == dof_state_revision;
    }

    void updateSnapshot(const void* mesh_id,
                        const PointerSnapshot& ptrs,
                        const Signature& sig,
                        std::uint64_t dof_state_revision) {
        mesh_identity = mesh_id;
        pointers = ptrs;
        signature = sig;
        handler_revision = dof_state_revision;
        has_snapshot = true;
        last_seen_revision = relevant_revision;
    }
};
#else
struct DofHandler::MeshCacheState {};
#endif

// =============================================================================
// Construction
// =============================================================================

DofHandler::DofHandler() = default;
DofHandler::~DofHandler() = default;

DofHandler::DofHandler(DofHandler&& other) noexcept
    : dof_map_(std::move(other.dof_map_))
    , partition_(std::move(other.partition_))
    , entity_dof_map_(std::move(other.entity_dof_map_))
    , ghost_manager_(std::move(other.ghost_manager_))
    , ghost_dofs_cache_(std::move(other.ghost_dofs_cache_))
    , ghost_cache_valid_(other.ghost_cache_valid_)
    , finalized_(other.finalized_)
    , dof_state_revision_(other.dof_state_revision_)
    , my_rank_(other.my_rank_)
    , world_size_(other.world_size_)
#if FE_HAS_MPI
    , mpi_comm_(other.mpi_comm_)
#endif
    , neighbor_ranks_(std::move(other.neighbor_ranks_))
    , global_numbering_(other.global_numbering_)
    , no_global_collectives_(other.no_global_collectives_)
#if FE_HAS_MPI
    , ghost_exchange_mpi_(std::move(other.ghost_exchange_mpi_))
#endif
    , mesh_cache_(std::move(other.mesh_cache_))
    , n_cells_(other.n_cells_)
    , spatial_dim_(other.spatial_dim_)
    , num_components_(other.num_components_)
    , cell_edge_orient_offsets_(std::move(other.cell_edge_orient_offsets_))
    , cell_edge_orient_data_(std::move(other.cell_edge_orient_data_))
    , cell_face_orient_offsets_(std::move(other.cell_face_orient_offsets_))
    , cell_face_orient_data_(std::move(other.cell_face_orient_data_))
    , spatial_dof_coords_(std::move(other.spatial_dof_coords_))
    , spatial_dof_coord_dim_(other.spatial_dof_coord_dim_)
{
    other.finalized_ = false;
    other.ghost_cache_valid_ = false;
    other.dof_state_revision_ = 0;
    other.spatial_dof_coord_dim_ = 0;
}

DofHandler& DofHandler::operator=(DofHandler&& other) noexcept {
    if (this != &other) {
        dof_map_ = std::move(other.dof_map_);
        partition_ = std::move(other.partition_);
        entity_dof_map_ = std::move(other.entity_dof_map_);
        ghost_manager_ = std::move(other.ghost_manager_);
#if FE_HAS_MPI
        ghost_exchange_mpi_ = std::move(other.ghost_exchange_mpi_);
#endif
        mesh_cache_ = std::move(other.mesh_cache_);
        ghost_dofs_cache_ = std::move(other.ghost_dofs_cache_);
        ghost_cache_valid_ = other.ghost_cache_valid_;
        finalized_ = other.finalized_;
        dof_state_revision_ = other.dof_state_revision_;
        my_rank_ = other.my_rank_;
        world_size_ = other.world_size_;
#if FE_HAS_MPI
        mpi_comm_ = other.mpi_comm_;
#endif
	        neighbor_ranks_ = std::move(other.neighbor_ranks_);
	        global_numbering_ = other.global_numbering_;
	        no_global_collectives_ = other.no_global_collectives_;
	        n_cells_ = other.n_cells_;
	        spatial_dim_ = other.spatial_dim_;
	        num_components_ = other.num_components_;
        cell_edge_orient_offsets_ = std::move(other.cell_edge_orient_offsets_);
        cell_edge_orient_data_ = std::move(other.cell_edge_orient_data_);
        cell_face_orient_offsets_ = std::move(other.cell_face_orient_offsets_);
        cell_face_orient_data_ = std::move(other.cell_face_orient_data_);
        spatial_dof_coords_ = std::move(other.spatial_dof_coords_);
        spatial_dof_coord_dim_ = other.spatial_dof_coord_dim_;

        other.finalized_ = false;
        other.ghost_cache_valid_ = false;
        other.dof_state_revision_ = 0;
        other.spatial_dof_coord_dim_ = 0;
    }
    return *this;
}

// =============================================================================
// DOF Distribution - Mesh-Independent API (Primary Implementation)
// =============================================================================

void DofHandler::distributeDofs(const MeshTopologyInfo& topology,
                                 const DofLayoutInfo& layout,
                                 const DofDistributionOptions& options) {
    checkNotFinalized();
    const auto view = MeshTopologyView::from(topology);
    distributeDofsCore(view, layout, options);
}

void DofHandler::distributeDofs(const MeshTopologyInfo& topology,
                                const spaces::FunctionSpace& space,
                                const DofDistributionOptions& options)
{
    checkNotFinalized();

    FE_THROW_IF(topology.n_cells <= 0, FEException,
                "DofHandler::distributeDofs(MeshTopologyInfo, FunctionSpace): topology has no cells");

    if (space.is_variable_order()) {
        my_rank_ = options.my_rank;
        world_size_ = options.world_size;
        global_numbering_ = options.global_numbering;
        no_global_collectives_ = options.no_global_collectives;
#if FE_HAS_MPI
        mpi_comm_ = options.mpi_comm;
#endif
        n_cells_ = topology.n_cells;
        spatial_dim_ = topology.dim;
        num_components_ = static_cast<LocalIndex>(
            std::max(1, (space.continuity() == Continuity::H_curl || space.continuity() == Continuity::H_div)
                            ? 1
                            : space.value_dimension()));
        dof_map_.setMyRank(my_rank_);
        ++dof_state_revision_;
        const auto view = MeshTopologyView::from(topology);
        return distributeVariableOrderDofs(view, space, options);
    }

    // Infer cell vertex count from cell 0.
    const auto cell0 = topology.getCellVertices(0);
    FE_THROW_IF(cell0.empty(), FEException,
                "DofHandler::distributeDofs(MeshTopologyInfo, FunctionSpace): cell 0 has no vertices");
    const int n_verts = static_cast<int>(cell0.size());

    const auto continuity = space.continuity();
    const int dim = topology.dim;
    const int order = space.polynomial_order();
    const auto total_dofs = static_cast<LocalIndex>(space.dofs_per_element());

    DofLayoutInfo layout{};

    const bool is_conforming =
        (continuity == Continuity::C0 || continuity == Continuity::C1 ||
         continuity == Continuity::H_curl || continuity == Continuity::H_div);
    layout.is_continuous = is_conforming;
    layout.tensor_face_dof_layout = false;

    if (continuity == Continuity::C0 || continuity == Continuity::C1) {
        layout = DofLayoutInfo::Lagrange(order, dim, n_verts);
        layout.total_dofs_per_element = total_dofs;
        layout.num_components = space.value_dimension();
        return distributeDofs(topology, layout, options);
    }

    if (continuity == Continuity::H_curl || continuity == Continuity::H_div) {
        const auto& elem = space.element();
        const auto& b = elem.basis();
        FE_THROW_IF(!b.is_vector_valued(), FEException,
                    "DofHandler::distributeDofs: H(curl)/H(div) space requires a vector-valued basis");
        const auto* vb = dynamic_cast<const basis::VectorBasisFunction*>(&b);
        FE_THROW_IF(vb == nullptr, FEException,
                    "DofHandler::distributeDofs: vector basis is not a VectorBasisFunction");

        const auto assoc = vb->dof_associations();
        const auto cell_type = infer_element_type_from_cell(dim, static_cast<std::size_t>(n_verts));
        const auto ref = (cell_type != ElementType::Unknown)
                             ? elements::ReferenceElement::create(cell_type)
                             : elements::ReferenceElement{};

        const std::size_t num_verts_ref = static_cast<std::size_t>(n_verts);
        const std::size_t num_edges_ref = ref.num_edges();
        const std::size_t num_faces_ref = ref.num_faces();

        std::vector<LocalIndex> per_vertex(num_verts_ref, 0);
        std::vector<LocalIndex> per_edge(num_edges_ref, 0);
        std::vector<LocalIndex> per_face(num_faces_ref, 0);
        LocalIndex interior = 0;

        for (const auto& a : assoc) {
            switch (a.entity_type) {
                case basis::DofEntity::Vertex: {
                    if (a.entity_id >= 0 && static_cast<std::size_t>(a.entity_id) < per_vertex.size()) {
                        per_vertex[static_cast<std::size_t>(a.entity_id)] += 1;
                    }
                    break;
                }
                case basis::DofEntity::Edge: {
                    if (a.entity_id >= 0 && static_cast<std::size_t>(a.entity_id) < per_edge.size()) {
                        per_edge[static_cast<std::size_t>(a.entity_id)] += 1;
                    }
                    break;
                }
                case basis::DofEntity::Face: {
                    if (a.entity_id >= 0 && static_cast<std::size_t>(a.entity_id) < per_face.size()) {
                        per_face[static_cast<std::size_t>(a.entity_id)] += 1;
                    }
                    break;
                }
                case basis::DofEntity::Interior:
                default:
                    interior += 1;
                    break;
            }
        }

        auto uniform_or_zero = [](const std::vector<LocalIndex>& counts) -> std::optional<LocalIndex> {
            if (counts.empty()) return LocalIndex{0};
            LocalIndex v = counts[0];
            for (const auto c : counts) {
                if (c != v) return std::nullopt;
            }
            return v;
        };

        const auto vtx = uniform_or_zero(per_vertex);
        const auto edg = uniform_or_zero(per_edge);
        const auto fac = uniform_or_zero(per_face);
        FE_THROW_IF(!vtx.has_value() || !edg.has_value() || !fac.has_value(), FEException,
                    "DofHandler::distributeDofs: non-uniform per-entity DOF counts are not supported");

        layout.dofs_per_vertex = vtx.value_or(0);
        layout.dofs_per_edge = edg.value_or(0);
        layout.dofs_per_face = fac.value_or(0);
        layout.dofs_per_cell = interior;
        layout.num_components = 1; // coefficients on vector basis functions
        layout.is_continuous = true;
        layout.total_dofs_per_element = total_dofs;
        layout.tensor_face_dof_layout = false;

        return distributeDofs(topology, layout, options);
    }

    // Default: treat as DG.
    layout.is_continuous = false;
    layout.dofs_per_vertex = 0;
    layout.dofs_per_edge = 0;
    layout.dofs_per_face = 0;
    layout.dofs_per_tri_face = 0;
    layout.dofs_per_quad_face = 0;
    layout.tensor_face_dof_layout = false;

    const auto nc = static_cast<LocalIndex>(std::max(1, space.value_dimension()));
    layout.num_components = nc;
    if (nc > 1 && (total_dofs % nc) == 0) {
        layout.dofs_per_cell = total_dofs / nc;
    } else {
        layout.dofs_per_cell = total_dofs;
        layout.num_components = 1;
    }
    layout.total_dofs_per_element = total_dofs;
    distributeDofs(topology, layout, options);
}

void DofHandler::distributeDofsCore(const MeshTopologyView& topology,
                                     const DofLayoutInfo& layout,
                                     const DofDistributionOptions& options) {
		    my_rank_ = options.my_rank;
		    world_size_ = options.world_size;
		    global_numbering_ = options.global_numbering;
		    no_global_collectives_ = options.no_global_collectives;
#if FE_HAS_MPI
	    mpi_comm_ = options.mpi_comm;
#else
	    if (world_size_ > 1) {
	        throw FEException("DofHandler::distributeDofs: MPI world_size>1 but FE is built without MPI support");
    }
#endif
    n_cells_ = topology.n_cells;
    spatial_dim_ = topology.dim;
    num_components_ = static_cast<LocalIndex>(std::max(1, layout.num_components));
    dof_map_.setMyRank(my_rank_);
    ++dof_state_revision_;

    // Clear any previously cached orientation metadata.
    cell_edge_orient_offsets_.clear();
    cell_edge_orient_data_.clear();
    cell_face_orient_offsets_.clear();
    cell_face_orient_data_.clear();

    const MeshTopologyView* topo = &topology;
    MeshTopologyInfo derived_topology;
    MeshTopologyView derived_view;

    const bool need_edges = layout.is_continuous && layout.dofs_per_edge > 0;
    const bool need_faces = layout.is_continuous && layout.has_face_dofs();
    const bool missing_edges = need_edges && (topology.n_edges <= 0 ||
                                              topology.cell2edge_offsets.empty() ||
                                              topology.cell2edge_data.empty());
    const bool missing_faces = need_faces && (topology.n_faces <= 0 ||
                                              topology.cell2face_offsets.empty() ||
                                              topology.cell2face_data.empty());

    if (missing_edges || missing_faces) {
        if (options.topology_completion == TopologyCompletion::RequireComplete) {
            if (missing_edges) {
                throw FEException("DofHandler::distributeDofs: edge-interior DOFs require mesh-provided edge connectivity (cell2edge, edge2vertex, and n_edges > 0)");
            }
            if (missing_faces) {
                throw FEException("DofHandler::distributeDofs: face-interior DOFs require mesh-provided face connectivity (cell2face, face2vertex, and n_faces > 0)");
            }
        }

        derived_topology = topology.materialize();
        if (missing_edges) {
            derive_edge_connectivity(derived_topology);
        }
        if (missing_faces) {
            derive_face_connectivity(derived_topology);
        }
        derived_view = MeshTopologyView::from(derived_topology);
        topo = &derived_view;
    }

    neighbor_ranks_.clear();
    if (world_size_ > 1) {
        if (!topo->neighbor_ranks.empty()) {
            neighbor_ranks_.assign(topo->neighbor_ranks.begin(), topo->neighbor_ranks.end());
        } else if (!topo->cell_owner_ranks.empty()) {
            std::unordered_set<int> nbrs;
            nbrs.reserve(topo->cell_owner_ranks.size());
            for (int r : topo->cell_owner_ranks) {
                if (r >= 0 && r != my_rank_) {
                    nbrs.insert(r);
                }
            }
            neighbor_ranks_.assign(nbrs.begin(), nbrs.end());
        }
        // Normalize: remove self, sort, deduplicate.
        neighbor_ranks_.erase(std::remove(neighbor_ranks_.begin(), neighbor_ranks_.end(), my_rank_),
                              neighbor_ranks_.end());
        std::sort(neighbor_ranks_.begin(), neighbor_ranks_.end());
        neighbor_ranks_.erase(std::unique(neighbor_ranks_.begin(), neighbor_ranks_.end()),
                              neighbor_ranks_.end());
    }

    if (world_size_ > 1) {
        if (layout.is_continuous) {
            distributeCGDofsParallel(*topo, layout, options);
        } else {
            distributeDGDofsParallel(*topo, layout, options);
        }
    } else {
        if (layout.is_continuous) {
            distributeCGDofs(*topo, layout, options);
        } else {
            distributeDGDofs(*topo, layout, options);
        }
    }

    // Optional: compute per-cell edge/face orientations for entity-oriented spaces.
    // This is independent of DOF numbering/renumbering and depends only on
    // canonical vertex ordering + reference element topology.
    if (layout.is_continuous &&
        options.use_canonical_ordering &&
        (layout.dofs_per_edge > 0 || layout.has_face_dofs())) {
        const auto n_cells = static_cast<std::size_t>(std::max<GlobalIndex>(topo->n_cells, 0));

        cell_edge_orient_offsets_.assign(n_cells + 1u, MeshOffset{0});
        cell_face_orient_offsets_.assign(n_cells + 1u, MeshOffset{0});

        for (GlobalIndex c = 0; c < topo->n_cells; ++c) {
            const auto cell_verts = topo->getCellVertices(c);
            const auto base_type = infer_element_type_from_cell(topo->dim, cell_verts.size());
            if (base_type == ElementType::Unknown) {
                cell_edge_orient_offsets_[static_cast<std::size_t>(c) + 1u] =
                    static_cast<MeshOffset>(cell_edge_orient_data_.size());
                cell_face_orient_offsets_[static_cast<std::size_t>(c) + 1u] =
                    static_cast<MeshOffset>(cell_face_orient_data_.size());
                continue;
            }

            const auto ref = elements::ReferenceElement::create(base_type);

            // Edge orientations in reference-edge order.
            cell_edge_orient_offsets_[static_cast<std::size_t>(c)] =
                static_cast<MeshOffset>(cell_edge_orient_data_.size());
            if (layout.dofs_per_edge > 0) {
                for (std::size_t le = 0; le < ref.num_edges(); ++le) {
                    const auto& en = ref.edge_nodes(le);
                    if (en.size() != 2u) {
                        cell_edge_orient_data_.push_back(+1);
                        continue;
                    }
                    const auto lv0 = static_cast<std::size_t>(en[0]);
                    const auto lv1 = static_cast<std::size_t>(en[1]);
                    if (lv0 >= cell_verts.size() || lv1 >= cell_verts.size()) {
                        cell_edge_orient_data_.push_back(+1);
                        continue;
                    }
                    const GlobalIndex gv0 = cell_verts[lv0];
                    const GlobalIndex gv1 = cell_verts[lv1];

                    auto gid_of = [&](GlobalIndex v) -> gid_t {
                        const auto vv = static_cast<std::size_t>(v);
                        if (vv < topo->vertex_gids.size()) {
                            return topo->vertex_gids[vv];
                        }
                        return static_cast<gid_t>(v);
                    };

                    const bool forward = (gv0 >= 0 && gv1 >= 0) ? (gid_of(gv0) <= gid_of(gv1)) : true;
                    cell_edge_orient_data_.push_back(forward ? +1 : -1);
                }
            }
            cell_edge_orient_offsets_[static_cast<std::size_t>(c) + 1u] =
                static_cast<MeshOffset>(cell_edge_orient_data_.size());

            // Face orientations in reference-face order.
            cell_face_orient_offsets_[static_cast<std::size_t>(c)] =
                static_cast<MeshOffset>(cell_face_orient_data_.size());
            if (layout.has_face_dofs() &&
                !topo->cell2face_offsets.empty() &&
                !topo->face2vertex_offsets.empty() &&
                !topo->face2vertex_data.empty()) {
                const auto cell_faces = topo->getCellFaces(c);
                for (std::size_t lf = 0; lf < ref.num_faces() && lf < cell_faces.size(); ++lf) {
                    const GlobalIndex fid = cell_faces[lf];
                    const auto global = topo->getFaceVertices(fid);
                    const auto& fn = ref.face_nodes(lf);

                    spaces::OrientationManager::FaceOrientation orient{};
                    if (fn.size() == 3u && global.size() == 3u) {
                        std::array<int, 3> local_v{};
                        std::array<int, 3> global_v{};
                        for (std::size_t i = 0; i < 3u; ++i) {
                            const auto lv = static_cast<std::size_t>(fn[i]);
                            local_v[i] = (lv < cell_verts.size()) ? static_cast<int>(cell_verts[lv]) : -1;
                            global_v[i] = static_cast<int>(global[i]);
                        }
                        orient = spaces::OrientationManager::triangle_face_orientation(local_v, global_v);
                    } else if (fn.size() == 4u && global.size() == 4u) {
                        std::array<int, 4> local_v{};
                        std::array<int, 4> global_v{};
                        for (std::size_t i = 0; i < 4u; ++i) {
                            const auto lv = static_cast<std::size_t>(fn[i]);
                            local_v[i] = (lv < cell_verts.size()) ? static_cast<int>(cell_verts[lv]) : -1;
                            global_v[i] = static_cast<int>(global[i]);
                        }
                        orient = spaces::OrientationManager::quad_face_orientation(local_v, global_v);
                    } else {
                        // Unsupported/degenerate face topology; keep default.
                        orient.sign = +1;
                    }
                    cell_face_orient_data_.push_back(std::move(orient));
                }
            }
            cell_face_orient_offsets_[static_cast<std::size_t>(c) + 1u] =
                static_cast<MeshOffset>(cell_face_orient_data_.size());
        }
    }

    const bool explicit_spatial =
        options.numbering == DofNumberingStrategy::Morton ||
        options.numbering == DofNumberingStrategy::Hilbert;
    bool apply_default_spatial =
        options.enable_spatial_locality_ordering &&
        options.numbering == DofNumberingStrategy::Sequential;

    // MPI-safe spatial renumbering is implemented for OwnerContiguous numbering only.
    // Keep default behavior compatible with other global numbering modes by
    // auto-disabling default-on spatial ordering in unsupported MPI configurations.
    if (world_size_ > 1 && explicit_spatial &&
        global_numbering_ != GlobalNumberingMode::OwnerContiguous) {
        throw FEException(
            "DofHandler::distributeDofs: explicit MPI spatial DOF numbering requires "
            "GlobalNumberingMode::OwnerContiguous.");
    }
    if (world_size_ > 1 &&
        (global_numbering_ != GlobalNumberingMode::OwnerContiguous ||
         options.reproducible_across_communicators)) {
        apply_default_spatial = false;
    }

    if (explicit_spatial || apply_default_spatial) {
        cacheSpatialDofCoordinates(*topo, layout);
    } else {
        clearSpatialDofCoordinates();
    }

    if (options.numbering != DofNumberingStrategy::Sequential) {
        renumberDofs(options.numbering);
    } else if (apply_default_spatial) {
        const auto curve_strategy = (options.spatial_curve == SpatialCurveType::Hilbert)
                                        ? DofNumberingStrategy::Hilbert
                                        : DofNumberingStrategy::Morton;
        renumberDofs(curve_strategy);
    }

    clearSpatialDofCoordinates();
}

bool DofHandler::hasCellOrientations() const noexcept
{
    return !cell_edge_orient_offsets_.empty() || !cell_face_orient_offsets_.empty();
}

std::span<const spaces::OrientationManager::Sign>
DofHandler::cellEdgeOrientations(GlobalIndex cell_id) const
{
    if (cell_id < 0) {
        return {};
    }
    const auto cid = static_cast<std::size_t>(cell_id);
    if (cid + 1u >= cell_edge_orient_offsets_.size()) {
        return {};
    }
    const auto begin = static_cast<std::size_t>(cell_edge_orient_offsets_[cid]);
    const auto end = static_cast<std::size_t>(cell_edge_orient_offsets_[cid + 1u]);
    if (begin > end || end > cell_edge_orient_data_.size()) {
        return {};
    }
    return {cell_edge_orient_data_.data() + begin, end - begin};
}

std::span<const spaces::OrientationManager::FaceOrientation>
DofHandler::cellFaceOrientations(GlobalIndex cell_id) const
{
    if (cell_id < 0) {
        return {};
    }
    const auto cid = static_cast<std::size_t>(cell_id);
    if (cid + 1u >= cell_face_orient_offsets_.size()) {
        return {};
    }
    const auto begin = static_cast<std::size_t>(cell_face_orient_offsets_[cid]);
    const auto end = static_cast<std::size_t>(cell_face_orient_offsets_[cid + 1u]);
    if (begin > end || end > cell_face_orient_data_.size()) {
        return {};
    }
    return {cell_face_orient_data_.data() + begin, end - begin};
}

void DofHandler::copyCellOrientationsFrom(const DofHandler& other)
{
    checkNotFinalized();
    ++dof_state_revision_;

    const auto n_cells = static_cast<std::size_t>(std::max<GlobalIndex>(dof_map_.getNumCells(), 0));

    if (!other.cell_edge_orient_offsets_.empty()) {
        FE_THROW_IF(other.cell_edge_orient_offsets_.size() != n_cells + 1u, FEException,
                    "DofHandler::copyCellOrientationsFrom: edge orientation table cell count mismatch");
        cell_edge_orient_offsets_ = other.cell_edge_orient_offsets_;
        cell_edge_orient_data_ = other.cell_edge_orient_data_;
    } else {
        cell_edge_orient_offsets_.clear();
        cell_edge_orient_data_.clear();
    }

    if (!other.cell_face_orient_offsets_.empty()) {
        FE_THROW_IF(other.cell_face_orient_offsets_.size() != n_cells + 1u, FEException,
                    "DofHandler::copyCellOrientationsFrom: face orientation table cell count mismatch");
        cell_face_orient_offsets_ = other.cell_face_orient_offsets_;
        cell_face_orient_data_ = other.cell_face_orient_data_;
    } else {
        cell_face_orient_offsets_.clear();
        cell_face_orient_data_.clear();
    }
}

void DofHandler::cacheCellOrientations(const MeshTopologyView& topology,
                                       bool need_edge_orientations,
                                       bool need_face_orientations)
{
    cell_edge_orient_offsets_.clear();
    cell_edge_orient_data_.clear();
    cell_face_orient_offsets_.clear();
    cell_face_orient_data_.clear();

    if (!need_edge_orientations && !need_face_orientations) {
        return;
    }

    const auto n_cells = static_cast<std::size_t>(std::max<GlobalIndex>(topology.n_cells, 0));
    cell_edge_orient_offsets_.assign(n_cells + 1u, MeshOffset{0});
    cell_face_orient_offsets_.assign(n_cells + 1u, MeshOffset{0});

    auto vertex_gid = [&](GlobalIndex v) -> gid_t {
        const auto sv = static_cast<std::size_t>(v);
        if (sv < topology.vertex_gids.size()) {
            return topology.vertex_gids[sv];
        }
        return static_cast<gid_t>(v);
    };

    for (GlobalIndex c = 0; c < topology.n_cells; ++c) {
        const auto sc = static_cast<std::size_t>(c);
        const auto cell_verts = topology.getCellVertices(c);
        const auto base_type = infer_element_type_from_cell(topology.dim, cell_verts.size());
        if (base_type == ElementType::Unknown) {
            cell_edge_orient_offsets_[sc + 1u] = static_cast<MeshOffset>(cell_edge_orient_data_.size());
            cell_face_orient_offsets_[sc + 1u] = static_cast<MeshOffset>(cell_face_orient_data_.size());
            continue;
        }

        const auto ref = elements::ReferenceElement::create(base_type);

        cell_edge_orient_offsets_[sc] = static_cast<MeshOffset>(cell_edge_orient_data_.size());
        if (need_edge_orientations) {
            for (std::size_t le = 0; le < ref.num_edges(); ++le) {
                const auto& en = ref.edge_nodes(le);
                if (en.size() != 2u) {
                    cell_edge_orient_data_.push_back(+1);
                    continue;
                }

                const auto lv0 = static_cast<std::size_t>(en[0]);
                const auto lv1 = static_cast<std::size_t>(en[1]);
                if (lv0 >= cell_verts.size() || lv1 >= cell_verts.size()) {
                    cell_edge_orient_data_.push_back(+1);
                    continue;
                }

                const auto gv0 = cell_verts[lv0];
                const auto gv1 = cell_verts[lv1];
                cell_edge_orient_data_.push_back(vertex_gid(gv0) <= vertex_gid(gv1) ? +1 : -1);
            }
        }
        cell_edge_orient_offsets_[sc + 1u] = static_cast<MeshOffset>(cell_edge_orient_data_.size());

        cell_face_orient_offsets_[sc] = static_cast<MeshOffset>(cell_face_orient_data_.size());
        if (need_face_orientations &&
            !topology.cell2face_offsets.empty() &&
            !topology.face2vertex_offsets.empty() &&
            !topology.face2vertex_data.empty()) {
            const auto cell_faces = topology.getCellFaces(c);
            for (std::size_t lf = 0; lf < ref.num_faces() && lf < cell_faces.size(); ++lf) {
                const auto face_vertices = topology.getFaceVertices(cell_faces[lf]);
                const auto& fn = ref.face_nodes(lf);

                spaces::OrientationManager::FaceOrientation orient{};
                if (fn.size() == 3u && face_vertices.size() == 3u) {
                    std::array<int, 3> local_v{};
                    std::array<int, 3> global_v{};
                    for (std::size_t i = 0; i < 3u; ++i) {
                        const auto lv = static_cast<std::size_t>(fn[i]);
                        local_v[i] = (lv < cell_verts.size()) ? static_cast<int>(cell_verts[lv]) : -1;
                        global_v[i] = static_cast<int>(face_vertices[i]);
                    }
                    orient = spaces::OrientationManager::triangle_face_orientation(local_v, global_v);
                } else if (fn.size() == 4u && face_vertices.size() == 4u) {
                    std::array<int, 4> local_v{};
                    std::array<int, 4> global_v{};
                    for (std::size_t i = 0; i < 4u; ++i) {
                        const auto lv = static_cast<std::size_t>(fn[i]);
                        local_v[i] = (lv < cell_verts.size()) ? static_cast<int>(cell_verts[lv]) : -1;
                        global_v[i] = static_cast<int>(face_vertices[i]);
                    }
                    orient = spaces::OrientationManager::quad_face_orientation(local_v, global_v);
                } else {
                    orient.sign = +1;
                }
                cell_face_orient_data_.push_back(std::move(orient));
            }
        }
        cell_face_orient_offsets_[sc + 1u] = static_cast<MeshOffset>(cell_face_orient_data_.size());
    }
}

void DofHandler::distributeVariableOrderDofs(const MeshTopologyView& topology,
                                             const spaces::FunctionSpace& space,
                                             const DofDistributionOptions& options) {
    if (world_size_ > 1) {
        return distributeVariableOrderDofsParallel(topology, space, options);
    }
    FE_CHECK_ARG(options.numbering == DofNumberingStrategy::Sequential,
                 "DofHandler::distributeVariableOrderDofs currently supports sequential numbering only");

    ghost_manager_.reset();
    ghost_dofs_cache_.clear();
    ghost_cache_valid_ = false;
    clearSpatialDofCoordinates();
    cell_edge_orient_offsets_.clear();
    cell_edge_orient_data_.clear();
    cell_face_orient_offsets_.clear();
    cell_face_orient_data_.clear();

    const MeshTopologyView* topo = &topology;
    MeshTopologyInfo derived_topology;
    MeshTopologyView derived_view;

    std::vector<VariableCellLayout> cell_layouts;
    cell_layouts.reserve(static_cast<std::size_t>(std::max<GlobalIndex>(topology.n_cells, 0)));

    bool need_edges = false;
    bool need_faces = false;
    bool need_edge_orientations = false;
    bool need_face_orientations = false;
    std::size_t max_cell_total_dofs = 0;

    for (GlobalIndex c = 0; c < topology.n_cells; ++c) {
        cell_layouts.push_back(build_variable_cell_layout(space, topology, c));
        const auto& layout = cell_layouts.back();
        need_edges = need_edges || std::any_of(layout.edge_counts.begin(),
                                               layout.edge_counts.end(),
                                               [](LocalIndex n) { return n > 0; });
        need_faces = need_faces || std::any_of(layout.face_counts.begin(),
                                               layout.face_counts.end(),
                                               [](LocalIndex n) { return n > 0; });
        if (layout.continuity == Continuity::H_curl || layout.continuity == Continuity::H_div) {
            need_edge_orientations = need_edge_orientations ||
                                     std::any_of(layout.edge_counts.begin(),
                                                 layout.edge_counts.end(),
                                                 [](LocalIndex n) { return n > 0; });
            need_face_orientations = need_face_orientations ||
                                     std::any_of(layout.face_counts.begin(),
                                                 layout.face_counts.end(),
                                                 [](LocalIndex n) { return n > 0; });
        }
        max_cell_total_dofs = std::max(max_cell_total_dofs, space.dofs_per_element(c));
    }

    const bool missing_edges = need_edges &&
                               (topology.n_edges <= 0 ||
                                topology.cell2edge_offsets.empty() ||
                                topology.cell2edge_data.empty());
    const bool missing_faces = need_faces &&
                               (topology.n_faces <= 0 ||
                                topology.cell2face_offsets.empty() ||
                                topology.cell2face_data.empty());

    if (missing_edges || missing_faces) {
        if (options.topology_completion == TopologyCompletion::RequireComplete) {
            if (missing_edges) {
                throw FEException("DofHandler::distributeVariableOrderDofs: edge DOFs require mesh-provided edge connectivity");
            }
            if (missing_faces) {
                throw FEException("DofHandler::distributeVariableOrderDofs: face DOFs require mesh-provided face connectivity");
            }
        }

        derived_topology = topology.materialize();
        if (missing_edges) {
            derive_edge_connectivity(derived_topology);
        }
        if (missing_faces) {
            derive_face_connectivity(derived_topology);
        }
        derived_view = MeshTopologyView::from(derived_topology);
        topo = &derived_view;

        cell_layouts.clear();
        cell_layouts.reserve(static_cast<std::size_t>(std::max<GlobalIndex>(topo->n_cells, 0)));
        for (GlobalIndex c = 0; c < topo->n_cells; ++c) {
            cell_layouts.push_back(build_variable_cell_layout(space, *topo, c));
        }
    }

    const auto n_vertices = static_cast<std::size_t>(std::max<GlobalIndex>(topo->n_vertices, 0));
    const auto n_edges = static_cast<std::size_t>(std::max<GlobalIndex>(topo->n_edges, 0));
    const auto n_faces = static_cast<std::size_t>(std::max<GlobalIndex>(topo->n_faces, 0));
    const auto n_cells = static_cast<std::size_t>(std::max<GlobalIndex>(topo->n_cells, 0));

    std::vector<GlobalIndex> shared_vertex_count(n_vertices, GlobalIndex{-1});
    std::vector<GlobalIndex> shared_edge_count(n_edges, GlobalIndex{-1});
    std::vector<GlobalIndex> shared_face_count(n_faces, GlobalIndex{-1});

    for (GlobalIndex c = 0; c < topo->n_cells; ++c) {
        const auto sc = static_cast<std::size_t>(c);
        const auto& layout = cell_layouts[sc];
        const auto cell_verts = topo->getCellVertices(c);

        for (std::size_t lv = 0; lv < layout.vertex_counts.size(); ++lv) {
            FE_CHECK_ARG(lv < cell_verts.size(),
                         "DofHandler::distributeVariableOrderDofs: local vertex index out of range");
            reduce_variable_entity_count(layout.vertex_counts[lv],
                                         shared_vertex_count[static_cast<std::size_t>(cell_verts[lv])]);
        }

        if (!layout.edge_counts.empty()) {
            const auto cell_edges = topo->getCellEdges(c);
            FE_CHECK_ARG(cell_edges.size() >= layout.edge_counts.size(),
                         "DofHandler::distributeVariableOrderDofs: cell edge connectivity does not match reference edge count");
            for (std::size_t le = 0; le < layout.edge_counts.size(); ++le) {
                reduce_variable_entity_count(layout.edge_counts[le],
                                             shared_edge_count[static_cast<std::size_t>(cell_edges[le])]);
            }
        }

        if (!layout.face_counts.empty()) {
            const auto cell_faces = topo->getCellFaces(c);
            FE_CHECK_ARG(cell_faces.size() >= layout.face_counts.size(),
                         "DofHandler::distributeVariableOrderDofs: cell face connectivity does not match reference face count");
            for (std::size_t lf = 0; lf < layout.face_counts.size(); ++lf) {
                reduce_variable_entity_count(layout.face_counts[lf],
                                             shared_face_count[static_cast<std::size_t>(cell_faces[lf])]);
            }
        }
    }

    for (auto& n : shared_vertex_count) {
        if (n < 0) n = 0;
    }
    for (auto& n : shared_edge_count) {
        if (n < 0) n = 0;
    }
    for (auto& n : shared_face_count) {
        if (n < 0) n = 0;
    }

    const auto nc = static_cast<GlobalIndex>(std::max<LocalIndex>(1, num_components_));

    std::vector<GlobalIndex> vertex_first_dof(n_vertices, -1);
    std::vector<GlobalIndex> edge_first_dof(n_edges, -1);
    std::vector<GlobalIndex> face_first_dof(n_faces, -1);
    std::vector<GlobalIndex> cell_private_first_dof(n_cells, -1);
    std::vector<LocalIndex> cell_private_count(n_cells, 0);

    entity_dof_map_ = std::make_unique<EntityDofMap>();
    entity_dof_map_->reserve(topo->n_vertices, topo->n_edges, topo->n_faces, topo->n_cells);

    GlobalIndex next_dof = 0;

    for (GlobalIndex v = 0; v < topo->n_vertices; ++v) {
        const auto count = static_cast<LocalIndex>(shared_vertex_count[static_cast<std::size_t>(v)]);
        if (count <= 0) continue;
        vertex_first_dof[static_cast<std::size_t>(v)] = next_dof;
        std::vector<GlobalIndex> dofs;
        dofs.reserve(static_cast<std::size_t>(count));
        for (LocalIndex d = 0; d < count; ++d) {
            dofs.push_back(next_dof++);
        }
        entity_dof_map_->setVertexDofs(v, dofs);
    }

    for (GlobalIndex e = 0; e < topo->n_edges; ++e) {
        const auto count = static_cast<LocalIndex>(shared_edge_count[static_cast<std::size_t>(e)]);
        if (count <= 0) continue;
        edge_first_dof[static_cast<std::size_t>(e)] = next_dof;
        std::vector<GlobalIndex> dofs;
        dofs.reserve(static_cast<std::size_t>(count));
        for (LocalIndex d = 0; d < count; ++d) {
            dofs.push_back(next_dof++);
        }
        entity_dof_map_->setEdgeDofs(e, dofs);
    }

    for (GlobalIndex f = 0; f < topo->n_faces; ++f) {
        const auto count = static_cast<LocalIndex>(shared_face_count[static_cast<std::size_t>(f)]);
        if (count <= 0) continue;
        face_first_dof[static_cast<std::size_t>(f)] = next_dof;
        std::vector<GlobalIndex> dofs;
        dofs.reserve(static_cast<std::size_t>(count));
        for (LocalIndex d = 0; d < count; ++d) {
            dofs.push_back(next_dof++);
        }
        entity_dof_map_->setFaceDofs(f, dofs);
    }

    for (GlobalIndex c = 0; c < topo->n_cells; ++c) {
        const auto sc = static_cast<std::size_t>(c);
        const auto& layout = cell_layouts[sc];
        const auto cell_verts = topo->getCellVertices(c);
        const auto cell_edges = layout.edge_counts.empty() ? std::span<const MeshIndex>{} : topo->getCellEdges(c);
        const auto cell_faces = layout.face_counts.empty() ? std::span<const MeshIndex>{} : topo->getCellFaces(c);

        LocalIndex private_count = layout.cell_interior_dofs;
        for (std::size_t lv = 0; lv < layout.vertex_counts.size(); ++lv) {
            const auto gv = static_cast<std::size_t>(cell_verts[lv]);
            private_count += std::max<LocalIndex>(
                0, layout.vertex_counts[lv] - static_cast<LocalIndex>(shared_vertex_count[gv]));
        }
        for (std::size_t le = 0; le < layout.edge_counts.size(); ++le) {
            const auto ge = static_cast<std::size_t>(cell_edges[le]);
            private_count += std::max<LocalIndex>(
                0, layout.edge_counts[le] - static_cast<LocalIndex>(shared_edge_count[ge]));
        }
        for (std::size_t lf = 0; lf < layout.face_counts.size(); ++lf) {
            const auto gf = static_cast<std::size_t>(cell_faces[lf]);
            private_count += std::max<LocalIndex>(
                0, layout.face_counts[lf] - static_cast<LocalIndex>(shared_face_count[gf]));
        }

        cell_private_count[sc] = private_count;
        if (private_count > 0) {
            cell_private_first_dof[sc] = next_dof;
            std::vector<GlobalIndex> dofs;
            dofs.reserve(static_cast<std::size_t>(private_count));
            for (LocalIndex d = 0; d < private_count; ++d) {
                dofs.push_back(next_dof++);
            }
            entity_dof_map_->setCellInteriorDofs(c, dofs);
        }
    }

    const GlobalIndex scalar_total_dofs = next_dof;
    const GlobalIndex total_dofs = scalar_total_dofs * nc;

    dof_map_.reserve(topo->n_cells, static_cast<LocalIndex>(max_cell_total_dofs));
    for (GlobalIndex c = 0; c < topo->n_cells; ++c) {
        const auto sc = static_cast<std::size_t>(c);
        const auto& layout = cell_layouts[sc];
        const auto cell_verts = topo->getCellVertices(c);
        const auto cell_edges = layout.edge_counts.empty() ? std::span<const MeshIndex>{} : topo->getCellEdges(c);
        const auto cell_faces = layout.face_counts.empty() ? std::span<const MeshIndex>{} : topo->getCellFaces(c);

        std::vector<GlobalIndex> scalar_cell_dofs;
        scalar_cell_dofs.reserve(static_cast<std::size_t>(layout.scalar_dofs_per_element));
        const GlobalIndex private_base = cell_private_first_dof[sc];
        LocalIndex private_cursor = 0;

        for (const auto& tag : layout.dof_tags) {
            GlobalIndex dof = -1;
            switch (tag.kind) {
                case VariableLocalDofKind::Vertex: {
                    const auto gv = static_cast<std::size_t>(cell_verts[static_cast<std::size_t>(tag.local_entity_id)]);
                    const auto shared = static_cast<LocalIndex>(shared_vertex_count[gv]);
                    if (tag.ordinal < shared) {
                        dof = vertex_first_dof[gv] + static_cast<GlobalIndex>(tag.ordinal);
                    }
                    break;
                }
                case VariableLocalDofKind::Edge: {
                    const auto ge = static_cast<std::size_t>(cell_edges[static_cast<std::size_t>(tag.local_entity_id)]);
                    const auto shared = static_cast<LocalIndex>(shared_edge_count[ge]);
                    if (tag.ordinal < shared) {
                        dof = edge_first_dof[ge] + static_cast<GlobalIndex>(tag.ordinal);
                    }
                    break;
                }
                case VariableLocalDofKind::Face: {
                    const auto gf = static_cast<std::size_t>(cell_faces[static_cast<std::size_t>(tag.local_entity_id)]);
                    const auto shared = static_cast<LocalIndex>(shared_face_count[gf]);
                    if (tag.ordinal < shared) {
                        dof = face_first_dof[gf] + static_cast<GlobalIndex>(tag.ordinal);
                    }
                    break;
                }
                case VariableLocalDofKind::Cell:
                default:
                    break;
            }

            if (dof < 0) {
                FE_CHECK_ARG(private_base >= 0,
                             "DofHandler::distributeVariableOrderDofs: missing cell-private DOF block");
                dof = private_base + static_cast<GlobalIndex>(private_cursor++);
            }
            scalar_cell_dofs.push_back(dof);
        }

        FE_CHECK_ARG(private_cursor == cell_private_count[sc],
                     "DofHandler::distributeVariableOrderDofs: cell-private DOF count mismatch");

        std::vector<GlobalIndex> cell_dofs;
        if (nc > 1) {
            cell_dofs.reserve(scalar_cell_dofs.size() * static_cast<std::size_t>(nc));
            for (GlobalIndex comp = 0; comp < nc; ++comp) {
                const GlobalIndex offset = comp * scalar_total_dofs;
                for (const auto dof : scalar_cell_dofs) {
                    cell_dofs.push_back(dof + offset);
                }
            }
        } else {
            cell_dofs = std::move(scalar_cell_dofs);
        }

        FE_CHECK_ARG(cell_dofs.size() == space.dofs_per_element(c),
                     "DofHandler::distributeVariableOrderDofs: cell DOF count does not match the selected element");
        dof_map_.setCellDofs(c, cell_dofs);
    }

    dof_map_.setNumDofs(total_dofs);
    dof_map_.setNumLocalDofs(total_dofs);
    partition_ = DofPartition(0, total_dofs, {});
    partition_.setGlobalSize(total_dofs);

    if (nc > 1) {
        auto expanded_entity = std::make_unique<EntityDofMap>();
        expanded_entity->reserve(topo->n_vertices, topo->n_edges, topo->n_faces, topo->n_cells);

        auto expand_range = [&](GlobalIndex base, LocalIndex count) -> std::vector<GlobalIndex> {
            std::vector<GlobalIndex> out;
            if (count <= 0 || base < 0) {
                return out;
            }
            out.reserve(static_cast<std::size_t>(count) * static_cast<std::size_t>(nc));
            for (GlobalIndex comp = 0; comp < nc; ++comp) {
                const GlobalIndex offset = comp * scalar_total_dofs;
                for (LocalIndex d = 0; d < count; ++d) {
                    out.push_back(base + static_cast<GlobalIndex>(d) + offset);
                }
            }
            return out;
        };

        for (GlobalIndex v = 0; v < topo->n_vertices; ++v) {
            const auto sv = static_cast<std::size_t>(v);
            if (shared_vertex_count[sv] <= 0 || vertex_first_dof[sv] < 0) continue;
            expanded_entity->setVertexDofs(v, expand_range(vertex_first_dof[sv], static_cast<LocalIndex>(shared_vertex_count[sv])));
        }
        for (GlobalIndex e = 0; e < topo->n_edges; ++e) {
            const auto se = static_cast<std::size_t>(e);
            if (shared_edge_count[se] <= 0 || edge_first_dof[se] < 0) continue;
            expanded_entity->setEdgeDofs(e, expand_range(edge_first_dof[se], static_cast<LocalIndex>(shared_edge_count[se])));
        }
        for (GlobalIndex f = 0; f < topo->n_faces; ++f) {
            const auto sf = static_cast<std::size_t>(f);
            if (shared_face_count[sf] <= 0 || face_first_dof[sf] < 0) continue;
            expanded_entity->setFaceDofs(f, expand_range(face_first_dof[sf], static_cast<LocalIndex>(shared_face_count[sf])));
        }
        for (GlobalIndex c = 0; c < topo->n_cells; ++c) {
            const auto sc = static_cast<std::size_t>(c);
            if (cell_private_count[sc] <= 0 || cell_private_first_dof[sc] < 0) continue;
            expanded_entity->setCellInteriorDofs(c, expand_range(cell_private_first_dof[sc], cell_private_count[sc]));
        }

        entity_dof_map_ = std::move(expanded_entity);
    }

    entity_dof_map_->buildReverseMapping();
    entity_dof_map_->finalize();
    cacheCellOrientations(*topo, need_edge_orientations, need_face_orientations);
}

void DofHandler::buildVariableOrderConstraints(const MeshTopologyInfo& topology_info,
                                               const spaces::FunctionSpace& space,
                                               constraints::AffineConstraints& constraints) const {
    FE_CHECK_ARG(space.is_variable_order(),
                 "DofHandler::buildVariableOrderConstraints requires a variable-order space");

    const auto* adaptive = dynamic_cast<const spaces::AdaptiveSpace*>(&space);
    FE_CHECK_ARG(adaptive != nullptr,
                 "DofHandler::buildVariableOrderConstraints currently requires AdaptiveSpace");

    FE_CHECK_NOT_NULL(getEntityDofMap(),
                      "DofHandler::buildVariableOrderConstraints entity_dof_map");

    MeshTopologyView topo = MeshTopologyView::from(topology_info);
    MeshTopologyInfo derived_topology;

    if (topo.dim == 2 && (topo.n_edges <= 0 || topo.cell2edge_offsets.empty() || topo.cell2edge_data.empty())) {
        derived_topology = topology_info;
        derive_edge_connectivity(derived_topology);
        topo = MeshTopologyView::from(derived_topology);
    }
    if (topo.dim == 3 && (topo.n_faces <= 0 || topo.cell2face_offsets.empty() || topo.cell2face_data.empty())) {
        if (derived_topology.n_cells == 0) {
            derived_topology = topology_info;
        }
        derive_face_connectivity(derived_topology);
        topo = MeshTopologyView::from(derived_topology);
    }

    auto vertex_id = [&](MeshIndex local_vertex) -> gid_t {
        const auto sv = static_cast<std::size_t>(local_vertex);
        if (!topo.vertex_gids.empty() && sv < topo.vertex_gids.size()) {
            return topo.vertex_gids[sv];
        }
        return static_cast<gid_t>(local_vertex);
    };

    auto interface_vertices = [&](GlobalIndex cell_id, int local_face_id) {
        const auto cell_verts = topo.getCellVertices(cell_id);
        const ElementType cell_type = infer_element_type_from_cell(topo.dim, cell_verts.size());
        FE_CHECK_ARG(cell_type != ElementType::Unknown,
                     "DofHandler::buildVariableOrderConstraints: unsupported cell topology");
        const auto ref = elements::ReferenceElement::create(cell_type);
        FE_CHECK_ARG(local_face_id >= 0 && static_cast<std::size_t>(local_face_id) < ref.num_faces(),
                     "DofHandler::buildVariableOrderConstraints: local face index out of range");
        const auto& face_nodes = ref.face_nodes(static_cast<std::size_t>(local_face_id));
        std::vector<gid_t> ids;
        ids.reserve(face_nodes.size());
        for (auto lv : face_nodes) {
            FE_CHECK_ARG(static_cast<std::size_t>(lv) < cell_verts.size(),
                         "DofHandler::buildVariableOrderConstraints: face vertex index out of range");
            ids.push_back(vertex_id(cell_verts[static_cast<std::size_t>(lv)]));
        }
        return ids;
    };

    auto trace_global_dofs = [&](GlobalIndex cell_id, const spaces::TraceSpace& trace_space) {
        std::vector<GlobalIndex> dofs;
        const auto cell_dofs = getCellDofs(cell_id);
        const auto local = trace_space.face_dof_indices();
        dofs.reserve(local.size());
        for (int lid : local) {
            FE_CHECK_ARG(lid >= 0 && static_cast<std::size_t>(lid) < cell_dofs.size(),
                         "DofHandler::buildVariableOrderConstraints: trace local DOF out of range");
            dofs.push_back(cell_dofs[static_cast<std::size_t>(lid)]);
        }
        return dofs;
    };

    auto make_point_map = [](ElementType face_type,
                             const std::vector<gid_t>& src_vertices,
                             const std::vector<gid_t>& dst_vertices) -> spaces::SpaceInterpolation::ReferencePointMap {
        FE_CHECK_ARG(src_vertices.size() == dst_vertices.size(),
                     "DofHandler::buildVariableOrderConstraints: interface vertex size mismatch");

        if (face_type == ElementType::Line2) {
            FE_CHECK_ARG(src_vertices.size() == 2u,
                         "DofHandler::buildVariableOrderConstraints: line interface expects 2 vertices");
            if (src_vertices == dst_vertices) {
                return {};
            }
            FE_CHECK_ARG(src_vertices[0] == dst_vertices[1] && src_vertices[1] == dst_vertices[0],
                         "DofHandler::buildVariableOrderConstraints: unsupported edge orientation");
            return [](const spaces::FunctionSpace::Value& xi_dst) {
                spaces::FunctionSpace::Value xi_src = xi_dst;
                xi_src[0] = -xi_src[0];
                return xi_src;
            };
        }

        const auto verts = canonical_face_vertices(face_type);
        FE_CHECK_ARG(!verts.empty(),
                     "DofHandler::buildVariableOrderConstraints: unsupported face shape");

        std::vector<int> dst_to_src(dst_vertices.size(), -1);
        for (std::size_t j = 0; j < dst_vertices.size(); ++j) {
            auto it = std::find(src_vertices.begin(), src_vertices.end(), dst_vertices[j]);
            FE_CHECK_ARG(it != src_vertices.end(),
                         "DofHandler::buildVariableOrderConstraints: interface vertex sets do not match");
            dst_to_src[j] = static_cast<int>(std::distance(src_vertices.begin(), it));
        }

        const auto map = compute_face_affine_from_vertex_map(verts, dst_to_src);
        return [map](const spaces::FunctionSpace::Value& xi_dst) {
            return map.apply(xi_dst);
        };
    };

    auto make_value_transform = [](const spaces::TraceSpace& src_trace,
                                   const spaces::TraceSpace& dst_trace) -> spaces::SpaceInterpolation::ValueTransform {
        if (src_trace.field_type() != FieldType::Scalar ||
            dst_trace.field_type() != FieldType::Scalar ||
            src_trace.trace_kind() == spaces::TraceKind::Value ||
            dst_trace.trace_kind() == spaces::TraceKind::Value) {
            return {};
        }

        Real sign = Real(1);
        if (src_trace.trace_kind() == spaces::TraceKind::Normal &&
            dst_trace.trace_kind() == spaces::TraceKind::Normal) {
            sign = src_trace.face_normal().dot(dst_trace.face_normal());
        } else if (src_trace.trace_kind() == spaces::TraceKind::Tangential &&
                   dst_trace.trace_kind() == spaces::TraceKind::Tangential) {
            sign = src_trace.face_tangent().dot(dst_trace.face_tangent());
        }
        const Real scale = (sign >= Real(0)) ? Real(1) : Real(-1);
        if (std::abs(scale - Real(1)) < Real(1e-12)) {
            return {};
        }

        return [scale](const spaces::FunctionSpace::Value& value) {
            spaces::FunctionSpace::Value out = value;
            out[0] *= scale;
            return out;
        };
    };

    struct InterfacePair {
        GlobalIndex entity_id{-1};
        std::array<GlobalIndex, 2> cells{{-1, -1}};
        std::array<int, 2> local_faces{{-1, -1}};
        ElementType face_type{ElementType::Unknown};
    };

    std::vector<InterfacePair> interfaces;
    if (topo.dim == 2) {
        std::vector<std::vector<std::pair<GlobalIndex, int>>> edge_cells(
            static_cast<std::size_t>(std::max<GlobalIndex>(0, topo.n_edges)));
        for (GlobalIndex c = 0; c < topo.n_cells; ++c) {
            const auto cell_edges = topo.getCellEdges(c);
            for (std::size_t le = 0; le < cell_edges.size(); ++le) {
                const auto se = static_cast<std::size_t>(cell_edges[le]);
                FE_CHECK_ARG(se < edge_cells.size(),
                             "DofHandler::buildVariableOrderConstraints: cell edge index out of range");
                edge_cells[se].push_back({c, static_cast<int>(le)});
            }
        }
        for (std::size_t e = 0; e < edge_cells.size(); ++e) {
            if (edge_cells[e].size() != 2u) {
                continue;
            }
            interfaces.push_back(InterfacePair{
                static_cast<GlobalIndex>(e),
                {edge_cells[e][0].first, edge_cells[e][1].first},
                {edge_cells[e][0].second, edge_cells[e][1].second},
                ElementType::Line2});
        }
    } else if (topo.dim == 3) {
        std::vector<std::vector<std::pair<GlobalIndex, int>>> face_cells(
            static_cast<std::size_t>(std::max<GlobalIndex>(0, topo.n_faces)));
        for (GlobalIndex c = 0; c < topo.n_cells; ++c) {
            const auto cell_faces = topo.getCellFaces(c);
            const auto cell_verts = topo.getCellVertices(c);
            const auto cell_type = infer_element_type_from_cell(topo.dim, cell_verts.size());
            const auto ref = elements::ReferenceElement::create(cell_type);
            for (std::size_t lf = 0; lf < cell_faces.size(); ++lf) {
                const auto sf = static_cast<std::size_t>(cell_faces[lf]);
                FE_CHECK_ARG(sf < face_cells.size(),
                             "DofHandler::buildVariableOrderConstraints: cell face index out of range");
                face_cells[sf].push_back({c, static_cast<int>(lf)});
            }
        }
        for (GlobalIndex f = 0; f < topo.n_faces; ++f) {
            const auto sf = static_cast<std::size_t>(f);
            if (sf >= face_cells.size() || face_cells[sf].size() != 2u) {
                continue;
            }
            const auto fv = topo.getFaceVertices(f);
            ElementType face_type = ElementType::Unknown;
            if (fv.size() == 3u) face_type = ElementType::Triangle3;
            else if (fv.size() == 4u) face_type = ElementType::Quad4;
            interfaces.push_back(InterfacePair{
                f,
                {face_cells[sf][0].first, face_cells[sf][1].first},
                {face_cells[sf][0].second, face_cells[sf][1].second},
                face_type});
        }
    }

    for (const auto& interface : interfaces) {
        GlobalIndex coarse_cell = interface.cells[0];
        GlobalIndex fine_cell = interface.cells[1];
        int coarse_face = interface.local_faces[0];
        int fine_face = interface.local_faces[1];

        const int order0 = space.polynomial_order(interface.cells[0]);
        const int order1 = space.polynomial_order(interface.cells[1]);
        if (order0 == order1) {
            continue;
        }
        if (order0 > order1) {
            coarse_cell = interface.cells[1];
            fine_cell = interface.cells[0];
            coarse_face = interface.local_faces[1];
            fine_face = interface.local_faces[0];
        }

        auto coarse_space = adaptive->element_space_ptr(coarse_cell);
        auto fine_space = adaptive->element_space_ptr(fine_cell);
        FE_CHECK_NOT_NULL(coarse_space.get(),
                          "DofHandler::buildVariableOrderConstraints coarse_space");
        FE_CHECK_NOT_NULL(fine_space.get(),
                          "DofHandler::buildVariableOrderConstraints fine_space");

        spaces::TraceSpace coarse_trace(coarse_space, coarse_face);
        spaces::TraceSpace fine_trace(fine_space, fine_face);

        const auto coarse_dofs = trace_global_dofs(coarse_cell, coarse_trace);
        const auto fine_dofs = trace_global_dofs(fine_cell, fine_trace);
        if (coarse_dofs.empty() || fine_dofs.empty()) {
            continue;
        }

        const bool vector_hierarchical_trace =
            (coarse_space->continuity() == Continuity::H_curl ||
             coarse_space->continuity() == Continuity::H_div) &&
            coarse_space->continuity() == fine_space->continuity();

        if (vector_hierarchical_trace) {
            std::unordered_set<GlobalIndex> coarse_dof_set(coarse_dofs.begin(), coarse_dofs.end());
            for (const GlobalIndex slave : fine_dofs) {
                if (coarse_dof_set.count(slave) > 0 || constraints.isConstrained(slave)) {
                    continue;
                }
                constraints.addLine(slave);
            }
            continue;
        }

        const auto coarse_vertices = interface_vertices(coarse_cell, coarse_face);
        const auto fine_vertices = interface_vertices(fine_cell, fine_face);
        const auto point_map = make_point_map(interface.face_type, coarse_vertices, fine_vertices);
        const auto value_transform = make_value_transform(coarse_trace, fine_trace);

        const auto transfer =
            spaces::SpaceInterpolation::build_transfer_operator(coarse_trace,
                                                                fine_trace,
                                                                point_map,
                                                                value_transform);

        std::unordered_set<GlobalIndex> coarse_dof_set(coarse_dofs.begin(), coarse_dofs.end());
        for (std::size_t row = 0; row < fine_dofs.size(); ++row) {
            const GlobalIndex slave = fine_dofs[row];
            if (coarse_dof_set.count(slave) > 0 || constraints.isConstrained(slave)) {
                continue;
            }

            bool has_master = false;
            constraints.addLine(slave);
            for (std::size_t col = 0; col < coarse_dofs.size(); ++col) {
                const Real weight = transfer(row, col);
                if (std::abs(weight) <= Real(1e-12)) {
                    continue;
                }
                constraints.addEntry(slave, coarse_dofs[col], weight);
                has_master = true;
            }

            FE_CHECK_ARG(has_master,
                         "DofHandler::buildVariableOrderConstraints: generated an empty constraint row");
        }
    }
}

void DofHandler::distributeVariableOrderDofsParallel(const MeshTopologyView& topology,
                                                     const spaces::FunctionSpace& space,
                                                     const DofDistributionOptions& options) {
#if !FE_HAS_MPI
    (void)topology;
    (void)space;
    (void)options;
    throw FEException("DofHandler::distributeVariableOrderDofsParallel: FE built without MPI support");
#else
    FE_CHECK_ARG(world_size_ > 1,
                 "DofHandler::distributeVariableOrderDofsParallel requires MPI execution");
    FE_CHECK_ARG(options.numbering == DofNumberingStrategy::Sequential,
                 "DofHandler::distributeVariableOrderDofsParallel currently supports sequential numbering only");
    FE_CHECK_ARG(global_numbering_ == GlobalNumberingMode::OwnerContiguous,
                 "DofHandler::distributeVariableOrderDofsParallel currently supports OwnerContiguous global numbering only");
    FE_CHECK_ARG(!topology.vertex_gids.empty() &&
                     topology.vertex_gids.size() == static_cast<std::size_t>(topology.n_vertices),
                 "DofHandler::distributeVariableOrderDofsParallel requires vertex_gids");

    ghost_manager_.reset();
    ghost_dofs_cache_.clear();
    ghost_cache_valid_ = false;
    clearSpatialDofCoordinates();
    cell_edge_orient_offsets_.clear();
    cell_edge_orient_data_.clear();
    cell_face_orient_offsets_.clear();
    cell_face_orient_data_.clear();

    const MeshTopologyView* topo = &topology;
    MeshTopologyInfo derived_topology;
    MeshTopologyView derived_view;

    std::vector<VariableCellLayout> cell_layouts;
    cell_layouts.reserve(static_cast<std::size_t>(std::max<GlobalIndex>(topology.n_cells, 0)));

    bool need_edges = false;
    bool need_faces = false;
    bool has_edge_entities = false;
    bool has_face_entities = false;
    bool need_edge_orientations = false;
    bool need_face_orientations = false;
    std::size_t max_cell_total_dofs = 0;
    for (GlobalIndex c = 0; c < topology.n_cells; ++c) {
        cell_layouts.push_back(build_variable_cell_layout(space, topology, c));
        const auto& layout = cell_layouts.back();
        has_edge_entities = has_edge_entities || !layout.edge_counts.empty();
        has_face_entities = has_face_entities || !layout.face_counts.empty();
        need_edges = need_edges || std::any_of(layout.edge_counts.begin(),
                                               layout.edge_counts.end(),
                                               [](LocalIndex n) { return n > 0; });
        need_faces = need_faces || std::any_of(layout.face_counts.begin(),
                                               layout.face_counts.end(),
                                               [](LocalIndex n) { return n > 0; });
        if (layout.continuity == Continuity::H_curl || layout.continuity == Continuity::H_div) {
            need_edge_orientations = need_edge_orientations ||
                                     std::any_of(layout.edge_counts.begin(),
                                                 layout.edge_counts.end(),
                                                 [](LocalIndex n) { return n > 0; });
            need_face_orientations = need_face_orientations ||
                                     std::any_of(layout.face_counts.begin(),
                                                 layout.face_counts.end(),
                                                 [](LocalIndex n) { return n > 0; });
        }
        max_cell_total_dofs = std::max(max_cell_total_dofs, space.dofs_per_element(c));
    }

    const bool missing_edges = has_edge_entities &&
                               (topology.n_edges <= 0 ||
                                topology.cell2edge_offsets.empty() ||
                                topology.cell2edge_data.empty() ||
                                topology.edge2vertex_data.size() <
                                    static_cast<std::size_t>(2) * static_cast<std::size_t>(std::max<GlobalIndex>(0, topology.n_edges)));
    const bool missing_faces = has_face_entities &&
                               (topology.n_faces <= 0 ||
                                topology.cell2face_offsets.empty() ||
                                topology.cell2face_data.empty() ||
                                topology.face2vertex_offsets.empty() ||
                                topology.face2vertex_data.empty());

    if (missing_edges || missing_faces) {
        if (options.topology_completion == TopologyCompletion::RequireComplete) {
            if (missing_edges) {
                throw FEException("DofHandler::distributeVariableOrderDofsParallel: edge DOFs require mesh-provided edge connectivity");
            }
            if (missing_faces) {
                throw FEException("DofHandler::distributeVariableOrderDofsParallel: face DOFs require mesh-provided face connectivity");
            }
        }

        derived_topology = topology.materialize();
        if (missing_edges) {
            derive_edge_connectivity(derived_topology);
        }
        if (missing_faces) {
            derive_face_connectivity(derived_topology);
        }
        derived_view = MeshTopologyView::from(derived_topology);
        topo = &derived_view;

        cell_layouts.clear();
        cell_layouts.reserve(static_cast<std::size_t>(std::max<GlobalIndex>(topo->n_cells, 0)));
        for (GlobalIndex c = 0; c < topo->n_cells; ++c) {
            cell_layouts.push_back(build_variable_cell_layout(space, *topo, c));
        }
    }

    struct GidHash {
        std::size_t operator()(gid_t gid) const noexcept {
            return static_cast<std::size_t>(mix_u64(static_cast<std::uint64_t>(gid)));
        }
    };

    const auto n_vertices = static_cast<std::size_t>(std::max<GlobalIndex>(0, topo->n_vertices));
    const auto n_edges = static_cast<std::size_t>(std::max<GlobalIndex>(0, topo->n_edges));
    const auto n_faces = static_cast<std::size_t>(std::max<GlobalIndex>(0, topo->n_faces));
    const auto n_cells = static_cast<std::size_t>(std::max<GlobalIndex>(0, topo->n_cells));

    std::vector<gid_t> vertex_keys = std::vector<gid_t>(topo->vertex_gids.begin(), topo->vertex_gids.end());
    std::vector<int> vertex_touch(n_vertices, -1);
    std::vector<gid_t> vertex_cell_gid_candidate(n_vertices, std::numeric_limits<gid_t>::max());
    std::vector<int> vertex_cell_owner_candidate(n_vertices, -1);
    std::vector<LocalIndex> vertex_local_count(n_vertices, std::numeric_limits<LocalIndex>::max());

    std::vector<EdgeKey> edge_keys;
    std::vector<int> edge_touch;
    std::vector<gid_t> edge_cell_gid_candidate;
    std::vector<int> edge_cell_owner_candidate;
    std::vector<LocalIndex> edge_local_count;
    if (has_edge_entities && topo->n_edges > 0) {
        edge_keys.resize(n_edges);
        edge_touch.assign(n_edges, -1);
        edge_cell_gid_candidate.assign(n_edges, std::numeric_limits<gid_t>::max());
        edge_cell_owner_candidate.assign(n_edges, -1);
        edge_local_count.assign(n_edges, std::numeric_limits<LocalIndex>::max());
        for (GlobalIndex e = 0; e < topo->n_edges; ++e) {
            const auto [v0, v1] = topo->getEdgeVertices(e);
            FE_CHECK_ARG(v0 >= 0 && v1 >= 0,
                         "DofHandler::distributeVariableOrderDofsParallel: invalid edge2vertex entry");
            const gid_t gid0 = topo->vertex_gids[static_cast<std::size_t>(v0)];
            const gid_t gid1 = topo->vertex_gids[static_cast<std::size_t>(v1)];
            edge_keys[static_cast<std::size_t>(e)] = EdgeKey{std::min(gid0, gid1), std::max(gid0, gid1)};
        }
    }

    std::vector<std::uint8_t> face_vertex_count(n_faces, 0);
    std::vector<GlobalIndex> tri_face_slot(n_faces, -1);
    std::vector<GlobalIndex> quad_face_slot(n_faces, -1);
    std::vector<FaceKey> tri_face_keys;
    std::vector<int> tri_face_touch;
    std::vector<gid_t> tri_face_cell_gid_candidate;
    std::vector<int> tri_face_cell_owner_candidate;
    std::vector<LocalIndex> tri_face_local_count;
    std::vector<FaceKey> quad_face_keys;
    std::vector<int> quad_face_touch;
    std::vector<gid_t> quad_face_cell_gid_candidate;
    std::vector<int> quad_face_cell_owner_candidate;
    std::vector<LocalIndex> quad_face_local_count;

    if (has_face_entities && topo->n_faces > 0) {
        for (GlobalIndex f = 0; f < topo->n_faces; ++f) {
            const auto verts = topo->getFaceVertices(f);
            FE_CHECK_ARG(verts.size() == 3u || verts.size() == 4u,
                         "DofHandler::distributeVariableOrderDofsParallel: unsupported face vertex count");
            face_vertex_count[static_cast<std::size_t>(f)] = static_cast<std::uint8_t>(verts.size());

            FaceKey key{};
            key.n = static_cast<std::uint8_t>(verts.size());
            for (std::size_t i = 0; i < verts.size(); ++i) {
                key.gids[i] = topo->vertex_gids[static_cast<std::size_t>(verts[i])];
            }
            std::sort(key.gids.begin(), key.gids.begin() + key.n);
            for (std::size_t i = static_cast<std::size_t>(key.n); i < key.gids.size(); ++i) {
                key.gids[i] = gid_t{0};
            }

            const auto sf = static_cast<std::size_t>(f);
            if (verts.size() == 3u) {
                tri_face_slot[sf] = static_cast<GlobalIndex>(tri_face_keys.size());
                tri_face_keys.push_back(key);
                tri_face_touch.push_back(-1);
                tri_face_cell_gid_candidate.push_back(std::numeric_limits<gid_t>::max());
                tri_face_cell_owner_candidate.push_back(-1);
                tri_face_local_count.push_back(std::numeric_limits<LocalIndex>::max());
            } else {
                quad_face_slot[sf] = static_cast<GlobalIndex>(quad_face_keys.size());
                quad_face_keys.push_back(key);
                quad_face_touch.push_back(-1);
                quad_face_cell_gid_candidate.push_back(std::numeric_limits<gid_t>::max());
                quad_face_cell_owner_candidate.push_back(-1);
                quad_face_local_count.push_back(std::numeric_limits<LocalIndex>::max());
            }
        }
    }

    auto reduce_local_count = [](LocalIndex count, LocalIndex& target) {
        target = std::min(target, count);
    };

    for (GlobalIndex c = 0; c < topo->n_cells; ++c) {
        const auto sc = static_cast<std::size_t>(c);
        const auto cgid = topo->getCellGid(c);
        const int cell_owner = topo->getCellOwnerRank(c, my_rank_);
        const bool owned_cell = (cell_owner == my_rank_);
        const auto& layout = cell_layouts[sc];
        const auto cell_verts = topo->getCellVertices(c);

        for (std::size_t lv = 0; lv < layout.vertex_counts.size(); ++lv) {
            const auto gv = static_cast<std::size_t>(cell_verts[lv]);
            if (cgid < vertex_cell_gid_candidate[gv] ||
                (cgid == vertex_cell_gid_candidate[gv] && cell_owner < vertex_cell_owner_candidate[gv])) {
                vertex_cell_gid_candidate[gv] = cgid;
                vertex_cell_owner_candidate[gv] = cell_owner;
            }
            reduce_local_count(layout.vertex_counts[lv], vertex_local_count[gv]);
            if (owned_cell) {
                vertex_touch[gv] = my_rank_;
            }
        }

        if (!layout.edge_counts.empty()) {
            const auto cell_edges = topo->getCellEdges(c);
            FE_CHECK_ARG(cell_edges.size() >= layout.edge_counts.size(),
                         "DofHandler::distributeVariableOrderDofsParallel: cell edge connectivity mismatch");
            for (std::size_t le = 0; le < layout.edge_counts.size(); ++le) {
                const auto ge = static_cast<std::size_t>(cell_edges[le]);
                if (cgid < edge_cell_gid_candidate[ge] ||
                    (cgid == edge_cell_gid_candidate[ge] && cell_owner < edge_cell_owner_candidate[ge])) {
                    edge_cell_gid_candidate[ge] = cgid;
                    edge_cell_owner_candidate[ge] = cell_owner;
                }
                reduce_local_count(layout.edge_counts[le], edge_local_count[ge]);
                if (owned_cell) {
                    edge_touch[ge] = my_rank_;
                }
            }
        }

        if (!layout.face_counts.empty()) {
            const auto cell_faces = topo->getCellFaces(c);
            FE_CHECK_ARG(cell_faces.size() >= layout.face_counts.size(),
                         "DofHandler::distributeVariableOrderDofsParallel: cell face connectivity mismatch");
            for (std::size_t lf = 0; lf < layout.face_counts.size(); ++lf) {
                const auto sf = static_cast<std::size_t>(cell_faces[lf]);
                if (tri_face_slot[sf] >= 0) {
                    const auto slot = static_cast<std::size_t>(tri_face_slot[sf]);
                    if (cgid < tri_face_cell_gid_candidate[slot] ||
                        (cgid == tri_face_cell_gid_candidate[slot] &&
                         cell_owner < tri_face_cell_owner_candidate[slot])) {
                        tri_face_cell_gid_candidate[slot] = cgid;
                        tri_face_cell_owner_candidate[slot] = cell_owner;
                    }
                    reduce_local_count(layout.face_counts[lf], tri_face_local_count[slot]);
                    if (owned_cell) {
                        tri_face_touch[slot] = my_rank_;
                    }
                } else if (quad_face_slot[sf] >= 0) {
                    const auto slot = static_cast<std::size_t>(quad_face_slot[sf]);
                    if (cgid < quad_face_cell_gid_candidate[slot] ||
                        (cgid == quad_face_cell_gid_candidate[slot] &&
                         cell_owner < quad_face_cell_owner_candidate[slot])) {
                        quad_face_cell_gid_candidate[slot] = cgid;
                        quad_face_cell_owner_candidate[slot] = cell_owner;
                    }
                    reduce_local_count(layout.face_counts[lf], quad_face_local_count[slot]);
                    if (owned_cell) {
                        quad_face_touch[slot] = my_rank_;
                    }
                }
            }
        }
    }

    auto finalize_local_counts = [](std::vector<LocalIndex>& counts) {
        for (auto& c : counts) {
            if (c == std::numeric_limits<LocalIndex>::max()) {
                c = 0;
            }
        }
    };
    finalize_local_counts(vertex_local_count);
    finalize_local_counts(edge_local_count);
    finalize_local_counts(tri_face_local_count);
    finalize_local_counts(quad_face_local_count);

    for (std::size_t v = 0; v < n_vertices; ++v) {
        if (vertex_cell_gid_candidate[v] == std::numeric_limits<gid_t>::max()) {
            vertex_cell_gid_candidate[v] = gid_t{-1};
            vertex_cell_owner_candidate[v] = -1;
        }
    }
    for (std::size_t e = 0; e < edge_cell_gid_candidate.size(); ++e) {
        if (edge_cell_gid_candidate[e] == std::numeric_limits<gid_t>::max()) {
            edge_cell_gid_candidate[e] = gid_t{-1};
            edge_cell_owner_candidate[e] = -1;
        }
    }
    for (std::size_t f = 0; f < tri_face_cell_gid_candidate.size(); ++f) {
        if (tri_face_cell_gid_candidate[f] == std::numeric_limits<gid_t>::max()) {
            tri_face_cell_gid_candidate[f] = gid_t{-1};
            tri_face_cell_owner_candidate[f] = -1;
        }
    }
    for (std::size_t f = 0; f < quad_face_cell_gid_candidate.size(); ++f) {
        if (quad_face_cell_gid_candidate[f] == std::numeric_limits<gid_t>::max()) {
            quad_face_cell_gid_candidate[f] = gid_t{-1};
            quad_face_cell_owner_candidate[f] = -1;
        }
    }

    std::vector<int> neighbors = neighbor_ranks_;
    if (!topo->cell_owner_ranks.empty()) {
        for (int r : topo->cell_owner_ranks) {
            if (r >= 0 && r != my_rank_) {
                neighbors.push_back(r);
            }
        }
    }
    std::sort(neighbors.begin(), neighbors.end());
    neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
    if (neighbors.empty()) {
        neighbors.reserve(static_cast<std::size_t>(std::max(0, world_size_ - 1)));
        for (int r = 0; r < world_size_; ++r) {
            if (r != my_rank_) neighbors.push_back(r);
        }
    }

    std::vector<int> rank_to_order_storage;
    std::span<const int> rank_to_order;
    if (options.reproducible_across_communicators) {
        rank_to_order_storage = compute_stable_rank_order(mpi_comm_,
                                                         my_rank_,
                                                         world_size_,
                                                         topo->cell_gids,
                                                         topo->cell_owner_ranks,
                                                         no_global_collectives_,
                                                         /*tag_base=*/43100);
        rank_to_order = rank_to_order_storage;
    }

    std::vector<LocalIndex> vertex_shared_count;
    std::vector<LocalIndex> edge_shared_count;
    std::vector<LocalIndex> tri_face_shared_count;
    std::vector<LocalIndex> quad_face_shared_count;
    reduce_min_counts_with_neighbors<gid_t, GidHash>(mpi_comm_,
                                                     neighbors,
                                                     vertex_keys,
                                                     vertex_local_count,
                                                     vertex_shared_count,
                                                     /*tag_base=*/43110);
    if (!edge_keys.empty()) {
        reduce_min_counts_with_neighbors<EdgeKey, EdgeKeyHash>(mpi_comm_,
                                                               neighbors,
                                                               edge_keys,
                                                               edge_local_count,
                                                               edge_shared_count,
                                                               /*tag_base=*/43120);
    }
    if (!tri_face_keys.empty()) {
        reduce_min_counts_with_neighbors<FaceKey, FaceKeyHash>(mpi_comm_,
                                                               neighbors,
                                                               tri_face_keys,
                                                               tri_face_local_count,
                                                               tri_face_shared_count,
                                                               /*tag_base=*/43130);
    }
    if (!quad_face_keys.empty()) {
        reduce_min_counts_with_neighbors<FaceKey, FaceKeyHash>(mpi_comm_,
                                                               neighbors,
                                                               quad_face_keys,
                                                               quad_face_local_count,
                                                               quad_face_shared_count,
                                                               /*tag_base=*/43140);
    }

    std::vector<LocalIndex> face_shared_count(n_faces, 0);
    for (std::size_t sf = 0; sf < n_faces; ++sf) {
        if (tri_face_slot[sf] >= 0) {
            face_shared_count[sf] = tri_face_shared_count[static_cast<std::size_t>(tri_face_slot[sf])];
        } else if (quad_face_slot[sf] >= 0) {
            face_shared_count[sf] = quad_face_shared_count[static_cast<std::size_t>(quad_face_slot[sf])];
        }
    }

    auto build_ordinal_offsets = [](std::span<const LocalIndex> counts) {
        std::vector<std::size_t> offsets(counts.size() + 1u, 0);
        for (std::size_t i = 0; i < counts.size(); ++i) {
            offsets[i + 1u] = offsets[i] + static_cast<std::size_t>(std::max<LocalIndex>(0, counts[i]));
        }
        return offsets;
    };

    const auto vertex_dof_offsets = build_ordinal_offsets(vertex_shared_count);
    const auto edge_dof_offsets = build_ordinal_offsets(edge_shared_count);
    const auto tri_face_dof_offsets = build_ordinal_offsets(tri_face_shared_count);
    const auto quad_face_dof_offsets = build_ordinal_offsets(quad_face_shared_count);

    std::vector<OrdinalKey<gid_t>> vertex_dof_keys;
    std::vector<int> vertex_dof_touch;
    std::vector<gid_t> vertex_dof_candidate_gid;
    std::vector<int> vertex_dof_candidate_owner;
    vertex_dof_keys.reserve(vertex_dof_offsets.back());
    vertex_dof_touch.reserve(vertex_dof_offsets.back());
    vertex_dof_candidate_gid.reserve(vertex_dof_offsets.back());
    vertex_dof_candidate_owner.reserve(vertex_dof_offsets.back());
    for (std::size_t v = 0; v < n_vertices; ++v) {
        for (LocalIndex ord = 0; ord < vertex_shared_count[v]; ++ord) {
            vertex_dof_keys.push_back(OrdinalKey<gid_t>{vertex_keys[v], static_cast<std::int32_t>(ord)});
            vertex_dof_touch.push_back(vertex_touch[v]);
            vertex_dof_candidate_gid.push_back(vertex_cell_gid_candidate[v]);
            vertex_dof_candidate_owner.push_back(vertex_cell_owner_candidate[v]);
        }
    }

    std::vector<OrdinalKey<EdgeKey>> edge_dof_keys;
    std::vector<int> edge_dof_touch;
    std::vector<gid_t> edge_dof_candidate_gid;
    std::vector<int> edge_dof_candidate_owner;
    edge_dof_keys.reserve(edge_dof_offsets.back());
    edge_dof_touch.reserve(edge_dof_offsets.back());
    edge_dof_candidate_gid.reserve(edge_dof_offsets.back());
    edge_dof_candidate_owner.reserve(edge_dof_offsets.back());
    for (std::size_t e = 0; e < n_edges; ++e) {
        for (LocalIndex ord = 0; ord < edge_shared_count[e]; ++ord) {
            edge_dof_keys.push_back(OrdinalKey<EdgeKey>{edge_keys[e], static_cast<std::int32_t>(ord)});
            edge_dof_touch.push_back(edge_touch[e]);
            edge_dof_candidate_gid.push_back(edge_cell_gid_candidate[e]);
            edge_dof_candidate_owner.push_back(edge_cell_owner_candidate[e]);
        }
    }

    std::vector<OrdinalKey<FaceKey>> tri_face_dof_keys;
    std::vector<int> tri_face_dof_touch;
    std::vector<gid_t> tri_face_dof_candidate_gid;
    std::vector<int> tri_face_dof_candidate_owner;
    tri_face_dof_keys.reserve(tri_face_dof_offsets.back());
    tri_face_dof_touch.reserve(tri_face_dof_offsets.back());
    tri_face_dof_candidate_gid.reserve(tri_face_dof_offsets.back());
    tri_face_dof_candidate_owner.reserve(tri_face_dof_offsets.back());
    for (std::size_t f = 0; f < tri_face_keys.size(); ++f) {
        for (LocalIndex ord = 0; ord < tri_face_shared_count[f]; ++ord) {
            tri_face_dof_keys.push_back(OrdinalKey<FaceKey>{tri_face_keys[f], static_cast<std::int32_t>(ord)});
            tri_face_dof_touch.push_back(tri_face_touch[f]);
            tri_face_dof_candidate_gid.push_back(tri_face_cell_gid_candidate[f]);
            tri_face_dof_candidate_owner.push_back(tri_face_cell_owner_candidate[f]);
        }
    }

    std::vector<OrdinalKey<FaceKey>> quad_face_dof_keys;
    std::vector<int> quad_face_dof_touch;
    std::vector<gid_t> quad_face_dof_candidate_gid;
    std::vector<int> quad_face_dof_candidate_owner;
    quad_face_dof_keys.reserve(quad_face_dof_offsets.back());
    quad_face_dof_touch.reserve(quad_face_dof_offsets.back());
    quad_face_dof_candidate_gid.reserve(quad_face_dof_offsets.back());
    quad_face_dof_candidate_owner.reserve(quad_face_dof_offsets.back());
    for (std::size_t f = 0; f < quad_face_keys.size(); ++f) {
        for (LocalIndex ord = 0; ord < quad_face_shared_count[f]; ++ord) {
            quad_face_dof_keys.push_back(OrdinalKey<FaceKey>{quad_face_keys[f], static_cast<std::int32_t>(ord)});
            quad_face_dof_touch.push_back(quad_face_touch[f]);
            quad_face_dof_candidate_gid.push_back(quad_face_cell_gid_candidate[f]);
            quad_face_dof_candidate_owner.push_back(quad_face_cell_owner_candidate[f]);
        }
    }

    std::vector<LocalIndex> cell_private_count(n_cells, 0);
    for (GlobalIndex c = 0; c < topo->n_cells; ++c) {
        const auto sc = static_cast<std::size_t>(c);
        const auto& layout = cell_layouts[sc];
        const auto cell_verts = topo->getCellVertices(c);
        const auto cell_edges = layout.edge_counts.empty() ? std::span<const MeshIndex>{} : topo->getCellEdges(c);
        const auto cell_faces = layout.face_counts.empty() ? std::span<const MeshIndex>{} : topo->getCellFaces(c);

        LocalIndex private_count = layout.cell_interior_dofs;
        for (std::size_t lv = 0; lv < layout.vertex_counts.size(); ++lv) {
            private_count += std::max<LocalIndex>(
                0, layout.vertex_counts[lv] - vertex_shared_count[static_cast<std::size_t>(cell_verts[lv])]);
        }
        for (std::size_t le = 0; le < layout.edge_counts.size(); ++le) {
            private_count += std::max<LocalIndex>(
                0, layout.edge_counts[le] - edge_shared_count[static_cast<std::size_t>(cell_edges[le])]);
        }
        for (std::size_t lf = 0; lf < layout.face_counts.size(); ++lf) {
            private_count += std::max<LocalIndex>(
                0, layout.face_counts[lf] - face_shared_count[static_cast<std::size_t>(cell_faces[lf])]);
        }
        cell_private_count[sc] = private_count;
    }

    const auto cell_private_offsets = build_ordinal_offsets(cell_private_count);
    std::vector<CellPrivateKey> cell_private_keys;
    std::vector<int> cell_private_owner_input;
    cell_private_keys.reserve(cell_private_offsets.back());
    cell_private_owner_input.reserve(cell_private_offsets.back());
    for (GlobalIndex c = 0; c < topo->n_cells; ++c) {
        const auto sc = static_cast<std::size_t>(c);
        const gid_t cgid = topo->getCellGid(c);
        const int owner = topo->getCellOwnerRank(c, my_rank_);
        for (LocalIndex ord = 0; ord < cell_private_count[sc]; ++ord) {
            cell_private_keys.push_back(CellPrivateKey{cgid, static_cast<std::int32_t>(ord)});
            cell_private_owner_input.push_back(owner);
        }
    }

    gid_t n_global_vertex_dofs = 0;
    gid_t n_global_edge_dofs = 0;
    gid_t n_global_tri_face_dofs = 0;
    gid_t n_global_quad_face_dofs = 0;
    gid_t n_global_cell_private_dofs = 0;
    std::vector<gid_t> vertex_dof_global_id;
    std::vector<int> vertex_dof_owner_rank;
    std::vector<gid_t> edge_dof_global_id;
    std::vector<int> edge_dof_owner_rank;
    std::vector<gid_t> tri_face_dof_global_id;
    std::vector<int> tri_face_dof_owner_rank;
    std::vector<gid_t> quad_face_dof_global_id;
    std::vector<int> quad_face_dof_owner_rank;
    std::vector<gid_t> cell_private_global_id;

    assign_contiguous_ids_and_owners_with_neighbors<OrdinalKey<gid_t>,
                                                    OrdinalKeyHash<gid_t, GidHash>,
                                                    std::less<OrdinalKey<gid_t>>>(
        mpi_comm_,
        my_rank_,
        world_size_,
        neighbors,
        options.ownership,
        vertex_dof_keys,
        vertex_dof_touch,
        vertex_dof_candidate_gid,
        vertex_dof_candidate_owner,
        rank_to_order,
        no_global_collectives_,
        vertex_dof_global_id,
        vertex_dof_owner_rank,
        n_global_vertex_dofs,
        /*tag_base=*/43150);
    assign_contiguous_ids_and_owners_with_neighbors<OrdinalKey<EdgeKey>,
                                                    OrdinalKeyHash<EdgeKey, EdgeKeyHash>,
                                                    std::less<OrdinalKey<EdgeKey>>>(
        mpi_comm_,
        my_rank_,
        world_size_,
        neighbors,
        options.ownership,
        edge_dof_keys,
        edge_dof_touch,
        edge_dof_candidate_gid,
        edge_dof_candidate_owner,
        rank_to_order,
        no_global_collectives_,
        edge_dof_global_id,
        edge_dof_owner_rank,
        n_global_edge_dofs,
        /*tag_base=*/43160);
    assign_contiguous_ids_and_owners_with_neighbors<OrdinalKey<FaceKey>,
                                                    OrdinalKeyHash<FaceKey, FaceKeyHash>,
                                                    std::less<OrdinalKey<FaceKey>>>(
        mpi_comm_,
        my_rank_,
        world_size_,
        neighbors,
        options.ownership,
        tri_face_dof_keys,
        tri_face_dof_touch,
        tri_face_dof_candidate_gid,
        tri_face_dof_candidate_owner,
        rank_to_order,
        no_global_collectives_,
        tri_face_dof_global_id,
        tri_face_dof_owner_rank,
        n_global_tri_face_dofs,
        /*tag_base=*/43170);
    assign_contiguous_ids_and_owners_with_neighbors<OrdinalKey<FaceKey>,
                                                    OrdinalKeyHash<FaceKey, FaceKeyHash>,
                                                    std::less<OrdinalKey<FaceKey>>>(
        mpi_comm_,
        my_rank_,
        world_size_,
        neighbors,
        options.ownership,
        quad_face_dof_keys,
        quad_face_dof_touch,
        quad_face_dof_candidate_gid,
        quad_face_dof_candidate_owner,
        rank_to_order,
        no_global_collectives_,
        quad_face_dof_global_id,
        quad_face_dof_owner_rank,
        n_global_quad_face_dofs,
        /*tag_base=*/43180);
    assign_global_ordinals_with_neighbors<CellPrivateKey,
                                          CellPrivateKeyHash,
                                          std::less<CellPrivateKey>>(
        mpi_comm_,
        my_rank_,
        world_size_,
        neighbors,
        cell_private_keys,
        cell_private_owner_input,
        rank_to_order,
        no_global_collectives_,
        cell_private_global_id,
        n_global_cell_private_dofs,
        /*tag_base=*/43190);

    const gid_t vertex_offset = 0;
    const gid_t edge_offset = n_global_vertex_dofs;
    const gid_t tri_face_offset = checked_nonneg_add(edge_offset, n_global_edge_dofs, "variable-order tri-face offset");
    const gid_t quad_face_offset = checked_nonneg_add(tri_face_offset, n_global_tri_face_dofs, "variable-order quad-face offset");
    const gid_t cell_private_offset = checked_nonneg_add(quad_face_offset, n_global_quad_face_dofs, "variable-order cell-private offset");
    const gid_t scalar_total_dofs = checked_nonneg_add(cell_private_offset,
                                                       n_global_cell_private_dofs,
                                                       "variable-order scalar total dofs");

    const gid_t nc = static_cast<gid_t>(std::max<LocalIndex>(1, num_components_));
    const gid_t global_total_dofs = checked_nonneg_mul(scalar_total_dofs,
                                                       nc,
                                                       "variable-order total dofs with components");
    std::vector<gid_t> component_offsets(static_cast<std::size_t>(nc), gid_t{0});
    for (gid_t comp = 0; comp < nc; ++comp) {
        component_offsets[static_cast<std::size_t>(comp)] =
            checked_nonneg_mul(scalar_total_dofs, comp, "variable-order component offset");
    }

    entity_dof_map_ = std::make_unique<EntityDofMap>();
    entity_dof_map_->reserve(topo->n_vertices, topo->n_edges, topo->n_faces, topo->n_cells);

    auto expand_scalar_ids = [&](std::span<const GlobalIndex> scalar_ids) {
        std::vector<GlobalIndex> expanded;
        expanded.reserve(scalar_ids.size() * static_cast<std::size_t>(nc));
        for (gid_t comp = 0; comp < nc; ++comp) {
            const gid_t offset = component_offsets[static_cast<std::size_t>(comp)];
            for (auto dof : scalar_ids) {
                expanded.push_back(dof + static_cast<GlobalIndex>(offset));
            }
        }
        return expanded;
    };

    auto make_scalar_span = [](const std::vector<GlobalIndex>& values) -> std::span<const GlobalIndex> {
        return std::span<const GlobalIndex>(values.data(), values.size());
    };

    std::vector<std::vector<GlobalIndex>> vertex_scalar_dofs(n_vertices);
    for (std::size_t v = 0; v < n_vertices; ++v) {
        auto& list = vertex_scalar_dofs[v];
        list.reserve(static_cast<std::size_t>(vertex_shared_count[v]));
        for (LocalIndex ord = 0; ord < vertex_shared_count[v]; ++ord) {
            const auto idx = vertex_dof_offsets[v] + static_cast<std::size_t>(ord);
            list.push_back(static_cast<GlobalIndex>(vertex_offset + vertex_dof_global_id[idx]));
        }
        if (!list.empty()) {
            entity_dof_map_->setVertexDofs(static_cast<GlobalIndex>(v), expand_scalar_ids(make_scalar_span(list)));
        }
    }

    std::vector<std::vector<GlobalIndex>> edge_scalar_dofs(n_edges);
    for (std::size_t e = 0; e < n_edges; ++e) {
        auto& list = edge_scalar_dofs[e];
        list.reserve(static_cast<std::size_t>(edge_shared_count[e]));
        for (LocalIndex ord = 0; ord < edge_shared_count[e]; ++ord) {
            const auto idx = edge_dof_offsets[e] + static_cast<std::size_t>(ord);
            list.push_back(static_cast<GlobalIndex>(edge_offset + edge_dof_global_id[idx]));
        }
        if (!list.empty()) {
            entity_dof_map_->setEdgeDofs(static_cast<GlobalIndex>(e), expand_scalar_ids(make_scalar_span(list)));
        }
    }

    std::vector<std::vector<GlobalIndex>> face_scalar_dofs(n_faces);
    for (std::size_t sf = 0; sf < n_faces; ++sf) {
        auto& list = face_scalar_dofs[sf];
        if (tri_face_slot[sf] >= 0) {
            const auto slot = static_cast<std::size_t>(tri_face_slot[sf]);
            list.reserve(static_cast<std::size_t>(tri_face_shared_count[slot]));
            for (LocalIndex ord = 0; ord < tri_face_shared_count[slot]; ++ord) {
                const auto idx = tri_face_dof_offsets[slot] + static_cast<std::size_t>(ord);
                list.push_back(static_cast<GlobalIndex>(tri_face_offset + tri_face_dof_global_id[idx]));
            }
        } else if (quad_face_slot[sf] >= 0) {
            const auto slot = static_cast<std::size_t>(quad_face_slot[sf]);
            list.reserve(static_cast<std::size_t>(quad_face_shared_count[slot]));
            for (LocalIndex ord = 0; ord < quad_face_shared_count[slot]; ++ord) {
                const auto idx = quad_face_dof_offsets[slot] + static_cast<std::size_t>(ord);
                list.push_back(static_cast<GlobalIndex>(quad_face_offset + quad_face_dof_global_id[idx]));
            }
        }
        if (!list.empty()) {
            entity_dof_map_->setFaceDofs(static_cast<GlobalIndex>(sf), expand_scalar_ids(make_scalar_span(list)));
        }
    }

    std::vector<std::vector<GlobalIndex>> cell_private_scalar_dofs(n_cells);
    for (std::size_t c = 0; c < n_cells; ++c) {
        auto& list = cell_private_scalar_dofs[c];
        list.reserve(static_cast<std::size_t>(cell_private_count[c]));
        for (LocalIndex ord = 0; ord < cell_private_count[c]; ++ord) {
            const auto idx = cell_private_offsets[c] + static_cast<std::size_t>(ord);
            list.push_back(static_cast<GlobalIndex>(cell_private_offset + cell_private_global_id[idx]));
        }
        if (!list.empty()) {
            entity_dof_map_->setCellInteriorDofs(static_cast<GlobalIndex>(c), expand_scalar_ids(make_scalar_span(list)));
        }
    }

    std::unordered_map<GlobalIndex, int> owner_by_scalar_dof;
    owner_by_scalar_dof.reserve(static_cast<std::size_t>(std::max<gid_t>(scalar_total_dofs / 2, gid_t{32})));
    auto record_scalar_owner = [&owner_by_scalar_dof](GlobalIndex dof, int owner) {
        const auto [it, inserted] = owner_by_scalar_dof.emplace(dof, owner);
        if (!inserted && it->second != owner) {
            throw FEException("DofHandler::distributeVariableOrderDofsParallel: inconsistent owner assignment");
        }
    };

    dof_map_.reserve(topo->n_cells, static_cast<LocalIndex>(max_cell_total_dofs));
    for (GlobalIndex c = 0; c < topo->n_cells; ++c) {
        const auto sc = static_cast<std::size_t>(c);
        const auto& layout = cell_layouts[sc];
        const auto cell_verts = topo->getCellVertices(c);
        const auto cell_edges = layout.edge_counts.empty() ? std::span<const MeshIndex>{} : topo->getCellEdges(c);
        const auto cell_faces = layout.face_counts.empty() ? std::span<const MeshIndex>{} : topo->getCellFaces(c);

        std::vector<GlobalIndex> scalar_cell_dofs;
        scalar_cell_dofs.reserve(static_cast<std::size_t>(layout.scalar_dofs_per_element));
        LocalIndex private_cursor = 0;

        for (const auto& tag : layout.dof_tags) {
            GlobalIndex scalar_dof = -1;
            switch (tag.kind) {
                case VariableLocalDofKind::Vertex: {
                    const auto gv = static_cast<std::size_t>(cell_verts[static_cast<std::size_t>(tag.local_entity_id)]);
                    if (tag.ordinal < vertex_shared_count[gv]) {
                        const auto idx = vertex_dof_offsets[gv] + static_cast<std::size_t>(tag.ordinal);
                        scalar_dof = static_cast<GlobalIndex>(vertex_offset + vertex_dof_global_id[idx]);
                        record_scalar_owner(scalar_dof, vertex_dof_owner_rank[idx]);
                    }
                    break;
                }
                case VariableLocalDofKind::Edge: {
                    const auto ge = static_cast<std::size_t>(cell_edges[static_cast<std::size_t>(tag.local_entity_id)]);
                    if (tag.ordinal < edge_shared_count[ge]) {
                        const auto idx = edge_dof_offsets[ge] + static_cast<std::size_t>(tag.ordinal);
                        scalar_dof = static_cast<GlobalIndex>(edge_offset + edge_dof_global_id[idx]);
                        record_scalar_owner(scalar_dof, edge_dof_owner_rank[idx]);
                    }
                    break;
                }
                case VariableLocalDofKind::Face: {
                    const auto gf = static_cast<std::size_t>(cell_faces[static_cast<std::size_t>(tag.local_entity_id)]);
                    if (tag.ordinal < face_shared_count[gf]) {
                        if (tri_face_slot[gf] >= 0) {
                            const auto slot = static_cast<std::size_t>(tri_face_slot[gf]);
                            const auto idx = tri_face_dof_offsets[slot] + static_cast<std::size_t>(tag.ordinal);
                            scalar_dof = static_cast<GlobalIndex>(tri_face_offset + tri_face_dof_global_id[idx]);
                            record_scalar_owner(scalar_dof, tri_face_dof_owner_rank[idx]);
                        } else if (quad_face_slot[gf] >= 0) {
                            const auto slot = static_cast<std::size_t>(quad_face_slot[gf]);
                            const auto idx = quad_face_dof_offsets[slot] + static_cast<std::size_t>(tag.ordinal);
                            scalar_dof = static_cast<GlobalIndex>(quad_face_offset + quad_face_dof_global_id[idx]);
                            record_scalar_owner(scalar_dof, quad_face_dof_owner_rank[idx]);
                        }
                    }
                    break;
                }
                case VariableLocalDofKind::Cell:
                default:
                    break;
            }

            if (scalar_dof < 0) {
                FE_CHECK_ARG(private_cursor < cell_private_count[sc],
                             "DofHandler::distributeVariableOrderDofsParallel: cell-private DOF count mismatch");
                const auto idx = cell_private_offsets[sc] + static_cast<std::size_t>(private_cursor++);
                scalar_dof = static_cast<GlobalIndex>(cell_private_offset + cell_private_global_id[idx]);
                record_scalar_owner(scalar_dof, cell_private_owner_input[idx]);
            }
            scalar_cell_dofs.push_back(scalar_dof);
        }

        FE_CHECK_ARG(private_cursor == cell_private_count[sc],
                     "DofHandler::distributeVariableOrderDofsParallel: cell-private cursor mismatch");

        std::vector<GlobalIndex> cell_dofs;
        if (nc > 1) {
            cell_dofs.reserve(scalar_cell_dofs.size() * static_cast<std::size_t>(nc));
            for (gid_t comp = 0; comp < nc; ++comp) {
                const gid_t offset = component_offsets[static_cast<std::size_t>(comp)];
                for (auto dof : scalar_cell_dofs) {
                    cell_dofs.push_back(dof + static_cast<GlobalIndex>(offset));
                }
            }
        } else {
            cell_dofs = std::move(scalar_cell_dofs);
        }

        FE_CHECK_ARG(cell_dofs.size() == space.dofs_per_element(c),
                     "DofHandler::distributeVariableOrderDofsParallel: cell DOF count mismatch");
        dof_map_.setCellDofs(c, cell_dofs);
    }

    std::shared_ptr<std::unordered_map<GlobalIndex, int>> owner_map_ptr =
        std::make_shared<std::unordered_map<GlobalIndex, int>>();
    owner_map_ptr->reserve(owner_by_scalar_dof.size() * static_cast<std::size_t>(std::max<gid_t>(1, nc)));
    for (const auto& [scalar_dof, owner] : owner_by_scalar_dof) {
        for (gid_t comp = 0; comp < nc; ++comp) {
            const GlobalIndex dof = scalar_dof + static_cast<GlobalIndex>(component_offsets[static_cast<std::size_t>(comp)]);
            owner_map_ptr->emplace(dof, owner);
        }
    }

    dof_map_.setNumDofs(global_total_dofs);
    dof_map_.setDofOwnership([owner_map_ptr](GlobalIndex dof) -> int {
        const auto it = owner_map_ptr->find(dof);
        return (it != owner_map_ptr->end()) ? it->second : -1;
    });

    std::vector<GlobalIndex> owned_dofs;
    std::unordered_map<GlobalIndex, int> ghost_owner_map;
    owned_dofs.reserve(owner_map_ptr->size());
    ghost_owner_map.reserve(owner_map_ptr->size());
    for (const auto& [dof, owner] : *owner_map_ptr) {
        if (owner == my_rank_) {
            owned_dofs.push_back(dof);
        } else if (owner >= 0) {
            ghost_owner_map.emplace(dof, owner);
        }
    }
    std::sort(owned_dofs.begin(), owned_dofs.end());
    owned_dofs.erase(std::unique(owned_dofs.begin(), owned_dofs.end()), owned_dofs.end());

    std::vector<GlobalIndex> ghost_dofs;
    std::vector<int> ghost_owners;
    ghost_dofs.reserve(ghost_owner_map.size());
    ghost_owners.reserve(ghost_owner_map.size());
    for (const auto& [dof, owner] : ghost_owner_map) {
        ghost_dofs.push_back(dof);
        ghost_owners.push_back(owner);
    }
    std::vector<std::size_t> ghost_perm(ghost_dofs.size());
    std::iota(ghost_perm.begin(), ghost_perm.end(), std::size_t{0});
    std::sort(ghost_perm.begin(), ghost_perm.end(), [&](std::size_t a, std::size_t b) {
        return ghost_dofs[a] < ghost_dofs[b];
    });
    std::vector<GlobalIndex> ghost_sorted;
    std::vector<int> owners_sorted;
    ghost_sorted.reserve(ghost_dofs.size());
    owners_sorted.reserve(ghost_owners.size());
    for (auto idx : ghost_perm) {
        ghost_sorted.push_back(ghost_dofs[idx]);
        owners_sorted.push_back(ghost_owners[idx]);
    }

    dof_map_.setNumLocalDofs(static_cast<GlobalIndex>(owned_dofs.size()));
    partition_ = DofPartition(IndexSet(std::move(owned_dofs)), IndexSet(ghost_sorted));
    partition_.setGlobalSize(global_total_dofs);

    ghost_manager_ = std::make_unique<GhostDofManager>();
    ghost_manager_->setGhostDofs(ghost_sorted, owners_sorted);

    entity_dof_map_->buildReverseMapping();
    entity_dof_map_->finalize();
    cacheCellOrientations(*topo, need_edge_orientations, need_face_orientations);
#endif
}

void DofHandler::distributeCGDofs(const MeshTopologyView& topology,
                                   const DofLayoutInfo& layout,
                                   const DofDistributionOptions& options) {
    // ==========================================================================
    // CG DOF Distribution: Share DOFs across mesh entities
    // ==========================================================================

    // Phase 1: Assign DOFs to vertices (shared based on vertex GID)
    // Phase 2: Assign DOFs to edges (shared using canonical ordering)
    // Phase 3: Assign DOFs to faces (shared using canonical ordering)
    // Phase 4: Assign DOFs to cell interiors (unique per cell)
    // Phase 5: Build cell-to-DOF mapping

    const auto nc = static_cast<GlobalIndex>(std::max<LocalIndex>(1, num_components_));
    GlobalIndex next_dof = 0;
    const auto n_vertices = static_cast<std::size_t>(std::max<GlobalIndex>(topology.n_vertices, 0));
    const auto n_edges = static_cast<std::size_t>(std::max<GlobalIndex>(topology.n_edges, 0));
    const auto n_faces = static_cast<std::size_t>(std::max<GlobalIndex>(topology.n_faces, 0));
    const auto n_cells = static_cast<std::size_t>(std::max<GlobalIndex>(topology.n_cells, 0));

    if (layout.dofs_per_edge > 0) {
        if (topology.n_edges <= 0 || topology.cell2edge_offsets.empty() || topology.cell2edge_data.empty()) {
            throw FEException("DofHandler::distributeDofs: edge-interior DOFs require cell2edge connectivity (and n_edges > 0)");
        }
    }
    if (layout.has_face_dofs()) {
        if (topology.n_faces <= 0 || topology.cell2face_offsets.empty() || topology.cell2face_data.empty()) {
            throw FEException("DofHandler::distributeDofs: face-interior DOFs require cell2face connectivity (and n_faces > 0)");
        }
    }

    // Create EntityDofMap to track entity-to-DOF associations
    entity_dof_map_ = std::make_unique<EntityDofMap>();
    entity_dof_map_->reserve(topology.n_vertices, topology.n_edges,
                             topology.n_faces, topology.n_cells);

    // -------------------------------------------------------------------------
    // Phase 1: Vertex DOFs (shared across all cells touching the vertex)
    // -------------------------------------------------------------------------
    // Map: vertex_local_id -> first DOF for that vertex
    std::vector<GlobalIndex> vertex_first_dof(n_vertices, -1);

    if (layout.dofs_per_vertex > 0) {
        for (GlobalIndex v = 0; v < topology.n_vertices; ++v) {
            vertex_first_dof[static_cast<std::size_t>(v)] = next_dof;
            std::vector<GlobalIndex> v_dofs;
            for (LocalIndex d = 0; d < layout.dofs_per_vertex; ++d) {
                v_dofs.push_back(next_dof++);
            }
            entity_dof_map_->setVertexDofs(v, v_dofs);
        }
    }

    // -------------------------------------------------------------------------
    // Phase 2: Edge DOFs (shared using canonical ordering by min vertex GID)
    // -------------------------------------------------------------------------
    // Map: edge_local_id -> first DOF for that edge
    std::vector<GlobalIndex> edge_first_dof(n_edges, -1);

    if (layout.dofs_per_edge > 0 && topology.n_edges > 0) {
        for (GlobalIndex e = 0; e < topology.n_edges; ++e) {
            edge_first_dof[static_cast<std::size_t>(e)] = next_dof;
            std::vector<GlobalIndex> e_dofs;
            for (LocalIndex d = 0; d < layout.dofs_per_edge; ++d) {
                e_dofs.push_back(next_dof++);
            }
            entity_dof_map_->setEdgeDofs(e, e_dofs);
        }
    }

    // -------------------------------------------------------------------------
    // Phase 3: Face DOFs (shared using canonical ordering by min vertex GID)
    // -------------------------------------------------------------------------
    std::vector<GlobalIndex> face_first_dof(n_faces, -1);
    std::vector<LocalIndex> face_dof_count(n_faces, 0);

    if (layout.has_face_dofs() && topology.n_faces > 0) {
        for (GlobalIndex f = 0; f < topology.n_faces; ++f) {
            const auto fv = topology.getFaceVertices(f);
            const LocalIndex dofs_on_face = layout.face_dofs_for_vertex_count(fv.size());
            face_dof_count[static_cast<std::size_t>(f)] = dofs_on_face;
            if (dofs_on_face <= 0) {
                continue;
            }
            face_first_dof[static_cast<std::size_t>(f)] = next_dof;
            std::vector<GlobalIndex> f_dofs;
            for (LocalIndex d = 0; d < dofs_on_face; ++d) {
                f_dofs.push_back(next_dof++);
            }
            entity_dof_map_->setFaceDofs(f, f_dofs);
        }
    }

    // -------------------------------------------------------------------------
    // Phase 4: Cell interior DOFs (unique per cell)
    // -------------------------------------------------------------------------
    std::vector<GlobalIndex> cell_first_interior_dof(n_cells, -1);

    if (layout.dofs_per_cell > 0) {
        for (GlobalIndex c = 0; c < topology.n_cells; ++c) {
            cell_first_interior_dof[static_cast<std::size_t>(c)] = next_dof;
            std::vector<GlobalIndex> c_dofs;
            for (LocalIndex d = 0; d < layout.dofs_per_cell; ++d) {
                c_dofs.push_back(next_dof++);
            }
            entity_dof_map_->setCellInteriorDofs(c, c_dofs);
        }
    }

    const GlobalIndex scalar_total_dofs = next_dof;
    const GlobalIndex total_dofs = scalar_total_dofs * nc;

    // -------------------------------------------------------------------------
    // Phase 5: Build cell-to-DOF mapping (assemble from entity DOFs)
    // -------------------------------------------------------------------------
    dof_map_.reserve(topology.n_cells, layout.total_dofs_per_element);

    for (GlobalIndex c = 0; c < topology.n_cells; ++c) {
        std::vector<GlobalIndex> cell_dofs;
        cell_dofs.reserve(layout.total_dofs_per_element);

        // Add vertex DOFs
        auto cell_verts = topology.getCellVertices(c);
        for (auto v : cell_verts) {
            if (vertex_first_dof[static_cast<std::size_t>(v)] >= 0) {
                for (LocalIndex d = 0; d < layout.dofs_per_vertex; ++d) {
                    cell_dofs.push_back(vertex_first_dof[static_cast<std::size_t>(v)] + d);
                }
            }
        }
        const std::size_t vertex_dofs_count = cell_dofs.size();

        // Add edge DOFs (requires cell2edge connectivity when dofs_per_edge > 0).
        if (layout.dofs_per_edge > 0 && !topology.cell2edge_offsets.empty()) {
            auto cell_edges = topology.getCellEdges(c);

            // Prefer reference-element edge ordering when topology provides edge vertices.
            const bool have_edge_vertices =
                !topology.edge2vertex_data.empty() &&
                topology.edge2vertex_data.size() >=
                    static_cast<std::size_t>(2) * static_cast<std::size_t>(topology.n_edges);

            auto get_edge_vertices = [&](GlobalIndex edge_id) -> std::pair<GlobalIndex, GlobalIndex> {
                if (edge_id < 0 || edge_id >= topology.n_edges) {
                    return {-1, -1};
                }
                const auto idx = static_cast<std::size_t>(edge_id) * 2;
                if (idx + 1 >= topology.edge2vertex_data.size()) {
                    return {-1, -1};
                }
                return {topology.edge2vertex_data[idx], topology.edge2vertex_data[idx + 1]};
            };

            auto infer_base_type = [&](std::size_t n_verts) -> ElementType {
                if (topology.dim == 2 && n_verts == 3) return ElementType::Triangle3;
                if (topology.dim == 2 && n_verts == 4) return ElementType::Quad4;
                if (topology.dim == 3 && n_verts == 4) return ElementType::Tetra4;
                if (topology.dim == 3 && n_verts == 8) return ElementType::Hex8;
                if (topology.dim == 3 && n_verts == 6) return ElementType::Wedge6;
                if (topology.dim == 3 && n_verts == 5) return ElementType::Pyramid5;
                return ElementType::Unknown;
            };

            const auto base_type = infer_base_type(cell_verts.size());
            const bool can_use_reference =
                have_edge_vertices && base_type != ElementType::Unknown &&
                cell_edges.size() >= elements::ReferenceElement::create(base_type).num_edges();

            if (can_use_reference) {
                const auto ref = elements::ReferenceElement::create(base_type);

                auto find_edge_id = [&](GlobalIndex v0, GlobalIndex v1) -> GlobalIndex {
                    for (auto e : cell_edges) {
                        auto [a, b] = get_edge_vertices(e);
                        if ((a == v0 && b == v1) || (a == v1 && b == v0)) {
                            return e;
                        }
                    }
                    return -1;
                };

                const auto n_ref_edges = ref.num_edges();
                for (std::size_t le = 0; le < n_ref_edges; ++le) {
                    const auto& edge_nodes = ref.edge_nodes(le);
                    if (edge_nodes.size() != 2) continue;

                    const auto lv0 = static_cast<std::size_t>(edge_nodes[0]);
                    const auto lv1 = static_cast<std::size_t>(edge_nodes[1]);
                    if (lv0 >= cell_verts.size() || lv1 >= cell_verts.size()) continue;

                    const GlobalIndex gv0 = cell_verts[lv0];
                    const GlobalIndex gv1 = cell_verts[lv1];

                    const GlobalIndex e = find_edge_id(gv0, gv1);
                    if (e < 0) {
                        // Fall back to the provided per-cell edge ordering if mapping fails.
                        break;
                    }

                    // For scalar Lagrange edge-interior DOFs, reverse ordering when the
                    // reference edge direction opposes the canonical (min-GID→max-GID) direction.
                    bool forward = true;
                    if (layout.dofs_per_edge > 1 && options.use_canonical_ordering) {
                        if (gv0 >= 0 && gv1 >= 0) {
                            auto get_gid = [&](GlobalIndex v) -> gid_t {
                                const auto vv = static_cast<std::size_t>(v);
                                if (vv < topology.vertex_gids.size()) {
                                    return topology.vertex_gids[vv];
                                }
                                return static_cast<gid_t>(v);
                            };
                            forward = (get_gid(gv0) <= get_gid(gv1));
                        }
                    }

                    if (layout.dofs_per_edge <= 1 || forward) {
                        for (LocalIndex d = 0; d < layout.dofs_per_edge; ++d) {
                            cell_dofs.push_back(edge_first_dof[static_cast<std::size_t>(e)] + d);
                        }
                    } else {
                        for (LocalIndex d = 0; d < layout.dofs_per_edge; ++d) {
                            const auto rd = static_cast<GlobalIndex>(layout.dofs_per_edge - 1 - d);
                            cell_dofs.push_back(edge_first_dof[static_cast<std::size_t>(e)] + rd);
                        }
                    }
                }

                // If we didn't add all expected edge DOFs, fall back to legacy ordering.
                const auto expected_edge_dofs =
                    static_cast<std::size_t>(ref.num_edges()) * static_cast<std::size_t>(layout.dofs_per_edge);
                if (layout.dofs_per_edge > 0 &&
                    (cell_dofs.size() < vertex_dofs_count + expected_edge_dofs)) {
                    // Reset back to vertex-only and re-add edges using legacy behavior.
                    cell_dofs.resize(vertex_dofs_count);
                    for (auto e : cell_edges) {
                        for (LocalIndex d = 0; d < layout.dofs_per_edge; ++d) {
                            cell_dofs.push_back(edge_first_dof[static_cast<std::size_t>(e)] + d);
                        }
                    }
                }
            } else {
                // Legacy behavior: assume cell2edge_data already matches element-local ordering.
                for (auto e : cell_edges) {
                    for (LocalIndex d = 0; d < layout.dofs_per_edge; ++d) {
                        cell_dofs.push_back(edge_first_dof[static_cast<std::size_t>(e)] + d);
                    }
                }
            }
        }

        // Add face DOFs (requires cell2face connectivity when face-interior DOFs exist).
        if (layout.has_face_dofs() && !topology.cell2face_offsets.empty()) {
            auto cell_faces = topology.getCellFaces(c);

            auto infer_base_type = [&](std::size_t n_verts) -> ElementType {
                if (topology.dim == 2 && n_verts == 3) return ElementType::Triangle3;
                if (topology.dim == 2 && n_verts == 4) return ElementType::Quad4;
                if (topology.dim == 3 && n_verts == 4) return ElementType::Tetra4;
                if (topology.dim == 3 && n_verts == 8) return ElementType::Hex8;
                if (topology.dim == 3 && n_verts == 6) return ElementType::Wedge6;
                if (topology.dim == 3 && n_verts == 5) return ElementType::Pyramid5;
                return ElementType::Unknown;
            };

            const auto base_type = infer_base_type(cell_verts.size());
            const bool can_orient_faces =
                options.use_canonical_ordering &&
                layout.tensor_face_dof_layout &&
                (base_type != ElementType::Unknown) &&
                !topology.face2vertex_offsets.empty() &&
                !topology.face2vertex_data.empty();

            const int poly_order = can_orient_faces ? (static_cast<int>(layout.dofs_per_edge) + 1) : 0;

            auto get_face_vertices = [&](GlobalIndex face_id) {
                return topology.getFaceVertices(face_id);
            };

            const auto ref = (base_type != ElementType::Unknown)
                                 ? elements::ReferenceElement::create(base_type)
                                 : elements::ReferenceElement{};

            for (std::size_t lf = 0; lf < cell_faces.size(); ++lf) {
                const GlobalIndex f = cell_faces[lf];
                const auto sf = static_cast<std::size_t>(f);
                if (sf >= face_dof_count.size()) {
                    throw FEException("DofHandler::distributeDofs: face id out of range while assembling face DOFs");
                }
                const LocalIndex dofs_on_face = face_dof_count[sf];
                if (dofs_on_face <= 0) {
                    continue;
                }

                if (!can_orient_faces || dofs_on_face == 1) {
                    for (LocalIndex d = 0; d < dofs_on_face; ++d) {
                        cell_dofs.push_back(face_first_dof[sf] + d);
                    }
                    continue;
                }

                const auto face_vertices = get_face_vertices(f);
                if (face_vertices.empty()) {
                    throw FEException("DofHandler::distributeDofs: missing face2vertex connectivity for face orientation");
                }

                if (lf >= ref.num_faces()) {
                    throw FEException("DofHandler::distributeDofs: cell2face ordering does not match reference element face count");
                }

                const auto& fn = ref.face_nodes(lf);
                std::vector<int> local_to_global;
                if (face_vertices.size() == 3u) {
                    if (fn.size() != 3u) {
                        throw FEException("DofHandler::distributeDofs: expected triangle face in reference element");
                    }

                    std::array<int, 3> local{};
                    std::array<int, 3> global{};
                    for (std::size_t i = 0; i < 3u; ++i) {
                        const auto lv = static_cast<std::size_t>(fn[i]);
                        if (lv >= cell_verts.size()) {
                            throw FEException("DofHandler::distributeDofs: triangle face vertex index out of range");
                        }
                        local[i] = static_cast<int>(cell_verts[lv]);
                        global[i] = static_cast<int>(face_vertices[i]);
                    }

                    const auto orient = spaces::OrientationManager::triangle_face_orientation(local, global);
                    local_to_global = compute_scalar_face_interior_local_to_global(
                        ElementType::Triangle3, poly_order, orient.vertex_perm);
                } else if (face_vertices.size() == 4u) {
                    if (fn.size() != 4u) {
                        throw FEException("DofHandler::distributeDofs: expected quad face in reference element");
                    }

                    std::array<int, 4> local{};
                    std::array<int, 4> global{};
                    for (std::size_t i = 0; i < 4u; ++i) {
                        const auto lv = static_cast<std::size_t>(fn[i]);
                        if (lv >= cell_verts.size()) {
                            throw FEException("DofHandler::distributeDofs: quad face vertex index out of range");
                        }
                        local[i] = static_cast<int>(cell_verts[lv]);
                        global[i] = static_cast<int>(face_vertices[i]);
                    }

                    const auto orient = spaces::OrientationManager::quad_face_orientation(local, global);
                    local_to_global = compute_scalar_face_interior_local_to_global(
                        ElementType::Quad4, poly_order, orient.vertex_perm);
                } else {
                    throw FEException("DofHandler::distributeDofs: unsupported face shape for face-orientation handling");
                }

                if (static_cast<LocalIndex>(local_to_global.size()) != dofs_on_face) {
                    throw FEException("DofHandler::distributeDofs: face interior permutation size mismatch");
                }

                for (LocalIndex l = 0; l < dofs_on_face; ++l) {
                    const int g = local_to_global[static_cast<std::size_t>(l)];
                    cell_dofs.push_back(face_first_dof[sf] + static_cast<GlobalIndex>(g));
                }
            }
        }

        // Add cell interior DOFs
        if (layout.dofs_per_cell > 0) {
            for (LocalIndex d = 0; d < layout.dofs_per_cell; ++d) {
                cell_dofs.push_back(cell_first_interior_dof[static_cast<std::size_t>(c)] + d);
            }
        }

        if (nc > 1) {
            std::vector<GlobalIndex> expanded;
            expanded.reserve(cell_dofs.size() * static_cast<std::size_t>(nc));
            for (GlobalIndex comp = 0; comp < nc; ++comp) {
                const GlobalIndex offset = static_cast<GlobalIndex>(comp * scalar_total_dofs);
                for (auto dof : cell_dofs) {
                    expanded.push_back(dof + offset);
                }
            }
            cell_dofs = std::move(expanded);
        }

        dof_map_.setCellDofs(c, cell_dofs);
    }

    // Set total DOF counts
    dof_map_.setNumDofs(total_dofs);
    dof_map_.setNumLocalDofs(total_dofs);  // All local for serial

    // Build partition (all owned, no ghosts for serial case)
    partition_ = DofPartition(0, total_dofs, {});
    partition_.setGlobalSize(total_dofs);

    if (nc > 1) {
        auto expanded_entity = std::make_unique<EntityDofMap>();
        expanded_entity->reserve(topology.n_vertices, topology.n_edges, topology.n_faces, topology.n_cells);

        auto expand_entity_range = [&](GlobalIndex base, LocalIndex dofs_per_entity) -> std::vector<GlobalIndex> {
            std::vector<GlobalIndex> out;
            if (dofs_per_entity <= 0) return out;
            out.reserve(static_cast<std::size_t>(dofs_per_entity) * static_cast<std::size_t>(nc));
            for (GlobalIndex comp = 0; comp < nc; ++comp) {
                const GlobalIndex offset = static_cast<GlobalIndex>(comp * scalar_total_dofs);
                for (LocalIndex d = 0; d < dofs_per_entity; ++d) {
                    out.push_back(base + d + offset);
                }
            }
            return out;
        };

        for (GlobalIndex v = 0; v < topology.n_vertices; ++v) {
            if (layout.dofs_per_vertex <= 0 || vertex_first_dof[static_cast<std::size_t>(v)] < 0) continue;
            expanded_entity->setVertexDofs(v, expand_entity_range(vertex_first_dof[static_cast<std::size_t>(v)], layout.dofs_per_vertex));
        }
        for (GlobalIndex e = 0; e < topology.n_edges; ++e) {
            if (layout.dofs_per_edge <= 0 || edge_first_dof[static_cast<std::size_t>(e)] < 0) continue;
            expanded_entity->setEdgeDofs(e, expand_entity_range(edge_first_dof[static_cast<std::size_t>(e)], layout.dofs_per_edge));
        }
        for (GlobalIndex f = 0; f < topology.n_faces; ++f) {
            const auto sf = static_cast<std::size_t>(f);
            if (sf >= face_dof_count.size() || face_dof_count[sf] <= 0 || face_first_dof[sf] < 0) continue;
            expanded_entity->setFaceDofs(f, expand_entity_range(face_first_dof[sf], face_dof_count[sf]));
        }
        for (GlobalIndex c = 0; c < topology.n_cells; ++c) {
            if (layout.dofs_per_cell <= 0 || cell_first_interior_dof[static_cast<std::size_t>(c)] < 0) continue;
            expanded_entity->setCellInteriorDofs(c, expand_entity_range(cell_first_interior_dof[static_cast<std::size_t>(c)], layout.dofs_per_cell));
        }

        entity_dof_map_ = std::move(expanded_entity);
    }

    entity_dof_map_->buildReverseMapping();
    entity_dof_map_->finalize();
}

void DofHandler::distributeDGDofs(const MeshTopologyView& topology,
                                   const DofLayoutInfo& layout,
                                   const DofDistributionOptions& /*options*/) {
    // ==========================================================================
    // DG DOF Distribution: No sharing, all DOFs unique per cell
    // ==========================================================================

    const auto nc = static_cast<GlobalIndex>(std::max<LocalIndex>(1, num_components_));
    const auto dofs_per_cell_scalar = static_cast<GlobalIndex>(std::max<LocalIndex>(0, layout.dofs_per_cell));
    const auto scalar_total = topology.n_cells * dofs_per_cell_scalar;
    const auto total_dofs = scalar_total * nc;

    // Create EntityDofMap (only cell interior DOFs for DG)
    entity_dof_map_ = std::make_unique<EntityDofMap>();
    entity_dof_map_->reserve(0, 0, 0, topology.n_cells);

    // Reserve storage
    const auto dofs_per_cell_total = static_cast<LocalIndex>(dofs_per_cell_scalar * nc);
    dof_map_.reserve(topology.n_cells, dofs_per_cell_total);

    // Assign DOFs cell-by-cell
    for (GlobalIndex c = 0; c < topology.n_cells; ++c) {
        const auto base = static_cast<GlobalIndex>(c) * dofs_per_cell_scalar;
        std::vector<GlobalIndex> cell_dofs;
        cell_dofs.reserve(static_cast<std::size_t>(dofs_per_cell_total));

        for (GlobalIndex comp = 0; comp < nc; ++comp) {
            const auto offset = static_cast<GlobalIndex>(comp * scalar_total);
            for (GlobalIndex d = 0; d < dofs_per_cell_scalar; ++d) {
                cell_dofs.push_back(base + d + offset);
            }
        }

        dof_map_.setCellDofs(c, cell_dofs);
        entity_dof_map_->setCellInteriorDofs(c, cell_dofs);
    }

    // Set total DOF counts
    dof_map_.setNumDofs(total_dofs);
    dof_map_.setNumLocalDofs(total_dofs);

    // Build partition
    partition_ = DofPartition(0, total_dofs, {});
    partition_.setGlobalSize(total_dofs);

    // Finalize EntityDofMap
    entity_dof_map_->buildReverseMapping();
    entity_dof_map_->finalize();
}

// =============================================================================
// DOF Distribution - Distributed (MPI) Implementation
// =============================================================================

void DofHandler::distributeCGDofsParallel(const MeshTopologyView& topology,
                                         const DofLayoutInfo& layout,
                                         const DofDistributionOptions& options) {
#if !FE_HAS_MPI
    (void)topology;
    (void)layout;
    (void)options;
    throw FEException("DofHandler::distributeCGDofsParallel: FE built without MPI support");
#else
    if (world_size_ <= 1) {
        distributeCGDofs(topology, layout, options);
        return;
    }

    if (topology.vertex_gids.empty() ||
        topology.vertex_gids.size() != static_cast<std::size_t>(topology.n_vertices)) {
        throw FEException("DofHandler::distributeCGDofsParallel: vertex_gids are required for distributed CG numbering");
    }

    if (layout.dofs_per_edge > 0) {
        if (topology.n_edges <= 0 || topology.cell2edge_offsets.empty() || topology.cell2edge_data.empty()) {
            throw FEException("DofHandler::distributeCGDofsParallel: edge-interior DOFs require cell2edge connectivity (and n_edges > 0)");
        }
        if (topology.edge2vertex_data.size() <
            static_cast<std::size_t>(2) * static_cast<std::size_t>(topology.n_edges)) {
            throw FEException("DofHandler::distributeCGDofsParallel: edge-interior DOFs require edge2vertex_data for distributed CG numbering");
        }
    }
    if (layout.has_face_dofs()) {
        if (topology.n_faces <= 0 || topology.cell2face_offsets.empty() || topology.cell2face_data.empty()) {
            throw FEException("DofHandler::distributeCGDofsParallel: face-interior DOFs require cell2face connectivity (and n_faces > 0)");
        }
        if (topology.face2vertex_offsets.empty() || topology.face2vertex_data.empty()) {
            throw FEException("DofHandler::distributeCGDofsParallel: face-interior DOFs require face2vertex connectivity for distributed CG numbering");
        }
    }

    // Create EntityDofMap to track entity-to-DOF associations
    entity_dof_map_ = std::make_unique<EntityDofMap>();
    entity_dof_map_->reserve(topology.n_vertices, topology.n_edges, topology.n_faces, topology.n_cells);

    // -------------------------------------------------------------------------
    // Build per-entity keys, touch flags (owned-cell incidence), and CellOwner candidates.
    // -------------------------------------------------------------------------
    const auto n_vertices = static_cast<std::size_t>(topology.n_vertices);
    std::vector<gid_t> vertex_keys(n_vertices, gid_t{-1});
    std::vector<int> vertex_touch(n_vertices, -1);
    std::vector<gid_t> vertex_cell_gid_candidate(n_vertices, std::numeric_limits<gid_t>::max());
    std::vector<int> vertex_cell_owner_candidate(n_vertices, -1);

    for (std::size_t v = 0; v < n_vertices; ++v) {
        vertex_keys[v] = topology.vertex_gids[v];
    }

    const auto n_edges = static_cast<std::size_t>(topology.n_edges);
    std::vector<EdgeKey> edge_keys;
    std::vector<int> edge_touch;
    std::vector<gid_t> edge_cell_gid_candidate;
    std::vector<int> edge_cell_owner_candidate;

    if (layout.dofs_per_edge > 0 && topology.n_edges > 0) {
        edge_keys.resize(n_edges);
        edge_touch.assign(n_edges, -1);
        edge_cell_gid_candidate.assign(n_edges, std::numeric_limits<gid_t>::max());
        edge_cell_owner_candidate.assign(n_edges, -1);

        for (GlobalIndex e = 0; e < topology.n_edges; ++e) {
            const auto [v0, v1] = topology.getEdgeVertices(e);
            if (v0 < 0 || v1 < 0) {
                throw FEException("DofHandler::distributeCGDofsParallel: invalid edge2vertex entry");
            }
            const auto sv0 = static_cast<std::size_t>(v0);
            const auto sv1 = static_cast<std::size_t>(v1);
            if (sv0 >= topology.vertex_gids.size() || sv1 >= topology.vertex_gids.size()) {
                throw FEException("DofHandler::distributeCGDofsParallel: edge vertex index out of range for vertex_gids");
            }
            const gid_t gid0 = topology.vertex_gids[sv0];
            const gid_t gid1 = topology.vertex_gids[sv1];
            edge_keys[static_cast<std::size_t>(e)] = EdgeKey{std::min(gid0, gid1), std::max(gid0, gid1)};
        }
    }

    const auto n_faces = static_cast<std::size_t>(topology.n_faces);
    std::vector<LocalIndex> face_dof_count(n_faces, 0);
    std::vector<std::uint8_t> face_vertex_count(n_faces, 0);
    std::vector<GlobalIndex> tri_face_slot(n_faces, -1);
    std::vector<GlobalIndex> quad_face_slot(n_faces, -1);

    std::vector<FaceKey> tri_face_keys;
    std::vector<int> tri_face_touch;
    std::vector<gid_t> tri_face_cell_gid_candidate;
    std::vector<int> tri_face_cell_owner_candidate;

    std::vector<FaceKey> quad_face_keys;
    std::vector<int> quad_face_touch;
    std::vector<gid_t> quad_face_cell_gid_candidate;
    std::vector<int> quad_face_cell_owner_candidate;

    auto get_face_vertices = [&](GlobalIndex face_id) {
        return topology.getFaceVertices(face_id);
    };

    if (layout.has_face_dofs() && topology.n_faces > 0) {
        for (GlobalIndex f = 0; f < topology.n_faces; ++f) {
            const auto verts = get_face_vertices(f);
            if (verts.empty()) {
                throw FEException("DofHandler::distributeCGDofsParallel: face2vertex connectivity missing for face");
            }
            if (verts.size() != 3u && verts.size() != 4u) {
                throw FEException("DofHandler::distributeCGDofsParallel: unsupported face vertex count");
            }

            const LocalIndex dofs_on_face = layout.face_dofs_for_vertex_count(verts.size());
            face_dof_count[static_cast<std::size_t>(f)] = dofs_on_face;
            face_vertex_count[static_cast<std::size_t>(f)] = static_cast<std::uint8_t>(verts.size());
            if (dofs_on_face <= 0) {
                continue;
            }

            FaceKey key{};
            key.n = static_cast<std::uint8_t>(verts.size());
            for (std::size_t i = 0; i < verts.size(); ++i) {
                const auto v = verts[i];
                if (v < 0) {
                    throw FEException("DofHandler::distributeCGDofsParallel: negative face vertex index");
                }
                const auto sv = static_cast<std::size_t>(v);
                if (sv >= topology.vertex_gids.size()) {
                    throw FEException("DofHandler::distributeCGDofsParallel: face vertex index out of range for vertex_gids");
                }
                key.gids[i] = topology.vertex_gids[sv];
            }
            std::sort(key.gids.begin(), key.gids.begin() + key.n);
            for (std::size_t i = static_cast<std::size_t>(key.n); i < key.gids.size(); ++i) {
                key.gids[i] = gid_t{0};
            }
            const auto sf = static_cast<std::size_t>(f);
            if (verts.size() == 3u) {
                tri_face_slot[sf] = static_cast<GlobalIndex>(tri_face_keys.size());
                tri_face_keys.push_back(key);
                tri_face_touch.push_back(-1);
                tri_face_cell_gid_candidate.push_back(std::numeric_limits<gid_t>::max());
                tri_face_cell_owner_candidate.push_back(-1);
            } else {
                quad_face_slot[sf] = static_cast<GlobalIndex>(quad_face_keys.size());
                quad_face_keys.push_back(key);
                quad_face_touch.push_back(-1);
                quad_face_cell_gid_candidate.push_back(std::numeric_limits<gid_t>::max());
                quad_face_cell_owner_candidate.push_back(-1);
            }
        }
    }

    // Scan cells to populate touch flags and CellOwner candidates.
    for (GlobalIndex c = 0; c < topology.n_cells; ++c) {
        const gid_t cgid = topology.getCellGid(c);
        const int cell_owner = topology.getCellOwnerRank(c, my_rank_);
        const bool owned_cell = (cell_owner == my_rank_);

        for (auto v : topology.getCellVertices(c)) {
            const auto sv = static_cast<std::size_t>(v);
            if (sv >= n_vertices) {
                throw FEException("DofHandler::distributeCGDofsParallel: cell vertex index out of range");
            }
            if (cgid < vertex_cell_gid_candidate[sv] ||
                (cgid == vertex_cell_gid_candidate[sv] && cell_owner < vertex_cell_owner_candidate[sv])) {
                vertex_cell_gid_candidate[sv] = cgid;
                vertex_cell_owner_candidate[sv] = cell_owner;
            }
            if (owned_cell) {
                vertex_touch[sv] = my_rank_;
            }
        }

        if (!edge_keys.empty()) {
            for (auto e : topology.getCellEdges(c)) {
                const auto se = static_cast<std::size_t>(e);
                if (se >= n_edges) {
                    throw FEException("DofHandler::distributeCGDofsParallel: cell edge index out of range");
                }
                if (cgid < edge_cell_gid_candidate[se] ||
                    (cgid == edge_cell_gid_candidate[se] && cell_owner < edge_cell_owner_candidate[se])) {
                    edge_cell_gid_candidate[se] = cgid;
                    edge_cell_owner_candidate[se] = cell_owner;
                }
                if (owned_cell) {
                    edge_touch[se] = my_rank_;
                }
            }
        }

        if (!tri_face_keys.empty() || !quad_face_keys.empty()) {
            for (auto f : topology.getCellFaces(c)) {
                const auto sf = static_cast<std::size_t>(f);
                if (sf >= n_faces) {
                    throw FEException("DofHandler::distributeCGDofsParallel: cell face index out of range");
                }
                if (face_dof_count[sf] <= 0) {
                    continue;
                }

                if (tri_face_slot[sf] >= 0) {
                    const auto slot = static_cast<std::size_t>(tri_face_slot[sf]);
                    if (cgid < tri_face_cell_gid_candidate[slot] ||
                        (cgid == tri_face_cell_gid_candidate[slot] &&
                         cell_owner < tri_face_cell_owner_candidate[slot])) {
                        tri_face_cell_gid_candidate[slot] = cgid;
                        tri_face_cell_owner_candidate[slot] = cell_owner;
                    }
                    if (owned_cell) {
                        tri_face_touch[slot] = my_rank_;
                    }
                } else if (quad_face_slot[sf] >= 0) {
                    const auto slot = static_cast<std::size_t>(quad_face_slot[sf]);
                    if (cgid < quad_face_cell_gid_candidate[slot] ||
                        (cgid == quad_face_cell_gid_candidate[slot] &&
                         cell_owner < quad_face_cell_owner_candidate[slot])) {
                        quad_face_cell_gid_candidate[slot] = cgid;
                        quad_face_cell_owner_candidate[slot] = cell_owner;
                    }
                    if (owned_cell) {
                        quad_face_touch[slot] = my_rank_;
                    }
                } else {
                    throw FEException("DofHandler::distributeCGDofsParallel: missing shape-specific face slot");
                }
            }
        }
    }

    // Fill missing candidates for isolated entities (should be rare).
    for (std::size_t v = 0; v < n_vertices; ++v) {
        if (vertex_cell_gid_candidate[v] == std::numeric_limits<gid_t>::max()) {
            vertex_cell_gid_candidate[v] = gid_t{-1};
            vertex_cell_owner_candidate[v] = -1;
        }
    }
    for (std::size_t e = 0; e < edge_cell_gid_candidate.size(); ++e) {
        if (edge_cell_gid_candidate[e] == std::numeric_limits<gid_t>::max()) {
            edge_cell_gid_candidate[e] = gid_t{-1};
            edge_cell_owner_candidate[e] = -1;
        }
    }
    for (std::size_t f = 0; f < tri_face_cell_gid_candidate.size(); ++f) {
        if (tri_face_cell_gid_candidate[f] == std::numeric_limits<gid_t>::max()) {
            tri_face_cell_gid_candidate[f] = gid_t{-1};
            tri_face_cell_owner_candidate[f] = -1;
        }
    }
    for (std::size_t f = 0; f < quad_face_cell_gid_candidate.size(); ++f) {
        if (quad_face_cell_gid_candidate[f] == std::numeric_limits<gid_t>::max()) {
            quad_face_cell_gid_candidate[f] = gid_t{-1};
            quad_face_cell_owner_candidate[f] = -1;
        }
    }

	    struct GidHash {
	        std::size_t operator()(gid_t gid) const noexcept {
	            return static_cast<std::size_t>(mix_u64(static_cast<std::uint64_t>(gid)));
	        }
	    };

	    std::vector<int> neighbors = neighbor_ranks_;
	    if (!topology.cell_owner_ranks.empty()) {
	        for (int r : topology.cell_owner_ranks) {
	            if (r >= 0 && r != my_rank_) {
	                neighbors.push_back(r);
	            }
	        }
	    }
	    std::sort(neighbors.begin(), neighbors.end());
	    neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
	    if (neighbors.empty()) {
	        // Fallback for mesh-independent distributed workflows that do not provide
	        // neighbor information: communicate with all ranks (correct but not scalable).
	        neighbors.reserve(static_cast<std::size_t>(std::max(0, world_size_ - 1)));
	        for (int r = 0; r < world_size_; ++r) {
	            if (r != my_rank_) {
	                neighbors.push_back(r);
	            }
	        }
	    }

	    std::vector<int> rank_to_order_storage;
	    std::span<const int> rank_to_order;
	    if (options.reproducible_across_communicators) {
	        rank_to_order_storage = compute_stable_rank_order(mpi_comm_,
	                                                         my_rank_,
	                                                         world_size_,
	                                                         topology.cell_gids,
	                                                         topology.cell_owner_ranks,
	                                                         no_global_collectives_,
	                                                         /*tag_base=*/41800);
	        rank_to_order = rank_to_order_storage;
	    }

	    // -------------------------------------------------------------------------
	    // Assign global entity IDs and owners per entity type.
	    // -------------------------------------------------------------------------
	    std::vector<gid_t> vertex_global_id;
	    std::vector<int> vertex_owner_rank;
	    gid_t n_global_vertices = 0;

	    std::vector<gid_t> edge_global_id;
	    std::vector<int> edge_owner_rank;
	    gid_t n_global_edges = 0;

	    std::vector<gid_t> tri_face_global_id;
	    std::vector<int> tri_face_owner_rank;
	    gid_t n_global_tri_faces = 0;

	    std::vector<gid_t> quad_face_global_id;
	    std::vector<int> quad_face_owner_rank;
	    gid_t n_global_quad_faces = 0;

	    std::vector<gid_t> cell_global_id;
	    std::vector<int> cell_owner_rank_out;
	    gid_t n_global_cells = 0;

		    if (global_numbering_ == GlobalNumberingMode::OwnerContiguous) {
		        assign_contiguous_ids_and_owners_with_neighbors<gid_t, GidHash, std::less<gid_t>>(
		            mpi_comm_,
		            my_rank_,
		            world_size_,
		            neighbors,
	            options.ownership,
	            vertex_keys,
	            vertex_touch,
	            vertex_cell_gid_candidate,
	            vertex_cell_owner_candidate,
	            rank_to_order,
	            no_global_collectives_,
	            vertex_global_id,
	            vertex_owner_rank,
	            n_global_vertices,
	            /*tag_base=*/42000);

	        if (!edge_keys.empty()) {
	            assign_contiguous_ids_and_owners_with_neighbors<EdgeKey, EdgeKeyHash, std::less<EdgeKey>>(
	                mpi_comm_,
	                my_rank_,
	                world_size_,
	                neighbors,
	                options.ownership,
	                edge_keys,
	                edge_touch,
	                edge_cell_gid_candidate,
	                edge_cell_owner_candidate,
	                rank_to_order,
	                no_global_collectives_,
	                edge_global_id,
	                edge_owner_rank,
	                n_global_edges,
	                /*tag_base=*/42010);
	        }

	        if (!tri_face_keys.empty()) {
	            assign_contiguous_ids_and_owners_with_neighbors<FaceKey, FaceKeyHash, std::less<FaceKey>>(
	                mpi_comm_,
	                my_rank_,
	                world_size_,
	                neighbors,
	                options.ownership,
	                tri_face_keys,
	                tri_face_touch,
	                tri_face_cell_gid_candidate,
	                tri_face_cell_owner_candidate,
	                rank_to_order,
	                no_global_collectives_,
	                tri_face_global_id,
	                tri_face_owner_rank,
	                n_global_tri_faces,
	                /*tag_base=*/42020);
	        }

	        if (!quad_face_keys.empty()) {
	            assign_contiguous_ids_and_owners_with_neighbors<FaceKey, FaceKeyHash, std::less<FaceKey>>(
	                mpi_comm_,
	                my_rank_,
	                world_size_,
	                neighbors,
	                options.ownership,
	                quad_face_keys,
	                quad_face_touch,
	                quad_face_cell_gid_candidate,
	                quad_face_cell_owner_candidate,
	                rank_to_order,
	                no_global_collectives_,
	                quad_face_global_id,
	                quad_face_owner_rank,
	                n_global_quad_faces,
	                /*tag_base=*/42025);
	        }

	        if (layout.dofs_per_cell > 0) {
	            const auto n_cells = static_cast<std::size_t>(topology.n_cells);
	            std::vector<gid_t> cell_keys(n_cells, gid_t{-1});
	            cell_owner_rank_out.assign(n_cells, -1);

	            for (GlobalIndex c = 0; c < topology.n_cells; ++c) {
	                const auto sc = static_cast<std::size_t>(c);
	                const gid_t cgid = topology.getCellGid(c);
	                const int owner = topology.getCellOwnerRank(c, my_rank_);
	                cell_keys[sc] = cgid;
	                cell_owner_rank_out[sc] = owner;
	            }

	            assign_global_ordinals_with_neighbors<gid_t, GidHash, std::less<gid_t>>(
	                mpi_comm_,
	                my_rank_,
	                world_size_,
	                neighbors,
	                cell_keys,
	                cell_owner_rank_out,
	                rank_to_order,
	                no_global_collectives_,
		                cell_global_id,
		                n_global_cells,
		                /*tag_base=*/42030);
		        }
		    } else if (global_numbering_ == GlobalNumberingMode::GlobalIds) {
		        // Process-count independent IDs derived from global entity IDs (may be sparse).
		        vertex_global_id.assign(vertex_keys.begin(), vertex_keys.end());
		        assign_owners_with_neighbors<gid_t, GidHash>(mpi_comm_,
		                                                     my_rank_,
	                                                     world_size_,
	                                                     neighbors,
	                                                     options.ownership,
	                                                     vertex_keys,
	                                                     vertex_touch,
	                                                     vertex_cell_gid_candidate,
	                                                     vertex_cell_owner_candidate,
	                                                     rank_to_order,
	                                                     vertex_owner_rank,
	                                                     /*tag_base=*/42000);

	        if (!edge_keys.empty()) {
	            if (topology.edge_gids.size() != static_cast<std::size_t>(topology.n_edges)) {
	                throw FEException("DofHandler::distributeCGDofsParallel: global_numbering=GlobalIds requires edge_gids (size must equal n_edges)");
	            }
	            edge_global_id.assign(topology.edge_gids.begin(), topology.edge_gids.end());
	            assign_owners_with_neighbors<EdgeKey, EdgeKeyHash>(mpi_comm_,
	                                                              my_rank_,
	                                                              world_size_,
	                                                              neighbors,
	                                                              options.ownership,
	                                                              edge_keys,
	                                                              edge_touch,
	                                                              edge_cell_gid_candidate,
	                                                              edge_cell_owner_candidate,
	                                                              rank_to_order,
	                                                              edge_owner_rank,
	                                                              /*tag_base=*/42010);
	        }

	        if (!tri_face_keys.empty() || !quad_face_keys.empty()) {
	            if (topology.face_gids.size() != static_cast<std::size_t>(topology.n_faces)) {
	                throw FEException("DofHandler::distributeCGDofsParallel: global_numbering=GlobalIds requires face_gids (size must equal n_faces)");
	            }
	        }

	        if (!tri_face_keys.empty()) {
	            tri_face_global_id.assign(tri_face_keys.size(), gid_t{-1});
	            for (std::size_t sf = 0; sf < n_faces; ++sf) {
	                if (tri_face_slot[sf] >= 0) {
	                    tri_face_global_id[static_cast<std::size_t>(tri_face_slot[sf])] = topology.face_gids[sf];
	                }
	            }
	            assign_owners_with_neighbors<FaceKey, FaceKeyHash>(mpi_comm_,
	                                                              my_rank_,
	                                                              world_size_,
	                                                              neighbors,
	                                                              options.ownership,
	                                                              tri_face_keys,
	                                                              tri_face_touch,
	                                                              tri_face_cell_gid_candidate,
	                                                              tri_face_cell_owner_candidate,
	                                                              rank_to_order,
	                                                              tri_face_owner_rank,
	                                                              /*tag_base=*/42020);
	        }

	        if (!quad_face_keys.empty()) {
	            quad_face_global_id.assign(quad_face_keys.size(), gid_t{-1});
	            for (std::size_t sf = 0; sf < n_faces; ++sf) {
	                if (quad_face_slot[sf] >= 0) {
	                    quad_face_global_id[static_cast<std::size_t>(quad_face_slot[sf])] = topology.face_gids[sf];
	                }
	            }
	            assign_owners_with_neighbors<FaceKey, FaceKeyHash>(mpi_comm_,
	                                                              my_rank_,
	                                                              world_size_,
	                                                              neighbors,
	                                                              options.ownership,
	                                                              quad_face_keys,
	                                                              quad_face_touch,
	                                                              quad_face_cell_gid_candidate,
	                                                              quad_face_cell_owner_candidate,
	                                                              rank_to_order,
	                                                              quad_face_owner_rank,
	                                                              /*tag_base=*/42025);
	        }

		        if (layout.dofs_per_cell > 0) {
		            if (topology.cell_gids.size() != static_cast<std::size_t>(topology.n_cells)) {
		                throw FEException("DofHandler::distributeCGDofsParallel: global_numbering=GlobalIds requires cell_gids (size must equal n_cells)");
		            }
	            cell_global_id.assign(topology.cell_gids.begin(), topology.cell_gids.end());
	            cell_owner_rank_out.assign(static_cast<std::size_t>(topology.n_cells), -1);
	            for (GlobalIndex c = 0; c < topology.n_cells; ++c) {
	                cell_owner_rank_out[static_cast<std::size_t>(c)] = topology.getCellOwnerRank(c, my_rank_);
	            }
	        }

	        auto global_max_gid = [&](std::span<const gid_t> gids, int tag_base, const char* label) -> gid_t {
	            gid_t local_max = gid_t{-1};
	            for (const auto gid : gids) {
	                if (gid < 0) {
	                    throw FEException(std::string("DofHandler::distributeCGDofsParallel: ") + label + " contains negative global IDs");
	                }
	                local_max = std::max(local_max, gid);
	            }
	            gid_t global_max = local_max;
	            if (!no_global_collectives_) {
	                fe_mpi_check(MPI_Allreduce(&local_max, &global_max, 1, MPI_INT64_T, MPI_MAX, mpi_comm_),
	                             "MPI_Allreduce (max gid) in DofHandler::distributeCGDofsParallel");
	            } else {
	                global_max = mpi_allreduce_max_no_collectives(mpi_comm_, my_rank_, world_size_, local_max, tag_base);
	            }
	            return global_max;
	        };

	        const gid_t max_v_gid = global_max_gid(vertex_global_id, /*tag_base=*/41700, "vertex_gids");
	        n_global_vertices = (max_v_gid >= 0) ? checked_nonneg_add(max_v_gid, gid_t{1}, "vertex gid range") : gid_t{0};

	        if (layout.dofs_per_edge > 0) {
	            const gid_t max_e_gid = global_max_gid(edge_global_id, /*tag_base=*/41710, "edge_gids");
	            n_global_edges = (max_e_gid >= 0) ? checked_nonneg_add(max_e_gid, gid_t{1}, "edge gid range") : gid_t{0};
	        }

	        if (!tri_face_keys.empty()) {
	            const gid_t max_f_gid = global_max_gid(tri_face_global_id, /*tag_base=*/41720, "triangle face_gids");
	            n_global_tri_faces =
	                (max_f_gid >= 0) ? checked_nonneg_add(max_f_gid, gid_t{1}, "triangle face gid range") : gid_t{0};
	        }

	        if (!quad_face_keys.empty()) {
	            const gid_t max_f_gid = global_max_gid(quad_face_global_id, /*tag_base=*/41725, "quadrilateral face_gids");
	            n_global_quad_faces =
	                (max_f_gid >= 0) ? checked_nonneg_add(max_f_gid, gid_t{1}, "quadrilateral face gid range") : gid_t{0};
	        }

		        if (layout.dofs_per_cell > 0) {
		            const gid_t max_c_gid = global_max_gid(cell_global_id, /*tag_base=*/41730, "cell_gids");
		            n_global_cells = (max_c_gid >= 0) ? checked_nonneg_add(max_c_gid, gid_t{1}, "cell gid range") : gid_t{0};
		        }
		    } else {
		        // Process-count independent dense IDs via distributed compaction (contiguous 0..N-1).
		        assign_owners_with_neighbors<gid_t, GidHash>(mpi_comm_,
		                                                     my_rank_,
		                                                     world_size_,
		                                                     neighbors,
		                                                     options.ownership,
		                                                     vertex_keys,
		                                                     vertex_touch,
		                                                     vertex_cell_gid_candidate,
		                                                     vertex_cell_owner_candidate,
		                                                     rank_to_order,
		                                                     vertex_owner_rank,
		                                                     /*tag_base=*/42000);

		        if (!edge_keys.empty()) {
		            assign_owners_with_neighbors<EdgeKey, EdgeKeyHash>(mpi_comm_,
		                                                              my_rank_,
		                                                              world_size_,
		                                                              neighbors,
		                                                              options.ownership,
		                                                              edge_keys,
		                                                              edge_touch,
		                                                              edge_cell_gid_candidate,
		                                                              edge_cell_owner_candidate,
		                                                              rank_to_order,
		                                                              edge_owner_rank,
		                                                              /*tag_base=*/42010);
		        }

		        if (!tri_face_keys.empty()) {
		            assign_owners_with_neighbors<FaceKey, FaceKeyHash>(mpi_comm_,
		                                                              my_rank_,
		                                                              world_size_,
		                                                              neighbors,
		                                                              options.ownership,
		                                                              tri_face_keys,
		                                                              tri_face_touch,
		                                                              tri_face_cell_gid_candidate,
		                                                              tri_face_cell_owner_candidate,
		                                                              rank_to_order,
		                                                              tri_face_owner_rank,
		                                                              /*tag_base=*/42020);
		        }

		        if (!quad_face_keys.empty()) {
		            assign_owners_with_neighbors<FaceKey, FaceKeyHash>(mpi_comm_,
		                                                              my_rank_,
		                                                              world_size_,
		                                                              neighbors,
		                                                              options.ownership,
		                                                              quad_face_keys,
		                                                              quad_face_touch,
		                                                              quad_face_cell_gid_candidate,
		                                                              quad_face_cell_owner_candidate,
		                                                              rank_to_order,
		                                                              quad_face_owner_rank,
		                                                              /*tag_base=*/42025);
		        }

		        if (layout.dofs_per_cell > 0) {
		            const auto n_cells = static_cast<std::size_t>(topology.n_cells);
		            std::vector<gid_t> cell_keys(n_cells, gid_t{-1});
		            cell_owner_rank_out.assign(n_cells, -1);
		            for (GlobalIndex c = 0; c < topology.n_cells; ++c) {
		                const auto sc = static_cast<std::size_t>(c);
		                cell_keys[sc] = topology.getCellGid(c);
		                cell_owner_rank_out[sc] = topology.getCellOwnerRank(c, my_rank_);
		            }
		            cell_global_id.clear();
			            assign_dense_global_ordinals_compact_auto<gid_t, GidHash, std::less<gid_t>>(
			                mpi_comm_,
			                my_rank_,
			                world_size_,
			                rank_to_order,
			                cell_keys,
			                no_global_collectives_,
			                cell_global_id,
			                n_global_cells,
			                /*tag_base=*/42600);
		        }

		        vertex_global_id.clear();
			        assign_dense_global_ordinals_compact_auto<gid_t, GidHash, std::less<gid_t>>(
			            mpi_comm_,
			            my_rank_,
			            world_size_,
			            rank_to_order,
			            vertex_keys,
			            no_global_collectives_,
			            vertex_global_id,
			            n_global_vertices,
			            /*tag_base=*/42700);

		        if (!edge_keys.empty()) {
		            edge_global_id.clear();
			            assign_dense_global_ordinals_compact_auto<EdgeKey, EdgeKeyHash, std::less<EdgeKey>>(
			                mpi_comm_,
			                my_rank_,
			                world_size_,
			                rank_to_order,
			                edge_keys,
			                no_global_collectives_,
			                edge_global_id,
			                n_global_edges,
			                /*tag_base=*/42800);
		        }

		        if (!tri_face_keys.empty()) {
		            tri_face_global_id.clear();
			            assign_dense_global_ordinals_compact_auto<FaceKey, FaceKeyHash, std::less<FaceKey>>(
			                mpi_comm_,
			                my_rank_,
			                world_size_,
			                rank_to_order,
			                tri_face_keys,
			                no_global_collectives_,
			                tri_face_global_id,
			                n_global_tri_faces,
			                /*tag_base=*/42900);
		        }

		        if (!quad_face_keys.empty()) {
		            quad_face_global_id.clear();
			            assign_dense_global_ordinals_compact_auto<FaceKey, FaceKeyHash, std::less<FaceKey>>(
			                mpi_comm_,
			                my_rank_,
			                world_size_,
			                rank_to_order,
			                quad_face_keys,
			                no_global_collectives_,
			                quad_face_global_id,
			                n_global_quad_faces,
			                /*tag_base=*/42905);
		        }
		    }

	    std::vector<int> face_owner_rank(n_faces, -1);
	    for (std::size_t sf = 0; sf < n_faces; ++sf) {
	        if (tri_face_slot[sf] >= 0) {
	            const auto slot = static_cast<std::size_t>(tri_face_slot[sf]);
	            face_owner_rank[sf] = tri_face_owner_rank[slot];
	        } else if (quad_face_slot[sf] >= 0) {
	            const auto slot = static_cast<std::size_t>(quad_face_slot[sf]);
	            face_owner_rank[sf] = quad_face_owner_rank[slot];
	        }
	    }

	    const gid_t vertex_dofs_total =
	        checked_nonneg_mul(n_global_vertices, static_cast<gid_t>(layout.dofs_per_vertex), "vertex dof block");
	    const gid_t edge_dofs_total =
	        checked_nonneg_mul(n_global_edges, static_cast<gid_t>(layout.dofs_per_edge), "edge dof block");
	    const gid_t tri_face_dofs_total =
	        checked_nonneg_mul(n_global_tri_faces,
	                           static_cast<gid_t>(layout.face_dofs_for_vertex_count(3u)),
	                           "triangle face dof block");
	    const gid_t quad_face_dofs_total =
	        checked_nonneg_mul(n_global_quad_faces,
	                           static_cast<gid_t>(layout.face_dofs_for_vertex_count(4u)),
	                           "quadrilateral face dof block");
	    const gid_t face_dofs_total =
	        checked_nonneg_add(tri_face_dofs_total, quad_face_dofs_total, "face dof block");
	    const gid_t cell_dofs_total =
	        checked_nonneg_mul(n_global_cells, static_cast<gid_t>(layout.dofs_per_cell), "cell dof block");

	    const gid_t scalar_total_dofs = checked_nonneg_add(
	        checked_nonneg_add(vertex_dofs_total, edge_dofs_total, "vertex+edge dof blocks"),
	        checked_nonneg_add(face_dofs_total, cell_dofs_total, "face+cell dof blocks"),
	        "total dof blocks");

	    const gid_t nc = static_cast<gid_t>(std::max<LocalIndex>(1, num_components_));
	    const gid_t global_total_dofs =
	        checked_nonneg_mul(scalar_total_dofs, nc, "total dof blocks (components)");

	    std::vector<gid_t> component_offsets(static_cast<std::size_t>(nc), gid_t{0});
	    for (gid_t comp = 0; comp < nc; ++comp) {
	        component_offsets[static_cast<std::size_t>(comp)] =
	            checked_nonneg_mul(scalar_total_dofs, comp, "component offset");
	    }

	    if (options.validate_parallel) {
	        validate_entity_assignments_with_neighbors<gid_t, GidHash, std::less<gid_t>>(
	            mpi_comm_,
	            my_rank_,
	            neighbors,
	            vertex_keys,
	            vertex_touch,
	            vertex_global_id,
	            vertex_owner_rank,
	            /*tag_base=*/42500,
	            "vertex");

	        if (!edge_keys.empty()) {
	            validate_entity_assignments_with_neighbors<EdgeKey, EdgeKeyHash, std::less<EdgeKey>>(
	                mpi_comm_,
	                my_rank_,
	                neighbors,
	                edge_keys,
	                edge_touch,
	                edge_global_id,
	                edge_owner_rank,
	                /*tag_base=*/42510,
	                "edge");
	        }

	        if (!tri_face_keys.empty()) {
	            validate_entity_assignments_with_neighbors<FaceKey, FaceKeyHash, std::less<FaceKey>>(
	                mpi_comm_,
	                my_rank_,
	                neighbors,
	                tri_face_keys,
	                tri_face_touch,
	                tri_face_global_id,
	                tri_face_owner_rank,
	                /*tag_base=*/42520,
	                "triangle face");
	        }

	        if (!quad_face_keys.empty()) {
	            validate_entity_assignments_with_neighbors<FaceKey, FaceKeyHash, std::less<FaceKey>>(
	                mpi_comm_,
	                my_rank_,
	                neighbors,
	                quad_face_keys,
	                quad_face_touch,
	                quad_face_global_id,
	                quad_face_owner_rank,
	                /*tag_base=*/42525,
	                "quadrilateral face");
	        }
	    }

	    // -------------------------------------------------------------------------
	    // Assign global DOF IDs for local entities using the global ordinals.
	    // -------------------------------------------------------------------------
	    const auto n_cells = static_cast<std::size_t>(topology.n_cells);
	    std::vector<GlobalIndex> vertex_first_dof(n_vertices, -1);
	    if (layout.dofs_per_vertex > 0) {
	        for (std::size_t sv = 0; sv < n_vertices; ++sv) {
	            const auto v = static_cast<GlobalIndex>(sv);
	            const gid_t base = vertex_global_id[sv] * static_cast<gid_t>(layout.dofs_per_vertex);
	            vertex_first_dof[sv] = static_cast<GlobalIndex>(base);
	            std::vector<GlobalIndex> v_dofs;
	            v_dofs.reserve(static_cast<std::size_t>(layout.dofs_per_vertex) * static_cast<std::size_t>(nc));
	            for (gid_t comp = 0; comp < nc; ++comp) {
	                const gid_t offset = component_offsets[static_cast<std::size_t>(comp)];
	                for (LocalIndex d = 0; d < layout.dofs_per_vertex; ++d) {
	                    v_dofs.push_back(static_cast<GlobalIndex>(base + static_cast<gid_t>(d) + offset));
                }
            }
            entity_dof_map_->setVertexDofs(v, v_dofs);
	        }
	    }

	    std::vector<GlobalIndex> edge_first_dof(n_edges, -1);
	    if (layout.dofs_per_edge > 0 && n_edges > 0) {
	        for (std::size_t se = 0; se < n_edges; ++se) {
	            const auto e = static_cast<GlobalIndex>(se);
	            const gid_t base = vertex_dofs_total + edge_global_id[se] * static_cast<gid_t>(layout.dofs_per_edge);
	            edge_first_dof[se] = static_cast<GlobalIndex>(base);
	            std::vector<GlobalIndex> e_dofs;
	            e_dofs.reserve(static_cast<std::size_t>(layout.dofs_per_edge) * static_cast<std::size_t>(nc));
	            for (gid_t comp = 0; comp < nc; ++comp) {
	                const gid_t offset = component_offsets[static_cast<std::size_t>(comp)];
	                for (LocalIndex d = 0; d < layout.dofs_per_edge; ++d) {
                    e_dofs.push_back(static_cast<GlobalIndex>(base + static_cast<gid_t>(d) + offset));
                }
            }
            entity_dof_map_->setEdgeDofs(e, e_dofs);
	        }
	    }

	    const gid_t tri_face_dof_offset = vertex_dofs_total + edge_dofs_total;
	    const gid_t quad_face_dof_offset = tri_face_dof_offset + tri_face_dofs_total;

	    std::vector<GlobalIndex> face_first_dof(n_faces, -1);
	    if (layout.has_face_dofs() && n_faces > 0) {
	        for (std::size_t sf = 0; sf < n_faces; ++sf) {
	            const LocalIndex dofs_on_face = face_dof_count[sf];
	            if (dofs_on_face <= 0) {
	                continue;
	            }

	            const auto f = static_cast<GlobalIndex>(sf);
	            gid_t base = gid_t{-1};
	            if (face_vertex_count[sf] == 3u && tri_face_slot[sf] >= 0) {
	                base = tri_face_dof_offset +
	                       tri_face_global_id[static_cast<std::size_t>(tri_face_slot[sf])] *
	                           static_cast<gid_t>(dofs_on_face);
	            } else if (face_vertex_count[sf] == 4u && quad_face_slot[sf] >= 0) {
	                base = quad_face_dof_offset +
	                       quad_face_global_id[static_cast<std::size_t>(quad_face_slot[sf])] *
	                           static_cast<gid_t>(dofs_on_face);
	            } else {
	                throw FEException("DofHandler::distributeCGDofsParallel: missing face ordinal while assigning face DOFs");
	            }

	            face_first_dof[sf] = static_cast<GlobalIndex>(base);
	            std::vector<GlobalIndex> f_dofs;
	            f_dofs.reserve(static_cast<std::size_t>(dofs_on_face) * static_cast<std::size_t>(nc));
	            for (gid_t comp = 0; comp < nc; ++comp) {
	                const gid_t offset = component_offsets[static_cast<std::size_t>(comp)];
	                for (LocalIndex d = 0; d < dofs_on_face; ++d) {
	                    f_dofs.push_back(static_cast<GlobalIndex>(base + static_cast<gid_t>(d) + offset));
	                }
	            }
	            entity_dof_map_->setFaceDofs(f, f_dofs);
	        }
	    }

	    std::vector<GlobalIndex> cell_first_interior_dof(n_cells, -1);
	    if (layout.dofs_per_cell > 0) {
	        for (std::size_t sc = 0; sc < n_cells; ++sc) {
	            const auto c = static_cast<GlobalIndex>(sc);
	            const gid_t base = quad_face_dof_offset + quad_face_dofs_total +
	                               cell_global_id[sc] * static_cast<gid_t>(layout.dofs_per_cell);
	            cell_first_interior_dof[sc] = static_cast<GlobalIndex>(base);
	            std::vector<GlobalIndex> c_dofs;
	            c_dofs.reserve(static_cast<std::size_t>(layout.dofs_per_cell) * static_cast<std::size_t>(nc));
	            for (gid_t comp = 0; comp < nc; ++comp) {
	                const gid_t offset = component_offsets[static_cast<std::size_t>(comp)];
	                for (LocalIndex d = 0; d < layout.dofs_per_cell; ++d) {
                    c_dofs.push_back(static_cast<GlobalIndex>(base + static_cast<gid_t>(d) + offset));
                }
	            }
	            entity_dof_map_->setCellInteriorDofs(c, c_dofs);
	        }
	    }

    // -------------------------------------------------------------------------
    // Phase 5: Build cell-to-DOF mapping using existing orientation-aware logic.
    // -------------------------------------------------------------------------
	    dof_map_.reserve(topology.n_cells, layout.total_dofs_per_element);

	    for (std::size_t sc = 0; sc < n_cells; ++sc) {
	        const auto c = static_cast<GlobalIndex>(sc);
	        std::vector<GlobalIndex> cell_dofs;
	        cell_dofs.reserve(layout.total_dofs_per_element);

	        // Add vertex DOFs
	        auto cell_verts = topology.getCellVertices(c);
	        for (auto v : cell_verts) {
	            const auto sv = static_cast<std::size_t>(v);
	            if (sv < vertex_first_dof.size() && vertex_first_dof[sv] >= 0) {
	                for (LocalIndex d = 0; d < layout.dofs_per_vertex; ++d) {
	                    cell_dofs.push_back(vertex_first_dof[sv] + d);
	                }
	            }
	        }
        const std::size_t vertex_dofs_count = cell_dofs.size();

        // Add edge DOFs (requires cell2edge connectivity when dofs_per_edge > 0).
        if (layout.dofs_per_edge > 0 && !topology.cell2edge_offsets.empty()) {
            auto cell_edges = topology.getCellEdges(c);

            // Prefer reference-element edge ordering when topology provides edge vertices.
            const bool have_edge_vertices =
                !topology.edge2vertex_data.empty() &&
                topology.edge2vertex_data.size() >=
                    static_cast<std::size_t>(2) * static_cast<std::size_t>(topology.n_edges);

            auto get_edge_vertices = [&](GlobalIndex edge_id) -> std::pair<GlobalIndex, GlobalIndex> {
                if (edge_id < 0 || edge_id >= topology.n_edges) {
                    return {-1, -1};
                }
                const auto idx = static_cast<std::size_t>(edge_id) * 2;
                if (idx + 1 >= topology.edge2vertex_data.size()) {
                    return {-1, -1};
                }
                return {topology.edge2vertex_data[idx], topology.edge2vertex_data[idx + 1]};
            };

            auto infer_base_type = [&](std::size_t n_verts) -> ElementType {
                if (topology.dim == 2 && n_verts == 3) return ElementType::Triangle3;
                if (topology.dim == 2 && n_verts == 4) return ElementType::Quad4;
                if (topology.dim == 3 && n_verts == 4) return ElementType::Tetra4;
                if (topology.dim == 3 && n_verts == 8) return ElementType::Hex8;
                if (topology.dim == 3 && n_verts == 6) return ElementType::Wedge6;
                if (topology.dim == 3 && n_verts == 5) return ElementType::Pyramid5;
                return ElementType::Unknown;
            };

            const auto base_type = infer_base_type(cell_verts.size());
            const bool can_use_reference =
                have_edge_vertices && base_type != ElementType::Unknown &&
                cell_edges.size() >= elements::ReferenceElement::create(base_type).num_edges();

            if (can_use_reference) {
                const auto ref = elements::ReferenceElement::create(base_type);

                auto find_edge_id = [&](GlobalIndex v0, GlobalIndex v1) -> GlobalIndex {
                    for (auto e : cell_edges) {
                        auto [a, b] = get_edge_vertices(e);
                        if ((a == v0 && b == v1) || (a == v1 && b == v0)) {
                            return e;
                        }
                    }
                    return -1;
                };

                const auto n_ref_edges = ref.num_edges();
                for (std::size_t le = 0; le < n_ref_edges; ++le) {
                    const auto& edge_nodes = ref.edge_nodes(le);
                    if (edge_nodes.size() != 2) continue;

                    const auto lv0 = static_cast<std::size_t>(edge_nodes[0]);
                    const auto lv1 = static_cast<std::size_t>(edge_nodes[1]);
                    if (lv0 >= cell_verts.size() || lv1 >= cell_verts.size()) continue;

                    const GlobalIndex gv0 = cell_verts[lv0];
                    const GlobalIndex gv1 = cell_verts[lv1];

                    const GlobalIndex e = find_edge_id(gv0, gv1);
                    if (e < 0) {
                        // Fall back to the provided per-cell edge ordering if mapping fails.
                        break;
                    }

                    bool forward = true;
                    if (layout.dofs_per_edge > 1 && options.use_canonical_ordering) {
                        const auto gid0 = topology.vertex_gids[static_cast<std::size_t>(gv0)];
                        const auto gid1 = topology.vertex_gids[static_cast<std::size_t>(gv1)];
                        forward = (gid0 <= gid1);
                    }

	                    if (layout.dofs_per_edge <= 1 || forward) {
	                        for (LocalIndex d = 0; d < layout.dofs_per_edge; ++d) {
	                            const auto se = static_cast<std::size_t>(e);
	                            if (se < edge_first_dof.size()) {
	                                cell_dofs.push_back(edge_first_dof[se] + d);
	                            }
	                        }
	                    } else {
	                        for (LocalIndex d = 0; d < layout.dofs_per_edge; ++d) {
	                            const auto rd = static_cast<GlobalIndex>(layout.dofs_per_edge - 1 - d);
	                            const auto se = static_cast<std::size_t>(e);
	                            if (se < edge_first_dof.size()) {
	                                cell_dofs.push_back(edge_first_dof[se] + rd);
	                            }
	                        }
	                    }
                }

                const auto expected_edge_dofs =
                    static_cast<std::size_t>(ref.num_edges()) * static_cast<std::size_t>(layout.dofs_per_edge);
                if (layout.dofs_per_edge > 0 &&
                    (cell_dofs.size() < vertex_dofs_count + expected_edge_dofs)) {
	                    cell_dofs.resize(vertex_dofs_count);
	                    for (auto e : cell_edges) {
	                        for (LocalIndex d = 0; d < layout.dofs_per_edge; ++d) {
	                            const auto se = static_cast<std::size_t>(e);
	                            if (se < edge_first_dof.size()) {
	                                cell_dofs.push_back(edge_first_dof[se] + d);
	                            }
	                        }
	                    }
	                }
	            } else {
	                for (auto e : cell_edges) {
	                    for (LocalIndex d = 0; d < layout.dofs_per_edge; ++d) {
	                        const auto se = static_cast<std::size_t>(e);
	                        if (se < edge_first_dof.size()) {
	                            cell_dofs.push_back(edge_first_dof[se] + d);
	                        }
	                    }
	                }
	            }
	        }

        // Add face DOFs (requires cell2face connectivity when face interiors are present).
        if (layout.has_face_dofs() && !topology.cell2face_offsets.empty()) {
            auto cell_faces = topology.getCellFaces(c);

            auto infer_base_type = [&](std::size_t n_verts) -> ElementType {
                if (topology.dim == 2 && n_verts == 3) return ElementType::Triangle3;
                if (topology.dim == 2 && n_verts == 4) return ElementType::Quad4;
                if (topology.dim == 3 && n_verts == 4) return ElementType::Tetra4;
                if (topology.dim == 3 && n_verts == 8) return ElementType::Hex8;
                if (topology.dim == 3 && n_verts == 6) return ElementType::Wedge6;
                if (topology.dim == 3 && n_verts == 5) return ElementType::Pyramid5;
                return ElementType::Unknown;
            };

            const auto base_type = infer_base_type(cell_verts.size());
            const bool can_orient_faces =
                options.use_canonical_ordering &&
                layout.tensor_face_dof_layout &&
                (layout.dofs_per_face > 1 || layout.dofs_per_tri_face > 1 || layout.dofs_per_quad_face > 1) &&
                (base_type != ElementType::Unknown) &&
                !topology.face2vertex_offsets.empty() &&
                !topology.face2vertex_data.empty();

            const int poly_order = can_orient_faces ? (static_cast<int>(layout.dofs_per_edge) + 1) : 0;
            auto get_face_vertices = [&](GlobalIndex face_id) {
                return topology.getFaceVertices(face_id);
            };

	            for (std::size_t lf = 0; lf < static_cast<std::size_t>(cell_faces.size()); ++lf) {
		                const GlobalIndex f = cell_faces[lf];
		                const auto sf = static_cast<std::size_t>(f);
	                if (sf >= face_first_dof.size()) {
	                    throw FEException("DofHandler::distributeCGDofsParallel: face id out of range while assembling face DOFs");
	                }

	                const LocalIndex dofs_on_face =
	                    (sf < face_dof_count.size()) ? face_dof_count[sf] : LocalIndex{0};
	                if (dofs_on_face <= 0 || face_first_dof[sf] < 0) {
	                    continue;
	                }

	                if (!can_orient_faces || dofs_on_face <= 1) {
	                    for (LocalIndex d = 0; d < dofs_on_face; ++d) {
	                        cell_dofs.push_back(face_first_dof[sf] + d);
	                    }
	                    continue;
	                }

                const auto face_vertices = get_face_vertices(f);
                if (face_vertices.empty()) {
                    throw FEException("DofHandler::distributeCGDofsParallel: face vertex list missing while orienting face DOFs");
                }

                const auto ref = elements::ReferenceElement::create(base_type);
                if (lf >= ref.num_faces()) {
                    throw FEException("DofHandler::distributeCGDofsParallel: cell2face ordering does not match reference element face count");
                }

                const auto& fn = ref.face_nodes(lf);
                std::vector<int> local_to_global;
                if (face_vertices.size() == 3u) {
                    if (fn.size() != 3u) {
                        throw FEException("DofHandler::distributeCGDofsParallel: expected triangle face in reference element");
                    }

                    std::array<int, 3> local{};
                    std::array<int, 3> global{};
                    for (std::size_t i = 0; i < 3u; ++i) {
                        const auto lv = static_cast<std::size_t>(fn[i]);
                        if (lv >= cell_verts.size()) {
                            throw FEException("DofHandler::distributeCGDofsParallel: triangle face vertex index out of range");
                        }
                        local[i] = static_cast<int>(cell_verts[lv]);
                        global[i] = static_cast<int>(face_vertices[i]);
                    }

                    const auto orient = spaces::OrientationManager::triangle_face_orientation(local, global);
                    local_to_global = compute_scalar_face_interior_local_to_global(
                        ElementType::Triangle3, poly_order, orient.vertex_perm);
                } else if (face_vertices.size() == 4u) {
                    if (fn.size() != 4u) {
                        throw FEException("DofHandler::distributeCGDofsParallel: expected quad face in reference element");
                    }

                    std::array<int, 4> local{};
                    std::array<int, 4> global{};
                    for (std::size_t i = 0; i < 4u; ++i) {
                        const auto lv = static_cast<std::size_t>(fn[i]);
                        if (lv >= cell_verts.size()) {
                            throw FEException("DofHandler::distributeCGDofsParallel: quad face vertex index out of range");
                        }
                        local[i] = static_cast<int>(cell_verts[lv]);
                        global[i] = static_cast<int>(face_vertices[i]);
                    }

                    const auto orient = spaces::OrientationManager::quad_face_orientation(local, global);
                    local_to_global = compute_scalar_face_interior_local_to_global(
                        ElementType::Quad4, poly_order, orient.vertex_perm);
                } else {
                    throw FEException("DofHandler::distributeCGDofsParallel: unsupported face shape for face-orientation handling");
                }

                if (static_cast<LocalIndex>(local_to_global.size()) != dofs_on_face) {
                    throw FEException("DofHandler::distributeCGDofsParallel: face interior permutation size mismatch");
                }

                for (LocalIndex l = 0; l < dofs_on_face; ++l) {
                    const int g = local_to_global[static_cast<std::size_t>(l)];
                    cell_dofs.push_back(face_first_dof[sf] + static_cast<GlobalIndex>(g));
                }
	            }
	        }

	        // Add cell interior DOFs
	        if (layout.dofs_per_cell > 0) {
	            for (LocalIndex d = 0; d < layout.dofs_per_cell; ++d) {
	                cell_dofs.push_back(cell_first_interior_dof[sc] + d);
	            }
	        }

        if (nc > 1) {
            std::vector<GlobalIndex> expanded;
            expanded.reserve(cell_dofs.size() * static_cast<std::size_t>(nc));
            for (gid_t comp = 0; comp < nc; ++comp) {
                const GlobalIndex offset = static_cast<GlobalIndex>(
                    component_offsets[static_cast<std::size_t>(comp)]);
                for (auto dof : cell_dofs) {
                    expanded.push_back(dof + offset);
                }
            }
            cell_dofs = std::move(expanded);
        }

        dof_map_.setCellDofs(c, cell_dofs);
    }

    // -------------------------------------------------------------------------
    // Ownership + partition sets (owned, ghost, relevant) + ghost manager wiring.
    // -------------------------------------------------------------------------
    dof_map_.setNumDofs(global_total_dofs);

    // Local ordinal -> owner rank maps (used by DofMap::getDofOwner()).
    auto vertex_owner_by_ordinal = std::make_shared<std::unordered_map<gid_t, int>>();
    vertex_owner_by_ordinal->reserve(vertex_global_id.size());
    for (std::size_t i = 0; i < vertex_global_id.size(); ++i) {
        vertex_owner_by_ordinal->emplace(vertex_global_id[i], vertex_owner_rank[i]);
    }

    auto edge_owner_by_ordinal = std::make_shared<std::unordered_map<gid_t, int>>();
    edge_owner_by_ordinal->reserve(edge_global_id.size());
    for (std::size_t i = 0; i < edge_global_id.size(); ++i) {
        edge_owner_by_ordinal->emplace(edge_global_id[i], edge_owner_rank[i]);
    }

    auto tri_face_owner_by_ordinal = std::make_shared<std::unordered_map<gid_t, int>>();
    tri_face_owner_by_ordinal->reserve(tri_face_global_id.size());
    for (std::size_t i = 0; i < tri_face_global_id.size(); ++i) {
        tri_face_owner_by_ordinal->emplace(tri_face_global_id[i], tri_face_owner_rank[i]);
    }

    auto quad_face_owner_by_ordinal = std::make_shared<std::unordered_map<gid_t, int>>();
    quad_face_owner_by_ordinal->reserve(quad_face_global_id.size());
    for (std::size_t i = 0; i < quad_face_global_id.size(); ++i) {
        quad_face_owner_by_ordinal->emplace(quad_face_global_id[i], quad_face_owner_rank[i]);
    }

    auto cell_owner_by_ordinal = std::make_shared<std::unordered_map<gid_t, int>>();
    cell_owner_by_ordinal->reserve(cell_global_id.size());
    for (std::size_t i = 0; i < cell_global_id.size(); ++i) {
        cell_owner_by_ordinal->emplace(cell_global_id[i], cell_owner_rank_out[i]);
    }

    const auto dofs_per_vertex = static_cast<gid_t>(layout.dofs_per_vertex);
    const auto dofs_per_edge = static_cast<gid_t>(layout.dofs_per_edge);
    const auto tri_dofs_per_face = static_cast<gid_t>(layout.face_dofs_for_vertex_count(3u));
    const auto quad_dofs_per_face = static_cast<gid_t>(layout.face_dofs_for_vertex_count(4u));
    const auto dofs_per_cell = static_cast<gid_t>(layout.dofs_per_cell);

    dof_map_.setDofOwnership([=](GlobalIndex global_dof) -> int {
        if (global_dof < 0 || global_dof >= global_total_dofs) {
            return -1;
        }

        gid_t offset = 0;
        gid_t dof = static_cast<gid_t>(global_dof);
        if (nc > 1) {
            if (scalar_total_dofs <= 0) {
                return -1;
            }
            dof = dof % scalar_total_dofs;
        }

        if (dofs_per_vertex > 0 && dof < vertex_dofs_total) {
            const gid_t ord = dof / dofs_per_vertex;
            const auto it = vertex_owner_by_ordinal->find(ord);
            return (it != vertex_owner_by_ordinal->end()) ? it->second : -1;
        }
        offset += vertex_dofs_total;

        if (dofs_per_edge > 0 && dof < offset + edge_dofs_total) {
            const gid_t ord = (dof - offset) / dofs_per_edge;
            const auto it = edge_owner_by_ordinal->find(ord);
            return (it != edge_owner_by_ordinal->end()) ? it->second : -1;
        }
        offset += edge_dofs_total;

        if (tri_dofs_per_face > 0 && dof < offset + tri_face_dofs_total) {
            const gid_t ord = (dof - offset) / tri_dofs_per_face;
            const auto it = tri_face_owner_by_ordinal->find(ord);
            return (it != tri_face_owner_by_ordinal->end()) ? it->second : -1;
        }
        offset += tri_face_dofs_total;

        if (quad_dofs_per_face > 0 && dof < offset + quad_face_dofs_total) {
            const gid_t ord = (dof - offset) / quad_dofs_per_face;
            const auto it = quad_face_owner_by_ordinal->find(ord);
            return (it != quad_face_owner_by_ordinal->end()) ? it->second : -1;
        }
        offset += quad_face_dofs_total;

        if (dofs_per_cell > 0 && dof < offset + cell_dofs_total) {
            const gid_t ord = (dof - offset) / dofs_per_cell;
            const auto it = cell_owner_by_ordinal->find(ord);
            return (it != cell_owner_by_ordinal->end()) ? it->second : -1;
        }

        return -1;
    });

    const std::size_t approx_local_dofs_scalar =
        static_cast<std::size_t>(std::max<GlobalIndex>(0, topology.n_vertices)) * static_cast<std::size_t>(layout.dofs_per_vertex) +
        static_cast<std::size_t>(std::max<GlobalIndex>(0, topology.n_edges)) * static_cast<std::size_t>(layout.dofs_per_edge) +
        std::accumulate(face_dof_count.begin(),
                        face_dof_count.end(),
                        std::size_t{0},
                        [](std::size_t sum, LocalIndex count) {
                            return sum + static_cast<std::size_t>(std::max<LocalIndex>(0, count));
                        }) +
        static_cast<std::size_t>(std::max<GlobalIndex>(0, topology.n_cells)) * static_cast<std::size_t>(layout.dofs_per_cell);
    const std::size_t approx_local_dofs =
        approx_local_dofs_scalar * static_cast<std::size_t>(std::max<gid_t>(1, nc));

    std::vector<GlobalIndex> owned_dofs;
    owned_dofs.reserve(approx_local_dofs);

    auto add_entity_dofs = [&](const std::vector<GlobalIndex>& first_dof,
                               std::span<const int> owner_rank,
                               LocalIndex dofs_per_entity) {
        if (dofs_per_entity <= 0) return;
        for (std::size_t i = 0; i < first_dof.size(); ++i) {
            if (owner_rank[i] != my_rank_) continue;
            const auto base = first_dof[i];
            for (gid_t comp = 0; comp < nc; ++comp) {
                const GlobalIndex offset = static_cast<GlobalIndex>(
                    component_offsets[static_cast<std::size_t>(comp)]);
                for (LocalIndex d = 0; d < dofs_per_entity; ++d) {
                    owned_dofs.push_back(base + static_cast<GlobalIndex>(d) + offset);
                }
            }
        }
    };

    auto add_entity_dofs_variable = [&](const std::vector<GlobalIndex>& first_dof,
                                        std::span<const int> owner_rank,
                                        std::span<const LocalIndex> dof_count) {
        for (std::size_t i = 0; i < first_dof.size(); ++i) {
            const LocalIndex dofs_per_entity = dof_count[i];
            if (dofs_per_entity <= 0 || owner_rank[i] != my_rank_ || first_dof[i] < 0) continue;
            const auto base = first_dof[i];
            for (gid_t comp = 0; comp < nc; ++comp) {
                const GlobalIndex offset = static_cast<GlobalIndex>(
                    component_offsets[static_cast<std::size_t>(comp)]);
                for (LocalIndex d = 0; d < dofs_per_entity; ++d) {
                    owned_dofs.push_back(base + static_cast<GlobalIndex>(d) + offset);
                }
            }
        }
    };

    add_entity_dofs(vertex_first_dof, vertex_owner_rank, layout.dofs_per_vertex);
    add_entity_dofs(edge_first_dof, edge_owner_rank, layout.dofs_per_edge);
    add_entity_dofs_variable(face_first_dof, face_owner_rank, face_dof_count);

    if (layout.dofs_per_cell > 0) {
        std::vector<int> cell_owner_local(static_cast<std::size_t>(topology.n_cells), -1);
        for (GlobalIndex c = 0; c < topology.n_cells; ++c) {
            cell_owner_local[static_cast<std::size_t>(c)] = topology.getCellOwnerRank(c, my_rank_);
        }
        add_entity_dofs(cell_first_interior_dof, cell_owner_local, layout.dofs_per_cell);
    }

    std::sort(owned_dofs.begin(), owned_dofs.end());
    owned_dofs.erase(std::unique(owned_dofs.begin(), owned_dofs.end()), owned_dofs.end());

    std::unordered_map<GlobalIndex, int> ghost_owner_map;
    ghost_owner_map.reserve(approx_local_dofs);

    auto add_entity_ghosts = [&](const std::vector<GlobalIndex>& first_dof,
                                 std::span<const int> owner_rank,
                                 LocalIndex dofs_per_entity) {
        if (dofs_per_entity <= 0) return;
        for (std::size_t i = 0; i < first_dof.size(); ++i) {
            const int owner = owner_rank[i];
            if (owner == my_rank_) continue;
            const auto base = first_dof[i];
            for (gid_t comp = 0; comp < nc; ++comp) {
                const GlobalIndex offset = static_cast<GlobalIndex>(
                    component_offsets[static_cast<std::size_t>(comp)]);
                for (LocalIndex d = 0; d < dofs_per_entity; ++d) {
                    const GlobalIndex dof = base + static_cast<GlobalIndex>(d) + offset;
                    const auto [it, inserted] = ghost_owner_map.emplace(dof, owner);
                    if (!inserted && it->second != owner) {
                        throw FEException("DofHandler::distributeCGDofsParallel: inconsistent ghost owner assignment");
                    }
                }
            }
        }
    };

    auto add_entity_ghosts_variable = [&](const std::vector<GlobalIndex>& first_dof,
                                          std::span<const int> owner_rank,
                                          std::span<const LocalIndex> dof_count) {
        for (std::size_t i = 0; i < first_dof.size(); ++i) {
            const LocalIndex dofs_per_entity = dof_count[i];
            const int owner = owner_rank[i];
            if (dofs_per_entity <= 0 || owner == my_rank_ || first_dof[i] < 0) continue;
            const auto base = first_dof[i];
            for (gid_t comp = 0; comp < nc; ++comp) {
                const GlobalIndex offset = static_cast<GlobalIndex>(
                    component_offsets[static_cast<std::size_t>(comp)]);
                for (LocalIndex d = 0; d < dofs_per_entity; ++d) {
                    const GlobalIndex dof = base + static_cast<GlobalIndex>(d) + offset;
                    const auto [it, inserted] = ghost_owner_map.emplace(dof, owner);
                    if (!inserted && it->second != owner) {
                        throw FEException("DofHandler::distributeCGDofsParallel: inconsistent ghost owner assignment");
                    }
                }
            }
        }
    };

    add_entity_ghosts(vertex_first_dof, vertex_owner_rank, layout.dofs_per_vertex);
    add_entity_ghosts(edge_first_dof, edge_owner_rank, layout.dofs_per_edge);
    add_entity_ghosts_variable(face_first_dof, face_owner_rank, face_dof_count);

    if (layout.dofs_per_cell > 0) {
        std::vector<int> cell_owner_local(static_cast<std::size_t>(topology.n_cells), -1);
        for (GlobalIndex c = 0; c < topology.n_cells; ++c) {
            cell_owner_local[static_cast<std::size_t>(c)] = topology.getCellOwnerRank(c, my_rank_);
        }
        add_entity_ghosts(cell_first_interior_dof, cell_owner_local, layout.dofs_per_cell);
    }

    std::vector<GlobalIndex> ghost_dofs;
    std::vector<int> ghost_owners;
    ghost_dofs.reserve(ghost_owner_map.size());
    ghost_owners.reserve(ghost_owner_map.size());
    for (const auto& [dof, owner] : ghost_owner_map) {
        ghost_dofs.push_back(dof);
        ghost_owners.push_back(owner);
    }

    // Sort ghost list and keep owners aligned.
    std::vector<std::size_t> perm(ghost_dofs.size());
    std::iota(perm.begin(), perm.end(), std::size_t{0});
    std::sort(perm.begin(), perm.end(), [&](std::size_t a, std::size_t b) {
        return ghost_dofs[a] < ghost_dofs[b];
    });
    std::vector<GlobalIndex> ghost_sorted;
    std::vector<int> owners_sorted;
    ghost_sorted.reserve(ghost_dofs.size());
    owners_sorted.reserve(ghost_dofs.size());
    for (auto idx : perm) {
        ghost_sorted.push_back(ghost_dofs[idx]);
        owners_sorted.push_back(ghost_owners[idx]);
    }

    dof_map_.setNumLocalDofs(static_cast<GlobalIndex>(owned_dofs.size()));
    partition_ = DofPartition(IndexSet(std::move(owned_dofs)), IndexSet(ghost_sorted));
    partition_.setGlobalSize(global_total_dofs);

    ghost_manager_ = std::make_unique<GhostDofManager>();
    ghost_manager_->setGhostDofs(ghost_sorted, owners_sorted);

    // Finalize EntityDofMap
    entity_dof_map_->buildReverseMapping();
    entity_dof_map_->finalize();
#endif
}

void DofHandler::distributeDGDofsParallel(const MeshTopologyView& topology,
                                         const DofLayoutInfo& layout,
                                         const DofDistributionOptions& options) {
#if !FE_HAS_MPI
    (void)topology;
    (void)layout;
    (void)options;
    throw FEException("DofHandler::distributeDGDofsParallel: FE built without MPI support");
#else
    if (world_size_ <= 1) {
        distributeDGDofs(topology, layout, options);
        return;
    }

    if (topology.cell_gids.empty() && topology.n_cells > 0) {
        // Still allow, but numbering will fall back to local cell indices which is not globally unique.
        throw FEException("DofHandler::distributeDGDofsParallel: cell_gids are required for distributed DG numbering");
    }

	    const auto n_cells = static_cast<std::size_t>(topology.n_cells);
	    std::vector<gid_t> cell_keys(n_cells, gid_t{-1});
	    std::vector<int> cell_owner_rank(n_cells, -1);

	    for (GlobalIndex c = 0; c < topology.n_cells; ++c) {
	        const auto sc = static_cast<std::size_t>(c);
	        const gid_t cgid = topology.getCellGid(c);
	        const int owner = topology.getCellOwnerRank(c, my_rank_);
	        cell_keys[sc] = cgid;
	        cell_owner_rank[sc] = owner;
	    }

    struct GidHash {
        std::size_t operator()(gid_t gid) const noexcept {
            return static_cast<std::size_t>(mix_u64(static_cast<std::uint64_t>(gid)));
        }
    };

	    std::vector<gid_t> cell_global_id;
	    gid_t n_global_cells = 0;

	    std::vector<int> neighbors = neighbor_ranks_;
	    if (!topology.cell_owner_ranks.empty()) {
	        for (int r : topology.cell_owner_ranks) {
	            if (r >= 0 && r != my_rank_) {
	                neighbors.push_back(r);
	            }
	        }
	    }
	    std::sort(neighbors.begin(), neighbors.end());
	    neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
	    if (neighbors.empty()) {
	        neighbors.reserve(static_cast<std::size_t>(std::max(0, world_size_ - 1)));
	        for (int r = 0; r < world_size_; ++r) {
	            if (r != my_rank_) {
	                neighbors.push_back(r);
	            }
	        }
	    }

	    std::vector<int> rank_to_order_storage;
	    std::span<const int> rank_to_order;
	    if (options.reproducible_across_communicators) {
	        rank_to_order_storage = compute_stable_rank_order(mpi_comm_,
	                                                         my_rank_,
	                                                         world_size_,
	                                                         topology.cell_gids,
	                                                         topology.cell_owner_ranks,
	                                                         no_global_collectives_,
	                                                         /*tag_base=*/41810);
	        rank_to_order = rank_to_order_storage;
	    }

		    if (global_numbering_ == GlobalNumberingMode::OwnerContiguous) {
		        assign_global_ordinals_with_neighbors<gid_t, GidHash, std::less<gid_t>>(
		            mpi_comm_,
		            my_rank_,
		            world_size_,
	            neighbors,
	            cell_keys,
	            cell_owner_rank,
	            rank_to_order,
	            no_global_collectives_,
		            cell_global_id,
		            n_global_cells,
		            /*tag_base=*/42100);
		    } else if (global_numbering_ == GlobalNumberingMode::GlobalIds) {
		        // Process-count independent IDs derived from cell_gids (may be sparse).
		        cell_global_id.assign(cell_keys.begin(), cell_keys.end());

	        gid_t local_max = gid_t{-1};
	        for (const auto gid : cell_global_id) {
	            if (gid < 0) {
	                throw FEException("DofHandler::distributeDGDofsParallel: global_numbering=GlobalIds requires non-negative cell_gids");
	            }
	            local_max = std::max(local_max, gid);
	        }

	        gid_t global_max = local_max;
	        if (!no_global_collectives_) {
	            fe_mpi_check(MPI_Allreduce(&local_max, &global_max, 1, MPI_INT64_T, MPI_MAX, mpi_comm_),
	                         "MPI_Allreduce (max cell_gid) in DofHandler::distributeDGDofsParallel");
	        } else {
	            global_max = mpi_allreduce_max_no_collectives(mpi_comm_, my_rank_, world_size_, local_max, /*tag_base=*/41740);
		        }
		        n_global_cells = (global_max >= 0) ? checked_nonneg_add(global_max, gid_t{1}, "cell gid range") : gid_t{0};
		    } else {
		        // Process-count independent dense IDs via distributed compaction (contiguous 0..N-1).
			        assign_dense_global_ordinals_compact_auto<gid_t, GidHash, std::less<gid_t>>(
			            mpi_comm_,
			            my_rank_,
			            world_size_,
			            rank_to_order,
			            cell_keys,
			            no_global_collectives_,
			            cell_global_id,
			            n_global_cells,
			            /*tag_base=*/43100);
		    }

    const gid_t scalar_total_dofs =
        checked_nonneg_mul(n_global_cells, static_cast<gid_t>(layout.dofs_per_cell), "DG cell dof block");
    const gid_t nc = static_cast<gid_t>(std::max<LocalIndex>(1, num_components_));
    const gid_t global_total_dofs =
        checked_nonneg_mul(scalar_total_dofs, nc, "DG cell dof block (components)");

    std::vector<gid_t> component_offsets(static_cast<std::size_t>(nc), gid_t{0});
    for (gid_t comp = 0; comp < nc; ++comp) {
        component_offsets[static_cast<std::size_t>(comp)] =
            checked_nonneg_mul(scalar_total_dofs, comp, "component offset");
    }

    entity_dof_map_ = std::make_unique<EntityDofMap>();
    entity_dof_map_->reserve(0, 0, 0, topology.n_cells);

    dof_map_.reserve(topology.n_cells, layout.total_dofs_per_element);

		    std::vector<GlobalIndex> cell_first_dof(n_cells, -1);
		    for (std::size_t sc = 0; sc < n_cells; ++sc) {
	        const auto c = static_cast<GlobalIndex>(sc);
	        const gid_t base = cell_global_id[sc] * static_cast<gid_t>(layout.dofs_per_cell);
	        cell_first_dof[sc] = static_cast<GlobalIndex>(base);

        std::vector<GlobalIndex> cell_dofs;
        cell_dofs.reserve(static_cast<std::size_t>(layout.dofs_per_cell) * static_cast<std::size_t>(nc));
        for (gid_t comp = 0; comp < nc; ++comp) {
            const gid_t offset = component_offsets[static_cast<std::size_t>(comp)];
            for (LocalIndex d = 0; d < layout.dofs_per_cell; ++d) {
                cell_dofs.push_back(static_cast<GlobalIndex>(base + static_cast<gid_t>(d) + offset));
            }
        }
	        dof_map_.setCellDofs(c, cell_dofs);
	        entity_dof_map_->setCellInteriorDofs(c, cell_dofs);
	    }

    dof_map_.setNumDofs(global_total_dofs);

    auto cell_owner_by_ordinal = std::make_shared<std::unordered_map<gid_t, int>>();
    cell_owner_by_ordinal->reserve(cell_global_id.size());
    for (std::size_t i = 0; i < cell_global_id.size(); ++i) {
        cell_owner_by_ordinal->emplace(cell_global_id[i], cell_owner_rank[i]);
    }

    const auto dofs_per_cell = static_cast<gid_t>(layout.dofs_per_cell);
    dof_map_.setDofOwnership([=](GlobalIndex global_dof) -> int {
        if (global_dof < 0 || global_dof >= global_total_dofs) {
            return -1;
        }
        if (dofs_per_cell <= 0) {
            return -1;
        }
        gid_t dof = static_cast<gid_t>(global_dof);
        if (nc > 1) {
            if (scalar_total_dofs <= 0) {
                return -1;
            }
            dof = dof % scalar_total_dofs;
        }
        const gid_t ord = dof / dofs_per_cell;
        const auto it = cell_owner_by_ordinal->find(ord);
        return (it != cell_owner_by_ordinal->end()) ? it->second : -1;
    });

    const std::size_t approx_local_dofs =
        static_cast<std::size_t>(std::max<GlobalIndex>(0, topology.n_cells)) *
        static_cast<std::size_t>(layout.dofs_per_cell) *
        static_cast<std::size_t>(std::max<gid_t>(1, nc));

    std::vector<GlobalIndex> owned_dofs;
    owned_dofs.reserve(approx_local_dofs);

    std::unordered_map<GlobalIndex, int> ghost_owner_map;
    ghost_owner_map.reserve(approx_local_dofs);

		    for (std::size_t sc = 0; sc < n_cells; ++sc) {
		        const int owner = cell_owner_rank[sc];
		        const auto base = cell_first_dof[sc];
		        for (gid_t comp = 0; comp < nc; ++comp) {
	            const GlobalIndex offset = static_cast<GlobalIndex>(
	                component_offsets[static_cast<std::size_t>(comp)]);
            for (LocalIndex d = 0; d < layout.dofs_per_cell; ++d) {
                const GlobalIndex dof = base + static_cast<GlobalIndex>(d) + offset;
                if (owner == my_rank_) {
                    owned_dofs.push_back(dof);
                } else {
                    ghost_owner_map.emplace(dof, owner);
                }
            }
        }
    }

    std::sort(owned_dofs.begin(), owned_dofs.end());
    owned_dofs.erase(std::unique(owned_dofs.begin(), owned_dofs.end()), owned_dofs.end());

    std::vector<GlobalIndex> ghost_dofs;
    std::vector<int> ghost_owners;
    ghost_dofs.reserve(ghost_owner_map.size());
    ghost_owners.reserve(ghost_owner_map.size());
    for (const auto& [dof, owner] : ghost_owner_map) {
        ghost_dofs.push_back(dof);
        ghost_owners.push_back(owner);
    }

    std::vector<std::size_t> perm(ghost_dofs.size());
    std::iota(perm.begin(), perm.end(), std::size_t{0});
    std::sort(perm.begin(), perm.end(), [&](std::size_t a, std::size_t b) {
        return ghost_dofs[a] < ghost_dofs[b];
    });
    std::vector<GlobalIndex> ghost_sorted;
    std::vector<int> owners_sorted;
    ghost_sorted.reserve(ghost_dofs.size());
    owners_sorted.reserve(ghost_dofs.size());
    for (auto idx : perm) {
        ghost_sorted.push_back(ghost_dofs[idx]);
        owners_sorted.push_back(ghost_owners[idx]);
    }

    dof_map_.setNumLocalDofs(static_cast<GlobalIndex>(owned_dofs.size()));
    partition_ = DofPartition(IndexSet(std::move(owned_dofs)), IndexSet(ghost_sorted));
    partition_.setGlobalSize(global_total_dofs);

    ghost_manager_ = std::make_unique<GhostDofManager>();
    ghost_manager_->setGhostDofs(ghost_sorted, owners_sorted);

    entity_dof_map_->buildReverseMapping();
    entity_dof_map_->finalize();
#endif
}

// =============================================================================
// DOF Distribution - Mesh-Based API (Convenience Wrappers)
// =============================================================================

#if DOFHANDLER_HAS_MESH

#if 0
struct DofHandler::MeshCacheState : MeshObserver {
    struct PointerSnapshot {
        const void* cell2vertex_offsets{nullptr};
        std::size_t cell2vertex_offsets_size{0};
        const void* cell2vertex{nullptr};
        std::size_t cell2vertex_size{0};
        const void* vertex_gids{nullptr};
        std::size_t vertex_gids_size{0};
        const void* cell_gids{nullptr};
        std::size_t cell_gids_size{0};
        const void* edge2vertex{nullptr};
        std::size_t edge2vertex_size{0}; // number of edges (pairs)
        const void* edge_gids{nullptr};
        std::size_t edge_gids_size{0};
        const void* face2vertex_offsets{nullptr};
        std::size_t face2vertex_offsets_size{0};
        const void* face2vertex{nullptr};
        std::size_t face2vertex_size{0};
        const void* face_gids{nullptr};
        std::size_t face_gids_size{0};

        bool operator==(const PointerSnapshot& other) const noexcept {
            return cell2vertex_offsets == other.cell2vertex_offsets &&
                   cell2vertex_offsets_size == other.cell2vertex_offsets_size &&
                   cell2vertex == other.cell2vertex &&
                   cell2vertex_size == other.cell2vertex_size &&
                   vertex_gids == other.vertex_gids &&
                   vertex_gids_size == other.vertex_gids_size &&
                   cell_gids == other.cell_gids &&
                   cell_gids_size == other.cell_gids_size &&
                   edge2vertex == other.edge2vertex &&
                   edge2vertex_size == other.edge2vertex_size &&
                   edge_gids == other.edge_gids &&
                   edge_gids_size == other.edge_gids_size &&
                   face2vertex_offsets == other.face2vertex_offsets &&
                   face2vertex_offsets_size == other.face2vertex_offsets_size &&
                   face2vertex == other.face2vertex &&
                   face2vertex_size == other.face2vertex_size &&
                   face_gids == other.face_gids &&
                   face_gids_size == other.face_gids_size;
        }
    };

    struct Signature {
        DofLayoutInfo layout{};
        DofNumberingStrategy numbering{DofNumberingStrategy::Sequential};
        OwnershipStrategy ownership{OwnershipStrategy::LowestRank};
        GlobalNumberingMode global_numbering{GlobalNumberingMode::OwnerContiguous};
        TopologyCompletion topology_completion{TopologyCompletion::DeriveMissing};
        bool use_canonical_ordering{true};
        bool validate_parallel{false};
        bool reproducible_across_communicators{false};
        bool no_global_collectives{false};
        int my_rank{0};
        int world_size{1};
#if FE_HAS_MPI
        MPI_Comm mpi_comm{MPI_COMM_WORLD};
#endif

        static bool layout_equal(const DofLayoutInfo& a, const DofLayoutInfo& b) noexcept {
            return a.dofs_per_vertex == b.dofs_per_vertex &&
                   a.dofs_per_edge == b.dofs_per_edge &&
                   a.dofs_per_face == b.dofs_per_face &&
                   a.dofs_per_tri_face == b.dofs_per_tri_face &&
                   a.dofs_per_quad_face == b.dofs_per_quad_face &&
                   a.dofs_per_cell == b.dofs_per_cell &&
                   a.num_components == b.num_components &&
                   a.is_continuous == b.is_continuous &&
                   a.tensor_face_dof_layout == b.tensor_face_dof_layout &&
                   a.total_dofs_per_element == b.total_dofs_per_element;
        }

        bool operator==(const Signature& other) const noexcept {
            return layout_equal(layout, other.layout) &&
                   numbering == other.numbering &&
                   ownership == other.ownership &&
                   global_numbering == other.global_numbering &&
                   topology_completion == other.topology_completion &&
                   use_canonical_ordering == other.use_canonical_ordering &&
                   validate_parallel == other.validate_parallel &&
                   reproducible_across_communicators == other.reproducible_across_communicators &&
                   no_global_collectives == other.no_global_collectives &&
                   my_rank == other.my_rank &&
                   world_size == other.world_size
#if FE_HAS_MPI
                   && mpi_comm == other.mpi_comm
#endif
                ;
        }
    };

	    ScopedSubscription subscription{};
	    const void* mesh_identity{nullptr};
	    std::uint64_t relevant_revision{0};
	    std::uint64_t last_seen_revision{0};
	    bool has_snapshot{false};
    PointerSnapshot pointers{};
    Signature signature{};

    ~MeshCacheState() override { detach(); }

    void on_mesh_event(MeshEvent event) override {
        switch (event) {
            case MeshEvent::TopologyChanged:
            case MeshEvent::PartitionChanged:
            case MeshEvent::AdaptivityApplied:
                ++relevant_revision;
                break;
            default:
                break;
        }
	    }
	
	    void attach(MeshEventBus& new_bus) {
	        if (subscription.bus() == &new_bus && subscription.is_active()) return;
	        subscription = ScopedSubscription(&new_bus, this);
	    }
	
	    void detach() {
	        subscription.unsubscribe();
	    }

    static PointerSnapshot capturePointers(const MeshBase& mesh) {
        PointerSnapshot snap;
        snap.cell2vertex_offsets = mesh.cell2vertex_offsets().data();
        snap.cell2vertex_offsets_size = mesh.cell2vertex_offsets().size();
        snap.cell2vertex = mesh.cell2vertex().data();
        snap.cell2vertex_size = mesh.cell2vertex().size();
        snap.vertex_gids = mesh.vertex_gids().data();
        snap.vertex_gids_size = mesh.vertex_gids().size();
        snap.cell_gids = mesh.cell_gids().data();
        snap.cell_gids_size = mesh.cell_gids().size();
        snap.edge2vertex = mesh.edge2vertex().data();
        snap.edge2vertex_size = mesh.edge2vertex().size();
        snap.edge_gids = mesh.edge_gids().data();
        snap.edge_gids_size = mesh.edge_gids().size();
        snap.face2vertex_offsets = mesh.face2vertex_offsets().data();
        snap.face2vertex_offsets_size = mesh.face2vertex_offsets().size();
        snap.face2vertex = mesh.face2vertex().data();
        snap.face2vertex_size = mesh.face2vertex().size();
        snap.face_gids = mesh.face_gids().data();
        snap.face_gids_size = mesh.face_gids().size();
        return snap;
    }

    static Signature makeSignature(const DofLayoutInfo& layout, const DofDistributionOptions& options) {
        Signature sig;
        sig.layout = layout;
        sig.numbering = options.numbering;
        sig.ownership = options.ownership;
        sig.global_numbering = options.global_numbering;
        sig.topology_completion = options.topology_completion;
        sig.use_canonical_ordering = options.use_canonical_ordering;
        sig.validate_parallel = options.validate_parallel;
        sig.reproducible_across_communicators = options.reproducible_across_communicators;
        sig.no_global_collectives = options.no_global_collectives;
        sig.my_rank = options.my_rank;
        sig.world_size = options.world_size;
#if FE_HAS_MPI
        sig.mpi_comm = options.mpi_comm;
#endif
        return sig;
    }

    [[nodiscard]] bool canSkip(const void* mesh_id,
                               const PointerSnapshot& ptrs,
                               const Signature& sig) const noexcept {
        return has_snapshot &&
               mesh_identity == mesh_id &&
               pointers == ptrs &&
               signature == sig &&
               last_seen_revision == relevant_revision;
    }

    void updateSnapshot(const void* mesh_id, const PointerSnapshot& ptrs, const Signature& sig) {
        mesh_identity = mesh_id;
        pointers = ptrs;
        signature = sig;
        has_snapshot = true;
        last_seen_revision = relevant_revision;
    }
};
#endif

void DofHandler::distributeDofs(const MeshBase& mesh,
                                const spaces::FunctionSpace& space,
                                const DofDistributionOptions& options) {
    checkNotFinalized();

    if (space.is_variable_order()) {
        throw FEException("DofHandler::distributeDofs(MeshBase): variable-order spaces are currently supported through the mesh-topology API only");
    }

    if (mesh.n_cells() == 0) {
        throw FEException("DofHandler::distributeDofs(MeshBase): mesh has no cells");
    }

    DofDistributionOptions opts = options;
    opts.my_rank = 0;
    opts.world_size = 1;
#if FE_HAS_MPI
    opts.mpi_comm = MPI_COMM_WORLD;
#endif

    // Build DofLayoutInfo from FunctionSpace
    DofLayoutInfo layout;
    const auto continuity = space.continuity();
    layout.is_continuous =
        (continuity == Continuity::C0 || continuity == Continuity::C1 ||
         continuity == Continuity::H_curl || continuity == Continuity::H_div);
    layout.tensor_face_dof_layout = false;
    layout.num_components = (continuity == Continuity::H_curl || continuity == Continuity::H_div)
                                ? 1
                                : space.value_dimension();

    const int order = space.polynomial_order();
    const auto total_dofs = static_cast<LocalIndex>(space.dofs_per_element());
    const auto cell0 = mesh.cell_vertices_span(0);
    const int n_verts = static_cast<int>(cell0.second);
    if (n_verts <= 0) {
        throw FEException("DofHandler::distributeDofs(MeshBase): cell 0 has no vertices");
    }

    if (continuity == Continuity::C0 || continuity == Continuity::C1) {
        layout = DofLayoutInfo::Lagrange(order, mesh.dim(), n_verts);
        layout.total_dofs_per_element = total_dofs;
        layout.num_components = space.value_dimension();
    } else if (continuity == Continuity::H_curl || continuity == Continuity::H_div) {
        const auto& elem = space.element();
        const auto& b = elem.basis();
        FE_THROW_IF(!b.is_vector_valued(), FEException,
                    "DofHandler::distributeDofs(MeshBase): H(curl)/H(div) space requires a vector-valued basis");
        const auto* vb = dynamic_cast<const basis::VectorBasisFunction*>(&b);
        FE_THROW_IF(vb == nullptr, FEException,
                    "DofHandler::distributeDofs(MeshBase): vector basis is not a VectorBasisFunction");

        const auto assoc = vb->dof_associations();
        const auto cell_type = infer_element_type_from_cell(mesh.dim(), static_cast<std::size_t>(n_verts));
        const auto ref = (cell_type != ElementType::Unknown)
                             ? elements::ReferenceElement::create(cell_type)
                             : elements::ReferenceElement{};

        const std::size_t num_verts_ref = static_cast<std::size_t>(n_verts);
        const std::size_t num_edges_ref = ref.num_edges();
        const std::size_t num_faces_ref = ref.num_faces();

        std::vector<LocalIndex> per_vertex(num_verts_ref, 0);
        std::vector<LocalIndex> per_edge(num_edges_ref, 0);
        std::vector<LocalIndex> per_face(num_faces_ref, 0);
        LocalIndex interior = 0;

        for (const auto& a : assoc) {
            switch (a.entity_type) {
                case basis::DofEntity::Vertex:
                    if (a.entity_id >= 0 && static_cast<std::size_t>(a.entity_id) < per_vertex.size()) {
                        per_vertex[static_cast<std::size_t>(a.entity_id)] += 1;
                    }
                    break;
                case basis::DofEntity::Edge:
                    if (a.entity_id >= 0 && static_cast<std::size_t>(a.entity_id) < per_edge.size()) {
                        per_edge[static_cast<std::size_t>(a.entity_id)] += 1;
                    }
                    break;
                case basis::DofEntity::Face:
                    if (a.entity_id >= 0 && static_cast<std::size_t>(a.entity_id) < per_face.size()) {
                        per_face[static_cast<std::size_t>(a.entity_id)] += 1;
                    }
                    break;
                case basis::DofEntity::Interior:
                default:
                    interior += 1;
                    break;
            }
        }

        auto uniform_or_zero = [](const std::vector<LocalIndex>& counts) -> std::optional<LocalIndex> {
            if (counts.empty()) return LocalIndex{0};
            LocalIndex v = counts[0];
            for (const auto c : counts) {
                if (c != v) return std::nullopt;
            }
            return v;
        };

        const auto vtx = uniform_or_zero(per_vertex);
        const auto edg = uniform_or_zero(per_edge);
        const auto fac = uniform_or_zero(per_face);
        FE_THROW_IF(!vtx.has_value() || !edg.has_value() || !fac.has_value(), FEException,
                    "DofHandler::distributeDofs(MeshBase): non-uniform per-entity DOF counts are not supported");

        layout.dofs_per_vertex = vtx.value_or(0);
        layout.dofs_per_edge = edg.value_or(0);
        layout.dofs_per_face = fac.value_or(0);
        layout.dofs_per_cell = interior;
        layout.num_components = 1;
        layout.is_continuous = true;
        layout.total_dofs_per_element = total_dofs;
        layout.tensor_face_dof_layout = false;
    } else {
        layout.dofs_per_vertex = 0;
        layout.dofs_per_edge = 0;
        layout.dofs_per_face = 0;
        layout.dofs_per_tri_face = 0;
        layout.dofs_per_quad_face = 0;
        const auto nc = static_cast<LocalIndex>(std::max(1, space.value_dimension()));
        layout.num_components = nc;
        if (nc > 1 && (total_dofs % nc) == 0) {
            layout.dofs_per_cell = total_dofs / nc;
        } else {
            layout.dofs_per_cell = total_dofs;
            layout.num_components = 1;
        }
        layout.total_dofs_per_element = total_dofs;
    }

    if (!mesh_cache_) {
        mesh_cache_ = std::make_unique<MeshCacheState>();
    }
    mesh_cache_->attach(mesh.event_bus());

    const auto ptrs = MeshCacheState::capturePointers(mesh);
    const auto sig = MeshCacheState::makeSignature(layout, opts);
    if (mesh_cache_->canSkip(&mesh, ptrs, sig, dof_state_revision_)) {
        return;
    }

    MeshTopologyView topo;
    topo.n_cells = static_cast<GlobalIndex>(mesh.n_cells());
    topo.n_vertices = static_cast<GlobalIndex>(mesh.n_vertices());
    topo.dim = mesh.dim();
    topo.cell2vertex_offsets = mesh.cell2vertex_offsets();
    topo.cell2vertex_data = mesh.cell2vertex();
    topo.vertex_gids = mesh.vertex_gids();
    topo.vertex_coords = mesh.X_ref();
    topo.cell_gids = mesh.cell_gids();

    const bool need_edges = layout.is_continuous && layout.dofs_per_edge > 0;
    const bool need_faces = layout.is_continuous && layout.has_face_dofs();

    CellToEntityCSR cell2edge;
    CellToEntityCSR cell2face;

    bool missing_edges = false;
    bool missing_faces = false;

    if (need_edges) {
        if (topo.dim == 2) {
            // In the Mesh library, (n-1)-entities are stored as faces; for 2D meshes, those faces are edges.
            const bool have_edges = (mesh.n_faces() > 0) &&
                                    !mesh.face2vertex_offsets().empty() &&
                                    !mesh.face2vertex().empty();
            if (!have_edges) {
                missing_edges = true;
            } else {
                topo.n_edges = static_cast<GlobalIndex>(mesh.n_faces());
                topo.edge_gids = mesh.face_gids();
                topo.edge2vertex_data = mesh.face2vertex();

                const auto f2v_offsets = mesh.face2vertex_offsets();
                const auto f2v = mesh.face2vertex();
                if (f2v_offsets.size() != static_cast<std::size_t>(mesh.n_faces()) + 1u) {
                    throw FEException("DofHandler::distributeDofs(MeshBase): invalid face2vertex_offsets size for 2D edge mapping");
                }
                for (std::size_t f = 0; f + 1 < f2v_offsets.size(); ++f) {
                    const auto begin = static_cast<std::size_t>(f2v_offsets[f]);
                    const auto end = static_cast<std::size_t>(f2v_offsets[f + 1]);
                    if (end < begin || end > f2v.size() || (end - begin) != 2u) {
                        throw FEException("DofHandler::distributeDofs(MeshBase): expected 2 vertices per face for 2D edge mapping");
                    }
                }

                static_assert(sizeof(std::array<MeshIndex, 2>) == sizeof(MeshIndex) * 2,
                              "std::array<MeshIndex,2> must be tightly packed");
                const auto* pairs = reinterpret_cast<const std::array<MeshIndex, 2>*>(f2v.data());
                cell2edge = buildCellToEdgesRefOrder(topo.dim,
                                                     topo.cell2vertex_offsets,
                                                     topo.cell2vertex_data,
                                                     std::span<const std::array<MeshIndex, 2>>(pairs, static_cast<std::size_t>(mesh.n_faces())));
                topo.cell2edge_offsets = cell2edge.offsets;
                topo.cell2edge_data = cell2edge.data;
            }
        } else {
            const bool have_edges = (mesh.n_edges() > 0) && !mesh.edge2vertex().empty();
            if (!have_edges) {
                missing_edges = true;
            } else {
                topo.n_edges = static_cast<GlobalIndex>(mesh.n_edges());
                topo.edge_gids = mesh.edge_gids();
                const auto& e2v = mesh.edge2vertex();
                static_assert(sizeof(std::array<MeshIndex, 2>) == sizeof(MeshIndex) * 2,
                              "std::array<MeshIndex,2> must be tightly packed");
                const auto* flat = e2v.empty() ? nullptr : reinterpret_cast<const MeshIndex*>(e2v.data());
                topo.edge2vertex_data = std::span<const MeshIndex>(flat, static_cast<std::size_t>(2u * e2v.size()));
                cell2edge = buildCellToEdgesRefOrder(topo.dim,
                                                     topo.cell2vertex_offsets,
                                                     topo.cell2vertex_data,
                                                     std::span<const std::array<MeshIndex, 2>>(e2v.data(), e2v.size()));
                topo.cell2edge_offsets = cell2edge.offsets;
                topo.cell2edge_data = cell2edge.data;
            }
        }
    }

    if (need_faces) {
        const bool have_faces = (mesh.n_faces() > 0) &&
                                !mesh.face2vertex_offsets().empty() &&
                                !mesh.face2vertex().empty();
        if (!have_faces) {
            missing_faces = true;
        } else {
            topo.n_faces = static_cast<GlobalIndex>(mesh.n_faces());
            topo.face_gids = mesh.face_gids();
            topo.face2vertex_offsets = mesh.face2vertex_offsets();
            topo.face2vertex_data = mesh.face2vertex();
            cell2face = buildCellToFacesRefOrder(topo.dim,
                                                 topo.cell2vertex_offsets,
                                                 topo.cell2vertex_data,
                                                 topo.face2vertex_offsets,
                                                 topo.face2vertex_data);
            topo.cell2face_offsets = cell2face.offsets;
            topo.cell2face_data = cell2face.data;
        }
    }

    if ((missing_edges && need_edges) || (missing_faces && need_faces)) {
        if (opts.topology_completion == TopologyCompletion::RequireComplete) {
            if (missing_edges && need_edges) {
                throw FEException("DofHandler::distributeDofs(MeshBase): edge-interior DOFs require Mesh edges (n_edges>0 and edge2vertex populated)");
            }
            if (missing_faces && need_faces) {
                throw FEException("DofHandler::distributeDofs(MeshBase): face-interior DOFs require Mesh faces (n_faces>0 and face2vertex populated)");
            }
        }

        auto derived = topo.materialize();
        if (missing_edges && need_edges) {
            derive_edge_connectivity(derived);
        }
        if (missing_faces && need_faces) {
            derive_face_connectivity(derived);
        }
        const auto derived_view = MeshTopologyView::from(derived);
        distributeDofsCore(derived_view, layout, opts);
    } else {
        distributeDofsCore(topo, layout, opts);
    }

    mesh_cache_->updateSnapshot(&mesh, ptrs, sig, dof_state_revision_);
}

void DofHandler::distributeDofs(const Mesh& mesh,
                                const spaces::FunctionSpace& space,
                                const DofDistributionOptions& options) {
    checkNotFinalized();

    if (space.is_variable_order()) {
        throw FEException("DofHandler::distributeDofs(Mesh): variable-order spaces are currently supported through the mesh-topology API only");
    }

    const auto& local_mesh = mesh.local_mesh();
    if (local_mesh.n_cells() == 0) {
        throw FEException("DofHandler::distributeDofs(Mesh): local mesh has no cells");
    }

    DofDistributionOptions opts = options;
    opts.my_rank = mesh.rank();
    opts.world_size = mesh.world_size();
#if FE_HAS_MPI && defined(MESH_HAS_MPI)
    opts.mpi_comm = mesh.mpi_comm();
#endif

    // Build DofLayoutInfo from FunctionSpace
    DofLayoutInfo layout;
    const auto continuity = space.continuity();
    layout.is_continuous =
        (continuity == Continuity::C0 || continuity == Continuity::C1 ||
         continuity == Continuity::H_curl || continuity == Continuity::H_div);
    layout.tensor_face_dof_layout = false;
    layout.num_components = (continuity == Continuity::H_curl || continuity == Continuity::H_div)
                                ? 1
                                : space.value_dimension();

    const int order = space.polynomial_order();
    const auto total_dofs = static_cast<LocalIndex>(space.dofs_per_element());
    const auto cell0 = local_mesh.cell_vertices_span(0);
    const int n_verts = static_cast<int>(cell0.second);
    if (n_verts <= 0) {
        throw FEException("DofHandler::distributeDofs(Mesh): cell 0 has no vertices");
    }

    if (continuity == Continuity::C0 || continuity == Continuity::C1) {
        layout = DofLayoutInfo::Lagrange(order, local_mesh.dim(), n_verts);
        layout.total_dofs_per_element = total_dofs;
        layout.num_components = space.value_dimension();
    } else if (continuity == Continuity::H_curl || continuity == Continuity::H_div) {
        const auto& elem = space.element();
        const auto& b = elem.basis();
        FE_THROW_IF(!b.is_vector_valued(), FEException,
                    "DofHandler::distributeDofs(Mesh): H(curl)/H(div) space requires a vector-valued basis");
        const auto* vb = dynamic_cast<const basis::VectorBasisFunction*>(&b);
        FE_THROW_IF(vb == nullptr, FEException,
                    "DofHandler::distributeDofs(Mesh): vector basis is not a VectorBasisFunction");

        const auto assoc = vb->dof_associations();
        const auto cell_type = infer_element_type_from_cell(local_mesh.dim(), static_cast<std::size_t>(n_verts));
        const auto ref = (cell_type != ElementType::Unknown)
                             ? elements::ReferenceElement::create(cell_type)
                             : elements::ReferenceElement{};

        const std::size_t num_verts_ref = static_cast<std::size_t>(n_verts);
        const std::size_t num_edges_ref = ref.num_edges();
        const std::size_t num_faces_ref = ref.num_faces();

        std::vector<LocalIndex> per_vertex(num_verts_ref, 0);
        std::vector<LocalIndex> per_edge(num_edges_ref, 0);
        std::vector<LocalIndex> per_face(num_faces_ref, 0);
        LocalIndex interior = 0;

        for (const auto& a : assoc) {
            switch (a.entity_type) {
                case basis::DofEntity::Vertex:
                    if (a.entity_id >= 0 && static_cast<std::size_t>(a.entity_id) < per_vertex.size()) {
                        per_vertex[static_cast<std::size_t>(a.entity_id)] += 1;
                    }
                    break;
                case basis::DofEntity::Edge:
                    if (a.entity_id >= 0 && static_cast<std::size_t>(a.entity_id) < per_edge.size()) {
                        per_edge[static_cast<std::size_t>(a.entity_id)] += 1;
                    }
                    break;
                case basis::DofEntity::Face:
                    if (a.entity_id >= 0 && static_cast<std::size_t>(a.entity_id) < per_face.size()) {
                        per_face[static_cast<std::size_t>(a.entity_id)] += 1;
                    }
                    break;
                case basis::DofEntity::Interior:
                default:
                    interior += 1;
                    break;
            }
        }

        auto uniform_or_zero = [](const std::vector<LocalIndex>& counts) -> std::optional<LocalIndex> {
            if (counts.empty()) return LocalIndex{0};
            LocalIndex v = counts[0];
            for (const auto c : counts) {
                if (c != v) return std::nullopt;
            }
            return v;
        };

        const auto vtx = uniform_or_zero(per_vertex);
        const auto edg = uniform_or_zero(per_edge);
        const auto fac = uniform_or_zero(per_face);
        FE_THROW_IF(!vtx.has_value() || !edg.has_value() || !fac.has_value(), FEException,
                    "DofHandler::distributeDofs(Mesh): non-uniform per-entity DOF counts are not supported");

        layout.dofs_per_vertex = vtx.value_or(0);
        layout.dofs_per_edge = edg.value_or(0);
        layout.dofs_per_face = fac.value_or(0);
        layout.dofs_per_cell = interior;
        layout.num_components = 1;
        layout.is_continuous = true;
        layout.total_dofs_per_element = total_dofs;
        layout.tensor_face_dof_layout = false;
    } else {
        layout.dofs_per_vertex = 0;
        layout.dofs_per_edge = 0;
        layout.dofs_per_face = 0;
        layout.dofs_per_tri_face = 0;
        layout.dofs_per_quad_face = 0;
        const auto nc = static_cast<LocalIndex>(std::max(1, space.value_dimension()));
        layout.num_components = nc;
        if (nc > 1 && (total_dofs % nc) == 0) {
            layout.dofs_per_cell = total_dofs / nc;
        } else {
            layout.dofs_per_cell = total_dofs;
            layout.num_components = 1;
        }
        layout.total_dofs_per_element = total_dofs;
    }

    if (!mesh_cache_) {
        mesh_cache_ = std::make_unique<MeshCacheState>();
    }
    mesh_cache_->attach(local_mesh.event_bus());

    const auto ptrs = MeshCacheState::capturePointers(local_mesh);
    const auto sig = MeshCacheState::makeSignature(layout, opts);
    if (mesh_cache_->canSkip(&mesh, ptrs, sig, dof_state_revision_)) {
        return;
    }

    MeshTopologyView topo;
    topo.n_cells = static_cast<GlobalIndex>(local_mesh.n_cells());
    topo.n_vertices = static_cast<GlobalIndex>(local_mesh.n_vertices());
    topo.dim = local_mesh.dim();
    topo.cell2vertex_offsets = local_mesh.cell2vertex_offsets();
    topo.cell2vertex_data = local_mesh.cell2vertex();
    topo.vertex_gids = local_mesh.vertex_gids();
    topo.vertex_coords = local_mesh.X_ref();
    topo.cell_gids = local_mesh.cell_gids();

    std::vector<int> cell_owner_ranks(static_cast<std::size_t>(topo.n_cells), opts.my_rank);
    for (GlobalIndex c = 0; c < topo.n_cells; ++c) {
        cell_owner_ranks[static_cast<std::size_t>(c)] =
            static_cast<int>(mesh.owner_rank_cell(static_cast<index_t>(c)));
    }
    topo.cell_owner_ranks = cell_owner_ranks;

    std::vector<int> neighbor_ranks;
    neighbor_ranks.reserve(mesh.neighbor_ranks().size());
    for (auto r : mesh.neighbor_ranks()) {
        const int rr = static_cast<int>(r);
        if (rr != opts.my_rank) {
            neighbor_ranks.push_back(rr);
        }
    }
    std::sort(neighbor_ranks.begin(), neighbor_ranks.end());
    neighbor_ranks.erase(std::unique(neighbor_ranks.begin(), neighbor_ranks.end()), neighbor_ranks.end());
    topo.neighbor_ranks = neighbor_ranks;

    const bool need_edges = layout.is_continuous && layout.dofs_per_edge > 0;
    const bool need_faces = layout.is_continuous && layout.has_face_dofs();

    CellToEntityCSR cell2edge;
    CellToEntityCSR cell2face;

    bool missing_edges = false;
    bool missing_faces = false;

    if (need_edges) {
        if (topo.dim == 2) {
            const bool have_edges = (local_mesh.n_faces() > 0) &&
                                    !local_mesh.face2vertex_offsets().empty() &&
                                    !local_mesh.face2vertex().empty();
            if (!have_edges) {
                missing_edges = true;
            } else {
                topo.n_edges = static_cast<GlobalIndex>(local_mesh.n_faces());
                topo.edge_gids = local_mesh.face_gids();
                topo.edge2vertex_data = local_mesh.face2vertex();

                const auto f2v_offsets = local_mesh.face2vertex_offsets();
                const auto f2v = local_mesh.face2vertex();
                if (f2v_offsets.size() != static_cast<std::size_t>(local_mesh.n_faces()) + 1u) {
                    throw FEException("DofHandler::distributeDofs(Mesh): invalid face2vertex_offsets size for 2D edge mapping");
                }
                for (std::size_t f = 0; f + 1 < f2v_offsets.size(); ++f) {
                    const auto begin = static_cast<std::size_t>(f2v_offsets[f]);
                    const auto end = static_cast<std::size_t>(f2v_offsets[f + 1]);
                    if (end < begin || end > f2v.size() || (end - begin) != 2u) {
                        throw FEException("DofHandler::distributeDofs(Mesh): expected 2 vertices per face for 2D edge mapping");
                    }
                }

                static_assert(sizeof(std::array<MeshIndex, 2>) == sizeof(MeshIndex) * 2,
                              "std::array<MeshIndex,2> must be tightly packed");
                const auto* pairs = reinterpret_cast<const std::array<MeshIndex, 2>*>(f2v.data());
                cell2edge = buildCellToEdgesRefOrder(topo.dim,
                                                     topo.cell2vertex_offsets,
                                                     topo.cell2vertex_data,
                                                     std::span<const std::array<MeshIndex, 2>>(pairs, static_cast<std::size_t>(local_mesh.n_faces())));
                topo.cell2edge_offsets = cell2edge.offsets;
                topo.cell2edge_data = cell2edge.data;
            }
        } else {
            const bool have_edges = (local_mesh.n_edges() > 0) && !local_mesh.edge2vertex().empty();
            if (!have_edges) {
                missing_edges = true;
            } else {
                topo.n_edges = static_cast<GlobalIndex>(local_mesh.n_edges());
                topo.edge_gids = local_mesh.edge_gids();
                const auto& e2v = local_mesh.edge2vertex();
                static_assert(sizeof(std::array<MeshIndex, 2>) == sizeof(MeshIndex) * 2,
                              "std::array<MeshIndex,2> must be tightly packed");
                const auto* flat = e2v.empty() ? nullptr : reinterpret_cast<const MeshIndex*>(e2v.data());
                topo.edge2vertex_data = std::span<const MeshIndex>(flat, static_cast<std::size_t>(2u * e2v.size()));
                cell2edge = buildCellToEdgesRefOrder(topo.dim,
                                                     topo.cell2vertex_offsets,
                                                     topo.cell2vertex_data,
                                                     std::span<const std::array<MeshIndex, 2>>(e2v.data(), e2v.size()));
                topo.cell2edge_offsets = cell2edge.offsets;
                topo.cell2edge_data = cell2edge.data;
            }
        }
    }

    if (need_faces) {
        const bool have_faces = (local_mesh.n_faces() > 0) &&
                                !local_mesh.face2vertex_offsets().empty() &&
                                !local_mesh.face2vertex().empty();
        if (!have_faces) {
            missing_faces = true;
        } else {
            topo.n_faces = static_cast<GlobalIndex>(local_mesh.n_faces());
            topo.face_gids = local_mesh.face_gids();
            topo.face2vertex_offsets = local_mesh.face2vertex_offsets();
            topo.face2vertex_data = local_mesh.face2vertex();
            cell2face = buildCellToFacesRefOrder(topo.dim,
                                                 topo.cell2vertex_offsets,
                                                 topo.cell2vertex_data,
                                                 topo.face2vertex_offsets,
                                                 topo.face2vertex_data);
            topo.cell2face_offsets = cell2face.offsets;
            topo.cell2face_data = cell2face.data;
        }
    }

    if ((missing_edges && need_edges) || (missing_faces && need_faces)) {
        if (opts.topology_completion == TopologyCompletion::RequireComplete) {
            if (missing_edges && need_edges) {
                throw FEException("DofHandler::distributeDofs(Mesh): edge-interior DOFs require Mesh edges (n_edges>0 and edge2vertex populated)");
            }
            if (missing_faces && need_faces) {
                throw FEException("DofHandler::distributeDofs(Mesh): face-interior DOFs require Mesh faces (n_faces>0 and face2vertex populated)");
            }
        }

        auto derived = topo.materialize();
        if (missing_edges && need_edges) {
            derive_edge_connectivity(derived);
        }
        if (missing_faces && need_faces) {
            derive_face_connectivity(derived);
        }
        const auto derived_view = MeshTopologyView::from(derived);
        distributeDofsCore(derived_view, layout, opts);
    } else {
        distributeDofsCore(topo, layout, opts);
    }

    mesh_cache_->updateSnapshot(&mesh, ptrs, sig, dof_state_revision_);
    return;

#if 0
    DofDistributionOptions opts = options;
    opts.my_rank = mesh.rank();
    opts.world_size = mesh.world_size();
#if FE_HAS_MPI
    opts.mpi_comm = mesh.mpi_comm();
#endif

    const auto& local_mesh = mesh.local_mesh();

    MeshTopologyInfo topology;
    topology.n_cells = static_cast<GlobalIndex>(local_mesh.n_cells());
    topology.n_vertices = static_cast<GlobalIndex>(local_mesh.n_vertices());
    topology.n_edges = 0;
    topology.n_faces = 0;
    topology.dim = local_mesh.dim();

    if (topology.n_cells <= 0) {
        throw FEException("DofHandler::distributeDofs(Mesh): local mesh has no cells");
    }

    // Build cell-to-vertex connectivity
    topology.cell2vertex_offsets.resize(topology.n_cells + 1);
    topology.cell2vertex_offsets[0] = 0;

    for (GlobalIndex c = 0; c < topology.n_cells; ++c) {
        auto verts = local_mesh.cell_vertices(static_cast<index_t>(c));
        topology.cell2vertex_offsets[c + 1] = topology.cell2vertex_offsets[c] +
                                              static_cast<GlobalIndex>(verts.size());
    }

    topology.cell2vertex_data.resize(topology.cell2vertex_offsets[topology.n_cells]);

    for (GlobalIndex c = 0; c < topology.n_cells; ++c) {
        auto verts = local_mesh.cell_vertices(static_cast<index_t>(c));
        auto offset = topology.cell2vertex_offsets[c];
        for (std::size_t i = 0; i < verts.size(); ++i) {
            topology.cell2vertex_data[offset + static_cast<GlobalIndex>(i)] =
                static_cast<GlobalIndex>(verts[i]);
        }
    }

    // Copy global IDs when available.
    topology.vertex_gids = std::vector<gid_t>(local_mesh.vertex_gids().begin(),
                                              local_mesh.vertex_gids().end());
    topology.cell_gids = std::vector<gid_t>(local_mesh.cell_gids().begin(),
                                            local_mesh.cell_gids().end());
    topology.edge_gids = std::vector<gid_t>(local_mesh.edge_gids().begin(),
                                            local_mesh.edge_gids().end());
    topology.face_gids = std::vector<gid_t>(local_mesh.face_gids().begin(),
                                            local_mesh.face_gids().end());

    // Cell owner ranks (owned/ghost flags) from Mesh.
    topology.cell_owner_ranks.resize(static_cast<std::size_t>(topology.n_cells), opts.my_rank);
    for (GlobalIndex c = 0; c < topology.n_cells; ++c) {
        topology.cell_owner_ranks[static_cast<std::size_t>(c)] =
            static_cast<int>(mesh.owner_rank_cell(static_cast<index_t>(c)));
    }

    // Neighbor ranks: use Mesh-provided neighbor set when available for
    // vertex/edge-only sharing cases (more complete than cell-owner inference).
    topology.neighbor_ranks.clear();
    topology.neighbor_ranks.reserve(mesh.neighbor_ranks().size());
    for (auto r : mesh.neighbor_ranks()) {
        const int rr = static_cast<int>(r);
        if (rr != opts.my_rank) {
            topology.neighbor_ranks.push_back(rr);
        }
    }

    // Build DofLayoutInfo from FunctionSpace
    DofLayoutInfo layout;
    auto continuity = space.continuity();
    layout.is_continuous = (continuity == Continuity::C0 || continuity == Continuity::C1);
    layout.num_components = space.value_dimension();

    const int order = space.polynomial_order();
    const auto total_dofs = static_cast<LocalIndex>(space.dofs_per_element());

    if (layout.is_continuous) {
        const int n_verts = static_cast<int>(local_mesh.cell_vertices(0).size());
        layout = DofLayoutInfo::Lagrange(order, topology.dim, n_verts);
        layout.total_dofs_per_element = total_dofs;
        layout.num_components = space.value_dimension();
    } else {
        layout.dofs_per_vertex = 0;
        layout.dofs_per_edge = 0;
        layout.dofs_per_face = 0;
        layout.dofs_per_tri_face = 0;
        layout.dofs_per_quad_face = 0;
        const auto nc = static_cast<LocalIndex>(std::max(1, layout.num_components));
        if (nc > 1 && (total_dofs % nc) == 0) {
            layout.dofs_per_cell = total_dofs / nc; // per component
        } else {
            layout.dofs_per_cell = total_dofs;
            layout.num_components = 1;
        }
        layout.total_dofs_per_element = total_dofs;
    }

    // Derive edge/face topology for higher-order CG when needed.
    const bool need_edges = layout.is_continuous && layout.dofs_per_edge > 0;
    const bool need_faces = layout.is_continuous && layout.has_face_dofs();

    if (need_edges || need_faces) {
        auto infer_base_type = [&](std::size_t n_verts) -> ElementType {
            if (topology.dim == 1 && n_verts == 2) return ElementType::Line2;
            if (topology.dim == 2 && n_verts == 3) return ElementType::Triangle3;
            if (topology.dim == 2 && n_verts == 4) return ElementType::Quad4;
            if (topology.dim == 3 && n_verts == 4) return ElementType::Tetra4;
            if (topology.dim == 3 && n_verts == 8) return ElementType::Hex8;
            if (topology.dim == 3 && n_verts == 6) return ElementType::Wedge6;
            if (topology.dim == 3 && n_verts == 5) return ElementType::Pyramid5;
            return ElementType::Unknown;
        };

        const auto base_type = infer_base_type(local_mesh.cell_vertices(0).size());
        if (base_type == ElementType::Unknown) {
            throw FEException("DofHandler::distributeDofs(Mesh): unsupported cell type for edge/face topology derivation");
        }
        if (topology.vertex_gids.size() != static_cast<std::size_t>(topology.n_vertices)) {
            throw FEException("DofHandler::distributeDofs(Mesh): vertex_gids are required to derive edge/face topology deterministically");
        }

        const auto ref = elements::ReferenceElement::create(base_type);

        struct EdgeKey {
            gid_t a;
            gid_t b;
            bool operator==(const EdgeKey& other) const noexcept { return a == other.a && b == other.b; }
        };
        struct EdgeKeyHash {
            std::size_t operator()(const EdgeKey& k) const noexcept {
                const std::size_t h1 = std::hash<gid_t>{}(k.a);
                const std::size_t h2 = std::hash<gid_t>{}(k.b);
                return h1 ^ (h2 + static_cast<std::size_t>(kHashSalt) + (h1 << 6) + (h1 >> 2));
            }
        };

        struct FaceKey {
            std::array<gid_t, 4> gids{};
            std::uint8_t n{0};
            bool operator==(const FaceKey& other) const noexcept { return n == other.n && gids == other.gids; }
        };
        struct FaceKeyHash {
            std::size_t operator()(const FaceKey& k) const noexcept {
                std::size_t seed = std::hash<std::uint8_t>{}(k.n);
                for (std::size_t i = 0; i < static_cast<std::size_t>(k.n); ++i) {
                    const std::size_t h = std::hash<gid_t>{}(k.gids[i]);
                    seed ^= h + static_cast<std::size_t>(kHashSalt) + (seed << 6) + (seed >> 2);
                }
                return seed;
            }
        };

        auto canonical_cycle = [&](const std::vector<GlobalIndex>& verts) -> std::vector<GlobalIndex> {
            if (verts.empty()) return {};
            const std::size_t n = verts.size();

            std::size_t start = 0;
            gid_t best_gid = topology.vertex_gids[static_cast<std::size_t>(verts[0])];
            GlobalIndex best_vid = verts[0];
            for (std::size_t i = 1; i < n; ++i) {
                const auto v = verts[i];
                const auto gid = topology.vertex_gids[static_cast<std::size_t>(v)];
                if (gid < best_gid || (gid == best_gid && v < best_vid)) {
                    best_gid = gid;
                    best_vid = v;
                    start = i;
                }
            }

            const std::size_t next = (start + 1u) % n;
            const std::size_t prev = (start + n - 1u) % n;
            const gid_t gid_next = topology.vertex_gids[static_cast<std::size_t>(verts[next])];
            const gid_t gid_prev = topology.vertex_gids[static_cast<std::size_t>(verts[prev])];

            const bool forward =
                (gid_next < gid_prev) ||
                (gid_next == gid_prev && verts[next] < verts[prev]);

            std::vector<GlobalIndex> out;
            out.reserve(n);
            for (std::size_t k = 0; k < n; ++k) {
                const std::size_t idx = forward ? ((start + k) % n)
                                                : ((start + n - k) % n);
                out.push_back(verts[idx]);
            }
            return out;
        };

        std::unordered_map<EdgeKey, GlobalIndex, EdgeKeyHash> edge_ids;
        edge_ids.reserve(static_cast<std::size_t>(topology.n_cells) * ref.num_edges());

        std::unordered_map<FaceKey, GlobalIndex, FaceKeyHash> face_ids;
        face_ids.reserve(static_cast<std::size_t>(topology.n_cells) * ref.num_faces());

        std::vector<std::array<GlobalIndex, 2>> edges;
        std::vector<std::vector<GlobalIndex>> faces;

        if (need_edges) {
            topology.cell2edge_offsets.resize(static_cast<std::size_t>(topology.n_cells) + 1u, 0);
            topology.cell2edge_data.clear();
            topology.cell2edge_data.reserve(static_cast<std::size_t>(topology.n_cells) * ref.num_edges());
        }

        if (need_faces) {
            topology.cell2face_offsets.resize(static_cast<std::size_t>(topology.n_cells) + 1u, 0);
            topology.cell2face_data.clear();
            topology.cell2face_data.reserve(static_cast<std::size_t>(topology.n_cells) * ref.num_faces());
        }

        for (GlobalIndex c = 0; c < topology.n_cells; ++c) {
            const auto cell_verts = topology.getCellVertices(c);

            if (need_edges) {
                topology.cell2edge_offsets[static_cast<std::size_t>(c)] =
                    static_cast<GlobalIndex>(topology.cell2edge_data.size());

                for (std::size_t le = 0; le < ref.num_edges(); ++le) {
                    const auto& en = ref.edge_nodes(le);
                    if (en.size() != 2u) {
                        throw FEException("DofHandler: unexpected edge node count while deriving topology");
                    }
                    const auto lv0 = static_cast<std::size_t>(en[0]);
                    const auto lv1 = static_cast<std::size_t>(en[1]);
                    if (lv0 >= cell_verts.size() || lv1 >= cell_verts.size()) {
                        throw FEException("DofHandler: edge vertex index out of range while deriving topology");
                    }

                    const GlobalIndex gv0 = cell_verts[lv0];
                    const GlobalIndex gv1 = cell_verts[lv1];
                    const gid_t gid0 = topology.vertex_gids[static_cast<std::size_t>(gv0)];
                    const gid_t gid1 = topology.vertex_gids[static_cast<std::size_t>(gv1)];

                    EdgeKey key{std::min(gid0, gid1), std::max(gid0, gid1)};
                    auto it = edge_ids.find(key);
                    GlobalIndex eid = -1;
                    if (it == edge_ids.end()) {
                        eid = static_cast<GlobalIndex>(edges.size());
                        edge_ids.emplace(key, eid);

                        if (gid0 <= gid1) {
                            edges.push_back({gv0, gv1});
                        } else {
                            edges.push_back({gv1, gv0});
                        }
                    } else {
                        eid = it->second;
                    }

                    topology.cell2edge_data.push_back(eid);
                }
            }

            if (need_faces) {
                topology.cell2face_offsets[static_cast<std::size_t>(c)] =
                    static_cast<GlobalIndex>(topology.cell2face_data.size());

                for (std::size_t lf = 0; lf < ref.num_faces(); ++lf) {
                    const auto& fn = ref.face_nodes(lf);
                    if (fn.size() != 3u && fn.size() != 4u) {
                        throw FEException("DofHandler: unexpected face node count while deriving topology");
                    }

                    std::vector<GlobalIndex> face_verts;
                    face_verts.reserve(fn.size());
                    FaceKey key{};
                    key.n = static_cast<std::uint8_t>(fn.size());
                    for (std::size_t i = 0; i < fn.size(); ++i) {
                        const auto lv = static_cast<std::size_t>(fn[i]);
                        if (lv >= cell_verts.size()) {
                            throw FEException("DofHandler: face vertex index out of range while deriving topology");
                        }
                        const GlobalIndex gv = cell_verts[lv];
                        face_verts.push_back(gv);
                        key.gids[i] = topology.vertex_gids[static_cast<std::size_t>(gv)];
                    }
                    std::sort(key.gids.begin(), key.gids.begin() + key.n);

                    auto it = face_ids.find(key);
                    GlobalIndex fid = -1;
                    if (it == face_ids.end()) {
                        fid = static_cast<GlobalIndex>(faces.size());
                        face_ids.emplace(key, fid);
                        faces.push_back(canonical_cycle(face_verts));
                    } else {
                        fid = it->second;
                    }

                    topology.cell2face_data.push_back(fid);
                }
            }
        }

        if (need_edges) {
            topology.cell2edge_offsets[static_cast<std::size_t>(topology.n_cells)] =
                static_cast<GlobalIndex>(topology.cell2edge_data.size());
            topology.n_edges = static_cast<GlobalIndex>(edges.size());
            topology.edge2vertex_data.resize(static_cast<std::size_t>(2) * static_cast<std::size_t>(topology.n_edges));
            for (GlobalIndex e = 0; e < topology.n_edges; ++e) {
                topology.edge2vertex_data[static_cast<std::size_t>(2 * e + 0)] = edges[static_cast<std::size_t>(e)][0];
                topology.edge2vertex_data[static_cast<std::size_t>(2 * e + 1)] = edges[static_cast<std::size_t>(e)][1];
            }
        }

        if (need_faces) {
            topology.cell2face_offsets[static_cast<std::size_t>(topology.n_cells)] =
                static_cast<GlobalIndex>(topology.cell2face_data.size());
            topology.n_faces = static_cast<GlobalIndex>(faces.size());

            topology.face2vertex_offsets.clear();
            topology.face2vertex_data.clear();
            topology.face2vertex_offsets.reserve(static_cast<std::size_t>(topology.n_faces) + 1u);
            topology.face2vertex_offsets.push_back(0);

            for (const auto& fv : faces) {
                topology.face2vertex_data.insert(topology.face2vertex_data.end(), fv.begin(), fv.end());
                topology.face2vertex_offsets.push_back(static_cast<GlobalIndex>(topology.face2vertex_data.size()));
            }
        }
    }

    distributeDofsCore(topology, layout, opts);
#endif
}

void DofHandler::distributeDofs(const MeshBase& mesh,
                                const svmp::InterfaceMesh& interface_mesh,
                                const spaces::FunctionSpace& space,
                                const DofDistributionOptions& options) {
    checkNotFinalized();

    const auto* mortar_space = dynamic_cast<const spaces::MortarSpace*>(&space);
    FE_THROW_IF(mortar_space == nullptr, FEException,
                "DofHandler::distributeDofs(MeshBase, InterfaceMesh): space must be a MortarSpace");
    FE_THROW_IF(mortar_space->interface_space().continuity() != Continuity::L2, FEException,
                "DofHandler::distributeDofs(MeshBase, InterfaceMesh): first mortar pass requires a discontinuous interface space");
    FE_THROW_IF(interface_mesh.n_faces() == 0u, FEException,
                "DofHandler::distributeDofs(MeshBase, InterfaceMesh): interface mesh has no faces");

    my_rank_ = options.my_rank;
    world_size_ = options.world_size;
    global_numbering_ = options.global_numbering;
#if FE_HAS_MPI
    mpi_comm_ = options.mpi_comm;
#else
    if (world_size_ > 1) {
        throw FEException("DofHandler::distributeDofs(MeshBase, InterfaceMesh): MPI world_size>1 but FE is built without MPI support");
    }
#endif

    n_cells_ = static_cast<GlobalIndex>(interface_mesh.n_faces());
    spatial_dim_ = interface_mesh.spatial_dim();
    num_components_ = static_cast<LocalIndex>(std::max(1, space.value_dimension()));
    dof_map_.setMyRank(my_rank_);
    ++dof_state_revision_;

    cell_edge_orient_offsets_.clear();
    cell_edge_orient_data_.clear();
    cell_face_orient_offsets_.clear();
    cell_face_orient_data_.clear();
    clearSpatialDofCoordinates();

    const auto dofs_per_face = static_cast<LocalIndex>(space.dofs_per_element());
    FE_THROW_IF(dofs_per_face <= 0, FEException,
                "DofHandler::distributeDofs(MeshBase, InterfaceMesh): mortar space has no face DOFs");

    const GlobalIndex n_iface_faces = static_cast<GlobalIndex>(interface_mesh.n_faces());
    const GlobalIndex total_dofs =
        n_iface_faces * static_cast<GlobalIndex>(dofs_per_face);

    dof_map_ = DofMap(n_iface_faces, total_dofs, dofs_per_face);
    dof_map_.setMyRank(my_rank_);
    dof_map_.setNumDofs(total_dofs);
    dof_map_.setNumLocalDofs(total_dofs);
    dof_map_.setDofOwnership([owner = my_rank_](GlobalIndex) { return owner; });

    entity_dof_map_ = std::make_unique<EntityDofMap>();
    entity_dof_map_->reserve(/*vertices=*/0,
                             /*edges=*/0,
                             static_cast<GlobalIndex>(mesh.n_faces()),
                             static_cast<GlobalIndex>(mesh.n_cells()));

    std::vector<GlobalIndex> face_dofs(static_cast<std::size_t>(dofs_per_face));
    for (GlobalIndex lf = 0; lf < n_iface_faces; ++lf) {
        const GlobalIndex first = lf * static_cast<GlobalIndex>(dofs_per_face);
        for (LocalIndex d = 0; d < dofs_per_face; ++d) {
            face_dofs[static_cast<std::size_t>(d)] = first + static_cast<GlobalIndex>(d);
        }
        dof_map_.setCellDofs(lf, face_dofs);

        const auto volume_face = static_cast<GlobalIndex>(
            interface_mesh.volume_face(static_cast<svmp::index_t>(lf)));
        FE_THROW_IF(volume_face < 0 || volume_face >= static_cast<GlobalIndex>(mesh.n_faces()),
                    FEException,
                    "DofHandler::distributeDofs(MeshBase, InterfaceMesh): interface face maps to invalid volume face");
        entity_dof_map_->setFaceDofs(volume_face, face_dofs);
    }

    entity_dof_map_->buildReverseMapping();
    entity_dof_map_->finalize();

    partition_ = DofPartition(0, total_dofs, {});
    partition_.setGlobalSize(total_dofs);
    ghost_manager_.reset();
}

void DofHandler::distributeDofs(const Mesh& mesh,
                                const svmp::InterfaceMesh& interface_mesh,
                                const spaces::FunctionSpace& space,
                                const DofDistributionOptions& options) {
    DofDistributionOptions opts = options;
    opts.my_rank = mesh.rank();
    opts.world_size = mesh.world_size();
#if FE_HAS_MPI && defined(MESH_HAS_MPI)
    opts.mpi_comm = mesh.mpi_comm();
#endif
    distributeDofs(mesh.local_mesh(), interface_mesh, space, opts);
}

void DofHandler::distributeDofsInternal(const MeshBase& mesh,
                                         const spaces::FunctionSpace& space,
                                         const DofDistributionOptions& options,
                                         bool /*is_distributed*/) {
    // Delegate to the public mesh-based API
    distributeDofs(mesh, space, options);
}

#else

// Stub implementations when Mesh library not available

void DofHandler::distributeDofs(const MeshBase& /*mesh*/,
                                const spaces::FunctionSpace& /*space*/,
                                const DofDistributionOptions& /*options*/) {
    throw FEException("DofHandler::distributeDofs: Mesh library not available. "
                      "Use distributeDofs(MeshTopologyInfo, DofLayoutInfo, options) instead.");
}

void DofHandler::distributeDofs(const Mesh& /*mesh*/,
                                const spaces::FunctionSpace& /*space*/,
                                const DofDistributionOptions& /*options*/) {
    throw FEException("DofHandler::distributeDofs: Mesh library not available. "
                      "Use distributeDofs(MeshTopologyInfo, DofLayoutInfo, options) instead.");
}

void DofHandler::distributeDofs(const MeshBase& /*mesh*/,
                                const svmp::InterfaceMesh& /*interface_mesh*/,
                                const spaces::FunctionSpace& /*space*/,
                                const DofDistributionOptions& /*options*/) {
    throw FEException("DofHandler::distributeDofs(MeshBase, InterfaceMesh): Mesh library not available.");
}

void DofHandler::distributeDofs(const Mesh& /*mesh*/,
                                const svmp::InterfaceMesh& /*interface_mesh*/,
                                const spaces::FunctionSpace& /*space*/,
                                const DofDistributionOptions& /*options*/) {
    throw FEException("DofHandler::distributeDofs(Mesh, InterfaceMesh): Mesh library not available.");
}

void DofHandler::distributeDofsInternal(const MeshBase& /*mesh*/,
                                         const spaces::FunctionSpace& /*space*/,
                                         const DofDistributionOptions& /*options*/,
                                         bool /*is_distributed*/) {
    throw FEException("DofHandler::distributeDofsInternal: Mesh library not available.");
}

#endif // DOFHANDLER_HAS_MESH

void DofHandler::setDofMap(DofMap dof_map) {
    checkNotFinalized();
    ++dof_state_revision_;
    dof_map_ = std::move(dof_map);
    clearSpatialDofCoordinates();
}

void DofHandler::setPartition(DofPartition partition) {
    checkNotFinalized();
    partition_ = std::move(partition);
    ghost_cache_valid_ = false;
}

void DofHandler::setEntityDofMap(std::unique_ptr<EntityDofMap> entity_dof_map) {
    checkNotFinalized();
    entity_dof_map_ = std::move(entity_dof_map);
    clearSpatialDofCoordinates();
}

void DofHandler::clearSpatialDofCoordinates() noexcept
{
    spatial_dof_coords_.clear();
    spatial_dof_coord_dim_ = 0;
}

void DofHandler::cacheSpatialDofCoordinates(const MeshTopologyView& topology,
                                            const DofLayoutInfo& layout)
{
    clearSpatialDofCoordinates();

    const GlobalIndex n_dofs = dof_map_.getNumDofs();
    if (n_dofs <= 0) {
        return;
    }

    const int dim = (topology.dim > 0) ? topology.dim : 3;
    const std::size_t dim_u = static_cast<std::size_t>(dim);
    const std::size_t required_coords =
        static_cast<std::size_t>(std::max<GlobalIndex>(topology.n_vertices, 0)) * dim_u;
    if (topology.vertex_coords.size() < required_coords || dim_u == 0u) {
        return;
    }

    spatial_dof_coord_dim_ = dim;
    spatial_dof_coords_.assign(static_cast<std::size_t>(n_dofs) * dim_u, 0.0);
    std::vector<std::uint8_t> assigned(static_cast<std::size_t>(n_dofs), 0u);

    const auto vertex_coords = [&](GlobalIndex v) -> std::array<double, 3> {
        std::array<double, 3> xyz{0.0, 0.0, 0.0};
        if (v < 0 || v >= topology.n_vertices) {
            return xyz;
        }
        const auto vv = static_cast<std::size_t>(v);
        const auto base = vv * dim_u;
        for (std::size_t d = 0; d < dim_u && d < 3u; ++d) {
            xyz[d] = static_cast<double>(topology.vertex_coords[base + d]);
        }
        return xyz;
    };

    const auto centroid = [&](std::span<const MeshIndex> verts) -> std::array<double, 3> {
        std::array<double, 3> xyz{0.0, 0.0, 0.0};
        if (verts.empty()) {
            return xyz;
        }
        for (const auto v : verts) {
            const auto p = vertex_coords(static_cast<GlobalIndex>(v));
            xyz[0] += p[0];
            xyz[1] += p[1];
            xyz[2] += p[2];
        }
        const auto inv = 1.0 / static_cast<double>(verts.size());
        xyz[0] *= inv;
        xyz[1] *= inv;
        xyz[2] *= inv;
        return xyz;
    };

    const auto assign = [&](GlobalIndex dof, const std::array<double, 3>& xyz) {
        if (dof < 0 || dof >= n_dofs) {
            return;
        }
        const auto did = static_cast<std::size_t>(dof);
        const auto base = did * dim_u;
        for (std::size_t d = 0; d < dim_u; ++d) {
            spatial_dof_coords_[base + d] = xyz[d];
        }
        assigned[did] = 1u;
    };

    if (entity_dof_map_) {
        for (GlobalIndex v = 0; v < entity_dof_map_->numVertices(); ++v) {
            const auto xyz = vertex_coords(v);
            for (const auto d : entity_dof_map_->getVertexDofs(v)) {
                assign(d, xyz);
            }
        }

        if (layout.dofs_per_edge > 0 && !topology.edge2vertex_data.empty()) {
            const GlobalIndex n_edges = std::min<GlobalIndex>(entity_dof_map_->numEdges(),
                                                              static_cast<GlobalIndex>(topology.edge2vertex_data.size() / 2u));
            for (GlobalIndex e = 0; e < n_edges; ++e) {
                const auto [v0, v1] = topology.getEdgeVertices(e);
                if (v0 < 0 || v1 < 0) {
                    continue;
                }
                const auto p0 = vertex_coords(static_cast<GlobalIndex>(v0));
                const auto p1 = vertex_coords(static_cast<GlobalIndex>(v1));
                const std::array<double, 3> xyz{
                    0.5 * (p0[0] + p1[0]),
                    0.5 * (p0[1] + p1[1]),
                    0.5 * (p0[2] + p1[2]),
                };
                for (const auto d : entity_dof_map_->getEdgeDofs(e)) {
                    assign(d, xyz);
                }
            }
        }

        if (layout.has_face_dofs() &&
            !topology.face2vertex_offsets.empty() &&
            !topology.face2vertex_data.empty()) {
            const GlobalIndex n_faces = std::min<GlobalIndex>(entity_dof_map_->numFaces(), topology.n_faces);
            for (GlobalIndex f = 0; f < n_faces; ++f) {
                const auto xyz = centroid(topology.getFaceVertices(f));
                for (const auto d : entity_dof_map_->getFaceDofs(f)) {
                    assign(d, xyz);
                }
            }
        }

        if (layout.dofs_per_cell > 0) {
            const GlobalIndex n_cells = std::min<GlobalIndex>(entity_dof_map_->numCells(), topology.n_cells);
            for (GlobalIndex c = 0; c < n_cells; ++c) {
                const auto xyz = centroid(topology.getCellVertices(c));
                for (const auto d : entity_dof_map_->getCellInteriorDofs(c)) {
                    assign(d, xyz);
                }
            }
        }
    }

    // Fallback assignment for any unassigned DOFs (e.g., DG layouts).
    for (GlobalIndex c = 0; c < topology.n_cells; ++c) {
        const auto xyz = centroid(topology.getCellVertices(c));
        for (const auto d : dof_map_.getCellDofs(c)) {
            if (d < 0 || d >= n_dofs) {
                continue;
            }
            if (assigned[static_cast<std::size_t>(d)] != 0u) {
                continue;
            }
            assign(d, xyz);
        }
    }

    // Final deterministic fallback when topology data is incomplete.
    for (GlobalIndex d = 0; d < n_dofs; ++d) {
        if (assigned[static_cast<std::size_t>(d)] != 0u) {
            continue;
        }
        const std::array<double, 3> xyz{
            static_cast<double>(d),
            static_cast<double>(d),
            static_cast<double>(d),
        };
        assign(d, xyz);
    }
}

void DofHandler::renumberDofs(DofNumberingStrategy strategy) {
    checkNotFinalized();
    const GlobalIndex n_dofs = dof_map_.getNumDofs();
    if (n_dofs <= 0) {
        ghost_cache_valid_ = false;
        return;
    }

    std::vector<GlobalIndex> perm(static_cast<std::size_t>(n_dofs));
    std::iota(perm.begin(), perm.end(), GlobalIndex{0});

    auto rebuild_entity_map = [&](std::span<const GlobalIndex> numbering) {
        if (!entity_dof_map_) {
            return;
        }

        const auto& old_map = *entity_dof_map_;
        auto new_map = std::make_unique<EntityDofMap>();
        new_map->reserve(old_map.numVertices(), old_map.numEdges(), old_map.numFaces(), old_map.numCells());

        auto remap_span = [&](std::span<const GlobalIndex> dofs) -> std::vector<GlobalIndex> {
            std::vector<GlobalIndex> out;
            out.reserve(dofs.size());
            for (auto d : dofs) {
                if (d < 0 || d >= n_dofs) {
                    throw FEException("DofHandler::renumberDofs: entity map DOF out of range");
                }
                out.push_back(numbering[static_cast<std::size_t>(d)]);
            }
            return out;
        };

        for (GlobalIndex v = 0; v < old_map.numVertices(); ++v) {
            auto vdofs = remap_span(old_map.getVertexDofs(v));
            new_map->setVertexDofs(v, vdofs);
        }
        for (GlobalIndex e = 0; e < old_map.numEdges(); ++e) {
            auto edofs = remap_span(old_map.getEdgeDofs(e));
            new_map->setEdgeDofs(e, edofs);
        }
        for (GlobalIndex f = 0; f < old_map.numFaces(); ++f) {
            auto fdofs = remap_span(old_map.getFaceDofs(f));
            new_map->setFaceDofs(f, fdofs);
        }
        for (GlobalIndex c = 0; c < old_map.numCells(); ++c) {
            auto cdofs = remap_span(old_map.getCellInteriorDofs(c));
            new_map->setCellInteriorDofs(c, cdofs);
        }

        new_map->buildReverseMapping();
        new_map->finalize();
        entity_dof_map_ = std::move(new_map);
    };

    switch (strategy) {
        case DofNumberingStrategy::Sequential:
            break; // identity

        case DofNumberingStrategy::Interleaved: {
            if (num_components_ <= 1) {
                break;
            }
            const auto nc = static_cast<GlobalIndex>(num_components_);
            if (nc <= 0 || (n_dofs % nc) != 0) {
                throw FEException("DofHandler::renumberDofs: Interleaved numbering requires n_dofs divisible by num_components");
            }
            const InterleavedNumbering strat(num_components_);
            perm = strat.computeNumbering(n_dofs, {}, {});
            break;
        }

        case DofNumberingStrategy::Block: {
            if (num_components_ <= 1) {
                break;
            }
            const auto nc = static_cast<GlobalIndex>(num_components_);
            if (nc <= 0 || (n_dofs % nc) != 0) {
                throw FEException("DofHandler::renumberDofs: Block numbering requires n_dofs divisible by num_components");
            }
            const GlobalIndex block = n_dofs / nc;
            std::vector<GlobalIndex> sizes(static_cast<std::size_t>(nc), block);
            const BlockNumbering strat(std::move(sizes));
            perm = strat.computeNumbering(n_dofs, {}, {});
            break;
        }

        case DofNumberingStrategy::Hierarchical: {
            if (!entity_dof_map_) {
                throw FEException("DofHandler::renumberDofs: Hierarchical numbering requires an EntityDofMap");
            }

            std::vector<GlobalIndex> ordering;
            ordering.reserve(static_cast<std::size_t>(n_dofs));

            for (GlobalIndex v = 0; v < entity_dof_map_->numVertices(); ++v) {
                auto dofs = entity_dof_map_->getVertexDofs(v);
                ordering.insert(ordering.end(), dofs.begin(), dofs.end());
            }
            for (GlobalIndex e = 0; e < entity_dof_map_->numEdges(); ++e) {
                auto dofs = entity_dof_map_->getEdgeDofs(e);
                ordering.insert(ordering.end(), dofs.begin(), dofs.end());
            }
            for (GlobalIndex f = 0; f < entity_dof_map_->numFaces(); ++f) {
                auto dofs = entity_dof_map_->getFaceDofs(f);
                ordering.insert(ordering.end(), dofs.begin(), dofs.end());
            }
            for (GlobalIndex c = 0; c < entity_dof_map_->numCells(); ++c) {
                auto dofs = entity_dof_map_->getCellInteriorDofs(c);
                ordering.insert(ordering.end(), dofs.begin(), dofs.end());
            }

            if (ordering.size() != static_cast<std::size_t>(n_dofs)) {
                throw FEException("DofHandler::renumberDofs: EntityDofMap does not cover all DOFs for hierarchical renumbering");
            }

            perm.assign(static_cast<std::size_t>(n_dofs), GlobalIndex{-1});
            GlobalIndex next = 0;
            for (auto old_dof : ordering) {
                if (old_dof < 0 || old_dof >= n_dofs) {
                    throw FEException("DofHandler::renumberDofs: DOF out of range in hierarchical ordering");
                }
                perm[static_cast<std::size_t>(old_dof)] = next++;
            }

            for (auto& v : perm) {
                if (v < 0) {
                    v = next++;
                }
            }
            break;
        }

        case DofNumberingStrategy::Morton:
        case DofNumberingStrategy::Hilbert: {
            if (world_size_ > 1) {
                FE_THROW_IF(global_numbering_ != GlobalNumberingMode::OwnerContiguous,
                            FEException,
                            "DofHandler::renumberDofs: MPI spatial renumbering requires OwnerContiguous global numbering");
#if !FE_HAS_MPI
                throw FEException("DofHandler::renumberDofs: MPI spatial renumbering requires FE_HAS_MPI");
#else
                const auto old_owned = partition_.locallyOwned().toVector();
                std::unordered_map<GlobalIndex, GlobalIndex> owned_old_to_new;
                owned_old_to_new.reserve(old_owned.size());

                const auto owned_count_local = static_cast<std::int64_t>(old_owned.size());
                std::vector<std::int64_t> owned_counts(static_cast<std::size_t>(world_size_), 0);
                fe_mpi_check(MPI_Allgather(&owned_count_local, 1, MPI_INT64_T,
                                           owned_counts.data(), 1, MPI_INT64_T,
                                           mpi_comm_),
                             "MPI_Allgather (owned counts) in DofHandler::renumberDofs");

                GlobalIndex owned_begin = 0;
                for (int r = 0; r < my_rank_; ++r) {
                    FE_THROW_IF(owned_counts[static_cast<std::size_t>(r)] < 0,
                                FEException,
                                "DofHandler::renumberDofs: negative owned count gathered from rank");
                    owned_begin += static_cast<GlobalIndex>(owned_counts[static_cast<std::size_t>(r)]);
                }
                const GlobalIndex owned_end = owned_begin + static_cast<GlobalIndex>(owned_count_local);

                if (!old_owned.empty()) {
                    FE_THROW_IF(owned_end < owned_begin,
                                FEException,
                                "DofHandler::renumberDofs: invalid owned contiguous range");
                    FE_THROW_IF(owned_end - owned_begin != static_cast<GlobalIndex>(old_owned.size()),
                                FEException,
                                "DofHandler::renumberDofs: owned contiguous range size mismatch");

                    const int curve_dim = std::max(1, std::min(3, spatial_dof_coord_dim_ > 0 ? spatial_dof_coord_dim_ : 3));
                    std::vector<std::array<double, 3>> raw_coords(old_owned.size(), {0.0, 0.0, 0.0});
                    std::array<double, 3> min_xyz{
                        std::numeric_limits<double>::max(),
                        std::numeric_limits<double>::max(),
                        std::numeric_limits<double>::max(),
                    };
                    std::array<double, 3> max_xyz{
                        std::numeric_limits<double>::lowest(),
                        std::numeric_limits<double>::lowest(),
                        std::numeric_limits<double>::lowest(),
                    };

                    const bool has_coords =
                        spatial_dof_coord_dim_ > 0 &&
                        !spatial_dof_coords_.empty() &&
                        spatial_dof_coords_.size() >=
                            static_cast<std::size_t>(n_dofs) * static_cast<std::size_t>(spatial_dof_coord_dim_);

                    for (std::size_t idx = 0; idx < old_owned.size(); ++idx) {
                        const auto old_dof = old_owned[idx];

                        std::array<double, 3> xyz{
                            static_cast<double>(old_dof),
                            static_cast<double>(old_dof),
                            static_cast<double>(old_dof),
                        };
                        if (has_coords && old_dof >= 0) {
                            const auto base = static_cast<std::size_t>(old_dof) *
                                              static_cast<std::size_t>(spatial_dof_coord_dim_);
                            for (int d = 0; d < curve_dim; ++d) {
                                xyz[static_cast<std::size_t>(d)] =
                                    spatial_dof_coords_[base + static_cast<std::size_t>(d)];
                            }
                        }

                        raw_coords[idx] = xyz;
                        for (int d = 0; d < curve_dim; ++d) {
                            min_xyz[static_cast<std::size_t>(d)] =
                                std::min(min_xyz[static_cast<std::size_t>(d)], xyz[static_cast<std::size_t>(d)]);
                            max_xyz[static_cast<std::size_t>(d)] =
                                std::max(max_xyz[static_cast<std::size_t>(d)], xyz[static_cast<std::size_t>(d)]);
                        }
                    }

                    std::vector<double> normalized_coords;
                    normalized_coords.resize(old_owned.size() * static_cast<std::size_t>(curve_dim), 0.0);
                    for (std::size_t idx = 0; idx < old_owned.size(); ++idx) {
                        for (int d = 0; d < curve_dim; ++d) {
                            const double lo = min_xyz[static_cast<std::size_t>(d)];
                            const double hi = max_xyz[static_cast<std::size_t>(d)];
                            double u = 0.0;
                            if (hi > lo) {
                                u = (raw_coords[idx][static_cast<std::size_t>(d)] - lo) / (hi - lo);
                            }
                            u = std::max(0.0, std::min(1.0, u));
                            normalized_coords[idx * static_cast<std::size_t>(curve_dim) + static_cast<std::size_t>(d)] = u;
                        }
                    }

                    SpaceFillingCurveNumbering local_strategy(
                        (strategy == DofNumberingStrategy::Hilbert)
                            ? SpaceFillingCurveNumbering::CurveType::Hilbert
                            : SpaceFillingCurveNumbering::CurveType::Morton,
                        curve_dim);
                    local_strategy.setCoordinates(normalized_coords, curve_dim);
                    const auto local_perm =
                        local_strategy.computeNumbering(static_cast<GlobalIndex>(old_owned.size()), {}, {});

                    for (std::size_t local_old = 0; local_old < old_owned.size(); ++local_old) {
                        const auto old_dof = old_owned[local_old];
                        const auto local_new = local_perm[local_old];
                        FE_THROW_IF(local_new < 0 ||
                                        local_new >= static_cast<GlobalIndex>(old_owned.size()),
                                    FEException,
                                    "DofHandler::renumberDofs: invalid local spatial permutation entry");
                        const auto new_dof = owned_begin + local_new;
                        perm[static_cast<std::size_t>(old_dof)] = new_dof;
                        owned_old_to_new.emplace(old_dof, new_dof);
                    }
                }

                const auto old_ghost = partition_.ghost().toVector();
                {
                    std::vector<int> neighbors;
                    neighbors.reserve(static_cast<std::size_t>(std::max(0, world_size_ - 1)));
                    for (int r = 0; r < world_size_; ++r) {
                        if (r != my_rank_) {
                            neighbors.push_back(r);
                        }
                    }

                    std::vector<int> rank_to_neighbor(static_cast<std::size_t>(world_size_), -1);
                    for (std::size_t i = 0; i < neighbors.size(); ++i) {
                        rank_to_neighbor[static_cast<std::size_t>(neighbors[i])] = static_cast<int>(i);
                    }

                    std::vector<std::vector<GlobalIndex>> request_send(neighbors.size());
                    for (auto old_dof : old_ghost) {
                        FE_THROW_IF(old_dof < 0 || old_dof >= n_dofs,
                                    FEException,
                                    "DofHandler::renumberDofs: ghost DOF out of range");
                        perm[static_cast<std::size_t>(old_dof)] = INVALID_GLOBAL_INDEX;
                    }
                    for (auto old_dof : old_ghost) {
                        int owner = -1;
                        if (ghost_manager_) {
                            owner = ghost_manager_->getDofOwner(old_dof);
                        }
                        if (owner < 0 || owner >= world_size_) {
                            owner = dof_map_.getDofOwner(old_dof);
                        }

                        if (owner == my_rank_) {
                            const auto it = owned_old_to_new.find(old_dof);
                            if (it != owned_old_to_new.end()) {
                                perm[static_cast<std::size_t>(old_dof)] = it->second;
                            }
                            continue;
                        }
                        if (owner < 0 || owner >= world_size_) {
                            continue;
                        }

                        const int ni = rank_to_neighbor[static_cast<std::size_t>(owner)];
                        if (ni < 0) {
                            continue;
                        }
                        request_send[static_cast<std::size_t>(ni)].push_back(old_dof);
                    }

                    std::vector<std::vector<GlobalIndex>> request_recv;
                    mpi_neighbor_exchange_bytes(mpi_comm_,
                                                std::span<const int>(neighbors.data(), neighbors.size()),
                                                std::span<const std::vector<GlobalIndex>>(request_send.data(), request_send.size()),
                                                request_recv,
                                                /*tag_counts=*/43910,
                                                /*tag_data=*/43911);

                    std::vector<std::vector<GlobalIndex>> response_send(neighbors.size());
                    for (std::size_t ni = 0; ni < neighbors.size(); ++ni) {
                        const auto& incoming = request_recv[ni];
                        auto& outgoing = response_send[ni];
                        outgoing.reserve(incoming.size());
                        for (auto old_dof : incoming) {
                            const auto it = owned_old_to_new.find(old_dof);
                            if (it == owned_old_to_new.end()) {
                                // Defer the hard failure to the requester side so all ranks still
                                // complete the response exchange (avoids MPI deadlocks on throw).
                                outgoing.push_back(INVALID_GLOBAL_INDEX);
                            } else {
                                outgoing.push_back(it->second);
                            }
                        }
                    }

                    std::vector<std::vector<GlobalIndex>> response_recv;
                    mpi_neighbor_exchange_bytes(mpi_comm_,
                                                std::span<const int>(neighbors.data(), neighbors.size()),
                                                std::span<const std::vector<GlobalIndex>>(response_send.data(), response_send.size()),
                                                response_recv,
                                                /*tag_counts=*/43912,
                                                /*tag_data=*/43913);

                    for (std::size_t ni = 0; ni < neighbors.size(); ++ni) {
                        const auto& requested = request_send[ni];
                        const auto& received = response_recv[ni];
                        FE_THROW_IF(received.size() != requested.size(),
                                    FEException,
                                    "DofHandler::renumberDofs: response size mismatch in ghost ID remap");
                        for (std::size_t k = 0; k < requested.size(); ++k) {
                            const auto old_dof = requested[k];
                            const auto new_dof = received[k];
                            FE_THROW_IF(old_dof < 0 || old_dof >= n_dofs,
                                        FEException,
                                        "DofHandler::renumberDofs: ghost request DOF out of range");
                            FE_THROW_IF(new_dof == INVALID_GLOBAL_INDEX,
                                        FEException,
                                        "DofHandler::renumberDofs: owner failed to resolve ghost DOF remap request");
                            perm[static_cast<std::size_t>(old_dof)] = new_dof;
                        }
                    }

                    for (auto old_dof : old_ghost) {
                        FE_THROW_IF(perm[static_cast<std::size_t>(old_dof)] == INVALID_GLOBAL_INDEX,
                                    FEException,
                                    "DofHandler::renumberDofs: failed to resolve ghost DOF remap");
                    }
                }

                break;
#endif
            }

            SpaceFillingCurveNumbering strategy_impl(
                (strategy == DofNumberingStrategy::Hilbert)
                    ? SpaceFillingCurveNumbering::CurveType::Hilbert
                    : SpaceFillingCurveNumbering::CurveType::Morton,
                spatial_dof_coord_dim_ > 0 ? spatial_dof_coord_dim_ : 3);

            const std::size_t required_coords =
                static_cast<std::size_t>(n_dofs) *
                static_cast<std::size_t>(std::max(1, spatial_dof_coord_dim_));
            if (!spatial_dof_coords_.empty() && spatial_dof_coords_.size() >= required_coords) {
                strategy_impl.setCoordinates(std::span<const double>(spatial_dof_coords_.data(),
                                                                     required_coords),
                                             std::max(1, spatial_dof_coord_dim_));
            }
            perm = strategy_impl.computeNumbering(n_dofs, {}, {});
            break;
        }

        case DofNumberingStrategy::CuthillMcKee: {
            // For multi-component interleaved DOFs (e.g., vel_x,vel_y,vel_z,pres per node),
            // operate at the NODE level so that all components of each node stay together.
            // This preserves the interleaved layout that the FSILS backend requires for
            // efficient SpMV access patterns.
            const auto nc = static_cast<GlobalIndex>(num_components_ > 1 ? num_components_ : 1);
            if (nc > 1 && (n_dofs % nc) == 0) {
                // Node-level RCM: build adjacency on nodes, then expand to DOFs
                const auto n_nodes = n_dofs / nc;
                const auto nn = static_cast<std::size_t>(n_nodes);
                std::vector<std::vector<GlobalIndex>> node_adj_lists(nn);

                const GlobalIndex n_cells = dof_map_.getNumCells();
                for (GlobalIndex c = 0; c < n_cells; ++c) {
                    const auto cell_dofs = dof_map_.getCellDofs(c);
                    // Extract unique node indices from cell DOFs
                    std::vector<GlobalIndex> cell_nodes;
                    cell_nodes.reserve(cell_dofs.size() / static_cast<std::size_t>(nc));
                    for (const auto d : cell_dofs) {
                        if (d < 0 || d >= n_dofs) continue;
                        const auto node = d / nc;
                        if (cell_nodes.empty() || cell_nodes.back() != node) {
                            // Check if already in list (handles non-sequential DOF ordering)
                            bool found = false;
                            for (const auto n : cell_nodes) {
                                if (n == node) { found = true; break; }
                            }
                            if (!found) cell_nodes.push_back(node);
                        }
                    }
                    // Record node adjacency
                    for (std::size_t i = 0; i < cell_nodes.size(); ++i) {
                        for (std::size_t j = i + 1; j < cell_nodes.size(); ++j) {
                            const auto ni = cell_nodes[i];
                            const auto nj = cell_nodes[j];
                            node_adj_lists[static_cast<std::size_t>(ni)].push_back(nj);
                            node_adj_lists[static_cast<std::size_t>(nj)].push_back(ni);
                        }
                    }
                }

                // Deduplicate and convert to CSR
                std::vector<GlobalIndex> adjacency_csr(nn + 1, 0);
                std::vector<GlobalIndex> adj_indices;
                for (std::size_t i = 0; i < nn; ++i) {
                    auto& neighbors = node_adj_lists[i];
                    std::sort(neighbors.begin(), neighbors.end());
                    neighbors.erase(std::unique(neighbors.begin(), neighbors.end()),
                                    neighbors.end());
                    adjacency_csr[i + 1] = adjacency_csr[i]
                                           + static_cast<GlobalIndex>(neighbors.size());
                    adj_indices.insert(adj_indices.end(), neighbors.begin(), neighbors.end());
                }

                CuthillMcKeeNumbering rcm(/*reverse=*/true);
                auto node_perm = rcm.computeNumbering(n_nodes, adjacency_csr, adj_indices);

                // Expand node permutation to DOF permutation preserving interleaving:
                // node old_node -> new_node: DOFs old_node*nc+k -> new_node*nc+k
                perm.resize(static_cast<std::size_t>(n_dofs));
                for (GlobalIndex old_node = 0; old_node < n_nodes; ++old_node) {
                    const auto new_node = node_perm[static_cast<std::size_t>(old_node)];
                    for (GlobalIndex k = 0; k < nc; ++k) {
                        perm[static_cast<std::size_t>(old_node * nc + k)] = new_node * nc + k;
                    }
                }
            } else {
                // Single-component: operate directly on DOFs
                const auto n = static_cast<std::size_t>(n_dofs);
                std::vector<std::vector<GlobalIndex>> adj_lists(n);

                const GlobalIndex n_cells = dof_map_.getNumCells();
                for (GlobalIndex c = 0; c < n_cells; ++c) {
                    const auto cell_dofs = dof_map_.getCellDofs(c);
                    for (std::size_t i = 0; i < cell_dofs.size(); ++i) {
                        const auto di = cell_dofs[i];
                        if (di < 0 || di >= n_dofs) continue;
                        for (std::size_t j = i + 1; j < cell_dofs.size(); ++j) {
                            const auto dj = cell_dofs[j];
                            if (dj < 0 || dj >= n_dofs) continue;
                            adj_lists[static_cast<std::size_t>(di)].push_back(dj);
                            adj_lists[static_cast<std::size_t>(dj)].push_back(di);
                        }
                    }
                }

                std::vector<GlobalIndex> adjacency_csr(n + 1, 0);
                std::vector<GlobalIndex> adj_indices;
                for (std::size_t i = 0; i < n; ++i) {
                    auto& neighbors = adj_lists[i];
                    std::sort(neighbors.begin(), neighbors.end());
                    neighbors.erase(std::unique(neighbors.begin(), neighbors.end()),
                                    neighbors.end());
                    adjacency_csr[i + 1] = adjacency_csr[i]
                                           + static_cast<GlobalIndex>(neighbors.size());
                    adj_indices.insert(adj_indices.end(), neighbors.begin(), neighbors.end());
                }

                CuthillMcKeeNumbering rcm(/*reverse=*/true);
                perm = rcm.computeNumbering(n_dofs, adjacency_csr, adj_indices);
            }
            break;
        }

        default:
            throw FEException("DofHandler::renumberDofs: unsupported strategy");
    }

    bool identity = true;
    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        if (perm[static_cast<std::size_t>(i)] != i) {
            identity = false;
            break;
        }
    }
    if (identity) {
        ghost_cache_valid_ = false;
        return;
    }

    ++dof_state_revision_;

    // In parallel, ownership/partitioning must be permuted along with the DOF IDs.
    // We precompute owner information for the permuted locally relevant set so
    // that we can rebuild stable owned/ghost sets and a backend-friendly owner map.
    std::unordered_map<GlobalIndex, int> owner_by_new_dof;
    std::vector<GlobalIndex> new_owned;
    std::vector<GlobalIndex> new_ghost;
    std::vector<int> new_ghost_owners;

    if (world_size_ > 1) {
        const auto relevant_old = partition_.locallyRelevant().toVector();
        owner_by_new_dof.reserve(relevant_old.size());

        for (auto old_dof : relevant_old) {
            if (old_dof < 0 || old_dof >= n_dofs) {
                throw FEException("DofHandler::renumberDofs: relevant DOF out of range before renumbering");
            }
            const int owner = dof_map_.getDofOwner(old_dof);
            const auto new_dof = perm[static_cast<std::size_t>(old_dof)];
            const auto [it, inserted] = owner_by_new_dof.emplace(new_dof, owner);
            if (!inserted && it->second != owner) {
                throw FEException("DofHandler::renumberDofs: inconsistent owner mapping while renumbering");
            }
        }

        const auto owned_old = partition_.locallyOwned().toVector();
        new_owned.reserve(owned_old.size());
        for (auto old_dof : owned_old) {
            new_owned.push_back(perm[static_cast<std::size_t>(old_dof)]);
        }
        std::sort(new_owned.begin(), new_owned.end());
        new_owned.erase(std::unique(new_owned.begin(), new_owned.end()), new_owned.end());

        const auto ghost_old = partition_.ghost().toVector();
        new_ghost.reserve(ghost_old.size());
        new_ghost_owners.reserve(ghost_old.size());
        for (auto old_dof : ghost_old) {
            const auto new_dof = perm[static_cast<std::size_t>(old_dof)];
            new_ghost.push_back(new_dof);
            const auto it = owner_by_new_dof.find(new_dof);
            new_ghost_owners.push_back(it != owner_by_new_dof.end() ? it->second : dof_map_.getDofOwner(old_dof));
        }

        // Sort ghosts and keep owners aligned.
        std::vector<std::size_t> ghost_perm(new_ghost.size());
        std::iota(ghost_perm.begin(), ghost_perm.end(), std::size_t{0});
        std::sort(ghost_perm.begin(), ghost_perm.end(), [&](std::size_t a, std::size_t b) {
            return new_ghost[a] < new_ghost[b];
        });
        std::vector<GlobalIndex> ghost_sorted;
        std::vector<int> owners_sorted;
        ghost_sorted.reserve(new_ghost.size());
        owners_sorted.reserve(new_ghost.size());
        for (auto idx : ghost_perm) {
            ghost_sorted.push_back(new_ghost[idx]);
            owners_sorted.push_back(new_ghost_owners[idx]);
        }
        new_ghost = std::move(ghost_sorted);
        new_ghost_owners = std::move(owners_sorted);
    }

    applyNumbering(dof_map_, perm);
    dof_map_.setMyRank(my_rank_);
    rebuild_entity_map(perm);

    if (world_size_ > 1) {
        // Rebuild partition sets after renumbering.
        dof_map_.setNumLocalDofs(static_cast<GlobalIndex>(new_owned.size()));
        partition_ = DofPartition(IndexSet(std::move(new_owned)), IndexSet(new_ghost));
        partition_.setGlobalSize(n_dofs);

        // Rebuild ownership lookup.
        // For OwnerContiguous global numbering we can recover owner rank for any global DOF
        // from the per-rank owned-range prefix sums, which keeps ownership queries valid even
        // for DOFs that are not locally relevant on this rank.
        if (global_numbering_ == GlobalNumberingMode::OwnerContiguous) {
            std::vector<std::int64_t> owned_counts(static_cast<std::size_t>(world_size_), 0);
            const auto owned_count_local = static_cast<std::int64_t>(partition_.localOwnedSize());
#if FE_HAS_MPI
            fe_mpi_check(MPI_Allgather(&owned_count_local, 1, MPI_INT64_T,
                                       owned_counts.data(), 1, MPI_INT64_T,
                                       mpi_comm_),
                         "MPI_Allgather (owned counts) in DofHandler::renumberDofs ownership rebuild");
#else
            owned_counts[0] = owned_count_local;
#endif

            auto owned_prefix = std::make_shared<std::vector<GlobalIndex>>(
                static_cast<std::size_t>(world_size_) + 1u, GlobalIndex{0});
            for (int r = 0; r < world_size_; ++r) {
                FE_THROW_IF(owned_counts[static_cast<std::size_t>(r)] < 0,
                            FEException,
                            "DofHandler::renumberDofs: negative owned count gathered while rebuilding ownership");
                (*owned_prefix)[static_cast<std::size_t>(r) + 1u] =
                    (*owned_prefix)[static_cast<std::size_t>(r)] +
                    static_cast<GlobalIndex>(owned_counts[static_cast<std::size_t>(r)]);
            }

            dof_map_.setDofOwnership([owned_prefix](GlobalIndex dof) -> int {
                if (dof < 0 || dof >= owned_prefix->back()) {
                    return -1;
                }
                const auto it = std::upper_bound(owned_prefix->begin() + 1, owned_prefix->end(), dof);
                return static_cast<int>((it - owned_prefix->begin()) - 1);
            });
        } else {
            // Fallback to locally relevant ownership info when no global owner-contiguous
            // ranges are available.
            auto owner_map_ptr = std::make_shared<std::unordered_map<GlobalIndex, int>>(std::move(owner_by_new_dof));
            dof_map_.setDofOwnership([owner_map_ptr](GlobalIndex dof) -> int {
                const auto it = owner_map_ptr->find(dof);
                return (it != owner_map_ptr->end()) ? it->second : -1;
            });
        }

        if (global_numbering_ == GlobalNumberingMode::OwnerContiguous) {
            for (std::size_t i = 0; i < new_ghost.size(); ++i) {
                new_ghost_owners[i] = dof_map_.getDofOwner(new_ghost[i]);
            }
        }

        // Reset ghost manager (scatter contexts must be rebuilt after renumbering).
        if (ghost_manager_) {
            ghost_manager_->setMyRank(my_rank_);
            ghost_manager_->setGhostDofs(new_ghost, new_ghost_owners);
            ghost_manager_->setOwnedDofs(partition_.locallyOwned().toVector());
        }
    }

    ghost_cache_valid_ = false;
}

void DofHandler::finalize() {
    checkNotFinalized();

    // Finalize the DOF map
    dof_map_.finalize();

    finalized_ = true;
}

// =============================================================================
// Query Methods
// =============================================================================

DofMap& DofHandler::getDofMapMutable() {
    checkNotFinalized();
    return dof_map_;
}

std::optional<std::pair<GlobalIndex, GlobalIndex>>
DofHandler::getLocalDofRange() const noexcept {
    auto range = partition_.locallyOwned().contiguousRange();
    if (range) {
        return std::make_pair(range->begin, range->end);
    }
    return std::nullopt;
}

std::span<const GlobalIndex> DofHandler::getGhostDofs() const {
    if (!ghost_cache_valid_) {
        ghost_dofs_cache_ = partition_.ghost().toVector();
        ghost_cache_valid_ = true;
    }
    return ghost_dofs_cache_;
}

	DofHandler::Statistics DofHandler::getStatistics() const {
	    Statistics stats;
    stats.total_dofs = dof_map_.getNumDofs();
    stats.local_owned_dofs = dof_map_.getNumLocalDofs();
    stats.ghost_dofs = partition_.ghostSize();

    if (n_cells_ > 0) {
        GlobalIndex min_dofs = std::numeric_limits<GlobalIndex>::max();
        GlobalIndex max_dofs = 0;
        GlobalIndex total_cell_dofs = 0;

        for (GlobalIndex c = 0; c < n_cells_; ++c) {
            auto n = dof_map_.getNumCellDofs(c);
            min_dofs = std::min(min_dofs, static_cast<GlobalIndex>(n));
            max_dofs = std::max(max_dofs, static_cast<GlobalIndex>(n));
            total_cell_dofs += n;
        }

        stats.min_dofs_per_cell = min_dofs;
        stats.max_dofs_per_cell = max_dofs;
        stats.avg_dofs_per_cell = static_cast<double>(total_cell_dofs) / n_cells_;
    }

	    return stats;
	}

		#if FE_HAS_MPI
		struct DofHandler::GhostExchangeContextMPI {
		    MPI_Comm comm{MPI_COMM_NULL};
		    MPI_Comm neighbor_comm{MPI_COMM_NULL};
		    int tag{43010};
		    std::vector<int> neighbors;
		    std::vector<int> send_counts;
		    std::vector<int> recv_counts;
		    std::vector<int> send_displs;
		    std::vector<int> recv_displs;
		    std::vector<std::size_t> send_offsets;
		    std::vector<std::size_t> recv_offsets;
		    std::vector<std::vector<std::size_t>> send_indices;
		    std::vector<std::vector<std::size_t>> recv_indices;
		    std::vector<double> send_buffer;
		    std::vector<double> recv_buffer;
		    std::vector<MPI_Request> persistent_reqs;
		    bool persistent_ready{false};

		    ~GhostExchangeContextMPI() {
		        int finalized = 0;
		        MPI_Finalized(&finalized);
		        if (finalized) return;
		        if (persistent_ready) {
		            for (auto& req : persistent_reqs) {
		                if (req != MPI_REQUEST_NULL) {
		                    MPI_Request_free(&req);
		                    req = MPI_REQUEST_NULL;
		                }
		            }
		        }
		        if (neighbor_comm != MPI_COMM_NULL && neighbor_comm != MPI_COMM_WORLD && neighbor_comm != MPI_COMM_SELF) {
		            MPI_Comm_free(&neighbor_comm);
		            neighbor_comm = MPI_COMM_NULL;
		        }
		    }

		    void clearPersistent() {
		        if (!persistent_ready) return;
		        int finalized = 0;
		        MPI_Finalized(&finalized);
		        if (!finalized) {
		            for (auto& req : persistent_reqs) {
	                if (req != MPI_REQUEST_NULL) {
	                    MPI_Request_free(&req);
	                    req = MPI_REQUEST_NULL;
	                }
	            }
	        }
		        persistent_reqs.clear();
		        persistent_ready = false;
		    }

		    void buildPersistent() {
		        if (persistent_ready) return;

#if MPI_VERSION >= 4
		        if (neighbor_comm != MPI_COMM_NULL) {
		            persistent_reqs.assign(1u, MPI_REQUEST_NULL);
		            const void* send_ptr = send_buffer.empty() ? nullptr : static_cast<const void*>(send_buffer.data());
		            void* recv_ptr = recv_buffer.empty() ? nullptr : static_cast<void*>(recv_buffer.data());
		            const int* send_counts_ptr = send_counts.empty() ? nullptr : send_counts.data();
		            const int* send_displs_ptr = send_displs.empty() ? nullptr : send_displs.data();
		            const int* recv_counts_ptr = recv_counts.empty() ? nullptr : recv_counts.data();
		            const int* recv_displs_ptr = recv_displs.empty() ? nullptr : recv_displs.data();
		            fe_mpi_check(MPI_Neighbor_alltoallv_init(send_ptr,
		                                                   send_counts_ptr,
		                                                   send_displs_ptr,
		                                                   MPI_DOUBLE,
		                                                   recv_ptr,
		                                                   recv_counts_ptr,
		                                                   recv_displs_ptr,
		                                                   MPI_DOUBLE,
		                                                   neighbor_comm,
		                                                   MPI_INFO_NULL,
		                                                   &persistent_reqs[0]),
		                         "MPI_Neighbor_alltoallv_init in GhostExchangeContextMPI::buildPersistent");
		            persistent_ready = true;
		            return;
		        }
#endif

		        persistent_reqs.assign(neighbors.size() * 2u, MPI_REQUEST_NULL);
		        for (std::size_t i = 0; i < neighbors.size(); ++i) {
		            const int peer = neighbors[i];
		            const int recv_n = recv_counts[i];
		            const int send_n = send_counts[i];
		            double* recv_ptr = recv_buffer.empty() ? nullptr : (recv_buffer.data() + recv_offsets[i]);
		            double* send_ptr = send_buffer.empty() ? nullptr : (send_buffer.data() + send_offsets[i]);
		            fe_mpi_check(MPI_Recv_init(recv_ptr,
		                                      recv_n,
		                                      MPI_DOUBLE,
		                                      peer,
		                                      tag,
		                                      comm,
		                                      &persistent_reqs[i]),
		                         "MPI_Recv_init in GhostExchangeContextMPI::buildPersistent");
		            fe_mpi_check(MPI_Send_init(send_ptr,
		                                      send_n,
		                                      MPI_DOUBLE,
		                                      peer,
		                                      tag,
		                                      comm,
		                                      &persistent_reqs[neighbors.size() + i]),
		                         "MPI_Send_init in GhostExchangeContextMPI::buildPersistent");
		        }
		        persistent_ready = true;
		    }
		};
		#endif

	// =============================================================================
	// Parallel Support
	// =============================================================================

	void DofHandler::buildScatterContexts() {
    if (!finalized_) {
        throw FEException("DofHandler::buildScatterContexts: must finalize first");
    }
    if (world_size_ <= 1) {
        return; // serial: nothing to build
    }

#if !FE_HAS_MPI
    throw FEException("DofHandler::buildScatterContexts: FE built without MPI support");
#else
    if (!ghost_manager_) {
        throw FEException("DofHandler::buildScatterContexts: ghost_manager is not initialized");
    }

    // Build packed-owned ordering (stable, sorted by global DOF ID).
    const auto owned_dofs = partition_.locallyOwned().toVector();
    const auto ghost_dofs = partition_.ghost().toVector();

    std::vector<int> ghost_owners;
    ghost_owners.reserve(ghost_dofs.size());

    // Neighbor ranks for scalable neighbor-only communication.
    // Include any ghost owners even if the caller did not provide explicit neighbors.
    std::vector<int> neighbors = neighbor_ranks_;
    std::unordered_map<int, std::size_t> neighbor_to_index;
    neighbor_to_index.reserve(neighbors.size() + 8u);
    for (std::size_t i = 0; i < neighbors.size(); ++i) {
        neighbor_to_index.emplace(neighbors[i], i);
    }

    auto ensure_neighbor = [&](int r) {
        if (r == my_rank_) return;
        if (r < 0 || r >= world_size_) {
            throw FEException("DofHandler::buildScatterContexts: invalid neighbor rank");
        }
        const auto it = neighbor_to_index.find(r);
        if (it != neighbor_to_index.end()) return;
        neighbor_to_index.emplace(r, neighbors.size());
        neighbors.push_back(r);
    };

    for (auto dof : ghost_dofs) {
        const int owner = dof_map_.getDofOwner(dof);
        if (owner < 0 || owner >= world_size_) {
            throw FEException("DofHandler::buildScatterContexts: invalid owner rank for ghost DOF");
        }
        if (owner == my_rank_) {
            throw FEException("DofHandler::buildScatterContexts: ghost set contains locally owned DOF");
        }
        ghost_owners.push_back(owner);
        ensure_neighbor(owner);
    }

    // Normalize neighbor list.
    std::sort(neighbors.begin(), neighbors.end());
    neighbors.erase(std::remove(neighbors.begin(), neighbors.end(), my_rank_), neighbors.end());
    neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
    neighbor_to_index.clear();
    neighbor_to_index.reserve(neighbors.size());
    for (std::size_t i = 0; i < neighbors.size(); ++i) {
        neighbor_to_index.emplace(neighbors[i], i);
    }

    std::vector<std::vector<GlobalIndex>> requests_by_neighbor(neighbors.size());
    for (std::size_t i = 0; i < ghost_dofs.size(); ++i) {
        const int owner = ghost_owners[i];
        const auto it = neighbor_to_index.find(owner);
        if (it == neighbor_to_index.end()) {
            throw FEException("DofHandler::buildScatterContexts: missing owner in neighbor list");
        }
        requests_by_neighbor[it->second].push_back(ghost_dofs[i]);
    }
    for (auto& list : requests_by_neighbor) {
        std::sort(list.begin(), list.end());
        list.erase(std::unique(list.begin(), list.end()), list.end());
    }

    // Reinitialize ghost manager in a mesh-independent, deterministic way.
	    ghost_manager_->setMyRank(my_rank_);
	    ghost_manager_->setGhostDofs(ghost_dofs, ghost_owners);
	    ghost_manager_->setOwnedDofs(owned_dofs);

	    // Neighbor-only request exchange:
	    // Each rank sends to each neighbor the list of ghost DOFs it needs from that neighbor.
	    // Each rank receives from each neighbor the list of DOFs that neighbor needs from us.
    constexpr int tag_counts = 41010;
    constexpr int tag_data = 41011;

    std::vector<int> send_counts(neighbors.size(), 0);
    for (std::size_t i = 0; i < neighbors.size(); ++i) {
        const auto& list = requests_by_neighbor[i];
        if (list.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw FEException("DofHandler::buildScatterContexts: ghost request list too large for MPI int counts");
        }
        send_counts[i] = static_cast<int>(list.size());
    }

    std::vector<int> recv_counts(neighbors.size(), 0);
    std::vector<MPI_Request> count_reqs;
    count_reqs.reserve(neighbors.size() * 2u);
    for (std::size_t i = 0; i < neighbors.size(); ++i) {
        MPI_Request req{};
        fe_mpi_check(MPI_Isend(&send_counts[i], 1, MPI_INT, neighbors[i], tag_counts, mpi_comm_, &req),
                     "MPI_Isend (counts) in DofHandler::buildScatterContexts");
        count_reqs.push_back(req);
        fe_mpi_check(MPI_Irecv(&recv_counts[i], 1, MPI_INT, neighbors[i], tag_counts, mpi_comm_, &req),
                     "MPI_Irecv (counts) in DofHandler::buildScatterContexts");
        count_reqs.push_back(req);
    }
    fe_mpi_check(MPI_Waitall(static_cast<int>(count_reqs.size()), count_reqs.data(), MPI_STATUSES_IGNORE),
                 "MPI_Waitall (counts) in DofHandler::buildScatterContexts");

    std::vector<std::vector<GlobalIndex>> incoming_requests(neighbors.size());
    for (std::size_t i = 0; i < neighbors.size(); ++i) {
        const int n = recv_counts[i];
        if (n < 0) {
            throw FEException("DofHandler::buildScatterContexts: negative receive count");
        }
        incoming_requests[i].resize(static_cast<std::size_t>(n));
    }

    std::vector<MPI_Request> data_reqs;
    data_reqs.reserve(neighbors.size() * 2u);
    for (std::size_t i = 0; i < neighbors.size(); ++i) {
        MPI_Request req{};
        const auto& send_list = requests_by_neighbor[i];
        fe_mpi_check(MPI_Isend(send_list.empty() ? nullptr : send_list.data(),
                               send_counts[i],
                               MPI_INT64_T,
                               neighbors[i],
                               tag_data,
                               mpi_comm_,
                               &req),
                     "MPI_Isend (data) in DofHandler::buildScatterContexts");
        data_reqs.push_back(req);

        auto& recv_list = incoming_requests[i];
        fe_mpi_check(MPI_Irecv(recv_list.empty() ? nullptr : recv_list.data(),
                               recv_counts[i],
                               MPI_INT64_T,
                               neighbors[i],
                               tag_data,
                               mpi_comm_,
                               &req),
                     "MPI_Irecv (data) in DofHandler::buildScatterContexts");
        data_reqs.push_back(req);
    }
    fe_mpi_check(MPI_Waitall(static_cast<int>(data_reqs.size()), data_reqs.data(), MPI_STATUSES_IGNORE),
                 "MPI_Waitall (data) in DofHandler::buildScatterContexts");

    // Union "shared" DOFs per neighbor = (ghosts we receive from them) U (DOFs they request from us).
    for (std::size_t i = 0; i < neighbors.size(); ++i) {
        auto& requested = incoming_requests[i];
        std::sort(requested.begin(), requested.end());
        requested.erase(std::unique(requested.begin(), requested.end()), requested.end());

        for (auto dof : requested) {
            const int owner = dof_map_.getDofOwner(dof);
            if (owner != my_rank_) {
                throw FEException("DofHandler::buildScatterContexts: received request for DOF not owned by this rank");
            }
        }

        const auto& need_from_neighbor = requests_by_neighbor[i];
        std::vector<GlobalIndex> shared;
        shared.reserve(need_from_neighbor.size() + requested.size());
        std::merge(need_from_neighbor.begin(), need_from_neighbor.end(),
                   requested.begin(), requested.end(),
                   std::back_inserter(shared));
        shared.erase(std::unique(shared.begin(), shared.end()), shared.end());
        if (!shared.empty()) {
            ghost_manager_->addSharedDofsWithNeighbor(neighbors[i], shared);
        }
    }

	    ghost_manager_->buildGhostExchange();

	    // Build reusable MPI schedules + index maps for fast ghost exchange.
	    ghost_exchange_mpi_ = std::make_unique<GhostExchangeContextMPI>();
	    ghost_exchange_mpi_->comm = mpi_comm_;
	    ghost_exchange_mpi_->tag = 43010;
	    ghost_exchange_mpi_->clearPersistent();

	    const auto& sched = ghost_manager_->getCommSchedule();
		    ghost_exchange_mpi_->neighbors = sched.neighbor_ranks;
		    ghost_exchange_mpi_->send_counts.resize(sched.neighbor_ranks.size(), 0);
		    ghost_exchange_mpi_->recv_counts.resize(sched.neighbor_ranks.size(), 0);
		    ghost_exchange_mpi_->send_displs.resize(sched.neighbor_ranks.size(), 0);
		    ghost_exchange_mpi_->recv_displs.resize(sched.neighbor_ranks.size(), 0);
		    ghost_exchange_mpi_->send_offsets.resize(sched.neighbor_ranks.size(), 0);
		    ghost_exchange_mpi_->recv_offsets.resize(sched.neighbor_ranks.size(), 0);
		    ghost_exchange_mpi_->send_indices.assign(sched.neighbor_ranks.size(), {});
		    ghost_exchange_mpi_->recv_indices.assign(sched.neighbor_ranks.size(), {});

		    std::size_t send_total = 0;
		    std::size_t recv_total = 0;
		    for (std::size_t i = 0; i < sched.neighbor_ranks.size(); ++i) {
		        if (send_total > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
		            recv_total > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
		            throw FEException("DofHandler::buildScatterContexts: ghost exchange displacements too large for MPI int");
		        }
		        ghost_exchange_mpi_->send_offsets[i] = send_total;
		        ghost_exchange_mpi_->recv_offsets[i] = recv_total;
		        ghost_exchange_mpi_->send_displs[i] = static_cast<int>(send_total);
		        ghost_exchange_mpi_->recv_displs[i] = static_cast<int>(recv_total);

		        const auto& send_list = sched.send_lists[i];
		        const auto& recv_list = sched.recv_lists[i];

	        if (send_list.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
	            recv_list.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
	            throw FEException("DofHandler::buildScatterContexts: ghost exchange list too large for MPI int counts");
	        }
	        ghost_exchange_mpi_->send_counts[i] = static_cast<int>(send_list.size());
	        ghost_exchange_mpi_->recv_counts[i] = static_cast<int>(recv_list.size());

	        ghost_exchange_mpi_->send_indices[i].reserve(send_list.size());
	        for (auto dof : send_list) {
	            const auto idx = ghost_manager_->ownedIndex(dof);
	            if (!idx.has_value()) {
	                throw FEException("DofHandler::buildScatterContexts: missing owned index for send DOF");
	            }
	            ghost_exchange_mpi_->send_indices[i].push_back(*idx);
	        }

	        ghost_exchange_mpi_->recv_indices[i].reserve(recv_list.size());
	        for (auto dof : recv_list) {
	            const auto idx = ghost_manager_->ghostIndex(dof);
	            if (!idx.has_value()) {
	                throw FEException("DofHandler::buildScatterContexts: missing ghost index for recv DOF");
	            }
	            ghost_exchange_mpi_->recv_indices[i].push_back(*idx);
	        }

	        send_total += send_list.size();
	        recv_total += recv_list.size();
	    }

		    ghost_exchange_mpi_->send_buffer.assign(send_total, 0.0);
		    ghost_exchange_mpi_->recv_buffer.assign(recv_total, 0.0);

#if MPI_VERSION >= 3
			    ghost_exchange_mpi_->neighbor_comm = MPI_COMM_NULL;
			    const int deg = static_cast<int>(ghost_exchange_mpi_->neighbors.size());
			    const int* nbrs = (deg == 0) ? nullptr : ghost_exchange_mpi_->neighbors.data();
			    fe_mpi_check(MPI_Dist_graph_create_adjacent(ghost_exchange_mpi_->comm,
			                                              deg,
			                                              nbrs,
			                                              MPI_UNWEIGHTED,
			                                              deg,
			                                              nbrs,
			                                              MPI_UNWEIGHTED,
			                                              MPI_INFO_NULL,
			                                              /*reorder=*/0,
			                                              &ghost_exchange_mpi_->neighbor_comm),
			                 "MPI_Dist_graph_create_adjacent in DofHandler::buildScatterContexts");
#endif
		#endif
		}

	#if FE_HAS_MPI
	void DofHandler::syncGhostValuesMPI(std::span<const double> owned_values,
	                                    std::span<double> ghost_values) {
	    if (world_size_ <= 1) {
	        return;
	    }
	    if (!finalized_) {
	        throw FEException("DofHandler::syncGhostValuesMPI: must finalize first");
	    }
	    if (!ghost_exchange_mpi_) {
	        throw FEException("DofHandler::syncGhostValuesMPI: scatter contexts not built; call buildScatterContexts()");
	    }

		    if (ghost_values.size() != static_cast<std::size_t>(partition_.ghostSize())) {
		        throw FEException("DofHandler::syncGhostValuesMPI: ghost_values size mismatch");
		    }

		    auto& ctx = *ghost_exchange_mpi_;

			    // Pack.
			    for (std::size_t i = 0; i < ctx.neighbors.size(); ++i) {
			        const auto off = ctx.send_offsets[i];
			        const auto& idx = ctx.send_indices[i];
		        for (std::size_t j = 0; j < idx.size(); ++j) {
	            if (idx[j] >= owned_values.size()) {
	                throw FEException("DofHandler::syncGhostValuesMPI: owned_values too small for packed-owned index map");
	            }
	            ctx.send_buffer[off + j] = owned_values[idx[j]];
			        }
			    }

#if MPI_VERSION >= 3
			    if (ctx.neighbor_comm == MPI_COMM_NULL) {
			        throw FEException("DofHandler::syncGhostValuesMPI: neighbor communicator not available; call buildScatterContexts()");
			    }

			    MPI_Request req = MPI_REQUEST_NULL;
			    const void* send_ptr = ctx.send_buffer.empty() ? nullptr : static_cast<const void*>(ctx.send_buffer.data());
			    void* recv_ptr = ctx.recv_buffer.empty() ? nullptr : static_cast<void*>(ctx.recv_buffer.data());
			    const int* send_counts_ptr = ctx.send_counts.empty() ? nullptr : ctx.send_counts.data();
			    const int* send_displs_ptr = ctx.send_displs.empty() ? nullptr : ctx.send_displs.data();
			    const int* recv_counts_ptr = ctx.recv_counts.empty() ? nullptr : ctx.recv_counts.data();
			    const int* recv_displs_ptr = ctx.recv_displs.empty() ? nullptr : ctx.recv_displs.data();

			    fe_mpi_check(MPI_Ineighbor_alltoallv(send_ptr,
			                                         send_counts_ptr,
			                                         send_displs_ptr,
			                                         MPI_DOUBLE,
			                                         recv_ptr,
			                                         recv_counts_ptr,
			                                         recv_displs_ptr,
			                                         MPI_DOUBLE,
			                                         ctx.neighbor_comm,
			                                         &req),
			                 "MPI_Ineighbor_alltoallv in DofHandler::syncGhostValuesMPI");
			    fe_mpi_check(MPI_Wait(&req, MPI_STATUS_IGNORE),
			                 "MPI_Wait (neighbor alltoallv) in DofHandler::syncGhostValuesMPI");
#else
			    // Nonblocking point-to-point exchange.
			    std::vector<MPI_Request> reqs(ctx.neighbors.size() * 2u, MPI_REQUEST_NULL);
			    for (std::size_t i = 0; i < ctx.neighbors.size(); ++i) {
			        const int peer = ctx.neighbors[i];
			        const int recv_n = ctx.recv_counts[i];
			        const int send_n = ctx.send_counts[i];
			        double* recv_ptr = ctx.recv_buffer.empty() ? nullptr : (ctx.recv_buffer.data() + ctx.recv_offsets[i]);
			        double* send_ptr = ctx.send_buffer.empty() ? nullptr : (ctx.send_buffer.data() + ctx.send_offsets[i]);

			        fe_mpi_check(MPI_Irecv(recv_ptr, recv_n, MPI_DOUBLE, peer, ctx.tag, ctx.comm, &reqs[i]),
			                     "MPI_Irecv in DofHandler::syncGhostValuesMPI");
			        fe_mpi_check(MPI_Isend(send_ptr, send_n, MPI_DOUBLE, peer, ctx.tag, ctx.comm, &reqs[ctx.neighbors.size() + i]),
			                     "MPI_Isend in DofHandler::syncGhostValuesMPI");
			    }
			    fe_mpi_check(MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE),
			                 "MPI_Waitall in DofHandler::syncGhostValuesMPI");
#endif

			    // Unpack.
			    for (std::size_t i = 0; i < ctx.neighbors.size(); ++i) {
			        const auto off = ctx.recv_offsets[i];
		        const auto& idx = ctx.recv_indices[i];
	        for (std::size_t j = 0; j < idx.size(); ++j) {
	            if (idx[j] >= ghost_values.size()) {
	                throw FEException("DofHandler::syncGhostValuesMPI: ghost index out of range");
	            }
	            ghost_values[idx[j]] = ctx.recv_buffer[off + j];
	        }
	    }
	}

	void DofHandler::syncGhostValuesMPIPersistent(std::span<const double> owned_values,
	                                              std::span<double> ghost_values) {
	    if (world_size_ <= 1) {
	        return;
	    }
	    if (!finalized_) {
	        throw FEException("DofHandler::syncGhostValuesMPIPersistent: must finalize first");
	    }
	    if (!ghost_exchange_mpi_) {
	        throw FEException("DofHandler::syncGhostValuesMPIPersistent: scatter contexts not built; call buildScatterContexts()");
	    }

	    if (ghost_values.size() != static_cast<std::size_t>(partition_.ghostSize())) {
	        throw FEException("DofHandler::syncGhostValuesMPIPersistent: ghost_values size mismatch");
	    }

		    auto& ctx = *ghost_exchange_mpi_;
		    ctx.buildPersistent();

	    // Pack.
	    for (std::size_t i = 0; i < ctx.neighbors.size(); ++i) {
	        const auto off = ctx.send_offsets[i];
	        const auto& idx = ctx.send_indices[i];
	        for (std::size_t j = 0; j < idx.size(); ++j) {
	            if (idx[j] >= owned_values.size()) {
	                throw FEException("DofHandler::syncGhostValuesMPIPersistent: owned_values too small for packed-owned index map");
	            }
	            ctx.send_buffer[off + j] = owned_values[idx[j]];
	        }
	    }

	    fe_mpi_check(MPI_Startall(static_cast<int>(ctx.persistent_reqs.size()), ctx.persistent_reqs.data()),
	                 "MPI_Startall in DofHandler::syncGhostValuesMPIPersistent");
	    fe_mpi_check(MPI_Waitall(static_cast<int>(ctx.persistent_reqs.size()), ctx.persistent_reqs.data(), MPI_STATUSES_IGNORE),
	                 "MPI_Waitall in DofHandler::syncGhostValuesMPIPersistent");

	    // Unpack.
	    for (std::size_t i = 0; i < ctx.neighbors.size(); ++i) {
	        const auto off = ctx.recv_offsets[i];
	        const auto& idx = ctx.recv_indices[i];
	        for (std::size_t j = 0; j < idx.size(); ++j) {
	            if (idx[j] >= ghost_values.size()) {
	                throw FEException("DofHandler::syncGhostValuesMPIPersistent: ghost index out of range");
	            }
	            ghost_values[idx[j]] = ctx.recv_buffer[off + j];
	        }
	    }
	}
	#endif

	void DofHandler::setRankInfo(int my_rank, int world_size) {
	    my_rank_ = my_rank;
	    world_size_ = world_size;
	}

// =============================================================================
// Internal Helpers
// =============================================================================

void DofHandler::checkNotFinalized() const {
    if (finalized_) {
        throw FEException("DofHandler: operation not allowed after finalization");
    }
}

} // namespace dofs
} // namespace FE
} // namespace svmp
