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
#include "Spaces/FunctionSpace.h"
#include "Spaces/OrientationManager.h"

// Mesh convenience overloads require the Mesh library to be linked, not just headers present.
// Standalone FE builds inside the svMultiPhysics repo still see Mesh headers, so __has_include
// alone is insufficient. Prefer an explicit compile definition when provided.
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#  include "Mesh/Mesh.h"
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

    if (order == 1) {
        // P1: DOFs only at vertices
        info.dofs_per_vertex = 1;
        info.dofs_per_edge = 0;
        info.dofs_per_face = 0;
        info.dofs_per_cell = 0;
        info.total_dofs_per_element = static_cast<LocalIndex>(num_verts_per_cell);
    } else if (order == 2) {
        // P2: DOFs at vertices + 1 per edge
        info.dofs_per_vertex = 1;
        info.dofs_per_edge = 1;
        info.dofs_per_face = 0;
        info.dofs_per_cell = 0;
        // Total depends on element type (tri: 6, quad: 9, tet: 10, hex: 27)
        // For simplicial elements: n_verts + n_edges
        if (dim == 1) {
            // 1D segment: 2 vertices + 1 interior node = 3
            info.dofs_per_edge = 0;
            info.dofs_per_cell = 1;
            info.total_dofs_per_element = 3;
        } else if (dim == 2) {
            if (num_verts_per_cell == 3) {
                info.total_dofs_per_element = 6;  // Triangle
            } else if (num_verts_per_cell == 4) {
                // Quad Q2: 4 vertices + 4 edge midpoints + 1 cell interior = 9
                info.dofs_per_cell = 1;
                info.total_dofs_per_element = 9;
            } else {
                throw FEException("DofLayoutInfo::Lagrange: unsupported 2D element for order 2");
            }
        } else if (dim == 3) {
            if (num_verts_per_cell == 4) {
                info.total_dofs_per_element = 10;  // Tet
            } else if (num_verts_per_cell == 8) {
                // Hex Q2: 8 vertices + 12 edge midpoints + 6 face centers + 1 cell center = 27
                info.dofs_per_face = 1;
                info.dofs_per_cell = 1;
                info.total_dofs_per_element = 27;
            } else {
                throw FEException("DofLayoutInfo::Lagrange: unsupported 3D element for order 2");
            }
        } else {
            throw FEException("DofLayoutInfo::Lagrange: unsupported dimension for order 2");
        }
    } else if (order == 3) {
        // P3: vertices + 2 per edge + (optional) face/cell interior DOFs.
        //
        // NOTE: In 2D, "face interior" DOFs are cell-interior (bubble) DOFs
        // and should be counted in dofs_per_cell (there is no separate face entity).
        info.dofs_per_vertex = 1;
        info.dofs_per_edge = 2;
        info.dofs_per_face = 0;
        info.dofs_per_cell = 0;

        if (dim == 1) {
            // 1D segment: (p+1) nodes, with (p-1) interior nodes
            info.dofs_per_edge = 0;
            info.dofs_per_cell = 2;
            info.total_dofs_per_element = 4;
        } else if (dim == 2) {
            if (num_verts_per_cell == 3) {
                // Triangle P3: 3 vertices + 3 edges*2 + 1 cell interior = 10
                info.dofs_per_cell = 1;
                info.total_dofs_per_element = 10;
            } else if (num_verts_per_cell == 4) {
                // Quad Q3: 4 vertices + 4 edges*2 + 4 cell interior = 16
                info.dofs_per_cell = 4;
                info.total_dofs_per_element = 16;
            } else {
                throw FEException("DofLayoutInfo::Lagrange: unsupported 2D element for order 3");
            }
        } else if (dim == 3) {
            if (num_verts_per_cell == 4) {
                // Tetra P3: 4 vertices + 6 edges*2 + 4 faces*1 = 20
                info.dofs_per_face = 1;
                info.dofs_per_cell = 0;
                info.total_dofs_per_element = 20;
            } else if (num_verts_per_cell == 8) {
                // Hex Q3: 8 vertices + 12 edges*2 + 6 faces*4 + 1 cell*8 = 64
                info.dofs_per_face = 4;
                info.dofs_per_cell = 8;
                info.total_dofs_per_element = 64;
            } else {
                throw FEException("DofLayoutInfo::Lagrange: unsupported 3D element for order 3");
            }
        } else {
            throw FEException("DofLayoutInfo::Lagrange: unsupported dimension for order 3");
        }
    } else {
        throw FEException("DofLayoutInfo::Lagrange: order > 3 not implemented");
    }

    // total_dofs_per_element describes the full field (all components).
    if (info.num_components > 1 && info.total_dofs_per_element > 0) {
        info.total_dofs_per_element = static_cast<LocalIndex>(
            static_cast<GlobalIndex>(info.total_dofs_per_element) *
            static_cast<GlobalIndex>(info.num_components));
    }

    return info;
}

DofLayoutInfo DofLayoutInfo::DG(int order, int num_verts_per_cell, int num_components) {
    DofLayoutInfo info;
    info.is_continuous = false;
    info.num_components = num_components;

    // For DG, all DOFs are cell-interior (no sharing)
    info.dofs_per_vertex = 0;
    info.dofs_per_edge = 0;
    info.dofs_per_face = 0;

    // Total DOFs based on polynomial order and element type
    // For simplicial elements in 2D: (p+1)(p+2)/2
    // For simplicial elements in 3D: (p+1)(p+2)(p+3)/6
    if (num_verts_per_cell == 3) {
        info.dofs_per_cell = static_cast<LocalIndex>((order + 1) * (order + 2) / 2);
    } else if (num_verts_per_cell == 4 && order <= 1) {
        // Could be quad or tet
        info.dofs_per_cell = static_cast<LocalIndex>(num_verts_per_cell);
    } else {
        // Fallback
        info.dofs_per_cell = static_cast<LocalIndex>(num_verts_per_cell);
    }

    info.total_dofs_per_element = static_cast<LocalIndex>(
        static_cast<GlobalIndex>(info.dofs_per_cell) * static_cast<GlobalIndex>(std::max(1, num_components)));

    return info;
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
                   a.dofs_per_cell == b.dofs_per_cell &&
                   a.num_components == b.num_components &&
                   a.is_continuous == b.is_continuous &&
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
{
    other.finalized_ = false;
    other.ghost_cache_valid_ = false;
    other.dof_state_revision_ = 0;
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

        other.finalized_ = false;
        other.ghost_cache_valid_ = false;
        other.dof_state_revision_ = 0;
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

    const MeshTopologyView* topo = &topology;
    MeshTopologyInfo derived_topology;
    MeshTopologyView derived_view;

    const bool need_edges = layout.is_continuous && layout.dofs_per_edge > 0;
    const bool need_faces = layout.is_continuous && layout.dofs_per_face > 0;
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

    if (options.numbering != DofNumberingStrategy::Sequential) {
        renumberDofs(options.numbering);
    }
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
    if (layout.dofs_per_face > 0) {
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

    if (layout.dofs_per_face > 0 && topology.n_faces > 0) {
        for (GlobalIndex f = 0; f < topology.n_faces; ++f) {
            face_first_dof[static_cast<std::size_t>(f)] = next_dof;
            std::vector<GlobalIndex> f_dofs;
            for (LocalIndex d = 0; d < layout.dofs_per_face; ++d) {
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

        // Add face DOFs (requires cell2face connectivity when dofs_per_face > 0).
        if (layout.dofs_per_face > 0 && !topology.cell2face_offsets.empty()) {
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
                (layout.dofs_per_face > 1) &&
                (base_type != ElementType::Unknown) &&
                !topology.face2vertex_offsets.empty() &&
                !topology.face2vertex_data.empty();

            const int poly_order = static_cast<int>(layout.dofs_per_edge) + 1;
            const int face_grid = poly_order - 1; // interior nodes per axis for tensor faces

            auto get_face_vertices = [&](GlobalIndex face_id) {
                return topology.getFaceVertices(face_id);
            };

            auto quad_interior_permutation_local_to_global = [&](const std::vector<int>& vertex_perm) -> std::vector<int> {
                // vertex_perm maps canonical corner index -> local corner index.
                // We find the corresponding dihedral transform on a (p+1)x(p+1) grid
                // and restrict it to the (p-1)x(p-1) interior.
                if (poly_order < 2 || face_grid <= 0) {
                    return {};
                }
                const int p = poly_order;
                const int m = face_grid;

                struct Transform { bool swap; bool flip_u; bool flip_v; };
                auto apply_transform = [&](int i, int j, const Transform& t) -> std::pair<int, int> {
                    int u = i;
                    int v = j;
                    if (t.swap) {
                        std::swap(u, v);
                    }
                    if (t.flip_u) {
                        u = p - u;
                    }
                    if (t.flip_v) {
                        v = p - v;
                    }
                    return {u, v};
                };

                auto corner_index = [&](int i, int j) -> int {
                    if (i == 0 && j == 0) return 0;
                    if (i == p && j == 0) return 1;
                    if (i == p && j == p) return 2;
                    if (i == 0 && j == p) return 3;
                    return -1;
                };

                Transform matched{false, false, false};
                bool found = false;
                for (int swap = 0; swap <= 1 && !found; ++swap) {
                    for (int fu = 0; fu <= 1 && !found; ++fu) {
                        for (int fv = 0; fv <= 1 && !found; ++fv) {
                            const Transform t{swap != 0, fu != 0, fv != 0};
                            bool ok = true;
                            for (int cidx = 0; cidx < 4; ++cidx) {
                                int ci = 0, cj = 0;
                                switch (cidx) {
                                    case 0: ci = 0; cj = 0; break;
                                    case 1: ci = p; cj = 0; break;
                                    case 2: ci = p; cj = p; break;
                                    case 3: ci = 0; cj = p; break;
                                }
                                const auto [ti, tj] = apply_transform(ci, cj, t);
                                const int mapped = corner_index(ti, tj);
                                if (mapped < 0 || static_cast<std::size_t>(cidx) >= vertex_perm.size() ||
                                    mapped != vertex_perm[static_cast<std::size_t>(cidx)]) {
                                    ok = false;
                                    break;
                                }
                            }
                            if (ok) {
                                matched = t;
                                found = true;
                            }
                        }
                    }
                }

                if (!found) {
                    throw FEException("DofHandler::distributeDofs: failed to match quad face orientation to a canonical dihedral transform");
                }

                std::vector<int> local_to_global(static_cast<std::size_t>(m * m), -1);
                for (int gj = 0; gj < m; ++gj) {
                    for (int gi = 0; gi < m; ++gi) {
                        const int g = gj * m + gi;        // canonical interior index (v-major)
                        const int ci = gi + 1;            // canonical grid index in [1,p-1]
                        const int cj = gj + 1;
                        const auto [li, lj] = apply_transform(ci, cj, matched);
                        if (li <= 0 || li >= p || lj <= 0 || lj >= p) {
                            throw FEException("DofHandler::distributeDofs: quad face interior map produced boundary index");
                        }
                        const int lgi = li - 1;
                        const int lgj = lj - 1;
                        const int l = lgj * m + lgi;
                        if (l < 0 || l >= m * m) {
                            throw FEException("DofHandler::distributeDofs: quad face interior permutation out of range");
                        }
                        local_to_global[static_cast<std::size_t>(l)] = g;
                    }
                }

                for (std::size_t i = 0; i < local_to_global.size(); ++i) {
                    if (local_to_global[i] < 0) {
                        throw FEException("DofHandler::distributeDofs: quad face interior permutation incomplete");
                    }
                }
                return local_to_global;
            };

            const auto ref = (base_type != ElementType::Unknown)
                                 ? elements::ReferenceElement::create(base_type)
                                 : elements::ReferenceElement{};

            for (std::size_t lf = 0; lf < cell_faces.size(); ++lf) {
                const GlobalIndex f = cell_faces[lf];

                if (!can_orient_faces || layout.dofs_per_face == 1) {
                    for (LocalIndex d = 0; d < layout.dofs_per_face; ++d) {
                        cell_dofs.push_back(face_first_dof[static_cast<std::size_t>(f)] + d);
                    }
                    continue;
                }

                const auto face_vertices = get_face_vertices(f);
                if (face_vertices.empty()) {
                    throw FEException("DofHandler::distributeDofs: missing face2vertex connectivity for face orientation");
                }

                if (face_vertices.size() == 4u) {
                    if (face_grid <= 0 ||
                        static_cast<LocalIndex>(face_grid * face_grid) != layout.dofs_per_face) {
                        throw FEException("DofHandler::distributeDofs: quad face dofs_per_face does not match (p-1)^2 for tensor Lagrange");
                    }

                    if (lf >= ref.num_faces()) {
                        throw FEException("DofHandler::distributeDofs: cell2face ordering does not match reference element face count");
                    }
                    const auto& fn = ref.face_nodes(lf);
                    if (fn.size() != 4u) {
                        throw FEException("DofHandler::distributeDofs: expected quad face in reference element");
                    }

                    std::array<int, 4> local{};
                    std::array<int, 4> global{};
                    for (std::size_t i = 0; i < 4u; ++i) {
                        const auto lv = static_cast<std::size_t>(fn[i]);
                        if (lv >= cell_verts.size()) {
                            throw FEException("DofHandler::distributeDofs: face vertex index out of range while orienting face DOFs");
                        }
                        local[i] = static_cast<int>(cell_verts[lv]);
                        global[i] = static_cast<int>(face_vertices[i]);
                    }

                    const auto orient = spaces::OrientationManager::quad_face_orientation(local, global);
                    const auto local_to_global = quad_interior_permutation_local_to_global(orient.vertex_perm);
                    if (static_cast<LocalIndex>(local_to_global.size()) != layout.dofs_per_face) {
                        throw FEException("DofHandler::distributeDofs: quad face interior permutation size mismatch");
                    }

                    for (LocalIndex l = 0; l < layout.dofs_per_face; ++l) {
                        const int g = local_to_global[static_cast<std::size_t>(l)];
                        cell_dofs.push_back(face_first_dof[static_cast<std::size_t>(f)] + static_cast<GlobalIndex>(g));
                    }
                } else {
                    // Simplex faces currently only have 0/1 interior DOFs in the supported layouts.
                    for (LocalIndex d = 0; d < layout.dofs_per_face; ++d) {
                        cell_dofs.push_back(face_first_dof[static_cast<std::size_t>(f)] + d);
                    }
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
            if (layout.dofs_per_face <= 0 || face_first_dof[static_cast<std::size_t>(f)] < 0) continue;
            expanded_entity->setFaceDofs(f, expand_entity_range(face_first_dof[static_cast<std::size_t>(f)], layout.dofs_per_face));
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
    if (layout.dofs_per_face > 0) {
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
    std::vector<FaceKey> face_keys;
    std::vector<int> face_touch;
    std::vector<gid_t> face_cell_gid_candidate;
    std::vector<int> face_cell_owner_candidate;

    auto get_face_vertices = [&](GlobalIndex face_id) {
        return topology.getFaceVertices(face_id);
    };

    if (layout.dofs_per_face > 0 && topology.n_faces > 0) {
        face_keys.resize(n_faces);
        face_touch.assign(n_faces, -1);
        face_cell_gid_candidate.assign(n_faces, std::numeric_limits<gid_t>::max());
        face_cell_owner_candidate.assign(n_faces, -1);

        for (GlobalIndex f = 0; f < topology.n_faces; ++f) {
            const auto verts = get_face_vertices(f);
            if (verts.empty()) {
                throw FEException("DofHandler::distributeCGDofsParallel: face2vertex connectivity missing for face");
            }
            if (verts.size() != 3u && verts.size() != 4u) {
                throw FEException("DofHandler::distributeCGDofsParallel: unsupported face vertex count");
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
            face_keys[static_cast<std::size_t>(f)] = key;
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

        if (!face_keys.empty()) {
            for (auto f : topology.getCellFaces(c)) {
                const auto sf = static_cast<std::size_t>(f);
                if (sf >= n_faces) {
                    throw FEException("DofHandler::distributeCGDofsParallel: cell face index out of range");
                }
                if (cgid < face_cell_gid_candidate[sf] ||
                    (cgid == face_cell_gid_candidate[sf] && cell_owner < face_cell_owner_candidate[sf])) {
                    face_cell_gid_candidate[sf] = cgid;
                    face_cell_owner_candidate[sf] = cell_owner;
                }
                if (owned_cell) {
                    face_touch[sf] = my_rank_;
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
    for (std::size_t f = 0; f < face_cell_gid_candidate.size(); ++f) {
        if (face_cell_gid_candidate[f] == std::numeric_limits<gid_t>::max()) {
            face_cell_gid_candidate[f] = gid_t{-1};
            face_cell_owner_candidate[f] = -1;
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

	    std::vector<gid_t> face_global_id;
	    std::vector<int> face_owner_rank;
	    gid_t n_global_faces = 0;

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

	        if (!face_keys.empty()) {
	            assign_contiguous_ids_and_owners_with_neighbors<FaceKey, FaceKeyHash, std::less<FaceKey>>(
	                mpi_comm_,
	                my_rank_,
	                world_size_,
	                neighbors,
	                options.ownership,
	                face_keys,
	                face_touch,
	                face_cell_gid_candidate,
	                face_cell_owner_candidate,
	                rank_to_order,
	                no_global_collectives_,
	                face_global_id,
	                face_owner_rank,
	                n_global_faces,
	                /*tag_base=*/42020);
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

	        if (!face_keys.empty()) {
	            if (topology.face_gids.size() != static_cast<std::size_t>(topology.n_faces)) {
	                throw FEException("DofHandler::distributeCGDofsParallel: global_numbering=GlobalIds requires face_gids (size must equal n_faces)");
	            }
	            face_global_id.assign(topology.face_gids.begin(), topology.face_gids.end());
	            assign_owners_with_neighbors<FaceKey, FaceKeyHash>(mpi_comm_,
	                                                              my_rank_,
	                                                              world_size_,
	                                                              neighbors,
	                                                              options.ownership,
	                                                              face_keys,
	                                                              face_touch,
	                                                              face_cell_gid_candidate,
	                                                              face_cell_owner_candidate,
	                                                              rank_to_order,
	                                                              face_owner_rank,
	                                                              /*tag_base=*/42020);
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

	        if (layout.dofs_per_face > 0) {
	            const gid_t max_f_gid = global_max_gid(face_global_id, /*tag_base=*/41720, "face_gids");
	            n_global_faces = (max_f_gid >= 0) ? checked_nonneg_add(max_f_gid, gid_t{1}, "face gid range") : gid_t{0};
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

		        if (!face_keys.empty()) {
		            assign_owners_with_neighbors<FaceKey, FaceKeyHash>(mpi_comm_,
		                                                              my_rank_,
		                                                              world_size_,
		                                                              neighbors,
		                                                              options.ownership,
		                                                              face_keys,
		                                                              face_touch,
		                                                              face_cell_gid_candidate,
		                                                              face_cell_owner_candidate,
		                                                              rank_to_order,
		                                                              face_owner_rank,
		                                                              /*tag_base=*/42020);
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

		        if (!face_keys.empty()) {
		            face_global_id.clear();
			            assign_dense_global_ordinals_compact_auto<FaceKey, FaceKeyHash, std::less<FaceKey>>(
			                mpi_comm_,
			                my_rank_,
			                world_size_,
			                rank_to_order,
			                face_keys,
			                no_global_collectives_,
			                face_global_id,
			                n_global_faces,
			                /*tag_base=*/42900);
		        }
		    }

	    const gid_t vertex_dofs_total =
	        checked_nonneg_mul(n_global_vertices, static_cast<gid_t>(layout.dofs_per_vertex), "vertex dof block");
	    const gid_t edge_dofs_total =
	        checked_nonneg_mul(n_global_edges, static_cast<gid_t>(layout.dofs_per_edge), "edge dof block");
	    const gid_t face_dofs_total =
	        checked_nonneg_mul(n_global_faces, static_cast<gid_t>(layout.dofs_per_face), "face dof block");
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

	        if (!face_keys.empty()) {
	            validate_entity_assignments_with_neighbors<FaceKey, FaceKeyHash, std::less<FaceKey>>(
	                mpi_comm_,
	                my_rank_,
	                neighbors,
	                face_keys,
	                face_touch,
	                face_global_id,
	                face_owner_rank,
	                /*tag_base=*/42520,
	                "face");
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

	    std::vector<GlobalIndex> face_first_dof(n_faces, -1);
	    if (layout.dofs_per_face > 0 && n_faces > 0) {
	        for (std::size_t sf = 0; sf < n_faces; ++sf) {
	            const auto f = static_cast<GlobalIndex>(sf);
	            const gid_t base = vertex_dofs_total + edge_dofs_total +
	                               face_global_id[sf] * static_cast<gid_t>(layout.dofs_per_face);
	            face_first_dof[sf] = static_cast<GlobalIndex>(base);
	            std::vector<GlobalIndex> f_dofs;
	            f_dofs.reserve(static_cast<std::size_t>(layout.dofs_per_face) * static_cast<std::size_t>(nc));
	            for (gid_t comp = 0; comp < nc; ++comp) {
	                const gid_t offset = component_offsets[static_cast<std::size_t>(comp)];
	                for (LocalIndex d = 0; d < layout.dofs_per_face; ++d) {
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
	            const gid_t base = vertex_dofs_total + edge_dofs_total + face_dofs_total +
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

        // Add face DOFs (requires cell2face connectivity when dofs_per_face > 0).
        if (layout.dofs_per_face > 0 && !topology.cell2face_offsets.empty()) {
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
                (layout.dofs_per_face > 1) &&
                (base_type != ElementType::Unknown) &&
                !topology.face2vertex_offsets.empty() &&
                !topology.face2vertex_data.empty();

            const int poly_order = static_cast<int>(layout.dofs_per_edge) + 1;
            const int face_grid = poly_order - 1;

            auto quad_interior_permutation_local_to_global = [&](const std::vector<int>& vertex_perm) -> std::vector<int> {
                if (poly_order < 2 || face_grid <= 0) {
                    return {};
                }
                const int p = poly_order;
                const int m = face_grid;

                struct Transform { bool swap; bool flip_u; bool flip_v; };
                auto apply_transform = [&](int i, int j, const Transform& t) -> std::pair<int, int> {
                    int u = i;
                    int v = j;
                    if (t.swap) {
                        std::swap(u, v);
                    }
                    if (t.flip_u) {
                        u = p - u;
                    }
                    if (t.flip_v) {
                        v = p - v;
                    }
                    return {u, v};
                };

                auto corner_index = [&](int i, int j) -> int {
                    if (i == 0 && j == 0) return 0;
                    if (i == p && j == 0) return 1;
                    if (i == p && j == p) return 2;
                    if (i == 0 && j == p) return 3;
                    return -1;
                };

                const std::array<std::pair<int, int>, 4> corners = {
                    std::make_pair(0, 0), std::make_pair(p, 0),
                    std::make_pair(p, p), std::make_pair(0, p)
                };

                Transform best{false, false, false};
                bool found = false;
                for (int swap = 0; swap <= 1; ++swap) {
                    for (int fu = 0; fu <= 1; ++fu) {
                        for (int fv = 0; fv <= 1; ++fv) {
	                            Transform t{swap != 0, fu != 0, fv != 0};
	                            std::array<int, 4> mapped{};
	                            for (std::size_t cidx = 0; cidx < corners.size(); ++cidx) {
	                                const auto [ii, jj] = apply_transform(corners[cidx].first, corners[cidx].second, t);
	                                mapped[cidx] = corner_index(ii, jj);
	                            }
	                            bool ok = true;
	                            for (std::size_t cidx = 0; cidx < corners.size(); ++cidx) {
	                                if (mapped[cidx] < 0 || mapped[cidx] >= 4) {
	                                    ok = false;
	                                    break;
	                                }
	                                if (vertex_perm[cidx] != mapped[cidx]) {
	                                    ok = false;
	                                    break;
	                                }
	                            }
                            if (ok) {
                                best = t;
                                found = true;
                                break;
                            }
                        }
                        if (found) break;
                    }
                    if (found) break;
                }
                if (!found) {
                    throw FEException("DofHandler::distributeCGDofsParallel: unable to match quad face vertex permutation");
                }

                std::vector<int> perm;
                perm.reserve(static_cast<std::size_t>(m * m));

                auto interior_index = [&](int i, int j) -> int {
                    return (j - 1) * m + (i - 1);
                };

                for (int j = 1; j <= p - 1; ++j) {
                    for (int i = 1; i <= p - 1; ++i) {
                        const auto [u, v] = apply_transform(i, j, best);
                        perm.push_back(interior_index(u, v));
                    }
                }
                return perm;
            };

	            for (std::size_t lf = 0; lf < static_cast<std::size_t>(cell_faces.size()); ++lf) {
	                const GlobalIndex f = cell_faces[lf];
	                const auto sf = static_cast<std::size_t>(f);
	                if (sf >= face_first_dof.size()) {
	                    throw FEException("DofHandler::distributeCGDofsParallel: face id out of range while assembling face DOFs");
	                }

	                if (!can_orient_faces) {
	                    for (LocalIndex d = 0; d < layout.dofs_per_face; ++d) {
	                        cell_dofs.push_back(face_first_dof[sf] + d);
	                    }
	                    continue;
	                }

                const auto face_vertices = get_face_vertices(f);
                if (face_vertices.empty()) {
                    throw FEException("DofHandler::distributeCGDofsParallel: face vertex list missing while orienting face DOFs");
                }

                if (face_vertices.size() == 4u) {
                    if (poly_order < 2 || face_grid <= 0 ||
                        static_cast<LocalIndex>(face_grid * face_grid) != layout.dofs_per_face) {
                        throw FEException("DofHandler::distributeCGDofsParallel: quad face dofs_per_face does not match (p-1)^2 for tensor Lagrange");
                    }

                    const auto ref = elements::ReferenceElement::create(base_type);
                    if (lf >= ref.num_faces()) {
                        throw FEException("DofHandler::distributeCGDofsParallel: cell2face ordering does not match reference element face count");
                    }
                    const auto& fn = ref.face_nodes(lf);
                    if (fn.size() != 4u) {
                        throw FEException("DofHandler::distributeCGDofsParallel: expected quad face in reference element");
                    }

                    std::array<int, 4> local{};
                    std::array<int, 4> global{};
                    for (std::size_t i = 0; i < 4u; ++i) {
                        const auto lv = static_cast<std::size_t>(fn[i]);
                        if (lv >= cell_verts.size()) {
                            throw FEException("DofHandler::distributeCGDofsParallel: face vertex index out of range while orienting face DOFs");
                        }
                        local[i] = static_cast<int>(cell_verts[lv]);
                        global[i] = static_cast<int>(face_vertices[i]);
                    }

                    const auto orient = spaces::OrientationManager::quad_face_orientation(local, global);
                    const auto local_to_global = quad_interior_permutation_local_to_global(orient.vertex_perm);
                    if (static_cast<LocalIndex>(local_to_global.size()) != layout.dofs_per_face) {
                        throw FEException("DofHandler::distributeCGDofsParallel: quad face interior permutation size mismatch");
                    }

	                    for (LocalIndex l = 0; l < layout.dofs_per_face; ++l) {
	                        const int g = local_to_global[static_cast<std::size_t>(l)];
	                        cell_dofs.push_back(face_first_dof[sf] + static_cast<GlobalIndex>(g));
	                    }
	                } else {
	                    for (LocalIndex d = 0; d < layout.dofs_per_face; ++d) {
	                        cell_dofs.push_back(face_first_dof[sf] + d);
	                    }
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

    auto face_owner_by_ordinal = std::make_shared<std::unordered_map<gid_t, int>>();
    face_owner_by_ordinal->reserve(face_global_id.size());
    for (std::size_t i = 0; i < face_global_id.size(); ++i) {
        face_owner_by_ordinal->emplace(face_global_id[i], face_owner_rank[i]);
    }

    auto cell_owner_by_ordinal = std::make_shared<std::unordered_map<gid_t, int>>();
    cell_owner_by_ordinal->reserve(cell_global_id.size());
    for (std::size_t i = 0; i < cell_global_id.size(); ++i) {
        cell_owner_by_ordinal->emplace(cell_global_id[i], cell_owner_rank_out[i]);
    }

    const auto dofs_per_vertex = static_cast<gid_t>(layout.dofs_per_vertex);
    const auto dofs_per_edge = static_cast<gid_t>(layout.dofs_per_edge);
    const auto dofs_per_face = static_cast<gid_t>(layout.dofs_per_face);
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

        if (dofs_per_face > 0 && dof < offset + face_dofs_total) {
            const gid_t ord = (dof - offset) / dofs_per_face;
            const auto it = face_owner_by_ordinal->find(ord);
            return (it != face_owner_by_ordinal->end()) ? it->second : -1;
        }
        offset += face_dofs_total;

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
        static_cast<std::size_t>(std::max<GlobalIndex>(0, topology.n_faces)) * static_cast<std::size_t>(layout.dofs_per_face) +
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

    add_entity_dofs(vertex_first_dof, vertex_owner_rank, layout.dofs_per_vertex);
    add_entity_dofs(edge_first_dof, edge_owner_rank, layout.dofs_per_edge);
    add_entity_dofs(face_first_dof, face_owner_rank, layout.dofs_per_face);

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

    add_entity_ghosts(vertex_first_dof, vertex_owner_rank, layout.dofs_per_vertex);
    add_entity_ghosts(edge_first_dof, edge_owner_rank, layout.dofs_per_edge);
    add_entity_ghosts(face_first_dof, face_owner_rank, layout.dofs_per_face);

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
                   a.dofs_per_cell == b.dofs_per_cell &&
                   a.num_components == b.num_components &&
                   a.is_continuous == b.is_continuous &&
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
    layout.is_continuous = (continuity == Continuity::C0 || continuity == Continuity::C1);
    layout.num_components = space.value_dimension();

    const int order = space.polynomial_order();
    const auto total_dofs = static_cast<LocalIndex>(space.dofs_per_element());
    const auto cell0 = mesh.cell_vertices_span(0);
    const int n_verts = static_cast<int>(cell0.second);
    if (n_verts <= 0) {
        throw FEException("DofHandler::distributeDofs(MeshBase): cell 0 has no vertices");
    }

    if (layout.is_continuous) {
        layout = DofLayoutInfo::Lagrange(order, mesh.dim(), n_verts);
        layout.total_dofs_per_element = total_dofs;
        layout.num_components = space.value_dimension();
    } else {
        layout.dofs_per_vertex = 0;
        layout.dofs_per_edge = 0;
        layout.dofs_per_face = 0;
        const auto nc = static_cast<LocalIndex>(std::max(1, layout.num_components));
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
    topo.cell_gids = mesh.cell_gids();

    const bool need_edges = layout.is_continuous && layout.dofs_per_edge > 0;
    const bool need_faces = layout.is_continuous && layout.dofs_per_face > 0;

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
    layout.is_continuous = (continuity == Continuity::C0 || continuity == Continuity::C1);
    layout.num_components = space.value_dimension();

    const int order = space.polynomial_order();
    const auto total_dofs = static_cast<LocalIndex>(space.dofs_per_element());
    const auto cell0 = local_mesh.cell_vertices_span(0);
    const int n_verts = static_cast<int>(cell0.second);
    if (n_verts <= 0) {
        throw FEException("DofHandler::distributeDofs(Mesh): cell 0 has no vertices");
    }

    if (layout.is_continuous) {
        layout = DofLayoutInfo::Lagrange(order, local_mesh.dim(), n_verts);
        layout.total_dofs_per_element = total_dofs;
        layout.num_components = space.value_dimension();
    } else {
        layout.dofs_per_vertex = 0;
        layout.dofs_per_edge = 0;
        layout.dofs_per_face = 0;
        const auto nc = static_cast<LocalIndex>(std::max(1, layout.num_components));
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
    const bool need_faces = layout.is_continuous && layout.dofs_per_face > 0;

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
    const bool need_faces = layout.is_continuous && layout.dofs_per_face > 0;

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
}

void DofHandler::setPartition(DofPartition partition) {
    checkNotFinalized();
    partition_ = std::move(partition);
    ghost_cache_valid_ = false;
}

void DofHandler::setEntityDofMap(std::unique_ptr<EntityDofMap> entity_dof_map) {
    checkNotFinalized();
    entity_dof_map_ = std::move(entity_dof_map);
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
    rebuild_entity_map(perm);

    if (world_size_ > 1) {
        // Rebuild partition sets after renumbering.
        dof_map_.setNumLocalDofs(static_cast<GlobalIndex>(new_owned.size()));
        partition_ = DofPartition(IndexSet(std::move(new_owned)), IndexSet(new_ghost));
        partition_.setGlobalSize(n_dofs);

        // Update local ownership function to reflect new DOF IDs for locally relevant DOFs.
        auto owner_map_ptr = std::make_shared<std::unordered_map<GlobalIndex, int>>(std::move(owner_by_new_dof));
        dof_map_.setDofOwnership([owner_map_ptr](GlobalIndex dof) -> int {
            const auto it = owner_map_ptr->find(dof);
            return (it != owner_map_ptr->end()) ? it->second : -1;
        });

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
