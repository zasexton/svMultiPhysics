#ifndef SVMP_FE_CORE_MPICOLLECTIVETRACE_H
#define SVMP_FE_CORE_MPICOLLECTIVETRACE_H

#include "Core/FEConfig.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if FE_HAS_MPI
#  include <mpi.h>
#endif

namespace svmp {
namespace FE {
namespace debug {

[[nodiscard]] inline bool mpiCollectiveTraceEnabled() noexcept
{
    const char* env = std::getenv("SVMP_MPI_COLLECTIVE_TRACE");
    return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
}

[[nodiscard]] inline std::uint64_t nextMpiCollectiveTraceSeq() noexcept
{
    static std::atomic<std::uint64_t> seq{0};
    return seq.fetch_add(1, std::memory_order_relaxed) + 1u;
}

#if FE_HAS_MPI
[[nodiscard]] inline const char* mpiDatatypeName(MPI_Datatype datatype) noexcept
{
    if (datatype == MPI_INT) return "MPI_INT";
    if (datatype == MPI_DOUBLE) return "MPI_DOUBLE";
    if (datatype == MPI_UNSIGNED_LONG_LONG) return "MPI_UNSIGNED_LONG_LONG";
    if (datatype == MPI_LONG_DOUBLE) return "MPI_LONG_DOUBLE";
    if (datatype == MPI_CHAR) return "MPI_CHAR";
    if (datatype == MPI_INTEGER) return "MPI_INTEGER";
    return "MPI_DATATYPE_UNKNOWN";
}

inline void traceMpiCollective(const char* phase,
                               std::uint64_t seq,
                               const char* label,
                               int count,
                               MPI_Datatype datatype,
                               MPI_Op,
                               MPI_Comm comm,
                               int rc = MPI_SUCCESS) noexcept
{
    if (!mpiCollectiveTraceEnabled()) {
        return;
    }

    int initialized = 0;
    MPI_Initialized(&initialized);
    int rank = -1;
    int size = -1;
    if (initialized && comm != MPI_COMM_NULL) {
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);
    }

    int datatype_size = 0;
    if (initialized) {
        MPI_Type_size(datatype, &datatype_size);
    }

    std::fprintf(stderr,
                 "[MPI_TRACE] phase=%s seq=%llu rank=%d size=%d label=%s count=%d datatype=%s datatype_size=%d rc=%d\n",
                 phase,
                 static_cast<unsigned long long>(seq),
                 rank,
                 size,
                 label != nullptr ? label : "<unknown>",
                 count,
                 mpiDatatypeName(datatype),
                 datatype_size,
                 rc);
    std::fflush(stderr);
}
#endif

} // namespace debug
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CORE_MPICOLLECTIVETRACE_H
