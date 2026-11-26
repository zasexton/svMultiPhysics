/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#pragma once

#include "Core/FEException.h"

#include <array>
#include <iomanip>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <string_view>

#if defined(FE_HAS_TRILINOS)
// Tpetra can own an MPI communicator even in a serial FE configuration.
#include <Teuchos_config.h>
#endif
#if FE_HAS_MPI || (defined(FE_HAS_FSILS) && FE_HAS_FSILS) || \
    (defined(FE_HAS_TRILINOS) && defined(HAVE_TEUCHOS_MPI))
#define SVMP_FE_MATRIX_ASSEMBLY_MPI
#include <mpi.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

namespace svmp::FE::backends {

// A worker or a single MPI rank cannot throw across the next assembly barrier.
// Retain its first failing owned coefficient until collective finalization;
// zeroing or rebuilding the matrix starts a fresh assembly.
class MatrixAssemblyError {
public:
    void reject(std::string_view backend, GlobalIndex row, GlobalIndex col, Real value,
                bool distributed)
    {
        const Entry entry{row, col, value};
        bool defer = distributed;
#ifdef _OPENMP
        defer = defer || omp_get_level() > 0;
#endif
        if (!defer) {
            fail(backend, entry, 0);
        }
        std::lock_guard lock(mutex_);
        if (!first_) {
            first_ = entry;
        }
    }

#ifdef SVMP_FE_MATRIX_ASSEMBLY_MPI
    void check(std::string_view backend, MPI_Comm comm = MPI_COMM_NULL) const
#else
    void check(std::string_view backend) const
#endif
    {
        std::optional<Entry> entry;
        {
            std::lock_guard lock(mutex_);
            entry = first_;
        }
        int rank = 0;
#ifdef SVMP_FE_MATRIX_ASSEMBLY_MPI
        // A null communicator also covers serial matrices constructed before MPI_Init.
        if (comm != MPI_COMM_NULL) {
            int size = 1;
            MPI_Comm_size(comm, &size);
            if (size > 1) {
                MPI_Comm_rank(comm, &rank);
                const int local_source = entry ? rank : size;
                int source = size;
                MPI_Allreduce(&local_source, &source, 1, MPI_INT, MPI_MIN, comm);
                if (source == size) {
                    return;
                }
                std::array<long long, 2> location{};
                double coefficient = 0;
                if (rank == source) {
                    location = {static_cast<long long>(entry->row), static_cast<long long>(entry->column)};
                    coefficient = static_cast<double>(entry->value);
                }
                MPI_Bcast(location.data(), 2, MPI_LONG_LONG_INT, source, comm);
                MPI_Bcast(&coefficient, 1, MPI_DOUBLE, source, comm);
                fail(backend, {static_cast<GlobalIndex>(location[0]), static_cast<GlobalIndex>(location[1]),
                               static_cast<Real>(coefficient)},
                     source);
            }
        }
#endif
        if (entry) {
            fail(backend, *entry, rank);
        }
    }

    void reset()
    {
        std::lock_guard lock(mutex_);
        first_.reset();
    }

private:
    struct Entry {
        GlobalIndex row, column;
        Real value;
    };

    [[noreturn]] static void fail(std::string_view backend, const Entry& entry, int rank)
    {
        std::ostringstream message;
        message << backend << " matrix assembly on rank " << rank << ": coefficient "
                << std::setprecision(std::numeric_limits<Real>::max_digits10) << entry.value
                << " at owned FE row " << entry.row << ", column " << entry.column
                << " is missing from the matrix sparsity pattern";
        FE_THROW(FEException, message.str());
    }

    mutable std::mutex mutex_;
    std::optional<Entry> first_;
};

} // namespace svmp::FE::backends

#undef SVMP_FE_MATRIX_ASSEMBLY_MPI
