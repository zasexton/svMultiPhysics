/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef SVMP_MESH_COMM_H
#define SVMP_MESH_COMM_H

#include "MeshTypes.h"

#ifdef MESH_HAS_MPI
#include <mpi.h>
#endif

namespace svmp {

/**
 * @brief Small communicator wrapper that exists in both serial and MPI builds.
 *
 * This type allows user code to pass a "communicator-like" object without
 * needing `#ifdef MESH_HAS_MPI` guards. In MPI builds it wraps an `MPI_Comm`.
 * In serial builds it behaves as a single-rank communicator.
 */
class MeshComm {
public:
  MeshComm() = default;

#ifdef MESH_HAS_MPI
  explicit MeshComm(MPI_Comm comm) : comm_(comm) {
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized && comm_ != MPI_COMM_NULL) {
      MPI_Comm_rank(comm_, &rank_);
      MPI_Comm_size(comm_, &size_);
    }
  }
#endif

  static MeshComm self() {
#ifdef MESH_HAS_MPI
    return MeshComm(MPI_COMM_SELF);
#else
    return MeshComm();
#endif
  }

  static MeshComm world() {
#ifdef MESH_HAS_MPI
    return MeshComm(MPI_COMM_WORLD);
#else
    return MeshComm();
#endif
  }

  rank_t rank() const noexcept { return rank_; }
  int size() const noexcept { return size_; }
  bool is_parallel() const noexcept { return size_ > 1; }

#ifdef MESH_HAS_MPI
  MPI_Comm native() const noexcept { return comm_; }
#endif

private:
#ifdef MESH_HAS_MPI
  MPI_Comm comm_ = MPI_COMM_SELF;
#endif
  rank_t rank_ = 0;
  int size_ = 1;
};

} // namespace svmp

#endif // SVMP_MESH_COMM_H
