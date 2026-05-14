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

#ifndef FSI_LINEAR_SOLVER_H 
#define FSI_LINEAR_SOLVER_H 

#include "CmMod.h"
#include "Array3.h"

#include "mpi.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <vector>

/// SELECTED_REAL_KIND(P,R) returns the kind value of a real data type with 
///
///  1) decimal precision of at least P digits, 
///
///  2) exponent range of at least R
///
/// \code {.f}
/// INTEGER, PARAMETER :: LSRP4  = SELECTED_REAL_KIND(6, 37)
/// INTEGER, PARAMETER :: LSRP8  = SELECTED_REAL_KIND(15, 307)      // C++ double
/// INTEGER, PARAMETER :: LSRP16 = SELECTED_REAL_KIND(33, 4931)     // C++ long double
/// INTEGER, PARAMETER :: LSRP = LSRP8
/// \endcode
///
/// The enums here replicate the PARAMETERs defined
/// in FSILS_STRUCT.h.
//
namespace fe_fsi_linear_solver {

/// Index type for large mesh support. Using int64_t allows meshes
/// with more than 2^31 nodes / non-zeros.
using fsils_int = int64_t;

enum class BcType
{
  BC_TYPE_Dir = 0,
  BC_TYPE_Neu = 1
};

enum class BcopType
{
  BCOP_TYPE_ADD = 0,
  BCOP_TYPE_PRE = 1
};

enum LinearSolverType
{
  LS_TYPE_CG = 798,
  LS_TYPE_GMRES = 797, 
  LS_TYPE_NS = 796, 
  LS_TYPE_BICGS = 795
};

enum class SchurPreconditionerType : std::uint8_t
{
  DIAG_L = 0,
  BLOCKDIAG_L = 1,
  ILU_L = 2,
  ALGEBRAIC_SHAT = 3
};

enum class SchurMomentumApproximationType : std::uint8_t
{
  DIAG_K = 0,
  BLOCKDIAG_K = 1,
  ILU_K = 2,
  ASM_K = 3
};

class FSILS_commuType 
{
  public:
    struct CollectiveStats {
      std::uint64_t allreduce_calls{0};
      std::uint64_t allreduce_words{0};
      double allreduce_time{0.0};
    };

    /// Free of created          (USE)
    int foC;         

    /// If this the master       (USE)    
    int masF;            
    
    /// Master ID                (USE)
    int master;          

    /// ID of this proc.         (USE)
    int task;            

    /// Task in FORTRAN indexing (USE)
    int tF;              

    /// Total number of tasks    (USE)
    int nTasks;          

    /// MPI communicator         (IN)
    MPI_Comm comm;       

    CollectiveStats collective_stats{};
};

inline void fsils_reset_collective_stats(FSILS_commuType& commu)
{
  commu.collective_stats = FSILS_commuType::CollectiveStats{};
}

inline void fsils_record_allreduce(FSILS_commuType& commu, int count, double duration)
{
  if (commu.nTasks <= 1 || count <= 0) {
    return;
  }
  commu.collective_stats.allreduce_calls += 1u;
  commu.collective_stats.allreduce_words += static_cast<std::uint64_t>(count);
  commu.collective_stats.allreduce_time += duration;
}

inline FSILS_commuType::CollectiveStats
fsils_collective_delta(const FSILS_commuType::CollectiveStats& before,
                       const FSILS_commuType::CollectiveStats& after) noexcept
{
  FSILS_commuType::CollectiveStats delta{};
  delta.allreduce_calls =
      (after.allreduce_calls >= before.allreduce_calls)
          ? (after.allreduce_calls - before.allreduce_calls)
          : 0u;
  delta.allreduce_words =
      (after.allreduce_words >= before.allreduce_words)
          ? (after.allreduce_words - before.allreduce_words)
          : 0u;
  delta.allreduce_time =
      (after.allreduce_time >= before.allreduce_time)
          ? (after.allreduce_time - before.allreduce_time)
          : 0.0;
  return delta;
}

inline int fsils_allreduce(const void* sendbuf,
                           void* recvbuf,
                           int count,
                           MPI_Datatype datatype,
                           MPI_Op op,
                           FSILS_commuType& commu)
{
  if (commu.nTasks <= 1 || count <= 0) {
    return MPI_SUCCESS;
  }
  const double tp0 = MPI_Wtime();
  const int rc = MPI_Allreduce(sendbuf, recvbuf, count, datatype, op, commu.comm);
  fsils_record_allreduce(commu, count, MPI_Wtime() - tp0);
  return rc;
}

inline int fsils_allreduce_in_place(void* buffer,
                                    int count,
                                    MPI_Datatype datatype,
                                    MPI_Op op,
                                    FSILS_commuType& commu)
{
  if (commu.nTasks <= 1 || count <= 0) {
    return MPI_SUCCESS;
  }
  const double tp0 = MPI_Wtime();
  const int rc = MPI_Allreduce(MPI_IN_PLACE, buffer, count, datatype, op, commu.comm);
  fsils_record_allreduce(commu, count, MPI_Wtime() - tp0);
  return rc;
}

inline int fsils_allreduce_sum(const void* sendbuf,
                               void* recvbuf,
                               int count,
                               MPI_Datatype datatype,
                               FSILS_commuType& commu)
{
  return fsils_allreduce(sendbuf, recvbuf, count, datatype, MPI_SUM, commu);
}

inline int fsils_allreduce_sum_in_place(void* buffer,
                                        int count,
                                        MPI_Datatype datatype,
                                        FSILS_commuType& commu)
{
  return fsils_allreduce_in_place(buffer, count, datatype, MPI_SUM, commu);
}

inline const char* fsils_solver_type_name(LinearSolverType type) noexcept
{
  switch (type) {
    case LinearSolverType::LS_TYPE_CG:
      return "CG";
    case LinearSolverType::LS_TYPE_GMRES:
      return "GMRES";
    case LinearSolverType::LS_TYPE_NS:
      return "BlockSchur";
    case LinearSolverType::LS_TYPE_BICGS:
      return "BICGS";
  }
  return "Unknown";
}

class FSILS_faceType
{
  public:
    /// Free or created                (USE)
    bool foC = false;

    /// Neu: P/Q coupling              (USE)
    bool coupledFlag = false;

    /// Neu: shared between proces     (USE)
    bool sharedFlag = false;

    /// Included in the computations   (IN)
    bool incFlag = false;

    /// Number of nodes                (IN)
    int nNo = 0;

    /// Degrees of freedom for val     (IN)
    int dof = 0;

    /// Dir/Neu                        (IN)
    BcType bGrp = BcType::BC_TYPE_Dir;

    /// Only for data alignment
    int reserved = 0;

    /// Global node number             (IN)
    Vector<int> glob;

    /// ||Sai||**2._LSRP                   (USE)
    double nS = 0.0;

    /// Neu: P = res*Q                 (IN)
    double res = 0.0;

    /// nodal Sai for Neu              (IN)
    Array<double> val;

    /// Neu W*Sai                      (TMP)
    Array<double> valM;
};

struct FSILS_reducedSparseEntry
{
    fsils_int node = -1;
    int full_component = -1;
    double value = 0.0;
};

class FSILS_reducedFieldUpdateType
{
  public:
    bool active = false;
    double sigma = 0.0;
    double nS = 0.0;
    int grouped_coupling_id = -1;
    std::vector<int> active_components;
    std::vector<FSILS_reducedSparseEntry> left;
    std::vector<FSILS_reducedSparseEntry> right;
    std::vector<FSILS_reducedSparseEntry> left_owned;
    std::vector<FSILS_reducedSparseEntry> right_owned;
    std::vector<FSILS_reducedSparseEntry> left_scaled;
    std::vector<FSILS_reducedSparseEntry> right_scaled;
    std::vector<FSILS_reducedSparseEntry> left_scaled_owned;
    std::vector<FSILS_reducedSparseEntry> right_scaled_owned;
    bool has_face_cache = false;
    FSILS_faceType left_face;
    FSILS_faceType right_face;
};

class FSILS_groupedBorderedFieldCouplingType
{
  public:
    bool active = false;
    int grouped_coupling_id = -1;
    std::vector<double> aux_matrix;
    std::vector<FSILS_reducedFieldUpdateType> modes;
    std::vector<FSILS_faceType> left_faces;
    std::vector<FSILS_faceType> right_faces;
    std::vector<double> add_dense_coeff;
    std::vector<double> pre_dense_coeff;
};

inline int fsils_reduced_local_component(const FSILS_reducedFieldUpdateType& update,
                                         int full_component,
                                         int current_dof,
                                         int system_dof) noexcept
{
  if (full_component < 0 || current_dof <= 0) {
    return -1;
  }

  if (system_dof <= 0) {
    system_dof = current_dof;
  }

  if (current_dof == system_dof) {
    return (full_component < current_dof) ? full_component : -1;
  }

  if (!update.active_components.empty()) {
    const int active_size =
        std::min(current_dof, static_cast<int>(update.active_components.size()));
    for (int i = 0; i < active_size; ++i) {
      if (update.active_components[static_cast<std::size_t>(i)] == full_component) {
        return i;
      }
    }
    return -1;
  }

  return (full_component < current_dof) ? full_component : -1;
}

/// @brief Modified in:
///
///  fsils_lhs_create()
//
class FSILS_lhsType 
{
  public:
    /// Free of created                     (USE)
    bool foC = false; 

    bool debug_active = false; 

    /// Global number of nodes              (IN)
    fsils_int gnNo = 0;

    /// Number of nodes                     (IN)
    fsils_int nNo = 0;

    /// Total degrees of freedom per node in the parent FE system.
    int system_dof = 0;

    /// Number of non-zero in lhs           (IN)
    fsils_int nnz = 0;

    ///  Number of faces                     (IN)
    int nFaces = 0;

    /// nNo of this proc                    (USE)
    fsils_int mynNo = 0;

    /// nNo of shared with lower proc       (USE)
    fsils_int shnNo = 0;

    /// True when the matrix stores PETSc-like owned rows only. Ghost nodes may
    /// still be present as local columns and vector halo entries.
    bool owned_row_operator = false;

    /// Explicit owner-to-ghost halo plan used by PETSc-like owned-row operators.
    ///
    /// Node ids are in FSILS internal ordering. `owned_halo_send_nodes[i]` are
    /// owned nodes whose values this rank sends to `owned_halo_neighbor_ranks[i]`.
    /// `owned_halo_recv_nodes[i]` are local ghost nodes overwritten by values
    /// received from that same rank.
    std::vector<int> owned_halo_neighbor_ranks;
    std::vector<std::vector<fsils_int>> owned_halo_send_nodes;
    std::vector<std::vector<fsils_int>> owned_halo_recv_nodes;
    mutable std::vector<double> owned_halo_send_buffer;
    mutable std::vector<double> owned_halo_recv_buffer;

    /// Column pointer                      (USE)
    Vector<fsils_int> colPtr;

    /// Row pointer                         (USE)
    Array<fsils_int> rowPtr;

    /// Diagonal pointer                    (USE)
    Vector<fsils_int> diagPtr;

    /// Mapping of nodes                    (USE)
    Vector<int> map;

    /// Global node id for each old/local node index (old ordering).
    Vector<int> gNodes;

    /// Stable FE global node id for each old/local node index. This is used
    /// only by diagnostics when the FSILS backend node id is partition-local.
    Vector<int> debug_global_nodes;

    FSILS_commuType commu;

    std::vector<FSILS_faceType> face;
    int native_face_rank_one_count = 0;
    std::vector<FSILS_reducedFieldUpdateType> reduced_updates;
    std::vector<FSILS_groupedBorderedFieldCouplingType> grouped_bordered_field_couplings;
    std::vector<int> native_face_pc_active_indices;
    std::vector<double> native_face_pc_dense_coeff;
    std::vector<int> reduced_update_pc_active_indices;
    std::vector<double> reduced_update_pc_inner_inv;
    bool use_exact_grouped_bordered_pre_in_add_bc_mul = false;
    bool use_reduced_face_cache_in_add_bc_mul = false;
};

class FSILS_subLsType 
{
  public:
    struct SolverStats {
      int solve_calls{0};
      int iterations_total{0};
      int max_iterations{0};
      int restart_cycles_total{0};
      int max_restart_cycles{0};
      std::uint64_t collective_calls{0};
      std::uint64_t collective_words{0};
      double collective_time{0.0};
      double setup_time{0.0};
      double solve_time{0.0};

      void reset() noexcept
      {
        *this = SolverStats{};
      }

      void record_call(int iterations,
                       int restart_cycles,
                       const FSILS_commuType::CollectiveStats& collective_delta,
                       double setup_seconds,
                       double solve_seconds) noexcept
      {
        solve_calls += 1;
        iterations_total += std::max(0, iterations);
        max_iterations = std::max(max_iterations, std::max(0, iterations));
        restart_cycles_total += std::max(0, restart_cycles);
        max_restart_cycles = std::max(max_restart_cycles, std::max(0, restart_cycles));
        collective_calls += collective_delta.allreduce_calls;
        collective_words += collective_delta.allreduce_words;
        collective_time += collective_delta.allreduce_time;
        setup_time += std::max(0.0, setup_seconds);
        solve_time += std::max(0.0, solve_seconds);
      }
    };

    /// Successful solving          (OUT)
    bool suc;       

    //int suc;       // Successful solving          (OUT)

    /// Maximum iteration           (IN)
    int mItr;      

    /// Space dimension             (IN)
    int sD;        

    /// Number of iteration         (OUT)
    int itr;       

    /// Number of Ax multiply       (OUT)
    int cM;        

    /// Number of |x| norms         (OUT)
    int cN;        

    /// Number of <x.y> dot products(OUT)
    int cD;        

    /// Only for data alignment     (-)
    int reserve;   

    /// Absolute tolerance          (IN)
    double absTol; 

    /// Relative tolerance          (IN)
    double relTol; 

    /// Initial norm of residual    (OUT)
    double iNorm;  

    /// Final norm of residual      (OUT)
    double fNorm;  

    /// Original RHS norm to use for convergence targets when the linear
    /// system has been row-scaled before entering the Krylov solver.
    double convergence_ref_norm{-1.0};

    /// Res. rduction in last itr.  (OUT)
    double dB;     

    /// Calling duration            (OUT)
    double callD;

    /// When true, do not terminate GMRES early on heuristic stagnation or
    /// adaptive-restart criteria before the requested tolerance is met.
    bool exact_convergence{false};

    /// When true, skip selective GMRES reorthogonalization for this solve.
    /// This is used only for severe-stall recovery paths where a larger
    /// Krylov space is more important than extra orthogonality work.
    bool disable_reorth{false};

    /// Schur-complement preconditioner selection used by the BlockSchur path.
    SchurPreconditionerType schur_preconditioner{SchurPreconditionerType::ALGEBRAIC_SHAT};

    /// Momentum-side approximation used when building algebraic Schur operators.
    SchurMomentumApproximationType schur_momentum_approximation{SchurMomentumApproximationType::ILU_K};

    /// Pre-allocated workspace arrays for iterative solvers.
    /// These are lazily resized on first use and reused on subsequent calls
    /// when the dimensions (nNo, dof, sD) remain unchanged.
    struct SolverWorkspace {
      int nNo = 0, dof = 0, sD_alloc = 0;
      Array<double> h;
      Array3<double> u3;        ///< Krylov basis for vector GMRES
      Array<double> u2;         ///< Krylov basis for scalar GMRES
      Array<double> X2, unCondU; ///< vector GMRES solution & precond workspace
      Vector<double> Xs;        ///< scalar GMRES solution
      Vector<double> y, c, s, err;

      // GMRES scratch (avoid per-call allocations).
      std::vector<double> h_col;
      std::vector<double> dot_thread;
      std::vector<double> basis_panel_v;

      // Recycled/deflation subspace for GMRES (vector version).
      // Stores U (correction space) and C = A*U (orthonormal columns) for
      // deflated/recycling GMRES variants. Both are stored in the same
      // (dof, nNo) layout as the GMRES unknown.
      Array3<double> recycle_U3;
      Array3<double> recycle_C3;
      Vector<double> recycle_y;
      std::vector<double> recycle_score;
      std::vector<int> recycle_drop_streak;
      int recycle_v_nNo = 0, recycle_v_dof = 0;
      int recycle_k_alloc = 0;
      int recycle_k = 0;

      /// Ensure dot-thread scratch has at least nthreads*stride doubles.
      void ensure_gmres_dot_thread(const int nthreads, const int stride)
      {
        if (nthreads <= 0 || stride <= 0) {
          return;
        }
        const size_t needed = static_cast<size_t>(nthreads) * static_cast<size_t>(stride);
        if (dot_thread.size() < needed) {
          dot_thread.resize(needed);
        }
      }

      /// Ensure vector-GMRES tile-major basis shadow storage is available.
      /// Layout is scalar-entry major: panel[entry * (sD + 1) + basis_index].
      void ensure_gmres_basis_panel_v(const int dof_, const int nNo_, const int sD_)
      {
        if (dof_ <= 0 || nNo_ <= 0 || sD_ < 0) {
          return;
        }
        const std::size_t entries = static_cast<std::size_t>(dof_) * static_cast<std::size_t>(nNo_);
        const std::size_t stride = static_cast<std::size_t>(sD_ + 1);
        const std::size_t needed = entries * stride;
        if (basis_panel_v.size() < needed) {
          basis_panel_v.resize(needed);
        }
      }

      /// Ensure workspace arrays are allocated for given dimensions.
      /// Only reallocates when dimensions change (typically once).
      void ensure_gmres_v(int dof_, int nNo_, int sD_) {
        if (dof_ == dof && nNo_ == nNo && sD_ == sD_alloc) return;
        h.resize(sD_+1, sD_);
        u3.resize(dof_, nNo_, sD_+1);
        X2.resize(dof_, nNo_);
        unCondU.resize(dof_, nNo_);
        y.resize(sD_); c.resize(sD_); s.resize(sD_); err.resize(sD_+1);
        h_col.resize(sD_+2);
        dof = dof_; nNo = nNo_; sD_alloc = sD_;
      }

      void ensure_gmres_s(int nNo_, int sD_) {
        if (dof == -1 && nNo_ == nNo && sD_ == sD_alloc) return;
        h.resize(sD_+1, sD_);
        u2.resize(nNo_, sD_+1);
        Xs.resize(nNo_);
        y.resize(sD_); c.resize(sD_); s.resize(sD_); err.resize(sD_+1);
        h_col.resize(sD_+2);
        dof = -1; nNo = nNo_; sD_alloc = sD_;
      }

      void ensure_recycle_v(int dof_, int nNo_, int k_) {
        if (k_ <= 0) {
          recycle_k = 0;
          return;
        }
        if (recycle_v_dof == dof_ && recycle_v_nNo == nNo_ && recycle_k_alloc >= k_) {
          if (recycle_score.size() < static_cast<size_t>(recycle_k_alloc)) {
            recycle_score.resize(static_cast<size_t>(recycle_k_alloc), 0.0);
          }
          if (recycle_drop_streak.size() < static_cast<size_t>(recycle_k_alloc)) {
            recycle_drop_streak.resize(static_cast<size_t>(recycle_k_alloc), 0);
          }
          return;
        }
        recycle_U3.resize(dof_, nNo_, k_);
        recycle_C3.resize(dof_, nNo_, k_);
        recycle_y.resize(k_);
        recycle_score.assign(static_cast<size_t>(k_), 0.0);
        recycle_drop_streak.assign(static_cast<size_t>(k_), 0);
        recycle_v_dof = dof_;
        recycle_v_nNo = nNo_;
        recycle_k_alloc = k_;
        recycle_k = 0;
      }

      // BiCGS workspace (vector version)
      Array<double> bicgs_P, bicgs_Rh, bicgs_X, bicgs_V, bicgs_S, bicgs_T;
      int bicgs_v_nNo = 0, bicgs_v_dof = 0;

      void ensure_bicgs_v(int dof_, int nNo_) {
        if (dof_ == bicgs_v_dof && nNo_ == bicgs_v_nNo) return;
        bicgs_P.resize(dof_, nNo_); bicgs_Rh.resize(dof_, nNo_);
        bicgs_X.resize(dof_, nNo_); bicgs_V.resize(dof_, nNo_);
        bicgs_S.resize(dof_, nNo_); bicgs_T.resize(dof_, nNo_);
        bicgs_v_dof = dof_; bicgs_v_nNo = nNo_;
      }

      // BiCGS workspace (scalar version)
      Vector<double> bicgs_Ps, bicgs_Rhs, bicgs_Xs, bicgs_Vs, bicgs_Ss, bicgs_Ts;
      int bicgs_s_nNo = 0;

      void ensure_bicgs_s(int nNo_) {
        if (nNo_ == bicgs_s_nNo) return;
        bicgs_Ps.resize(nNo_); bicgs_Rhs.resize(nNo_);
        bicgs_Xs.resize(nNo_); bicgs_Vs.resize(nNo_);
        bicgs_Ss.resize(nNo_); bicgs_Ts.resize(nNo_);
        bicgs_s_nNo = nNo_;
      }

      // CG workspace (vector version)
      Array<double> cg_P, cg_KP, cg_X;
      int cg_v_nNo = 0, cg_v_dof = 0;

      void ensure_cg_v(int dof_, int nNo_) {
        if (dof_ == cg_v_dof && nNo_ == cg_v_nNo) return;
        cg_P.resize(dof_, nNo_); cg_KP.resize(dof_, nNo_);
        cg_X.resize(dof_, nNo_);
        cg_v_dof = dof_; cg_v_nNo = nNo_;
      }

      // CG workspace (scalar version)
      Vector<double> cg_Ps, cg_KPs, cg_Xs;
      int cg_s_nNo = 0;

      void ensure_cg_s(int nNo_) {
        if (nNo_ == cg_s_nNo) return;
        cg_Ps.resize(nNo_); cg_KPs.resize(nNo_);
        cg_Xs.resize(nNo_);
        cg_s_nNo = nNo_;
      }
    } ws;

    SolverStats stats{};
};

class FSILS_lsType 
{
  public:
    struct BlockSchurStats {
      int outer_iterations{0};
      std::uint64_t collective_calls_total{0};
      std::uint64_t collective_calls_max_per_outer{0};
      std::uint64_t collective_words_total{0};
      std::uint64_t collective_words_max_per_outer{0};
      double collective_time_total{0.0};
      double collective_time_max_per_outer{0.0};

      void reset() noexcept
      {
        *this = BlockSchurStats{};
      }

      void record_outer_iteration(const FSILS_commuType::CollectiveStats& collective_delta) noexcept
      {
        outer_iterations += 1;
        collective_calls_total += collective_delta.allreduce_calls;
        collective_calls_max_per_outer =
            std::max(collective_calls_max_per_outer, collective_delta.allreduce_calls);
        collective_words_total += collective_delta.allreduce_words;
        collective_words_max_per_outer =
            std::max(collective_words_max_per_outer, collective_delta.allreduce_words);
        collective_time_total += collective_delta.allreduce_time;
        collective_time_max_per_outer =
            std::max(collective_time_max_per_outer, collective_delta.allreduce_time);
      }
    };

    /// Free of created             (USE)
    int foC;                     

    /// Which one of LS             (IN)
    LinearSolverType LS_type;    

    /// Contribution of mom. res.   (OUT)
    int Resm;                    

    /// Contribution of cont. res.  (OUT)
    int Resc;                    

    /// When true, fsils_solve() treats the Ri array as already stored in the
    /// internal FSILS node ordering and skips the old->internal / internal->old
    /// permutation copies around the solver kernel.
    bool ri_internal_order{false};
    
    FSILS_subLsType GM;
    FSILS_subLsType CG;
    FSILS_subLsType RI;
    BlockSchurStats blockschur_stats{};

    struct SolveWorkspace {
      int dof{0};
      fsils_int nNo{0};
      Array<double> R;
      Array<double> Wr;
      Array<double> Wc;

      void ensure(int dof_, fsils_int nNo_)
      {
        if (dof_ == dof && nNo_ == nNo) {
          return;
        }
        R.resize(dof_, nNo_);
        Wr.resize(dof_, nNo_);
        Wc.resize(dof_, nNo_);
        dof = dof_;
        nNo = nNo_;
      }
    } solve_ws;

    /// Block layout for fractional-step (BlockSchur) solver.
    /// Populated by the FE backend before calling fsils_solve.
    /// When mom_ncomp > 0, the block solver uses these indices instead of the
    /// legacy assumption nsd = dof - 1 (momentum = first dof-1, constraint = last).
    int mom_start{0};    ///< First per-node component of field-A (momentum) block
    int mom_ncomp{0};    ///< Number of field-A components (0 = use legacy nsd = dof-1)
    int con_start{0};    ///< First per-node component of field-B (constraint) block
    int con_ncomp{0};    ///< Number of field-B components (typically 1)
    int blockschur_min_outer_iterations{0};
    bool use_coupled_outer_fgmres_scalar{false};
};


};

#endif
