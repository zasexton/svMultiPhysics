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

class FSILS_commuType 
{
  public:
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
};

class FSILS_cSType
{
  public:
    /// The processor to communicate with
    int iP = -1;

    /// Number of data to be commu
    int n = 0;

    /// Pointer to the data for commu
    Vector<int> ptr;
};

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

    /// Number of non-zero in lhs           (IN)
    fsils_int nnz = 0;

    ///  Number of faces                     (IN)
    int nFaces = 0;

    /// nNo of this proc                    (USE)
    fsils_int mynNo = 0;

    /// nNo of shared with lower proc       (USE)
    fsils_int shnNo = 0;

    /// Number of communication requests    (USE)
    int nReq = 0;

    /// Column pointer                      (USE)
    Vector<fsils_int> colPtr;

    /// Row pointer                         (USE)
    Array<fsils_int> rowPtr;

    /// Diagonal pointer                    (USE)
    Vector<fsils_int> diagPtr;

    /// Mapping of nodes                    (USE)
    Vector<int> map;

    FSILS_commuType commu;

    std::vector<FSILS_cSType> cS;

    std::vector<FSILS_faceType> face;

    /// Pre-allocated communication buffers (sized once in fsils_lhs_create)
    int nmax_commu = 0;      ///< max shared nodes across all comm partners
    mutable std::vector<MPI_Request> commu_sReq;
    mutable std::vector<MPI_Request> commu_rReq;
    mutable std::vector<double> commu_sB;  ///< send buffer (flat, sized for max usage)
    mutable std::vector<double> commu_rB;  ///< recv buffer (flat, sized for max usage)
    int commu_dof_capacity = 0;    ///< current dof capacity for vector buffers
};

class FSILS_subLsType 
{
  public:
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

    /// Res. rduction in last itr.  (OUT)
    double dB;     

    /// Calling duration            (OUT)
    double callD;

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

      /// Ensure workspace arrays are allocated for given dimensions.
      /// Only reallocates when dimensions change (typically once).
      void ensure_gmres_v(int dof_, int nNo_, int sD_) {
        if (dof_ == dof && nNo_ == nNo && sD_ == sD_alloc) return;
        h.resize(sD_+1, sD_);
        u3.resize(dof_, nNo_, sD_+1);
        X2.resize(dof_, nNo_);
        unCondU.resize(dof_, nNo_);
        y.resize(sD_); c.resize(sD_); s.resize(sD_); err.resize(sD_+1);
        dof = dof_; nNo = nNo_; sD_alloc = sD_;
      }

      void ensure_gmres_s(int nNo_, int sD_) {
        if (dof == -1 && nNo_ == nNo && sD_ == sD_alloc) return;
        h.resize(sD_+1, sD_);
        u2.resize(nNo_, sD_+1);
        Xs.resize(nNo_);
        y.resize(sD_); c.resize(sD_); s.resize(sD_); err.resize(sD_+1);
        dof = -1; nNo = nNo_; sD_alloc = sD_;
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
};

class FSILS_lsType 
{
  public:
    /// Free of created             (USE)
    int foC;                     

    /// Which one of LS             (IN)
    LinearSolverType LS_type;    

    /// Contribution of mom. res.   (OUT)
    int Resm;                    

    /// Contribution of cont. res.  (OUT)
    int Resc;                    
    
    FSILS_subLsType GM;
    FSILS_subLsType CG;
    FSILS_subLsType RI;
};


};

#endif

