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

// The functions here reproduce the subroutines defined in svFSILS/LHS.f.

#include "lhs.h"
#include "CmMod.h"
#include "DebugMsg.h"

#include "mpi.h"
#include <unordered_map>

namespace fe_fsi_linear_solver {

/// @brief Modifies:
///
///  lhs.foC 
///  lhs.gnNo 
///  lhs.nNo 
///  lhs.nnz 
///  lhs.commu 
///  lhs.nFaces 
///  lhs.mynNo 
///
///  lhs.colPtr
///  lhs.rowPtr
///  lhs.diagPtr
///  lhs.map
///  lhs.face
//
void fsils_lhs_create(FSILS_lhsType& lhs, FSILS_commuType& commu, int gnNo, int nNo, int nnz, Vector<int>& gNodes,  
       Vector<int> &rowPtr, Vector<int>& colPtr, int nFaces)
{
  #define n_debug_fsils_lhs_create
  #ifdef debug_fsils_lhs_create
  DebugMsg dmsg(__func__,  lhs.commu.task);
  dmsg.banner();
  #endif

  lhs.foC = true; 
  lhs.gnNo = gnNo;
  lhs.nNo = nNo;
  lhs.nnz = nnz;
  lhs.commu = commu;
  lhs.nFaces = nFaces;
  #ifdef debug_fsils_lhs_create
  dmsg << "gnNo: " << gnNo;
  dmsg << "nNo: " << nNo;
  dmsg << "nnz: " << nnz;
  dmsg << "nFaces: " << nFaces;
  #endif

  int nTasks = commu.nTasks;
  auto comm = commu.comm;
  auto tF = commu.tF;
  #ifdef debug_fsils_lhs_create
  dmsg << "nTasks: " << nTasks;
  dmsg << "tF: " << tF;
  #endif

  lhs.colPtr.resize(nnz); 
  lhs.rowPtr.resize(2,nNo); 
  lhs.diagPtr.resize(nNo);
  lhs.map.resize(nNo); 
  lhs.face.resize(nFaces);

  // For a sequential simulation. 
  //
  if (nTasks == 1) {

    for (fsils_int i = 0; i < nnz; i++) {
      lhs.colPtr(i) = colPtr(i);
    }

    for (fsils_int Ac = 0; Ac < nNo; Ac++) {
      fsils_int s = rowPtr(Ac);
      fsils_int e = rowPtr(Ac+1) - 1;

      for (fsils_int i = s; i <= e; i++) {
        int a = colPtr(i);
        if (Ac == a) {
          lhs.diagPtr(Ac) = i;
          break;
        }
      }

      lhs.rowPtr(0,Ac) = s;
      lhs.rowPtr(1,Ac) = e;
      lhs.map(Ac) = Ac;
    }

    lhs.mynNo = nNo;
    return; 
  }

  // Phase 6A: Scalable P2P boundary discovery
  // Step 1: Exchange node count and range (min/max global ID) per rank
  // instead of the full O(maxnNo * nTasks) Allgatherv.
  //
  int my_info[3]; // {nNo, min_gid, max_gid}
  my_info[0] = nNo;
  my_info[1] = gNodes(0);
  my_info[2] = gNodes(0);
  for (int i = 1; i < nNo; i++) {
    if (gNodes(i) < my_info[1]) my_info[1] = gNodes(i);
    if (gNodes(i) > my_info[2]) my_info[2] = gNodes(i);
  }
  std::vector<int> all_info(3 * nTasks);
  MPI_Allgather(my_info, 3, cm_mod::mpint, all_info.data(), 3, cm_mod::mpint, comm);

  // Step 2: Identify candidate neighbors (overlapping global ID ranges)
  std::vector<int> candidates;
  for (int i = 0; i < nTasks; i++) {
    if (i == tF) continue;
    int other_min = all_info[3*i + 1];
    int other_max = all_info[3*i + 2];
    if (other_min <= my_info[2] && other_max >= my_info[1]) {
      candidates.push_back(i);
    }
  }

  // Step 3: Exchange actual node lists only with candidates via P2P
  int maxnNo = 0;
  for (int i = 0; i < nTasks; i++) {
    if (all_info[3*i] > maxnNo) maxnNo = all_info[3*i];
  }

  // Allocate storage only for candidates + self
  int nCandidates = static_cast<int>(candidates.size());
  // Map from rank -> candidate index (or -1)
  std::vector<int> rank_to_cand(nTasks, -1);
  int self_cand_idx = nCandidates; // self stored at index nCandidates
  for (int c = 0; c < nCandidates; c++) {
    rank_to_cand[candidates[c]] = c;
  }
  rank_to_cand[tF] = self_cand_idx;

  // aNodes now sized (maxnNo, nCandidates+1) — candidates + self
  Array<int> aNodes(maxnNo, nCandidates + 1);
  aNodes = -1;

  // Fill self column
  for (int i = 0; i < nNo; i++) {
    aNodes(i, self_cand_idx) = gNodes(i);
  }

  // P2P exchange with candidates
  std::vector<MPI_Request> send_reqs(nCandidates);
  std::vector<MPI_Request> recv_reqs(nCandidates);
  // Send our node list, receive theirs
  Vector<int> send_buf(maxnNo);
  send_buf = -1;
  for (int i = 0; i < nNo; i++) {
    send_buf(i) = gNodes(i);
  }

  for (int c = 0; c < nCandidates; c++) {
    int other_nNo = all_info[3*candidates[c]];
    MPI_Irecv(&aNodes(0, c), other_nNo, cm_mod::mpint, candidates[c], 2, comm, &recv_reqs[c]);
    MPI_Isend(send_buf.data(), nNo, cm_mod::mpint, candidates[c], 2, comm, &send_reqs[c]);
  }

  if (nCandidates > 0) {
    MPI_Waitall(nCandidates, recv_reqs.data(), MPI_STATUSES_IGNORE);
  }

  Vector<int> ltg(nNo);

  std::unordered_map<int,int> gtlPtr;
  gtlPtr.reserve(nNo);

  for (int a = 0; a < nNo; a++) {
     int Ac = gNodes(a);
     gtlPtr[Ac] = a;
  }

  // Including the nodes shared by processors with higher ID at the end,
  // and including the nodes shared by lower processors IDs at the front.
  // shnNo is counter for lower ID and mynNo is counter for higher ID
  //
  lhs.mynNo = nNo;
  lhs.shnNo = 0;
  #ifdef debug_fsils_lhs_create
  dmsg << "lhs.mynNo: " << lhs.mynNo;
  dmsg << "tF: " << tF;
  #endif

  // Iterate candidates from highest rank to lowest (reverse order)
  for (int c = nCandidates - 1; c >= 0; c--) {
    int rank_i = candidates[c];
    int other_nNo = all_info[3*rank_i];

    for (int a = 0; a < other_nNo; a++) {
      // Global node number in processor rank_i at location a
      int Ac = aNodes(a, c);
      // Exit if this is the last node
      if (Ac == -1) {
        break;
      }

      // Corresponding local node in current processor.
      auto gtl_it = gtlPtr.find(Ac);
      if (gtl_it != gtlPtr.end()) {
        int localNodeIndex = gtl_it->second;
        // If this node has not been included already
        if (aNodes(localNodeIndex, self_cand_idx) != -1) {
          // If the processor ID is lower, it is appended to the beginning
          if (rank_i < tF) {
            ltg(lhs.shnNo) = Ac;
            lhs.shnNo = lhs.shnNo + 1;
          // If the processor ID is higher, it is appended to the end
          } else {
            ltg(lhs.mynNo-1) = Ac;
            lhs.mynNo = lhs.mynNo - 1;
          }
          aNodes(localNodeIndex, self_cand_idx) = -1;
        }
      }
    }
  }

  #ifdef debug_fsils_lhs_create
  dmsg << "lhs.shnNo: " << lhs.shnNo;
  dmsg << "lhs.mynNo: " << lhs.mynNo;
  #endif

  // Now including the local nodes that are left behind
  //
  int j = lhs.shnNo + 1;

  for (int a = 0; a < nNo; a++) {
    int Ac = aNodes(a, self_cand_idx);
    //  If this node has not been included already
    if (Ac != -1) {
      ltg(j-1) = Ac;
      j = j + 1;
    }
  }

  if (j != lhs.mynNo+1) {
    throw std::runtime_error("FSILS: Unexpected behavior: j=" + std::to_string(j) + " lhs.mynNo: " + std::to_string(lhs.mynNo) + ".");
    MPI_Finalize();
  }

  // Having the new ltg pointer, map is constructed
  //
  gtlPtr.clear();
  gtlPtr.reserve(nNo);
  for (int a = 0; a < nNo; a++) {
     int Ac = ltg(a);
     gtlPtr[Ac] = a;
  }

  for (int a = 0; a < nNo; a++) {
     int Ac = gNodes(a);
     lhs.map(a) = gtlPtr[Ac];
  }

  // Based on the new ordering of the nodes, rowPtr and colPtr are constructed
  //
  for (fsils_int a = 0; a < nNo; a++) {
    int Ac = lhs.map(a);
    lhs.rowPtr(0,Ac) = rowPtr(a);
    lhs.rowPtr(1,Ac) = rowPtr(a+1) - 1;
  }

  for (fsils_int i = 0; i < nnz; i++) {
    lhs.colPtr(i) = lhs.map(colPtr(i));
  }

  // diagPtr points to the diagonal entries of LHS
  for (fsils_int Ac = 0; Ac < nNo; Ac++) {
    for (fsils_int i = lhs.rowPtr(0,Ac); i <= lhs.rowPtr(1,Ac); i++) {
      fsils_int a = lhs.colPtr(i);
      if (Ac == a) {
        lhs.diagPtr(Ac) = i;
        break;
      }
    }
  }

  // Constructing the communication data structure based on the ltg
  // P2P exchange of reordered ltg with candidates
  //
  for (int i = 0; i < nNo; i++) {
    aNodes(i, self_cand_idx) = ltg(i);
  }
  for (int i = nNo; i < maxnNo; i++) {
    aNodes(i, self_cand_idx) = -1;
  }

  // Wait for first-round sends to complete before reusing buffers
  if (nCandidates > 0) {
    MPI_Waitall(nCandidates, send_reqs.data(), MPI_STATUSES_IGNORE);
  }

  for (int c = 0; c < nCandidates; c++) {
    int other_nNo = all_info[3*candidates[c]];
    MPI_Irecv(&aNodes(0, c), other_nNo, cm_mod::mpint, candidates[c], 3, comm, &recv_reqs[c]);
    MPI_Isend(&aNodes(0, self_cand_idx), nNo, cm_mod::mpint, candidates[c], 3, comm, &send_reqs[c]);
  }

  if (nCandidates > 0) {
    MPI_Waitall(nCandidates, recv_reqs.data(), MPI_STATUSES_IGNORE);
  }

  // Count shared nodes with each candidate
  std::vector<int> shared_count(nCandidates, 0);
  lhs.nReq = 0;

  for (int c = 0; c < nCandidates; c++) {
    int other_nNo = all_info[3*candidates[c]];
    for (int a = 0; a < other_nNo; a++) {
      int Ac = aNodes(a, c);
      if (Ac == -1) {
        break;
      }
      if (gtlPtr.count(Ac)) {
        shared_count[c]++;
      }
    }

    if (shared_count[c] != 0) {
      lhs.nReq = lhs.nReq + 1;
    }
  }
  #ifdef debug_fsils_lhs_create
  dmsg << "lhs.nReq: " << lhs.nReq;
  #endif

  lhs.cS.resize(lhs.nReq);

  // Now that we know which processor is communicating to which, we can
  // setup the handles and structures
  //
  // lhs.cS[j].iP is the processor to communicate with.
  //
  #ifdef debug_fsils_lhs_create
  dmsg << "Setup the handles ...";
  #endif
  j = 0;
  for (int c = 0; c < nCandidates; c++) {
    int a = shared_count[c];
    if (a != 0) {
      lhs.cS[j].iP = candidates[c];
      lhs.cS[j].n = a;
      lhs.cS[j].ptr.resize(a);
      j = j + 1;
    }
  }

  // Pre-allocate communication buffers
  if (lhs.nReq > 0) {
    lhs.nmax_commu = std::max_element(lhs.cS.begin(), lhs.cS.end(),
        [](const FSILS_cSType& a, const FSILS_cSType& b){ return a.n < b.n; })->n;
    lhs.commu_sReq.resize(lhs.nReq);
    lhs.commu_rReq.resize(lhs.nReq);
    // Pre-allocate scalar comm buffers (nmax * nReq)
    size_t scalar_sz = static_cast<size_t>(lhs.nmax_commu) * lhs.nReq;
    lhs.commu_sB.resize(scalar_sz);
    lhs.commu_rB.resize(scalar_sz);
    lhs.commu_dof_capacity = 1;
  }

  // Order of nodes in ptr is based on the node order in processor
  // with higher ID. ptr is calculated for tF+1:nTasks and will be
  // sent over.
  //
  #ifdef debug_fsils_lhs_create
  dmsg << "Order of nodes ...";
  #endif
  MPI_Status status;

  for (int i = 0; i < lhs.nReq; i++) {
    int iP = lhs.cS[i].iP;

    if (iP < tF) {
      MPI_Recv(lhs.cS[i].ptr.data(), lhs.cS[i].n, cm_mod::mpint, iP, 1,  comm, &status);

      for (int j = 0; j < lhs.cS[i].n; j++) {
        lhs.cS[i].ptr[j] = gtlPtr[lhs.cS[i].ptr[j]];
      }
    } else {
      // This is a counter for the shared nodes
      j = 0;
      int cand_idx = rank_to_cand[iP];
      int other_nNo = all_info[3*iP];
      for (int a = 0; a < other_nNo; a++) {
        // Global node number in processor iP at location a
        int Ac = aNodes(a, cand_idx);
        // Exit if this is the last node
        if (Ac == -1) {
          break;
        }
        // Corresponding local node in current processor
        auto gtl_it2 = gtlPtr.find(Ac);
        if (gtl_it2 != gtlPtr.end()) {
          lhs.cS[i].ptr[j] = Ac;
          j = j + 1;
        }
      }

      MPI_Send(lhs.cS[i].ptr.data(), lhs.cS[i].n, cm_mod::mpint, iP, 1, comm);

      for (int j = 0; j < lhs.cS[i].n; j++) {
        lhs.cS[i].ptr[j] = gtlPtr[lhs.cS[i].ptr[j]];
      }
    }
  }

  // Wait for second-round sends to complete before local buffers go out of scope
  if (nCandidates > 0) {
    MPI_Waitall(nCandidates, send_reqs.data(), MPI_STATUSES_IGNORE);
  }
}

//----------------
// fsils_lhs_free
//----------------
//
void fsils_lhs_free(FSILS_lhsType& lhs)
{
  for (int faIn = 0; faIn < lhs.nFaces; faIn++) {
    if (lhs.face[faIn].foC) {
      fsils_bc_free(lhs, faIn);
    }
  }

  for (int i = 0; i < lhs.nReq; i++) {
    //IF (ALLOCATED(lhs.cS(i).ptr)) DEALLOCATE(lhs.cS(i).ptr)
  }

  lhs.foC = false;
  lhs.gnNo   = 0;
  lhs.nNo    = 0;
  lhs.nnz    = 0;
  lhs.nFaces = 0;

  lhs.colPtr.clear();
  lhs.rowPtr.clear();
  lhs.diagPtr.clear();
  lhs.map.clear();
  lhs.cS.clear();
  lhs.face.clear();
  lhs.nmax_commu = 0;
  lhs.commu_sReq.clear();
  lhs.commu_rReq.clear();
  lhs.commu_sB.clear();
  lhs.commu_rB.clear();
  lhs.commu_dof_capacity = 0;
}


};

