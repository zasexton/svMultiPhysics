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

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fe_fsi_linear_solver {

namespace {

constexpr int rank_derived_ownership = -1;

} // namespace

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
static void fsils_lhs_create_impl(FSILS_lhsType& lhs, FSILS_commuType& commu, int gnNo, int nNo, int nnz,
       Vector<int>& gNodes, Vector<int> &rowPtr, Vector<int>& colPtr, int nFaces, int explicit_owned_nNo)
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
  lhs.owned_row_operator = (explicit_owned_nNo >= 0);
  #ifdef debug_fsils_lhs_create
  dmsg << "gnNo: " << gnNo;
  dmsg << "nNo: " << nNo;
  dmsg << "nnz: " << nnz;
  dmsg << "nFaces: " << nFaces;
  #endif

  int nTasks = commu.nTasks;
  auto comm = commu.comm;
  auto tF = commu.tF;
  const bool use_explicit_owned_nodes = explicit_owned_nNo >= 0;
  if (use_explicit_owned_nodes && explicit_owned_nNo > nNo) {
    throw std::runtime_error("FSILS: explicit owned node count exceeds local node count.");
  }
  #ifdef debug_fsils_lhs_create
  dmsg << "nTasks: " << nTasks;
  dmsg << "tF: " << tF;
  #endif

  lhs.colPtr.resize(nnz); 
  lhs.rowPtr.resize(2,nNo); 
  lhs.diagPtr.resize(nNo);
  lhs.map.resize(nNo); 
  lhs.gNodes.resize(nNo);
  lhs.face.resize(nFaces);

  for (fsils_int a = 0; a < nNo; ++a) {
    lhs.gNodes(a) = gNodes(a);
  }

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

  if (use_explicit_owned_nodes) {
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

    lhs.mynNo = explicit_owned_nNo;
    lhs.shnNo = lhs.mynNo;
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
  int self_cand_idx = nCandidates; // self stored at index nCandidates

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

  // Classify shared nodes into front (shared with lower ranks) and back (shared with higher ranks).
  // The original FSILS code appended these in processor encounter order, which makes the internal
  // permutation depend on communicator topology. Keep the same front/local/back contract, but make
  // the node order within the shared bands canonical in global-node space.
  std::vector<int> lower_shared_nodes;
  std::vector<int> higher_shared_nodes;
  lower_shared_nodes.reserve(nNo);
  higher_shared_nodes.reserve(nNo);

  // Iterate candidates from highest rank to lowest to preserve the original marking logic, but do
  // not finalize the internal order yet.
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
          // Classify by processor ID; the final internal order is canonicalized below.
          if (rank_i < tF) {
            lower_shared_nodes.push_back(Ac);
          } else {
            higher_shared_nodes.push_back(Ac);
          }
          aNodes(localNodeIndex, self_cand_idx) = -1;
        }
      }
    }
  }

  std::sort(lower_shared_nodes.begin(), lower_shared_nodes.end());
  lower_shared_nodes.erase(std::unique(lower_shared_nodes.begin(), lower_shared_nodes.end()),
                           lower_shared_nodes.end());
  std::sort(higher_shared_nodes.begin(), higher_shared_nodes.end());
  higher_shared_nodes.erase(std::unique(higher_shared_nodes.begin(), higher_shared_nodes.end()),
                            higher_shared_nodes.end());

  lhs.shnNo = static_cast<int>(lower_shared_nodes.size());
  lhs.mynNo = nNo - static_cast<int>(higher_shared_nodes.size());

  #ifdef debug_fsils_lhs_create
  dmsg << "lhs.shnNo: " << lhs.shnNo;
  dmsg << "lhs.mynNo: " << lhs.mynNo;
  #endif

  for (int i = 0; i < lhs.shnNo; ++i) {
    ltg(i) = lower_shared_nodes[static_cast<std::size_t>(i)];
  }

  // Now include the local nodes left behind in their original local order.
  int j = lhs.shnNo;

  for (int a = 0; a < nNo; a++) {
    int Ac = aNodes(a, self_cand_idx);
    //  If this node has not been included already
    if (Ac != -1) {
      ltg(j) = Ac;
      j = j + 1;
    }
  }

  if (j != lhs.mynNo) {
    throw std::runtime_error("FSILS: Unexpected behavior: j=" + std::to_string(j) + " lhs.mynNo: " + std::to_string(lhs.mynNo) + ".");
    MPI_Finalize();
  }

  for (std::size_t i = 0; i < higher_shared_nodes.size(); ++i) {
    ltg(lhs.mynNo + static_cast<int>(i)) = higher_shared_nodes[i];
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

  // Wait for first-round sends to complete before local buffers go out of scope.
  // FE solve-time communication is handled by the explicit owned_halo_* plan;
  // this rank-derived path only needs the exchanged node lists to build the
  // internal ordering above.
  if (nCandidates > 0) {
    MPI_Waitall(nCandidates, send_reqs.data(), MPI_STATUSES_IGNORE);
  }
}

void fsils_lhs_create(FSILS_lhsType& lhs, FSILS_commuType& commu, int gnNo, int nNo, int nnz, Vector<int>& gNodes,
       Vector<int> &rowPtr, Vector<int>& colPtr, int nFaces)
{
  fsils_lhs_create_impl(lhs, commu, gnNo, nNo, nnz, gNodes, rowPtr, colPtr, nFaces, rank_derived_ownership);
}

void fsils_lhs_create_with_explicit_owned_nodes(FSILS_lhsType& lhs, FSILS_commuType& commu, int gnNo, int nNo,
       int nnz, Vector<int>& gNodes, Vector<int>& rowPtr, Vector<int>& colPtr, int nFaces, int owned_nNo)
{
  if (owned_nNo < 0) {
    throw std::runtime_error("FSILS: explicit owned node count must be non-negative.");
  }
  fsils_lhs_create_impl(lhs, commu, gnNo, nNo, nnz, gNodes, rowPtr, colPtr, nFaces, owned_nNo);
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

  lhs.foC = false;
  lhs.gnNo   = 0;
  lhs.nNo    = 0;
  lhs.nnz    = 0;
  lhs.nFaces = 0;
  lhs.owned_row_operator = false;

  lhs.colPtr.clear();
  lhs.rowPtr.clear();
  lhs.diagPtr.clear();
  lhs.map.clear();
  lhs.face.clear();
  lhs.owned_halo_neighbor_ranks.clear();
  lhs.owned_halo_send_nodes.clear();
  lhs.owned_halo_recv_nodes.clear();
  lhs.owned_halo_send_buffer.clear();
  lhs.owned_halo_recv_buffer.clear();
}


};
