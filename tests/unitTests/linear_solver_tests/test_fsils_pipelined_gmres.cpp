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

#include "FE/Backends/FSILS/liner_solver/gmres.h"

#include "gtest/gtest.h"

namespace {

void build_diag_lhs(fe_fsi_linear_solver::FSILS_lhsType& lhs, const fe_fsi_linear_solver::fsils_int n)
{
  using namespace fe_fsi_linear_solver;
  lhs.nNo = n;
  lhs.gnNo = n;
  lhs.mynNo = n;
  lhs.shnNo = 0;
  lhs.nnz = n;
  lhs.nReq = 0;
  lhs.nFaces = 0;
  lhs.face.clear();

  lhs.commu.nTasks = 1;
  lhs.commu.task = 0;
  lhs.commu.master = 0;
  lhs.commu.masF = true;
  lhs.commu.comm = MPI_COMM_SELF;

  lhs.rowPtr.resize(2, n);
  lhs.colPtr.resize(n);
  for (fsils_int i = 0; i < n; ++i) {
    lhs.rowPtr(0, i) = i;
    lhs.rowPtr(1, i) = i;
    lhs.colPtr(i) = i;
  }
}

} // namespace

TEST(FSILS_PipelinedGMRES, ScalarIdentityMatchesClassic)
{
  using namespace fe_fsi_linear_solver;

  FSILS_lhsType lhs{};
  const fsils_int nNo = 8;
  build_diag_lhs(lhs, nNo);

  Vector<double> Val(lhs.nnz);
  for (fsils_int i = 0; i < lhs.nnz; ++i) {
    Val(i) = 1.0;
  }

  Vector<double> rhs(nNo);
  for (fsils_int i = 0; i < nNo; ++i) {
    rhs(i) = 0.25 * static_cast<double>(i + 1);
  }

  FSILS_subLsType classic{};
  classic.mItr = 4;
  classic.sD = 10;
  classic.absTol = 1e-14;
  classic.relTol = 1e-14;
  classic.pipelined_gmres = false;

  Vector<double> r_classic = rhs;
  gmres::gmres_s(lhs, classic, /*dof*/1, Val, r_classic);
  ASSERT_TRUE(classic.suc);

  FSILS_subLsType piped{};
  piped.mItr = classic.mItr;
  piped.sD = classic.sD;
  piped.absTol = classic.absTol;
  piped.relTol = classic.relTol;
  piped.pipelined_gmres = true;

  Vector<double> r_pipe = rhs;
  gmres::gmres_s(lhs, piped, /*dof*/1, Val, r_pipe);
  ASSERT_TRUE(piped.suc);

  for (fsils_int i = 0; i < nNo; ++i) {
    EXPECT_NEAR(r_classic(i), rhs(i), 1e-12);
    EXPECT_NEAR(r_pipe(i), rhs(i), 1e-12);
    EXPECT_NEAR(r_pipe(i), r_classic(i), 1e-12);
  }
}

TEST(FSILS_PipelinedGMRES, VectorIdentityMatchesClassic)
{
  using namespace fe_fsi_linear_solver;

  FSILS_lhsType lhs{};
  const fsils_int nNo = 6;
  build_diag_lhs(lhs, nNo);

  const int dof = 2;
  Array<double> Val(dof * dof, lhs.nnz);
  Val = 0.0;
  for (fsils_int i = 0; i < lhs.nnz; ++i) {
    Val(0, i) = 1.0;
    Val(3, i) = 1.0;
  }

  Array<double> rhs(dof, nNo);
  for (fsils_int i = 0; i < nNo; ++i) {
    rhs(0, i) = 1.0 + 0.1 * static_cast<double>(i);
    rhs(1, i) = -0.5 + 0.05 * static_cast<double>(i);
  }

  FSILS_subLsType classic{};
  classic.mItr = 4;
  classic.sD = 10;
  classic.absTol = 1e-14;
  classic.relTol = 1e-14;
  classic.pipelined_gmres = false;

  Array<double> r_classic = rhs;
  gmres::gmres_v(lhs, classic, dof, Val, r_classic);
  ASSERT_TRUE(classic.suc);

  FSILS_subLsType piped{};
  piped.mItr = classic.mItr;
  piped.sD = classic.sD;
  piped.absTol = classic.absTol;
  piped.relTol = classic.relTol;
  piped.pipelined_gmres = true;

  Array<double> r_pipe = rhs;
  gmres::gmres_v(lhs, piped, dof, Val, r_pipe);
  ASSERT_TRUE(piped.suc);

  for (fsils_int i = 0; i < nNo; ++i) {
    EXPECT_NEAR(r_classic(0, i), rhs(0, i), 1e-12);
    EXPECT_NEAR(r_classic(1, i), rhs(1, i), 1e-12);
    EXPECT_NEAR(r_pipe(0, i), rhs(0, i), 1e-12);
    EXPECT_NEAR(r_pipe(1, i), rhs(1, i), 1e-12);
    EXPECT_NEAR(r_pipe(0, i), r_classic(0, i), 1e-12);
    EXPECT_NEAR(r_pipe(1, i), r_classic(1, i), 1e-12);
  }
}
