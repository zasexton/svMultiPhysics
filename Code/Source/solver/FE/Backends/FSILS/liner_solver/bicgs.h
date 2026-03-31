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

#ifndef FSI_LINEAR_SOLVER_BICGS_H
#define FSI_LINEAR_SOLVER_BICGS_H

#include "fils_struct.hpp"

namespace bicgs {

void bicgsv(fe_fsi_linear_solver::FSILS_lhsType& lhs, fe_fsi_linear_solver::FSILS_subLsType& ls, const int dof,
            const Array<double>& K, Array<double>& R);

void bicgss(fe_fsi_linear_solver::FSILS_lhsType& lhs, fe_fsi_linear_solver::FSILS_subLsType& ls,
            const Vector<double>& K, Vector<double>& R);

// NEW: VMS compatible Asymmetric Schur Complement Solver
void schur(fe_fsi_linear_solver::FSILS_lhsType& lhs,
           fe_fsi_linear_solver::FSILS_subLsType& ls,
           fe_fsi_linear_solver::FSILS_subLsType& momentum_ls,
           const int nsd,
           const Array<double>& K, const Array<double>& D, const Array<double>& G,
           const Vector<double>& L, Vector<double>& R);

/// Multi-component constraint Schur complement solver.
/// Solves (L - D*H*G) P = R where the constraint has con_ncomp components.
/// D(con_ncomp*mom_ncomp, nnz), G(mom_ncomp*con_ncomp, nnz), L(con_ncomp*con_ncomp, nnz).
/// R(con_ncomp, nNo) is both the RHS input and solution output.
void schur_mc(fe_fsi_linear_solver::FSILS_lhsType& lhs,
              fe_fsi_linear_solver::FSILS_subLsType& ls,
              fe_fsi_linear_solver::FSILS_subLsType& momentum_ls,
              int mom_ncomp, int con_ncomp,
              const Array<double>& K, const Array<double>& D, const Array<double>& G,
              const Array<double>& L, Array<double>& R);

} // namespace bicgs

#endif
