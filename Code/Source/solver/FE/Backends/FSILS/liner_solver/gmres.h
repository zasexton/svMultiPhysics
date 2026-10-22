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

#include "fils_struct.hpp"
#include "distributed_solver_bundles.h"

namespace gmres {

void gmres(const fe_fsi_linear_solver::distributed_solver_bundles::VectorLinearSystem& system,
    fe_fsi_linear_solver::FSILS_subLsType& ls, const Array<double>& R, Array<double>& X);

void gmres(fe_fsi_linear_solver::FSILS_lhsType& lhs, fe_fsi_linear_solver::FSILS_subLsType& ls, const int dof,
    const Array<double>& Val, const Array<double>& R, Array<double>& X);

void gmres_s(const fe_fsi_linear_solver::distributed_solver_bundles::ScalarLinearSystem& system,
    fe_fsi_linear_solver::FSILS_subLsType& ls, Vector<double>& R,
    const Vector<double>* row_scaling = nullptr);

void gmres_s(fe_fsi_linear_solver::FSILS_lhsType& lhs, fe_fsi_linear_solver::FSILS_subLsType& ls, const int dof,
    const Vector<double>& Val, Vector<double>& R);

void gmres_v(const fe_fsi_linear_solver::distributed_solver_bundles::VectorLinearSystem& system,
    fe_fsi_linear_solver::FSILS_subLsType& ls, Array<double>& R,
    const Array<double>* row_scaling = nullptr);

void gmres_v(fe_fsi_linear_solver::FSILS_lhsType& lhs, fe_fsi_linear_solver::FSILS_subLsType& ls, const int dof,
    const Array<double>& Val, Array<double>& R);

namespace test {

double fused_dot_zz_v_for_test(const int dof,
    const fe_fsi_linear_solver::fsils_int mynNo,
    Array3<double>& u,
    const int i,
    const Array<double>& u_next,
    std::vector<double>& h_col,
    const int num_threads = 1);

/// Test-only: when true, the serial Gram-Schmidt dot for dof 6 and 8 uses the
/// generic fused_dot_zz_v_dyn kernel (the reference of the two-lane kernel).
/// Default false.
void set_dot_dynamic_kernel_forced(bool forced) noexcept;

/// Test-only: when true, the serial Gram-Schmidt update for dof > 4 uses the
/// loop of fused_update_norm_v_dyn (the reference of the register-blocked
/// kernel).  Default false.
void set_update_dynamic_kernel_forced(bool forced) noexcept;

/// The Gram-Schmidt update and norm of the GMRES iteration:
/// u_next -= sum_j h_factors[j] * u(:,:,j) for j = 0..i, then the norm of the
/// owned entries.
double fused_update_norm_v_for_test(const int dof,
    const fe_fsi_linear_solver::fsils_int nNo,
    const fe_fsi_linear_solver::fsils_int mynNo,
    fe_fsi_linear_solver::FSILS_commuType& commu,
    Array3<double>& u,
    const int i,
    Array<double>& u_next,
    const std::vector<double>& h_factors);

void fused_update_v_inplace_for_test(const int dof,
    const fe_fsi_linear_solver::fsils_int nNo,
    Array3<double>& u,
    const int i,
    Array<double>& u_next,
    const std::vector<double>& h_factors);

void apply_givens_rotation_for_test(Array<double>& h,
    Vector<double>& c,
    Vector<double>& s,
    Vector<double>& err,
    const int i);

bool backsolve_hessenberg_for_test(const Array<double>& h,
    const Vector<double>& rhs,
    const int last_i,
    const double initial_norm,
    const double eps,
    Vector<double>& y);

} // namespace test

};
