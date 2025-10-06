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

#include "heatf.h"

#include "all_fun.h"
#include "lhsa.h"
#include "mat_fun.h"
#include "nn.h"
#include "utils.h"

#include <math.h>

namespace heatf {

void b_heatf(ComMod& com_mod, const int eNoN, const double w, const Vector<double>& N, const Vector<double>& y,
    const double h, const Vector<double>& nV, Array<double>& lR, Array3<double>& lK, int species_index)
{
  const int nsd = com_mod.nsd;
  const int cEq = com_mod.cEq;
  const auto& eq = com_mod.eq[cEq];
  const double dt = com_mod.dt;

  double wl = w * eq.af * eq.gam * dt;
  int m = eq.dof;
  double udn = 0.0;

  for (int i = 0; i < nsd; i++) {
    udn = udn + y(i)*nV(i);
  }

  udn = 0.5 * (udn - fabs(udn));
  int is_start = (species_index >= 0 ? species_index : 0);
  int is_end   = (species_index >= 0 ? species_index : m-1);
  for (int is = is_start; is <= is_end; ++is) {
    int sidx = eq.s + is;
    double T = y(sidx);
    double T1  = h - udn*T;
    for (int a = 0; a < eNoN; a++) {
      lR(is,a) = lR(is,a) + w*N(a)*T1;
      for (int b = 0; b < eNoN; b++) {
        int blk = is*m + is;
        lK(blk,a,b) = lK(blk,a,b) - wl*N(a)*N(b)*udn;
      }
    }
  }
}

void construct_heatf(ComMod& com_mod, const mshType& lM, const Array<double>& Ag, const Array<double>& Yg)
{
  #define n_debug_construct_heatf 
  #ifdef debug_construct_heatf
  DebugMsg dmsg(__func__, com_mod.cm.idcm());
  dmsg.banner();
  #endif

  using namespace consts;

  const int nsd  = com_mod.nsd;
  const int tDof = com_mod.tDof;
  const int dof = com_mod.dof;
  const int cEq = com_mod.cEq;
  const auto& eq = com_mod.eq[cEq];
  auto& cDmn = com_mod.cDmn;

  int eNoN = lM.eNoN;
  #ifdef debug_construct_heatf
  dmsg << "cEq: " << cEq;
  dmsg << "cDmn: " << cDmn;
  #endif

  Vector<int> ptr(eNoN);
  Vector<double> N(eNoN);
  Array<double> xl(nsd,eNoN), al(tDof,eNoN), yl(tDof,eNoN), Nx(nsd,eNoN), lR(dof,eNoN);
  Array3<double> lK(dof*dof,eNoN,eNoN);
  Array<double> ksix(nsd,nsd);

  for (int e = 0; e < lM.nEl; e++) {
    // Update domain and proceed if domain phys and eqn phys match
    cDmn = all_fun::domain(com_mod, lM, cEq, e);
    auto cPhys = eq.dmn[cDmn].phys;
    if (cPhys != EquationType::phys_heatF) {
      continue;
    }

    // Update shape functions for NURBS
    if (lM.eType == ElementType::NRB) {
      //CALL NRBNNX(lM, e)
    }

    // Create local copies
    //
    for (int a = 0; a < eNoN; a++) {
      int Ac = lM.IEN(a,e);
      ptr(a) = Ac;

      for (int i = 0; i < nsd; i++) {
        xl(i,a) = com_mod.x(i,Ac);
      }

      for (int i = 0; i < tDof; i++) {
        al(i,a) = Ag(i,Ac);
        yl(i,a) = Yg(i,Ac);
      }
    }

    // Gauss integration
    //
    lR = 0.0;
    lK = 0.0;
    double Jac{0.0};

    for (int g = 0; g < lM.nG; g++) {
      if (g == 0 || !lM.lShpF) {
        auto Nx_g = lM.Nx.slice(g);
        nn::gnn(eNoN, nsd, nsd, Nx_g, xl, Nx, Jac, ksix);
        if (utils::is_zero(Jac)) {
          throw std::runtime_error("[construct_heatf] Jacobian for element " + std::to_string(e) + " is < 0.");
        }
      }

      double w = lM.w(g) * Jac;
      N = lM.N.col(g);

      if (nsd == 3) {
        heatf_3d(com_mod, eNoN, w, N, Nx, al, yl, ksix, lR, lK);

      } else if (nsd == 2) {
        heatf_2d(com_mod, eNoN, w, N, Nx, al, yl, ksix, lR, lK);
      }
    } // for g = 0

    eq.linear_algebra->assemble(com_mod, eNoN, ptr, lK, lR);

  } // for e = 0
}

void heatf_2d(ComMod& com_mod, const int eNoN, const double w, const Vector<double>& N, const Array<double>& Nx,
    const Array<double>& al, const Array<double>& yl, const Array<double>& ksix, Array<double>& lR, Array3<double>& lK)
{
  #define n_debug_heatf_2d 
  #ifdef debug_heatf_2d 
  DebugMsg dmsg(__func__, com_mod.cm.idcm());
  dmsg.banner();
  dmsg << "w: " << w;
  #endif

  static Vector<double> ct({4.0, 1.0, 3.0, 1.0});

  using namespace consts;
  using namespace mat_fun;

  const int nsd = com_mod.nsd;
  const int cEq = com_mod.cEq;
  auto& eq = com_mod.eq[cEq];
  const int cDmn = com_mod.cDmn;
  auto& dmn = eq.dmn[cDmn];
  const double dt = com_mod.dt;
  const int m = eq.dof;

  double T1 = eq.af * eq.gam * dt;
  double amd = eq.am / T1;
  double nu_base = dmn.prop.at(PhysicalProperyType::conductivity);
  double s_base = dmn.prop.at(PhysicalProperyType::source_term);
  double wl = w * T1;

  #ifdef debug_heats_2d 
  dmsg << "nu: " << nu;
  dmsg << "s: " << s;
  dmsg << "T1: " << T1;
  dmsg << "i: " << i;
  dmsg << "wl: " << wl;
  #endif

  Vector<double> u(nsd), udNx(eNoN);
  Vector<double> Tg(m), Td(m), udTx(m);
  Array<double> Tx(nsd,m);

  for (int a = 0; a < eNoN; a++) {
    u(0) = u(0) + N(a)*yl(0,a);
    u(1) = u(1) + N(a)*yl(1,a);
    for (int is = 0; is < m; ++is) {
      int sidx = eq.s + is;
      Td(is) = Td(is) + N(a)*al(sidx,a);
      Tg(is) = Tg(is) + N(a)*yl(sidx,a);
      Tx(0,is) = Tx(0,is) + Nx(0,a)*yl(sidx,a);
      Tx(1,is) = Tx(1,is) + Nx(1,a)*yl(sidx,a);
    }
  }

  // Add reaction term s += R(Tg) -> Td -= R(Tg)
  Array<double> kr_2d(m,m);
  kr_2d = 0.0;
  if (dmn.has_reaction_rhs) {
    Vector<double> state(m), deriv(m);
    for (int is = 0; is < m; ++is) state(is) = Tg(is);
    dmn.reaction_rhs(com_mod.time, state, deriv);
    for (int is = 0; is < m; ++is) {
      Td(is) = Td(is) - deriv(is);
    }
    if (dmn.has_reaction_jac) {
      dmn.reaction_jac(com_mod.time, state, kr_2d);
    }
  }

  if (com_mod.mvMsh) {
    for (int a = 0; a < eNoN; a++) {
      u(0) = u(0) - N(a)*yl(4,a);
      u(1) = u(1) - N(a)*yl(5,a);
      u(2) = u(2) - N(a)*yl(6,a);
    }
  } 

  double kU = u(0)*u(0)*ksix(0,0) + u(1)*u(0)*ksix(1,0) + u(0)*u(1)*ksix(0,1) + u(1)*u(1)*ksix(1,1);
  double kS = ksix(0,0)*ksix(0,0) + ksix(1,0)*ksix(1,0) + ksix(0,1)*ksix(0,1) + ksix(1,1)*ksix(1,1);
  Vector<double> tauM(m), Tp(m), nu_eff(m);
  for (int is = 0; is < m; ++is) {
    double nTx = ksix(0,0)*Tx(0,is)*Tx(0,is) + ksix(1,1)*Tx(1,is)*Tx(1,is) + (ksix(0,1) + ksix(1,0))*Tx(0,is)*Tx(1,is);
    if (utils::is_zero(nTx)) nTx = std::numeric_limits<double>::epsilon();
    udTx(is) = u(0)*Tx(0,is) + u(1)*Tx(1,is);
    double nu = nu_base;
    double Tpc = fabs(Td(is) + udTx(is));
    nu = nu + 0.5 * Tpc / sqrt(nTx);
    tauM(is) = ct(3) / sqrt((ct(0)/(dt*dt)) + ct(1)*kU + ct(2)*nu*nu*kS);
    Tp(is) = -tauM(is) * (Td(is) + udTx(is));
    nu_eff(is) = nu;
  }

  for (int a = 0; a < eNoN; a++) {
    udNx(a) = u(0)*Nx(0,a) + u(1)*Nx(1,a);
  }
  // Build base diffusion matrix D (m x m) from domain parameters
  Array<double> D(m,m);
  D = 0.0;
  if (dmn.diffusion_matrix_flat.size() == m*m) {
    for (int i = 0, k = 0; i < m; ++i) {
      for (int j = 0; j < m; ++j, ++k) D(i,j) = dmn.diffusion_matrix_flat(k);
    }
  } else if (dmn.species_diffusivity.size() == m) {
    for (int i = 0; i < m; ++i) D(i,i) = dmn.species_diffusivity(i);
  } else {
    for (int i = 0; i < m; ++i) D(i,i) = nu_base;
  }

  for (int a = 0; a < eNoN; a++) {
    for (int is = 0; is < m; ++is) {
      // Diffusion residual: base cross-diffusion + augmented diagonal
      double diff_res = 0.0;
      for (int js = 0; js < m; ++js) {
        diff_res += D(is,js) * (Nx(0,a)*Tx(0,js) + Nx(1,a)*Tx(1,js));
      }
      diff_res += (Nx(0,a)*Tx(0,is) + Nx(1,a)*Tx(1,is)) * (nu_eff(is) - nu_base);

      lR(is,a) = lR(is,a) + w*(N(a)*(Td(is) + udTx(is)) + diff_res - udNx(a)*Tp(is));
      for (int b = 0; b < eNoN; b++) {
        // Base diffusion blocks
        for (int js = 0; js < m; ++js) {
          int ij = is*m + js;
          lK(ij,a,b) = lK(ij,a,b) + wl * D(is,js) * (Nx(0,a)*Nx(0,b) + Nx(1,a)*Nx(1,b));
        }
        // Augmented diagonal diffusion
        int ii = is*m + is;
        lK(ii,a,b) = lK(ii,a,b) + wl * (nu_eff(is) - nu_base) * (Nx(0,a)*Nx(0,b) + Nx(1,a)*Nx(1,b));
        // Time/mass + convection
        lK(ii,a,b) = lK(ii,a,b) + wl * (N(a) + tauM(is)*udNx(a)) * (N(b)*amd + udNx(b));
        // Reaction Jacobian full block: off-diagonals included
        if (dmn.has_reaction_rhs && dmn.has_reaction_jac) {
          for (int js = 0; js < m; ++js) {
            int ij = is*m + js;
            lK(ij,a,b) = lK(ij,a,b) - wl * (N(a) + tauM(is)*udNx(a)) * kr_2d(is,js) * N(b);
          }
        }
      }
    }
  }
}


void heatf_3d(ComMod& com_mod, const int eNoN, const double w, const Vector<double>& N, const Array<double>& Nx,
    const Array<double>& al, const Array<double>& yl, const Array<double>& ksix, Array<double>& lR, Array3<double>& lK)
{
  #define n_debug_heatf_3d 
  #ifdef debug_heatf_3d 
  DebugMsg dmsg(__func__, com_mod.cm.idcm());
  dmsg.banner();
  dmsg << "w: " << w;
  dmsg << "N: " << N;
  dmsg << "Nx: " << Nx;
  dmsg << "yl: " << yl;
  dmsg << "al: " << al;
  #endif

  static Vector<double> ct({4.0, 1.0, 3.0, 1.0});

  using namespace consts;
  using namespace mat_fun;

  const int nsd = com_mod.nsd;
  const int cEq = com_mod.cEq;
  auto& eq = com_mod.eq[cEq];
  const int cDmn = com_mod.cDmn;
  auto& dmn = eq.dmn[cDmn];
  const double dt = com_mod.dt;
  const int m = eq.dof;

  double T1 = eq.af * eq.gam * dt;
  double amd = eq.am / T1;
  double nu_base = dmn.prop.at(PhysicalProperyType::conductivity);
  double s_base = dmn.prop.at(PhysicalProperyType::source_term);
  double wl = w * T1;

  #ifdef debug_heatf_3d 
  dmsg;
  dmsg << "eNoN: " << eNoN;
  dmsg << "nu: " << nu;
  dmsg << "s: " << s;
  dmsg << "T1: " << T1;
  dmsg << "i: " << i;
  dmsg << "wl: " << wl;
  dmsg << "ct: " << ct;
  #endif

  Vector<double> u(nsd), udNx(eNoN);
  Vector<double> Tg(m), Td(m), udTx(m);
  Array<double> Tx(nsd,m);

  for (int a = 0; a < eNoN; a++) {
    u(0) = u(0) + N(a)*yl(0,a);
    u(1) = u(1) + N(a)*yl(1,a);
    u(2) = u(2) + N(a)*yl(2,a);

    for (int is = 0; is < m; ++is) {
      int sidx = eq.s + is;
      Td(is) = Td(is) + N(a)*al(sidx,a);
      Tg(is) = Tg(is) + N(a)*yl(sidx,a);
      Tx(0,is) = Tx(0,is) + Nx(0,a)*yl(sidx,a);
      Tx(1,is) = Tx(1,is) + Nx(1,a)*yl(sidx,a);
      Tx(2,is) = Tx(2,is) + Nx(2,a)*yl(sidx,a);
    }
  }

  // Add reaction term s += R(Tg) -> Td -= R(Tg)
  Array<double> kr_3d(m,m);
  kr_3d = 0.0;
  if (dmn.has_reaction_rhs) {
    Vector<double> state(m), deriv(m);
    for (int is = 0; is < m; ++is) state(is) = Tg(is);
    dmn.reaction_rhs(com_mod.time, state, deriv);
    for (int is = 0; is < m; ++is) Td(is) -= deriv(is);
    if (dmn.has_reaction_jac) {
      dmn.reaction_jac(com_mod.time, state, kr_3d);
    }
  }

  if (com_mod.mvMsh) {
    for (int a = 0; a < eNoN; a++) {
      u(0) = u(0) - N(a)*yl(4,a);
      u(1) = u(1) - N(a)*yl(5,a);
      u(2) = u(2) - N(a)*yl(6,a);
    }
  }

  double kU = u(0)*u(0)*ksix(0,0) + u(1)*u(0)*ksix(1,0) + u(2)*u(0)*ksix(2,0) + u(0)*u(1)*ksix(0,1) + 
              u(1)*u(1)*ksix(1,1) + u(2)*u(1)*ksix(2,1) + u(0)*u(2)*ksix(0,2) + u(1)*u(2)*ksix(1,2) + 
              u(2)*u(2)*ksix(2,2);

  double kS = ksix(0,0)*ksix(0,0) + ksix(1,0)*ksix(1,0) + ksix(2,0)*ksix(2,0) + ksix(0,1)*ksix(0,1) + 
              ksix(1,1)*ksix(1,1) + ksix(2,1)*ksix(2,1) + ksix(0,2)*ksix(0,2) + ksix(1,2)*ksix(1,2) + 
              ksix(2,2)*ksix(2,2);

  Vector<double> tauM(m), Tp(m), nu_eff(m);
  for (int is = 0; is < m; ++is) {
    double nTx = ksix(0,0)*Tx(0,is)*Tx(0,is) + ksix(1,1)*Tx(1,is)*Tx(1,is) + ksix(2,2)*Tx(2,is)*Tx(2,is) +
                 (ksix(0,1) + ksix(1,0))*Tx(0,is)*Tx(1,is) + (ksix(0,2) + ksix(2,0))*Tx(0,is)*Tx(2,is) +
                 (ksix(1,2) + ksix(2,1))*Tx(1,is)*Tx(2,is);
    if (utils::is_zero(nTx)) nTx = std::numeric_limits<double>::epsilon();
    udTx(is) = u(0)*Tx(0,is) + u(1)*Tx(1,is) + u(2)*Tx(2,is);
    double nu = nu_base;
    double Tpc = fabs(Td(is) + udTx(is));
    nu = nu + 0.5 * Tpc / sqrt(nTx);
    tauM(is) = ct(3) / sqrt((ct(0)/(dt*dt)) + ct(1)*kU + ct(2)*nu*nu*kS);
    Tp(is) = -tauM(is) * (Td(is) + udTx(is));
    nu_eff(is) = nu;
  }

  #ifdef debug_heatf_3d 
  dmsg << "u: " << u;
  dmsg << "Tx: " << Tx;
  dmsg << "kU: " << kU;
  dmsg << "kS: " << kS;
  dmsg << "nTx: " << nTx<< std::endl;
  #endif

  if (utils::is_zero(nTx)) {
    nTx = std::numeric_limits<double>::epsilon();
  }

  double udTx = u(0)*Tx(0) + u(1)*Tx(1) + u(2)*Tx(2);
  double Tp = fabs(Td + udTx);
  nu = nu + 0.5 * Tp / sqrt(nTx);
  double tauM = ct(3) / sqrt((ct(0)/(dt*dt)) + ct(1)*kU + ct(2)*nu*nu*kS);
  Tp = -tauM*(Td + udTx);

  for (int a = 0; a < eNoN; a++) {
    udNx(a) = u(0)*Nx(0,a) + u(1)*Nx(1,a) + u(2)*Nx(2,a);
  }

  // Base diffusion matrix D for 3D
  Array<double> D(m,m);
  D = 0.0;
  if (dmn.diffusion_matrix_flat.size() == m*m) {
    for (int i = 0, k = 0; i < m; ++i) {
      for (int j = 0; j < m; ++j, ++k) D(i,j) = dmn.diffusion_matrix_flat(k);
    }
  } else if (dmn.species_diffusivity.size() == m) {
    for (int i = 0; i < m; ++i) D(i,i) = dmn.species_diffusivity(i);
  } else {
    for (int i = 0; i < m; ++i) D(i,i) = nu_base;
  }

  for (int a = 0; a < eNoN; a++) {
    for (int is = 0; is < m; ++is) {
      double diff_res = 0.0;
      for (int js = 0; js < m; ++js) {
        diff_res += D(is,js) * (Nx(0,a)*Tx(0,js) + Nx(1,a)*Tx(1,js) + Nx(2,a)*Tx(2,js));
      }
      diff_res += (Nx(0,a)*Tx(0,is) + Nx(1,a)*Tx(1,is) + Nx(2,a)*Tx(2,is)) * (nu_eff(is) - nu_base);
      lR(is,a) = lR(is,a) + w*(N(a)*(Td(is) + udTx(is)) + diff_res - udNx(a)*Tp(is));
      for (int b = 0; b < eNoN; b++) {
        for (int js = 0; js < m; ++js) {
          int ij = is*m + js;
          lK(ij,a,b) = lK(ij,a,b) + wl * D(is,js) * (Nx(0,a)*Nx(0,b) + Nx(1,a)*Nx(1,b) + Nx(2,a)*Nx(2,b));
        }
        int ii = is*m + is;
        lK(ii,a,b) = lK(ii,a,b) + wl * (nu_eff(is) - nu_base) * (Nx(0,a)*Nx(0,b) + Nx(1,a)*Nx(1,b) + Nx(2,a)*Nx(2,b));
        lK(ii,a,b) = lK(ii,a,b) + wl * (N(a) + tauM(is)*udNx(a)) * (N(b)*amd + udNx(b));
        if (dmn.has_reaction_rhs && dmn.has_reaction_jac) {
          for (int js = 0; js < m; ++js) {
            int ij = is*m + js;
            lK(ij,a,b) = lK(ij,a,b) - wl * (N(a) + tauM(is)*udNx(a)) * kr_3d(is,js) * N(b);
          }
        }
      }
    }
  }
}

};
