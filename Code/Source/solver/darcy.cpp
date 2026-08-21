/*
 This code implements the Darcy Equation for 2D and 3D
 problems in perfusion of porous media.
 -------------------------------------------------------------
 Assumptions:
    - Homogeneous Permeability
    - Homogeneous Density
    - Isotropic Permeability
    - Assumptions of Stokes Flow
    - Steady-State
 -------------------------------------------------------------
 Strong form of the Single-Compartment Darcy equation:
    u = -K∇(P)
    ∇⋅u = β0(P_source - P) - β1(P - P_sink)
 where:
    u -> Volume flux vector
    K -> Permeability tensor
    P -> Pressure
 -------------------------------------------------------------
 Weak form of the Single-Compartment Darcy equation:
    -∫(∇q∇P)dΩ - λ∫qPdΩ = ∫qFdΩ - ∫q∇P⋅nvdΓ
 where:
    q -> Test function
    λ -> (β0 + β1)/K
    F -> -(β0(P_source) + β1(P_sink))/K
    n -> Normal vector to the boundary
*/

#include "darcy.h"

#include "all_fun.h"
#include "mat_fun.h"
#include "nn.h"
#include "utils.h"

namespace darcy {

void b_darcy(ComMod& com_mod, const int eNoN, const double w, const Vector<double>& N, const double h, Array<double>& lR)
{
  for (int a = 0; a < eNoN; a++) {
    lR(0,a) = lR(0,a) + w * N(a) * h;
  }
}

void construct_darcy(ComMod& com_mod, const mshType& lM, const SolutionStates& solutions)
{
  const auto& Ag = solutions.intermediate.get_acceleration();
  const auto& Yg = solutions.intermediate.get_velocity();
  #define n_debug_construct_darcy
  #ifdef debug_construct_darcy
  DebugMsg dmsg(__func__, com_mod.cm.idcm());
  dmsg.banner();
  #endif

  using namespace consts;

  const int nsd = com_mod.nsd;
  const int tDof = com_mod.tDof;
  const int dof = com_mod.dof;
  const int cEq = com_mod.cEq;
  const auto& eq = com_mod.eq[cEq];
  auto& cDmn = com_mod.cDmn;

  int eNoN = lM.eNoN;
  int insd = nsd;
  if (lM.lFib) {
    insd = 1;
  }
  #ifdef debug_construct_darcy
  dmsg << "cEq: " << cEq;
  dmsg << "cDmn: " << cDmn;
  dmsg << "insd: " << insd;
  #endif

  Vector<int> ptr(eNoN);
  Vector<double> N(eNoN);
  Array<double> xl(nsd, eNoN), al(tDof, eNoN), yl(tDof, eNoN);
  Array<double> Nx(insd, eNoN), lR(dof, eNoN);
  Array3<double> lK(dof * dof, eNoN, eNoN);
  Array<double> ksix(nsd, nsd);

  for (int e = 0; e < lM.nEl; e++) {
    cDmn = all_fun::domain(com_mod, lM, cEq, e);
    auto cPhys = eq.dmn[cDmn].phys;
    if (cPhys != EquationType::phys_darcy) {
      continue;
    }

    // Update shape function for NURBS
    if (lM.eType == ElementType::NRB) {
      //CALL NRMNNX(lm, e)
    }

    // Create local copies
    for (int a = 0; a < eNoN; a++) {
      int Ac = lM.IEN(a, e);
      ptr(a) = Ac;

      for (int i = 0; i < nsd; i++) {
        xl(i, a) = com_mod.x(i, Ac);
      }

      for (int i = 0; i < tDof; i++) {
        al(i, a) = Ag(i, Ac);
        yl(i, a) = Yg(i, Ac);
      }
    }

    // Gauss integration
    lR = 0.0;
    lK = 0.0;
    double Jac{0.0};

    for (int g = 0; g < lM.nG; g++) {
      if (g == 0 || !lM.lShpF) {
        auto Nx_g = lM.Nx.slice(g);
        nn::gnn(eNoN, nsd, insd, Nx_g, xl, Nx, Jac, ksix);
        if (utils::is_zero(Jac)) {
          throw std::runtime_error(
              "[construct_darcy] Jacobian for element " + std::to_string(e) + " is < 0.");
        }
      }

      double w = lM.w(g) * Jac;
      N = lM.N.col(g);

      if (insd == 3) {
        darcy_3d(com_mod, eNoN, w, N, Nx, al, yl, lR, lK);
      } else if (insd == 2) {
        darcy_2d(com_mod, eNoN, w, N, Nx, al, yl, lR, lK);
      } else if (insd == 1) {
        darcy_1d(com_mod, eNoN, w, N, Nx, al, yl, lR, lK);
      } else {
        throw std::runtime_error("[construct_darcy] insd must be 1, 2 or 3.");
      }
    }

    eq.linear_algebra->assemble(com_mod, eNoN, ptr, lK, lR);
  }
}

void darcy_1d(ComMod& com_mod, const int eNoN, const double w, const Vector<double>& N, const Array<double>& Nx,
              const Array<double>& al, const Array<double>& yl, Array<double>& lR, Array3<double>& lK)
{
  using namespace consts;

  const int cEq = com_mod.cEq;
  auto& eq = com_mod.eq[cEq];
  const int cDmn = com_mod.cDmn;
  auto& dmn = eq.dmn[cDmn];
  const double dt = com_mod.dt;
  const int i = eq.s;

  double k = dmn.prop.at(PhysicalProperyType::permeability);
  double source = dmn.prop.at(PhysicalProperyType::source_term);
  double beta_0 = dmn.prop.at(PhysicalProperyType::media_compressibility);
  double rho_0 = dmn.prop.at(PhysicalProperyType::fluid_density);
  double mu = dmn.prop.at(PhysicalProperyType::darcy_fluid_viscosity);

  double T1 = eq.af * eq.gam * dt;
  double amd = eq.am / T1;
  double wl = w * T1;

  double Pd = -source;
  double Px = 0.0;

  for (int a = 0; a < eNoN; a++) {
    Pd = Pd + N(a) * al(i, a);
    Px = Px + Nx(0, a) * yl(i, a);
  }

  for (int a = 0; a < eNoN; a++) {
    lR(0, a) = lR(0, a) + w * (rho_0 * beta_0 * N(a) * Pd +
                               (((k * rho_0) / mu) * (Nx(0, a) * Px)));
    for (int b = 0; b < eNoN; b++) {
      lK(0, a, b) = lK(0, a, b) + wl * (rho_0 * beta_0 * N(a) * N(b) * amd +
                                       ((((rho_0 * k) / mu) * (Nx(0, a) * Nx(0, b)))));
    }
  }
}

void darcy_2d(ComMod& com_mod, const int eNoN, const double w, const Vector<double>& N, const Array<double>& Nx,
              const Array<double>& al, const Array<double>& yl, Array<double>& lR, Array3<double>& lK)
{
  #define n_debug_darcy_2d
  #ifdef debug_darcy_2d
  DebugMsg dmsg(__func__, com_mod.cm.idcm());
  dmsg.banner();
  dmsg << "w: " << w;
  #endif
  using namespace consts;

  const int nsd = com_mod.nsd;
  const int cEq = com_mod.cEq;
  auto& eq = com_mod.eq[cEq];
  const int cDmn = com_mod.cDmn;
  auto& dmn = eq.dmn[cDmn];
  const double dt = com_mod.dt;
  const int i = eq.s;

  double k = dmn.prop.at(PhysicalProperyType::permeability);
  double source = dmn.prop.at(PhysicalProperyType::source_term);
  double beta_0 = dmn.prop.at(PhysicalProperyType::media_compressibility);
  double rho_0 = dmn.prop.at(PhysicalProperyType::fluid_density);
  double mu = dmn.prop.at(PhysicalProperyType::darcy_fluid_viscosity);

  double T1 = eq.af * eq.gam * dt;
  double amd = eq.am / T1;
  double wl = w * T1;

  #ifdef debug_darcy_2d
  dmsg << "k: " << k;
  dmsg << "source: " << source;
  dmsg << "T1: " << T1;
  dmsg << "i: " << i;
  dmsg << "wl: " << wl;
  #endif

  double Pd = -source;
  Vector<double> Px(nsd);

  for (int a = 0; a < eNoN; a++) {
    Pd = Pd + N(a)*al(i,a);
    Px(0) = Px(0) + Nx(0,a)*yl(i,a);
    Px(1) = Px(1) + Nx(1,a)*yl(i,a);
  }

  for (int a = 0; a < eNoN; a++) {
    lR(0,a) = lR(0,a) + w*(rho_0*beta_0*N(a)*Pd + (((k*rho_0)/mu)*(Nx(0,a)*Px(0)
                                                                   + Nx(1,a)*Px(1))));
    for (int b = 0; b < eNoN; b++) {
      lK(0,a,b) = lK(0,a,b) + wl*(rho_0*beta_0*N(a)*N(b)*amd +
                                  ((((rho_0*k)/mu)*(Nx(0,a)*Nx(0,b) +
                                                    Nx(1,a)*Nx(1,b)))));
    }
  }
}

void darcy_3d(ComMod& com_mod, const int eNoN, const double w, const Vector<double>& N, const Array<double>& Nx,
              const Array<double>& al, const Array<double>& yl, Array<double>& lR, Array3<double>& lK)
{
  #define n_debug_darcy_3d
  #ifdef debug_darcy_3d
  DebugMsg dmsg(__func__, com_mod.cm.idcm());
  dmsg.banner();
  dmsg << "w: " << w;
  #endif
  using namespace consts;

  const int nsd = com_mod.nsd;
  const int cEq = com_mod.cEq;
  auto& eq = com_mod.eq[cEq];
  const int cDmn = com_mod.cDmn;
  auto& dmn = eq.dmn[cDmn];
  const double dt = com_mod.dt;
  const int i = eq.s;

  double k = dmn.prop.at(PhysicalProperyType::permeability);
  double source = dmn.prop.at(PhysicalProperyType::source_term);
  double beta_0 = dmn.prop.at(PhysicalProperyType::media_compressibility);
  double rho_0 = dmn.prop.at(PhysicalProperyType::fluid_density);
  double mu = dmn.prop.at(PhysicalProperyType::darcy_fluid_viscosity);

  double T1 = eq.af * eq.gam * dt;
  double amd = eq.am / T1;
  double wl = w * T1;

  #ifdef debug_darcy_3d
  dmsg << "k: " << k;
  dmsg << "source: " << source;
  dmsg << "T1: " << T1;
  dmsg << "i: " << i;
  dmsg << "wl: " << wl;
  #endif

  double Pd = -source;
  Vector<double> Px(nsd);

  for (int a = 0; a < eNoN; a++) {
    Pd = Pd + N(a) * al(i,a);
    Px(0) = Px(0) + Nx(0,a) * yl(i,a);
    Px(1) = Px(1) + Nx(1,a) * yl(i,a);
    Px(2) = Px(2) + Nx(2,a) * yl(i,a);
  }

  for (int a = 0; a < eNoN; a++) {
    lR(0,a) = lR(0, a) + w*(rho_0*beta_0*N(a)*Pd +
                            ((k*rho_0)/mu)*(Nx(0,a)*Px(0) + Nx(1,a)*Px(1) + Nx(2,a)*Px(2)));
    for (int b = 0; b < eNoN; b++) {
      lK(0,a,b) = lK(0,a,b) + wl*(rho_0*beta_0*N(a)*N(b)*amd +
                                  ((k*rho_0)/mu)*(Nx(0,a)*Nx(0,b) + Nx(1,a)*Nx(1,b) +
                                                  Nx(2,a)*Nx(2,b)));
    }
  }
}

}