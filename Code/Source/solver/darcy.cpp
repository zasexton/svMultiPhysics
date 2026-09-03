#include "darcy.h"

#include "all_fun.h"
#include "mat_fun.h"
#include "nn.h"
#include "utils.h"

namespace darcy {
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
 * Strong form of the Single-Compartment Darcy equation:
 * \f[ u = -K\nabla P \f]
 * \f[ \nabla \cdot u = \beta_0(P_{source} - P) - \beta_1(P - P_{sink}) \f]
 * Note: See equations 8(a)/(b) in https://doi.org/10.1007/s10439-020-02681-z
 * 
 * Combined Strong form:
 * \f[ -\nabla \cdot (K \nabla P) - \beta_0(P_{source} - P) + \beta_1(P - P_{sink}) = 0 \f]
 * 
 * Where:
 *  - \f$ u \f$ : Darcy flux
 *  - \f$ K \f$ : Permeability tensor
 *  - \f$ P \f$ : Pressure
 *  - \f$ P_{source} \f$ : Source pressure (e.g., arterial pressure) 
 *  - \f$ P_{sink} \f$ : Sink pressure (e.g., venous/extraction pressure)
 *  - \f$ \beta_0 \f$ : Source coupling term (describes conductance of flow entering myocardium)
 *  - \f$ \beta_1 \f$ : Sink coupling term (describes conductance of flow exiting myocardium)
 * 
 * -------------------------------------------------------------
 * 
 * Weak form of the Single-Compartment Darcy equation:
 * \f[ -\int_{\Omega} (\nabla q \cdot \nabla P) d\Omega - \lambda \int_{\Omega} q P d\Omega = \int_{\Omega} q F d\Omega - \int_{\Gamma} q (\nabla P \cdot n) d\Gamma \f]
 * 
 * Where:
 *  - \f$ q \f$ : Test function
 *  - \f$ \lambda \f$ : \f$ \frac{\beta_0 + \beta_1}{K} \f$
 *  - \f$ F \f$ : \f$ -\frac{\beta_0 P_{source} + \beta_1 P_{sink}}{K} \f$
 *  - \f$ n \f$ : Normal vector to the boundary
 *  - \f$ \Omega \f$ : Computational domain
 *  - \f$ \Gamma \f$ : Domain boundary
*/

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
      } else {
        throw std::runtime_error("[construct_darcy] insd must be 2 or 3.");
      }
    }

    eq.linear_algebra->assemble(com_mod, eNoN, ptr, lK, lR);
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

  double k = dmn.prop.at(PhysicalPropertyType::darcy_permeability);
  double source = dmn.prop.at(PhysicalPropertyType::source_term);
  double beta_0 = dmn.prop.at(PhysicalPropertyType::darcy_media_compressibility);
  double rho_0 = dmn.prop.at(PhysicalPropertyType::fluid_density);
  double mu = dmn.prop.at(PhysicalPropertyType::darcy_fluid_viscosity);

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

  double p_dot = 0.0;
  Vector<double> Px(nsd);

  for (int a = 0; a < eNoN; a++) {
    p_dot = p_dot + N(a)*al(i,a);
    Px(0) = Px(0) + Nx(0,a)*yl(i,a);
    Px(1) = Px(1) + Nx(1,a)*yl(i,a);
  }

  for (int a = 0; a < eNoN; a++) {
    lR(0,a) = lR(0,a) +
        w * (rho_0 * N(a) * (beta_0 * p_dot - source) +
             ((k * rho_0) / mu) *
                 (Nx(0,a) * Px(0) + Nx(1,a) * Px(1)));
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

  double k = dmn.prop.at(PhysicalPropertyType::darcy_permeability);
  double source = dmn.prop.at(PhysicalPropertyType::source_term);
  double beta_0 = dmn.prop.at(PhysicalPropertyType::darcy_media_compressibility);
  double rho_0 = dmn.prop.at(PhysicalPropertyType::fluid_density);
  double mu = dmn.prop.at(PhysicalPropertyType::darcy_fluid_viscosity);

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

  double p_dot = 0.0;
  Vector<double> Px(nsd);

  for (int a = 0; a < eNoN; a++) {
    p_dot = p_dot + N(a) * al(i,a);
    Px(0) = Px(0) + Nx(0,a) * yl(i,a);
    Px(1) = Px(1) + Nx(1,a) * yl(i,a);
    Px(2) = Px(2) + Nx(2,a) * yl(i,a);
  }

  for (int a = 0; a < eNoN; a++) {
    lR(0,a) = lR(0, a) +
        w * (rho_0 * N(a) * (beta_0 * p_dot - source) +
             ((k * rho_0) / mu) *
                 (Nx(0,a) * Px(0) + Nx(1,a) * Px(1) +
                  Nx(2,a) * Px(2)));
    for (int b = 0; b < eNoN; b++) {
      lK(0,a,b) = lK(0,a,b) + wl*(rho_0*beta_0*N(a)*N(b)*amd +
                                  ((k*rho_0)/mu)*(Nx(0,a)*Nx(0,b) + Nx(1,a)*Nx(1,b) +
                                                  Nx(2,a)*Nx(2,b)));
    }
  }
}

}