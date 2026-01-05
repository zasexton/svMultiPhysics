# Vortex Shedding Benchmark (2D Cylinder-in-Channel, Re = 100)

This benchmark is a classic validation case for incompressible finite-element Navier–Stokes solvers: **2D flow past a circular cylinder in a rectangular channel**. At **Re = 100**, the wake develops a stable, time-periodic **von Kármán vortex street**. The benchmark is commonly known as the **DFG / Schäfer–Turek “flow around a cylinder”** case (often referenced as *2D-2, periodic*).

The goal is to verify that your solver reproduces:
- **periodic lift oscillations**,
- realistic **drag levels**,
- correct **shedding frequency / Strouhal number**,
- consistent **pressure drop** across the cylinder.

---

## Problem definition

### Geometry (2D)
A rectangular channel with a circular hole:
- Channel:  
  \[
  \Omega_{\text{box}} = [0,\,2.2]\times[0,\,0.41]
  \]
- Cylinder:
    - center: \((x_c,y_c) = (0.2,\,0.2)\)
    - radius: \(r = 0.05\) (diameter \(D = 0.1\))
- Fluid domain:
  \[
  \Omega = \Omega_{\text{box}} \setminus B_{0.05}(0.2,\,0.2)
  \]

Useful clearances:
- inlet → cylinder: \(x_c - r = 0.15\)
- cylinder → outlet: \(2.2 - (x_c + r) = 1.95\)
- cylinder → bottom wall: \(y_c - r = 0.15\)
- cylinder → top wall: \(0.41 - (y_c + r) = 0.16\)

---

## Governing equations (incompressible Navier–Stokes)

Solve for velocity \(u(x,t)\) and pressure \(p(x,t)\):
\[
u_t + (u\cdot\nabla)u - \nu \Delta u + \nabla p = 0,\qquad
\nabla\cdot u = 0.
\]

### Material parameters
This benchmark is typically run in a unit-density setting:
- Density: \(\rho = 1\)
- Kinematic viscosity: \(\nu = 0.001\)

---

## Boundary conditions

Let the boundaries be:
- Inlet: \(x=0\)
- Outlet: \(x=2.2\)
- Walls: \(y=0\) and \(y=0.41\)
- Cylinder: \(\partial B_r\)

### Inlet (prescribed parabolic inflow)
\[
u(0,y) = \left(\frac{4U\,y(0.41-y)}{0.41^2},\,0\right),\quad U=1.5.
\]
Here, \(U\) is the **maximum** inlet velocity.

Mean inlet velocity for this parabola:
\[
U_{\text{mean}}=\frac{2}{3}U = 1.0.
\]

### Walls + cylinder (no slip)
\[
u = 0.
\]

### Outlet (traction / “do-nothing” outflow)
A common form:
\[
\nu\,\partial_\eta u - p\,\eta = 0,
\]
where \(\eta\) is the outward unit normal at the outlet.

> Note: If your code uses a different outflow model (e.g., weak backflow stabilization, pressure-outlet), document it—outflow treatment can affect the wake.

---

## Reynolds number

Characteristic length: cylinder diameter \(D=0.1\).  
Using \(U_{\text{mean}} = 1.0\):
\[
\mathrm{Re} = \frac{U_{\text{mean}}\,D}{\nu}
= \frac{1.0 \cdot 0.1}{0.001}
= 100.
\]

At Re=100, the wake becomes **time-periodic**, giving lift oscillations and alternating vortices.

---

## What to measure (benchmark outputs)

### 1) Drag and lift on the cylinder
Compute the traction on the cylinder boundary:
\[
\sigma(u,p) = -pI + 2\nu\,\varepsilon(u),\qquad
\varepsilon(u)=\frac{1}{2}(\nabla u + \nabla u^T).
\]
Force on the cylinder:
\[
F = \int_{\Gamma_{\text{cyl}}} \sigma(u,p)\,n\,ds,
\]
then:
- Drag \(F_D = F\cdot e_x\)
- Lift \(F_L = F\cdot e_y\)

Common dimensionless coefficients:
\[
C_D = \frac{2F_D}{\rho\,U_{\text{mean}}^2\,D},\qquad
C_L = \frac{2F_L}{\rho\,U_{\text{mean}}^2\,D}.
\]

### 2) Pressure difference (two probe points)
A standard choice:
- \(a_1 = (0.15,\,0.2)\)
- \(a_2 = (0.25,\,0.2)\)
  \[
  \Delta p(t) = p(a_1) - p(a_2).
  \]

### 3) Shedding frequency and Strouhal number
Measure the oscillation period \(T\) of \(C_L(t)\), then \(f=1/T\).  
\[
\mathrm{St} = \frac{fD}{U_{\text{mean}}}.
\]

---

## What to expect (qualitative)

After startup from rest, you should see:
- an initial transient as the wake forms,
- then a **stable periodic limit cycle**:
    - \(C_L(t)\) oscillates with alternating sign,
    - \(C_D(t)\) oscillates around a positive mean with smaller amplitude,
    - the wake shows alternating vortices convecting downstream.

Typical ballpark results (dependent on discretization/outflow/timestep):
- Mean drag coefficient: **~3.1–3.3**
- Lift coefficient amplitude: **~0.9–1.1**
- Strouhal number: **~0.29–0.31**

If your results drift significantly:
- too coarse mesh near the cylinder,
- too large timestep,
- insufficient stabilization (or excessive numerical diffusion),
- outflow reflections / backflow issues,
  are the most common culprits.

---

## Recommended simulation setup

### Initial condition
- Start from rest: \(u(x,0) = 0\)

### Time integration
- Use a stable, at least second-order method if possible (e.g., BDF2, Crank–Nicolson)
- Typical timestep range for good accuracy:
    - **Δt ~ 1e-3 to 5e-3** (use smaller for higher-order, or if forces are noisy)

### Spatial discretization
- Stable velocity/pressure pairing (examples):
    - Taylor–Hood \(P_2/P_1\)
    - stabilized equal-order (PSPG/grad-div as appropriate)
- Mesh guidance:
    - refine strongly near the cylinder and in the near wake,
    - ensure adequate resolution of boundary layers and shear layers.
- A practical check: further mesh/time refinement should change mean \(C_D\), lift amplitude, and St only slightly.

### Outflow notes
At Re=100, backflow can occur intermittently. If your solver supports it, consider:
- weak backflow stabilization,
- convective outflow variants,
- or sufficient downstream length (this benchmark already provides a long wake region).

---

## Run procedure (suggested)

1. Generate/Load mesh for the domain (channel minus cylinder).
2. Assign boundary markers:
    - inlet, outlet, top wall, bottom wall, cylinder
3. Run transient simulation until the wake becomes periodic:
    - monitor \(C_L(t)\) until peak-to-peak amplitude stabilizes
4. Once periodic, compute statistics over one (or several) full cycles:
    - min/max/mean/amplitude of \(C_D\) and \(C_L\)
    - shedding period \(T\), frequency \(f\), Strouhal St
    - \(\Delta p(t)\)

---

## Outputs (recommended files)

At minimum:
- `forces.csv`  
  columns: `t, Cd, Cl`
- `pressure_probes.csv`  
  columns: `t, dp`
- `summary.json` (or `.txt`)  
  includes:
    - mean/min/max/amplitude for Cd, Cl
    - period T, frequency f, Strouhal St
    - mesh size info + dt + discretization choices

Optional:
- VTK/VTU time series for visualization (`u`, `p`, vorticity)
- extracted boundary polylines (`.vtp`) for BC assignment:
    - `bc_inlet.vtp`, `bc_outlet.vtp`, `bc_top.vtp`, `bc_bottom.vtp`, `bc_cylinder.vtp`

---

## Verification checklist

- [ ] Lift signal \(C_L(t)\) is clearly periodic (not decaying to 0, not chaotic).
- [ ] Vorticity field shows alternating vortices shed from the cylinder.
- [ ] Mean drag is within expected range and converges under refinement.
- [ ] Strouhal number stabilizes and is consistent under dt/mesh refinement.
- [ ] Boundary conditions are correctly applied (especially no-slip on cylinder/walls).

---

## Common failure modes

- **No vortex shedding (steady wake)**:
    - too much numerical diffusion (over-stabilization),
    - timestep too large,
    - mesh too coarse in the near wake.
- **Unstable / exploding simulation**:
    - insufficient stabilization,
    - outflow/backflow issues,
    - too aggressive nonlinear solver tolerances.
- **Wrong frequency**:
    - dt too large,
    - outflow reflections,
    - under-resolved shear layer separation.

---
