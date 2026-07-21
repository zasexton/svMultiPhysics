# Conservative phase-indicator transport contract

## Selected representation

The conservative unknown is a bounded liquid indicator `q`, not the signed
distance field. `q=1` denotes liquid and `q=0` denotes the exterior phase. For
positive lumped nodal control volumes `m_i`, the declared discrete liquid
measure is

\[
M_q = \sum_i m_i q_i.
\]

The selected AD-4 direction is a coupled representation: continuous P1 `q`
provides conservative phase transport, while the signed-distance field
provides interface geometry. Geometry reconciliation must preserve the
accepted `q` measure and must remain inside the maintenance transaction.

The reusable correction kernel is implemented in
`LevelSetConservativePhaseTransport`. The first geometry-aware P1 operator is
implemented in `LevelSetConservativePhaseOperator`. It constructs lumped nodal
control volumes and the full algebraic gradient graph from the actual physical
mapping and quadrature. Production application wiring, geometry
reconciliation, distributed ownership, and the required transport benchmark
matrix remain separate qualification steps. Until those steps are complete,
the existing signed-distance transport must continue to report that it is not
locally conservative.

## P1 finite-element flux construction

For the assembled gradient matrix

\[
C_{ij}=\int_\Omega N_i\nabla N_j\,dx,
\]

the strong conservative CG contribution is

\[
-\sum_j C_{ij}\cdot(q_j u_j).
\]

Every unordered edge `i<j` is stored once. The central mass-transfer rate into
`i` is

\[
T_{ij}=C_{ji}\cdot(q_i u_i)-C_{ij}\cdot(q_j u_j),
\qquad T_{ji}=-T_{ij}.
\]

The part not represented by pair transfers is the physical-boundary remainder

\[
B_i=-q_i u_i\cdot b_i,
\qquad b_i=\sum_j C_{ji}=\int_{\partial\Omega}N_i n\,ds.
\]

Writing advective indicator transport in conservative form adds the explicit
discrete-divergence source

\[
S_i=q_i\sum_j C_{ij}\cdot u_j.
\]

Consequently `sum_edges(T_ij) + B_i + S_i` equals the lumped strong-CG
advective operator node by node. For constant `q`, the boundary, edge, and
divergence terms cancel exactly even for a nodal velocity field with nonzero
discrete divergence.

The symmetric low-order graph-viscosity coefficient is

\[
d_{ij}=\max\left(\left|C_{ij}\cdot u_j\right|,
                  \left|C_{ji}\cdot u_i\right|\right).
\]

It makes both directed off-diagonal update coefficients nonnegative. The
forward-Euler stage is accepted only when

\[
\max_i \frac{\Delta t}{m_i}
\sum_{j\ne i}\left(d_{ij}-C_{ij}\cdot u_j\right)\le 1.
\]

The low-order edge transfer adds `d_ij(q_j-q_i)`. The raw antidiffusive
transfer removes exactly that addition, so the unlimited target is the
lumped central operator and the accepted correction remains pairwise
conservative.

The graph builder validates partition of unity, the zero gradient sum,
positive mapped quadrature and control volumes, the global row-sum identity,
and closure of the summed lumped volumes against mapped physical measure.
It records geometry, topology, ownership, numbering, and degree-of-freedom
layout revisions. On multiple ranks, only owned cells contribute. Nodal
coefficients are reduced on the field communicator, sparse edge fragments are
merged in canonical order, and each edge is assigned one logical owner from
its endpoint ownership. Endpoint ownership is compared across ranks before the
graph is accepted. The resulting sparse graph is replicated because the
current level-set state uses globally indexed replicated vectors; every rank
therefore obtains the same single-edge ledger without counting ghost cells.
Rank-local preflight and assembly failures are synchronized before later
collectives so all ranks return the same failing-rank diagnostic.

The two-rank operator test checks mapped mass, cancellation of the artificial
partition-boundary remainder, unique owner coverage, identical graph and
limited state on both ranks, bounded transport across the shared interface,
constant preservation, and a fault injected on only one owned cell. Replacing
the replicated sparse merge with owner-to-owner exchange is a scalability
optimization that must preserve these public graph and ledger invariants. A
separate rank-local invalid-option test guards the collective failure path.

## Fully discrete edge update

Each interior algebraic edge is stored once in canonical orientation `i<j`.
A positive transfer `F_ij` is received by node `i` and removed from node `j`,
so `F_ji=-F_ij` by construction. The low-order stage is

\[
m_i q_i^L = m_i q_i^n + B_i + S_i + \sum_j L_{ij},
\]

where `B_i` is the integrated physical-boundary transfer, `S_i` is the
explicit discrete-divergence source, and `L_ij=-L_ji` is a bounded low-order
interior transfer. `S_i` is zero for a discretely divergence-compatible
advecting field.

For raw antidiffusive transfers `F_ij=-F_ji`, define

\[
P_i^+ = \sum_j \max(0,F_{ij}), \qquad
P_i^- = \sum_j \min(0,F_{ij}),
\]

and the admissible mass increments

\[
Q_i^+ = m_i(q_i^{max}-q_i^L), \qquad
Q_i^- = m_i(q_i^{min}-q_i^L).
\]

The nodal factors are

\[
R_i^+ = \min(1,Q_i^+/P_i^+), \qquad
R_i^- = \min(1,Q_i^-/P_i^-),
\]

with a factor of one when the corresponding denominator is zero. The
symmetric edge factor is

\[
\alpha_{ij} =
\begin{cases}
\min(R_i^+,R_j^-), & F_{ij}>0,\\
\min(R_i^-,R_j^+), & F_{ij}<0,\\
1, & F_{ij}=0.
\end{cases}
\]

The accepted update is

\[
m_i q_i^{n+1} = m_i q_i^L + \sum_j \alpha_{ij}F_{ij}.
\]

Because the same `alpha_ij` multiplies both orientations, every interior
correction cancels pairwise. The limiter changes neither the external physical
flux nor the declared divergence source.

## Ledger and fail-closed invariants

The stage result retains, for every node, the previous, low-order, raw-target,
and limited indicator; boundary, divergence, low-order, raw, and limited mass
contributions; both nodal factors; and a local balance residual for each
stage. Every edge record retains its raw and limited transfers, shared factor,
and pair-cancellation residuals.

The stage is rejected when any of these conditions fails:

- control-volume masses are not finite and positive;
- bounds are not ordered inside `[0,1]` or exclude the previous indicator;
- an edge is duplicated, reversed, self-connected, or out of range;
- the low-order predictor violates its local invariant domain;
- the limited state violates its local bounds;
- interior cancellation or any nodal/global balance does not close to the
  scaled invariant tolerance; or
- a constant input is not preserved by the low-order and accepted stages when
  constant preservation is required. Boundary data that intentionally change
  a constant state require an explicit stage-level waiver, which is retained
  in the result.

No global phase shift appears in this algorithm.

## Primary method references

- D. Kuzmin and M. Quezada de Luna, “Algebraic entropy fixes and convex
  limiting for continuous finite element discretizations of scalar hyperbolic
  conservation laws,” 2020, <https://arxiv.org/abs/2003.12007>.
- D. Kuzmin, M. Quezada de Luna, D. Ketcheson, and J. Grüll,
  “Bound-preserving flux limiting for high-order explicit Runge–Kutta time
  discretizations of hyperbolic conservation laws,” 2020,
  <https://arxiv.org/abs/2009.01133>.
