# WP-3/WP-7 accepted-state coercivity-policy prerequisite qualification record

- Source commit: `cb6cf91a090414eef020e3c30924b0b30570ed27`
- Frozen matrix: `free_surface_wp3_wp7_symmetric_nitsche_accepted_state_floor_prerequisite_v3`
- Outcome: **PASS**
- Distinct tests: 33
- Serial and distributed groups: 5
- Quantitative evidence: **PASS** (8 checks)
- Group recorded properties: **PASS** (0 checks)
- Final provenance: **PASS**

Exact aggregate-trace certification plus a predeclared symmetric-Nitsche energy floor c*=1/4 for every accepted current state of the supported production Navier-Stokes viscous/Nitsche subform, whose module supplies the bulk viscous energy K. The generic FE gate is conditional on an installed caller-supplied coercive bulk form and does not independently prove that bulk hypothesis. This matrix does not close FSR-16, FSR-07, WP-3, WP-7, or Q1, and does not prove that every cut or mesh-family state will be accepted..

## Accepted-state symmetric-Nitsche prerequisite

Exact aggregate-trace certification plus a predeclared symmetric-Nitsche energy floor c*=1/4 for every accepted current state of the supported production Navier-Stokes viscous/Nitsche subform, whose module supplies the bulk viscous energy K. The generic FE gate is conditional on an installed caller-supplied coercive bulk form and does not independently prove that bulk hypothesis. This matrix does not close FSR-16, FSR-07, WP-3, WP-7, or Q1, and does not prove that every cut or mesh-family state will be accepted.

The accepted-state floor is exactly `0.25` for the supported production subform. The emitted route digest and exact certificate digest do not bind that floor; policy-signature and certificate-cache provenance do. FSR-16, FSR-07, WP-3, WP-7, and Q1 remain open.

- Aggregate-trace evidence: **PASS**
- Accepted cases: 108
- Maximum trace upper bound: 1.3865887291231187
- Minimum accepted-state energy floor: 0.25
- Minimum conservative sampled eigenvalue gap: 0.7499999999993009
