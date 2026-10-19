# Free-surface refactor design (2026-10-19)

Design studies behind decision D42 in `Documentation/free_surface_program_tracker.md`.
They supersede the target architecture of `free_surface_architecture_refactoring_plan_20260904.md`
where the two differ. Line numbers refer to commit `e7080344`.

- `formulation_design.md`: inventory of the Navier–Stokes VMS module and register, the target
  formulation of about 470 lines in the form language (free-surface terms as in tracker §6.1),
  missing form-language constructs, option schemas, bitwise-identity risks.
- `infrastructure_design.md`: where the free-surface machinery lives in ApplicationDriver,
  NewtonSolver and FESystem; the target FE services (level-set geometry, maintenance pipeline,
  field extension, generated-state registry and outer fixed point, step participants,
  constraint plan, Newton hooks); form-language lowering rules; the staged migration S0–S13;
  section 7 aligns its names with the formulation design.
- `levelset_placement.md`: which level-set code in Application belongs in FE, which stays as
  input translation, and which is physics; staged moves C1–C4 mapped onto S0–S13.

Rules for every step: results stay bitwise identical on the default path (kernel and constraint
manifests, the arm64 reference set, CTest, the two-rank MPI matrix, 2/4-rank parity), except the
final, separate expression-tree cleanup, which is validated against tolerances.
