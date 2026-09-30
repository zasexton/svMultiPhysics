# Free-surface benchmarks

Short, self-contained benchmarks for the free-surface boundary condition.
They supply the evidence required by decision D6 in
`Documentation/free_surface_program_tracker.md`. Each result is recorded as:

- the commit hash;
- the benchmark's driver and tolerance file from this directory;
- one results row in the tracker (§2.1 and §8);
- raw outputs in group storage, for accepted results only.

No other record is required.

## Layout

One subdirectory per benchmark:

```text
free_surface_benchmarks/
  <benchmark>/
    README.md          physical setup, reference solution, source of each tolerance
    generate_case.py   writes solver.xml and the mesh for one resolution level
    tolerances.json    acceptance criteria, fixed before first use
    verify.py          reads solver output, computes the metrics, applies tolerances.json
```

`generate_case.py` takes the resolution level and an output directory, so a
refinement study is a loop over levels. Generated cases and solver output
go to `$SCRATCH`, never into this directory.

## Tolerance files

`tolerances.json` lists one entry per quantity:

```json
{
  "benchmark": "static_drop_2d",
  "milestone": "M2",
  "levels": {"R_over_h": [8, 16, 32, 64]},
  "criteria": [
    {
      "quantity": "pressure_jump_relative_error",
      "definition": "|(p_in - p_out) - gamma/R| / (gamma/R) on the relaxed state",
      "limit": 0.01,
      "at_level": 32,
      "minimum_observed_order": 1.0,
      "source": "Decision D1 working criterion (tracker section 5, M2)"
    }
  ]
}
```

Each criterion names:

- the quantity and how it is computed;
- its limit, and the level where the limit applies;
- any required convergence rate;
- where the value comes from: a decision, the literature, or an analytic reference.

Tolerances are set before the first run. If a tolerance proves unrealistic,
change it once, with a one-line justification in the tracker.

## Rules for every benchmark

- **No tuning (P1).** A benchmark must not require a numerical parameter chosen per case or per mesh. Physical inputs (surface tension, viscosity, contact angle, slip length) come from the case definition.
- **Relaxed states (D1, D3).** Static equilibria start from the sampled analytic shape and run until the flow relaxes, typically 5–10 viscous times. Metrics are taken from the relaxed state, and the history of the maximum velocity is reported.
- **Surface tension inputs.** Set `Geometry_tangent_policy=RefreshedFrozenQuadrature`, which unfitted surface tension requires. Check the time step against the capillary limit `sqrt(rho*h^3/(2*pi*gamma))`.
- **Cost.** Each resolution level should run in minutes to about an hour on one node, submitted through Slurm with an explicit time limit and begin/end/fail mail.
- **Launching the solver.** Start every MPI-linked binary with `mpiexec -n <N> --bind-to none` (or `srun --mpi=pmix -n <N> --exact`), never bare.
  - Submit with `sbatch --export=NONE` and set the environment in the job script.
  - Reason: this Open MPI build uses Slurm's PMI when started without a launcher. A job submitted from inside an interactive `sh_dev` session inherits that session's `srun` contact variables, so a bare process contacts the interactive `srun` and hangs. That `srun` prints `PMK_KVS_Barrier task count inconsistent` in the user's terminal.

## Planned benchmarks

| Benchmark | Milestone | Purpose |
|---|---|---|
| `tank_at_rest` | M1 | Hydrostatic balance with a flat interface, 2D and 3D |
| `linear_sloshing_2d` | M1 | Frequency and damping against linear theory. Reuses `open_vessel_free_surface/unfitted_level_set/linear_sloshing_2d`. |
| `static_drop_2d` | M2 | Laplace pressure and spurious currents. Capillary-route comparison (D2). |
| `capillary_wave_2d` | M3 | Frequency and damping against Prosperetti |
| `sessile_drop_2d` | M4 | Relaxation to the Young angle with Navier slip (D4) |
