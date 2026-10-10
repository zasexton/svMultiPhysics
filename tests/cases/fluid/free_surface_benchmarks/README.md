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
- **Level-set advection (D9).** Moving-interface benchmarks advect `phi` with the harmonic PDE extension of the fluid velocity, monolithically coupled: `Velocity_source=prescribed_data`, `Velocity_field_name=LevelSetAdvectionVelocity`, `Auto_register_velocity_field=true`, `Source_velocity_field_name=Velocity`, `Advection_velocity_extension_method=pde_harmonic`, `Advection_velocity_extension_coupling=monolithic` (`linear_sloshing_2d` README, "Level-set advection velocity").
- **Space and time (D10, D11).** Gate spatial convergence with the time-step error removed and run a separate time-step study; volume criteria gate the maximum deviation over the run.
- **Cost.** Each resolution level should run in minutes to about an hour on one node, submitted through Slurm with an explicit time limit and begin/end/fail mail.
- **Launching the solver.** Start every MPI-linked binary with `mpiexec -n <N> --bind-to none` (or `srun --mpi=pmix -n <N> --exact`), never bare.
  - Submit with `sbatch --export=NONE` and set the environment in the job script.
  - Reason: this Open MPI build uses Slurm's PMI when started without a launcher. A job submitted from inside an interactive `sh_dev` session inherits that session's `srun` contact variables, so a bare process contacts the interactive `srun` and hangs. That `srun` prints `PMK_KVS_Barrier task count inconsistent` in the user's terminal.
- **Long runs on 4 ranks.** Since the MPI correctness fixes (`ac273512`), runs with FSILS linear algebra match serial to round-off on 1, 2, 4 and 8 ranks. Long runs (static drop R/h ≥ 16, capillary wave λ/h ≥ 32, sessile drop) therefore use 4 MPI ranks on one node, for example 3.1× faster at R/h = 32.
  - Leave `<Ghost_layers>` unset: the solver derives 8 layers in 2D and 12 in 3D for an aggregating free surface (fewer layers drop constraint-fill columns and can make the aggregation roots depend on the partition).
  - Decks with Eigen linear algebra (for example `linear_sloshing_2d`, `tank_at_rest`) stay serial.
  - Bitwise comparisons between solver builds stay serial, because rank counts change the round-off.
- **JIT object cache.** Benchmark run scripts set `SVMP_JIT_CPU=x86-64-v3` and `SVMP_CACHE_PROFILE=L1d:32768,L1i:32768,L2:1048576,L3:16777216`, so one kernel cache serves both the Skylake and the Milan nodes of the partition. Outputs are bitwise identical to the default host target, and cold runs skip 10–15 s of kernel compilation (tracker D17; `Code/Source/solver/FE/Docs/BuildOptimization.md`). Bitwise comparisons between solver builds use the default target.
- **Requesting resources.** amarsden allows at most 8000 MB per CPU. A request above that silently adds CPUs: a one-task job with `--mem=8G` gets 2 CPUs, half of them idle. Request at most 8000 MB × CPUs (for example `--mem=7G` serial, `--mem=8G` for 4 ranks). Check finished jobs with `seff <jobid>`.
- **Run configurations (tracker D35).**
  - **Gated protocol runs** keep FSILS GMRES with row-column scaling and the default generalized-α predictor, so their results stay comparable with earlier runs.
  - **Development runs** may use the opt-in aggregation multigrid preconditioner (`<Right_preconditioner>amg</Right_preconditioner>` in `<LS type="GMRES">`) and the opt-in rate-extrapolation predictor (`Generalized_alpha_predictor=RateExtrapolation`, `Generalized_alpha_predictor_fields=phi`). Both give the same results on every rank count up to round-off, and change results only within the linear and nonlinear tolerances.
  - **3D production runs** use 8 ranks × 3 assembly threads per node:

    ```
    #SBATCH --nodes=1 --ntasks=8 --cpus-per-task=3
    export OMP_NUM_THREADS=1 SVMP_ASSEMBLY_THREADS=$SLURM_CPUS_PER_TASK
    mpiexec -n $SLURM_NTASKS --map-by slot:PE=$SLURM_CPUS_PER_TASK --bind-to core svmultiphysics solver.xml
    ```

    The assembly and geometry threads give bitwise-identical results for any thread count. The multigrid preconditioner is recommended there from R/h = 16 on. The opt-in free-surface partition weighting (`<Partition_weighting>free_surface</Partition_weighting>`) is allowed; it changes the partition and hence the results at round-off, which the run record must state.

## Planned benchmarks

| Benchmark | Milestone | Purpose |
|---|---|---|
| `tank_at_rest` | M1 | Hydrostatic balance with a flat interface, 2D and 3D |
| `linear_sloshing_2d` | M1 | Frequency and damping against linear theory. Reuses `open_vessel_free_surface/unfitted_level_set/linear_sloshing_2d`. |
| `static_drop_2d` | M2 | Laplace pressure and spurious currents. Capillary-route comparison (D2). |
| `capillary_wave_2d` | M3 | Frequency and damping against Prosperetti |
| `sessile_drop_2d` | M4 | Relaxation to the Young angle with Navier slip (D4) |
| `ren_e_2d` | M4 | Ren–E moving contact line (DynamicRenE), advancing and receding: law consistency on 3 meshes (l_s/h = 2, 4, 8) and 3 time steps |
