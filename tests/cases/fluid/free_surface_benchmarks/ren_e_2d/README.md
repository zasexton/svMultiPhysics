# 2D Ren-E moving contact line (tracker M4)

A 2D liquid cap on the flat bottom wall relaxes under the Ren-E contact-line law

    V_CL = gamma * M * (cos(theta_e) - cos(theta_d)),    V_CL > 0 advancing,

with the line mobility `M` imposed as line friction `1/M` together with the
variational Young term (`Contact_line_model` `DynamicRenE`), Navier slip on the
wetted wall and a strong normal-only wall velocity (decision D4).  Everything
else is the sessile-drop deck (`../sessile_drop_2d`): SurfaceStress with
small-cut aggregation, harmonic PDE velocity extension (D15), kinematic
reconciliation (D14), sign-definite patch bounds (D21), generalized-alpha
(rho_inf = 0.5) and FSILS GMRES.

It refines the 2026-08-30 pilot (job 41286834; master audit L1819: cap of radius
0.3 in a unit box, theta_e = 90 deg, theta_0 = 95/85 deg, M = 1, slip 0.1,
resolution 16, dt = 1e-3, 20 steps, gate 0.5 on the relative law error).

## Protocol

| quantity | value | reason |
|---|---|---|
| R, rho, gamma | 1, 1, 1 | units; the sessile-drop scaling |
| mu | from La = rho gamma 2R / mu^2 = 12 | as static_drop_2d and sessile_drop_2d |
| theta_e | 90 deg | the pilot |
| theta_0 | 105 deg (advancing), 75 deg (receding), equal liquid area | 15 deg: a law speed 0.26 gamma M, three times the pilot's |
| M | 1 | the pilot |
| slip length | R/4, fixed | l_s/h = 2, 4, 8 on R/h = 8, 16, 32 |
| dt | dt0 = 1/800, dt0/2, dt0/4 | dt0 is below the capillary limit of R/h = 32 |
| end time | 0.25 | the line moves under a sizable law speed |
| outputs | every 0.0125 (20) | |

`generate_case.py --level {8,16,32} --case {advancing,receding} --dt-divisor {1,2,4} --output-dir DIR`
writes one run (18 runs in all).  Runs are 2D; serial or up to 4 ranks.

## Measurement (verify.py)

At every saved state the two phi = 0 roots on y = 0 are found.  At each root the
dynamic angle is that of the P1 level-set gradient in the wall triangle holding
the root (the LinearCorner fragment normal), the prediction is the Ren-E speed at
that angle, the wall fluid speed is the outward wall-tangential velocity
interpolated at the root, and the geometric speed is the central difference of
the outward contact position.  Errors are relative to the prediction, RMS over
the window t >= 0.025 and both contact points.

## Criteria (tolerances.json, registered before the protocol runs)

* Signs: wall fluid speed and geometric speed have the sign of the prediction in
  every window sample of every run.
* Accuracy at the finest pair (R/h = 32, dt0/4): both RMS relative errors <= 10%
  for the advancing and the receding case.
* Mesh convergence at dt0/4: both RMS errors decrease strictly over R/h = 8, 16, 32.
* Time step at R/h = 32: the mean contact displacement at t = 0.25 changes by at
  most 2% between dt0/2 and dt0/4.
* Liquid area: maximum relative drift <= 1e-4 in every run (D11).
* Completion: every run reaches t = 0.25; an outer-cap failure is retried once
  with `SVMP_GENERATED_STATE_OUTER_DYNAMIC_RELAXATION=1` for diagnosis only.

The 10% and 2% values are proposals (no published discrete benchmark exists for
this constitutive check): 10% is a fifth of the pilot's loose gate and the size of
error at which a law speed is still quantitatively useful; 2% mirrors the D19
time-step criteria of the static drop and capillary wave in relative terms.  The
convergence requirement is the D1 working criterion (error decreasing with h).

## Result (2026-10-07, protocol job 46900681, binary 4cbf2643-ltopgo, commit 73c095a9)

**FAIL on finest accuracy only.**  All 18 runs complete with no outer-cap failure.
The signs are correct and the area drift is at most 4e-7.  The errors decrease
strictly with h, and the time-step change is below 1%.  The RMS law error at
dt0/4 on R/h = 8, 16, 32 is:

* advancing: 0.307, 0.242, 0.171
* receding: 0.584, 0.434, 0.346

The verify angle is the solver's own operator angle (`dynamic_contact_operator_angle`).
The wall fluid speed agrees with the geometric contact speed to 1-2%.  The measured
speed stays below the law, by a factor of 0.43 / 0.58 / 0.66 for receding at all
times, and by 0.66-0.75 early for advancing.

A diagnosis-only sensitivity at R/h = 16, dt0 (job 46906511) changed one parameter
at a time:

* M/10: the speed ratio rises to 1.03 (advancing) and 0.88 (receding).
* mu/10: the ratio becomes 1.05-1.17 (advancing) and 0.80 (receding).
* l_s x4: little change.

The deficit is therefore set by mu*M.  The weak point force balance shares the
uncompensated Young force between the line friction 1/M and the viscous and slip
stress in the contact element, and this share decays only slowly with h.  The
criteria above are unchanged.
