# Examples

Scenario scripts for the start-up physics in RAPID2D, one stage at a time and then the
whole chain. Each script sets a configuration and flags, runs, and writes plots and an
animation.

## Running

```
julia --project=examples -e 'using Pkg; Pkg.instantiate()'   # once
julia --project=examples examples/townsend_avalanche.jl
```

The examples environment develops RAPID2D from the parent directory (the `[sources]` entry
of `Project.toml`, which needs Julia 1.11 or later). It also adds `Plots`, which activates
the plotting extension; the mp4 uses the FFMPEG that ships with it. Outputs go to
`examples/output/<name>/`.

### Verdicts

A script with a prediction checks itself against it. It saves its figure as
`<name>__PASS.png` or `<name>__FAIL.png`, removes the figure of the other verdict, and heads
the figure with what it checked. The output folder therefore shows the verdict of the last
run. The scenario scripts have no prediction and save their figures without a verdict.

Known failures, and the work each belongs to:

| figure | what fails | cause |
|---|---|---|
| `force_balance_control/position_control__FAIL` | the column is not held; it reaches the wall | the force and velocity model: the global J×B force is an explicit kick on an accumulated velocity. The coupled solve is not the cause. |

## Scenarios

| script | field and geometry | physics on | simulated time | check |
|---|---|---|---|---|
| `townsend_avalanche.jl` | single-quadrupole null, box wall | atomic reactions, transport | 0.8 ms | scenario, no verdict |
| `selfE_avalanche.jl` | single-quadrupole null, box wall | as `townsend_avalanche.jl`, plus E∥ cancellation, mean E×B, turbulent E×B mixing | 4 ms | scenario, no verdict |
| `current_diffusion.jl` | pure toroidal field | a current filament with Ampère, with and without the inductive E; single-filament L/R reference with the electrons' kinetic inductance | 2 × 20 ms | the current follows the (L + L_kin)/R circuit within 1 % of saturation |
| `force_balance_control.jl` | pure toroidal field | J×B hoop force; curved vertical-field PID position control | 2 × 2 ms | the controller holds the current centroid within 5 cm of 1.5 m after 1 ms |
| `full_startup.jl` | single-quadrupole null, box wall | every module except the global J×B force | 10 ms | scenario, no verdict |
| `kstar_reference.jl` | KSTAR, time-varying external field | self-E model, Ampère off | 40 ms | scenario, no verdict |

## Coupled-step verification (`coupled_step/`)

Small problems for the coupled step, in which the electron parallel momentum, Ampère's law
and the circuits of coils and conducting structures advance together. Each isolates one
mechanism and plots the simulation against a prediction that does not come from the code.
They share `coupled_step/common.jl`: a uniform column at rest in a pure toroidal field (the
column of `current_diffusion.jl`, at Te = 1 eV), toroidal loops around it, and a lumped
model of the column as one plasma loop, which gives the circuit problems their reference
curves. The two motion problems need no model: a superconducting loop or shell keeps its
flux, and that flux is computed from the simulated current.
`test/regression/coupled_step_test.jl` checks the same problems.

| script | setup | prediction |
|---|---|---|
| `two_coils.jl` | two coupled loops, no plasma | exp(−M⁻¹R t) I(0), and the backward-Euler recursion step for step |
| `density_doubling.jl` | a driven column, alone and with a superconducting loop beside it; n doubled at a fixed drift | the fluxes cannot jump: I_p rises under 2 %, the drift halves, the loop current holds (lumped model) |
| `coil_driven_column.jl` | a 10 V coil drives the column; no loop voltage | two coupled circuits, with the electrons' kinetic inductance (lumped model) |
| `density_growth_dt.jl` | as `density_doubling.jl`, with n growing at 200/s; Δt = 5 and 2.5 µs | the loop keeps its flux (lumped model); the error is one step of growth, halving with Δt |
| `column_pushed_toward_loop.jl` | the column pushed at 200 m/s toward a superconducting loop outside the wall | I_c = −Φ_p/L_c; the loop pushes the column back |
| `column_shifted_in_shell.jl` | the column moved up one cell inside a shell of 24 superconducting filaments | the shell's flux-conserving currents push it back down |

### The coupled solve's iteration

Within a step the coupled solve iterates on the boundary flux and the coil currents (see
`coupled_step/common.jl`). These scripts compare the default solve with the same equations
solved directly (`method = :direct`, no iteration), step by step, each step against the direct
solve's current of that step (floored at 1e-3 of the run's peak). A case whose direct run did
not converge fails. Each figure shows four things:
- where the column and the conductors sit;
- the plasma current of the two runs;
- the first step's induced-field error against the number of iterations, for the relaxed
  iteration (the default before Anderson mixing) and for the default;
- that error over the grid.

| script | setup | check |
|---|---|---|
| `picard_center_column.jl` | a dense, hot column in the middle of the default domain | the default stays within 1 % of the direct solve's current at every step |
| `picard_kstar_inboard_limited.jl` | the KSTAR grid and first wall; a dense column 4 cm from the inboard wall | as above |
| `picard_tight_box.jl` | a column filling a box wall one cell inside the grid | as above |
| `picard_filament_shell.jl` | a dense column inside a shell of 24 copper filaments | as above |
| `picard_regime_map.jl` | the KSTAR grid and first wall; inboard-limited columns of radius 0.25–0.45 m, from 1e16 m⁻³ and 2 eV to 1e19 m⁻³ and 20 eV | every case within 1 % |
| `gate_crossing_dense_column.jl` | dense columns at rest under the default 1 A gate | the run steps as the run with the gate at 0, within 1 % |

Each script writes one figure to `examples/output/coupled_step/`:

```
julia --project=examples examples/coupled_step/two_coils.jl
```

## Input data

- `data/SingleQuad_LV=+5.dat`: a BREAK-format field file. It holds B_R, B_Z, ψ and the loop
  voltage on a 141 × 121 (R, Z) grid over 0.8–2.2 m × −1.2–1.2 m: a single-quadrupole null
  with a 5 V loop voltage. One file is a static field. A directory of such files, one per
  time slice, is interpolated in time.
- `data/box_wall.dat`: the rectangular wall, 1–2 m × −1–1 m.

The examples pass both through `SimulationConfig(inputs = InputPaths(field = …, wall = …))`.

`kstar_reference.jl` needs the KSTAR inputs, which are not in the repository. Set
`RAPID_INPUT_PATH` to the directory that holds `KSTAR/Reference_201x201/` and
`KSTAR_First_Wall.dat`.

## Step callbacks

`run_simulation!` calls `callback_before_step(RP)` once the step's inputs are set from the
state at tⁿ (external fields, `Jϕ`), before it solves; `callback_after_step(RP)` runs at its
end. Anything callable with `RP` works; any other state it needs, it captures:

```julia
ne_before = similar(RP.plasma.ne)
dne = Float64[]
run_simulation!(RP;
    callback_before_step = rp -> copyto!(ne_before, rp.plasma.ne),
    callback_after_step = rp -> push!(dne, maximum(abs, rp.plasma.ne .- ne_before)),
)
```

Mutate what a callback captures (`copyto!`, `push!`, `x[] = …` on a `Ref`) rather than
reassigning it. At the top level of a script, `n += 1` inside a callback fails unless `n` is
declared `global`.

A step refreshes the collision rates and transport coefficients at its end, before
`callback_after_step` runs. A callback that changes the plasma state (densities,
temperatures, velocities) calls `RAPID2D.update_transport_quantities!(RP)` afterwards, or the
next step runs on the rates of the old state.

The examples use three forms:

- **Plain functions:** `townsend_avalanche.jl` defines `count_before_step` and
  `record_growth_rate`, which measure the net growth rate of each step.
- **A callable struct:** in `force_balance_control.jl`, `CurvedBzControl` keeps the PID
  state and sets the controller field after each step.
- **A closure:** `full_startup.jl` gets one from `recorder(rec)` in `common.jl`, which
  records the closed-field-line node count.

## Shared pieces

`common.jl` holds what the scripts share:

- `setup`: config, then flags, then `initialize!`.
- `pure_toroidal(E0)`, the `ManualSetup` of a pure toroidal field (no poloidal field,
  Eϕ = E0·R̄/R), and the initial columns used by `current_diffusion.jl` and
  `force_balance_control.jl`.
- `run!`: runs to the end, or reports a failure mid-run and keeps the snapshots taken so far.
- The plots: `plot_traces`, `plot_dashboard`, `plot_snapshots2D`, `animate2D`.
- `StepRecord` and `recorder`, a per-step record for `callback_after_step`.
