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

## Scenarios

| script | field and geometry | physics on | simulated time |
|---|---|---|---|
| `townsend_avalanche.jl` | single-quadrupole null, box wall | atomic reactions, transport | 0.8 ms |
| `selfE_avalanche.jl` | single-quadrupole null, box wall | as `townsend_avalanche.jl`, plus E∥ cancellation, mean E×B, turbulent E×B mixing | 4 ms |
| `current_diffusion.jl` | pure toroidal field | a current filament with Ampère, with and without the inductive E; analytic L/R reference | 2 × 20 ms |
| `force_balance_control.jl` | pure toroidal field | J×B hoop force; curved vertical-field PID position control | 2 × 2 ms |
| `full_startup.jl` | single-quadrupole null, box wall | every module except the global J×B force | 10 ms |
| `kstar_reference.jl` | KSTAR, time-varying external field | self-E model, Ampère off | 40 ms |

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
- The pure-toroidal-field helpers and the initial columns used by `current_diffusion.jl` and
  `force_balance_control.jl`.
- `run!`: runs to the end, or reports a failure mid-run and keeps the snapshots taken so far.
- The plots: `plot_traces`, `plot_dashboard`, `plot_snapshots2D`, `animate2D`.
- `StepRecord` and `recorder`, a per-step record for `callback_after_step`.
