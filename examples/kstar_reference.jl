# The KSTAR reference start-up (Reference_201x201) from seed electrons to growing closed flux
# surfaces, over the 0–59 ms of its time-varying external field: Townsend avalanche, the
# self-E phase (E∥ cancellation, E×B mixing), the plasma current with Ampère and the
# inductive E, then closed flux surfaces. The 201×201 input is interpolated onto the 30×50
# grid.
#
# Two pieces hold the closed region once it forms:
# - On closed surfaces the electron parallel flow is constrained so that each surface carries
#   the current, not the density (`SurfaceFluxConstraint`, solved inside the coupled u∥–ψ
#   block).
# - The prescribed field is a feedforward scenario; it does not follow the current column,
#   which cannot move in the model. From 40 ms an ideal external vertical field
#   (`IdealVerticalFieldControl`) cancels the column's radial force estimate, so the run
#   evolves under quasi-equilibrium. Its own induced field drives current too.
#
# Every other flag is at its default. The KSTAR inputs are not shipped here: set
# RAPID_INPUT_PATH to the directory that holds KSTAR/Reference_201x201/ and KSTAR_First_Wall.dat.
#
#   RAPID_INPUT_PATH=/path/to/Input julia --project=examples examples/kstar_reference.jl
#   julia --project=examples examples/plot_run.jl examples/output/kstar_reference   # the figures again, from its files

include("common.jl")
using RAPID2D: SurfaceFluxConstraint, IdealVerticalFieldControl
isempty(INPUT_PATH) && error("set RAPID_INPUT_PATH to the directory holding KSTAR/Reference_201x201/ and KSTAR_First_Wall.dat")

name = "kstar_reference"
config = SimulationConfig{Float64}(
    Input_path = INPUT_PATH, device_Name = "KSTAR", shot_Name = "Reference_201x201",
    NR = 30, NZ = 50, R0B0 = -2.7 * 1.8, prefilled_gas_pressure = 4.0e-3,
    dt = 20.0e-6, t_end_s = 59.0e-3, snap0D_Δt_s = 100.0e-6, snap2D_Δt_s = 200.0e-6,   # the field file ends at 59 ms
    Output_path = output_dir(name),
)
flags = (; Ampere = true, closed_surface_flow = SurfaceFluxConstraint(), vertical_field_control = IdealVerticalFieldControl(t_on_s = 40.0e-3))

RP = setup(config; flags...)
run!(RP)
summarize(RP)

# the figures read the run's files, as examples/plot_run.jl does to draw them again without running
title = "KSTAR reference, closed-surface constraint + ideal vertical field from 40 ms"
plot_run_figures(
    load_run(config.Output_path; prefix = config.Output_prefix); title,
    table_times_ms = vcat(36:2:58, 59), snapshot_times_ms = [30.0, 38.0, 42.0, 46.0, 50.0, 54.0, 59.0],
)
