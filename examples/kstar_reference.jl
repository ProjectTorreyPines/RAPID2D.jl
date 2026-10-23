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
RP = setup(
    config;
    Ampere = true,
    closed_surface_flow = SurfaceFluxConstraint(),
    vertical_field_control = IdealVerticalFieldControl(t_on_s = 40.0e-3),
)

run!(RP)
summarize(RP)

# the current column every 2 ms once closed surfaces can form, with the magnetic LCFS (through
# an X-point or touching the wall): its geometric centre R_geo, from which the Shafranov shift
# Δ = R_axis − R_geo is measured, its area centroid R_area, and what limits it
s0, s2 = RP.diagnostics.snaps0D, RP.diagnostics.snaps2D
t2 = [x.time_s for x in s2]
magnetic_of(x) = snapshot_lcfs(RP, x).magnetic
println("  t     |I_tor|   closed   R_cen   R_axis   R_geo  R_area  R_axis−R_geo  limited by    a₁      ⟨Tₑ⟩")
for tms in vcat(36:2:58, 59)
    x, m = nearest_snap0D(RP, tms * 1.0e-3), magnetic_of(s2[argmin(abs.(t2 .- tms * 1.0e-3))])
    R_geo, R_area, by = m === nothing ? (NaN, NaN, "–") : (m.R_geo, m.R_area, string(m.limited_by))
    @printf(
        "%5.1f  %6.2f kA %5.2f kA  %.3f m  %.3f m  %.3f  %.3f   %+6.3f m   %-8s  %5.2f mT  %5.2f eV\n", 1.0e3x.time_s,
        1.0e-3abs(x.I_tor), 1.0e-3abs(x.I_closed), x.R_cen_by_Jϕ, x.R_axis, R_geo, R_area, x.R_axis - R_geo, by, 1.0e3x.ctrl_a1, x.Te_eV
    )
end

out = config.Output_path
title = "KSTAR reference, closed-surface constraint + ideal vertical field from 40 ms"
t = s0.time_s .* 1.0e3
p1 = plot(t, 1.0e-3abs.(s0.I_tor); lw = 2, label = "I_tor", ylabel = "|I| (kA)", legend = :topleft)
plot!(p1, t, 1.0e-3abs.(s0.I_closed); lw = 2, label = "on closed lines")
p2 = plot(t, 1.0e3s0.ctrl_a1; lw = 2, ylabel = "control a₁ = B_Z (mT)", legend = false)
p3 = plot(t, s0.R_cen_by_Jϕ; lw = 2, label = "current centroid", ylabel = "R (m)", legend = :bottomright)
plot!(p3, t, s0.R_axis; lw = 2, label = "magnetic axis")
R_geo = [(m = magnetic_of(x); m === nothing ? NaN : m.R_geo) for x in s2]
plot!(p3, 1.0e3 .* t2, R_geo; lw = 2, label = "magnetic LCFS, geometric centre")
p4 = plot(t, s0.Te_eV; lw = 2, ylabel = "⟨Tₑ⟩ (eV)", xlabel = "t (ms)", legend = false)
savefig(
    plot(p1, p2, p3, p4; layout = (4, 1), size = (700, 1000), left_margin = 8Plots.mm, plot_title = title, plot_titlefontsize = 10),
    joinpath(out, "column.png")
)
plot_traces(RP; file = joinpath(out, "traces.png"))
plot_dashboard(RP; file = joinpath(out, "dashboard.png"))
plot_fields_snapshots(RP, [30.0, 38.0, 42.0, 46.0, 50.0, 54.0, 59.0]; file = joinpath(out, "fields_snapshots.png"), title)
animate_fields(RP; file = joinpath(out, "fields.mp4"), title)
println("outputs in ", out)
