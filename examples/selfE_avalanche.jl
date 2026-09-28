# The avalanche of townsend_avalanche.jl with the self-electric-field model on: parallel E∥
# cancellation (E_para_self_ES), the mean E×B drift and turbulent E×B mixing. The space
# charge limits the growth and spreads the plasma along the poloidal field.
#
#   julia --project=examples examples/selfE_avalanche.jl

include("common.jl")

name = "selfE_avalanche"
config = SimulationConfig{Float64}(
    inputs = InputPaths(field = SINGLE_QUAD, wall = BOX_WALL),
    NR = 30, NZ = 50, R0B0 = 3.0, prefilled_gas_pressure = 2.0e-3,
    dt = 2.0e-6, t_end_s = 4.0e-3, snap0D_Δt_s = 20.0e-6, snap2D_Δt_s = 100.0e-6,
    Output_path = output_dir(name),
)
RP = setup(
    config;
    src = true, Atomic_Collision = true, Coulomb_Collision = true,
    diffu = true, convec = true, ud_evolve = true, Te_evolve = true, Ti_evolve = false,
    update_ni_independently = false, Gas_evolve = false,
    mean_ExB = true, turb_ExB_mixing = true, E_para_self_ES = true,   # the self-E model
    E_para_self_EM = false, Ampere = false,
    FLF_nstep = 100,
)

@time run_simulation!(RP)
summarize(RP)

out = config.Output_path
plot_traces(RP; file = joinpath(out, "traces.png"))
plot_dashboard(RP; file = joinpath(out, "dashboard.png"))
plot_snapshots2D(RP, :ne, [0.0, 1.0, 2.0, 4.0]; file = joinpath(out, "ne_snapshots.png"))
animate2D(RP, [:ne, :Te_eV, :E_para_tot, :D_pol]; file = joinpath(out, "snaps2D.mp4"))
println("outputs in ", out)
