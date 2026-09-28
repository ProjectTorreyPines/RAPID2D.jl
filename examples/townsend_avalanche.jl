# Townsend avalanche in a single-quadrupole null field (static B, 5 V loop voltage).
# Atomic reactions and transport only: self-E and Ampère are off, so the density grows
# exponentially along B without saturation. A pair of step callbacks, written as plain
# functions below, measures the net growth rate of every step.
#
#   julia --project=examples examples/townsend_avalanche.jl

include("common.jl")

name = "townsend_avalanche"
config = SimulationConfig{Float64}(
    inputs = InputPaths(field = SINGLE_QUAD, wall = BOX_WALL),
    NR = 30, NZ = 50, R0B0 = 3.0, prefilled_gas_pressure = 2.0e-3,
    dt = 2.0e-6, t_end_s = 0.8e-3, snap0D_Δt_s = 10.0e-6, snap2D_Δt_s = 40.0e-6,
    Output_path = output_dir(name),
)
RP = setup(
    config;
    src = true, Atomic_Collision = true, Coulomb_Collision = false,
    diffu = true, convec = true, ud_evolve = true, Te_evolve = true, Ti_evolve = false,
    update_ni_independently = false, Gas_evolve = false,
    mean_ExB = false, turb_ExB_mixing = false,
    E_para_self_ES = false, E_para_self_EM = false, Ampere = false,
    FLF_nstep = 100,
)

# Net growth rate of each step, γ = ln(N_after / N_before) / Δt, with N the electron count
# inside the wall: one callback counts before the step, the other after it.
electron_count(RP) = sum(RP.plasma.ne .* RP.G.inVol2D)

const N_before = Ref(0.0)
const growth = (t = Float64[], γ = Float64[])

function count_before_step(RP)
    N_before[] = electron_count(RP)
    return nothing
end

function record_growth_rate(RP)
    push!(growth.t, RP.time_s)
    push!(growth.γ, log(electron_count(RP) / N_before[]) / RP.dt)
    return nothing
end

@time run_simulation!(RP; callback_before_step = count_before_step, callback_after_step = record_growth_rate)
summarize(RP)
@printf("net growth rate over the last step: %.3g s⁻¹\n", growth.γ[end])

out = config.Output_path
fig = plot(growth.t .* 1.0e3, growth.γ; lw = 2, xlabel = "t (ms)", ylabel = "net growth rate γ (s⁻¹)", legend = false)
savefig(fig, joinpath(out, "growth_rate.png"))
plot_traces(RP; file = joinpath(out, "traces.png"))
plot_dashboard(RP; file = joinpath(out, "dashboard.png"))
plot_snapshots2D(RP, :ne, [0.0, 0.2, 0.4, 0.8]; file = joinpath(out, "ne_snapshots.png"))
animate2D(RP, [:ne, :Te_eV, :ue_para]; file = joinpath(out, "snaps2D.mp4"))
println("outputs in ", out)
