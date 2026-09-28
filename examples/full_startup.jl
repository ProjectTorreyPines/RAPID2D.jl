# The whole start-up chain in one run: seed electrons → Townsend avalanche → self-E phase
# (E∥ cancellation, E×B mixing) → plasma current with Ampère and the inductive E → closed
# flux surfaces, in the single-quadrupole field of townsend_avalanche.jl and
# selfE_avalanche.jl. Every module is on except Global_JxB_Force.
#
# The step hook records the number of grid nodes on closed field lines, which no snapshot
# carries, so the closed-surface formation time comes out of the run.
#
#   julia --project=examples examples/full_startup.jl

include("common.jl")

name = "full_startup"
config = SimulationConfig{Float64}(
    inputs = InputPaths(field = SINGLE_QUAD, wall = BOX_WALL),
    NR = 30, NZ = 50, R0B0 = 3.0, prefilled_gas_pressure = 2.0e-3,
    dt = 2.0e-6, t_end_s = 10.0e-3, snap0D_Δt_s = 20.0e-6, snap2D_Δt_s = 100.0e-6,
    Output_path = output_dir(name),
)
RP = setup(
    config;
    src = true, Atomic_Collision = true, Coulomb_Collision = true, Spitzer_Resistivity = true,
    diffu = true, convec = true, ud_evolve = true, Te_evolve = true, Ti_evolve = true,
    update_ni_independently = true, Gas_evolve = true,
    mean_ExB = true, turb_ExB_mixing = true, E_para_self_ES = true,     # self-E
    Ampere = true, Ampere_Itor_threshold = 0.1, Ampere_nstep = 1,         # self-B
    E_para_self_EM = true,                                                # inductive E
    Global_JxB_Force = false,
    FLF_nstep = 50,
)

rec = StepRecord()
run!(RP; callback_after_step = recorder(rec))
summarize(RP)

k = findfirst(>(0), rec.n_closed)
isnothing(k) ? println("no closed field lines formed") :
    @printf("first closed field lines at t = %.3f ms (I_tor = %.3g A)\n", rec.t[k] * 1.0e3, rec.I_tor[k])

out = config.Output_path
plot_traces(RP; file = joinpath(out, "traces.png"))
plot_dashboard(RP; file = joinpath(out, "dashboard.png"))
p1 = plot(rec.t .* 1.0e3, rec.I_tor; lw = 2, ylabel = "I_tor (A)", legend = false)
p2 = plot(rec.t .* 1.0e3, rec.n_closed; lw = 2, ylabel = "nodes on closed lines", xlabel = "t (ms)", legend = false)
savefig(plot(p1, p2; layout = (2, 1), size = (700, 600)), joinpath(out, "closed_surface.png"))
plot_snapshots2D(RP, :ne, [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]; file = joinpath(out, "ne_snapshots.png"))
plot_snapshots2D(RP, :Jϕ, [2.0, 4.0, 6.0, 8.0, 10.0]; file = joinpath(out, "Jphi_snapshots.png"))
animate2D(RP, [:ne, :Te_eV, :Jϕ, :E_para_tot]; file = joinpath(out, "snaps2D.mp4"))
println("outputs in ", out)
