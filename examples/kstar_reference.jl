# A KSTAR start-up with its time-varying external field (Reference_201x201,
# 0–59 ms) and the self-E model on, Ampère off. The 201×201 input is interpolated onto the
# 30×50 grid. The KSTAR inputs are not shipped here: set RAPID_INPUT_PATH to the directory
# that holds KSTAR/Reference_201x201/ and KSTAR_First_Wall.dat.
#
#   RAPID_INPUT_PATH=/path/to/Input julia --project=examples examples/kstar_reference.jl

include("common.jl")
isempty(INPUT_PATH) && error("set RAPID_INPUT_PATH to the directory holding KSTAR/Reference_201x201/ and KSTAR_First_Wall.dat")

name = "kstar_reference"
config = SimulationConfig{Float64}(
    Input_path = INPUT_PATH, device_Name = "KSTAR", shot_Name = "Reference_201x201",
    NR = 30, NZ = 50, R0B0 = -2.7 * 1.8, prefilled_gas_pressure = 4.0e-3,
    dt = 10.0e-6, t_end_s = 40.0e-3, snap0D_Δt_s = 100.0e-6, snap2D_Δt_s = 500.0e-6,
    Output_path = output_dir(name),
)
RP = setup(
    config;
    src = true, Atomic_Collision = true, Coulomb_Collision = true,
    diffu = true, convec = true, ud_evolve = true, Te_evolve = true, Ti_evolve = false,
    update_ni_independently = false, Gas_evolve = false,
    mean_ExB = true, turb_ExB_mixing = true, E_para_self_ES = true,
    E_para_self_EM = false, Ampere = false,
    FLF_nstep = 50,
)

run!(RP)
summarize(RP)

out = config.Output_path
plot_traces(RP; file = joinpath(out, "traces.png"))
plot_dashboard(RP; file = joinpath(out, "dashboard.png"))
plot_snapshots2D(RP, :ne, [0.0, 10.0, 20.0, 30.0, 40.0]; file = joinpath(out, "ne_snapshots.png"))
animate2D(RP, [:ne, :Te_eV, :E_para_tot]; file = joinpath(out, "snaps2D.mp4"))
println("outputs in ", out)
