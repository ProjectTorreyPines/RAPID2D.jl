# Current diffusion in a plasma filament driven by a loop voltage, in a pure toroidal field
# with Coulomb collisions as the only resistivity. Two runs: with the inductive back-EMF
# (E_para_self_EM) the current rises on the L/R time of the loop; without it, it jumps to the
# collisional saturation value. The current plot adds the analytic L/R curve of a circular loop.
#
#   julia --project=examples examples/current_diffusion.jl

include("common.jl")

const R_FIL, A_FIL, N_FIL = 1.5, 0.3, 1.0e16   # filament major/minor radius [m], density [m⁻³]

function filament_run(name; E_para_self_EM::Bool)
    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 30, NZ = 50, R0B0 = 3.0,
        prefilled_gas_pressure = 0.0,          # vacuum: no neutrals
        dt = 25.0e-6, t_end_s = 20.0e-3, snap0D_Δt_s = 50.0e-6, snap2D_Δt_s = 500.0e-6,
        Output_path = output_dir(name),
    )
    RP = setup(
        config;
        Ampere = true, Ampere_Itor_threshold = 1.0e-3, E_para_self_EM,
        ud_evolve = true, Coulomb_Collision = true,
        Atomic_Collision = false, src = false, convec = false, diffu = false,
        Te_evolve = false, Ti_evolve = false, Gas_evolve = false, update_ni_independently = false,
        Include_ud_convec_term = false, Include_ud_diffu_term = false, Include_ud_pressure_term = false,
        Include_Te_convec_term = false,
        E_para_self_ES = false, mean_ExB = false, turb_ExB_mixing = false,
        FLF_nstep = 100,
    )
    toroidal_field!(RP; E0 = 0.3)
    set_column!(RP, tophat(RP.G; cenR = R_FIL, radius = A_FIL, n0 = N_FIL))
    @time run_simulation!(RP)
    summarize(RP)
    return RP
end

# Circular-loop L/R reference: L = μ0 R (ln(8R/a) − 2 + ¼), R_p from the Coulomb-drag
# saturation current at the applied loop voltage.
function lr_reference(RP::RAPID)
    G, F, pla = RP.G, RP.fields, RP.plasma
    ee, me, μ0 = RP.config.constants.ee, RP.config.constants.me, RP.config.constants.μ0
    inw = G.nodes.in_wall_nids
    ue_sat = @. -ee * F.Eϕ_ext / (me * pla.ν_ei_eff)
    ue_sat[.!isfinite.(ue_sat)] .= 0.0
    LV = mean(F.LV_ext[inw])
    R_p = LV / sum(-ee .* pla.ne[inw] .* ue_sat[inw] .* G.dR .* G.dZ)
    L_p = μ0 * R_FIL * (log(8 * R_FIL / A_FIL) - 2 + 0.25)
    @printf("L/R reference: LV = %.3f V, R_p = %.3e Ω, L_p = %.3e H, τ = %.2f ms\n", LV, R_p, L_p, L_p / R_p * 1.0e3)
    t = RP.diagnostics.snaps0D.time_s
    return t, @. (LV / R_p) * (1 - exp(-t * R_p / L_p))
end

RP_noEM = filament_run("current_diffusion/no_Eind"; E_para_self_EM = false)
RP_EM = filament_run("current_diffusion/with_Eind"; E_para_self_EM = true)

out = output_dir("current_diffusion")
t_ref, I_ref = lr_reference(RP_EM)
fig = plot(xlabel = "t (ms)", ylabel = "I_pla (A)", legend = :bottomright, size = (700, 450))
plot!(fig, RP_noEM.diagnostics.snaps0D.time_s .* 1.0e3, RP_noEM.diagnostics.snaps0D.I_tor; lw = 2, label = "without E_ind")
plot!(fig, RP_EM.diagnostics.snaps0D.time_s .* 1.0e3, RP_EM.diagnostics.snaps0D.I_tor; lw = 2, label = "with E_ind")
plot!(fig, t_ref .* 1.0e3, I_ref; lw = 2, ls = :dash, c = :black, label = "analytic L/R")
savefig(fig, joinpath(out, "current.png"))
animate2D(["without E_ind" => RP_noEM, "with E_ind" => RP_EM], [:Jϕ]; file = joinpath(out, "Jphi.mp4"))
println("outputs in ", out)
