# Current rise in a plasma filament driven by a loop voltage, in a pure toroidal field with
# Coulomb collisions as the only resistivity. Two runs: without the inductive E∥
# (E_para_self_EM) the current saturates on the electron momentum-relaxation time 1/ν; with
# it, the current rises on the L/R time of the loop, where L includes the electrons' inertia
# (their kinetic inductance). The current plot adds that single-filament L/R curve.
#
#   julia --project=examples examples/current_diffusion.jl

include("common.jl")

const R_FIL, A_FIL, N_FIL = 1.5, 0.3, 1.0e16   # filament major/minor radius [m], density [m⁻³]
const E0 = 0.3                                  # toroidal E at the mean R [V/m]

function filament_run(name; E_para_self_EM::Bool)
    config = SimulationConfig{Float64}(
        device_Name = "manual", manual = pure_toroidal(E0), NR = 30, NZ = 50, R0B0 = 3.0,
        prefilled_gas_pressure = 0.0,          # vacuum: no neutrals
        dt = 25.0e-6, t_end_s = 20.0e-3, snap0D_Δt_s = 50.0e-6, snap2D_Δt_s = 500.0e-6,
        Output_path = output_dir(name),
    )
    RP = setup(
        config;
        # A benchmark setting: Ampère from the first step. The column starts formed at I = 0, so
        # under any threshold its first step accelerates with no back-EMF (to ~550 A here) and
        # the next takes that current back. Runs that grow from breakdown keep the default (1 A).
        Ampere = true, Ampere_Itor_threshold = 0.0, E_para_self_EM,
        ud_evolve = true, Coulomb_Collision = true,
        Atomic_Collision = false, src = false, convec = false, diffu = false,
        Te_evolve = false, Ti_evolve = false, Gas_evolve = false, update_ni_independently = false,
        Include_ud_convec_term = false, Include_ud_diffu_term = false, Include_ud_pressure_term = false,
        Include_Te_convec_term = false,
        E_para_self_ES = false, mean_ExB = false, turb_ExB_mixing = false,
        FLF_nstep = 100,
    )
    set_column!(RP, tophat(RP.G; cenR = R_FIL, radius = A_FIL, n0 = N_FIL))
    @time run_simulation!(RP)
    summarize(RP)
    return RP
end

# Single-filament circuit (L + L_kin) dI/dt + R_p I = V, returned with and without L_kin, where
#   L     = μ0 R (ln(8R/a) − 2 + li/2), li = 1/2   (uniform current, circular loop)
#   L_kin = mₑ 2πR / (n e² πa²)                     (electron inertia; L_kin/R_p = 1/ν)
#   R_p   = V / I_sat, I_sat from the Coulomb-drag saturation drift at the applied field.
function lr_reference(RP::RAPID)
    G, F, pla = RP.G, RP.fields, RP.plasma
    ee, me, μ0 = RP.config.constants.ee, RP.config.constants.me, RP.config.constants.μ0
    col = findall(>(0), pla.ne)
    ue_sat = @. -ee * F.Eϕ_ext[col] / (me * pla.ν_ei_eff[col])
    LV = mean(F.LV_ext[col])
    R_p = LV / sum(@. -ee * pla.ne[col] * ue_sat * G.dR * G.dZ)
    L_p = μ0 * R_FIL * (log(8 * R_FIL / A_FIL) - 2 + 0.25)
    L_kin = me / (N_FIL * ee^2) * (2π * R_FIL / (π * A_FIL^2))
    τ = (L_p + L_kin) / R_p
    @printf(
        "L/R reference: LV = %.3f V, R_p = %.3e Ω, L_p = %.3e H, L_kin = %.3e H, τ = %.2f ms (L_p/R_p = %.2f ms)\n",
        LV, R_p, L_p, L_kin, τ * 1.0e3, L_p / R_p * 1.0e3
    )
    t = RP.diagnostics.snaps0D.time_s
    return t, @.((LV / R_p) * (1 - exp(-t / τ))), @.((LV / R_p) * (1 - exp(-t * R_p / L_p)))
end

RP_noEM = filament_run("current_diffusion/no_Eind"; E_para_self_EM = false)
RP_EM = filament_run("current_diffusion/with_Eind"; E_para_self_EM = true)

out = output_dir("current_diffusion")
t_ref, I_ref, I_ref_noLkin = lr_reference(RP_EM)
fig = plot(xlabel = "t (ms)", ylabel = "I_pla (A)", legend = :bottomright, size = (700, 450))
plot!(fig, RP_noEM.diagnostics.snaps0D.time_s .* 1.0e3, RP_noEM.diagnostics.snaps0D.I_tor; lw = 2, label = "without E_ind")
plot!(fig, RP_EM.diagnostics.snaps0D.time_s .* 1.0e3, RP_EM.diagnostics.snaps0D.I_tor; lw = 2, label = "with E_ind")
plot!(fig, t_ref .* 1.0e3, I_ref; lw = 2, ls = :dash, c = :black, label = "analytic (L + L_kin)/R")
plot!(fig, t_ref .* 1.0e3, I_ref_noLkin; lw = 1.5, ls = :dot, c = :gray40, label = "analytic L/R (no L_kin)")
# As test/regression/inductance_test.jl: with the inductive field the current follows the
# circuit within 1 % of its saturation value at every snapshot.
gap = maximum(abs.(RP_EM.diagnostics.snaps0D.I_tor .- I_ref)) / maximum(I_ref)
plot!(fig; top_margin = 6Plots.mm)
save_with_verdict(
    fig, out, "current", gap < 0.01,
    @sprintf("with E_ind the current follows the (L + L_kin)/R circuit within %.2g %% of its saturation at every snapshot (needs 1 %%)", 100gap),
)
animate2D(["without E_ind" => RP_noEM, "with E_ind" => RP_EM], [:Jϕ]; file = joinpath(out, "Jphi.mp4"))
println("outputs in ", out)
