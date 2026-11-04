# Ohmic heating where only Coulomb collisions remain, against 0D theory. Once the gas has burnt through, the
# resistivity is Spitzer's, η ∝ lnΛ Tₑ^{−3/2}; the current's Ohmic heat goes to the electrons, and the e–i exchange
# passes part of it on to the ions.
#
# 2a  A uniform plasma under a fixed parallel field, without induction. Each node follows the ODE
#         ḋ = −(eE/mₑ + ν_eff d)(1 + mₑ/mᵢ),   (3/2) Ṫₑ = mₑ ν_eff d²/e − (3/2) a (Tₑ − Tᵢ),   (3/2) Ṫᵢ = (3/2) a (Tₑ − Tᵢ)
#     for the relative drift d, ν_eff = ξ ν_ei and a = 2mₑmᵢ/(mₑ + mᵢ)² ν_ei. In the quasi-steady drift, with lnΛ
#     fixed and no exchange, Tₑ(t) = [T₀^{−1/2} − σ₀E²t/(3neT₀^{3/2})]^{−2}: a fixed field with nothing to limit the
#     current runs away in finite time.
# 2b  The filament of current_diffusion.jl at n = 1e18 m⁻³: its current first steady at fixed temperatures, then Tₑ
#     and Tᵢ free, against the circuit (L + L_kin) İ = V − R(Tₑ) I with the column's energy balance. As the plasma
#     heats, R falls and the current turns toward the inductive limit. The skin current and J ∝ E ∝ 1/R heat the
#     inner edge most; Tₑ peaks there and the current gathers there, ahead of the uniform model. Tₑ conducts along
#     the current's own closed surfaces, and the minimum at their axis rises with the rest of the column.
#
#   julia --project=examples examples/ohmic_heating.jl

include("common.jl")

# ── theory (internal/docs/src/reference/coulomb-collisions.md) ─────────────────────────────────────────────────────

const XI = 0.510469472194728   # the Spitzer–Härm factor at Z = 1

"The e–i Coulomb logarithm: NRL p.34(b), Z = 1, cgs inside, floored at 1."
function lnΛ_ei(n, Te, Ti, mi, me, mp)
    ncgs = n * 1.0e-6
    l = Te < Ti * me / mi ? 16.0 - log(sqrt(ncgs) * Ti^(-1.5) * (mi / mp)) :
        Te > 10.0 ? 24.0 - log(sqrt(ncgs) / Te) : 23.0 - log(sqrt(ncgs) * Te^(-1.5))
    return max(l, 1.0)
end

"ν_ei(n, Tₑ, Tᵢ) for the constants `c` and ion mass `mi`."
function nu_ei(c, mi)
    kν = sqrt(2) * c.ee^4 / (12 * π^1.5 * c.eps0^2 * sqrt(c.me))
    return (n, Te, Ti) -> kν * n * lnΛ_ei(n, Te, Ti, mi, c.me, c.mp) * (c.ee * Te)^(-1.5)
end

"RK4 of `rhs` from `s0`, sampled at the increasing `ts`, with steps no longer than `hmax(s)`."
function rk4(rhs, s0, ts; hmax)
    s, t, out = s0, 0.0, typeof(s0)[]
    for tt in ts
        while t < tt - 1.0e-15
            h = min(hmax(s), tt - t)
            k1 = rhs(s)
            k2 = rhs(s .+ (h / 2) .* k1)
            k3 = rhs(s .+ (h / 2) .* k2)
            k4 = rhs(s .+ h .* k3)
            s = s .+ (h / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
            t += h
        end
        push!(out, s)
    end
    return out
end

# ── 2a: a uniform plasma under a fixed field ──────────────────────────────────────────────────────────────────────

function ohmic_box(; n, T0, E0, dt, t_end)
    config = SimulationConfig{Float64}(;
        device_Name = "manual", manual = ManualSetup{Float64}(BR = 0.0, BZ = 0.0, Eϕ = E0),
        NR = 9, NZ = 9, R_min = 1.0, R_max = 2.0, Z_min = -0.5, Z_max = 0.5,
        wall_R = [1.15, 1.85, 1.85, 1.15], wall_Z = [-0.35, -0.35, 0.35, 0.35],
        prefilled_gas_pressure = 0.0, R0B0 = 1.0, dt, t_end_s = t_end, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
        Output_path = mktempdir(),
    )
    RP = setup(
        config; Implicit = true, diffu = false, convec = false, src = false,
        Atomic_Collision = false, Coulomb_Collision = true, Spitzer_Resistivity = true,
        mean_ExB = false, turb_ExB_mixing = false, charge_sheet_field = false, E_para_cancellation = false,
        E_para_induction = false, Ampere = false, Te_evolve = true, Ti_evolve = true, ud_evolve = true, Gas_evolve = false,
        update_ni_independently = true, secondary_electron = false, negative_n_correction = false,
        Include_ud_pressure_term = false, Include_ud_convec_term = false, Include_ud_diffu_term = false,
        Include_Te_convec_term = false, Include_Te_diffu_term = false, Include_heat_flux_term = false,
        vacuum_relaxation = (state = false, factor = 1.0e2),
    )
    G, pla = RP.G, RP.plasma
    inw = G.nodes.in_wall_nids
    RP.fields.LV_presc .= 2π .* G.R2D .* E0              # a uniform E (a loop voltage would give E ∝ 1/R)
    RAPID2D.update_prescribed_fields!(RP)
    RAPID2D.combine_field_sources!(RP)
    pla.ne .= 0.0
    vec(pla.ne)[inw] .= n
    pla.ni .= pla.ne
    fill!(pla.Te_eV, T0)
    fill!(pla.Ti_eV, T0)
    fill!(pla.ue_para, 0.0)
    fill!(pla.ui_para, 0.0)
    RAPID2D.update_transport_quantities!(RP)
    return RP
end

"Run a box; t, Tₑ, Tᵢ, d at its centre node every step, and the 0D reference at the same times."
function box_run(; n, T0, E0, dt, t_end)
    RP = ohmic_box(; n, T0, E0, dt, t_end)
    k = RP.G.nodes.in_wall_nids[cld(length(RP.G.nodes.in_wall_nids), 2)]
    t, Te, Ti, d = Float64[], Float64[], Float64[], Float64[]
    run_simulation!(
        RP; callback_after_step = function (rp)
            p = rp.plasma
            push!(t, rp.time_s); push!(Te, p.Te_eV[k]); push!(Ti, p.Ti_eV[k]); push!(d, p.ue_para[k] - p.ui_para[k])
            return nothing
        end
    )
    c, mi = RP.config.constants, RAPID2D.bulk_ion_mass(RP)
    ν = nu_ei(c, mi)
    rate(Te, Ti) = 2 * c.me * mi / (c.me + mi)^2 * ν(n, Te, Ti)
    rhs(s) = (
        -(c.ee * E0 / c.me + XI * ν(n, s[2], s[3]) * s[1]) * (1 + c.me / mi),
        (c.me * XI * ν(n, s[2], s[3]) * s[1]^2 / c.ee) / 1.5 - rate(s[2], s[3]) * (s[2] - s[3]),
        rate(s[2], s[3]) * (s[2] - s[3]),
    )
    ode = rk4(rhs, (0.0, T0, T0), t; hmax = s -> min(0.02 / (XI * ν(n, s[2], s[3])), 1.0e-8))
    # the reduced limit: quasi-steady drift, lnΛ fixed at T₀, no exchange
    σ0 = n * c.ee^2 / (c.me * XI * ν(n, T0, T0))
    pole = 3 * n * c.ee * T0^1.5 / (σ0 * E0^2) * T0^(-0.5)
    reduced = [tt < pole ? (T0^(-0.5) - σ0 * E0^2 * tt / (3 * n * c.ee * T0^1.5))^(-2) : NaN for tt in t]
    return (; t, Te, Ti, d, ode, reduced, pole)
end

out = output_dir("ohmic_heating")

# (a) the heating dominates: n = 1e17, E = 0.1 V/m; (b) the ions follow: n = 1e19, E = 1 V/m
runs_a = box_run(; n = 1.0e17, T0 = 4.0, E0 = 0.1, dt = 0.5e-6, t_end = 0.9e-3)
runs_b = box_run(; n = 1.0e19, T0 = 4.0, E0 = 1.0, dt = 0.5e-6, t_end = 1.2e-3)
p1 = plot(xlabel = "t (ms)", ylabel = "T (eV)", title = "n = 1e17 m⁻³, E = 0.1 V/m", legend = :topleft)
plot!(p1, runs_a.t .* 1.0e3, runs_a.Te; lw = 2, label = "Tₑ RAPID2D")
plot!(p1, runs_a.t .* 1.0e3, getindex.(runs_a.ode, 2); lw = 1.5, ls = :dash, c = :black, label = "Tₑ ODE")
plot!(p1, runs_a.t .* 1.0e3, runs_a.reduced; lw = 1.2, ls = :dot, c = :gray40, label = "quasi-steady, no exchange")
plot!(p1, runs_a.t .* 1.0e3, runs_a.Ti; lw = 2, label = "Tᵢ RAPID2D")
plot!(p1, runs_a.t .* 1.0e3, getindex.(runs_a.ode, 3); lw = 1.5, ls = :dash, c = :black, label = "")
p2 = plot(xlabel = "t (ms)", ylabel = "T (eV)", title = "n = 1e19 m⁻³, E = 1 V/m", legend = :topleft)
plot!(p2, runs_b.t .* 1.0e3, runs_b.Te; lw = 2, label = "Tₑ RAPID2D")
plot!(p2, runs_b.t .* 1.0e3, getindex.(runs_b.ode, 2); lw = 1.5, ls = :dash, c = :black, label = "ODE")
plot!(p2, runs_b.t .* 1.0e3, runs_b.Ti; lw = 2, label = "Tᵢ RAPID2D")
plot!(p2, runs_b.t .* 1.0e3, getindex.(runs_b.ode, 3); lw = 1.5, ls = :dash, c = :black, label = "")

# convergence: the end-point error inside one branch of the logarithm (Tₑ < 9 eV)
dts = [2.0e-6, 1.0e-6, 0.5e-6, 0.25e-6]
errs = map(dts) do dt
    r = box_run(; n = 1.0e19, T0 = 4.0, E0 = 1.0, dt, t_end = 0.7e-3)
    maximum(abs(r.Te[end] - r.ode[end][2]) / r.ode[end][2] for _ in 1:1)
end
p3 = plot(
    dts .* 1.0e6, errs; xscale = :log10, yscale = :log10, marker = :circle, lw = 2, xlabel = "Δt (µs)",
    ylabel = "|Tₑ − ODE| / ODE at 0.7 ms", label = "RAPID2D", legend = :topleft, title = "first order in Δt"
)
plot!(p3, dts .* 1.0e6, errs[1] .* dts ./ dts[1]; ls = :dash, c = :black, label = "∝ Δt")
order = log2(errs[end - 1] / errs[end])
fig_a = plot(p1, p2, p3; layout = (1, 3), size = (1500, 450), left_margin = 5Plots.mm, bottom_margin = 6Plots.mm)
save_with_verdict(
    fig_a, out, "uniform_box", 0.85 < order < 1.15 && errs[end] < 5.0e-3,
    @sprintf("a uniform plasma under a fixed field follows the Coulomb-only ODE, first order in Δt (order %.2f, %.2g %% at 0.25 µs)", order, 100errs[end]),
)

# ── 2b: the inductive filament ────────────────────────────────────────────────────────────────────────────────────

const R_FIL, A_FIL, N_FIL, T_FIL = 1.5, 0.3, 1.0e18, 10.0

function filament_heating(; t_prep = 40.0e-3, t_heat = 10.0e-3, dt = 25.0e-6)
    config = SimulationConfig{Float64}(;
        device_Name = "manual", manual = pure_toroidal(0.3), NR = 30, NZ = 50, R0B0 = 3.0, prefilled_gas_pressure = 0.0,
        dt, t_end_s = t_prep, snap0D_Δt_s = 1.0e-4, snap2D_Δt_s = 1.0e-3, Output_path = output_dir("ohmic_heating/filament"),
    )
    RP = setup(
        config; Ampere = true, Ampere_Itor_threshold = 0.0, E_para_induction = true, ud_evolve = true,
        Coulomb_Collision = true, Atomic_Collision = false, src = false, convec = false, diffu = false,
        Te_evolve = false, Ti_evolve = false, Gas_evolve = false, update_ni_independently = false,
        Include_ud_convec_term = false, Include_ud_diffu_term = false, Include_ud_pressure_term = false,
        Include_Te_convec_term = false, Include_Te_diffu_term = true, Include_heat_flux_term = false,
        charge_sheet_field = false, E_para_cancellation = false, mean_ExB = false,
        turb_ExB_mixing = false, FLF_nstep = 100, vacuum_relaxation = (state = false, factor = 1.0e2),
    )
    set_column!(RP, tophat(RP.G; cenR = R_FIL, radius = A_FIL, n0 = N_FIL); Te_eV = T_FIL, Ti_eV = T_FIL)
    run_simulation!(RP)                                 # the steady current at fixed temperatures
    G, c = RP.G, RP.config.constants
    col = findall(>(0), vec(RP.plasma.ne))
    I0 = RP.diagnostics.snaps0D.I_tor[end]
    RP.flags.Te_evolve = true
    RP.flags.Ti_evolve = true
    RP.t_end_s = t_prep + t_heat
    w = vec(G.inVol2D)[col] .* N_FIL
    t, I, Te, Ti, spread, Te_min = Float64[], Float64[], Float64[], Float64[], Float64[], Float64[]
    run_simulation!(
        RP; callback_after_step = function (rp)
            rp.step % 4 == 0 || return nothing
            p = rp.plasma
            push!(t, rp.time_s - t_prep); push!(I, rp.diagnostics.snaps0D.I_tor[end])
            push!(Te, sum(w .* vec(p.Te_eV)[col]) / sum(w)); push!(Ti, sum(w .* vec(p.Ti_eV)[col]) / sum(w))
            push!(spread, maximum(vec(p.Te_eV)[col]) - minimum(vec(p.Te_eV)[col]))
            push!(Te_min, minimum(vec(p.Te_eV)[col]))
            return nothing
        end
    )
    # the uniform column's circuit and energy balance
    mi = RAPID2D.bulk_ion_mass(RP)
    ν = nu_ei(c, mi)
    LV = mean(vec(RP.fields.LV_presc)[col])
    R_cells, dA = vec(G.R2D)[col], G.dR * G.dZ
    σ(Te, Ti) = N_FIL * c.ee^2 / (c.me * XI * ν(N_FIL, Te, Ti))
    R_p(Te, Ti) = 2π / (σ(Te, Ti) * sum(dA ./ R_cells))
    Vol = sum(2π .* R_cells .* dA)
    L = c.μ0 * R_FIL * (log(8 * R_FIL / A_FIL) - 2 + 0.25)
    L_kin = c.me / (N_FIL * c.ee^2) * (2π * R_FIL / (π * A_FIL^2))
    rate(Te, Ti) = 2 * c.me * mi / (c.me + mi)^2 * ν(N_FIL, Te, Ti)
    rhs(s) = (
        (LV - R_p(s[2], s[3]) * s[1]) / (L + L_kin),
        (s[1]^2 * R_p(s[2], s[3]) / Vol / (N_FIL * c.ee)) / 1.5 - rate(s[2], s[3]) * (s[2] - s[3]),
        rate(s[2], s[3]) * (s[2] - s[3]),
    )
    ode = rk4(rhs, (I0, T_FIL, T_FIL), t; hmax = _ -> 1.0e-6)
    @printf("2b: steady I = %.5g A, the uniform column's %.5g A; L/R = %.2f ms\n", I0, LV / R_p(T_FIL, T_FIL), (L + L_kin) / R_p(T_FIL, T_FIL) * 1.0e3)
    return (; RP, t, I, Te, Ti, spread, Te_min, ode, I0, I_sat = LV / R_p(T_FIL, T_FIL), inductive = LV / (L + L_kin))
end

fil = filament_heating()
tms = fil.t .* 1.0e3
q1 = plot(xlabel = "t after the temperatures are freed (ms)", ylabel = "I (A)", legend = :topleft, title = "current")
plot!(q1, tms, fil.I; lw = 2, label = "RAPID2D")
plot!(q1, tms, getindex.(fil.ode, 1); lw = 1.5, ls = :dash, c = :black, label = "0D circuit + energy")
plot!(q1, tms, fil.I0 .+ fil.inductive .* fil.t; lw = 1, ls = :dot, c = :gray40, label = "inductive limit I₀ + V t/(L + L_kin)")
q2 = plot(xlabel = "t (ms)", ylabel = "T (eV)", legend = :topleft, title = "column averages")
plot!(q2, tms, fil.Te; lw = 2, label = "⟨Tₑ⟩ RAPID2D")
plot!(q2, tms, getindex.(fil.ode, 2); lw = 1.5, ls = :dash, c = :black, label = "0D")
plot!(q2, tms, fil.Ti; lw = 2, label = "⟨Tᵢ⟩ RAPID2D")
plot!(q2, tms, getindex.(fil.ode, 3); lw = 1.5, ls = :dash, c = :black, label = "")
plot!(q2, tms, fil.spread; lw = 1.2, ls = :dashdot, label = "Tₑ max − min in the column")
plot!(q2, tms, fil.Te_min; lw = 1.2, ls = :dot, c = :purple, label = "min Tₑ in the column")
gapI = 100 .* (fil.I .- getindex.(fil.ode, 1)) ./ getindex.(fil.ode, 1)
gapTe = 100 .* (fil.Te .- getindex.(fil.ode, 2)) ./ getindex.(fil.ode, 2)
gapTi = 100 .* (fil.Ti .- getindex.(fil.ode, 3)) ./ getindex.(fil.ode, 3)
q3 = plot(
    tms, [gapI gapTe gapTi]; lw = 2, label = ["I" "⟨Tₑ⟩" "⟨Tᵢ⟩"], xlabel = "t (ms)", ylabel = "RAPID2D − 0D (%)",
    title = "difference from the uniform model", legend = :topleft
)
vline!(q3, [3.0]; c = :gray50, ls = :dash, label = "the regression test's window")
G = fil.RP.G
q4 = heatmap(
    G.R1D, G.Z1D, fil.RP.plasma.Te_eV'; aspect_ratio = :equal, xlabel = "R (m)", ylabel = "Z (m)",
    title = @sprintf("Tₑ (eV) at %.0f ms", tms[end]), xlims = (1.1, 1.9), ylims = (-0.4, 0.4)
)
q5 = heatmap(
    G.R1D, G.Z1D, fil.RP.plasma.Jϕ'; aspect_ratio = :equal, xlabel = "R (m)", ylabel = "Z (m)",
    title = "Jϕ (A/m²)", xlims = (1.1, 1.9), ylims = (-0.4, 0.4)
)
fig_b = plot(q1, q2, q3, q4, q5; layout = @layout([a b c; d e]), size = (1500, 900), left_margin = 5Plots.mm, bottom_margin = 5Plots.mm)
win = findall(<=(3.0), tms)
rising = all(diff(fil.Te_min) .>= -1.0e-6)
ok = abs(fil.I0 - fil.I_sat) / fil.I_sat < 5.0e-3 && maximum(abs, gapI[win]) < 5 && maximum(abs, gapTe[win]) < 1 &&
    maximum(abs, gapTi[win]) < 1 && rising
save_with_verdict(
    fig_b, out, "filament", ok,
    @sprintf(
        "steady current %.2f %% from the uniform column's; over 3 ms of heating I, ⟨Tₑ⟩, ⟨Tᵢ⟩ within %.1f, %.1f, %.1f %% of the 0D circuit (needs 5, 1, 1); with Tₑ conduction on, the column's minimum %s",
        100 * (fil.I0 - fil.I_sat) / fil.I_sat, maximum(abs, gapI[win]), maximum(abs, gapTe[win]), maximum(abs, gapTi[win]),
        rising ? "never falls" : "falls"
    ),
)
println("outputs in ", out)
