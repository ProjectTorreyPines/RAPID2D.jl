# Inductive current rise with Ohmic heating where only Coulomb collisions remain (case 2b of
# internal/docs/src/notes/plans/PLAN_ohmic-heating-tests.md): the filament of inductance_test.jl at n = 1e18 m⁻³, its
# current first brought to the steady state at fixed temperatures, then Tₑ and Tᵢ evolving, against the 0D circuit
#     (L + L_kin) İ = V − R(Tₑ) I,   R = 2π/(σ(Tₑ) Σ dA/R) over the column's cells,
#     (3/2) n e Ṫₑ = I² R/Vol − (3/2) n e a (Tₑ − Tᵢ),   (3/2) n e Ṫᵢ = (3/2) n e a (Tₑ − Tᵢ),
# L = μ₀R₀(ln(8R₀/a) − 2 + lᵢ/2), lᵢ = 1/2, L_kin = mₑ 2πR₀/(n e² πa²), σ = n e²/(mₑ ξ ν_ei), a = 2mₑmᵢ/(mₑ + mᵢ)² ν_ei
# (internal/docs/src/reference/inductive-current-rise.md §4.3, coulomb-collisions.md). As the plasma heats, R falls as
# Tₑ^{−3/2} and the current turns toward the inductive limit. In a torus J ∝ E ∝ 1/R heats the inner side more, so Tₑ
# spreads across the column and the current gathers there: the window stays at 3 ms. The first test runs without Tₑ
# conduction, the second with it, over 9 ms, where its operator once cooled the minimum at the axis of the column's own
# surfaces (issues/anisotropic-conduction-cools-a-minimum.md).

@testsnippet OhmicFilamentTheory begin
    "The e–i Coulomb logarithm as the reference states it: NRL p.34(b), Z = 1, cgs inside, floored at 1."
    function lnΛ_ei(n, Te, Ti, mi, me, mp)
        ncgs = n * 1.0e-6
        l = Te < Ti * me / mi ? 16.0 - log(sqrt(ncgs) * Ti^(-1.5) * (mi / mp)) :
            Te > 10.0 ? 24.0 - log(sqrt(ncgs) / Te) : 23.0 - log(sqrt(ncgs) * Te^(-1.5))
        return max(l, 1.0)
    end

    """
    The uniform column's circuit: its steady current `I_sat(Te, Ti)` under the loop voltage `LV`, and `evolve(I0, Te0,
    Ti0)`, a function of t returning (I, Tₑ, Tᵢ) by RK4. `R_cells` and `dA` are the column's cells; constants are `c`'s.
    """
    function filament_circuit(c, mi; n, LV, R_cells, dA, R0, a)
        me, ee, mp, eps0, μ0 = c.me, c.ee, c.mp, c.eps0, c.μ0
        kν = sqrt(2) * ee^4 / (12 * π^1.5 * eps0^2 * sqrt(me))
        ξ = 0.510469472194728
        ν_ei(Te, Ti) = kν * n * lnΛ_ei(n, Te, Ti, mi, me, mp) * (ee * Te)^(-1.5)
        σ(Te, Ti) = n * ee^2 / (me * ξ * ν_ei(Te, Ti))
        geometric = sum(dA ./ R_cells)
        R_p(Te, Ti) = 2π / (σ(Te, Ti) * geometric)
        Vol = sum(2π .* R_cells .* dA)
        L = μ0 * R0 * (log(8 * R0 / a) - 2 + 0.25)
        L_kin = me / (n * ee^2) * (2π * R0 / (π * a^2))
        function rhs(s)
            I, Te, Ti = s
            rate = 2 * me * mi / (me + mi)^2 * ν_ei(Te, Ti)
            heat = I^2 * R_p(Te, Ti) / Vol / (n * ee)
            return ((LV - R_p(Te, Ti) * I) / (L + L_kin), heat / 1.5 - rate * (Te - Ti), rate * (Te - Ti))
        end
        function evolve(I0, Te0, Ti0)
            s, t = (Float64(I0), Float64(Te0), Float64(Ti0)), 0.0
            return function (t_to)
                while t < t_to - 1.0e-12
                    h = min(1.0e-6, t_to - t)
                    k1 = rhs(s)
                    k2 = rhs(s .+ (h / 2) .* k1)
                    k3 = rhs(s .+ (h / 2) .* k2)
                    k4 = rhs(s .+ h .* k3)
                    s = s .+ (h / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
                    t += h
                end
                return s
            end
        end
        return (; I_sat = (Te, Ti) -> LV / R_p(Te, Ti), evolve)
    end
end

@testitem "Ohmic heating of an inductive filament: the steady current, then Tₑ, Tᵢ and the current rising as the circuit and energy balance say" tags = [:regression] setup = [RegressionCommon, OhmicFilamentTheory] begin
    using RAPID2D.Statistics
    R0, a, n, T0 = 1.5, 0.3, 1.0e18, 10.0
    t_prep, t_heat, dt = 40.0e-3, 3.0e-3, 25.0e-6
    config = regression_config(;
        manual = ManualSetup{Float64}(BR = 0.0, BZ = 0.0, Eϕ = 0.3), NR = 30, NZ = 50, R0B0 = 3.0,
        prefilled_gas_pressure = 0.0, dt, t_end_s = t_prep, snap0D_Δt_s = 1.0e-4, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    RP.flags = SimulationFlags{Float64}(;
        Ampere = true, Ampere_Itor_threshold = 0.0, E_para_induction = true, ud_evolve = true, Implicit = true,
        Coulomb_Collision = true, Spitzer_Resistivity = true, Atomic_Collision = false, src = false,
        convec = false, diffu = false, Gas_evolve = false, update_ni_independently = false,
        Te_evolve = false, Ti_evolve = false,                 # the first phase: the steady current at fixed temperatures
        Include_ud_convec_term = false, Include_ud_diffu_term = false, Include_ud_pressure_term = false,
        Include_Te_convec_term = false, Include_Te_diffu_term = false, Include_heat_flux_term = false,
        charge_sheet_field = false, E_para_cancellation = false, mean_ExB = false, turb_ExB_mixing = false,
        Global_JxB_Force = false, FLF_nstep = 100, vacuum_relaxation = (state = false, factor = 1.0e2),
    )
    initialize!(RP)
    G, pla = RP.G, RP.plasma
    column = tophat_blob(G; cenR = R0, radius = a, n0 = n)
    pla.ne .= column
    pla.ni .= column
    fill!(pla.Te_eV, T0)
    fill!(pla.Ti_eV, T0)
    fill!(pla.ue_para, 0.0)
    fill!(pla.ui_para, 0.0)
    col = findall(>(0), vec(column))
    theory = filament_circuit(
        RP.config.constants, RAPID2D.bulk_ion_mass(RP); n, LV = mean(vec(RP.fields.LV_presc)[col]),
        R_cells = vec(G.R2D)[col], dA = G.dR * G.dZ, R0, a,
    )

    run_simulation!(RP)
    I_tor = RP.diagnostics.snaps0D.I_tor
    I0 = I_tor[end]
    @test abs(I0 - I_tor[end - 10]) / I0 < 1.0e-3                    # steady over the last ms
    @test I0 ≈ theory.I_sat(T0, T0) rtol = 5.0e-3                    # the uniform column's Ohmic current

    # the second phase: both temperatures free, from that current
    RP.flags.Te_evolve = true
    RP.flags.Ti_evolve = true
    RP.t_end_s = t_prep + t_heat
    ode = theory.evolve(I0, T0, T0)
    w = vec(G.inVol2D)[col] .* n
    errs = (I = Float64[], Te = Float64[], Ti = Float64[])
    last = Ref((0.0, 0.0, 0.0))
    run_simulation!(
        RP; callback_after_step = function (rp)
            rp.step % 4 == 0 || return nothing
            p = rp.plasma
            I, Te, Ti = ode(rp.time_s - t_prep)
            Te_c = sum(w .* vec(p.Te_eV)[col]) / sum(w)
            Ti_c = sum(w .* vec(p.Ti_eV)[col]) / sum(w)
            push!(errs.I, abs(rp.diagnostics.snaps0D.I_tor[end] - I) / I)
            push!(errs.Te, abs(Te_c - Te) / Te)
            push!(errs.Ti, abs(Ti_c - Ti) / Ti)
            last[] = (rp.diagnostics.snaps0D.I_tor[end], Te_c, Ti_c)
            return nothing
        end,
    )
    @test length(errs.I) >= 25
    I_end, Te_end, Ti_end = last[]
    @test Te_end > 2.5T0 && Ti_end > 1.3T0 && I_end > 1.25I0     # the regime: Tₑ several times, the current well up
    @test maximum(errs.I) < 5.0e-2                                  # 3.4 % at 3 ms: the current gathers where the edge runs hot
    @test maximum(errs.Te) < 1.0e-2                                 # 0.4 %
    @test maximum(errs.Ti) < 1.0e-2                                 # 0.4 %
end

@testitem "Ohmic heating of an inductive filament with the electrons' conduction on: the axis's Tₑ minimum is never deepened, the column's energy as the circuit says" tags = [:regression] setup = [RegressionCommon, OhmicFilamentTheory] begin
    # the current's own poloidal field closes the column's surfaces about an axis; the electrons' parallel
    # conduction, projected by it, must not cool a Tₑ minimum there (PLAN_monotone-conduction.md §7). The
    # density does not diffuse, so Tₑ conducts and rides no drift: the column's energy is the circuit's.
    using RAPID2D.Statistics
    R0, a, n, T0 = 1.5, 0.3, 1.0e18, 10.0
    t_prep, t_heat, dt = 40.0e-3, 9.0e-3, 25.0e-6
    config = regression_config(;
        manual = ManualSetup{Float64}(BR = 0.0, BZ = 0.0, Eϕ = 0.3), NR = 30, NZ = 50, R0B0 = 3.0,
        prefilled_gas_pressure = 0.0, dt, t_end_s = t_prep, snap0D_Δt_s = 1.0e-4, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    RP.flags = SimulationFlags{Float64}(;
        Ampere = true, Ampere_Itor_threshold = 0.0, E_para_induction = true, ud_evolve = true, Implicit = true,
        Coulomb_Collision = true, Spitzer_Resistivity = true, Atomic_Collision = false, src = false,
        convec = false, diffu = false, Gas_evolve = false, update_ni_independently = false,
        Te_evolve = false, Ti_evolve = false,
        Include_ud_convec_term = false, Include_ud_diffu_term = false, Include_ud_pressure_term = false,
        Include_Te_convec_term = false, Include_Te_diffu_term = true, Include_heat_flux_term = false,
        charge_sheet_field = false, E_para_cancellation = false, mean_ExB = false, turb_ExB_mixing = false,
        Global_JxB_Force = false, FLF_nstep = 100, vacuum_relaxation = (state = false, factor = 1.0e2),
    )
    initialize!(RP)
    G, pla = RP.G, RP.plasma
    column = tophat_blob(G; cenR = R0, radius = a, n0 = n)
    pla.ne .= column
    pla.ni .= column
    fill!(pla.Te_eV, T0)
    fill!(pla.Ti_eV, T0)
    fill!(pla.ue_para, 0.0)
    fill!(pla.ui_para, 0.0)
    col = findall(>(0), vec(column))
    theory = filament_circuit(
        RP.config.constants, RAPID2D.bulk_ion_mass(RP); n, LV = mean(vec(RP.fields.LV_presc)[col]),
        R_cells = vec(G.R2D)[col], dA = G.dR * G.dZ, R0, a,
    )
    run_simulation!(RP)
    I0 = RP.diagnostics.snaps0D.I_tor[end]

    RP.flags.Te_evolve = true
    RP.flags.Ti_evolve = true
    RP.t_end_s = t_prep + t_heat
    ode = theory.evolve(I0, T0, T0)
    w = vec(G.inVol2D)[col] .* n
    samples = (t = Float64[], Te_min = Float64[], I = Float64[], Te = Float64[])
    run_simulation!(
        RP; callback_after_step = function (rp)
            rp.step % 4 == 0 || return nothing
            p = rp.plasma
            I, Te, _ = ode(rp.time_s - t_prep)
            push!(samples.t, rp.time_s - t_prep)
            push!(samples.Te_min, minimum(vec(p.Te_eV)[col]))
            push!(samples.I, abs(rp.diagnostics.snaps0D.I_tor[end] - I) / I)
            push!(samples.Te, abs(sum(w .* vec(p.Te_eV)[col]) / sum(w) - Te) / Te)
            return nothing
        end,
    )
    @test length(samples.t) >= 80
    @test samples.Te_min[end] > 4 * T0                                   # the coldest node heats with the rest
    @test all(diff(samples.Te_min) .>= -1.0e-6)                          # and never cools
    early = samples.t .<= 3.0e-3
    @test maximum(samples.Te[early]) < 1.0e-2                            # 0.7 % at 3 ms
    @test maximum(samples.Te) < 6.0e-2                                   # 4.6 % at 9 ms: the column is not uniform
    @test maximum(samples.I) < 6.0e-2                                    # 4.9 % at 9 ms
end
