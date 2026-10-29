# Ohmic heating where only Coulomb collisions remain: a uniform plasma under a fixed parallel field, no induction,
# against an independent 0D ODE (case 2a of internal/docs/src/notes/plans/PLAN_ohmic-heating-tests.md).
#
# Relative drift d = u_e − u_i, the ions' recoil included (both start at rest, total momentum zero):
#     ḋ = −(eE/mₑ + ν_eff d)(1 + mₑ/mᵢ)
#     (3/2) Ṫₑ = mₑ ν_eff d²/e − (3/2) a (Tₑ − Tᵢ)
#     (3/2) Ṫᵢ = (3/2) a (nₑ/nᵢ)(Tₑ − Tᵢ)
# ν_eff = ξ ν_ei, ξ = 0.5105 at Z = 1, ν_ei = √2 e⁴ n lnΛ / (12 π^{3/2} ε₀² √mₑ (eTₑ)^{3/2}), lnΛ NRL p.34(b),
# a = 2mₑmᵢ/(mₑ + mᵢ)² ν_ei (internal/docs/src/reference/coulomb-collisions.md). The quasi-steady drift heats by
# σE² with σ ∝ Tₑ^{3/2}/lnΛ: without exchange and induction a fixed field runs away in finite time.

@testsnippet OhmicHeatingCase begin
    using RAPID2D

    """
    A uniform plasma in a box under the uniform parallel field `E0`: a pure toroidal field (bϕ = 1), only Coulomb
    collisions, nothing transported, no gas, no induction, the vacuum relaxation off.
    """
    function ohmic_box(; n, T0, E0, dt, t_end, Te_evolve = true, Ti_evolve = true)
        config = SimulationConfig{Float64}(;
            device_Name = "manual", manual = ManualSetup{Float64}(BR = 0.0, BZ = 0.0, Eϕ = E0),
            NR = 9, NZ = 9, R_min = 1.0, R_max = 2.0, Z_min = -0.5, Z_max = 0.5,
            wall_R = [1.15, 1.85, 1.85, 1.15], wall_Z = [-0.35, -0.35, 0.35, 0.35],
            prefilled_gas_pressure = 0.0, R0B0 = 1.0, dt = dt, t_end_s = t_end,
            snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0, Output_path = mktempdir(),
        )
        RP = RAPID{Float64}(config)
        RP.flags = SimulationFlags{Float64}(;
            Implicit = true, diffu = false, convec = false, src = false,
            Atomic_Collision = false, Coulomb_Collision = true, Spitzer_Resistivity = true,
            mean_ExB = false, turb_ExB_mixing = false, charge_sheet_field = false, E_para_cancellation = false,
            E_para_induction = false, Ampere = false, Te_evolve, Ti_evolve, ud_evolve = true, Gas_evolve = false,
            update_ni_independently = true, secondary_electron = false, negative_n_correction = false,
            Include_ud_pressure_term = false, Include_ud_convec_term = false, Include_ud_diffu_term = false,
            Include_Te_convec_term = false, Include_Te_diffu_term = false, Include_heat_flux_term = false,
            vacuum_relaxation = (state = false, factor = 1.0e2),
        )
        initialize!(RP)
        G, pla, F = RP.G, RP.plasma, RP.fields
        inw = G.nodes.in_wall_nids
        F.LV_presc .= 2π .* G.R2D .* E0                 # a uniform E: the manual loop voltage would give E ∝ 1/R
        RAPID2D.update_prescribed_fields!(RP)
        RAPID2D.combine_field_sources!(RP)
        pla.ne .= 0.0
        vec(pla.ne)[inw] .= n
        pla.ni .= pla.ne
        pla.Te_eV .= T0
        pla.Ti_eV .= T0
        pla.ue_para .= 0.0
        pla.ui_para .= 0.0
        update_transport_quantities!(RP)
        return RP
    end

    "The e–i Coulomb logarithm as the reference states it: NRL p.34(b), Z = 1, cgs inside, floored at 1."
    function lnΛ_ei(n, Te, Ti, mi, me, mp)
        ncgs = n * 1.0e-6
        l = Te < Ti * me / mi ? 16.0 - log(sqrt(ncgs) * Ti^(-1.5) * (mi / mp)) :
            Te > 10.0 ? 24.0 - log(sqrt(ncgs) / Te) : 23.0 - log(sqrt(ncgs) * Te^(-1.5))
        return max(l, 1.0)
    end

    """
    The 0D reference for `RP`'s plasma under the field `E`: a function `(t) -> (d, Te, Ti)`, integrated by RK4 from
    rest at `T0` in steps far below 1/ν. Constants are `RP`'s; the formulas are the reference page's.
    """
    function coulomb_ode(RP, n, E, T0; Te_evolve = true, Ti_evolve = true)
        c = RP.config.constants
        me, ee, mp, eps0 = c.me, c.ee, c.mp, c.eps0
        mi = RAPID2D.bulk_ion_mass(RP)
        ξ = 0.510469472194728
        kν = sqrt(2) * ee^4 / (12 * π^1.5 * eps0^2 * sqrt(me))
        recoil = 1 + me / mi
        function rhs(s)
            d, Te, Ti = s
            ν = kν * n * lnΛ_ei(n, Te, Ti, mi, me, mp) * (ee * Te)^(-1.5)
            a = 2 * me * mi / (me + mi)^2 * ν
            dTe = Te_evolve ? (me * ξ * ν * d^2 / ee) / 1.5 - a * (Te - Ti) : 0.0
            dTi = Ti_evolve ? a * (Te - Ti) : 0.0
            return (-(ee * E / me + ξ * ν * d) * recoil, dTe, dTi)
        end
        return function (t_end)
            s, t = (0.0, Float64(T0), Float64(T0)), 0.0
            while t < t_end
                ν = kν * n * lnΛ_ei(n, s[2], s[3], mi, me, mp) * (ee * s[2])^(-1.5)
                h = min(0.02 / (ξ * ν), t_end - t, 1.0e-8)
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

    "The first time the reference's Tₑ reaches `T_stop`, on a grid of `step`, by bisection."
    function time_to_reach(ode, T_stop; t_hi, step = 4.0e-6)
        lo, hi = 0.0, t_hi
        for _ in 1:40
            mid = (lo + hi) / 2
            ode(mid)[2] < T_stop ? (lo = mid) : (hi = mid)
        end
        return step * floor(lo / step)
    end

    "Run `RP` to its end; the drift and the temperatures at one in-wall node, and their spread over the in-wall nodes."
    function run_box!(RP)
        run_simulation!(RP)
        pla, inw = RP.plasma, RP.G.nodes.in_wall_nids
        k = inw[cld(length(inw), 2)]
        spread(f) = maximum(abs, vec(f)[inw] .- f[k])
        return (
            d = pla.ue_para[k] - pla.ui_para[k], Te = pla.Te_eV[k], Ti = pla.Ti_eV[k],
            spread = max(spread(pla.Te_eV), spread(pla.ue_para) / max(abs(pla.ue_para[k]), 1.0)), t = RP.time_s,
        )
    end
end

@testitem "a uniform plasma under a fixed parallel field heats as the Coulomb-only ODE says, to first order in Δt" setup = [OhmicHeatingCase] begin
    # Ohmic heating against the exchange: dominant at n = 1e17, E = 0.1 V/m; the ions follow at n = 1e19, E = 1 V/m.
    # The window ends below Tₑ = 9 eV, inside one branch of the logarithm, so the error is the scheme's alone.
    for (n, E0) in ((1.0e17, 0.1), (1.0e19, 1.0))
        T0 = 4.0
        probe = ohmic_box(; n, T0, E0, dt = 1.0e-6, t_end = 1.0e-6)
        inw = probe.G.nodes.in_wall_nids
        @test all(≈(E0; rtol = 1.0e-12), probe.fields.E_para_tot[inw])
        ode = coulomb_ode(probe, n, E0, T0)
        t_end = time_to_reach(ode, 8.5; t_hi = 5.0e-3)
        @test t_end > 50.0e-6
        errs = Float64[]
        for dt in (1.0e-6, 0.5e-6, 0.25e-6)
            RP = ohmic_box(; n, T0, E0, dt, t_end)
            r = run_box!(RP)
            @test r.t ≈ t_end rtol = 1.0e-9
            @test r.spread <= 1.0e-9 * r.Te                 # every node the same plasma
            @test all(≈(n; rtol = 1.0e-12), RP.plasma.ne[inw])   # nothing moved or born
            d, Te, Ti = ode(r.t)
            @test Te > 2T0 && 3.0e-3 < abs(d) / sqrt(2 * RP.config.constants.ee * Te / RP.config.constants.me) < 0.3
            push!(errs, max(abs(r.Te - Te) / Te, abs(r.Ti - Ti) / Ti, abs(r.d - d) / abs(d)))
        end
        @test errs[end] < 5.0e-3
        @test 1.6 < errs[1] / errs[2] < 2.4 && 1.6 < errs[2] / errs[3] < 2.4   # first order
    end
end

@testitem "the step's energy: the thermal and kinetic gain is the field's trapezoid work less half the ions' kick; momentum is kept" setup = [OhmicHeatingCase] begin
    # Per step and volume, with E constant and nothing floored: ΔU = J̄ E Δt − ½ I_ei Δuᵢ, U = (3/2) e (nₑTₑ + nᵢTᵢ) +
    # ½ mₑ nₑ u_e² + ½ ρ uᵢ², J̄ = e n (ūᵢ − ū_e). The e–i exchange cancels inside U; the field's impulses on the two
    # species cancel, so the total momentum stays zero.
    n, E0 = 1.0e19, 1.0
    RP = ohmic_box(; n, T0 = 4.0, E0, dt = 1.0e-6, t_end = 40.0e-6)
    pla, k = RP.plasma, RP.G.nodes.in_wall_nids[1]
    me, ee = RP.config.constants.me, RP.config.constants.ee
    U(p) = 1.5 * ee * (p.ne[k] * p.Te_eV[k] + p.ni_tot[k] * p.Ti_eV[k]) + 0.5 * me * p.ne[k] * p.ue_para[k]^2 + 0.5 * p.mni_tot[k] * p.ui_para[k]^2
    before = Ref{Any}(nothing)
    gaps, works, momenta = Float64[], Float64[], Float64[]
    run_simulation!(
        RP;
        callback_before_step = rp -> (p = rp.plasma; before[] = (U = U(p), ue = p.ue_para[k], ui = p.ui_para[k]); nothing),
        callback_after_step = function (rp)
            p, b = rp.plasma, before[]
            ΔU = U(p) - b.U
            W = ee * n * (0.5 * (b.ui + p.ui_para[k]) - 0.5 * (b.ue + p.ue_para[k])) * E0 * rp.dt
            push!(gaps, (ΔU - (W - 0.5 * p.I_ei[k] * (p.ui_para[k] - b.ui))) / ΔU)
            push!(works, W / ΔU)
            push!(momenta, (me * p.ne[k] * p.ue_para[k] + p.mni_tot[k] * p.ui_para[k]) / (me * p.ne[k] * abs(p.ue_para[k])))
            return nothing
        end,
    )
    @test length(gaps) == 40
    @test maximum(abs, gaps) < 1.0e-9
    @test maximum(abs, momenta) < 1.0e-9
    @test all(w -> 0.99 < w < 1.01, works[10:end])   # the ions' kick is order mₑ/mᵢ of the work
end

@testitem "fixed temperatures, no induction: the drift saturates as d_sat (1 − e^{−ν_eff (1 + mₑ/mᵢ) t}), to first order in Δt" setup = [OhmicHeatingCase] begin
    n, T0, E0 = 1.0e17, 5.0, 0.1
    t_end = 20.0e-6                                     # about three relaxation times
    errs = Float64[]
    for dt in (1.0e-6, 0.5e-6, 0.25e-6)
        RP = ohmic_box(; n, T0, E0, dt, t_end, Te_evolve = false, Ti_evolve = false)
        r = run_box!(RP)
        ode = coulomb_ode(RP, n, E0, T0; Te_evolve = false, Ti_evolve = false)
        d = ode(r.t)[1]
        @test r.Te == T0 && r.Ti == T0
        push!(errs, abs(r.d - d) / abs(d))
    end
    @test errs[end] < 5.0e-3
    @test 1.6 < errs[1] / errs[2] < 2.4 && 1.6 < errs[2] / errs[3] < 2.4
end

@testitem "a longer trajectory across the logarithm's 10 eV branch: the error stays small and falls with Δt" setup = [OhmicHeatingCase] begin
    # 5 eV to about 25 eV in 1 ms (n = 1e17, E = 0.1 V/m); the logarithm drops by 0.15 at 10 eV
    n, T0, E0, t_end = 1.0e17, 5.0, 0.1, 1.0e-3
    errs = Float64[]
    for dt in (1.0e-6, 0.5e-6)
        RP = ohmic_box(; n, T0, E0, dt, t_end)
        r = run_box!(RP)
        d, Te, Ti = coulomb_ode(RP, n, E0, T0)(r.t)
        @test Te > 20.0
        push!(errs, max(abs(r.Te - Te) / Te, abs(r.Ti - Ti) / Ti, abs(r.d - d) / abs(d)))
    end
    @test errs[1] < 2.0e-2
    @test errs[2] < 0.7 * errs[1]
end
