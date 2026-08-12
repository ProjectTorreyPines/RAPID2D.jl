@testsnippet ExpRBTeFixtures begin
    using RAPID2D: Electron_RRCs, ExpRB, ForwardEuler, update_RRCs!, update_Te!,
        update_electron_heating_powers!, exprb_bern
    using RAPID2D: h5open        # HDF5 is RAPID2D's dependency, not the test env's

    # A 0D-like configuration: no transport in the energy equation, so A_LHS is
    # the diagonal alone and the scheme under test is the only thing acting. The
    # drift is frozen (`ud_evolve = false`) so that with constant rate
    # coefficients the energy equation is exactly linear in Tₑ — which is what
    # turns "is it accurate" into "is it the closed-form answer".
    # The drift has to be fast enough that frictional heating beats the constant
    # inelastic losses. With K held constant on every surface the excitation and
    # ionization sinks never switch off with falling Ē, so a slow drift has no
    # positive equilibrium at all — Te_sat comes out negative and the run walks
    # off the table. Not a physical statement, just what a constant-rate table
    # implies; the real table's thresholds are what normally prevent it.
    function te_RAPID(;
            Te_eV, u_para = -5.0e6, E_para = -50.0, pressure = 5.0e-3,
            implicit = true
        )
        config = SimulationConfig{Float64}(
            NR = 8, NZ = 8, R_min = 0.8, R_max = 2.2, Z_min = -1.2, Z_max = 1.2,
            dt = 1.0e-8, t_end_s = 1.0e-6, R0B0 = 1.0,
            prefilled_gas_pressure = pressure,
            snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
            wall_R = [1.0, 2.0, 2.0, 1.0], wall_Z = [-1.0, -1.0, 1.0, 1.0],
        )
        config.Output_path = mktempdir(; cleanup = false)
        config.min_Te = 1.0e-8          # the clamps must never be what saves a run
        config.max_Te = 1.0e8
        RP = RAPID{Float64}(config)
        RP.flags.Atomic_Collision = true
        RP.flags.src = true
        RP.flags.Coulomb_Collision = false
        RP.flags.Te_evolve = true
        RP.flags.ud_evolve = false
        RP.flags.Implicit = implicit
        RP.flags.Include_Te_diffu_term = false
        RP.flags.Include_Te_convec_term = false
        RP.flags.Include_heat_flux_term = false
        initialize!(RP)
        RP.config.min_Te = 1.0e-8
        RP.config.max_Te = 1.0e8
        RP.plasma.Te_eV .= Te_eV
        RP.plasma.ne .= 1.0e16
        RP.plasma.ni .= 1.0e16
        RP.plasma.ue_para .= u_para
        # P_drag reads ueR/ueϕ/ueZ, not ue_para. Sync or all heating vanishes.
        RP.plasma.ueR .= u_para .* RP.fields.bR
        RP.plasma.ueϕ .= u_para .* RP.fields.bϕ
        RP.plasma.ueZ .= u_para .* RP.fields.bZ
        RP.fields.E_para_tot .= E_para
        # `ePowers.drag` charges its Coulomb half unconditionally — it sits inside
        # `Atomic_Collision`, not inside `Coulomb_Collision` — so initialization's
        # leftover sptz_fac·ν_ei keeps contributing even with Coulomb off. It is
        # 2.6e-6 of the drag here and irrelevant physically, but it is a term in Tₑ
        # that the closed form below does not model, and this test's whole premise
        # is that the problem IS the closed form. Zero it explicitly.
        RP.plasma.sptz_fac .= 0.0
        RP.plasma.ν_ei .= 0.0
        return RP
    end

    # Every RATE surface (K_iz, K_diss_iz, K_mom, ...) a constant ⟹ the ν's built
    # from them are Tₑ-independent, so P_drag, P_dilution, P_iz and P_diss_iz stay
    # exactly linear (or constant) in Tₑ with no help needed.
    #
    # L_ela/L_exc are the one place a FLAT surface would NOT do that anymore.
    # Since Task B3, P_ela and P_exc carry a cold-target factor
    # `1 − 1.5·T_gas/Ē` applied to `Kerg_ela(Ē)`/`Kerg_exc(Ē)`, so a flat Kerg
    # would leave P_ela/P_exc only WEAKLY Tₑ-dependent (through that tiny
    # correction alone) and, worse, genuinely NONLINEAR (Ē sits in a
    # denominator). A table that is instead EXACTLY `b·Ē` cancels the
    # denominator exactly:
    #   n_gas·b·Ē·(1 − 1.5T_gas/Ē) = n_gas·b·(Ē − 1.5T_gas)
    #                              = n_gas·b·1.5·(Tₑ − T_gas) + n_gas·b·½mₑu∥²/e,
    # which — because Ē = 1.5Tₑ + ½mₑu∥²/e is itself linear in Tₑ — is EXACTLY
    # linear, not the "Ē_ela ≃ Ē" approximation update_electron_heating_powers!
    # documents for the real table. `b_erg` is shared by L_ela and L_exc: it
    # only needs to make the two energy sinks dominate P_drag by enough that a
    # positive Tₑ_sat exists (drag ∝ u∥², and so does the cold-target's ½mₑu∥²/e
    # floor loss, so the ratio between them does not depend on which u∥ is
    # picked, only on b_erg vs K). L_diss_exc stays FLAT — production applies no
    # cold-target factor to it — but at the SAME small scale as b_erg, not at
    # the rate-coefficient scale K, or its raw constant loss alone would swamp
    # everything else the way L_ela/L_exc used to before this fix.
    function with_constant_surfaces!(RP; K, b_erg)
        EoverP = collect(range(1.0, 1000.0, 8))
        Erg_eV = collect(10 .^ range(-3, 3, 16))
        flat = fill(K, length(EoverP), length(Erg_eV))
        flat_small = fill(b_erg, length(EoverP), length(Erg_eV))
        prop_to_Ē = [b_erg * E for _ in EoverP, E in Erg_eV]
        path = joinpath(mktempdir(; cleanup = false), "eRRCs_EoverP_Erg.h5")
        h5open(path, "w") do fid
            fid["EoverP"] = EoverP
            fid["Erg_eV"] = Erg_eV
            # Electron_RRCs now reads the full 2026-08 group ledger (Task B1); every
            # (E/p, Ē) surface it reads must exist in the file, even the ones this
            # test does not exercise, or the constructor throws on a missing dataset.
            for name in (
                    "K_iz", "K_diss_iz", "K_exc", "K_diss_exc",
                    "K_mom", "K_mom_by_ela", "K_mom_by_exc", "K_mom_by_diss_exc",
                    "K_mom_by_iz", "K_mom_by_diss_iz",
                    "L_tot",
                )
                fid[name] = copy(flat)
            end
            fid["L_ela"] = copy(prop_to_Ē)
            fid["L_exc"] = copy(prop_to_Ē)
            fid["L_diss_exc"] = copy(flat_small)
        end
        RP.eRRCs = Electron_RRCs(
            path, joinpath(dirname(dirname(pathof(RAPID2D))), "RRC_data", "eRRCs_T_ud.h5")
        )
        return RP
    end

    # The closed form of that linear problem. P(Tₑ) is engineered to be EXACTLY
    # affine by `with_constant_surfaces!` above, so two probe evaluations
    # through the PRODUCTION power path pin A and 𝔅 exactly — no term-by-term
    # re-derivation to drift out of sync with update_electron_heating_powers!
    # itself (which is what happened here across the Task B1-B3 ledger
    # migration: this test's old hand-derived 𝔅 quietly stopped matching what
    # the code computes, and nothing caught it until Task B4's Jacobian fix
    # made the mismatch visible).
    #
    # Returned PER NODE. n_H2_gas varies across the grid, so ν and therefore λ do
    # too — every node relaxes on its own clock. Te_sat does not vary (A and 𝔅 are
    # both ∝ ν, so the ratio is ν-free), but a grid average would still smear the
    # residual transient across a spread of λ and lose the exactness this test is
    # for. Compare node by node.
    function linear_te_problem(RP)
        pla = RP.plasma
        ee = RP.config.constants.ee
        Te_ref = copy(pla.Te_eV)

        pla.Te_eV .= Te_ref .+ 1.0
        update_RRCs!(RP); update_electron_heating_powers!(RP)
        P_hi = copy(pla.ePowers.tot)

        pla.Te_eV .= Te_ref .- 1.0
        update_RRCs!(RP); update_electron_heating_powers!(RP)
        P_lo = copy(pla.ePowers.tot)

        pla.Te_eV .= Te_ref                      # restore state AND powers/rates
        update_RRCs!(RP); update_electron_heating_powers!(RP)

        𝔅 = @. (P_lo - P_hi) / 2.0                # P(Tₑ) = A − 𝔅·Tₑ, slope = −𝔅
        A = @. (P_hi + P_lo) / 2.0 + 𝔅 * Te_ref
        return (Te_sat = A ./ 𝔅, eig = @.(-(2 / 3) * 𝔅 / ee))
    end

    # The exact trajectory of dTe/dt = λ(Te − Te_sat), node by node.
    exact_Te(p, Te₀, t) = @. p.Te_sat + (Te₀ - p.Te_sat) * exp(p.eig * t)

    # `refresh_rates`: production calls `update_RRCs!` once per OUTER time step,
    # immediately before the physics that consumes it (see
    # `internal/docs/src/notes/design/rrc-single-evaluation-point.md`), so a
    # multi-step march that actually represents that sequence must do the same.
    # Defaulted off to leave the OTHER testitems in this file — which march the
    # REAL production table, not a synthetic one, and were passing before this
    # fix — bit-for-bit unchanged; only the linear-problem test below opts in.
    # Whether it matters depends on the table: under the OLD Kerg_ela-free
    # model, and under any FLAT synthetic surface, ν is Tₑ-independent so
    # leaving it stale is harmless; under `with_constant_surfaces!`'s L_ela/L_exc
    # (linear in Ē, hence in Tₑ) it is not — P_en_ela staying pinned at
    # Ē(Te-at-last-refresh) while only the cold-target factor tracks the LIVE Tₑ
    # is exactly what breaks the exact-linearity this file is built on.
    function march!(RP, dt, nsteps; refresh_rates = false)
        for _ in 1:nsteps
            RP.dt = dt
            refresh_rates && update_RRCs!(RP)
            update_Te!(RP)
        end
        return RP
    end
end

@testitem "ExpRB Tₑ: exact on the linear problem, at every Δt, both directions" setup = [ExpRBTeFixtures] begin
    using RAPID2D: ExpRB, FullLinearResponse, update_RRCs!

    # With constant rate coefficients and a frozen drift the energy equation is
    # dTe/dt = λ(Te − Te_sat) with λ constant, and ExpRB is EXACT for that at any
    # step. So this is not an accuracy test with a tolerance chosen to pass — it
    # is an identity, and 1e-12 is the room left for round-off in assembling it.
    #
    # Both directions, because the approach to the fixed point is not symmetric:
    # cooling starts with a large |P| and heating with a small one, so they weight
    # the transient differently even though they share a λ.
    #
    # Note what this does NOT cover: both directions have λ < 0. Measured on the
    # real table too (see exprb_te_ladder_test.jl), heating toward the fixed point
    # is a relaxation from below, not a runaway, so B's growth branch stays
    # unexercised by physics here. One formula spans both signs with no branch to
    # take, and the kernel tests cover z > 0 directly — but that is the argument,
    # not a measurement.
    #
    # `exprb_eigenvalue = FullLinearResponse`, not the default `PartialLinearResponse`.
    # Since Task B3 moved most of P_ela's and P_exc's Tₑ dependence INSIDE
    # Kerg_ela(Ē)/Kerg_exc(Ē), `PartialLinearResponse` discards exactly the part of
    # ∂P/∂Tₑ that `with_constant_surfaces!` engineers to dominate here (see its
    # comment) — an intentional, documented omission (physics.jl's
    # `_eig_Te_from_known_rates!`), not a bug, but it means the fitted z no longer
    # matches this problem's TRUE 𝔅 under that depth, and exactness is specifically
    # what `FullLinearResponse` exists to recover.
    K, b_erg = 2.0e-15, 5.0e-35
    probe = with_constant_surfaces!(te_RAPID(; Te_eV = 5.0); K, b_erg)
    probe.flags.scheme.atomic = ExpRB
    probe.flags.exprb_eigenvalue = FullLinearResponse
    update_RRCs!(probe)
    p = linear_te_problem(probe)
    inw = probe.G.nodes.in_wall_nids
    @test all(<(0), p.eig[inw])                          # a relaxation, in both cases
    τ = 1 / minimum(abs, p.eig[inw])                     # the SLOWEST node sets the run
    Te_sat = p.Te_sat[first(inw)]

    # Both solve paths. `Implicit = false` is not a place where the fit stops
    # applying, only one where there is no matrix to put B on the diagonal of —
    # there it divides the increment instead, and the two must agree because the
    # closed form does not know which one ran.
    for (label, Te₀_factor) in (("cool-down", 3.0), ("heat-up", 0.05)),
            implicit in (true, false)
        Te₀ = Te₀_factor * Te_sat
        t_end = 12τ

        for nsteps in (2048, 64, 4, 1)                 # down to ONE step for the whole run
            dt = t_end / nsteps
            RP = with_constant_surfaces!(
                te_RAPID(; Te_eV = Te₀, implicit = implicit); K, b_erg
            )
            RP.flags.scheme.atomic = ExpRB
            RP.flags.exprb_eigenvalue = FullLinearResponse
            update_RRCs!(RP)
            march!(RP, dt, nsteps; refresh_rates = true)

            exact = exact_Te(linear_te_problem(RP), Te₀, t_end)
            @test RP.plasma.Te_eV[inw] ≈ exact[inw] rtol = 1.0e-12

            # Never rescued by a clamp, and never past the fixed point.
            @test all(RP.plasma.Te_eV[inw] .> RP.config.min_Te)
            @test all(RP.plasma.Te_eV[inw] .< RP.config.max_Te)
            if Te₀ > Te_sat
                @test all(RP.plasma.Te_eV[inw] .>= Te_sat)     # cooled, never undershot
            else
                @test all(RP.plasma.Te_eV[inw] .<= Te_sat)     # heated, never overshot
            end

            # The gate is not vacuous: a coefficient that happened to sit near 1
            # would pass everything above. Forward Euler on the same problem at
            # this step does not.
            if nsteps == 4
                fe = with_constant_surfaces!(te_RAPID(; Te_eV = Te₀); K, b_erg)
                update_RRCs!(fe)
                march!(fe, dt, nsteps; refresh_rates = true)
                @test maximum(abs, fe.plasma.Te_eV[inw] .- exact[inw]) /
                    maximum(exact[inw]) > 0.5
            end
        end
        @info "ExpRB exact on the linear problem" label Te₀ Te_sat
    end
end

@testitem "ExpRB Tₑ: off is bit-for-bit the current scheme" setup = [ExpRBTeFixtures] begin
    using RAPID2D: ForwardEuler, ExpRB, update_RRCs!

    # bern(0) = 1 makes the fallback exact rather than approximate, but the code must
    # also not perturb the arithmetic on the way — the manuscript's figures are
    # pinned to current behaviour, so "unchanged" has to mean every bit.
    for implicit in (true, false)
        a = te_RAPID(; Te_eV = 9.3, implicit = implicit)
        update_RRCs!(a)
        march!(a, 2.0e-8, 50)

        b = te_RAPID(; Te_eV = 9.3, implicit = implicit)
        @test b.flags.scheme.atomic === ForwardEuler
        update_RRCs!(b)
        march!(b, 2.0e-8, 50)

        @test a.plasma.Te_eV == b.plasma.Te_eV
    end
end

@testitem "ExpRB Tₑ: the diagonal is bern(z), and the sparsity pattern is untouched" setup = [ExpRBTeFixtures] begin
    using RAPID2D: ExpRB, update_RRCs!, update_electron_heating_powers!,
        update_electron_power_jacobian!, exprb_bern, exprb_cap_exponent

    # The change to the assembled system is meant to be two coefficients, not a
    # new operator: A_LHS gains diag(B − 1) and the RHS multiplies Tₑⁿ by B. If
    # the pattern moved, `factorize!` would stop reusing its symbolic
    # factorization and the cost claim would be wrong.
    RP = te_RAPID(; Te_eV = 9.3)
    RP.flags.Include_Te_diffu_term = true          # a real operator to sit beside
    RP.flags.scheme.atomic = ExpRB
    RP.dt = 1.0e-6
    update_RRCs!(RP)

    off = te_RAPID(; Te_eV = 9.3)
    off.flags.Include_Te_diffu_term = true
    off.dt = 1.0e-6
    update_RRCs!(off)

    update_Te!(RP)
    pattern_on = copy(RP.operators.A_LHS.matrix)
    update_Te!(off)
    pattern_off = copy(off.operators.A_LHS.matrix)

    @test size(pattern_on) == size(pattern_off)
    @test pattern_on.colptr == pattern_off.colptr      # identical sparsity
    @test pattern_on.rowval == pattern_off.rowval
    @test pattern_on != pattern_off                    # but different values

    # And the diagonal difference is exactly bern(z) − 1.
    update_electron_heating_powers!(RP)
    update_electron_power_jacobian!(RP)
    B = exprb_bern.(exprb_cap_exponent.(RP.plasma.exprb.eig_Te .* RP.dt))
    @test all(B .> 0)
    @test any(B .!= 1)                                 # the fit is actually doing something
end
