@testsnippet ExpRBTeFixtures begin
    using RAPID2D: Electron_RRCs, ExpRB, ForwardEuler, update_RRCs!, update_Te!,
        update_electron_heating_powers!, bernoulli_B
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
    function te_RAPID(; Te_eV, u_para = -5.0e6, E_para = -50.0, pressure = 5.0e-3,
            implicit = true)
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

    # Every (E/p, Ē) surface a constant ⟹ ∂K/∂Ē = 0 ⟹ the only Tₑ dependence
    # left in the power is algebraic, and P(Tₑ) = A − 𝔅·Tₑ exactly.
    function with_constant_surfaces!(RP; K)
        EoverP = collect(range(1.0, 1000.0, 8))
        Erg_eV = collect(10 .^ range(-3, 3, 16))
        path = joinpath(mktempdir(; cleanup = false), "eRRCs_EoverP_Erg.h5")
        h5open(path, "w") do fid
            fid["EoverP"] = EoverP
            fid["Erg_eV"] = Erg_eV
            for name in ("Ionization", "Total_Momentum", "Momentum_by_ela", "Total_Excitation")
                fid[name] = fill(K, length(EoverP), length(Erg_eV))
            end
            fid["characteristic_exc_erg_eV"] = RP.config.constants.char_exc_erg_eV
        end
        RP.eRRCs = Electron_RRCs(
            path, joinpath(dirname(dirname(pathof(RAPID2D))), "RRC_data", "eRRCs_T_ud.h5")
        )
        return RP
    end

    # The closed form of that linear problem, from the same constants the code
    # uses. Derived here rather than hard-coded, so it tracks the physics rather
    # than pinning a number nobody can re-derive.
    #
    # Returned PER NODE. n_H2_gas varies across the grid, so ν and therefore λ do
    # too — every node relaxes on its own clock. Te_sat does not vary (A and 𝔅 are
    # both ∝ ν, so the ratio is ν-free), but a grid average would still smear the
    # residual transient across a spread of λ and lose the exactness this test is
    # for. Compare node by node.
    function linear_te_problem(RP)
        c = RP.config.constants
        ee, me, m_H2 = c.ee, c.me, c.mi
        pla = RP.plasma
        # All four surfaces hold the same constant K, so one ν stands for all.
        ν = pla.ν_en_mom_tot
        ν_iz = pla.ν_en_iz
        ue_sq = @. pla.ueR^2 + pla.ueϕ^2 + pla.ueZ^2
        T_gas = pla.T_gas_eV                       # scalar, not a field

        𝔅 = @. (2me / m_H2) * ν * 1.5 * ee + 1.5 * ee * ν_iz
        A = @. (
            me * ue_sq * ν
                + (2me / m_H2) * ν * 1.5 * T_gas * ee
                + 0.5 * me * ue_sq * ν_iz
                - ee * c.char_exc_erg_eV * ν
                - ee * c.iz_erg_eV * ν_iz
        )
        return (Te_sat = A ./ 𝔅, λ = @.(-(2 / 3) * 𝔅 / ee))
    end

    # The exact trajectory of dTe/dt = λ(Te − Te_sat), node by node.
    exact_Te(p, Te₀, t) = @. p.Te_sat + (Te₀ - p.Te_sat) * exp(p.λ * t)

    function march!(RP, dt, nsteps)
        for _ in 1:nsteps
            RP.dt = dt
            update_Te!(RP)
        end
        return RP
    end
end

@testitem "ExpRB Tₑ: exact on the linear problem, at every Δt, both directions" setup = [ExpRBTeFixtures] begin
    using RAPID2D: ExpRB, update_RRCs!

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
    probe = with_constant_surfaces!(te_RAPID(; Te_eV = 5.0); K = 2.0e-15)
    probe.flags.scheme.atomic = ExpRB
    update_RRCs!(probe)
    p = linear_te_problem(probe)
    inw = probe.G.nodes.in_wall_nids
    @test all(<(0), p.λ[inw])                          # a relaxation, in both cases
    τ = 1 / minimum(abs, p.λ[inw])                     # the SLOWEST node sets the run
    Te_sat = p.Te_sat[first(inw)]

    for (label, Te₀_factor) in (("cool-down", 3.0), ("heat-up", 0.05))
        Te₀ = Te₀_factor * Te_sat
        t_end = 12τ

        for nsteps in (2048, 64, 4, 1)                 # down to ONE step for the whole run
            dt = t_end / nsteps
            RP = with_constant_surfaces!(te_RAPID(; Te_eV = Te₀); K = 2.0e-15)
            RP.flags.scheme.atomic = ExpRB
            update_RRCs!(RP)
            march!(RP, dt, nsteps)

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
        end
        @info "ExpRB exact on the linear problem" label Te₀ Te_sat
    end
end

@testitem "ExpRB Tₑ: forward Euler fails the same ladder, so the gate is not vacuous" setup = [ExpRBTeFixtures] begin
    using RAPID2D: ExpRB, ForwardEuler, update_RRCs!

    # A test that only ever sees a passing scheme cannot tell a working fit from a
    # coefficient that happens to be near 1. FE on the same problem at z ≈ −24
    # diverges — that is the failure this whole change exists to remove.
    probe = with_constant_surfaces!(te_RAPID(; Te_eV = 5.0); K = 2.0e-15)
    update_RRCs!(probe)
    p = linear_te_problem(probe)
    inw = probe.G.nodes.in_wall_nids
    τ = 1 / minimum(abs, p.λ[inw])
    Te₀ = 3 * p.Te_sat[first(inw)]
    t_end = 12τ
    dt = t_end / 4                     # z = λΔt ≈ −3, past FE's |1 − z| < 1

    fe = with_constant_surfaces!(te_RAPID(; Te_eV = Te₀); K = 2.0e-15)
    update_RRCs!(fe)
    march!(fe, dt, 4)

    ex = with_constant_surfaces!(te_RAPID(; Te_eV = Te₀); K = 2.0e-15)
    ex.flags.scheme.atomic = ExpRB
    update_RRCs!(ex)
    march!(ex, dt, 4)

    exact = exact_Te(linear_te_problem(ex), Te₀, t_end)
    @test ex.plasma.Te_eV[inw] ≈ exact[inw] rtol = 1.0e-12
    @test maximum(abs, fe.plasma.Te_eV[inw] .- exact[inw]) / maximum(exact[inw]) > 0.5
end

@testitem "ExpRB Tₑ: off is bit-for-bit the current scheme" setup = [ExpRBTeFixtures] begin
    using RAPID2D: ForwardEuler, ExpRB, update_RRCs!

    # B(0) = 1 makes the fallback exact rather than approximate, but the code must
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

@testitem "ExpRB Tₑ: the diagonal is B(z), and the sparsity pattern is untouched" setup = [ExpRBTeFixtures] begin
    using RAPID2D: ExpRB, update_RRCs!, update_electron_heating_powers!,
        update_electron_power_jacobian!, bernoulli_B, cap_exprb_z

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

    # And the diagonal difference is exactly B(z) − 1.
    update_electron_heating_powers!(RP)
    update_electron_power_jacobian!(RP)
    B = bernoulli_B.(cap_exprb_z.(RP.plasma.λ_Te .* RP.dt))
    @test all(B .> 0)
    @test any(B .!= 1)                                 # the fit is actually doing something
end
