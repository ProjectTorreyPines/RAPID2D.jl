@testsnippet ExpRBIonFixtures begin
    using RAPID2D: ExpRB, ForwardEuler, Theta, FullLinearResponse, update_ui_para!, update_Ti!,
        update_ion_heating_powers!, update_ion_power_jacobian!, ion_rate_jacobian,
        get_H2_ion_RRC, exprb_bern, exprb_cap_exponent, bulk_ion_mass, bulk_ion_charge

    # 0-D-like: no ion transport, so the only thing acting on u_i∥ and T_i is the
    # local rate the scheme fits.
    function ion_RAPID(; Ti_eV = 3.0, ui = -2.0e4, E_para = -50.0, pressure = 5.0e-3)
        config = SimulationConfig{Float64}(
            NR = 8, NZ = 8, R_min = 0.8, R_max = 2.2, Z_min = -1.2, Z_max = 1.2,
            dt = 1.0e-8, t_end_s = 1.0e-6, R0B0 = 1.0,
            prefilled_gas_pressure = pressure,
            snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
            wall_R = [1.0, 2.0, 2.0, 1.0], wall_Z = [-1.0, -1.0, 1.0, 1.0],
        )
        config.Output_path = mktempdir(; cleanup = false)
        config.min_Te = 1.0e-8
        config.max_Te = 1.0e8
        RP = RAPID{Float64}(config)
        RP.flags.Atomic_Collision = true
        RP.flags.src = true
        RP.flags.Coulomb_Collision = false
        RP.flags.Ti_evolve = true
        RP.flags.ud_evolve = true
        # This file measures the full ∂f/∂y; PartialLinearResponse is a different question.
        RP.flags.exprb_eigenvalue = FullLinearResponse
        initialize!(RP)
        RP.config.min_Te = 1.0e-8
        RP.config.max_Te = 1.0e8
        RP.plasma.ne .= 1.0e16
        RP.plasma.ni .= 1.0e16
        RP.plasma.Ti_eV .= Ti_eV
        RP.plasma.ui_para .= ui
        RP.plasma.uiR .= ui .* RP.fields.bR
        RP.plasma.uiϕ .= ui .* RP.fields.bϕ
        RP.plasma.uiZ .= ui .* RP.fields.bZ
        RP.fields.E_para_tot .= E_para
        RP.plasma.ν_ei .= 0.0
        return RP
    end

    # The rate `update_ui_para!` divides by, rebuilt here so the closed form below
    # is not just the code read back.
    function ion_drag_rate(RP)
        pla = RP.plasma
        K_ela = get_H2_ion_RRC(RP, :Elastic)
        K_cx = get_H2_ion_RRC(RP, :Charge_Exchange)
        ν = @. pla.n_H2_gas * (0.5 * K_ela + K_cx)
        if RP.flags.src
            Z_i = bulk_ion_charge(RP)
            @. ν += Z_i * pla.ν_en_iz
        end
        return ν
    end

    function march_ui!(RP, dt, nsteps)
        for _ in 1:nsteps
            RP.dt = dt
            update_ui_para!(RP)
        end
        return RP
    end
end

@testitem "ExpRB ion drift: exact on the frozen-friction problem" setup = [ExpRBIonFixtures] begin
    using RAPID2D: ExpRB

    # `update_ui_para!` is the decay family: du/dt = a − ν·u. ONE step, because
    # that is the only place the frozen-coefficient claim actually holds here —
    # unlike the electron frequencies, which `update_RRCs!` materializes once per
    # step, the ion rates are queried LIVE from `(T_i, |u_i∥|)` inside this very
    # function, so ν moves with the unknown across a march. See the ladder below.
    RP0 = ion_RAPID()
    inw = RP0.G.nodes.in_wall_nids
    ν = ion_drag_rate(RP0)
    a = RP0.config.constants.ee * RP0.fields.E_para_tot[1] / bulk_ion_mass(RP0)
    u₀ = RP0.plasma.ui_para[1]
    τ = 1 / maximum(ν[inw])

    for t_end in (0.1τ, 2τ, 8τ, 200τ)        # z from −0.1 to −200 in a single step
        RP = ion_RAPID()
        RP.flags.scheme.decay = ExpRB
        march_ui!(RP, t_end, 1)
        exact = @. (a / ν) + (u₀ - a / ν) * exp(-ν * t_end)
        @test RP.plasma.ui_para[inw] ≈ exact[inw] rtol = 1.0e-10
    end
end

@testitem "ExpRB ion drift: self-converges onto the same saturated drift" setup = [ExpRBIonFixtures] begin
    using RAPID2D: ExpRB

    # What survives ν moving with u across a march: a coarse run must land where a
    # resolved one does, and must approach the saturated drift from one side.
    #
    # Deliberately NOT asserted: that ExpRB beats the hard-coded backward Euler at
    # a coarse step. It does not, and the reason is structural — θ_fit(z) < 1 on a
    # decay branch, so ExpRB is less implicit than BE and overshoots more once the
    # step outruns the rate. ExpRB's gain on this equation is the fitted weight
    # where the step resolves the friction, not monotonicity at a coarse one.
    inw = ion_RAPID().G.nodes.in_wall_nids
    τ = 1 / maximum(ion_drag_rate(ion_RAPID())[inw])
    t_end = 8τ

    march(nsteps) = begin
        RP = ion_RAPID()
        RP.flags.scheme.decay = ExpRB
        march_ui!(RP, t_end / nsteps, nsteps)
        RP.plasma.ui_para[first(inw)]
    end

    ref = march(20000)
    for nsteps in (2048, 256)
        @test abs(march(nsteps) - ref) / abs(ref) < 0.05
    end

    # At coarser steps it overshoots PAST the saturated drift — 9.6 % at 64 steps —
    # and that is not the fit failing. ExpRB is exact for a frozen ν; the target
    # a/ν itself retreats as |u| grows and the charge-exchange rate with it, so an
    # exact landing on the stale target is past the eventual one. Recorded rather
    # than forbidden: the same mechanism sets the electron drift's coarse-step
    # overshoot, and pinning a bound here would just pin today's rates.
    coarse = march(64)
    @info "ion drift, unresolved step" ref coarse rel = abs(coarse - ref) / abs(ref)
    @test isfinite(coarse) && coarse < 0
end

@testitem "ExpRB ion drift: off is bit-for-bit the hard-coded backward Euler" setup = [ExpRBIonFixtures] begin
    using RAPID2D: Theta

    a = ion_RAPID()
    @test a.flags.scheme.decay === Theta
    march_ui!(a, 2.0e-8, 40)
    b = ion_RAPID()
    march_ui!(b, 2.0e-8, 40)
    @test a.plasma.ui_para == b.plasma.ui_para
end

@testitem "ion_rate_jacobian: matches a central difference, and is zero off-table" setup = [ExpRBIonFixtures] begin
    using RAPID2D: ion_rate_jacobian, get_H2_ion_RRC

    # T_i is this table's own first axis, so there is no chain rule to get wrong —
    # which is exactly why a sign or an axis swap would go unnoticed without this.
    RP = ion_RAPID(; Ti_eV = 3.0)
    inw = RP.G.nodes.in_wall_nids
    for reaction in (:Elastic, :Charge_Exchange)
        an = ion_rate_jacobian(RP, reaction)
        h = 1.0e-5 * 3.0
        RP.plasma.Ti_eV .= 3.0 + h
        hi = copy(get_H2_ion_RRC(RP, reaction))
        RP.plasma.Ti_eV .= 3.0 - h
        lo = copy(get_H2_ion_RRC(RP, reaction))
        RP.plasma.Ti_eV .= 3.0
        fd = (hi .- lo) ./ (2h)
        scale = max(maximum(abs, fd[inw]), eps())
        @test maximum(abs, an[inw] .- fd[inw]) / scale < 1.0e-6
        @test !all(iszero, an[inw])
    end

    # Above the table the value is frozen, so the honest derivative is 0 — not the
    # boundary cell's slope, which is what a clamped derivative view would return.
    hot = ion_RAPID(; Ti_eV = 1.0e7)
    @test all(iszero, ion_rate_jacobian(hot, :Elastic))
end

@testitem "eig_Ti: agrees with a central difference of the assembled ion power" setup = [ExpRBIonFixtures] begin
    using RAPID2D: ExpRB, update_ion_heating_powers!, update_ion_power_jacobian!

    # The oracle that catches a term present in iPowers.tot and missing from the
    # Jacobian — the failure a hand-written product rule actually has.
    function eig_fd(RP, Ti, h)
        ee = RP.config.constants.ee
        P(T) = (RP.plasma.Ti_eV .= T; update_ion_heating_powers!(RP); copy(RP.plasma.iPowers.tot))
        hi, lo = P(Ti + h), P(Ti - h)
        RP.plasma.Ti_eV .= Ti
        return (2 / 3) .* (hi .- lo) ./ (2h) ./ ee
    end

    for Ti in (0.5, 3.0, 20.0), coulomb in (false, true)
        RP = ion_RAPID(; Ti_eV = Ti)
        RP.flags.Coulomb_Collision = coulomb
        coulomb && (RP.plasma.ν_ei .= 1.0e5)
        RP.flags.scheme.atomic = ExpRB
        update_ion_heating_powers!(RP)
        update_ion_power_jacobian!(RP)
        eig = copy(RP.plasma.exprb.eig_Ti)
        fd = eig_fd(RP, Ti, 1.0e-5 * Ti)

        inw = RP.G.nodes.in_wall_nids
        scale = max(maximum(abs, fd[inw]), eps())
        @test maximum(abs, eig[inw] .- fd[inw]) / scale < 1.0e-5
        @test !all(iszero, eig[inw])
        @test all(<(0), eig[inw])          # collisional relaxation toward the gas
    end
end

@testitem "ExpRB ion energy: stays physical at a step forward Euler inverts" setup = [ExpRBIonFixtures] begin
    using RAPID2D: ExpRB, ForwardEuler, update_ion_heating_powers!,
        update_ion_power_jacobian!

    # Δt derived from the measured eig_Ti, so this states a fact about the scheme
    # rather than a guess about the rates. At Δt = 10τ forward Euler's
    # amplification is |1 + λΔt| = 9: a relaxation becomes a sign flip, and only
    # the clamp keeps the run on the page.
    probe = ion_RAPID(; Ti_eV = 30.0)
    probe.flags.scheme.atomic = ExpRB
    update_ion_heating_powers!(probe)
    update_ion_power_jacobian!(probe)
    inw = probe.G.nodes.in_wall_nids
    @test all(<(0), probe.plasma.exprb.eig_Ti[inw])
    Δt = 10 / maximum(abs, probe.plasma.exprb.eig_Ti[inw])

    ref = ion_RAPID(; Ti_eV = 30.0)
    ref.flags.scheme.atomic = ExpRB
    for _ in 1:4000
        ref.dt = Δt / 400
        update_Ti!(ref)
    end
    Ti_fix = ref.plasma.Ti_eV[first(inw)]

    ex = ion_RAPID(; Ti_eV = 30.0)
    ex.flags.scheme.atomic = ExpRB
    ex.dt = Δt
    update_Ti!(ex)
    ex_Ti = ex.plasma.Ti_eV[first(inw)]

    fe = ion_RAPID(; Ti_eV = 30.0)
    @test fe.flags.scheme.atomic === ForwardEuler
    fe.dt = Δt
    update_Ti!(fe)
    fe_Ti = fe.plasma.Ti_eV[first(inw)]

    @info "ion energy at Δt = 10τ" Ti_fix ex_Ti fe_Ti
    # ExpRB stays on the physical side: between where it started and where it is
    # going, never past. One step cannot resolve a nonlinear relaxation, so it does
    # NOT land on Ti_fix — the frozen λ aims at the root of its own linearisation.
    @test Ti_fix < ex_Ti < 30.0
    # Forward Euler leaves the interval entirely and is caught by min_Te. Note what
    # is NOT asserted: that ExpRB ends up closer to Ti_fix. It does not here, and
    # distance to the fixed point is the wrong metric anyway — it rewards a scheme
    # that inverts all the way to the clamp for landing near it by accident.
    @test fe_Ti <= ref.config.min_Te
end

@testitem "ExpRB ion energy: off is bit-for-bit forward Euler" setup = [ExpRBIonFixtures] begin
    using RAPID2D: ForwardEuler

    a = ion_RAPID(; Ti_eV = 12.0)
    @test a.flags.scheme.atomic === ForwardEuler
    b = ion_RAPID(; Ti_eV = 12.0)
    for _ in 1:30
        a.dt = b.dt = 2.0e-8
        update_Ti!(a)
        update_Ti!(b)
    end
    @test a.plasma.Ti_eV == b.plasma.Ti_eV
end
