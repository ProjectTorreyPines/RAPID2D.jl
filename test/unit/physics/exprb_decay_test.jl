@testitem "exprb_theta: the weight a ledger should record, valid at every z" begin
    using RAPID2D: exprb_theta, exprb_bern

    # θ(z) = (1 − bern(z))/z is what `ExpRB` amounts to when read as a θ-scheme. It
    # is never used to BUILD the scheme — 1 − θz cancels — but a consumer that
    # stores ∫…dt ≈ Δt[(1−θ)(…)ⁿ + θ(…)ⁿ⁺¹] has to know which quadrature ran.
    @test exprb_theta(0.0) == 0.5
    @test exprb_theta(0.0f0) === 0.5f0

    # Strictly inside (0, 1) everywhere, so a ledger never reads an invalid
    # weight — this is what lets `_check_implicit_weight`'s contract survive a
    # per-cell θ.
    zs = range(-200.0, 200.0, 4001)
    θs = exprb_theta.(zs)
    @test all(0 .< θs .< 1)
    @test all(<(0), diff(θs))                     # monotone decreasing in z

    # The limits ImplicitWeights assigns by hand, recovered as limits.
    @test exprb_theta(-1.0e6) ≈ 1.0 rtol = 1.0e-5   # stiff decay  → BE
    @test exprb_theta(1.0e6) ≈ 0.0 atol = 1.0e-5    # fast growth  → FE

    # The series branch joins the closed form continuously and beats it near zero.
    @test exprb_theta(1.0e-4) ≈ (1 - exprb_bern(1.0e-4)) / 1.0e-4 rtol = 1.0e-9
    θ_big(z) = Float64((big(1.0) - big(z) / expm1(big(z))) / big(z))
    for z in (1.0e-12, -1.0e-9, 1.0e-6, -1.0e-5, 1.0e-3, -2.0)
        @test exprb_theta(z) ≈ θ_big(z) rtol = 1.0e-13
    end
end

@testsnippet ExpRBDecayFixtures begin
    using RAPID2D: ExpRB, Theta, update_ue_para!, update_RRCs!, exprb_bern, exprb_theta

    function ud_RAPID(;
            u₀ = 0.0, E_para = -50.0, pressure = 5.0e-3, implicit = true,
            coulomb = false
        )
        config = SimulationConfig{Float64}(
            NR = 6, NZ = 6, R_min = 0.8, R_max = 2.2, Z_min = -1.2, Z_max = 1.2,
            dt = 1.0e-8, t_end_s = 1.0, R0B0 = 1.0,
            prefilled_gas_pressure = pressure,
            snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
            wall_R = [1.0, 2.0, 2.0, 1.0], wall_Z = [-1.0, -1.0, 1.0, 1.0],
        )
        config.Output_path = mktempdir(; cleanup = false)
        RP = RAPID{Float64}(config)
        RP.flags.Atomic_Collision = true
        RP.flags.src = true
        RP.flags.Coulomb_Collision = coulomb
        RP.flags.ud_evolve = true
        RP.flags.ud_method = "Xsec"
        RP.flags.Te_evolve = false
        RP.flags.Implicit = implicit
        RP.flags.Include_ud_convec_term = false
        RP.flags.Include_ud_pressure_term = false
        RP.flags.Include_ud_diffu_term = false
        initialize!(RP)
        RP.plasma.Te_eV .= 5.0
        RP.plasma.ne .= 1.0e16
        RP.plasma.ni .= 1.0e16
        RP.plasma.ue_para .= u₀
        RP.fields.E_para_tot .= E_para
        update_RRCs!(RP)
        return RP
    end

    # du/dt = qE/m − ν u with ν frozen: linear, so ExpRB is exact at any step.
    function drift_problem(RP)
        c = RP.config.constants
        pla = RP.plasma
        ν = @. pla.ν_en_iz + pla.ν_en_mom_tot + pla.ν_ei_eff
        u_sat = @. c.qe * RP.fields.E_para_tot / (c.me * ν)
        return (u_sat = u_sat, eig = -ν)
    end

    function march_ud!(RP, dt, nsteps)
        for _ in 1:nsteps
            RP.dt = dt
            update_ue_para!(RP)
        end
        return RP
    end
end

@testitem "ExpRB decay: exact on the frozen-friction drift, at every Δt" setup = [ExpRBDecayFixtures] begin
    using RAPID2D: ExpRB, Theta

    # The equation ImplicitWeights points to as the reason `decay` defaults to BE:
    # friction-dominated, and BE lands on u_sat where CN rings about it. ExpRB
    # does better than land on it — with ν frozen the problem is linear, so it
    # reproduces the trajectory exactly, which BE (first order) does not.
    probe = ud_RAPID()
    p = drift_problem(probe)
    inw = probe.G.nodes.in_wall_nids
    τ = 1 / minimum(abs, p.eig[inw])
    t_end = 10τ

    # Both solve paths: with no matrix, bern divides the increment and bern(−z) scales
    # uⁿ — the same two coefficients the assembled form puts on either side. The
    # closed form cannot tell them apart, so neither may the answer.
    for (label, u₀) in (("from rest", 0.0), ("from above u_sat", 3 * p.u_sat[first(inw)])),
            implicit in (true, false)
        for nsteps in (1024, 32, 1)
            dt = t_end / nsteps
            RP = ud_RAPID(; u₀ = u₀, implicit = implicit)
            RP.flags.scheme.decay = ExpRB
            march_ud!(RP, dt, nsteps)

            q = drift_problem(RP)
            exact = @. q.u_sat + (u₀ - q.u_sat) * exp(q.eig * t_end)
            @test RP.plasma.ue_para[inw] ≈ exact[inw] rtol = 1.0e-12
        end
        @info "ExpRB decay exact on the drift" label u₀
    end
end

@testitem "ExpRB decay: beats backward Euler where BE is only first order" setup = [ExpRBDecayFixtures] begin
    using RAPID2D: ExpRB, Theta

    # BE is L-stable and lands on u_sat, which is why it is the current default —
    # but it is first order, so mid-transient it is wrong by O(Δt). ExpRB is exact
    # there. Compared at a step that resolves the friction, which is where the
    # accuracy actually matters and where the two must differ.
    probe = ud_RAPID()
    p = drift_problem(probe)
    inw = probe.G.nodes.in_wall_nids
    τ = 1 / minimum(abs, p.eig[inw])
    t_end = τ                      # mid-transient: the fixed point cannot hide the error
    dt = t_end / 8                 # z ≈ −0.125, well resolved

    be = ud_RAPID(); march_ud!(be, dt, 8)                        # θ_imp.decay = 1
    ex = ud_RAPID(); ex.flags.scheme.decay = ExpRB; march_ud!(ex, dt, 8)

    q = drift_problem(ex)
    exact = @. q.u_sat + (0.0 - q.u_sat) * exp(q.eig * t_end)
    err(x) = maximum(abs, x[inw] .- exact[inw]) / maximum(abs, exact[inw])

    @info "ExpRB vs BE mid-transient" err(be.plasma.ue_para) err(ex.plasma.ue_para)
    @test err(ex.plasma.ue_para) < 1.0e-12
    @test err(be.plasma.ue_para) > 100 * max(err(ex.plasma.ue_para), eps())
end

@testitem "ExpRB decay: off is bit-for-bit, both solve paths" setup = [ExpRBDecayFixtures] begin
    using RAPID2D: Theta, ExpRB

    for implicit in (true, false), coulomb in (false, true)
        a = ud_RAPID(; implicit = implicit, coulomb = coulomb)
        march_ud!(a, 2.0e-8, 40)
        b = ud_RAPID(; implicit = implicit, coulomb = coulomb)
        @test b.flags.scheme.decay === Theta
        march_ud!(b, 2.0e-8, 40)
        @test a.plasma.ue_para == b.plasma.ue_para
        @test a.plasma.Rue_ei == b.plasma.Rue_ei
    end
end

@testitem "ExpRB decay: Rue_ei records the quadrature that actually ran" setup = [ExpRBDecayFixtures] begin
    using RAPID2D: ExpRB, exprb_theta

    # Rue_ei is a LEDGER: it stores ν_ei_eff(u_i − [(1−θ)uⁿ + θuⁿ⁺¹]), the momentum
    # exchange integrated over the step. Under ExpRB the quadrature is no longer
    # θ_imp.decay but the per-cell θ(z) — leaving the constant in place would book
    # an exchange the momentum equation did not perform.
    RP = ud_RAPID(; coulomb = true, u₀ = -1.0e5)
    RP.flags.scheme.decay = ExpRB
    RP.dt = 5.0e-7
    u_prev = copy(RP.plasma.ue_para)
    ν_sum = @. RP.plasma.ν_en_iz + RP.plasma.ν_en_mom_tot + RP.plasma.ν_ei_eff
    update_ue_para!(RP)

    θ = exprb_theta.(-ν_sum .* RP.dt)
    expected = @. RP.plasma.ν_ei_eff * (
        RP.plasma.ui_para - ((1 - θ) * u_prev + θ * RP.plasma.ue_para)
    )
    inw = RP.G.nodes.in_wall_nids
    @test RP.plasma.Rue_ei[inw] ≈ expected[inw] rtol = 1.0e-12
    @test any(θ[inw] .!= 1.0)          # genuinely not backward Euler at this step
    @test all(0 .< θ[inw] .< 1)
end
