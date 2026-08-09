@testsnippet ExpRBLadderFixtures begin
    using RAPID2D: ExpRB, ForwardEuler, update_RRCs!, update_Te!, exprb_bern, exprb_cap_exponent

    # The real table, at the E/p of the design note's Fig 7 cool-down. E/p is not
    # a knob — it is |E∥|/(n_gas·T_gas·e) — so the field is solved for instead of
    # guessed, which also keeps the case pinned if the fixture's pressure moves.
    function ladder_RAPID(; Te_eV, EoverP = 63.0, u_para = -1.0e6, pressure = 5.0e-3)
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
        RP.flags.Coulomb_Collision = false
        RP.flags.Te_evolve = true
        RP.flags.ud_evolve = false
        RP.flags.Implicit = true
        RP.flags.Include_Te_diffu_term = false
        RP.flags.Include_Te_convec_term = false
        RP.flags.Include_heat_flux_term = false
        initialize!(RP)
        RP.plasma.Te_eV .= Te_eV
        RP.plasma.ne .= 1.0e16
        RP.plasma.ni .= 1.0e16
        RP.plasma.ue_para .= u_para
        RP.plasma.ueR .= u_para .* RP.fields.bR
        RP.plasma.ueϕ .= u_para .* RP.fields.bϕ
        RP.plasma.ueZ .= u_para .* RP.fields.bZ
        # See exprb_te_test.jl: drag's Coulomb half is not gated on the flag.
        RP.plasma.sptz_fac .= 0.0
        RP.plasma.ν_ei .= 0.0
        ee = RP.config.constants.ee
        @. RP.fields.E_para_tot = -EoverP * RP.plasma.n_H2_gas * RP.plasma.T_gas_eV * ee
        return RP
    end

    # One step of the real loop, reduced to what moves Tₑ: re-materialize the
    # rates at the current state (what `update_transport_quantities!` does at the
    # end of every iteration), then advance. Returns the trajectory extremes and
    # the z range, which is the diagnostic the design note admits was never
    # measured in a real run.
    function march_physical!(RP, dt, nsteps)
        inw = RP.G.nodes.in_wall_nids
        lo, hi = Inf, -Inf
        z_lo, z_hi = Inf, -Inf
        for _ in 1:nsteps
            RP.dt = dt
            update_RRCs!(RP)
            update_Te!(RP)
            te = @view RP.plasma.Te_eV[inw]
            lo = min(lo, minimum(te)); hi = max(hi, maximum(te))
            if RP.flags.scheme.atomic === ExpRB
                z = @view RP.plasma.exprb.eig_Te[inw]
                z_lo = min(z_lo, minimum(z) * dt); z_hi = max(z_hi, maximum(z) * dt)
            end
        end
        return (
            Te = sum(RP.plasma.Te_eV[inw]) / length(inw),
            Te_min = lo, Te_max = hi, z_min = z_lo, z_max = z_hi,
        )
    end

    function run_ladder(; Te₀, dts, t_end, scheme)
        return map(dts) do dt
            RP = ladder_RAPID(; Te_eV = Te₀)
            RP.flags.scheme.atomic = scheme
            r = march_physical!(RP, dt, round(Int, t_end / dt))
            (dt = dt, r...)
        end
    end
end

@testitem "ExpRB Tₑ: converges to one answer across a 630× Δt ladder, cool-down and heat-up" setup = [ExpRBLadderFixtures] begin
    using RAPID2D: ExpRB, ForwardEuler

    # The reference is the ladder's OWN finest step, not BreakdownDynamics: BD is
    # separately generated data with its own sampling noise, and what has to agree
    # is the converged limit, not two independent codes digit for digit.
    #
    # Δt snapped so every rung lands exactly on t_end — `ceil(t_end/dt)` overshoots
    # by up to one step, which at the coarse end is a third more integration
    # attributed to the scheme rather than to the driver.
    t_end = 3.0e-4                       # the Fig 7 cool-down duration
    dts = [1.0e-7, 1.0e-6, 1.0e-5, 6.0e-5]     # 1× … 600× the reference step

    for (label, Te₀) in (("cool-down", 19.19), ("heat-up", 0.026))
        rungs = run_ladder(; Te₀, dts, t_end, scheme = ExpRB)
        ref = first(rungs)

        for r in rungs
            @test isfinite(r.Te)
            @test r.Te_min > 1.001 * 1.0e-3         # min_Te default; never clamped
            @test r.Te_max < 0.999 * 500.0          # max_Te default; never clamped
        end

        errs = [abs(r.Te - ref.Te) / ref.Te for r in rungs[2:end]]
        @info "ExpRB Δt ladder" label Te₀ ref.Te errs rungs[end].z_min rungs[end].z_max

        # Every rung lands on the same answer, and coarsening never helps.
        @test all(<(0.02), errs)
        @test issorted(errs)

        # No overshoot past the converged value, in the direction that matters.
        # This is the symptom the change exists to remove: at this step FE takes
        # 19.19 eV up to 40.7 before collapsing onto the min_Te clamp.
        if Te₀ > ref.Te
            @test all(r -> r.Te_max <= Te₀ * 1.001, rungs)       # cooling, never spikes up
        else
            @test all(r -> r.Te_max <= ref.Te * 1.05, rungs)     # heating, lands not rings
        end
    end
end

@testitem "ExpRB Tₑ: forward Euler breaks on the same ladder" setup = [ExpRBLadderFixtures] begin
    using RAPID2D: ExpRB, ForwardEuler

    # The failure this change removes. At Δt = 6e-5 s the design note measures FE
    # at +67.9 % with a 14.5 eV upward excursion on the cool-down; whatever the
    # exact number here, FE must not match ExpRB, or the ladder above is passing
    # for a reason unrelated to the scheme.
    t_end = 3.0e-4
    fine, coarse = 1.0e-7, 6.0e-5

    ref = only(run_ladder(; Te₀ = 19.19, dts = [fine], t_end, scheme = ExpRB))
    fe = only(run_ladder(; Te₀ = 19.19, dts = [coarse], t_end, scheme = ForwardEuler))
    ex = only(run_ladder(; Te₀ = 19.19, dts = [coarse], t_end, scheme = ExpRB))

    fe_err = abs(fe.Te - ref.Te) / ref.Te
    ex_err = abs(ex.Te - ref.Te) / ref.Te
    @info "FE vs ExpRB at 600× the reference step" ref.Te fe.Te ex.Te fe_err ex_err fe.Te_max

    @test ex_err < 0.02
    @test fe_err > 10 * ex_err          # FE is not merely worse, it is a different answer
end

@testitem "ExpRB Tₑ: a real run's z stays negative and far below the cap" setup = [ExpRBLadderFixtures] begin
    using RAPID2D: ExpRB, EXPRB_MAX_EXPONENT

    # The design note concedes that whether |λ_reaction| dominates the transport
    # norm "has NOT been measured [in a real 2D run] and should be". This is that
    # measurement for the local half: what z the atomic power actually reaches.
    #
    # Measured at Δt = 6e-5 s (600× the reference step): z ∈ [−2.91, −2.74] on the
    # cool-down and [−5.97, −2.91] on the heat-up. Two things follow, and the
    # second corrects an expectation this branch was built on.
    #
    #  1. The +30 cap is a backstop, not part of the answer. If this test ever
    #     fails, that has stopped being true.
    #  2. z is NEGATIVE throughout, on BOTH directions. Heating up toward the
    #     fixed point does not make ∂P/∂Tₑ positive — the approach is a relaxation
    #     from below, not a runaway — so **B's growth branch is not exercised by
    #     the Tₑ equation at all.** Nothing in this file should be read as evidence
    #     about it. The real-physics case for z > 0 is the continuity equation's
    #     ionization source, covered in exprb_growth_test.jl: at z = 10 in one step
    #     ExpRB reproduces e¹⁰ while Crank–Nicolson returns a negative density.
    r = only(run_ladder(; Te₀ = 19.19, dts = [6.0e-5], t_end = 3.0e-4, scheme = ExpRB))
    @info "z = eig_Te·Δt over a 600× step run" r.z_min r.z_max EXPRB_MAX_EXPONENT
    @test r.z_max < EXPRB_MAX_EXPONENT
    @test r.z_max < 0                   # decay everywhere — see (2) above
end
