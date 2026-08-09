@testsnippet ExpRBGrowthFixtures begin
    using RAPID2D: ExpRB, Theta, update_RRCs!, solve_electron_continuity_equation!,
        reaction_θ, check_reaction_counts, net_electron_count, net_ion_count,
        bernoulli_B, exprb_theta, cap_exprb_z

    # Pure growth: no transport in the continuity equation, so the only thing
    # acting is the ionization source and dn/dt = +ν_iz·n exactly.
    function growth_RAPID(;
            n₀ = 1.0e14, Te_eV = 8.0, EoverP = 100.0, u_para = -1.0e6,
            pressure = 5.0e-3, implicit = true, diffu = false
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
        RP.flags.Coulomb_Collision = false
        RP.flags.Te_evolve = false
        RP.flags.ud_evolve = false
        RP.flags.Implicit = implicit
        RP.flags.diffu = diffu
        RP.flags.convec = false
        initialize!(RP)
        RP.plasma.Te_eV .= Te_eV
        RP.plasma.ne .= n₀
        RP.plasma.ni .= n₀
        RP.plasma.ue_para .= u_para
        ee = RP.config.constants.ee
        @. RP.fields.E_para_tot = -EoverP * RP.plasma.n_H2_gas * RP.plasma.T_gas_eV * ee
        update_RRCs!(RP)
        return RP
    end

    function march_ne!(RP, dt, nsteps)
        for _ in 1:nsteps
            RP.dt = dt
            solve_electron_continuity_equation!(RP)
        end
        return RP
    end
end

@testitem "ExpRB growth: exact exponential, down to one step for the whole run" setup = [ExpRBGrowthFixtures] begin
    using RAPID2D: ExpRB, Theta

    # dn/dt = +ν_iz·n with ν_iz independent of n, so this is the EXACT local
    # Jacobian and ExpRB reproduces e^(νt) at any step. This is the branch where
    # every θ has a pole — BE's at z = 1, CN's at z = 2 — and past it they return
    # NEGATIVE densities, which is a different kind of failure from a large error.
    RP0 = growth_RAPID()
    inw = RP0.G.nodes.in_wall_nids
    ν = copy(RP0.plasma.ν_en_iz)
    n₀ = 1.0e14
    t_end = 10 / maximum(ν[inw])            # ~10 e-foldings

    # Both solve paths: `Implicit = false` is not a place where the fit stops
    # applying, only one where there is no matrix — and it is the path the 0D
    # comparison drivers actually run.
    for implicit in (true, false), nsteps in (512, 16, 1)   # z up to ~10 at one step
        RP = growth_RAPID(; n₀ = n₀, implicit = implicit)
        RP.flags.scheme.growth = ExpRB
        march_ne!(RP, t_end / nsteps, nsteps)
        exact = @. n₀ * exp(ν * t_end)
        @test RP.plasma.ne[inw] ≈ exact[inw] rtol = 1.0e-12
        @test all(>(0), RP.plasma.ne[inw])
    end

    # Crank–Nicolson, the current default, on the coarsest rung: past its pole.
    cn = growth_RAPID(; n₀ = n₀)
    march_ne!(cn, t_end, 1)
    z = maximum(ν[inw]) * t_end
    @info "growth branch at one step" z minimum(cn.plasma.ne[inw]) maximum(cn.plasma.ne[inw])
    @test z > 2                              # genuinely past CN's pole
    @test any(<(0), cn.plasma.ne[inw])       # …and it returns negative densities
end

@testitem "ExpRB growth: off is bit-for-bit, with and without transport" setup = [ExpRBGrowthFixtures] begin
    using RAPID2D: Theta, ExpRB

    for implicit in (true, false), diffu in (false, true)
        a = growth_RAPID(; implicit = implicit, diffu = diffu)
        march_ne!(a, 1.0e-7, 30)
        b = growth_RAPID(; implicit = implicit, diffu = diffu)
        @test b.flags.scheme.growth === Theta
        march_ne!(b, 1.0e-7, 30)
        @test a.plasma.ne == b.plasma.ne
        @test a.reactions.counts.iz == b.reactions.counts.iz
    end
end

@testitem "ExpRB growth: sits beside a transport operator without moving the pattern" setup = [ExpRBGrowthFixtures] begin
    using RAPID2D: ExpRB

    # The continuity LHS is II − Δt(θ∇𝐃∇ − θ∇𝐮 + θν) and the fit replaces only the
    # last term, as a diagonal deviation. If that moved the sparsity pattern,
    # `factorize!` would stop reusing its symbolic factorization every step — the
    # cost claim depends on it, and the diffusion case is where it would show.
    on = growth_RAPID(; diffu = true); on.flags.scheme.growth = ExpRB
    off = growth_RAPID(; diffu = true)
    on.dt = off.dt = 1.0e-6
    solve_electron_continuity_equation!(on)
    solve_electron_continuity_equation!(off)

    A_on, A_off = on.operators.A_LHS.matrix, off.operators.A_LHS.matrix
    @test A_on.colptr == A_off.colptr
    @test A_on.rowval == A_off.rowval
    @test A_on != A_off
    @test all(isfinite, on.plasma.ne)
    @test all(>(0), on.plasma.ne[on.G.nodes.in_wall_nids])
end

@testitem "ExpRB growth: the event count records the quadrature that ran" setup = [ExpRBGrowthFixtures] begin
    using RAPID2D: ExpRB, reaction_θ, exprb_theta, check_reaction_counts

    # `ReactionCounts` exists so one ionization cannot make an electron and an ion
    # at different rates: the electron solve publishes ONE count and every other
    # consumer reads it. That contract survives ExpRB only if the count is formed
    # with the weight the solve actually used, which is now a per-cell θ(z) rather
    # than θ_imp.growth.
    RP = growth_RAPID()
    RP.flags.scheme.growth = ExpRB
    RP.dt = 2.0e-5
    prev = copy(RP.plasma.ne)
    ν = copy(RP.plasma.ν_en_iz)
    solve_electron_continuity_equation!(RP)

    # That θ IS the fitted weight is asserted in "the ledger reads the fitted
    # weight on BOTH solve paths"; what is checked here is the count built from it.
    θ = reaction_θ(RP, :iz)
    inw = RP.G.nodes.in_wall_nids
    counts = check_reaction_counts(RP)
    expected = @. RP.dt * ((1 - θ) * prev + θ * RP.plasma.ne) * ν
    @test counts.iz[inw] ≈ expected[inw] rtol = 1.0e-12

    # Growth is where θ_fit falls BELOW ½ — the fitted weight leans explicit as
    # the step outruns the rate, which is the opposite of what stiffness intuition
    # suggests and exactly why a hand-picked constant cannot cover both signs.
    @test all(<(0.5), θ[inw])
end

@testitem "ExpRB growth: electrons and ions still agree on the event count" setup = [ExpRBGrowthFixtures] begin
    using RAPID2D: ExpRB, check_reaction_counts, net_electron_count, net_ion_count

    # The identity `ReactionCounts` guarantees, asserted under the new scheme: the
    # ion equation reads the published count and divides by Δt, so it cannot
    # disagree with the electron equation regardless of which weight formed it.
    # Also the discrete statement of "one ionization makes one electron and one
    # ion" — Δnₑ from the solve must equal the count it published.
    #
    # BOTH solve paths. `Implicit = false` is not a place where the fit stops
    # applying — the explicit branch runs the same two coefficients, so the ledger
    # owes the same quadrature. Two earlier tests each covered half of this
    # (one loops over `implicit` but checks only `ne`; this one checked the count
    # but only implicitly), and the gap between them is exactly where the count
    # and the solve were free to disagree.
    for implicit in (true, false)
        RP = growth_RAPID(; implicit = implicit)
        RP.flags.scheme.growth = ExpRB
        RP.dt = 1.0e-5
        prev = copy(RP.plasma.ne)
        solve_electron_continuity_equation!(RP)

        counts = check_reaction_counts(RP)
        inw = RP.G.nodes.in_wall_nids
        Δne = RP.plasma.ne .- prev
        @test Δne[inw] ≈ net_electron_count(counts)[inw] rtol = 1.0e-10
        @test net_ion_count(counts, :H2⁺)[inw] ≈ net_electron_count(counts)[inw] rtol = 1.0e-14
    end
end

@testitem "ExpRB growth: the ledger reads the fitted weight on BOTH solve paths" setup = [ExpRBGrowthFixtures] begin
    using RAPID2D: ExpRB, Theta, reaction_θ, exprb_theta, cap_exprb_z

    # `reaction_θ` answered `Implicit || return 0` before consulting `scheme`, which
    # was right while "explicit" and "unweighted" meant the same thing. ExpRB
    # separates them: the explicit branch of the continuity solve applies B(z) and
    # therefore integrated the source with θ_fit(z), not with 0. A ledger formed at
    # 0 under-reports every ionization — measurably, not marginally.
    #
    # Asked AFTER the solve, because that is what the question means: `reaction_θ`
    # reports the quadrature the step performed, reading the z the solve stored.
    # `expected` is still derived independently from ν and Δt, so this remains an
    # oracle rather than a restatement of the cache.
    for implicit in (true, false)
        RP = growth_RAPID(; implicit = implicit)
        RP.flags.scheme.growth = ExpRB
        RP.dt = 2.0e-5
        inw = RP.G.nodes.in_wall_nids
        expected = exprb_theta.(cap_exprb_z.(RP.plasma.ν_en_iz .* RP.dt))
        solve_electron_continuity_equation!(RP)

        θ = reaction_θ(RP, :iz)
        @test θ isa AbstractMatrix                    # per cell on both paths
        @test θ[inw] ≈ expected[inw] rtol = 1.0e-14
        @test all(0 .< θ[inw] .< 1)
    end

    # And the pre-existing meaning of `Implicit = false` is untouched where the
    # family still runs a θ-scheme: a forward-Euler source really is integrated at
    # θ = 0, and that must keep reading 0 rather than the growth family's constant.
    off = growth_RAPID(; implicit = false)
    @test off.flags.scheme.growth === Theta
    @test reaction_θ(off, :iz) == 0.0
    on = growth_RAPID(; implicit = true)
    @test reaction_θ(on, :iz) == on.flags.θ_imp.growth
end

@testitem "ExpRB growth: the ledger counts the growth the cap allowed, not the one it refused" setup = [ExpRBGrowthFixtures] begin
    using RAPID2D: ExpRB, Theta, EXPRB_Z_MAX

    # One ionization makes one electron, so with no transport the published count
    # IS the density increase — an identity the quadrature satisfies by
    # construction, at any θ and at any step. It is what lets the ion source and
    # the gas sink read `N.iz` instead of re-deriving the growth.
    #
    # Above the cap the solve advances with `z_cap` while the ledger multiplied by
    # the uncapped `ν_en_iz`, and the count came out `(z/z_cap)×` too large — 4/3
    # at `z = 40`. Ion production and neutral depletion then exceed the electrons
    # actually born, on a branch that only warns.
    function counted_vs_born(RP, dt)
        RP.dt = dt
        n_before = copy(RP.plasma.ne)
        solve_electron_continuity_equation!(RP)
        inw = RP.G.nodes.in_wall_nids
        return check_reaction_counts(RP).iz[inw], (RP.plasma.ne .- n_before)[inw]
    end

    ν_probe = maximum(growth_RAPID().plasma.ν_en_iz[growth_RAPID().G.nodes.in_wall_nids])

    for implicit in (true, false)
        # z ≈ 40: past EXPRB_Z_MAX, so the solve caps and the ledger must follow.
        RP = growth_RAPID(; implicit = implicit)
        RP.flags.scheme.growth = ExpRB
        @test ν_probe * (40 / ν_probe) > EXPRB_Z_MAX
        counted, born = counted_vs_born(RP, 40 / ν_probe)
        @test counted ≈ born rtol = 1.0e-12

        # z ≈ 2: below the cap, where the identity already held. The fix must not
        # buy the capped branch at the cost of the ordinary one.
        ok = growth_RAPID(; implicit = implicit)
        ok.flags.scheme.growth = ExpRB
        counted, born = counted_vs_born(ok, 2 / ν_probe)
        @test counted ≈ born rtol = 1.0e-12
    end

    # The identity is a property of the quadrature, not of ExpRB: Crank–Nicolson
    # satisfies it too, below its pole. Asserting it here says the ledger contract
    # is what the cap broke, not the ledger.
    cn = growth_RAPID()
    @test cn.flags.scheme.growth === Theta
    counted, born = counted_vs_born(cn, 1 / ν_probe)
    @test counted ≈ born rtol = 1.0e-12
end
