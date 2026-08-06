# The ion continuity equation, as the workflow actually calls it.
#
# On master `solve_ion_continuity_equation!` warned once and returned, so `ni`
# was frozen whenever `update_ni_independently` was true and slaved to `ne/Zeff`
# otherwise. Nothing else in the code knows that: `ν_ei` is built from `ni`
# (`initialization.jl`), so a frozen `ni` quietly corrupts the Coulomb collision
# frequency as soon as `ne` moves.
#
# These tests are about the wiring — that the species axis, the policy and the
# wall reach the real solver. The mixing physics is pinned in
# `ion_transport_test.jl`, the operator and batch solve in
# `ion_transport_solve_test.jl`.

@testsnippet IonRun begin
    "A discharge-shaped setup with a box wall and ions free to evolve."
    function ion_case(; NR = 25, NZ = 25, ne = 1.0e15, Te = 5.0, kw...)
        config = SimulationConfig{Float64}(
            device_Name = "manual", NR = NR, NZ = NZ,
            R_min = 1.0, R_max = 2.0, Z_min = -0.5, Z_max = 0.5,
            wall_R = [1.15, 1.85, 1.85, 1.15], wall_Z = [-0.35, -0.35, 0.35, 0.35],
            prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-8,
            t_end_s = 1.0e-6, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0;
            kw...
        )
        RP = RAPID{Float64}(config)
        initialize!(RP)
        RP.flags.update_ni_independently = true
        RP.plasma.ne .= ne
        RP.plasma.ni .= ne
        RP.plasma.Te_eV .= Te
        RP.plasma.Ti_eV .= 1.0
        return RP
    end

    "Σ J·n over in-wall nodes — the measure the operator conserves."
    inventory(RP, n) = sum(vec(RP.G.Jacob)[RP.G.nodes.in_wall_nids] .* vec(n)[RP.G.nodes.in_wall_nids])
end

@testitem "The ion continuity equation moves ni" setup = [IonRun] begin
    RP = ion_case()
    update_transport_quantities!(RP)
    # The ion source reads rates the electron solve publishes, so a standalone
    # call has to stand the producer up first — see `reset_reaction_counts!`.
    RAPID2D.update_reaction_counts!(RP)
    before = copy(RP.plasma.ni)

    solve_ion_continuity_equation!(RP)

    @test RP.plasma.ni != before
    @test all(isfinite, RP.plasma.ni)
    @test all(>=(0), RP.plasma.ni)
end

@testitem "Slaving ions to electrons is untouched" setup = [IonRun] begin
    # The default path in every existing test. Whatever the ion solver does, this
    # branch must keep behaving exactly as it did, or the whole suite moves.
    #
    # It used to be ambiguous which behaviour that was. Two places slaved ions and
    # they disagreed: the step wrote `ni .= ne ./ Zeff`, then
    # `treat_electron_outside_wall!` overwrote it with `ni .= ne`, so the
    # Zeff-aware line was dead code in `run_simulation!`. Both now call
    # `slave_ions_to_electrons!`, which is `ne/Z` for the declared species — and
    # H₂⁺ has Z = 1, so what actually runs is still `ni = ne`, bitwise.
    RP = ion_case()
    RP.flags.update_ni_independently = false
    # A hand-set Z_eff is a statement about the single-fluid closure. It is not
    # what quasineutrality balances against, so it must not move `ni` at all.
    RP.plasma.Zeff .= 1.3
    run_simulation!(RP)

    @test RP.plasma.ni == RP.plasma.ne
    @test RAPID2D.bulk_ion_charge(RP) == 1
end

@testitem "Ionization enters the ion equation at the electron rate" setup = [IonRun] begin
    # Ni_iz = Ne_iz: each ionization makes one ion AND one electron. The rate is
    # set by the ELECTRON density, so for ions it is a pure explicit source — but
    # "the electron density" means the one the electron equation itself ionized
    # at, `(1−θ)nⁿ + θnⁿ⁺¹`, NOT whatever `pla.ne` holds once that solve returns.
    # This asserted the latter, and so pinned the defect: at θ = ½ the ion gained
    # θ(Δtν)²n more than the electron every step.
    #
    # Run in the workflow's order, because that is the order the identity holds
    # in — `solve_electron_continuity_equation!` is what fills `prev_n`.
    RP = ion_case()
    RP.flags.diffu = false
    RP.flags.convec = false
    RP.flags.src = true
    update_transport_quantities!(RP)

    ne_before = copy(RP.plasma.ne)
    ni_before = copy(RP.plasma.ni)
    solve_electron_continuity_equation!(RP)
    solve_ion_continuity_equation!(RP)

    inw = RP.G.nodes.in_wall_nids
    @test any(>(0), RP.plasma.ν_en_iz[inw])      # the source is not silently zero

    # One event, one of each. Loose by the standards of this file because the two
    # sides are not computed the same way: the electron gain comes back through a
    # sparse LU solve while the ion gain, with transport off, is accumulated
    # directly. The identity is exact in exact arithmetic; 1e-12 is below the
    # round-trip.
    @test (RP.plasma.ne - ne_before)[inw] ≈ (RP.plasma.ni - ni_before)[inw] rtol = 1.0e-10

    # and it is the θ-weighted rate, written out so the scheme itself is pinned
    θ = RP.flags.θ_imp.growth
    n_star = @. (1 - θ) * ne_before + θ * RP.plasma.ne
    expected = @. ni_before + RP.dt * n_star * RP.plasma.ν_en_iz
    @test RP.plasma.ni[inw] ≈ expected[inw] rtol = 1.0e-12
end

@testitem "Turning the source off leaves the ion source out" setup = [IonRun] begin
    RP = ion_case()
    RP.flags.diffu = false
    RP.flags.convec = false
    RP.flags.src = false
    update_transport_quantities!(RP)

    before = copy(RP.plasma.ni)
    solve_ion_continuity_equation!(RP)
    @test RP.plasma.ni == before
end

@testitem "The transport policy is a type, and one species makes them agree" setup = [IonRun] begin
    using RAPID2D: SharedEffectiveTransport, PerSpeciesTransport, IonTransportPolicy

    # H2+ is the only ion species today, so the two policies must be
    # indistinguishable. That is what allows the default to be chosen on physical
    # grounds now and revisited when a second species is appended.
    @test RAPID2D.SimulationFlags{Float64}().ion_transport_policy isa IonTransportPolicy

    function final_ni(policy)
        RP = ion_case()
        RP.flags.ion_transport_policy = policy
        run_simulation!(RP)
        return copy(RP.plasma.ni)
    end

    @test final_ni(SharedEffectiveTransport()) == final_ni(PerSpeciesTransport())
end

@testitem "Ions never appear outside the wall" setup = [IonRun] begin
    RP = ion_case()
    run_simulation!(RP)

    outside = setdiff(1:(RP.G.NR * RP.G.NZ), RP.G.nodes.in_wall_nids)
    @test all(==(0.0), vec(RP.plasma.ni)[outside])
end

@testitem "A reflecting ion wall conserves ions; an absorbing one does not" setup = [IonRun] begin
    # The Robin condition's first appearance in a production solve. With the
    # source off, ion number is exactly conserved at albedo 1 and only decreases
    # at albedo 0.
    function surviving(albedo)
        RP = ion_case(; ion_wall_albedo = albedo)
        RP.flags.src = false
        RP.flags.convec = false
        update_transport_quantities!(RP)
        start = inventory(RP, RP.plasma.ni)
        for _ in 1:50
            solve_ion_continuity_equation!(RP)
        end
        return inventory(RP, RP.plasma.ni) / start
    end

    fracs = surviving.((1.0, 0.9, 0.0))
    @test fracs[1] ≈ 1.0 rtol = 1.0e-11
    @test issorted(fracs; rev = true)
    @test fracs[end] < 1.0
end

@testitem "Ions absorbed at the wall are booked as ion loss" setup = [IonRun] begin
    # The wall-aware operator never writes outside the wall, so the old
    # `treat_ion_outside_wall!` accounting -- which reads the density it finds on
    # out-of-wall nodes -- would silently report zero loss. The Robin debit has
    # to be booked where it is taken: at the face.
    RP = ion_case(; ion_wall_albedo = 0.0)
    RP.flags.src = false
    RP.flags.convec = false
    update_transport_quantities!(RP)

    start = inventory(RP, RP.plasma.ni)
    RP.diagnostics.Ntracker.cum0D_Ni_loss = 0.0
    for _ in 1:50
        solve_ion_continuity_equation!(RP)
    end
    lost = start - inventory(RP, RP.plasma.ni)

    # Σ J·n is 1/2π of a particle count; the ledger books true particles
    @test RP.diagnostics.Ntracker.cum0D_Ni_loss > 0
    @test RP.diagnostics.Ntracker.cum0D_Ni_loss ≈ 2π * lost * RP.G.dR * RP.G.dZ rtol = 1.0e-6
end

@testitem "Diffusion off means the ion operator is not assembled" setup = [IonRun] begin
    # `flags.diffu` has to reach the ion equation too, or a run configured for
    # source-only physics silently gains transport.
    RP = ion_case()
    RP.flags.diffu = false
    RP.flags.convec = false
    RP.flags.src = false
    update_transport_quantities!(RP)

    RP.plasma.ni[RP.G.nodes.in_wall_nids] .= 1.0e15 .* rand(length(RP.G.nodes.in_wall_nids))
    before = copy(RP.plasma.ni)
    for _ in 1:5
        solve_ion_continuity_equation!(RP)
    end
    @test RP.plasma.ni == before
end

@testitem "A species collides with the BULK, not with itself" setup = [IonRun] begin
    using RAPID2D: IonSpecies, ion_transport_channels, shared_turbulent_channel,
        channel_D_para, channel_D_perp

    # Ti and the fluid velocity are shared — the reaction set carries one ion
    # temperature and one ion drift, and density is the only per-species state. So
    # everything separating two ion species arrives through m and Z.
    #
    # `transport.νi_coulomb` is the SELF-collision rate (NRL Plasma Formulary
    # p.28, ν_i ∝ Z⁴μ^-1/2 n λ T^-3/2), which is what `initialization.jl` computes
    # for the bulk. A trace species does not collide with itself — it collides
    # with the bulk, and NRL's ion–ion test-particle rate (p.33) scales as
    #
    #     ν_z ∝ Z_z²Z_i² n_i (μ_i^½/μ_z)(1 + μ_i/μ_z)
    #
    # With D∥ built by Einstein, D = T/(m_z ν_z), that gives
    #
    #     D∥_z / D∥_bulk = 2√(μ_i/μ_z) / [ (Z_z/Z_i)² (1 + μ_i/μ_z) ]
    #
    # A strongly coupled species diffuses LESS, and being carried along by the bulk
    # is convection (the shared u_i) plus the pinch term — not a shared D. See
    # internal/docs/src/details/impurity-transport-basics.md §4.
    RP = ion_case()
    RP.flags.Atomic_Collision = false          # Coulomb-only, so the ratio is clean
    update_transport_quantities!(RP)
    turb = shared_turbulent_channel(RP)
    m_p = RP.config.constants.mi / 2
    ee = RP.config.constants.ee
    μ_i, Z_i = 2, 1                            # H₂⁺, the bulk

    coll(m, Z) = ion_transport_channels(RP, IonSpecies(:probe, m, Z), turb)[1]
    bohm(m, Z) = ion_transport_channels(RP, IonSpecies(:probe, m, Z), turb)[2]
    inw = RP.G.nodes.in_wall_nids
    at(x) = x[inw]
    ref = coll(2m_p, 1)
    D_ratio(μ_z, Z_z) = 2 * sqrt(μ_i / μ_z) / ((Z_z / Z_i)^2 * (1 + μ_i / μ_z))

    # the channel's own speed is now the mean speed, so it sees mass alone…
    @test at(coll(12m_p, 1).v_para) ≈ at(ref.v_para) ./ sqrt(6) rtol = 1.0e-12
    @test at(coll(2m_p, 6).v_para) ≈ at(ref.v_para) rtol = 1.0e-12
    # …and it is the SAME field the wall reads, which is the point of the basis
    @test coll(12m_p, 1).v_para == coll(12m_p, 1).vm_para

    # …but D∥ follows the ion–ion scaling, which is NOT separable in m and Z
    for (μ_z, Z_z) in ((12, 1), (2, 6), (12, 6), (16, 8), (1, 1))
        got = at(channel_D_para(coll(μ_z * m_p, Z_z))) ./ at(channel_D_para(ref))
        @test got ≈ fill(D_ratio(μ_z, Z_z), length(got)) rtol = 1.0e-9
    end

    # the bulk reproduces itself exactly, so nothing about H₂⁺ moves
    @test at(channel_D_para(coll(2m_p, 1))) == at(channel_D_para(ref))

    # Absolute, not a ratio: D∥ must BE Einstein at the species' own mass against
    # the species' own rate. Two independent things — NRL's test-particle
    # conversion and D = T/(mν) — have to compose, and a ratio test would pass if
    # both were wrong by the same factor.
    for (μ_z, Z_z) in ((12, 1), (2, 6), (12, 6), (16, 8))
        m_z = μ_z * m_p
        μ = 2m_p / m_z
        C_z = (Z_z / Z_i)^2 * sqrt(μ) * (1 + μ) / 2
        want = @. RP.plasma.Ti_eV * ee / (m_z * C_z * RP.transport.νi_coulomb)
        @test at(channel_D_para(coll(m_z, Z_z))) ≈ at(want) rtol = 1.0e-12
    end

    # Structural consequences of that form, read off the CODE and not off a local
    # formula. Charge and mass enter through different routes, so each is pinned on
    # its own: ν ∝ Z² at fixed mass, and D ∝ 1/√m once the impurity is heavy enough
    # that (1 + μ) → 1 — an infinitely heavy species stops diffusing rather than
    # settling at a mass-free floor.
    @test at(channel_D_para(coll(2m_p, 6))) ≈ at(channel_D_para(coll(2m_p, 3))) ./ 4 rtol = 1.0e-9
    # deliberately unphysical masses: the 1/√m asymptote needs μ ≪ 1 to separate
    # from the (1 + μ) factor, which is still worth 0.15 % at C⁶⁺
    @test at(channel_D_para(coll(4.0e4 * m_p, 6))) ≈
        at(channel_D_para(coll(1.0e4 * m_p, 6))) ./ 2 rtol = 1.0e-3
    @test only(unique(at(channel_D_para(coll(1.0e6 * m_p, 6))))) <
        only(unique(at(channel_D_para(coll(12m_p, 6))))) / 100

    # Bohm is mass-free but NOT charge-free: ρ_s²ω_ci = Te/(ZeB)
    @test at(channel_D_perp(bohm(12m_p, 1))) ≈ at(channel_D_perp(bohm(2m_p, 1))) rtol = 1.0e-12
    @test at(channel_D_perp(bohm(2m_p, 6))) ≈ at(channel_D_perp(bohm(2m_p, 1))) ./ 6 rtol = 1.0e-12
end

@testitem "The Z² scaling applies to Coulomb collisions, not to the neutral gas" setup = [IonRun] begin
    using RAPID2D.Statistics
    using RAPID2D: IonSpecies, ion_transport_channels, shared_turbulent_channel

    # νi_eff is ion-neutral plus ion-ion. Only the Coulomb half carries Z²; a
    # charge-exchange or elastic collision with H₂ does not care that the ion is
    # six times charged. Scaling the whole sum would overstate an impurity's
    # collisionality by Z² through the entire gas-dominated early discharge —
    # which is most of a burn-through.
    m_p = 1.6726e-27
    ratios = map((true, false)) do coulomb
        RP = ion_case()
        RP.flags.Coulomb_Collision = coulomb
        update_transport_quantities!(RP)
        turb = shared_turbulent_channel(RP)
        λ(Z) = ion_transport_channels(RP, IonSpecies(:probe, 2m_p, Z), turb)[1].λ_para
        inw = RP.G.nodes.in_wall_nids
        return mean(λ(1)[inw] ./ λ(6)[inw])
    end

    # with Coulomb collisions the ratio moves toward 36; with none it must be 1,
    # because then νi_eff is purely ion-neutral
    @test ratios[2] ≈ 1.0 rtol = 1.0e-12
    @test ratios[1] > 1.0
end

@testitem "Bohm's 1/Z is a modelling choice, so it has a flag" setup = [IonRun] begin
    using RAPID2D: IonSpecies, ion_transport_channels, shared_turbulent_channel,
        channel_D_perp

    # D_B = Te/(16 Z e B) follows from reading Bohm as a random walk of ρ_s per
    # ~1.3 gyro-periods, and it is consistent with NRL p.28's r_i ∝ Z⁻¹. But NRL
    # p.29 states Bohm itself as D_B = ckT/16eB — an ELECTRON quantity with no Z
    # in it. Bohm is an anomalous coefficient, not a derivation, so the per-charge
    # 1/Z is a modelling choice and belongs behind a flag rather than baked in.
    m_p = 1.6726e-27
    D⊥(RP, turb, Z) = channel_D_perp(
        ion_transport_channels(RP, IonSpecies(:probe, 2m_p, Z), turb)[2]
    )
    v⊥(RP, turb, Z) = ion_transport_channels(RP, IonSpecies(:probe, 2m_p, Z), turb)[2].v_perp

    RP = ion_case()
    @test RP.flags.bohm_charge_scaling            # default: unchanged behaviour
    update_transport_quantities!(RP)
    turb = shared_turbulent_channel(RP)
    inw = RP.G.nodes.in_wall_nids
    @test D⊥(RP, turb, 6)[inw] ≈ D⊥(RP, turb, 1)[inw] ./ 6 rtol = 1.0e-12

    off = ion_case()
    off.flags.bohm_charge_scaling = false
    update_transport_quantities!(off)
    turb_off = shared_turbulent_channel(off)
    @test D⊥(off, turb_off, 6)[inw] ≈ D⊥(off, turb_off, 1)[inw] rtol = 1.0e-12

    # The flag moves λ⊥ only. v⊥ = c_s/8 has no Z in it either way, so the wall
    # ceiling — which depends on speeds alone — must not notice the flag at all.
    for Z in (1, 6)
        @test v⊥(off, turb_off, Z)[inw] ≈ v⊥(RP, turb, Z)[inw] rtol = 1.0e-12
    end

    # …and at Z = 1 the flag is a no-op, so no existing single-species result moves
    @test D⊥(off, turb_off, 1)[inw] ≈ D⊥(RP, turb, 1)[inw] rtol = 1.0e-12
end

@testitem "The wall speed is the population's MEAN speed, not the D-convention speed" setup = [IonRun] begin
    using RAPID2D: IonSpecies, ion_transport_channels, shared_turbulent_channel,
        channel_ceiling

    # Γ = ¼·n·v̄·|b̂·n̂| is an integral over the Maxwellian, so it needs ⟨|v|⟩ —
    # a function of (T, m) and of nothing else. The channel's own `v` is
    # bookkeeping married to `λ` so that ½vλ reproduces D, and which split was
    # chosen is a property of how the code got D, not of the population. Deriving
    # v̄ from it makes the wall depend on that choice: the collisional channel
    # declares vp = √(2T/m) while the gas channel declares vth = √(T/m), and one
    # global v̄/v ratio cannot be right for both.
    RP = ion_case()
    update_transport_quantities!(RP)
    turb = shared_turbulent_channel(RP)
    mi = RP.config.constants.mi
    ee = RP.config.constants.ee
    coll = ion_transport_channels(RP, IonSpecies(:H2⁺, mi, 1), turb)[1]

    inw = RP.G.nodes.in_wall_nids
    v̄ = @. sqrt(8 * RP.plasma.Ti_eV * ee / (π * mi))
    @test channel_ceiling(coll, 1.0, 0.0, (1, 0))[inw] ≈ 0.25 .* v̄[inw] rtol = 1.0e-12

    # …and the magnetic projection rides on top of it unchanged: v⊥ = 0 for a
    # collisional channel, so a grazing face sees ¼v̄∥·|b̂·n̂| and nothing else.
    @test all(iszero, coll.v_perp)
    b = 0.6
    @test channel_ceiling(coll, b, 0.0, (1, 0))[inw] ≈ 0.25 .* v̄[inw] .* b rtol = 1.0e-12
end

@testitem "The collisional channel is exact in both limits, and composes at D" setup = [IonRun] begin
    using RAPID2D: IonSpecies, ion_transport_channels, shared_turbulent_channel,
        channel_D_para, channel_D_perp, maxwellian_mean_speed

    # Two mechanisms bound a parallel step: a collision rate and a geometric
    # length. Each has an exact closed form on its own, and the model is that they
    # compose as diffusivities. All three claims are checked here against values
    # built from (T, m, ν, L) alone — nothing is compared to a previous formula.
    #
    #   rate alone     D∥ = T/(mν)          Einstein
    #   length alone   D∥ = ½·vm·L          free streaming, ⟨|v_∥|⟩ = vm/2
    #   both           1/D∥ = 1/D_ν + 1/D_L
    #
    # The last line is the whole content of composing at the level of D. It holds
    # for no other pairing of moment and coefficient, which is why it is worth
    # asserting directly rather than through the value it happens to produce.
    RP = ion_case()
    update_transport_quantities!(RP)
    RP.transport.Dpara0 = 0.0                  # no additive floor riding along

    mi = RP.config.constants.mi
    ee = RP.config.constants.ee
    inw = RP.G.nodes.in_wall_nids
    H2⁺ = IonSpecies(:H2⁺, mi, 1)
    # The ExB channel is mass- and charge-free, so the caller builds it once and
    # hands the same object to every species — that is what lets a mixture return
    # it untouched instead of averaging a constant.
    turb_ref = shared_turbulent_channel(RP)
    channels(ν, L) = begin
        fill!(RP.transport.νi_neutral, ν)
        fill!(RP.transport.νi_coulomb, 0.0)
        fill!(RP.transport.L_mixing, L)
        ion_transport_channels(RP, H2⁺, turb_ref)
    end
    D(ν, L) = channel_D_para(channels(ν, L)[1])[inw]

    vm = maxwellian_mean_speed.(RP.plasma.Ti_eV, mi)[inw]
    T_J = (RP.plasma.Ti_eV .* ee)[inw]
    ν, L = 3.0e5, 0.37                         # L_mixing = 0 means "unset", i.e. unbounded

    @test D(ν, 0.0) ≈ T_J ./ (mi .* ν) rtol = 1.0e-12          # Einstein
    @test D(0.0, L) ≈ 0.5 .* vm .* L rtol = 1.0e-12            # free streaming
    @test 1 ./ D(ν, L) ≈ 1 ./ D(ν, 0.0) .+ 1 ./ D(0.0, L) rtol = 1.0e-12

    # Neither mechanism present is not a small diffusivity but no channel at all:
    # free streaming with nothing to bound it is convection, and the equation
    # already carries it as −∇·(n𝐮_i).
    @test all(iszero, D(0.0, 0.0))

    # A parallel channel contributes nothing across the field, in every limit —
    # otherwise the Bohm channel it sits beside would be double counting D⊥. And
    # the collision rate is the collisional channel's business alone: Bohm reads
    # (Te, B, m, Z) and nothing else, so it must not move when ν or L does.
    bohm_ref = channel_D_perp(channels(ν, L)[2])
    for (νi, Li) in ((ν, L), (ν, 0.0), (0.0, L), (0.0, 0.0))
        chs = channels(νi, Li)
        @test all(iszero, chs[1].v_perp) && all(iszero, chs[1].λ_perp)
        @test all(iszero, chs[1].vm_perp)
        @test channel_D_perp(chs[2]) == bohm_ref
        @test chs[3] === turb_ref               # handed through, not rebuilt
    end
end
