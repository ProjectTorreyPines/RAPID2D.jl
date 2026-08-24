# The first step must run on the state the caller handed us, not on the one
# `initialize!` happened to build.
#
# `plasma.ν_en_*` and `plasma.P_en_*` are materialized by `update_RRCs!` and read —
# never re-queried — by everything inside `advance_timestep!` (`types.jl:366`). The
# refresh sits at the END of the `run_simulation!` loop body, so the loop MAINTAINS
# that invariant but historically never ESTABLISHED it: an initial condition assigned
# between `initialize!` and `run_simulation!` — the only way to set one, there being no
# `SimulationConfig` entry for `Te`/`u∥` — was missed by exactly the first step.
#
# Background and measurements: internal/docs/src/notes/issues/stale-rrcs-on-first-step.md
#
# The fixture is shared with `known_defects_test.jl`; `resync = false` is the shape a
# caller who does NOT know about the invariant produces.

@testitem "a state assigned after initialize! reaches the first step" setup = [AtomicOnlyOneStep] begin
    Te0, dt = 5.0, 1.0e-6

    # Same run twice: once relying on the library, once with the hand re-sync every
    # driver in this repo performs. The property under test is that the hand re-sync
    # is a NO-OP — stated without naming the function that provides it, so the test
    # survives any later restructuring of where the invariant gets established.
    lib = atomic_only(; dt, Te0, resync = false)
    hand = atomic_only(; dt, Te0, resync = true)

    # Premise: `damping_func` is identically 1 here, so the extra
    # `update_transport_quantities!` in `hand` cannot differ through the one part of that
    # function which is not idempotent (`ue_para *= damping_func`). Note it is built from
    # `fitted_wall`, NOT the `wall_R/Z` that defines `in_wall_nids` — and at this
    # resolution the fitted wall spans the whole grid, so every node reads as inside.
    # If a fixture change breaks that, `==` below starts measuring damping instead.
    @test all(isone, hand.damping_func)

    run_simulation!(lib)
    run_simulation!(hand)

    @test lib.plasma.Te_eV == hand.plasma.Te_eV
    @test lib.plasma.ne == hand.plasma.ne
    @test lib.plasma.ue_para == hand.plasma.ue_para
end

@testitem "splitting a run in two does not change the answer" begin
    # The re-sync above must not re-dose the out-wall damping. `ue_para`, `ui_para` and
    # `mean_ExB_R/Z` are multiplied by `damping_func` IN PLACE (`transport.jl:198-205`)
    # — the one part of `update_transport_quantities!` that accumulates rather than
    # recomputes, `Dpara`/`Dperp` being rebuilt from scratch first. Damping expresses a
    # suppression profile, applied once per state production; a re-derivation that
    # applies it again squares it.
    #
    # Stated as the user-visible invariant rather than as a property of the call:
    # `RP.t_end_s = …; run_simulation!(RP)` is the documented resume idiom (see the
    # SEQUENTIAL blocks in `physics_test.jl`), and a resumed run is handed a state its
    # predecessor already damped. Two half-runs must therefore equal one whole run, bit
    # for bit.
    #
    # The geometry matters: the atomic fixture above has `damping_func ≡ 1` and cannot
    # see any of this. Here the wall sits strictly inside the domain, so a band of nodes
    # carries 0 < damping_func < 1, and `ue_para` is nonzero there. Out-wall velocities
    # reach in-wall nodes through the convection/diffusion stencil before
    # `treat_electron_outside_wall!` clears the band, so "outside the wall" is not the
    # same as "cannot matter".
    FT = Float64
    function damped_geometry(t_end)
        config = SimulationConfig{FT}(
            NR = 20, NZ = 28, R_min = 0.1, R_max = 0.5, Z_min = -0.4, Z_max = 0.4,
            dt = 1.0e-6, t_end_s = t_end, R0B0 = 1.0,
            Dpara0 = 10.0, Dperp0 = 0.1, prefilled_gas_pressure = 5.0e-3,
            wall_R = [0.15, 0.45, 0.45, 0.15], wall_Z = [-0.35, -0.35, 0.35, 0.35],
            snap0D_Δt_s = 3.0e-6, snap2D_Δt_s = 3.0e-6,
        )
        config.Output_path = mktempdir(; cleanup = false)
        RP = RAPID{FT}(config)
        RP.flags = SimulationFlags{FT}(
            convec = true, diffu = true, ud_evolve = true, src = false,
            Te_evolve = false, Ti_evolve = false, Ampere = false,
            E_para_self_ES = false, E_para_self_EM = false, Gas_evolve = false,
            update_ni_independently = false, Include_ud_convec_term = false,
            Coulomb_Collision = false, negative_n_correction = false,
        )
        initialize!(RP)
        G = RP.G
        @. RP.plasma.ne = 1.0e6 * exp(-((G.R2D - 0.3)^2 / 5.0e-4 + G.Z2D^2 / 2.0e-3))
        RP.plasma.ne[G.nodes.on_out_wall_nids] .= 0.0
        RP.plasma.ue_para .= 1.0e6          # nonzero OUTSIDE the wall too
        RP.fields.BR_ext .= 10.0e-4
        RP.fields.BZ_ext .= 20.0e-4
        RAPID2D.combine_external_and_self_fields!(RP)
        return RP
    end

    whole = damped_geometry(6.0e-6)
    split = damped_geometry(3.0e-6)

    # Premise: this geometry really does damp, partially, on a real band of nodes.
    # Without it the assertions below are vacuous — which is how the defect they pin
    # stayed invisible to the fixture above.
    d = whole.damping_func
    @test count(x -> 1.0e-3 < x < 0.999, d) > 0
    @test all(isone, d[whole.G.nodes.in_wall_nids])

    run_simulation!(whole)

    run_simulation!(split)
    split.t_end_s = 6.0e-6
    run_simulation!(split)                  # resumes; must not re-damp what it inherits

    @test whole.step == split.step
    @test whole.plasma.ne == split.plasma.ne
    @test whole.plasma.ue_para == split.plasma.ue_para

    # The other half of the same rule, isolated: entering the loop must not itself be a
    # damping event. `t_end_s = 0` runs the entry and no step at all, so anything that
    # moves here moved before any physics did.
    entry = damped_geometry(6.0e-6)
    entry.t_end_s = 0.0
    u_before = copy(entry.plasma.ue_para)
    ui_before = copy(entry.plasma.ui_para)

    run_simulation!(entry)
    run_simulation!(entry)                  # `step` is still 0, so the entry runs again

    @test entry.step == 0
    @test entry.plasma.ue_para == u_before
    @test entry.plasma.ui_para == ui_before
end

@testitem "DEFECT: a hand re-sync and the library's disagree where the wall damping bites" begin
    using RAPID2D: update_transport_quantities!

    # issues/stale-rrcs-on-first-step.md §0.3
    #
    # The two ways of establishing the invariant are not interchangeable on a geometry
    # that damps. `update_transport_quantities!` queries the rate tables at the TOP of
    # the function and damps `ue_para` at the BOTTOM, so a driver calling it by hand
    # gets rates at the undamped velocity and a damped state, while the library's entry
    # call (`damp_state = false`) gets rates at whatever velocity it is handed. Measured
    # 2.0 % on `ν_en_mom_tot` at out-wall nodes.
    #
    # Fixing it needs either the ordering repaired inside that function or a marker
    # recording whether a state carries its damping — both larger than the defect they
    # would close, and both tangled with out-wall damping being a stand-in for wall
    # boundary conditions that is meant to disappear (plans/PLAN_wall-robin-numerics.md).
    # Pinned rather than fixed, so it is visible and cannot rot into a silent pass.
    FT = Float64
    function damped(t_end)
        config = SimulationConfig{FT}(
            NR = 20, NZ = 28, R_min = 0.1, R_max = 0.5, Z_min = -0.4, Z_max = 0.4,
            dt = 1.0e-6, t_end_s = t_end, R0B0 = 1.0,
            Dpara0 = 10.0, Dperp0 = 0.1, prefilled_gas_pressure = 5.0e-3,
            wall_R = [0.15, 0.45, 0.45, 0.15], wall_Z = [-0.35, -0.35, 0.35, 0.35],
            snap0D_Δt_s = 3.0e-6, snap2D_Δt_s = 3.0e-6,
        )
        config.Output_path = mktempdir(; cleanup = false)
        RP = RAPID{FT}(config)
        RP.flags = SimulationFlags{FT}(
            convec = true, diffu = true, ud_evolve = true, src = false,
            Te_evolve = false, Ti_evolve = false, Ampere = false,
            E_para_self_ES = false, E_para_self_EM = false, Gas_evolve = false,
            update_ni_independently = false, Include_ud_convec_term = false,
            Coulomb_Collision = false, negative_n_correction = false,
        )
        initialize!(RP)
        G = RP.G
        @. RP.plasma.ne = 1.0e6 * exp(-((G.R2D - 0.3)^2 / 5.0e-4 + G.Z2D^2 / 2.0e-3))
        RP.plasma.ne[G.nodes.on_out_wall_nids] .= 0.0
        RP.plasma.ue_para .= 1.0e6
        RP.fields.BR_ext .= 10.0e-4
        RP.fields.BZ_ext .= 20.0e-4
        RAPID2D.combine_external_and_self_fields!(RP)
        return RP
    end

    lib = damped(3.0e-6)
    hand = damped(3.0e-6)
    update_transport_quantities!(hand)      # the documented workaround

    # The premise: this geometry damps, so the two paths CAN diverge here.
    @test count(x -> 1.0e-3 < x < 0.999, lib.damping_func) > 0

    run_simulation!(lib)
    run_simulation!(hand)

    # INTENDED: the workaround is redundant, so it cannot change the answer.
    @test_broken lib.plasma.ue_para == hand.plasma.ue_para
    # Pinned from the other side so this cannot start recording some unrelated drift:
    # in-wall density stays together to round-off even while the out-wall states differ.
    inw = lib.G.nodes.in_wall_nids
    @test isapprox(lib.plasma.ne[inw], hand.plasma.ne[inw]; rtol = 1.0e-10)
end

@testitem "5 eV electrons cool and ionize on the first step" setup = [AtomicOnlyOneStep] begin
    # The user-visible face of the same defect, and the one that made it look like
    # physics: with the rates left at their `initialize!` state (Ē = 0.039 eV) every
    # inelastic channel is BELOW THRESHOLD. `P_en_diss_exc` and `ν_en_iz_tot` come back
    # EXACTLY zero, and the two that survive are 1.5e3 / 4.7e3 times too small. A flat
    # first step followed by a normal second one reads as a real sub-`dt` transient.
    #
    # Both assertions are magnitude statements on purpose. The stale rates are not
    # bit-zero across the board — elastic recoil and excitation still return 2.3e-20 W
    # and 9.7e-20 W — so a bare `Te < Te0` is satisfied by a 5e-7 eV drift and pins
    # nothing. (Measured before the fix: ΔTe = 5.0e-7 eV against the 1.3e-2 eV a synced
    # step removes.)
    Te0, dt = 5.0, 1.0e-6
    RP = atomic_only(; dt, Te0, resync = false)
    k = first(RP.G.nodes.in_wall_nids)

    run_simulation!(RP)

    # Cooling within an order of magnitude of the 1.3e-2 eV a synced step delivers.
    @test Te0 - RP.plasma.Te_eV[k] > 1.0e-3
    # `ν_en_iz_tot` is exactly zero below threshold, so a stale step leaves the density
    # bit-identical to its initial value: the avalanche source is missing outright.
    @test RP.plasma.ne[k] > 1.0e15
end
