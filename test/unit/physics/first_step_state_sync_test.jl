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

    run_simulation!(lib)
    run_simulation!(hand)

    @test lib.plasma.Te_eV == hand.plasma.Te_eV
    @test lib.plasma.ne == hand.plasma.ne
    @test lib.plasma.ue_para == hand.plasma.ue_para
end

@testsnippet ResumeFixtures begin
    # A wall strictly inside the domain, a nonzero drift everywhere (outside the wall too),
    # convection and diffusion on, nothing else: the shape every resume test needs, since an
    # entry that touched the state, or a step that read the wrong operator, would show.
    function wall_geometry(t_end)
        FT = Float64
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
end

@testitem "splitting a run in two does not change the answer" setup = [ResumeFixtures] begin
    # Stated as the user-visible invariant rather than as a property of the call:
    # `RP.t_end_s = …; run_simulation!(RP)` is the documented resume idiom (see the
    # SEQUENTIAL blocks in `physics_test.jl`). Entering the loop re-derives the
    # coefficients and the operator cache from the state it is handed — on every entry, a
    # resumed run included — and since that refresh rebuilds everything from the current
    # state, re-deriving on an untouched state changes nothing: two half-runs must equal one
    # whole run, bit for bit. (This used to fail through the out-wall damping, which
    # multiplied `ue_para`, `ui_para` and `mean_ExB_R/Z` in place on every entry; nothing is
    # damped outside the wall any more.)
    whole = wall_geometry(6.0e-6)
    split = wall_geometry(3.0e-6)

    run_simulation!(whole)

    run_simulation!(split)
    split.t_end_s = 6.0e-6
    run_simulation!(split)                  # resumes from the state it inherits

    @test whole.step == split.step
    @test whole.plasma.ne == split.plasma.ne
    @test whole.plasma.ue_para == split.plasma.ue_para

    # The other half of the same rule, isolated: entering the loop must not itself
    # change the state. `t_end_s = 0` runs the entry and no step at all, so anything
    # that moves here moved before any physics did.
    entry = wall_geometry(6.0e-6)
    entry.t_end_s = 0.0
    u_before = copy(entry.plasma.ue_para)
    ui_before = copy(entry.plasma.ui_para)

    run_simulation!(entry)
    run_simulation!(entry)                  # `step` is still 0, so the entry runs again

    @test entry.step == 0
    @test entry.plasma.ue_para == u_before
    @test entry.plasma.ui_para == ui_before
end

@testitem "a flag changed between two runs reaches the first resumed step" setup = [ResumeFixtures] begin
    # `flags.upwind` selects the interior scheme of the cached convection operator. The cache
    # is refreshed at the END of every step, so a flag changed between two runs would leave
    # the first resumed step on the old scheme if the entry re-derived only for a fresh run
    # (Copilot's review of #23). It re-derives on every entry, so a hand re-sync before the
    # resume must be a no-op — the same statement the fresh-run test above makes.
    lib = wall_geometry(3.0e-6)
    hand = wall_geometry(3.0e-6)
    run_simulation!(lib)
    run_simulation!(hand)
    for RP in (lib, hand)
        RP.flags.upwind = false
        RP.t_end_s = 6.0e-6
    end
    RAPID2D.update_transport_quantities!(hand)          # the hand re-sync
    run_simulation!(lib)
    run_simulation!(hand)
    @test lib.plasma.ne == hand.plasma.ne
    @test lib.plasma.ue_para == hand.plasma.ue_para
    @test lib.transport.C_e_upwind == false

    # and the flag did reach the step: the upwind continuation is a different answer
    up = wall_geometry(3.0e-6)
    run_simulation!(up)
    up.t_end_s = 6.0e-6
    run_simulation!(up)
    @test up.plasma.ne != lib.plasma.ne
end

@testitem "nothing is damped outside the wall: no damping_func, no Damp_Transp_outWall" begin
    # The out-wall damping was a stand-in for wall boundary conditions: it pulled D, the
    # drifts, the loop voltage and the temperatures down over a band of nodes outside the
    # wall so that whole-grid operators reading that band saw something tame. No operator
    # reads the band any more (every transport operator lives on in-wall rows), so the
    # device is gone: no field on `RAPID`, no flag, and the two evolve-inside-only stubs
    # that never had an implementation go with it.
    @test !(:damping_func in fieldnames(RAPID{Float64}))
    @test !(:Damp_Transp_outWall in fieldnames(SimulationFlags{Float64}))
    @test !(:evolve_ud_inWall_only in fieldnames(SimulationFlags{Float64}))
    @test !(:evolve_Te_inWall_only in fieldnames(SimulationFlags{Float64}))
    @test !isdefined(RAPID2D, :cal_damping_function_outside_wall)
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
