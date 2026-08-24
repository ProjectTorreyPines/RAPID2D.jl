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

    # Premise: `damping_func` is identically 1 on this geometry, so the extra
    # `update_transport_quantities!` in `hand` cannot differ through the one part of
    # that function which is not idempotent (`ue_para *= damping_func`,
    # `transport.jl:195`). If a future fixture change breaks this, the `==` below
    # would start measuring damping rather than the invariant.
    @test all(isone, hand.damping_func)

    run_simulation!(lib)
    run_simulation!(hand)

    @test lib.plasma.Te_eV == hand.plasma.Te_eV
    @test lib.plasma.ne == hand.plasma.ne
    @test lib.plasma.ue_para == hand.plasma.ue_para
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
