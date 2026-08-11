# Defects that are documented, reproducible, and not fixed yet.
#
# Each item below is `@test_broken`, so it records Broken today and turns into an
# "Unexpected Pass" failure the moment someone fixes it — which is the point. The
# failure names the file, and the file names the issue note, so the fix and the
# evidence stay attached to each other.
#
# Every setup drives `run_simulation!` from outside with a single step, and switches
# every transport channel off so that what remains is the local atomic reactions. No
# internal function is called to stage the state, because two of these defects are
# precisely that an internal ordering is not what a caller would assume.

@testsnippet AtomicOnlyOneStep begin
    using RAPID2D: update_transport_quantities!

    # All of transport, both self-field models, and the ExB channels off. What is left
    # is the local reaction/heating system on a fixed neutral background.
    const TRANSPORT_OFF = (
        :diffu, :convec, :upwind, :mean_ExB, :turb_ExB_mixing,
        :Include_Te_diffu_term, :Include_Te_convec_term, :Include_heat_flux_term,
        :Include_ud_convec_term, :Include_ud_pressure_term, :Include_ud_diffu_term,
        :Ampere, :E_para_self_ES, :E_para_self_EM, :Coulomb_Collision, :Gas_evolve,
    )

    function atomic_only(; dt, nsteps = 1, Te0 = 0.03, θ_growth = 0.5)
        config = SimulationConfig{Float64}(
            NR = 6, NZ = 6, R_min = 0.8, R_max = 2.2, Z_min = -1.2, Z_max = 1.2,
            dt = dt, t_end_s = nsteps * dt, R0B0 = 1.0,
            prefilled_gas_pressure = 2.0e-3,
            snap0D_Δt_s = dt, snap2D_Δt_s = nsteps * dt,
            wall_R = [1.0, 2.0, 2.0, 1.0], wall_Z = [-1.0, -1.0, 1.0, 1.0],
        )
        config.Output_path = mktempdir(; cleanup = false)
        config.min_Te, config.max_Te = 1.0e-8, 1.0e8
        RP = RAPID{Float64}(config)
        for f in (:Atomic_Collision, :src, :ud_evolve, :Te_evolve, :Implicit)
            setproperty!(RP.flags, f, true)
        end
        foreach(f -> setproperty!(RP.flags, f, false), TRANSPORT_OFF)
        RP.flags.θ_imp.growth = θ_growth
        initialize!(RP)
        RP.config.min_Te, RP.config.max_Te = 1.0e-8, 1.0e8
        inw = RP.G.nodes.in_wall_nids
        RP.plasma.Te_eV .= Te0
        RP.plasma.Ti_eV .= Te0
        RP.plasma.ne .= 0.0
        RP.plasma.ni .= 0.0
        RP.plasma.ne[inw] .= 1.0e15
        RP.plasma.ni[inw] .= 1.0e15
        RP.plasma.ue_para .= 0.0
        RP.plasma.sptz_fac .= 0.0
        RP.plasma.ν_ei .= 0.0
        # The re-sync every driver in `test/tmp_regression/` performs, because
        # `initialize!` computed its rates for the state replaced just above.
        update_transport_quantities!(RP)
        return RP
    end
end

@testitem "DEFECT: one step of acceleration does not heat the electrons" setup = [AtomicOnlyOneStep] begin
    # issues/drag-heating-lags-the-momentum-solve.md
    #
    # `update_ue_para!` writes `ue_para` early in the step; `P_drag` reads
    # `ueR/ueϕ/ueZ`, which are only assembled from it at the END of the step in
    # `update_transport_quantities!`. So the energy equation is always handed the
    # previous step's velocity, and on the very first step that velocity is zero.
    #
    # Physically: electrons that went from rest to ~10⁵–10⁶ m/s against a neutral
    # background did work on it and must be hotter. They are not.
    Te0 = 0.03
    RP = atomic_only(; dt = 1.0e-5, Te0)
    k = first(RP.G.nodes.in_wall_nids)

    run_simulation!(RP)

    # The premise: the electrons really did accelerate. If this ever fails the
    # @test_broken below is measuring nothing.
    @test abs(RP.plasma.ue_para[k]) > 1.0e5

    # The defect, twice: no drag power was booked, and so no heating happened.
    @test_broken RP.plasma.ePowers.drag[k] > 0
    @test_broken RP.plasma.Te_eV[k] > Te0
end

@testitem "DEFECT: an imposed E_para_tot does not survive a single step" setup = [AtomicOnlyOneStep] begin
    # issues/failures-that-do-not-announce-themselves.md §2
    #
    # `advance_timestep!` rebuilds `E_para_tot` from the external loop voltage every
    # step (`combine_external_and_self_fields!` → `calculate_parallel_electric_field!`),
    # so a caller imposing a parallel drive this way is silently overridden — sign
    # included. The idiom appears in scenario setups across the repo.
    RP = atomic_only(; dt = 1.0e-6)
    imposed = -0.5
    RP.fields.E_para_tot .= imposed

    run_simulation!(RP)

    @test_broken all(≈(imposed), RP.fields.E_para_tot)
end

@testitem "DEFECT: a density past the growth pole kills the run inside diagnostics" setup = [AtomicOnlyOneStep] begin
    # issues/failures-that-do-not-announce-themselves.md §4
    #
    # `measure_snap0D!` computes `log(1 + cum_src/prev_total_Ne)` (diagnostics.jl:109)
    # with no guard, while the very next line wraps the analogous loss-rate expression
    # in try/catch. A θ-scheme past its growth pole returns a negative density, the
    # argument goes negative, and `log` throws — so a physics failure is reported as a
    # diagnostics failure, with a stack trace pointing at the wrong file.
    #
    # Backward Euler (θ = 1) has its pole at z = ν_iz·Δt = 1. Δt = 4e-4 s puts a
    # discharge that reaches ν_iz ≈ 4e3 1/s well past it.
    RP = atomic_only(; dt = 4.0e-4, nsteps = 6, Te0 = 10.0, θ_growth = 1.0)
    RP.plasma.ue_para .= -1.0e6

    # The premise: this really is past the pole. Measured z ≈ 1.78.
    @test maximum(RP.plasma.ν_en_iz[RP.G.nodes.in_wall_nids]) * RP.dt > 1

    err = try
        run_simulation!(RP)
        nothing
    catch e
        e
    end

    # Whatever the scheme does to the density, the run should end by its own rules
    # rather than by an unguarded `log`.
    @test_broken err === nothing
    # …and while it does not, pin WHICH failure, so this cannot quietly start
    # recording some unrelated crash as the same defect.
    err === nothing || @test err isa DomainError
end
