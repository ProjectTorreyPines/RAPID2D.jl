@testsnippet SingleEvalFixtures begin
    using RAPID2D: ExpRB, FullLinearResponse, advance_timestep!,
        update_transport_quantities!

    # The cached rates and every derivative surface built from them: exactly what
    # `update_RRCs!` writes, and what nothing else is allowed to write (its own
    # docstring lists these eight `plasma.*` fields as the consumer-facing set).
    const CACHED_RATES = (
        :ν_en_iz, :ν_en_diss_iz, :ν_en_iz_tot, :ν_en_mom_tot, :ν_en_mom_ela,
        :P_en_ela, :P_en_exc, :P_en_diss_exc,
    )
    const CACHED_JACOBIANS = (:iz, :diss_iz, :mom_tot, :mom_ela, :ela_erg, :exc_erg, :diss_exc_erg)

    # Above the ionization threshold on purpose: at Ē ≈ 5 eV every ionization rate and
    # derivative is exactly zero, and a test that poisons zeros proves nothing.
    function single_eval_RAPID(; Te_eV = 10.0, u_para = -1.0e6)
        config = SimulationConfig{Float64}(
            NR = 6, NZ = 6, R_min = 0.8, R_max = 2.2, Z_min = -1.2, Z_max = 1.2,
            dt = 1.0e-7, t_end_s = 1.0e-5, R0B0 = 1.0,
            prefilled_gas_pressure = 2.0e-3,
            snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
            wall_R = [1.0, 2.0, 2.0, 1.0], wall_Z = [-1.0, -1.0, 1.0, 1.0],
        )
        config.Output_path = mktempdir(; cleanup = false)
        RP = RAPID{Float64}(config)
        for f in (:Atomic_Collision, :src, :ud_evolve, :Te_evolve, :Implicit)
            setproperty!(RP.flags, f, true)
        end
        for f in (
                :Coulomb_Collision, :diffu, :convec, :Include_Te_diffu_term,
                :Include_Te_convec_term, :Include_heat_flux_term, :Ampere,
                :E_para_self_ES, :E_para_self_EM, :mean_ExB, :turb_ExB_mixing,
            )
            setproperty!(RP.flags, f, false)
        end
        # FullLinearResponse is the setting that depends on this invariant, so it is
        # the one the test runs: it reads `dν_dTe`, which only `update_RRCs!` writes.
        RP.flags.scheme.growth = ExpRB
        RP.flags.scheme.atomic = ExpRB
        RP.flags.exprb_eigenvalue = FullLinearResponse
        initialize!(RP)
        inw = RP.G.nodes.in_wall_nids
        RP.plasma.Te_eV .= Te_eV
        RP.plasma.Ti_eV .= Te_eV
        RP.plasma.ne .= 0.0
        RP.plasma.ni .= 0.0
        RP.plasma.ne[inw] .= 1.0e15
        RP.plasma.ni[inw] .= 1.0e15
        RP.plasma.ue_para .= u_para
        RP.plasma.sptz_fac .= 0.0
        RP.plasma.ν_ei .= 0.0
        update_transport_quantities!(RP)     # the step's one evaluation point
        return RP
    end

    # Values no rate table can return, so their survival is unambiguous.
    function poison!(pla)
        for (i, f) in enumerate(CACHED_RATES)
            getproperty(pla, f) .= 1000.0 + i
        end
        for (i, f) in enumerate(CACHED_JACOBIANS)
            getproperty(pla.dν_dTe, f) .= -(2000.0 + i)
        end
        return pla
    end

    intact(pla) =
        all(all(==(1000.0 + i), getproperty(pla, f)) for (i, f) in enumerate(CACHED_RATES)) &&
        all(
        all(==(-(2000.0 + i)), getproperty(pla.dν_dTe, f))
            for (i, f) in enumerate(CACHED_JACOBIANS)
    )
end

@testitem "one rate evaluation per step: the advance reads the cache, never the tables" setup = [SingleEvalFixtures] begin
    using RAPID2D: advance_timestep!, update_transport_quantities!

    # `transport.jl` states it: one RRC evaluation per step, at the top, and the whole
    # of the next `advance_timestep!` reads `plasma.ν_en_*` rather than re-querying,
    # "so a step cannot mix coefficients from two states".
    #
    # This is not a tidiness rule. `FullLinearResponse` assembles λ from `dν_dTe`
    # cached BEFORE `update_ue_para!` moves u∥. Re-querying between the momentum solve
    # and `update_Te!` — which looks like a staleness fix — hands it a λ from the
    # post-momentum state, and on a cold start the two disagree in sign: the fresher
    # one reaches +1.7e6 1/s, saturates the exponent cap, and φ₁ multiplies the
    # increment by ~3.6e11.
    #
    # No counter is needed to pin it. Poison the cache with values no table can
    # return; if anything downstream re-queries, they are overwritten.
    RP = single_eval_RAPID()
    poison!(RP.plasma)

    advance_timestep!(RP)

    @test intact(RP.plasma)
end

@testitem "one rate evaluation per step: and the top of the next step does refresh" setup = [SingleEvalFixtures] begin
    using RAPID2D: update_transport_quantities!

    # Without this the test above would pass on code that never writes the cache at
    # all — it pins "nothing overwrote the poison", which is only meaningful if
    # something is supposed to, one call later.
    RP = single_eval_RAPID()
    poison!(RP.plasma)
    @test intact(RP.plasma)

    update_transport_quantities!(RP)

    @test !intact(RP.plasma)
    inw = RP.G.nodes.in_wall_nids
    @test all(>(0), RP.plasma.ν_en_iz[inw])          # a real rate, not the marker
    @test !all(iszero, RP.plasma.dν_dTe.iz[inw])     # and its derivative surface
end
