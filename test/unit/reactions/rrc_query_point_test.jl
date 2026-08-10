# `update_rate_jacobian!` is only a derivative of `get_electron_RRC` if the two ask
# the table the same question. That was two hand-maintained copies of the same four
# lines — including the no-gas guard, whose whole purpose is to keep a NaN out of a
# product that is about to be multiplied by zero. One definition, two callers.

@testsnippet QueryPointFixtures begin
    using RAPID2D: _eRRC_query_point, get_electron_RRC, update_rate_jacobian!,
        update_RRCs!, ExpRB, FullLinearResponse

    function qp_RAPID(; Te_eV = 5.0, u_para = -1.0e6, E_para = -50.0, pressure = 5.0e-3)
        config = SimulationConfig{Float64}(
            NR = 6, NZ = 6, R_min = 0.8, R_max = 2.2, Z_min = -1.2, Z_max = 1.2,
            dt = 1.0e-8, t_end_s = 1.0e-6, R0B0 = 1.0,
            prefilled_gas_pressure = pressure,
            snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
            wall_R = [1.0, 2.0, 2.0, 1.0], wall_Z = [-1.0, -1.0, 1.0, 1.0],
        )
        config.Output_path = mktempdir(; cleanup = false)
        RP = RAPID{Float64}(config)
        RP.flags.Atomic_Collision = true
        RP.flags.src = true
        RP.flags.scheme.atomic = ExpRB
        RP.flags.exprb_eigenvalue = FullLinearResponse
        initialize!(RP)
        RP.plasma.Te_eV .= Te_eV
        RP.plasma.ne .= 1.0e16
        RP.plasma.ni .= 1.0e16
        RP.plasma.ue_para .= u_para
        RP.fields.E_para_tot .= E_para
        return RP
    end
end

@testitem "RRC query point: one definition of (E/p, Ē), typed in FT" setup = [QueryPointFixtures] begin
    using RAPID2D: _eRRC_query_point

    RP = qp_RAPID()
    pla, ee, me = RP.plasma, RP.config.constants.ee, RP.config.constants.me
    EoverP, Ē = _eRRC_query_point(RP)

    # Ē = (3/2)Tₑ + ½mₑu∥²/e, and E/p = |E∥|/(n_gas·T_gas·e). Written out rather
    # than read back from the source: a shared helper that shares the wrong formula
    # is worse than two copies of the right one.
    @test Ē ≈ (1.5 .* pla.Te_eV .+ 0.5 .* me .* pla.ue_para .^ 2 ./ ee)
    @test EoverP ≈ abs.(RP.fields.E_para_tot ./ (pla.n_H2_gas .* pla.T_gas_eV .* ee))
    @test eltype(EoverP) === eltype(Ē) === Float64
    @test !all(iszero, EoverP)            # otherwise the guard test below is vacuous
end

@testitem "RRC query point: the no-gas guard belongs to both callers" setup = [QueryPointFixtures] begin
    using RAPID2D: _eRRC_query_point, get_electron_RRC, update_rate_jacobian!, update_RRCs!

    # With n_gas = 0 and E∥ ≠ 0 the ratio is Inf; with both zero it is NaN. Either
    # poisons `ν = n_gas·K` through 0·NaN and lands a singular row in the Ampère
    # matrix — and it poisons `∂ν/∂Tₑ` by exactly the same route. The value path
    # guarded it and the Jacobian path guarded it separately; now they cannot
    # disagree about whether they did.
    RP = qp_RAPID()
    dead = RP.G.nodes.in_wall_nids[1:2:end]
    RP.plasma.n_H2_gas[dead] .= 0.0

    EoverP, _ = _eRRC_query_point(RP)
    @test all(isfinite, EoverP)
    @test all(iszero, EoverP[dead])

    K = get_electron_RRC(RP, :Ionization)
    @test all(isfinite, K)

    update_RRCs!(RP)
    @test all(isfinite, RP.plasma.ν_en_iz)
    @test all(isfinite, RP.plasma.dν_dTe.iz)
    @test all(iszero, RP.plasma.ν_en_iz[dead])
    @test all(iszero, RP.plasma.dν_dTe.iz[dead])
end
