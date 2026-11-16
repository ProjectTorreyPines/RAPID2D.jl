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

    # Ē = (3/2)Tₑ + ½mₑu∥²/e, and E/p = |E∥|/(n_gas·T_table·e), with the gas temperature
    # at which the TABLE's pressure is a density. Written out rather than read back from
    # the source: a shared helper that shares the wrong formula is worse than two copies
    # of the right one.
    @test Ē ≈ (1.5 .* pla.Te_eV .+ 0.5 .* me .* pla.ue_para .^ 2 ./ ee)
    @test EoverP ≈ abs.(RP.fields.E_para_tot ./ (pla.n_H2_gas .* RP.eRRCs.reference_T_gas_eV .* ee))
    # the shipped table says it: BreakdownDynamics scans at n_gas = p / (0.026 eV e)
    @test RP.eRRCs.reference_T_gas_eV == 0.026
    @test eltype(EoverP) === eltype(Ē) === Float64
    @test !all(iszero, EoverP)            # otherwise the guard test below is vacuous
end

@testitem "RRC query point: the run's gas temperature does not move it" setup = [QueryPointFixtures] begin
    using RAPID2D: _eRRC_query_point, get_electron_RRC

    # The rates are functions of E/N. The table's E/p axis is E/N at the table's own gas
    # temperature, so at a given gas density and field the run's gas temperature cannot
    # enter: a hotter gas at the same DENSITY collides exactly as often.
    RP = qp_RAPID(; E_para = -1.0)        # E/p = 200, inside the table: K_iz can tell
    EoverP_room, Ē_room = _eRRC_query_point(RP)
    K_room = get_electron_RRC(RP, :K_iz)
    @test !all(iszero, K_room)            # otherwise the comparison below is vacuous

    RP.plasma.T_gas_eV = 2 * RP.plasma.T_gas_eV          # the density stays as it is
    EoverP_hot, Ē_hot = _eRRC_query_point(RP)
    @test EoverP_hot == EoverP_room
    @test Ē_hot == Ē_room
    @test get_electron_RRC(RP, :K_iz) == K_room
end

@testitem "RRC query point: the table's own gas temperature is read from the file" setup = [QueryPointFixtures] begin
    using RAPID2D: _eRRC_query_point, Electron_RRCs, h5open, delete_object

    # A table scanned at another reference temperature says so, and the query follows the
    # table, not the run.
    RP = qp_RAPID(; E_para = -1.0)
    data_dir = joinpath(pkgdir(RAPID2D), "RRC_data")
    path = joinpath(mktempdir(), "eRRCs_EoverP_Erg.h5")
    cp(joinpath(data_dir, "eRRCs_EoverP_Erg.h5"), path)
    chmod(path, 0o644)
    T_table = 0.03
    h5open(path, "r+") do fid
        delete_object(fid, "reference_T_gas_eV")
        fid["reference_T_gas_eV"] = T_table
    end
    EoverP_shipped, _ = _eRRC_query_point(RP)
    RP.eRRCs = Electron_RRCs(path, joinpath(data_dir, "eRRCs_T_ud.h5"))
    @test RP.eRRCs.reference_T_gas_eV == T_table

    pla, ee = RP.plasma, RP.config.constants.ee
    EoverP, _ = _eRRC_query_point(RP)
    @test EoverP ≈ abs.(RP.fields.E_para_tot ./ (pla.n_H2_gas .* T_table .* ee))
    @test EoverP ≈ EoverP_shipped .* (0.026 / T_table)
end

@testitem "RRC query point: a table that does not name its gas temperature is refused" setup = [QueryPointFixtures] begin
    using RAPID2D: Electron_RRCs, h5open, delete_object

    # No default: without `reference_T_gas_eV` the table's E/p cannot be turned into E/N.
    data_dir = joinpath(pkgdir(RAPID2D), "RRC_data")
    path = joinpath(mktempdir(), "eRRCs_EoverP_Erg.h5")
    cp(joinpath(data_dir, "eRRCs_EoverP_Erg.h5"), path)
    chmod(path, 0o644)
    h5open(fid -> delete_object(fid, "reference_T_gas_eV"), path, "r+")
    @test_throws "reference_T_gas_eV" Electron_RRCs(path, joinpath(data_dir, "eRRCs_T_ud.h5"))
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

    K = get_electron_RRC(RP, :K_iz)
    @test all(isfinite, K)

    update_RRCs!(RP)
    @test all(isfinite, RP.plasma.ν_en_iz)
    @test all(isfinite, RP.plasma.dν_dTe.iz)
    @test all(iszero, RP.plasma.ν_en_iz[dead])
    @test all(iszero, RP.plasma.dν_dTe.iz[dead])
end
