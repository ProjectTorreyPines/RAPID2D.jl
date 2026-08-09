@testsnippet JacobianFixtures begin
    using RAPID2D: RRC_EoverP_Erg, Electron_RRCs, load_electron_RRCs, ExpRB, ForwardEuler

    # A RAPID with a workable plasma state, small enough to be cheap.
    function jac_RAPID(; Te_eV = 5.0, ne = 1.0e16, pressure = 5.0e-3)
        config = SimulationConfig{Float64}(
            NR = 8, NZ = 8, R_min = 0.8, R_max = 2.2, Z_min = -1.2, Z_max = 1.2,
            dt = 1.0e-8, t_end_s = 1.0e-6, R0B0 = 1.0,
            prefilled_gas_pressure = pressure,
            snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
            wall_R = [1.0, 2.0, 2.0, 1.0], wall_Z = [-1.0, -1.0, 1.0, 1.0],
        )
        config.Output_path = mktempdir(; cleanup = false)
        RP = RAPID{Float64}(config)
        RP.flags.Atomic_Collision = true
        RP.flags.src = true
        initialize!(RP)
        RP.plasma.Te_eV .= Te_eV
        RP.plasma.ne .= ne
        RP.plasma.ni .= ne
        RP.plasma.ue_para .= -1.0e5
        RP.fields.E_para_tot .= -50.0
        return RP
    end

    # Replace the (E/p, Ē) surfaces with synthetic ones, keeping the real (T, u_d)
    # half — those are diagnostic-only. Written as a temporary HDF5 and read back
    # through the production loader, so no test-only constructor is added to src/
    # and the loader itself gets exercised as a side effect.
    using RAPID2D: h5open        # HDF5 is RAPID2D's dependency, not the test env's

    function with_synthetic_surfaces(RP; K_of_Ē, EoverP = collect(range(1.0, 1000.0, 24)),
            Erg_eV = collect(10 .^ range(-3, 3, 48)))
        data = [K_of_Ē(E) for _ in EoverP, E in Erg_eV]
        path = joinpath(mktempdir(; cleanup = false), "eRRCs_EoverP_Erg.h5")
        h5open(path, "w") do fid
            fid["EoverP"] = EoverP
            fid["Erg_eV"] = Erg_eV
            for name in ("Ionization", "Total_Momentum", "Momentum_by_ela", "Total_Excitation")
                fid[name] = copy(data)
            end
            fid["characteristic_exc_erg_eV"] = RP.config.constants.char_exc_erg_eV
        end
        real_T_ud = joinpath(dirname(dirname(pathof(RAPID2D))), "RRC_data", "eRRCs_T_ud.h5")
        RP.eRRCs = Electron_RRCs(path, real_T_ud)
        return RP
    end
end

@testitem "update_RRCs!: ∂ν/∂Tₑ is materialized beside ν, at the same point" setup = [JacobianFixtures] begin
    using RAPID2D: update_RRCs!, ExpRB

    # ν and ∂ν/∂Tₑ must come from ONE evaluation of the tables. A consumer that
    # re-queries gets the same physical coefficient at two different plasma
    # states, and the discrete energy budget stops closing — the same argument
    # `rrc-single-evaluation-point.md` makes for the frequencies themselves,
    # which applies with equal force to their derivatives.
    a, b = 2.0e-15, 3.5e-17
    RP = with_synthetic_surfaces(
        jac_RAPID(); K_of_Ē = Ē -> a + b * Ē
    )
    RP.flags.scheme.atomic = ExpRB
    update_RRCs!(RP)

    n_gas = RP.plasma.n_H2_gas
    inw = RP.G.nodes.in_wall_nids
    # ∂Ē/∂Tₑ = 3/2 at fixed u, and the table is linear in Ē, so this is exact.
    expected = (@. n_gas * 1.5 * b)[inw]
    for f in (:iz, :mom_tot, :mom_ela, :exc_eff)
        @test getfield(RP.plasma.dν_dTe, f)[inw] ≈ expected rtol = 1.0e-12
    end

    # The frequencies themselves are unchanged by the derivative path existing.
    me, ee = RP.config.constants.me, RP.config.constants.ee
    Ē = @. 1.5 * RP.plasma.Te_eV + 0.5 * me * RP.plasma.ue_para^2 / ee
    @test RP.plasma.ν_en_mom_tot ≈ (@. n_gas * (a + b * Ē)) rtol = 1.0e-12
end

@testitem "update_RRCs!: the derivative costs nothing when the scheme is off" setup = [JacobianFixtures] begin
    using RAPID2D: update_RRCs!, ForwardEuler, ExpRB

    # Default off must mean the table derivatives are never even queried, not
    # merely ignored — otherwise every existing run pays for a feature it does
    # not use.
    RP = with_synthetic_surfaces(
        jac_RAPID(); K_of_Ē = Ē -> 1.0e-15 + 1.0e-17 * Ē
    )
    @test RP.flags.scheme.atomic === ForwardEuler
    update_RRCs!(RP)
    @test all(iszero, RP.plasma.dν_dTe.iz)
    @test all(iszero, RP.plasma.dν_dTe.mom_tot)
    @test all(iszero, RP.plasma.dν_dTe.mom_ela)
    @test all(iszero, RP.plasma.dν_dTe.exc_eff)

    # And the frequencies are bit-identical to a run that never heard of ExpRB —
    # this is the flag-off regression in miniature.
    ν_off = copy(RP.plasma.ν_en_mom_tot)
    RP.flags.scheme.atomic = ExpRB
    update_RRCs!(RP)
    @test RP.plasma.ν_en_mom_tot == ν_off
    @test !all(iszero, RP.plasma.dν_dTe.mom_tot)
end

@testitem "update_RRCs!: no ionization derivative where there is no ionization" setup = [JacobianFixtures] begin
    using RAPID2D: update_RRCs!, ExpRB

    # ν_en_iz is zeroed outside the wall, so its derivative must be too — a
    # diagonal perturbation on a node whose rate is identically zero is a
    # fabricated dependence.
    RP = with_synthetic_surfaces(
        jac_RAPID(); K_of_Ē = Ē -> 1.0e-15 + 1.0e-17 * Ē
    )
    RP.flags.scheme.atomic = ExpRB
    update_RRCs!(RP)

    out = RP.G.nodes.on_out_wall_nids
    @test !isempty(out)
    @test all(iszero, RP.plasma.ν_en_iz[out])
    @test all(iszero, RP.plasma.dν_dTe.iz[out])
end

@testitem "update_RRCs!: real tables, derivative matches a central difference of ν" setup = [JacobianFixtures] begin
    using RAPID2D: update_RRCs!, ExpRB

    # The end-to-end check of the chain rule on the production table: perturb Tₑ,
    # re-run the real update_RRCs!, and difference the frequencies it produced.
    # This is where a missing or doubled ∂Ē/∂Tₑ = 3/2 shows up.
    RP = jac_RAPID(; Te_eV = 8.0)
    RP.flags.scheme.atomic = ExpRB
    update_RRCs!(RP)
    analytic = deepcopy(RP.plasma.dν_dTe)

    h = 1.0e-4 * 8.0
    RP.plasma.Te_eV .= 8.0 + h
    update_RRCs!(RP)
    hi = (
        iz = copy(RP.plasma.ν_en_iz), mom_tot = copy(RP.plasma.ν_en_mom_tot),
        mom_ela = copy(RP.plasma.ν_en_mom_ela), exc_eff = copy(RP.plasma.ν_en_exc_eff),
    )
    RP.plasma.Te_eV .= 8.0 - h
    update_RRCs!(RP)
    lo = (
        iz = copy(RP.plasma.ν_en_iz), mom_tot = copy(RP.plasma.ν_en_mom_tot),
        mom_ela = copy(RP.plasma.ν_en_mom_ela), exc_eff = copy(RP.plasma.ν_en_exc_eff),
    )

    inw = RP.G.nodes.in_wall_nids
    for f in (:iz, :mom_tot, :mom_ela, :exc_eff)
        fd = (getfield(hi, f)[inw] .- getfield(lo, f)[inw]) ./ (2h)
        an = getfield(analytic, f)[inw]
        scale = max(maximum(abs, fd), eps())
        @test maximum(abs, an .- fd) / scale < 1.0e-6
        @test !all(iszero, an)      # a zero Jacobian would pass the line above vacuously
    end
end
