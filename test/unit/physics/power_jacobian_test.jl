@testsnippet PowerJacobianFixtures begin
    using RAPID2D: RRC_EoverP_Erg, Electron_RRCs, ExpRB, ForwardEuler, LinearResponse,
        update_RRCs!, update_electron_heating_powers!, update_electron_power_jacobian!
    using RAPID2D: h5open        # HDF5 is RAPID2D's dependency, not the test env's

    function pj_RAPID(;
            Te_eV = 5.0, u_para = -1.0e5, E_para = -50.0,
            pressure = 5.0e-3, coulomb = false, heat_flux = false
        )
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
        RP.flags.Coulomb_Collision = coulomb
        RP.flags.Include_Te_diffu_term = false
        RP.flags.Include_Te_convec_term = false
        # This file measures the full ∂f/∂y; FrozenResponse is a different question.
        RP.flags.exprb_eigenvalue = LinearResponse
        RP.flags.Include_heat_flux_term = heat_flux
        initialize!(RP)
        RP.plasma.Te_eV .= Te_eV
        RP.plasma.ne .= 1.0e16
        RP.plasma.ni .= 1.0e16
        RP.plasma.ue_para .= u_para
        # P_drag reads ueR/ueϕ/ueZ, NOT ue_para — sync them or every heating term
        # silently reads a zero drift (the trap recorded in the design note §7).
        RP.plasma.ueR .= u_para .* RP.fields.bR
        RP.plasma.ueϕ .= u_para .* RP.fields.bϕ
        RP.plasma.ueZ .= u_para .* RP.fields.bZ
        RP.fields.E_para_tot .= E_para
        if !coulomb
            # `ePowers.drag`'s Coulomb half is charged inside `Atomic_Collision`,
            # not inside `Coulomb_Collision`, so initialization's leftover
            # sptz_fac·ν_ei survives the flag being off. "Coulomb off" has to mean
            # the terms are absent, or the Jacobian comparison is against a power
            # that carries a contribution nobody asked for.
            RP.plasma.sptz_fac .= 0.0
            RP.plasma.ν_ei .= 0.0
        end
        return RP
    end

    function with_surfaces(RP; K_of_Ē)
        EoverP = collect(range(1.0, 1000.0, 24))
        Erg_eV = collect(10 .^ range(-3, 3, 48))
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
        RP.eRRCs = Electron_RRCs(
            path, joinpath(dirname(dirname(pathof(RAPID2D))), "RRC_data", "eRRCs_T_ud.h5")
        )
        return RP
    end

    # eig_Te at the current state, through the production path.
    function eig_at(RP)
        update_RRCs!(RP)
        update_electron_heating_powers!(RP)
        update_electron_power_jacobian!(RP)
        return copy(RP.plasma.exprb.eig_Te)
    end

    # The design note §6.1 reference: central-difference the WHOLE assembled power.
    # Kept as the oracle rather than the implementation because it re-derives
    # nothing — any term added to ePowers.tot is differentiated automatically,
    # which is exactly what makes it survive a rewrite of the term list.
    function eig_fd(RP; h_rel = 1.0e-5)
        ee = RP.config.constants.ee
        Te0 = copy(RP.plasma.Te_eV)
        h = @. h_rel * max(abs(Te0), 1.0)

        RP.plasma.Te_eV .= Te0 .+ h
        update_RRCs!(RP); update_electron_heating_powers!(RP)
        P_hi = copy(RP.plasma.ePowers.tot)

        RP.plasma.Te_eV .= Te0 .- h
        update_RRCs!(RP); update_electron_heating_powers!(RP)
        P_lo = copy(RP.plasma.ePowers.tot)

        RP.plasma.Te_eV .= Te0                      # restore state AND powers
        update_RRCs!(RP); update_electron_heating_powers!(RP)
        return @. (2 / 3) * (P_hi - P_lo) / (2h) / ee
    end
end

@testitem "eig_Te: exact against a hand-computed Jacobian on an Ē-linear table" setup = [PowerJacobianFixtures] begin
    using RAPID2D: ExpRB

    # Bilinear interpolation reproduces K = a + b·Ē exactly, so ∂K/∂Ē = b to
    # machine precision and every term of ∂P/∂Tₑ closes in hand-written algebra.
    # That makes this sharper than the finite-difference oracle, which truncation
    # caps near 1e-6: a dropped product rule or a missing ∂Ē/∂Tₑ = 3/2 shows up
    # here at the twelfth digit, not the sixth.
    a, b = 2.0e-15, 3.5e-17
    RP = with_surfaces(pj_RAPID(; Te_eV = 5.0); K_of_Ē = Ē -> a + b * Ē)
    RP.flags.scheme.atomic = ExpRB
    eig = eig_at(RP)

    cnst = RP.config.constants
    ee, me, char_exc_erg_eV, iz_erg_eV = cnst.ee, cnst.me, cnst.char_exc_erg_eV, cnst.iz_erg_eV
    m_H2 = cnst.mi
    pla = RP.plasma
    inw = RP.G.nodes.in_wall_nids

    ue_sq = @. pla.ueR^2 + pla.ueϕ^2 + pla.ueZ^2
    ν, dν = pla.ν_en_mom_tot, pla.dν_dTe.mom_tot          # all four surfaces are equal here
    ν_ela, dν_ela = pla.ν_en_mom_ela, pla.dν_dTe.mom_ela
    ν_exc, dν_exc = pla.ν_en_exc_eff, pla.dν_dTe.exc_eff
    ν_iz, dν_iz = pla.ν_en_iz, pla.dν_dTe.iz

    dP = @. (
        me * ue_sq * dν                                               # ∂P_drag/∂Tₑ
            - (2 * me / m_H2) * 1.5 * ee * (ν_ela + (pla.Te_eV - pla.T_gas_eV) * dν_ela)
            - ee * char_exc_erg_eV * dν_exc
            - ee * iz_erg_eV * dν_iz
            - (dν_iz * (1.5 * pla.Te_eV * ee - 0.5 * me * ue_sq) + 1.5 * ee * ν_iz)
    )
    @test eig[inw] ≈ ((2 / 3) .* dP ./ ee)[inw] rtol = 1.0e-12

    # A cooling-dominated state has λ < 0, and that sign is the whole point: it is
    # what B(z) reads to decide between damping and amplifying the FE increment.
    @test all(<(0), eig[inw])
end

@testitem "eig_Te: agrees with a central difference of the real assembled power" setup = [PowerJacobianFixtures] begin
    using RAPID2D: ExpRB

    # The oracle of design note §6.1, on the production table. This is the test
    # that catches a term present in ePowers.tot but missing from the Jacobian —
    # the failure mode hand-written product rules actually have. It also survives
    # the bd-reaction-schema-migration Stage B rewrite untouched, which is why it
    # is worth carrying permanently rather than deleting once this lands.
    for Te in (1.0, 5.0, 20.0), E_para in (-20.0, -200.0)
        RP = pj_RAPID(; Te_eV = Te, E_para = E_para)
        RP.flags.scheme.atomic = ExpRB
        eig = eig_at(RP)
        fd = eig_fd(RP)

        inw = RP.G.nodes.in_wall_nids
        scale = max(maximum(abs, fd[inw]), eps())
        @test maximum(abs, eig[inw] .- fd[inw]) / scale < 1.0e-5
        @test !all(iszero, eig[inw])          # otherwise the line above is vacuous
    end
end

@testitem "eig_Te: zero wherever the power it linearises is zero" setup = [PowerJacobianFixtures] begin
    using RAPID2D: ExpRB, ForwardEuler

    # update_electron_heating_powers! zeroes every power outside the wall. A
    # diagonal perturbation there would describe a relaxation rate for a node
    # whose power is identically zero.
    RP = pj_RAPID()
    RP.flags.scheme.atomic = ExpRB
    eig = eig_at(RP)
    out = RP.G.nodes.on_out_wall_nids
    @test !isempty(out)
    @test all(iszero, RP.plasma.ePowers.tot[out])
    @test all(iszero, eig[out])

    # With the scheme off, eig_Te is never written at all — B(0) = 1 then makes the
    # whole term vanish from the update rather than contribute a stale value.
    off = pj_RAPID()
    @test off.flags.scheme.atomic === ForwardEuler
    update_RRCs!(off)
    update_electron_heating_powers!(off)
    update_electron_power_jacobian!(off)
    @test all(iszero, off.plasma.exprb.eig_Te)
end

@testitem "eig_Te: the ν_ei omission is announced, not hidden" setup = [PowerJacobianFixtures] begin
    using RAPID2D: ExpRB
    const Warn = Base.CoreLogging.Warn      # Logging is stdlib, not a test dependency

    # ∂ν_ei/∂Tₑ is not a table lookup — it is Spitzer-like and lives outside the
    # RRC path — so P_equi's and P_drag's Coulomb halves are left out of the
    # Jacobian. That under-damps rather than over-damps, and the fixed point does
    # not depend on λ at all, so it is tolerable. It is not tolerable silently:
    # an omitted term someone later measures as an accuracy loss must be findable.
    RP = pj_RAPID(; coulomb = true)
    RP.flags.scheme.atomic = ExpRB
    @test_logs (:warn,) match_mode = :any eig_at(RP)

    # Coulomb off is the breakdown configuration and warns about nothing.
    quiet = pj_RAPID(; coulomb = false)
    quiet.flags.scheme.atomic = ExpRB
    @test_logs min_level = Warn eig_at(quiet)
end

@testitem "eig_Te: the equilibration term is differentiated, not lumped in with ∂ν_ei/∂Tₑ" setup = [PowerJacobianFixtures] begin
    using RAPID2D: ExpRB

    # `P_equi = 2μ(3/2)e(Tₑ − T_i)ν_ei` is subtracted from `ePowers.tot`, so it
    # carries a Tₑ derivative that needs no table at all: `−2μ(3/2)e·ν_ei` at
    # frozen ν_ei, and unconditionally stabilizing. Only `∂ν_ei/∂Tₑ` is
    # Spitzer-like, and the omission had been written wide enough to drop both.
    # `update_ion_power_jacobian!` keeps this exact term, so the two Jacobians
    # disagreed about one piece of physics.
    #
    # The central-difference oracle above runs Coulomb-OFF, where P_equi is
    # identically zero on both sides — which is how a missing term survived an
    # oracle built to catch missing terms.
    for Te in (1.0, 5.0), ν_ei in (1.0e8, 1.0e10)
        RP = pj_RAPID(; Te_eV = Te, coulomb = true)
        RP.flags.scheme.atomic = ExpRB
        RP.plasma.ν_ei .= ν_ei
        RP.plasma.Ti_eV .= 0.3 * Te

        eig = eig_at(RP)
        fd = eig_fd(RP)
        inw = RP.G.nodes.in_wall_nids
        scale = max(maximum(abs, fd[inw]), eps())
        @test maximum(abs, eig[inw] .- fd[inw]) / scale < 1.0e-5
    end

    # Its sign is not a matter of regime: −ν_ei times positive constants, so it can
    # only damp. That is the whole reason omitting it was tolerable-but-wrong.
    hot = pj_RAPID(; Te_eV = 5.0, coulomb = true)
    hot.flags.scheme.atomic = ExpRB
    hot.plasma.ν_ei .= 1.0e10
    cold = pj_RAPID(; Te_eV = 5.0, coulomb = true)
    cold.flags.scheme.atomic = ExpRB
    cold.plasma.ν_ei .= 0.0
    inw = hot.G.nodes.in_wall_nids
    @test all(eig_at(hot)[inw] .< eig_at(cold)[inw])
end

@testitem "eig_Te: the heat-flux omission is announced too" setup = [PowerJacobianFixtures] begin
    using RAPID2D: ExpRB
    const Warn = Base.CoreLogging.Warn

    # `ePowers.heat` is the one power with NEITHER an implicit half (nothing adds
    # it to A_LHS) nor a place in eig_Te, so with the flag on it is plain forward
    # Euler inside a step that is otherwise fitted. Its `−Tₑ(𝐮·∇ln n)` half is
    # genuinely pointwise in Tₑ and could join the diagonal; its `−∇⋅(Tₑ𝐮)` half
    # is an operator and could not, and splitting one nonlocal term in half is a
    # measurement this change did not make. Left out — but not left silent.
    RP = pj_RAPID(; coulomb = false, heat_flux = true)
    RP.flags.scheme.atomic = ExpRB
    @test_logs (:warn,) match_mode = :any eig_at(RP)

    # Off by default, and then there is nothing to announce.
    quiet = pj_RAPID(; coulomb = false)
    @test !quiet.flags.Include_heat_flux_term
    quiet.flags.scheme.atomic = ExpRB
    @test_logs min_level = Warn eig_at(quiet)
end
