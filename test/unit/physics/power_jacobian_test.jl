@testsnippet PowerJacobianFixtures begin
    using RAPID2D: RRC_EoverP_Erg, Electron_RRCs, ExpRB, ForwardEuler,
        PartialLinearResponse, FullLinearResponse,
        update_RRCs!, update_electron_heating_powers!, update_electron_power_jacobian!
    using RAPID2D: h5open        # HDF5 is RAPID2D's dependency, not the test env's

    function pj_RAPID(;
            Te_eV = 5.0, u_para = -1.0e5, E_para = -50.0,
            pressure = 5.0e-3, coulomb = false, heat_flux = false,
            depth = FullLinearResponse
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
        # This file measures the full ∂f/∂y; PartialLinearResponse gets its own
        # comparison in linear_response_depth_test.jl, and its own warning case below.
        RP.flags.exprb_eigenvalue = depth
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

    # `L_of_Ē` defaults to `K_of_Ē` so callers doing exact hand-algebra keep one number
    # everywhere. Anything that compares a RESIDUAL against a SCALE must pass both:
    # `K_*` is [m³/s] (~1e-15) and `L_*` is [W·m³] (~1e-35), so one value for both puts
    # the energy sinks 20 orders above the particle terms and every check normalised by
    # a maximum stops seeing anything but the sinks.
    function with_surfaces(RP; K_of_Ē, L_of_Ē = K_of_Ē)
        EoverP = collect(range(1.0, 1000.0, 24))
        Erg_eV = collect(10 .^ range(-3, 3, 48))
        K_data = [K_of_Ē(E) for _ in EoverP, E in Erg_eV]
        L_data = [L_of_Ē(E) for _ in EoverP, E in Erg_eV]
        path = joinpath(mktempdir(; cleanup = false), "eRRCs_EoverP_Erg.h5")
        h5open(path, "w") do fid
            fid["EoverP"] = EoverP
            fid["Erg_eV"] = Erg_eV
            # Electron_RRCs now reads the full 2026-08 group ledger (Task B1); every
            # (E/p, Ē) surface it reads must exist in the file, even the ones this
            # test does not exercise, or the constructor throws on a missing dataset.
            for name in (
                    "K_iz", "K_diss_iz", "K_exc", "K_diss_exc",
                    "K_mom", "K_mom_by_ela", "K_mom_by_exc", "K_mom_by_diss_exc",
                    "K_mom_by_iz", "K_mom_by_diss_iz",
                )
                fid[name] = copy(K_data)
            end
            for name in ("L_ela", "L_exc", "L_diss_exc", "L_tot")
                fid[name] = copy(L_data)
            end
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
    #
    # K and L are given their own magnitudes -- [m³/s] and [W·m³] -- because `≈` on
    # arrays is norm-based. Feeding both from one number put the energy sinks 20 orders
    # above every particle term, and then `rtol = 1e-12` on the norm could not see them:
    # dropping `diss_iz_erg_eV·∂ν_DI/∂Tₑ` from the P_DI line left this item green.
    # Verified by mutation, and it is caught here now.
    a, b = 2.0e-15, 3.5e-17
    a_L, b_L = 2.0e-35, 3.5e-37
    RP = with_surfaces(
        pj_RAPID(; Te_eV = 5.0); K_of_Ē = Ē -> a + b * Ē, L_of_Ē = Ē -> a_L + b_L * Ē
    )
    RP.flags.scheme.atomic = ExpRB
    eig = eig_at(RP)

    cnst = RP.config.constants
    ee, me, iz_erg_eV, diss_iz_erg_eV = cnst.ee, cnst.me, cnst.iz_erg_eV, cnst.diss_iz_erg_eV
    pla = RP.plasma
    inw = RP.G.nodes.in_wall_nids

    ue_sq = @. pla.ueR^2 + pla.ueϕ^2 + pla.ueZ^2
    ν, dν = pla.ν_en_mom_tot, pla.dν_dTe.mom_tot          # every K_* surface is equal here
    ν_iz, dν_iz = pla.ν_en_iz, pla.dν_dTe.iz
    ν_diss_iz, dν_diss_iz = pla.ν_en_diss_iz, pla.dν_dTe.diss_iz
    P_ela, dP_ela = pla.P_en_ela, pla.dν_dTe.ela_erg
    P_exc, dP_exc = pla.P_en_exc, pla.dν_dTe.exc_erg
    dP_diss_exc = pla.dν_dTe.diss_exc_erg

    # Same cold-target factor update_electron_heating_powers! shares between P_ela
    # and P_exc, and the same product rule _eig_Te_from_linear_response! applies to
    # both.
    Ē_eV = @. 1.5 * pla.Te_eV + 0.5 * me * pla.ue_para^2 / ee
    Ē_floor = first(RP.eRRCs.Kerg_ela.Erg_eV)
    Ē_safe = @. max(Ē_eV, Ē_floor)
    cold = @. 1 - 1.5 * pla.T_gas_eV / Ē_safe

    dP = @. (
        me * ue_sq * dν                                               # ∂P_drag/∂Tₑ
            - (dP_ela * cold + P_ela * 2.25 * pla.T_gas_eV / Ē_safe^2)     # P_ela
            - (dP_exc * cold + P_exc * 2.25 * pla.T_gas_eV / Ē_safe^2)     # P_exc
            - dP_diss_exc                                                  # P_diss_exc
            - ee * (iz_erg_eV * dν_iz + diss_iz_erg_eV * dν_diss_iz)       # P_iz, P_DI
            - (
            (dν_iz + dν_diss_iz) * (1.5 * pla.Te_eV * ee - 0.5 * me * ue_sq)
                + 1.5 * ee * (ν_iz + ν_diss_iz)
        )
    )
    @test eig[inw] ≈ ((2 / 3) .* dP ./ ee)[inw] rtol = 1.0e-12

    # A cooling-dominated state has λ < 0, and that sign is the whole point: it is
    # what bern(z) reads to decide between damping and amplifying the FE increment.
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

    # With the scheme off, eig_Te is never written at all — bern(0) = 1 then makes the
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

@testitem "Power Jacobian: FullLinearResponse matches a finite difference of the new sinks" begin
    using RAPID2D
    using RAPID2D: update_RRCs!, update_electron_heating_powers!,
        update_electron_power_jacobian!, FullLinearResponse, ExpRB

    function powered(Te0)
        config = SimulationConfig{Float64}(
            NR = 6, NZ = 6, prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0,
            dt = 1.0e-8, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
        )
        config.Output_path = mktempdir(; cleanup = false)
        RP = RAPID{Float64}(config)
        RP.flags.scheme.atomic = ExpRB
        RP.flags.exprb_eigenvalue = FullLinearResponse
        initialize!(RP)
        RP.flags.Atomic_Collision = true
        RP.flags.src = true
        RP.plasma.Te_eV .= Te0
        RP.fields.E_para_tot .= 30.0
        update_RRCs!(RP)
        update_electron_heating_powers!(RP)
        return RP
    end

    Te0 = 5.0
    h = 1.0e-4 * Te0
    RP = powered(Te0)
    update_electron_power_jacobian!(RP)
    ee = RP.config.constants.ee
    # eig_Te = (2/3e)·∂P/∂Tₑ, so undo the prefactor to compare against ΔP/ΔTₑ.
    analytic = @. RP.plasma.exprb.eig_Te * 1.5 * ee

    Pp = powered(Te0 + h).plasma.ePowers.tot
    Pm = powered(Te0 - h).plasma.ePowers.tot
    numeric = @. (Pp - Pm) / (2h)

    inw = RP.G.nodes.in_wall_nids
    scale = maximum(abs.(numeric[inw]))
    @test maximum(abs.(analytic[inw] .- numeric[inw])) < 1.0e-3 * scale
end

@testitem "Power Jacobian: PartialLinearResponse still cannot produce a growth branch" begin
    using RAPID2D
    using RAPID2D: update_RRCs!, update_electron_power_jacobian!,
        PartialLinearResponse, ExpRB
    # The depth's guarantee: it sums rates that are non-negative in any state the model
    # describes, so λ ≤ 0 and there is no pole. The cold-target factor is the one new
    # term that could break it — it flips sign below T_gas — so check both sides.
    for Te0 in (0.001, 5.0)
        config = SimulationConfig{Float64}(
            NR = 6, NZ = 6, prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0,
            dt = 1.0e-8, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
        )
        config.Output_path = mktempdir(; cleanup = false)
        RP = RAPID{Float64}(config)
        RP.flags.scheme.atomic = ExpRB
        RP.flags.exprb_eigenvalue = PartialLinearResponse
        initialize!(RP)
        RP.flags.Atomic_Collision = true
        RP.flags.src = true
        RP.plasma.Te_eV .= Te0
        update_RRCs!(RP)
        update_electron_power_jacobian!(RP)
        @test all(<=(0.0), RP.plasma.exprb.eig_Te)
        # `<=` alone admits all-zeros, which a mutation that never writes `eig_Te`
        # would also pass.
        @test any(<(0.0), RP.plasma.exprb.eig_Te)
    end
end

@testitem "eig_Te: the heat-flux omission is announced too" setup = [PowerJacobianFixtures] begin
    using RAPID2D: ExpRB, PartialLinearResponse, FullLinearResponse
    const Warn = Base.CoreLogging.Warn

    # `ePowers.heat` is the one power with NEITHER an implicit half (nothing adds
    # it to A_LHS) nor a place in eig_Te, so with the flag on it is plain forward
    # Euler inside a step that is otherwise fitted. Its `−Tₑ(𝐮·∇ln n)` half is
    # genuinely pointwise in Tₑ and could join the diagonal; its `−∇⋅(Tₑ𝐮)` half
    # is an operator and could not, and splitting one nonlocal term in half is a
    # measurement this change did not make. Left out — but not left silent.
    #
    # It is omitted by BOTH response depths — it is not a derivative anyone declined
    # to take — so the announcement belongs to the dispatch, not to one branch of it.
    # The default depth omits strictly more, and used to be the silent one.
    for depth in (FullLinearResponse, PartialLinearResponse)
        RP = pj_RAPID(; coulomb = false, heat_flux = true, depth = depth)
        RP.flags.scheme.atomic = ExpRB
        @test_logs (:warn,) match_mode = :any eig_at(RP)

        # Off by default, and then there is nothing to announce.
        quiet = pj_RAPID(; coulomb = false, depth = depth)
        @test !quiet.flags.Include_heat_flux_term
        quiet.flags.scheme.atomic = ExpRB
        @test_logs min_level = Warn eig_at(quiet)
    end
end

@testitem "eig_Te: below the table's Ē floor the frozen cold-target factor has no derivative" setup = [PowerJacobianFixtures] begin
    using RAPID2D: ExpRB, PartialLinearResponse, FullLinearResponse

    # `update_electron_heating_powers!` evaluates the cold-target factor at
    # `max(Ē, Ē_floor)`, and the value path clamps to the table's bottom row below
    # that. So for Ē < Ē_floor BOTH the factor and the coefficients are constant in
    # Tₑ and ∂P/∂Tₑ is exactly zero. A Jacobian that still differentiates
    # 1.5·T_gas/Ē as 2.25·T_gas/Ē_safe² is not the derivative of the power it
    # linearises, and the error is not marginal: 1/Ē_floor² = 1e6.
    #
    # With u∥ = 0, Ē = (3/2)Tₑ. On the shipped table's bottom row K_iz and L_exc are
    # identically zero and L_ela is not, so P_en_ela alone carries the term here —
    # dilution, P_exc and P_drag all vanish and cannot mask the check.
    for depth in (PartialLinearResponse, FullLinearResponse)
        RP = pj_RAPID(; Te_eV = 2.0e-4, u_para = 0.0, depth = depth)
        RP.flags.scheme.atomic = ExpRB
        inw = RP.G.nodes.in_wall_nids

        Ē_floor = first(RP.eRRCs.Kerg_ela.Erg_eV)
        @test 1.5 * RP.plasma.Te_eV[first(inw)] < Ē_floor      # the premise of the item

        eig = eig_at(RP)
        @test !all(iszero, RP.plasma.P_en_ela[inw])            # or the check is vacuous
        @test all(iszero, eig[inw])
        @test eig[inw] ≈ eig_fd(RP)[inw] atol = 1.0e-12

        # Live again once Ē clears the floor, so the fix cannot be "always zero".
        RP.plasma.Te_eV .= 1.0
        @test !all(iszero, eig_at(RP)[inw])
    end
end

@testitem "eig_Te: the Ē floor is inferrable and the known-rates path allocates nothing" setup = [PowerJacobianFixtures] begin
    using RAPID2D: ExpRB, PartialLinearResponse, mean_energy_floor,
        _eig_Te_from_known_rates!, update_RRCs!, update_electron_heating_powers!

    # `RAPID.eRRCs` is declared as the ABSTRACT `AbstractSpeciesRRCs{FT}`, so reading a
    # field off it directly infers `Any`. An `Any` scalar operand inside `@.` stops
    # `Broadcast.combine_eltypes` from producing a concrete eltype, and the fused
    # kernel falls back to per-element dynamic dispatch — a cost proportional to the
    # grid. Every other rate consumer in this package reaches the tables through a
    # function barrier for exactly this reason; the Ē-floor lookup has to as well.
    RP = pj_RAPID(; depth = PartialLinearResponse)
    RP.flags.scheme.atomic = ExpRB
    @test @inferred(mean_energy_floor(RP)) isa Float64
    @test mean_energy_floor(RP) == first(RP.eRRCs.Kerg_ela.Erg_eV)

    # This path was allocation-free before the 2026-08 ledger landed; it then grew two
    # whole-grid temporaries per step, and it runs every step. What is pinned here is
    # that nothing GRID-SIZED is allocated, measured by running the same function on
    # grids that differ by 9x in node count and requiring the byte count not to move.
    #
    # It is not zero. Reading a field off `RP.eRRCs` — declared abstract — dispatches
    # dynamically, and its `Float64` result comes back boxed: one small allocation per
    # call, independent of the grid. Removing it would mean caching the floor, which
    # the fixtures above make stale the moment they swap `RP.eRRCs`, or making the
    # field concrete. A boxed scalar per step is the cheaper of those.
    function alloc_at(NR, NZ)
        config = SimulationConfig{Float64}(
            NR = NR, NZ = NZ, R_min = 0.8, R_max = 2.2, Z_min = -1.2, Z_max = 1.2,
            dt = 1.0e-8, t_end_s = 1.0e-6, R0B0 = 1.0, prefilled_gas_pressure = 5.0e-3,
            snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
            wall_R = [1.0, 2.0, 2.0, 1.0], wall_Z = [-1.0, -1.0, 1.0, 1.0],
        )
        config.Output_path = mktempdir(; cleanup = false)
        R = RAPID{Float64}(config)
        R.flags.Atomic_Collision = true
        R.flags.src = true
        R.flags.Coulomb_Collision = false
        R.flags.exprb_eigenvalue = PartialLinearResponse
        initialize!(R)
        R.flags.scheme.atomic = ExpRB
        R.plasma.Te_eV .= 5.0
        R.plasma.ne .= 1.0e16
        R.plasma.ni .= 1.0e16
        update_RRCs!(R)
        update_electron_heating_powers!(R)
        _eig_Te_from_known_rates!(R)                    # compile before measuring
        return @allocated(_eig_Te_from_known_rates!(R)), length(R.plasma.Te_eV)
    end

    small, n_small = alloc_at(8, 8)
    large, n_large = alloc_at(24, 24)
    @test n_large > 8 * n_small                         # the premise of the comparison
    @test small == large                                # nothing grid-sized survives
    @test large < 256                                   # and what remains is one scalar
end

@testitem "heating powers: the cold-target factor costs no grid temporary" setup = [PowerJacobianFixtures] begin
    using RAPID2D: ExpRB, PartialLinearResponse, update_RRCs!,
        update_electron_heating_powers!

    # `update_electron_heating_powers!` runs EVERY step whatever the scheme flags say,
    # so a grid temporary here is paid on every step of every run. The 2026-08 ledger
    # branch briefly added two — `Ē_eV` and `cold_factor` — and they are now folded
    # into the two consuming broadcasts, with the shared factor parked in
    # `ePowers.ela` between them rather than in a fresh array.
    #
    # This is NOT a zero-allocation contract, and pretending otherwise would make the
    # test a lie: the function still builds `ue_mag_sq`, `ue_dot_ui` and the transport
    # branches' own temporaries. What is pinned is how many WHOLE-GRID arrays it
    # builds, counted against a same-process measurement of one such array rather than
    # against a byte constant.
    #
    # **Counted, not bounded in bytes, because bytes are not portable here.** An
    # earlier version of this test asserted `< 85 B/node`, measured locally at 81.7,
    # and it failed CI on macOS at 91.3 while passing on ubuntu. Same Julia 1.12.7 on
    # both runners; the difference is one whole-grid array's worth of inlining
    # decision, and the local machine on 1.12.6 differed again. Observed baselines:
    #
    #     local  1.12.6 aarch64   81.7 B/node   (24x24)   ~10 arrays
    #     ubuntu 1.12.7 x86_64    < 85          (24x24)   ~10 arrays
    #     macOS  1.12.7 aarch64   91.3          (24x24)   ~11 arrays
    #
    # The platform spread (~9 B/node) is the same size as the signal this test exists
    # to catch (the two temporaries the fold removed, ~16 B/node), so **no absolute
    # byte bound can both pass everywhere and catch a two-array regression**. The
    # ceiling below is therefore deliberately loose: it catches a gross regression —
    # fusion breaking, or a handful of temporaries returning — and does not pretend to
    # catch one or two. The sharp check is the measurement recorded in the commit that
    # removed them (24x24: 56448 -> 47072 B, exactly two grid arrays).
    function bytes_per_node(NR, NZ)
        config = SimulationConfig{Float64}(
            NR = NR, NZ = NZ, R_min = 0.8, R_max = 2.2, Z_min = -1.2, Z_max = 1.2,
            dt = 1.0e-8, t_end_s = 1.0e-6, R0B0 = 1.0, prefilled_gas_pressure = 5.0e-3,
            snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
            wall_R = [1.0, 2.0, 2.0, 1.0], wall_Z = [-1.0, -1.0, 1.0, 1.0],
        )
        config.Output_path = mktempdir(; cleanup = false)
        R = RAPID{Float64}(config)
        R.flags.Atomic_Collision = true
        R.flags.src = true
        R.flags.exprb_eigenvalue = PartialLinearResponse
        initialize!(R)
        R.flags.scheme.atomic = ExpRB
        R.plasma.Te_eV .= 5.0
        R.plasma.ne .= 1.0e16
        R.plasma.ni .= 1.0e16
        R.plasma.ue_para .= -1.0e5
        R.fields.E_para_tot .= -50.0
        update_RRCs!(R)
        update_electron_heating_powers!(R)              # compile before measuring
        n = length(R.plasma.Te_eV)
        # What one whole-grid array costs on THIS machine, including its header —
        # the unit the ceiling is expressed in.
        one_grid_array = @allocated(similar(R.plasma.Te_eV))
        return @allocated(update_electron_heating_powers!(R)) / n,
            one_grid_array / n
    end

    (small, unit_small) = bytes_per_node(24, 24)
    (large, unit_large) = bytes_per_node(48, 48)

    # Grid-independence of the RATE is the premise: if the per-node cost itself moved
    # with the grid, bounding it at one size would say nothing about the other. The
    # residual gap is the function's fixed overhead spread over more nodes.
    @test isapprox(small, large; rtol = 0.1)

    # Fewer than 15 whole-grid arrays per call. Baselines above sit at 10-11, so this
    # tolerates the platform spread and a future inlining change, while a fusion
    # failure — where each `@.` stops fusing and materialises its operands — lands
    # well past it.
    @test small / unit_small < 15.0
    @test large / unit_large < 15.0
end

@testitem "eig_Te: PartialLinearResponse is exact once the surfaces stop responding" setup = [PowerJacobianFixtures] begin
    using RAPID2D: ExpRB, PartialLinearResponse

    # The DEFAULT depth drops the response THROUGH the coefficients on purpose, so a
    # finite difference of the real assembled power is not its oracle in general. That
    # is why the only item that touched this path was a transcription of the
    # implementation — and it says so itself, which means it cannot catch a term both
    # copies drop.
    #
    # Make the dropped part identically zero instead. With Ē-independent surfaces
    # ∂K/∂Ē = 0, so Partial and the true ∂P/∂Tₑ must agree EXACTLY, and `eig_fd`
    # becomes a genuine oracle for TWO of the three things Partial keeps: the
    # cold-target slope and the dilution rate. Mutation-checked — deleting either is
    # caught, at 43x and 5.0e5x tolerance respectively.
    #
    # **It does NOT cover the equilibration term, and an earlier version of this
    # comment claimed it did** on the grounds that `Coulomb_Collision = true` made it
    # "live rather than assumed". It is not a fixture problem and cannot be fixed by
    # one: `2μ(3/2)e·ν_ei` is 4.7e-18 here against a `scale` of 2.4e6 set by the
    # cold-target slope, so the term sits 24 orders below the tolerance and stays there
    # for any `ν_ei` — 10¹⁰ still leaves 18 orders. The mass ratio μ ≈ 2.7e-4 and the
    # `e` are what put it there.
    #
    # That term has its own item, which drives `ν_ei` directly at 1e8 and 1e10:
    # "eig_Te: the equilibration term is differentiated, not lumped in with ∂ν_ei/∂Tₑ".
    # Two tests, two regimes; neither pretends to be the other.
    # Both units, at their real magnitudes. With one value for K and L alike the energy
    # sinks land 20 orders above every particle term, `scale` below is set entirely by
    # the cold-target slope, and the dilution and equilibration terms this item exists
    # to cover become unfalsifiable — verified by mutation, not assumed.
    RP = with_surfaces(
        pj_RAPID(; coulomb = true, depth = PartialLinearResponse);
        K_of_Ē = _ -> 1.0e-15, L_of_Ē = _ -> 1.0e-35
    )
    RP.flags.scheme.atomic = ExpRB
    inw = RP.G.nodes.in_wall_nids

    analytic = eig_at(RP)
    numeric = eig_fd(RP)
    @test !all(iszero, analytic[inw])                   # or the comparison is vacuous
    scale = maximum(abs.(numeric[inw]))
    @test scale > 0.0
    @test maximum(abs.(analytic[inw] .- numeric[inw])) < 1.0e-6 * scale
end
