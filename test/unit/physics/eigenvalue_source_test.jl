@testsnippet EigSourceFixtures begin
    using RAPID2D: ExpRB, ForwardEuler, FrozenResponse, LinearResponse, EigenvalueSource,
        update_RRCs!, update_electron_heating_powers!, update_electron_power_jacobian!,
        update_ion_heating_powers!, update_ion_power_jacobian!,
        solve_electron_continuity_equation!, validate_scheme_flags, SimulationFlags

    function eig_RAPID(;
            Te_eV = 5.0, u_para = -1.0e5, EoverP = 100.0, pressure = 5.0e-3,
            coulomb = false, source = FrozenResponse
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
        RP.flags.diffu = false
        RP.flags.convec = false
        RP.flags.Include_Te_diffu_term = false
        RP.flags.Include_Te_convec_term = false
        RP.flags.Include_heat_flux_term = false
        RP.flags.exprb_eigenvalue = source
        initialize!(RP)
        RP.plasma.Te_eV .= Te_eV
        RP.plasma.Ti_eV .= 0.5 * Te_eV
        RP.plasma.ne .= 1.0e16
        RP.plasma.ni .= 1.0e16
        RP.plasma.ue_para .= u_para
        # P_drag reads ueR/ueϕ/ueZ, not ue_para.
        RP.plasma.ueR .= u_para .* RP.fields.bR
        RP.plasma.ueϕ .= u_para .* RP.fields.bϕ
        RP.plasma.ueZ .= u_para .* RP.fields.bZ
        ee = RP.config.constants.ee
        @. RP.fields.E_para_tot = -EoverP * RP.plasma.n_H2_gas * RP.plasma.T_gas_eV * ee
        if !coulomb
            RP.plasma.sptz_fac .= 0.0
            RP.plasma.ν_ei .= 0.0
        end
        return RP
    end

    function eig_Te_at(RP)
        RP.flags.scheme.atomic = ExpRB
        update_RRCs!(RP)
        update_electron_heating_powers!(RP)
        update_electron_power_jacobian!(RP)
        return copy(RP.plasma.exprb.eig_Te)
    end
end

@testitem "eigenvalue source: the default asks for nothing it cannot read off" setup = [EigSourceFixtures] begin
    using RAPID2D: FrozenResponse, LinearResponse, SimulationFlags

    # `FrozenResponse` uses the rate the model already states as `ν·y`; `LinearResponse`
    # adds `∂ν/∂y` from the tables. Both produce a rate — the axis is whether a
    # derivative was taken, not whether the answer is one.
    @test SimulationFlags{Float64}().exprb_eigenvalue === FrozenResponse
    @test FrozenResponse !== LinearResponse
end

@testitem "eigenvalue source: growth cannot tell the two apart" setup = [EigSourceFixtures] begin
    using RAPID2D: ExpRB, FrozenResponse, LinearResponse

    # `f = ν_iz·n` is EXACTLY linear in n and `∂ν_iz/∂n = 0`, so the stated rate
    # and the full derivative are the same number. That makes the continuity
    # equation the invariant this policy must not touch: bit for bit, not close.
    for source in (FrozenResponse, LinearResponse)
        RP = eig_RAPID(; source = source)
        RP.flags.scheme.growth = ExpRB
        RP.dt = 2.0e-5
        update_RRCs!(RP)
        solve_electron_continuity_equation!(RP)
        source === FrozenResponse && (global ref_ne = copy(RP.plasma.ne))
        source === FrozenResponse && (global ref_z = copy(RP.plasma.exprb.z_growth))
        source === FrozenResponse && (global ref_iz = copy(RP.reactions.counts.iz))
        if source === LinearResponse
            @test RP.plasma.ne == ref_ne
            @test RP.plasma.exprb.z_growth == ref_z
            @test RP.reactions.counts.iz == ref_iz
        end
    end
end

@testitem "eigenvalue source: FrozenResponse cannot produce a growth branch on Tₑ" setup = [EigSourceFixtures] begin
    using RAPID2D: FrozenResponse, LinearResponse

    # 𝔅 = (2mₑ/m_H₂)ν_ela(3/2)e + (3/2)e·ν_iz + 2μ(3/2)e·ν_ei is a sum of
    # non-negative rates, so −(2/3e)𝔅 ≤ 0 wherever the rates are. That is the
    # whole point of the default: no pole, no cap, no positive band, structurally.
    #
    # The band is real and LinearResponse finds it — measured +1.67e5 1/s at
    # Tₑ ≈ 0.3 eV with u∥ ≈ −4.5e5 — so this is a genuine difference, not a
    # tolerance.
    for Te in (0.05, 0.3, 1.0, 5.0), u in (-1.0e5, -4.5e5)
        known = eig_Te_at(eig_RAPID(; Te_eV = Te, u_para = u, source = FrozenResponse))
        inw = eig_RAPID().G.nodes.in_wall_nids
        @test all(<=(0), known[inw])
    end

    # …and at the band's peak the two genuinely disagree about the sign.
    lr = eig_Te_at(eig_RAPID(; Te_eV = 0.3, u_para = -4.5e5, source = LinearResponse))
    kr = eig_Te_at(eig_RAPID(; Te_eV = 0.3, u_para = -4.5e5, source = FrozenResponse))
    inw = eig_RAPID().G.nodes.in_wall_nids
    @test any(>(0), lr[inw])
    @test all(<=(0), kr[inw])
end

@testitem "eigenvalue source: FrozenResponse is the 𝔅 of the terms carrying an explicit Tₑ" setup = [EigSourceFixtures] begin
    using RAPID2D: FrozenResponse

    # Written out here from `update_electron_heating_powers!` rather than from the
    # implementation, so this catches a term that moves between the two.
    RP = eig_RAPID(; Te_eV = 3.0, coulomb = true, source = FrozenResponse)
    RP.plasma.ν_ei .= 1.0e8
    c = RP.config.constants
    eig = eig_Te_at(RP)

    pla = RP.plasma
    m_H2 = c.mi
    m_i = RAPID2D.bulk_ion_mass(RP)
    μ = m_i * c.me / (m_i + c.me)^2
    𝔅 = @. (2 * c.me / m_H2) * pla.ν_en_mom_ela * 1.5 * c.ee +   # P_ela
        1.5 * c.ee * pla.ν_en_iz +                                # P_dilution
        2 * μ * 1.5 * c.ee * pla.ν_ei                             # P_equi
    expected = @. -(2 / 3) * 𝔅 / c.ee

    inw = RP.G.nodes.in_wall_nids
    @test eig[inw] ≈ expected[inw] rtol = 1.0e-12
    @test !all(iszero, eig[inw])
end

@testitem "eigenvalue source: FrozenResponse never asks update_RRCs! for a derivative" setup = [EigSourceFixtures] begin
    using RAPID2D: ExpRB, FrozenResponse, LinearResponse

    # The derivative surfaces are the expensive half of the branch. Under
    # FrozenResponse nothing reads them, so nothing should pay to build them.
    kr = eig_RAPID(; source = FrozenResponse)
    kr.flags.scheme.atomic = ExpRB
    update_RRCs!(kr)
    @test all(iszero, kr.plasma.dν_dTe.iz)
    @test all(iszero, kr.plasma.dν_dTe.mom_tot)

    lr = eig_RAPID(; source = LinearResponse)
    lr.flags.scheme.atomic = ExpRB
    update_RRCs!(lr)
    @test !all(iszero, lr.plasma.dν_dTe.iz)
end

@testitem "eigenvalue source: only LinearResponse needs a differentiable rate" setup = [EigSourceFixtures] begin
    using RAPID2D: ExpRB, FrozenResponse, LinearResponse, SimulationFlags, validate_scheme_flags

    # The legacy rate paths are refused for LinearResponse because there is no
    # dK/dĒ to take. FrozenResponse takes no derivative at all, so the same pairing is
    # fine — refusing it would narrow what works for no reason.
    for (field, bad) in ((:Ionz_method, "Townsend_coeff"), (:ud_method, "Lloyd_fit"))
        lr = SimulationFlags{Float64}()
        lr.scheme.atomic = ExpRB
        lr.exprb_eigenvalue = LinearResponse
        setproperty!(lr, field, bad)
        @test_throws ArgumentError validate_scheme_flags(lr)

        kr = SimulationFlags{Float64}()
        kr.scheme.atomic = ExpRB
        kr.exprb_eigenvalue = FrozenResponse
        setproperty!(kr, field, bad)
        @test validate_scheme_flags(kr) === kr
    end
end

@testitem "eigenvalue source: LinearResponse is refused where no one implements it" setup = [EigSourceFixtures] begin
    using RAPID2D: ExpRB, LinearResponse, FrozenResponse, SimulationFlags, validate_scheme_flags

    # `update_ue_para!` uses λ = −ν_sum, which is the stated rate. Its linear
    # response is −(mₑu²/e)·∂ν/∂Ē — measured at up to 122 % of what is used, so
    # not negligible — and nothing computes it. Selecting LinearResponse there
    # would silently get FrozenResponse, which is the defect class this branch spent
    # its review fixing.
    lr = SimulationFlags{Float64}()
    lr.scheme.decay = ExpRB
    lr.exprb_eigenvalue = LinearResponse
    lr.Ampere = false                       # keep the coupled-solver refusal out of it
    err = try
        validate_scheme_flags(lr)
        nothing
    catch e
        sprint(showerror, e)
    end
    @test err !== nothing
    @test occursin("decay", err)
    @test occursin("update_ue_para!", err)

    # The same pairing under the default is exactly what runs today.
    kr = SimulationFlags{Float64}()
    kr.scheme.decay = ExpRB
    kr.Ampere = false
    @test kr.exprb_eigenvalue === FrozenResponse
    @test validate_scheme_flags(kr) === kr
end
