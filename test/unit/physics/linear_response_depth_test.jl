@testsnippet ResponseDepthFixtures begin
    using RAPID2D: ExpRB, ForwardEuler, PartialLinearResponse, FullLinearResponse, LinearResponseDepth,
        update_RRCs!, update_electron_heating_powers!, update_electron_power_jacobian!,
        update_ion_heating_powers!, update_ion_power_jacobian!,
        solve_electron_continuity_equation!, validate_scheme_flags, SimulationFlags

    function eig_RAPID(;
            Te_eV = 5.0, u_para = -1.0e5, EoverP = 100.0, pressure = 5.0e-3,
            coulomb = false, source = PartialLinearResponse
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

@testitem "linear-response depth: the default asks for nothing it cannot read off" setup = [ResponseDepthFixtures] begin
    using RAPID2D: PartialLinearResponse, FullLinearResponse, SimulationFlags

    # `PartialLinearResponse` uses the rate the model already states as `ν·y`; `FullLinearResponse`
    # adds `∂ν/∂y` from the tables. Both produce a rate — the axis is whether a
    # derivative was taken, not whether the answer is one.
    @test SimulationFlags{Float64}().exprb_eigenvalue === PartialLinearResponse
    @test PartialLinearResponse !== FullLinearResponse
end

@testitem "linear-response depth: growth cannot tell the two apart" setup = [ResponseDepthFixtures] begin
    using RAPID2D: ExpRB, PartialLinearResponse, FullLinearResponse

    # `f = ν_iz·n` is EXACTLY linear in n and `∂ν_iz/∂n = 0`, so the stated rate
    # and the full derivative are the same number. That makes the continuity
    # equation the invariant this policy must not touch: bit for bit, not close.
    for source in (PartialLinearResponse, FullLinearResponse)
        RP = eig_RAPID(; source = source)
        RP.flags.scheme.growth = ExpRB
        RP.dt = 2.0e-5
        update_RRCs!(RP)
        solve_electron_continuity_equation!(RP)
        source === PartialLinearResponse && (global ref_ne = copy(RP.plasma.ne))
        source === PartialLinearResponse && (global ref_z = copy(RP.plasma.exprb.z_growth))
        source === PartialLinearResponse && (global ref_iz = copy(RP.reactions.counts.iz))
        if source === FullLinearResponse
            @test RP.plasma.ne == ref_ne
            @test RP.plasma.exprb.z_growth == ref_z
            @test RP.reactions.counts.iz == ref_iz
        end
    end
end

@testitem "linear-response depth: PartialLinearResponse cannot produce a growth branch on Tₑ" setup = [ResponseDepthFixtures] begin
    using RAPID2D: PartialLinearResponse, FullLinearResponse

    # 𝔅 = (2mₑ/m_H₂)ν_ela(3/2)e + (3/2)e·ν_iz + 2μ(3/2)e·ν_ei is a sum of
    # non-negative rates, so −(2/3e)𝔅 ≤ 0 wherever the rates are. That is the whole
    # point of the default: no pole, no cap, no positive band, structurally. The
    # band FullLinearResponse finds below is real, so this is a genuine difference
    # and not a tolerance.
    for Te in (0.05, 0.3, 1.0, 5.0), u in (-1.0e5, -4.5e5)
        known = eig_Te_at(eig_RAPID(; Te_eV = Te, u_para = u, source = PartialLinearResponse))
        inw = eig_RAPID().G.nodes.in_wall_nids
        @test all(<=(0), known[inw])
    end

    # …and at the band's peak the two genuinely disagree about the sign.
    lr = eig_Te_at(eig_RAPID(; Te_eV = 0.3, u_para = -4.5e5, source = FullLinearResponse))
    kr = eig_Te_at(eig_RAPID(; Te_eV = 0.3, u_para = -4.5e5, source = PartialLinearResponse))
    inw = eig_RAPID().G.nodes.in_wall_nids
    @test any(>(0), lr[inw])
    @test all(<=(0), kr[inw])
end

@testitem "linear-response depth: PartialLinearResponse is the 𝔅 of the terms carrying an explicit Tₑ" setup = [ResponseDepthFixtures] begin
    using RAPID2D: PartialLinearResponse

    # Written out here from `update_electron_heating_powers!` rather than from the
    # implementation, so this catches a term that moves between the two.
    RP = eig_RAPID(; Te_eV = 3.0, coulomb = true, source = PartialLinearResponse)
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

@testitem "linear-response depth: PartialLinearResponse never asks update_RRCs! for a derivative" setup = [ResponseDepthFixtures] begin
    using RAPID2D: ExpRB, PartialLinearResponse, FullLinearResponse

    # The derivative surfaces are the expensive half of the branch. Under
    # PartialLinearResponse nothing reads them, so nothing should pay to build them.
    kr = eig_RAPID(; source = PartialLinearResponse)
    kr.flags.scheme.atomic = ExpRB
    update_RRCs!(kr)
    @test all(iszero, kr.plasma.dν_dTe.iz)
    @test all(iszero, kr.plasma.dν_dTe.mom_tot)

    lr = eig_RAPID(; source = FullLinearResponse)
    lr.flags.scheme.atomic = ExpRB
    update_RRCs!(lr)
    @test !all(iszero, lr.plasma.dν_dTe.iz)
end

@testitem "linear-response depth: the ion Jacobian's default branch is the stated rate alone" setup = [ResponseDepthFixtures] begin
    using RAPID2D: ExpRB, PartialLinearResponse, FullLinearResponse, update_RRCs!,
        update_ion_power_jacobian!, get_H2_ion_RRC, bulk_ion_charge, bulk_ion_mass

    # `P_atomic = ν_a·ΔE` splits by the product rule into `−ν_a·(3/2)e` (ΔE's own
    # T_i, written down in the model) and `∂ν_a/∂T_i·ΔE` (the tables). Partial keeps
    # the first and drops the second — so under Partial the ion eigenvalue must be
    # the collision frequency and nothing else, exactly.
    function ion_ready(source; coulomb = false)
        RP = eig_RAPID(; source = source, coulomb = coulomb)
        RP.flags.scheme.atomic = ExpRB
        RP.plasma.ui_para .= -2.0e4
        RP.plasma.uiR .= RP.plasma.ui_para .* RP.fields.bR
        RP.plasma.uiϕ .= RP.plasma.ui_para .* RP.fields.bϕ
        RP.plasma.uiZ .= RP.plasma.ui_para .* RP.fields.bZ
        coulomb && (RP.plasma.ν_ei .= 1.0e8)
        update_RRCs!(RP)
        update_ion_power_jacobian!(RP)
        return RP
    end

    for coulomb in (false, true)
        kr = ion_ready(PartialLinearResponse; coulomb = coulomb)
        pla, cnst = kr.plasma, kr.config.constants
        ee, me = cnst.ee, cnst.me
        mi, Z_i = bulk_ion_mass(kr), bulk_ion_charge(kr)
        K_ela, K_cx = get_H2_ion_RRC(kr, :Elastic), get_H2_ion_RRC(kr, :Charge_Exchange)

        ν_a = @. pla.n_H2_gas * (0.5 * K_ela + K_cx)
        @. ν_a += Z_i * pla.ν_en_iz                          # src is on in this fixture
        expected = @. -ν_a * 1.5 * ee
        if coulomb
            @. expected -= (2 * (mi * me / (mi + me)^2)) * 1.5 * ee * pla.ν_ei
        end
        @. expected *= 2 / 3 / ee

        inw = kr.G.nodes.in_wall_nids
        @test pla.exprb.eig_Ti[inw] ≈ expected[inw] rtol = 1.0e-12
        @test all(<(0), pla.exprb.eig_Ti[inw])            # a rate sum: it can only damp
    end

    # And the branch is a real fork, not dead code: with ΔE ≠ 0 and the ion tables
    # sloping, Full lands somewhere else.
    lr = ion_ready(FullLinearResponse)
    kr = ion_ready(PartialLinearResponse)
    inw = lr.G.nodes.in_wall_nids
    @test !all(≈(0), lr.plasma.exprb.eig_Ti[inw] .- kr.plasma.exprb.eig_Ti[inw])
end

@testitem "linear-response depth: only FullLinearResponse needs a differentiable rate" setup = [ResponseDepthFixtures] begin
    using RAPID2D: ExpRB, PartialLinearResponse, FullLinearResponse, SimulationFlags, validate_scheme_flags

    # The legacy rate paths are refused for FullLinearResponse because there is no
    # dK/dĒ to take. PartialLinearResponse takes no derivative at all, so the same pairing is
    # fine — refusing it would narrow what works for no reason.
    for (field, bad) in ((:Ionz_method, "Townsend_coeff"), (:ud_method, "Lloyd_fit"))
        lr = SimulationFlags{Float64}()
        lr.scheme.atomic = ExpRB
        lr.exprb_eigenvalue = FullLinearResponse
        setproperty!(lr, field, bad)
        @test_throws ArgumentError validate_scheme_flags(lr)

        kr = SimulationFlags{Float64}()
        kr.scheme.atomic = ExpRB
        kr.exprb_eigenvalue = PartialLinearResponse
        setproperty!(kr, field, bad)
        @test validate_scheme_flags(kr) === kr
    end
end

@testitem "linear-response depth: FullLinearResponse is refused where no one implements it" setup = [ResponseDepthFixtures] begin
    using RAPID2D: ExpRB, FullLinearResponse, PartialLinearResponse, SimulationFlags, validate_scheme_flags

    # `update_ue_para!` uses λ = −ν_sum, the stated rate. Its linear response
    # −(mₑu²/e)·∂ν/∂Ē is not computed anywhere, so selecting FullLinearResponse
    # there would silently deliver PartialLinearResponse — the defect class this
    # branch spent its review fixing.
    lr = SimulationFlags{Float64}()
    lr.scheme.decay = ExpRB
    lr.exprb_eigenvalue = FullLinearResponse
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
    @test kr.exprb_eigenvalue === PartialLinearResponse
    @test validate_scheme_flags(kr) === kr
end
