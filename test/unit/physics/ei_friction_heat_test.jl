# The heat of the e–i friction: the electrons get mₑ ν_ei^eff |u_e − u_i|², Braginskii's −R·(u_e − u_i).
#
# Two terms in the electrons' energy make it: −u_e·R, their own kinetic energy the friction removes,
# and +u_i·R, the work of the moving ions (elastic scattering off a heavy ion keeps |v − u_i|, so the
# collisions change the electrons' energy at u_i·R, zero only for ions at rest). The ions take the
# friction's work on their kinetic energy and no heat, to order mₑ/mᵢ.
# internal/docs/src/notes/issues/ei-friction-heat-partition.md.

@testsnippet EiFrictionCase begin
    "A uniform plasma in a box, electrons and ions drifting apart along B, no field, no gas unless asked."
    function friction_case(; dt = 1.0e-8, t_end_s = 1.0e-6, atomic = false, ue = 1.0e6, ui = 5.0e4, n = 1.0e18, T = 10.0)
        config = SimulationConfig{Float64}(;
            device_Name = "manual", NR = 9, NZ = 9, R_min = 1.0, R_max = 2.0, Z_min = -0.5, Z_max = 0.5,
            wall_R = [1.15, 1.85, 1.85, 1.15], wall_Z = [-0.35, -0.35, 0.35, 0.35],
            prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = dt, t_end_s = t_end_s,
            snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0, Output_path = mktempdir(),
        )
        config.manual.Eϕ = 0.0
        RP = RAPID{Float64}(config)
        RP.flags = SimulationFlags{Float64}(
            Implicit = true, diffu = false, convec = false, src = false,
            Atomic_Collision = atomic, Coulomb_Collision = true, Spitzer_Resistivity = true,
            mean_ExB = false, turb_ExB_mixing = false, charge_sheet_field = false, E_para_cancellation = false, E_para_induction = false,
            Ampere = false, Te_evolve = true, ud_evolve = true, Ti_evolve = true, Gas_evolve = false,
            update_ni_independently = true, secondary_electron = false, negative_n_correction = false,
            Include_ud_pressure_term = false, Include_ud_convec_term = false, Include_ud_diffu_term = false,
            Include_Te_convec_term = false, Include_Te_diffu_term = false, Include_heat_flux_term = false,
        )
        initialize!(RP)
        G, pla = RP.G, RP.plasma
        inw = G.nodes.in_wall_nids
        pla.ne .= 0.0
        vec(pla.ne)[inw] .= n
        pla.ni .= pla.ne
        pla.Te_eV .= T
        pla.Ti_eV .= T
        pla.ue_para .= ue
        pla.ui_para .= ui
        update_transport_quantities!(RP)
        return RP
    end

    "Σ V (½ mₑ nₑ u_e² + ½ ρ u_i² + (3/2) e (nₑ Tₑ + n_I Tᵢ)) over the in-wall nodes."
    function pair_energy(RP)
        pla, G = RP.plasma, RP.G
        inw = G.nodes.in_wall_nids
        me, ee = RP.config.constants.me, RP.config.constants.ee
        V = vec(G.inVol2D)[inw]
        e = 0.5 .* me .* vec(pla.ne)[inw] .* vec(pla.ue_para)[inw] .^ 2 .+
            0.5 .* vec(pla.mni_tot)[inw] .* vec(pla.ui_para)[inw] .^ 2 .+
            1.5 .* ee .* (vec(pla.ne)[inw] .* vec(pla.Te_eV)[inw] .+ vec(pla.ni_tot)[inw] .* vec(pla.Ti_eV)[inw])
        return sum(V .* e)
    end
end

@testitem "the e–i friction heats the electrons by the kinetic energy it took from the relative drift: −I_ei·(ū_e − u_i), the same in every frame" setup = [EiFrictionCase] begin
    using RAPID2D: update_electron_heating_powers!, update_ue_para!
    RP = friction_case()
    pla = RP.plasma
    inw = RP.G.nodes.in_wall_nids
    me, dt = RP.config.constants.me, RP.dt
    Δu0 = copy(pla.ue_para .- pla.ui_para)
    ue0 = copy(pla.ue_para)
    update_ue_para!(RP)
    update_electron_heating_powers!(RP)
    Δu1 = pla.ue_para .- pla.ui_para
    # the friction's work on the relative drift over the step, per electron (no field, no gas)
    expected = @. -me * pla.Rue_ei * (0.5 * (ue0 + pla.ue_para) - pla.ui_para)
    @test pla.ePowers.drag[inw] ≈ expected[inw] rtol = 1.0e-12
    @test all(>(0), pla.ePowers.drag[inw])
    # at ν Δt ≪ 1 it is Braginskii's mₑ ν_ei^eff |u_e − u_i|², the electrons' −u_e·R plus the ions' work u_i·R
    @test maximum(pla.ν_ei_eff[inw]) * dt < 2.0e-2
    @test pla.ePowers.drag[inw] ≈ (me .* pla.ν_ei_eff .* Δu0 .^ 2)[inw] rtol = 3.0e-2
    # and it is exactly the drift's kinetic energy lost: ½ mₑ (Δu₀² − Δu₁²) per electron, for ions much heavier
    @test (pla.ePowers.drag .* dt)[inw] ≈ (0.5 .* me .* (Δu0 .^ 2 .- Δu1 .^ 2))[inw] rtol = 1.0e-3
    P0 = copy(pla.ePowers.drag)
    # the same relative drift in another frame: the same heat, where −u_e·R alone would cool
    RF = friction_case(; ue = 1.0e6 - 2.0e6, ui = 5.0e4 - 2.0e6)
    update_ue_para!(RF)
    update_electron_heating_powers!(RF)
    @test RF.plasma.ePowers.drag[inw] ≈ P0[inw] rtol = 1.0e-9
    # u_e not evolved: the prescribed drift's friction, mₑ ν_ei^eff |u_e − u_i|², at once
    RP.flags.ud_evolve = false
    update_electron_heating_powers!(RP)
    rel_sq = @. (pla.ueR - pla.uiR)^2 + (pla.ueϕ - pla.uiϕ)^2 + (pla.ueZ - pla.uiZ)^2
    @test pla.ePowers.drag[inw] ≈ (me .* rel_sq .* pla.ν_ei_eff)[inw] rtol = 1.0e-12
end

@testitem "the friction's heat with the gas on, and without Coulomb collisions" setup = [EiFrictionCase] begin
    using RAPID2D: update_electron_heating_powers!, update_ue_para!
    RG = friction_case(; atomic = true)
    plg = RG.plasma
    inw = RG.G.nodes.in_wall_nids
    me = RG.config.constants.me
    ue0 = copy(plg.ue_para)
    update_ue_para!(RG)
    update_electron_heating_powers!(RG)
    ue_sq = @. plg.ueR^2 + plg.ueϕ^2 + plg.ueZ^2
    friction = @. -me * plg.Rue_ei * (0.5 * (ue0 + plg.ue_para) - plg.ui_para)
    # the gas, at rest, heats by mₑ|u_e|²ν_mom beside the friction's
    @test plg.ePowers.drag[inw] ≈ (me .* ue_sq .* plg.ν_en_mom_tot .+ friction)[inw] rtol = 1.0e-12
    # without Coulomb collisions no friction heat, whatever rate initialization left
    RG.flags.Coulomb_Collision = false
    update_ue_para!(RG)
    update_electron_heating_powers!(RG)
    @test plg.ePowers.drag[inw] ≈ (me .* ue_sq .* plg.ν_en_mom_tot)[inw] rtol = 1.0e-12
end

@testitem "a stiff friction (ν Δt ≫ 1) gives the electrons the drift's kinetic energy, not ν Δt times it" setup = [EiFrictionCase] begin
    using RAPID2D: update_electron_heating_powers!, update_ue_para!
    # cold and dense: ν_ei Δt ~ 1e4. The relative drift relaxes within the step; the heat is what it held
    RP = friction_case(; n = 5.0e17, T = 2.0e-2, dt = 2.0e-5, ue = 2.0e5, ui = 0.0)
    pla = RP.plasma
    inw = RP.G.nodes.in_wall_nids
    me, dt = RP.config.constants.me, RP.dt
    @test minimum(pla.ν_ei_eff[inw]) * dt > 1.0e3
    Δu0 = copy(pla.ue_para .- pla.ui_para)
    update_ue_para!(RP)
    update_electron_heating_powers!(RP)
    Δu1 = pla.ue_para .- pla.ui_para
    @test all(abs.(Δu1[inw]) .< 1.0e-2 .* abs.(Δu0[inw]))
    @test (pla.ePowers.drag .* dt)[inw] ≈ (0.5 .* me .* (Δu0 .^ 2 .- Δu1 .^ 2))[inw] rtol = 1.0e-3
end

@testitem "an isolated electron–ion pair keeps its energy: the friction's kinetic energy is the electrons' heat, at every Δt" setup = [EiFrictionCase] begin
    # no gas, no field, no transport: the friction and the temperature exchange alone. The heat is the
    # kinetic energy the friction took over each step, −I_ei·(ū_e − u_i), so the total is kept at any
    # Δt (to the ions' share, mₑ/mᵢ of it). With −u_e·R alone the pair would gain mₑ ν (u_e − u_i) u_i.
    t_end = 3.0e-6
    errs = Float64[]
    for dt in (2.0e-8, 1.0e-8, 5.0e-9)
        RP = friction_case(; dt = dt, t_end_s = t_end)
        E0 = pair_energy(RP)
        u0 = copy(RP.plasma.ue_para)
        run_simulation!(RP)
        inw = RP.G.nodes.in_wall_nids
        @test all(abs.(vec(RP.plasma.ue_para .- RP.plasma.ui_para)[inw]) .< 0.5 .* abs.(vec(u0 .- 5.0e4)[inw]))   # relaxed
        push!(errs, abs(pair_energy(RP) - E0) / E0)
    end
    # the heat is the friction's work over each step: the pair keeps its energy at every Δt
    @test all(<(1.0e-6), errs)
end

@testitem "a cold dense plasma keeps a positive e–i Coulomb logarithm: lnΛ ≥ 1, so the friction heats and the exchange relaxes" setup = [EiFrictionCase] begin
    using RAPID2D: update_electron_heating_powers!, ei_relaxation_rate!
    # nₑ = 5e17 m⁻³ at the 1 meV floor: the raw NRL expression is −0.8, which made ν_ei negative, the
    # friction cool and the e–i exchange run away from equilibrium (KSTAR at 58.9 ms, 2026-10-28)
    RP = friction_case(; n = 5.0e17, T = 1.0e-3)
    pla = RP.plasma
    inw = RP.G.nodes.in_wall_nids
    @test all(>=(1.0), pla.lnΛ[inw])
    @test all(>(0.0), pla.ν_ei[inw]) && all(>(0.0), pla.ν_ei_eff[inw])
    RAPID2D.update_ue_para!(RP)
    update_electron_heating_powers!(RP)
    @test all(>(0.0), pla.ePowers.drag[inw])
    a = ei_relaxation_rate!(similar(pla.ne), RP)
    @test all(>(0.0), a[inw])
    # the relaxation stays between the two temperatures
    pla.Te_eV .= 5.0
    pla.Ti_eV .= 1.0e-3
    RAPID2D.apply_ei_exchange!(RP)
    @test all(T -> 1.0e-3 <= T <= 5.0, pla.Te_eV[inw]) && all(T -> 1.0e-3 <= T <= 5.0, pla.Ti_eV[inw])
end
