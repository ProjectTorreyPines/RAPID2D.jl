# The common mode of the parallel transport (the ambipolar closure). The pressure-driven parallel
# flow is carried once: by u∥ for the share 1 − w_amb, and for the share w_amb by the common-mode
# coefficient D_common, the drag-only D_a under the sound speed's ceilings, which the electrons and
# every ion species take alike, leaving the wall at the Bohm flux c_s n.
# internal/docs/src/notes/design/space-charge-layers-v4.md §5.

@testitem "common mode: the parallel pressure is carried once, at every coupling weight" setup = [CoulombReference] tags = [:physics] begin
    using RAPID2D: calculate_electron_acceleration_by_pressure, ue_pressure_acceleration
    RP = uniform_box([("H2", MH2, 1, 1.0e17)]; ne = 1.0e17, Te = 5.0, Ti = 1.0, dt = 1.0e-6)
    pla, tp, G = RP.plasma, RP.transport, RP.G
    @. pla.ne = 1.0e17 * (1 + 0.5 * G.Z2D)              # a gradient along the (vertical) field
    pla.ni .= pla.ne
    fill!(pla.ue_para, 1.0e5)                           # the acceleration limiter scales with u∥
    inw = G.nodes.in_wall_nids
    for w in (0.0, 0.5, 1.0)
        fill!(pla.ambipolar_weight, w)
        update_transport_quantities!(RP)                # the gradient is upwinded on the velocity this writes
        full = calculate_electron_acceleration_by_pressure(RP)
        @test any(!iszero, full[inw])
        # u∥ carries 1 − w of the pressure drive, the common mode w of it
        @test ue_pressure_acceleration(RP) ≈ (1 - w) .* full
        @test tp.Dpara[inw] ≈ (1 - w) .* tp.Dpara0 .+ w .* tp.Dpara_common[inw]
    end
end

@testitem "common mode: fully coupled, the electrons and one ion species share the parallel channel and its wall speed" setup = [CoulombReference] tags = [:physics] begin
    using RAPID2D: electron_wall_channels
    RP = uniform_box([("H2", MH2, 1, 1.0e17)]; ne = 1.0e17, Te = 5.0, Ti = 1.0, dt = 1.0e-6)
    pla, tp = RP.plasma, RP.transport
    fill!(pla.ambipolar_weight, 1.0)
    update_transport_quantities!(RP)
    inw = RP.G.nodes.in_wall_nids
    c_s = @. sqrt(EE * (pla.Te_eV + pla.Ti_eV) / pla.mi_mean)
    e_channels = electron_wall_channels(RP)
    i_channels = ion_transport_channels(RP, tp.ion_species[1], shared_turbulent_channel(RP))
    e_para, i_para = e_channels[1][1], i_channels[1]
    @test channel_D_para(e_para)[inw] ≈ tp.Dpara_common[inw]
    @test channel_D_para(i_para)[inw] ≈ tp.Dpara_common[inw]
    # both deliver the Bohm flux c_s n to the wall: the one-sided ¼v̄n of v̄ = 4c_s
    @test e_para.vm_para[inw] ≈ 4 .* c_s[inw]
    @test i_para.vm_para[inw] ≈ 4 .* c_s[inw]
    # the electrons' Bohm channel moves at the ions' speed
    @test e_channels[2][1].vm_perp[inw] ≈ i_channels[2].vm_perp[inw]
end

@testitem "common mode: finite without gas, under the sound speed's ceilings, and continuous as the coupling vanishes" setup = [CoulombReference] tags = [:physics] begin
    RP = uniform_box([("H2", MH2, 1, 1.0e17)]; ne = 1.0e17, Te = 5.0, Ti = 1.0, dt = 1.0e-6)
    pla, tp = RP.plasma, RP.transport
    pla.n_H2_gas .= 0.0
    fill!(pla.ambipolar_weight, 1.0)
    update_transport_quantities!(RP)
    inw = RP.G.nodes.in_wall_nids
    @test all(isinf, tp.Dpara_amb[inw])                  # nothing drags the common motion
    c_s = @. sqrt(EE * (pla.Te_eV + pla.Ti_eV) / pla.mi_mean)
    ceiling = RAPID2D.wall_step_ceiling.(c_s, RP.flf.Lc_forward, RP.flf.Lc_backward)
    @test all(isfinite, tp.Dpara_common[inw]) && all(tp.Dpara_common[inw] .<= ceiling[inw])
    # the coupling switched on by a hair moves the coefficients by a hair
    D_common = copy(tp.Dpara_common)
    fill!(pla.ambipolar_weight, 0.0)
    update_transport_quantities!(RP)
    D0 = copy(tp.Dpara)
    fill!(pla.ambipolar_weight, 1.0e-9)
    update_transport_quantities!(RP)
    @test tp.Dpara[inw] ≈ D0[inw] .+ 1.0e-9 .* (D_common[inw] .- D0[inw]) rtol = 1.0e-12
end

@testitem "common mode: the electrons conduct with their own coefficient whatever the coupling" setup = [CoulombReference] tags = [:physics] begin
    RP = uniform_box([("H2", MH2, 1, 1.0e17)]; ne = 1.0e17, Te = 5.0, Ti = 1.0, dt = 1.0e-6)
    pla, tp = RP.plasma, RP.transport
    inw = RP.G.nodes.in_wall_nids
    conduction(w) = (fill!(pla.ambipolar_weight, w); update_transport_quantities!(RP); copy(tp.Dpara_cond))
    D0, D1 = conduction(0.0), conduction(1.0)
    own = [k for k in inw if D0[k] >= tp.Dpara_common[k]]
    @test !isempty(own)
    @test D1[own] == D0[own]
end

@testitem "common mode: without the coupling the pressure is all u∥'s and each species diffuses on its own" setup = [CoulombReference] tags = [:physics] begin
    using RAPID2D: calculate_electron_acceleration_by_pressure, ue_pressure_acceleration, update_space_charge_weights!
    RP = uniform_box([("H2", MH2, 1, 1.0e17)]; ne = 1.0e17, Te = 5.0, Ti = 1.0, dt = 1.0e-6)
    pla, tp, G = RP.plasma, RP.transport, RP.G
    @. pla.ne = 1.0e17 * (1 + 0.5 * G.Z2D)
    pla.ni .= pla.ne
    fill!(pla.ue_para, 1.0e5)
    RP.flags.ambipolar_coupling = false
    update_space_charge_weights!(RP)
    update_transport_quantities!(RP)
    inw = G.nodes.in_wall_nids
    @test all(iszero, pla.ambipolar_weight)
    full = calculate_electron_acceleration_by_pressure(RP)
    @test any(!iszero, full[inw]) && ue_pressure_acceleration(RP) == full
    @test all(==(tp.Dpara0), tp.Dpara[inw])
    # the ions' own: the drag of the gas alone, under the thermal ceiling
    i_para = ion_transport_channels(RP, tp.ion_species[1], shared_turbulent_channel(RP))[1]
    vm = @. sqrt(8 * EE * pla.Ti_eV / (π * MH2))
    own = @. min(EE * pla.Ti_eV / (MH2 * tp.νi_neutral) + tp.Dpara0, RAPID2D.wall_step_ceiling(vm, RP.flf.Lc_forward, RP.flf.Lc_backward))
    @test channel_D_para(i_para)[inw] ≈ own[inw]
end

@testitem "common mode: the work its share of the pressure force would have done is booked, and a gate retry puts it back" setup = [CoulombReference] tags = [:physics] begin
    using RAPID2D: update_ue_para!, calculate_electron_acceleration_by_pressure, below_gate_entry_state,
        restore_below_gate_entry_state!, measure_snap0D!, Snapshot0D
    flags = SimulationFlags{Float64}(;
        src = false, Te_evolve = false, Ti_evolve = false, Gas_evolve = false, diffu = false, convec = false,
        charge_sheet_field = false, E_para_cancellation = false, E_para_induction = false, Ampere = false,
        Include_ud_convec_term = false, Include_ud_diffu_term = false,
    )
    RP = uniform_box([("H2", MH2, 1, 1.0e17)]; ne = 1.0e17, Te = 5.0, Ti = 1.0, dt = 1.0e-6, flags)
    pla, G = RP.plasma, RP.G
    @. pla.ne = 1.0e17 * (1 + 0.5 * G.Z2D)
    pla.ni .= pla.ne
    me = RP.config.constants.me
    function step(w)
        fill!(pla.ue_para, 1.0e5)
        fill!(pla.ambipolar_weight, w)
        update_transport_quantities!(RP)
        RP.fields.E_para_tot .= 0.0
        a_p = calculate_electron_acceleration_by_pressure(RP)
        u_old = copy(pla.ue_para)
        before = RP.diagnostics.ambipolar_exchange
        update_ue_para!(RP)
        booked = RP.diagnostics.ambipolar_exchange - before
        want = RP.dt * sum(G.inVol2D[k] * me * pla.ne[k] * w * a_p[k] * (u_old[k] + pla.ue_para[k]) / 2 for k in G.nodes.in_wall_nids)
        return booked, want
    end
    @test step(0.0) == (0.0, 0.0)
    for w in (0.5, 1.0)
        booked, want = step(w)
        @test booked != 0
        @test booked ≈ want rtol = 1.0e-12
    end
    # the snapshot reads the ledger; a step solved again above the gate puts it back
    snap = Snapshot0D{Float64}()
    measure_snap0D!(RP, snap)
    @test snap.ambipolar_exchange == RP.diagnostics.ambipolar_exchange
    entry = below_gate_entry_state(RP)
    RP.diagnostics.ambipolar_exchange += 1.0
    restore_below_gate_entry_state!(RP, entry)
    @test RP.diagnostics.ambipolar_exchange == snap.ambipolar_exchange
end

@testitem "common mode: the ∇Tₑ share of its drive is 0 at uniform Tₑ and 1 at uniform density" setup = [CoulombReference] tags = [:physics] begin
    using RAPID2D: ambipolar_gradTe_share
    RP = uniform_box([("H2", MH2, 1, 1.0e17)]; ne = 1.0e17, Te = 5.0, Ti = 1.0, dt = 1.0e-6)
    pla, G = RP.plasma, RP.G
    fill!(pla.ambipolar_weight, 1.0)
    @. pla.ne = 1.0e17 * (1 + 0.5 * G.Z2D)
    @test ambipolar_gradTe_share(RP) == 0
    pla.ne .= 1.0e17
    @. pla.Te_eV = 5.0 * (1 + 0.5 * G.Z2D)
    @test ambipolar_gradTe_share(RP) ≈ 1
    @. pla.ne = 1.0e17 * (1 + 0.5 * G.Z2D)
    @test 0 < ambipolar_gradTe_share(RP) < 1
    # nothing coupled, nothing to share
    fill!(pla.ambipolar_weight, 0.0)
    @test ambipolar_gradTe_share(RP) == 0
end

@testitem "common mode: its drive down ∇(Tₑ + Tᵢ) is a drift both species take, at most the sound speed" setup = [CoulombReference] tags = [:physics] begin
    using RAPID2D: calculate_para_grad_of_scalar_F, sound_speed
    RP = uniform_box([("H2", MH2, 1, 1.0e17)]; ne = 1.0e17, Te = 5.0, Ti = 1.0, dt = 1.0e-6)
    pla, tp, G, F = RP.plasma, RP.transport, RP.G, RP.fields
    inw = G.nodes.in_wall_nids
    @. pla.Te_eV = 5.0 * (1 + 0.5 * G.Z2D)              # a temperature gradient along the field
    fill!(pla.ambipolar_weight, 1.0)
    update_transport_quantities!(RP)
    ∇lnT = calculate_para_grad_of_scalar_F(RP, log.(pla.Te_eV .+ pla.Ti_eV); upwind = false)
    c_s = sound_speed.(pla.Te_eV, pla.Ti_eV, pla.mi_mean)
    @test tp.u_para_thermal[inw] ≈ (@. -clamp(tp.Dpara_common * ∇lnT, -c_s, c_s))[inw]
    @test any(!iszero, tp.u_para_thermal[inw])
    # the electrons and the ions move with it alike: no current
    @test pla.ueR .- pla.ue_para .* F.bR ≈ pla.uiR .- pla.ui_para .* F.bR
    @test pla.ueZ .- pla.ue_para .* F.bZ ≈ pla.uiZ .- pla.ui_para .* F.bZ
    # uncoupled, or isothermal, none
    fill!(pla.ambipolar_weight, 0.0)
    update_transport_quantities!(RP)
    @test all(iszero, tp.u_para_thermal)
    fill!(pla.ambipolar_weight, 1.0)
    fill!(pla.Te_eV, 5.0)
    update_transport_quantities!(RP)
    @test all(iszero, tp.u_para_thermal[inw])
end

@testitem "common mode: the pair's pressure and drag carry the electrons each ion brings, Z̄, and so does its sound speed at the wall" setup = [CoulombReference] tags = [:physics] begin
    using RAPID2D: electron_wall_channels
    MHE = 4.002602 * 1.6605390666e-27
    # He²⁺ alone, nₑ = 2nᵢ; then with H₂⁺, Z̄ = Σ zₛnₛ/Σ nₛ = 4/3 at the density-weighted mass
    for (ions, Z̄) in (([("He2", MHE, 2, 0.5e17)], 2.0), ([("H2", MH2, 1, 0.5e17), ("He2", MHE, 2, 0.25e17)], 4 / 3))
        RP = uniform_box(ions; ne = 1.0e17, Te = 5.0, Ti = 1.0, dt = 1.0e-6)
        pla, tp = RP.plasma, RP.transport
        fill!(pla.ambipolar_weight, 1.0)
        update_transport_quantities!(RP)
        inw = RP.G.nodes.in_wall_nids
        @test RAPID2D.mean_ion_charge(RP)[inw] ≈ fill(Z̄, length(inw))
        ν_en = pla.ν_en_mom_tot .+ pla.ν_en_iz_tot
        @test tp.Dpara_amb[inw] ≈ (@. EE * (Z̄ * pla.Te_eV + pla.Ti_eV) / (Z̄ * ME * ν_en + pla.mi_mean * tp.νi_neutral))[inw]
        c_s = @. sqrt(EE * (Z̄ * pla.Te_eV + pla.Ti_eV) / pla.mi_mean)
        e_para = electron_wall_channels(RP)[1][1]
        i_para = ion_transport_channels(RP, tp.ion_species[1], shared_turbulent_channel(RP))[1]
        @test e_para.vm_para[inw] ≈ 4 .* c_s[inw]
        @test i_para.vm_para[inw] ≈ 4 .* c_s[inw]
    end
end

@testitem "common mode: with one singly charged species the mean charge is exactly one" setup = [CoulombReference] tags = [:physics] begin
    RP = uniform_box([("H2", MH2, 1, 1.0e17)]; ne = 1.0e17, Te = 5.0, Ti = 1.0, dt = 1.0e-6)
    @test all(==(1.0), RAPID2D.mean_ion_charge(RP))
end

@testitem "Bohm: the electrons cross the field as the charge they balance, so the cross-field flux carries no current" setup = [CoulombReference] tags = [:physics] begin
    using RAPID2D: channel_D_perp
    MHE = 4.002602 * 1.6605390666e-27
    # He²⁺ alone: the electrons at the ions' D_B/2
    RP = uniform_box([("He2", MHE, 2, 0.5e17)]; ne = 1.0e17, Te = 5.0, Ti = 1.0, dt = 1.0e-6)
    update_transport_quantities!(RP)
    tp, pla, F = RP.transport, RP.plasma, RP.fields
    inw = RP.G.nodes.in_wall_nids
    bohm_ion = ion_transport_channels(RP, tp.ion_species[1], shared_turbulent_channel(RP))[2]
    @test tp.Dperp[inw] ≈ (tp.Dperp0 .+ channel_D_perp(bohm_ion))[inw]
    # H₂⁺ and He²⁺: Σ zₛnₛ (D_B/zₛ) / Σ zₛnₛ = D_B Σ nₛ/nₑ
    RP = uniform_box([("H2", MH2, 1, 0.5e17), ("He2", MHE, 2, 0.25e17)]; ne = 1.0e17, Te = 5.0, Ti = 1.0, dt = 1.0e-6)
    update_transport_quantities!(RP)
    tp, pla, F = RP.transport, RP.plasma, RP.fields
    D_B = @. abs(pla.Te_eV / (16 * F.Bϕ))
    @test tp.Dperp[inw] ≈ (tp.Dperp0 .+ 0.75 .* D_B)[inw]
    # without the per-charge 1/Z every species and the electrons take D_B
    RP.flags.bohm_charge_scaling = false
    update_transport_quantities!(RP)
    @test tp.Dperp[inw] ≈ (tp.Dperp0 .+ D_B)[inw]
end
