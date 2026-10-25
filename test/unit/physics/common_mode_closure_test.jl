# The common mode of the parallel transport (the ambipolar closure). The pressure-driven parallel
# flow is carried once: by u∥ for the share 1 − w_amb, and for the share w_amb by the common-mode
# coefficient D_common, the drag-only D_a under the sound speed's ceilings, which the electrons and
# every ion species take alike, with the sound speed at the wall.
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
    @test e_para.vm_para[inw] ≈ c_s[inw] && i_para.vm_para[inw] ≈ c_s[inw]
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
