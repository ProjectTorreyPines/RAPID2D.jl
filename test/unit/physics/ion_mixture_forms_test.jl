# The ion species as a mixture (`update_charge_states!` and its consumers): the sums the
# current, the collisions and the masses read, with a toy species written straight into the
# transport buffers beside the bulk H2[1] (a second declared species is still refused).

@testsnippet MixtureCase begin
    "The bulk H2[1] at `n_bulk`, and a toy species of charge `z_toy`, mass 6 mᵢ, at `n_toy`."
    function mixture_case(; n_bulk = 1.0e18, n_toy = 1.0e17, z_toy = 6, ne = n_bulk + z_toy * n_toy, Te = 100.0)
        config = SimulationConfig{Float64}(;
            device_Name = "manual", NR = 9, NZ = 9, R_min = 1.0, R_max = 2.0, Z_min = -0.5, Z_max = 0.5,
            wall_R = [1.15, 1.85, 1.85, 1.15], wall_Z = [-0.35, -0.35, 0.35, 0.35],
            prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-8, t_end_s = 1.0e-6,
            snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0, Output_path = mktempdir(; cleanup = false),
        )
        RP = RAPID{Float64}(config)
        initialize!(RP)
        pla, tp = RP.plasma, RP.transport
        pla.ne .= ne
        pla.ni .= n_bulk
        pla.Te_eV .= Te
        pla.Ti_eV .= 1.0
        mi = RP.config.constants.mi
        tp.ion_species = [tp.ion_species[1], RAPID2D.IonRecord(:toy, 6mi, z_toy)]
        tp.ion_N = zeros(RP.G.NR * RP.G.NZ, 2)
        tp.ion_S = zeros(RP.G.NR * RP.G.NZ, 2)
        tp.ion_N[:, 2] .= n_toy
        empty!(tp.ion_solvers)
        return RP
    end
end

@testitem "The species are summed node by node, and their means weight by density" setup = [MixtureCase] begin
    RP = mixture_case()
    update_charge_states!(RP)
    pla, mi = RP.plasma, RP.config.constants.mi
    @test all(≈(1.1e18), pla.ni_sum)
    @test all(≈(1.6e18), pla.Zni_sum)          # 1e18 + 6 × 1e17
    @test all(≈(4.6e18), pla.Z2ni_sum)         # 1e18 + 36 × 1e17
    @test all(≈(mi * 1.0e18 + 6mi * 1.0e17), pla.mni_sum)
    @test all(≈((mi * 1.0e18 + 6mi * 1.0e17) / 1.1e18), pla.mi_mean)
    @test all(≈(4.6 / 1.6), pla.Zeff)

    # one species: its own mass and charge, exactly
    RP1 = mixture_case(n_toy = 0.0)
    RP1.transport.ion_species = RP1.transport.ion_species[1:1]
    update_charge_states!(RP1)
    @test all(==(mi), RP1.plasma.mi_mean) && all(==(1.0), RP1.plasma.Zeff)
    @test RP1.plasma.Zni_sum == RP1.plasma.ni
end

@testitem "Where there are no ions the means are the first species'" setup = [MixtureCase] begin
    RP = mixture_case(n_bulk = 0.0, n_toy = 0.0)
    update_charge_states!(RP)
    pla = RP.plasma
    @test all(iszero, pla.ni_sum) && all(iszero, pla.Zni_sum) && all(iszero, pla.Z2ni_sum) && all(iszero, pla.mni_sum)
    @test all(==(RP.config.constants.mi), pla.mi_mean)
    @test all(==(1.0), pla.Zeff)
end

@testitem "The current carries the charge of every species" setup = [MixtureCase] begin
    RP = mixture_case()
    pla, F, c = RP.plasma, RP.fields, RP.config.constants
    pla.ue_para .= -1.0e5
    pla.ui_para .= 2.0e3
    RAPID2D.update_Jϕ!(RP)
    @test pla.Jϕ ≈ @.((c.qe * pla.ne * pla.ue_para + c.ee * 1.6e18 * pla.ui_para) * F.bϕ)
end

@testitem "Electrons scatter on Σ z² n, and Spitzer takes the composition" setup = [MixtureCase] begin
    # Te = 100 eV is above 10 Zeff², where lnΛ carries no charge, so the two runs share it
    mix = mixture_case()
    bulk = mixture_case(n_toy = 0.0, ne = 1.6e18)
    foreach(update_coulomb_collision_parameters!, (mix, bulk))
    @test mix.plasma.lnΛ == bulk.plasma.lnΛ
    @test mix.plasma.ν_ei ≈ 4.6 .* bulk.plasma.ν_ei rtol = 1.0e-12
    Z = 4.6 / 1.6
    @test all(≈((1 + 1.198Z + 0.222Z^2) / (1 + 2.966Z + 0.753Z^2)), mix.plasma.sptz_fac)
end

@testitem "The ion–ion rate is the mean ion's own" setup = [MixtureCase] begin
    RP = mixture_case()
    update_coulomb_collision_parameters!(RP)
    pla, mi, mp = RP.plasma, RP.config.constants.mi, RP.config.constants.mp
    Z2_mean = (1.0e18 * 1 + 1.0e17 * 36) / 1.1e18
    m_mean = (mi * 1.0e18 + 6mi * 1.0e17) / 1.1e18
    lnΛ_ii = RAPID2D.ion_ion_coulomb_log(1.1e18, 1.0, sqrt(Z2_mean), m_mean / mp)
    @test all(≈(lnΛ_ii), pla.lnΛ_ii)
    @test all(≈(4.8e-8 * Z2_mean^2 * (m_mean / mp)^(-0.5) * 1.1e18 * 1.0e-6 * lnΛ_ii), pla.ν_ii)
end

@testitem "A lone species of any charge is its own Coulomb reference" setup = [IonRun] begin
    using RAPID2D: IonRecord, set_ion_species!, ion_transport_channels, shared_turbulent_channel, channel_D_para
    # the channel's rate is ν_ii scaled by the test-particle ratio against the mean ion: 1 for
    # a lone species, its charge included, so D∥ is Einstein against ν_ii itself
    RP = ion_case()
    RP.flags.Atomic_Collision = false
    mi, ee = RP.config.constants.mi, RP.config.constants.ee
    set_ion_species!(RP, [IonRecord(:C6, 6mi, 6)])
    update_transport_quantities!(RP)
    ch = ion_transport_channels(RP, RP.transport.ion_species[1], shared_turbulent_channel(RP))[1]
    inw = RP.G.nodes.in_wall_nids
    want = @. RP.plasma.Ti_eV * ee / (6mi * RP.transport.νi_coulomb)
    @test channel_D_para(ch)[inw] ≈ want[inw] rtol = 1.0e-12
end

@testitem "A species with no inventory does not shape the operator where no ion is" begin
    using RAPID2D: DiffusionChannel, mixture_channel
    ch1 = DiffusionChannel(fill(2.0, 2, 2), fill(3.0, 2, 2), fill(4.0, 2, 2), fill(5.0, 2, 2); vm_para = fill(2.0, 2, 2), vm_perp = fill(4.0, 2, 2))
    ch2 = DiffusionChannel(fill(6.0, 2, 2), fill(1.0, 2, 2), fill(8.0, 2, 2), fill(1.0, 2, 2); vm_para = fill(6.0, 2, 2), vm_perp = fill(8.0, 2, 2))
    # species 1 is present at one node; species 2 nowhere: the empty nodes take species 1
    w1 = [1.0 0.0; 0.0 0.0]
    mix = mixture_channel([ch1, ch2], [w1, zeros(2, 2)])
    @test all(mix.v_para .≈ 2.0) && all(mix.v_perp .≈ 4.0)
    @test all(mix.λ_para .≈ 3.0) && all(mix.λ_perp .≈ 5.0)
    # with neither present anywhere, both
    both = mixture_channel([ch1, ch2], [zeros(2, 2), zeros(2, 2)])
    @test all(both.v_para .≈ 4.0)
end

@testitem "The e–i friction is an exchange: momentum per volume is kept exactly, whatever the scheme" begin
    using RAPID2D: ExpRB, Theta
    # Coulomb the only channel, no drive: the ions take the opposite of the impulse the
    # electrons recorded, so mₑnₑuₑ + ρuᵢ is kept to round-off under θ and ExpRB, and with
    # nₑ ≠ nᵢ, and both velocities meet at the centre of mass
    function relax(scheme; θ = 1.0, ne = 1.0e18, ni = 1.0e18)
        config = SimulationConfig{Float64}(NR = 12, NZ = 12, prefilled_gas_pressure = 5.0e-3, R0B0 = 1.0, dt = 1.0e-9)
        RP = RAPID{Float64}(config)
        RP.flags = SimulationFlags{Float64}(
            ud_evolve = true, Implicit = true, Atomic_Collision = false, Coulomb_Collision = true,
            Spitzer_Resistivity = true, src = false, Te_evolve = false, Ti_evolve = false, Gas_evolve = false,
            diffu = false, convec = false, Ampere = false, E_para_self_ES = false, E_para_self_EM = false,
            update_ni_independently = false, Include_ud_convec_term = false, Include_ud_pressure_term = false,
            Include_ud_diffu_term = false,
        )
        RP.flags.scheme.decay = scheme
        RP.flags.θ_imp.decay = θ
        initialize!(RP)
        RP.plasma.ne .= ne
        RP.plasma.ni .= ni
        RP.plasma.Te_eV .= 10.0
        RP.plasma.Ti_eV .= 1.0
        RAPID2D.update_transport_quantities!(RP)
        RP.fields.E_para_tot .= 0.0
        inw = RP.G.nodes.in_wall_nids
        RP.dt = 1 / (sum(RP.plasma.ν_ei_eff[inw]) / length(inw))
        me, mi = RP.config.constants.me, RP.config.constants.mi
        ue0, ui0 = 1.0e5, 0.0
        RP.plasma.ue_para .= ue0
        RP.plasma.ui_para .= ui0
        for _ in 1:60
            RAPID2D.update_ue_para!(RP)
            RAPID2D.update_ui_para!(RP)
        end
        ue = sum(RP.plasma.ue_para[inw]) / length(inw)
        ui = sum(RP.plasma.ui_para[inw]) / length(inw)
        p0 = me * ne * ue0 + mi * ni * ui0
        u_cm = p0 / (me * ne + mi * ni)
        return (drift = (me * ne * ue + mi * ni * ui) / p0 - 1, ue = ue / u_cm, ui = ui / u_cm)
    end
    for (scheme, θ) in ((Theta, 1.0), (Theta, 0.5), (ExpRB, 1.0)), (ne, ni) in ((1.0e18, 1.0e18), (1.3e18, 1.0e18))
        r = relax(scheme; θ, ne, ni)
        @test abs(r.drift) < 1.0e-13
        @test r.ue ≈ 1 rtol = 1.0e-8
        @test r.ui ≈ 1 rtol = 1.0e-8
    end
end

@testitem "Ions born at rest dilute the drift by their mass, without atomic collisions" setup = [MixtureCase] begin
    using RAPID2D: update_ui_para!, reset_reaction_counts!
    # the born mass Δρ = Σ mₛ ΔNₛ (this step's counts) keeps ρ uᵢ: u → u (1 − Δρ/ρ), with ρ
    # the density after the step; nothing else acts here (no field, no drag, no friction)
    RP = mixture_case(n_toy = 0.0)
    RP.flags.Atomic_Collision = false
    RP.flags.Coulomb_Collision = false
    RP.flags.src = true
    RAPID2D.update_charge_states!(RP)
    pla, rx, mi = RP.plasma, RP.reactions, RP.config.constants.mi
    RP.fields.E_para_tot .= 0.0
    pla.ui_para .= 2.0e3
    reset_reaction_counts!(RP)
    rx.counts.iz .= 2.0e17                 # 20 % of the ions are this step's births…
    rx.counts.diz .= 0.5e17                # …and DI's ion is booked to H2[1] too
    union!(rx.published, (:iz, :diz))
    update_ui_para!(RP)
    @test pla.ui_para ≈ fill(2.0e3 * (1 - 2.5e17 / 1.0e18), size(pla.ui_para))
    # the toy species is unregistered: no reaction makes it, so it does not dilute
    @test RAPID2D.born_ion_mass(RP) ≈ fill(mi * 2.5e17, size(pla.ni))
end
