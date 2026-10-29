# Tₑ's and Tᵢ's compression, −(2/3) T ∇·u, the θ-step of its local rate node by node: θ where that step keeps a
# margin, the nearest weight that does elsewhere (x = (2/3) Δt |∇·u|; θₖ = min(θ, 1/(2x)) converging, max(θ, 1 − 1/(2x))
# expanding), so a converging flow heats instead of flipping the solution's sign and a fast expansion cools without
# going negative; a resolved rate keeps θ's step.
# internal/docs/src/notes/issues/compression-implicit-sign-flip.md.

@testsnippet CompressionCase begin
    using RAPID2D: cache_electron_operators!, ue_Te_operators, update_electron_heating_powers!, update_Te!,
        update_ion_heating_powers!, update_Ti!, publish_electron_carriage!
    using RAPID2D.SparseArrays: sparse, spdiagm
    using RAPID2D.LinearAlgebra: I

    "The weight of node k's compression: θ where its θ-step keeps a margin, the nearest weight that does elsewhere."
    capped_weight(d, θ, dt) = (x = (2 / 3) * dt * abs(d); d < 0 ? min(θ, 0.5 / x) : max(θ, 1 - 0.5 / x))

    """
    The step from T⁻ with the advection θ-weighted and the compression by `weight(d)`:
    (I + θ Δt A_adv + (2/3) Δt diag(θₖ ∇·u)) T⁺ = T⁻ − (1 − θ) Δt A_adv T⁻ − (2/3) Δt (1 − θₖ) ∇·u T⁻.
    """
    function theta_step(RP, T⁻; weight = d -> capped_weight(d, RP.flags.θ_imp.transport, RP.dt))
        pops = ue_Te_operators(RP)
        d, dt, θ = vec(pops.div_u), RP.dt, RP.flags.θ_imp.transport
        A = sparse(pops.A_adv.matrix)
        θk = weight.(d)
        L = I + θ * dt * A + spdiagm(0 => (2 / 3) * dt .* θk .* d)
        return L \ (T⁻ .- (1 - θ) * dt .* (A * T⁻) .- (2 / 3) * dt .* (1 .- θk) .* d .* T⁻), d
    end

    """
    Uniform electrons and slaved ions in a slab box (R ≈ 10 m), carried by the flow u = −(g/2)(R − 10, Z), nothing
    else acting on Tₑ and Tᵢ: ∇·u ≈ −g, so g > 0 converges.
    """
    function compression_RP(; g, dt = 1.0e-6, θ = 0.5, T0 = 20.0)
        config = SimulationConfig{Float64}(;
            device_Name = "manual", NR = 21, NZ = 21, R_min = 9.5, R_max = 10.5, Z_min = -0.5, Z_max = 0.5,
            wall_R = [9.55, 10.45, 10.45, 9.55], wall_Z = [-0.45, -0.45, 0.45, 0.45],
            prefilled_gas_pressure = 0.0, R0B0 = 10.0, dt, t_end_s = dt, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
        )
        config.Output_path = mktempdir(; cleanup = false)
        config.min_Te = 1.0e-3
        RP = RAPID{Float64}(config)
        RP.flags = SimulationFlags{Float64}(;
            Implicit = true, Atomic_Collision = false, Coulomb_Collision = false, src = false, diffu = false, convec = true,
            Te_evolve = true, Ti_evolve = true, ud_evolve = false, update_ni_independently = false, Ampere = false,
            Include_Te_convec_term = true, Include_Te_diffu_term = false, Include_heat_flux_term = false,
            mean_ExB = false, turb_ExB_mixing = false, charge_sheet_field = false, E_para_cancellation = false,
            E_para_induction = false, vacuum_relaxation = (state = false, factor = 1.0e2),
        )
        initialize!(RP)
        RP.flags.θ_imp.transport = θ
        G, pla = RP.G, RP.plasma
        pla.ne .= 0.0
        pla.ne[G.nodes.in_wall_nids] .= 1.0e17
        pla.ni .= pla.ne
        pla.Te_eV .= T0
        pla.Ti_eV .= T0
        pla.ue_para .= 0.0
        pla.ueR .= -(g / 2) .* (G.R2D .- 10.0)
        pla.ueZ .= -(g / 2) .* G.Z2D
        pla.uiR .= pla.ueR
        pla.uiZ .= pla.ueZ
        cache_electron_operators!(RP)
        return RP
    end

    "One Tₑ stage; returns Tₑ before and after, the θ-step's own result and the compression ∇·u."
    function te_stage!(RP)
        T⁻ = copy(vec(RP.plasma.Te_eV))
        expected, div_u = theta_step(RP, T⁻)
        update_electron_heating_powers!(RP)
        update_Te!(RP)
        return T⁻, copy(vec(RP.plasma.Te_eV)), expected, div_u
    end
end

@testitem "compression, a resolved rate: θ's own step for a converging and a diverging flow, a non-uniform Tₑ too" setup = [CompressionCase] begin
    for g in (2.0e4, -2.0e4), uniform in (true, false)
        RP = compression_RP(; g)
        inw = RP.G.nodes.in_wall_nids
        uniform || (RP.plasma.Te_eV .*= 1 .+ 0.3 .* sin.(4 .* (RP.G.R2D .- 10)) .* cos.(3 .* RP.G.Z2D))
        T⁻, T⁺, expected, d = te_stage!(RP)
        @test sign(g) * sum(d[inw]) < 0
        # every rate resolved: the weight is θ itself
        θ = RP.flags.θ_imp.transport
        @test all(k -> capped_weight(d[k], θ, RP.dt) == θ, inw)
        @test T⁺[inw] ≈ expected[inw] rtol = 1.0e-9
        @test T⁺[inw] ≈ theta_step(RP, T⁻; weight = _ -> θ)[1][inw] rtol = 1.0e-9
    end
end

@testitem "compression beyond a resolved rate: a converging flow heats and an expanding one cools, Tₑ positive both ways" setup = [CompressionCase] begin
    # g Δt = 6: θ = ½ would give a diagonal of −1 converging, a numerator of −1 expanding
    for g in (6.0e6, -6.0e6)
        RP = compression_RP(; g)
        inw = RP.G.nodes.in_wall_nids
        T⁻, T⁺, expected, d = te_stage!(RP)
        @test maximum(abs, d[inw]) * (2 / 3) * 0.5 * RP.dt > 1
        @test T⁺[inw] ≈ expected[inw] rtol = 1.0e-9
        @test all(T⁺[inw] .> 0)
        g > 0 ? @test(all(T⁺[inw] .> T⁻[inw])) : @test(all(T⁺[inw] .< T⁻[inw]))
    end
end

@testitem "compression: Tᵢ alike, carried by the electrons' flow when slaved" setup = [CompressionCase] begin
    for g in (2.0e4, -2.0e4, 6.0e6, -6.0e6)
        RP = compression_RP(; g)
        inw = RP.G.nodes.in_wall_nids
        publish_electron_carriage!(RP)
        T⁻ = copy(vec(RP.plasma.Ti_eV))
        expected, _ = theta_step(RP, T⁻)
        update_Ti!(RP)
        T⁺ = vec(RP.plasma.Ti_eV)
        @test T⁺[inw] ≈ expected[inw] rtol = 1.0e-9
        @test all(T⁺[inw] .> 0)
    end
end
