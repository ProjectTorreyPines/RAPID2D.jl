# Tₑ's and Tᵢ's compression, −(2/3) T ∇·u, split by its sign: expansion (∇·u > 0, a decay) backward Euler, compression
# (∇·u < 0, a growth) explicit, so the step's diagonal never falls below one and a converging flow heats instead of
# flipping the solution's sign:
#     (1 + (2/3) Δt (∇·u)₊) T⁺ = (1 − (2/3) Δt (∇·u)₋) Tⁿ.
# internal/docs/src/notes/issues/compression-implicit-sign-flip.md.

@testsnippet CompressionCase begin
    using RAPID2D: cache_electron_operators!, ue_Te_operators, update_electron_heating_powers!, update_Te!,
        update_ion_heating_powers!, update_Ti!, publish_electron_carriage!
    using RAPID2D.SparseArrays: sparse, spdiagm
    using RAPID2D.LinearAlgebra: I

    """
    The step a uniform T⁻ takes: the advection θ-weighted (zero on the uniform T⁻, so on the LHS only), the compression
    by its sign, (I + θ Δt A_adv + (2/3) Δt diag((∇·u)₊)) T⁺ = (1 − (2/3) Δt (∇·u)₋) T⁻.
    """
    function split_step(RP, T⁻)
        pops = ue_Te_operators(RP)
        d, dt, θ = vec(pops.div_u), RP.dt, RP.flags.θ_imp.transport
        L = I + θ * dt * sparse(pops.A_adv.matrix) + spdiagm(0 => (2 / 3) * dt .* max.(d, 0.0))
        return L \ ((1 .- (2 / 3) * dt .* min.(d, 0.0)) .* T⁻), d
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

    "One Tₑ stage; returns Tₑ before and after, the split step's own result and the compression ∇·u."
    function te_stage!(RP)
        T⁻ = copy(vec(RP.plasma.Te_eV))
        expected, div_u = split_step(RP, T⁻)
        update_electron_heating_powers!(RP)
        update_Te!(RP)
        return T⁻, copy(vec(RP.plasma.Te_eV)), expected, div_u
    end
end

@testitem "compression by its sign: a converging flow heats Tₑ explicitly, a diverging one cools it backward Euler" setup = [CompressionCase] begin
    for g in (2.0e4, -2.0e4)
        RP = compression_RP(; g)
        inw = RP.G.nodes.in_wall_nids
        T⁻, T⁺, expected, d = te_stage!(RP)
        @test sign(g) * sum(d[inw]) < 0
        @test T⁺[inw] ≈ expected[inw] rtol = 1.0e-10
        # node by node the compression alone, up to the advection of what it made non-uniform
        a = (2 / 3) * RP.dt .* d
        local_only = @. ifelse(d < 0, (1 - a) * T⁻, T⁻ / (1 + a))
        @test T⁺[inw] ≈ local_only[inw] rtol = 1.0e-5
    end
end

@testitem "compression by its sign: where (2/3) θ Δt |∇·u| exceeds one, Tₑ still heats and stays positive" setup = [CompressionCase] begin
    # g Δt = 6: the θ = ½ diagonal 1 + (2/3) θ Δt ∇·u would be −1, the old solve's sign flipped
    RP = compression_RP(; g = 6.0e6)
    inw = RP.G.nodes.in_wall_nids
    T⁻, T⁺, expected, d = te_stage!(RP)
    @test minimum(d[inw]) * (2 / 3) * 0.5 * RP.dt < -1
    @test T⁺[inw] ≈ expected[inw] rtol = 1.0e-10
    @test all(T⁺[inw] .> T⁻[inw])
    @test minimum(T⁺[inw]) > 2 * T⁻[inw[1]]
end

@testitem "compression by its sign: Tᵢ alike, carried by the electrons' flow when slaved" setup = [CompressionCase] begin
    for g in (2.0e4, -2.0e4, 6.0e6)
        RP = compression_RP(; g)
        inw = RP.G.nodes.in_wall_nids
        publish_electron_carriage!(RP)
        T⁻ = copy(vec(RP.plasma.Ti_eV))
        expected, _ = split_step(RP, T⁻)
        update_Ti!(RP)
        T⁺ = vec(RP.plasma.Ti_eV)
        @test T⁺[inw] ≈ expected[inw] rtol = 1.0e-10
        @test all(T⁺[inw] .> 0)
    end
end
