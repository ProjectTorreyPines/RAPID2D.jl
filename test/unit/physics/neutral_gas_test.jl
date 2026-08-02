# Kinetic diffusivity of the neutral H2 fill.
#
# The mean free path competes three ways: gas-gas elastic collisions, destruction
# by electron-impact ionization, and — once the gas is thin enough that a molecule
# would otherwise cross the vessel unimpeded — the wall itself. All three are
# summed as rates (Matthiessen), so whichever is shortest wins.

@testitem "Neutral gas diffusivity: elastic mean free path scales as 1/n" begin
    using RAPID2D: neutral_gas_diffusivity

    # D = ½·v_th·λ at fixed T, so D inherits λ's density scaling exactly.
    # Inf characteristic length disables the wall term to isolate the gas term.
    D1 = neutral_gas_diffusivity(1.0e18, 0.026, 0.0, Inf)
    D2 = neutral_gas_diffusivity(5.0e17, 0.026, 0.0, Inf)
    @test D2 / D1 ≈ 2.0 rtol = 1.0e-12
end

@testitem "Neutral gas diffusivity: D scales as sqrt(T_gas)" begin
    using RAPID2D: neutral_gas_diffusivity

    # λ = 1/(√2·n·σ) carries no temperature, so all of D's T-dependence comes from
    # v_th ∝ √T. This is the assertion that fails for the MATLAB formulation, where
    # D is written as D_NTP·n_NTP/n and v_th cancels out of the elastic limit
    # entirely — leaving 273 K hard-coded. See claudedocs/impurity_model_equations_v2.md.
    Da = neutral_gas_diffusivity(1.0e18, 0.026, 0.0, Inf)
    Db = neutral_gas_diffusivity(1.0e18, 4 * 0.026, 0.0, Inf)
    @test Db / Da ≈ 2.0 rtol = 1.0e-12
end

@testitem "Neutral gas diffusivity: ionization shortens the mean free path" begin
    using RAPID2D: neutral_gas_diffusivity, neutral_gas_thermal_speed

    # A molecule destroyed by electron impact never completes its free path, so
    # ν_iz adds to the collision rate. This is the mechanism behind neutral
    # shielding: where the plasma is hot, the gas cannot penetrate.
    n, T, νiz = 6.4e17, 0.026, 1.0e4
    D0 = neutral_gas_diffusivity(n, T, 0.0, Inf)
    Diz = neutral_gas_diffusivity(n, T, νiz, Inf)
    @test Diz < D0

    # exact combination law, without hard-coding σ: recover ν_elastic from D0
    vth = neutral_gas_thermal_speed(T)
    ν_el = vth^2 / (2 * D0)
    @test Diz ≈ D0 * ν_el / (ν_el + νiz) rtol = 1.0e-12
end

@testitem "Neutral gas diffusivity: never transports faster than free streaming" begin
    using RAPID2D: neutral_gas_diffusivity, neutral_gas_thermal_speed

    # At breakdown pressures the gas-gas mean free path is METRES — larger than the
    # vessel — so the flow is free-molecular, not diffusive. Left uncapped, D grows
    # without bound and the diffusive crossing time L²/D drops BELOW the ballistic
    # floor L/v_th, which nothing can beat. Adding the wall as a collision channel
    # (1/λ += 1/L, i.e. Knudsen diffusion) restores the correct limit.
    L, T = 1.09, 0.026
    vth = neutral_gas_thermal_speed(T)
    for n in (1.0e15, 1.0e17, 6.4e17, 1.0e19)
        D = neutral_gas_diffusivity(n, T, 0.0, L)
        @test L^2 / D ≥ L / vth
    end

    # in the collisionless limit the wall alone sets the path
    @test neutral_gas_diffusivity(1.0e8, T, 0.0, L) ≈ 0.5 * vth * L rtol = 1.0e-4
end

@testitem "Neutral gas diffusivity: stays finite where the gas is fully burnt" begin
    using RAPID2D: neutral_gas_diffusivity, neutral_gas_thermal_speed

    # n_gas → 0 makes the elastic path infinite. The wall term keeps λ finite, so
    # no special-casing (and no NaN scrubbing) is needed on burnt-out cells.
    L, T = 1.09, 0.026
    D = neutral_gas_diffusivity(0.0, T, 0.0, L)
    @test isfinite(D)
    @test D ≈ 0.5 * neutral_gas_thermal_speed(T) * L rtol = 1.0e-12
end
