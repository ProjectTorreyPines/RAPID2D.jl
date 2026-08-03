# The ion–ion Coulomb logarithm.
#
# `ν_ii` is the bulk self-collision rate that every ion transport channel scales
# from (`ion_transport.jl`), so its Coulomb logarithm sets the whole per-species
# D∥ ladder. On master it borrowed `plasma.lnΛ` — the ELECTRON–ion logarithm,
# which in the branch that fires during burn-through is built from `ne` and `Te`.
# With Te ≫ Ti that is not a small error: the log carries T^(-3/2) inside it.
#
# NRL Plasma Formulary 2023, p.34(c):
#
#   λ_ii' = 23 − ln[ (ZZ'(μ+μ')/(μT' + μ'T)) · (nZ²/T + n'Z'²/T')^(1/2) ]
#
# with n in cm⁻³ and T in eV.

@testitem "Ion–ion lnΛ reproduces NRL p.34(c) term by term" begin
    using RAPID2D: ion_ion_coulomb_log

    # Written out longhand, in NRL's own units, so the test fails if any factor
    # inside the log is dropped or misplaced.
    nrl(n_cm3, T, Z, μ, n′_cm3, T′, Z′, μ′) =
        23.0 - log(
        (Z * Z′ * (μ + μ′) / (μ * T′ + μ′ * T)) *
            sqrt(n_cm3 * Z^2 / T + n′_cm3 * Z′^2 / T′)
    )

    for (n, T, Z, μ, n′, T′, Z′, μ′) in (
            (1.0e18, 1.0, 1, 2.0, 1.0e18, 1.0, 1, 2.0),      # H₂⁺ on itself
            (1.0e18, 1.0, 1, 2.0, 1.0e16, 5.0, 6, 12.0),     # C⁶⁺ trace in H₂⁺
            (5.0e19, 12.0, 2, 4.0, 3.0e18, 3.0, 1, 1.0),     # He²⁺ / H⁺
        )
        @test ion_ion_coulomb_log(n, T, Z, μ, n′, T′, Z′, μ′) ≈
            nrl(n * 1.0e-6, T, Z, μ, n′ * 1.0e-6, T′, Z′, μ′) rtol = 1.0e-12
    end
end

@testitem "Ion–ion lnΛ: the self-collision method is the two-species one at i=i′" begin
    using RAPID2D: ion_ion_coulomb_log

    for (n, T, Z, μ) in ((1.0e18, 1.0, 1, 2.0), (1.0e15, 0.3, 6, 12.0))
        @test ion_ion_coulomb_log(n, T, Z, μ) ≈
            ion_ion_coulomb_log(n, T, Z, μ, n, T, Z, μ) rtol = 1.0e-12
        # …and that reduction is 23 − ln(√2·Z³·√n[cm⁻³]·T^(-3/2))
        @test ion_ion_coulomb_log(n, T, Z, μ) ≈
            23.0 - log(sqrt(2.0) * Z^3 * sqrt(n * 1.0e-6) * T^(-1.5)) rtol = 1.0e-12
    end
end

@testitem "Ion–ion lnΛ is symmetric under swapping the two species" begin
    using RAPID2D: ion_ion_coulomb_log

    a = (1.0e18, 1.0, 1, 2.0)
    b = (1.0e16, 5.0, 6, 12.0)
    @test ion_ion_coulomb_log(a..., b...) ≈ ion_ion_coulomb_log(b..., a...) rtol = 1.0e-12
end

@testitem "Ion–ion lnΛ stays physical where the Coulomb picture breaks" begin
    using RAPID2D: ion_ion_coulomb_log

    # Cold and dense enough that the raw NRL expression goes NEGATIVE. A negative
    # logarithm makes ν_ii negative, which is an anti-collision — the run does not
    # merely lose accuracy, it inverts. Floor it instead.
    raw = 23.0 - log(sqrt(2.0) * 1.0^3 * sqrt(1.0e21 * 1.0e-6) * 0.026^(-1.5))
    @test raw < 1.0                                     # the regime exists
    @test ion_ion_coulomb_log(1.0e21, 0.026, 1, 2.0) ≥ 1.0

    # Degenerate inputs must not propagate: a continuity solve can hand back n ≤ 0.
    @test isfinite(ion_ion_coulomb_log(0.0, 1.0, 1, 2.0))
    @test isfinite(ion_ion_coulomb_log(-1.0e-40, 1.0, 1, 2.0))
    @test isfinite(ion_ion_coulomb_log(1.0e18, 0.0, 1, 2.0))
end

@testitem "ν_ii is built from the ion–ion logarithm, not the electron one" begin
    using RAPID2D: ion_ion_coulomb_log

    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 7, NZ = 7,
        R_min = 1.0, R_max = 2.0, Z_min = -0.5, Z_max = 0.5,
        wall_R = [1.15, 1.85, 1.85, 1.15], wall_Z = [-0.35, -0.35, 0.35, 0.35],
        prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-8,
        t_end_s = 1.0e-6, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)

    ne, Te, Ti = 1.0e18, 10.0, 1.0
    RP.plasma.ne .= ne
    RP.plasma.ni .= ne
    RP.plasma.Te_eV .= Te
    RP.plasma.Ti_eV .= Ti
    update_coulomb_collision_parameters!(RP)

    mi, mp = RP.config.constants.mi, RP.config.constants.mp
    μ = mi / mp

    # The new field holds NRL p.34(c) for the bulk species colliding with itself
    expected = ion_ion_coulomb_log(ne, Ti, 1, μ)
    @test all(≈(expected; rtol = 1.0e-12), RP.plasma.lnΛ_ii)

    # Te ≫ Ti ⇒ the two logarithms genuinely differ; the old code used the wrong one
    @test !isapprox(RP.plasma.lnΛ_ii[4, 4], RP.plasma.lnΛ[4, 4]; rtol = 1.0e-3)
    @test RP.plasma.lnΛ_ii[4, 4] < RP.plasma.lnΛ[4, 4]   # hotter T inside a −ln

    # ν_ii must now carry it (NRL p.28 self-collision rate)
    ν_expected = 4.8e-8 * 1.0^4 * μ^(-0.5) * (ne * 1.0e-6) * expected * Ti^(-1.5)
    @test all(≈(ν_expected; rtol = 1.0e-12), RP.plasma.ν_ii)

    # ν_ei is an ELECTRON–ion rate and must keep the electron logarithm
    @test all(isfinite, RP.plasma.ν_ei)
    @test all(>(0.0), RP.plasma.ν_ei)
end

@testitem "Correcting ν_ii changes the ion parallel diffusivity it feeds" begin
    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 7, NZ = 7,
        R_min = 1.0, R_max = 2.0, Z_min = -0.5, Z_max = 0.5,
        wall_R = [1.15, 1.85, 1.85, 1.15], wall_Z = [-0.35, -0.35, 0.35, 0.35],
        prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-8,
        t_end_s = 1.0e-6, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)
    RP.flags.Coulomb_Collision = true
    RP.plasma.ne .= 1.0e18
    RP.plasma.ni .= 1.0e18
    RP.plasma.Te_eV .= 10.0
    RP.plasma.Ti_eV .= 1.0
    update_transport_quantities!(RP)

    # The correction is not cosmetic: with Te = 10 eV and Ti = 1 eV the electron
    # logarithm overstates ν_ii by tens of percent, and ν_ii is what the shared
    # D∥ is built from.
    ratio = RP.plasma.lnΛ[4, 4] / RP.plasma.lnΛ_ii[4, 4]
    @test ratio > 1.2
    @test all(>(0.0), RP.transport.νi_coulomb)
    @test RP.transport.νi_coulomb[4, 4] == RP.plasma.ν_ii[4, 4]
end
