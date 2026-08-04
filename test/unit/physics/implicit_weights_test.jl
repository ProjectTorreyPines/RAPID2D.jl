@testitem "θ_imp: one weight per operator character, with the default each needs" begin
    using RAPID2D: ImplicitWeights

    # The θ of the θ-scheme is not one number. Families of terms want different
    # weights, and what decides which is the SIGN of the eigenvalue. Before this
    # struct the code said so in three incompatible ways: a shared
    # `Implicit_weight`, a hand-rolled `θ_gas`, and a hardcoded `1.0` inline.
    w = ImplicitWeights{Float64}()

    # λ < 0 and well-resolved: second-order accuracy is worth having. CN.
    @test w.transport == 0.5

    # λ > 0. A-stability does not apply — the true solution grows, so |g| > 1 is
    # correct — and what matters is that g = (1+(1−θ)z)/(1−θz) stay positive and
    # finite. BE's pole sits at Δtν = 1 against CN's at 2, and BE is first-order
    # where CN is second, so for a GROWTH term the usual argument reverses. CN.
    @test w.growth == 0.5

    # λ < 0 and stiff. CN is A-stable but not L-stable: g → −1 rather than 0 as
    # |λ|Δt grows, so stiff modes ring instead of damping, and a friction-driven
    # equation rings about its saturated value instead of landing on it. BE.
    @test w.decay == 1.0
    @test w.gas == 1.0

    @test ImplicitWeights{Float32}().transport isa Float32
end

@testitem "θ_imp: BE beats CN on decay, CN beats BE on growth" begin
    using RAPID2D: ImplicitWeights

    # The amplification factor of the scalar problem ḟ = λf, which is the whole
    # reason the families are split by sign rather than by which equation they
    # appear in.
    g(θ, z) = (1 + (1 - θ) * z) / (1 - θ * z)
    CN, BE, FE = 0.5, 1.0, 0.0

    # DECAY (z < 0): BE damps, CN rings. This is L-stability.
    @test g(BE, -1.0e6) ≈ 0.0 atol = 1.0e-5
    @test g(CN, -1.0e6) ≈ -1.0 atol = 1.0e-5
    @test 0 < g(BE, -100.0) < 1                 # positive and monotone
    @test g(CN, -100.0) < 0                     # sign flip every step

    # GROWTH (z > 0): the reverse. BE's pole is at z = 1, CN's at z = 2, so
    # between them BE has already gone negative while CN is still tracking.
    @test g(BE, 1.5) < 0
    @test g(CN, 1.5) > 0
    # and CN is the more accurate of the two below the poles, from the correct side
    for z in (0.05, 0.1, 0.5)
        @test abs(g(CN, z) - exp(z)) < abs(g(BE, z) - exp(z))
        @test g(CN, z) > exp(z)                 # both overshoot...
        @test g(BE, z) > g(CN, z)               # ...BE by more
    end
    # FE is the only unconditionally positive one, at the cost of always undershooting
    @test g(FE, 5.0) > 0
    @test g(FE, 0.5) < exp(0.5)
end

@testitem "θ_imp: a weight outside [0, 1] is rejected where it is set" begin
    using RAPID2D: ImplicitWeights

    # 0 = forward Euler, ½ = Crank-Nicolson, 1 = backward Euler. Outside that the
    # scheme is not a θ-scheme, and the failure it produces downstream (a growing
    # mode, a negative density) says nothing about where it came from.
    @test_throws ArgumentError ImplicitWeights{Float64}(transport = 1.5)
    @test_throws ArgumentError ImplicitWeights{Float64}(growth = -0.1)
    @test_throws ArgumentError ImplicitWeights{Float64}(gas = NaN)

    w = ImplicitWeights{Float64}()
    @test_throws ArgumentError w.decay = 2.0
    @test w.decay == 1.0             # and the rejected write left nothing behind

    # the endpoints themselves are legal
    @test ImplicitWeights{Float64}(transport = 0.0).transport == 0.0
    @test ImplicitWeights{Float64}(transport = 1.0).transport == 1.0
end

@testitem "θ_imp: each family is tunable one at a time" begin
    using RAPID2D

    RP = RAPID{Float64}(SimulationConfig{Float64}(; NR = 5, NZ = 5, dt = 1.0e-6))

    # The point of the struct: one place to reach, and reaching it for one family
    # leaves the others alone. Before, moving transport off CN also moved every
    # atomic rate with it, because they shared `Implicit_weight`.
    RP.flags.θ_imp.transport = 1.0
    @test RP.flags.θ_imp.transport == 1.0
    @test RP.flags.θ_imp.growth == 0.5
    @test RP.flags.θ_imp.decay == 1.0
    @test RP.flags.θ_imp.gas == 1.0

    RP.flags.θ_imp.growth = 1.0
    @test RP.flags.θ_imp.transport == 1.0
    @test RP.flags.θ_imp.growth == 1.0
end

@testitem "Ionization creates one ion per electron, at every θ_growth" begin
    using RAPID2D

    # One ionization event makes one electron AND one ion. That identity is exact
    # in the continuous equations, so it must survive discretization — but the two
    # species are advanced by separate solves, and nothing enforces that they
    # weight the same source the same way.
    #
    # The electron equation carries ν_iz inside its θ-scheme, gaining
    #     Δt·ν·[(1−θ)nₑⁿ + θnₑⁿ⁺¹]
    # while the ion equation takes ν_iz as a plain explicit source built from
    # whatever `pla.ne` holds when it runs — which is nₑⁿ⁺¹, because the electron
    # solve went first. At θ = 1 those coincide by accident. At θ < 1 they do not,
    # and the ion gains more than the electron by θ(Δtν)²n every step: a
    # systematic ni excess that grows exactly where the avalanche is fastest.
    for θ_growth in (1.0, 0.5, 0.0)
        config = SimulationConfig{Float64}(;
            device_Name = "manual", NR = 7, NZ = 7,
            R_min = 1.0, R_max = 2.0, Z_min = -0.5, Z_max = 0.5,
            wall_R = [1.15, 1.85, 1.85, 1.15], wall_Z = [-0.35, -0.35, 0.35, 0.35],
            prefilled_gas_pressure = 4.0e-2, R0B0 = 1.0, dt = 1.0e-5,
            t_end_s = 5.0e-5, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
        )
        RP = RAPID{Float64}(config)
        initialize!(RP)

        # Source only: no transport, no wall, nothing that can make the two
        # species differ for any reason except how they weight ionization.
        RP.flags.diffu = false
        RP.flags.convec = false
        RP.flags.src = true
        RP.flags.update_ni_independently = true
        RP.flags.Implicit = true
        RP.flags.θ_imp.growth = θ_growth

        RP.plasma.ne .= 1.0e14
        RP.plasma.ni .= 1.0e14
        RP.plasma.Te_eV .= 5.0
        RP.plasma.Ti_eV .= 1.0
        # Δt·ν ≈ 0.5 — the regime the mismatch actually lives in. A rate small
        # enough to be safe is also small enough to hide the defect.
        RP.plasma.ν_en_iz .= 5.0e4

        for _ in 1:5
            RAPID2D.solve_electron_continuity_equation!(RP)
            RAPID2D.solve_ion_continuity_equation!(RP)
            RP.plasma.ν_en_iz .= 5.0e4     # hold the rate; only the scheme is under test
        end

        @test RP.plasma.ne ≈ RP.plasma.ni rtol = 1.0e-12
    end
end
