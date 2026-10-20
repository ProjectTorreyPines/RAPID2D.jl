# Rate coefficients by model (`rate_models.jl`): fits in a temperature, and the tables wrapped
# as `TabulatedRate`, each evaluated in the context of the colliding pair.

@testitem "A fit's rate inside its range is its formula" begin
    using RAPID2D: ConstantRate, ArrheniusRate, LogPolynomialRate, ElectronDriftContext,
        HeavyPairContext, rate_coefficient, rate_coefficient!

    T = [0.5, 1.0, 3.0, 10.0, 40.0]
    ctx = ElectronDriftContext(T, zeros(5))
    @test rate_coefficient(ConstantRate(2.0e-15), ctx) == fill(2.0e-15, 5)

    # element by element: the rates span twelve decades, which a norm-wise ≈ would not see
    arrhenius = ArrheniusRate(3.0e-14, 0.5, 12.0, (0.1, 100.0))
    @test all(isapprox.(rate_coefficient(arrhenius, ctx), @.(3.0e-14 * T^0.5 * exp(-12.0 / T)); rtol = 1.0e-12))

    coefficients = (-30.0, 2.0, -0.3, 0.01)
    fit = LogPolynomialRate(coefficients, (0.1, 100.0))
    lnT = log.(T)
    reference = @. exp(coefficients[1] + coefficients[2] * lnT + coefficients[3] * lnT^2 + coefficients[4] * lnT^3)
    @test all(isapprox.(rate_coefficient(fit, ctx), reference; rtol = 1.0e-12))

    # in place, the same numbers; a heavy pair reads its temperature the same way
    out = zeros(5)
    @test rate_coefficient!(out, fit, HeavyPairContext(T, zeros(5))) === out
    @test out == rate_coefficient(fit, ctx)
end

@testitem "Outside a fit's range the rate is frozen and its derivative zero" begin
    using RAPID2D: ArrheniusRate, LogPolynomialRate, ElectronDriftContext, rate_coefficient,
        rate_derivative!, derivative_support, HasDerivative

    T_lo, T_hi = 1.0, 50.0
    fits = (ArrheniusRate(3.0e-14, -0.5, 12.0, (T_lo, T_hi)), LogPolynomialRate((-30.0, 2.0, -0.3, 0.01), (T_lo, T_hi)))
    T = [0.0, 0.2, T_lo, 7.0, T_hi, 80.0, 1.0e4]      # T = 0 where Tₑ is damped outside the wall
    ctx = ElectronDriftContext(T, zeros(length(T)))
    for fit in fits
        K = rate_coefficient(fit, ctx)
        @test K[1] == K[2] == K[3]
        @test K[5] == K[6] == K[7]
        @test all(isfinite, K) && all(>(0), K)

        dK = rate_derivative!(similar(T), fit, ctx)
        @test dK[[1, 2, 6, 7]] == zeros(4)
        # inside, the slope of the rate itself
        for Tm in (2.0, 7.0, 30.0)
            h = 1.0e-6 * Tm
            around = ElectronDriftContext([Tm - h, Tm, Tm + h], zeros(3))
            K_around = rate_coefficient(fit, around)
            @test rate_derivative!(zeros(3), fit, around)[2] ≈ (K_around[3] - K_around[1]) / (2h) rtol = 1.0e-6
        end
        @test derivative_support(fit) === HasDerivative()
    end

    # a one-term log-polynomial is a constant, with no slope anywhere
    flat = LogPolynomialRate((-33.0,), (T_lo, T_hi))
    @test rate_coefficient(flat, ctx) ≈ fill(exp(-33.0), length(T))
    @test rate_derivative!(similar(T), flat, ctx) == zeros(length(T))

    @test_throws ArgumentError LogPolynomialRate((-30.0, 1.0), (2.0, 1.0))
    @test_throws ArgumentError ArrheniusRate(1.0e-14, 0.0, 0.0, (0.0, 10.0))
end

@testitem "A tabulated rate is its table's answer, in the context of the table's axes" begin
    using RAPID2D: RRC_EoverP_Erg, RRC_T_ud, RRC_T_ud_gFac, TabulatedRate, ElectronImpactContext,
        ElectronDriftContext, HeavyPairContext, rate_coefficient, rate_coefficient!, rate_derivative!,
        derivative_support, HasDerivative, NoDerivative

    EoverP = [1.0, 3.0, 10.0, 30.0, 100.0]
    Erg_eV = [0.1, 1.0, 10.0, 100.0]
    eop = RRC_EoverP_Erg(EoverP, Erg_eV, [p^0.3 * E^1.5 for p in EoverP, E in Erg_eV])
    # on knots, between them, and outside on each axis
    p = [1.0, 2.0, 30.0, 0.1, 500.0, 5.0, 5.0]
    E = [0.1, 0.5, 100.0, 1.0, 1.0, 0.01, 1.0e3]
    ctx = ElectronImpactContext(p, E)
    @test rate_coefficient(TabulatedRate(eop), ctx) == eop.itp((p, E))
    out = zeros(length(p))
    @test rate_coefficient!(out, TabulatedRate(eop), ctx) === out
    @test out == eop.itp((p, E))
    # differentiated at the query the caller has clamped onto the Ē axis
    Ec = clamp.(E, first(Erg_eV), last(Erg_eV))
    @test rate_derivative!(zeros(length(p)), TabulatedRate(eop), ElectronImpactContext(p, Ec)) == eop.dK_dĒ((p, Ec))

    T_eV = [0.5, 1.0, 5.0, 20.0]
    u = [0.0, 1.0e4, 1.0e5]
    tud = RRC_T_ud(T_eV, u, [t * (1 + v / 1.0e5) for t in T_eV, v in u])
    T = [0.5, 3.0, 20.0, 0.1, 50.0]
    v = [0.0, 5.0e3, 1.0e5, 2.0e5, 1.0e4]
    for pair in (ElectronDriftContext(T, v), HeavyPairContext(T, v))
        @test rate_coefficient(TabulatedRate(tud), pair) == tud.itp((T, v))
    end
    Tc = clamp.(T, first(T_eV), last(T_eV))
    @test rate_derivative!(zeros(5), TabulatedRate(tud), HeavyPairContext(Tc, v)) == tud.dK_dT((Tc, v))
    @test derivative_support(TabulatedRate(eop)) === derivative_support(TabulatedRate(tud)) === HasDerivative()

    # a table answers only in the context of its axes; a 3-D table has none yet
    @test_throws MethodError rate_coefficient(TabulatedRate(eop), ElectronDriftContext(T, v))
    @test_throws MethodError rate_coefficient(TabulatedRate(tud), ctx)
    g3 = RRC_T_ud_gFac(T_eV, u, [0.5, 1.0], rand(4, 3, 2))
    @test_throws MethodError rate_coefficient(TabulatedRate(g3), HeavyPairContext(T, v))
    @test derivative_support(TabulatedRate(g3)) === NoDerivative()
end

@testitem "The electron and ion queries are their tables' answers, with or without gas" begin
    using RAPID2D: get_electron_RRC, get_H2_ion_RRC, _eRRC_query_point

    config = SimulationConfig{Float64}(
        NR = 8, NZ = 8, R_min = 0.8, R_max = 2.2, Z_min = -1.2, Z_max = 1.2, dt = 1.0e-8, t_end_s = 1.0e-6,
        R0B0 = 1.0, prefilled_gas_pressure = 5.0e-3, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
        wall_R = [1.0, 2.0, 2.0, 1.0], wall_Z = [-1.0, -1.0, 1.0, 1.0], Output_path = mktempdir(; cleanup = false),
    )
    RP = RAPID{Float64}(config)
    RP.flags.Atomic_Collision = true
    initialize!(RP)
    pla = RP.plasma
    pla.Te_eV .= 5.0
    pla.ue_para .= -1.0e5
    pla.Ti_eV .= 1.0
    pla.ui_para .= 2.0e3
    RP.fields.E_para_tot .= -50.0
    pla.n_H2_gas[1:3, :] .= 0.0              # no gas: the query point's guard
    @test get_electron_RRC(RP, :K_iz) == RP.eRRCs.K_iz.itp(_eRRC_query_point(RP))
    @test get_electron_RRC(RP, :Halpha) == RP.eRRCs.Halpha.itp((pla.Te_eV, abs.(pla.ue_para)))
    @test get_H2_ion_RRC(RP, :Elastic) == RP.iRRCs.Elastic.itp((pla.Ti_eV, abs.(pla.ui_para)))
    @test all(isfinite, get_electron_RRC(RP, :K_mom))
end

@testitem "Rate evaluations are inferred, entry points and kernels" begin
    using RAPID2D: ConstantRate, ArrheniusRate, LogPolynomialRate, TabulatedRate, RRC_EoverP_Erg,
        ElectronDriftContext, ElectronImpactContext, rate_coefficient, rate_coefficient!, rate_derivative!,
        arrhenius_rate, arrhenius_slope, log_polynomial_rate, log_polynomial_slope

    T = [0.5, 3.0, 20.0]
    ctx = ElectronDriftContext(T, zeros(3))
    arrhenius = ArrheniusRate(3.0e-14, 0.5, 12.0, (0.1, 100.0))
    fit = LogPolynomialRate((-30.0, 2.0, -0.3), (0.1, 100.0))
    for m in (ConstantRate(1.0e-15), arrhenius, fit)
        @test @inferred(rate_coefficient(m, ctx)) isa Vector{Float64}
        @test @inferred(rate_coefficient!(zeros(3), m, ctx)) isa Vector{Float64}
        @test @inferred(rate_derivative!(zeros(3), m, ctx)) isa Vector{Float64}
    end
    @test @inferred(arrhenius_rate(2.0, 3.0e-14, 0.5, 12.0, 0.1, 100.0)) isa Float64
    @test @inferred(arrhenius_slope(2.0, 3.0e-14, 0.5, 12.0, 0.1, 100.0)) isa Float64
    @test @inferred(log_polynomial_rate(2.0, fit.coefficients, 0.1, 100.0)) isa Float64
    @test @inferred(log_polynomial_slope(2.0, fit.coefficients, (2.0, -0.6), 0.1, 100.0)) isa Float64

    eop = RRC_EoverP_Erg([1.0, 10.0, 100.0], [0.1, 1.0, 10.0], [p * E for p in [1.0, 10.0, 100.0], E in [0.1, 1.0, 10.0]])
    ectx = ElectronImpactContext([2.0, 20.0], [0.5, 5.0])
    @test @inferred(rate_coefficient(TabulatedRate(eop), ectx)) isa Vector{Float64}
    @test @inferred(rate_coefficient!(zeros(2), TabulatedRate(eop), ectx)) isa Vector{Float64}
    @test @inferred(rate_derivative!(zeros(2), TabulatedRate(eop), ectx)) isa Vector{Float64}
end
