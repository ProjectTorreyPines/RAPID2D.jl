# The species registry and the reaction catalog (`species_registry.jl`, `reaction_catalog.jl`,
# `hydrogen_network.jl`): which reactions and collisions a species list switches on, the checks
# a reaction passes, how rates carry across hydrogen isotopes, an element's ladder from its
# parent's methods, and the HYDHEL fits against the values the database prints.

@testitem "The species list decides which hydrogen reactions and collisions are active" begin
    using RAPID2D: assemble_species_system, load_electron_RRCs, load_H2_Ion_RRCs, PlasmaConstants
    using RAPID2D.Particles: H, H2, H3

    eRRCs, iRRCs, constants = load_electron_RRCs(), load_H2_Ion_RRCs(), PlasmaConstants{Float64}()
    assemble(list) = assemble_species_system(list, Protium(), constants, eRRCs, iRRCs)
    reaction_labels(sys) = Set(a.reaction.label for a in sys.reactions)
    collision_labels(sys) = Set(a.collision.label for a in sys.collisions)

    # the default list: eight reactions, the H₃⁺ ones off for lack of H3[1], nothing to warn of
    default = assemble((H2[0], H[0], H2[1], H[1]))
    @test reaction_labels(default) == Set([:iz, :diz, :dex, :de2, :di2, :dr2, :izH0, :recH1])
    @test collision_labels(default) == Set([:cxH, :elH2, :cxH2])
    @test default.report.inactive == [label => [Symbol("H3[1]")] for label in (:h3, :dr3a, :dr3b, :de3)]
    @test isempty(default.report.warnings)

    # H3[1] listed: its formation and destruction join, and nothing is off
    with_H3 = assemble((H2[0], H[0], H2[1], H[1], H3[1]))
    @test reaction_labels(with_H3) == union(reaction_labels(default), Set([:h3, :dr3a, :dr3b, :de3]))
    @test isempty(with_H3.report.inactive)

    # a default species left out: every reaction and collision involving it goes, with a warning
    without_H1 = assemble((H2[0], H[0], H2[1]))
    @test reaction_labels(without_H1) == Set([:iz, :dex, :dr2])
    @test collision_labels(without_H1) == Set([:elH2, :cxH2])
    @test any(w -> startswith(w, "diz ") && endswith(w, "lacks H[1]"), without_H1.report.warnings)
    @test !any(w -> startswith(w, "h3 "), without_H1.report.warnings)   # H3[1] is not a default

    # H2[1] alone: none of its reactions has all its species, and nothing makes electrons
    alone = assemble((H2[1],))
    @test isempty(alone.reactions)
    @test "no active reaction makes electrons" in alone.report.warnings

    # registry: ions and neutrals apart, columns in list order
    reg = default.registry
    @test [r.label for r in reg.ions] == [Symbol("H2[1]"), Symbol("H[1]")]
    @test [r.label for r in reg.neutrals] == [Symbol("H2[0]"), Symbol("H[0]")]
    @test reg.ion_column[RAPID2D.species_key(H[1])] == 2
    @test reg.neutral_column[RAPID2D.species_key(H2[0])] == 1
end

@testitem "A reaction conserves charge and nuclei, and its producer fits its reactants" begin
    using RAPID2D: Reaction, ElectronSolve, LocalNetwork, RateData, SameForIsotopes, ConstantRate,
        EventEnergy, CapturedElectron, build_registry, assemble_species_system, load_electron_RRCs,
        load_H2_Ion_RRCs, PlasmaConstants, Ion
    using RAPID2D.Particles: H, H2, H3, electron

    rate = RateData(ConstantRate(1.0e-14), SameForIsotopes(), "test")
    energy = EventEnergy(1.0e-18)
    network(sp) = LocalNetwork(implicit = sp)

    # balanced, then broken one way at a time
    @test Reaction(:ok, (electron => 1, H2[1] => 1) => (electron => 2, H[1] => 2); rate, producer = network(H2[1]), energy) isa Reaction
    @test_throws ArgumentError Reaction(:charge, (electron => 1, H2[0] => 1) => (electron => 1, H2[1] => 1); rate, producer = ElectronSolve(), energy)
    @test_throws ArgumentError Reaction(:nuclei, (electron => 1, H2[1] => 1) => (electron => 1, H[1] => 1); rate, producer = network(H2[1]), energy)
    @test_throws ArgumentError Reaction(:ternary, (electron => 2, H[1] => 1) => (electron => 1, H[0] => 1); rate, producer = network(H[1]), energy)
    @test_throws ArgumentError Reaction(:count, (electron => 1, H2[1] => 1) => (electron => 0, H[1] => 2); rate, producer = network(H2[1]), energy)
    @test_throws ArgumentError Reaction(:twice, (electron => 1, H[1] => 1) => (H[0] => 1, H[0] => 1); rate, producer = network(H[1]), energy)

    # the producer: the electron solve counts impact on a neutral; the network is implicit in a
    # reactant ion, and holds a heavy pair's other reactant frozen
    @test_throws ArgumentError Reaction(:ion, (electron => 1, H2[1] => 1) => (electron => 2, H[1] => 2); rate, producer = ElectronSolve(), energy)
    @test_throws ArgumentError Reaction(:implicit, (electron => 1, H2[1] => 1) => (electron => 2, H[1] => 2); rate, producer = network(H[1]), energy)
    @test_throws ArgumentError Reaction(:pair, (H2[1] => 1, H2[0] => 1) => (H3[1] => 1, H[0] => 1); rate, producer = network(H2[1]))
    @test Reaction(
        :pair, (H2[1] => 1, H2[0] => 1) => (H3[1] => 1, H[0] => 1); rate, producer = LocalNetwork(implicit = H2[1], frozen = H2[0])
    ) isa Reaction
    @test_throws ArgumentError Reaction(:no_energy, (electron => 1, H[1] => 1) => (H[0] => 1,); rate, producer = network(H[1]))
    @test Reaction(:captured, (electron => 1, H[1] => 1) => (H[0] => 1,); rate, producer = network(H[1]), energy = CapturedElectron()) isa Reaction

    # the list: species only, each once, in a charge state its parent has
    constants = PlasmaConstants{Float64}()
    @test_throws ArgumentError build_registry((H2[0], :H2), Protium(), constants)
    @test_throws ArgumentError build_registry((H2[0], H2[0]), Protium(), constants)
    @test_throws ArgumentError build_registry((H2[0], Ion(H3, 2)), Protium(), constants)
    @test_throws ArgumentError build_registry((), Protium(), constants)

    # a rate given in the configuration replaces the catalog's; a label no active entry has is refused
    eRRCs, iRRCs = load_electron_RRCs(), load_H2_Ion_RRCs()
    list = (H2[0], H[0], H2[1], H[1])
    sys = assemble_species_system(list, Protium(), constants, eRRCs, iRRCs; rates = (:de2 => ConstantRate(1.0e-13),))
    de2 = only(filter(a -> a.reaction.label === :de2, sys.reactions))
    @test de2.rate === ConstantRate(1.0e-13) && de2.source == "config.rates"
    @test_throws ArgumentError assemble_species_system(list, Protium(), constants, eRRCs, iRRCs; rates = (:h3 => ConstantRate(1.0e-15),))
    @test_throws ArgumentError assemble_species_system(list, Protium(), constants, eRRCs, iRRCs; rates = (:de2 => 1.0e-13,))
end

@testitem "A parent that leaves out part of its API is refused" begin
    using RAPID2D: build_registry, PlasmaConstants

    struct Bare <: RAPID2D.Molecule end
    RAPID2D.allowed_charges(::Bare) = 0:1
    RAPID2D.composition(::Bare) = (:B => 2,)
    RAPID2D.parent_name(::Bare) = "B2"
    # no neutral_mass
    @test_throws ArgumentError build_registry((Bare()[0], Bare()[1]), Protium(), PlasmaConstants{Float64}())
end

@testitem "Masses and labels follow the hydrogen isotope, and so do heavy-pair rates by their rule" begin
    using RAPID2D: assemble_species_system, load_electron_RRCs, load_H2_Ion_RRCs, PlasmaConstants,
        species_mass, rate_coefficient, ElectronDriftContext, HeavyPairContext, TemperatureScaled,
        ConstantRate, RateData, ReferenceOnly, resolve_rate, LogPolynomialRate, Collision,
        ResonantChargeExchange
    using RAPID2D.Particles: H, H2, H3

    eRRCs, iRRCs, constants = load_electron_RRCs(), load_H2_Ion_RRCs(), PlasmaConstants{Float64}()
    list = (H2[0], H[0], H2[1], H[1], H3[1])
    hydrogen = assemble_species_system(list, Protium(), constants, eRRCs, iRRCs)
    deuterium = assemble_species_system(list, Deuterium(), constants, eRRCs, iRRCs)

    # labels and masses: the nuclei and the electrons, an ion z electron masses short
    @test [r.label for r in deuterium.registry.ions] == [Symbol("D2[1]"), Symbol("D[1]"), Symbol("D3[1]")]
    @test hydrogen.registry.ions[2].mass ≈ constants.mp rtol = 1.0e-15
    @test hydrogen.registry.neutrals[1].mass ≈ 2 * (constants.mp + constants.me) rtol = 1.0e-15
    @test deuterium.registry.ions[1].mass ≈ 2 * RAPID2D.DEUTERON_MASS + constants.me rtol = 1.0e-15

    # the same reactions either way; electron impact keeps its rate
    @test [a.reaction.label for a in hydrogen.reactions] == [a.reaction.label for a in deuterium.reactions]
    T = [0.5, 3.0, 10.0, 100.0]
    electron_ctx = ElectronDriftContext(T, zeros(4))
    for (a, b) in zip(hydrogen.reactions, deuterium.reactions)
        a.reaction.label in (:de2, :di2, :dr2, :dr3a, :dr3b, :de3, :izH0, :recH1) || continue
        @test rate_coefficient(a.rate, electron_ctx) == rate_coefficient(b.rate, electron_ctx)
    end

    # a heavy pair at equal velocity: the deuterium rate at T is protium's at T μ_H/μ_D
    reduced(a, b, iso) = (ma = species_mass(a, iso, constants); mb = species_mass(b, iso, constants); ma * mb / (ma + mb))
    factor = reduced(H[1], H[0], Protium()) / reduced(H[1], H[0], Deuterium())
    cx_H = only(filter(a -> a.collision.label === :cxH, hydrogen.collisions))
    cx_D = only(filter(a -> a.collision.label === :cxH, deuterium.collisions))
    @test cx_D.rate isa TemperatureScaled && cx_D.rate.factor == factor
    @test factor ≈ 0.5 rtol = 2.0e-3
    @test rate_coefficient(cx_D.rate, HeavyPairContext(T, zeros(4))) == rate_coefficient(cx_H.rate, HeavyPairContext(factor .* T, zeros(4)))

    # Langevin capture: about the measured 2.1e-9 cm³/s for H₂⁺ + H₂, and ∝ μ^(-1/2)
    h3(sys) = only(filter(a -> a.reaction.label === :h3, sys.reactions)).rate
    @test h3(hydrogen).K ≈ 2.1e-15 rtol = 0.02
    @test h3(deuterium).K / h3(hydrogen).K ≈ sqrt(reduced(H2[1], H2[0], Protium()) / reduced(H2[1], H2[0], Deuterium())) rtol = 1.0e-12

    # data that hold for one isotope alone are refused for another
    model = ConstantRate(1.0e-14)
    collision = Collision(:x, H[1], H[0]; rate = RateData(model, ReferenceOnly(Protium()), "test"), moment = ResonantChargeExchange())
    @test resolve_rate(collision.rate, collision, hydrogen.registry, constants) === model
    @test_throws ArgumentError resolve_rate(collision.rate, collision, deuterium.registry, constants)
end

@testitem "An element's ladder enters through its parent's methods, without changes to the package" begin
    using RAPID2D: assemble_species_system, load_electron_RRCs, load_H2_Ion_RRCs, PlasmaConstants,
        LadderRates, ConstantRate, ElectronSolve, LocalNetwork, species_report
    using RAPID2D.Particles: H2

    struct ToyElement <: RAPID2D.Element end
    RAPID2D.allowed_charges(::ToyElement) = 0:2
    RAPID2D.composition(::ToyElement) = (:X => 1,)
    RAPID2D.parent_name(::ToyElement) = "X"
    RAPID2D.neutral_mass(::ToyElement, ::RAPID2D.AbstractHydrogenIsotope, constants::RAPID2D.PlasmaConstants) = 20 * constants.mp
    RAPID2D.ladder_rates(::ToyElement, ::Type{FT}) where {FT} = LadderRates(
        (ConstantRate(FT(1.0e-14)), ConstantRate(FT(2.0e-15))), (ConstantRate(FT(3.0e-19)), ConstantRate(FT(4.0e-19))),
        (FT(7.0), FT(15.0)), ("toy ionization", "toy recombination"),
    )
    X = ToyElement()

    sys = assemble_species_system((H2[0], H2[1], X[0], X[1], X[2]), Protium(), PlasmaConstants{Float64}(), load_electron_RRCs(), load_H2_Ion_RRCs())
    ladder = Dict(a.reaction.label => a for a in sys.reactions if occursin(r"^(iz|rec)X\d$", string(a.reaction.label)))
    @test Set(keys(ladder)) == Set([:izX0, :izX1, :recX1, :recX2])
    @test ladder[:izX0].reaction.producer === ElectronSolve()             # from the neutral, with its electrons
    @test ladder[:izX1].reaction.producer === LocalNetwork(implicit = X[1])
    @test ladder[:recX2].reaction.producer === LocalNetwork(implicit = X[2])
    @test ladder[:izX1].rate === ConstantRate(2.0e-15)
    @test ladder[:izX1].reaction.energy.electron_loss == 15.0 * RAPID2D.PlasmaConstants{Float64}().ee
    @test sys.registry.ions[2].mass ≈ 20 * PlasmaConstants{Float64}().mp - PlasmaConstants{Float64}().me
    @test occursin("izX1    e + X[1] → 2 e + X[2]", species_report(sys))

    # an element without ladder rates is refused
    struct Rateless <: RAPID2D.Element end
    RAPID2D.allowed_charges(::Rateless) = 0:1
    RAPID2D.composition(::Rateless) = (:Y => 1,)
    RAPID2D.parent_name(::Rateless) = "Y"
    RAPID2D.neutral_mass(::Rateless, ::RAPID2D.AbstractHydrogenIsotope, constants::RAPID2D.PlasmaConstants) = 9 * constants.mp
    @test_throws ArgumentError assemble_species_system(
        (H2[0], Rateless()[0], Rateless()[1]), Protium(), PlasmaConstants{Float64}(), load_electron_RRCs(), load_H2_Ion_RRCs()
    )
end

@testitem "The HYDHEL fits give the values the database prints, and a captured electron's energy" begin
    using RAPID2D: LogPolynomialRate, ElectronDriftContext, rate_coefficient, event_energy!,
        CapturedElectron, ConstantRate, PlasmaConstants
    R = RAPID2D

    # ⟨σv⟩ [cm³/s] HYDHEL prints at a fit's lower bound and at its maximum: a miscopied
    # coefficient, weighted by (ln T)⁸, would miss them by orders of magnitude
    K_cgs(fit, T) = 1.0e6 * rate_coefficient(LogPolynomialRate(Float64, fit), ElectronDriftContext([T], [0.0]))[1]
    at_lower_bound = (
        (R.HYDHEL_2_1_5, 2.0, 7.22e-12), (R.HYDHEL_2_2_11, 3.16, 5.23e-12), (R.HYDHEL_2_2_12, 0.2, 1.23e-12),
        (R.HYDHEL_2_2_14, 0.1, 2.23e-7), (R.HYDHEL_2_2_15, 0.1, 1.05e-7), (R.HYDHEL_2_2_16, 1.26, 9.46e-13),
    )
    for (fit, T, printed) in at_lower_bound
        @test K_cgs(fit, T) ≈ printed rtol = 0.06
    end
    Ts = exp.(range(log(0.1), log(1.0e4); length = 4001))
    for (fit, printed) in ((R.HYDHEL_2_1_5, 3.11e-8), (R.HYDHEL_2_2_11, 9.06e-9), (R.HYDHEL_2_2_12, 1.15e-7), (R.HYDHEL_2_2_16, 4.18e-7))
        @test maximum(T -> K_cgs(fit, T), Ts) ≈ printed rtol = 0.02
    end

    # a captured electron's energy, e T (3/2 + d ln K/d ln T), against AMJUEL's H.8 2.2.14 fit of
    # ⟨σvE⟩ [eV cm³/s], ln⟨σvE⟩ = h₀ + h₁ ln T, made from HYDHEL's b₀ and b₁ alone (b₂…b₈ ~ 10⁻⁸)
    ee = PlasmaConstants{Float64}().ee
    T = [0.3, 1.0, 10.0, 100.0]
    ctx = ElectronDriftContext(T, zeros(4))
    dr2 = LogPolynomialRate(Float64, R.HYDHEL_2_2_14)
    energy = event_energy!(zeros(4), CapturedElectron(), dr2, ctx, ee)
    h0, h1 = -1.681368547011e+1, 3.964355004318e-1
    weighted = @. exp(h0 + h1 * log(T)) * 1.0e-6                        # eV m³/s
    @test energy ./ ee ≈ weighted ./ rate_coefficient(dr2, ctx) rtol = 1.0e-4
    # a constant rate: the electrons' mean energy, (3/2) e T
    @test event_energy!(zeros(4), CapturedElectron(), ConstantRate(1.0e-19), ctx, ee) ≈ 1.5 * ee .* T rtol = 1.0e-14
end

@testitem "initialize! assembles the species list and writes its report" begin
    using RAPID2D: initialize_species!
    using RAPID2D.Particles: H, H2

    dir = mktempdir()
    config = SimulationConfig{Float64}(; NR = 10, NZ = 10, Output_path = dir, Output_prefix = "run_")
    RP = RAPID{Float64}(config)
    initialize_species!(RP)
    @test isnothing(RP.species)

    RP.config.species = (H2[0], H2[1], H[0])    # H[1] left out: warned of
    @test_logs (:info, r"Species: 1 ions, 2 neutrals, 3 reactions, 2 collisions") (:warn, r"diz") match_mode = :any initialize_species!(RP)
    @test RP.species isa SpeciesSystem
    report = read(joinpath(dir, "run_species.txt"), String)
    @test occursin("dr2     e + H2[1] → 2 H[0]", report)
    @test occursin("Warnings", report) && occursin("lacks H[1]", report)
end
