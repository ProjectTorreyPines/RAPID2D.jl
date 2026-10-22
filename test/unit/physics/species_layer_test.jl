# The species layer: parents indexed by charge state, identities, the records the transport
# carries, the reaction-count contractions generated from the stoichiometry table, and the
# per-species ion ledger entries recorded beside the totals.

@testitem "A parent indexed by charge state is a neutral or an ion, by type" begin
    using RAPID2D: Neutral, Ion, NeutralSpecies, IonSpecies, AbstractSpecies, allowed_charges,
        species_label
    using RAPID2D.Particles: H, H2, H3

    @test H2[0] isa Neutral && H2[0] isa NeutralSpecies && H2[0] isa AbstractSpecies
    @test H2[1] isa Ion && H2[1] isa IonSpecies && H2[1].z == 1
    @test H[0] isa Neutral && H[1] isa Ion && H3[1] isa Ion

    # a literal charge state resolves the type at compile time, which is what lets
    # "neutral or ion" select an equation by dispatch
    neutral_molecule() = H2[0]
    molecular_ion() = H2[1]
    @test only(Base.return_types(neutral_molecule, ())) <: Neutral
    @test only(Base.return_types(molecular_ion, ())) <: Ion

    # charge states a parent does not have are errors, not silent species
    @test_throws ArgumentError H3[0]          # the neutral H₃ is not bound
    @test_throws ArgumentError H2[2]
    @test_throws ArgumentError H[2]
    @test allowed_charges(H) == 0:1

    @test species_label(H2[1]) === Symbol("H2[1]")
    @test species_label(H[0]) === Symbol("H[0]")
    @test isbitstype(typeof(H2[1]))
end

@testitem "A species' identity is its composition and charge, not its label" begin
    using RAPID2D: species_key, SpeciesKey
    using RAPID2D.Particles: H, H2, H3

    @test species_key(H2[1]) == SpeciesKey((:H => 2,), 1)
    @test species_key(H3[1]) == SpeciesKey((:H => 3,), 1)
    @test species_key(H[0]) == SpeciesKey((:H => 1,), 0)
    @test species_key(H2[1]) != species_key(H2[0])
    @test species_key(H2[1]) != species_key(H[1])
    @test length(Set([species_key(H2[1]), species_key(H2[1]), species_key(H[1])])) == 2
end

@testitem "An ion record carries identity, label, mass and charge" begin
    using RAPID2D: IonRecord, species_key, SpeciesKey, UNREGISTERED_COMPOSITION
    using RAPID2D.Particles: H2

    mi = 3.3475e-27
    r = IonRecord(H2[1]; mass = mi)
    @test r.key == species_key(H2[1])
    @test r.label === Symbol("H2[1]")
    @test r.mass === mi                       # stored as given, not recomputed
    @test r.charge == 1
    @test r isa IonRecord{Float64}

    # the three-argument form is a test probe no reaction table can match
    probe = IonRecord(:probe, 6mi, 6)
    @test probe.key == SpeciesKey(UNREGISTERED_COMPOSITION, 6)
    @test probe.charge == 6

    @test_throws ArgumentError IonRecord(:bad, -1.0, 1)
    @test_throws ArgumentError IonRecord(:bad, 1.0, 0)
end

@testitem "The reaction-count contractions keep the hand-written expressions, signed zeros included" begin
    using RAPID2D: ReactionCounts, net_electron_count, net_H2_gas_count, net_ion_count, species_key
    using RAPID2D.Particles: H, H2

    bits(x) = reinterpret.(UInt64, x)
    N = ReactionCounts{Float64}(dims = (2, 3))
    N.iz .= [1.0 -0.0 0.0; 2.5 0.0 1.0e-300]
    N.diz .= [0.5 0.0 -0.0; -0.0 -0.0 3.0e-300]

    @test bits(net_electron_count(N)) == bits(N.iz .+ N.diz)
    @test bits(net_H2_gas_count(N)) == bits(-(N.iz .+ N.diz))
    for k in eachindex(N.iz)
        @test bits(net_H2_gas_count(N, k)) == bits(-(N.iz[k] + N.diz[k]))
    end
    @test bits(net_ion_count(N, species_key(H2[1]))) == bits(N.iz .+ N.diz)
    @test net_ion_count(N, species_key(H[1])) === nothing

    # a fresh array, never an alias of a stored count
    Δ = net_ion_count(N, species_key(H2[1]))
    Δ .= 0.0
    @test N.iz[1, 1] == 1.0
end

@testitem "Per-species ion ledger entries sum to the totals" begin
    using RAPID2D: IonRecord, solve_ion_continuity_equation!, update_transport_quantities!,
        PerSpeciesTransport, layout_ion_species!

    config = SimulationConfig{Float64}(;
        device_Name = "manual", NR = 21, NZ = 21, R_min = 1.0, R_max = 2.0, Z_min = -0.5, Z_max = 0.5,
        wall_R = [1.15, 1.85, 1.85, 1.15], wall_Z = [-0.35, -0.35, 0.35, 0.35],
        prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-8, t_end_s = 1.0e-6,
        snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0, Output_path = mktempdir(; cleanup = false),
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)
    RP.flags.update_ni_independently = true
    RP.flags.src = false
    RP.flags.ion_pinch = true
    RP.flags.ion_transport_policy = PerSpeciesTransport()
    G, pla = RP.G, RP.plasma
    @. pla.ni = 1.0e18 * exp(-((G.R2D - 1.5)^2 + G.Z2D^2) / 0.02)
    vec(pla.ni)[setdiff(1:(G.NR * G.NZ), G.nodes.in_wall_nids)] .= 0.0
    pla.ne .= pla.ni
    fill!(pla.Te_eV, 10.0); fill!(pla.Ti_eV, 1.0)
    update_transport_quantities!(RP)

    tracker = RP.diagnostics.Ntracker

    # one species: the entry is the total (the entries follow the species list)
    tracker.cum0D_Ni_loss = 0.0
    fill!(tracker.cum0D_Ni_loss_s, 0.0)
    solve_ion_continuity_equation!(RP)
    @test length(tracker.cum0D_Ni_loss_s) == 1
    @test tracker.cum0D_Ni_loss > 0
    @test tracker.cum0D_Ni_loss_s[1] == tracker.cum0D_Ni_loss

    # two species, laid out past the one-species guard: the entries grow with the list and
    # sum to the total to round-off (their own reductions)
    tp = RP.transport
    layout_ion_species!(RP, [tp.ion_species[1], IonRecord(:C6, 6 * tp.ion_species[1].mass, 6)])
    vec(pla.n_ions[2])[G.nodes.in_wall_nids] .= 1.0e15
    tracker.cum0D_Ni_loss = 0.0
    fill!(tracker.cum0D_Ni_loss_s, 0.0)
    for _ in 1:5
        solve_ion_continuity_equation!(RP)
    end
    @test length(tracker.cum0D_Ni_loss_s) == 2
    @test all(>(0), tracker.cum0D_Ni_loss_s)
    @test sum(tracker.cum0D_Ni_loss_s) ≈ tracker.cum0D_Ni_loss rtol = 1.0e-12
end

@testitem "The contraction generator builds the expressions the accessors run" begin
    using RAPID2D: _column_channels, _contraction, _contraction_at

    # the table's columns: both channels make one electron and burn one molecule
    @test _column_channels(row -> row.electron) == ([:iz, :diz], 1)
    @test _column_channels(row -> row.H2_gas) == ([:iz, :diz], -1)
    # the two rows are equal values, so a column with mixed coefficients is built from the
    # call order: +1, −1, +1, −1
    calls = Ref(0)
    @test_throws ErrorException _column_channels(row -> (calls[] += 1; isodd(calls[]) ? 1 : -1))

    whole(ch) = :(N.$ch)
    at(ch) = :(N.$ch[k])
    @test _contraction([:iz, :diz], 1, whole) == :(N.iz .+ N.diz)
    @test _contraction([:iz, :diz], -1, whole) == :(-(N.iz .+ N.diz))
    @test _contraction([:iz], 1, whole) == :(copy(N.iz))           # never an alias
    @test _contraction([:iz, :diz], 2, whole) == :(2 .* (N.iz .+ N.diz))
    @test _contraction_at([:iz, :diz], -1, at) == :(-(N.iz[k] + N.diz[k]))
    @test _contraction_at([:iz], 3, at) == :(3 * N.iz[k])
end

@testitem "Species methods answer at run time, not only for literal species" begin
    using RAPID2D: species_label, species_key, charge, composition
    using RAPID2D.Particles: H, H2, H3

    # through a Vector{Any}, so nothing is folded at compile time
    sps = Any[H[0], H[1], H2[0], H2[1], H3[1]]
    @test map(species_label, sps) == Symbol.(["H[0]", "H[1]", "H2[0]", "H2[1]", "H3[1]"])
    @test map(charge, sps) == [0, 1, 0, 1, 1]
    @test map(sp -> composition(sp.parent), sps) ==
        [(:H => 1,), (:H => 1,), (:H => 2,), (:H => 2,), (:H => 3,)]
    @test length(unique(map(species_key, sps))) == 5
end
