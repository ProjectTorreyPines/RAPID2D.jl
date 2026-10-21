# The species store (`plasma.n_ions`, `plasma.n_gas`): each ion parent's charge states as the
# planes of one array, each parent's neutral gas as a matrix, laid out from the run's species;
# `ni` and `n_H2_gas` read where they lie, the solvers reach the store through their columns.

@testsnippet StoreCase begin
    using RAPID2D: LadderRates, ConstantRate
    using RAPID2D.Particles: H, H2, H3

    "A small discharge-shaped run; `species` is its list, if any, and `hydrogen` its isotope."
    function store_case(; species = nothing, hydrogen = Protium())
        config = SimulationConfig{Float64}(;
            device_Name = "manual", NR = 9, NZ = 9, R_min = 1.0, R_max = 2.0, Z_min = -0.5, Z_max = 0.5,
            wall_R = [1.15, 1.85, 1.85, 1.15], wall_Z = [-0.35, -0.35, 0.35, 0.35],
            prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-8, t_end_s = 1.0e-6,
            snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0, Output_path = mktempdir(; cleanup = false), species, hydrogen,
        )
        RP = RAPID{Float64}(config)
        initialize!(RP)
        return RP
    end

    "A peaked ion profile inside the wall, electrons on it, ions free to evolve, no sources."
    function peaked!(RP)
        G, pla = RP.G, RP.plasma
        RP.flags.update_ni_independently = true
        RP.flags.src = false
        @. pla.ni = 1.0e18 * exp(-((G.R2D - 1.5)^2 + G.Z2D^2) / 0.02)
        vec(pla.ni)[setdiff(1:(G.NR * G.NZ), G.nodes.in_wall_nids)] .= 0.0
        pla.ne .= pla.ni
        fill!(pla.Te_eV, 10.0)
        fill!(pla.Ti_eV, 1.0)
        return RP
    end

    # A toy element X[0:3] with its ladder: three ion charge states, a neutral of the gas
    struct ToyLadder <: RAPID2D.Element end
    RAPID2D.allowed_charges(::ToyLadder) = 0:3
    RAPID2D.composition(::ToyLadder) = (:X => 1,)
    RAPID2D.parent_name(::ToyLadder) = "X"
    RAPID2D.neutral_mass(::ToyLadder, ::RAPID2D.AbstractHydrogenIsotope, constants::RAPID2D.PlasmaConstants) = 12 * constants.mp
    RAPID2D.ladder_rates(::ToyLadder, ::RAPID2D.RateClosure, ::Type{FT}) where {FT} = LadderRates(
        (ConstantRate(FT(4.0e-14)), ConstantRate(FT(2.0e-14)), ConstantRate(FT(1.0e-14))),
        (ConstantRate(FT(1.0e-14)), ConstantRate(FT(3.0e-14)), ConstantRate(FT(5.0e-14))),
        (FT(11.3), FT(24.4), FT(47.9)), ("toy", "toy"),
    )
    const X = ToyLadder()
end

@testitem "Without a list the store holds the transport's H₂⁺ and the H₂ gas; `ni` and `n_H2_gas` read them" setup = [StoreCase] begin
    RP = store_case()
    pla, layout = RP.plasma, RP.store_layout
    @test layout.ion_parents == ["H2"] && layout.gas_parents == ["H2"]
    @test size.(pla.n_ions) == [(9, 9, 1)] && size.(pla.n_gas) == [(9, 9)]
    @test parent(pla.ni) === pla.n_ions[1] && pla.n_H2_gas === pla.n_gas[1]
    @test layout.transport_slots == [(1, 1)]

    # written through its name, a density lands in the store
    pla.ni .= 3.0
    @test all(==(3.0), pla.n_ions[1])
    # the names are inferred where they are written as fields
    read_ni(p) = p.ni
    read_gas(p) = p.n_H2_gas
    @test @inferred(read_ni(pla)) isa SubArray && @inferred(read_gas(pla)) isa Matrix
    @test :ni in propertynames(pla) && :n_H2_gas in propertynames(pla)

    # initialize! again builds a new plasma, and its store
    initialize!(RP)
    @test parent(RP.plasma.ni) === RP.plasma.n_ions[1] && RP.plasma.n_H2_gas === RP.plasma.n_gas[1]

    # the store is written into, never rebound
    pla = RP.plasma
    @test_throws ArgumentError pla.ni = zeros(9, 9)
    @test_throws ArgumentError pla.n_H2_gas = zeros(9, 9)
    @test_throws ArgumentError pla.n_ions = [zeros(9, 9, 1)]
    @test_throws ArgumentError pla.ni_slot = (1, 1)
    pla.Te_eV = fill(3.0, 9, 9)                          # other fields as before
    @test all(==(3.0), pla.Te_eV)
end

@testitem "With a list the store holds its parents in order, each ion parent's charge states as planes" setup = [StoreCase] begin
    using RAPID2D: species_key

    # H first, H2[1] the second ion parent, X with three charge states
    RP = store_case(species = (H[0], H[1], H2[0], H2[1], H3[1], X[0], X[1], X[2], X[3]))
    pla, layout, reg = RP.plasma, RP.store_layout, RP.species.registry
    @test layout.ion_parents == ["H", "H2", "H3", "X"] && layout.gas_parents == ["H", "H2", "X"]
    @test size.(pla.n_ions) == [(9, 9, 1), (9, 9, 1), (9, 9, 1), (9, 9, 3)]
    @test [layout.ion_slots[reg.ion_column[species_key(sp)]] for sp in (H[1], H2[1], H3[1], X[1], X[2], X[3])] ==
        [(1, 1), (2, 1), (3, 1), (4, 1), (4, 2), (4, 3)]
    @test [r.charge for r in layout.ion_records[4]] == [1, 2, 3]
    @test parent(pla.ni) === pla.n_ions[2] && pla.ni_slot == (2, 1)
    @test pla.n_H2_gas === pla.n_gas[2] && layout.transport_slots == [(2, 1)]
    # the transport moves the list's H₂⁺; what initialize! set before the list carried over
    @test RP.transport.ion_species[1] === reg.ions[reg.ion_column[species_key(H2[1])]]
    @test all(>(0), pla.n_H2_gas) && pla.ni == pla.ne
    @test all(iszero, pla.n_ions[1]) && all(iszero, pla.n_ions[4]) && all(iszero, pla.n_gas[3])

    # deuterium: the list's D₂⁺, at its own mass
    D = store_case(species = (H2[0], H2[1]), hydrogen = Deuterium())
    @test D.transport.ion_species[1].label === Symbol("D2[1]") && D.store_layout.ion_parents == ["D2"]
    @test D.transport.ion_species[1].mass ≈ 2 * RAPID2D.DEUTERON_MASS + D.config.constants.me rtol = 1.0e-15

    # a parent's charge states are 1…Z with none missing, and the H₂ gas the neutral model evolves is listed
    @test_throws ArgumentError store_case(species = (H2[0], H2[1], X[0], X[1], X[3]))
    @test_throws ArgumentError store_case(species = (H[0], H[1], H2[1]))
end

@testitem "A new layout carries each density to its species' new place, and the bulk's to the new bulk" setup = [StoreCase] begin
    using RAPID2D: IonRecord, initialize_species!, set_ion_species!

    # every plane and every gas its own values
    RP = store_case(species = (H2[0], H2[1], X[0], X[1], X[2], X[3]))
    pla = RP.plasma
    ramp = reshape(1.0:81.0, 9, 9)
    for (k, A) in enumerate(pla.n_ions), z in axes(A, 3)
        A[:, :, z] .= (10k + z) .* ramp
    end
    for (j, g) in enumerate(pla.n_gas)
        g .= 100j .* ramp
    end
    ions_H2, ions_X, gas_H2, gas_X = copy(pla.ni), copy(pla.n_ions[2]), copy(pla.n_gas[1]), copy(pla.n_gas[2])

    # the same species in another order, H besides: every parent moves, and the densities with it
    RP.config.species = (X[0], X[1], X[2], X[3], H2[0], H2[1], H[0], H[1])
    initialize_species!(RP)
    pla, layout = RP.plasma, RP.store_layout
    @test layout.ion_parents == ["X", "H2", "H"] && layout.gas_parents == ["X", "H2", "H"]
    @test pla.n_ions[1] == ions_X && pla.ni == ions_H2 && parent(pla.ni) === pla.n_ions[2]
    @test pla.n_gas[1] == gas_X && pla.n_H2_gas == gas_H2 && pla.n_H2_gas === pla.n_gas[2]
    @test all(iszero, pla.n_ions[3]) && all(iszero, pla.n_gas[3])            # H, new

    # without a list the transport's ion declared another one: the bulk density goes over to it
    RP = store_case()
    RP.plasma.ni .= ramp
    gas = copy(RP.plasma.n_H2_gas)
    set_ion_species!(RP, [IonRecord(:C6, 6 * RP.transport.ion_species[1].mass, 6)])
    pla = RP.plasma
    @test pla.ni == ramp && parent(pla.ni) === pla.n_ions[1] && pla.n_H2_gas == gas
end

@testitem "The mixture's sums take every stored ion, moved by the transport or not" setup = [StoreCase] begin
    RP = store_case(species = (H2[0], H2[1], X[0], X[1], X[2], X[3]))
    pla = RP.plasma
    pla.ni .= 1.0e18
    pla.n_ions[2][:, :, 2] .= 1.0e17                    # X[2], which the transport does not move yet
    update_charge_states!(RP)
    @test all(≈(1.1e18), pla.ni_tot)
    @test all(≈(1.0e18 + 2 * 1.0e17), pla.Zni_tot)
    @test all(≈(1.0e18 + 4 * 1.0e17), pla.Z2ni_tot)
    @test all(≈(1.4e18 / 1.2e18), pla.Zeff)                 # Σ z² n / Σ z n
    m_H2, m_X2 = RP.store_layout.ion_records[1][1].mass, RP.store_layout.ion_records[2][2].mass
    @test all(≈(m_H2 * 1.0e18 + m_X2 * 1.0e17), pla.mni_tot)
end

@testitem "The ion solve gathers the transport's species from the store and scatters them back" setup = [StoreCase] begin
    using RAPID2D: IonRecord, layout_ion_species!, solve_ion_continuity_equation!, update_transport_quantities!

    # two species past the one-species guard, each a parent of its own
    RP = peaked!(store_case())
    G, pla, tp = RP.G, RP.plasma, RP.transport
    layout_ion_species!(RP, [tp.ion_species[1], IonRecord(:C6, 6 * tp.ion_species[1].mass, 6)])
    @test RP.store_layout.transport_slots == [(1, 1), (2, 1)]
    vec(view(pla.n_ions[2], :, :, 1))[G.nodes.in_wall_nids] .= 1.0e15
    update_transport_quantities!(RP)
    before = deepcopy(pla.n_ions)
    solve_ion_continuity_equation!(RP)
    @test parent(pla.ni) === pla.n_ions[1]
    @test all(s -> pla.n_ions[s] != before[s], 1:2)                                # both moved
    @test all(s -> vec(pla.n_ions[s]) == tp.ion_N[:, s], 1:2)                      # the solve's columns, scattered back

    # with a list the transport's H₂⁺ is the second ion parent: the solve moves that plane
    RP = peaked!(store_case(species = (H[0], H[1], H2[0], H2[1])))
    pla = RP.plasma
    update_transport_quantities!(RP)
    before = copy(pla.ni)
    solve_ion_continuity_equation!(RP)
    @test parent(pla.ni) === pla.n_ions[2] && pla.ni != before
    @test vec(pla.ni) == RP.transport.ion_N[:, 1]
end

@testitem "The band outside the wall is cleared and negatives floored on every charge plane" setup = [StoreCase] begin
    using RAPID2D: clear_ions_outside_wall!, correct_negative_densities!, grow_species_entries!

    RP = store_case(species = (H2[0], H2[1], X[0], X[1], X[2], X[3]))
    G, pla = RP.G, RP.plasma
    out, inside = G.nodes.on_out_wall_nids, G.nodes.in_wall_nids
    foreach(A -> fill!(A, 1.0), pla.n_ions)
    clear_ions_outside_wall!(RP)
    for A in pla.n_ions, z in axes(A, 3)
        @test all(iszero, view(A, :, :, z)[out]) && all(==(1.0), view(A, :, :, z)[inside])
    end

    # negatives on the transported H₂⁺ and on X[3]: both floored, each booked to the total, the
    # transported one to its own entry as well
    RP.flags.update_ni_independently = true
    RP.flags.negative_n_correction = true
    pla.ne .= 1.0
    tracker = RP.diagnostics.Ntracker
    grow_species_entries!(tracker, 1)
    tracker.cum0D_Ni_loss = 0.0
    fill!(tracker.cum0D_Ni_loss_s, 0.0)
    k1, k2 = inside[1], inside[2]
    pla.ni[k1] = -2.0
    view(pla.n_ions[2], :, :, 3)[k2] = -3.0
    correct_negative_densities!(RP)
    @test pla.ni[k1] == 0 && view(pla.n_ions[2], :, :, 3)[k2] == 0 && all(>=(0), pla.n_ions[2])
    vol(k) = 2π * G.Jacob[k] * G.dR * G.dZ
    @test tracker.cum0D_Ni_loss_s[1] ≈ -2.0 * vol(k1) rtol = 1.0e-14
    @test tracker.cum0D_Ni_loss ≈ -2.0 * vol(k1) - 3.0 * vol(k2) rtol = 1.0e-14
end

@testitem "A step put back restores the store in place; a store laid out anew during a step is refused" setup = [StoreCase] begin
    using RAPID2D: StepState, restore_step_state!, IonRecord, layout_ion_species!

    RP = store_case(species = (H2[0], H2[1], X[0], X[1], X[2], X[3]))
    pla = RP.plasma
    pla.ni .= 2.0
    pla.n_ions[2][:, :, 3] .= 5.0
    held_ni, held_gas = pla.ni, pla.n_H2_gas             # views and matrices held across the step
    gas = copy(held_gas)
    ck = StepState(RP)
    pla.ni .= -1.0
    pla.n_ions[2] .= 0.0
    pla.n_H2_gas .= 0.0
    restore_step_state!(RP, ck)
    @test all(==(2.0), held_ni) && parent(held_ni) === RP.plasma.n_ions[1]
    @test all(==(5.0), RP.plasma.n_ions[2][:, :, 3]) && held_gas == gas && held_gas === RP.plasma.n_H2_gas

    # a layout changed between the save and the restore cannot be put back into
    RP = store_case()
    ck = StepState(RP)
    layout_ion_species!(RP, [RP.transport.ion_species[1], IonRecord(:C6, 6 * RP.transport.ion_species[1].mass, 6)])
    @test_throws ErrorException restore_step_state!(RP, ck)
end

@testitem "Slaved ions are one species: another stored ion with density is refused" setup = [StoreCase] begin
    using RAPID2D: slave_ions_to_electrons!

    RP = store_case(species = (H2[0], H2[1], X[0], X[1], X[2], X[3]))
    RP.flags.update_ni_independently = false
    RP.plasma.ne .= 1.0e18
    slave_ions_to_electrons!(RP)
    @test all(==(1.0e18), RP.plasma.ni)
    RP.plasma.n_ions[2][:, :, 1] .= 1.0e15
    @test_throws ArgumentError slave_ions_to_electrons!(RP)
end
