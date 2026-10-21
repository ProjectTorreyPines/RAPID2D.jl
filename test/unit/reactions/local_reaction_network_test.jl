# The local reaction network (`local_network.jl`), in a few independent cells (0D): conservation
# of charge, nuclei and ion mass by the events, positivity at long steps, the analytic decay and
# steady state, the frozen partner, the blocks, and the joint budget that refuses a step whole.

@testsnippet LocalNetworkCase begin
    using LinearAlgebra: I
    using RAPID2D: assemble_species_system, load_electron_RRCs, load_H2_Ion_RRCs, PlasmaConstants,
        local_network_plan, NetworkInputs, NetworkAccumulator, solve_local_network!, network_rates!,
        network_energies!, ElectronDriftContext, HeavyPairContext, species_key
    using RAPID2D.Particles: H, H2, H3

    constants = PlasmaConstants{Float64}()
    tables = (load_electron_RRCs(), load_H2_Ion_RRCs())

    # Cells at electron temperatures Te: the plan, densities, and the inputs, rates and energies from
    # the rows' own models, the electron solve adding no ions and using no neutrals. The keywords
    # (rates, processes) go to the assembly
    function network_case(list; Te = [2.0, 8.0, 30.0], ne = fill(1.0e18, length(Te)), kwargs...)
        sys = assemble_species_system(list, Protium(), constants, tables...; kwargs...)
        plan = local_network_plan(sys)
        cells = length(Te)
        n_ion = [fill(1.0e17 * s, cells) for s in 1:plan.n_ion]
        n_neutral = [fill(1.0e19, cells) for _ in 1:plan.n_neutral]
        electron_ctx = ElectronDriftContext(Te, zeros(cells))
        K = network_rates!([zeros(cells) for _ in plan.rows], plan, electron_ctx, HeavyPairContext(fill(0.5, cells), zeros(cells)))
        energy = network_energies!([zeros(cells) for _ in plan.rows], plan, electron_ctx, constants.ee)
        inputs = NetworkInputs(K, energy, ne, [zeros(cells) for _ in 1:plan.n_ion], [zeros(cells) for _ in 1:plan.n_neutral])
        return (; sys, plan, n_ion, n_neutral, inputs, acc = NetworkAccumulator{Float64}(plan, (cells,)))
    end

    ion_column(case, sp) = case.sys.registry.ion_column[species_key(sp)]
    neutral_column(case, sp) = case.sys.registry.neutral_column[species_key(sp)]
    row_index(case, label) = findfirst(row -> row.label === label, case.plan.rows)

    # The network's matrix in one cell over all ion columns, I − Σⱼ νⱼ transferⱼ e_i(j)ᵀ, with
    # transferⱼ = Δt Kⱼ n_cat,j
    function network_matrix(case, k, Δt)
        plan = case.plan
        matrix = Matrix{Float64}(I, plan.n_ion, plan.n_ion)
        for (j, row) in enumerate(plan.rows), (s, ν) in row.ion_change
            catalyst = row.frozen == 0 ? case.inputs.ne[k] : case.n_neutral[row.frozen][k]
            matrix[s, row.implicit] -= ν * Δt * case.inputs.K[j][k] * catalyst
        end
        return matrix
    end
end

@testitem "The network's events conserve charge, each element's nuclei and the ion mass" setup = [LocalNetworkCase] begin
    using RAPID2D: Accept
    using RAPID2D.Particles: H, H2, H3

    case = network_case((H2[0], H[0], H2[1], H[1], H3[1]))
    reg = case.sys.registry
    old_ion, old_neutral = deepcopy(case.n_ion), deepcopy(case.n_neutral)
    @test solve_local_network!(case.acc, case.n_ion, case.n_neutral, case.plan, case.inputs, 1.0e-6) === Accept()

    for k in 1:3
        Δion = [case.n_ion[s][k] - old_ion[s][k] for s in 1:case.plan.n_ion]
        Δneutral = [case.n_neutral[n][k] - old_neutral[n][k] for n in 1:case.plan.n_neutral]
        scale = sum(old_ion[s][k] for s in 1:case.plan.n_ion)
        # charge: the ions gain what the electrons gain
        @test sum(r.charge * Δion[s] for (s, r) in enumerate(reg.ions)) ≈ case.acc.born_electrons[k] - case.acc.lost_electrons[k] atol = 1.0e-13 * scale
        # nuclei: all hydrogen, counted in the ions and the neutrals
        nuclei(record) = last(only(record.key.composition))
        @test sum(nuclei(r) * Δion[s] for (s, r) in enumerate(reg.ions)) + sum(nuclei(r) * Δneutral[n] for (n, r) in enumerate(reg.neutrals)) ≈ 0 atol = 1.0e-13 * scale
        # ion mass: taken in from neutrals at rest, less what left with the ions
        @test sum(r.mass * Δion[s] for (s, r) in enumerate(reg.ions)) ≈ case.acc.ion_mass_born[k] - case.acc.ion_mass_lost[k] rtol = 1.0e-10
        # every row's events are what the densities moved by
        for (s, _) in enumerate(reg.ions)
            from_events = sum(sum(ν * case.acc.events[j][k] for (col, ν) in row.ion_change if col == s; init = 0.0) for (j, row) in enumerate(case.plan.rows))
            @test Δion[s] ≈ from_events rtol = 1.0e-12 atol = 1.0e-13 * scale
        end
    end
end

@testitem "At a step a hundred H2[1] lifetimes long the ions stay positive" setup = [LocalNetworkCase] begin
    using RAPID2D: Accept
    using RAPID2D.Particles: H, H2, H3

    case = network_case((H2[0], H[0], H2[1], H[1], H3[1]))
    loss = [sum(case.inputs.K[j][k] * case.inputs.ne[k] for (j, row) in enumerate(case.plan.rows) if row.implicit == ion_column(case, H2[1]) && row.frozen == 0) for k in 1:3]
    Δt = 100 / minimum(loss)
    @test solve_local_network!(case.acc, case.n_ion, case.n_neutral, case.plan, case.inputs, Δt) === Accept()
    @test all(all(>=(0), n) for n in case.n_ion)
    @test all(all(>=(0), n) for n in case.n_neutral)
end

@testitem "H2[1] decays as (1 + λΔt)⁻ⁿ, and under a constant source settles at S/λ" setup = [LocalNetworkCase] begin
    using RAPID2D: ConstantRate, Accept
    using RAPID2D.Particles: H, H2

    rates = (
        :de2 => ConstantRate(5.0e-14), :de2b => ConstantRate(0.0), :di2 => ConstantRate(2.0e-14), :dr2 => ConstantRate(1.0e-14),
        :recH1 => ConstantRate(0.0),
    )
    case = network_case((H2[0], H[0], H2[1], H[1]); Te = [5.0], ne = [2.0e18], rates)
    s = ion_column(case, H2[1])
    λ = (5.0e-14 + 2.0e-14 + 1.0e-14) * 2.0e18
    Δt = 0.5 / λ
    n_start = case.n_ion[s][1]
    for _ in 1:20
        solve_local_network!(case.acc, case.n_ion, case.n_neutral, case.plan, case.inputs, Δt)
    end
    @test case.n_ion[s][1] ≈ n_start / (1 + λ * Δt)^20 rtol = 1.0e-12

    # a source S from the electron solve: n⁺ = (n + SΔt)/(1 + λΔt) settles at S/λ, whatever Δt
    S = 3.0e21
    case.inputs.ion_increment[s][1] = S * Δt
    for _ in 1:200
        @test solve_local_network!(case.acc, case.n_ion, case.n_neutral, case.plan, case.inputs, Δt) === Accept()
    end
    @test case.n_ion[s][1] ≈ S / λ rtol = 1.0e-12
end

@testitem "h3 holds H2[0] frozen and takes it out by its events" setup = [LocalNetworkCase] begin
    using RAPID2D: ConstantRate, Accept
    using RAPID2D.Particles: H, H2, H3

    off = ConstantRate(0.0)
    rates = (:de2 => off, :de2b => off, :di2 => off, :dr2 => off, :recH1 => off, :dr3a => off, :dr3b => off, :de3 => off)
    case = network_case((H2[0], H[0], H2[1], H[1], H3[1]); Te = [5.0], rates)
    K = only(case.inputs.K[row_index(case, :h3)])
    n_H2 = case.n_neutral[neutral_column(case, H2[0])][1]
    n_start = case.n_ion[ion_column(case, H2[1])][1]
    Δt = 1.0e-4

    # the matrix: an M-matrix, positive diagonal, transfers ≤ 0 and a nonnegative inverse (H2[0]
    # brings nuclei to the ions, so it is not diagonally dominant in them)
    matrix = network_matrix(case, 1, Δt)
    @test all(matrix[i, i] > 0 for i in axes(matrix, 1))
    @test all(matrix[i, j] <= 0 for i in axes(matrix, 1), j in axes(matrix, 2) if i != j)
    @test all(>=(0), inv(matrix))

    @test solve_local_network!(case.acc, case.n_ion, case.n_neutral, case.plan, case.inputs, Δt) === Accept()
    n_end = n_start / (1 + Δt * K * n_H2)
    @test case.n_ion[ion_column(case, H2[1])][1] ≈ n_end rtol = 1.0e-14
    h3_events = only(case.acc.events[row_index(case, :h3)])
    @test h3_events ≈ Δt * K * n_H2 * n_end rtol = 1.0e-14
    # H2[0] loses the events; H3[1] and H[0] gain them
    @test case.n_neutral[neutral_column(case, H2[0])][1] ≈ n_H2 - h3_events rtol = 1.0e-14
    @test case.n_neutral[neutral_column(case, H[0])][1] ≈ 1.0e19 + h3_events rtol = 1.0e-14
    @test case.n_ion[ion_column(case, H3[1])][1] ≈ 3.0e17 + h3_events rtol = 1.0e-14
    # the ion mass: H2[0] in at rest, H[0] out with the ions
    reg = case.sys.registry
    @test only(case.acc.ion_mass_born) ≈ h3_events * reg.neutrals[neutral_column(case, H2[0])].mass rtol = 1.0e-14
end

@testitem "Blocks follow the flow of production, a ladder's cycle solved as one" setup = [LocalNetworkCase] begin
    using RAPID2D: LadderRates, ConstantRate, NetworkBlock, LocalNetworkPlan, Accept, ThomasSolve, PivotedLUSolve,
        EliminationSolve, SingleBlock
    using RAPID2D.Particles: H, H2, H3

    # hydrogen with H3[1]: no cycle, H2[1] before H3[1] before H[1]
    case = network_case((H2[0], H[0], H2[1], H[1], H3[1]))
    order = [only(b.columns) for b in case.plan.blocks]
    @test order == [ion_column(case, H2[1]), ion_column(case, H3[1]), ion_column(case, H[1])]

    # a toy ladder X[1] ⇄ X[2] ⇄ X[3], ionization up and recombination down: one tridiagonal block
    struct Ladder3 <: RAPID2D.Element end
    RAPID2D.allowed_charges(::Ladder3) = 0:3
    RAPID2D.composition(::Ladder3) = (:X => 1,)
    RAPID2D.parent_name(::Ladder3) = "X"
    RAPID2D.neutral_mass(::Ladder3, ::RAPID2D.AbstractHydrogenIsotope, constants::RAPID2D.PlasmaConstants) = 12 * constants.mp
    RAPID2D.ladder_rates(::Ladder3, ::RAPID2D.RateClosure, ::Type{FT}) where {FT} = LadderRates(
        (ConstantRate(FT(4.0e-14)), ConstantRate(FT(2.0e-14)), ConstantRate(FT(1.0e-14))),
        (ConstantRate(FT(1.0e-14)), ConstantRate(FT(3.0e-14)), ConstantRate(FT(5.0e-14))),
        (FT(11.3), FT(24.4), FT(47.9)), ("toy", "toy"),
    )
    X = Ladder3()
    ladder = network_case((H2[0], H2[1], X[0], X[1], X[2], X[3]))
    block = only(filter(b -> length(b.columns) > 1, ladder.plan.blocks))
    @test block.columns == [ion_column(ladder, X[z]) for z in 1:3] && block.solver isa ThomasSolve

    # Thomas, elimination, pivoted LU, the whole cell as one block and a dense solve agree
    Δt = 1.0e-5
    dense = ladder.plan.n_ion
    expected = [network_matrix(ladder, k, Δt) \ [ladder.n_ion[s][k] for s in 1:dense] for k in 1:3]
    with(solver) = LocalNetworkPlan(
        ladder.plan.rows, ladder.plan.reactions,
        Tuple(NetworkBlock(b.columns, solver, b.rows, b.inflow) for b in ladder.plan.blocks), ladder.plan.n_ion, ladder.plan.n_neutral,
    )
    coupled = local_network_plan(ladder.sys; block_structure = SingleBlock())
    for plan in (ladder.plan, with(EliminationSolve()), with(PivotedLUSolve()), coupled)
        n_ion = deepcopy(ladder.n_ion)
        @test solve_local_network!(ladder.acc, n_ion, deepcopy(ladder.n_neutral), plan, ladder.inputs, Δt) === Accept()
        @test all([n_ion[s][k] for s in 1:dense] ≈ expected[k] for k in 1:3)
    end
end

@testitem "The hydrogen ions in production order are the whole cell's system, solved by substitution" setup = [LocalNetworkCase] begin
    using RAPID2D: Accept, ScalarSolve, EliminationSolve, SingleBlock
    using RAPID2D.Particles: H, H2, H3

    # No reaction turns H3[1] or H[1] back into an upstream ion, so the cell's 3×3 system is lower
    # triangular in the order H2[1], H3[1], H[1]: three divisions in turn solve it exactly
    case = network_case((H2[0], H[0], H2[1], H[1], H3[1]))
    order = [ion_column(case, sp) for sp in (H2[1], H3[1], H[1])]
    @test [only(b.columns) for b in case.plan.blocks] == order
    @test all(b -> b.solver isa ScalarSolve, case.plan.blocks)
    coupled = local_network_plan(case.sys; block_structure = SingleBlock())
    @test only(coupled.blocks).columns == order && only(coupled.blocks).solver isa EliminationSolve
    for k in 1:3
        matrix = network_matrix(case, k, 1.0e-6)
        @test all(iszero, matrix[i, j] for (a, i) in enumerate(order), (b, j) in enumerate(order) if b > a)
    end

    for Δt in (1.0e-6, 1.0e-3)                  # a step short, and one far beyond the H2[1] lifetime
        expected = [network_matrix(case, k, Δt) \ [case.n_ion[s][k] for s in 1:case.plan.n_ion] for k in 1:3]
        results = map((case.plan, coupled)) do plan
            n_ion, acc = deepcopy(case.n_ion), NetworkAccumulator{Float64}(plan, (3,))
            @test solve_local_network!(acc, n_ion, deepcopy(case.n_neutral), plan, case.inputs, Δt) === Accept()
            @test all([n_ion[s][k] for s in 1:case.plan.n_ion] ≈ expected[k] for k in 1:3)
            @test all(all(>=(0), n) for n in n_ion)
            (n_ion, acc.events)
        end
        @test all(isapprox.(results[1][1], results[2][1]; rtol = 1.0e-12))
        @test all(isapprox.(results[1][2], results[2][2]; rtol = 1.0e-12))
    end

    # neither way allocates in the cell loop
    function allocated(plan, inputs, n_ion0, n_neutral0)
        acc, n_ion, n_neutral = NetworkAccumulator{Float64}(plan, (3,)), deepcopy(n_ion0), deepcopy(n_neutral0)
        solve_local_network!(acc, n_ion, n_neutral, plan, inputs, 1.0e-6)
        return @allocated solve_local_network!(acc, n_ion, n_neutral, plan, inputs, 1.0e-6)
    end
    @test allocated(case.plan, case.inputs, case.n_ion, case.n_neutral) == 0
    @test allocated(coupled, case.inputs, case.n_ion, case.n_neutral) == 0
end

@testitem "A reaction back up the production order closes a cycle: one block, fed by the block before it" setup = [LocalNetworkCase] begin
    using RAPID2D: ReactionProcess, RateClosure, MaxwellianClosure, RateSources, RateData, SameForIsotopes,
        ConstantRate, LocalNetwork, Accept, ScalarSolve, SingleBlock
    using RAPID2D.Particles: H, H2, H3

    # a toy association H[1] + H2[0] → H3[1] turns H[1] back into H3[1]: the two become one cycle,
    # downstream of H2[1], whose rows into them are the cycle's inflow
    struct ToyAssociation <: ReactionProcess end
    RAPID2D.process_label(::ToyAssociation) = :toy
    RAPID2D.stoichiometry(::ToyAssociation) = (H[1] => 1, H2[0] => 1) => (H3[1] => 1,)
    RAPID2D.producer(::ToyAssociation) = LocalNetwork(implicit = H[1], frozen = H2[0])
    RAPID2D.rate_data(::ToyAssociation, ::MaxwellianClosure, ::RateSources) = RateData(ConstantRate(1.0e-15), SameForIsotopes(), "toy")
    RAPID2D.electron_energy(::ToyAssociation, ::RateClosure, ::RateSources) = nothing
    case = network_case((H2[0], H[0], H2[1], H[1], H3[1]); processes = (ToyAssociation(),))
    @test length(case.plan.blocks) == 2
    upstream, cycle = case.plan.blocks
    @test only(upstream.columns) == ion_column(case, H2[1]) && upstream.solver isa ScalarSolve
    @test sort(cycle.columns) == sort([ion_column(case, H3[1]), ion_column(case, H[1])])
    @test !isempty(cycle.inflow)

    # the blocks in turn, the whole cell as one block and a dense solve agree, at a step short and
    # one far beyond the H2[1] lifetime
    coupled = local_network_plan(case.sys; block_structure = SingleBlock())
    for Δt in (1.0e-6, 1.0e-3)
        expected = [network_matrix(case, k, Δt) \ [case.n_ion[s][k] for s in 1:case.plan.n_ion] for k in 1:3]
        for plan in (case.plan, coupled)
            n_ion = deepcopy(case.n_ion)
            acc = NetworkAccumulator{Float64}(plan, (3,))
            @test solve_local_network!(acc, n_ion, deepcopy(case.n_neutral), plan, case.inputs, Δt) === Accept()
            @test all([n_ion[s][k] for s in 1:case.plan.n_ion] ≈ expected[k] for k in 1:3)
            @test all(all(>=(0), n) for n in n_ion)
        end
    end
end

@testitem "Pivoted LU solves a block that is no M-matrix, a zero first pivot included" begin
    using RAPID2D: block_solve!, PivotedLUSolve

    # the block in the leading rows of a larger scratch, as the network holds it; the rest is NaN
    # so that touching it would show
    A = [0.0 2.0 1.0; 1.0 1.0 0.0; 3.0 0.0 1.0]
    x = [1.0, -2.0, 0.5]
    matrix, rhs = fill(NaN, 4, 4), fill(NaN, 4)
    matrix[1:3, 1:3] .= A
    rhs[1:3] .= A * x
    block_solve!(PivotedLUSolve(), matrix, rhs, 3)
    @test rhs[1:3] ≈ x rtol = 1.0e-12
    @test isnan(rhs[4])
end

@testitem "The electron solve's use of a neutral and the network's share one budget; a refused step changes nothing" setup = [LocalNetworkCase] begin
    using RAPID2D: ConstantRate, Accept, Reject
    using RAPID2D.Particles: H, H2, H3

    off = ConstantRate(0.0)
    rates = (:de2 => off, :de2b => off, :di2 => off, :dr2 => off, :recH1 => off, :dr3a => off, :dr3b => off, :de3 => off)
    case = network_case((H2[0], H[0], H2[1], H[1], H3[1]); rates)
    Δt = 1.0e-3
    h3 = row_index(case, :h3)
    g = neutral_column(case, H2[0])

    # what h3 alone takes of H2[0] in each cell
    trial = NetworkAccumulator{Float64}(case.plan, (3,))
    @test solve_local_network!(trial, deepcopy(case.n_ion), deepcopy(case.n_neutral), case.plan, case.inputs, Δt) === Accept()
    taken = copy(trial.events[h3])

    # the electron solve takes the rest of the third cell and a little more: refused, untouched
    case.inputs.donor_use[g] .= [0.0, 0.0, case.n_neutral[g][3] - taken[3] * (1 - 1.0e-9)]
    before = (deepcopy(case.n_ion), deepcopy(case.n_neutral))
    @test solve_local_network!(case.acc, case.n_ion, case.n_neutral, case.plan, case.inputs, Δt) === Reject(:budget)
    @test (case.n_ion, case.n_neutral) == before

    # a little less: the two fit
    case.inputs.donor_use[g][3] = case.n_neutral[g][3] - taken[3] * (1 + 1.0e-9)
    @test solve_local_network!(case.acc, case.n_ion, case.n_neutral, case.plan, case.inputs, Δt) === Accept()

    # more electrons captured than the cell holds: refused
    starved = network_case((H2[0], H[0], H2[1], H[1]); ne = [1.0e16, 1.0e16, 1.0e16])
    before = (deepcopy(starved.n_ion), deepcopy(starved.n_neutral))
    @test solve_local_network!(starved.acc, starved.n_ion, starved.n_neutral, starved.plan, starved.inputs, 1.0) === Reject(:budget)
    @test (starved.n_ion, starved.n_neutral) == before
end

@testitem "An electron solve that takes more of an ion than the cell holds is refused, and changes nothing" setup = [LocalNetworkCase] begin
    using RAPID2D: Reject
    using RAPID2D.Particles: H, H2

    case = network_case((H2[0], H[0], H2[1], H[1]))
    s = ion_column(case, H[1])
    case.inputs.ion_increment[s][2] = -2 * case.n_ion[s][2]     # the second cell only
    before = (deepcopy(case.n_ion), deepcopy(case.n_neutral))
    @test solve_local_network!(case.acc, case.n_ion, case.n_neutral, case.plan, case.inputs, 1.0e-6) === Reject(:positivity)
    @test (case.n_ion, case.n_neutral) == before
end
