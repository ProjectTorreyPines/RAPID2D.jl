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
    # the rows' own models, the electron solve adding no ions and using no neutrals
    function network_case(list; Te = [2.0, 8.0, 30.0], ne = fill(1.0e18, length(Te)), rates = ())
        sys = assemble_species_system(list, Protium(), constants, tables...; rates)
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

    rates = (:de2 => ConstantRate(5.0e-14), :di2 => ConstantRate(2.0e-14), :dr2 => ConstantRate(1.0e-14), :recH1 => ConstantRate(0.0))
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
    rates = (:de2 => off, :di2 => off, :dr2 => off, :recH1 => off, :dr3a => off, :dr3b => off, :de3 => off)
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
    using RAPID2D: LadderRates, ConstantRate, NetworkBlock, LocalNetworkPlan, Accept
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
    RAPID2D.ladder_rates(::Ladder3, ::Type{FT}) where {FT} = LadderRates(
        (ConstantRate(FT(4.0e-14)), ConstantRate(FT(2.0e-14)), ConstantRate(FT(1.0e-14))),
        (ConstantRate(FT(1.0e-14)), ConstantRate(FT(3.0e-14)), ConstantRate(FT(5.0e-14))),
        (FT(11.3), FT(24.4), FT(47.9)), ("toy", "toy"),
    )
    X = Ladder3()
    ladder = network_case((H2[0], H2[1], X[0], X[1], X[2], X[3]))
    block = only(filter(b -> length(b.columns) > 1, ladder.plan.blocks))
    @test block.columns == [ion_column(ladder, X[z]) for z in 1:3] && block.tridiagonal

    # Thomas, pivoted LU and a dense solve of the whole matrix agree
    Δt = 1.0e-5
    dense = ladder.plan.n_ion
    expected = [network_matrix(ladder, k, Δt) \ [ladder.n_ion[s][k] for s in 1:dense] for k in 1:3]
    as_lu = LocalNetworkPlan(
        ladder.plan.rows, ladder.plan.reactions,
        [NetworkBlock(b.columns, false, b.rows, b.inflow) for b in ladder.plan.blocks], ladder.plan.n_ion, ladder.plan.n_neutral,
    )
    for plan in (ladder.plan, as_lu)
        n_ion = deepcopy(ladder.n_ion)
        @test solve_local_network!(ladder.acc, n_ion, deepcopy(ladder.n_neutral), plan, ladder.inputs, Δt) === Accept()
        @test all([n_ion[s][k] for s in 1:dense] ≈ expected[k] for k in 1:3)
    end
end

@testitem "The electron solve's use of a neutral and the network's share one budget; a refused step changes nothing" setup = [LocalNetworkCase] begin
    using RAPID2D: ConstantRate, Accept, Reject
    using RAPID2D.Particles: H, H2, H3

    off = ConstantRate(0.0)
    rates = (:de2 => off, :di2 => off, :dr2 => off, :recH1 => off, :dr3a => off, :dr3b => off, :de3 => off)
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
