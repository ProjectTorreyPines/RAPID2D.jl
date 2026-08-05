# Which transport equation do the ion species share?
#
# Ions are collisional with each other 342× faster than they are transported
# (τ_ii = 1.51 ms vs τ_transport = 0.515 s at ne = 1e15, Ti = 1 eV), so friction
# forbids the relative drift that species-dependent D∥ would produce. That argues
# for one shared effective operator. But D∥ carries 73 % of the ion flux reaching
# the wall, so the approximation is not obviously small either.
#
# Both readings are implemented and selected by TYPE, not by a symbol compared at
# each call site. These tests fix the contract of the one dispatch point,
# `ion_transport_groups`, and of the mixing rule underneath it.

@testitem "A channel every species shares passes through untouched" begin
    using RAPID2D: DiffusionChannel, mixture_channel

    # The turbulent ExB channel is mass- and charge-free (v_E = E_pol/B_tot), so
    # every ion species carries literally the SAME object. Mixing must recognise
    # that and return it — both because the shared policy then costs exactly
    # nothing for this mechanism, and because a weighted mean of a constant is not
    # bit-exact in floating point at the density ratios a discharge spans.
    ch = DiffusionChannel(fill(3.0, 4, 5), fill(0.5, 4, 5), fill(7.0, 4, 5), fill(0.25, 4, 5); vm_para = fill(3.0, 4, 5), vm_perp = fill(7.0, 4, 5))
    weights = [fill(1.0e14, 4, 5), fill(9.0e17, 4, 5), fill(2.0, 4, 5)]

    @test mixture_channel([ch, ch, ch], weights) === ch
    @test mixture_channel([ch], [weights[1]]) === ch
end

@testitem "Equal-valued but distinct channels mix back to their common value" begin
    using RAPID2D: DiffusionChannel, mixture_channel

    # Same claim without the identity short-circuit: the general path must also
    # reproduce a constant, to rounding. Weights span 18 orders, which is what a
    # burn-through impurity fraction actually looks like.
    mk() = DiffusionChannel(fill(3.0, 4, 5), fill(0.5, 4, 5), fill(7.0, 4, 5), fill(0.25, 4, 5); vm_para = fill(3.0, 4, 5), vm_perp = fill(7.0, 4, 5))
    mix = mixture_channel([mk(), mk(), mk()], [fill(1.0e14, 4, 5), fill(9.0e17, 4, 5), fill(2.0, 4, 5)])

    @test mix.v_para ≈ fill(3.0, 4, 5)
    @test mix.λ_para ≈ fill(0.5, 4, 5)
    @test mix.v_perp ≈ fill(7.0, 4, 5)
    @test mix.λ_perp ≈ fill(0.25, 4, 5)
end

@testitem "Mixing weights the speed and the diffusivity, and derives the step from both" begin
    using RAPID2D: DiffusionChannel, mixture_channel, channel_D_para, channel_D_perp

    # Two mechanisms are averaged in the two quantities that have an exact
    # justification, and λ = 2D/v follows:
    #   v  — a one-sided wall flux is Σ ¼v̄_s·n_s, linear in v
    #   D  — a diffusive flux is Σ D_s∇n_s, linear in D
    # Averaging λ directly instead would conserve neither.
    ch1 = DiffusionChannel(fill(2.0, 3, 3), fill(3.0, 3, 3), fill(4.0, 3, 3), fill(5.0, 3, 3); vm_para = fill(2.0, 3, 3), vm_perp = fill(4.0, 3, 3))
    ch2 = DiffusionChannel(fill(6.0, 3, 3), fill(1.0, 3, 3), fill(8.0, 3, 3), fill(1.0, 3, 3); vm_para = fill(6.0, 3, 3), vm_perp = fill(8.0, 3, 3))
    w = [fill(1.0, 3, 3), fill(3.0, 3, 3)]

    mix = mixture_channel([ch1, ch2], w)

    @test all(mix.v_para .≈ (1 * 2 + 3 * 6) / 4)          # = 5
    @test all(channel_D_para(mix) .≈ (1 * 3 + 3 * 3) / 4) # = 3, both species have D∥ = 3
    @test all(mix.λ_para .≈ 2 * 3 / 5)                    # = 1.2, not (3+1)/2 nor (1·3+3·1)/4

    @test all(mix.v_perp .≈ (1 * 4 + 3 * 8) / 4)          # = 7
    @test all(channel_D_perp(mix) .≈ (1 * 10 + 3 * 4) / 4) # = 5.5
    @test all(mix.λ_perp .≈ 2 * 5.5 / 7)
end

@testitem "Mixing averages the wall speed on its own, not through v" begin
    using RAPID2D: DiffusionChannel, mixture_channel

    # Two species can carry different v̄/v ratios — a hydrogen ion and a heavy
    # impurity have different masses, and each channel's v is set by how its D was
    # obtained. So v̄ has to be averaged as itself. Scaling the averaged v would
    # reintroduce, inside the mixture, exactly the shared-ratio assumption the
    # per-channel v̄ exists to remove.
    ch1 = DiffusionChannel(
        fill(2.0, 2, 2), fill(3.0, 2, 2), fill(4.0, 2, 2), fill(5.0, 2, 2);
        vm_para = fill(20.0, 2, 2), vm_perp = fill(4.0, 2, 2)     # ratio 10
    )
    ch2 = DiffusionChannel(
        fill(6.0, 2, 2), fill(1.0, 2, 2), fill(8.0, 2, 2), fill(1.0, 2, 2);
        vm_para = fill(6.0, 2, 2), vm_perp = fill(8.0, 2, 2)      # ratio 1
    )
    mix = mixture_channel([ch1, ch2], [fill(1.0, 2, 2), fill(3.0, 2, 2)])

    @test all(mix.vm_para .≈ (1 * 20 + 3 * 6) / 4)         # = 9.5
    @test all(mix.v_para .≈ (1 * 2 + 3 * 6) / 4)          # = 5, and 9.5 is not a multiple of it
    @test all(mix.vm_perp .≈ (1 * 4 + 3 * 8) / 4)          # = 7

    # …and the ceiling the mixture reports is the density-weighted sum of the
    # ceilings the species would have reported, exactly, at a head-on face
    using RAPID2D: channel_ceiling
    head_on = (ch) -> channel_ceiling(ch, 1.0, 0.0, (1, 0))
    @test all(4 .* head_on(mix) .≈ 1 .* head_on(ch1) .+ 3 .* head_on(ch2))
end

@testitem "Mixing is per cell, and a species that is absent nowhere contributes" begin
    using RAPID2D: DiffusionChannel, mixture_channel

    ch1 = DiffusionChannel(fill(2.0, 2, 3), fill(3.0, 2, 3), fill(4.0, 2, 3), fill(5.0, 2, 3); vm_para = fill(2.0, 2, 3), vm_perp = fill(4.0, 2, 3))
    ch2 = DiffusionChannel(fill(6.0, 2, 3), fill(1.0, 2, 3), fill(8.0, 2, 3), fill(1.0, 2, 3); vm_para = fill(6.0, 2, 3), vm_perp = fill(8.0, 2, 3))

    # species 2 present only in the right-hand column
    w2 = zeros(2, 3)
    w2[:, 3] .= 1.0
    mix = mixture_channel([ch1, ch2], [ones(2, 3), w2])

    @test all(mix.v_para[:, 1:2] .≈ 2.0)     # pure species 1
    @test all(mix.v_para[:, 3] .≈ 4.0)       # equal parts: (2+6)/2
end

@testitem "An empty cell falls back to the unweighted mean instead of NaN" begin
    using RAPID2D: DiffusionChannel, mixture_channel

    # Σn = 0 happens on every node outside the plasma at t = 0, and a 0/0 there
    # would poison the whole matrix. The fallback must also not be D = 0: a cell
    # with no ions yet is exactly where ions are about to diffuse INTO, and a zero
    # diffusivity there is an artificial barrier at the plasma edge.
    ch1 = DiffusionChannel(fill(2.0, 2, 2), fill(3.0, 2, 2), fill(4.0, 2, 2), fill(5.0, 2, 2); vm_para = fill(2.0, 2, 2), vm_perp = fill(4.0, 2, 2))
    ch2 = DiffusionChannel(fill(6.0, 2, 2), fill(1.0, 2, 2), fill(8.0, 2, 2), fill(1.0, 2, 2); vm_para = fill(6.0, 2, 2), vm_perp = fill(8.0, 2, 2))

    mix = mixture_channel([ch1, ch2], [zeros(2, 2), zeros(2, 2)])

    @test all(isfinite, mix.v_para)
    @test all(isfinite, mix.λ_para)
    @test all(mix.v_para .≈ 4.0)             # (2 + 6)/2
    @test all(mix.v_perp .≈ 6.0)             # (4 + 8)/2
    @test all(mix.λ_para .> 0)               # not a transport barrier
end

@testitem "Mixing is scale-invariant in the weights" begin
    using RAPID2D: DiffusionChannel, mixture_channel

    # A low-density discharge starts at ne = 1e6 and ends 9 orders higher. The
    # mixture must depend on the density RATIO only, or the effective transport
    # would drift during a run for no physical reason.
    ch1 = DiffusionChannel(fill(2.0, 3, 2), fill(3.0, 3, 2), fill(4.0, 3, 2), fill(5.0, 3, 2); vm_para = fill(2.0, 3, 2), vm_perp = fill(4.0, 3, 2))
    ch2 = DiffusionChannel(fill(6.0, 3, 2), fill(1.0, 3, 2), fill(8.0, 3, 2), fill(1.0, 3, 2); vm_para = fill(6.0, 3, 2), vm_perp = fill(8.0, 3, 2))

    lo = mixture_channel([ch1, ch2], [fill(1.0e6, 3, 2), fill(3.0e6, 3, 2)])
    hi = mixture_channel([ch1, ch2], [fill(1.0e15, 3, 2), fill(3.0e15, 3, 2)])

    @test lo.v_para ≈ hi.v_para
    @test lo.λ_perp ≈ hi.λ_perp
end

@testitem "The mixture conserves the wall flux exactly head-on and grazing" begin
    using RAPID2D: DiffusionChannel, mixture_channel, channel_ceiling

    # The ceiling is ¼√(v̄⊥² + (v̄∥² − v̄⊥²)b_n²) — linear in v only at the two
    # limits, where it reduces to ¼v̄∥ and ¼v̄⊥. There the density-weighted mean
    # speed reproduces Σ n_s·¼v̄_s EXACTLY. In between it blends in quadrature and
    # cannot; that error is the price of the shared policy and is pinned below.
    ch1 = DiffusionChannel(fill(2.0, 3, 3), fill(3.0, 3, 3), fill(4.0, 3, 3), fill(5.0, 3, 3); vm_para = fill(2.0, 3, 3), vm_perp = fill(4.0, 3, 3))
    ch2 = DiffusionChannel(fill(6.0, 3, 3), fill(1.0, 3, 3), fill(8.0, 3, 3), fill(1.0, 3, 3); vm_para = fill(6.0, 3, 3), vm_perp = fill(8.0, 3, 3))
    chans = [ch1, ch2]
    w = [fill(1.0, 3, 3), fill(3.0, 3, 3)]
    n_tot = w[1] .+ w[2]
    mix = mixture_channel(chans, w)

    # head-on: b̂ ∥ n̂
    per_species = sum(w[s] .* channel_ceiling(chans[s], 1.0, 0.0, (1, 0)) for s in 1:2)
    @test all(per_species .≈ n_tot .* channel_ceiling(mix, 1.0, 0.0, (1, 0)))

    # grazing: b̂ ⊥ n̂
    per_species = sum(w[s] .* channel_ceiling(chans[s], 0.0, 1.0, (1, 0)) for s in 1:2)
    @test all(per_species .≈ n_tot .* channel_ceiling(mix, 0.0, 1.0, (1, 0)))

    # 45°, where the quadrature blend bites: the mixture UNDER-delivers, because
    # √ is concave and the mean speed of a mixture is not the mixture of speeds
    b = sqrt(0.5)
    per_species = sum(w[s] .* channel_ceiling(chans[s], b, b, (1, 0)) for s in 1:2)
    shared = n_tot .* channel_ceiling(mix, b, b, (1, 0))
    err = maximum(abs.(shared .- per_species) ./ per_species)
    @test all(shared .< per_species)
    @test err < 0.01
end

@testitem "Mixing rejects a ragged set of channels" begin
    using RAPID2D: DiffusionChannel, mixture_channel

    ch1 = DiffusionChannel(fill(2.0, 3, 3), fill(3.0, 3, 3), fill(4.0, 3, 3), fill(5.0, 3, 3); vm_para = fill(2.0, 3, 3), vm_perp = fill(4.0, 3, 3))
    ch2 = DiffusionChannel(fill(2.0, 4, 3), fill(3.0, 4, 3), fill(4.0, 4, 3), fill(5.0, 4, 3); vm_para = fill(2.0, 4, 3), vm_perp = fill(4.0, 4, 3))

    @test_throws DimensionMismatch mixture_channel([ch1, ch2], [ones(3, 3), ones(3, 3)])
    @test_throws DimensionMismatch mixture_channel([ch1, ch1], [ones(3, 3), ones(4, 3)])
    @test_throws ArgumentError mixture_channel(typeof(ch1)[], Matrix{Float64}[])
    @test_throws ArgumentError mixture_channel([ch1, ch1], [ones(3, 3)])
end

@testitem "Per-species policy hands every species its own operator, untouched" begin
    using RAPID2D: DiffusionChannel, PerSpeciesTransport, ion_transport_groups

    mk(v) = DiffusionChannel(fill(v, 2, 2), fill(1.0, 2, 2), fill(2v, 2, 2), fill(0.5, 2, 2); vm_para = fill(v, 2, 2), vm_perp = fill(2v, 2, 2))
    per_species = [[mk(1.0), mk(10.0)], [mk(2.0), mk(20.0)], [mk(3.0), mk(30.0)]]
    w = [fill(1.0, 2, 2), fill(2.0, 2, 2), fill(3.0, 2, 2)]

    groups = ion_transport_groups(PerSpeciesTransport(), per_species, w)

    @test length(groups) == 3
    @test [g.sids for g in groups] == [[1], [2], [3]]
    # the channels are passed through, not rebuilt: this policy approximates nothing
    for s in 1:3
        @test groups[s].channels[1] === per_species[s][1]
        @test groups[s].channels[2] === per_species[s][2]
    end
end

@testitem "Shared policy collapses to one operator and mixes mechanism by mechanism" begin
    using RAPID2D: DiffusionChannel, SharedEffectiveTransport, ion_transport_groups,
        mixture_channel

    mk(v) = DiffusionChannel(fill(v, 2, 2), fill(1.0, 2, 2), fill(2v, 2, 2), fill(0.5, 2, 2); vm_para = fill(v, 2, 2), vm_perp = fill(2v, 2, 2))
    per_species = [[mk(1.0), mk(10.0)], [mk(2.0), mk(20.0)], [mk(3.0), mk(30.0)]]
    w = [fill(1.0, 2, 2), fill(2.0, 2, 2), fill(3.0, 2, 2)]

    groups = ion_transport_groups(SharedEffectiveTransport(), per_species, w)

    @test length(groups) == 1
    @test groups[1].sids == [1, 2, 3]

    # Two mechanisms in, two mechanisms out. Mixing the TENSORS instead would
    # collapse them to one and destroy the per-mechanism wall ceilings, which add
    # across mechanisms but average across species.
    @test length(groups[1].channels) == 2
    @test groups[1].channels[1].v_para ≈ mixture_channel([p[1] for p in per_species], w).v_para
    @test groups[1].channels[2].v_para ≈ mixture_channel([p[2] for p in per_species], w).v_para
end

@testitem "With one species the two policies are bit-identical" begin
    using RAPID2D: DiffusionChannel, SharedEffectiveTransport, PerSpeciesTransport,
        ion_transport_groups

    # The regression that lets the default change without changing today's answer:
    # H₂⁺ is the only ion species in the code, so switching policy must be a no-op
    # until a second species is appended.
    only_one = [
        [
            DiffusionChannel(fill(3.0, 4, 6), fill(1.5, 4, 6), fill(9.0, 4, 6), fill(0.1, 4, 6); vm_para = fill(3.0, 4, 6), vm_perp = fill(9.0, 4, 6)),
            DiffusionChannel(zeros(4, 6), zeros(4, 6), fill(4.0, 4, 6), fill(2.0, 4, 6); vm_para = zeros(4, 6), vm_perp = fill(4.0, 4, 6)),
        ],
    ]
    w = [fill(5.0e14, 4, 6)]

    shared = ion_transport_groups(SharedEffectiveTransport(), only_one, w)
    per = ion_transport_groups(PerSpeciesTransport(), only_one, w)

    @test length(shared) == length(per) == 1
    @test shared[1].sids == per[1].sids == [1]
    for m in 1:2
        @test shared[1].channels[m].v_para == per[1].channels[m].v_para
        @test shared[1].channels[m].λ_para == per[1].channels[m].λ_para
        @test shared[1].channels[m].v_perp == per[1].channels[m].v_perp
        @test shared[1].channels[m].λ_perp == per[1].channels[m].λ_perp
    end
end

@testitem "Grouping rejects species that disagree about their mechanisms" begin
    using RAPID2D: DiffusionChannel, SharedEffectiveTransport, PerSpeciesTransport,
        ion_transport_groups

    mk() = DiffusionChannel(ones(2, 2), ones(2, 2), ones(2, 2), ones(2, 2); vm_para = ones(2, 2), vm_perp = ones(2, 2))
    ragged = [[mk(), mk()], [mk()]]
    w = [ones(2, 2), ones(2, 2)]

    # A species missing a mechanism is a construction bug, not a physical statement
    # that it lacks that transport — a species with no Bohm channel passes a
    # zero-speed channel instead.
    @test_throws ArgumentError ion_transport_groups(SharedEffectiveTransport(), ragged, w)
    @test_throws ArgumentError ion_transport_groups(PerSpeciesTransport(), ragged, w)

    ok = [[mk()], [mk()]]
    @test_throws ArgumentError ion_transport_groups(SharedEffectiveTransport(), ok, [ones(2, 2)])
    @test_throws ArgumentError ion_transport_groups(PerSpeciesTransport(), ok, [ones(2, 2)])
end

@testitem "An ion species carries the mass and charge its channels need" begin
    using RAPID2D: IonSpecies

    h2p = IonSpecies(:H2⁺, 2 * 1.6726e-27, 1)
    c6p = IonSpecies(:C⁶⁺, 12 * 1.6726e-27, 6)

    @test h2p.name === :H2⁺
    @test c6p.charge == 6
    @test h2p.mass < c6p.mass
    # mass and charge vary INDEPENDENTLY across the periodic table, which is why
    # Zeff alone cannot stand in for a species: C⁶⁺ and H₂⁺ differ 6× in charge
    # and 6× in mass, and no single scalar recovers both.
    @test c6p.mass / h2p.mass ≈ c6p.charge / h2p.charge

    @test_throws ArgumentError IonSpecies(:junk, -1.0, 1)
    @test_throws ArgumentError IonSpecies(:junk, 1.0, 0)
end
