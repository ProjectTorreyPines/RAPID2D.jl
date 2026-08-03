# How many transport equations do the ion species need?
#
# Ions exchange momentum with each other far faster than they are transported:
# at ne = 1e15 m⁻³, Ti = 1 eV, τ_ii = 1.51 ms against τ_transport ≈ 0.515 s, a
# ratio of 342. Friction damps any relative drift between species 342× faster
# than transport can build one, so ion species do not diffuse independently —
# they drag each other, which is the same physics as impurity pinch and
# screening. On that reading one effective equation is not a shortcut but the
# correct limit.
#
# Against that: D∥ carries 73 % of the ion flux that reaches the wall at those
# conditions (convection 11 %, D⊥ 15 %), and D∥ is the one mechanism where the
# species genuinely differ. So the mixing error lands on the dominant term.
#
# Both readings are therefore implemented, and which one runs is a matter of
# TYPE. No call site below the dispatch point asks which policy is in force.

"""
    IonSpecies{FT}

One ion species: what its transport channels need to know about it.

`mass` and `charge` vary independently across the periodic table — C⁶⁺ and H₂⁺
differ 6× in both, but O⁸⁺ and H₂⁺ differ 8× in charge and 8× in mass while C⁴⁺
and He²⁺ share neither ratio. That independence is why `Zeff` alone cannot stand
in for a species: one scalar cannot carry two independent numbers.
"""
struct IonSpecies{FT <: AbstractFloat}
    name::Symbol
    mass::FT
    charge::Int

    function IonSpecies{FT}(name::Symbol, mass::FT, charge::Int) where {FT <: AbstractFloat}
        mass > zero(FT) || throw(ArgumentError("$name: mass must be positive, got $mass"))
        charge > 0 || throw(ArgumentError("$name: an ion carries charge ≥ 1, got $charge"))
        return new{FT}(name, mass, charge)
    end
end

IonSpecies(name::Symbol, mass::Real, charge::Integer) =
    IonSpecies{float(typeof(mass))}(name, float(mass), Int(charge))

"""
    IonTransportPolicy

Whether ion species share a transport operator, and which one.

The choice is a type rather than a flag value so that it resolves at
[`ion_transport_groups`](@ref) and nowhere else. Everything downstream —
assembly, factorization, the solve — is written once against the groups and
never learns which policy produced them.
"""
abstract type IonTransportPolicy end

"""
    SharedEffectiveTransport()

One operator for every ion species, built from density-weighted effective
channels.

Justified by `τ_ii ≪ τ_transport`: friction couples the species into a single
fluid long before transport can separate them. Costs one factorization for any
number of species, with the species entering as extra right-hand sides.

Mixing happens **mechanism by mechanism**, not on the assembled tensor. That
keeps each mechanism's wall ceiling separable — ceilings *add* across mechanisms
while densities *average* across species, and collapsing to a tensor first would
lose the distinction.
"""
struct SharedEffectiveTransport <: IonTransportPolicy end

"""
    PerSpeciesTransport()

Each ion species gets its own operator, from its own channels, with nothing
averaged.

The reference against which [`SharedEffectiveTransport`](@ref) is measured, and
the fallback if the mixing error ever turns out to matter. Costs one
factorization per species.
"""
struct PerSpeciesTransport <: IonTransportPolicy end

"""
    IonTransportGroup{FT}

A set of ion species that will be advanced by one shared operator, together with
the channels that operator is assembled from.

`sids` index into the species list; `channels` holds one entry per transport
mechanism, in the order the caller supplied them.
"""
struct IonTransportGroup{FT <: AbstractFloat}
    sids::Vector{Int}
    channels::Vector{DiffusionChannel{FT}}
end

"""
    ion_transport_groups(policy, channels_per_species, weights) -> Vector{IonTransportGroup}

Map ion species onto transport operators. **The one place a policy is consulted.**

`channels_per_species[s][m]` is species `s`'s channel for mechanism `m`; every
species must list the same mechanisms in the same order. A species without a
given mechanism passes a zero-speed channel rather than omitting it — a missing
entry is a construction bug, not a physical statement. `weights[s]` is that
species' density field.

| policy | groups | factorizations | right-hand sides |
|---|---|---|---|
| [`SharedEffectiveTransport`](@ref) | 1 | 1 | one per species |
| [`PerSpeciesTransport`](@ref) | one per species | one per species | 1 each |

With a single species the two agree exactly, so the default may change without
changing today's answer.
"""
function ion_transport_groups end

function ion_transport_groups(::PerSpeciesTransport, channels_per_species, weights)
    _check_species_channels(channels_per_species, weights)
    return [
        IonTransportGroup(Int[s], collect(channels_per_species[s]))
            for s in eachindex(channels_per_species)
    ]
end

function ion_transport_groups(::SharedEffectiveTransport, channels_per_species, weights)
    nmech = _check_species_channels(channels_per_species, weights)
    mixed = [
        mixture_channel([chs[m] for chs in channels_per_species], collect(weights))
            for m in 1:nmech
    ]
    return [IonTransportGroup(collect(eachindex(channels_per_species)), mixed)]
end

"""
    wall_absorption_speeds(channels_with_directions, faces, albedo) -> Vector

Robin coefficient `v_absorb = ¼v̄_n·(1 − R)` [m/s] for every entry of `faces`,
summed over the mechanisms in `channels_with_directions`.

`albedo` is the fraction returned to the plasma: a scalar for a uniform surface,
or one value per face where the surface is not uniform. `R = 1` gives exactly
zero — a reflective wall, and a matrix bit-identical to one assembled with no
wall term at all.

**Sampled per face, not per cell.** The ceiling depends on `b̂·n̂`, so the two
faces of a staircase corner generally carry different speeds even though they
share an owning cell and its plasma state. There are only four possible
outward normals, so the ceiling field is built once per direction and indexed,
rather than recomputed for each of the thousands of faces.
"""
function wall_absorption_speeds(
        channels_with_directions,
        faces::AbstractVector{WallFace{FT}},
        albedo
    ) where {FT <: AbstractFloat}
    in_range(r) = zero(FT) <= r <= one(FT)
    if albedo isa AbstractVector
        length(albedo) == length(faces) ||
            throw(DimensionMismatch("got $(length(albedo)) albedos for $(length(faces)) faces"))
        all(in_range, albedo) ||
            throw(ArgumentError("every albedo must lie in [0, 1]"))
    else
        in_range(albedo) ||
            throw(ArgumentError("albedo must lie in [0, 1], got $albedo"))
    end

    ceilings = Dict{Tuple{Int, Int}, Matrix{FT}}()
    v_absorb = Vector{FT}(undef, length(faces))
    for (k, f) in enumerate(faces)
        c = get!(() -> total_ceiling(channels_with_directions, f.outward), ceilings, f.outward)
        R = albedo isa AbstractVector ? albedo[k] : albedo
        v_absorb[k] = (one(FT) - FT(R)) * c[f.nid]
    end
    return v_absorb
end

"""
    ion_transport_operator(G, group, directions; faces, albedo, cross_terms) -> (A, v_absorb)

Assemble the wall-aware `∇·(𝐃∇·)` a group of ion species will share, and the
Robin coefficients that go with it.

`directions[m]` is the `(bR, bZ)` the group's `m`-th channel is aligned with —
the full `b̂` for a collisional mechanism, `b̂_pol` for a turbulent one. Directions
are field properties, so they are the same for every species and live outside the
group.

The tensor and the wall ceiling are built from **one** channel list, which is the
reason this exists as a function rather than two calls at each site: a matrix
assembled from one set of channels and a boundary condition from another is a
mismatch nothing downstream can detect.

Omitting `faces` gives the reflective operator and an empty coefficient vector.
"""
function ion_transport_operator(
        G::GridGeometry{FT}, group::IonTransportGroup{FT}, directions;
        faces::Union{Nothing, AbstractVector{WallFace{FT}}} = nothing,
        albedo = zero(FT),
        cross_terms::Symbol = :drop,
    ) where {FT <: AbstractFloat}
    length(directions) == length(group.channels) || throw(
        ArgumentError(
            "each of the group's $(length(group.channels)) mechanisms needs a direction, " *
                "got $(length(directions))"
        )
    )
    cwd = [
        (group.channels[m], directions[m][1], directions[m][2])
            for m in eachindex(group.channels)
    ]
    D_RR, D_RZ, D_ZZ = total_tensor(cwd)

    isnothing(faces) &&
        return build_wall_diffusion_matrix(G, D_RR, D_RZ, D_ZZ; cross_terms), FT[]

    v_absorb = wall_absorption_speeds(cwd, faces, albedo)
    A = build_wall_diffusion_matrix(
        G, D_RR, D_RZ, D_ZZ;
        cross_terms = cross_terms, faces = faces, v_absorb = v_absorb
    )
    return A, v_absorb
end

"""
    solve_ion_group!(N, group, A, solver, dt; θ = 1, S = nothing) -> N

Advance every species of `group` through the θ-scheme

```
    (𝐈 − θΔt𝐀)nⁿ⁺¹ = nⁿ + (1−θ)Δt𝐀nⁿ + ΔtS
```

with **one** factorization. `N` is `(NR·NZ) × Nspecies` with species as columns —
the layout `reshape` gives an `(NR, NZ, Nspecies)` density array for free — and
only the columns in `group.sids` are read or written.

The batch is the whole point of grouping. A second species in a group costs a
backsolve (68–488 µs across the measured grid sizes) against a factorization
(2.0–14.0 ms), so a shared operator over ten species is roughly the price of one.
"""
function solve_ion_group!(
        N::AbstractMatrix{FT}, group::IonTransportGroup{FT},
        A::SparseMatrixCSC{FT}, solver::AbstractLinearSolver{FT}, dt::Real;
        θ::Real = 1, S::Union{Nothing, AbstractMatrix{FT}} = nothing,
    ) where {FT <: AbstractFloat}
    Ng = size(N, 1)
    size(A, 1) == Ng ||
        throw(DimensionMismatch("operator is $(size(A, 1))×$(size(A, 2)) but N has $Ng rows"))
    maximum(group.sids) <= size(N, 2) ||
        throw(BoundsError(N, (:, maximum(group.sids))))

    sids = group.sids
    B = N[:, sids]
    θ < 1 && (B .+= FT((1 - θ) * dt) .* (A * B))
    isnothing(S) || (B .+= FT(dt) .* view(S, :, sids))

    factorize!(solver, I - FT(θ * dt) * A)
    X = similar(B)
    solve!(X, solver, B)
    @views N[:, sids] .= X
    return N
end

"Validate the ragged-array contract shared by every policy; return the mechanism count."
function _check_species_channels(channels_per_species, weights)
    isempty(channels_per_species) &&
        throw(ArgumentError("no ion species to transport"))
    length(channels_per_species) == length(weights) || throw(
        ArgumentError(
            "got $(length(channels_per_species)) species but $(length(weights)) density fields"
        )
    )
    nmech = length(first(channels_per_species))
    nmech > 0 || throw(ArgumentError("an ion species needs at least one transport mechanism"))
    all(chs -> length(chs) == nmech, channels_per_species) || throw(
        ArgumentError(
            "every species must list the same mechanisms in the same order; got lengths " *
                "$(map(length, channels_per_species)). A species without a mechanism passes " *
                "a zero-speed channel, it does not omit the entry"
        )
    )
    return nmech
end
