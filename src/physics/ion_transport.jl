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
