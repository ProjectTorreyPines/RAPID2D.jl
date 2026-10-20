export AbstractSpecies, NeutralSpecies, IonSpecies, Neutral, Ion, IonRecord, SpeciesKey, species_key
export IonTransportPolicy, SharedEffectiveTransport, PerSpeciesTransport

# What a species is, and whether ion species share a transport equation.
#
# Declarations only, included before `types.jl`, which types a flag and a state field with
# them. A species is a parent particle (an element or a molecule) in one charge state,
# written `H2[1]`; charge state 0 is the neutral. Neutral and ion are different TYPES
# because they solve different equations; the charge state of an ion is data, so the
# charge states of one element share one type. The species types are for setup and
# dispatch: the solvers see `IonRecord`s and column indices, never these types.

# ── parents ─────────────────────────────────────────────────────────────────────────────

"""
    AbstractParent

A particle that exists in one or more charge states: an [`Element`](@ref) (a charge-state
ladder 0…Z) or a [`Molecule`](@ref). Index it with a charge state to get a species:
`H2[1]`.
"""
abstract type AbstractParent end

"An atomic parent: its charge states form a ladder 0…Z."
abstract type Element <: AbstractParent end

"A molecular parent: a few charge states, each with reactions of its own."
abstract type Molecule <: AbstractParent end

struct Hydrogen <: Element end
struct H2Molecule <: Molecule end
struct H3Molecule <: Molecule end

"The charge states a parent exists in."
function allowed_charges end
allowed_charges(::Hydrogen) = 0:1
allowed_charges(::H2Molecule) = 0:1
allowed_charges(::H3Molecule) = 1:1     # the neutral H₃ is not bound

"Nuclei per particle, by element: `(:H => 2,)` for H2."
function composition end
composition(::Hydrogen) = (:H => 1,)
composition(::H2Molecule) = (:H => 2,)
composition(::H3Molecule) = (:H => 3,)

parent_name(::Hydrogen) = "H"
parent_name(::H2Molecule) = "H2"
parent_name(::H3Molecule) = "H3"

# ── species ─────────────────────────────────────────────────────────────────────────────

"A parent in one charge state."
abstract type AbstractSpecies end

"A neutral species: it diffuses by its own operator."
abstract type NeutralSpecies <: AbstractSpecies end

"""
    IonSpecies

A charged species: it moves with the ion group (one shared operator, one right-hand-side
column each). The concrete type is [`Ion`](@ref); what the solvers carry is an
[`IonRecord`](@ref).
"""
abstract type IonSpecies <: AbstractSpecies end

"The neutral of `parent` (charge state 0)."
struct Neutral{P <: AbstractParent} <: NeutralSpecies
    parent::P
end

"`parent` in charge state `z ≥ 1`."
struct Ion{P <: AbstractParent} <: IonSpecies
    parent::P
    z::Int
end

"""
    parent[z]

The species of `parent` in charge state `z`: a [`Neutral`](@ref) for `z = 0`, an
[`Ion`](@ref) otherwise. Throws for a charge state the parent does not have.
"""
function Base.getindex(p::AbstractParent, z::Integer)
    z in allowed_charges(p) ||
        throw(ArgumentError("$(parent_name(p)) has charge states $(allowed_charges(p)), not $z"))
    return iszero(z) ? Neutral(p) : Ion(p, Int(z))
end

charge(::Neutral) = 0
charge(sp::Ion) = sp.z

"The display label, `H2[1]`. Identity is the [`SpeciesKey`](@ref), not the label."
species_label(sp::AbstractSpecies) = Symbol(parent_name(sp.parent), "[", charge(sp), "]")

"""
    SpeciesKey(composition, charge)

A species' identity: its nuclei by element and its charge. Reaction tables and source
lookups match on this, not on labels. An empty composition marks a species no reaction
table knows (a test probe), which therefore receives no reaction source.
"""
struct SpeciesKey
    composition::Tuple{Vararg{Pair{Symbol, Int}}}
    charge::Int
end

species_key(sp::AbstractSpecies) = SpeciesKey(composition(sp.parent), charge(sp))

const UNREGISTERED_COMPOSITION = ()

"""
    IonRecord{FT}(key, label, mass, charge)
    IonRecord(sp::Ion; mass)
    IonRecord(label::Symbol, mass, charge)

What the ion transport carries for one species: identity, display label, mass and charge.

`mass` and `charge` vary independently across the periodic table (C⁶⁺ and H₂⁺ differ 6× in
both, O⁸⁺ and H₂⁺ 8× in charge and 8× in mass), which is why `Zeff` alone cannot stand in
for a species. The three-argument form builds an unregistered record (a test probe): it
has no key a reaction table can match.

Exactly one record is declared at a time; see [`set_ion_species!`](@ref) for what a second
one needs first.
"""
struct IonRecord{FT <: AbstractFloat}
    key::SpeciesKey
    label::Symbol
    mass::FT
    charge::Int

    function IonRecord{FT}(key::SpeciesKey, label::Symbol, mass::FT, charge::Int) where {FT <: AbstractFloat}
        mass > zero(FT) || throw(ArgumentError("$label: mass must be positive, got $mass"))
        charge > 0 || throw(ArgumentError("$label: an ion carries charge ≥ 1, got $charge"))
        return new{FT}(key, label, mass, charge)
    end
end

IonRecord(sp::Ion; mass::Real) =
    IonRecord{float(typeof(mass))}(species_key(sp), species_label(sp), float(mass), sp.z)

IonRecord(label::Symbol, mass::Real, charge::Integer) = IonRecord{float(typeof(mass))}(
    SpeciesKey(UNREGISTERED_COMPOSITION, Int(charge)), label, float(mass), Int(charge)
)

"""
    RAPID2D.Particles

The parent particles by their usual names, `H`, `H2`, `H3`, for writing species as
`H2[1]`. Not exported: `using RAPID2D.Particles: H, H2`.
"""
module Particles
    import ..RAPID2D: Hydrogen, H2Molecule, H3Molecule
    const H = Hydrogen()
    const H2 = H2Molecule()
    const H3 = H3Molecule()
end

# The two charge averages a species MIXTURE needs — Z̄ = Σn_z Z_z/Σn_z per ion and
# Z_eff = Σn_z Z_z²/n_e per electron — come back with the mixture; their forms, degenerate
# cases and the 99:1 / 90:10 H₂⁺–C⁶⁺ numbers are in
# internal/docs/src/notes/TODO/ion-inventory-multi-species.md.

# ── transport policies ──────────────────────────────────────────────────────────────────

"""
    IonTransportPolicy

Whether ion species share a transport operator, and which one.

The choice is a type rather than a flag value so that it resolves at
`ion_transport_groups` and nowhere else. Everything downstream — assembly,
factorization, the solve — is written once against the groups and never learns
which policy produced them.
"""
abstract type IonTransportPolicy end

"""
    SharedEffectiveTransport()

One operator for every ion species, built from density-weighted effective
channels. **The default.**

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

Measured against the shared policy on a 25×25 box wall (H₂⁺ against a 6× heavier
carbon ion, 20 steps onto a fully absorbing wall, fraction of inventory
surviving): the two differ by ±10 % at a 50:50 mix, and at a realistic 99:1 mix
the bulk is untouched (+0.2 %) while the whole error lands on the trace species
(−18.5 %) — which is what friction coupling says should physically happen to a
trace impurity in a hydrogen background.
"""
struct PerSpeciesTransport <: IonTransportPolicy end
