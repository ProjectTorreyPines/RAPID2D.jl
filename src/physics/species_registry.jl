export NeutralRecord, SpeciesRegistry, SpeciesSystem

# The species of a run as the solvers see them: a record per species, ions and neutrals apart,
# built once from the list a run is given. Included before `types.jl`, whose `RAPID` holds the
# assembled species system.

"""
    NeutralRecord{FT}(key, label, mass)

What a neutral species' equation carries: identity, display label and mass.
"""
struct NeutralRecord{FT <: AbstractFloat}
    key::SpeciesKey
    label::Symbol
    mass::FT
end

"""
    SpeciesRegistry

A run's species as records, built once from its list by [`build_registry`](@ref): the ions as
a vector of [`IonRecord`](@ref)s (their columns in the ion group, in list order), the neutrals
as a tuple of [`NeutralRecord`](@ref)s, and each one's column by identity. Masses and labels are
those of the run's hydrogen isotope.
"""
struct SpeciesRegistry{FT <: AbstractFloat, NT <: Tuple, H <: AbstractHydrogenIsotope}
    ions::Vector{IonRecord{FT}}
    neutrals::NT
    ion_column::Dict{SpeciesKey, Int}
    neutral_column::Dict{SpeciesKey, Int}
    isotope::H
end

"Whether the registry has species `sp`."
has_species(reg::SpeciesRegistry, sp::Ion) = haskey(reg.ion_column, species_key(sp))
has_species(reg::SpeciesRegistry, sp::Neutral) = haskey(reg.neutral_column, species_key(sp))

"""
    build_registry(list, isotope, constants) -> SpeciesRegistry

The registry of the species `list`, a tuple such as `(H2[0], H[0], H2[1], H[1])`. Throws for an
entry that is not a species, a species listed twice, a charge state its parent does not have,
or a parent that leaves out part of its API.
"""
function build_registry(list::Tuple, iso::AbstractHydrogenIsotope, constants::PlasmaConstants{FT}) where {FT <: AbstractFloat}
    isempty(list) && throw(ArgumentError("the species list is empty"))
    foreach(sp -> check_species(sp, iso, constants), list)
    allunique(map(species_key, list)) ||
        throw(ArgumentError("a species is listed twice in ($(join(map(sp -> species_label(sp, iso), list), ", ")))"))

    record(sp::Ion) = IonRecord{FT}(species_key(sp), species_label(sp, iso), FT(species_mass(sp, iso, constants)), charge(sp))
    record(sp::Neutral) = NeutralRecord{FT}(species_key(sp), species_label(sp, iso), FT(species_mass(sp, iso, constants)))
    ions = IonRecord{FT}[map(record, filter(sp -> sp isa Ion, list))...]
    neutrals = map(record, filter(sp -> sp isa Neutral, list))
    return SpeciesRegistry(
        ions, neutrals,
        Dict{SpeciesKey, Int}(r.key => k for (k, r) in enumerate(ions)),
        Dict{SpeciesKey, Int}(r.key => k for (k, r) in enumerate(neutrals)),
        iso,
    )
end

# A parent defines allowed_charges, composition, parent_name and neutral_mass
function check_species(sp, iso, constants)
    sp isa AbstractSpecies ||
        throw(ArgumentError("$(repr(sp)) in the species list is not a species; write one as `H2[1]` (`using RAPID2D.Particles: H2`)"))
    p = sp.parent
    for (f, args) in ((allowed_charges, (p,)), (composition, (p,)), (parent_name, (p,)), (neutral_mass, (p, iso, constants)))
        hasmethod(f, Tuple{map(typeof, args)...}) || throw(
            ArgumentError(
                "$(nameof(typeof(p))) does not define `$(nameof(f))`: a parent defines allowed_charges, " *
                    "composition, parent_name and neutral_mass"
            )
        )
    end
    charge(sp) in allowed_charges(p) ||
        throw(ArgumentError("$(parent_name(p)) has charge states $(allowed_charges(p)), not $(charge(sp))"))
    return nothing
end

"""
    SpeciesSystem

The species a run carries and what happens among them, assembled by `initialize!` from
`config.species` ([`assemble_species_system`](@ref)): the [`SpeciesRegistry`](@ref), the active
reactions and collisions with their rates for the run's isotope, and what the assembly left
out ([`AssemblyReport`](@ref)). Constant over a run.
"""
struct SpeciesSystem{R <: SpeciesRegistry, RX <: Tuple, CX <: Tuple, A}
    registry::R
    reactions::RX
    collisions::CX
    report::A
end
