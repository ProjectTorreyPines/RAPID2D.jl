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

# ── the species store's layout ──────────────────────────────────────────────────────────

"""
    SpeciesStoreLayout{FT}

What the species store holds (`plasma.n_ions`, `plasma.n_gas`): the ion parents, each with its
charge states z = 1…Z as `n_ions[k][:, :, z]`, and the parents of the neutral gas as
`n_gas[j]`, with their records; and where the solvers' species columns lie in it — the
registry's (`ion_slots`, `gas_slots`, the reaction network's columns) and the transport's
(`transport_slots`). Built by [`species_store_layout`](@ref); constant over a run.
"""
struct SpeciesStoreLayout{FT <: AbstractFloat}
    ion_parents::Vector{String}
    ion_records::Vector{Vector{IonRecord{FT}}}
    gas_parents::Vector{String}
    gas_records::Vector{NeutralRecord{FT}}
    ion_slots::Vector{Tuple{Int, Int}}
    gas_slots::Vector{Int}
    transport_slots::Vector{Tuple{Int, Int}}
end

# The parent a record belongs to: its nuclei, or the record alone when it has none (a test probe)
store_parent(r::IonRecord) = isempty(r.key.composition) ? (r.key.composition, r.label) : (r.key.composition, Symbol())
parent_display_name(label::Symbol) = String(first(split(String(label), '[')))

"""
    species_store_layout(ions, neutrals, transport) -> SpeciesStoreLayout

The layout for the ion records `ions` and the neutral records `neutrals`, in the solvers' column
order, and the transport's ion records `transport`, each one of `ions`. The ions group by parent
(their nuclei), in the order the parents first appear; a parent's charge states run 1…Z with none
missing, so that `n_ions[k][:, :, z]` is charge state z. A record with no nuclei (a test probe)
is a parent of its own, one plane. Each neutral is a parent of the gas.
"""
function species_store_layout(
        ions::AbstractVector{IonRecord{FT}}, neutrals, transport::AbstractVector{IonRecord{FT}},
    ) where {FT <: AbstractFloat}
    parents = unique(map(store_parent, ions))
    ion_records = [sort!([r for r in ions if store_parent(r) == p]; by = r -> r.charge) for p in parents]
    for (p, records) in zip(parents, ion_records)
        isempty(first(p)) && continue
        charges = [r.charge for r in records]
        charges == 1:length(charges) || throw(
            ArgumentError(
                "$(parent_display_name(first(records).label)) has charge states $(join(charges, ", ")) in the list: " *
                    "a parent's ions are stored by charge state, 1…Z with none missing"
            )
        )
    end
    slot = Dict{Tuple{SpeciesKey, Symbol}, Tuple{Int, Int}}(
        (r.key, r.label) => (k, z) for (k, records) in enumerate(ion_records) for (z, r) in enumerate(records)
    )
    find_slot(r) = get(slot, (r.key, r.label)) do
        throw(ArgumentError("the transport moves $(r.label), which is not among the stored ions"))
    end
    return SpeciesStoreLayout{FT}(
        [parent_display_name(first(records).label) for records in ion_records], ion_records,
        [parent_display_name(r.label) for r in neutrals], NeutralRecord{FT}[neutrals...],
        Tuple{Int, Int}[slot[(r.key, r.label)] for r in ions], collect(eachindex(neutrals)),
        Tuple{Int, Int}[find_slot(r) for r in transport],
    )
end

# The layout of a run without a species list: the transport's ions, and the H₂ gas
function legacy_store_layout(transport::AbstractVector{IonRecord{FT}}, constants::PlasmaConstants) where {FT <: AbstractFloat}
    return species_store_layout(transport, (h2_gas_record(FT, constants),), transport)
end

# The ion a run starts with, and the one the transport moves without a species list: H₂⁺ at
# `constants.mi`
default_ion_record(::Type{FT}, constants::PlasmaConstants) where {FT <: AbstractFloat} =
    IonRecord(Particles.H2[1]; mass = FT(constants.mi))

# The layout a run starts with, before `initialize!` declares the transport's species: the
# default H₂⁺, and the H₂ gas
default_store_layout(constants::PlasmaConstants{FT}) where {FT <: AbstractFloat} =
    species_store_layout([default_ion_record(FT, constants)], (h2_gas_record(FT, constants),), IonRecord{FT}[])

function h2_gas_record(::Type{FT}, constants::PlasmaConstants) where {FT <: AbstractFloat}
    h2 = Particles.H2[0]
    return NeutralRecord{FT}(species_key(h2), species_label(h2, Protium()), FT(species_mass(h2, Protium(), constants)))
end
