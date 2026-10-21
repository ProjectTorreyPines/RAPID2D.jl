# The species store: every species' density, the ions of each parent by charge state and the neutral
# gas of each parent (`plasma.n_ions`, `plasma.n_gas`), laid out once from the run's species; the
# solvers reach it through their species columns (`RP.store_layout`).

"""
    build_species_store!(RP) -> RP

Lay the species store out for the run's species: those of `config.species` in its order, or,
without a list, the transport's ions and the H₂ gas ([`species_store_layout`](@ref)). Each ion
parent gets an `NR × NZ × Z` array, each gas parent an `NR × NZ` matrix; a species the old
layout held as well keeps its density, every other starts at zero. `plasma.ni` is then the
transport's first species and `plasma.n_H2_gas` the H₂ gas.
"""
function build_species_store!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    pla, tp = RP.plasma, RP.transport
    layout = if !isnothing(RP.species)
        reg = RP.species.registry
        haskey(reg.neutral_column, species_key(Particles.H2[0])) ||
            throw(ArgumentError("config.species lacks H2[0], the H₂ gas the neutral model evolves"))
        species_store_layout(reg.ions, reg.neutrals, tp.ion_species)
    elseif isempty(tp.ion_species)          # before the transport's species are declared
        default_store_layout(RP.config.constants)
    else
        legacy_store_layout(tp.ion_species, RP.config.constants)
    end
    dims = size(pla.ne)
    n_ions = [zeros(FT, dims..., length(records)) for records in layout.ion_records]
    n_gas = [zeros(FT, dims) for _ in layout.gas_records]
    ni_slot = isempty(layout.transport_slots) ? default_bulk_slot(layout) : first(layout.transport_slots)

    # what the old layout held carries over, species by species; and `plasma.ni` keeps its density
    # when the transport's first species is declared another one
    old, old_bulk = RP.store_layout, getfield(pla, :ni_slot)
    new_slot = Dict{Tuple{SpeciesKey, Symbol}, Tuple{Int, Int}}(
        store_identity(r) => (k, z) for (k, records) in enumerate(layout.ion_records) for (z, r) in enumerate(records)
    )
    for (k, records) in enumerate(old.ion_records), (z, r) in enumerate(records)
        slot = get(new_slot, store_identity(r), nothing)
        isnothing(slot) || copyto!(view(n_ions[slot[1]], :, :, slot[2]), ion_plane(pla, (k, z)))
    end
    haskey(new_slot, store_identity(old.ion_records[old_bulk[1]][old_bulk[2]])) ||
        copyto!(view(n_ions[ni_slot[1]], :, :, ni_slot[2]), ion_plane(pla, old_bulk))
    for (j, r) in enumerate(old.gas_records)
        i = findfirst(g -> g.key == r.key, layout.gas_records)
        isnothing(i) || copyto!(n_gas[i], getfield(pla, :n_gas)[j])
    end

    # the store's own layout: rebinding these is refused everywhere else
    setfield!(pla, :n_ions, n_ions)
    setfield!(pla, :n_gas, n_gas)
    setfield!(pla, :ni_slot, ni_slot)
    setfield!(pla, :h2_slot, findfirst(g -> g.key == species_key(Particles.H2[0]), layout.gas_records))
    RP.store_layout = layout
    return RP
end

# A species' identity in the store: its key (nuclei and charge), or its key and label for a
# record with no nuclei (test probes share their keys)
store_identity(r::IonRecord) = isempty(r.key.composition) ? (r.key, r.label) : (r.key, Symbol())

# The bulk ion before the transport declares one: H2[1] if the store holds it, else the first ion
function default_bulk_slot(layout::SpeciesStoreLayout)
    h2 = species_key(Particles.H2[1])
    for (k, records) in enumerate(layout.ion_records), (z, r) in enumerate(records)
        r.key == h2 && return (k, z)
    end
    return (1, 1)
end

# f(n, z, m) for every stored ion species but the bulk (`plasma.ni`): its density plane, its
# charge and its mass. Every ion the store holds counts, moved by the transport or not
function foreach_other_ion(f, RP::RAPID{FT}) where {FT <: AbstractFloat}
    pla = RP.plasma
    bulk = getfield(pla, :ni_slot)
    for (k, records) in enumerate(RP.store_layout.ion_records), (z, r) in enumerate(records)
        (k, z) == bulk && continue
        f(ion_plane(pla, (k, z)), FT(r.charge), r.mass)
    end
    return nothing
end

# The transport's species between the store and the columns of the batch solve: gathered into
# `N` before it, scattered back after
function gather_ion_densities!(N::AbstractMatrix, RP::RAPID)
    for (s, slot) in enumerate(RP.store_layout.transport_slots)
        copyto!(view(N, :, s), ion_plane(RP.plasma, slot))
    end
    return N
end

function scatter_ion_densities!(RP::RAPID, N::AbstractMatrix)
    for (s, slot) in enumerate(RP.store_layout.transport_slots)
        copyto!(ion_plane(RP.plasma, slot), view(N, :, s))
    end
    return RP
end

# Every stored ion species set to zero on the band outside the wall, plane by plane
function clear_ions_outside_wall!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    out = RP.G.nodes.on_out_wall_nids
    for A in RP.plasma.n_ions, z in axes(A, 3)
        view(A, :, :, z)[out] .= zero(FT)
    end
    return RP
end

# Two plasma states hold stores of one layout: the same parents, charge states and slots, so that
# one is copied into the other in place and the slots mean the same species
function check_store_layout(a::PlasmaState, b::PlasmaState)
    same = map(size, getfield(a, :n_ions)) == map(size, getfield(b, :n_ions)) &&
        map(size, getfield(a, :n_gas)) == map(size, getfield(b, :n_gas)) &&
        getfield(a, :ni_slot) == getfield(b, :ni_slot) && getfield(a, :h2_slot) == getfield(b, :h2_slot)
    same || error("the species store changed its layout during the step; its densities cannot be put back")
    return nothing
end
