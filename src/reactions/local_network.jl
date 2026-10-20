# The local reaction network: the reactions the electron solve does not count, solved cell by
# cell, implicit in the ions they consume, block by block in the order production flows.

"""
    NetworkRow{FT}

A reaction of the local network as the cell loop reads it: the ion column whose new density its
events take (`implicit`), its partner (`frozen = 0` the electrons, else a neutral column held at
its step-entry density), and what one event changes: ion and neutral columns, electrons, and the
ion mass it takes in from a neutral at rest (`mass_born`) and lets out with the ions' velocity
(`mass_lost`).
"""
struct NetworkRow{FT <: AbstractFloat}
    label::Symbol
    implicit::Int
    frozen::Int
    ion_change::Vector{Pair{Int, Int}}
    neutral_change::Vector{Pair{Int, Int}}
    electron_change::Int
    mass_born::FT
    mass_lost::FT
end

"""
    NetworkBlock

Ion columns the network couples both ways (a strongly connected component of its production
graph), solved together: by Thomas when every transfer joins neighbours in `columns`, else by
pivoted LU. `rows` are the rows whose implicit column is in the block, `inflow` the rows of
earlier blocks that produce one of its columns.
"""
struct NetworkBlock
    columns::Vector{Int}
    tridiagonal::Bool
    rows::Vector{Int}
    inflow::Vector{Int}
end

"""
    LocalNetworkPlan

The local network of a species system, built once by [`local_network_plan`](@ref): its rows, the
active reactions they come from (whose rates and energies are evaluated outside the cell loop),
and the blocks of ion columns in the order production flows.
"""
struct LocalNetworkPlan{FT <: AbstractFloat, RX <: Tuple}
    rows::Vector{NetworkRow{FT}}
    reactions::RX
    blocks::Vector{NetworkBlock}
    n_ion::Int
    n_neutral::Int
end

"""
    local_network_plan(system::SpeciesSystem) -> LocalNetworkPlan

The rows of the system's network reactions on its registry's columns, and their blocks: the
strongly connected components of the production graph `implicit → product ion` (Tarjan), in
topological order, so that a block's inflow comes from blocks already solved.
"""
function local_network_plan(sys::SpeciesSystem{<:SpeciesRegistry{FT}}) where {FT <: AbstractFloat}
    reg = sys.registry
    reactions = filter(a -> a.reaction.producer isa LocalNetwork, sys.reactions)
    rows = NetworkRow{FT}[map(a -> network_row(a.reaction, reg), reactions)...]
    n_ion = length(reg.ions)

    successors = [Int[] for _ in 1:n_ion]
    for row in rows, (s, ν) in row.ion_change
        (s != row.implicit && ν > 0 && !(s in successors[row.implicit])) && push!(successors[row.implicit], s)
    end
    components = strongly_connected_components(successors)
    blocks = map(components) do component
        columns = sort(component; by = s -> (reg.ions[s].charge, s))
        network_block(columns, rows)
    end
    return LocalNetworkPlan(rows, reactions, blocks, n_ion, length(reg.neutrals))
end

function network_row(r::Reaction, reg::SpeciesRegistry{FT}) where {FT <: AbstractFloat}
    p = r.producer
    ion, neutral = Dict{Int, Int}(), Dict{Int, Int}()
    mass_ions = zero(FT)          # the ion fluid's mass change per event
    mass_born = zero(FT)          # the neutral reactants' mass, taken in at rest
    for (side, sign) in ((r.reactants, -1), (r.products, 1)), (x, n) in side
        if x isa Ion
            col = reg.ion_column[species_key(x)]
            ion[col] = get(ion, col, 0) + sign * n
            mass_ions += sign * n * reg.ions[col].mass
        elseif x isa Neutral
            col = reg.neutral_column[species_key(x)]
            neutral[col] = get(neutral, col, 0) + sign * n
            sign < 0 && (mass_born += n * reg.neutrals[col].mass)
        end
    end
    changes(d) = sort!([k => v for (k, v) in d if v != 0])
    implicit = reg.ion_column[species_key(p.implicit)]
    get(ion, implicit, 0) < 0 ||
        throw(ArgumentError("$(r.label): the network takes its implicit species by the events, and this reaction does not consume it"))
    return NetworkRow{FT}(
        r.label, implicit,
        isnothing(p.frozen) ? 0 : reg.neutral_column[species_key(p.frozen)],
        changes(ion), changes(neutral), electron_change(r), mass_born, mass_born - mass_ions,
    )
end

function network_block(columns::Vector{Int}, rows::Vector{<:NetworkRow})
    inside(s) = s in columns
    block_rows = [j for (j, row) in enumerate(rows) if inside(row.implicit)]
    inflow = [j for (j, row) in enumerate(rows) if !inside(row.implicit) && any(entry -> inside(first(entry)) && last(entry) > 0, row.ion_change)]
    position(s) = findfirst(==(s), columns)
    tridiagonal = all(
        abs(position(s) - position(rows[j].implicit)) <= 1 for j in block_rows for (s, _) in rows[j].ion_change if inside(s)
    )
    # in a cycle no event may add to the block's ions: its matrix is then diagonally dominant by
    # columns, an M-matrix, and the ions stay nonnegative
    length(columns) == 1 || all(sum(ν for (s, ν) in rows[j].ion_change if inside(s)) <= 0 for j in block_rows) || throw(
        ArgumentError("the reactions $(join((rows[j].label for j in block_rows), ", ")) form a cycle that multiplies its ions")
    )
    return NetworkBlock(columns, tridiagonal, block_rows, inflow)
end

# The strongly connected components of the graph `successors` (Tarjan), sources first: Tarjan
# completes a component after every component it reaches
function strongly_connected_components(successors::Vector{Vector{Int}})
    n = length(successors)
    index, low = zeros(Int, n), zeros(Int, n)
    on_stack = falses(n)
    stack, components = Int[], Vector{Int}[]
    counter = 0
    function visit(v)
        counter += 1
        index[v] = low[v] = counter
        push!(stack, v)
        on_stack[v] = true
        for w in successors[v]
            if index[w] == 0
                visit(w)
                low[v] = min(low[v], low[w])
            elseif on_stack[w]
                low[v] = min(low[v], index[w])
            end
        end
        if low[v] == index[v]
            component = Int[]
            while true
                w = pop!(stack)
                on_stack[w] = false
                push!(component, w)
                w == v && break
            end
            push!(components, component)
        end
        return nothing
    end
    for v in 1:n
        index[v] == 0 && visit(v)
    end
    return reverse!(components)
end

# ── a step ──────────────────────────────────────────────────────────────────────────────

"""
    NetworkInputs(K, energy, ne, ion_increment, donor_use)

What a step of the local network is given, cell by cell: each row's rate K [m³/s] and electron
energy per event [J], the catalyst electron density nₑ [m⁻³], each ion column's increment from the
electron solve [m⁻³], and each neutral column's consumption by it [m⁻³].
"""
struct NetworkInputs{V <: AbstractVector, D <: AbstractArray}
    K::V
    energy::V
    ne::D
    ion_increment::V
    donor_use::V
end

"""
    NetworkAccumulator{FT}(plan, dims)

What a step of the local network did, cell by cell: electrons born and captured [m⁻³], each row's
events [m⁻³], the electrons' energy over the events [J/m³], the ion mass taken in from neutrals at
rest and let out with the ions' velocity [kg/m³], and each neutral column's change [m⁻³].
"""
struct NetworkAccumulator{FT <: AbstractFloat, D <: AbstractArray{FT}}
    born_electrons::D
    lost_electrons::D
    events::Vector{D}
    electron_energy::D
    ion_mass_born::D
    ion_mass_lost::D
    neutral_change::Vector{D}
end

function NetworkAccumulator{FT}(plan::LocalNetworkPlan, dims::Tuple) where {FT <: AbstractFloat}
    field() = zeros(FT, dims)
    return NetworkAccumulator(
        field(), field(), [field() for _ in plan.rows], field(), field(), field(), [field() for _ in 1:plan.n_neutral],
    )
end

"""
    solve_local_network!(acc, n_ion, n_neutral, plan, inputs, Δt) -> StepVerdict

A step of the local network in every cell, backward Euler in the ions its rows consume:

    n⁺ₛ = nₛ + Δnₛ + Σⱼ νⱼₛ Δt Kⱼ n_cat,j n⁺_i(j)

with Δnₛ the electron solve's increment, i(j) row j's implicit ion and n_cat,j its catalyst, the
electrons or the frozen neutral at its step-entry density. Block by block in the order production
flows, a block's inflow from earlier ones on its right side; the events Nⱼ = Δt Kⱼ n_cat,j n⁺_i(j)
then move the neutrals and fill `acc`. A block is one species or a cycle whose events add no ions
to it, so its matrix is an M-matrix and nonnegative densities stay nonnegative.

Refused with `n_ion` and `n_neutral` untouched (and `acc` meaningless) when a cell consumes more of
a neutral than it holds, its use by the electron solve counted in, or captures more electrons than
`inputs.ne` (`Reject(:budget)`), or ends with a negative ion (`Reject(:positivity)`).
"""
function solve_local_network!(
        acc::NetworkAccumulator{FT}, n_ion::AbstractVector{<:AbstractArray{FT}}, n_neutral::AbstractVector{<:AbstractArray{FT}},
        plan::LocalNetworkPlan{FT}, inp::NetworkInputs, Δt::FT,
    ) where {FT <: AbstractFloat}
    cells = eachindex(inp.ne)
    largest = maximum(b -> length(b.columns), plan.blocks; init = 1)
    verdict = @with_pool pool begin
        n_new = acquire!(pool, FT, length(cells), plan.n_ion)   # committed once every cell passes
        transfer = acquire!(pool, FT, length(plan.rows))       # Δt K n_cat of each row in the cell
        block_matrix = acquire!(pool, FT, largest, largest)
        block_rhs = acquire!(pool, FT, largest)
        cell_verdict = Accept()
        for (cell, k) in enumerate(cells)
            for (j, row) in enumerate(plan.rows)
                catalyst = row.frozen == 0 ? inp.ne[k] : n_neutral[row.frozen][k]
                transfer[j] = Δt * inp.K[j][k] * catalyst
            end
            for block in plan.blocks
                solve_block!(n_new, cell, k, block, plan.rows, transfer, n_ion, inp.ion_increment, block_matrix, block_rhs)
            end
            cell_verdict = cell_events!(acc, n_new, cell, k, plan, transfer, inp, n_neutral)
            cell_verdict isa Accept || break
        end
        if cell_verdict isa Accept
            for s in 1:plan.n_ion, (cell, k) in enumerate(cells)
                n_ion[s][k] = n_new[cell, s]
            end
            for col in 1:plan.n_neutral, k in cells
                n_neutral[col][k] += acc.neutral_change[col][k]
            end
        end
        cell_verdict
    end
    return verdict
end

stoichiometric(changes::Vector{Pair{Int, Int}}, col::Int) = (i = findfirst(p -> first(p) == col, changes); isnothing(i) ? 0 : last(changes[i]))

# One block in cell k: (I − Σⱼ νⱼ transferⱼ e_i(j)ᵀ) n⁺ = n + Δn + inflow
function solve_block!(n_new, cell, k, block::NetworkBlock, rows, transfer, n_ion, ion_increment, block_matrix, block_rhs)
    FT = eltype(n_new)
    columns = block.columns
    if length(columns) == 1
        s = only(columns)
        diag, rhs = one(FT), n_ion[s][k] + ion_increment[s][k]
        for j in block.rows
            diag -= stoichiometric(rows[j].ion_change, s) * transfer[j]
        end
        for j in block.inflow
            rhs += stoichiometric(rows[j].ion_change, s) * transfer[j] * n_new[cell, rows[j].implicit]
        end
        n_new[cell, s] = rhs / diag
        return nothing
    end
    n = length(columns)
    position(s) = findfirst(==(s), columns)
    fill!(view(block_matrix, 1:n, 1:n), zero(FT))
    for q in 1:n
        block_matrix[q, q] = one(FT)
        block_rhs[q] = n_ion[columns[q]][k] + ion_increment[columns[q]][k]
    end
    for j in block.rows
        i = position(rows[j].implicit)
        for (s, ν) in rows[j].ion_change
            q = position(s)
            isnothing(q) || (block_matrix[q, i] -= ν * transfer[j])
        end
    end
    for j in block.inflow, (s, ν) in rows[j].ion_change
        q = position(s)
        isnothing(q) || (block_rhs[q] += ν * transfer[j] * n_new[cell, rows[j].implicit])
    end
    block.tridiagonal ? thomas!(block_matrix, block_rhs, n) : gepp!(block_matrix, block_rhs, n)
    for q in 1:n
        n_new[cell, columns[q]] = block_rhs[q]
    end
    return nothing
end

# A tridiagonal system in its first n rows by elimination without pivoting (an M-matrix needs
# none); the right side becomes the solution
function thomas!(matrix, rhs, n)
    for q in 2:n
        f = matrix[q, q - 1] / matrix[q - 1, q - 1]
        matrix[q, q] -= f * matrix[q - 1, q]
        rhs[q] -= f * rhs[q - 1]
    end
    rhs[n] /= matrix[n, n]
    for q in (n - 1):-1:1
        rhs[q] = (rhs[q] - matrix[q, q + 1] * rhs[q + 1]) / matrix[q, q]
    end
    return rhs
end

# A system in its first n rows by Gaussian elimination with partial pivoting, in place; the right
# side becomes the solution
function gepp!(matrix, rhs, n)
    for col in 1:n
        p = col
        for i in (col + 1):n
            abs(matrix[i, col]) > abs(matrix[p, col]) && (p = i)
        end
        if p != col
            for q in col:n
                matrix[col, q], matrix[p, q] = matrix[p, q], matrix[col, q]
            end
            rhs[col], rhs[p] = rhs[p], rhs[col]
        end
        for i in (col + 1):n
            f = matrix[i, col] / matrix[col, col]
            for q in col:n
                matrix[i, q] -= f * matrix[col, q]
            end
            rhs[i] -= f * rhs[col]
        end
    end
    for i in n:-1:1
        s = rhs[i]
        for q in (i + 1):n
            s -= matrix[i, q] * rhs[q]
        end
        rhs[i] = s / matrix[i, i]
    end
    return rhs
end

# The events of cell k from the solved ions, their effects, and the cell's checks
function cell_events!(acc, n_new, cell, k, plan, transfer, inp, n_neutral)
    FT = eltype(n_new)
    born = lost = energy = mass_born = mass_lost = zero(FT)
    for col in 1:plan.n_neutral
        acc.neutral_change[col][k] = zero(FT)
    end
    for (j, row) in enumerate(plan.rows)
        events = transfer[j] * n_new[cell, row.implicit]
        acc.events[j][k] = events
        row.electron_change > 0 && (born += row.electron_change * events)
        row.electron_change < 0 && (lost -= row.electron_change * events)
        energy += events * inp.energy[j][k]
        mass_born += events * row.mass_born
        mass_lost += events * row.mass_lost
        for (col, ν) in row.neutral_change
            acc.neutral_change[col][k] += ν * events
        end
    end
    acc.born_electrons[k], acc.lost_electrons[k], acc.electron_energy[k] = born, lost, energy
    acc.ion_mass_born[k], acc.ion_mass_lost[k] = mass_born, mass_lost

    for s in 1:plan.n_ion
        n_new[cell, s] >= zero(FT) || return Reject(:positivity)
    end
    lost <= inp.ne[k] || return Reject(:budget)
    for col in 1:plan.n_neutral
        # what the network takes of the neutral, and what the electron solve took
        taken = inp.donor_use[col][k]
        for (j, row) in enumerate(plan.rows), (n, ν) in row.neutral_change
            n == col && ν < 0 && (taken -= ν * acc.events[j][k])
        end
        taken <= n_neutral[col][k] || return Reject(:budget)
    end
    return Accept()
end

# ── rates and energies, outside the cell loop ───────────────────────────────────────────

"""
    network_rates!(K, plan, electron_ctx, heavy_ctx) -> K

Each row's rate [m³/s] into `K[j]`: electron impact in `electron_ctx`, a heavy pair in `heavy_ctx`.
"""
function network_rates!(K::AbstractVector, plan::LocalNetworkPlan, electron_ctx::AbstractRateContext, heavy_ctx::AbstractRateContext)
    map(plan.reactions, ntuple(identity, Val(length(plan.reactions)))) do a, j
        rate_coefficient!(K[j], a.rate, has_electron(a.reaction.reactants) ? electron_ctx : heavy_ctx)
        nothing
    end
    return K
end

"""
    network_energies!(energy_per_event, plan, electron_ctx, ee) -> energy_per_event

Each row's electron energy per event [J] into `energy_per_event[j]` ([`event_energy!`](@ref)), zero
for a reaction without electrons.
"""
function network_energies!(energy_per_event::AbstractVector, plan::LocalNetworkPlan, electron_ctx::AbstractRateContext, ee)
    map(plan.reactions, ntuple(identity, Val(length(plan.reactions)))) do a, j
        model = a.reaction.energy
        out = energy_per_event[j]
        isnothing(model) ? fill!(out, zero(eltype(out))) : event_energy!(out, model, a.rate, electron_ctx, ee)
        nothing
    end
    return energy_per_event
end
