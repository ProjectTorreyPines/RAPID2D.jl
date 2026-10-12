# Averages along the field line through each node. The line of an open node runs from wall to
# wall, that of a closed node once around, and both carry the weight dl/B_pol of the flux-surface
# average, so on a closed line the average is the flux-surface average at the node's level.
# internal/docs/src/notes/plans/PLAN_flf-contours.md §5.5, §6.

"""
    FieldLineAverage{FT}

Averages along the field line through each of a set of nodes, weighted by dl/B_pol: from wall
to wall on an open line, once around on a closed one, where it is the flux-surface average at
the node's level. Build with [`field_line_average`](@ref); apply with [`line_average`](@ref).

Fields read by consumers: `nodes` (linear indices), `valid` (the line was measured, not ended
at a B_pol null or by the budget), `closed`. The weights are internal.
"""
struct FieldLineAverage{FT <: AbstractFloat}
    nodes::Vector{Int}
    valid::Vector{Bool}
    closed::Vector{Bool}
    weights::SparseMatrixCSC{FT, Int}          # grid nodes × lines; the column of a valid line sums to 1
    dims::Tuple{Int, Int}
end

"""
    field_line_average(g, nodes; tracer = MarchingSquaresTracer(), max_length) -> FieldLineAverage
    field_line_average(RP; kwargs...)

The field line through each node of `nodes` (linear indices of the grid of `g`), traced both
ways by `tracer` (a node it cannot follow by its fallback), as weights for
[`line_average`](@ref). The `RP` form takes the in-wall nodes, the run's wall and its tracer.
"""
function field_line_average(
        g::FieldLineGeometry{FT}, nodes::AbstractVector{<:Integer};
        tracer::FieldLineTracer = MarchingSquaresTracer(), max_length::Real = default_max_length(g),
    ) where {FT <: AbstractFloat}
    NR, NZ = size(g.ψ)
    N, nl = NR * NZ, length(nodes)
    colptr = Vector{Int}(undef, nl + 1)
    colptr[1] = 1
    rowval, nzval = Int[], FT[]
    valid, closed = fill(false, nl), fill(false, nl)
    @safe_with_pool pool begin                  # a trace may throw; the pool must survive it
        acc = zeros!(pool, FT, N)               # one line's weight on each node
        stamp = zeros!(pool, Int, N)            # the last line that reached each node
        reached = acquire!(pool, Int, N)        # the nodes this line reached
        lw = LineWeights(acc, stamp, reached, g)
        for (m, k) in pairs(nodes)
            ok, closed[m] = trace_line!(lw, m, tracer, g, mod1(k, NR), cld(k, NR), max_length)
            if ok && lw.total > 0
                for node in sort!(view(reached, 1:lw.nreached))
                    push!(rowval, node)
                    push!(nzval, acc[node] / lw.total)
                    acc[node] = zero(FT)
                end
                valid[m] = true
            else
                acc[view(reached, 1:lw.nreached)] .= zero(FT)
            end
            colptr[m + 1] = length(rowval) + 1
        end
        nothing
    end
    return FieldLineAverage(collect(Int, nodes), valid, closed, SparseMatrixCSC(N, nl, colptr, rowval, nzval), (NR, NZ))
end

function field_line_average(RP::RAPID; tracer::FieldLineTracer = RP.flags.field_line_tracer, kwargs...)
    g = FieldLineGeometry(RP; nodal_B = needs_nodal_B(tracer))
    return field_line_average(g, RP.G.nodes.in_wall_nids; tracer, kwargs...)
end

"""
    line_average(fla, f) -> Vector
    line_average(fla, f, ω) -> Vector

The average of the grid field `f` along each line of `fla` (in the order of `fla.nodes`), or the
weighted average ⟨ω f⟩/⟨ω⟩. `NaN` on lines that were not measured.
"""
function line_average(fla::FieldLineAverage, f::AbstractVecOrMat)
    avg = transpose(fla.weights) * vec(f)
    avg[.!fla.valid] .= NaN
    return avg
end

line_average(fla::FieldLineAverage, f::AbstractVecOrMat, ω::AbstractVecOrMat) =
    line_average(fla, vec(ω) .* vec(f)) ./ line_average(fla, ω)

"""
    to_grid(fla, values) -> Matrix

Values of the lines (one per node of `fla`) on their nodes; zero elsewhere.
"""
function to_grid(fla::FieldLineAverage, values::AbstractVector)
    out = zeros(eltype(values), fla.dims)
    out[fla.nodes] .= values
    return out
end

# Both directions from node (i, j) into the column `m` of `lw`: whether the line was measured,
# and whether it is closed. A node the tracer cannot follow goes to its fallback.
function trace_line!(lw, m, tracer, g, i, j, max_length)
    start_line!(lw, m)
    e = trace_from_node!(start_direction!(lw), tracer, g, i, j, 1; max_length)
    if e.termination === :degenerate
        # the attempt's weights and marks must not hide the fallback's
        for node in view(lw.reached, 1:lw.nreached)
            lw.acc[node] = 0
            lw.stamp[node] = 0
        end
        return trace_line!(lw, m, fallback_tracer(tracer), g, i, j, max_length)
    end
    if e.termination === :closed
        end_direction!(lw, true)
        finish_line!(lw)
        return true, true
    end
    e.termination === :wall || return false, false
    end_direction!(lw, false)
    e = trace_from_node!(start_direction!(lw), tracer, g, i, j, -1; max_length)
    e.termination === :wall || return false, false
    end_direction!(lw, false)
    finish_line!(lw)
    return true, false
end

# A visitor that spreads the trapezoidal weight (dl/2 on each side) / B_pol of every vertex of a
# line bilinearly onto the nodes. The start node is visited first in both directions; its weight
# waits for both, and the wall end of a direction takes its one half-segment.
mutable struct LineWeights{FT, G}
    acc::Vector{FT}
    stamp::Vector{Int}
    reached::Vector{Int}
    g::G
    column::Int
    nreached::Int
    total::FT
    n::Int                                      # vertices of this direction
    R::FT                                       # the vertex waiting for its second half-segment
    Z::FT
    w::FT                                       # 1/B_pol there
    half::FT                                    # its first half-segment
    Rn::FT                                      # the start node, 1/B_pol there and its weight
    Zn::FT
    wn::FT
    node_weight::FT
end
function LineWeights(acc::AbstractVector{FT}, stamp, reached, g) where {FT}
    nan = FT(NaN)
    return LineWeights(acc, stamp, reached, g, 0, 0, zero(FT), 0, nan, nan, nan, zero(FT), nan, nan, nan, zero(FT))
end

function start_line!(lw::LineWeights{FT}, m) where {FT}
    lw.column, lw.nreached, lw.total, lw.node_weight = m, 0, zero(FT), zero(FT)
    return lw
end
start_direction!(lw::LineWeights) = (lw.n = 0; lw)

function visit!(lw::LineWeights, R, Z, Bpol, Bϕ)
    w = inv(Bpol)
    if lw.n == 0
        lw.Rn, lw.Zn, lw.wn = R, Z, w
    else
        d = hypot(R - lw.R, Z - lw.Z)
        if lw.n == 1
            lw.node_weight += d / 2 * lw.wn
        else
            spread!(lw, lw.R, lw.Z, (lw.half + d / 2) * lw.w)
        end
        lw.half = d / 2
    end
    lw.R, lw.Z, lw.w = R, Z, w
    lw.n += 1
    return lw
end

# the last vertex of a direction: the wall end, or the start node again on a closed line
function end_direction!(lw::LineWeights, closed::Bool)
    lw.n >= 2 || return lw
    closed ? (lw.node_weight += lw.half * lw.wn) : spread!(lw, lw.R, lw.Z, lw.half * lw.w)
    return lw
end
finish_line!(lw::LineWeights) = spread!(lw, lw.Rn, lw.Zn, lw.node_weight)
