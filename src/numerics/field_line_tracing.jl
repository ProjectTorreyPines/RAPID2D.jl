# Following field lines in the poloidal plane. In axisymmetry the poloidal projection of a field
# line is a ψ contour, so a trace follows one: from a grid node in either direction along B_pol
# (field-line lengths), or around an O-point at a given level (flux surfaces). A tracer reports
# every vertex of its path to a visitor, `visit!(v, R, Z, Bpol, Bϕ)`, which integrates what it
# needs as the path goes; no path is stored. The tracer policies are in `field_line_tracers.jl`.
# internal/docs/src/notes/plans/PLAN_flf-contours.md §6.

"""
    psi_interpolant(G, ψ)

Bicubic interpolant of the grid flux `ψ`, with analytic gradient and Hessian. It reproduces
cubic polynomials exactly.
"""
function psi_interpolant(G::GridGeometry, ψ::AbstractMatrix)
    Rg = range(first(G.R1D), last(G.R1D); length = G.NR)
    Zg = range(first(G.Z1D), last(G.Z1D); length = G.NZ)
    return cubic_interp((Rg, Zg), ψ; bc = CubicFit(), extrap = ExtendExtrap())
end

"""
    FieldLineGeometry(G, ψ; Bϕ = zero, inside = the grid)
    FieldLineGeometry(RP)

What every tracer reads: the uniform grid, the flux `ψ` on its nodes with its bicubic
interpolant, the toroidal field `Bϕ` on the nodes, and the wall as a set of cells: `inside[i, j]`
for the cell spanning nodes i..i+1 × j..j+1. By default every cell of the grid is inside; the
`RP` form takes the wall from `cell_state`.
"""
struct FieldLineGeometry{FT <: AbstractFloat, I}
    R1D::Vector{FT}
    Z1D::Vector{FT}
    dR::FT
    dZ::FT
    ψ::Matrix{FT}
    itp::I
    Bϕ::Matrix{FT}
    inside::Matrix{Bool}
end

function FieldLineGeometry(
        G::GridGeometry{FT}, ψ::AbstractMatrix;
        Bϕ::AbstractMatrix = zeros(FT, G.NR, G.NZ), inside::AbstractMatrix{Bool} = trues(G.NR, G.NZ),
    ) where {FT <: AbstractFloat}
    size(ψ) == size(Bϕ) == size(inside) == (G.NR, G.NZ) ||
        throw(DimensionMismatch("ψ, Bϕ and inside must be $((G.NR, G.NZ))"))
    return FieldLineGeometry(
        collect(FT, G.R1D), collect(FT, G.Z1D), FT(G.dR), FT(G.dZ),
        Matrix{FT}(ψ), psi_interpolant(G, ψ), Matrix{FT}(Bϕ), Matrix{Bool}(inside),
    )
end

FieldLineGeometry(RP::RAPID) = FieldLineGeometry(RP.G, RP.fields.ψ; Bϕ = RP.fields.Bϕ, inside = RP.G.cell_state .>= 0)

# whether cell (i, j) is a cell of the grid inside the wall
@inline function cell_inside(g::FieldLineGeometry, i::Int, j::Int)
    NR, NZ = size(g.ψ)
    return 1 <= i < NR && 1 <= j < NZ && @inbounds g.inside[i, j]
end

"""
    PathLengths{FT}()

A visitor that measures a traced path: the poloidal length `Lpol` = ∫ dl, the length along B
`Lc` = ∫ (B/B_pol) dl, the smallest `min_Bpol` met, and the number of vertices `n`. Trapezoidal
between vertices.
"""
mutable struct PathLengths{FT <: AbstractFloat}
    Lpol::FT
    Lc::FT
    min_Bpol::FT
    n::Int
    R::FT                                  # the last vertex, and B/B_pol there
    Z::FT
    ratio::FT
end
PathLengths{FT}() where {FT} = PathLengths{FT}(zero(FT), zero(FT), FT(Inf), 0, FT(NaN), FT(NaN), FT(NaN))

function visit!(p::PathLengths, R, Z, Bpol, Bϕ)
    ratio = sqrt(Bpol^2 + Bϕ^2) / Bpol
    if p.n > 0
        d = hypot(R - p.R, Z - p.Z)
        p.Lpol += d
        p.Lc += d * (p.ratio + ratio) / 2
    end
    p.min_Bpol = min(p.min_Bpol, Bpol)
    p.n += 1
    p.R, p.Z, p.ratio = R, Z, ratio
    return p
end

# ── marching squares ─────────────────────────────────────────────────────────────────

# Cell (i, j) has corners 1–4 counter-clockwise from node (i, j); edge e joins corners e and
# e % 4 + 1 (1 bottom, 2 right, 3 top, 4 left). A node is above the level when ψ ≥ level.
@inline ms_corner(i, j, c) = c == 1 ? (i, j) : c == 2 ? (i + 1, j) : c == 3 ? (i + 1, j + 1) : (i, j + 1)
@inline ms_above(ψ, L, n) = @inbounds ψ[n[1], n[2]] >= L
@inline ms_crosses(ψ, L, i, j, e) = ms_above(ψ, L, ms_corner(i, j, e)) != ms_above(ψ, L, ms_corner(i, j, e % 4 + 1))
# the cell across edge e, and the edge it is entered by
@inline ms_across(i, j, e) = e == 1 ? (i, j - 1, 3) : e == 2 ? (i + 1, j, 4) : e == 3 ? (i, j + 1, 1) : (i - 1, j, 2)
# one id per grid edge: horizontal (i, j)–(i+1, j) odd, vertical (i, j)–(i, j+1) even
@inline function ms_edge_id(NR, i, j, e)
    e == 1 && return 2 * ((j - 1) * NR + i) - 1
    e == 2 && return 2 * ((j - 1) * NR + i + 1)
    e == 3 && return 2 * (j * NR + i) - 1
    return 2 * ((j - 1) * NR + i)
end

# Going along B_pol, ψ rises to the right (B_pol is ∇ψ turned a quarter counter-clockwise). A
# contour that enters a cell through edge e has corner e % 4 + 1 on its right, so it goes along
# B_pol when that corner is above the level, and against it otherwise.
@inline ms_forward(ψ, L, i, j, e) = ms_above(ψ, L, ms_corner(i, j, e % 4 + 1))

# The edge by which the contour entering cell (i, j) through edge `ein` leaves it, 0 if none. A
# saddle (corners alternating above and below) connects by its centre: the corners on the other
# side from the centre are cut off, each with its two edges.
function ms_exit_edge(ψ, L, i, j, ein)
    if ms_crosses(ψ, L, i, j, 1) && ms_crosses(ψ, L, i, j, 2) && ms_crosses(ψ, L, i, j, 3) && ms_crosses(ψ, L, i, j, 4)
        @inbounds centre_above = (ψ[i, j] + ψ[i + 1, j] + ψ[i + 1, j + 1] + ψ[i, j + 1]) / 4 >= L
        # of the two corners of the entry edge, the one cut off is the one unlike the centre
        c = ms_above(ψ, L, ms_corner(i, j, ein)) == centre_above ? ein % 4 + 1 : ein
        return c == ein ? (ein + 2) % 4 + 1 : c       # corner c lies on edges c − 1 and c
    end
    for e in 1:4
        e != ein && ms_crosses(ψ, L, i, j, e) && return e
    end
    return 0
end

# The point where the contour crosses edge e of cell (i, j), and its fraction t along the edge.
@inline function ms_point(g::FieldLineGeometry, L, i, j, e)
    a, b = ms_corner(i, j, e), ms_corner(i, j, e % 4 + 1)
    @inbounds va, vb = g.ψ[a[1], a[2]], g.ψ[b[1], b[2]]
    t = (L - va) / (vb - va)
    return g.R1D[a[1]] + t * (g.R1D[b[1]] - g.R1D[a[1]]), g.Z1D[a[2]] + t * (g.Z1D[b[2]] - g.Z1D[a[2]]), t, a, b
end

# That point with B_pol and Bϕ there.
@inline function ms_vertex(g::FieldLineGeometry, L, i, j, e)
    R, Z, t, a, b = ms_point(g, L, i, j, e)
    gr = gradient(g.itp, (R, Z))
    @inbounds Bϕ = g.Bϕ[a[1], a[2]] + t * (g.Bϕ[b[1], b[2]] - g.Bϕ[a[1], a[2]])
    return R, Z, hypot(gr[1], gr[2]) / R, Bϕ
end

# From the contour's crossing of edge `e` of cell (i, j) (already visited), cell by cell until
# it enters a cell outside the wall (:wall), comes back through the edge `start` (:closed), or
# runs out of `max_length` or of cells (:trace_limit). Returns the termination and the
# poloidal length walked.
function ms_march!(v, g::FieldLineGeometry{FT}, L, i, j, e, start, max_length) where {FT}
    NR, NZ = size(g.ψ)
    R, Z, _, _, _ = ms_point(g, L, i, j, e)
    Lpol = zero(FT)
    for _ in 1:(2 * NR * NZ)
        cell_inside(g, i, j) || return :wall, Lpol
        eo = ms_exit_edge(g.ψ, L, i, j, e)
        eo == 0 && return :trace_limit, Lpol
        Rn, Zn, Bpol, Bϕ = ms_vertex(g, L, i, j, eo)
        visit!(v, Rn, Zn, Bpol, Bϕ)
        Lpol += hypot(Rn - R, Zn - Z)
        ms_edge_id(NR, i, j, eo) == start && return :closed, Lpol
        Lpol > max_length && return :trace_limit, Lpol
        R, Z = Rn, Zn
        i, j, e = ms_across(i, j, eo)
    end
    return :trace_limit, Lpol
end

"""
    trace_from_node!(visitor, tracer, g, i, j, sgn; max_length = Inf) -> termination

Follow the field line through node (i, j) along `sgn` B_pol (`sgn = ±1`), reporting each
vertex to `visitor`, the node first. Terminations: `:wall` (the path ends on the wall),
`:closed` (it came back to the node; the node is visited again last), `:null` (B_pol vanished),
`:trace_limit` (the poloidal length passed `max_length`, or the tracer's own step budget ran
out), `:degenerate` (the tracer cannot follow this level from this node: with marching squares,
a node whose level is a single point of the bilinear ψ, as next to an O-point).
"""
function trace_from_node!(v, ::MarchingSquaresTracer, g::FieldLineGeometry{FT}, i::Int, j::Int, sgn::Int; max_length::Real = Inf) where {FT}
    ψ = g.ψ
    NR, NZ = size(ψ)
    L = ψ[i, j]
    # An edge from the node to a neighbour below its level carries the crossing at the node
    # itself; of the two cells beside that edge, the contour goes along B_pol into one.
    for (ia, ja, ea, ib, jb, eb, nbr) in (
            (i, j, 1, i, j - 1, 3, (i + 1, j)), (i, j, 4, i - 1, j, 2, (i, j + 1)),
            (i - 1, j, 1, i - 1, j - 1, 3, (i - 1, j)), (i, j - 1, 4, i - 1, j - 1, 2, (i, j - 1)),
        )
        (1 <= nbr[1] <= NR && 1 <= nbr[2] <= NZ) || continue
        ms_above(ψ, L, nbr) && continue
        # either cell may lie off the grid; its orientation reads only the edge's own two nodes
        ci, cj, ce = (ms_forward(ψ, L, ia, ja, ea) == (sgn > 0)) ? (ia, ja, ea) : (ib, jb, eb)
        start = ms_edge_id(NR, ci, cj, ce)
        visit!(v, g.R1D[i], g.Z1D[j], ms_node_field(g, i, j)...)
        term, Lpol = ms_march!(v, g, L, ci, cj, ce, start, max_length)
        # a closed path of no length is a point: the level does not leave the node
        return term === :closed && Lpol == 0 ? :degenerate : term
    end
    return :degenerate                       # no neighbour below: the level is the node alone
end

@inline function ms_node_field(g::FieldLineGeometry, i, j)
    R = g.R1D[i]
    gr = gradient(g.itp, (R, g.Z1D[j]))
    return hypot(gr[1], gr[2]) / R, g.Bϕ[i, j]
end
