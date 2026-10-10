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
    FieldLineGeometry(G, ψ; Bϕ = nothing, inside = nothing, nodal_B = false, BR, BZ)
    FieldLineGeometry(RP; nodal_B = false)

What every tracer reads: the uniform grid, the flux `ψ` on its nodes (kept, not copied) with its
bicubic interpolant, the toroidal field `Bϕ` on the nodes (`nothing`: none), and the wall as a
set of cells: `inside[i, j]` for the cell spanning nodes i..i+1 × j..j+1 (`nothing`: the whole
grid). The `RP` form takes the run's Bϕ and its wall from `cell_state`. `nodal_B = true` adds
bicubic interpolants of the nodal `BR`, `BZ` (by default central differences of ψ; the `RP` form
takes the run's) and `Bϕ`, for a tracer that follows B rather than ψ ([`needs_nodal_B`](@ref)).
"""
struct FieldLineGeometry{FT <: AbstractFloat, M <: AbstractMatrix{FT}, I, P, W, B}
    R1D::Vector{FT}
    Z1D::Vector{FT}
    dR::FT
    dZ::FT
    ψ::M
    ψ_scale::FT                             # largest |ψ| on the grid: what rounding is relative to
    itp::I
    Bϕ::P                                   # nothing, or Bϕ on the nodes
    inside::W                               # nothing (the whole grid), or the in-wall cells
    nodal_B::B                              # nothing, or (; BR, BZ, Bϕ) interpolants
end

function FieldLineGeometry(
        G::GridGeometry{FT}, ψ::AbstractMatrix{FT};
        Bϕ::Union{AbstractMatrix, Nothing} = nothing, inside::Union{AbstractMatrix{Bool}, Nothing} = nothing,
        nodal_B::Bool = false, BR::Union{AbstractMatrix, Nothing} = nothing, BZ::Union{AbstractMatrix, Nothing} = nothing,
    ) where {FT <: AbstractFloat}
    dims = (G.NR, G.NZ)
    (size(ψ) == dims && (Bϕ === nothing || size(Bϕ) == dims) && (inside === nothing || size(inside) == dims)) ||
        throw(DimensionMismatch("ψ, Bϕ and inside must be $dims"))
    R1D, Z1D = collect(FT, G.R1D), collect(FT, G.Z1D)
    B = nothing
    if nodal_B
        if BR === nothing || BZ === nothing
            BR, BZ = zeros(FT, dims), zeros(FT, dims)
            calculate_B_from_ψ!(G, Matrix{FT}(ψ), BR, BZ)
        end
        B = (
            BR = my_interpolation(R1D, Z1D, Matrix{FT}(BR)), BZ = my_interpolation(R1D, Z1D, Matrix{FT}(BZ)),
            Bϕ = my_interpolation(R1D, Z1D, Bϕ === nothing ? zeros(FT, dims) : Matrix{FT}(Bϕ)),
        )
    end
    return FieldLineGeometry(R1D, Z1D, FT(G.dR), FT(G.dZ), ψ, FT(maximum(abs, ψ)), psi_interpolant(G, ψ), Bϕ, inside, B)
end

FieldLineGeometry(RP::RAPID; nodal_B::Bool = false) = FieldLineGeometry(
    RP.G, RP.fields.ψ; Bϕ = RP.fields.Bϕ, inside = RP.G.cell_state .>= 0, nodal_B, BR = RP.fields.BR, BZ = RP.fields.BZ,
)

# The poloidal-length budget of a field-line trace: three diagonals of the grid.
default_max_length(g::FieldLineGeometry{FT}) where {FT} =
    FT(3) * sqrt((maximum(g.R1D) - minimum(g.R1D))^2 + (maximum(g.Z1D) - minimum(g.Z1D))^2)

# the most steps one trace may take within `max_length`
step_budget(::MarchingSquaresTracer, g::FieldLineGeometry, max_length) = 2 * length(g.ψ)
step_budget(p::CubicTracer, g::FieldLineGeometry, max_length) = p.max_steps
step_budget(p::RK4Tracer, g::FieldLineGeometry{FT}, max_length) where {FT} =
    floor(Int, max_length / (FT(p.step_cells) * min(abs(g.dR), abs(g.dZ))))

# the visitor that measures a trace's lengths: RK4 keeps its right-endpoint sums
path_lengths(::FieldLineTracer, ::Type{FT}) where {FT} = PathLengths{FT}()
path_lengths(::RK4Tracer, ::Type{FT}) where {FT} = StepLengths{FT}()

# whether `:closed` means the path came back to its start (a contour tracer), so that one closed
# direction closes the line; RK4 closes on turning angle, which an open line can reach
closes_by_return(::FieldLineTracer) = true
closes_by_return(::RK4Tracer) = false

# the tracer for a node a tracer cannot follow (`:degenerate`)
fallback_tracer(::MarchingSquaresTracer) = CubicTracer()

# whether cell (i, j) is a cell of the grid inside the wall
@inline function cell_inside(g::FieldLineGeometry, i::Int, j::Int)
    NR, NZ = size(g.ψ)
    (1 <= i < NR && 1 <= j < NZ) || return false
    return g.inside === nothing || @inbounds g.inside[i, j]
end

# Bϕ at a point, bilinear between the nodes
@inline function toroidal_field(g::FieldLineGeometry{FT}, x) where {FT}
    g.Bϕ === nothing && return zero(FT)
    NR, NZ = size(g.Bϕ)
    i = clamp(floor(Int, (x[1] - g.R1D[1]) / g.dR) + 1, 1, NR - 1)
    j = clamp(floor(Int, (x[2] - g.Z1D[1]) / g.dZ) + 1, 1, NZ - 1)
    t = (x[1] - g.R1D[i]) / g.dR
    u = (x[2] - g.Z1D[j]) / g.dZ
    @inbounds return (1 - t) * (1 - u) * g.Bϕ[i, j] + t * (1 - u) * g.Bϕ[i + 1, j] +
        (1 - t) * u * g.Bϕ[i, j + 1] + t * u * g.Bϕ[i + 1, j + 1]
end

# The cell indices a chord can start in, along one axis, from the position `f` in cells and the
# displacement Δ: on a grid line the cell on the side it moves to (both when it moves along it).
@inline function start_cells(f, Δ)
    k = round(f)
    abs(f - k) <= 64 * eps(typeof(f)) * max(one(f), abs(f)) || return (floor(Int, f) + 1, floor(Int, f) + 1)
    return Δ > 0 ? (Int(k) + 1, Int(k) + 1) : Δ < 0 ? (Int(k), Int(k)) : (Int(k) + 1, Int(k))
end

# The fraction along the chord from x to xn at which it first enters a cell outside the wall,
# or `nothing`. It walks every cell the chord crosses, so no corner of the wall is stepped over.
function chord_exit(g::FieldLineGeometry{FT}, x, xn) where {FT}
    R1, Z1 = g.R1D[1], g.Z1D[1]
    ΔR, ΔZ = xn[1] - x[1], xn[2] - x[2]
    # the cell the chord starts into: from a grid line (a node), the one on the side it leaves by;
    # along a grid line, an inside one of the two
    is, js = start_cells((x[1] - R1) / g.dR, ΔR), start_cells((x[2] - Z1) / g.dZ, ΔZ)
    i, j = 0, 0
    for a in is, b in js
        cell_inside(g, a, b) && ((i, j) = (a, b); break)
    end
    i == 0 && return zero(FT)
    tR = ΔR == 0 ? FT(Inf) : (R1 + (ΔR > 0 ? i : i - 1) * g.dR - x[1]) / ΔR
    tZ = ΔZ == 0 ? FT(Inf) : (Z1 + (ΔZ > 0 ? j : j - 1) * g.dZ - x[2]) / ΔZ
    dtR = ΔR == 0 ? FT(Inf) : g.dR / abs(ΔR)
    dtZ = ΔZ == 0 ? FT(Inf) : g.dZ / abs(ΔZ)
    while true
        if tR <= tZ
            tR > 1 && return nothing
            s = tR
            i += ΔR > 0 ? 1 : -1
            tR += dtR
        else
            tZ > 1 && return nothing
            s = tZ
            j += ΔZ > 0 ? 1 : -1
            tZ += dtZ
        end
        cell_inside(g, i, j) || return s
    end
    return
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

# back to no path, for the next trace
function reset!(p::PathLengths{FT}) where {FT}
    p.Lpol, p.Lc, p.min_Bpol, p.n = zero(FT), zero(FT), FT(Inf), 0
    return p
end

function visit!(p::PathLengths, R, Z, Bpol, Bϕ)
    ratio = sqrt(Bpol^2 + Bϕ^2) / Bpol
    if p.n > 0
        d = sqrt((R - p.R)^2 + (Z - p.Z)^2)
        p.Lpol += d
        p.Lc += d * (p.ratio + ratio) / 2
    end
    p.min_Bpol = min(p.min_Bpol, Bpol)
    p.n += 1
    p.R, p.Z, p.ratio = R, Z, ratio
    return p
end

"""
    StepLengths{FT}()

[`PathLengths`](@ref) as RAPID2D's first field-line following summed them, kept for
[`RK4Tracer`](@ref): each step adds its chord times B/B_pol at its end, and the smallest B_pol is
taken over the step ends, not the start.
"""
mutable struct StepLengths{FT <: AbstractFloat}
    Lpol::FT
    Lc::FT
    min_Bpol::FT
    n::Int
    R::FT
    Z::FT
end
StepLengths{FT}() where {FT} = StepLengths{FT}(zero(FT), zero(FT), FT(Inf), 0, FT(NaN), FT(NaN))

function reset!(p::StepLengths{FT}) where {FT}
    p.Lpol, p.Lc, p.min_Bpol, p.n = zero(FT), zero(FT), FT(Inf), 0
    return p
end

function visit!(p::StepLengths, R, Z, Bpol, Bϕ)
    if p.n > 0
        dl_pol = sqrt((R - p.R)^2 + (Z - p.Z)^2)
        Btot = sqrt(Bpol^2 + Bϕ^2)
        p.Lpol += dl_pol
        p.Lc += dl_pol * Btot / Bpol
        p.min_Bpol = min(p.min_Bpol, Bpol)
    end
    p.n += 1
    p.R, p.Z = R, Z
    return p
end

# The lengths a trace publishes for each way it can end (`FLF_TERMINATIONS`): the measured ones
# at a wall or after one circuit; at a B_pol null the line closes toroidally, adding 2πR to Lc
# with Lpol unbounded; when the budget ran out the distance along B is unknown, Lc = Inf.
function published_lengths(p, termination::Symbol, R_end)
    FT = typeof(p.Lpol)
    termination === :null && return (; Lpol = FT(Inf), Lc = p.Lc + FT(2π) * R_end, min_Bpol = zero(FT))
    termination === :trace_limit && return (; Lpol = p.Lpol, Lc = FT(Inf), min_Bpol = p.min_Bpol)
    return (; Lpol = p.Lpol, Lc = p.Lc, min_Bpol = p.min_Bpol)
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
    @inbounds v1, v2, v3, v4 = ψ[i, j], ψ[i + 1, j], ψ[i + 1, j + 1], ψ[i, j + 1]
    a1, a2, a3, a4 = v1 >= L, v2 >= L, v3 >= L, v4 >= L
    if a1 == a3 && a2 == a4 && a1 != a2
        # of the two corners of the entry edge, the one cut off is the one unlike the centre
        centre_above = (v1 + v2 + v3 + v4) / 4 >= L
        entry_above = ein == 1 ? a1 : ein == 2 ? a2 : ein == 3 ? a3 : a4
        c = entry_above == centre_above ? ein % 4 + 1 : ein
        return c == ein ? (ein + 2) % 4 + 1 : c       # corner c lies on edges c − 1 and c
    end
    ein != 1 && a1 != a2 && return 1
    ein != 2 && a2 != a3 && return 2
    ein != 3 && a3 != a4 && return 3
    ein != 4 && a4 != a1 && return 4
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
    Bϕ = g.Bϕ === nothing ? zero(R) : @inbounds g.Bϕ[a[1], a[2]] + t * (g.Bϕ[b[1], b[2]] - g.Bϕ[a[1], a[2]])
    return R, Z, sqrt(gr[1]^2 + gr[2]^2) / R, Bϕ
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
        cell_inside(g, i, j) || return (; termination = :wall, R, Z), Lpol
        eo = ms_exit_edge(g.ψ, L, i, j, e)
        eo == 0 && return (; termination = :trace_limit, R, Z), Lpol
        Rn, Zn, Bpol, Bϕ = ms_vertex(g, L, i, j, eo)
        visit!(v, Rn, Zn, Bpol, Bϕ)
        Lpol += sqrt((Rn - R)^2 + (Zn - Z)^2)
        R, Z = Rn, Zn
        ms_edge_id(NR, i, j, eo) == start && return (; termination = :closed, R, Z), Lpol
        Lpol > max_length && return (; termination = :trace_limit, R, Z), Lpol
        i, j, e = ms_across(i, j, eo)
    end
    return (; termination = :trace_limit, R, Z), Lpol
end

"""
    trace_from_node!(visitor, tracer, g, i, j, sgn; max_length = Inf) -> (; termination, R, Z)

Follow the field line through node (i, j) along `sgn` B_pol (`sgn = ±1`), reporting each
vertex to `visitor`, the node first, and return how and where the path ended. Terminations: `:wall` (the path ends on the wall),
`:closed` (it came back to the node; the node is visited again last), `:null` (B_pol vanished),
`:trace_limit` (the poloidal length passed `max_length`, or the tracer's own step budget ran
out), `:degenerate` (the tracer cannot follow this level from this node: with marching squares,
a node whose level is a single point of the bilinear ψ, as next to an O-point).
"""
function trace_from_node!(v, ::MarchingSquaresTracer, g::FieldLineGeometry{FT}, i::Int, j::Int, sgn::Int; max_length::Real = Inf) where {FT}
    ψ = g.ψ
    L = ψ[i, j]
    seed = ms_node_seed(ψ, L, i, j, sgn)
    seed === nothing && return (; termination = :degenerate, R = g.R1D[i], Z = g.Z1D[j])   # the level is the node alone
    if !cell_inside(g, seed[1], seed[2])
        # Nodes at the level count as above it, so a contour along a grid line runs in the cells on
        # its lower side. When those are outside the wall, the level a rounding above the node's
        # value runs in the cells on the other side.
        up = ms_node_seed(ψ, nextfloat(L), i, j, sgn)
        up !== nothing && cell_inside(g, up[1], up[2]) && ((L, seed) = (nextfloat(L), up))
    end
    ci, cj, ce = seed
    visit!(v, g.R1D[i], g.Z1D[j], ms_node_field(g, i, j)...)
    e, Lpol = ms_march!(v, g, L, ci, cj, ce, ms_edge_id(size(ψ, 1), ci, cj, ce), max_length)
    # a closed path of no length is a point: the level does not leave the node
    return e.termination === :closed && Lpol == 0 ? (; termination = :degenerate, e.R, e.Z) : e
end

# The cell and entry edge by which the contour of `L` leaves node (i, j) along `sgn` B_pol, or
# `nothing`: an edge from the node to a neighbour on the other side of the level carries the
# crossing at the node itself, and of the two cells beside it the contour goes along B_pol into
# one (either may lie off the grid; its orientation reads only the edge's own two nodes).
function ms_node_seed(ψ, L, i, j, sgn)
    NR, NZ = size(ψ)
    node_above = ms_above(ψ, L, (i, j))
    for (ia, ja, ea, ib, jb, eb, nbr) in (
            (i, j, 1, i, j - 1, 3, (i + 1, j)), (i, j, 4, i - 1, j, 2, (i, j + 1)),
            (i - 1, j, 1, i - 1, j - 1, 3, (i - 1, j)), (i, j - 1, 4, i - 1, j - 1, 2, (i, j - 1)),
        )
        (1 <= nbr[1] <= NR && 1 <= nbr[2] <= NZ) || continue
        ms_above(ψ, L, nbr) == node_above && continue
        return ms_forward(ψ, L, ia, ja, ea) == (sgn > 0) ? (ia, ja, ea) : (ib, jb, eb)
    end
    return nothing
end

@inline function ms_node_field(g::FieldLineGeometry, i, j)
    R = g.R1D[i]
    gr = gradient(g.itp, (R, g.Z1D[j]))
    return sqrt(gr[1]^2 + gr[2]^2) / R, g.Bϕ === nothing ? zero(R) : g.Bϕ[i, j]
end

"""
    trace_around!(visitor, tracer, g, level, o; max_length = Inf) -> (; termination, R, Z)

Follow the contour of `level` around the O-point `o` (`(; R, Z, ψ)`), starting where it crosses
the midplane outward of `o`, until it comes back there (`:closed`). As for
[`trace_from_node!`](@ref), plus `:unseeded` when no crossing is found. With marching squares
the midplane is the grid row nearest `o`; otherwise it is Z = o.Z on the bicubic ψ.
"""
function trace_around!(v, ::MarchingSquaresTracer, g::FieldLineGeometry{FT}, level, o; max_length::Real = Inf) where {FT}
    ψ = g.ψ
    NR, NZ = size(ψ)
    unseeded = (; termination = :unseeded, R = FT(o.R), Z = FT(o.Z))
    (g.R1D[1] <= o.R < g.R1D[end] && g.Z1D[1] <= o.Z <= g.Z1D[end]) || return unseeded
    ia = floor(Int, (o.R - g.R1D[1]) / g.dR) + 1
    jr = clamp(round(Int, (o.Z - g.Z1D[1]) / g.dZ) + 1, 1, NZ)
    for i in (ia + 1):NR
        ms_above(ψ, level, (i - 1, jr)) == ms_above(ψ, level, (i, jr)) && continue
        # the edge is the bottom of cell (i − 1, jr) and the top of cell (i − 1, jr − 1): along B_pol into one
        ci, cj, ce = ms_forward(ψ, level, i - 1, jr, 1) ? (i - 1, jr, 1) : (i - 1, jr - 1, 3)
        R, Z, Bpol, Bϕ = ms_vertex(g, level, ci, cj, ce)
        visit!(v, R, Z, Bpol, Bϕ)
        return first(ms_march!(v, g, level, ci, cj, ce, ms_edge_id(NR, ci, cj, ce), max_length))
    end
    return unseeded
end

# ── the bicubic level set ────────────────────────────────────────────────────────

# The unit tangent along sgn·B_pol and R/|∇ψ| (= 1/B_pol) at x, or `nothing` where ∇ψ = 0.
function tangent_and_weight(itp, x, sgn)
    g = gradient(itp, x)
    n = hypot(g[1], g[2])
    n > 0 || return nothing
    return (-sgn * g[2] / n, sgn * g[1] / n), x[1] / n
end

# Newton from x back onto `level` along ∇ψ; `nothing` if it moves farther than h or does not converge.
function project_to_level(itp, level, x, h)
    x_pred = x
    for _ in 1:8
        v = itp(x) - level
        g = gradient(itp, x)
        g2 = g[1]^2 + g[2]^2
        g2 > 0 || return nothing
        x = (x[1] - v * g[1] / g2, x[2] - v * g[2] / g2)
        hypot(x[1] - x_pred[1], x[2] - x_pred[2]) <= h || return nothing
        # on the level to a fraction of the step, or to ψ's own rounding where |∇ψ| is too small
        # for that (a large flux near an O-point)
        (abs(v) / sqrt(g2) < max(1.0e-10, 64 * eps(typeof(v))) * h || abs(v) <= 16 * eps(abs(level))) && return x
    end
    return nothing
end

# whether ψ changes by no more than its rounding over a step h at x: B_pol is zero there
flat(g::FieldLineGeometry, x, h) = (gr = gradient(g.itp, x); hypot(gr[1], gr[2]) * h <= 64 * eps(g.ψ_scale))

trace_from_node!(v, p::CubicTracer, g::FieldLineGeometry, i::Int, j::Int, sgn::Int; max_length::Real = Inf) =
    cubic_trace!(v, p, g, (g.R1D[i], g.Z1D[j]), g.itp((g.R1D[i], g.Z1D[j])), sgn, max_length)

# From x0 on `level` along sgn·B_pol. Closed when, after more than half a turn, the path crosses
# the line through x0 normal to its tangent, forward and within two steps of x0.
function cubic_trace!(v, p::CubicTracer, g::FieldLineGeometry{FT}, x0, level, sgn, max_length) where {FT}
    itp = g.itp
    h_max = p.max_step_cells * min(g.dR, g.dZ)
    h_min = FT(1.0e-6) * h_max
    tw0 = tangent_and_weight(itp, x0, sgn)
    (tw0 === nothing || flat(g, x0, h_max)) && return (; termination = :null, R = x0[1], Z = x0[2])
    t0, w0 = tw0
    visit!(v, x0[1], x0[2], inv(w0), toroidal_field(g, x0))
    t, w = t0, w0
    x, h, turned, Lpol = x0, h_max, zero(FT), zero(FT)
    for _ in 1:p.max_steps
        # predictor along the tangent, corrector back onto the level; halve until the tangent
        # turns little and the integrand R/|∇ψ| changes little
        xn, tn, wn, turn = x, t, w, zero(FT)
        while true
            xn = project_to_level(itp, level, (x[1] + h * t[1], x[2] + h * t[2]), h)
            tw = xn === nothing ? nothing : tangent_and_weight(itp, xn, sgn)
            if tw !== nothing
                tn, wn = tw
                turn = atan(t[1] * tn[2] - t[2] * tn[1], t[1] * tn[1] + t[2] * tn[2])
                abs(turn) <= p.max_turn && abs(wn - w) <= p.max_weight_change * max(w, wn) && break
            end
            h /= 2
            # B_pol is zero to rounding, or the contour cannot be resolved here (smaller than the
            # shortest step): a null, or a failed measurement
            h < h_min && return (; termination = flat(g, x, h_max) ? :null : :trace_limit, R = x[1], Z = x[2])
        end
        s = chord_exit(g, x, xn)
        if s !== nothing
            xb = (x[1] + s * (xn[1] - x[1]), x[2] + s * (xn[2] - x[2]))
            gb = gradient(itp, xb)
            visit!(v, xb[1], xb[2], hypot(gb[1], gb[2]) / xb[1], toroidal_field(g, xb))
            return (; termination = :wall, R = xb[1], Z = xb[2])
        end
        turned += turn
        if abs(turned) > π
            a = (x[1] - x0[1]) * t0[1] + (x[2] - x0[2]) * t0[2]
            b = (xn[1] - x0[1]) * t0[1] + (xn[2] - x0[2]) * t0[2]
            if a < 0 <= b
                f = a / (a - b)
                if hypot(x[1] + f * (xn[1] - x[1]) - x0[1], x[2] + f * (xn[2] - x[2]) - x0[2]) <= 2 * h_max
                    visit!(v, x0[1], x0[2], inv(w0), toroidal_field(g, x0))
                    return (; termination = :closed, R = x0[1], Z = x0[2])
                end
            end
        end
        visit!(v, xn[1], xn[2], inv(wn), toroidal_field(g, xn))
        Lpol += hypot(xn[1] - x[1], xn[2] - x[2])
        Lpol > max_length && return (; termination = :trace_limit, R = xn[1], Z = xn[2])
        x, t, w = xn, tn, wn
        h = min(2h, h_max)
    end
    return (; termination = :trace_limit, R = x[1], Z = x[2])
end

function trace_around!(v, p::Union{CubicTracer, RK4Tracer}, g::FieldLineGeometry{FT}, level, o; max_length::Real = Inf) where {FT}
    x0 = midplane_crossing(g, level, o)
    x0 === nothing && return (; termination = :unseeded, R = FT(o.R), Z = FT(o.Z))
    tw = tangent_and_weight(g.itp, x0, 1)
    tw === nothing && return (; termination = :null, R = x0[1], Z = x0[2])
    return trace_from_point!(v, p, g, x0, level, tw[1][2] >= 0 ? 1 : -1, max_length)    # upward
end

trace_from_point!(v, p::CubicTracer, g::FieldLineGeometry, x0, level, sgn, max_length) =
    cubic_trace!(v, p, g, x0, level, sgn, max_length)

# The first crossing of `level` outward of `o` along Z = o.Z on the bicubic ψ, to bisection accuracy.
function midplane_crossing(g::FieldLineGeometry{FT}, level, o) where {FT}
    f(R) = g.itp((R, o.Z)) - level
    f_axis = f(o.R)
    a, step = FT(o.R), g.dR / 4
    while a < g.R1D[end]
        b = min(a + step, g.R1D[end])
        if sign(f(b)) != sign(f_axis)
            for _ in 1:60
                c = (a + b) / 2
                sign(f(c)) == sign(f_axis) ? (a = c) : (b = c)
            end
            return ((a + b) / 2, FT(o.Z))
        end
        a = b
    end
    return nothing
end

# ── RK4 along the nodal B ────────────────────────────────────────────────────────────

"""
    advance_step_along_b_rz_plane(dl, R, Z, interp_BR, interp_BZ)

Advance one step along magnetic field line using 4th-order Runge-Kutta integration.

This function implements the RK4 algorithm for integrating field lines in the poloidal
plane, following the magnetic field direction with step size `dl`.

# Arguments
- `dl::FT`: Step size (positive for forward, negative for backward)
- `R::FT`: Current radial position
- `Z::FT`: Current vertical position
- `interp_BR`: Radial magnetic field interpolation function
- `interp_BZ`: Vertical magnetic field interpolation function

# Returns
- `next_R::FT`: New radial position
- `next_Z::FT`: New vertical position

# Algorithm
Uses classical RK4 integration:
1. k1 = f(t, y)
2. k2 = f(t + h/2, y + k1*h/2)
3. k3 = f(t + h/2, y + k2*h/2)
4. k4 = f(t + h, y + k3*h)
5. y_new = y + h*(k1 + 2*k2 + 2*k3 + k4) / 6

Where f(t,y) = B(R,Z)/|B_pol(R,Z)| is the unit field direction.
"""
function advance_step_along_b_rz_plane(
        dl::FT,
        R::FT,
        Z::FT,
        interp_BR::FastInterpolations.AbstractInterpolantND,
        interp_BZ::FastInterpolations.AbstractInterpolantND
    ) where {FT <: AbstractFloat}
    half = FT(0.5)
    two_FT = FT(2.0)

    # RK4 Step 1
    BR1 = interp_BR(R, Z)
    BZ1 = interp_BZ(R, Z)
    Bpol1 = sqrt(BR1^two_FT + BZ1^two_FT)
    R_RK1 = dl * BR1 / Bpol1
    Z_RK1 = dl * BZ1 / Bpol1

    # RK4 Step 2
    R2 = R + half * R_RK1
    Z2 = Z + half * Z_RK1
    BR2 = interp_BR(R2, Z2)
    BZ2 = interp_BZ(R2, Z2)
    Bpol2 = sqrt(BR2^two_FT + BZ2^two_FT)
    R_RK2 = dl * BR2 / Bpol2
    Z_RK2 = dl * BZ2 / Bpol2

    # RK4 Step 3
    R3 = R + half * R_RK2
    Z3 = Z + half * Z_RK2
    BR3 = interp_BR(R3, Z3)
    BZ3 = interp_BZ(R3, Z3)
    Bpol3 = sqrt(BR3^two_FT + BZ3^two_FT)
    R_RK3 = dl * BR3 / Bpol3
    Z_RK3 = dl * BZ3 / Bpol3

    # RK4 Step 4
    R4 = R + R_RK3
    Z4 = Z + Z_RK3
    BR4 = interp_BR(R4, Z4)
    BZ4 = interp_BZ(R4, Z4)
    Bpol4 = sqrt(BR4^two_FT + BZ4^two_FT)
    R_RK4 = dl * BR4 / Bpol4
    Z_RK4 = dl * BZ4 / Bpol4

    # Combine RK4 steps
    next_R = R + (R_RK1 + two_FT * R_RK2 + two_FT * R_RK3 + R_RK4) / FT(6.0)
    next_Z = Z + (Z_RK1 + two_FT * Z_RK2 + two_FT * Z_RK3 + Z_RK4) / FT(6.0)

    return next_R, next_Z
end

"""
    is_in_wall_by_cell_state(R, Z, Rmin, Zmin, NZ, inv_dR, inv_dZ, cell_state)

Check if point (R,Z) is inside the computational domain (not in wall).

Uses cell state array to determine if a point is inside the valid computational
region or hits a wall boundary.

# Arguments
- `R::FT`: Radial coordinate
- `Z::FT`: Vertical coordinate
- `Rmin::FT`: Minimum radial coordinate
- `Zmin::FT`: Minimum vertical coordinate
- `NZ::Int`: Number of vertical grid points
- `inv_dR::FT`: Inverse radial grid spacing
- `inv_dZ::FT`: Inverse vertical grid spacing
- `cell_state::Vector{Bool}`: Cell state array (true = valid, false = wall)

# Returns
- `state::Bool`: true if point is valid (not in wall), false if in wall
"""
function is_in_wall_by_cell_state(
        R::FT, Z::FT, Rmin::FT, Zmin::FT, NR::Int,
        inv_dR::FT, inv_dZ::FT, cell_state::AbstractMatrix{Bool}
    ) where {FT <: AbstractFloat}

    # Convert coordinates to grid indices
    Rid = floor(Int, (R - Rmin) * inv_dR) + 1
    Zid = floor(Int, (Z - Zmin) * inv_dZ) + 1

    # Bound R FIRST: the linear index alone folds `Rid = NR+1` onto the valid index of
    # `(1, Zid+1)`, so a rightward escape used to read as "still inside".
    if Rid < 1 || Rid > NR
        return false  # Outside domain = wall
    end

    # With R bounded, the linear check below is exactly a bound on Zid.
    nid = (Zid - 1) * NR + Rid
    if nid < 1 || nid > length(cell_state)
        return false  # Outside domain = wall
    end

    return cell_state[nid]
end

trace_from_node!(v, p::RK4Tracer, g::FieldLineGeometry, i::Int, j::Int, sgn::Int; max_length::Real = Inf) =
    trace_from_point!(v, p, g, (g.R1D[i], g.Z1D[j]), nothing, sgn, max_length)

# RK4 does not read the level
function trace_from_point!(v, p::RK4Tracer, g::FieldLineGeometry{FT}, x0, level, sgn, max_length) where {FT}
    B = g.nodal_B
    B === nothing && throw(ArgumentError("RK4Tracer follows the nodal B: build the geometry with `nodal_B = true`"))
    step_size = FT(p.step_cells) * min(abs(g.dR), abs(g.dZ))
    max_Lpol = isfinite(max_length) ? FT(max_length) : default_max_length(g)
    NR = size(g.ψ, 1)
    Rmin, Zmin, inv_dR, inv_dZ = g.R1D[1], g.Z1D[1], one(FT) / g.dR, one(FT) / g.dZ
    cells = g.inside === nothing ? trues(size(g.ψ)) : g.inside
    wall_checker = (R, Z) -> is_in_wall_by_cell_state(R, Z, Rmin, Zmin, NR, inv_dR, inv_dZ, cells)
    return rk4_trace!(
        v, FT(x0[1]), FT(x0[2]), sgn, B.BR, B.BZ, B.Bϕ, step_size, floor(Int, max_Lpol / step_size), max_Lpol, wall_checker,
    )
end

# RK4 in fixed steps from (R0, Z0): the wall is checked where each step starts, the budget is
# `max_steps` steps or `max_Lpol` of poloidal length, and the line has closed once the direction
# of its steps has turned through 2π. Each step end is visited with the interpolated B there.
function rk4_trace!(
        v, R0::FT, Z0::FT, direction::Int, interp_BR, interp_BZ, interp_Bphi,
        step_size::FT, max_steps::Int, max_Lpol::FT, wall_checker;
        detect_closure::Bool = true, closure_tolerance::FT = FT(1.0e-6),
    ) where {FT <: AbstractFloat}
    R_current, Z_current = R0, Z0
    Lpol = zero(FT)
    total_angle = zero(FT)
    prev_R, prev_Z = R0, Z0
    dl = direction * step_size

    # `Bpol = 0` at the start: purely toroidal, a closed circle
    BR = interp_BR(R_current, Z_current)
    BZ = interp_BZ(R_current, Z_current)
    Bpol = sqrt(BR^2 + BZ^2)
    Bpol == 0 && return (; termination = :null, R = R_current, Z = Z_current)
    visit!(v, R_current, Z_current, Bpol, interp_Bphi(R_current, Z_current))

    for step in 1:max_steps
        wall_checker(R_current, Z_current) || return (; termination = :wall, R = R_current, Z = Z_current)
        R_prev, Z_prev = R_current, Z_current
        R_current, Z_current = advance_step_along_b_rz_plane(dl, R_current, Z_current, interp_BR, interp_BZ)

        # The RK4 stages divide by Bpol at their lookahead points, so a step poking into a
        # Bpol = 0 region comes back NaN: a null at the last position actually reached.
        if !(isfinite(R_current) && isfinite(Z_current))
            return (; termination = :null, R = R_prev, Z = Z_prev)
        end

        BR = interp_BR(R_current, Z_current)
        BZ = interp_BZ(R_current, Z_current)
        Bphi = interp_Bphi(R_current, Z_current)
        Bpol = sqrt(BR^2 + BZ^2)
        Bpol == 0 && return (; termination = :null, R = R_current, Z = Z_current)
        visit!(v, R_current, Z_current, Bpol, Bphi)
        Lpol += sqrt((R_current - R_prev)^2 + (Z_current - Z_prev)^2)
        Lpol > max_Lpol && return (; termination = :trace_limit, R = R_current, Z = Z_current)

        # closure: the angle between consecutive steps, accumulated
        if detect_closure && step > 2
            v1_R, v1_Z = R_prev - prev_R, Z_prev - prev_Z
            v2_R, v2_Z = R_current - R_prev, Z_current - Z_prev
            cross_prod = v1_R * v2_Z - v1_Z * v2_R
            dot_prod = v1_R * v2_R + v1_Z * v2_Z
            if dot_prod != 0 || cross_prod != 0
                total_angle += atan(cross_prod, dot_prod)
                abs(total_angle) >= 2π - closure_tolerance && return (; termination = :closed, R = R_current, Z = Z_current)
            end
            prev_R, prev_Z = R_prev, Z_prev
        end
    end
    return (; termination = :trace_limit, R = R_current, Z = Z_current)
end

# Helper function to create 2D interpolation that matches MATLAB's griddedInterpolant behavior
function my_interpolation(R1D::Vector{FT}, Z1D::Vector{FT}, data_2d::Matrix{FT}; method::Symbol = :cubic) where {FT <: AbstractFloat}
    @assert method in (:nearst, :linear, :cubic) "Invalid interpolation method: $method"

    r1d = range(R1D[1], stop = R1D[end], length = length(R1D))
    z1d = range(Z1D[1], stop = Z1D[end], length = length(Z1D))

    # `ClampExtrap`, not the default `NoExtrap`: RK4 evaluates up to one step BEYOND
    # the current position before the next wall check can stop the trace, so a trace on
    # the last grid node queries just outside the domain — which `NoExtrap` answered
    # with a run-killing DomainError. Outside the grid there is no field data, so the
    # edge IS a wall; the clamped step is discarded by the very next wall check, and
    # flat clamping cannot diverge the way extending the cubic would.
    extrap = ClampExtrap()
    if method == :nearst
        itp = constant_interp((r1d, z1d), data_2d; extrap)
    elseif method == :linear
        itp = linear_interp((r1d, z1d), data_2d; extrap)
    elseif method == :cubic
        itp = cubic_interp((r1d, z1d), data_2d; extrap)
    end
    return itp
end
