# Surface functions on the closed region, and the electron parallel flux they carry. On closed
# flux surfaces the electron parallel particle flux per B is a surface function,
# n u∥ / B is a function of ψ. Its poloidal part is (n u∥/B) ∇ψ × ∇ϕ = ∇H × ∇ϕ with H′ = n u∥/B, so the flux through
# a control-volume face is 2π (H(ψ_end) − H(ψ_start)) at the face's two corners (the dual vertices),
# and the divergence of every control volume telescopes to zero. The constraint domain is the
# support of H on the admissible nodes, which leaves every face between it and the rest at one H.
#
# Levels are the flux from the axis, Δψ = s (ψ − ψ_axis), signed to grow outward: zero on the axis
# and positive on every surface around it, s = +1 about a minimum of ψ and −1 about a maximum. The
# LCFS bisection, the sublevel sweep's order and the double-null tolerance are all in Δψ, so none of
# them depends on the sign of the current.

"""
    closed_surface_basis(G, ψ, admissible; inside = nothing, gram_weights = nothing, nknots = default)
        -> ClosedSurfaceBasis or nothing

The constraint domain and its surface functions for the nodes marked `admissible` (an `(NR, NZ)`
mask: closed field lines inside the wall, not empty). Nodes on the grid's perimeter are never
admitted; nodes the admissible ones enclose are (the node on the O-point, where field-line
following meets a null).
- The LCFS is the smallest corner level of any face between an admissible node and the rest,
  and no higher than the outermost level the bicubic tracer closes around the O-point inside the
  wall cells `inside`. Ω_c is every admissible node with a corner inside the LCFS, so the faces
  between Ω_c and the rest have both corners at or beyond it and carry no flux.
- `nothing` when the region has no O-point, more than one, or no level that closes.
- `nknots` basis functions (default: one per grid spacing, plus one) are merged, nearest the
  axis first, until each covers two nodes and the Gram matrix of `gram_weights` (nodal, default 1) has a condition
  number below 10⁸; a single remaining level is one constant function.
"""
function closed_surface_basis(
        G::GridGeometry{FT}, ψ::AbstractMatrix{FT}, admissible::AbstractMatrix{Bool};
        inside::Union{Nothing, AbstractMatrix{Bool}} = nothing, gram_weights = nothing,
        nknots::Union{Nothing, Int} = nothing,
    ) where {FT <: AbstractFloat}
    NR, NZ = size(ψ)
    A = copy(admissible)
    A[[1, NR], :] .= false
    A[:, [1, NZ]] .= false
    fill_holes!(A)
    region = findall(vec(A))
    isempty(region) && return nothing
    # one interpolant for the O-point and for the tracer that verifies the LCFS
    g = FieldLineGeometry(G, ψ; inside)
    o = find_o_point(G, ψ, region; itp = g.itp)
    (o === nothing || !o.converged) && return nothing
    count_o_points(G, ψ, region, g.itp) > 1 && return nothing
    ψ_axis = FT(o.ψ)
    s = sign(sum(ψ[region]) - length(region) * ψ_axis)
    s == 0 && return nothing
    # Δψ at the dual corners: zero on the axis, growing outward
    dc = [s * ((ψ[i, j] + ψ[i + 1, j] + ψ[i, j + 1] + ψ[i + 1, j + 1]) / 4 - ψ_axis) for i in 1:(NR - 1), j in 1:(NZ - 1)]
    Δψ_lcfs = lcfs_level(g, o, s, first_boundary_corner(A, dc))
    Δψ_lcfs === nothing && return nothing
    lin = LinearIndices((NR, NZ))
    nodes = [lin[i, j] for j in 2:(NZ - 1), i in 2:(NR - 1) if A[i, j] && corner_min(dc, i, j) < Δψ_lcfs]
    sort!(nodes)
    isempty(nodes) && return nothing
    active = falses(NR - 1, NZ - 1)
    for k in nodes
        i, j = mod1(k, NR), cld(k, NR)
        for (a, c) in ((i - 1, j - 1), (i, j - 1), (i - 1, j), (i, j))
            dc[a, c] < Δψ_lcfs && (active[a, c] = true)
        end
    end
    ψN = [s * (ψ[k] - ψ_axis) / Δψ_lcfs for k in nodes]
    V = [FT(2π) * G.R2D[k] * G.dR * G.dZ for k in nodes]
    gram_w = gram_weights === nothing ? ones(FT, length(nodes)) : FT.(vec(gram_weights)[nodes])
    n0 = nknots === nothing ? default_surface_count(nodes) + 1 : nknots
    knots = merge_knots(collect(range(zero(FT), one(FT); length = max(n0, 2))), ψN, gram_w)
    axis = (; R = FT(o.R), Z = FT(o.Z), ψ = ψ_axis)
    return ClosedSurfaceBasis{FT}(
        nodes, V, axis, ψ_axis + s * Δψ_lcfs, ψN, dc ./ Δψ_lcfs, active, knots,
        basis_matrix(knots, ψN; beyond = :zero), basis_matrix(knots, ψN; beyond = :constant), (NR, NZ),
    )
end

# Admit the non-admissible nodes the admissible ones enclose: those not reached from the grid's
# perimeter through non-admissible nodes. Field-line following ends at the null of a node on the
# O-point and leaves it out; the LCFS must come from the region's outer edge.
function fill_holes!(A::AbstractMatrix{Bool})
    NR, NZ = size(A)
    reached = falses(NR, NZ)
    stack = Tuple{Int, Int}[]
    for j in 1:NZ, i in 1:NR
        (i in (1, NR) || j in (1, NZ)) && !A[i, j] && (reached[i, j] = true; push!(stack, (i, j)))
    end
    while !isempty(stack)
        i, j = pop!(stack)
        for (a, c) in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1))
            (1 <= a <= NR && 1 <= c <= NZ) || continue
            (A[a, c] || reached[a, c]) && continue
            reached[a, c] = true
            push!(stack, (a, c))
        end
    end
    A .|= .!reached
    return A
end

# the smallest of the four corners of node (i, j)'s control volume
@inline corner_min(dc, i, j) = min(dc[i - 1, j - 1], dc[i, j - 1], dc[i - 1, j], dc[i, j])

# The smallest corner level of the faces between an admissible node and a node that is not: below
# it, H changes only across faces whose two sides are admissible.
function first_boundary_corner(A::AbstractMatrix{Bool}, dc::AbstractMatrix{FT}) where {FT}
    NR, NZ = size(A)
    Δψ = FT(Inf)
    for j in 2:(NZ - 1), i in 1:(NR - 1)          # R faces (i, j)–(i + 1, j): corners (i, j − 1), (i, j)
        A[i, j] == A[i + 1, j] && continue
        Δψ = min(Δψ, dc[i, j - 1], dc[i, j])
    end
    for j in 1:(NZ - 1), i in 2:(NR - 1)          # Z faces (i, j)–(i, j + 1): corners (i − 1, j), (i, j)
        A[i, j] == A[i, j + 1] && continue
        Δψ = min(Δψ, dc[i - 1, j], dc[i, j])
    end
    return Δψ
end

# The LCFS's flux from the axis, Δψ = s(ψ − ψ_axis), at or below `Δψ0`, whose contour the bicubic
# tracer closes around `o` inside the wall: Δψ0 itself, or the outermost closed level between the
# axis and Δψ0, found from a level confirmed closed by bisection. A trace that hits the wall, runs
# out of budget, meets a null or is not seeded counts as not closed. `nothing` when no level closes.
function lcfs_level(g::FieldLineGeometry{FT}, o, s, Δψ0) where {FT}
    isfinite(Δψ0) && Δψ0 > 0 || return nothing
    closes(Δψ) = trace_around!(PathLengths{FT}(), CubicTracer(), g, FT(o.ψ) + s * Δψ, o).termination === :closed
    closes(Δψ0) && return Δψ0
    lo = Δψ0
    for _ in 1:8
        lo /= 2
        closes(lo) && break
    end
    closes(lo) || return nothing
    hi = Δψ0
    for _ in 1:40
        mid = (lo + hi) / 2
        closes(mid) ? (lo = mid) : (hi = mid)
    end
    return lo
end

# The configurations `magnetic_lcfs` names, in the order of their codes in the 2D snapshots
# (`lcfs_configuration`: 0 none, else the index here).
const LCFS_CONFIGURATIONS = (:limited, :single_null, :double_null)

# A visitor that keeps the extent of a traced contour: its smallest and largest R and Z.
mutable struct ContourExtent{FT <: AbstractFloat}
    R_min::FT
    R_max::FT
    Z_min::FT
    Z_max::FT
end
ContourExtent{FT}() where {FT} = ContourExtent{FT}(FT(Inf), FT(-Inf), FT(Inf), FT(-Inf))
function visit!(e::ContourExtent, R, Z, Bpol, Bϕ)
    e.R_min, e.R_max = min(e.R_min, R), max(e.R_max, R)
    e.Z_min, e.Z_max = min(e.Z_min, Z), max(e.Z_max, Z)
    return e
end

"""
    magnetic_lcfs(G, ψ, o, in_wall; g = FieldLineGeometry(G, ψ), double_null_tolerance = 0.01)
        -> (; ψ_lcfs, ψ_trace, configuration, xpoints, contact, geometric_centre) or nothing

The last closed flux surface around the O-point `o` (`(; R, Z, ψ)`) as the field and the wall set
it, with no density or field-line following behind it (the closed-surface basis's LCFS stops at its
admissible region instead). Computed on request; the step does not call it.

- `configuration`: `:limited` when the surfaces nested around `o` touch the wall first, at the wall
  node `contact` (`(; R, Z)`, NaN otherwise); `:single_null` when they meet an X-point first;
  `:double_null` when a second X-point's flux from the axis is within `double_null_tolerance`
  (relative) of the first's. `xpoints` holds the X-points on the LCFS, `(; R, Z, ψ)`, nearest the
  axis in flux first.
- On the grid ([`sublevel_sweep`](@ref)): the nodes in the wall (`in_wall`, NR × NZ) join their
  neighbours' sublevel sets in increasing flux from the axis, Δψ = s(ψ − ψ_axis) ≥ 0, which grows
  outward (s = +1 about a minimum of ψ, −1 about a maximum). The set of `o` passes a saddle where a node joins it to a set whose lowest node is
  more than two cells from `o`, and touches the wall where it takes a wall node.
- On the bicubic interpolant: each saddle is the X-point Newton reaches from its node; a contact's
  level is the outermost the tracer of `g` closes around `o` below the node's, as for the basis.
- `ψ_trace`: a level at or just inside `ψ_lcfs` that the tracer closes around `o` (a separatrix
  runs through a null): within 10⁻⁴ … 10⁻¹ of the separatrix's flux from the axis, the closest
  that closes.
- `geometric_centre`: `(; R, Z)` = ((R_max + R_min)/2, (Z_max + Z_min)/2) of the LCFS, the
  reference of the Shafranov shift: the extent of the contour traced at `ψ_trace` and of the
  X-points on the LCFS, which the separatrix runs through.

`nothing` when `o` is not an extremum of the interpolant, or no level closes.
"""
function magnetic_lcfs(
        G::GridGeometry{FT}, ψ::AbstractMatrix{FT}, o, in_wall::AbstractMatrix{Bool};
        g::FieldLineGeometry{FT} = FieldLineGeometry(G, ψ), double_null_tolerance::Real = 0.01,
    ) where {FT <: AbstractFloat}
    H = hessian(g.itp, (o.R, o.Z))
    H[1, 1] * H[2, 2] - H[1, 2] * H[2, 1] > 0 || return nothing
    s = sign(H[1, 1] + H[2, 2])                       # +1 about a minimum of ψ, −1 about a maximum
    ψ_axis = FT(o.ψ)
    tol = FT(double_null_tolerance)
    sweep = sublevel_sweep(G, ψ, o, in_wall, s, tol)
    sweep === nothing && return nothing
    if isempty(sweep.saddles)                         # the wall first: a limited plasma
        k = sweep.contact
        Δψ_lcfs = lcfs_level(g, o, s, s * (ψ[k] - ψ_axis))
        Δψ_lcfs === nothing && return nothing
        ψ_lcfs = ψ_axis + s * Δψ_lcfs
        xpoints = NamedTuple{(:R, :Z, :ψ), NTuple{3, FT}}[]
        geometric_centre = lcfs_geometric_centre(g, o, ψ_lcfs, xpoints)
        return (; ψ_lcfs, ψ_trace = ψ_lcfs, configuration = :limited, xpoints, contact = (; R = G.R2D[k], Z = G.Z2D[k]), geometric_centre)
    end
    xpoints = saddle_xpoints(G, ψ, g, sweep.saddles)
    Δψ_of(p) = s * (p.ψ - ψ_axis)
    sort!(xpoints; by = Δψ_of)
    Δψ_x = Δψ_of(xpoints[1])
    Δψ_x > 0 || return nothing
    filter!(p -> Δψ_of(p) <= Δψ_x * (1 + tol), xpoints)
    Δψ_trace = level_inside(g, o, s, Δψ_x)
    Δψ_trace === nothing && return nothing
    configuration = length(xpoints) >= 2 ? :double_null : :single_null
    contact = (; R = FT(NaN), Z = FT(NaN))
    ψ_trace = ψ_axis + s * Δψ_trace
    geometric_centre = lcfs_geometric_centre(g, o, ψ_trace, xpoints)
    return (; ψ_lcfs = ψ_axis + s * Δψ_x, ψ_trace, configuration, xpoints, contact, geometric_centre)
end

# ((R_max + R_min)/2, (Z_max + Z_min)/2) of the contour the tracer follows around `o` at `ψ_trace`
# and of `xpoints`; NaN if the trace does not close.
function lcfs_geometric_centre(g::FieldLineGeometry{FT}, o, ψ_trace, xpoints) where {FT}
    e = ContourExtent{FT}()
    trace_around!(e, CubicTracer(), g, ψ_trace, o).termination === :closed || return (; R = FT(NaN), Z = FT(NaN))
    for x in xpoints
        visit!(e, x.R, x.Z, zero(FT), zero(FT))
    end
    return (; R = (e.R_min + e.R_max) / 2, Z = (e.Z_min + e.Z_max) / 2)
end

"""
    sublevel_sweep(G, ψ, o, in_wall, s, tol) -> (; saddles, contact) or nothing

The sublevel sets of Δψ = s(ψ − ψ_axis) on the nodes in the wall, grown in increasing Δψ by
union–find: each node joins its neighbours' sets, and each set keeps its lowest node. Where a node
would join the set of `o` (lowest node within two cells of `o`) to another set, the set of `o` has
reached a saddle: the node is recorded and the two sets are kept apart, so that the sweep goes on
to (1 + `tol`) times the first saddle's level for a second one. The first wall node (a neighbour
out of the wall, or on the grid's edge) the set of `o` takes ends the sweep as `contact` (0: none).
`saddles` (node indices) is empty when the wall came first; `nothing` when neither happens. The
scratch is from the pool.
"""
function sublevel_sweep(G::GridGeometry{FT}, ψ::AbstractMatrix{FT}, o, in_wall::AbstractMatrix{Bool}, s, tol::FT) where {FT}
    NR, NZ = size(ψ)
    ψ_axis = FT(o.ψ)
    near_o(k) = abs(G.R2D[k] - o.R) <= 2 * G.dR && abs(G.Z2D[k] - o.Z) <= 2 * G.dZ
    on_wall(i, j) = i in (1, NR) || j in (1, NZ) || !(in_wall[i - 1, j] && in_wall[i + 1, j] && in_wall[i, j - 1] && in_wall[i, j + 1])
    contact = Ref(0)            # outside the pool block: an index read from a pooled array is still an index
    saddles = @with_pool pool begin
        Δψ = acquire!(pool, FT, NR * NZ)
        for k in eachindex(Δψ)
            Δψ[k] = s * (ψ[k] - ψ_axis)
        end
        n = count(in_wall)
        order = acquire!(pool, Int, n)
        m = 0
        for k in eachindex(in_wall)
            in_wall[k] || continue
            m += 1
            order[m] = k
        end
        sort!(order; by = k -> Δψ[k], alg = QuickSort)
        parent = zeros!(pool, Int, NR * NZ)           # 0: not in a set yet
        lowest = acquire!(pool, Int, NR * NZ)          # a set's lowest node, kept at its root
        saddle_nodes = acquire!(pool, Int, n)
        n_saddle, first_level = 0, FT(Inf)
        for k in order
            Δψ[k] > first_level * (1 + tol) && break
            i, j = mod1(k, NR), cld(k, NR)
            mine, other = near_o(k), false
            for (a, b) in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1))
                (1 <= a <= NR && 1 <= b <= NZ) || continue
                nb = (b - 1) * NR + a
                parent[nb] == 0 && continue
                near_o(lowest[set_root!(parent, nb)]) ? (mine = true) : (other = true)
            end
            if mine && other
                n_saddle += 1
                saddle_nodes[n_saddle] = k
                first_level = min(first_level, Δψ[k])
            elseif mine && on_wall(i, j)
                contact[] = k
                break
            end
            parent[k], lowest[k] = k, k
            for (a, b) in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1))
                (1 <= a <= NR && 1 <= b <= NZ) || continue
                nb = (b - 1) * NR + a
                parent[nb] == 0 && continue
                r = set_root!(parent, nb)
                (r == k || (mine && !near_o(lowest[r]))) && continue     # the set of `o` stays apart
                parent[r] = k
                Δψ[lowest[r]] < Δψ[lowest[k]] && (lowest[k] = lowest[r])
            end
        end
        Int[saddle_nodes[i] for i in 1:n_saddle]       # a fresh vector leaves the block, never a pooled one
    end
    isempty(saddles) && contact[] == 0 && return nothing
    return (; saddles, contact = contact[])
end

# A level the tracer closes around `o` just inside the separatrix at flux `Δψ_x` from the axis: the
# first of 10⁻⁴, 10⁻³, 10⁻², 10⁻¹ below it that closes (beside a null the nearest can fail), else
# the outermost closed level by bisection.
function level_inside(g::FieldLineGeometry{FT}, o, s, Δψ_x) where {FT}
    for gap in (FT(1.0e-4), FT(1.0e-3), FT(1.0e-2), FT(1.0e-1))
        Δψ = Δψ_x * (1 - gap)
        trace_around!(PathLengths{FT}(), CubicTracer(), g, FT(o.ψ) + s * Δψ, o).termination === :closed && return Δψ
    end
    return lcfs_level(g, o, s, Δψ_x * (1 - FT(1.0e-1)))
end

# The root of node k's set, halving the path on the way.
function set_root!(parent, k)
    while parent[k] != k
        parent[k] = parent[parent[k]]
        k = parent[k]
    end
    return k
end

# The X-points Newton reaches from the saddle nodes of a sweep, each kept once (within two cells):
# a converged root within two cells of its node, with a negative Hessian determinant. When no
# search converges, the first node stands for its X-point.
function saddle_xpoints(G::GridGeometry{FT}, ψ::AbstractMatrix{FT}, g, saddles) where {FT}
    xpoints = NamedTuple{(:R, :Z, :ψ), NTuple{3, FT}}[]
    tol, h = max(FT(1.0e-10), 64 * eps(FT)), min(G.dR, G.dZ)
    for k in saddles
        x0 = (G.R2D[k], G.Z2D[k])
        x, converged, det = newton_critical_point(g.itp, x0, h; tol, maxit = 50)
        (converged && det < 0 && abs(x[1] - x0[1]) <= 2 * G.dR && abs(x[2] - x0[2]) <= 2 * G.dZ) || continue
        any(p -> hypot((x[1] - p.R) / G.dR, (x[2] - p.Z) / G.dZ) < 2, xpoints) && continue
        push!(xpoints, (; R = FT(x[1]), Z = FT(x[2]), ψ = FT(g.itp(x))))
    end
    isempty(xpoints) && push!(xpoints, (; R = G.R2D[saddles[1]], Z = G.Z2D[saddles[1]], ψ = ψ[saddles[1]]))
    return xpoints
end

# The knot weights at x: the values of the basis functions of `knots` (row: point, column: knot);
# beyond the last knot zero (`:zero`) or the last basis function's 1 (`:constant`).
# A single knot is one constant function on [0, 1].
function basis_matrix(knots::Vector{FT}, x::Vector{FT}; beyond::Symbol) where {FT}
    M = length(knots)
    I, J, Vs = Int[], Int[], FT[]
    for (r, xr) in enumerate(x)
        xr > 1 && beyond === :zero && continue
        xc = clamp(xr, zero(FT), one(FT))
        if M == 1
            push!(I, r); push!(J, 1); push!(Vs, one(FT))
            continue
        end
        s = min(searchsortedlast(knots, xc), M - 1)
        t = (xc - knots[s]) / (knots[s + 1] - knots[s])
        push!(I, r); push!(J, s); push!(Vs, 1 - t)
        push!(I, r); push!(J, s + 1); push!(Vs, t)
    end
    return dropzeros!(sparse(I, J, Vs, length(x), M))
end

# Merge basis functions, the interior knot nearest the axis first, until each covers two nodes and
# the weighted Gram matrix is well conditioned; one knot left is one constant.
function merge_knots(knots::Vector{FT}, x::Vector{FT}, gram_w::Vector{FT}) where {FT}
    while length(knots) > 2
        knot_w = basis_matrix(knots, x; beyond = :zero)
        covered = vec(sum(knot_w .> 0; dims = 1))
        gram = Matrix(knot_w' * Diagonal(gram_w) * knot_w)
        all(>=(2), covered) && cond(gram) < 1.0e8 && return knots
        bad = findfirst(<(2), covered)
        s = bad === nothing ? argmin(diag(gram)) : bad
        deleteat!(knots, clamp(s, 2, length(knots) - 1))
    end
    knot_w = basis_matrix(knots, x; beyond = :zero)
    all(>=(2), vec(sum(knot_w .> 0; dims = 1))) && cond(Matrix(knot_w' * Diagonal(gram_w) * knot_w)) < 1.0e8 && return knots
    return [zero(FT)]
end

# ∫₀^min(x, 1) of the piecewise-linear function with values vk at the knots (one knot: a constant)
function basis_integral(knots::Vector{FT}, vk::AbstractVector, x::FT) where {FT}
    x = clamp(x, zero(FT), one(FT))
    length(knots) == 1 && return vk[1] * x
    acc = zero(promote_type(FT, eltype(vk)))
    for s in 1:(length(knots) - 1)
        a, b = knots[s], knots[s + 1]
        x <= a && break
        e = min(x, b)
        ve = vk[s] + (vk[s + 1] - vk[s]) * (e - a) / (b - a)
        acc += (vk[s] + ve) / 2 * (e - a)
    end
    return acc
end

"""
    stream_function_fluxes(basis, Γ_over_B) -> (ΓR, ΓZ)

The poloidal particle flux of n u∥ b through every control-volume face of the grid, for the
surface function n u∥/B with values `Γ_over_B` at the knots: `ΓR[i, j]` through the face between nodes
(i, j) and (i + 1, j) along +R, `ΓZ[i, j]` between (i, j) and (i, j + 1) along +Z, each
2π (H(ψ_end) − H(ψ_start)) at the face's corners (ψ per radian, B_R = −∂_Zψ/R, B_Z = ∂_Rψ/R).
H varies only at the corners of Ω_c inside the LCFS (`corner_active`), so every face outside Ω_c
and between Ω_c and the rest carries none.
"""
function stream_function_fluxes(b::ClosedSurfaceBasis{FT}, Γ_over_B::AbstractVector) where {FT}
    NR, NZ = b.dims
    J = b.ψ_lcfs - b.axis.ψ
    H_lcfs = J * basis_integral(b.knots, Γ_over_B, one(FT))
    H = [b.corner_active[c] ? J * basis_integral(b.knots, Γ_over_B, b.corner_ψN[c]) : H_lcfs for c in CartesianIndices(b.corner_ψN)]
    T = promote_type(FT, eltype(Γ_over_B))
    ΓR = zeros(T, NR - 1, NZ)
    ΓZ = zeros(T, NR, NZ - 1)
    for j in 2:(NZ - 1), i in 1:(NR - 1)
        ΓR[i, j] = 2π * (H[i, j - 1] - H[i, j])
    end
    for j in 1:(NZ - 1), i in 2:(NR - 1)
        ΓZ[i, j] = 2π * (H[i, j] - H[i - 1, j])
    end
    return ΓR, ΓZ
end

"""
    face_outflow(ΓR, ΓZ) -> (NR, NZ) array

The net outflow of every control volume from face fluxes laid out as in
[`stream_function_fluxes`](@ref).
"""
function face_outflow(ΓR::AbstractMatrix{T}, ΓZ::AbstractMatrix{T}) where {T}
    NR, NZ = size(ΓR, 1) + 1, size(ΓR, 2)
    out = zeros(T, NR, NZ)
    for j in 1:NZ, i in 1:(NR - 1)
        out[i, j] += ΓR[i, j]
        out[i + 1, j] -= ΓR[i, j]
    end
    for j in 1:(NZ - 1), i in 1:NR
        out[i, j] += ΓZ[i, j]
        out[i, j + 1] -= ΓZ[i, j]
    end
    return out
end

"""
    pressure_operator(basis, g) -> B_p (M × length(basis.nodes))

The pressure's generalized force on the surface functions, B_p τ ≈ ∫ h_s τ B·∇g dV for the
nodal τ = e T_e on Ω_c and g = ln n on the grid. It is built on the stream-function face fluxes
of each basis function, Σ_f Γ_f^(s) Δg_f τ̄_f, applied to the part of τ that is not a surface function
(τ minus its volume-weighted projection on `basis.knot_weights_held`): the surface part gives exactly zero, as
∫ h_s τ̄ B·∇g dV does, so a constant or surface temperature exerts no force. The Galerkin row
takes −B_p τ / m_e; the matching compression power of node i is τ_i (B_pᵀ Γ_over_B)_i, Γ_over_B the knot
values of n u∥/B.
"""
function pressure_operator(b::ClosedSurfaceBasis{FT}, g::AbstractMatrix) where {FT}
    NR, NZ = b.dims
    M, N = length(b.knots), length(b.nodes)
    col = zeros(Int, NR * NZ)
    col[b.nodes] .= 1:N
    X = zeros(FT, M, N)
    I1, I2 = zeros(FT, M), zeros(FT, M)
    scale = 2π * (b.ψ_lcfs - b.axis.ψ)
    # a basis function's H at a corner: ∫₀^x of it, x the corner's ψN′ inside the LCFS, 1 elsewhere
    x(i, j) = b.corner_active[i, j] ? b.corner_ψN[i, j] : one(FT)
    # every basis function's flux 2π (H(c1) − H(c2)) through the face from node a to h, at once
    function face!(c1, c2, a, h, Δg)
        (b.corner_active[c1...] || b.corner_active[c2...]) || return
        basis_integrals!(I1, b.knots, x(c1...))
        basis_integrals!(I2, b.knots, x(c2...))
        for s in 1:M
            Γ = scale * (I1[s] - I2[s])
            iszero(Γ) && continue
            (col[a] > 0 && col[h] > 0) || error("a face carrying flux leaves Ω_c")
            X[s, col[a]] += Γ * Δg / 2
            X[s, col[h]] += Γ * Δg / 2
        end
        return
    end
    for j in 2:(NZ - 1), i in 1:(NR - 1)          # +R faces, corners (i, j − 1) and (i, j)
        a = i + (j - 1) * NR
        face!((i, j - 1), (i, j), a, a + 1, g[i + 1, j] - g[i, j])
    end
    for j in 1:(NZ - 1), i in 2:(NR - 1)          # +Z faces, corners (i, j) and (i − 1, j)
        a = i + (j - 1) * NR
        face!((i, j), (i - 1, j), a, a + NR, g[i, j + 1] - g[i, j])
    end
    # remove the surface part of τ: X (I − S), S = W_h (W_hᵀ diag(V) W_h)⁻¹ W_hᵀ diag(V),
    # W_h = `knot_weights_held`
    knot_w_held = Matrix(b.knot_weights_held)
    vol_knot_w = b.V .* knot_w_held
    return X .- ((X * knot_w_held) / (knot_w_held' * vol_knot_w)) * vol_knot_w'
end

# ∫₀^min(x, 1) of each basis function of `knots` into v; one knot is one constant
function basis_integrals!(v::AbstractVector, knots::Vector{FT}, x::FT) where {FT}
    x = clamp(x, zero(FT), one(FT))
    length(knots) == 1 && (v[1] = x; return v)
    fill!(v, zero(FT))
    for t in 1:(length(knots) - 1)
        a = knots[t]
        x <= a && break
        h = knots[t + 1] - a
        d = min(x, knots[t + 1]) - a
        # on [a, a + h] basis function t falls from 1 to 0 and t + 1 rises from 0 to 1
        v[t] += d - d^2 / (2h)
        v[t + 1] += d^2 / (2h)
    end
    return v
end

# ── diagnostics on Ω_c ───────────────────────────────────────────────────────────

"""
    galerkin_flux(basis, n, u, B) -> Γ_over_B

The surface flux closest to the electron velocity `u` in the kinetic-energy metric: the Galerkin
projection (Wᵀ diag(V B²/n) W) Γ_over_B = Wᵀ diag(V B) u over Ω_c, W = `knot_weights` and Γ_over_B the knot
values of n u∥/B. A velocity that is a surface flux, u = (B/n) W Γ_over_B, gives back its Γ_over_B. `n`, `u`, `B` are grid arrays.
"""
function galerkin_flux(b::ClosedSurfaceBasis, n::AbstractVecOrMat, u::AbstractVecOrMat, B::AbstractVecOrMat)
    Bk, nk = B[b.nodes], n[b.nodes]
    knot_w = b.knot_weights
    return Matrix(knot_w' * Diagonal(b.V .* Bk .^ 2 ./ nk) * knot_w) \ (knot_w' * (b.V .* Bk .* u[b.nodes]))
end

"""
    surface_residual(basis, n, u, B; Γ_over_B = galerkin_flux(basis, n, u, B)) -> Real

How far n u/B on Ω_c is from the surface function W Γ_over_B (W = `knot_weights`), as the volume-weighted rms of the
difference over that of n u/B: 0 for a surface flux. Beyond the LCFS, where the surface function ends, all of
n u/B counts.
"""
function surface_residual(
        b::ClosedSurfaceBasis, n::AbstractVecOrMat, u::AbstractVecOrMat, B::AbstractVecOrMat;
        Γ_over_B = galerkin_flux(b, n, u, B),
    )
    f = n[b.nodes] .* u[b.nodes] ./ B[b.nodes]
    total = sum(b.V .* f .^ 2)
    iszero(total) && return zero(total)
    return sqrt(sum(b.V .* (f .- b.knot_weights * Γ_over_B) .^ 2) / total)
end

"""
    electron_kinetic_energy(G, n, u, nodes; me) -> Real

½ mₑ n u² summed over the toroidal control volumes 2πR ΔR ΔZ of `nodes` [J].
"""
electron_kinetic_energy(G::GridGeometry, n::AbstractVecOrMat, u::AbstractVecOrMat, nodes; me::Real) =
    sum(me / 2 * n[k] * u[k]^2 * 2π * G.R2D[k] * G.dR * G.dZ for k in nodes; init = zero(me * G.dR))

"""
    surface_statistics(basis, ne, Zni, ue, ui, Te_eV; me, ee) -> (; n_min, drift, imbalance)

Per basis function of `basis`, over the Ω_c nodes it covers: the smallest electron density, the largest
electron–ion drift against the electron thermal speed, |uₑ − uᵢ| / √(e Tₑ / mₑ), and the largest
relative charge imbalance |nₑ − Σ zₛ nₛ| / nₑ, with `Zni` the ion charge density Σ zₛ nₛ.
"""
function surface_statistics(
        b::ClosedSurfaceBasis{FT}, ne, Zni, ue, ui, Te_eV; me::Real, ee::Real,
    ) where {FT}
    M = length(b.knots)
    n_min, drift, imbalance = fill(FT(Inf), M), zeros(FT, M), zeros(FT, M)
    rows, vals = rowvals(b.knot_weights), nonzeros(b.knot_weights)
    for s in 1:M, p in nzrange(b.knot_weights, s)
        iszero(vals[p]) && continue
        k = b.nodes[rows[p]]
        n_min[s] = min(n_min[s], ne[k])
        drift[s] = max(drift[s], abs(ue[k] - ui[k]) / sqrt(ee * Te_eV[k] / me))
        imbalance[s] = max(imbalance[s], abs(ne[k] - Zni[k]) / ne[k])
    end
    return (; n_min, drift, imbalance)
end

"""
    current_outside_constraint(basis, closed, J) -> Real

The share of the toroidal current `J` (grid array) on the closed nodes `closed` that the
constraint leaves out: nodes outside Ω_c, and the edge layer (Ω_c beyond the LCFS), where n u∥/B
is not constrained.
"""
function current_outside_constraint(b::ClosedSurfaceBasis, closed::AbstractVector{<:Integer}, J::AbstractVecOrMat)
    total = sum(J[k] for k in closed; init = zero(eltype(J)))
    iszero(total) && return zero(total)
    lost = zero(total)
    for k in closed
        r = searchsortedfirst(b.nodes, k)
        (r > length(b.nodes) || b.nodes[r] != k || b.ψN[r] > 1) && (lost += J[k])
    end
    return lost / total
end

# ── the constrained momentum step ────────────────────────────────────────────────

# the constrained rows (inside the LCFS): their positions in Ω_c, their grid nodes, their knot weights
function inside_lcfs(b::ClosedSurfaceBasis)
    sel = findall(<=(1), b.ψN)
    return sel, b.nodes[sel], b.knot_weights[sel, :]
end

"""
    constrained_system(sys, S, basis) -> ConstrainedSystem

`sys` when it was built for the pattern of `S` and of `basis`'s basis functions inside the LCFS; a new
[`ConstrainedSystem`](@ref) otherwise, or for `sys === nothing`.
"""
function constrained_system(
        sys::Union{Nothing, ConstrainedSystem{FT}}, S::SparseMatrixCSC{FT}, b::ClosedSurfaceBasis,
    ) where {FT <: AbstractFloat}
    _, cu, knot_w_in = inside_lcfs(b)
    key = hash((size(S), S.colptr, S.rowval, cu, knot_w_in.colptr, knot_w_in.rowval))
    sys !== nothing && sys.key == key && return sys
    return ConstrainedSystem(S, cu, knot_w_in, key)
end

function ConstrainedSystem(S::SparseMatrixCSC{FT}, cu::Vector{Int}, knot_w_in::SparseMatrixCSC, key::UInt) where {FT <: AbstractFloat}
    L, M = size(S, 1), size(knot_w_in, 2)
    Sr = rowvals(S)
    c_of = zeros(Int, L)                  # a row's position in cu, 0 when it keeps its equation
    c_of[cu] .= eachindex(cu)
    col_S = zeros(Int, length(Sr))
    for j in 1:L, p in nzrange(S, j)
        col_S[p] = j
    end
    basis_row = copy(rowvals(knot_w_in))
    basis_col = [s for s in 1:M for _ in nzrange(knot_w_in, s)]
    on_row = [Int[] for _ in cu]          # S's stored entries on each constrained row
    I, J = Int[], Int[]
    for p in eachindex(Sr)
        c = c_of[Sr[p]]
        c == 0 ? (push!(I, Sr[p]); push!(J, col_S[p])) : push!(on_row[c], p)
    end
    append!(I, cu)
    append!(J, cu)
    for (c, s) in zip(basis_row, basis_col)
        push!(I, cu[c], L + s)         # T at (k, L + s), the shift at (L + s, k)
        push!(J, L + s, cu[c])
        for p in on_row[c]
            push!(I, L + s)
            push!(J, col_S[p])
        end
    end
    A = sparse(I, J, ones(FT, length(I)), L + M, L + M)
    fill!(nonzeros(A), zero(FT))
    pos(r, c) = first(nzrange(A, c)) + searchsortedfirst(view(rowvals(A), nzrange(A, c)), r) - 1
    pos_S = [c_of[Sr[p]] == 0 ? pos(Sr[p], col_S[p]) : 0 for p in eachindex(Sr)]
    gal = [(p, g) for g in eachindex(basis_row) for p in on_row[basis_row[g]]]
    return ConstrainedSystem{FT}(
        key, L, cu, A, pos_S, [pos(k, k) for k in cu], basis_row, basis_col,
        [pos(cu[c], L + s) for (c, s) in zip(basis_row, basis_col)], [pos(L + s, cu[c]) for (c, s) in zip(basis_row, basis_col)],
        first.(gal), last.(gal), [pos(L + basis_col[g], col_S[p]) for (p, g) in gal],
        zeros(FT, length(basis_row)), zeros(FT, M), one(FT), SparseLUSolver{FT}(), true,
    )
end

"""
    update_constrained_system!(sys, S, basis, n, B; shift = nothing) -> sys

Write the values of `S` (on the pattern `sys` was built for), of the constraint at the density
`n` and field `B` (grid arrays) and of the Galerkin rows into `sys`, and refactorize it.
`shift`, one value per constrained row, is added to those rows' diagonal of S, which only the
Galerkin rows read.
"""
function update_constrained_system!(
        sys::ConstrainedSystem{FT}, S::SparseMatrixCSC{FT}, b::ClosedSurfaceBasis, n::AbstractVecOrMat, B::AbstractVecOrMat;
        shift::Union{Nothing, AbstractVector} = nothing,
    ) where {FT <: AbstractFloat}
    sel, cu, knot_w_in = inside_lcfs(b)
    cu == sys.cu || throw(ArgumentError("update_constrained_system!: the basis is not the one the system was built for"))
    knot_w_vals, Sv, nz = nonzeros(knot_w_in), nonzeros(S), nonzeros(sys.matrix)
    length(Sv) == length(sys.pos_S) || throw(ArgumentError("update_constrained_system!: S is not on the system's pattern"))
    # the Galerkin row coefficients W V B, each row normalized by its sum
    fill!(sys.qs, zero(FT))
    for g in eachindex(sys.q)
        c = sys.basis_row[g]
        sys.q[g] = knot_w_vals[g] * b.V[sel[c]] * B[cu[c]]
        sys.qs[sys.basis_col[g]] += abs(sys.q[g])
    end
    sys.q ./= view(sys.qs, sys.basis_col)
    sys.scale = sum(B[k] / n[k] for k in cu) / length(cu)
    fill!(nz, zero(FT))
    for p in eachindex(Sv)
        sys.pos_S[p] > 0 && (nz[sys.pos_S[p]] = Sv[p])
    end
    nz[sys.pos_I] .= one(FT)
    for g in eachindex(sys.q)
        k = cu[sys.basis_row[g]]
        nz[sys.pos_T[g]] = -B[k] / (n[k] * sys.scale) * knot_w_vals[g]
        shift === nothing || (nz[sys.pos_shift[g]] += sys.q[g] * shift[sys.basis_row[g]])
    end
    for i in eachindex(sys.gal_S)
        nz[sys.gal_pos[i]] += sys.q[sys.gal_basis[i]] * Sv[sys.gal_S[i]]
    end
    factorize!(sys.solver, sys.matrix; reuse = !sys.fresh)
    sys.fresh = false
    return sys
end

"""
    constrained_rhs(sys, r; extra = nothing) -> Vector

The right-hand side of `sys` for S x = r: r with the constrained rows' zero, then the Galerkin
rows' Q r + `extra`, normalized as their rows are.
"""
function constrained_rhs(sys::ConstrainedSystem{FT}, r::AbstractVector; extra::Union{Nothing, AbstractVector} = nothing) where {FT}
    L = sys.L
    y = zeros(promote_type(FT, eltype(r)), L + length(sys.qs))
    y[1:L] .= r
    y[sys.cu] .= zero(FT)
    for g in eachindex(sys.q)
        y[L + sys.basis_col[g]] += sys.q[g] * r[sys.cu[sys.basis_row[g]]]
    end
    extra === nothing || (y[(L + 1):end] .+= extra ./ sys.qs)
    return y
end

"""
    constrained_momentum_solve(A, r, basis, n, B; extra = nothing) -> (u, Γ_over_B)

Solve the per-electron momentum step A u = r on the grid's nodes (`A` sparse, vectors over all
nodes) with the surface-flux constraint on the nodes of Ω_c inside the LCFS: there
u = (B/n) W Γ_over_B, W = `knot_weights` and Γ_over_B the knot values of n u∥/B, and their momentum rows are
replaced by the Galerkin rows Q (A u − r) = `extra` (zero by default), Q = Wᵀ diag(V B). Nodes beyond the LCFS and outside Ω_c
keep their rows, so their current is not removed. The residual of the replaced rows then does
no work on the constrained flow. `basis === nothing` is the plain solve, with an empty Γ_over_B.
A run keeps the system between steps ([`constrained_system`](@ref)); this builds it afresh.
"""
function constrained_momentum_solve(
        A::SparseMatrixCSC{FT}, r::AbstractVector, b::Union{Nothing, ClosedSurfaceBasis}, n::AbstractVecOrMat, B::AbstractVecOrMat;
        extra::Union{Nothing, AbstractVector} = nothing,
    ) where {FT <: AbstractFloat}
    b === nothing && return (A \ r, FT[])
    sys = update_constrained_system!(constrained_system(nothing, A, b), A, b, n, B)
    y = constrained_rhs(sys, r; extra)
    x = solve!(similar(y), sys.solver, y)
    return x[1:sys.L], x[(sys.L + 1):end] ./ sys.scale
end
