# Flux-surface averages on the (R, Z) grid.
#
# A connected component of a ψ contour, revolved toroidally, is a flux surface made of many
# field lines: closed around an O-point, or open from wall to wall (not handled yet). Its
# average carries the volume measure dl/B_pol,
#
#     ⟨f⟩ = ∮ f dl/B_pol / ∮ dl/B_pol,        dV/dψ = 2π ∮ dl/B_pol   (ψ per radian),
#
# which on a closed surface is the flux-surface average. The plain arclength average along a
# field line is ⟨B f⟩/⟨B⟩.
# internal/docs/src/notes/plans/PLAN_flux-surface-average.md;
# internal/docs/src/notes/design/closed-surface-quasineutrality.md §9.4.

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
    find_o_point(G, ψ, region; itp = psi_interpolant(G, ψ), tol = 1e-10, maxit = 50) -> (; R, Z, ψ, converged) or nothing

The O-point of a closed region (`region` holds linear node indices): the extremum of the
bicubic interpolant of `ψ`, found by damped Newton on ∇ψ = 0 from the region's node farthest
in ψ from the region's edge. `converged` requires a definite Hessian (an extremum, not an
X-point) and a step below `tol` grid spacings. `nothing` when `region` is empty.
"""
function find_o_point(
        G::GridGeometry{FT}, ψ::AbstractMatrix{FT}, region::AbstractVector{<:Integer};
        itp = nothing, tol::Real = 1.0e-10, maxit::Int = 50,
    ) where {FT <: AbstractFloat}
    isempty(region) && return nothing
    itp === nothing && (itp = psi_interpolant(G, ψ))
    k0 = o_point_seed(G, ψ, region)
    x = (G.R2D[k0], G.Z2D[k0])
    h = min(G.dR, G.dZ)
    converged = false
    for _ in 1:maxit
        g = gradient(itp, x)
        H = hessian(itp, x)
        det = H[1, 1] * H[2, 2] - H[1, 2] * H[2, 1]
        det == 0 && break
        δ = ((H[2, 2] * g[1] - H[1, 2] * g[2]) / det, (H[1, 1] * g[2] - H[2, 1] * g[1]) / det)
        # Backtrack until |∇ψ| decreases: a full Newton step can overshoot on a coarse grid.
        g2 = g[1]^2 + g[2]^2
        α = one(FT)
        xn = (x[1] - α * δ[1], x[2] - α * δ[2])
        while α > FT(1.0e-3)
            gn = gradient(itp, xn)
            gn[1]^2 + gn[2]^2 < g2 && break
            α /= 2
            xn = (x[1] - α * δ[1], x[2] - α * δ[2])
        end
        step = hypot(xn[1] - x[1], xn[2] - x[2])
        x = xn
        if step < tol * h
            Hn = hessian(itp, x)
            converged = Hn[1, 1] * Hn[2, 2] - Hn[1, 2] * Hn[2, 1] > 0
            break
        end
    end
    inside = first(G.R1D) <= x[1] <= last(G.R1D) && first(G.Z1D) <= x[2] <= last(G.Z1D)
    return (; R = x[1], Z = x[2], ψ = itp(x), converged = converged && inside)
end

# The region's node farthest in ψ from the mean ψ of its edge nodes (those with a neighbour
# outside the region): the node closest to the O-point of a closed region.
function o_point_seed(G::GridGeometry, ψ::AbstractMatrix, region::AbstractVector{<:Integer})
    NR, NZ = G.NR, G.NZ
    in_region = falses(NR * NZ)
    in_region[region] .= true
    edge_sum, edge_count = zero(eltype(ψ)), 0
    for k in region
        i, j = mod1(k, NR), cld(k, NR)
        neighbours = ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1))
        on_edge = any(n -> !(1 <= n[1] <= NR && 1 <= n[2] <= NZ) || !in_region[(n[2] - 1) * NR + n[1]], neighbours)
        if on_edge
            edge_sum += ψ[k]
            edge_count += 1
        end
    end
    ψ_edge = edge_count > 0 ? edge_sum / edge_count : sum(ψ[region]) / length(region)
    return region[argmax(abs.(ψ[region] .- ψ_edge))]
end

"""
    surface_levels(o, ψ, region; nsurf = default) -> (; ψ_axis, ψ_edge, ψN, ψ)

ψ levels of `nsurf` closed surfaces between the O-point `o` and the region's edge, the region
node farthest in ψ from the axis. Levels sit at the centres of `nsurf` equal steps of the
normalized flux ψN = (ψ − ψ_axis)/(ψ_edge − ψ_axis), so none is the axis or the edge itself.
The default gives about one surface per grid spacing across the region.
"""
function surface_levels(
        o, ψ::AbstractMatrix{FT}, region::AbstractVector{<:Integer};
        nsurf::Int = default_surface_count(region),
    ) where {FT <: AbstractFloat}
    nsurf >= 1 || throw(ArgumentError("nsurf must be positive, got $nsurf"))
    ψ_axis = FT(o.ψ)
    ψ_edge = ψ[region[argmax(abs.(ψ[region] .- ψ_axis))]]
    ψN = (FT.(1:nsurf) .- FT(0.5)) ./ nsurf
    return (; ψ_axis, ψ_edge, ψN, ψ = ψ_axis .+ ψN .* (ψ_edge - ψ_axis))
end

# About one surface per grid spacing across a roughly round region.
default_surface_count(region) = max(2, round(Int, sqrt(length(region) / π)))

"""
    FluxSurfaceAveragePolicy

How the weights of a flux-surface average are built from the grid. Every policy yields the
same object, so consumers never learn which one built it.
"""
abstract type FluxSurfaceAveragePolicy end

"""
    MarchingSquaresAverage()

Each surface is the closed contour of its level, traced by marching squares on the grid ψ
(`IMASutils.contour_from_midplane!`, from the O-point outward along its midplane). The
average is the trapezoidal ∮ f dl/B_pol along the contour, with f bilinear between the
nodes and B_pol = |∇ψ|/R from the bicubic ψ. No scatter from where nodes sit, second order
in the grid. A level whose contour lies inside one cell, or does not close, is marked
invalid. **The default.**
"""
struct MarchingSquaresAverage <: FluxSurfaceAveragePolicy end

"""
    HatBinningAverage()

Each region node is shared between the two surfaces whose levels bracket its ψ (linear hats
in normalized flux), weighted by its cell volume. No geometry: the average of a surface is
a volume-weighted mean of the nodes near its level, so it is a convex average but carries
the scatter of where the nodes happen to sit. The baseline the contour policies are judged
against.
"""
struct HatBinningAverage <: FluxSurfaceAveragePolicy end

"""
    FluxSurfaceAverage{FT, P}

Averages over the closed flux surfaces of one region, and their way back to the grid. Build
with [`flux_surface_average`](@ref); apply with [`surface_average`](@ref) and
[`to_grid`](@ref).

Fields read by consumers: `policy`, `axis` (`(; R, Z, ψ, converged)`), `ψ` and `ψN` (the
level of each surface), `valid` (whether the surface was found), `dVdψ` (2π∮dl/B_pol, per
radian of ψ). The weights are internal.
"""
struct FluxSurfaceAverage{FT <: AbstractFloat, P <: FluxSurfaceAveragePolicy}
    policy::P
    axis::@NamedTuple{R::FT, Z::FT, ψ::FT, converged::Bool}
    ψ_edge::FT
    ψ::Vector{FT}
    ψN::Vector{FT}
    valid::Vector{Bool}
    dVdψ::Vector{FT}
    weights::SparseMatrixCSC{FT, Int}          # nodes × surfaces; the column of a valid surface sums to 1
    grid_weights::SparseMatrixCSC{FT, Int}     # nodes × surfaces, interpolation in ψN; invalid columns empty
    dims::Tuple{Int, Int}
end

"""
    flux_surface_average(G, ψ, region; policy = MarchingSquaresAverage(), nsurf) -> FluxSurfaceAverage or nothing
    flux_surface_average(RP; kwargs...)

Averages over `nsurf` closed flux surfaces of the region `region` (linear node indices),
between its O-point and its edge ([`surface_levels`](@ref)). The `RP` form uses the closed
nodes of the last field-line analysis. `nothing` when the region is empty; every surface
invalid when the region has no O-point (`axis.converged == false`).
"""
function flux_surface_average(
        G::GridGeometry{FT}, ψ::AbstractMatrix{FT}, region::AbstractVector{<:Integer};
        policy::FluxSurfaceAveragePolicy = MarchingSquaresAverage(),
        nsurf::Int = default_surface_count(region),
    ) where {FT <: AbstractFloat}
    isempty(region) && return nothing
    itp = psi_interpolant(G, ψ)
    o = find_o_point(G, ψ, region; itp)
    lv = surface_levels(o, ψ, region; nsurf)
    # Without an O-point there are no closed surfaces to average over: report, do not guess.
    weights, dVdψ, valid = o.converged ? surface_weights(policy, G, ψ, region, o, lv, itp) :
        (spzeros(FT, length(ψ), nsurf), fill(FT(NaN), nsurf), fill(false, nsurf))
    grid_weights = level_interpolation(ψ, region, lv, valid)
    axis = (; R = FT(o.R), Z = FT(o.Z), ψ = FT(o.ψ), converged = o.converged)
    return FluxSurfaceAverage(policy, axis, lv.ψ_edge, lv.ψ, lv.ψN, valid, dVdψ, weights, grid_weights, size(ψ))
end

flux_surface_average(RP::RAPID; kwargs...) =
    flux_surface_average(RP.G, RP.fields.ψ, RP.flf.closed_surface_nids; kwargs...)

"""
    surface_average(fsa, f) -> Vector
    surface_average(fsa, f, ω) -> Vector

The average of the grid field `f` (an `(NR, NZ)` array or its vector) on each surface, or the
weighted average ⟨ω f⟩/⟨ω⟩. `NaN` on surfaces that were not found.
"""
function surface_average(fsa::FluxSurfaceAverage, f::AbstractVecOrMat)
    avg = transpose(fsa.weights) * vec(f)
    avg[.!fsa.valid] .= NaN
    return avg
end

surface_average(fsa::FluxSurfaceAverage, f::AbstractVecOrMat, ω::AbstractVecOrMat) =
    surface_average(fsa, vec(ω) .* vec(f)) ./ surface_average(fsa, ω)

"""
    to_grid(fsa, profile) -> Matrix

A profile on the surfaces (one value per surface) back on the grid: linear in normalized
flux between the valid surfaces, and extrapolated linearly from the two nearest beyond the
first and the last (half a level at most), so a profile linear in ψN, as any smooth field is
near the axis, comes back exactly. Nodes outside the region are zero.
"""
to_grid(fsa::FluxSurfaceAverage, profile::AbstractVector) =
    reshape(fsa.grid_weights * ifelse.(fsa.valid, profile, zero(eltype(profile))), fsa.dims)

# ── weights of each policy ───────────────────────────────────────────────────────────

# Linear-interpolation weights in ψN from the levels `ψN_levels` to the region's nodes: rows
# are nodes, columns are levels. Beyond the first and the last level the weights are held
# constant (`extrapolate = false`, a partition into hats) or continue the end segments. Level
# `s` is written to column `columns[s]` of a matrix with `ncol` columns.
function hat_weights(
        ψ::AbstractMatrix{FT}, region, ψ_axis, ψ_edge, ψN_levels;
        extrapolate::Bool = false, columns = eachindex(ψN_levels), ncol::Int = length(ψN_levels),
    ) where {FT}
    n = length(ψN_levels)
    I, J, V = Int[], Int[], FT[]
    for k in region
        x = (ψ[k] - ψ_axis) / (ψ_edge - ψ_axis)
        if n == 1 || (!extrapolate && x <= ψN_levels[1])
            push!(I, k); push!(J, columns[1]); push!(V, one(FT))
        elseif !extrapolate && x >= ψN_levels[n]
            push!(I, k); push!(J, columns[n]); push!(V, one(FT))
        else
            s = clamp(searchsortedlast(ψN_levels, x), 1, n - 1)
            t = (x - ψN_levels[s]) / (ψN_levels[s + 1] - ψN_levels[s])
            push!(I, k, k); push!(J, columns[s], columns[s + 1]); push!(V, one(FT) - t, t)
        end
    end
    return sparse(I, J, V, length(ψ), ncol)
end

# The way back to the grid uses only the surfaces that were found.
function level_interpolation(ψ::AbstractMatrix{FT}, region, lv, valid) where {FT}
    ids = findall(valid)
    isempty(ids) && return spzeros(FT, length(ψ), length(lv.ψ))
    return hat_weights(ψ, region, lv.ψ_axis, lv.ψ_edge, lv.ψN[ids]; extrapolate = true, columns = ids, ncol = length(lv.ψ))
end

function surface_weights(::MarchingSquaresAverage, G::GridGeometry{FT}, ψ, region, o, lv, itp) where {FT}
    ψm = ψ isa Matrix ? ψ : Matrix(ψ)
    Rc_cache, Zc_cache = IMASutils.contour_cache(G.R1D, G.Z1D)
    nsurf, N = length(lv.ψ), G.NR * G.NZ
    colptr = Vector{Int}(undef, nsurf + 1)
    colptr[1] = 1
    rowval, nzval = Int[], FT[]
    dVdψ = fill(FT(NaN), nsurf)
    valid = fill(false, nsurf)
    hint = (Ref(1), Ref(1))                     # successive contour points share cells
    @with_pool pool begin
        acc = zeros!(pool, FT, N)               # one surface's weight on each node
        stamp = zeros!(pool, Int, N)            # the last surface that reached each node
        reached = acquire!(pool, Int, N)        # the nodes this surface reached
        for s in 1:nsurf
            Rc, Zc = closed_contour!(Rc_cache, Zc_cache, ψm, G, lv.ψ[s], o)
            if Rc !== nothing
                m = length(Rc) - 1              # the last point repeats the first
                nreached, total = 0, zero(FT)
                for j in 1:m
                    jm = j == 1 ? m : j - 1
                    dl = (hypot(Rc[j + 1] - Rc[j], Zc[j + 1] - Zc[j]) + hypot(Rc[j] - Rc[jm], Zc[j] - Zc[jm])) / 2
                    g = gradient(itp, (Rc[j], Zc[j]); hint)
                    w = dl * Rc[j] / hypot(g[1], g[2])  # dl/B_pol with B_pol = |∇ψ|/R
                    nreached = add_bilinear!(acc, stamp, reached, nreached, G, Rc[j], Zc[j], w, s)
                    total += w
                end
                nodes = sort!(view(reached, 1:nreached))
                for node in nodes
                    push!(rowval, node)
                    push!(nzval, acc[node] / total)
                    acc[node] = zero(FT)
                end
                dVdψ[s] = 2π * total
                valid[s] = true
            end
            colptr[s + 1] = length(rowval) + 1
        end
        nothing
    end
    return SparseMatrixCSC(N, nsurf, colptr, rowval, nzval), dVdψ, valid
end

# The closed contour of `level` around the O-point `o`, or `nothing` when marching squares
# finds none (a contour inside one cell, an open contour, or a saddle it cannot connect).
function closed_contour!(Rc_cache, Zc_cache, ψ::Matrix, G::GridGeometry, level, o)
    # marching squares starts from the axis cell, which must lie inside the grid
    (G.R1D[1] <= o.R < G.R1D[end] && G.Z1D[1] <= o.Z < G.Z1D[end]) || return nothing, nothing
    Rc, Zc = try
        IMASutils.contour_from_midplane!(Rc_cache, Zc_cache, ψ, G.R1D, G.Z1D, level, o.R, o.Z, o.ψ)
    catch err
        err isa ErrorException || rethrow()
        return nothing, nothing
    end
    length(Rc) >= 4 || return nothing, nothing
    closed = isapprox(first(Rc), last(Rc); atol = 1.0e-9 * G.dR) && isapprox(first(Zc), last(Zc); atol = 1.0e-9 * G.dZ)
    return closed ? (Rc, Zc) : (nothing, nothing)
end

# Add `w` times the bilinear weights of the point (R, Z) on its four cell corners into `acc`,
# listing in `reached` the nodes surface `s` reaches for the first time.
function add_bilinear!(acc, stamp, reached, nreached, G::GridGeometry, R, Z, w, s)
    # the grid is uniform: the cell follows from the spacing
    i = clamp(floor(Int, (R - G.R1D[1]) / G.dR) + 1, 1, G.NR - 1)
    j = clamp(floor(Int, (Z - G.Z1D[1]) / G.dZ) + 1, 1, G.NZ - 1)
    t = (R - G.R1D[i]) / G.dR
    u = (Z - G.Z1D[j]) / G.dZ
    k = (j - 1) * G.NR + i
    for (node, c) in ((k, (1 - t) * (1 - u)), (k + 1, t * (1 - u)), (k + G.NR, (1 - t) * u), (k + G.NR + 1, t * u))
        if stamp[node] != s
            stamp[node] = s
            nreached += 1
            reached[nreached] = node
        end
        acc[node] += w * c
    end
    return nreached
end

# ── reference ──────────────────────────────────────────────────────────────────────

function surface_weights(::HatBinningAverage, G::GridGeometry{FT}, ψ, region, o, lv, itp) where {FT}
    H = hat_weights(ψ, region, lv.ψ_axis, lv.ψ_edge, lv.ψN)       # nodes × surfaces
    mass = transpose(H) * vec(G.inVol2D)                           # ∫ h_s dV
    valid = mass .> 0
    scale = [m > 0 ? inv(m) : zero(FT) for m in mass]
    weights = sparse(Diagonal(vec(G.inVol2D)) * H * Diagonal(scale))
    # ∫ h_s dV ≈ V'(ψ_s) Δψ, Δψ the level spacing in ψ (hats at the ends also take the clamped tails)
    Δψ = abs(lv.ψ_edge - lv.ψ_axis) / length(lv.ψ)
    dVdψ = mass ./ Δψ
    return weights, dVdψ, collect(valid)
end
