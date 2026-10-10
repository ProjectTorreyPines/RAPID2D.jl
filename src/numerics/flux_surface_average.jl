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
    find_o_point(G, ψ, region; itp = psi_interpolant(G, ψ), tol, maxit = 50) -> (; R, Z, ψ, converged) or nothing

The O-point of a closed region (`region` holds linear node indices): the extremum of the
bicubic interpolant of `ψ`, found by damped Newton on ∇ψ = 0 from the region's node farthest
in ψ from the region's edge. `converged` requires a full Newton correction below `tol` grid
spacings (by default 1e-10, or 64 eps of the grid's float type if larger), a definite Hessian (an extremum, not an X-point), and a cell with a corner in the
region: an extremum outside the region is not its O-point. A line search that cannot lower
|∇ψ| stops the search unconverged. `nothing` when `region` is empty.
"""
function find_o_point(
        G::GridGeometry{FT}, ψ::AbstractMatrix{FT}, region::AbstractVector{<:Integer};
        itp = nothing, tol::Real = max(1.0e-10, 64 * eps(FT)), maxit::Int = 50,
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
        if hypot(δ[1], δ[2]) < tol * h          # the full correction is already negligible
            x = (x[1] - δ[1], x[2] - δ[2])
            converged = det > 0
            break
        end
        # Backtrack until |∇ψ| decreases: a full Newton step can overshoot on a coarse grid.
        g2 = g[1]^2 + g[2]^2
        α, decreased = one(FT), false
        xn = (x[1] - δ[1], x[2] - δ[2])
        while α > FT(1.0e-3)
            gn = gradient(itp, xn)
            decreased = gn[1]^2 + gn[2]^2 < g2
            decreased && break
            α /= 2
            xn = (x[1] - α * δ[1], x[2] - α * δ[2])
        end
        decreased || break
        x = xn
    end
    return (; R = x[1], Z = x[2], ψ = itp(x), converged = converged && in_region_cell(G, region, x))
end

# Whether the cell containing `x` has a corner in `region`.
function in_region_cell(G::GridGeometry, region, x)
    (G.R1D[1] <= x[1] <= G.R1D[end] && G.Z1D[1] <= x[2] <= G.Z1D[end]) || return false
    i = clamp(floor(Int, (x[1] - G.R1D[1]) / G.dR) + 1, 1, G.NR - 1)
    j = clamp(floor(Int, (x[2] - G.Z1D[1]) / G.dZ) + 1, 1, G.NZ - 1)
    k = (j - 1) * G.NR + i
    return any(c -> c in region, (k, k + 1, k + G.NR, k + G.NR + 1))
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
    ContourAverage(tracer = CubicTracer())

Each surface is the contour of its level, followed around the O-point by `tracer` (a
[`FieldLineTracer`](@ref)) from where it crosses the outward midplane. The average is the
trapezoidal ∮ f dl/B_pol along the path, f bilinear between nodes and B_pol as the tracer gives it.
A level whose path does not close, leaves the wall, or does not wind once around the O-point is
marked invalid.

- `CubicTracer()`, **the default**: the bicubic level set, not limited by the cells, so a surface
  a few cells across, or one beside an X-point, keeps its length and dV/dψ.
- `MarchingSquaresTracer()`: the bilinear contour, second order in the grid and the fastest. On a
  coarse grid a small surface is a polygon of a handful of vertices: ratio averages stay
  accurate, but dV/dψ (and q) lose several per cent. A contour inside one cell is not found.
- `RK4Tracer()`: follows B rather than the level, for comparison only.
"""
struct ContourAverage{T <: FieldLineTracer} <: FluxSurfaceAveragePolicy
    tracer::T
end
ContourAverage() = ContourAverage(CubicTracer())

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
level of each surface), `valid` (whether the surface was found), `dVdψ` (2π∮dl/B_pol per
radian of ψ: the magnitude |dV/dψ|, positive whether ψ rises or falls away from the axis).
The weights are internal.
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
    flux_surface_average(G, ψ, region; policy = ContourAverage(), nsurf, inside) -> FluxSurfaceAverage or nothing
    flux_surface_average(RP; kwargs...)

Averages over `nsurf` closed flux surfaces of the region `region` (linear node indices),
between its O-point and its edge ([`surface_levels`](@ref)). `inside` is the wall as a set of
cells ([`FieldLineGeometry`](@ref)); `nothing` is the whole grid. The `RP` form uses the closed nodes
of the last field-line analysis and the run's wall. `nothing` when the region is empty; every
surface invalid when the region has no O-point (`axis.converged == false`).
"""
function flux_surface_average(
        G::GridGeometry{FT}, ψ::AbstractMatrix{FT}, region::AbstractVector{<:Integer};
        policy::FluxSurfaceAveragePolicy = ContourAverage(),
        nsurf::Int = default_surface_count(region),
        inside::Union{AbstractMatrix{Bool}, Nothing} = nothing,
    ) where {FT <: AbstractFloat}
    isempty(region) && return nothing
    g = FieldLineGeometry(G, ψ; inside, nodal_B = needs_nodal_B(policy))
    o = find_o_point(G, ψ, region; itp = g.itp)
    lv = surface_levels(o, ψ, region; nsurf)
    # Without an O-point there are no closed surfaces to average over: report, do not guess.
    weights, dVdψ, valid = o.converged ? surface_weights(policy, G, g, region, o, lv) :
        (spzeros(FT, length(ψ), nsurf), fill(FT(NaN), nsurf), fill(false, nsurf))
    grid_weights = level_interpolation(ψ, region, lv, valid)
    axis = (; R = FT(o.R), Z = FT(o.Z), ψ = FT(o.ψ), converged = o.converged)
    return FluxSurfaceAverage(policy, axis, lv.ψ_edge, lv.ψ, lv.ψN, valid, dVdψ, weights, grid_weights, size(ψ))
end

flux_surface_average(RP::RAPID; kwargs...) =
    flux_surface_average(RP.G, RP.fields.ψ, RP.flf.closed_surface_nids; inside = RP.G.cell_state .>= 0, kwargs...)

needs_nodal_B(::FluxSurfaceAveragePolicy) = false
needs_nodal_B(p::ContourAverage) = needs_nodal_B(p.tracer)

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
first and the last valid ones (half a level when the end surfaces are valid, farther when
they are not). A profile linear in ψN comes back exactly; a surface profile smooth in the
minor radius is a function of r² ∝ ψN, so it is linear in ψN near the axis. With a single
valid surface the profile comes back constant. Nodes outside the region are zero.
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

# Each surface traced around the O-point, its trapezoidal weights dl/B_pol spread bilinearly onto
# the nodes as the path goes; the columns of a nodes × surfaces matrix.
function surface_weights(p::ContourAverage, G::GridGeometry{FT}, g::FieldLineGeometry, region, o, lv) where {FT}
    N, nsurf = G.NR * G.NZ, length(lv.ψ)
    colptr = Vector{Int}(undef, nsurf + 1)
    colptr[1] = 1
    rowval, nzval = Int[], FT[]
    dVdψ = fill(FT(NaN), nsurf)
    valid = fill(false, nsurf)
    @safe_with_pool pool begin                  # a trace may throw; the pool must survive it
        acc = zeros!(pool, FT, N)               # one surface's weight on each node
        stamp = zeros!(pool, Int, N)            # the last surface that reached each node
        reached = acquire!(pool, Int, N)        # the nodes this surface reached
        sw = SurfaceWeights(acc, stamp, reached, g, o)
        for s in 1:nsurf
            start_surface!(sw, s)
            e = trace_around!(sw, p.tracer, g, lv.ψ[s], o)
            nodes = view(reached, 1:sw.nreached)
            if e.termination === :closed && abs(sw.winding) == 1
                close_surface!(sw)
                for node in sort!(view(reached, 1:sw.nreached))
                    push!(rowval, node)
                    push!(nzval, acc[node] / sw.total)
                    acc[node] = zero(FT)
                end
                dVdψ[s] = 2π * sw.total
                valid[s] = true
            else
                acc[nodes] .= zero(FT)
            end
            colptr[s + 1] = length(rowval) + 1
        end
        nothing
    end
    return SparseMatrixCSC(N, nsurf, colptr, rowval, nzval), dVdψ, valid
end

# A visitor that spreads the trapezoidal weight (dl/2 on each side) / B_pol of every vertex of a
# closed path bilinearly onto the nodes, and counts how many times the path winds around the
# O-point (signed crossings of the ray from it towards +R). The first vertex's weight waits for
# the closing segment.
mutable struct SurfaceWeights{FT, G}
    acc::Vector{FT}
    stamp::Vector{Int}
    reached::Vector{Int}
    g::G
    Ro::FT                                      # the O-point
    Zo::FT
    column::Int
    nreached::Int
    total::FT
    n::Int
    winding::Int
    R::FT                                       # the vertex waiting for its second half-segment
    Z::FT
    w::FT                                       # 1/B_pol there
    half::FT                                    # its first half-segment
    first_half::FT                              # the first vertex's second half-segment
    R1::FT
    Z1::FT
    w1::FT
end
function SurfaceWeights(acc::AbstractVector{FT}, stamp, reached, g, o) where {FT}
    nan = FT(NaN)
    return SurfaceWeights(acc, stamp, reached, g, FT(o.R), FT(o.Z), 0, 0, zero(FT), 0, 0, nan, nan, nan, zero(FT), zero(FT), nan, nan, nan)
end

function start_surface!(sw::SurfaceWeights{FT}, s) where {FT}
    sw.column, sw.nreached, sw.total, sw.n, sw.winding = s, 0, zero(FT), 0, 0
    return sw
end

function visit!(sw::SurfaceWeights, R, Z, Bpol, Bϕ)
    w = inv(Bpol)
    if sw.n == 0
        sw.R1, sw.Z1, sw.w1 = R, Z, w
    else
        d = hypot(R - sw.R, Z - sw.Z)
        if sw.n == 1
            sw.first_half = d / 2
        else
            spread!(sw, sw.R, sw.Z, (sw.half + d / 2) * sw.w)
        end
        za, zb = sw.Z - sw.Zo, Z - sw.Zo
        if (za < 0) != (zb < 0) && sw.R + (R - sw.R) * za / (za - zb) > sw.Ro
            sw.winding += zb < 0 ? -1 : 1
        end
        sw.half = d / 2
    end
    sw.R, sw.Z, sw.w = R, Z, w
    sw.n += 1
    return sw
end

# the path came back to its first vertex: give that vertex both its half-segments
close_surface!(sw::SurfaceWeights) = spread!(sw, sw.R1, sw.Z1, (sw.first_half + sw.half) * sw.w1)

# Add `w` times the bilinear weights of the point (R, Z) on its four cell corners into the
# visitor's `acc`, listing in `reached` the nodes its column reaches for the first time.
function spread!(v, R, Z, w)
    g = v.g
    NR, NZ = size(g.ψ)
    i = clamp(floor(Int, (R - g.R1D[1]) / g.dR) + 1, 1, NR - 1)
    j = clamp(floor(Int, (Z - g.Z1D[1]) / g.dZ) + 1, 1, NZ - 1)
    t = (R - g.R1D[i]) / g.dR
    u = (Z - g.Z1D[j]) / g.dZ
    k = (j - 1) * NR + i
    for (node, c) in ((k, (1 - t) * (1 - u)), (k + 1, t * (1 - u)), (k + NR, (1 - t) * u), (k + NR + 1, t * u))
        if v.stamp[node] != v.column
            v.stamp[node] = v.column
            v.nreached += 1
            v.reached[v.nreached] = node
        end
        v.acc[node] += w * c
    end
    v.total += w
    return v
end

function surface_weights(::HatBinningAverage, G::GridGeometry{FT}, g::FieldLineGeometry, region, o, lv) where {FT}
    ψ = g.ψ
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
