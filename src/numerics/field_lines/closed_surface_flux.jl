# Surface functions on the closed region, and the electron parallel flux they carry. On closed
# flux surfaces the electron parallel particle flux per B is a surface function,
# n u∥ / B = K_e(ψ). Its poloidal part is K_e ∇ψ × ∇ϕ = ∇H × ∇ϕ with H′ = K_e, so the flux through
# a control-volume face is 2π (H(ψ_end) − H(ψ_start)) at the face's two corners (the dual vertices),
# and the divergence of every control volume telescopes to zero. The constraint domain is the
# support of H on the admissible nodes, which leaves every face between it and the rest at one H.

"""
    closed_surface_basis(G, ψ, admissible; inside = nothing, weights = nothing, nknots = default)
        -> ClosedSurfaceBasis or nothing

The constraint domain and its surface functions for the nodes marked `admissible` (an `(NR, NZ)`
mask: closed field lines inside the wall, not empty). Nodes on the grid's perimeter are never
admitted; nodes the admissible ones enclose are (the node on the O-point, where field-line
following meets a null).
- The cut is the smallest corner level of any face between an admissible node and the rest,
  and no higher than the outermost level the bicubic tracer closes around the O-point inside the
  wall cells `inside`. Ω_c is every admissible node with a corner inside the cut, so the faces
  between Ω_c and the rest have both corners at or beyond it and carry no flux.
- `nothing` when the region has no O-point, more than one, or no level that closes.
- `nknots` hats (default: one per grid spacing, plus one) are merged, nearest the axis first,
  until each covers two nodes and the Gram matrix of `weights` (nodal, default 1) has a condition
  number below 10⁸; a single remaining level is one constant function.
"""
function closed_surface_basis(
        G::GridGeometry{FT}, ψ::AbstractMatrix{FT}, admissible::AbstractMatrix{Bool};
        inside::Union{Nothing, AbstractMatrix{Bool}} = nothing, weights = nothing,
        nknots::Union{Nothing, Int} = nothing,
    ) where {FT <: AbstractFloat}
    NR, NZ = size(ψ)
    A = copy(admissible)
    A[[1, NR], :] .= false
    A[:, [1, NZ]] .= false
    fill_holes!(A)
    region = findall(vec(A))
    isempty(region) && return nothing
    # one interpolant for the O-point and for the tracer that verifies the cut
    g = FieldLineGeometry(G, ψ; inside)
    o = find_o_point(G, ψ, region; itp = g.itp)
    (o === nothing || !o.converged) && return nothing
    count_o_points(G, ψ, region, g.itp) > 1 && return nothing
    ψ_axis = FT(o.ψ)
    s = sign(sum(ψ[region]) - length(region) * ψ_axis)
    s == 0 && return nothing
    # flux from the axis, growing outward, at the dual corners
    dc = [s * ((ψ[i, j] + ψ[i + 1, j] + ψ[i, j + 1] + ψ[i + 1, j + 1]) / 4 - ψ_axis) for i in 1:(NR - 1), j in 1:(NZ - 1)]
    L = first_boundary_corner(A, dc)
    L = closed_cut(g, o, s, L)
    L === nothing && return nothing
    lin = LinearIndices((NR, NZ))
    nodes = [lin[i, j] for j in 2:(NZ - 1), i in 2:(NR - 1) if A[i, j] && corner_min(dc, i, j) < L]
    sort!(nodes)
    isempty(nodes) && return nothing
    active = falses(NR - 1, NZ - 1)
    for k in nodes
        i, j = mod1(k, NR), cld(k, NR)
        for (a, c) in ((i - 1, j - 1), (i, j - 1), (i - 1, j), (i, j))
            dc[a, c] < L && (active[a, c] = true)
        end
    end
    ψN = [s * (ψ[k] - ψ_axis) / L for k in nodes]
    V = [FT(2π) * G.R2D[k] * G.dR * G.dZ for k in nodes]
    w = weights === nothing ? ones(FT, length(nodes)) : FT.(vec(weights)[nodes])
    n0 = nknots === nothing ? default_surface_count(nodes) + 1 : nknots
    knots = merge_knots(collect(range(zero(FT), one(FT); length = max(n0, 2))), ψN, w)
    axis = (; R = FT(o.R), Z = FT(o.Z), ψ = ψ_axis)
    return ClosedSurfaceBasis{FT}(
        nodes, V, axis, ψ_axis + s * L, ψN, dc ./ L, active, knots,
        hat_matrix(knots, ψN; beyond = :zero), hat_matrix(knots, ψN; beyond = :constant), (NR, NZ),
    )
end

# Admit the non-admissible nodes the admissible ones enclose: those not reached from the grid's
# perimeter through non-admissible nodes. Field-line following ends at the null of a node on the
# O-point and leaves it out; the cut must come from the region's outer edge.
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
    L = FT(Inf)
    for j in 2:(NZ - 1), i in 1:(NR - 1)          # R faces (i, j)–(i + 1, j): corners (i, j − 1), (i, j)
        A[i, j] == A[i + 1, j] && continue
        L = min(L, dc[i, j - 1], dc[i, j])
    end
    for j in 1:(NZ - 1), i in 2:(NR - 1)          # Z faces (i, j)–(i, j + 1): corners (i − 1, j), (i, j)
        A[i, j] == A[i, j + 1] && continue
        L = min(L, dc[i - 1, j], dc[i, j])
    end
    return L
end

# The cut L (flux from the axis) at or below `L0` whose contour the bicubic tracer closes around `o`
# inside the wall: L0 itself, or the outermost closed level between the axis and L0, found from a
# level confirmed closed by bisection. A trace that hits the wall, runs out of budget, meets a null
# or is not seeded counts as not closed. `nothing` when no level closes.
function closed_cut(g::FieldLineGeometry{FT}, o, s, L0) where {FT}
    isfinite(L0) && L0 > 0 || return nothing
    closes(L) = trace_around!(PathLengths{FT}(), CubicTracer(), g, FT(o.ψ) + s * L, o).termination === :closed
    closes(L0) && return L0
    lo = L0
    for _ in 1:8
        lo /= 2
        closes(lo) && break
    end
    closes(lo) || return nothing
    hi = L0
    for _ in 1:40
        mid = (lo + hi) / 2
        closes(mid) ? (lo = mid) : (hi = mid)
    end
    return lo
end

# Hat values of `knots` at x; beyond the last knot zero (`:zero`) or the last hat's 1 (`:constant`).
# A single knot is one constant function on [0, 1].
function hat_matrix(knots::Vector{FT}, x::Vector{FT}; beyond::Symbol) where {FT}
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

# Merge hats, the interior knot nearest the axis first, until each hat covers two nodes and the
# weighted Gram matrix is well conditioned; one knot left is one constant.
function merge_knots(knots::Vector{FT}, x::Vector{FT}, w::Vector{FT}) where {FT}
    while length(knots) > 2
        P = hat_matrix(knots, x; beyond = :zero)
        covered = vec(sum(P .> 0; dims = 1))
        gram = Matrix(P' * Diagonal(w) * P)
        all(>=(2), covered) && cond(gram) < 1.0e8 && return knots
        bad = findfirst(<(2), covered)
        s = bad === nothing ? argmin(diag(gram)) : bad
        deleteat!(knots, clamp(s, 2, length(knots) - 1))
    end
    P = hat_matrix(knots, x; beyond = :zero)
    all(>=(2), vec(sum(P .> 0; dims = 1))) && cond(Matrix(P' * Diagonal(w) * P)) < 1.0e8 && return knots
    return [zero(FT)]
end

# ∫₀^min(x, 1) of the piecewise-linear function with values Kk at the knots (one knot: a constant)
function hat_integral(knots::Vector{FT}, Kk::AbstractVector, x::FT) where {FT}
    x = clamp(x, zero(FT), one(FT))
    length(knots) == 1 && return Kk[1] * x
    acc = zero(promote_type(FT, eltype(Kk)))
    for s in 1:(length(knots) - 1)
        a, b = knots[s], knots[s + 1]
        x <= a && break
        e = min(x, b)
        Ke = Kk[s] + (Kk[s + 1] - Kk[s]) * (e - a) / (b - a)
        acc += (Kk[s] + Ke) / 2 * (e - a)
    end
    return acc
end

"""
    stream_function_fluxes(basis, K) -> (ΓR, ΓZ)

The poloidal particle flux of n u∥ b = K_e B through every control-volume face of the grid, for
the surface function with values `K` at the knots: `ΓR[i, j]` through the face between nodes
(i, j) and (i + 1, j) along +R, `ΓZ[i, j]` between (i, j) and (i, j + 1) along +Z, each
2π (H(ψ_end) − H(ψ_start)) at the face's corners (ψ per radian, B_R = −∂_Zψ/R, B_Z = ∂_Rψ/R).
H varies only at the corners of Ω_c inside the cut (`corner_active`), so every face outside Ω_c
and between Ω_c and the rest carries none.
"""
function stream_function_fluxes(b::ClosedSurfaceBasis{FT}, K::AbstractVector) where {FT}
    NR, NZ = b.dims
    J = b.ψ_cut - b.axis.ψ
    H_cut = J * hat_integral(b.knots, K, one(FT))
    H = [b.corner_active[c] ? J * hat_integral(b.knots, K, b.corner_ψN[c]) : H_cut for c in CartesianIndices(b.corner_ψN)]
    T = promote_type(FT, eltype(K))
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
of each hat, Σ_f Γ_f^(s) Δg_f τ̄_f, applied to the part of τ that is not a surface function
(τ minus its volume-weighted projection on `basis.Pτ`): the surface part gives exactly zero, as
∫ h_s τ̄ B·∇g dV does, so a constant or surface temperature exerts no force. The Galerkin row
takes −B_p τ / m_e; the matching compression power of node i is τ_i (B_pᵀ K)_i.
"""
function pressure_operator(b::ClosedSurfaceBasis{FT}, g::AbstractMatrix) where {FT}
    NR, NZ = b.dims
    M, N = length(b.knots), length(b.nodes)
    col = zeros(Int, NR * NZ)
    col[b.nodes] .= 1:N
    X = zeros(FT, M, N)
    for s in 1:M
        ΓR, ΓZ = stream_function_fluxes(b, [t == s ? one(FT) : zero(FT) for t in 1:M])
        for j in 1:NZ, i in 1:(NR - 1)
            Γ = ΓR[i, j]
            iszero(Γ) && continue
            a, h = col[i + (j - 1) * NR], col[i + 1 + (j - 1) * NR]
            (a > 0 && h > 0) || error("a face carrying flux leaves Ω_c")
            c = Γ * (g[i + 1, j] - g[i, j]) / 2
            X[s, a] += c
            X[s, h] += c
        end
        for j in 1:(NZ - 1), i in 1:NR
            Γ = ΓZ[i, j]
            iszero(Γ) && continue
            a, h = col[i + (j - 1) * NR], col[i + j * NR]
            (a > 0 && h > 0) || error("a face carrying flux leaves Ω_c")
            c = Γ * (g[i, j + 1] - g[i, j]) / 2
            X[s, a] += c
            X[s, h] += c
        end
    end
    # remove the surface part of τ: X (I − S), S = Pτ (Pτᵀ W Pτ)⁻¹ Pτᵀ W
    Pτ = Matrix(b.Pτ)
    WPτ = b.V .* Pτ
    return X .- ((X * Pτ) / (Pτ' * WPτ)) * WPτ'
end

# ── diagnostics on Ω_c ───────────────────────────────────────────────────────────

"""
    galerkin_flux(basis, n, u, B) -> K

The surface flux closest to the electron velocity `u` in the kinetic-energy metric: the Galerkin
projection (Pᵀ diag(V B²/n) P) K = Pᵀ diag(V B) u over Ω_c. A velocity that is a surface flux,
u = (B/n) P K, gives back its K. `n`, `u`, `B` are grid arrays.
"""
function galerkin_flux(b::ClosedSurfaceBasis, n::AbstractVecOrMat, u::AbstractVecOrMat, B::AbstractVecOrMat)
    Bk, nk = B[b.nodes], n[b.nodes]
    return Matrix(b.P' * Diagonal(b.V .* Bk .^ 2 ./ nk) * b.P) \ (b.P' * (b.V .* Bk .* u[b.nodes]))
end

"""
    surface_residual(basis, n, u, B; K = galerkin_flux(basis, n, u, B)) -> Real

How far n u/B on Ω_c is from the surface function P K, as the volume-weighted rms of the
difference over that of n u/B: 0 for a surface flux. Beyond the cut, where K_e ends, all of
n u/B counts.
"""
function surface_residual(
        b::ClosedSurfaceBasis, n::AbstractVecOrMat, u::AbstractVecOrMat, B::AbstractVecOrMat;
        K = galerkin_flux(b, n, u, B),
    )
    f = n[b.nodes] .* u[b.nodes] ./ B[b.nodes]
    total = sum(b.V .* f .^ 2)
    iszero(total) && return zero(total)
    return sqrt(sum(b.V .* (f .- b.P * K) .^ 2) / total)
end

"""
    electron_kinetic_energy(G, n, u, nodes; me) -> Real

½ mₑ n u² summed over the toroidal control volumes 2πR ΔR ΔZ of `nodes` [J].
"""
electron_kinetic_energy(G::GridGeometry, n::AbstractVecOrMat, u::AbstractVecOrMat, nodes; me::Real) =
    sum(me / 2 * n[k] * u[k]^2 * 2π * G.R2D[k] * G.dR * G.dZ for k in nodes; init = zero(me * G.dR))

"""
    surface_statistics(basis, ne, ni, ue, ui, Te_eV; me, ee) -> (; n_min, drift, imbalance)

Per hat of `basis`, over the Ω_c nodes it covers: the smallest electron density, the largest
electron–ion drift against the electron thermal speed, |uₑ − uᵢ| / √(e Tₑ / mₑ), and the largest
relative density imbalance |nₑ − nᵢ| / nₑ.
"""
function surface_statistics(
        b::ClosedSurfaceBasis{FT}, ne, ni, ue, ui, Te_eV; me::Real, ee::Real,
    ) where {FT}
    M = length(b.knots)
    n_min, drift, imbalance = fill(FT(Inf), M), zeros(FT, M), zeros(FT, M)
    rows, vals = rowvals(b.P), nonzeros(b.P)
    for s in 1:M, p in nzrange(b.P, s)
        iszero(vals[p]) && continue
        k = b.nodes[rows[p]]
        n_min[s] = min(n_min[s], ne[k])
        drift[s] = max(drift[s], abs(ue[k] - ui[k]) / sqrt(ee * Te_eV[k] / me))
        imbalance[s] = max(imbalance[s], abs(ne[k] - ni[k]) / ne[k])
    end
    return (; n_min, drift, imbalance)
end

"""
    current_outside_constraint(basis, closed, J) -> Real

The share of the toroidal current `J` (grid array) on the closed nodes `closed` that the
constraint leaves out: nodes outside Ω_c, and Ω_c nodes beyond the cut, where K_e = 0.
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

"""
    constrained_momentum_solve(A, r, basis, n, B; extra = nothing) -> (u, K)

Solve the per-electron momentum step A u = r on the grid's nodes (`A` sparse, vectors over all
nodes) with the surface-flux constraint on the nodes of Ω_c inside the cut: there
u = (B/n) P K, and their momentum rows are replaced by the Galerkin rows
Q (A u − r) = `extra` (zero by default), Q = Pᵀ diag(V B). Nodes beyond the cut and outside Ω_c
keep their rows, so their current is not removed. The residual of the replaced rows then does
no work on the constrained flow. `basis === nothing` is the plain solve, with an empty K.
"""
function constrained_momentum_solve(
        A::SparseMatrixCSC{FT}, r::AbstractVector, b::Union{Nothing, ClosedSurfaceBasis}, n::AbstractVector, B::AbstractVector;
        extra::Union{Nothing, AbstractVector} = nothing,
    ) where {FT <: AbstractFloat}
    b === nothing && return (A \ r, FT[])
    N = size(A, 1)
    sel = findall(<=(1), b.ψN)
    cu = b.nodes[sel]
    Pin = b.P[sel, :]
    M = size(Pin, 2)
    nk, Bk = n[cu], B[cu]
    κ = mean(Bk ./ nk)                                   # K is solved as κ K, of the order of u
    Q = Pin' * Diagonal(b.V[sel] .* Bk)
    qs = vec(sum(abs, Q; dims = 2))                      # Galerkin rows normalized
    Qs = Diagonal(one(FT) ./ qs) * Q
    keep = ones(FT, N)
    keep[cu] .= zero(FT)
    Arows = Diagonal(keep) * A + Diagonal(one(FT) .- keep)   # constrained rows: u_k − (T K)_k = 0
    T = sparse(Diagonal((Bk ./ nk) ./ κ) * Pin)
    rows, cols, vals = findnz(T)
    Tcols = sparse(cu[rows], cols, -vals, N, M)
    Aaug = [Arows Tcols; sparse(Qs * A[cu, :]) spzeros(FT, M, M)]
    rhs = vcat(keep .* r, Qs * r[cu] .+ (extra === nothing ? zero(FT) : extra ./ qs))
    x = Aaug \ rhs
    return x[1:N], x[(N + 1):end] ./ κ
end
