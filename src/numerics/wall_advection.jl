# (u·∇)f for a per-particle variable (u∥, Te), derived from the SAME mass flux the continuity
# equation uses:
#
#     u·∇f ≡ [ ∇·(n u f) − f ∇·(n u) ] / n
#
# so that u∥ and Te are advected by the particles that actually move, the interior
# discretisation matches continuity, and at an outflow wall face the two terms cancel exactly
# (nothing is imposed on f there; inflow faces contribute nothing because the face operator
# reads nothing from outside). Rows on nodes without plasma (n ≤ n_floor) are empty.
# internal/docs/src/notes/design/wall-flux-channels.md §2.5–2.6.

"""
    advection_operator(A_conv, n; n_floor) -> SparseMatrixCSC

`(u·∇f)_i = [(A_conv·diag(n)·f)_i − f_i·(A_conv·n)_i] / n_i` from the face-flux divergence `A_conv`
(`build_face_flux_divergence`) and the density vector `n`. Rows with `n_i ≤ n_floor` are zero.
Annihilates constants exactly; reduces to the nodal upwind `u·∇` for uniform `n` and `u`.
"""
function advection_operator(
        A_conv::SparseMatrixCSC{FT, Int}, n::AbstractVector{FT}; n_floor::FT,
    ) where {FT <: AbstractFloat}
    An = A_conv * n
    inv_n = [ni > n_floor ? one(FT) / ni : zero(FT) for ni in n]
    return spdiagm(inv_n) * (A_conv * spdiagm(n) - spdiagm(An))
end

"""
    apply_advection(A_conv, n, f; n_floor) -> Vector

`(u·∇f)_i = [(A_conv·(n∘f))_i − f_i·(A_conv·n)_i] / n_i` without assembling the operator: two matvecs
instead of two sparse products. Rows with `n_i ≤ n_floor` are zero, exactly as in
[`advection_operator`](@ref); the two agree to rounding.
"""
function apply_advection(
        A_conv::SparseMatrixCSC{FT, Int}, n::AbstractVector{FT}, f::AbstractVector{FT}; n_floor::FT,
    ) where {FT <: AbstractFloat}
    Anf = A_conv * (n .* f)
    An = A_conv * n
    return [n[i] > n_floor ? (Anf[i] - f[i] * An[i]) / n[i] : zero(FT) for i in eachindex(n)]
end

"""
    advection_operator!(A_adv, A_conv, n; n_floor, work = similar(n)) -> A_adv

`advection_operator` written into `A_adv`, an operator on the same wall pattern as `A_conv`:
off the diagonal `inv_n_i·(A_ij·n_j)`, on it `inv_n_i·((A_ii·n_i) − (A_conv·n)_i)` — the
assembled product's own expressions. Rows with `n_i ≤ n_floor` are zero; `work` receives
`A_conv·n`.
"""
function advection_operator!(
        A_adv::DiscretizedOperator{FT}, A_conv::DiscretizedOperator{FT}, n::AbstractVector{FT};
        n_floor::FT, work::AbstractVector{FT} = similar(n),
    ) where {FT <: AbstractFloat}
    check_wall_pattern(A_adv)
    check_wall_pattern(A_conv)
    C, U = A_conv.matrix, A_adv.matrix
    (C.colptr == U.colptr && C.rowval == U.rowval) ||
        throw(ArgumentError("advection_operator!: A_adv and A_conv do not share a pattern"))
    length(n) == size(C, 2) == length(work) ||
        throw(DimensionMismatch("advection_operator!: n and work must have one entry per node"))
    mul!(work, C, n)
    nzU, nzC, rows = nonzeros(U), nonzeros(C), rowvals(C)
    @inbounds for j in 1:size(C, 2)
        nj = n[j]
        for k in nzrange(C, j)
            i = rows[k]
            inv_ni = n[i] > n_floor ? one(FT) / n[i] : zero(FT)
            nzU[k] = inv_ni * (nzC[k] * nj)
        end
    end
    k2c = A_conv.k2csc
    @inbounds for i in eachindex(n)
        inv_ni = n[i] > n_floor ? one(FT) / n[i] : zero(FT)
        kd = slot_position(k2c, i, SLOT_C)
        nzU[kd] = inv_ni * ((nzC[kd] * n[i]) - work[i])
    end
    return A_adv
end

apply_advection(A_conv::DiscretizedOperator{FT}, n::AbstractVector{FT}, f::AbstractVector{FT}; n_floor::FT) where {FT <: AbstractFloat} =
    apply_advection(A_conv.matrix, n, f; n_floor)

"""
    wall_divergence(G, uR, uZ) -> Matrix
    wall_divergence!(div, G, uR, uZ) -> div

`∇·u = (1/R)∂(R u_R)/∂R + ∂u_Z/∂Z` on in-wall nodes, central where both neighbours are in-wall
and one-sided where one is not; zero on on/out-wall nodes. Never reads the band outside the wall.
The `!` form rewrites every node of `div`.
"""
function wall_divergence(
        G::GridGeometry{FT}, uR::AbstractMatrix{FT}, uZ::AbstractMatrix{FT},
    ) where {FT <: AbstractFloat}
    return wall_divergence!(zeros(FT, G.NR, G.NZ), G, uR, uZ)
end

function wall_divergence!(
        div::AbstractMatrix{FT}, G::GridGeometry{FT}, uR::AbstractMatrix{FT}, uZ::AbstractMatrix{FT},
    ) where {FT <: AbstractFloat}
    NR, NZ = G.NR, G.NZ
    size(div) == (NR, NZ) || throw(DimensionMismatch("wall_divergence!: div must be NR × NZ"))
    J = G.Jacob
    fill!(div, zero(FT))
    for j in 1:NZ, i in 1:NR
        is_in_wall(G, i, j) || continue
        ip = is_in_wall(G, i + 1, j) ? i + 1 : i
        im = is_in_wall(G, i - 1, j) ? i - 1 : i
        dR = (ip - im) * G.dR
        dRterm = dR > 0 ? (J[ip, j] * uR[ip, j] - J[im, j] * uR[im, j]) / (J[i, j] * dR) : zero(FT)
        jp = is_in_wall(G, i, j + 1) ? j + 1 : j
        jm = is_in_wall(G, i, j - 1) ? j - 1 : j
        dZ = (jp - jm) * G.dZ
        dZterm = dZ > 0 ? (uZ[i, jp] - uZ[i, jm]) / dZ : zero(FT)
        div[i, j] = dRterm + dZterm
    end
    return div
end

"""
    wall_gradient(G, f) -> (∂f/∂R, ∂f/∂Z)

`∇f` on in-wall nodes, central where both neighbours are in-wall and one-sided where one is
not; zero on on/out-wall nodes. Never reads the band outside the wall — a central difference
there sees the empty band as a wall-ward drop and turns any coefficient built from it (the
ion pinch velocity) inward at every wall-adjacent cell.
"""
function wall_gradient(G::GridGeometry{FT}, f::AbstractMatrix{FT}) where {FT <: AbstractFloat}
    NR, NZ = G.NR, G.NZ
    gR = zeros(FT, NR, NZ)
    gZ = zeros(FT, NR, NZ)
    for j in 1:NZ, i in 1:NR
        is_in_wall(G, i, j) || continue
        ip = is_in_wall(G, i + 1, j) ? i + 1 : i
        im = is_in_wall(G, i - 1, j) ? i - 1 : i
        dR = (ip - im) * G.dR
        gR[i, j] = dR > 0 ? (f[ip, j] - f[im, j]) / dR : zero(FT)
        jp = is_in_wall(G, i, j + 1) ? j + 1 : j
        jm = is_in_wall(G, i, j - 1) ? j - 1 : j
        dZ = (jp - jm) * G.dZ
        gZ[i, j] = dZ > 0 ? (f[i, jp] - f[i, jm]) / dZ : zero(FT)
    end
    return gR, gZ
end
