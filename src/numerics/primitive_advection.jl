# (u·∇)f for a primitive variable, derived from the SAME mass flux the continuity equation uses:
#
#     u·∇f ≡ [ ∇·(n u f) − f ∇·(n u) ] / n
#
# so that u∥ and Te are advected by the particles that actually move, the interior
# discretisation matches continuity, and at an outflow wall face the two terms cancel exactly
# (nothing is imposed on f there; inflow faces contribute nothing because the face operator
# reads nothing from outside). Rows on nodes without plasma (n ≤ n_floor) are empty.
# internal/docs/src/notes/design/wall-flux-channels.md §2.5–2.6.

"""
    primitive_advection_operator(C, n; n_floor) -> SparseMatrixCSC

`(u·∇f)_i = [(C·diag(n)·f)_i − f_i·(C·n)_i] / n_i` from the face-flux divergence `C`
(`build_face_flux_divergence`) and the density vector `n`. Rows with `n_i ≤ n_floor` are zero.
Annihilates constants exactly; reduces to the nodal upwind `u·∇` for uniform `n` and `u`.
"""
function primitive_advection_operator(
        C::SparseMatrixCSC{FT, Int}, n::AbstractVector{FT}; n_floor::FT,
    ) where {FT <: AbstractFloat}
    Cn = C * n
    inv_n = [ni > n_floor ? one(FT) / ni : zero(FT) for ni in n]
    return spdiagm(inv_n) * (C * spdiagm(n) - spdiagm(Cn))
end

"""
    wall_divergence(G, uR, uZ) -> Matrix

`∇·u = (1/R)∂(R u_R)/∂R + ∂u_Z/∂Z` on in-wall nodes, central where both neighbours are in-wall
and one-sided where one is not; zero on on/out-wall nodes. Never reads the damped band.
"""
function wall_divergence(
        G::GridGeometry{FT}, uR::AbstractMatrix{FT}, uZ::AbstractMatrix{FT},
    ) where {FT <: AbstractFloat}
    NR, NZ = G.NR, G.NZ
    J = G.Jacob
    div = zeros(FT, NR, NZ)
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
