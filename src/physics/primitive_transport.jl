# u∥ and Te transport: the in-wall operators.
#
# Both are primitive (per-particle) variables: they are carried by the electrons that move,
# so their advection is derived from the same face mass flux the continuity equation uses,
# and their diffusion (turbulent viscosity / conduction) is a reflective in-wall operator —
# a wall neither reads nor damps a per-particle quantity. Nothing outside the wall is read.
# internal/docs/src/notes/design/wall-flux-channels.md §2.5–2.6.

"""
    cache_electron_operators!(RP)

Rebuild the per-step cache of the electron in-wall operators from the current `ueR`, `ueZ`
and tensor: `transport.C_e` (face-flux divergence of the electron velocity), `D_op_e`
(reflective `∇·D∇`) and `div_ue` (`∇·u_e`). `update_transport_quantities!` calls this last;
a caller that overwrites the velocities by hand afterwards must call it again before
stepping, or the step reads the operators of the velocities it replaced.
"""
function cache_electron_operators!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    pla, tp, G = RP.plasma, RP.transport, RP.G
    tp.C_e = build_face_flux_divergence(G, pla.ueR, pla.ueZ; upwind = RP.flags.upwind)
    tp.D_op_e = build_wall_diffusion_matrix(G, tp.DRR, tp.DRZ, tp.DZZ; cross_terms = :drop)
    tp.div_ue = wall_divergence(G, pla.ueR, pla.ueZ)
    return RP
end

"""
    electron_primitive_operators(RP) -> (U_op, D_op, div_u)

- `U_op`: `(u·∇)f` from the cached face flux of `(ueR, ueZ)` (`transport.C_e`) and the
  CURRENT `ne` (`primitive_advection_operator`; rows with `ne ≤ 1 m⁻³` are empty). Assembled
  on every call because `ne` moves within the step; the matrix is what the implicit solves
  need. A right-hand side only wants [`apply_primitive_advection`](@ref) on `transport.C_e`.
- `D_op`: the cached reflective in-wall `∇·D∇` (`transport.D_op_e`).
- `div_u`: the cached `∇·u` on in-wall nodes (`transport.div_ue`).

The cache is refreshed once per step by `update_transport_quantities!`.
"""
function electron_primitive_operators(RP::RAPID{FT}) where {FT <: AbstractFloat}
    pla, tp = RP.plasma, RP.transport
    U_op = primitive_advection_operator(tp.C_e, vec(pla.ne); n_floor = FT(1.0))
    return (U_op = U_op, D_op = tp.D_op_e, div_u = tp.div_ue)
end

"Apply `A` (a sparse matrix) to a 2-D field and return a 2-D field."
apply_op(A::SparseMatrixCSC{FT, Int}, f::AbstractMatrix{FT}) where {FT} = reshape(A * vec(f), size(f))
