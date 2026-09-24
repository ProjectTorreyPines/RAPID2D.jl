# u∥ and Te transport: the in-wall operators.
#
# Both are primitive (per-particle) variables: they are carried by the electrons that move,
# so their advection is derived from the same face mass flux the continuity equation uses,
# and their diffusion (turbulent viscosity / conduction) is a reflective in-wall operator —
# a wall neither reads nor damps a per-particle quantity. Nothing outside the wall is read.
# internal/docs/src/notes/design/wall-flux-channels.md §2.5–2.6.

"""
    electron_primitive_operators(RP) -> (U_op, D_op, div_u)

- `U_op`: `(u·∇)f` from the face flux of `(ueR, ueZ)` and the current `ne`
  (`primitive_advection_operator`; rows with `ne ≤ 1 m⁻³` are empty).
- `D_op`: reflective in-wall `∇·D∇` from `(DRR, DRZ, DZZ)` (`build_wall_diffusion_matrix`
  without faces: zero flux through every wall face).
- `div_u`: `∇·u` on in-wall nodes, one-sided at the wall (`wall_divergence`).

Rebuilt on every call; `ue_para`, `Te` and the heating powers each build their own copy
within a step (cost measured later, see PLAN_wall-flux-channels.md PR2b).
"""
function electron_primitive_operators(RP::RAPID{FT}) where {FT <: AbstractFloat}
    pla, tp, G = RP.plasma, RP.transport, RP.G
    C = build_face_flux_divergence(G, pla.ueR, pla.ueZ; upwind = RP.flags.upwind)
    U_op = primitive_advection_operator(C, vec(pla.ne); n_floor = FT(1.0))
    D_op = build_wall_diffusion_matrix(G, tp.DRR, tp.DRZ, tp.DZZ; cross_terms = :drop)
    div_u = wall_divergence(G, pla.ueR, pla.ueZ)
    return (U_op = U_op, D_op = D_op, div_u = div_u)
end

"Apply `A` (a sparse matrix) to a 2-D field and return a 2-D field."
apply_op(A::SparseMatrixCSC{FT, Int}, f::AbstractMatrix{FT}) where {FT} = reshape(A * vec(f), size(f))
