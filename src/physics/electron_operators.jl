# The electron in-wall operators: cached once per step, and what u∥ and Te solve with.
#
# u∥ and Te are per-particle variables: they are carried by the electrons that move,
# so their advection is derived from the same face mass flux the continuity equation uses,
# and their diffusion (turbulent viscosity / conduction) is a reflective in-wall operator —
# a wall neither reads nor damps a per-particle quantity. Nothing outside the wall is read.
# internal/docs/src/notes/design/wall-flux-channels.md §2.5–2.6.

"""
    cache_electron_operators!(RP)

Refresh the electron in-wall operators on `RP.operators` from the current `ueR`, `ueZ`,
tensor and `flags.upwind`, values only (they were allocated on the wall pattern by
`initialize!`): `A_conv_e` (face-flux divergence of the electron velocity, the scheme it was
built with recorded in `A_conv_e_upwind`), `A_diffu_e` (reflective `∇·D∇`, its anti-diffusive
pairs limited at the current `ne` by `flags.limit_antidiffusive_pairs`) and `div_ue`
(`∇·u_e`). `update_transport_quantities!` calls this last — at the end of every step and at
every `run_simulation!` entry; a caller that overwrites the velocities or the flag by hand and
then steps by hand must call it again, or the step reads the operators of the state it
replaced ([`electron_operator_cache`](@ref) refuses a changed flag).
"""
function cache_electron_operators!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    pla, op, tp, G = RP.plasma, RP.operators, RP.transport, RP.G
    check_electron_operators(op)
    if !closed_surface_active(RP)
        build_face_flux_divergence!(op.A_conv_e, G, pla.ueR, pla.ueZ; upwind = RP.flags.upwind)
        wall_divergence!(op.div_ue, G, pla.ueR, pla.ueZ)
    else
        # SurfaceFluxConstraint: on every face of the constraint domain the density moves with the
        # perpendicular drift only; the parallel flux there is the stream-function flux, which
        # leaves every control volume as it found it
        F = RP.fields
        mask = falses(G.NR, G.NZ)
        mask[RP.closed_surface.basis.nodes] .= true
        uR⊥, uZ⊥ = pla.ueR .- pla.ue_para .* F.bR, pla.ueZ .- pla.ue_para .* F.bZ
        build_face_flux_divergence!(op.A_conv_e, G, pla.ueR, pla.ueZ; upwind = RP.flags.upwind, faces_of = (; mask, uR = uR⊥, uZ = uZ⊥))
        wall_divergence!(op.div_ue, G, pla.ueR, pla.ueZ)
        div⊥ = similar(op.div_ue)
        wall_divergence!(div⊥, G, uR⊥, uZ⊥)
        op.div_ue[mask] .= div⊥[mask]
    end
    op.A_conv_e_upwind = RP.flags.upwind
    build_wall_diffusion_matrix!(op.A_diffu_e, G, tp.DRR, tp.DRZ, tp.DZZ; cross_terms = :drop)
    # lagged: the density at the start of the step sets the limiter for the density solve and
    # for the u∥ and Te operators built from it
    lim = RP.flags.limit_antidiffusive_pairs
    lim.state && limit_antidiffusive_pairs!(op.A_diffu_e, G, pla.ne; r0 = lim.factor)
    return RP
end

# The electron operators must have been allocated on the wall pattern by `initialize!`.
function check_electron_operators(op::Operators)
    all(is_on_wall_pattern, (op.A_conv_e, op.A_diffu_e, op.A_adv_e, op.A_visc_drift_e, op.A_LHS)) || throw(
        ArgumentError(
            "RP.operators was not built by initialize!: the electron in-wall operators " *
                "are not allocated on the wall pattern"
        )
    )
    return op
end

"""
    electron_operator_cache(RP) -> Operators

`RP.operators`, checked to hold electron operators a consumer may use: allocated by
`initialize!` on the wall pattern, with the wall faces on `RP.transport`, and refreshed with
the current `flags.upwind`. A flag changed since the refresh is refused rather than silently
applied to the old operators; the refresh (`update_transport_quantities!`, or
`cache_electron_operators!` alone) clears it.
"""
function electron_operator_cache(RP::RAPID{FT}) where {FT <: AbstractFloat}
    op = RP.operators
    isempty(RP.transport.wall_faces) && throw(
        ArgumentError(
            "transport.wall_faces is empty: this Transport was not built by initialize!, " *
                "so no wall-aware operator can be assembled"
        )
    )
    check_electron_operators(op)
    op.A_conv_e_upwind == RP.flags.upwind || throw(
        ArgumentError(
            "operators.A_conv_e was cached with upwind = $(op.A_conv_e_upwind) but flags.upwind is now " *
                "$(RP.flags.upwind): run update_transport_quantities! (or cache_electron_operators!) " *
                "after changing the scheme"
        )
    )
    return op
end

"""
    ue_Te_operators(RP) -> (A_adv, A_diffu, div_u)

The operators the u∥ and Te equations solve with, from the per-step cache on `RP.operators`:

- `A_adv`: `(u·∇)f` from the cached face flux `A_conv_e` and the CURRENT `ne`, written in
  place into `A_adv_e` (`advection_operator!`; rows with `ne ≤ 1 m⁻³` are empty) on every
  call, because `ne` moves within the step. A right-hand side only wants
  [`apply_advection`](@ref) on `A_conv_e`.
- `A_diffu`: what u∥ and Te diffuse with: the particle-weighted operator of the reflective
  in-wall `∇·D∇` (`A_diffu_e`) and the CURRENT `ne`, written into `A_visc_drift_e` on every
  call (`per_particle_operator!`). It is the viscosity (conduction for Te) plus the advection
  by the diffusive drift, in one matrix; the solves only ever need their sum.
- `div_u`: `∇·u` on in-wall nodes (`div_ue`).

All three are buffers, overwritten by the next refresh or call.
"""
function ue_Te_operators(RP::RAPID{FT}) where {FT <: AbstractFloat}
    pla, op = RP.plasma, electron_operator_cache(RP)
    advection_operator!(op.A_adv_e, op.A_conv_e, pla.ne; n_floor = FT(1.0))
    div_u = op.div_ue
    if closed_surface_active(RP) && !isempty(RP.closed_surface.solve.Γe_para_over_B)
        # the parallel carriage on the constraint domain is the stream-function flux, and its
        # compression the pressure operator's adjoint
        add_stream_function_transport!(op.A_adv_e, RP)
        compression, V = RP.closed_surface.solve.compression, RP.G.inVol2D
        div_u = copy(op.div_ue)
        for k in eachindex(compression)
            c = compression[k]
            (iszero(c) || !(pla.ne[k] > CLOSED_SURFACE_N_FLOOR)) && continue
            div_u[k] -= c / (pla.ne[k] * V[k])
        end
    end
    # u∥ and Te ride the particles the density diffusion moves. One matrix holds the viscosity
    # (conduction for Te) and the drift advection: M_kl = A_kl n_l/n_k splits exactly into
    # A_kl (n_l + n_k)/(2 n_k), the first with the face-mean density, and A_kl (n_l − n_k)/(2 n_k),
    # the second (electron-diffusive-transport.md §4).
    per_particle_operator!(op.A_visc_drift_e, op.A_diffu_e, pla.ne; n_floor = FT(1.0))
    return (A_adv = op.A_adv_e, A_diffu = op.A_visc_drift_e, div_u)
end


"""
    viscous_heating!(P, RP; M = ue_Te_operators(RP).A_diffu) -> P

The viscous heating of u∥, W per electron, into `P`: `mₑ Γ_M(u∥)` with `Γ_M` the
dissipation rate ([`dissipation_rate!`](@ref)) of the operator Te diffuses with
(`ue_Te_operators(RP).A_diffu`) and the current `ue_para`. In the continuum it is
`−Π:∇u / nₑ = mₑ ∇u∥·D∇u∥` with the stress `Π = −mₑ nₑ D ∇u∥`: Braginskii's viscous heating
with the anomalous viscosity `mₑ nₑ D`. `M` also carries u∥ with the diffusive drift, which
in the continuum moves `½u²` without loss; on the grid its share of `Γ_M` is a small
discretization error, and the energy identity holds for the full `M`, which is what is used
(electron-diffusive-transport.md §4). With it credited to Te, the energy per electron
`3/2 k_B Te + ½ mₑ u∥²` is mixed by the same operator as u∥ and Te, so the energy the
particles carry is conserved (§5). Pass `M` when the caller has just built it.
"""
function viscous_heating!(
        P::AbstractMatrix{FT}, RP::RAPID{FT}; M::DiscretizedOperator{FT} = ue_Te_operators(RP).A_diffu,
    ) where {FT <: AbstractFloat}
    dissipation_rate!(P, M, RP.plasma.ue_para)
    P .*= RP.config.constants.me
    return P
end
