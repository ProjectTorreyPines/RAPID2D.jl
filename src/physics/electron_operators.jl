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
built with recorded in `A_conv_e_upwind`), `A_diffu_e` (reflective `∇·D∇`) and `div_ue`
(`∇·u_e`). `update_transport_quantities!` calls this last — at the end of every step and at
every `run_simulation!` entry; a caller that overwrites the velocities or the flag by hand and
then steps by hand must call it again, or the step reads the operators of the state it
replaced ([`electron_operator_cache`](@ref) refuses a changed flag).
"""
function cache_electron_operators!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    pla, op, tp, G = RP.plasma, RP.operators, RP.transport, RP.G
    check_electron_operators(op)
    build_face_flux_divergence!(op.A_conv_e, G, pla.ueR, pla.ueZ; upwind = RP.flags.upwind)
    op.A_conv_e_upwind = RP.flags.upwind
    build_wall_diffusion_matrix!(op.A_diffu_e, G, tp.DRR, tp.DRZ, tp.DZZ; cross_terms = :drop)
    wall_divergence!(op.div_ue, G, pla.ueR, pla.ueZ)
    return RP
end

# The electron operators must have been allocated on the wall pattern by `initialize!`.
function check_electron_operators(op::Operators)
    all(is_on_wall_pattern, (op.A_conv_e, op.A_diffu_e, op.A_adv_e, op.A_mix_e, op.A_LHS)) || throw(
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
- `A_diffu`: what u∥ and Te diffuse with, by `flags.mixing_policy` ([`MixingPolicy`](@ref)):
  the particle-weighted operator of the reflective in-wall `∇·D∇` (`A_diffu_e`) and the
  CURRENT `ne`, written into `A_mix_e` on every call (`per_particle_operator!`), or the
  density operator itself.
- `div_u`: `∇·u` on in-wall nodes (`div_ue`).

All three are buffers, overwritten by the next refresh or call.
"""
function ue_Te_operators(RP::RAPID{FT}) where {FT <: AbstractFloat}
    pla, op = RP.plasma, electron_operator_cache(RP)
    n = vec(pla.ne)
    advection_operator!(op.A_adv_e, op.A_conv_e, n; n_floor = FT(1.0), work = op.ne_work)
    A_diffu = mixing_operator!(op, n, RP.flags.mixing_policy)
    return (A_adv = op.A_adv_e, A_diffu = A_diffu, div_u = op.div_ue)
end

# The default: u∥ and Te ride the particles the density diffusion moves.
function mixing_operator!(op::Operators{FT}, n::AbstractVector{FT}, ::ParticleMixing) where {FT <: AbstractFloat}
    return per_particle_operator!(op.A_mix_e, op.A_diffu_e, n; n_floor = FT(1.0), work = op.ne_work)
end

# ── the reference: the density operator applied to the variable itself ────────────────────
mixing_operator!(op::Operators, ::AbstractVector, ::VelocityDiffusion) = op.A_diffu_e
