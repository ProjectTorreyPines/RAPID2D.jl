# Electron wall: the Robin coefficients and the wall-aware operator for the electron
# continuity equation. The tensor is the transport tensor `tp.DRR/DRZ/DZZ`; only `v_absorb` is
# built from channels, because a kinetic ceiling is a property of the channel's speed
# moments, not of the tensor (`channel_ceiling`). Design: internal/docs/src/notes/design/
# electron-wall-absorption-options.md (option A) and wall-flux-channels.md §2.1.

"""
    electron_wall_channels(RP) -> [(channel, bR, bZ), ...]

The three channels the electron transport tensor is a sum of, in the channel basis, so
`wall_absorption_speeds` can take their kinetic ceilings. `parallel_collisional_channel`
with `D∥ = tp.Dpara` reproduces the tensor's parallel diffusivity exactly (λ = 2D/vp);
`bohm_channel` reproduces `Te/(16B)`; the turbulent channel is species-independent.
`Dperp0` (a bare base diffusivity) has no kinetic ceiling and is not a channel.
"""
function electron_wall_channels(RP::RAPID{FT}) where {FT <: AbstractFloat}
    pla, tp, F = RP.plasma, RP.transport, RP.fields
    me = RP.config.constants.me
    mi = bulk_ion_mass(RP)
    vp_e = maxwellian_most_probable_speed.(pla.Te_eV, me)
    chans = Any[
        (parallel_collisional_channel(vp_e, tp.Dpara), F.bR, F.bZ),
        (bohm_channel(pla.Te_eV, F.Bϕ, mi, 1), F.bR, F.bZ),
    ]
    turb = shared_turbulent_channel(RP)
    isnothing(turb) || push!(chans, (turb, F.bpol_R, F.bpol_Z))
    return chans
end

"""
    electron_wall_albedo(RP) -> FT

`config.electron_wall_albedo`, validated to lie in [0, 1]. Both wall channels read the albedo
through this — the diffusive builder validates too, but it does not run with `diffu = false`,
and an out-of-range value would otherwise turn the convective debit into a source.
"""
function electron_wall_albedo(RP::RAPID{FT}) where {FT <: AbstractFloat}
    a = FT(RP.config.electron_wall_albedo)
    zero(FT) <= a <= one(FT) ||
        throw(ArgumentError("electron_wall_albedo must lie in [0, 1], got $a"))
    return a
end

"Robin coefficient per wall face for electrons: Σ_channels ¼v̄_n · (1 − a_f)."
function electron_wall_absorption_speeds(RP::RAPID{FT}, faces) where {FT <: AbstractFloat}
    return wall_absorption_speeds(electron_wall_channels(RP), faces, electron_wall_albedo(RP))
end

"Wall-aware `∇·(𝐃∇·)` for electrons plus its Robin coefficients, from the transport tensor."
function electron_transport_operator(RP::RAPID{FT}, faces) where {FT <: AbstractFloat}
    tp = RP.transport
    v_absorb = electron_wall_absorption_speeds(RP, faces)
    A = build_wall_diffusion_matrix(
        RP.G, tp.DRR, tp.DRZ, tp.DZZ;
        cross_terms = :drop, faces = faces, v_absorb = v_absorb
    )
    return A, v_absorb
end

"Book what the electron Robin wall took this step, per face, into the electron tracker."
function book_electron_wall_loss!(
        RP::RAPID{FT}, faces, v_absorb::AbstractVector{FT},
        n_new::AbstractArray{FT}, n_prev::AbstractArray{FT}, θ
    ) where {FT <: AbstractFloat}
    isempty(faces) && return RP
    ledger = WallLedger{FT}(length(faces))
    accumulate_wall_absorption!(ledger, faces, v_absorb, n_new, RP.dt; n_prev = n_prev, θ = θ)
    Ntracker = RP.diagnostics.Ntracker
    Ntracker.cum0D_Ne_loss += sum(ledger.absorbed)
    for (k, f) in enumerate(faces)
        Ntracker.cum2D_Ne_loss[f.nid] += ledger.absorbed[k]
    end
    return RP
end
