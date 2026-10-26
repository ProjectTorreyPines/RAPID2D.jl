# Electron wall: the Robin coefficients and the wall-aware operator for the electron
# continuity equation. The tensor is the transport tensor `tp.DRR/DRZ/DZZ`; only `v_absorb` is
# built from channels, because a kinetic ceiling is a property of the channel's speed
# moments, not of the tensor (`channel_ceiling`). Design: internal/docs/src/notes/design/
# electron-wall-absorption-options.md (option A) and wall-flux-channels.md §2.1.

"""
    electron_wall_channels(RP) -> [(channel, bR, bZ), ...]

The three channels the electron transport tensor is a sum of, in the channel basis, so
`wall_absorption_speeds` can take their kinetic ceilings. The parallel channel carries
`D∥ = tp.Dpara`, the tensor's parallel diffusivity, and delivers to the wall what the ambipolar
coupling mixes, the electrons' own one-sided flux `¼v̄ₑn` and the common mode's Bohm flux `c_s n`
([`coupled_parallel_channel`](@ref)); the Bohm channel moves at the
ions' Bohm speeds, averaged over the species by the electrons each brings (`Z n`), so the
coupled pairs leave at one speed; the turbulent channel is species-independent. `Dperp0` (a
bare base diffusivity) has no kinetic ceiling and is not a channel.
"""
function electron_wall_channels(RP::RAPID{FT}) where {FT <: AbstractFloat}
    pla, tp, F = RP.plasma, RP.transport, RP.fields
    me = RP.config.constants.me
    vm_e = maxwellian_mean_speed.(pla.Te_eV, me)
    c_s = sound_speed.(pla.Te_eV, pla.Ti_eV, pla.mi_mean, mean_ion_charge(RP))
    species = tp.ion_species
    bohm = if length(species) == 1
        bohm_channel(pla.Te_eV, F.Bϕ, only(species).mass, 1)
    else
        weights = [s.charge .* ion_plane(pla, slot) for (s, slot) in zip(species, RP.store_layout.transport_slots)]
        mixture_channel([bohm_channel(pla.Te_eV, F.Bϕ, s.mass, 1) for s in species], weights)
    end
    chans = Any[
        (coupled_parallel_channel(vm_e, c_s, tp.Dpara, pla.ambipolar_weight), F.bR, F.bZ),
        (bohm, F.bR, F.bZ),
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

"""
Wall-aware `∇·(𝐃∇·)` for electrons plus its Robin coefficients, from the transport tensor: the
fresh reference. The continuity solve applies the same Robin wall as a diagonal rate on the
cached reflective operator (`operators.A_diffu_e`).
"""
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
