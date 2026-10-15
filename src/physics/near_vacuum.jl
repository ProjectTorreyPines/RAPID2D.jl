# Numerical treatments for nodes where the plasma is not resolved: the near-empty corners of the
# vessel during start-up, where the solves have almost no particles to act on. Not physics.
#
# - the near-vacuum relaxation of u∥ and Te (`flags.vacuum_relaxation`), used by the u∥ and Te solves
# - the negative-density floor after each step (`flags.negative_n_correction`)
#
# Next to what they act on instead: the anti-diffusive pair limiter of the electron diffusion
# stencil (`limit_antidiffusive_pairs!`, numerics/wall_diffusion.jl), the flux-limited diffusivity
# (transport.jl) and the clamp on the pressure acceleration (`calculate_electron_acceleration_by_pressure`).

# ── near-vacuum relaxation of u∥ and Te ──────────────────────────────────────────

"""
    vacuum_relaxation_weight(n, n_vac) -> weight

The backward-Euler weight `Δt·ν_vac` of the near-vacuum relaxation at density `n`:
`(n_vac/n)²`, capped at 1e12. Below `10·n_vac` the relaxed value keeps
`1/(1 + weight) = n²/(n² + n_vac²)` of itself. Above it the weight fades out smoothly in
`log n` and is exactly zero from `100·n_vac`.
"""
@inline function vacuum_relaxation_weight(n::FT, n_vac::FT) where {FT <: AbstractFloat}
    n >= 100 * n_vac && return zero(FT)
    weight = n > zero(FT) ? min((n_vac / n)^2, FT(1.0e12)) : FT(1.0e12)
    n <= 10 * n_vac && return weight
    x = log10(n / (10 * n_vac))                  # 0 → 1 over the last decade
    return weight * (one(FT) - x^2 * (3 - 2x))   # smoothstep from 1 to 0
end

"""
    vacuum_relaxation_weights(RP, n) -> Matrix

`vacuum_relaxation_weight` on every in-wall node from the density `n`, zero elsewhere and when
`flags.vacuum_relaxation` is off.

- **Why.** Where there is almost no plasma, u∥ and Te have nothing to hold them. The pressure
  force `Te ∇∥ ln n` and the heating per particle grow without bound there.
- **What.** The u∥ and Te solves add `−ν_vac (f − f_ref)`, implicit (backward Euler) whatever θ
  or ExpRB the other terms use. `f_ref` is 0 for u∥ and `vacuum_reference_Te` for Te. It is a
  numerical term, kept out of the collision sums, the e–i ledger and the friction heating.
- **Where.** Only where `n < 100·n_vac` (`flags.vacuum_relaxation.factor`). Above that the solves
  are unchanged bit for bit.
"""
function vacuum_relaxation_weights(RP::RAPID{FT}, n::AbstractMatrix{FT}) where {FT <: AbstractFloat}
    weights = zeros(FT, size(n))
    relaxation = RP.flags.vacuum_relaxation       # (state = on/off, factor = n_vac)
    if !relaxation.state
        return weights
    end
    n_vac = relaxation.factor
    @inbounds for k in RP.G.nodes.in_wall_nids
        weights[k] = vacuum_relaxation_weight(n[k], n_vac)
    end
    return weights
end

"""
    vacuum_reference_Te(RP, n, Te) -> Matrix

The density-weighted mean of `Te` over each in-wall node's in-wall 3×3 neighbourhood: the
temperature the near-vacuum relaxation pulls an almost empty node to, so it takes the
temperature of the plasma next to it. A neighbourhood with no density keeps its own `Te`.
"""
function vacuum_reference_Te(RP::RAPID{FT}, n::AbstractMatrix{FT}, Te::AbstractMatrix{FT}) where {FT <: AbstractFloat}
    G = RP.G
    Te_ref = copy(Te)
    @inbounds for k in G.nodes.in_wall_nids
        i, j = G.nodes.rid[k], G.nodes.zid[k]
        sum_nTe, sum_n = zero(FT), zero(FT)
        for dj in -1:1, di in -1:1
            ii, jj = i + di, j + dj
            is_in_wall(G, ii, jj) || continue
            n_nb = max(n[ii, jj], zero(FT))
            sum_nTe += n_nb * Te[ii, jj]
            sum_n += n_nb
        end
        if sum_n > zero(FT)
            Te_ref[k] = sum_nTe / sum_n
        end
    end
    return Te_ref
end

"""
    check_vacuum_relaxation_density!(RP) -> RP

At the start of a run: when the near-vacuum relaxation would not end at least a decade below the
initial plasma, `1000·n_vac > n̄`, lower n_vac (`flags.vacuum_relaxation.factor`) to `n̄/10⁴` with
a warning. That is the default's margin, 10² m⁻³ under the 10⁶ m⁻³ seed of `initialize_density!`.

`n̄ = Σ R n² / Σ R n` over the in-wall nodes (`R = G.Jacob`, the volume factor) is the density an
average electron sits in, so empty
corners do not pull it down. The minimum would, and they are where the relaxation is meant to act.
No plasma, or the relaxation off: nothing changes.
"""
function check_vacuum_relaxation_density!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    relaxation = RP.flags.vacuum_relaxation       # (state = on/off, factor = n_vac [m⁻³])
    if !relaxation.state
        return RP                                 # relaxation off: n_vac is never used
    end
    n_vac = relaxation.factor

    # n̄ = Σ R n² / Σ R n over the in-wall nodes
    in_wall = RP.G.nodes.in_wall_nids
    n = max.(RP.plasma.ne[in_wall], zero(FT))     # a negative transient counts as empty
    particles = RP.G.Jacob[in_wall] .* n          # per node, up to 2π dR dZ
    if sum(particles) <= zero(FT)
        return RP                                 # no plasma to protect
    end
    n_bar = sum((particles ./ sum(particles)) .* n)   # weights first: n² overflows Float32 above 1.8e19

    # the relaxation acts up to 100·n_vac: keep that a decade or more below n̄
    if 1000 * n_vac > n_bar
        n_vac_new = n_bar / 10_000
        @warn "flags.vacuum_relaxation: n_vac = $(n_vac) m⁻³ would damp the initial plasma " *
            "(particle-weighted mean density $(round(n_bar; sigdigits = 3)) m⁻³); " *
            "lowered to $(round(n_vac_new; sigdigits = 3)) m⁻³"
        RP.flags.vacuum_relaxation = (state = true, factor = n_vac_new)
    end
    return RP
end

# ── the negative-density floor ───────────────────────────────────────────────────

"""
    correct_negative_densities!(RP::RAPID{FT}) where FT<:AbstractFloat

After the step: floor negative densities and book what the floor ADDED as a negative loss,
so `Δ(Σ V·n) = src − loss` keeps holding; then re-slave the ions to the electrons when they
are not evolved independently.

- Electrons (`negative_n_correction`): a negative cell is raised to
  `min(0.1·|n|, 0.01·mean|n_neighbours|)`, and the particles that creates, `(n_new − n_old)·V`,
  are taken OFF `Ne_loss`. (They used to be added to it, which double-counted them.)
- Ions, when independent: a negative cell is set to zero and `n_old·V` (negative) is added
  to `Ni_loss` — the same sign convention.

Nothing outside the wall is touched. No transport operator has rows there, and the
ionization rates are zero there, so the band keeps the value the initial condition gave it;
`initialize_density!` sets it to zero. There is no zeroing pass any more, and nothing found
on the band is ever booked — the per-face ledgers already hold everything that left.
"""
function correct_negative_densities!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    @timeit RAPID_TIMER "correct_negative_densities!" begin
        Ntracker = RP.diagnostics.Ntracker
        G = RP.G
        cell_volume(nid) = FT(2π) * G.Jacob[nid] * G.dR * G.dZ

        if RP.flags.negative_n_correction
            ne = RP.plasma.ne
            neg_e = findall(<(zero(FT)), ne)
            if !isempty(neg_e)
                ori_ne = copy(ne)
                @inbounds for nid in neg_e
                    rid, zid = G.nodes.rid[nid], G.nodes.zid[nid]
                    ngh_rids = max(1, rid - 1):min(G.NR, rid + 1)
                    ngh_zids = max(1, zid - 1):min(G.NZ, zid + 1)
                    ne[nid] = min(FT(0.1) * abs(ori_ne[nid]), FT(0.01) * mean(abs.(ori_ne[ngh_rids, ngh_zids])))
                    added = (ne[nid] - ori_ne[nid]) * cell_volume(nid)    # > 0: what the floor created
                    Ntracker.cum0D_Ne_loss -= added
                    Ntracker.cum2D_Ne_loss[nid] -= added
                end
            end

            if RP.flags.update_ni_independently
                ni = RP.plasma.ni
                @inbounds for nid in findall(<(zero(FT)), ni)
                    removed = ni[nid] * cell_volume(nid)                  # < 0: a negative loss
                    Ntracker.cum0D_Ni_loss += removed
                    Ntracker.cum2D_Ni_loss[nid] += removed
                    ni[nid] = zero(FT)
                end
            end
        end

        if !RP.flags.update_ni_independently
            # Same definition the step uses; they disagreed before (ne/Zeff vs ne).
            slave_ions_to_electrons!(RP)
        end
        return RP
    end # @timeit
end
