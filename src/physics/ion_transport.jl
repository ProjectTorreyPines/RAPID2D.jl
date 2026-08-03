# From a set of ion species to an advanced density.
#
# The policy types and `IonSpecies` are declared in `ion_species.jl`, ahead of
# `types.jl`; this file is everything that consumes them — grouping, assembly and
# the batch solve.

"""
    IonTransportGroup{FT}

A set of ion species that will be advanced by one shared operator, together with
the channels that operator is assembled from.

`sids` index into the species list; `channels` holds one entry per transport
mechanism, in the order the caller supplied them.
"""
struct IonTransportGroup{FT <: AbstractFloat}
    sids::Vector{Int}
    channels::Vector{DiffusionChannel{FT}}
end

"""
    ion_transport_groups(policy, channels_per_species, weights) -> Vector{IonTransportGroup}

Map ion species onto transport operators. **The one place a policy is consulted.**

`channels_per_species[s][m]` is species `s`'s channel for mechanism `m`; every
species must list the same mechanisms in the same order. A species without a
given mechanism passes a zero-speed channel rather than omitting it — a missing
entry is a construction bug, not a physical statement. `weights[s]` is that
species' density field.

| policy | groups | factorizations | right-hand sides |
|---|---|---|---|
| [`SharedEffectiveTransport`](@ref) | 1 | 1 | one per species |
| [`PerSpeciesTransport`](@ref) | one per species | one per species | 1 each |

With a single species the two agree exactly, so the default may change without
changing today's answer.
"""
function ion_transport_groups end

function ion_transport_groups(::PerSpeciesTransport, channels_per_species, weights)
    _check_species_channels(channels_per_species, weights)
    return [
        IonTransportGroup(Int[s], collect(channels_per_species[s]))
            for s in eachindex(channels_per_species)
    ]
end

function ion_transport_groups(::SharedEffectiveTransport, channels_per_species, weights)
    nmech = _check_species_channels(channels_per_species, weights)
    mixed = [
        mixture_channel([chs[m] for chs in channels_per_species], collect(weights))
            for m in 1:nmech
    ]
    return [IonTransportGroup(collect(eachindex(channels_per_species)), mixed)]
end

"""
    wall_absorption_speeds(channels_with_directions, faces, albedo) -> Vector

Robin coefficient `v_absorb = ¼v̄_n·(1 − R)` [m/s] for every entry of `faces`,
summed over the mechanisms in `channels_with_directions`.

`albedo` is the fraction returned to the plasma: a scalar for a uniform surface,
or one value per face where the surface is not uniform. `R = 1` gives exactly
zero — a reflective wall, and a matrix bit-identical to one assembled with no
wall term at all.

**Sampled per face, not per cell.** The ceiling depends on `b̂·n̂`, so the two
faces of a staircase corner generally carry different speeds even though they
share an owning cell and its plasma state. There are only four possible
outward normals, so the ceiling field is built once per direction and indexed,
rather than recomputed for each of the thousands of faces.
"""
function wall_absorption_speeds(
        channels_with_directions,
        faces::AbstractVector{WallFace{FT}},
        albedo
    ) where {FT <: AbstractFloat}
    in_range(r) = zero(FT) <= r <= one(FT)
    if albedo isa AbstractVector
        length(albedo) == length(faces) ||
            throw(DimensionMismatch("got $(length(albedo)) albedos for $(length(faces)) faces"))
        all(in_range, albedo) ||
            throw(ArgumentError("every albedo must lie in [0, 1]"))
    else
        in_range(albedo) ||
            throw(ArgumentError("albedo must lie in [0, 1], got $albedo"))
    end

    ceilings = Dict{Tuple{Int, Int}, Matrix{FT}}()
    v_absorb = Vector{FT}(undef, length(faces))
    for (k, f) in enumerate(faces)
        c = get!(() -> total_ceiling(channels_with_directions, f.outward), ceilings, f.outward)
        R = albedo isa AbstractVector ? albedo[k] : albedo
        v_absorb[k] = (one(FT) - FT(R)) * c[f.nid]
    end
    return v_absorb
end

"""
    ion_transport_operator(G, group, directions; faces, albedo, cross_terms) -> (A, v_absorb)

Assemble the wall-aware `∇·(𝐃∇·)` a group of ion species will share, and the
Robin coefficients that go with it.

`directions[m]` is the `(bR, bZ)` the group's `m`-th channel is aligned with —
the full `b̂` for a collisional mechanism, `b̂_pol` for a turbulent one. Directions
are field properties, so they are the same for every species and live outside the
group.

The tensor and the wall ceiling are built from **one** channel list, which is the
reason this exists as a function rather than two calls at each site: a matrix
assembled from one set of channels and a boundary condition from another is a
mismatch nothing downstream can detect.

Omitting `faces` gives the reflective operator and an empty coefficient vector.
"""
function ion_transport_operator(
        G::GridGeometry{FT}, group::IonTransportGroup{FT}, directions;
        faces::Union{Nothing, AbstractVector{WallFace{FT}}} = nothing,
        albedo = zero(FT),
        cross_terms::Symbol = :drop,
    ) where {FT <: AbstractFloat}
    length(directions) == length(group.channels) || throw(
        ArgumentError(
            "each of the group's $(length(group.channels)) mechanisms needs a direction, " *
                "got $(length(directions))"
        )
    )
    cwd = [
        (group.channels[m], directions[m][1], directions[m][2])
            for m in eachindex(group.channels)
    ]
    D_RR, D_RZ, D_ZZ = total_tensor(cwd)
    # A non-finite diffusivity reaches the solver as `SingularException(0)` from a
    # factorization several calls away, which says nothing about which channel
    # produced it. Collisionless cells are the way in — `D = ½v²/ν` diverges — so
    # the check is worth one pass over the grid.
    all(isfinite, D_RR) && all(isfinite, D_RZ) && all(isfinite, D_ZZ) || throw(
        ArgumentError(
            "ion diffusivity is not finite on $(count(!isfinite, D_RR)) node(s); a " *
                "channel produced Inf or NaN, most likely a collisionless cell"
        )
    )

    isnothing(faces) &&
        return build_wall_diffusion_matrix(G, D_RR, D_RZ, D_ZZ; cross_terms), FT[]

    v_absorb = wall_absorption_speeds(cwd, faces, albedo)
    A = build_wall_diffusion_matrix(
        G, D_RR, D_RZ, D_ZZ;
        cross_terms = cross_terms, faces = faces, v_absorb = v_absorb
    )
    return A, v_absorb
end

"""
    solve_ion_group!(N, group, A, solver, dt; θ = 1, S = nothing) -> N

Advance every species of `group` through the θ-scheme

```
    (𝐈 − θΔt𝐀)nⁿ⁺¹ = nⁿ + (1−θ)Δt𝐀nⁿ + ΔtS
```

with **one** factorization. `N` is `(NR·NZ) × Nspecies` with species as columns —
the layout `reshape` gives an `(NR, NZ, Nspecies)` density array for free — and
only the columns in `group.sids` are read or written.

The batch is the whole point of grouping. A second species in a group costs a
backsolve (68–488 µs across the measured grid sizes) against a factorization
(2.0–14.0 ms), so a shared operator over ten species is roughly the price of one.
"""
function solve_ion_group!(
        N::AbstractMatrix{FT}, group::IonTransportGroup{FT},
        A::SparseMatrixCSC{FT}, solver::AbstractLinearSolver{FT}, dt::Real;
        θ::Real = 1, S::Union{Nothing, AbstractMatrix{FT}} = nothing,
    ) where {FT <: AbstractFloat}
    Ng = size(N, 1)
    size(A, 1) == Ng ||
        throw(DimensionMismatch("operator is $(size(A, 1))×$(size(A, 2)) but N has $Ng rows"))
    maximum(group.sids) <= size(N, 2) ||
        throw(BoundsError(N, (:, maximum(group.sids))))

    sids = group.sids
    B = N[:, sids]
    θ < 1 && (B .+= FT((1 - θ) * dt) .* (A * B))
    isnothing(S) || (B .+= FT(dt) .* view(S, :, sids))

    factorize!(solver, I - FT(θ * dt) * A)
    X = similar(B)
    solve!(X, solver, B)
    @views N[:, sids] .= X
    return N
end

"Validate the ragged-array contract shared by every policy; return the mechanism count."
function _check_species_channels(channels_per_species, weights)
    isempty(channels_per_species) &&
        throw(ArgumentError("no ion species to transport"))
    length(channels_per_species) == length(weights) || throw(
        ArgumentError(
            "got $(length(channels_per_species)) species but $(length(weights)) density fields"
        )
    )
    nmech = length(first(channels_per_species))
    nmech > 0 || throw(ArgumentError("an ion species needs at least one transport mechanism"))
    all(chs -> length(chs) == nmech, channels_per_species) || throw(
        ArgumentError(
            "every species must list the same mechanisms in the same order; got lengths " *
                "$(map(length, channels_per_species)). A species without a mechanism passes " *
                "a zero-speed channel, it does not omit the entry"
        )
    )
    return nmech
end

# ── the production ion equation ─────────────────────────────────────────────

"""
    ion_transport_channels(RP, species) -> Vector{DiffusionChannel}

The transport mechanisms `species` participates in, in the order
[`ion_channel_directions`](@ref) lists their axes.

Three mechanisms, and their species dependence is the whole argument for having a
policy at all:

| mechanism | axis | depends on the species? |
|---|---|---|
| collisional | `b̂` | **yes** — `D∥ = ½v_p²/ν` with `v_p = √(2T_i/m)` |
| Bohm | `b̂` | `D⊥ = T_e/16B` is mass-free; only the `v`/`λ` split moves |
| turbulent ExB | `b̂_pol` | no — `v_E = E_pol/B_tot` has neither mass nor charge in it |

The ExB channel is therefore built once and handed to every species as the same
object, which is what lets the mixture return it untouched.

`ν` is the bulk ion collision frequency for every species. Species-resolved
`ν_ss'` is a refinement the per-species policy could use; it is not what
separates the two policies today.
"""
function ion_transport_channels(RP::RAPID{FT}, species::IonSpecies{FT}, shared_turb) where {FT <: AbstractFloat}
    pla, tp, F = RP.plasma, RP.transport, RP.fields
    ee = RP.config.constants.ee

    # λ∥ is built directly as a length, not recovered from a diffusivity. Going
    # through `D∥ = ½v²/ν` and back through `λ = 2D/v` is exactly the lossy round
    # trip the channel basis exists to avoid, and it is what turns a collisionless
    # cell into `Inf/0`. Written as an inverse length, every degenerate limit falls
    # out instead of needing a case:
    #
    #   T_i = 0            v_p = 0 → ν/v_p = Inf → λ∥ = 0, and D∥ = ½v∥λ∥ = 0
    #   no collisions      λ∥ = L_field: free streaming to the wall along B
    #   no field length    nothing bounds a parallel step, so the COLLISIONAL
    #                      channel is absent. Free streaming is convection, and the
    #                      equation already carries it as −∇·(n𝐮_i).
    # T_i and the fluid velocity are shared — the reaction set carries one ion
    # temperature and one ion drift, and density is the only per-species state, so
    # everything separating two ion species arrives through m and Z.
    #
    # `νi_coulomb` is the SELF-collision rate (NRL Plasma Formulary p.28,
    # ν_i ∝ Z⁴μ^-½ n λ T^-3/2), which is what `update_coulomb_collision_parameters!`
    # computes for the bulk. A trace species does not collide with itself: it
    # collides with the BULK, and NRL's ion–ion test-particle rate (p.33) is
    #
    #     ν_s^{z|i} ∝ n_i Z_z²Z_i² λ (μ_i^½/μ_z)(1 + μ_i/μ_z) T^-3/2
    #
    # Dividing that by the same expression at z = i gives `coulomb_scale` below,
    # which is exactly 1 for the bulk. With v_p,z ∝ μ_z^-½ the mass then CANCELS
    # out of the diffusivity for a heavy impurity,
    #
    #     D∥_z / D∥_bulk = 2 / [ (Z_z/Z_i)² (1 + μ_i/μ_z) ]
    #
    # — using the self-collision μ^-½ instead makes C⁶⁺ 4.2× less diffusive.
    #
    # Only the Coulomb half of ν is scaled this way. An ion–neutral collision rate
    # goes as n₀σ√(T/m) (NRL p.39), so its contribution to 1/λ = ν/v is
    # species-independent outright, and scaling it by Z² would overstate an
    # impurity's collisionality through the whole gas-dominated early discharge.
    #
    # max(0, ·): the Ti equation is free to land microscopically below zero
    # (−1.3e-61 was observed), and `sqrt` of that is a DomainError, not a NaN.
    m_ref = RP.config.constants.mi
    mass_ratio = FT(m_ref / species.mass)
    coulomb_scale = FT(species.charge^2) * sqrt(mass_ratio) * (1 + mass_ratio) / 2
    v_p_ref = @. sqrt(max(zero(FT), 2 * pla.Ti_eV * ee / m_ref))
    v_p = @. v_p_ref * sqrt(mass_ratio)
    inv_λ = @. (tp.νi_neutral + coulomb_scale * tp.νi_coulomb) / v_p_ref +
        ifelse(tp.L_mixing > 0, 1 / tp.L_mixing, zero(FT))
    # `inv_λ > 0` is false for both Inf⁻¹ = 0 and for a NaN out of 0/0, so the two
    # degenerate cases above land on λ∥ = 0 without being enumerated.
    λ_para = @. ifelse(inv_λ > 0, 1 / inv_λ, zero(FT))
    if tp.Dpara0 > zero(FT)
        # A base diffusivity adds a step length, because ½v(λ + λ₀) = D + ½vλ₀
        @. λ_para += ifelse(v_p > 0, 2 * tp.Dpara0 / v_p, zero(FT))
    end
    collisional = DiffusionChannel(v_p, λ_para, zero.(v_p), zero.(v_p))

    bohm = bohm_channel(pla.Te_eV, F.Bϕ, species.mass, species.charge)
    if !all(isfinite, bohm.λ_perp)
        # ρ_s = c_s/ω_ci is 0/0 wherever B vanishes; no field means no gyro-step
        bohm = DiffusionChannel(
            bohm.v_para, bohm.λ_para, bohm.v_perp,
            (@. ifelse(isfinite(bohm.λ_perp), bohm.λ_perp, zero(FT)))
        )
    end
    if tp.Dperp0 > zero(FT)
        # A constant floor has no speed of its own, so it rides at the Bohm speed:
        # D⊥ picks it up exactly and the kinetic ceiling, which depends only on v,
        # is left alone.
        bohm = DiffusionChannel(
            bohm.v_para, bohm.λ_para, bohm.v_perp,
            (@. bohm.λ_perp + ifelse(bohm.v_perp > 0, 2 * tp.Dperp0 / bohm.v_perp, zero(FT)))
        )
    end

    channels = isnothing(shared_turb) ? [collisional, bohm] : [collisional, bohm, shared_turb]
    for (name, ch) in zip(ION_MECHANISMS, channels)
        _finite_channel(ch) || throw(
            ArgumentError(
                "the $name channel of $(species.name) is not finite. A field line that " *
                    "never reaches a wall gives an unbounded L_mixing, and a cell with no " *
                    "collisions gives an unbounded mean free path; both make D = ½vλ diverge"
            )
        )
    end
    return channels
end

"Mechanism names, in the order [`ion_transport_channels`](@ref) returns them."
const ION_MECHANISMS = ("collisional", "Bohm", "turbulent ExB")

_finite_channel(ch::DiffusionChannel) =
    all(isfinite, ch.v_para) && all(isfinite, ch.λ_para) &&
    all(isfinite, ch.v_perp) && all(isfinite, ch.λ_perp)

"""
    ion_channel_directions(RP) -> Vector{Tuple}

The `(bR, bZ)` each mechanism of [`ion_transport_channels`](@ref) is aligned
with. Directions are field properties, identical for every species, so they are
built once per step rather than per species.
"""
function ion_channel_directions(RP::RAPID{FT}) where {FT <: AbstractFloat}
    F = RP.fields
    dirs = [(F.bR, F.bZ), (F.bR, F.bZ)]
    RP.flags.turb_ExB_mixing && push!(dirs, (F.bpol_R, F.bpol_Z))
    return dirs
end

"The ExB channel every ion species shares, or `nothing` when it is switched off."
function shared_turbulent_channel(RP::RAPID{FT}) where {FT <: AbstractFloat}
    RP.flags.turb_ExB_mixing || return nothing
    f_para = FT(RP.config.turbulent_diffusion_fraction_along_bpol)
    return turbulent_ExB_channel(
        RP.fields.Epol_self, RP.fields.Btot, RP.transport.L_mixing,
        f_para, one(FT) - f_para
    )
end

"""
    solve_ion_continuity_equation!(RP)

Advance every ion species one step of

```
    ∂nₛ/∂t = ∇·(𝐃ₛ∇nₛ) + Sₛ
```

under the θ-scheme, grouped by `RP.flags.ion_transport_policy`.

The ionization source is `nₑ·ν_iz` — one ion per electron, at a rate set by the
**electron** density, so for ions it is a pure explicit source with no diagonal
counterpart. `ν_iz` is the value `update_RRCs!` materialized at the step-entry
state; the tables are not re-queried here.

The wall is the Robin condition of [`wall_absorption_speeds`](@ref) at
`config.ion_wall_albedo`, and what it takes is booked at the face. That matters
because the wall-aware operator never writes outside the wall, so the older
accounting — read whatever density is found on out-of-wall nodes — would report
exactly zero loss for a wall that is in fact draining.

Convection is `−∇·(n𝐮_i)` from `operators.∇𝐮_i`, built from the **ion**
velocities. It is shared by every group: `ν_ii` couples the species into one
fluid far faster than transport separates them, so a species-resolved `𝐮` would
be modelling a drift friction forbids.

The two wall treatments differ by channel, which is deliberate. Diffusion leaves
through the Robin face term at `¼v̄_n(1−R)` and is booked there; convection uses
the interior-sweeping operator, deposits on out-of-wall nodes, and is booked by
`treat_ion_outside_wall!`. The paths do not overlap — the Robin debit never
writes outside the wall — and a surface reached at `n𝐮·n̂` is not the same
boundary condition as one reached at `¼v̄n`.
"""
function solve_ion_continuity_equation!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    return @timeit RAPID_TIMER "solve_ion_continuity_equation!" begin
        tp, pla = RP.transport, RP.plasma
        dt = RP.dt
        θ = RP.flags.Implicit ? FT(RP.flags.Implicit_weight) : zero(FT)

        N, S = tp.ion_N, tp.ion_S
        N[:, 1] .= vec(pla.ni)
        fill!(S, zero(FT))
        if RP.flags.src
            @views S[:, 1] .= vec(pla.ne) .* vec(pla.ν_en_iz)
        end

        if !RP.flags.diffu && !RP.flags.convec
            N .+= dt .* S
        else
            groups, faces = ion_step_operators(RP)
            while length(tp.ion_solvers) < length(groups)
                push!(tp.ion_solvers, SparseLUSolver{FT}())
            end
            for (gi, (group, A, v_absorb)) in enumerate(groups)
                n_prev = N[:, group.sids]
                solve_ion_group!(N, group, A, tp.ion_solvers[gi], dt; θ = θ, S = S)
                book_ion_wall_loss!(RP, group, faces, v_absorb, N, n_prev, θ)
            end
        end

        vec(pla.ni) .= view(N, :, 1)
        return RP
    end
end

"""
    ion_step_operators(RP) -> (Vector{Tuple{group, A, v_absorb}}, faces)

The operators this step will invert, one per transport group, and the wall faces
they were built against.

With diffusion off there is nothing for a policy to partition — every species
sees the same (null) diffusion — so a single group covers them all and carries
only the convective term.
"""
function ion_step_operators(RP::RAPID{FT}) where {FT <: AbstractFloat}
    tp, G = RP.transport, RP.G
    ns = length(tp.ion_species)
    convection = RP.flags.convec ? RP.operators.∇𝐮_i.matrix : nothing

    if !RP.flags.diffu
        group = IonTransportGroup(collect(1:ns), DiffusionChannel{FT}[])
        Ng = G.NR * G.NZ
        A = isnothing(convection) ? spzeros(FT, Ng, Ng) : -convection
        return [(group, A, FT[])], WallFace{FT}[]
    end

    turb = shared_turbulent_channel(RP)
    per_species = [ion_transport_channels(RP, sp, turb) for sp in tp.ion_species]
    dirs = ion_channel_directions(RP)
    weights = [reshape(view(tp.ion_N, :, s), G.NR, G.NZ) for s in 1:ns]
    faces = wall_faces(G)
    albedo = FT(RP.config.ion_wall_albedo)

    ops = map(ion_transport_groups(RP.flags.ion_transport_policy, per_species, weights)) do group
        A, v_absorb = ion_transport_operator(G, group, dirs; faces = faces, albedo = albedo)
        return (group, isnothing(convection) ? A : A - convection, v_absorb)
    end
    return ops, faces
end

"Book what the Robin condition took this step, per face, into the ion tracker."
function book_ion_wall_loss!(
        RP::RAPID{FT}, group::IonTransportGroup{FT}, faces, v_absorb,
        N::AbstractMatrix{FT}, n_prev::AbstractMatrix{FT}, θ
    ) where {FT <: AbstractFloat}
    isempty(faces) && return RP
    ledger = WallLedger{FT}(length(faces))
    for (c, s) in enumerate(group.sids)
        accumulate_wall_absorption!(
            ledger, faces, v_absorb, view(N, :, s), RP.dt;
            n_prev = view(n_prev, :, c), θ = θ
        )
    end
    Ntracker = RP.diagnostics.Ntracker
    Ntracker.cum0D_Ni_loss += sum(ledger.absorbed)
    for (k, f) in enumerate(faces)
        Ntracker.cum2D_Ni_loss[f.nid] += ledger.absorbed[k]
    end
    return RP
end

"""
    set_ion_species!(RP, species) -> RP

Declare which ion species the transport solve advances, and size the work
buffers to match.

Species are columns of `ion_N`/`ion_S`, in the order given; column 1 is the
species `plasma.ni` mirrors. Appending a species here is the whole cost of
adding it to the transport solve — grouping, assembly and the batch solve are
already written for a list.
"""
function set_ion_species!(RP::RAPID{FT}, species::AbstractVector{IonSpecies{FT}}) where {FT <: AbstractFloat}
    isempty(species) && throw(ArgumentError("at least one ion species is required"))
    allunique(sp.name for sp in species) ||
        throw(ArgumentError("ion species names must be unique, got $(map(sp -> sp.name, species))"))
    tp = RP.transport
    tp.ion_species = collect(species)
    Ng = RP.G.NR * RP.G.NZ
    tp.ion_N = zeros(FT, Ng, length(species))
    tp.ion_S = zeros(FT, Ng, length(species))
    empty!(tp.ion_solvers)
    return RP
end
