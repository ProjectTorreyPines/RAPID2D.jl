# Wall-aware anisotropic diffusion operator.
#
# A whole-grid tensor builder that sweeps 2:N-1 with no wall awareness (the retired `∇𝐃∇`)
# lets material diffuse past the wall to be removed afterwards; `build_reflective_diffusion_matrix`
# knows about the wall but is 5-point and isotropic. This one is both.

"""
    build_wall_diffusion_matrix!(A, G, D_RR, D_RZ, D_ZZ; cross_terms = :drop, faces = nothing, v_absorb = nothing) -> A
    build_wall_diffusion_matrix(G, D_RR, D_RZ, D_ZZ; cross_terms = :drop, faces = nothing, v_absorb = nothing)

Nine-point `∇·(𝐃∇·)` with a **reflective** wall, on the full `NR·NZ` indexing. The `!` form
writes the values into `A`, an operator on the wall pattern (`build_wall_pattern`): every slot
is rewritten and a pair that `:drop` removes stays a stored zero. The allocating form fills a
fresh pattern and drops its stored zeros.

Takes the tensor directly rather than a `RAPID` object, so the operator can be
driven from a manufactured `(D_RR, D_RZ, D_ZZ)` with no physics model behind it —
which is how the wall behaviour gets tested at anisotropies a real discharge would
take a long time to reach.

Rows for nodes on or outside the wall are left empty, so those nodes never evolve.
Nodes off the grid count as not-in-wall, so a wall coinciding with the grid frame
works with no outside region at all.

## The Robin debit

Pass `faces` (from `wall_faces`) with a per-face `v_absorb`, and each wall face
subtracts its absorption from the owning cell's diagonal:

```
    diag_i −= Σ_{f ∈ ∂wall i}  (A_f/V_i)·v_absorb_f
```

Omitting them — or passing `v_absorb = 0` — gives a reflective wall, and the
matrix is then **bit-identical** to the version with no wall term at all.

`v_absorb ≥ 0` makes the diagonal only more negative, so the operator drains
rather than sources. The three familiar wall conditions become one formula: `0` is
reflective, `→ ∞` is Dirichlet, and `¼v̄(1−R)` is the physical case between them.

**The debit is per face, not per cell.** A staircase corner owns two outward faces
and takes both; with an oblique field their `v_absorb` generally differ, since it
depends on `b̂·n̂`, a face property. Nothing is ever written outside the wall — the
absorbed material leaves as a boundary term on the interior cell.

This replaces an implicit `v_absorb = D/(2Δx)`, a discretisation artefact rather
than a surface property: measured against the kinetic ceiling it runs 15–21× too
fast at production resolution and grows without bound as `Δx → 0`.

## The stencil, and what happens at the wall

Conservative, `J`-weighted, face-averaged discretisation with the coefficients
`CT_RR = J·D_RR/ΔR²`, `CT_RZ = J·D_RZ/(ΔR·ΔZ)`, `CT_ZZ = J·D_ZZ/ΔZ²` and
`inv_J[i,j]` on the row. The cardinal part is the usual five-point operator; the
cross-derivative part adds four groups, one per cardinal **face**, reaching
diagonal neighbours. Two rules handle them:

**A cross group belongs to its face.** Group `i+½` is the cross-derivative
contribution to the flux through the `i+½` face, so if that face is a wall face
the whole group is the boundary condition's business and is dropped. What remains
ambiguous is only groups whose own face is interior but whose arms reach outside —
the neighbourhood of a staircase corner.

**A group is two centred-difference pairs, and pairs are indivisible:**

```
    group(i+½) = C·[ (f[i,j+1] − f[i,j−1]) + (f[i+1,j+1] − f[i+1,j−1]) ]
                      ‾‾‾‾‾‾ pair A ‾‾‾‾‾‾    ‾‾‾‾‾‾‾ pair B ‾‾‾‾‾‾‾
```

Dropping a single **arm** is not an option: a pair contributes `1 − 1 = 0` on a
constant field, but one arm alone contributes `1`, so the row sum stops vanishing
and the operator manufactures material out of a uniform state. Both treatments act
on whole pairs, and both keep constants in the kernel:

| `cross_terms` | rule |
|---|---|
| `:drop` (default) | remove any pair containing a node that is not in-wall |
| `:reflect` | substitute the owning cell for a not-in-wall node |

## Measured: `:drop` conserves, `:reflect` does not

Zero row sums do **not** imply `Σ J·n` is conserved — that needs `J·A` symmetric,
which is exactly what a wall treatment can break. Measured as `max|JᵀA|` over
in-wall columns, normalised by `max(J)·max|A|`, at `D∥/D⊥ = 10 / 1000`:

| wall | `:drop` | `:reflect` |
|---|---|---|
| axis-aligned box | 1.7e-16 / 2.0e-16 | **0.169 / 0.206** |
| 45° diamond | 1.7e-16 / 2.0e-16 | **0.077 / 0.092** |
| L-shape, re-entrant corner | 2.3e-16 / 1.8e-16 | **0.171 / 0.209** |

Fifteen orders apart on every wall shape and anisotropy. At `D∥/D⊥ = 1` the two
coincide: with `D_RZ = 0` there is no cross term and no question.

Reflection is *not* conservative by construction, contrary to what its name
suggests. Substituting the owning cell keeps the pair a difference, so constants
stay in the kernel and row sums still vanish — but it moves `±C` onto the diagonal
with no matching change in any other row, and `J·A` stops being symmetric. Row-sum
conservation is strictly weaker than `Σ J·n` conservation, and only the latter is
the invariant this code needs. `:reflect` is kept so the comparison stays
reproducible; it must not be used for production transport.

**Not a strict M-matrix once `D_RZ ≠ 0`.** The cross terms carry the sign of
`D_RZ`, so roughly a quarter of the off-diagonals go negative (1054 of 4312 at
`D∥/D⊥ = 10`) and `I − θΔt·A` picks up positive off-diagonals. That is a property
of the standard 9-point cross-derivative stencil, not of the wall treatment, and
it is why positivity is asserted by *solving* rather than by inspecting signs.
Where the density is not resolved, [`limit_antidiffusive_pairs!`](@ref) scales those
pairs down.
"""
function build_wall_diffusion_matrix!(
        A::DiscretizedOperator{FT}, G::GridGeometry{FT},
        D_RR::AbstractMatrix{FT}, D_RZ::AbstractMatrix{FT}, D_ZZ::AbstractMatrix{FT};
        cross_terms::Symbol = :drop,
        faces::Union{Nothing, AbstractVector{WallFace{FT}}} = nothing,
        v_absorb::Union{Nothing, AbstractVector{FT}} = nothing,
    ) where {FT <: AbstractFloat}
    check_wall_pattern(A)
    _check_wall_diffusion_args(cross_terms, faces, v_absorb)
    NR, NZ = G.NR, G.NZ
    # the Robin debit per owning cell, so the diagonal is written once
    debit = wall_face_debit(FT, NR * NZ, faces, v_absorb)
    CTRR = @. G.Jacob * D_RR / (G.dR * G.dR)
    CTRZ = @. G.Jacob * D_RZ / (G.dR * G.dZ)
    CTZZ = @. G.Jacob * D_ZZ / (G.dZ * G.dZ)
    nid = G.nodes.nid
    nz, k2c = nonzeros(A.matrix), A.k2csc
    fill!(nz, zero(FT))

    @inbounds for j in 1:NZ, i in 1:NR
        is_in_wall(G, i, j) || continue
        row = nid[i, j]
        invJ = one(FT) / G.Jacob[i, j]
        diag = zero(FT)

        # ── cardinal arms: the five-point part ──────────────────────────────
        for (di, dj, s) in ((1, 0, SLOT_E), (-1, 0, SLOT_W), (0, 1, SLOT_N), (0, -1, SLOT_S))
            ii, jj = i + di, j + dj
            is_in_wall(G, ii, jj) || continue     # reflective: omit, do not zero
            CT = dj == 0 ? CTRR : CTZZ
            c = invJ * FT(0.5) * (CT[ii, jj] + CT[i, j])
            nz[slot_position(k2c, row, s)] += c
            diag -= c
        end

        # ── cross-derivative groups, one per cardinal face ──────────────────
        for (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1))
            ii, jj = i + di, j + dj
            # a wall face's flux belongs entirely to the boundary condition
            is_in_wall(G, ii, jj) || continue
            sgn = (di + dj) > 0 ? one(FT) : -one(FT)
            c = sgn * invJ * FT(0.125) * (CTRZ[ii, jj] + CTRZ[i, j])
            iszero(c) && continue                 # isotropic tensor: nothing to add

            # the pair direction is transverse to the face
            ti, tj = dj == 0 ? (0, 1) : (1, 0)
            for (oi, oj) in ((i, j), (ii, jj))
                pi_, pj_ = oi + ti, oj + tj
                mi_, mj_ = oi - ti, oj - tj
                in_p = is_in_wall(G, pi_, pj_)
                in_m = is_in_wall(G, mi_, mj_)
                if cross_terms === :drop
                    (in_p && in_m) || continue    # the pair goes as a unit
                    nz[slot_position(k2c, row, stencil_slot(pi_ - i, pj_ - j))] += c
                    nz[slot_position(k2c, row, stencil_slot(mi_ - i, mj_ - j))] -= c
                else
                    if in_p
                        nz[slot_position(k2c, row, stencil_slot(pi_ - i, pj_ - j))] += c
                    else
                        diag += c
                    end
                    if in_m
                        nz[slot_position(k2c, row, stencil_slot(mi_ - i, mj_ - j))] -= c
                    else
                        diag -= c
                    end
                end
            end
        end

        # ── the Robin debit ────────────────────────────────────────────────
        # v_absorb ≥ 0, so this only makes the diagonal more negative: the wall
        # drains and never sources, and the term cannot cost positivity.
        nz[slot_position(k2c, row, SLOT_C)] = diag - debit[row]
    end
    return A
end

function build_wall_diffusion_matrix(
        G::GridGeometry{FT},
        D_RR::AbstractMatrix{FT}, D_RZ::AbstractMatrix{FT}, D_ZZ::AbstractMatrix{FT};
        cross_terms::Symbol = :drop,
        faces::Union{Nothing, AbstractVector{WallFace{FT}}} = nothing,
        v_absorb::Union{Nothing, AbstractVector{FT}} = nothing,
    ) where {FT <: AbstractFloat}
    A = build_wall_diffusion_matrix!(build_wall_pattern(G), G, D_RR, D_RZ, D_ZZ; cross_terms, faces, v_absorb)
    return dropzeros!(A.matrix)
end

# the stencil directions that visit every node pair once, each with the slot that points back
const PAIR_SLOTS = ((SLOT_E, SLOT_W), (SLOT_N, SLOT_S), (SLOT_NE, SLOT_SW), (SLOT_NW, SLOT_SE))

"""
    limit_antidiffusive_pairs!(A, G, n; max_ratio) -> number of limited pairs

Limit the anti-diffusive node pairs of `A` where the density is not resolved.

- **Problem.** A negative off-diagonal `A_kl` moves particles from the lower density to the
  higher one. Across an unresolved cliff the low node loses `|A_kl| n_l`, more than it holds.
- **Transfers.** The product `R_k·A_kl` is the rate at which node `l` passes particles to node
  `k` per unit `n_l`, with `R_k = G.Jacob[k]` the major radius of node `k`, its volume factor.
  Conservation is `Σ_k R_k·A_kl = 0` for every `l`.
- **Fix.** Where the smaller transfer of a pair, `min(R_k·A_kl, R_l·A_lk)`, is negative and
  `n_high > max_ratio·n_low`, add `(1 − keep)·|min(R_k·A_kl, R_l·A_lk)|` to both transfers and
  take it off both diagonals, with `keep = max_ratio·n_low/n_high`. This is a symmetric pair
  diffusion, so the row sums and the conservation sums stay as they were. A negative entry ends
  up at least `keep` times itself.
- **Bounds.** The low node then loses at most `max_ratio |A_kl| n_low`. A negative entry of the
  per-particle operator, `A_kl n_l/n_k`, stays above `−max_ratio |A_kl|`. The weaker side of an
  asymmetric pair may turn positive, which only adds diffusion.
- **Where it acts.** Only pairs with a negative entry and a density ratio above `max_ratio`. A
  resolved density gives back `A` bit for bit. The Robin debit is not changed.

With cross terms `R_k·A_kl ≠ R_l·A_lk`, so scaling `A_kl` and `A_lk` by `keep` would not conserve.
`A` is an operator on the wall pattern of `G`. `n` is the node vector or the `(NR, NZ)` grid,
must be finite, and is read with negative values as zero. `max_ratio ≥ 1` must be finite.
Nothing is allocated.
"""
function limit_antidiffusive_pairs!(
        A::DiscretizedOperator{FT}, G::GridGeometry{FT}, n::AbstractVecOrMat{FT}; max_ratio::FT,
    ) where {FT <: AbstractFloat}
    check_wall_pattern(A)
    n_nodes = size(A.matrix, 1)
    (A.dims_rz == (G.NR, G.NZ) && length(n) == n_nodes) ||
        throw(DimensionMismatch("limit_antidiffusive_pairs!: A and n must be on the grid of G"))
    all(isfinite, n) || throw(ArgumentError("limit_antidiffusive_pairs!: n must be finite"))
    (isfinite(max_ratio) && max_ratio >= one(FT)) ||
        throw(ArgumentError("limit_antidiffusive_pairs!: max_ratio must be finite and ≥ 1, got $max_ratio"))
    NR, Jacob = G.NR, G.Jacob
    nz, k2c = nonzeros(A.matrix), A.k2csc
    n_limited = 0
    @inbounds for k in 1:n_nodes, (slot, slot_back) in PAIR_SLOTS
        p_kl = slot_position(k2c, k, slot)
        p_kl == 0 && continue
        di, dj = STENCIL_OFFSETS[slot]
        l = k + di + dj * NR
        p_lk = slot_position(k2c, l, slot_back)     # the pattern is structurally symmetric
        transfer_kl, transfer_lk = Jacob[k] * nz[p_kl], Jacob[l] * nz[p_lk]
        transfer_min = min(transfer_kl, transfer_lk)
        transfer_min < zero(FT) || continue
        n_low, n_high = minmax(max(n[k], zero(FT)), max(n[l], zero(FT)))
        n_high <= max_ratio * n_low && continue      # resolved (or both empty): untouched
        keep = max_ratio * n_low / n_high            # n_high > max_ratio·n_low ≥ 0, so n_high > 0
        # transfer + added written as (transfer − transfer_min) + keep·transfer_min: the negative
        # side becomes keep·transfer_min without cancellation
        new_kl = ((transfer_kl - transfer_min) + keep * transfer_min) / Jacob[k]
        new_lk = ((transfer_lk - transfer_min) + keep * transfer_min) / Jacob[l]
        _set_pair_entry!(nz, p_kl, slot_position(k2c, k, SLOT_C), new_kl)
        _set_pair_entry!(nz, p_lk, slot_position(k2c, l, SLOT_C), new_lk)
        n_limited += 1
    end
    return n_limited
end

# set the off-diagonal at `p_offdiag` to `value`, and keep the row sum through the diagonal at `p_diag`
@inline function _set_pair_entry!(nz, p_offdiag, p_diag, value)
    @inbounds nz[p_diag] -= value - nz[p_offdiag]
    @inbounds nz[p_offdiag] = value
    return nothing
end

function _check_wall_diffusion_args(cross_terms, faces, v_absorb)
    cross_terms in (:drop, :reflect) ||
        throw(ArgumentError("cross_terms must be :drop or :reflect, got :$cross_terms"))
    isnothing(faces) == isnothing(v_absorb) ||
        throw(ArgumentError("`faces` and `v_absorb` must be given together"))
    if !isnothing(faces)
        length(faces) == length(v_absorb) ||
            throw(DimensionMismatch("v_absorb must have one entry per wall face"))
        any(<(0), v_absorb) &&
            throw(ArgumentError("v_absorb must be non-negative: a wall cannot emit here"))
    end
    return nothing
end

"""
    wall_face_debit(FT, Ng, faces, v) -> Vector

`Σ_{f ∈ ∂wall i} (A_f/V_i)·v_f` per owning cell: a speed per wall face turned into a diagonal
rate per node. A staircase corner takes both of its faces. No faces: all zero.
"""
function wall_face_debit(::Type{FT}, Ng::Int, faces, v) where {FT <: AbstractFloat}
    debit = zeros(FT, Ng)
    isnothing(faces) && return debit
    for (f, vf) in zip(faces, v)
        debit[f.nid] += f.area_per_volume * vf
    end
    return debit
end
