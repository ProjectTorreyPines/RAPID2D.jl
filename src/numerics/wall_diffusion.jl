# Wall-aware anisotropic diffusion operator.
#
# Neither existing builder does this job. `compute_∇𝐃∇f_directly` and the assembled
# `∇𝐃∇` carry the full tensor but sweep 2:N-1 with no wall awareness, so material
# diffuses past the wall and is removed afterwards. `build_reflective_diffusion_matrix`
# knows about the wall but is 5-point and isotropic. This one is both.

"""
    build_wall_diffusion_matrix(G, D_RR, D_RZ, D_ZZ; cross_terms = :drop)

Nine-point `∇·(𝐃∇·)` with a **reflective** wall, on the full `NR·NZ` indexing.

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

Same conservative, `J`-weighted, face-averaged discretisation as
`compute_∇𝐃∇f_directly` (`CT_RR = J·D_RR/ΔR²`, `CT_RZ = J·D_RZ/(ΔR·ΔZ)`,
`CT_ZZ = J·D_ZZ/ΔZ²`). The cardinal part is the usual five-point operator; the
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
"""
function build_wall_diffusion_matrix(
        G::GridGeometry{FT},
        D_RR::AbstractMatrix{FT}, D_RZ::AbstractMatrix{FT}, D_ZZ::AbstractMatrix{FT};
        cross_terms::Symbol = :drop,
        faces::Union{Nothing, AbstractVector{WallFace{FT}}} = nothing,
        v_absorb::Union{Nothing, AbstractVector{FT}} = nothing,
    ) where {FT <: AbstractFloat}

    cross_terms in (:drop, :reflect) ||
        throw(ArgumentError("cross_terms must be :drop or :reflect, got :$cross_terms"))
    isnothing(faces) == isnothing(v_absorb) ||
        throw(ArgumentError("`faces` and `v_absorb` must be given together"))
    if !isnothing(faces)
        length(faces) == length(v_absorb) ||
            throw(DimensionMismatch("v_absorb must have one entry per wall face"))
        any(<(zero(FT)), v_absorb) &&
            throw(ArgumentError("v_absorb must be non-negative: a wall cannot emit here"))
    end

    # Accumulate the Robin debit per owning cell first, so the diagonal is written
    # once. A staircase corner contributes through both of its faces.
    debit = zeros(FT, G.NR * G.NZ)
    if !isnothing(faces)
        for (f, v) in zip(faces, v_absorb)
            debit[f.nid] += f.area_per_volume * v
        end
    end

    NR, NZ = G.NR, G.NZ
    Ng = NR * NZ
    CTRR = @. G.Jacob * D_RR / (G.dR * G.dR)
    CTRZ = @. G.Jacob * D_RZ / (G.dR * G.dZ)
    CTZZ = @. G.Jacob * D_ZZ / (G.dZ * G.dZ)
    nid = G.nodes.nid

    rows = Int[]
    cols = Int[]
    vals = FT[]
    sizehint!(rows, 9 * Ng)
    sizehint!(cols, 9 * Ng)
    sizehint!(vals, 9 * Ng)

    @inbounds for j in 1:NZ, i in 1:NR
        is_in_wall(G, i, j) || continue
        row = nid[i, j]
        invJ = one(FT) / G.Jacob[i, j]
        diag = zero(FT)

        # ── cardinal arms: the five-point part ──────────────────────────────
        for (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1))
            ii, jj = i + di, j + dj
            is_in_wall(G, ii, jj) || continue     # reflective: omit, do not zero
            CT = dj == 0 ? CTRR : CTZZ
            c = invJ * FT(0.5) * (CT[ii, jj] + CT[i, j])
            push!(rows, row)
            push!(cols, nid[ii, jj])
            push!(vals, c)
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
                    push!(rows, row)
                    push!(cols, nid[pi_, pj_])
                    push!(vals, c)
                    push!(rows, row)
                    push!(cols, nid[mi_, mj_])
                    push!(vals, -c)
                else
                    if in_p
                        push!(rows, row)
                        push!(cols, nid[pi_, pj_])
                        push!(vals, c)
                    else
                        diag += c
                    end
                    if in_m
                        push!(rows, row)
                        push!(cols, nid[mi_, mj_])
                        push!(vals, -c)
                    else
                        diag -= c
                    end
                end
            end
        end

        # ── the Robin debit ────────────────────────────────────────────────
        # v_absorb ≥ 0, so this only makes the diagonal more negative: the wall
        # drains and never sources, and the term cannot cost positivity.
        push!(rows, row)
        push!(cols, row)
        push!(vals, diag - debit[row])
    end

    return sparse(rows, cols, vals, Ng, Ng)
end
