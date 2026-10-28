# The per-particle operators: what a density operator does to the variable its particles carry.
#
# A conservative operator A advances a density, dn/dt = A n: the face-flux divergence (with
# its sign) or the density diffusion ∇·D∇. A variable f carried by those particles — u∥, Te,
# the energy per electron — then obeys df/dt = M f with
#
#     M = N⁻¹ (A N − diag(A n)),        N = diag(n),
#
# the one operator whose primitive form reproduces d(n f)/dt = A (n f) on every row:
# n (M f) + f (A n) = A (n f). Advection and mixing are the same construction, so it lives here
# once; `advection_operator!` applies it to the face-flux divergence, the mixing operator to the
# diffusion. The exchange of momentum between parcels turns kinetic energy into heat at the
# dissipation rate Γ_M(u) = ½ M u² − u ⊙ M u.
# internal/docs/src/reference/electron-diffusive-transport.md §5.

"""
    per_particle_operator!(M, A, n; n_floor) -> M

`M = N⁻¹ (A N − diag(A n))` written into `M`, an operator on the same wall pattern as `A`:
off the diagonal `A_kl n_l / n_k`, on it `(A_kk n_k − (A n)_k) / n_k`. It depends on the
off-diagonals of `A` alone, so a Robin debit on `A`'s diagonal cancels: the wall drains
particles, not the variable they carry. Rows with `n_k ≤ n_floor` are empty (the variable is
frozen where there is no plasma; [`floor_defect`](@ref) measures what that costs). Every slot
of the pattern is rewritten and a stored zero of `A` stays one. `n` is the node vector or the
`(NR, NZ)` grid itself, read in node order, and must be finite; `M` must not alias `A`. The
scratch (`1/n` and `A n`) comes from the task's array pool, so nothing is allocated after the
first call.
"""
per_particle_operator!(
    M::DiscretizedOperator{FT}, A::DiscretizedOperator{FT}, n::AbstractVecOrMat{FT}; n_floor::FT,
) where {FT <: AbstractFloat} = per_particle_operator!(M, (A,), (n,), n; n_floor)

"""
    per_particle_operator!(M, As, ws, w_total; n_floor) -> M

The per-particle operator of a mixture of carriers: groups of particles moved by their own
operators `As[g]` with densities `ws[g]`, one carried variable counted over `w_total` (which may
hold particles no group moves). Off the diagonal `Σ_g A_g,kl w_g,l / w_total,k`, on it the zero row
sum, so `w_total (M f) + f Σ_g A_g w_g = Σ_g A_g (w_g f)` on every row above the floor, and a node
where one group is absent still takes what it brings in. The floor is on `w_total`. One group whose
density is the total is the single-operator form, bit for bit. All on one wall pattern, `M` aliasing
none of `As`; nothing is allocated.
"""
function per_particle_operator!(
        M::DiscretizedOperator{FT}, As, ws, w_total::AbstractVecOrMat{FT}; n_floor::FT,
    ) where {FT <: AbstractFloat}
    (!isempty(As) && length(As) == length(ws)) ||
        throw(ArgumentError("per_particle_operator!: one density per operator, at least one of each"))
    check_wall_pattern(M)
    U = M.matrix
    Ng = size(U, 2)
    for (A, w) in zip(As, ws)
        check_wall_pattern(A)
        C = A.matrix
        (U !== C && nonzeros(U) !== nonzeros(C)) ||
            throw(ArgumentError("per_particle_operator!: M must not alias A"))
        (C.colptr == U.colptr && C.rowval == U.rowval) ||
            throw(ArgumentError("per_particle_operator!: M and A do not share a pattern"))
        length(w) == Ng ||
            throw(DimensionMismatch("per_particle_operator!: each density must have one entry per node"))
        all(isfinite, w) || throw(ArgumentError("per_particle_operator!: the densities must be finite"))
    end
    length(w_total) == Ng ||
        throw(DimensionMismatch("per_particle_operator!: the total density must have one entry per node"))
    all(isfinite, w_total) || throw(ArgumentError("per_particle_operator!: the total density must be finite"))
    nzU, rows, k2c = nonzeros(U), rowvals(U), M.k2csc
    # the checks above throw; the pool block below must not, since it has no `finally`
    @with_pool pool begin
        # 1/w_k once per row, zero on the floor rows
        inv_n = acquire!(pool, FT, Ng)
        @inbounds for k in 1:Ng
            inv_n[k] = w_total[k] > n_floor ? one(FT) / w_total[k] : zero(FT)
        end
        # Σ_g A_g,kl w_g,l into every slot, and Σ_g A_g w_g accumulated in the same column sweeps,
        # with `muladd` in the order `mul!` adds, so one group gives `mul!`'s bit for bit
        fill!(nzU, zero(FT))
        An = zeros!(pool, FT, Ng)
        for (A, w) in zip(As, ws)
            nzC = nonzeros(A.matrix)
            @inbounds for l in 1:Ng
                wl = w[l]
                for s in nzrange(U, l)
                    k = rows[s]
                    An[k] = muladd(nzC[s], wl, An[k])
                    nzU[s] += nzC[s] * wl
                end
            end
        end
        # the off-diagonals over the total, and the diagonal (Σ_g A_g,kk w_g,k − (Σ_g A_g w_g)_k) / w_k
        @inbounds for l in 1:Ng, s in nzrange(U, l)
            k = rows[s]
            k == l || (nzU[s] = inv_n[k] * nzU[s])
        end
        @inbounds for k in 1:Ng
            s = slot_position(k2c, k, SLOT_C)
            nzU[s] = inv_n[k] * (nzU[s] - An[k])
        end
        nothing
    end
    return M
end

"""
    add_conduction_correction!(M, A_part, A_cond, n, G; n_floor, n_total = n) -> M

Add to `M`, the per-particle operator of the particle diffusion `A_part`, the correction `C` that
makes the carried variable conduct (and diffuse momentum) with `A_cond` instead:

    S_kl = ½ (R_k ΔA_kl + R_l ΔA_lk),   ΔA = A_cond − A_part,
    (C f)_k = Σ_l S_kl (n_k + n_l)/2 (f_l − f_k) / (R_k n_k),

with `R_k = G.Jacob[k]`. The pair transfers are symmetric, so `Σ R n f` is kept and constants
stay in the kernel (the cross terms make `R A` asymmetric, so a split of `M` by face densities
would not keep it). The negative transfers are kept, as `A_part` and `A_cond` keep theirs: on an
oblique field they are what cancels the diffusion across `b`, so a profile constant along `b`
stays in the kernel to truncation order (dropping them conducted ≈ ½ΔD across the field at any
resolution). Where the density is not resolved, `limit_antidiffusive_pairs!` has bounded them in
both operators. A pair with a node at or below `n_floor` is skipped, its row being frozen. In a
uniform plasma with `R ΔA` symmetric (no cross terms) `M + C` is `A_cond`; `A_cond = A_part`
leaves `M` bit for bit.
For one group of a mixture (`per_particle_operator!` over several carriers), `n` is the group's
density and `n_total` the carriers' total: the transfers are the group's, `S (n_k + n_l)/2`, over
`R_k n_total,k`, and the floor is on the total, so `Σ R n_total f` is kept.
All three on the wall pattern of `G`, `M` aliasing neither; nothing is allocated.
"""
function add_conduction_correction!(
        M::DiscretizedOperator{FT}, A_part::DiscretizedOperator{FT}, A_cond::DiscretizedOperator{FT},
        n::AbstractVecOrMat{FT}, G::GridGeometry{FT}; n_floor::FT, n_total::AbstractVecOrMat{FT} = n,
    ) where {FT <: AbstractFloat}
    foreach(check_wall_pattern, (M, A_part, A_cond))
    P, Q, U = A_part.matrix, A_cond.matrix, M.matrix
    (P.colptr == Q.colptr == U.colptr && P.rowval == Q.rowval == U.rowval) ||
        throw(ArgumentError("add_conduction_correction!: M, A_part and A_cond do not share a pattern"))
    (nonzeros(U) !== nonzeros(P) && nonzeros(U) !== nonzeros(Q)) ||
        throw(ArgumentError("add_conduction_correction!: M must not alias A_part or A_cond"))
    (M.dims_rz == (G.NR, G.NZ) && length(n) == size(U, 1) == length(n_total)) ||
        throw(DimensionMismatch("add_conduction_correction!: M and n must be on the grid of G"))
    nzU, nzP, nzQ, k2c = nonzeros(U), nonzeros(P), nonzeros(Q), M.k2csc
    NR, R = G.NR, G.Jacob
    half = FT(0.5)
    @inbounds for k in 1:size(U, 1), (slot, slot_back) in PAIR_SLOTS
        p_kl = slot_position(k2c, k, slot)
        p_kl == 0 && continue
        di, dj = STENCIL_OFFSETS[slot]
        l = k + di + dj * NR
        (n_total[k] > n_floor && n_total[l] > n_floor) || continue
        p_lk = slot_position(k2c, l, slot_back)     # the pattern is structurally symmetric
        S = half * (R[k] * (nzQ[p_kl] - nzP[p_kl]) + R[l] * (nzQ[p_lk] - nzP[p_lk]))
        iszero(S) && continue
        n_mean = half * (n[k] + n[l])
        _add_pair_transfer!(nzU, p_kl, slot_position(k2c, k, SLOT_C), S * n_mean / (R[k] * n_total[k]))
        _add_pair_transfer!(nzU, p_lk, slot_position(k2c, l, SLOT_C), S * n_mean / (R[l] * n_total[l]))
    end
    return M
end

# add `value` to the off-diagonal at `p_offdiag`, and keep the row sum through the diagonal at `p_diag`
@inline function _add_pair_transfer!(nz, p_offdiag, p_diag, value)
    @inbounds nz[p_offdiag] += value
    @inbounds nz[p_diag] -= value
    return nothing
end

"""
    dissipation_rate!(Γ, M, u) -> Γ

`Γ_k = ½ Σ_l M_kl (u_l − u_k)²`, which for an `M` that annihilates constants equals
`½ (M u²)_k − u_k (M u)_k`: the rate at which the exchange `du/dt = M u` turns the kinetic
energy per particle into heat, so that `Σ_k V_k n_k Γ_k` is what `Σ V ½ n u²` loses under
`dn/dt = A n`. The edge sum is exactly zero at uniform `u` and non-negative when the
off-diagonals of `M` are; the 9-point cross terms can make it negative on a node. `Γ` and `u`
are node vectors or `(NR, NZ)` grids, in node order, and must not overlap: each row reads
its neighbours' `u`. Nothing is allocated.
"""
function dissipation_rate!(
        Γ::AbstractVecOrMat{FT}, M::DiscretizedOperator{FT}, u::AbstractVecOrMat{FT},
    ) where {FT <: AbstractFloat}
    check_wall_pattern(M)
    NR = M.dims_rz[1]
    Ng = size(M.matrix, 1)
    length(u) == Ng == length(Γ) ||
        throw(DimensionMismatch("dissipation_rate!: u and Γ must have one entry per node"))
    Base.mightalias(Γ, u) && throw(ArgumentError("dissipation_rate!: Γ must not overlap u"))
    nz, k2c = nonzeros(M.matrix), M.k2csc
    half = FT(0.5)
    @inbounds for r in 1:Ng
        ur = u[r]
        s = zero(FT)
        for slot in 2:9
            p = slot_position(k2c, r, slot)
            p == 0 && continue
            di, dj = STENCIL_OFFSETS[slot]
            d = u[r + di + dj * NR] - ur
            s += nz[p] * d * d
        end
        Γ[r] = half * s
    end
    return Γ
end

"""
    floor_defect(A, n, f, V; n_floor) -> FT

`Σ_{k: n_k ≤ n_floor} V_k [(A (n f))_k − f_k (A n)_k]`: the amount of `V n f` the per-particle
form does not conserve, because on a floor node `f` is frozen while `A` still moves particles
in. Zero when every node `A` reaches is above the floor. A diagnostic, not a correction.
"""
function floor_defect(
        A::DiscretizedOperator{FT}, n::AbstractVector{FT}, f::AbstractVector{FT}, V::AbstractVector{FT};
        n_floor::FT,
    ) where {FT <: AbstractFloat}
    Anf = A * (n .* f)
    An = A * n
    defect = zero(FT)
    for k in eachindex(n)
        n[k] <= n_floor || continue
        defect += V[k] * (Anf[k] - f[k] * An[k])
    end
    return defect
end
