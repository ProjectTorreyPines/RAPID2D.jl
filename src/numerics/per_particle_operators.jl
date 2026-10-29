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
    conduction_excess_operator!(A, G, Δ, b, w, discretization; n_total = w, n_floor, n = n_total) -> A

`∇·(w Δ b̂b̂ ∇·)` into `A`, an operator on the wall pattern: what a variable carried by the density `w`
conducts (and diffuses momentum) with beyond what its particles diffuse with, `Δ = D_cond − D_part` along
`b = (b_R, b_Z)`, the density inside the coefficient. Assembled by `discretization` like every
anisotropic operator, its limiter reading `n`, on the nodes whose `n_total` is above `n_floor`: the floor
is a wall for it, every face and pair reaching a floor node dropped on both sides. Grids are `(NR, NZ)`.
"""
function conduction_excess_operator!(
        A::DiscretizedOperator{FT}, G::GridGeometry{FT}, Δ::AbstractMatrix{FT}, b, w::AbstractMatrix{FT},
        discretization::AnisotropicDiscretization; n_total::AbstractMatrix{FT} = w, n_floor::FT,
        n::AbstractVecOrMat{FT} = n_total,
    ) where {FT <: AbstractFloat}
    bR, bZ = b
    @with_pool pool begin
        active = acquire!(pool, Bool, G.NR * G.NZ)
        @inbounds for k in eachindex(active)
            active[k] = n_total[k] > n_floor
        end
        χ = acquire!(pool, FT, size(Δ)...)
        @. χ = w * Δ
        RR, RZ, ZZ = acquire!(pool, FT, size(Δ)...), acquire!(pool, FT, size(Δ)...), acquire!(pool, FT, size(Δ)...)
        @. RR = χ * bR^2
        @. RZ = χ * bR * bZ
        @. ZZ = χ * bZ^2
        assemble_anisotropic_diffusion!(A, G, RR, RZ, ZZ, discretization; n, active_nodes = active)
        nothing
    end
    return A
end

"""
    add_conduction_excess!(M, A_excess, n_total; n_floor) -> M

Add to `M`, the per-particle operator of the particles' diffusion, `C = N_total⁻¹ A_excess`: the conduction
excess ([`conduction_excess_operator!`](@ref)) over the carriers' total density, off the diagonal, with the
zero row sum on it. Each face flux of `A_excess` is shared by its two rows, so `Σ J n_total f` is kept and
constants stay in the kernel; at uniform density `M + C` is the conduction operator of the whole tensor,
face by face (the same cross terms that cancel the diffusion across `b`). `A_excess` must have been assembled
on the nodes above the floor of the same `n_total`; rows at or below it stay as `M` has them. Both on one wall
pattern, `M` not aliasing `A_excess`; nothing is allocated.
"""
function add_conduction_excess!(
        M::DiscretizedOperator{FT}, A_excess::DiscretizedOperator{FT}, n_total::AbstractVecOrMat{FT}; n_floor::FT,
    ) where {FT <: AbstractFloat}
    check_wall_pattern(M)
    check_wall_pattern(A_excess)
    U, X = M.matrix, A_excess.matrix
    (U.colptr == X.colptr && U.rowval == X.rowval) ||
        throw(ArgumentError("add_conduction_excess!: M and A_excess do not share a pattern"))
    nonzeros(U) !== nonzeros(X) || throw(ArgumentError("add_conduction_excess!: M must not alias A_excess"))
    length(n_total) == size(U, 1) ||
        throw(DimensionMismatch("add_conduction_excess!: the total density must have one entry per node"))
    nzU, nzX, rows, k2c = nonzeros(U), nonzeros(X), rowvals(U), M.k2csc
    @inbounds for l in axes(U, 2), s in nzrange(U, l)
        k = rows[s]
        (k == l || !(n_total[k] > n_floor)) && continue
        _add_pair_transfer!(nzU, s, slot_position(k2c, k, SLOT_C), nzX[s] / n_total[k])
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
