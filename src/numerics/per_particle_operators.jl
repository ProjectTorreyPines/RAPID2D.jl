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
of the pattern is rewritten and a stored zero of `A` stays one. The scratch `A n` comes from
the task's array pool, so nothing is allocated after the first call. `M` must not alias `A`,
and `n` must be finite.
"""
function per_particle_operator!(
        M::DiscretizedOperator{FT}, A::DiscretizedOperator{FT}, n::AbstractVector{FT};
        n_floor::FT,
    ) where {FT <: AbstractFloat}
    check_wall_pattern(M)
    check_wall_pattern(A)
    C, U = A.matrix, M.matrix
    (U !== C && nonzeros(U) !== nonzeros(C)) ||
        throw(ArgumentError("per_particle_operator!: M must not alias A"))
    (C.colptr == U.colptr && C.rowval == U.rowval) ||
        throw(ArgumentError("per_particle_operator!: M and A do not share a pattern"))
    length(n) == size(C, 2) ||
        throw(DimensionMismatch("per_particle_operator!: n must have one entry per node"))
    all(isfinite, n) || throw(ArgumentError("per_particle_operator!: n must be finite"))
    nzU, nzC, rows = nonzeros(U), nonzeros(C), rowvals(C)
    @inbounds for j in 1:size(C, 2)
        nj = n[j]
        for k in nzrange(C, j)
            i = rows[k]
            inv_ni = n[i] > n_floor ? one(FT) / n[i] : zero(FT)
            nzU[k] = inv_ni * (nzC[k] * nj)
        end
    end
    k2c = A.k2csc
    # the checks above throw; the pool block below does not, since it has no `finally`
    @with_pool pool begin
        An = acquire!(pool, FT, length(n))
        mul!(An, C, n)
        @inbounds for i in eachindex(n)
            inv_ni = n[i] > n_floor ? one(FT) / n[i] : zero(FT)
            kd = slot_position(k2c, i, SLOT_C)
            nzU[kd] = inv_ni * ((nzC[kd] * n[i]) - An[i])
        end
        nothing
    end
    return M
end

"""
    dissipation_rate!(Γ, M, u) -> Γ

`Γ_k = ½ Σ_l M_kl (u_l − u_k)²`, which for an `M` that annihilates constants equals
`½ (M u²)_k − u_k (M u)_k`: the rate at which the exchange `du/dt = M u` turns the kinetic
energy per particle into heat, so that `Σ_k V_k n_k Γ_k` is what `Σ V ½ n u²` loses under
`dn/dt = A n`. The edge sum is exactly zero at uniform `u` and non-negative when the
off-diagonals of `M` are; the 9-point cross terms can make it negative on a node. Nothing is
allocated.
"""
function dissipation_rate!(
        Γ::AbstractVector{FT}, M::DiscretizedOperator{FT}, u::AbstractVector{FT},
    ) where {FT <: AbstractFloat}
    check_wall_pattern(M)
    NR = M.dims_rz[1]
    Ng = size(M.matrix, 1)
    length(u) == Ng == length(Γ) ||
        throw(DimensionMismatch("dissipation_rate!: u and Γ must have one entry per node"))
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
