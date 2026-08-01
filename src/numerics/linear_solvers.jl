# Cached-factorization linear solvers for θ-scheme transport/continuity solves.
#
# Contract: `factorize!(s, A)` whenever A's VALUES change (the sparsity pattern
# must stay fixed to keep the symbolic-reuse path valid), then `solve!(X, s, B)`
# with a vector or matrix RHS. Multi-RHS batch: the factorization is paid once,
# each extra RHS adds only a backsolve.
#
# Design record: Burnthrough0D claudedocs/transport2d_results.md §(b′) — with
# BLAS pinned to 1 thread (done at RAPID2D module load), `lu!` reuse beats the
# per-solve `\` path by ~25% and an Ng×6 batch costs ~1.2× a single solve.
# BandedLUSolver wins only on small grids (≲64²); it is kept as a per-problem
# selectable alternative with a residual-checked fallback.

abstract type AbstractLinearSolver{FT <: AbstractFloat} end

"""
    SparseLUSolver{FT}()

Sparse LU through the stdlib generic `LinearAlgebra.lu`/`lu!` (current engine:
the SuiteSparse UMFPACK build bundled with Julia). `factorize!` reuses the
symbolic analysis via `lu!` once a factorization exists.
"""
mutable struct SparseLUSolver{FT} <: AbstractLinearSolver{FT}
    F::Any   # UmfpackLU after first factorize!; `Any` avoids naming SparseArrays internals
end
SparseLUSolver{FT}() where {FT <: AbstractFloat} = SparseLUSolver{FT}(nothing)
SparseLUSolver() = SparseLUSolver{Float64}()

function factorize!(s::SparseLUSolver{FT}, A::SparseMatrixCSC{FT}; reuse::Bool = true) where {FT}
    (s.F === nothing || !reuse) ? (s.F = lu(A)) : lu!(s.F, A)
    return s
end

function solve!(X::AbstractVecOrMat{FT}, s::SparseLUSolver{FT}, B::AbstractVecOrMat{FT}) where {FT}
    ldiv!(X, s.F, B)
    return X
end

"""
    BandedLUSolver(A::SparseMatrixCSC; resid_tol=1e-10)

Hand-rolled no-pivot banded LU with SIMD batch backsolve (band storage
`AB[p+1+i-j, j] = A[i,j]`, precomputed inverse diagonal). No pivoting is backed
by the diagonal dominance of θ-matrices `I − θΔt·L`; strong anisotropy can break
strict dominance (measured harmless — residuals ≤1e-13 at margin −3e3, see
transport2d_results.md §(a)), so every `solve!` verifies the residual and falls
back to the internal `SparseLUSolver` when it exceeds `resid_tol`.

Bandwidth `p` is fixed from A's pattern at construction; `factorize!` accepts
any same-size matrix whose bandwidth stays ≤ p.
"""
mutable struct BandedLUSolver{FT} <: AbstractLinearSolver{FT}
    p::Int                              # bandwidth
    n::Int
    AB::Matrix{FT}                      # (2p+1, n) band store; row p+1 = diagonal
    invd::Vector{FT}                    # inverse of the U diagonal
    Xt::Matrix{FT}                      # (m, n) transposed-RHS scratch (batch axis contiguous)
    resid_tol::FT
    A_ref::SparseMatrixCSC{FT, Int}
    fallback::SparseLUSolver{FT}
end

function BandedLUSolver(A::SparseMatrixCSC{FT, Int}; resid_tol::Real = 1e-10) where {FT}
    n = size(A, 1)
    rows = rowvals(A)
    p = 0
    for j in 1:n, k in nzrange(A, j)
        p = max(p, abs(rows[k] - j))
    end
    return BandedLUSolver{FT}(p, n, zeros(FT, 2p + 1, n), zeros(FT, n),
                              zeros(FT, 0, 0), FT(resid_tol), A, SparseLUSolver{FT}())
end

function factorize!(s::BandedLUSolver{FT}, A::SparseMatrixCSC{FT}; reuse::Bool = true) where {FT}
    size(A, 1) == s.n || throw(DimensionMismatch("matrix size changed since construction"))
    AB, p, n = s.AB, s.p, s.n
    c = p + 1
    fill!(AB, zero(FT))
    rows, vals = rowvals(A), nonzeros(A)
    @inbounds for j in 1:n, k in nzrange(A, j)
        d = rows[k] - j
        abs(d) <= p || throw(ArgumentError("bandwidth grew beyond the construction pattern"))
        AB[c + d, j] = vals[k]
    end
    invd = s.invd
    @inbounds for k in 1:n
        kend = min(k + p, n)
        dinv = inv(AB[c, k])
        invd[k] = dinv
        for i in k+1:kend
            AB[c + i - k, k] *= dinv               # multiplier l_ik stored in place
        end
        for j in k+1:kend
            akj = AB[c + k - j, j]
            iszero(akj) && continue
            @simd for i in k+1:kend
                AB[c + i - j, j] = muladd(-AB[c + i - k, k], akj, AB[c + i - j, j])
            end
        end
    end
    s.A_ref = A
    return s
end

function solve!(X::AbstractMatrix{FT}, s::BandedLUSolver{FT}, B::AbstractMatrix{FT}) where {FT}
    n, m = size(B)
    n == s.n || throw(DimensionMismatch("RHS rows ≠ matrix size"))
    p, c, AB, invd = s.p, s.p + 1, s.AB, s.invd
    size(s.Xt) == (m, n) || (s.Xt = Matrix{FT}(undef, m, n))
    Xt = s.Xt
    @inbounds for i in 1:n, r in 1:m
        Xt[r, i] = B[i, r]
    end
    @inbounds for k in 1:n-1                       # L sweep (unit lower, column-wise)
        kend = min(k + p, n)
        for i in k+1:kend
            l = AB[c + i - k, k]
            iszero(l) && continue
            @simd for r in 1:m
                Xt[r, i] = muladd(-l, Xt[r, k], Xt[r, i])
            end
        end
    end
    @inbounds for j in n:-1:1                      # U sweep (finalize col j, push up)
        dj = invd[j]
        @simd for r in 1:m
            Xt[r, j] *= dj
        end
        for k in max(1, j - p):j-1
            u = AB[c + k - j, j]
            iszero(u) && continue
            @simd for r in 1:m
                Xt[r, k] = muladd(-u, Xt[r, j], Xt[r, k])
            end
        end
    end
    @inbounds for i in 1:n, r in 1:m
        X[i, r] = Xt[r, i]
    end
    # no-pivot safety net (one spmv per solve): fall back to SparseLU on excess residual
    resid = norm(s.A_ref * X .- B) / max(norm(B), eps(FT))
    resid > s.resid_tol && solve!(X, factorize!(s.fallback, s.A_ref), B)
    return X
end

function solve!(x::AbstractVector{FT}, s::BandedLUSolver{FT}, b::AbstractVector{FT}) where {FT}
    solve!(reshape(x, :, 1), s, reshape(b, :, 1))
    return x
end
