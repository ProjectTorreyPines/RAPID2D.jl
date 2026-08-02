# Cached-factorization linear solvers for θ-scheme transport/continuity solves.
#
# Contract: `factorize!(s, A)` whenever A's VALUES change (the sparsity pattern
# must stay fixed to keep the symbolic-reuse path valid), then `solve!(X, s, B)`
# with a vector or matrix RHS. Multi-RHS batch: the factorization is paid once,
# each extra RHS adds only a backsolve.
#
# Measured on the real electron-continuity matrix (default configuration, so the
# anisotropy Dpara/Dperp ≈ 1.7e5 is the one production actually assembles),
# `factorize!` + `solve!` against the per-step `\`:
#
#   grid      n       direct    SparseLU        BandedLU
#   40×80     3200    2897 µs   1989 µs 1.46×   1201 µs 2.41×
#   64×64     4096    4277 µs   2967 µs 1.44×   3000 µs 1.43×
#   80×160    12800  15553 µs  10475 µs 1.48×  13265 µs 1.17×
#   128×128   16384  20695 µs  14025 µs 1.48×  38223 µs 0.54×
#
# SparseLUSolver holds ~1.45× at every size and reproduces `\` to exactly zero
# relative error (it is the same UMFPACK factorization). BandedLUSolver wins only
# on small grids — it ties at 64² and is 1.9× SLOWER at 128² — so it is kept as a
# per-problem selectable alternative, not the default.
#
# For an extra RHS in a batch (`solve!` alone) SparseLU is 5-11× faster
# (68/95/354/488 µs vs 358/636/2628/5371 µs): BandedLU verifies its residual on
# every solve, and a single RHS cannot amortize that sparse matvec.
#
# BLAS thread count is immaterial here — 1 vs 8 threads gives 1.46× vs 1.49×
# (40×80) and 0.50× vs 0.53× (128²), because UMFPACK on these matrices is
# dominated by sparse bookkeeping rather than dense BLAS3 kernels. Note that the
# `LinearAlgebra.BLAS.set_num_threads(1)` at RAPID2D.jl top level sits outside an
# `__init__`, so it runs at precompile time only and does not take effect at
# runtime; that is a separate issue, and the numbers above hold either way.

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
    if s.F === nothing || !reuse
        s.F = lu(A)
    else
        try
            lu!(s.F, A)
        catch err
            # Sparse broadcast drops numerical zeros, so a value turning on (e.g.
            # convection starting from u=0) can GROW the assembled pattern between
            # steps. lu! then throws "pattern of the matrix changed" — recover with
            # a fresh symbolic analysis instead of failing the step.
            err isa ArgumentError || rethrow()
            s.F = lu(A)
        end
    end
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

function BandedLUSolver(A::SparseMatrixCSC{FT, Int}; resid_tol::Real = 1.0e-10) where {FT}
    n = size(A, 1)
    rows = rowvals(A)
    p = 0
    for j in 1:n, k in nzrange(A, j)
        p = max(p, abs(rows[k] - j))
    end
    return BandedLUSolver{FT}(
        p, n, zeros(FT, 2p + 1, n), zeros(FT, n),
        zeros(FT, 0, 0), FT(resid_tol), A, SparseLUSolver{FT}()
    )
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
        for i in (k + 1):kend
            AB[c + i - k, k] *= dinv               # multiplier l_ik stored in place
        end
        for j in (k + 1):kend
            akj = AB[c + k - j, j]
            iszero(akj) && continue
            @simd for i in (k + 1):kend
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
    @inbounds for k in 1:(n - 1)                       # L sweep (unit lower, column-wise)
        kend = min(k + p, n)
        for i in (k + 1):kend
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
        for k in max(1, j - p):(j - 1)
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
