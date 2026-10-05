# Anderson mixing for a fixed point x = g(x).

using LinearAlgebra

"""
    AndersonMixer{FT}(n, m; β = ones(n), W = ones(n), growth = 1e3, max_restarts = 3)

Anderson mixing (type II) for a fixed point x = g(x) of `n` unknowns. Given an iterate x and its
residual f = g(x) − x, [`anderson_step!`](@ref) steps to

    x + B f − (ΔX + B ΔF) γ,    γ = argmin ‖W (f − ΔF γ)‖₂,

where ΔX and ΔF hold the differences of the last `m` + 1 iterates and residuals, B = Diagonal(β)
and W = Diagonal(W). With `m = 0` this is the relaxed iteration x + B f. On an affine map it is
a truncated relative of GMRES on (𝟙 − T) x = c, and it can converge where the relaxed
iteration diverges.

It keeps the iterate with the smallest weighted residual ‖W f‖. A residual that is not finite,
or larger than `growth` times that smallest one, makes it restart: the history is dropped, B
is halved, and the next iterate is the relaxed step from the best one. After `max_restarts`
restarts it stops at the best iterate.
"""
mutable struct AndersonMixer{FT <: AbstractFloat}
    m::Int
    β::Vector{FT}
    W::Vector{FT}
    growth::FT
    max_restarts::Int
    xs::Vector{Vector{FT}}   # the last m + 1 iterates
    fs::Vector{Vector{FT}}   # and their residuals
    best_x::Vector{FT}
    best_f::Vector{FT}
    best_r::FT
    nrestart::Int
end

function AndersonMixer{FT}(
        n::Integer, m::Integer; β = ones(FT, n), W = ones(FT, n), growth::Real = 1.0e3, max_restarts::Integer = 3,
    ) where {FT <: AbstractFloat}
    m >= 0 || throw(ArgumentError("the Anderson memory m must be ≥ 0, got $m"))
    (length(β) == n && length(W) == n) || throw(DimensionMismatch("β and W must have length $n"))
    return AndersonMixer{FT}(
        Int(m), FT.(β), FT.(W), FT(growth), Int(max_restarts),
        Vector{FT}[], Vector{FT}[], FT[], FT[], FT(Inf), 0,
    )
end

"""
    anderson_step!(A::AndersonMixer, x, f; valid = true) -> (x_next, status)

The iterate after `x`, whose residual is `f`. `valid = false` marks an evaluation that is not
finite somewhere `f` does not see; it counts as a residual that is not finite. `status` says
what became of `x`:
- `:best`: finite, and the best so far;
- `:ok`: finite;
- `:restart`: not finite, or grown past `growth` times the best; `x_next` restarts from the best;
- `:exhausted`: as `:restart`, with the restarts used up; `x_next` is the best iterate;
- `:failed`: not finite, with no finite iterate before it; `x_next` is `x`.
"""
function anderson_step!(A::AndersonMixer{FT}, x::AbstractVector, f::AbstractVector; valid::Bool = true) where {FT}
    r = norm(A.W .* f)
    if !(valid && isfinite(r) && all(isfinite, x)) || r > A.growth * A.best_r
        isempty(A.best_x) && return (Vector{FT}(x), :failed)
        A.nrestart >= A.max_restarts && return (copy(A.best_x), :exhausted)
        A.nrestart += 1
        A.β ./= 2
        empty!(A.xs)
        empty!(A.fs)
        return (A.best_x .+ A.β .* A.best_f, :restart)
    end
    status = :ok
    if r < A.best_r
        A.best_x, A.best_f, A.best_r = Vector{FT}(x), Vector{FT}(f), r
        status = :best
    end
    push!(A.xs, Vector{FT}(x))
    push!(A.fs, Vector{FT}(f))
    if length(A.xs) > A.m + 1
        popfirst!(A.xs)
        popfirst!(A.fs)
    end
    x_next = x .+ A.β .* f
    k = length(A.xs) - 1
    if k > 0
        ΔX = reduce(hcat, [A.xs[i + 1] .- A.xs[i] for i in 1:k])
        ΔF = reduce(hcat, [A.fs[i + 1] .- A.fs[i] for i in 1:k])
        x_next .-= (ΔX .+ A.β .* ΔF) * anderson_lstsq(A.W .* ΔF, A.W .* f)
    end
    return (x_next, status)
end

"""
    best_iterate(A::AndersonMixer) -> x

The iterate with the smallest weighted residual so far.
"""
best_iterate(A::AndersonMixer) = copy(A.best_x)

"""
    anderson_solve!(evaluate!, A::AndersonMixer, x; max_iter, keep!, restore!) -> (iter, outcome)

Iterate from `x` with [`anderson_step!`](@ref) until an evaluation converges. `evaluate!(x)`
evaluates the map at `x` into the caller's state and returns `(f, converged, valid)`: the
residual, the caller's stopping test, and whether all of the evaluation is finite. `keep!()`
saves the state of each new best evaluation, and `restore!()` brings the best back.

`outcome` is
- `:converged`: the last evaluation converged, and the mixer kept it (`:best` or `:ok`); one it
  rejects is never accepted, whatever its stopping test says;
- `:stopped`: `max_iter` evaluations, or the restarts used up; the state is the best evaluation;
- `:failed`: the first evaluation is not finite.
"""
function anderson_solve!(evaluate!::E, A::AndersonMixer, x::AbstractVector; max_iter::Integer, keep!::K, restore!::R) where {E, K, R}
    iter = 0
    while true
        iter += 1
        f, converged, valid = evaluate!(x)
        x_next, status = anderson_step!(A, x, f; valid)
        status === :failed && return (iter, :failed)
        status === :best && keep!()
        converged && (status === :best || status === :ok) && return (iter, :converged)
        if iter >= max_iter || status === :exhausted
            restore!()
            return (iter, :stopped)
        end
        x = x_next
    end
    return
end

# argmin ‖M γ − b‖₂ by pivoted QR, the columns whose pivot falls under max(1e-10, n ε) of the
# largest left out (γ = 0 there): repeated or dependent differences give no direction. n ε is
# round-off; 1e-10 bounds the fit's conditioning in double precision.
function anderson_lstsq(M::AbstractMatrix{FT}, b::AbstractVector{FT}) where {FT}
    γ = zeros(FT, size(M, 2))
    F = qr(M, ColumnNorm())
    d = abs.(diag(F.R))
    (isempty(d) || !(d[1] > 0)) && return γ
    r = count(>(max(FT(1.0e-10), maximum(size(M)) * eps(FT)) * d[1]), d)
    y = (F.Q' * b)[1:r]
    γ[F.p[1:r]] .= UpperTriangular(F.R[1:r, 1:r]) \ y
    return γ
end
