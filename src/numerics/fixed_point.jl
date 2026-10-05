# A fixed point x = g(x): the solve that drives a stepper, and Newton's step for an affine g.

using LinearAlgebra

"""
    fixed_point_step!(stepper, x, f; valid = true) -> (x_next, status)

The iterate after `x`, whose residual is f = g(x) − x. `valid = false` marks an evaluation
that is not finite somewhere `f` does not see; it counts as a residual that is not finite.
`status` says what became of `x`:
- `:best`: finite, and the best so far;
- `:ok`: finite;
- `:stalled`: finite, but no better than the best, and no later step would be; `x_next` is `x`;
- `:restart`: not finite, or grown past what the stepper allows; `x_next` restarts from the best;
- `:exhausted`: as `:restart`, with no restart left; `x_next` is the best iterate;
- `:failed`: not finite, with no finite iterate before it; `x_next` is `x`.

The steppers are [`AndersonMixer`](@ref) and [`NewtonStepper`](@ref).
"""
function fixed_point_step! end

"""
    fixed_point_solve!(evaluate!, stepper, x; max_iter, keep!, restore!) -> (iter, outcome)

Iterate from `x` with [`fixed_point_step!`](@ref) until an evaluation converges.
`evaluate!(x)` evaluates the map at `x` into the caller's state and returns
`(f, converged, valid)`: the residual, the caller's stopping test, and whether all of the
evaluation is finite. `keep!()` saves the state of each new best evaluation, and `restore!()`
brings the best back.

`outcome` is
- `:converged`: the last evaluation converged and is finite (`:best`, `:ok` or `:stalled`);
  one the stepper rejects is never accepted, whatever its stopping test says;
- `:stopped`: `max_iter` evaluations, or the stepper can do no better; the state is the best
  evaluation;
- `:failed`: the first evaluation is not finite.
"""
function fixed_point_solve!(evaluate!::E, stepper, x::AbstractVector; max_iter::Integer, keep!::K, restore!::R) where {E, K, R}
    iter = 0
    while true
        iter += 1
        f, converged, valid = evaluate!(x)
        x_next, status = fixed_point_step!(stepper, x, f; valid)
        status === :failed && return (iter, :failed)
        status === :best && keep!()
        converged && (status === :best || status === :ok || status === :stalled) && return (iter, :converged)
        if iter >= max_iter || status === :exhausted || status === :stalled
            restore!()
            return (iter, :stopped)
        end
        x = x_next
    end
    return
end

"""
    NewtonStepper(K, W)

Newton's step for x = g(x), x + (𝟙 − T)⁻¹ f, with T the Jacobian of g and `K` a factorization
of W (𝟙 − T) W⁻¹. W = Diagonal(`W`) puts the unknowns in one unit, which keeps that matrix well
conditioned when they come in different units. On an affine map g(x) = T x + c the first step
lands on the fixed point, and later ones only remove rounding; a residual ‖W f‖ no smaller
than the best stalls.
"""
mutable struct NewtonStepper{FT <: AbstractFloat, KT}
    K::KT
    W::Vector{FT}
    best_r::FT
end
NewtonStepper(K, W::AbstractVector{FT}) where {FT <: AbstractFloat} = NewtonStepper{FT, typeof(K)}(K, Vector{FT}(W), FT(Inf))

function fixed_point_step!(S::NewtonStepper{FT}, x::AbstractVector, f::AbstractVector; valid::Bool = true) where {FT}
    r = norm(S.W .* f)
    valid && isfinite(r) && all(isfinite, x) || return (Vector{FT}(x), isinf(S.best_r) ? :failed : :exhausted)
    r < S.best_r || return (Vector{FT}(x), :stalled)
    S.best_r = r
    return (x .+ (S.K \ (S.W .* f)) ./ S.W, :best)
end
