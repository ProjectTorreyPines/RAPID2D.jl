# Exponential Rosenbrock–Euler coefficients for the local (diagonal) rates.
#
#     yⁿ⁺¹ = yⁿ + Δt·f(yⁿ)/B(λΔt),    B(z) = z/(eᶻ − 1)
#
# — forward Euler with `B` as a divisor on the increment, i.e. `y + hφ₁(hJ)f(y)`
# with `φ₁ = 1/B` (`exprb2`): second order, L-stable when stiff, and exact for
# `dy/dt = λy + S` at every `Δt`, growth included.
#
# The assembled form uses `B(z)` and `B(-z) = B(z) + z` directly and never forms
# the equivalent θ; see `exprb_theta` for why.
#
# A comment, not a docstring: Julia binds a string to the expression that follows
# it, and the next expression here is another string — so as a `"""..."""` block
# this bound to nothing and `?` never reached it.
#
# Derivation and measurements: internal/docs/src/notes/design/implicit-atomic-power.md
# §3.6–3.7. What is built and how it is gated: exprb-implementation.md.

"""
    EXPRB_MAX_EXPONENT

Upper cap on the exponent `z = λΔt`, growth side only.

`expm1` overflows at `z ≈ 709` (`Float64`) / `88` (`Float32`) and `B` then returns
exactly `0.0`, annihilating the diagonal. The cap sits far below that, where
conditioning starts to cost digits: the spread across a mixed-sign grid goes like
`|z₋|·e^(z₊)/z₊` over the negative and positive entries, so `z₊ ≈ 16` already puts
`κ` near `5e7`.

Decay is never capped — `B(z) → |z|` as `z → −∞` is bounded and is exactly the
"this cell fully relaxed inside the step" limit the scheme exists to capture.
"""
const EXPRB_MAX_EXPONENT = 30

"""
    exprb_cap_exponent(z)

Clamp `z = λΔt` from above at [`EXPRB_MAX_EXPONENT`](@ref), preserving the type of `z`.
"""
@inline exprb_cap_exponent(z::T) where {T <: AbstractFloat} = min(z, T(EXPRB_MAX_EXPONENT))

"""
    exprb_bern(z)

`B(z) = z/(eᶻ − 1)`, with `B(0) = 1`.

**Strictly positive and finite for every finite `z`** — that is what keeps the
assembled matrix an M-matrix at any `Δt`, where a θ-scheme's `1 − θz` changes sign
on a growth cell once `θz > 1`. `B(0) = 1` makes the forward-Euler fallback exact
rather than approximate.

No series expansion near zero: `expm1` already keeps `eᶻ − 1` accurate where it
cancels, measured max relative error **1.9e-16** over `z ∈ ±[1e-18, 200]` against
`BigFloat`. Expects `z` to have passed through [`exprb_cap_exponent`](@ref).
"""
@inline exprb_bern(z::T) where {T <: AbstractFloat} = iszero(z) ? one(T) : z / expm1(z)

"""
    exprb_theta(z)

The θ whose amplification matches `ExpRB`'s: `θ(z) = (1 − B(z))/z`, with `θ(0) = ½`.

**For ledgers only — never build the scheme from this.** The diagonal `1 − θz` is
a subtraction that cancels as `θz → 1`: ~1 % wrong by `z = 36` and gone past it.
A consumer recording `∫ … dt ≈ Δt[(1−θ)(…)ⁿ + θ(…)ⁿ⁺¹]` needs the quadrature the
step actually used, and here θ is an output rather than a divisor.

Monotone in `(0, 1)` at every `z`, with limits `θ → 1` (BE) as `z → −∞`, `½` (CN)
at `0`, `0` (FE) as `z → +∞`. Series below `|z| = 1e-4`, where `1 − B(z)` loses
digits to cancellation.
"""
@inline function exprb_theta(z::T) where {T <: AbstractFloat}
    return abs(z) < T(1.0e-4) ?
        evalpoly(z, (T(0.5), -T(1) / T(12), zero(T), T(1) / T(720))) :
        (one(T) - exprb_bern(z)) / z
end

"""
    _warn_if_exprb_capped(z) -> z

Warn once if any entry of an already-capped `z` sits at [`EXPRB_MAX_EXPONENT`](@ref) — a
cell asking to grow by more than `e³⁰` in one step means the *step* is the problem.
"""
function _warn_if_exprb_capped(z::AbstractArray{T}) where {T <: AbstractFloat}
    n = count(==(T(EXPRB_MAX_EXPONENT)), z)
    n > 0 && @warn "ExpRB: z = λΔt capped at $EXPRB_MAX_EXPONENT on $n cell(s). Those cells " *
        "would grow by more than e^$EXPRB_MAX_EXPONENT in one step — the step is the " *
        "problem, not the coefficient. Reduce Δt." maxlog = 1
    return z
end

export exprb_bern, exprb_theta, exprb_cap_exponent, EXPRB_MAX_EXPONENT
