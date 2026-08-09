"""
Exponential Rosenbrock–Euler coefficients for the local (diagonal) rates.

The scheme this file supports advances a term whose local Jacobian `λ = ∂f/∂y` is
known as

```math
y^{n+1} = y^{n} + \\frac{\\Delta t\\, f(y^{n})}{B(\\lambda\\Delta t)},
\\qquad B(z) = \\frac{z}{e^{z}-1}
```

— forward Euler with `B` as a divisor on the increment. It is
`y + hφ₁(hJ)f(y)` with `φ₁(z) = (eᶻ−1)/z = 1/B(z)`, i.e. exponential
Rosenbrock–Euler (`exprb2`): second order, L-stable when stiff, and **exact** for
the frozen-coefficient problem `dy/dt = λy + S` at every `Δt`, growth included.

`B` is the Bernoulli function of Scharfetter–Gummel, and the same coefficients
read as a θ-scheme give the exponentially-fitted weight
`θ(z) = 1/z − 1/(eᶻ−1)`. **That weight is never formed here** — building the
diagonal as `1 − θz` is a subtraction that cancels as `θz → 1`, ~1 % wrong by
`z = 36` and gone entirely past it. The identities used instead are

```math
1-\\theta(z)z = B(z), \\qquad 1+\\bigl(1-\\theta(z)\\bigr)z = B(-z) = B(z) + z,
\\qquad \\frac{B(-z)}{B(z)} = e^{z}.
```

Derivation, measurements and scope: `internal/docs/src/notes/design/implicit-atomic-power.md`
§3.6–3.7. What is built and how it is gated: `exprb-implementation.md`.
"""

"""
    EXPRB_Z_MAX

Upper cap on `z = λΔt` before it reaches [`bernoulli_B`](@ref).

Only *growth* needs capping. `expm1` overflows at `z ≈ 709` (`Float64`) and
`z ≈ 88` (`Float32`), and `B` then returns exactly `0.0` — the diagonal is
annihilated and the row is left to the transport operator alone. The cap sits far
below that, where conditioning starts to matter rather than where arithmetic
fails: the spread across a mixed-sign grid goes like `|z_decay|·e^(z_growth)/z_growth`,
so `z_growth ≈ 16` already puts `κ` near `5e7`. A cell whose temperature grows by
`e³⁰` inside one step is not a step anyone should be taking.

Decay is never capped — `B(z) → |z|` as `z → −∞` is bounded, benign, and exactly
the "this cell fully relaxed inside the step" limit the scheme exists to capture.
"""
const EXPRB_Z_MAX = 30

"""
    cap_exprb_z(z)

Clamp `z = λΔt` from above at [`EXPRB_Z_MAX`](@ref), preserving the type of `z`.
See that constant for why only the growth side is capped.
"""
@inline cap_exprb_z(z::T) where {T <: AbstractFloat} = min(z, T(EXPRB_Z_MAX))

"""
    bernoulli_B(z)

`B(z) = z/(eᶻ − 1)`, with `B(0) = 1`.

Strictly positive and finite for every finite `z`. That is the property the
assembled matrix depends on: with `B` on the diagonal the system stays an
M-matrix at any `Δt`, whereas a θ-scheme's `1 − θz` changes sign on a growth cell
once `θz > 1` and no amount of rescaling repairs it.

`B(0) = 1` is what makes falling back to forward Euler exact rather than
approximate — a term handed `z = 0` contributes nothing.

No series expansion near zero. `expm1` exists to keep `eᶻ − 1` accurate where it
cancels, so `z/expm1(z)` inherits that: measured max relative error **1.9e-16**
over `z ∈ ±[1e-18, 200]` against `BigFloat`. A truncated series would be worse.
`z` is expected to have passed through [`cap_exprb_z`](@ref).
"""
@inline bernoulli_B(z::T) where {T <: AbstractFloat} = iszero(z) ? one(T) : z / expm1(z)

"""
    _warn_if_z_capped(z) -> z

Warn once if any entry of an already-capped `z` sits at [`EXPRB_Z_MAX`](@ref).

The cap keeps the arithmetic sound, but a cell that wanted `z > 30` is asking to
grow by more than `e³⁰` in one step, and at that point the *step* is the problem,
not the coefficient. Silence would let a run report a plausible number built on a
step nothing resolves.
"""
function _warn_if_z_capped(z::AbstractArray{T}) where {T <: AbstractFloat}
    n = count(==(T(EXPRB_Z_MAX)), z)
    n > 0 && @warn "ExpRB: z = λΔt capped at $EXPRB_Z_MAX on $n cell(s). Those cells " *
        "would grow by more than e^$EXPRB_Z_MAX in one step — the step is the " *
        "problem, not the coefficient. Reduce Δt." maxlog = 1
    return z
end

export bernoulli_B, cap_exprb_z, EXPRB_Z_MAX
