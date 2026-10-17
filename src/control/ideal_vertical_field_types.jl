# The vertical-field control a run applies: an actuator model, not a coil. The flag
# `vertical_field_control` holds the policy, `RP.vertical_field_control` its state, and
# `control/ideal_vertical_field.jl` the update. The ideal control is composed of four choices,
# each a policy type: what force it cancels (`target`), with which vacuum field (`field`), how the
# field's flux change reaches the plasma (`induction`), and how fast it follows (`response`).

"""
    VerticalFieldControl

How a run controls the column's position with an external vertical field:
[`NoVerticalFieldControl`](@ref) (the default) or [`IdealVerticalFieldControl`](@ref).
"""
abstract type VerticalFieldControl end

"""
    NoVerticalFieldControl()

No control field: the run's code path is the one without control.
"""
struct NoVerticalFieldControl <: VerticalFieldControl end

"""
    VerticalFieldTarget

The force the ideal control cancels, computed with the control field left out:
- `MagneticForceTarget()` (default): the forces of the external, coil and plasma fields on Jϕ,
  the MATLAB-level target. For a one-signed current the uniform field that cancels it is
  −⟨B_Z⟩, the |Jϕ|dV-weighted mean of the field without control.
- `ReducedForceTarget()`: those, plus the pressure force on the column's moment ellipse and the
  toroidal closure: the reduced force of `column_force_balance`, whose `Bv_to_add` it cancels.
"""
abstract type VerticalFieldTarget end
struct MagneticForceTarget <: VerticalFieldTarget end
struct ReducedForceTarget <: VerticalFieldTarget end

"""
    VerticalControlField

The vacuum field the ideal control applies:
- `UniformVerticalField()` (default): a₁ ψ₁ with ψ₁ = R²/2, a uniform B_Z = a₁; it cancels the
  radial force.
- `VerticalAndRadialField()`: also a₂ ψ₂ with ψ₂ = R²Z/R_ref (B_Z = 2a₂Z/R_ref, B_R = −a₂R/R_ref;
  a₂ the −B_R at R_ref, the grid's middle R), solved jointly to cancel the vertical force too.
"""
abstract type VerticalControlField end
struct UniformVerticalField <: VerticalControlField end
struct VerticalAndRadialField <: VerticalControlField end

"""
    ControlInduction

How the control field's flux change reaches the plasma:
- `FaradayInduction()` (default): the vacuum flux as it is, E_ϕ = −Δψ_ctrl/(R Δt).
- `CSCompensatedInduction()`: plus a spatially constant flux, the field of a central solenoid
  outside its bore, that holds the control loop voltage at zero at the current centroid of the
  first turn-on (a fixed point; elsewhere the R² gradient of the vertical field's flux remains).
  B is unchanged.
"""
abstract type ControlInduction end
struct FaradayInduction <: ControlInduction end
struct CSCompensatedInduction <: ControlInduction end

"""
    ControlResponse

How the applied coefficients follow the target:
- `FirstOrderResponse(τ_s)` (default τ_s = 1e-4 s): a ← a + α (aim − a), α = 1 − exp(−Δt/τ),
  exact for a fixed aim at any Δt; a finite bandwidth, so a rising target is followed with a lag.
- `InstantaneousResponse()`: the coefficients are the (ramped, clipped) target of the step.
"""
abstract type ControlResponse end
struct FirstOrderResponse{FT <: AbstractFloat} <: ControlResponse
    τ_s::FT
    function FirstOrderResponse{FT}(τ_s::Real) where {FT <: AbstractFloat}
        (isfinite(τ_s) && τ_s > 0) || throw(ArgumentError("τ_s must be finite and positive, got $τ_s"))
        return new{FT}(τ_s)
    end
end
FirstOrderResponse(τ_s::Real) = FirstOrderResponse{Float64}(τ_s)
struct InstantaneousResponse <: ControlResponse end

"""
    IdealVerticalFieldControl{FT}(; t_on_s, target = MagneticForceTarget(),
        field = UniformVerticalField(), induction = FaradayInduction(),
        response = FirstOrderResponse(1e-4), ramp_s = 2e-3, I_on = 100, I_off = 50,
        coherence_on = 0.9, coherence_off = 0.8, coefficient_max = 5e-3, cond_max = 1e4)

An ideal full-state actuator: each step it computes, from the whole present state, the vacuum
field that cancels the column's force estimate (`target`) with the control field left out, and
applies it through `field`, `induction` and `response`. Quasi-equilibrium is assumed, not solved:
the actuator supplies the field the column needs wherever transport has put it. It has no
sensors, no model error and no coils; a PID controller or real coils are separate models.

The defaults are the simplest choices. For the KSTAR scenario question (does the closed region
persist and |I_p| keep rising under the scenario's loop voltage), the reference is
`target = ReducedForceTarget(), induction = CSCompensatedInduction()`.

- **Timing:** off before `t_on_s`; on while the column qualifies (|I| ≥ `I_on` and
  `current_coherence` ≥ `coherence_on` to turn on, below `I_off` or `coherence_off` to turn off,
  a net ∫Jϕ dV and, with `VerticalAndRadialField`, a system with condition number ≤ `cond_max`
  either way). Each turn-on ramps the target up over `ramp_s`; a coefficient beyond
  `coefficient_max` [T] is clipped (saturated); turned off, the coefficients relax to zero.

Refused with `Global_JxB_Force`, without Ampère, and in a run with coils (no circuit couples to
the control flux).
"""
struct IdealVerticalFieldControl{FT <: AbstractFloat, T <: VerticalFieldTarget, F <: VerticalControlField, I <: ControlInduction, R <: ControlResponse} <: VerticalFieldControl
    target::T
    field::F
    induction::I
    response::R
    t_on_s::FT
    ramp_s::FT
    I_on::FT
    I_off::FT
    coherence_on::FT
    coherence_off::FT
    coefficient_max::FT
    cond_max::FT
    function IdealVerticalFieldControl{FT}(;
            t_on_s::Real, target::VerticalFieldTarget = MagneticForceTarget(),
            field::VerticalControlField = UniformVerticalField(), induction::ControlInduction = FaradayInduction(),
            response::ControlResponse = FirstOrderResponse{FT}(1.0e-4), ramp_s::Real = 2.0e-3,
            I_on::Real = 100, I_off::Real = 50, coherence_on::Real = 0.9, coherence_off::Real = 0.8,
            coefficient_max::Real = 5.0e-3, cond_max::Real = 1.0e4,
        ) where {FT <: AbstractFloat}
        isnan(t_on_s) && throw(ArgumentError("t_on_s must be a time or Inf, not NaN"))
        (isfinite(ramp_s) && ramp_s > 0) || throw(ArgumentError("ramp_s must be finite and positive, got $ramp_s"))
        0 <= I_off < I_on || throw(ArgumentError("need 0 ≤ I_off < I_on, got I_off = $I_off, I_on = $I_on"))
        0 < coherence_off < coherence_on <= 1 ||
            throw(ArgumentError("need 0 < coherence_off < coherence_on ≤ 1, got $coherence_off, $coherence_on"))
        (isfinite(coefficient_max) && coefficient_max > 0) ||
            throw(ArgumentError("coefficient_max must be finite and positive, got $coefficient_max"))
        cond_max >= 1 || throw(ArgumentError("cond_max must be at least 1, got $cond_max"))
        return new{FT, typeof(target), typeof(field), typeof(induction), typeof(response)}(
            target, field, induction, response, t_on_s, ramp_s, I_on, I_off, coherence_on, coherence_off,
            coefficient_max, cond_max,
        )
    end
end
IdealVerticalFieldControl(; kwargs...) = IdealVerticalFieldControl{Float64}(; kwargs...)

"""
    VerticalFieldControlState{FT}

The control's state between steps: the applied coefficients `a` = (a₁, a₂) [T] and those of the
step before (`a_prev`), the flux offset `c` (and `c_prev`) of `CSCompensatedInduction`, the step
`dt_applied` the induced field was computed for, the target of the last update, its status code
(0 off, 1 waiting, 2 ramping, 3 on, 4 saturated, 5 turning off), whether the column qualifies,
the time it last turned on, `R_ref`, the compensation point, the force residual F₀ + A a
predicted when the coefficients were set, and the power ∫Jϕ E_ϕ,ctrl dV of the last accepted step
and the energy given to the plasma (from an external store, not heat).
"""
@kwdef mutable struct VerticalFieldControlState{FT <: AbstractFloat}
    a::Vector{FT} = zeros(FT, 2)
    a_prev::Vector{FT} = zeros(FT, 2)
    c::FT = zero(FT)
    c_prev::FT = zero(FT)
    dt_applied::FT = zero(FT)
    a_target::Vector{FT} = zeros(FT, 2)
    status::Int = 0
    qualifies::Bool = false
    t_turned_on::FT = FT(NaN)
    R_ref::FT = FT(NaN)
    R_comp::FT = FT(NaN)
    Z_comp::FT = FT(NaN)
    force_R_predictor::FT = FT(NaN)
    force_Z_predictor::FT = FT(NaN)
    power::FT = zero(FT)
    energy::FT = zero(FT)
end
