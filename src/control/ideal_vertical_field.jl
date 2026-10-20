# The ideal vertical-field actuator (`IdealVerticalFieldControl`): each step, from the state at tⁿ,
# the coefficients of a vacuum field that cancels the column's reduced force with the control left
# out, relaxed in physical time, written into the control arrays and combined into the total field
# before the step solves. Policy and state: `control/ideal_vertical_field_types.jl`.

_vertical_field_control_on(RP::RAPID) = RP.flags.vertical_field_control isa IdealVerticalFieldControl

# the two vacuum bases at (R, Z): flux, B_Z and B_R (B_Z = ψ_R/R, B_R = −ψ_Z/R)
_control_basis_ψ(R, Z, R_ref) = (R^2 / 2, R^2 * Z / R_ref)
_control_basis_BZ(R, Z, R_ref) = (one(R), 2Z / R_ref)
_control_basis_BR(R, Z, R_ref) = (zero(R), -R / R_ref)

"""
    refuse_unsupported_vertical_field_control(RP)

Throw for a run the vertical-field control cannot serve: with `Global_JxB_Force` (both would
move the column), without Ampère (the target needs the plasma's own field), or with coils (no
circuit couples to the control flux). Called by `initialize!` and at every update.
"""
function refuse_unsupported_vertical_field_control(RP::RAPID)
    _vertical_field_control_on(RP) || return RP
    RP.flags.Global_JxB_Force &&
        throw(ArgumentError("vertical_field_control = IdealVerticalFieldControl cannot run with Global_JxB_Force"))
    RP.flags.Ampere ||
        throw(ArgumentError("vertical_field_control = IdealVerticalFieldControl needs Ampere = true: its target uses the plasma's own field"))
    RP.coil_system.n_total > 0 &&
        throw(ArgumentError("vertical_field_control = IdealVerticalFieldControl cannot run with coils: no circuit couples to the control flux"))
    return RP
end

# the forces the target cancels, control left out, from the parts that `_column_forces` and the 0D
# snapshot both carry: radial by the target, vertical the magnetic forces either way
_target_force_R(::ReducedForceTarget, col) =
    col.force_R_presc + col.force_R_coils + col.force_R_pla + col.force_R_pressure + col.force_R_toroidal
_target_force_R(::MagneticForceTarget, col) = col.force_R_presc + col.force_R_coils + col.force_R_pla
_target_force_Z(col) = col.force_Z_presc + col.force_Z_coils + col.force_Z_pla

# the coefficients from A a = −F₀: the uniform field alone, or both bases jointly
function _target_coefficients(::UniformVerticalField, F₀, A, policy)
    return [-F₀[1] / A[1, 1], zero(eltype(A))], true
end
function _target_coefficients(::VerticalAndRadialField, F₀, A, policy)
    cond(A) <= policy.cond_max || return zeros(eltype(A), 2), false
    return A \ (-F₀), true
end

"""
    vertical_field_control_target(RP) -> NamedTuple

The coefficients a* = (a₁, a₂) that cancel the column's force F₀ (`target`, control field left
out) through A a* = −F₀, with A_{Rk} = ∫Jϕ B_{Z,k} dV and A_{Zk} = −∫Jϕ B_{R,k} dV. With
`UniformVerticalField`, a₁* = −F_R0/∫Jϕ dV (with `ReducedForceTarget`, `column_force_balance`'s
`Bv_to_add`) and a₂* = 0. Also returns F₀, A, the current, its coherence and centroid, and whether
the system can be solved (`solvable`: current above the Ampère threshold, a net ∫Jϕ dV, finite
forces and, for both bases, condition number within `cond_max`). Changes no state.
"""
function vertical_field_control_target(RP::RAPID{FT}) where {FT <: AbstractFloat}
    policy = RP.flags.vertical_field_control
    G, J = RP.G, RP.plasma.Jϕ
    st = RP.vertical_field_control
    R_ref = isnan(st.R_ref) ? (G.R1D[1] + G.R1D[end]) / 2 : st.R_ref
    I = sum(J) * G.dR * G.dZ
    nothing_to_hold = (;
        a = zeros(FT, 2), F = fill(FT(NaN), 2), A = fill(FT(NaN), 2, 2), I,
        coherence = FT(NaN), R_cen = FT(NaN), Z_cen = FT(NaN), solvable = false,
    )
    (iszero(I) || abs(I) < RP.flags.Ampere_Itor_threshold) && return nothing_to_hold

    BR_pla, BZ_pla = _plasma_poloidal_field(RP)
    BR_coils, BZ_coils = _coil_poloidal_field(RP)
    col = _column_forces(RP, BR_pla, BZ_pla, BR_coils, BZ_coils)
    F₀ = [_target_force_R(policy.target, col), _target_force_Z(col)]

    V = G.inVol2D
    A = zeros(FT, 2, 2)
    A[1, 1] = col.∫J                                            # B_Z,1 = 1
    A[1, 2] = sum(@. J * 2 * G.Z2D / R_ref * V)
    A[2, 2] = sum(@. J * G.R2D / R_ref * V)                     # −B_R,2 = R/R_ref; B_R,1 = 0
    solvable = col.net_current && all(isfinite, F₀) && all(isfinite, A)
    a = zeros(FT, 2)
    if solvable
        a, solvable = _target_coefficients(policy.field, F₀, A, policy)
    end
    return (; a, F = F₀, A, I, coherence = col.current_coherence, R_cen = col.R_cen_by_Jϕ, Z_cen = col.Z_cen_by_Jϕ, solvable)
end

"""
    update_vertical_field_control!(RP)

The control's step, from the state at tⁿ (`prepare_timestep!` calls it after `update_Jϕ!`):
1. the target ([`vertical_field_control_target`](@ref)) and the status: qualify with hysteresis,
   ramp the target up after each turn-on, clip at `coefficient_max`, aim at zero when off;
2. move the coefficients toward it (`response`);
3. write the control field and its induced E_ϕ, and combine the fields, so that the whole step
   (electron momentum, then ions and temperatures) sees the same control field. The combine also
   brings in the prescribed field at tⁿ, which the path without control takes in only inside the
   step: compare a controlled run with `t_on_s = Inf`, not with `NoVerticalFieldControl`.

The coefficients set at tⁿ hold over [tⁿ, tⁿ⁺¹], so the ramp is taken at tⁿ⁺¹ and the first-order
response to a fixed target is the same at the same time for any Δt. Turned off, coefficients below
10⁻⁶ `coefficient_max` are set to zero. The work of the step is booked after it is accepted
([`book_vertical_field_control_work!`](@ref)). Nothing changes without `IdealVerticalFieldControl`; the coefficients stay fixed
while the step solves (a gate retry does not call this again). `Jϕ` is not touched: the caller
recomputes it on the combined field.
"""
function update_vertical_field_control!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    _vertical_field_control_on(RP) || return RP
    refuse_unsupported_vertical_field_control(RP)
    policy, st, G, F = RP.flags.vertical_field_control, RP.vertical_field_control, RP.G, RP.fields
    isnan(st.R_ref) && (st.R_ref = (G.R1D[1] + G.R1D[end]) / 2)
    dt, t = RP.dt, RP.time_s

    # 1. target and status
    target = vertical_field_control_target(RP)
    qualifies = if t < policy.t_on_s || !target.solvable
        false
    elseif st.qualifies
        abs(target.I) >= policy.I_off && target.coherence >= policy.coherence_off
    else
        abs(target.I) >= policy.I_on && target.coherence >= policy.coherence_on
    end
    qualifies && !st.qualifies && (st.t_turned_on = t)
    st.qualifies = qualifies
    if qualifies
        ramp = clamp((t + dt - st.t_turned_on) / policy.ramp_s, zero(FT), one(FT))
        clipped = clamp.(target.a, -policy.coefficient_max, policy.coefficient_max)
        aim = ramp .* clipped
        st.status = any(abs.(target.a) .> policy.coefficient_max) ? 4 : ramp < 1 ? 2 : 3
    else
        aim = zeros(FT, 2)
        st.status = t < policy.t_on_s ? 0 : any(!iszero, st.a) ? 5 : 1
    end
    st.a_target .= target.a

    # 2. respond
    st.a_prev .= st.a
    _respond!(policy.response, st.a, aim, dt)
    if !qualifies && maximum(abs, st.a) < 1.0e-6 * policy.coefficient_max
        fill!(st.a, zero(FT))
        st.status = t < policy.t_on_s ? 0 : 1
    end
    if target.solvable
        residual = target.F .+ target.A * st.a
        st.force_R_predictor, st.force_Z_predictor = residual
    else
        st.force_R_predictor, st.force_Z_predictor = FT(NaN), FT(NaN)
    end

    # 3. the field, its induced E_ϕ over this step, and the combined total
    st.dt_applied = dt
    write_vertical_control_field!(RP)
    combine_field_sources!(RP)
    return RP
end

# the applied coefficients follow the aim: first order in physical time, or at once
function _respond!(r::FirstOrderResponse, a, aim, dt)
    α = -expm1(-dt / r.τ_s)
    @. a += α * (aim - a)
    return a
end
_respond!(::InstantaneousResponse, a, aim, dt) = (a .= aim)

"""
    book_vertical_field_control_work!(RP)

The control's work over the step just accepted (`solve_timestep!` calls it at its end, before
the snapshots): power ∫Jϕ E_ϕ,ctrl dV with the control field that acted over the step and the
current the momentum solve left, and energy += power × `dt_applied`. Energy exchanged with the
actuator's store, not heat.
"""
function book_vertical_field_control_work!(RP::RAPID)
    _vertical_field_control_on(RP) || return RP
    st, G = RP.vertical_field_control, RP.G
    st.power = sum(@. RP.plasma.Jϕ * RP.fields.Eϕ_ctrl * G.inVol2D)
    st.energy += st.power * st.dt_applied
    return RP
end

"""
    write_vertical_control_field!(RP)

The control arrays from the state's coefficients: ψ_ctrl = a₁ R²/2 + a₂ R²Z/R_ref, its B, and
E_ϕ,ctrl = −(ψ_ctrl − ψ_ctrl,prev)/(R Δt) from the change since the step before, over the step
`dt_applied` (zero before any step).
"""
function write_vertical_control_field!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    st, G, F = RP.vertical_field_control, RP.G, RP.fields
    a₁, a₂, R_ref = st.a[1], st.a[2], st.R_ref
    Δa₁, Δa₂ = a₁ - st.a_prev[1], a₂ - st.a_prev[2]
    R, Z, dt = G.R2D, G.Z2D, st.dt_applied
    @. F.ψ_ctrl = a₁ * R^2 / 2 + a₂ * R^2 * Z / R_ref
    @. F.BZ_ctrl = a₁ + a₂ * 2 * Z / R_ref
    @. F.BR_ctrl = -a₂ * R / R_ref
    if iszero(dt)                                               # no step interval yet: nothing induced
        fill!(F.Eϕ_ctrl, zero(FT))
    else
        @. F.Eϕ_ctrl = -(Δa₁ * R^2 / 2 + Δa₂ * R^2 * Z / R_ref) / (R * dt)
    end
    return RP
end

"""
    measure_vertical_field_control!(snap0D, RP)

The control's 0D values: status, coefficients and targets, the control field at the current
centroid, the force residual predicted when the coefficients were set and the one of the
present state (F₀ from the snapshot's `column_force_balance` values, plus A a), the force scale
μ0 I²/4, the current-weighted control loop voltage, the power and energy given to the plasma,
and the total external B_Z and decay index at the centroid. Zero without control.
"""
function measure_vertical_field_control!(snap0D, RP::RAPID{FT}) where {FT <: AbstractFloat}
    _vertical_field_control_on(RP) || return snap0D
    policy, st, G, F, J = RP.flags.vertical_field_control, RP.vertical_field_control, RP.G, RP.fields, RP.plasma.Jϕ
    snap0D.ctrl_status = st.status
    snap0D.ctrl_a1, snap0D.ctrl_a2 = st.a
    snap0D.ctrl_a1_target, snap0D.ctrl_a2_target = st.a_target
    snap0D.ctrl_force_R_predictor, snap0D.ctrl_force_Z_predictor = st.force_R_predictor, st.force_Z_predictor
    snap0D.ctrl_power, snap0D.ctrl_energy = st.power, st.energy

    dA, V = G.dR * G.dZ, G.inVol2D
    I = sum(J) * dA
    snap0D.ctrl_force_scale = RP.config.constants.μ0 * I^2 / 4
    snap0D.ctrl_loop_voltage = iszero(I) ? FT(NaN) : sum(@. J * 2π * G.R2D * F.Eϕ_ctrl) * dA / I
    R_cen, Z_cen = snap0D.R_cen_by_Jϕ, snap0D.Z_cen_by_Jϕ
    if isnan(R_cen) || isnan(st.R_ref)
        snap0D.ctrl_BZ_at_cen = snap0D.ctrl_BR_at_cen = FT(NaN)
        snap0D.ctrl_BZ_total_at_cen = snap0D.ctrl_decay_index_total = FT(NaN)
        snap0D.ctrl_force_R_residual = snap0D.ctrl_force_Z_residual = FT(NaN)
        return snap0D
    end
    BZ₁, BZ₂ = _control_basis_BZ(R_cen, Z_cen, st.R_ref)
    BR₁, BR₂ = _control_basis_BR(R_cen, Z_cen, st.R_ref)
    snap0D.ctrl_BZ_at_cen = st.a[1] * BZ₁ + st.a[2] * BZ₂
    snap0D.ctrl_BR_at_cen = st.a[1] * BR₁ + st.a[2] * BR₂
    snap0D.ctrl_force_R_residual = _target_force_R(policy.target, snap0D) + sum(@. J * F.BZ_ctrl * V)
    snap0D.ctrl_force_Z_residual = _target_force_Z(snap0D) - sum(@. J * F.BR_ctrl * V)
    # the decay index of the prescribed + control field (the control adds no ∂B_Z/∂R), from the
    # derivative, with the total B_Z in the denominator
    h = G.dR / 2
    dBZ_dR = (_bilinear(G, F.BZ_presc, R_cen + h, Z_cen) - _bilinear(G, F.BZ_presc, R_cen - h, Z_cen)) / (2h)
    BZ_total = _bilinear(G, F.BZ_presc, R_cen, Z_cen) + snap0D.ctrl_BZ_at_cen
    snap0D.ctrl_BZ_total_at_cen = BZ_total
    snap0D.ctrl_decay_index_total = iszero(BZ_total) ? FT(NaN) : -R_cen * dBZ_dR / BZ_total
    return snap0D
end
