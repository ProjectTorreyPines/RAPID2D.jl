# The ions' u∥ and Tᵢ carried by the particles that move them, as the electrons' are.
#
# A variable per ion is carried by the operator that moved the ion densities: the per-particle
# operator of the ion continuity's own (each transport group's diffusion, the ion velocity's
# convection), or of the electrons' where the ions are slaved to them. That move publishes its
# operators ([`IonCarriage`](@ref)); the start of a step retracts them, so nothing moved means
# nothing carried. Momentum is counted by mass, heat by number, over the whole ion fluid: a species
# the transport does not move dilutes, it does not carry.
# internal/docs/src/reference/governing-equations.md §4.1–4.2;
# internal/docs/src/notes/plans/PLAN_ion-momentum-energy-transport.md §2.

const ION_CARRIAGE_N_FLOOR = 1.0   # m⁻³, the electrons' floor of the per-particle rows

"""
    retract_ion_carriage!(RP) -> RP

Forget what moved the ion densities: the start of a step, a new layout, a new run.
"""
function retract_ion_carriage!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    car = RP.operators.ion_carriage
    car.source = NotCarried()
    empty!(car.parts)
    car.convec = false
    return RP
end

# Transport group `g`'s buffers, allocated on first use and kept: its density diffusion, the operator
# of its conduction excess and that excess's coefficient
function ion_group_buffers!(car::IonCarriage{FT}, G::GridGeometry{FT}, g::Integer) where {FT <: AbstractFloat}
    while length(car.group_buffers) < g
        pattern = build_wall_pattern(G)
        push!(car.group_buffers, (pattern, similar(pattern), zeros(FT, G.NR, G.NZ)))
    end
    return car.group_buffers[g]
end

"""
    ion_conduction_excess(buffers, b, D_part, D_cond) -> ConductionExcess or nothing

Where the ions conduct along `b = (bR, bZ)` with more than their particles diffuse with, `D_part`: the
excess `max(D_cond, D_part) − D_part` (reference (4.5)) into the group's coefficient buffer, with the
group's operator buffer it is assembled into when the carriage is built. `nothing` where no node differs.
"""
function ion_conduction_excess(buffers, b, D_part::AbstractMatrix{FT}, D_cond::AbstractMatrix{FT}) where {FT <: AbstractFloat}
    any(k -> D_cond[k] > D_part[k], eachindex(D_part)) || return nothing
    _, A, Δ = buffers
    @. Δ = max(D_cond, D_part) - D_part
    return ConductionExcess{FT}(Δ, b[1], b[2], A)
end

"""
    publish_electron_carriage!(RP) -> RP

Slaved ions (`update_ni_independently = false`) are moved by the electrons' operators, `nᵢ = nₑ/Z`:
publish those, the cached density diffusion `A_diffu_e` with the ions' own parallel conduction's
excess along B, and the electrons' convection, which [`ion_momentum_carriage`](@ref) reads through
`ue_Te_operators` (its closed-surface carriage included).
"""
function publish_electron_carriage!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    retract_ion_carriage!(RP)
    flags = RP.flags
    (flags.diffu || flags.convec) || return RP
    car, op, tp, G = RP.operators.ion_carriage, electron_operator_cache(RP), RP.transport, RP.G
    if flags.diffu
        excess = ion_conduction_excess(ion_group_buffers!(car, G, 1), (RP.fields.bR, RP.fields.bZ), tp.Dpara, tp.Dpara_cond_i)
        push!(car.parts, IonCarriagePart{FT}(collect(eachindex(tp.ion_species)), op.A_diffu_e, excess))
    end
    car.convec = flags.convec
    car.source = CarriedByElectronOperators()
    return RP
end

# The density each transported species carries a variable with, `weight(z, m) nₛ`: summed per part
# (what its diffusion moved), over every transported species (what the shared convection moved), and
# over the whole fluid, every stored ion. A weight of exactly one on the bulk keeps one species' sums
# its density, bit for bit.
function _carrier_densities!(ws, w_moved, w_total, RP::RAPID{FT}, weight) where {FT <: AbstractFloat}
    pla, tp, car = RP.plasma, RP.transport, RP.operators.ion_carriage
    slots = RP.store_layout.transport_slots
    carried(s) = FT(weight(tp.ion_species[s].charge, tp.ion_species[s].mass))
    for (g, part) in enumerate(car.parts)
        w = ws[g]
        fill!(w, zero(FT))
        for s in part.sids
            c, n = carried(s), ion_plane(pla, slots[s])
            @. w += c * n
        end
    end
    fill!(w_moved, zero(FT))
    for s in eachindex(tp.ion_species)
        c, n = carried(s), ion_plane(pla, slots[s])
        @. w_moved += c * n
    end
    c1 = FT(weight(bulk_ion_charge(RP), bulk_ion_mass(RP)))
    @. w_total = c1 * pla.ni
    foreach_other_ion(RP) do n, z, m
        c = FT(weight(z, m))
        @. w_total += c * n
    end
    return nothing
end

# The carried operators by `weight` into `M` and `A_adv`: per part the per-particle operator over
# the total and the conduction excess with the part's carriers in its coefficient, the convection's
# over the total of what moves. `nothing` for what did not move the densities.
function _carried_operators!(M, A_adv, RP::RAPID{FT}, weight) where {FT <: AbstractFloat}
    car = RP.operators.ion_carriage
    n_floor = FT(ION_CARRIAGE_N_FLOOR)
    has_parts = !isempty(car.parts)
    dims = (RP.G.NR, RP.G.NZ)
    update_charge_states!(RP)
    @with_pool pool begin
        ws = [acquire!(pool, FT, dims...) for _ in car.parts]
        w_moved, w_total = acquire!(pool, FT, dims...), acquire!(pool, FT, dims...)
        _carrier_densities!(ws, w_moved, w_total, RP, weight)
        if has_parts
            per_particle_operator!(M, [p.A_part for p in car.parts], ws, w_total; n_floor)
            for (p, w) in zip(car.parts, ws)
                x = p.excess
                isnothing(x) && continue
                # assembled like the densities' move: its discretization, its limiter on the step's density
                conduction_excess_operator!(
                    x.A, RP.G, x.Δ, (x.bR, x.bZ), w, RP.flags.anisotropic_discretization; n_total = w_total, n_floor,
                    n = RP.prev_n,
                )
                add_conduction_excess!(M, x.A, w_total; n_floor)
            end
        end
        if car.convec && car.source isa CarriedByIonOperators
            per_particle_operator!(A_adv, (car.A_conv,), (w_moved,), w_total; n_floor)
        elseif car.convec
            # slaved: the electrons' carriage, on their density (nᵢ = nₑ/Z), closed surfaces included
            copy_values!(A_adv, ue_Te_operators(RP).A_adv)
        end
        nothing
    end
    return (A_adv = car.convec ? A_adv : nothing, M = has_parts ? M : nothing)
end

"""
    ion_momentum_carriage(RP) -> (; A_adv, M) or nothing

What u_i∥ is carried with this step, by mass over the whole ion fluid: `A_adv`, the convection
(`(u·∇)` on the moving part), and `M`, the diffusive carriage and viscosity with the ions' own
conduction along B (`𝒯_D^ρ` of reference (4.9)); either `nothing` where its channel did not move the
densities, and `nothing` altogether when nothing did. Buffers, rewritten by the next call.
"""
function ion_momentum_carriage(RP::RAPID{FT}) where {FT <: AbstractFloat}
    car = RP.operators.ion_carriage
    car.source isa NotCarried && return nothing
    m_bulk = bulk_ion_mass(RP)
    return _carried_operators!(car.M_rho, car.A_adv_rho, RP, (z, m) -> m / m_bulk)
end

"""
    ion_heat_carriage(RP) -> (; A_adv, M, div_u) or nothing

What Tᵢ is carried with this step, by number over the whole ion fluid, as
[`ion_momentum_carriage`](@ref), and `div_u`, the compression per ion: `∇·u` of the velocity that
moved the densities times the moving part `n_T/n_I` (zero where there are no ions).
"""
function ion_heat_carriage(RP::RAPID{FT}) where {FT <: AbstractFloat}
    car = RP.operators.ion_carriage
    car.source isa NotCarried && return nothing
    ops = _carried_operators!(car.M_n, car.A_adv_n, RP, (z, m) -> one(m))
    pla = RP.plasma
    if !car.convec
        fill!(car.div_u_n, zero(FT))
    elseif car.source isa CarriedByIonOperators
        @with_pool pool begin
            n_moved = acquire!(pool, FT, size(pla.ni)...)
            fill!(n_moved, zero(FT))
            for slot in RP.store_layout.transport_slots
                n_moved .+= ion_plane(pla, slot)
            end
            @. car.div_u_n = ifelse(pla.ni_tot > zero(FT), car.div_u * (n_moved / pla.ni_tot), zero(FT))
            nothing
        end
    else
        car.div_u_n .= ue_Te_operators(RP).div_u
    end
    return (A_adv = ops.A_adv, M = ops.M, div_u = car.div_u_n)
end

"""
    carry_ui_para!(RP, carriage, ν_a, accel, relax_weight, decay_is_exprb) -> RP

uᵢ's update with its carriage (`update_ui_para!` after the births): the drag backward Euler or by
ExpRB's `bern`, the field's `accel`, the convection `A_adv` and the diffusive carriage and viscosity
`M` of [`ion_momentum_carriage`](@ref) under `θ_imp.decay` (explicit with `Implicit = false`), and the
near-vacuum relaxation backward Euler, as the electrons' u∥ ([`update_ue_para!`](@ref)). The
momentum the relaxation took out is booked in `diagnostics.near_vacuum.relax_momentum_i`.
"""
function carry_ui_para!(
        RP::RAPID{FT}, carriage, ν_a::AbstractMatrix{FT}, accel::AbstractMatrix{FT},
        relax_weight::AbstractMatrix{FT}, decay_is_exprb::Bool,
    ) where {FT <: AbstractFloat}
    pla, OP, dt = RP.plasma, RP.operators, RP.dt
    one_FT = one(FT)
    A_adv = isnothing(carriage) ? nothing : carriage.A_adv
    M = isnothing(carriage) ? nothing : carriage.M
    θ = RP.flags.Implicit ? RP.flags.θ_imp.decay : zero(FT)
    u_ref = vacuum_reference(RP, pla.ni_tot, pla.ui_para, relax_weight)
    # the drag's diagonal, and what it leaves of uⁿ on the right-hand side
    diag = if decay_is_exprb
        decay_exponent = @. exprb_cap_exponent(-ν_a * dt)
        bern_decay = exprb_bern.(decay_exponent)
        @. OP.RHS = (bern_decay + decay_exponent) * pla.ui_para + accel
        bern_decay
    else
        @. OP.RHS = pla.ui_para + accel
        @. one_FT + dt * ν_a
    end
    # the carriage's explicit part, (1 − θ)Δt (M − A_adv) uⁿ
    w = (one_FT - θ) * dt
    if !iszero(w)
        isnothing(A_adv) || (OP.RHS .-= w .* (A_adv * pla.ui_para))
        isnothing(M) || (OP.RHS .+= w .* (M * pla.ui_para))
    end
    @. OP.RHS += ifelse(relax_weight > zero(FT), relax_weight * u_ref, zero(FT))
    if RP.flags.Implicit && (!isnothing(A_adv) || !isnothing(M))
        # the shared LHS buffer on the wall pattern: values only, the solver keeps its analysis
        A = OP.A_LHS
        set_identity!(A)
        isnothing(A_adv) || add_scaled!(A, θ * dt, A_adv)
        isnothing(M) || add_scaled!(A, -θ * dt, M)
        add_diagonal!(A, vec(diag) .- one_FT)
        add_diagonal!(A, vec(relax_weight))
        @timeit RAPID_TIMER "ui_para LinearSolve" begin
            factorize!(OP.ui_solver, A)
            solve!(view(pla.ui_para, :), OP.ui_solver, view(OP.RHS, :))
        end
    else
        @. pla.ui_para = OP.RHS / (diag + relax_weight)
    end
    # the momentum the near-vacuum relaxation took out, before the e–i impulse
    RP.diagnostics.near_vacuum.relax_momentum_i +=
        vacuum_relaxation_exchange(RP, pla.mni_tot, relax_weight, pla.ui_para, u_ref)
    return RP
end

"""
    carry_Ti!(RP, heat, relax_weight) -> RP

Tᵢ's update with its carriage (`update_Ti!`), as the electrons' Tₑ ([`update_Te!`](@ref)): the
atomic power (ExpRB's `bern` on the diagonal when `scheme.atomic` is ExpRB), the convection and
diffusive carriage of [`ion_heat_carriage`](@ref) under `θ_imp.transport` and its compression by
[`add_compression!`](@ref), their explicit parts through `iPowers.conv` and `iPowers.diffu`, and the
near-vacuum relaxation backward
Euler. Without carriage (the relaxation alone) or with `Implicit = false` the update is diagonal and
nothing is factorized. The thermal energy the relaxation took out is booked, before the clamp, in
`diagnostics.near_vacuum.relax_energy_i`.
"""
function carry_Ti!(RP::RAPID{FT}, heat, relax_weight::AbstractMatrix{FT}) where {FT <: AbstractFloat}
    pla, OP, dt = RP.plasma, RP.operators, RP.dt
    ee = RP.config.constants.ee
    iP = pla.iPowers
    two_thirds = FT(2) / FT(3)
    θ = RP.flags.θ_imp.transport
    T_ref = vacuum_reference(RP, pla.ni_tot, pla.Ti_eV, relax_weight)
    A_adv = isnothing(heat) ? nothing : heat.A_adv
    M = isnothing(heat) ? nothing : heat.M
    bern = if RP.flags.scheme.atomic === ExpRB
        update_ion_power_jacobian!(RP)
        atomic_exponent = @. exprb_cap_exponent(pla.exprb.eig_Ti * dt)
        _warn_if_exprb_capped(atomic_exponent)
        exprb_bern.(atomic_exponent)
    else
        nothing
    end
    if RP.flags.Implicit && (!isnothing(A_adv) || !isnothing(M))
        A = OP.A_LHS   # the shared LHS buffer on the wall pattern: values only
        set_identity!(A)
        P_tilde = copy(iP.tot)
        if !isnothing(M)
            @. P_tilde -= θ * iP.diffu
            add_scaled!(A, -two_thirds * FT(1.5) * (dt * θ), M)
        end
        if !isnothing(A_adv)
            # P_conv = −1.5 (u·∇)Tᵢ − Tᵢ ∇·u: the advection on the LHS as +1.5 c A_adv, c = (2/3)Δt θ;
            # the compression θ-weighted node by node (`add_compression!`)
            @. P_tilde -= θ * iP.conv
            add_scaled!(A, two_thirds * (dt * θ) * FT(1.5), A_adv)
            add_compression!(A, P_tilde, pla.Ti_eV, heat.div_u, θ; dt, ee, bern)
        end
        if isnothing(bern)
            @. OP.RHS = pla.Ti_eV + two_thirds * (dt * P_tilde / ee)
        else
            add_diagonal!(A, vec(bern) .- one(FT))
            @. OP.RHS = bern * pla.Ti_eV + two_thirds * (dt * P_tilde / ee)
        end
        add_diagonal!(A, vec(relax_weight))
        @. OP.RHS += ifelse(relax_weight > zero(FT), relax_weight * T_ref, zero(FT))
        @timeit RAPID_TIMER "Ti_eV LinearSolve" begin
            factorize!(OP.Ti_solver, A)
            solve!(view(pla.Ti_eV, :), OP.Ti_solver, view(OP.RHS, :))
        end
    else
        # forward Euler (bern dividing the increment under ExpRB), the relaxation backward Euler
        b = isnothing(bern) ? one(FT) : bern
        @. pla.Ti_eV = ifelse(
            relax_weight > zero(FT),
            (b * pla.Ti_eV + two_thirds * iP.tot * dt / ee + relax_weight * T_ref) / (b + relax_weight),
            pla.Ti_eV + two_thirds * iP.tot * dt / ee / b,
        )
    end
    RP.diagnostics.near_vacuum.relax_energy_i +=
        FT(1.5) * ee * vacuum_relaxation_exchange(RP, pla.ni_tot, relax_weight, pla.Ti_eV, T_ref)
    return RP
end
