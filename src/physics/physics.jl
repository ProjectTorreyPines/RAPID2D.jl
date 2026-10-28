"""
Physics module for RAPID2D.

Contains functions related to plasma physics models, including:
- Electron and ion dynamics
- Collisions
- Reaction rates
- Power balance
"""

using TimerOutputs

# Use the global timer from the main module

# Export public functions
export update_ue_para!,
    update_ui_para!,
    update_Te!,
    update_Ti!,
    update_coulomb_collision_parameters!,
    update_electron_heating_powers!,
    update_electron_power_jacobian!,
    update_ion_power_jacobian!,
    update_ion_heating_powers!,
    solve_electron_continuity_equation!,
    solve_ion_continuity_equation!,
    update_charge_states!,
    apply_electron_density_boundary_conditions!,
    calculate_para_grad_of_scalar_F,
    calculate_grad_of_scalar_F,
    calculate_electron_acceleration_by_pressure,
    update_uMHD_by_global_JxB_force!,
    solve_combined_momentum_Ampere_equations_with_coils!

"""
    update_ue_para!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Update the parallel electron velocity: the θ-scheme, with `θ_imp.decay` on the friction and on
the nonlocal operators (advection, diffusion). With `Implicit = false` those operators are
explicit (θ = 0), and the update divides by the friction's diagonal instead of solving.
"""
function update_ue_para!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    @timeit RAPID_TIMER "update_ue_para!" begin
        one_FT = one(FT)   # hoisted for type stability
        @unpack qe, me = RP.config.constants
        dt = RP.dt
        pla = RP.plasma
        F = RP.fields

        # Total decay rate of the drift. ν_iz_tot belongs here because newborn electrons
        # (from EITHER ionization channel) enter at rest — dilution, not momentum
        # transfer to the gas. ν_ei_eff (= ξ_sptz·ν_ei) is the Coulomb half of the same
        # friction; its u_i∥ half is added as a source below. The combined
        # momentum-Ampère solvers build the identical sum.
        ν_sum_mom_iz_ei = @. pla.ν_en_iz_tot + pla.ν_en_mom_tot + pla.ν_ei_eff
        # u∥ → the local plasma's u∥ where there is almost no plasma: backward Euler, the
        # weight on the diagonal and the local mean (from uⁿ) on the right-hand side
        relax_weight = vacuum_relaxation_weights(RP, pla.ne)
        ue_ref = vacuum_reference(RP, pla.ne, pla.ue_para, relax_weight)

        # Backward Euler by default (θ_imp.decay = 1): friction-dominated, so at large
        # Δt BE lands on u∞ = S/ν while CN rings about it. ExpRB replaces the constant
        # with the friction's own fitted θ(z), z = −νΔt, per cell — a better local
        # weight where the step resolves ν, but still first order and LESS implicit
        # than BE (θ_fit < 1 on a decay branch). See exprb-implementation.md §5 phase 2.
        decay_is_exprb = RP.flags.scheme.decay === ExpRB
        decay_is_exprb && RP.flags.closed_surface_flow isa SurfaceFluxConstraint &&
            _refuse_exprb_decay(RP.flags, "SurfaceFluxConstraint", "solves u∥ with the constant θ_imp.decay")
        decay_is_exprb && _refuse_full_response_decay(RP.flags)
        decay_exponent = decay_is_exprb ? (@. exprb_cap_exponent(-ν_sum_mom_iz_ei * dt)) : nothing
        bern_decay = decay_is_exprb ? exprb_bern.(decay_exponent) : nothing
        # `θu` weights the FRICTION and the ledger that records it (per-cell under
        # ExpRB); `θ_op` weights the nonlocal operators, which the fit does not reach
        # and which keep today's constant either way. Splitting them keeps
        # `scheme.decay` from silently changing the transport treatment. `Implicit = false`
        # steps the nonlocal operators explicitly, θ_op = 0: the same equation, whose matrix
        # is then the friction's diagonal and needs no solve. The friction keeps its weight.
        θ_op = RP.flags.Implicit ? RP.flags.θ_imp.decay : zero(FT)
        θu = decay_is_exprb ? exprb_theta.(decay_exponent) : RP.flags.θ_imp.decay

        # (u·∇)u∥ from the face mass flux and the reflective in-wall viscosity: rows on
        # in-wall nodes only, nothing read outside the wall (`electron_operators.jl`).
        pops = ue_Te_operators(RP)

        # Rue_ei, part 1 (uⁿ). A LEDGER of the exchange over the step, so it must use
        # the quadrature the update actually performed — θ(z) under ExpRB.
        if RP.flags.Coulomb_Collision
            @. pla.Rue_ei = pla.ν_ei_eff * (pla.ui_para - (one_FT - θu) * pla.ue_para)
        end

        # The right-hand side, every term from uⁿ
        # #1: Electric acceleration term [qe*E_para_tot/me]
        accel_para_tilde = qe * F.E_para_tot / me

        # #2: Advection term (1-θ_op)*[-(𝐮⋅∇)*ue_para]. θ_op, not θu: the fitted
        # weight belongs to the friction's eigenvalue, not to a nonlocal operator.
        if RP.flags.Include_ud_convec_term
            accel_para_tilde .+= (one_FT - θ_op) * (-(pops.A_adv * pla.ue_para))
        end

        # #3: Pressure term [-∇∥(ne*Te)/(me*ne)], the share u∥ carries; the ambipolar one, kept
        # for its ledger
        a_amb = nothing
        if RP.flags.Include_ud_pressure_term
            shares = electron_pressure_shares(RP)
            accel_para_tilde .+= shares.ue
            a_amb = shares.amb
        end
        u_old = isnothing(a_amb) ? nothing : copy(pla.ue_para)   # for the exchange ledger
        ue_entry = copy(pla.ue_para)                               # uⁿ, for the friction's heat

        # #4: collision drag force  (1-θu)*[-(ν_en_iz_tot + ν_mom + ν_ei_eff)*ue_para]. Under
        # ExpRB uⁿ carries bern(−z) instead, applied to the RHS below.
        if !decay_is_exprb
            @. accel_para_tilde += (one_FT - θu) * (-ν_sum_mom_iz_ei * pla.ue_para)
        end

        # #5: momentum source from electron-ion collision [+sptz_fac*νei*ui_para]
        @. accel_para_tilde += (pla.ν_ei_eff * pla.ui_para)

        # #6: turbulent Diffusive term by ExB mixing (nonlocal — θ_op)
        if RP.flags.Include_ud_diffu_term
            accel_para_tilde .+= (one_FT - θ_op) * (pops.A_diffu * pla.ue_para)
        end

        # bern(−z) formed as bern(z) + z, not the algebraically equal
        # 1 − (1−θ)νΔt: that subtraction cancels to nothing exactly where the
        # true value is small but meaningful.
        OP = RP.operators
        if decay_is_exprb
            @. OP.RHS = (bern_decay + decay_exponent) * pla.ue_para + dt * accel_para_tilde
        else
            @. OP.RHS = pla.ue_para + dt * accel_para_tilde
        end
        @. OP.RHS += ifelse(relax_weight > 0, relax_weight * ue_ref, zero(FT))

        if RP.flags.Implicit
            # The shared LHS buffer on the wall pattern: values only, so the cached solver
            # keeps its symbolic analysis from step to step.
            A = OP.A_LHS
            set_identity!(A)
            RP.flags.Include_ud_convec_term && add_scaled!(A, θ_op * dt, pops.A_adv)
            if decay_is_exprb
                add_diagonal!(A, vec(bern_decay) .- one_FT)
            else
                add_diagonal!(A, vec(ν_sum_mom_iz_ei); scale = θu * dt)
            end
            add_diagonal!(A, vec(relax_weight))
            RP.flags.Include_ud_diffu_term && add_scaled!(A, -θ_op * dt, pops.A_diffu)

            @timeit RAPID_TIMER "ue_para LinearSolve" begin
                if !closed_surface_active(RP)
                    factorize!(OP.ue_solver, A)
                    solve!(view(pla.ue_para, :), OP.ue_solver, view(OP.RHS, :))
                else
                    # SurfaceFluxConstraint: n u∥/B a surface function on the closed region
                    surface_constrained_ue!(RP, A.matrix, vec(OP.RHS), θu)
                end
            end
        else
            # θ_op = 0: the matrix is the friction's diagonal, bern(z) under ExpRB
            if !closed_surface_active(RP)
                if decay_is_exprb
                    @. pla.ue_para = OP.RHS / (bern_decay + relax_weight)
                else
                    @. pla.ue_para = OP.RHS / (one_FT + θu * ν_sum_mom_iz_ei * dt + relax_weight)
                end
            else
                # SurfaceFluxConstraint: the same diagonal, solved with the constraint
                surface_constrained_ue!(RP, spdiagm(0 => vec(@. one_FT + θu * ν_sum_mom_iz_ei * dt + relax_weight)), vec(OP.RHS), θu)
            end
        end

        # Rue_ei, part 2 (uⁿ⁺¹)
        if RP.flags.Coulomb_Collision
            @. pla.Rue_ei += pla.ν_ei_eff * (-θu * pla.ue_para)
        end
        record_ei_impulse!(RP, ue_entry)   # before continuity moves nₑ

        # the momentum the near-vacuum relaxation took out
        RP.diagnostics.near_vacuum.relax_momentum +=
            me * vacuum_relaxation_exchange(RP, pla.ne, relax_weight, pla.ue_para, ue_ref)
        # the work the ambipolar share of the pressure force would have done
        isnothing(a_amb) || book_ambipolar_exchange!(RP, a_amb, u_old)

        return RP
    end # @timeit
end


"""
    update_ui_para!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Update the parallel ion velocity from the ions' momentum per volume ρ uᵢ, ρ = Σ mₛ nₛ after this
step's continuity. In order: the ions born at rest dilute it, the parallel field (on the ions'
charge per mass) and the neutral drag act on it, and it takes −I_ei, the opposite of the
impulse the electron momentum equation recorded, so the e–i exchange cancels exactly whatever
the scheme and the densities.

That friction is explicit in uᵢ, at the rate (mₑnₑ/ρ)ν_ei. With the electrons' friction backward
Euler or ExpRB the relative drift still decays at any Δt where the ions outweigh the electrons,
ρ > mₑnₑ (by mᵢ/mₑ ≈ 3700 in a quasineutral hydrogen plasma).

uᵢ is carried by the operators that moved the ion densities this step ([`ion_momentum_carriage`](@ref)):
convection and the diffusive carriage with viscosity, by mass, θ-weighted with `θ_imp.decay` as the
electrons' u∥, in the solve that takes the drag; the near-vacuum relaxation as the electrons'. With
nothing carried and no node to relax, the update is the local one alone.
"""
function update_ui_para!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    return @timeit RAPID_TIMER "update_ui_para!" begin
        update_charge_states!(RP)   # ρ and the means of the densities as they are now
        pla = RP.plasma
        dt = RP.dt
        ee = RP.config.constants.ee
        one_FT = one(FT)
        ρ = pla.mni_tot

        # Births at rest first: ρ u stays as it was, now carried by more mass
        if RP.flags.src
            Δρ = born_ion_mass(RP)
            @. pla.ui_para *= ifelse(ρ > zero(FT), max(zero(FT), one_FT - Δρ / ρ), one_FT)
        end

        # The neutral drag, one rate for every species (the H₂⁺ surfaces); ½ on elastic: those
        # collisions shed only half the momentum
        ν_a = zeros(FT, size(pla.ui_para))
        if RP.flags.Atomic_Collision
            K_ela = get_H2_ion_RRC(RP, RP.iRRCs, :Elastic)
            K_cx = get_H2_ion_RRC(RP, RP.iRRCs, :Charge_Exchange)
            @. ν_a = pla.n_H2_gas * (FT(0.5) * K_ela + K_cx)
            replace!(ν_a, NaN => zero(FT))
        end

        # What carries uᵢ: the operators that moved the densities, nothing if none did; and the
        # near-vacuum relaxation, u → the local plasma's mean where there is almost no plasma
        carriage = ion_momentum_carriage(RP)
        relax_weight = vacuum_relaxation_weights(RP, pla.ni_tot)
        transported = !isnothing(carriage) && (!isnothing(carriage.A_adv) || !isnothing(carriage.M))

        # The parallel field on the ions' charge per mass, e Σ zₛ nₛ / Σ mₛ nₛ (the means weight
        # by density, so one species gives e z/m exactly)
        @with_pool pool begin
            Z_mean = ion_mean!(acquire!(pool, FT, size(pla.ni)...), RP, (z, m) -> z)
            accel = @. dt * ee * Z_mean * RP.fields.E_para_tot / pla.mi_mean
            decay_is_exprb = RP.flags.scheme.decay === ExpRB
            if !transported && !any(>(zero(FT)), relax_weight)
                # the local update alone
                if decay_is_exprb
                    # the exponent carries the drag only: the e–i friction is the recorded impulse below
                    decay_exponent = @. exprb_cap_exponent(-ν_a * dt)
                    bern_decay = exprb_bern.(decay_exponent)
                    @. pla.ui_para = ((bern_decay + decay_exponent) * pla.ui_para + accel) / bern_decay
                else
                    @. pla.ui_para = (pla.ui_para + accel) / (one_FT + dt * ν_a)   # backward Euler
                end
            else
                carry_ui_para!(RP, carriage, ν_a, accel, relax_weight, decay_is_exprb)
            end
        end

        # The electrons' friction
        if RP.flags.Coulomb_Collision
            @. pla.ui_para -= ifelse(ρ > zero(FT), pla.I_ei / ρ, zero(FT))
        end
        return RP
    end
end

# The ion mass this step's reactions created, per volume: Σₛ mₛ ΔNₛ over the species, from the
# published reaction counts. Creations only: an ion destroyed takes its momentum with it.
# INTERIM(diz-ion-species): H₂⁺'s count holds DI's ions, so they are born with H₂⁺'s mass. Why,
# and what reverts when H⁺ becomes transportable: `REACTION_STOICHIOMETRY.diz`.
function born_ion_mass(RP::RAPID{FT}) where {FT <: AbstractFloat}
    counts = check_reaction_counts(RP)
    Δρ = zeros(FT, size(RP.plasma.ni))
    for species in RP.transport.ion_species
        Δn = net_ion_count(counts, species.key)
        isnothing(Δn) && continue
        @. Δρ += species.mass * max(Δn, zero(FT))
    end
    return Δρ
end

# The electrons' momentum from the ions over the step, per volume, at the density their momentum
# equation saw; zero without Coulomb collisions. The ions take −I_ei (`update_ui_para!`). And the
# friction's heat, per electron: the kinetic energy it took from the relative drift over the step,
# −I_ei·(ū_e − u_i)/(nₑ Δt), ū_e the mean of `ue_entry` (uⁿ) and the solved u∥. Braginskii's
# −R·(u_e − u_i): the electrons' own kinetic energy, −u_e·R, plus the moving ions' work on them, u_i·R.
# Exact for any ν Δt, so a stiff friction gives the drift's energy, not ν Δt times it. Floored at zero:
# where another force (E∥, the gas) reverses the relative drift within the step, the impulse at u^θ on
# the step's mean drift would be a cooling no collision makes; without one, nothing reverses it.
function record_ei_impulse!(RP::RAPID{FT}, ue_entry::AbstractMatrix{FT}) where {FT <: AbstractFloat}
    pla = RP.plasma
    if RP.flags.Coulomb_Collision
        me = RP.config.constants.me
        @. pla.I_ei = RP.dt * me * pla.ne * pla.Rue_ei
        @. pla.ei_friction_heat = max(zero(FT), -me * pla.Rue_ei * (FT(0.5) * (ue_entry + pla.ue_para) - pla.ui_para))
    else
        fill!(pla.I_ei, zero(FT))
        fill!(pla.ei_friction_heat, zero(FT))
    end
    return RP
end

"""
    ei_exchange_energy!(RP) -> Matrix

The energy the electrons give the ions by Coulomb collisions over the step, per volume
[J/m³], into `plasma.ei_exchange_energy`: the exact solution of their linear relaxation

    (3/2) e nₑ dTₑ/dt = −Q,   (3/2) e n_I dTᵢ/dt = +Q,   Q = (3/2) e nₑ a (Tₑ − Tᵢ)

over Δt from the temperatures and densities as they are, with
a = Σₛ 2mₑmₛ/(mₑ+mₛ)² νₑₛ and νₑₛ = ν_ei zₛ² nₛ/Σz²n ([`ei_relaxation_rate!`](@ref)):

    ΔE = (3/2) e nₑ a (Tₑ − Tᵢ) (1 − e^{−λΔt})/λ,   λ = a [Tₑ evolves] + a nₑ/n_I [Tᵢ evolves]

A temperature that does not evolve is a bath: it does not move and leaves λ. Zero without
Coulomb collisions and outside the wall. Its mean power over the step, per electron and per ion,
goes to `ePowers.equi` and `iPowers.equi` [W], the exchange the powers' `tot` leaves out.
[`apply_ei_exchange!`](@ref) gives it to both temperatures.
"""
function ei_exchange_energy!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    pla = RP.plasma
    ΔE = pla.ei_exchange_energy
    if !RP.flags.Coulomb_Collision
        fill!(ΔE, zero(FT))
        fill!(pla.ePowers.equi, zero(FT))
        fill!(pla.iPowers.equi, zero(FT))
        return ΔE
    end
    update_charge_states!(RP)
    ee = RP.config.constants.ee
    te, ti = RP.flags.Te_evolve, RP.flags.Ti_evolve
    @with_pool pool begin
        a = ei_relaxation_rate!(acquire!(pool, FT, size(pla.ni)...), RP)
        ΔE .= relaxation_energy.(a, pla.ne, pla.ni_tot, pla.Te_eV, pla.Ti_eV, RP.dt, te, ti, ee)
    end
    for k in RP.G.nodes.on_out_wall_nids
        ΔE[k] = zero(FT)
    end
    @. pla.ePowers.equi = ifelse(pla.ne > zero(FT), ΔE / (RP.dt * pla.ne), zero(FT))
    @. pla.iPowers.equi = ifelse(pla.ni_tot > zero(FT), ΔE / (RP.dt * pla.ni_tot), zero(FT))
    return ΔE
end

"""
    apply_ei_exchange!(RP) -> RP

The e–i exchange over the step as a substep of its own, after both temperatures' other terms
(`solve_timestep!`): [`ei_exchange_energy!`](@ref) from the temperatures those left, taken by the
ions and given by the electrons, each only if it evolves. The relaxation moves the two toward each
other without crossing, so it cannot carry a temperature its own terms cooled past the floor.
"""
function apply_ei_exchange!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    ΔE = ei_exchange_energy!(RP)
    RP.flags.Coulomb_Collision || return RP
    pla = RP.plasma
    ee = RP.config.constants.ee
    RP.flags.Te_evolve &&
        @. pla.Te_eV -= ifelse(pla.ne > zero(FT), ΔE / (FT(1.5) * ee * pla.ne), zero(FT))
    RP.flags.Ti_evolve &&
        @. pla.Ti_eV += ifelse(pla.ni_tot > zero(FT), ΔE / (FT(1.5) * ee * pla.ni_tot), zero(FT))
    return RP
end

"""
    ei_relaxation_rate!(a, RP) -> a

Σₛ 2mₑmₛ/(mₑ+mₛ)² νₑₛ node by node, νₑₛ = ν_ei zₛ² nₛ/Σz²n the electrons' collision rate on
species s: the rate at which the electron temperature relaxes onto the ions'. With one
species it is 2mₑmᵢ/(mₑ+mᵢ)² ν_ei.
"""
function ei_relaxation_rate!(a::AbstractMatrix{FT}, RP::RAPID{FT}) where {FT <: AbstractFloat}
    pla = RP.plasma
    me = RP.config.constants.me
    reduced(m) = FT(2) * m * me / (m + me)^2
    z1, m1 = FT(bulk_ion_charge(RP)), bulk_ion_mass(RP)
    # the weights zₛ² nₛ/Σz²n first, so one species gives its own coefficient exactly
    @. a = ifelse(pla.Z2ni_tot > zero(FT), (z1^2 * pla.ni / pla.Z2ni_tot) * reduced(m1), reduced(m1))
    foreach_other_ion(RP) do n, z, m
        @. a += ifelse(pla.Z2ni_tot > zero(FT), (z^2 * n / pla.Z2ni_tot) * reduced(m), zero(FT))
    end
    @. a *= pla.ν_ei
    return a
end

# (3/2) e nₑ a (Tₑ − Tᵢ) (1 − e^{−λΔt})/λ, the window → Δt as λ → 0; nothing where either
# population is absent
@inline function relaxation_energy(a, ne, nI, Te, Ti, dt, Te_evolves::Bool, Ti_evolves::Bool, ee)
    (ne > zero(ne) && nI > zero(nI)) || return zero(a)
    λ = (Te_evolves ? a : zero(a)) + (Ti_evolves ? a * ne / nI : zero(a))
    window = λ * dt > zero(a) ? -expm1(-λ * dt) / λ : dt
    return oftype(a, 1.5) * ee * ne * a * (Te - Ti) * window
end

"""
    update_Te!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Update the electron temperature based on energy balance equation.

This function evolves the electron temperature by solving the electron energy equation:
3/2 n_e ∂T_e/∂t = P_heat - P_loss + ∇·(κ_e ∇T_e) - 3/2 n_e u_e·∇T_e

where:
- P_heat includes ohmic heating and other power sources
- P_loss includes ionization, excitation and radiation losses
- κ_e is the thermal conductivity
- The last term represents convective transport

The e–i exchange is not here: it couples Tₑ and Tᵢ, and `solve_timestep!` takes it as a substep
of its own after both updates ([`apply_ei_exchange!`](@ref)).

The implementation supports both explicit and implicit time integration schemes.
"""
function update_Te!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    @timeit RAPID_TIMER "update_Te!" begin
        # Update power terms for energy equation
        update_electron_heating_powers!(RP)

        ee = RP.config.constants.ee
        dt = RP.dt
        pla = RP.plasma
        OP = RP.operators
        # Only the transport terms of the energy equation are θ-weighted here.
        # The atomic power enters `ePowers.tot`, which carries no operator for a θ
        # to multiply — `scheme.atomic` is what decides its treatment instead.
        θimp = RP.flags.θ_imp.transport

        two_thirds_FT = FT(2.0) / FT(3.0)

        # the near-vacuum relaxation, from the density and Te at this stage's entry
        relax_weight = vacuum_relaxation_weights(RP, pla.ne)
        Te_ref = vacuum_reference(RP, pla.ne, pla.Te_eV, relax_weight)

        # `bern(z)` with `z = exprb.eig_Te·Δt`. Branched rather than relying on bern(0) = 1, so
        # the off path does no arithmetic at all and "unchanged" holds structurally.
        atomic_is_exprb = RP.flags.scheme.atomic === ExpRB
        bern_atomic = if atomic_is_exprb
            update_electron_power_jacobian!(RP)
            atomic_exponent = @. exprb_cap_exponent(pla.exprb.eig_Te * dt)
            _warn_if_exprb_capped(atomic_exponent)
            exprb_bern.(atomic_exponent)
        else
            nothing
        end

        # Apply time integration method based on flag
        if RP.flags.Implicit
            A = OP.A_LHS   # the shared LHS buffer on the wall pattern: values only
            set_identity!(A)

            ePowers_tilde = copy(pla.ePowers.tot)

            # Calculate RHS
            # ePowers_tilde = ePowers already known at crruent time t
            # Note: diffu and conv will have (1-θimp) contribution
            # ePowers_tilde = pla.ePowers.tot - θimp * (pla.ePowers.diffu + pla.ePowers.conv)

            # Calculate LHS from the in-wall operators (`electron_operators.jl`)
            pops = ue_Te_operators(RP)
            if RP.flags.Include_Te_diffu_term
                # P_diffu = 1.5*∇·D∇Te
                @. ePowers_tilde -= θimp * pla.ePowers.diffu
                add_scaled!(A, -two_thirds_FT * FT(1.5) * (dt * θimp), pops.A_diffu)
            end

            if RP.flags.Include_Te_convec_term
                @. ePowers_tilde -= θimp * pla.ePowers.conv
                # P_conv = -1.5*(𝐮⋅∇)Te − Te*(∇⋅𝐮)   (≡ -1.5*∇⋅(𝐮 Te) + 0.5*Te*∇⋅𝐮): on the LHS
                # +1.5·c·A_adv and +c·diag(∇⋅𝐮), with c = (2/3)·Δt·θ
                c_conv = two_thirds_FT * (dt * θimp)
                add_scaled!(A, c_conv * FT(1.5), pops.A_adv)
                add_diagonal!(A, vec(pops.div_u); scale = c_conv)
            end

            # LHS written as a deviation from the identity, so the sparsity
            # pattern — and the cached symbolic factorization — is untouched.
            # The source needs no bern: with S = (2/3e)P⁰ − λTₑⁿ, bern(−z) = bern(z) + z
            # cancels the λTₑⁿ pieces, leaving today's RHS with Tₑⁿ scaled by bern.
            if atomic_is_exprb
                add_diagonal!(A, vec(bern_atomic) .- one(FT))
                OP.RHS .= bern_atomic .* pla.Te_eV +
                    two_thirds_FT * (dt * ePowers_tilde / ee)
            else
                OP.RHS .= pla.Te_eV + two_thirds_FT * (dt * ePowers_tilde / ee)
            end

            # Te → the local plasma's Te where there is almost no plasma (backward Euler)
            add_diagonal!(A, vec(relax_weight))
            @. OP.RHS += ifelse(relax_weight > 0, relax_weight * Te_ref, zero(FT))

            # Solve the linear system (cached factorization; pattern is step-stable)
            @timeit RAPID_TIMER "Te_eV LinearSolve" begin
                factorize!(OP.Te_solver, A)
                solve!(view(pla.Te_eV, :), OP.Te_solver, view(OP.RHS, :))
            end
        elseif atomic_is_exprb
            # Forward Euler with bern as a divisor on the increment. With the near-vacuum
            # relaxation the diagonal is bern + relax_weight, as in the implicit branch.
            @. pla.Te_eV = ifelse(
                relax_weight > 0,
                (bern_atomic * pla.Te_eV + two_thirds_FT * pla.ePowers.tot * dt / ee + relax_weight * Te_ref) / (bern_atomic + relax_weight),
                pla.Te_eV + two_thirds_FT * pla.ePowers.tot * dt / ee / bern_atomic,
            )
        else
            # Explicit method (forward Euler); the near-vacuum relaxation stays backward Euler
            @. pla.Te_eV = ifelse(
                relax_weight > 0,
                (pla.Te_eV + two_thirds_FT * pla.ePowers.tot * dt / ee + relax_weight * Te_ref) / (one(FT) + relax_weight),
                pla.Te_eV + two_thirds_FT * pla.ePowers.tot * dt / ee,
            )
        end

        # the thermal energy the near-vacuum relaxation took out (before the limits below)
        RP.diagnostics.near_vacuum.relax_energy +=
            FT(1.5) * ee * vacuum_relaxation_exchange(RP, pla.ne, relax_weight, pla.Te_eV, Te_ref)

        # Apply temperature limits
        @. pla.Te_eV = max(pla.Te_eV, RP.config.min_Te)
        @. pla.Te_eV = min(pla.Te_eV, RP.config.max_Te)

        return RP
    end # @timeit
end

"""
    update_Ti!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Advance the ion temperature, `(3/2) dTᵢ/dt = P_atomic − (3/2)(u·∇)Tᵢ − Tᵢ∇·u + (3/2)𝒯_D(Tᵢ) + P_visc` per ion
(reference (4.3), (4.8)). Carried by the operators that moved the ion densities
([`ion_heat_carriage`](@ref)), it is solved as the electrons' Tₑ ([`carry_Ti!`](@ref)); with nothing
carried and no node to relax, forward Euler on the atomic power (bern dividing the increment under
ExpRB). Then clamped to the electrons' bounds. The e–i exchange is not here: it couples Tₑ and Tᵢ, and
`solve_timestep!` takes it as a substep of its own after both updates ([`apply_ei_exchange!`](@ref)).
"""
function update_Ti!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    @timeit RAPID_TIMER "update_Ti!" begin
        # What carries Tᵢ (the operators that moved the densities, nothing if none did), and the
        # ion heating powers with it
        heat = ion_heat_carriage(RP)
        update_ion_heating_powers!(RP; heat)

        ee = RP.config.constants.ee
        dt = RP.dt
        pla = RP.plasma
        relax_weight = vacuum_relaxation_weights(RP, pla.ni_tot)
        transported = !isnothing(heat) && (!isnothing(heat.A_adv) || !isnothing(heat.M))

        if transported || any(>(zero(FT)), relax_weight)
            carry_Ti!(RP, heat, relax_weight)
        elseif RP.flags.scheme.atomic === ExpRB
            # Forward Euler with bern as a divisor on the increment, the same treatment
            # `update_Te!` gives the electron power and for the same reason: `iPowers.tot`
            # is a number from a table lookup with no operator for a θ to weigh.
            update_ion_power_jacobian!(RP)
            atomic_exponent = @. exprb_cap_exponent(pla.exprb.eig_Ti * dt)
            _warn_if_exprb_capped(atomic_exponent)
            @. pla.Ti_eV +=
                (FT(2.0) / FT(3.0)) * pla.iPowers.tot * dt / ee / exprb_bern(atomic_exponent)
        else
            @. pla.Ti_eV += (FT(2.0) / FT(3.0)) * pla.iPowers.tot * dt / (ee)
        end

        # Apply temperature limits (same as electrons for simplicity)
        @. pla.Ti_eV = max(pla.Ti_eV, RP.config.min_Te)
        @. pla.Ti_eV = min(pla.Ti_eV, RP.config.max_Te)

        return RP
    end # @timeit
end

"""
    mean_energy_floor(RP) -> FT

The bottom of the rate tables' `Ē` axis [eV] — below it `RRC_EoverP_Erg`'s
`ClampExtrap` freezes every coefficient and the interpolant stops meaning anything.

**A function barrier, not a convenience.** `RAPID.eRRCs` is declared as the abstract
`AbstractSpeciesRRCs{FT}`, and one `Any` scalar inside a `@.` costs the fused kernel its
specialization; read through [`erg_axis_bounds`](@ref), the floor is a concrete `FT`.
Hoisting also avoids the `first` trap: `Number` is iterable, so `first` under `@.` would
dot into `first.(Erg_eV)` — the identity on a vector, not a scalar floor.

Read off `Kerg_ela`, but it is every surface's floor: all of them share the one
`Erg_eV` vector the constructor read.
"""
@inline mean_energy_floor(RP::RAPID{FT}) where {FT <: AbstractFloat} =
    first(erg_axis_bounds(RP, :Kerg_ela))

"""
    cold_target_factor(Ē, T_gas_eV, Ē_floor)
    cold_target_slope(Ē, T_gas_eV, Ē_floor)

The `(1 − (3/2)T_gas/Ē)` that a cold-target energy ledger is consumed with, and its
`∂/∂Tₑ` — which is `∂/∂Ē` times `3/2`, since `Ē = (3/2)Tₑ + ½mₑu∥²/e`.

Defined as a pair because they are used as a pair: `P_ela` and `P_exc` share one
factor, and every eigenvalue path has to differentiate the factor those powers
actually applied. Two transcriptions of the same algebra is how the residual and its
Jacobian drift apart.

**The slope is not `2.25·T_gas/Ē²`.** Below `Ē_floor` the factor is frozen by `max`,
so its derivative there is exactly zero; differentiating the clamp instead reports
damping the power does not have.
"""
@inline cold_target_factor(Ē::T, T_gas_eV, Ē_floor) where {T} =
    one(T) - T(1.5) * T_gas_eV / max(Ē, Ē_floor)

@inline cold_target_slope(Ē::T, T_gas_eV, Ē_floor) where {T} =
    ifelse(Ē > Ē_floor, T(2.25) * T_gas_eV / Ē^2, zero(T))

"""
    update_electron_heating_powers!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Update electron heating power components for electron energy equation.

# Arguments
- `RP::RAPID{FT}`: The RAPID object containing simulation state

# Returns
- `RP`: The updated RAPID object

# Notes
- Calculates all electron power sources and sinks:
  - Diffusion and convection powers (if enabled)
  - Collision drag power
  - Elastic energy loss to neutrals (2me/M per momentum-transfer collision)
  - Heat generation from density gradients
  - Ionization, dissociative ionization, excitation, dissociative excitation,
    and dilution powers
- All powers stored in the RP.plasma.ePowers struct, except the e–i exchange `equi`, which
  [`ei_exchange_energy!`](@ref) writes
"""
function update_electron_heating_powers!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    @timeit RAPID_TIMER "update_electron_heating_powers!" begin
        # Extract physical constants + reaction energies (all in RP.config.constants).
        @unpack ee, qe, me, iz_erg_eV, diss_iz_erg_eV = RP.config.constants
        # The elastic recoil needs no separate neutral mass here: `P_en_ela` already
        # carries 2mₑ/M internally (see the ela block).

        # Alias common objects for readability
        pla = RP.plasma
        ePowers = RP.plasma.ePowers

        zero_FT = zero(FT)

        # Reset all power arrays to zero (precaution to avoid accumulation)
        ePowers.diffu .= zero_FT
        ePowers.conv .= zero_FT
        ePowers.heat .= zero_FT
        ePowers.drag .= zero_FT
        ePowers.ela .= zero_FT
        ePowers.iz .= zero_FT
        ePowers.exc .= zero_FT
        ePowers.diss_exc .= zero_FT
        ePowers.diss_iz .= zero_FT
        ePowers.dilution .= zero_FT
        ePowers.visc_heat .= zero_FT

        # The electron in-wall operators, cached once per step (`operators.A_conv_e`, `A_diffu_e`,
        # `div_ue`; rows on in-wall nodes only, so the excluded band is never read) and
        # checked to carry the current `upwind` scheme. These are right-hand sides, so (u·∇)
        # is applied matrix-free from the current `ne` rather than assembled — the assembly
        # is what the implicit solves pay for.
        op_e = electron_operator_cache(RP)
        n_e = vec(pla.ne)
        n_floor = one(FT)
        u∇(f) = reshape(apply_advection(op_e.A_conv_e, n_e, vec(f); n_floor), size(f))

        # The viscous heating is the energy partner of u∥'s diffusion, so it follows that and has
        # no switch of its own: on exactly when u∥ is diffused (off, no kinetic energy is erased;
        # on without it, the erased energy would be lost).
        visc_heat_on = RP.flags.ud_evolve && RP.flags.Include_ud_diffu_term
        # SurfaceFluxConstraint: on the closed region Te rides the stream-function flux and is
        # compressed by the pressure operator's adjoint, which only the operators `update_Te!`
        # solves with carry
        surface_carriage = RP.flags.Include_Te_convec_term && closed_surface_active(RP) && !isempty(RP.closed_surface.solve.Γe_para_over_B)
        # one build of the operators Te is solved with, for the powers below that take them
        pops = RP.flags.Include_Te_diffu_term || visc_heat_on || surface_carriage ? ue_Te_operators(RP) : nothing
        A_diffu = pops === nothing ? nothing : pops.A_diffu

        # If diffusion term is included in temperature equation
        if RP.flags.Include_Te_diffu_term
            # P_diffu = 1.5*(A_diffu Te): the operator `update_Te!` puts on its left-hand side,
            # so the explicit and the implicit halves of the θ-scheme agree
            ePowers.diffu .= ee * FT(1.5) * (A_diffu * pla.Te_eV)
        end

        # The viscous heating, mₑ Γ_M(u∥) per electron (−Π:∇u / nₑ with Π = −mₑ nₑ D ∇u∥): the
        # kinetic energy of the parallel flow the diffusion of momentum erased, from the
        # operator Te diffuses with and the step's u∥ (`electron_operators.jl`).
        if visc_heat_on
            viscous_heating!(ePowers.visc_heat, RP; M = A_diffu)
        end

        # If convection term is included in temperature equation
        if RP.flags.Include_Te_convec_term
            # P_conv = -1.5*∇⋅(𝐮 Te) + 0.5*Te*(∇⋅𝐮)  =  -1.5*(𝐮⋅∇)Te − Te*(∇⋅𝐮)
            if !surface_carriage
                ePowers.conv .= ee * (-FT(1.5) * u∇(pla.Te_eV) .- pla.Te_eV .* op_e.div_ue)
            else
                # the operators `update_Te!` puts on its left-hand side, so the explicit and the
                # implicit halves of the θ-scheme agree on the closed region too
                ePowers.conv .= ee * (-FT(1.5) * (pops.A_adv * pla.Te_eV) .- pla.Te_eV .* pops.div_u)
            end
        end

        if RP.flags.Include_heat_flux_term
            # NOTE: Assumption: 𝐪 ≈ p*𝐮 (heat flux is about in the order of pressure*velocity)
            # Pcond = Pheat = -∇⋅(Te * 𝐮) - Te*𝐮⋅∇(ln_n)
            #       = −(𝐮⋅∇)Te − Te ∇⋅𝐮 − Te (𝐮⋅∇) ln n, every term from the in-wall operators.
            # The floor keeps log finite on empty rows (ne = 0); those rows contribute nothing.
            ln_n = log.(max.(pla.ne, one(FT)))
            ePowers.heat .= ee * (-u∇(pla.Te_eV) .- pla.Te_eV .* op_e.div_ue .- pla.Te_eV .* u∇(ln_n))
        end


        # The e–i friction's heat, Braginskii's −R·(u_e − u_i) → mₑ ν_ei^eff |u_e − u_i|², the sum of two
        # terms of the electrons' energy: −u_e·R, the kinetic energy the friction takes from them, and
        # +u_i·R, the work of the moving ions (elastic scattering off a heavy ion keeps |v − u_i|). The
        # ions get the friction's work and no heat, to order mₑ/mᵢ. With u∥ evolved it is what the
        # momentum solve's friction took over the step (`record_ei_impulse!`); a prescribed drift's at once.
        if RP.flags.Coulomb_Collision
            if RP.flags.ud_evolve
                ePowers.drag .= pla.ei_friction_heat
            else
                @. ePowers.drag = me * ((pla.ueR - pla.uiR)^2 + (pla.ueϕ - pla.uiϕ)^2 + (pla.ueZ - pla.uiZ)^2) * pla.ν_ei_eff
            end
        end

        if RP.flags.Atomic_Collision
            # Every ν_en_* below is materialized by update_RRCs! at the step-entry state.
            # Reading them rather than re-querying is what makes the frictional heating
            # credited here use the *same* coefficient the momentum equation used to
            # remove that momentum, so the discrete energy budget closes.

            # Calculate velocity magnitudes for drag forces
            ue_mag_sq = @. pla.ueR^2 .+ pla.ueϕ^2 .+ pla.ueZ^2

            # the gas's drag heats by mₑ|u_e|²ν_mom: the gas is at rest, so −u_e·R is all of it
            @. ePowers.drag += me * ue_mag_sq * pla.ν_en_mom_tot

            # Elastic recoil. `P_en_ela = n_gas·Kerg_ela` already contains 2mₑ/M and the
            # e — do NOT reapply them. It is a COLD-TARGET coefficient (molecule at rest),
            # so unlike a (Tₑ − T_gas) form it does not vanish at thermal equilibrium; the
            # factor below is what restores that, and it is not optional. Substituting
            # Ē_ela ≃ Ē shows the factor IS (Tₑ − T_gas), recovered:
            #
            #   n_g·Kerg_ela·(1 − (3/2)T_gas/Ē)
            #     ≃ (2mₑ/M)·ν_mom_ela·[ (3/2)(Tₑ − T_gas)e + ½mₑu∥² ]
            #
            # — including the ½mₑu∥² the old form lost by approximating Ē as (3/2)Tₑ.
            # Below T_gas the factor is negative and the gas HEATS the electrons, which
            # the relaxation testitem's cold branch exercises.
            #
            # Ē is the table's own query coordinate, rebuilt here rather than read: it is
            # the same expression `_eRRC_query_point` uses.
            # The 1/Ē divergence is self-limiting only INSIDE the table: there,
            # Kerg_ela ∝ Ē (the Ē_ela ≃ Ē identity in the comment above), so the product
            # stays finite as Ē → 0. Below the table's bottom row, `RRC_EoverP_Erg.itp`
            # is ClampExtrap and freezes `Kerg_ela` at a finite boundary value, so the
            # cancellation fails and the factor diverges -- `max(Ē_eV, eps(FT))` would
            # turn that into a `-1.8e14`-scale factor instead of the intended guard.
            # Floor at the table's own bottom Ē row instead: the honest clamp, since
            # that is the point below which the interpolant stops meaning anything
            # (bounds the factor to ~-38 rather than ~-1.8e14). Ē_eV = 1e-3 eV is
            # Tₑ ≈ 7.7 K, so this floor guards a diverging solve, not a physical regime.
            #
            # Shared with the excitation term below -- same factor, same floor, computed
            # once -- so elastic and excitation cannot drift apart.
            Ē_floor = mean_energy_floor(RP)
            # The factor is parked in `ePowers.ela` as SCRATCH rather than built in a
            # fresh grid array, and Ē is folded into the same fused kernel instead of
            # being materialised first. This function runs every step whatever the
            # scheme flags say, and those two temporaries were the only per-step
            # allocation this branch added; both are gone. Ē is not transcribed again
            # either — it appears here exactly as often as before.
            #
            # **The factor is consumed at the excitation site below, both times.**
            # Keeping one array is what makes "same factor, same floor, computed once"
            # literal rather than a promise about two call sites agreeing.
            @. ePowers.ela = cold_target_factor(
                FT(1.5) * pla.Te_eV + FT(0.5) * me * pla.ue_para^2 / ee,
                pla.T_gas_eV, Ē_floor
            )

            # Excitation, tabulated. No constant survives here: the EXC group spans
            # 0.0441 eV (rot) to 14.9 eV, a factor 338 in per-event cost, so its mean cost
            # is a function of where the distribution sits and ran 0.059-9.72 eV across
            # the operating range against the 12.0 eV this used to hard-code.
            #
            # DEVIATION FROM THE LEDGER'S WRITTEN CONTRACT -- deliberate, and loud on
            # purpose. `L_exc`'s `consume_as` attribute in the HDF5 file says
            # `P = n_e * n_gas * L_exc`, no factor. Every other term in this function
            # consumes its ledger column literally; this is the one place that does not,
            # so it must announce itself or a reader cross-checking BD's own docs will
            # read the mismatch as a bug.
            #
            # Why: `Kerg_exc`, like `Kerg_ela`, is a ONE-WAY coefficient -- computed
            # against a stationary, ground-state H2 -- so with no correction it keeps
            # draining energy at Tₑ = T_gas, which is why (pre-fix) this equation settled
            # Tₑ at ~0.0111 eV instead of relaxing to T_gas. Only rotation matters here:
            # ΔE_rot = 0.0441 eV is 1.7x room_T_eV, so a real thermal population is
            # already pre-excited at 300 K; vibration (0.516 eV) and the electronic
            # channels (>= 11.2 eV) have no such population and need no correction.
            #
            # The form below is PHENOMENOLOGICAL, not exact detailed balance. The exact
            # rotational factor is `1 - exp(ΔE/Tₑ - ΔE/T_gas)`, but applied to the WHOLE
            # EXC group (rotation is not exported separately) it does not return to 1 at
            # high Tₑ -- it plateaus at 0.816, since superelastic never stops and its
            # rate relative to excitation is fixed by the gas Boltzmann ratio, not by Tₑ.
            # That would cut electronic excitation by a permanent 18.4%. The linear
            # cold-target form shares both correct limits (0 at Tₑ = T_gas, -> 1 at high
            # Tₑ) and costs only 0.28% at Tₑ = 9.2 eV instead of 18.4%; in the band where
            # rotation IS the whole group (Tₑ ~ 0.05-0.15 eV) the linear and exact forms
            # agree within ~15% (14% at 0.05 eV, 10% at 0.15 eV), so this is not
            # over-correcting where it matters. That agreement narrows fast below 0.05 eV
            # -- 34% low at Tₑ = 0.03 eV -- which is why the band above starts there and
            # not lower.
            #
            # Retire this the day BD exports `L_exc_rot` separately: the exact factor
            # can then be applied to the rotational part alone, leaving the rest of EXC
            # (vibration, electronic) on the literal `consume_as` contract.
            # `ePowers.ela` still holds the shared cold-target factor here.
            #
            # ORDER IS LOAD-BEARING: excitation must consume it BEFORE elastic
            # overwrites it. Swapping these two lines silently gives excitation a
            # spurious `P_en_ela` factor — caught by physics_test.jl's
            # "ePowers.exc ≈ P_en_exc .* cold", which is why that test is not
            # redundant with the elastic one beside it.
            @. ePowers.exc = pla.P_en_exc * ePowers.ela
            @. ePowers.ela = pla.P_en_ela * ePowers.ela

            # Dissociative excitation, charged separately because its energy split from
            # EXC is not recoverable afterwards: a B-excited molecule costs its full
            # 11.184 eV whether or not it later dissociates, so that energy stays in exc.
            # Stays RAW, no cold-target factor: the DISS group is the triplets,
            # thresholds 8.9-11.8 eV, with no thermal population at 300 K to return
            # energy from -- there is no superelastic channel here to correct for.
            @. ePowers.diss_exc = pla.P_en_diss_exc

            # For ionization
            if RP.flags.src
                # Two channels, per-event costs a factor 2.3 apart. Each is a SINGLE
                # channel with a SINGLE threshold, which is the one case where
                # e·ε·ν reconstructs the loss exactly.
                @. ePowers.iz = pla.ν_en_iz * iz_erg_eV * ee
                @. ePowers.diss_iz = pla.ν_en_diss_iz * diss_iz_erg_eV * ee

                # Dilution: BOTH channels create exactly one electron per event.
                @. ePowers.dilution = (pla.ν_en_iz + pla.ν_en_diss_iz) * (
                    FT(1.5) * pla.Te_eV * ee
                        - FT(0.5) * me * ue_mag_sq
                )
            end
        end

        # Calculate total power (sum of all components; the e–i exchange is applied on its own,
        # and `ei_exchange_energy!` writes its power to `equi`)
        @. ePowers.tot = (
            ePowers.drag + ePowers.conv + ePowers.heat + ePowers.diffu + ePowers.visc_heat
                - ePowers.ela - ePowers.dilution - ePowers.iz - ePowers.diss_iz
                - ePowers.exc - ePowers.diss_exc
        )

        # # Zero out power values outside the wall
        on_out_wall_nids = RP.G.nodes.on_out_wall_nids
        if !isempty(on_out_wall_nids)
            @views ePowers.tot[on_out_wall_nids] .= zero_FT
            @views ePowers.diffu[on_out_wall_nids] .= zero_FT
            @views ePowers.conv[on_out_wall_nids] .= zero_FT
            @views ePowers.drag[on_out_wall_nids] .= zero_FT
            @views ePowers.ela[on_out_wall_nids] .= zero_FT
            @views ePowers.dilution[on_out_wall_nids] .= zero_FT
            @views ePowers.iz[on_out_wall_nids] .= zero_FT
            @views ePowers.diss_iz[on_out_wall_nids] .= zero_FT
            @views ePowers.exc[on_out_wall_nids] .= zero_FT
            @views ePowers.diss_exc[on_out_wall_nids] .= zero_FT
            @views ePowers.heat[on_out_wall_nids] .= zero_FT
            @views ePowers.visc_heat[on_out_wall_nids] .= zero_FT
        end

        return RP
    end # @timeit
end

"""
    update_electron_power_jacobian!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Write `plasma.exprb.eig_Te = (2/3e)·∂P/∂Tₑ` [1/s], the local eigenvalue of the electron
energy equation — signed, so `z = λΔt` covers relaxation and runaway with one formula.
How completely it is assembled is [`LinearResponseDepth`](@ref).

Term for term against [`update_electron_heating_powers!`](@ref) — **edit them
together**. `test/unit/physics/power_jacobian_test.jl` finite-differences the real
assembled power, which is what catches a term present there and missing here.

Not in `λ` at either depth: `P_visc_heat` depends on u∥, not on Tₑ, so its derivative is
zero; `P_diffu` and `P_conv` are nonlocal and keep
`θ_imp.transport`; `P_heat` is nonlocal with no implicit half at all, so it stays
forward Euler — warned about here, since it is the dispatch's omission and not one
branch's. `FullLinearResponse` additionally warns about `∂ν_ei/∂Tₑ` for a prescribed drift
(`ud_evolve = false`), the one derivative it declines to take (Spitzer-like, not an RRC
surface — it under-damps the transient only, since the fixed point does not depend on `λ`);
with u∥ evolved the friction's heat is the momentum solve's work, with no Tₑ in it. What
`PartialLinearResponse` leaves out is the enum's subject, not a warning's.

Two traps. `Ē` is built from `ue_para` while `P_drag` and `P_dilution` use
`ue_mag_sq`; they differ once `mean_ExB` or diamagnetic drifts are on. And
`ue_mag_sq` is one step stale here — see
`internal/docs/src/notes/issues/drag-heating-lags-the-momentum-solve.md`.
"""
function update_electron_power_jacobian!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    @timeit RAPID_TIMER "update_electron_power_jacobian!" begin
        RP.flags.scheme.atomic === ExpRB || return RP

        # Neither depth carries P_heat, so the announcement belongs here rather than
        # inside one of them — the branch that omits MORE was the silent one.
        if RP.flags.Include_heat_flux_term
            @warn "scheme.atomic = ExpRB omits P_heat from exprb.eig_Te: its −∇⋅(Tₑ𝐮) half is an " *
                "operator, not a diagonal entry, and its pointwise −Tₑ(𝐮⋅∇ln n) half was " *
                "not measured separately. Nothing weights P_heat implicitly either, so " *
                "with Include_heat_flux_term on that power alone advances at forward " *
                "Euler inside a fitted step." maxlog = 1
        end

        # One branch per step, not per cell. Split into two functions because the
        # two policies share no arithmetic — one reads rates, the other
        # differentiates surfaces.
        if RP.flags.exprb_eigenvalue === PartialLinearResponse
            return _eig_Te_from_known_rates!(RP)
        end
        return _eig_Te_from_linear_response!(RP)
    end # @timeit
end

"""
    _eig_Te_from_known_rates!(RP)

`λ_Tₑ` from the terms whose `Tₑ` is written down, and nothing else:

```
(3/2)e·dTₑ/dt = A − 𝔅·Tₑ,
𝔅 = (P_en_ela + P_en_exc)·(9/4)T_gas/Ē²  +  (3/2)e·(ν_iz + ν_diss_iz)
```

from `P_ela`, `P_exc` and `P_dilution`; the e–i exchange is not in `ePowers.tot`, and the step
applies its exact solution on its own ([`ei_exchange_energy!`](@ref)). Pre-migration, with
`P_ela ∝ (Tₑ − T_gas)` exact and no other explicit `Tₑ`, this rearrangement of
[`update_electron_heating_powers!`](@ref) was algebraically exact, not a
linearisation. **That is no longer true.** `P_en_ela` is now
`n_H2_gas·Kerg_ela(Ē)·(1 − 1.5·T_gas/Ē)` with `Ē` linear in `Tₑ`, so treating
`A − 𝔅·Tₑ` as the whole `Tₑ`-response is a local linearisation of the
cold-target factor — `Kerg_ela(Ē)` and `Kerg_exc(Ē)`'s own `Tₑ`-dependence is
dropped here (see the note below on what that costs). `𝔅` still sums rates that
are non-negative in any state the model describes, so `λ = −(2/3e)𝔅 ≤ 0`: no pole
and no growth branch. Not a licence to drop the exponent cap downstream — `ν_ei` is
built from `ni`, which the continuity solve can land marginally below zero — only a
statement that the branch is not where a positive `λ` comes from.
`P_drag`, `P_diss_exc`, `P_iz` and `P_diss_iz` carry no explicit `Tₑ` — and since
2026-08 neither do most of `P_ela` and `P_exc`, whose responses now live inside
`Kerg_ela(Ē)` and `Kerg_exc(Ē)`; only the cold-target factor they share is written
down here — see [`LinearResponseDepth`](@ref) for what the rest costs.
"""
function _eig_Te_from_known_rates!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    pla = RP.plasma
    @unpack ee, me = RP.config.constants
    zero_FT = zero(FT)

    fill!(pla.exprb.eig_Te, zero_FT)

    if RP.flags.Atomic_Collision
        # P_ela's explicit Tₑ is now ONLY the cold-target factor: the rest lives inside
        # Kerg_ela(Ē), which this depth discards by construction. Differentiating
        #     P_ela = P_en_ela·(1 − (3/2)T_gas/Ē),  Ē = (3/2)Tₑ + ½mₑu∥²/e
        # at frozen P_en_ela gives +(9/4)·T_gas/Ē², and ePowers.tot SUBTRACTS P_ela, so
        # it enters negative — damping, no growth branch, same guarantee as before.
        #
        # It is much weaker than the pre-2026-08 −(2mₑ/M)·ν_ela·(3/2)e this replaced,
        # because that term's Tₑ was explicit and this one's mostly is not. That is the
        # depth's definition applied honestly, not an omission: `FullLinearResponse` is
        # where the rest of the response lives.
        # P_exc carries the SAME factor, so it has an explicit Tₑ too and this depth
        # must pick it up — the sinks share one `cold_target_factor` in
        # update_electron_heating_powers!, and the Jacobian mirrors that sharing.
        # P_diss_exc stays raw and contributes nothing here.
        # One fused broadcast, no grid temporaries: this runs every step, and
        # `cold_target_slope` is the same slope `update_electron_heating_powers!`'s
        # factor has — including the zero below the floor.
        Ē_floor = mean_energy_floor(RP)
        @. pla.exprb.eig_Te -= (pla.P_en_ela + pla.P_en_exc) * cold_target_slope(
            FT(1.5) * pla.Te_eV + FT(0.5) * me * pla.ue_para^2 / ee,
            pla.T_gas_eV, Ē_floor
        )
        # Dilution, both electron-producing channels.
        RP.flags.src && @. pla.exprb.eig_Te -= FT(1.5) * ee *
            (pla.ν_en_iz + pla.ν_en_diss_iz)
    end
    # (the e–i exchange is not here: the step applies its exact solution on its own)

    # Indexed rather than `@views … .= `: the SubArray broadcast builds for that is a
    # heap allocation, small but per step, and this function's contract is that it
    # makes none. `update_electron_heating_powers!` already zeroes these nodes' powers;
    # this keeps the eigenvalue from describing a relaxation rate for them.
    for k in RP.G.nodes.on_out_wall_nids
        pla.exprb.eig_Te[k] = zero_FT
    end

    @. pla.exprb.eig_Te *= FT(2.0) / FT(3.0) / ee
    return RP
end

"""
    _eig_Te_from_linear_response!(RP)

`λ_Tₑ = (2/3e)·∂P/∂Tₑ` in full: every explicit `Tₑ` of
[`_eig_Te_from_known_rates!`](@ref) **plus** the chain rule through
`Ē = (3/2)Tₑ + ½mₑu∥²/e` into every rate coefficient. A superset, and the only
part of it that can be positive.

Reads `plasma.dν_dTe`, which only `update_RRCs!` writes and only under this depth —
so a depth switched on after the last rate step is refused rather than served zeros.
Refreshing here instead is not an option: it would query the tables at a state the
frequencies were not evaluated at, which is the invariant
`internal/docs/src/notes/design/rrc-single-evaluation-point.md` exists to keep.
"""
function _eig_Te_from_linear_response!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    @timeit RAPID_TIMER "_eig_Te_from_linear_response!" begin
        pla = RP.plasma
        pla.dν_dTe.fresh || throw(
            ArgumentError(
                "exprb_eigenvalue = FullLinearResponse needs ∂ν/∂Tₑ from the same rate " *
                    "evaluation as the frequencies, and the last update_RRCs! ran under a " *
                    "policy that does not materialize it — so these surfaces are zero or " *
                    "from an earlier state. Set the depth before initialize!, or run one " *
                    "update_transport_quantities! after changing it."
            )
        )
        @unpack ee, me, iz_erg_eV, diss_iz_erg_eV = RP.config.constants
        zero_FT = zero(FT)
        dν = pla.dν_dTe

        fill!(pla.exprb.eig_Te, zero_FT)

        if RP.flags.Atomic_Collision
            # Same velocity magnitude update_electron_heating_powers! charges the
            # drag and the dilution with — NOT ue_para, which is what Ē is built
            # from. The two differ as soon as a perpendicular drift is on.
            ue_mag_sq = @. pla.ueR^2 + pla.ueϕ^2 + pla.ueZ^2
            Ē_eV = @. FT(1.5) * pla.Te_eV + FT(0.5) * me * pla.ue_para^2 / ee
            Ē_floor = mean_energy_floor(RP)
            # The factor the powers applied, and its slope — one definition each, from
            # the same pair `update_electron_heating_powers!` consumes.
            cold = @. cold_target_factor(Ē_eV, pla.T_gas_eV, Ē_floor)
            dcold = @. cold_target_slope(Ē_eV, pla.T_gas_eV, Ē_floor)

            @. pla.exprb.eig_Te += (
                me * ue_mag_sq * dν.mom_tot                                   # P_drag
                    # P_ela: product rule across the coefficient AND the cold-target factor.
                    - (dν.ela_erg * cold + pla.P_en_ela * dcold)
                    # P_exc carries the SAME cold-target factor as P_ela since the
                    # excitation sink gained it, so its derivative is a product rule too
                    # -- the same `dcold` by construction rather than by transcription.
                    - (dν.exc_erg * cold + pla.P_en_exc * dcold)
                    - dν.diss_exc_erg                                         # P_diss_exc (raw, no factor)
            )

            if RP.flags.src
                ν_new = @. pla.ν_en_iz + pla.ν_en_diss_iz
                dν_new = @. dν.iz + dν.diss_iz
                @. pla.exprb.eig_Te += -(
                    ee * (iz_erg_eV * dν.iz + diss_iz_erg_eV * dν.diss_iz)    # P_iz, P_DI
                        + dν_new * (FT(1.5) * pla.Te_eV * ee - FT(0.5) * me * ue_mag_sq)
                        + FT(1.5) * ee * ν_new                                # P_dilution
                )
            end
        end

        # (the e–i exchange is not here: the step applies its exact solution on its own)

        # `ePowers.drag`'s Coulomb half: with u∥ evolved, the friction's heat the momentum solve recorded,
        # no Tₑ in it; a prescribed drift heats by mₑ ν_ei^eff(Tₑ)|u_e − u_i|² at once, its Tₑ through ν_ei
        # left out
        if RP.flags.Coulomb_Collision && !RP.flags.ud_evolve
            @warn "scheme.atomic = ExpRB omits ∂ν_ei/∂Tₑ from exprb.eig_Te: with u∥ not evolved the " *
                "friction heats by mₑ ν_ei^eff(Tₑ)|u_e − u_i|², and ν_ei is Spitzer-like, not an RRC " *
                "surface. This under-damps the transient; the fixed point is unaffected." maxlog = 1
        end

        # No power outside the wall, so no rate of change of it either.
        on_out_wall_nids = RP.G.nodes.on_out_wall_nids
        if !isempty(on_out_wall_nids)
            @views pla.exprb.eig_Te[on_out_wall_nids] .= zero_FT
        end

        # (2/3e)·∂P/∂Tₑ: the power is per electron [W] and Tₑ is in eV, so ee
        # converts J/eV and the 2/3 comes from (3/2)e·∂Tₑ/∂t = P.
        @. pla.exprb.eig_Te *= FT(2.0) / FT(3.0) / ee

        return RP
    end # @timeit
end

"""
    update_ion_power_jacobian!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Write `plasma.exprb.eig_Ti = (2/3e)·∂P_i/∂T_i` [1/s], the ion sibling of
[`update_electron_power_jacobian!`](@ref).

Term for term against [`update_ion_heating_powers!`](@ref) — edit them together.
With `ν_a` the effective atomic collision frequency and `ΔE` the per-collision
energy change it multiplies:

- `P_atomic = ν_a·ΔE` gives `∂ν_a/∂T_i·ΔE − ν_a·(3/2)e`. The second half needs no
  table at all; the first comes from [`ion_rate_jacobian`](@ref).
- `Z·ν_iz` inside `ν_a` is the ELECTRON ionization rate and carries no `T_i`
  dependence, so it contributes to `ν_a` but not to `∂ν_a/∂T_i`.

The e–i exchange is not in `iPowers.tot`: the step applies its exact solution on its own
([`ei_exchange_energy!`](@ref)).

Rate coefficients are re-queried rather than read from `plasma`: unlike the electron
frequencies the ion ones are never materialized, so evaluating at the same
`(T_i, |u_i∥|)` is what keeps the Jacobian on the power it linearises.
"""
function update_ion_power_jacobian!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    @timeit RAPID_TIMER "update_ion_power_jacobian!" begin
        pla = RP.plasma
        RP.flags.scheme.atomic === ExpRB || return RP

        @unpack ee, me = RP.config.constants
        mi = bulk_ion_mass(RP)
        zero_FT = zero(FT)
        fill!(pla.exprb.eig_Ti, zero_FT)

        # The ion split falls out of the product rule: `P_atomic = ν_a·ΔE` gives
        # `−ν_a·(3/2)e` from ΔE's explicit T_i — the stated rate — and `dν_a·ΔE`
        # from the tables. Only the second is a linear response.
        linear_response = RP.flags.exprb_eigenvalue === FullLinearResponse

        if RP.flags.Atomic_Collision
            K_ela, K_cx = get_H2_ion_RRC(RP, :Elastic), get_H2_ion_RRC(RP, :Charge_Exchange)
            ν_a = @. pla.n_H2_gas * (FT(0.5) * K_ela + K_cx)
            if RP.flags.src
                Z_i = FT(bulk_ion_charge(RP))
                # INTERIM(diz-ion-species): ν_en_iz_tot, not ν_en_iz alone, because every
                # ion is booked as H₂⁺ and so DI dilutes this population too. Why, and
                # what reverts when H⁺ becomes transportable: `REACTION_STOICHIOMETRY.diz`.
                @. ν_a += Z_i * pla.ν_en_iz_tot      # electron rate: no T_i dependence
            end
            @. pla.exprb.eig_Ti -= ν_a * FT(1.5) * ee

            if linear_response
                ui_mag_sq = @. pla.uiR^2 + pla.uiϕ^2 + pla.uiZ^2
                ΔE = @. (
                    FT(0.5) * mi * ui_mag_sq - FT(1.5) * (pla.Ti_eV - pla.T_gas_eV) * ee
                )
                dK_ela = ion_rate_jacobian(RP, :Elastic)
                dK_cx = ion_rate_jacobian(RP, :Charge_Exchange)
                dν_a = @. pla.n_H2_gas * (FT(0.5) * dK_ela + dK_cx)
                @. pla.exprb.eig_Ti += dν_a * ΔE
            end
        end

        # (the e–i exchange is not here: the step applies its exact solution on its own)

        on_out_wall_nids = RP.G.nodes.on_out_wall_nids
        if !isempty(on_out_wall_nids)
            @views pla.exprb.eig_Ti[on_out_wall_nids] .= zero_FT
        end

        @. pla.exprb.eig_Ti *= FT(2.0) / FT(3.0) / ee
        return RP
    end # @timeit
end

"""
    update_ion_heating_powers!(RP::RAPID{FT}; heat = ion_heat_carriage(RP)) where {FT<:AbstractFloat}

Update ion heating power components for ion energy equation, W per ion:
- `atomic`, from elastic, charge exchange and ionization (MATLAB `Cal_Ion_Heating_Powers`);
- `conv` and `diffu`, the carriage of Tᵢ by the operators that moved the densities (`heat`,
  [`ion_heat_carriage`](@ref); zero when nothing moved them);
- `visc_heat`, `m̄ᵢ Γ_M(u_i∥)`, the viscous heating of u_i∥, on exactly when u_i∥ is diffused.

The e–i exchange is not here: [`ei_exchange_energy!`](@ref) gives its exact increment over the
step and writes its power to `iPowers.equi`.

# Arguments
- `RP::RAPID{FT}`: The RAPID object containing simulation state

# Returns
- `RP`: The updated RAPID object

# Notes
The power calculation includes:
- Energy change from atomic collisions: 0.5*mi*ui_mag_sq - 1.5*(Ti-T_gas)*ee
- Effective collision frequency: n_H2_gas*(0.5*elastic + charge_exchange + Z*ionization)
- Atomic power: collision_frequency * energy_change
- Total power: atomic + conv + diffu + visc_heat
- Sets power to zero outside wall boundaries
"""
function update_ion_heating_powers!(RP::RAPID{FT}; heat = ion_heat_carriage(RP)) where {FT <: AbstractFloat}
    @timeit RAPID_TIMER "update_ion_heating_powers!" begin
        # Define type-stable constants
        zero_FT = zero(FT)

        # Extract physical constants
        @unpack ee, me = RP.config.constants
        mi = bulk_ion_mass(RP)

        # Alias common objects for readability
        pla = RP.plasma
        iPowers = RP.plasma.iPowers

        # Reset ion power arrays to zero (precaution against accumulation)
        iPowers.atomic .= zero_FT
        iPowers.conv .= zero_FT
        iPowers.diffu .= zero_FT
        iPowers.visc_heat .= zero_FT

        # The carriage of Tᵢ by the operators that moved the densities (`ion_heat_carriage`):
        # P_diffu = 1.5 (M Tᵢ), P_conv = −1.5 (u·∇)Tᵢ − Tᵢ ∇·u, the ones `update_Ti!` puts on its
        # left-hand side, so the explicit and the implicit halves of the θ-scheme agree
        if !isnothing(heat)
            isnothing(heat.M) || (iPowers.diffu .= ee * FT(1.5) * (heat.M * pla.Ti_eV))
            isnothing(heat.A_adv) || (iPowers.conv .= ee * (-FT(1.5) * (heat.A_adv * pla.Ti_eV) .- pla.Ti_eV .* heat.div_u))
        end
        # The viscous heating, m̄ᵢ Γ_M(u_i∥) per ion: the kinetic energy of the parallel flow the
        # diffusion of its momentum erased, its energy partner, on exactly when u_i∥ is diffused
        if RP.flags.ud_evolve && RP.flags.diffu
            mom = ion_momentum_carriage(RP)
            if !isnothing(mom) && !isnothing(mom.M)
                dissipation_rate!(iPowers.visc_heat, mom.M, pla.ui_para)
                iPowers.visc_heat .*= pla.mi_mean
            end
        end

        if RP.flags.Atomic_Collision
            # Get ion reaction rate coefficients
            iRRC_cx = get_H2_ion_RRC(RP, :Charge_Exchange)
            iRRC_elastic = get_H2_ion_RRC(RP, :Elastic)

            # Calculate ion velocity magnitude squared
            ui_mag_sq = @. pla.uiR^2 + pla.uiϕ^2 + pla.uiZ^2

            # Calculate average energy change from atomic collisions
            # Energy balance: kinetic energy loss minus thermal energy change
            avg_erg_change_by_atomic_collision = @. (
                FT(0.5) * mi * ui_mag_sq - FT(1.5) * (pla.Ti_eV - pla.T_gas_eV) * ee
            )

            # Calculate effective atomic collision frequency
            # Note: 0.5 factor for elastic collisions (momentum transfer efficiency)
            eff_atomic_coll_freq = @. pla.n_H2_gas * (FT(0.5) * iRRC_elastic + iRRC_cx)

            # Ionization contribution, from the step-entry ν_en_iz_tot (update_RRCs!) that
            # the electron continuity and energy equations use — the electron rate governs
            # it. Z·ν_iz, not ν_iz: the events are counted per electron and this equation
            # is per ion, and one ion carries Z of them (see `update_ui_para!`).
            if RP.flags.src
                Z_i = FT(bulk_ion_charge(RP))
                # INTERIM(diz-ion-species): ν_en_iz_tot, not ν_en_iz alone, because every
                # ion is booked as H₂⁺ and so DI dilutes this population too. Why, and
                # what reverts when H⁺ becomes transportable: `REACTION_STOICHIOMETRY.diz`.
                @. eff_atomic_coll_freq += Z_i * pla.ν_en_iz_tot
            end

            # Calculate atomic power: collision frequency times energy change
            @. iPowers.atomic = eff_atomic_coll_freq * avg_erg_change_by_atomic_collision
        end

        # Calculate total ion heating power (the e–i exchange is applied on its own, and
        # `ei_exchange_energy!` writes its power to `equi`)
        @. iPowers.tot = iPowers.atomic + iPowers.conv + iPowers.diffu + iPowers.visc_heat

        # Set power to zero outside wall boundaries
        on_out_wall_nids = RP.G.nodes.on_out_wall_nids
        if !isempty(on_out_wall_nids)
            @views iPowers.tot[on_out_wall_nids] .= zero_FT
            @views iPowers.atomic[on_out_wall_nids] .= zero_FT
            @views iPowers.conv[on_out_wall_nids] .= zero_FT
            @views iPowers.diffu[on_out_wall_nids] .= zero_FT
            @views iPowers.visc_heat[on_out_wall_nids] .= zero_FT
        end

        return RP
    end # @timeit
end

"""
    solve_electron_continuity_equation!(RP::RAPID{FT}) where FT<:AbstractFloat

Solve the electron continuity equation to update electron density.
Uses either explicit or implicit time integration based on RP.flags.Implicit.
"""
function solve_electron_continuity_equation!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    @timeit RAPID_TIMER "solve_electron_continuity_equation!" begin
        # Alias for readability
        op = RP.operators
        # Get time step from RP
        dt = RP.dt
        pla = RP.plasma

        # Store previous density state for transport calculations
        RP.prev_n .= RP.plasma.ne

        # op.RHS accumulates the TRANSPORT terms only. Ionization is added at the
        # weighting step below instead, because the two families carry different
        # θ (`θ_imp.transport` vs `θ_imp.growth`) and so cannot share a sum.
        fill!(op.RHS, zero(FT))

        # The wall-aware operators: rows on in-wall nodes only, the cached reflective
        # A_diffu_e and face-flux A_conv_e (checked to carry the current `upwind` scheme), and
        # the wall as two diagonal rates per owning cell built from per-face speeds — the Robin
        # debit Σ_f (A_f/V_i)·v_absorb_f, and the albedo's return of the convective outflow
        # a·Σ_f (A_f/V_i)·max(u·n̂_f, 0). The ledger books each face with the same speeds, so
        # operator and ledger are one arithmetic. Each channel enters only with its flag.
        electron_operator_cache(RP)   # refuses a missing or stale cache; `op` is RP.operators
        faces_e = RP.transport.wall_faces
        Ng = length(pla.ne)
        v_e = zeros(FT, length(faces_e))
        robin_e = zeros(FT, Ng)
        returned_e = zeros(FT, Ng)
        if RP.flags.diffu
            update_charge_states!(RP)   # the Bohm channel's mean ion mass, as it is now
            v_absorb = electron_wall_absorption_speeds(RP, faces_e)
            robin_e .= wall_face_debit(FT, Ng, faces_e, v_absorb)
            v_e .+= v_absorb
        end
        if RP.flags.convec
            a_e = electron_wall_albedo(RP)
            v_out = face_outflow_speeds(RP.G, faces_e, pla.ueR, pla.ueZ)
            # A convective operator never written for these velocities would apply nothing
            # while the ledger books the outflow the faces see: refused.
            iszero(op.A_conv_e) && any(>(zero(FT)), v_out) && throw(
                ArgumentError(
                    "operators.A_conv_e is empty while wall faces see outflow: run " *
                        "update_transport_quantities! (or cache_electron_operators!) for these velocities"
                )
            )
            returned_e .= wall_face_debit(FT, Ng, faces_e, a_e .* v_out)
            v_e .+= (one(FT) - a_e) .* v_out
        end

        n_vec, rhs_vec = vec(pla.ne), vec(op.RHS)
        if RP.flags.diffu
            # ∇⋅𝐃⋅∇n with the Robin wall
            rhs_vec .+= op.A_diffu_e * n_vec .- robin_e .* n_vec
        end
        if RP.flags.convec
            # -∇⋅(n 𝐮), the albedo returning its share of the wall outflow
            rhs_vec .-= op.A_conv_e * n_vec .- returned_e .* n_vec
        end

        # A GROWTH eigenvalue, z = +ν_iz_tot·Δt (BOTH electron-producing channels —
        # newborn electrons dilute the drift the same way regardless of which channel
        # made them), and the EXACT local Jacobian since ν_iz_tot does not depend on n
        # — so ExpRB reproduces e^(νΔt) at any step, while every θ has a pole here (BE
        # at z = 1, CN at z = 2) past which it returns a negative density.
        #
        # Derived ONCE and stored, because `update_reaction_counts!` must weight
        # its ledger with the same z — cap included. Written whether or not `src`
        # is on, so `reaction_θ` never reads a previous step's.
        #
        # Unlike decay this z is POSITIVE and can reach the cap: a cell asking to
        # multiply its density by more than e³⁰ in one step is a step nothing
        # resolves, and it should say so.
        # Recorded, not re-read: `reaction_θ` must answer for the solve that formed the
        # ledger, and `scheme.growth` can move between the two calls.
        growth_is_exprb = RP.flags.scheme.growth === ExpRB
        pla.exprb.growth_fitted = growth_is_exprb
        growth_is_exprb && @. pla.exprb.z_growth = exprb_cap_exponent(pla.ν_en_iz_tot * dt)

        fit_growth = RP.flags.src && growth_is_exprb
        bern_growth = if fit_growth
            _warn_if_exprb_capped(pla.exprb.z_growth)
            exprb_bern.(pla.exprb.z_growth)
        else
            nothing
        end

        # update electron density
        if RP.flags.Implicit
            # 0 = forward Euler, ½ = Crank-Nicolson, 1 = backward Euler, per family.
            θ_tr = RP.flags.θ_imp.transport
            θ_gr = RP.flags.θ_imp.growth

            # Weight each family's explicit half in place, then close the RHS —
            # the same accumulation order the explicit branch below uses, so that
            # θ = 0 reproduces it bit for bit rather than merely algebraically.
            @. op.RHS *= (one(FT) - θ_tr)
            if fit_growth
                # nⁿ coefficient is bern(−z) = bern(z) + z. Unlike the decay branch, the
                # subtraction that cancels here is on the LHS, not this one — see
                # the assembly below.
                @. op.RHS = (bern_growth + pla.exprb.z_growth) * pla.ne + dt * op.RHS
            else
                if RP.flags.src
                    @. op.RHS += (one(FT) - θ_gr) * pla.ne * pla.ν_en_iz_tot
                end
                @. op.RHS = pla.ne + dt * op.RHS
            end

            # Build the LHS on the wall pattern, values only:
            #   I − Δt·[θ_tr·(A_diffu_e − diag(robin) − A_conv_e + diag(returned)) + θ_gr·diag(ν_iz_tot)]
            # Every term is gated by the SAME flag that gated its explicit half above: all
            # three used to be added unconditionally, so `flags.diffu = false` removed only the
            # explicit half and left θ·Δt of the diffusion still acting implicitly, and a run
            # that turned `src` off mid-way kept ionizing through a stale ν_en_iz_tot (BOTH
            # electron-producing channels, materialized by update_RRCs! at the step-entry
            # state — not re-queried here). The ion path honours the flags in full, which is
            # how the mismatch showed up.
            A = op.A_LHS
            set_identity!(A)
            if RP.flags.diffu
                add_scaled!(A, -dt * θ_tr, op.A_diffu_e)
                add_diagonal!(A, robin_e; scale = dt * θ_tr)
            end
            if RP.flags.convec
                add_scaled!(A, dt * θ_tr, op.A_conv_e)
                add_diagonal!(A, returned_e; scale = -dt * θ_tr)
            end
            (RP.flags.src && !fit_growth) && add_diagonal!(A, vec(pla.ν_en_iz_tot); scale = -dt * θ_gr)
            if fit_growth
                # bern(z) on the diagonal, as a deviation from the identity. This is the side
                # that cancels on a growth branch — the θ form's 1 − θνΔt passes through zero
                # at the poles — and bern(z) > 0 at every z keeps the matrix an M-matrix.
                add_diagonal!(A, vec(bern_growth) .- one(FT))
            end

            # Solve the linear system (cached factorization; the pattern never changes)
            @timeit RAPID_TIMER "ne LinearSolve`" begin
                factorize!(op.ne_solver, A)
                solve!(view(pla.ne, :), op.ne_solver, view(op.RHS, :))
            end
            book_electron_wall_loss!(RP, faces_e, v_e, pla.ne, RP.prev_n, θ_tr)
        elseif fit_growth
            # Same two coefficients as the assembled path: no matrix is not the same
            # as no fit. `op.RHS` still holds transport, which is not part of λ and
            # rides the increment like any other frozen source.
            @. pla.ne = ((bern_growth + pla.exprb.z_growth) * pla.ne + dt * op.RHS) / bern_growth
            book_electron_wall_loss!(RP, faces_e, v_e, pla.ne, RP.prev_n, zero(FT))
        else
            if RP.flags.src
                @. op.RHS += pla.ne * pla.ν_en_iz_tot
            end
            @. RP.plasma.ne += dt * op.RHS
            book_electron_wall_loss!(RP, faces_e, v_e, pla.ne, RP.prev_n, zero(FT))
        end

        # Publish how many ionizations this step made, from the θ and the nⁿ/nⁿ⁺¹
        # this solve just used. Everything else — the ion source, both particle
        # ledgers, the neutral-gas sink — reads that one number instead of
        # rebuilding it, so they cannot disagree about the event count.
        update_reaction_counts!(RP)
        return RP
    end # @timeit
end

"""
    book_ionization_sources!(RP::RAPID{FT}) where FT<:AbstractFloat

Add this step's ionization count to the particle ledgers: `Ne_src`, and `Ni_src` when the
ions are evolved independently, in both the 0-D and the 2-D cumulative trackers.

The count is the one `update_reaction_counts!` published for the solve that just ran — a
per-step number, already weighted with the θ the solve used, so nothing is recomputed here —
times the in-wall cell volumes. One ionization event makes one electron AND one ion, so the
ion ledger is fed the same number. The published rates are zero on the band outside the
wall, so nothing is ever booked there.
"""
function book_ionization_sources!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    @timeit RAPID_TIMER "book_ionization_sources!" begin
        # Bound outside the `@.`, which would otherwise broadcast the calls over `RP`.
        Δn_e = net_electron_count(check_reaction_counts(RP))
        Ne_iz = @. Δn_e * RP.G.inVol2D

        Ntracker = RP.diagnostics.Ntracker
        Ntracker.cum0D_Ne_src += sum(Ne_iz)
        @. Ntracker.cum2D_Ne_src += Ne_iz
        if RP.flags.update_ni_independently
            Ntracker.cum0D_Ni_src += sum(Ne_iz)
            @. Ntracker.cum2D_Ni_src += Ne_iz
            # per species, from each species' own count (the total above stays the
            # electron count, which equals Σ over species under the current table)
            counts = RP.reactions.counts
            grow_species_entries!(Ntracker, length(RP.transport.ion_species))
            for (s, sp) in enumerate(RP.transport.ion_species)
                Δn = net_ion_count(counts, sp.key)
                isnothing(Δn) || (Ntracker.cum0D_Ni_src_s[s] += sum(Δn .* RP.G.inVol2D))
            end
        end
        return RP
    end # @timeit
end

"""
    calculate_para_grad_of_scalar_F(RP::RAPID{FT}, F::Matrix{FT}; upwind::Bool=RP.flags.upwind) where {FT<:AbstractFloat}

Calculate the parallel gradient [∇∥ ≡ b⋅∇] of a scalar field F in the direction of the magnetic field.
Uses either upwind scheme (based on flow velocity) or central differences.

# Arguments
- `RP::RAPID{FT}`: The RAPID object containing simulation state
- `F::Matrix{FT}`: The scalar field whose parallel gradient is to be calculated
- `upwind::Bool=RP.flags.upwind`: whether to use flow direction to choose appropriate differencing

# Returns
- `Matrix{FT}`: The calculated parallel gradient field

# Notes
- When upwind=true, uses flow direction to choose appropriate differencing
- When upwind=false, uses central differencing for interior points
- Provides better numerical stability for advection-dominated problems when upwind=true
- Matrix indexing is F[i,j] where i is R-index and j is Z-index
"""
function calculate_para_grad_of_scalar_F(RP::RAPID{FT}, F::Matrix{FT}; upwind::Bool = RP.flags.upwind) where {FT <: AbstractFloat}
    NR, NZ = size(F)
    @assert NR > 1 && NZ > 1 "Grid size must be at least 2x2"

    # Define constants for type stability
    zero_FT = zero(FT)
    half = FT(0.5)
    eps_val = eps(FT)

    # Pre-compute inverse values for faster calculation
    inv_dR = one(FT) / RP.G.dR
    inv_dZ = one(FT) / RP.G.dZ

    # Initialize output array
    para_∇F = zeros(FT, NR, NZ)

    # Calculate parallel gradient for interior points
    if upwind
        # Upwind scheme based on flow velocity direction
        @inbounds for j in 2:(NZ - 1), i in 2:(NR - 1)

            # R-direction contribution
            if abs(RP.plasma.ueR[i, j]) < eps_val
                # Zero velocity: use central differencing for stability
                para_∇F[i, j] += RP.fields.bR[i, j] * (F[i + 1, j] - F[i - 1, j]) * (inv_dR * half)
            elseif RP.plasma.ueR[i, j] > zero_FT
                # Positive flow: backward difference (upwind)
                para_∇F[i, j] += RP.fields.bR[i, j] * (F[i, j] - F[i - 1, j]) * inv_dR
            else
                # Negative flow: forward difference (upwind)
                para_∇F[i, j] += RP.fields.bR[i, j] * (F[i + 1, j] - F[i, j]) * inv_dR
            end

            # Z-direction contribution
            if abs(RP.plasma.ueZ[i, j]) < eps_val
                # Zero velocity: use central differencing for stability
                para_∇F[i, j] += RP.fields.bZ[i, j] * (F[i, j + 1] - F[i, j - 1]) * (inv_dZ * half)
            elseif RP.plasma.ueZ[i, j] > zero_FT
                # Positive flow: backward difference (upwind)
                para_∇F[i, j] += RP.fields.bZ[i, j] * (F[i, j] - F[i, j - 1]) * inv_dZ
            else
                # Negative flow: forward difference (upwind)
                para_∇F[i, j] += RP.fields.bZ[i, j] * (F[i, j + 1] - F[i, j]) * inv_dZ
            end
        end
    else
        # Central difference scheme for interior points
        # This is more accurate for smooth solutions but may have stability issues for advection-dominated flows
        @inbounds for j in 2:(NZ - 1), i in 2:(NR - 1)
            para_∇F[i, j] = RP.fields.bR[i, j] * (F[i + 1, j] - F[i - 1, j]) * (inv_dR * half) +
                RP.fields.bZ[i, j] * (F[i, j + 1] - F[i, j - 1]) * (inv_dZ * half)
        end
    end

    # Handle boundaries with one-sided differences
    # Calculate R derivative contributions
    @inbounds for j in 1:NZ
        # Left boundary: forward difference
        i = 1
        para_∇F[i, j] += RP.fields.bR[i, j] * (F[i + 1, j] - F[1, j]) * inv_dR
        i = NR
        # Right boundary: backward difference
        para_∇F[i, j] += RP.fields.bR[i, j] * (F[i, j] - F[i - 1, j]) * inv_dR
    end
    # Bottom and Top boundary: central difference
    @inbounds for j in [1, NZ]
        for i in 2:(NR - 1)
            para_∇F[i, j] += RP.fields.bR[i, j] * (F[i + 1, j] - F[i - 1, j]) * (inv_dR * half)
        end
    end

    # Calculate Z derivative contributions
    @inbounds for i in 1:NR
        # Bottom boundary: forward difference
        j = 1
        para_∇F[i, j] += RP.fields.bZ[i, j] * (F[i, j + 1] - F[i, j]) * inv_dZ
        # Top boundary: backward difference
        j = NZ
        para_∇F[i, j] += RP.fields.bZ[i, j] * (F[i, j] - F[i, j - 1]) * inv_dZ
    end
    # Left and Right boundary: central difference
    @inbounds for i in [1, NR]
        for j in 2:(NZ - 1)
            para_∇F[i, j] += RP.fields.bZ[i, j] * (F[i, j + 1] - F[i, j - 1]) * (inv_dZ * half)
        end
    end

    return para_∇F
end


"""
    calculate_grad_of_scalar_F(RP::RAPID{FT}, F::Matrix{FT}; upwind::Bool=RP.flags.upwind) where {FT<:AbstractFloat}

Calculate the gradient components of a scalar field F in R and Z directions.
Returns gradF_R and gradF_Z as separate matrices.

# Arguments
- `RP::RAPID{FT}`: The RAPID object containing simulation state
- `F::Matrix{FT}`: The scalar field whose gradient is to be calculated
- `upwind::Bool=RP.flags.upwind`: Whether to use upwind differencing based on velocity field

# Returns
- Tuple of two matrices (gradF_R, gradF_Z): Components of the gradient in R and Z directions

# Notes
- When upwind=true, uses flow velocity direction to choose appropriate differencing scheme
- When upwind=false, uses standard central differencing for interior points with one-sided differences at boundaries
- Provides better numerical stability for advection-dominated problems when upwind=true
- Matrix indexing is F[i,j] where i is R-index and j is Z-index (Julia convention)
  which differs from MATLAB's (j,i) convention
"""
function calculate_grad_of_scalar_F(RP::RAPID{FT}, F::Matrix{FT}; upwind::Bool = RP.flags.upwind) where {FT <: AbstractFloat}
    NR, NZ = size(F)
    @assert NR > 1 && NZ > 1 "Grid size must be at least 2x2"

    # Define constants for type stability
    zero_FT = zero(FT)
    half = FT(0.5)
    eps_val = eps(FT)

    # Pre-compute inverse values for faster calculation
    inv_dR = one(FT) / RP.G.dR
    inv_dZ = one(FT) / RP.G.dZ

    # Initialize output arrays
    ∇F_R = zeros(FT, NR, NZ)
    ∇F_Z = zeros(FT, NR, NZ)


    # Calculate gradients for interior points
    if upwind
        # Upwind differencing scheme based on flow velocity
        @inbounds for j in 2:(NZ - 1), i in 2:(NR - 1)
            # R-direction gradient
            if abs(RP.plasma.ueR[i, j]) < eps_val
                # Zero velocity: use central differencing for stability
                ∇F_R[i, j] = (F[i + 1, j] - F[i - 1, j]) * (inv_dR * half)
            elseif RP.plasma.ueR[i, j] > zero_FT
                # Positive velocity: backward difference (upwind)
                ∇F_R[i, j] = (F[i, j] - F[i - 1, j]) * inv_dR
            else
                # Negative velocity: forward difference (upwind)
                ∇F_R[i, j] = (F[i + 1, j] - F[i, j]) * inv_dR
            end

            # Z-direction gradient
            if abs(RP.plasma.ueZ[i, j]) < eps_val
                # Zero velocity: use central differencing for stability
                ∇F_Z[i, j] = (F[i, j + 1] - F[i, j - 1]) * (inv_dZ * half)
            elseif RP.plasma.ueZ[i, j] > zero_FT
                # Positive velocity: backward difference (upwind)
                ∇F_Z[i, j] = (F[i, j] - F[i, j - 1]) * inv_dZ
            else
                # Negative velocity: forward difference (upwind)
                ∇F_Z[i, j] = (F[i, j + 1] - F[i, j]) * inv_dZ
            end
        end
    else
        # Standard central differencing scheme
        @inbounds for j in 2:(NZ - 1), i in 2:(NR - 1)
            # R-direction gradient
            ∇F_R[i, j] = (F[i + 1, j] - F[i - 1, j]) * (inv_dR * half)
            # Z-direction gradient
            ∇F_Z[i, j] = (F[i, j + 1] - F[i, j - 1]) * (inv_dZ * half)
        end
    end

    # Handle boundaries with one-sided differences
    # Calculate R derivative contributions
    @inbounds for j in 1:NZ
        # Left boundary: forward difference
        ∇F_R[1, j] = (F[2, j] - F[1, j]) * inv_dR
        # Right boundary: backward difference
        ∇F_R[NR, j] = (F[NR, j] - F[NR - 1, j]) * inv_dR
    end
    # Bottom and Top boundary: central difference
    @inbounds for j in [1, NZ]
        for i in 2:(NR - 1)
            ∇F_R[i, j] = (F[i + 1, j] - F[i - 1, j]) * (inv_dR * half)
        end
    end

    # Calculate Z derivative contributions
    @inbounds for i in 1:NR
        # Bottom boundary: forward difference
        ∇F_Z[i, 1] = (F[i, 2] - F[i, 1]) * inv_dZ
        # Top boundary: backward difference
        ∇F_Z[i, NZ] = (F[i, NZ] - F[i, NZ - 1]) * inv_dZ
    end
    # Left and Right boundary: central difference
    @inbounds for i in [1, NR]
        for j in 2:(NZ - 1)
            ∇F_Z[i, j] = (F[i, j + 1] - F[i, j - 1]) * (inv_dZ * half)
        end
    end

    return ∇F_R, ∇F_Z
end


"""
    calculate_electron_acceleration_by_pressure(RP::RAPID{FT}; num_SM::Int=2) where {FT<:AbstractFloat}

Calculate the electron pressure gradient force acceleration along the magnetic field.
This uses a smoothed density field to improve numerical stability.

# Arguments
- `RP::RAPID{FT}`: The RAPID object containing simulation state
- `num_SM::Int=2`: Number of smoothing iterations (if 0, no smoothing is applied)

# Returns
- `Matrix{FT}`: The pressure gradient acceleration field (m/s²)

# Notes
- Uses smoothed density field to avoid numerical issues with very low density regions
- Calculates both density gradient and temperature gradient contributions
- Limits the maximum acceleration to maintain numerical stability
- Setting `num_SM=0` bypasses smoothing, which may be desirable for specific use cases
"""
function calculate_electron_acceleration_by_pressure(RP::RAPID{FT}; num_SM::Int = 2) where {FT <: AbstractFloat}
    # alias
    @unpack ee, me = RP.config.constants

    # Smooth the density field to reduce numerical noise (skip if num_SM is 0)
    n_SM = smooth_data_2D(RP.plasma.ne; num_SM, weighting = RP.G.Jacob)
    n_SM[n_SM .< 0] .= zero(FT)

    # Calculate ln(n) gradients along B to avoid division by zero issues with low density
    # Calculate temperature gradients along B
    para_grad_ln_n = calculate_para_grad_of_scalar_F(RP, log.(n_SM))
    para_grad_Te_eV = calculate_para_grad_of_scalar_F(RP, RP.plasma.Te_eV)

    # Combine both terms for total pressure gradient acceleration
    accel_by_pressure = @. (
        - para_grad_ln_n * RP.plasma.Te_eV * ee / me
            - para_grad_Te_eV * ee / me
    )

    if RP.flags.limit_acceleration.state
        factor = RP.flags.limit_acceleration.factor
        max_abs_accel = factor .* maximum(abs.(RP.plasma.ue_para[RP.G.nodes.in_wall_nids])) ./ RP.dt
        clamp!(accel_by_pressure, -max_abs_accel, +max_abs_accel)
    end

    # Handle any NaN or Inf values that might arise
    accel_by_pressure[.!isfinite.(accel_by_pressure)] .= zero(FT)

    return accel_by_pressure
end


"""
    electron_pressure_shares(RP) -> (ue, amb)

The electron pressure acceleration ([`calculate_electron_acceleration_by_pressure`](@ref), after
its limiter) split by the coupling weight `w_amb = ambipolar_weight`: the share u∥ carries,
`(1 − w_amb) a_p`, and the ambipolar one, `w_amb a_p`, which both species diffuse with
`transport.Dpara_amb`, so the drive is carried once.
"""
function electron_pressure_shares(RP::RAPID{FT}) where {FT <: AbstractFloat}
    a_p = calculate_electron_acceleration_by_pressure(RP)
    w = RP.plasma.ambipolar_weight
    return (ue = (one(FT) .- w) .* a_p, amb = w .* a_p)
end

"The share of the electron pressure acceleration that u∥ carries ([`electron_pressure_shares`](@ref))."
ue_pressure_acceleration(RP::RAPID) = electron_pressure_shares(RP).ue

"""
    ambipolar_parallel_field(RP) -> Matrix

The parallel field that holds the electrons to the ions in the ambipolar pair, `E∥,amb = (mₑ/e) w_amb a_p`:
it takes the ambipolar share of the electron pressure acceleration
([`electron_pressure_shares`](@ref)) off the electrons and passes it to the ions, `≈ −w_amb ∇∥pₑ/(e nₑ)`
(Boltzmann) where the acceleration limiter does not bind. A diagnostic: no part of `E_para_tot`.
"""
function ambipolar_parallel_field(RP::RAPID{FT}) where {FT <: AbstractFloat}
    @unpack me, ee = RP.config.constants
    return (me / ee) .* electron_pressure_shares(RP).amb
end

"""
    book_ambipolar_exchange!(RP, a_amb, u_old) -> RP

Add to `diagnostics.ambipolar_exchange` the work the ambipolar share of the electron
pressure force would have done on u∥ over the step, at the step's mean velocity,
`Σ V mₑ nₑ a_amb (uⁿ + uⁿ⁺¹)/2 Δt` [J], `u_old` being uⁿ and `ue_para` uⁿ⁺¹. The ambipolar
field passes it from the electrons' pressure to the pair's motion, the ions', and no equation of
the electrons' energy sees it: a diagnostic.
"""
function book_ambipolar_exchange!(RP::RAPID{FT}, a_amb::AbstractMatrix{FT}, u_old::AbstractMatrix{FT}) where {FT <: AbstractFloat}
    pla, V = RP.plasma, RP.G.inVol2D
    me = RP.config.constants.me
    W = zero(FT)
    @inbounds for k in RP.G.nodes.in_wall_nids
        W += V[k] * me * pla.ne[k] * a_amb[k] * (u_old[k] + pla.ue_para[k]) / 2
    end
    RP.diagnostics.ambipolar_exchange += W * RP.dt
    return RP
end

"""
    ambipolar_gradTe_share(RP) -> FT

The share `∇Tₑ` makes of the electron pressure drive the ambipolar share takes, `w_amb`- and
`nₑ`-weighted over the in-wall volume: `Σ V n w |∂∥Tₑ| / Σ V n w (|Tₑ ∂∥ln n| + |∂∥Tₑ|)`, 0 where
nothing is coupled. The electrons' pressure only: the ambipolar temperature-gradient drift
`transport.u_para_amb_gradT` carries the pair's `∇ln(Z̄Tₑ + Tᵢ)`, the ions' temperature as well.
"""
function ambipolar_gradTe_share(RP::RAPID{FT}) where {FT <: AbstractFloat}
    pla, V = RP.plasma, RP.G.inVol2D
    w = pla.ambipolar_weight
    any(>(zero(FT)), w) || return zero(FT)
    n_SM = smooth_data_2D(pla.ne; num_SM = 2, weighting = RP.G.Jacob)
    grad_ln_n = calculate_para_grad_of_scalar_F(RP, log.(max.(n_SM, floatmin(FT))))
    grad_Te = calculate_para_grad_of_scalar_F(RP, pla.Te_eV)
    num, den = zero(FT), zero(FT)
    for k in RP.G.nodes.in_wall_nids
        node_weight = V[k] * pla.ne[k] * w[k]
        isfinite(grad_ln_n[k]) && isfinite(grad_Te[k]) || continue
        num += node_weight * abs(grad_Te[k])
        den += node_weight * (abs(pla.Te_eV[k] * grad_ln_n[k]) + abs(grad_Te[k]))
    end
    return den > zero(FT) ? num / den : zero(FT)
end

"""
    solve_Ampere_equation!(RP::RAPID{FT}, F::Fields{FT}=RP.fields; plasma::Bool=true, coils::Bool=true, update_Eϕ_coils_pla::Bool=true) where {FT<:AbstractFloat}

Solve Ampère's equation (Grad-Shafranov equation) for the field of the coils and the plasma (`ψ_coils_pla`, `BR/BZ_coils_pla`).

Solves: ΔGS * ψ = - μ₀R²Jϕ with boundary conditions from Green's function.
Updates ψ_coils_pla, and optionally Eϕ_coils_pla.

# Arguments
- `RP::RAPID{FT}`: RAPID simulation object containing plasma, grid, and operators
- `F::Fields{FT}`: Fields object to update (defaults to `RP.fields`)

# Keyword Arguments
- `plasma::Bool=true`: Include plasma current density (Jϕ) in the source term
- `coils::Bool=true`: Include the coils' currents in the source term
"""
function solve_Ampere_equation!(RP::RAPID{FT}, F::Fields{FT} = RP.fields; plasma::Bool = true, coils::Bool = true, update_Eϕ_coils_pla::Bool = true) where {FT <: AbstractFloat}
    @timeit RAPID_TIMER "solve_Ampere_equation" begin
        # Alias for readability
        pla = RP.plasma
        OP = RP.operators
        μ0 = RP.config.constants.μ0
        csys = RP.coil_system

        if plasma
            @. OP.RHS = - μ0 * RP.G.R2D * pla.Jϕ
        else
            @. OP.RHS = zero(FT)
        end

        if coils && csys.n_total > 0
            inside_Jϕ_coil_k = distribute_coil_currents_to_Jϕ(csys, RP.G)
            @. OP.RHS .+= -μ0 * RP.G.R2D * inside_Jϕ_coil_k
        end

        # Boudnary condition: calculate psi values at boundaries using the Green_inWall2bdy
        if plasma
            @views OP.RHS[RP.G.BDY_idx] .= (RP.G.Green_inWall2bdy * pla.Jϕ[RP.G.nodes.in_wall_nids]) * RP.G.dR * RP.G.dZ
        else
            OP.RHS[RP.G.BDY_idx] .= zero(FT)
        end

        if coils && csys.n_total > 0
            OP.RHS[RP.G.BDY_idx] .+= csys.Green_coils2bdy * csys.coils.current
        end

        old_ψ_coils_pla = copy(F.ψ_coils_pla)

        # solve Ampere's equation. ΔGS is built once by `initialize_operators!` and never
        # changes, so it is factorized on the first solve and only back-substituted after.
        OP.ΔGS_solver.F === nothing && factorize!(OP.ΔGS_solver, OP.ΔGS)
        ψ = similar(OP.RHS)
        solve!(vec(ψ), OP.ΔGS_solver, vec(OP.RHS))
        F.ψ_coils_pla = ψ

        # the magnetic field of ψ_coils_pla
        calculate_B_from_ψ!(RP.G, F.ψ_coils_pla, F.BR_coils_pla, F.BZ_coils_pla)

        if update_Eϕ_coils_pla && RP.flags.E_para_induction
            @. F.Eϕ_coils_pla = - (F.ψ_coils_pla - old_ψ_coils_pla) / (RP.G.R2D * RP.dt)
        end

        return RP
    end # @timeit
end

"""
    plasma_flux_on_grid(RP, Jϕ)

The flux ψ [Wb/rad] that the toroidal current density `Jϕ` alone makes on the grid: the
Grad–Shafranov solution with `Jϕ` as the source and its Green function on the boundary. With
[`coil_flux_on_grid`](@ref) it splits `ψ_coils_pla` by source, up to what the coupled solve's outer
iteration and the steps after it leave between the solved sum and the present currents.
"""
function plasma_flux_on_grid(RP::RAPID{FT}, Jϕ::AbstractMatrix{FT}) where {FT <: AbstractFloat}
    G, OP = RP.G, RP.operators
    RHS = @. -RP.config.constants.μ0 * G.R2D * Jϕ
    RHS[G.BDY_idx] .= (G.Green_inWall2bdy * Jϕ[G.nodes.in_wall_nids]) .* (G.dR * G.dZ)
    OP.ΔGS_solver.F === nothing && factorize!(OP.ΔGS_solver, OP.ΔGS)
    ψ = similar(RHS)
    solve!(vec(ψ), OP.ΔGS_solver, vec(RHS))
    return ψ
end

"""
    coil_flux_on_grid(RP, I)

The flux ψ [Wb/rad] that coil currents `I` alone make on the grid: the Grad–Shafranov
solution with the in-grid coils as sources and the coils' Green function on the boundary.
"""
function coil_flux_on_grid(RP::RAPID{FT}, I::AbstractVector{FT}) where {FT <: AbstractFloat}
    G, OP, csys = RP.G, RP.operators, RP.coil_system
    RHS = -RP.config.constants.μ0 .* G.R2D .* distribute_coil_currents_to_Jϕ(csys, G; currents = I)
    RHS[G.BDY_idx] .= csys.Green_coils2bdy * I
    OP.ΔGS_solver.F === nothing && factorize!(OP.ΔGS_solver, OP.ΔGS)
    ψ = similar(RHS)
    solve!(vec(ψ), OP.ΔGS_solver, vec(RHS))
    return ψ
end

"""
    coil_induced_Eϕ(RP, ΔI) -> Matrix

The toroidal field a change `ΔI` of the coil currents over the coils' last step induces on the
grid, −ψ_coils(ΔI)/(R Δt) with Δt = `coil_system.Δt` (`RP.dt`, or the last part of a step taken
in parts) and ψ_coils from [`coil_flux_on_grid`](@ref); zero for no change. Below the Ampère gate
it is the whole induced field; the 2D snapshots split the induced field with it.
"""
function coil_induced_Eϕ(RP::RAPID{FT}, ΔI::AbstractVector{FT}) where {FT <: AbstractFloat}
    (isempty(ΔI) || all(iszero, ΔI)) && return zeros(FT, RP.G.NR, RP.G.NZ)
    ψ = coil_flux_on_grid(RP, ΔI)
    return @. -ψ / (RP.G.R2D * RP.coil_system.Δt)
end

"""
    set_Eϕ_coils_pla_from_coils!(RP, ΔI)

Below the Ampère gate, set the induced field to the coils' alone, Eϕ_coils_pla = −ψ_coils(ΔI)/(R Δt)
with ΔI their change over the step, and E∥ from it; the prescribed and control parts of E∥ stay as the last
step projected it. There the plasma is not a source of induction: its own inductance is left
out (unstable taken explicitly), and so is its flux in the coils (their reaction, fed back
without L_p, is unstable); see `internal/docs/src/notes/design/coupled-step-coil-flux.md`
§5.3 and §10.1.
"""
function set_Eϕ_coils_pla_from_coils!(RP::RAPID{FT}, ΔI::AbstractVector{FT}) where {FT <: AbstractFloat}
    F = RP.fields
    F.Eϕ_coils_pla_prev .= F.Eϕ_coils_pla
    F.Eϕ_coils_pla .= coil_induced_Eϕ(RP, ΔI)
    if RP.flags.E_para_induction
        @. F.Eϕ_para_coils_pla = F.Eϕ_coils_pla * F.bϕ
    end
    combine_E_para!(RP)
    return RP
end

"""
    advance_coils!(RP; plasma = true) -> ΔI

Advance the coil currents one step outside the coupled solve and return their change, empty
without coils. Their circuits take the plasma's flux change since each coil's last update,
the plasma current taken at the start of the step; below the gate (`plasma = false`) they
advance as in vacuum (see `set_Eϕ_coils_pla_from_coils!`).
"""
function advance_coils!(RP::RAPID{FT}; plasma::Bool = true) where {FT <: AbstractFloat}
    csys = RP.coil_system
    csys.n_total == 0 && return FT[]
    θ_circuit = RP.flags.θ_imp.circuit
    if RP.dt != csys.Δt || csys.θimp != θ_circuit
        csys.Δt = RP.dt
        csys.θimp = θ_circuit
        calculate_circuit_matrices!(csys)
    end
    I_before = get_all_currents(csys)
    advance_LR_circuit_step!(csys, RP.G, RP.plasma.Jϕ, RP.time_s; plasma)
    return get_all_currents(csys) .- I_before
end


"""
    solve_Ampere_equation(RP::RAPID{FT}; plasma::Bool=true, coils::Bool=true) where {FT<:AbstractFloat}

Non-mutating version of `solve_Ampere_equation!` that returns a new Fields object.

Creates a new `Fields{FT}` object and solves Ampère's equation without modifying
the original fields in the RAPID object.

# See Also
- [`solve_Ampere_equation!`](@ref): In-place version that modifies the RAPID object
"""
function solve_Ampere_equation(RP::RAPID{FT}; plasma::Bool = true, coils::Bool = true) where {FT <: AbstractFloat}
    F = Fields{FT}(RP.G.NR, RP.G.NZ)
    solve_Ampere_equation!(RP, F; plasma, coils)
    return F
end

"""
    _refuse_exprb_decay(flags, who, why)

Throw if `scheme.decay = ExpRB` reaches code that cannot honour it.

The use-site backstop for what `validate_scheme_flags` refuses at `initialize!`.
Deliberately wider: flags stay mutable afterwards, and these callers are reached by
runtime condition or direct call, not by the configuration that was validated.

`why` completes "…cannot honour scheme.decay = ExpRB because it `why`".
"""
function _refuse_exprb_decay(flags::SimulationFlags, who::AbstractString, why::AbstractString)
    flags.scheme.decay === ExpRB && throw(
        ArgumentError(
            "$who cannot honour scheme.decay = ExpRB because it $why — only " *
                "update_ue_para! carries bern(z) today. Use scheme.decay = Theta."
        )
    )
    return nothing
end

"""
    _refuse_full_response_decay(flags)

Throw if `scheme.decay = ExpRB` is asked for `FullLinearResponse`.

The sibling of [`_refuse_exprb_decay`](@ref): nothing computes the momentum equation's
linear response, so honouring the flag would quietly deliver the partial one instead.
"""
function _refuse_full_response_decay(flags::SimulationFlags)
    flags.exprb_eigenvalue === FullLinearResponse && throw(
        ArgumentError(
            "scheme.decay = ExpRB cannot honour exprb_eigenvalue = FullLinearResponse: " *
                "update_ue_para! fits λ = −(ν_en_mom_tot + ν_en_iz_tot + ν_ei_eff), the " *
                "stated rate, and nothing computes the −(mₑu∥²/e)·∂ν/∂Ē that would " *
                "complete it. Reaching here means the depth was set after initialize!, " *
                "which validate_scheme_flags refuses. Use " *
                "exprb_eigenvalue = PartialLinearResponse, or scheme.decay = Theta."
        )
    )
    return nothing
end

"""
    coupled_residual_converged(f_ψb, f_I, R_bdy, coil_field, ψ_new, ψ_old, R, I_new, I_old, dt; tol, E_floor, I_floor)

The combined solve's stopping test, on the residual f = g(x) − x of its outer iteration over
the boundary flux and the coil currents, unweighted. Two parts are in the units of the induced
field E = −Δψ/(R Δt):
- the boundary flux, max|f_ψb / R_b| / Δt;
- the coils inside the grid, whose current enters ψ's interior source, max |f_I,c| · coil_field_c,
  where coil_field = L_c/(2π R_c Δt) there and 0 for coils outside the grid; those reach the
  plasma only through the boundary flux.

Both are measured against `tol` times the field the step induces, max|(ψ_new − ψ_old)/R| / Δt,
plus `E_floor` [V/m]. Every coil's current residual must also be under `tol` times its change
over the step, |I_new − I_old|, plus `I_floor` [A].

A residual or a step field that is not finite is not converged. This is a residual test, not
an error bound: the tests compare the accepted step with the step iterated to convergence.
Returns `(converged, E_residual, I_residual, finite)`, `finite` saying whether both residuals
and the step field are.
"""
function coupled_residual_converged(f_ψb, f_I, R_bdy, coil_field, ψ_new, ψ_old, R, I_new, I_old, dt; tol, E_floor, I_floor)
    E_bdy = maximum(abs, f_ψb ./ R_bdy) / dt
    E_coil = isempty(f_I) ? zero(E_bdy) : maximum(abs, f_I .* coil_field)
    E_step = maximum(abs, (ψ_new .- ψ_old) ./ R) / dt
    E_residual = max(E_bdy, E_coil)
    I_residual = isempty(f_I) ? zero(E_bdy) : maximum(abs, f_I)
    coils_settled = all(abs.(f_I) .<= tol .* abs.(I_new .- I_old) .+ I_floor)
    finite = isfinite(E_residual) && isfinite(I_residual) && isfinite(E_step)
    converged = finite && E_residual <= tol * E_step + E_floor && coils_settled
    return converged, E_residual, I_residual, finite
end

# Count the solve in RP.diagnostics.ampere_picard, and warn at a run's first unconverged one.
function record_picard!(RP::RAPID, iter::Int, converged::Bool, E_residual::Real, I_residual::Real, max_iter::Int)
    stats = RP.diagnostics.ampere_picard
    stats.nsolve += 1
    stats.niter += iter
    stats.last_niter = iter
    stats.last_E_residual = E_residual
    stats.last_I_residual = I_residual
    if !converged
        stats.nunconverged += 1
        stats.nunconverged == 1 &&
            @warn "Ampère Picard iteration stopped short of its tolerance after $iter block solves (max_iter = $max_iter, step $(RP.step)); later ones are only counted in RP.diagnostics.ampere_picard" E_residual I_residual
    end
    return nothing
end

"""
    uψ_coupling_terms(RP) -> (c_uψ, c_ψu)

The two coupling diagonals of the u∥–ψ block, per node: ψ's inductive drive on u∥,
c_uψ = q_e b_ϕ/(m_e R), and the electron current's source in Ampère's law,
c_ψu = μ0 R n_e q_e b_ϕ, zero on the boundary rows (ψ is set there).
"""
function uψ_coupling_terms(RP::RAPID{FT}) where {FT <: AbstractFloat}
    @unpack qe, me, μ0 = RP.config.constants
    c_uψ = vec(@. qe * RP.fields.bϕ / (me * RP.G.R2D))
    c_ψu = vec(@. μ0 * RP.G.R2D * RP.plasma.ne * qe * RP.fields.bϕ)
    c_ψu[RP.G.BDY_idx] .= zero(FT)
    return c_uψ, c_ψu
end

"""
    solve_combined_momentum_Ampere_equations_with_coils!(RP; method = AndersonOuterSolve(),
        tolerance = 1e-3, max_iter = 20, E_floor = 1e-6, I_floor = 1e-6)

Advance u∥, ψ_coils_pla and the coil currents together over one step, the coefficients (n, ν, b)
held at the step's start. u∥'s equation is [`update_ue_para!`](@ref)'s, term by term and with
its θ-weights (`θ_imp.decay`; advection and diffusion explicit when `Implicit = false`). The
induced field is the step's mean, −(ψⁿ⁺¹ − ψⁿ)/(RΔt), which carries no θ. The circuits take
`θ_imp.circuit`, as below the gate.

The block matrix of u∥ and ψ inside the domain (`CoupledBlock`, on a pattern fixed for the
run) is factorized once. With `SurfaceFluxConstraint()` on a closed region the surface flux
n u∥/B at the knots joins it as M more unknowns ([`coupled_block_solver!`](@ref)), so ψ and the coils are
solved with the constrained current (`BlockConstraint`, the default; `PostProjection` solves
u∥ again with the constraint after the block). The free boundary and the circuits close it by
an outer iteration on x = (boundary flux, coil currents):
- a block solve with the boundary flux as Dirichlet values and the coils inside the grid as ψ's
  source gives u∥ and ψ;
- from the plasma current J, the circuits give the coil currents, with the plasma term
  2π[ψ_pla(r_c; J) − `Coil.ψ_pla`];
- Green's functions give the boundary flux of J and those coil currents.

That map is affine, g(x) = T x + c. The default `method`, [`AndersonOuterSolve`](@ref), drives
f = g(x) − x to zero by Anderson mixing (`AndersonMixer`); a relative of GMRES, it can also
converge where the relaxed iteration x + w f diverges.
- **First iterate:** a block solve with the boundary flux of the extrapolated induced field
  and the coil currents at tⁿ.
- **Circuit forcing:** M Iⁿ + Δt V(tⁿ + Δt/2), taken once for the step.

[`DirectOuterSolve`](@ref) is the reference: T assembled and (𝟙 − T) x = c solved at once.

The iteration (`fixed_point_solve!`) stops when `coupled_residual_converged` holds. It accepts
that evaluation: u∥ and ψ of its block solve, and the coil currents of its J, with `Coil.ψ_pla`
set to ψ_pla(r_c; J). The circuits' flux balance then closes every step. A solve that does not
converge (`max_iter` block solves, the mixer's restarts used up, or Newton's step stalled)
accepts the evaluation with the smallest residual. It is counted in
`RP.diagnostics.ampere_picard`, and the run's first one warns. An evaluation that is not
finite, or that the stepper rejects, is never accepted; if the first is not finite, the solve
throws and leaves u∥, the fields and the coils as it found them (only the step's caches, the
block, its LU and the circuit matrices, are rebuilt).

Coil voltages given as functions are taken to be pure functions of time.
"""
function solve_combined_momentum_Ampere_equations_with_coils!(
        RP::RAPID{FT};
        method::OuterSolvePolicy = AndersonOuterSolve(),
        tolerance::FT = 1.0e-3,
        max_iter::Int = 20,
        E_floor::FT = FT(1.0e-6),
        I_floor::FT = FT(1.0e-6),
    ) where {FT <: AbstractFloat}
    _refuse_exprb_decay(
        RP.flags, "solve_combined_momentum_Ampere_equations_with_coils!", "takes the constant θ_imp.decay for the u∥ friction"
    )
    @timeit RAPID_TIMER "solve_combined_momentum_Ampere_equations_with_coils!" begin
        # Aliases for readability
        pla = RP.plasma
        F = RP.fields
        OP = RP.operators
        G = RP.G
        flags = RP.flags
        dt = RP.dt
        csys = RP.coil_system

        # Physical constants
        @unpack ee, me, μ0, qe = RP.config.constants
        update_charge_states!(RP)   # the ion charge density the currents read, as it is now

        # u∥'s θ-weights, as in update_ue_para!: θu on the friction and its ledger, θ_op on the
        # nonlocal operators (advection, diffusion), 0 when Implicit = false
        θu = flags.θ_imp.decay
        θ_op = flags.Implicit ? θu : zero(FT)

        # Factor for EM drive contribution
        # derived from [E_para_EM = -qe/me * (ψ^(n+1) - ψ^(n))/(R*Δt)  = -facEM * ((ψ^(n+1) - ψ^(n)))/Δt]
        facEM = (qe / me) * (F.bϕ ./ G.R2D)

        # accel_para_tilde from the state at the step's start
        accel_para_tilde = zeros(FT, G.NR, G.NZ) # Initialize acceleration field

        # pressure gradient contribution: [-∇∥(ne*Te)/(me*ne)], the share u∥ carries; the
        # ambipolar one, kept for its ledger
        a_amb = nothing
        if RP.flags.Include_ud_pressure_term
            shares = electron_pressure_shares(RP)
            accel_para_tilde .+= shares.ue
            a_amb = shares.amb
        end

        # convection -(𝐮⋅∇)u∥ and diffusion ∇⋅(D∇u∥), from the in-wall operators
        # update_ue_para! takes (`electron_operators.jl`): (1-θ) of each explicitly here, θ of
        # each inside A_u below
        pops = flags.Include_ud_convec_term || flags.Include_ud_diffu_term ? ue_Te_operators(RP) : nothing
        if flags.Include_ud_convec_term
            accel_para_tilde .+= (one(FT) - θ_op) * (-(pops.A_adv * pla.ue_para))
        end
        if flags.Include_ud_diffu_term
            accel_para_tilde .+= (one(FT) - θ_op) * (pops.A_diffu * pla.ue_para)
        end

        # Electric field contributions: (qe/me) (Eϕ_para_presc (+ Eϕ_para_ctrl) + ERZ_para_pla); the
        # induced part is solved with ψ
        if flags.E_para_cancellation
            @. accel_para_tilde += qe / me * (F.Eϕ_para_presc + F.ERZ_para_pla)
        else
            @. accel_para_tilde += qe / me * (F.Eϕ_para_presc)
        end
        if _vertical_field_control_on(RP)
            @. accel_para_tilde += qe / me * F.Eϕ_para_ctrl
        end

        # Effective electron collision frequency
        ν_sum_mom_iz_ei = pla.ν_en_mom_tot + pla.ν_en_iz_tot + pla.ν_ei_eff

        @. accel_para_tilde += (
            facEM / dt * F.ψ_coils_pla
                - (one(FT) - θu) * ν_sum_mom_iz_ei * pla.ue_para
                + pla.ν_ei_eff * pla.ui_para
        )

        # A_u = 𝟙 + Δt θ (ν + u·∇ − ∇·D∇), values only, on the wall pattern
        A_u = OP.A_u
        set_identity!(A_u)
        add_diagonal!(A_u, vec(ν_sum_mom_iz_ei); scale = dt * θu)
        # the near-vacuum relaxation update_ue_para! adds, inside the block so the current–ψ
        # relation it solves holds for the relaxed u∥: the weight here, the local mean in RHS_u
        relax_weight = vacuum_relaxation_weights(RP, pla.ne)
        ue_ref = vacuum_reference(RP, pla.ne, pla.ue_para, relax_weight)
        add_diagonal!(A_u, vec(relax_weight))
        flags.Include_ud_convec_term && add_scaled!(A_u, dt * θ_op, pops.A_adv)
        flags.Include_ud_diffu_term && add_scaled!(A_u, -dt * θ_op, pops.A_diffu)

        # Toroidal current density Jϕ @ t=(n-th step)
        Jϕ_pla_0 = @. (qe * pla.ne * pla.ue_para + pla.Zni_tot * ee * pla.ui_para) * F.bϕ

        # Predicted ψ: the step's mean induced field extrapolated linearly, 2 E(n) − E(n−1). Its
        # boundary values start the outer iteration.
        Eϕ_coils_pla_np1_pred = @. FT(2.0) * F.Eϕ_coils_pla - FT(1.0) * F.Eϕ_coils_pla_prev
        ψ_pred = @. F.ψ_coils_pla - dt * G.R2D * Eϕ_coils_pla_np1_pred
        old_ψ_coils_pla = copy(F.ψ_coils_pla)

        N, Nb, Nc = G.NR * G.NZ, length(G.BDY_idx), csys.n_total
        coils_I_n = Nc > 0 ? get_all_currents(csys) : FT[]
        # The plasma flux each coil last accounted for (Coil.ψ_pla). The circuit's
        # d/dt[2π ψ_pla(r_c)] is taken from it, not from the state at entry: that state already
        # carries what the steps since changed (ionization, transport, losses, motion), which
        # the coils have not seen yet.
        ψ_pla_coils_n = Nc > 0 ? coil_plasma_flux_memory(csys, G, Jϕ_pla_0) : FT[]

        # The circuit matrices for this Δt and the circuits' own θ-weight, as advance_coils! builds
        # them below the gate
        θ_circuit = flags.θ_imp.circuit
        if Nc > 0 && (RP.dt != csys.Δt || θ_circuit != csys.θimp)
            csys.Δt = RP.dt
            csys.θimp = θ_circuit
            calculate_circuit_matrices!(csys)
        end
        # The circuits' forcing, M Iⁿ + Δt V(tⁿ + Δt/2), once for the step: the map the outer
        # iteration solves must not change between its iterations.
        circuit_rhs_n = Nc > 0 ? calculate_LR_circuit_rhs_by_coils(csys, RP.time_s) : FT[]

        # The u∥–ψ block on its fixed pattern, so the step refactorizes values only. It does not
        # change inside the iteration below: factorized once per step here, and every iteration
        # only back-substitutes. With BlockConstraint on a closed region the surface flux joins it
        # (`sys`), with the terms its constrained rows take (`ct`).
        isnothing(OP.uψ_block) && (OP.uψ_block = CoupledBlock(A_u, OP.ΔGS))
        A_u_ψ = update_coupled_block!(OP.uψ_block, A_u, OP.ΔGS, uψ_coupling_terms(RP)...)
        solver, sys, ct = coupled_block_solver!(RP, A_u_ψ, θu)
        sol = Vector{FT}(undef, sys === nothing ? 2N : size(sys.matrix, 1))
        RHS_u = vec(@. pla.ue_para + dt * accel_para_tilde + ifelse(relax_weight > 0, relax_weight * ue_ref, zero(FT)))   # the same for every iteration
        sys === nothing || (RHS_u[ct.cu] .+= ct.ionization)
        RHS_ψ_ions = @. -μ0 * G.R2D * pla.Zni_tot * ee * pla.ui_para * F.bϕ
        RHS_ψ = similar(RHS_ψ_ions)

        # The outer unknowns x = (boundary flux, coil currents), mixed as the docstring says. The
        # least-squares weights put both in flux per unit R; a coil current counts as the flux it
        # makes on itself, L_c I_c/(2π R_c).
        R_bdy = G.R2D[G.BDY_idx]
        coil_R = FT[c.location.r for c in csys.coils]
        coil_L = FT[c.self_inductance for c in csys.coils]
        W = vcat(one(FT) ./ R_bdy, coil_L ./ (2π .* coil_R))
        stepper = outer_stepper(method, RP, W, solver, length(sol))
        # the field a coil's current residual makes on itself, for the coils inside the grid
        coil_field = zeros(FT, Nc)
        for c in csys.inside_domain_indices
            coil_field[c] = coil_L[c] / (2π * coil_R[c] * dt)
        end

        # one evaluation of the map, with its E and I residuals (and surface flux), and the best
        # one so far
        ue_k, ψ_k, Jϕ_k = similar(pla.ue_para), similar(F.ψ_coils_pla), similar(pla.ue_para)
        I_k, Φ_k, res_k = similar(coils_I_n), similar(ψ_pla_coils_n), zeros(FT, 2)
        Γ_over_B_k = zeros(FT, length(sol) - 2N)
        best = (ue = similar(ue_k), ψ = similar(ψ_k), I = similar(I_k), Φ = similar(Φ_k), Γ_over_B = similar(Γ_over_B_k), res = zeros(FT, 2))

        function evaluate!(x)
            ψ_bdy, I_in = view(x, 1:Nb), view(x, (Nb + 1):(Nb + Nc))

            # u∥ and ψ with the boundary flux held and the coils inside the grid as ψ's source
            RHS_ψ .= RHS_ψ_ions
            if Nc > 0
                RHS_ψ .+= -μ0 .* G.R2D .* distribute_coil_currents_to_Jϕ(csys, G; currents = Vector(I_in))
            end
            RHS_ψ[G.BDY_idx] .= ψ_bdy
            rhs = vcat(RHS_u, vec(RHS_ψ))
            solve!(sol, solver, sys === nothing ? rhs : constrained_rhs(sys, rhs; ct.extra))
            vec(ue_k) .= view(sol, 1:N)
            vec(ψ_k) .= view(sol, (N + 1):(2N))
            sys === nothing || (Γ_over_B_k .= view(sol, (2N + 1):length(sol)) ./ sys.scale)

            # the circuits and the boundary flux of that current
            @. Jϕ_k = (qe * pla.ne * ue_k + pla.Zni_tot * ee * pla.ui_para) * F.bϕ
            if Nc > 0
                Φ_k .= plasma_flux_at_coils(csys, G, Jϕ_k)
                I_k .= csys.inv_A_LR_circuit * (circuit_rhs_n .- 2π .* (Φ_k .- ψ_pla_coils_n))
            end
            ψ_bdy_new = (G.Green_inWall2bdy * Jϕ_k[G.nodes.in_wall_nids]) .* (G.dR * G.dZ)
            Nc > 0 && (ψ_bdy_new .+= csys.Green_coils2bdy * I_k)

            f = vcat(ψ_bdy_new .- ψ_bdy, I_k .- I_in)
            converged, res_k[1], res_k[2], finite = coupled_residual_converged(
                view(f, 1:Nb), view(f, (Nb + 1):(Nb + Nc)), R_bdy, coil_field, ψ_k, old_ψ_coils_pla, G.R2D,
                I_k, coils_I_n, dt; tol = tolerance, E_floor, I_floor,
            )
            # the evaluation is accepted whole: its induced field, u∥ and ψ must be finite too
            return f, converged, finite && all(isfinite, ue_k) && all(isfinite, ψ_k)
        end
        function keep!()
            best.ue .= ue_k
            best.ψ .= ψ_k
            best.I .= I_k
            best.Φ .= Φ_k
            best.Γ_over_B .= Γ_over_B_k
            best.res .= res_k
            return nothing
        end
        function restore!()
            ue_k .= best.ue
            ψ_k .= best.ψ
            I_k .= best.I
            Φ_k .= best.Φ
            Γ_over_B_k .= best.Γ_over_B
            res_k .= best.res
            return nothing
        end
        iter, outcome = fixed_point_solve!(evaluate!, stepper, vcat(ψ_pred[G.BDY_idx], coils_I_n); max_iter, keep!, restore!)
        outcome === :failed && error(
            "the coupled solve's first iterate is not finite (step $(RP.step)): u∥, ψ, the field " *
                "they induce over the step or the coil currents of the block solve hold NaN or Inf"
        )
        record_picard!(RP, iter, outcome === :converged, res_k[1], res_k[2], max_iter)

        # The accepted evaluation's fields. u∥, the fields and the coils are written from here on
        # only, so a solve that throws leaves them as it found them.
        # Rue_ei (electron-ion momentum exchange rate): its part at tⁿ, before u∥ moves on
        if RP.flags.Coulomb_Collision
            @. pla.Rue_ei = pla.ν_ei_eff * (pla.ui_para - (one(FT) - θu) * pla.ue_para)
        end
        u_old = isnothing(a_amb) ? nothing : copy(pla.ue_para)   # for the exchange ledger
        ue_entry = copy(pla.ue_para)                               # uⁿ, for the friction's heat
        F.ψ_coils_pla .= ψ_k
        if !closed_surface_active(RP)
            pla.ue_para .= ue_k
        elseif sys !== nothing
            # BlockConstraint: the accepted u∥ is constrained, and its surface flux is the step's
            commit_surface_flux!(RP, copy(Γ_over_B_k), ct.Bp)
            pla.ue_para .= ue_k
        else
            # PostProjection: u∥ solved again from uⁿ with the constraint, at the induced field the
            # coupled solve found; ψ stays that solve's
            c_uψ, _ = uψ_coupling_terms(RP)
            surface_constrained_ue!(RP, A_u.matrix, RHS_u .- c_uψ .* vec(ψ_k), θu)
        end
        RP.diagnostics.near_vacuum.relax_momentum +=
            me * vacuum_relaxation_exchange(RP, pla.ne, relax_weight, pla.ue_para, ue_ref)
        isnothing(a_amb) || book_ambipolar_exchange!(RP, a_amb, u_old)

        # Update self-consistent electric field: Eϕ = -∂ψ/∂t/R
        F.Eϕ_coils_pla_prev .= F.Eϕ_coils_pla
        @. F.Eϕ_coils_pla = -(F.ψ_coils_pla - old_ψ_coils_pla) / (G.R2D * dt)

        # Complete the Rue_ei calculation with second part (n+1 step contribution)
        if RP.flags.Coulomb_Collision
            @. pla.Rue_ei += pla.ν_ei_eff * (-θu * pla.ue_para)
        end
        record_ei_impulse!(RP, ue_entry)   # before continuity moves nₑ

        update_Jϕ!(RP)

        # The coil currents of the accepted current, with that current's flux as their memory
        if Nc > 0
            csys.time_s = RP.time_s + csys.Δt
            set_all_currents!(csys, I_k)
            csys.coils.ψ_pla = Φ_k
        end

        # Update magnetic fields from ψ_coils_pla
        calculate_B_from_ψ!(G, F.ψ_coils_pla, F.BR_coils_pla, F.BZ_coils_pla)

        return RP
    end # @timeit
end

# The outer iteration's stepper, by policy. W puts its unknowns in one unit: the edge flux per
# unit R, and a coil current as the flux it makes on itself, L_c I_c/(2π R_c). `solver` is the
# step's factorized block, of `nsol` unknowns.
function outer_stepper(p::AndersonOuterSolve, RP::RAPID{FT}, W::Vector{FT}, solver, nsol::Int) where {FT <: AbstractFloat}
    Nb, Nc = length(RP.G.BDY_idx), RP.coil_system.n_total
    return AndersonMixer{FT}(Nb + Nc, p.memory; β = vcat(fill(FT(p.relaxation_w), Nb), ones(FT, Nc)), W)
end

# ── The reference (DirectOuterSolve): the outer map's linear part assembled, and solved ─────
function outer_stepper(::DirectOuterSolve, RP::RAPID{FT}, W::Vector{FT}, solver, nsol::Int) where {FT <: AbstractFloat}
    T = coupled_map_jacobian(RP, solver; nsol)
    return NewtonStepper(lu(Diagonal(W) * (LinearAlgebra.I - T) / Diagonal(W)), W)
end

"""
    coupled_map_jacobian(RP, solver; nsol = 2N) -> T

The linear part T of the coupled solve's map g(x) = T x + c, x = (boundary flux, coil currents),
by columns: what a unit of each x_j gives back through the block solve (`solver`, factorized
for the step, of `nsol` unknowns: u∥ and ψ first, then the surface flux with BlockConstraint),
as the boundary flux and coil currents of the electron current it makes. The ions and every
forcing are in c. Costs N_b + N_c back-substitutions.
"""
function coupled_map_jacobian(RP::RAPID{FT}, solver::AbstractLinearSolver{FT}; nsol::Int = 2 * RP.G.NR * RP.G.NZ) where {FT <: AbstractFloat}
    G, csys = RP.G, RP.coil_system
    @unpack qe, μ0 = RP.config.constants
    N, Nb, Nc = G.NR * G.NZ, length(G.BDY_idx), csys.n_total
    dA = G.dR * G.dZ
    u_to_J = vec(@. qe * RP.plasma.ne * RP.fields.bϕ)   # the electron current of a unit u∥
    T = zeros(FT, Nb + Nc, Nb + Nc)
    for cols in Iterators.partition(1:(Nb + Nc), 64)
        # the block's right-hand sides: a unit flux on a boundary node, or a coil's unit current
        # as ψ's source inside the grid
        B = zeros(FT, nsol, length(cols))
        for (i, j) in enumerate(cols)
            if j <= Nb
                B[N + G.BDY_idx[j], i] = one(FT)
            else
                unit = zeros(FT, Nc)
                unit[j - Nb] = one(FT)
                B[(N + 1):(2N), i] .= -μ0 .* vec(G.R2D) .* vec(distribute_coil_currents_to_Jϕ(csys, G; currents = unit))
                B[N .+ G.BDY_idx, i] .= zero(FT)
            end
        end
        J = u_to_J .* view(solve!(similar(B), solver, B), 1:N, :)
        I_c = Nc > 0 ? csys.inv_A_LR_circuit * (-2π .* (csys.Green_grid2coils * J) .* dA) : zeros(FT, 0, length(cols))
        ψ_b = (G.Green_inWall2bdy * J[G.nodes.in_wall_nids, :]) .* dA
        Nc > 0 && (ψ_b .+= csys.Green_coils2bdy * I_c)
        T[1:Nb, cols] .= ψ_b
        T[(Nb + 1):end, cols] .= I_c
    end
    return T
end

"""
    coil_flux_change_by_plasma_displacement(RP, Jϕ_now, Jϕ_entry; θimp = 1)

The coil row of DYON's motional term (Kim et al. 2022, NF 62 126012, §2.1), written per grid
cell: 2π Σ_g (∂G_cg/∂R_g δR_g + ∂G_cg/∂Z_g δZ_g) I_g, with I_g the cell current and (δR, δZ)
its displacement over the step by the electron fluid velocity and the J×B acceleration. This
is MATLAB's `Ipla_dMcp`.

Not called. The circuits difference against the plasma flux they last used (`Coil.ψ_pla`),
which carries the motion one step later; adding this term would count it twice. It is the
candidate same-step predictor for fast vertical motion, used together with a stored flux
that includes it. See internal notes, design/coupled-step-coil-flux.md §4.4.
"""
function coil_flux_change_by_plasma_displacement(
        RP::RAPID{FT}, Jϕ_now::AbstractMatrix{FT}, Jϕ_entry::AbstractMatrix{FT}; θimp::FT = one(FT)
    ) where {FT <: AbstractFloat}
    G, pla, csys, dt = RP.G, RP.plasma, RP.coil_system, RP.dt
    I_cell = @. (θimp * Jϕ_now + (1 - θimp) * Jϕ_entry) * G.dR * G.dZ
    δR = @. pla.ueR * dt + 0.5 * pla.mean_aR_by_JxB * dt^2
    δZ = @. pla.ueZ * dt + 0.5 * pla.mean_aZ_by_JxB * dt^2
    return 2π * (csys.dGreen_dRg_grid2coils * vec(I_cell .* δR) + csys.dGreen_dZg_grid2coils * vec(I_cell .* δZ))
end


"""
    update_uMHD_by_global_JxB_force!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Update plasma MHD velocities using global JxB force balance.

This function computes the global toroidal force balance for closed flux surfaces
and updates the plasma mean velocities by applying JxB accelerations. The method
enforces momentum conservation across the entire plasma volume when magnetic
confinement creates closed flux surfaces.

# Physics
- Calculates global JxB forces on the plasma
- Applies momentum conservation constraints
- Updates plasma mean velocities (uMHD) based on electromagnetic forces
- Only active when closed flux surfaces are present

# Arguments
- `RP::RAPID{FT}`: RAPID simulation object containing plasma state and fields
"""
function update_uMHD_by_global_JxB_force!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    # Check if we have closed flux surfaces
    if !isempty(RP.flf.closed_surface_nids)

        @unpack me = RP.config.constants
        nids = vcat(RP.G.nodes.on_wall_nids, RP.G.nodes.in_wall_nids)
        pla = RP.plasma
        F = RP.fields
        update_charge_states!(RP)   # the ion mass density as it is now

        # Calculate total plasma mass by integrating over volume (the ions' Σ mₛ nₛ)
        sum_plasma_mass = sum(@. (pla.mni_tot[nids] + me * pla.ne[nids]) * RP.G.inVol2D[nids])
        sum_JxB_R = sum(@. pla.Jϕ[nids] * F.BZ[nids] * RP.G.inVol2D[nids])
        sum_JxB_Z = sum(@. -pla.Jϕ[nids] * F.BR[nids] * RP.G.inVol2D[nids])

        # Calculate mean accelerations
        if sum_plasma_mass > zero(FT)
            pla.mean_aR_by_JxB[nids] .= sum_JxB_R / sum_plasma_mass
            pla.mean_aZ_by_JxB[nids] .= sum_JxB_Z / sum_plasma_mass

            # TODO: Is this explicit way stable? Should we use the mean_aR and mean_aZ implicitly in some functions?
            pla.uMHD_R[nids] .+= pla.mean_aR_by_JxB[nids] * RP.dt
            pla.uMHD_Z[nids] .+= pla.mean_aZ_by_JxB[nids] * RP.dt
        end
    end

    return RP
end
