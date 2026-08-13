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
    calculate_electron_acceleration_by_convection,
    update_uMHD_by_global_JxB_force!,
    combine_Au_and_ΔGS_sparse_matrices,
    solve_combined_momentum_Ampere_equations_with_coils!

"""
    update_ue_para!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Update the parallel electron velocity.
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

        # Backward Euler by default (θ_imp.decay = 1): friction-dominated, so at large
        # Δt BE lands on u∞ = S/ν while CN rings about it. ExpRB replaces the constant
        # with the friction's own fitted θ(z), z = −νΔt, per cell — a better local
        # weight where the step resolves ν, but still first order and LESS implicit
        # than BE (θ_fit < 1 on a decay branch). See exprb-implementation.md §5 phase 2.
        decay_is_exprb = RP.flags.scheme.decay === ExpRB
        decay_is_exprb && _refuse_full_response_decay(RP.flags)
        decay_exponent = decay_is_exprb ? (@. exprb_cap_exponent(-ν_sum_mom_iz_ei * dt)) : nothing
        bern_decay = decay_is_exprb ? exprb_bern.(decay_exponent) : nothing
        # `θu` weights the FRICTION and the ledger that records it (per-cell under
        # ExpRB); `θ_op` weights the nonlocal operators, which the fit does not reach
        # and which keep today's constant either way. Splitting them keeps
        # `scheme.decay` from silently changing the transport treatment.
        θ_op = RP.flags.θ_imp.decay
        θu = decay_is_exprb ? exprb_theta.(decay_exponent) : θ_op

        # Rue_ei, part 1 (uⁿ). A LEDGER of the exchange over the step, so it must use
        # the quadrature the update actually performed — θ(z) under ExpRB.
        if RP.flags.Coulomb_Collision
            @. pla.Rue_ei = pla.ν_ei_eff * (pla.ui_para - (one_FT - θu) * pla.ue_para)
        end

        if RP.flags.Implicit
            OP = RP.operators
            @. OP.A_LHS = OP.II

            # #1: Electric acceleration term [qe*E_para_tot/me]
            accel_para_tilde = qe * F.E_para_tot / me

            # #2: Advection term (1-θ_op)*[-(𝐮⋅∇)*ue_para]. θ_op, not θu: the fitted
            # weight belongs to the friction's eigenvalue, not to a nonlocal operator.
            if RP.flags.Include_ud_convec_term
                accel_para_tilde .+= (one_FT - θ_op) * (-OP.𝐮∇ * pla.ue_para)
                @. OP.A_LHS += θ_op * dt * OP.𝐮∇
            end

            # #3: Pressure term [-∇∥(ne*Te)/(me*ne)]
            if RP.flags.Include_ud_pressure_term
                accel_para_tilde .+= calculate_electron_acceleration_by_pressure(RP)
            end

            # #4: collision drag force  (1-θu)*[-(ν_en_iz_tot + ν_mom + ν_ei_eff)*ue_para]
            if decay_is_exprb
                # uⁿ carries bern(−z), applied to the RHS below rather than here.
                OP.A_LHS += @views spdiagm((bern_decay .- one_FT)[:])
            else
                @. accel_para_tilde += (one_FT - θu) * (-ν_sum_mom_iz_ei * pla.ue_para)
                OP.A_LHS += @views spdiagm(θu * dt * ν_sum_mom_iz_ei[:])
            end

            # #5: momentum source from electron-ion collision [+sptz_fac*νei*ui_para]
            @. accel_para_tilde += (pla.ν_ei_eff * pla.ui_para)

            # #6: turbulent Diffusive term by ExB mixing (nonlocal — θ_op)
            if RP.flags.Include_ud_diffu_term
                accel_para_tilde .+= (one_FT - θ_op) * (OP.∇𝐃∇ * pla.ue_para)
                @. OP.A_LHS -= θ_op * dt * OP.∇𝐃∇
            end

            # bern(−z) formed as bern(z) + z, not the algebraically equal
            # 1 − (1−θ)νΔt: that subtraction cancels to nothing exactly where the
            # true value is small but meaningful.
            if decay_is_exprb
                @. OP.RHS = (bern_decay + decay_exponent) * pla.ue_para + dt * accel_para_tilde
            else
                @. OP.RHS = pla.ue_para + dt * accel_para_tilde
            end

            @timeit RAPID_TIMER "ue_para LinearSolve" begin
                pla.ue_para .= OP.A_LHS \ OP.RHS
            end
        else
            # Same two coefficients as the assembled path: 1/bern(z) divides the
            # increment, and bern(−z) = bern(z) + z scales uⁿ.
            inv_factor = decay_is_exprb ?
                (@. one_FT / bern_decay) :
                (@. one_FT / (one_FT + θu * ν_sum_mom_iz_ei * dt))
            u_coeff = decay_is_exprb ?
                (@. bern_decay + decay_exponent) :
                (@. one_FT - (one_FT - θu) * dt * ν_sum_mom_iz_ei)
            @. pla.ue_para = inv_factor * (
                pla.ue_para * u_coeff
                    + dt * (qe * F.E_para_tot / me + pla.ν_ei_eff * pla.ui_para)
            )

            if RP.flags.Include_ud_pressure_term
                accel_by_pressure = calculate_electron_acceleration_by_pressure(RP)
                @. pla.ue_para += inv_factor * dt * (accel_by_pressure)
            end

            if RP.flags.Include_ud_convec_term
                accel_by_grad_ud = calculate_electron_acceleration_by_convection(RP)
                @. pla.ue_para += inv_factor * dt * (accel_by_grad_ud)
            end
        end

        # Rue_ei, part 2 (uⁿ⁺¹)
        if RP.flags.Coulomb_Collision
            @. pla.Rue_ei += pla.ν_ei_eff * (-θu * pla.ue_para)
        end

        return RP
    end # @timeit
end


"""
    update_ui_para!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Update the parallel ion velocity.
"""
function update_ui_para!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    return @timeit RAPID_TIMER "update_ui_para!" begin
        # Alias
        cnst = RP.config.constants
        pla = RP.plasma
        m_i = bulk_ion_mass(RP)   # the ion this equation moves, not the default

        eff_atomic_coll_freq = zeros(FT, size(pla.ui_para))
        if RP.flags.Atomic_Collision
            iRRC_elastic = get_H2_ion_RRC(RP, RP.iRRCs, :Elastic)
            iRRC_cx = get_H2_ion_RRC(RP, RP.iRRCs, :Charge_Exchange)

            # ½ on elastic: those collisions shed only half the momentum.
            eff_atomic_coll_freq = @. pla.n_H2_gas * (FT(0.5) * iRRC_elastic + iRRC_cx)

            # Ions are born at rest, so ionization is a dilution rate: events per
            # second over the ions already present, n_e·ν_iz/(n_e/Z) = Z·ν_iz. Uses
            # the step-entry ν_en_iz_tot that continuity and the energy equation share.
            if RP.flags.src
                Z_i = FT(bulk_ion_charge(RP))
                # INTERIM(diz-ion-species): ν_en_iz_tot, not ν_en_iz alone, because every
                # ion is booked as H₂⁺ and so DI dilutes this population too. Why, and
                # what reverts when H⁺ becomes transportable: `REACTION_STOICHIOMETRY.diz`.
                @. eff_atomic_coll_freq += Z_i * pla.ν_en_iz_tot
            end

            # TODO: convection/pressure are ignored for ions; adding them needs a
            # Zeff review.
            replace!(eff_atomic_coll_freq, NaN => 0.0)
        end

        qi = cnst.ee

        one_FT = one(FT)
        if RP.flags.scheme.decay === ExpRB
            # Same sink form as `update_ue_para!`, same caveat — see there.
            #
            # The exponent carries the ATOMIC rate ONLY: Coulomb friction is added
            # explicitly after this update, so a Coulomb-stiff cell is stepped at
            # forward Euler whatever this says. The asymmetry predates this branch and
            # is identical under θ, so fixing it here would move the default. The
            # electron equation has no such gap — ν_ei_eff sits inside its exponent.
            decay_exponent = @. exprb_cap_exponent(-eff_atomic_coll_freq * RP.dt)
            bern_decay = exprb_bern.(decay_exponent)
            @. pla.ui_para = (
                (bern_decay + decay_exponent) * pla.ui_para +
                    RP.dt * qi * RP.fields.E_para_tot / m_i
            ) / bern_decay
        else
            # θ = 1 outright, not `θ_imp.decay`: this equation has never read the
            # weight store, so wiring it up would silently move anyone who set it.
            θ = one_FT  # Backward Euler
            @. pla.ui_para = (
                pla.ui_para * (one_FT - (one_FT - θ) * RP.dt * eff_atomic_coll_freq) +
                    RP.dt * qi * RP.fields.E_para_tot / m_i
            ) /
                (one_FT + θ * RP.dt * eff_atomic_coll_freq)
        end

        # Add electron-ion momentum transfer effect
        if RP.flags.Coulomb_Collision
            if RP.flags.Spitzer_Resistivity
                Rui_ei = @. pla.sptz_fac * (cnst.me / m_i) * pla.ν_ei * (pla.ue_para - pla.ui_para)
            else
                Rui_ei = @. (cnst.me / m_i) * pla.ν_ei * (pla.ue_para - pla.ui_para)
            end
            pla.ui_para .+= RP.dt * Rui_ei
        end
    end
end

"""
    update_Te!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Update the electron temperature based on energy balance equation.

This function evolves the electron temperature by solving the electron energy equation:
3/2 n_e ∂T_e/∂t = P_heat - P_loss + ∇·(κ_e ∇T_e) - 3/2 n_e u_e·∇T_e

where:
- P_heat includes ohmic heating and other power sources
- P_loss includes ionization, excitation, radiation, and equilibration losses
- κ_e is the thermal conductivity
- The last term represents convective transport

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
            if RP.flags.evolve_Te_inWall_only
                @warn "Implicit method for Te_evolve_inWall_only not implemented yet" maxlog = 1
            else
                @. OP.A_LHS = OP.II

                ePowers_tilde = copy(pla.ePowers.tot)

                # Calculate RHS
                # ePowers_tilde = ePowers already known at crruent time t
                # Note: diffu and conv will have (1-θimp) contribution
                # ePowers_tilde = pla.ePowers.tot - θimp * (pla.ePowers.diffu + pla.ePowers.conv)

                # Calculate LHS
                if RP.flags.Include_Te_diffu_term
                    # P_diffu = 1.5*∇𝐃∇*Te
                    @. ePowers_tilde -= θimp * pla.ePowers.diffu
                    @. OP.A_LHS -= two_thirds_FT * FT(1.5) * (dt * θimp * OP.∇𝐃∇)
                end

                if RP.flags.Include_Te_convec_term
                    # P_conv = -1.5*∇⋅(𝐮 Te) + 0.5*Te*(∇⋅𝐮)
                    @. ePowers_tilde -= θimp * pla.ePowers.conv
                    div_u = calculate_divergence(RP.G, pla.ueR, pla.ueZ)
                    OP.A_LHS .-= two_thirds_FT * (@views dt * θimp * (-FT(1.5) * OP.∇𝐮 + spdiagm(FT(0.5) * div_u[:])))
                end

                # LHS written as a deviation from the identity, so the sparsity
                # pattern — and the cached symbolic factorization — is untouched.
                # The source needs no bern: with S = (2/3e)P⁰ − λTₑⁿ, bern(−z) = bern(z) + z
                # cancels the λTₑⁿ pieces, leaving today's RHS with Tₑⁿ scaled by bern.
                if atomic_is_exprb
                    OP.A_LHS += @views spdiagm((bern_atomic .- one(FT))[:])
                    OP.RHS .= bern_atomic .* pla.Te_eV +
                        two_thirds_FT * (dt * ePowers_tilde / ee)
                else
                    OP.RHS .= pla.Te_eV + two_thirds_FT * (dt * ePowers_tilde / ee)
                end

                # Solve the linear system (cached factorization; pattern is step-stable)
                @timeit RAPID_TIMER "Te_eV LinearSolve" begin
                    factorize!(OP.Te_solver, OP.A_LHS.matrix)
                    solve!(view(pla.Te_eV, :), OP.Te_solver, view(OP.RHS, :))
                end
            end
        elseif atomic_is_exprb
            # Forward Euler with bern as a divisor on the increment.
            @. pla.Te_eV += two_thirds_FT * pla.ePowers.tot * dt / ee / bern_atomic
        else
            # Explicit method (forward Euler)
            @. pla.Te_eV += two_thirds_FT * pla.ePowers.tot * dt / ee
        end

        # Apply temperature limits
        @. pla.Te_eV = max(pla.Te_eV, RP.config.min_Te)
        @. pla.Te_eV = min(pla.Te_eV, RP.config.max_Te)

        return RP
    end # @timeit
end

"""
    update_Ti!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Update the ion temperature based on energy balance equation.

This function evolves the ion temperature by solving the ion energy equation:
(3/2)*∂Ti/∂t = P_ion_heating

where P_ion_heating includes atomic collision heating and equilibration with electrons.

# Arguments
- `RP::RAPID{FT}`: The RAPID object containing simulation state

# Returns
- `RP`: The updated RAPID object

# Notes
- Uses forward Euler time integration
- Calls update_ion_heating_powers! to compute heating terms
- Applies temperature limits after update
- Sets ion temperature equal to electron temperature if Ti_evolve is disabled
"""
function update_Ti!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    @timeit RAPID_TIMER "update_Ti!" begin
        # Update ion heating power terms
        update_ion_heating_powers!(RP)

        ee = RP.config.constants.ee
        dt = RP.dt
        pla = RP.plasma

        # Forward Euler with bern as a divisor on the increment, the same treatment
        # `update_Te!` gives the electron power and for the same reason: `iPowers.tot`
        # is a number from a table lookup with no operator for a θ to weigh.
        if RP.flags.scheme.atomic === ExpRB
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
`AbstractSpeciesRRCs{FT}`, so `RP.eRRCs.Kerg_ela.Erg_eV` infers `Any` at the call
site, and one `Any` scalar inside a `@.` costs the fused kernel its specialization.
The `::FT` is what makes it concrete again. Hoisting also avoids the `first` trap:
`Number` is iterable, so `first` under `@.` would dot into `first.(Erg_eV)` — the
identity on a vector, not a scalar floor.

Read off `Kerg_ela`, but it is every surface's floor: all of them share the one
`Erg_eV` vector the constructor read.
"""
@inline mean_energy_floor(RP::RAPID{FT}) where {FT <: AbstractFloat} =
    first(RP.eRRCs.Kerg_ela.Erg_eV)::FT

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
  - Temperature equilibration power with ions
- All powers stored in the RP.plasma.ePowers struct
"""
function update_electron_heating_powers!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    @timeit RAPID_TIMER "update_electron_heating_powers!" begin
        # Extract physical constants + reaction energies (all in RP.config.constants).
        @unpack ee, qe, me, iz_erg_eV, diss_iz_erg_eV = RP.config.constants
        # `m_i` is the ion mass electrons equilibrate with in the equi term below.
        # The elastic recoil no longer needs a separate neutral mass here: `P_en_ela`
        # already carries 2mₑ/M internally (see the ela block).
        m_i = bulk_ion_mass(RP)
        OP = RP.operators

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
        ePowers.equi .= zero_FT

        # If diffusion term is included in temperature equation
        if RP.flags.Include_Te_diffu_term
            # P_diffu = 1.5*∇𝐃∇*Te
            if RP.flags.Implicit
                ePowers.diffu .= ee * FT(1.5) * (OP.∇𝐃∇ * pla.Te_eV)
            else
                ePowers.diffu .= ee * FT(1.5) * compute_∇𝐃∇f_directly(RP, RP.plasma.Te_eV)
            end
        end

        # If convection term is included in temperature equation
        if RP.flags.Include_Te_convec_term
            # P_conv = -1.5*∇⋅(𝐮 Te) + 0.5*Te*(∇⋅𝐮)
            if RP.flags.Implicit
                ePowers.conv .= ee * (
                    -FT(1.5) * (OP.∇𝐮 * pla.Te_eV)
                        .+ FT(0.5) * pla.Te_eV .* calculate_divergence(RP.G, pla.ueR, pla.ueZ)
                )
            else
                ePowers.conv .= ee * (
                    -FT(1.5) * compute_∇f𝐮_directly(RP, pla.Te_eV)
                        .+ FT(0.5) * pla.Te_eV .* calculate_divergence(RP.G, pla.ueR, pla.ueZ)
                )
            end
        end

        if RP.flags.Include_heat_flux_term
            # NOTE: Assumption: 𝐪 ≈ p*𝐮 (heat flux is about in the order of pressure*velocity)
            # Pcond = Pheat = -∇⋅(Te * 𝐮) - Te*𝐮⋅∇(ln_n)
            ePowers.heat .= ee * (
                - compute_∇f𝐮_directly(RP, pla.Te_eV)
                    - pla.Te_eV * compute_𝐮∇f_directly(RP, log.(pla.ne))
            )
        end


        if RP.flags.Atomic_Collision
            # Every ν_en_* below is materialized by update_RRCs! at the step-entry state.
            # Reading them rather than re-querying is what makes the frictional heating
            # credited here use the *same* coefficient the momentum equation used to
            # remove that momentum, so the discrete energy budget closes.

            # Calculate velocity magnitudes for drag forces
            ue_mag_sq = @. pla.ueR^FT(2.0) .+ pla.ueϕ^FT(2.0) .+ pla.ueZ^FT(2.0)
            ue_dot_ui = @. pla.ueR * pla.uiR + pla.ueϕ * pla.uiϕ + pla.ueZ * pla.uiZ

            @. ePowers.drag = me * (
                ue_mag_sq * pla.ν_en_mom_tot
                    + (ue_mag_sq - ue_dot_ui) * pla.sptz_fac * pla.ν_ei
            )

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
            Ē_eV = @. FT(1.5) * pla.Te_eV + FT(0.5) * me * pla.ue_para^FT(2.0) / ee
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
            cold_factor = @. cold_target_factor(Ē_eV, pla.T_gas_eV, Ē_floor)
            @. ePowers.ela = pla.P_en_ela * cold_factor

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
            @. ePowers.exc = pla.P_en_exc * cold_factor

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


        # Equilibration power with ions (energy exchange from temperature differences)
        if RP.flags.Coulomb_Collision
            # Factor for energy transfer rate between electrons and ions
            @. ePowers.equi = (FT(2.0) * (m_i * me / (m_i + me)^2)) * FT(1.5) * ee * (pla.Te_eV - pla.Ti_eV) * pla.ν_ei
        end

        # Calculate total power (sum of all components)
        @. ePowers.tot = (
            ePowers.drag + ePowers.conv + ePowers.heat + ePowers.diffu
                - ePowers.ela - ePowers.dilution - ePowers.iz - ePowers.diss_iz
                - ePowers.exc - ePowers.diss_exc - ePowers.equi
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
            @views ePowers.equi[on_out_wall_nids] .= zero_FT
            @views ePowers.heat[on_out_wall_nids] .= zero_FT
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

Not in `λ` at either depth: `P_diffu` and `P_conv` are nonlocal and keep
`θ_imp.transport`; `P_heat` is nonlocal with no implicit half at all, so it stays
forward Euler — warned about here, since it is the dispatch's omission and not one
branch's. `FullLinearResponse` additionally warns about `∂ν_ei/∂Tₑ`, the one
derivative it declines to take (Spitzer-like, not an RRC surface — it under-damps the
transient only, since the fixed point does not depend on `λ`); what
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
𝔅 = (P_en_ela + P_en_exc)·(9/4)T_gas/Ē²  +  (3/2)e·(ν_iz + ν_diss_iz)  +  2μ·(3/2)e·ν_ei
```

from `P_ela`, `P_exc`, `P_dilution` and `P_equi`. Pre-migration, with
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
            FT(1.5) * pla.Te_eV + FT(0.5) * me * pla.ue_para^FT(2.0) / ee,
            pla.T_gas_eV, Ē_floor
        )
        # Dilution, both electron-producing channels.
        RP.flags.src && @. pla.exprb.eig_Te -= FT(1.5) * ee *
            (pla.ν_en_iz + pla.ν_en_diss_iz)
    end
    if RP.flags.Coulomb_Collision
        m_i = bulk_ion_mass(RP)
        μ_reduced = m_i * me / (m_i + me)^2
        @. pla.exprb.eig_Te -= (FT(2.0) * μ_reduced) * FT(1.5) * ee * pla.ν_ei
    end

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
            ue_mag_sq = @. pla.ueR^FT(2.0) + pla.ueϕ^FT(2.0) + pla.ueZ^FT(2.0)
            Ē_eV = @. FT(1.5) * pla.Te_eV + FT(0.5) * me * pla.ue_para^FT(2.0) / ee
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

        if RP.flags.Coulomb_Collision
            # ePowers.tot SUBTRACTS equi, so ∂/∂Tₑ of 2μ(3/2)e(Tₑ−T_i)ν_ei enters
            # negative. At frozen ν_ei this is pure algebra — no RRC surface — and
            # it can only damp, which is why leaving it out was an unforced loss
            # rather than a trade. `update_ion_power_jacobian!` carries the same
            # term; the two must agree about one piece of physics.
            #
            # bulk_ion_mass, not a neutral mass: one is the ion Tₑ equilibrates
            # with, the other the neutral it recoils off. Equal for H₂/H₂⁺ and
            # unequal for any other declared ion. Hoisted, because `@.` would
            # call it once per cell.
            m_i = bulk_ion_mass(RP)
            μ_reduced = m_i * me / (m_i + me)^2
            @. pla.exprb.eig_Te -= (FT(2.0) * μ_reduced) * FT(1.5) * ee * pla.ν_ei
        end

        # Gated on the terms being PRESENT, not on the flag. `ePowers.drag`'s
        # Coulomb half is charged unconditionally inside `Atomic_Collision`, using
        # whatever `sptz_fac·ν_ei` initialization left behind — so it survives
        # `Coulomb_Collision = false`, unlike `ePowers.equi`. A flag-gated warning
        # would stay silent in exactly the case where the omission is unannounced.
        # Indexed rather than broadcast: one grid-sized temporary per step is a lot
        # to allocate for a `maxlog = 1` decision.
        if any(i -> !iszero(pla.sptz_fac[i] * pla.ν_ei[i]), eachindex(pla.sptz_fac))
            @warn "scheme.atomic = ExpRB omits ∂ν_ei/∂Tₑ from exprb.eig_Te: ν_ei is Spitzer-like, " *
                "not an RRC surface, so the Tₑ dependence THROUGH ν_ei — in P_equi and in " *
                "P_drag's Coulomb half — is left out. P_equi's explicit (Tₑ−T_i) factor " *
                "is differentiated. This under-damps the transient; the fixed point is " *
                "λ-independent and unaffected." maxlog = 1
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
- `P_equi` gives `−2(m_i m_e/(m_i+m_e)²)(3/2)e·ν_ei`, with `∂ν_ei/∂T_i` omitted for
  the reason its electron counterpart is.

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
                ui_mag_sq = @. pla.uiR^FT(2.0) + pla.uiϕ^FT(2.0) + pla.uiZ^FT(2.0)
                ΔE = @. (
                    FT(0.5) * mi * ui_mag_sq - FT(1.5) * (pla.Ti_eV - pla.T_gas_eV) * ee
                )
                dK_ela = ion_rate_jacobian(RP, :Elastic)
                dK_cx = ion_rate_jacobian(RP, :Charge_Exchange)
                dν_a = @. pla.n_H2_gas * (FT(0.5) * dK_ela + dK_cx)
                @. pla.exprb.eig_Ti += dν_a * ΔE
            end
        end

        if RP.flags.Coulomb_Collision
            @. pla.exprb.eig_Ti -= (FT(2.0) * (mi * me / (mi + me)^2)) * FT(1.5) * ee * pla.ν_ei
        end

        on_out_wall_nids = RP.G.nodes.on_out_wall_nids
        if !isempty(on_out_wall_nids)
            @views pla.exprb.eig_Ti[on_out_wall_nids] .= zero_FT
        end

        @. pla.exprb.eig_Ti *= FT(2.0) / FT(3.0) / ee
        return RP
    end # @timeit
end

"""
    update_ion_heating_powers!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Update ion heating power components for ion energy equation.

This function calculates the ion power sources and sinks based on the MATLAB
`Cal_Ion_Heating_Powers` function, including:
- Atomic collision power (from elastic, charge exchange, and ionization)
- Equilibration power with electrons (if Coulomb collisions enabled)

# Arguments
- `RP::RAPID{FT}`: The RAPID object containing simulation state

# Returns
- `RP`: The updated RAPID object

# Notes
The power calculation includes:
- Energy change from atomic collisions: 0.5*mi*ui_mag_sq - 1.5*(Ti-T_gas)*ee
- Effective collision frequency: n_H2_gas*(0.5*elastic + charge_exchange + Z*ionization)
- Atomic power: collision_frequency * energy_change
- Equilibration power: matches electron equilibration power if Coulomb collisions enabled
- Total power: atomic + equilibration
- Sets power to zero outside wall boundaries
"""
function update_ion_heating_powers!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    @timeit RAPID_TIMER "update_ion_heating_powers!" begin
        # Define type-stable constants
        zero_FT = zero(FT)

        # TODO: add convection and diffusion terms for ions if needed

        # Extract physical constants
        @unpack ee, me = RP.config.constants
        mi = bulk_ion_mass(RP)

        # Alias common objects for readability
        pla = RP.plasma
        iPowers = RP.plasma.iPowers

        # Reset ion power arrays to zero (precaution against accumulation)
        iPowers.atomic .= zero_FT
        iPowers.equi .= zero_FT

        if RP.flags.Atomic_Collision
            # Get ion reaction rate coefficients
            iRRC_cx = get_H2_ion_RRC(RP, :Charge_Exchange)
            iRRC_elastic = get_H2_ion_RRC(RP, :Elastic)

            # Calculate ion velocity magnitude squared
            ui_mag_sq = @. pla.uiR^FT(2.0) + pla.uiϕ^FT(2.0) + pla.uiZ^FT(2.0)

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

        # Handle equilibration power with electrons
        if RP.flags.Coulomb_Collision
            @. iPowers.equi = (FT(2.0) * (mi * me / (mi + me)^2)) * FT(1.5) * ee * (pla.Te_eV - pla.Ti_eV) * pla.ν_ei
        end

        # Calculate total ion heating power
        @. iPowers.tot = iPowers.atomic + iPowers.equi

        # Set power to zero outside wall boundaries
        on_out_wall_nids = RP.G.nodes.on_out_wall_nids
        if !isempty(on_out_wall_nids)
            @views iPowers.tot[on_out_wall_nids] .= zero_FT
            @views iPowers.atomic[on_out_wall_nids] .= zero_FT
            @views iPowers.equi[on_out_wall_nids] .= zero_FT
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
        if RP.flags.src && RP.flags.Implicit
            # The implicit half of the ionization source needs ν_en_iz_tot (BOTH
            # electron-producing channels) as a diagonal operator. Assembled here
            # rather than in update_RRCs! so that a run with `src` off never builds
            # one. ν_en_iz_tot itself was materialized by update_RRCs! at the
            # step-entry state — do not re-query the table here.
            op.ν_en_iz_tot .= @views spdiagm(pla.ν_en_iz_tot[:])
        end

        if RP.flags.diffu
            # ∇⋅𝐃⋅∇n
            op.RHS .+= compute_∇𝐃∇f_directly(RP, pla.ne)
        end
        if RP.flags.convec
            # -∇⋅(n 𝐮)
            op.RHS .+= -compute_∇f𝐮_directly(RP, pla.ne)
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

            # Build LHS operator. Every term is gated by the SAME flag that gated
            # its explicit half above: all three used to be added unconditionally,
            # so `flags.diffu = false` removed only the explicit half and left θ·Δt
            # of the diffusion still acting implicitly, and a run that turned `src`
            # off mid-way kept ionizing through a stale ν_en_iz_tot. The ion path
            # honours the flags in full, which is how the mismatch showed up.
            #
            # Gated by ZEROING the weight rather than by branching, so this stays
            # the single fused broadcast it has always been. Four statements cost
            # 26 extra Ng-sized sparse temporaries per step at 100×200; they also
            # would have made the sparsity pattern flag-dependent, and the cached
            # factorization wants it stable.
            θ_d = RP.flags.diffu ? θ_tr : zero(FT)
            θ_c = RP.flags.convec ? θ_tr : zero(FT)
            θ_s = (RP.flags.src && !fit_growth) ? θ_gr : zero(FT)
            @. op.A_LHS = op.II - dt * (θ_d * op.∇𝐃∇ - θ_c * op.∇𝐮 + θ_s * op.ν_en_iz_tot)
            if fit_growth
                # bern(z) on the diagonal, as a deviation from the identity so the
                # pattern is untouched. This is the side that cancels on a growth
                # branch — the θ form's 1 − θνΔt passes through zero at the poles —
                # and bern(z) > 0 at every z keeps the matrix an M-matrix.
                op.A_LHS += @views spdiagm((bern_growth .- one(FT))[:])
            end

            # Solve the linear system (cached factorization; pattern is step-stable)
            @timeit RAPID_TIMER "ne LinearSolve`" begin
                factorize!(op.ne_solver, op.A_LHS.matrix)
                solve!(view(pla.ne, :), op.ne_solver, view(op.RHS, :))
            end
        elseif fit_growth
            # Same two coefficients as the assembled path: no matrix is not the same
            # as no fit. `op.RHS` still holds transport, which is not part of λ and
            # rides the increment like any other frozen source.
            @. pla.ne = ((bern_growth + pla.exprb.z_growth) * pla.ne + dt * op.RHS) / bern_growth
        else
            if RP.flags.src
                @. op.RHS += pla.ne * pla.ν_en_iz_tot
            end
            @. RP.plasma.ne += dt * op.RHS
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
    treat_electron_outside_wall!(RP::RAPID{FT}) where FT<:AbstractFloat

Apply boundary conditions for electrons outside the wall and track particle sources/losses.

This function performs three main operations:
1. **Track ionization sources**: Calculates electrons generated by ionization and adds to cumulative tracking
2. **Apply wall boundary conditions**:
- Sets electron density to zero outside the wall
- Sets electron temperature to room temperature outside the wall
3. **Correct negative densities**: Optionally removes negative densities and counts them as losses

# Arguments
- `RP::RAPID{FT}`: RAPID simulation object containing plasma state and geometry

# Details
- Uses implicit weighting when `RP.flags.Implicit = true` for ionization source calculation
- Tracks cumulative particle numbers in both 1D (volume-integrated) and 2D (spatially-resolved) formats
- Applies negative density correction when `RP.flags.negative_n_correction = true`
- Updates particle number tracker (`Ntracker`) for diagnostic purposes

# Returns
- `RP`: Modified RAPID object with updated electron density and particle tracking
"""
function treat_electron_outside_wall!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    @timeit RAPID_TIMER "treat_electron_outside_wall!" begin
        # The number of electrons ionization made this step, from the one array
        # that says so — already a per-step count, hence no Δt here. Bound outside
        # the `@.`, which would otherwise broadcast the calls themselves over `RP`.
        Δn_e = net_electron_count(check_reaction_counts(RP))
        Ne_iz = @. Δn_e * RP.G.inVol2D

        # Estimate electron loss outside the wall
        # TODO: How to accurately define the volume outside/on the wall?
        on_out_wall_nids = RP.G.nodes.on_out_wall_nids
        Ne_loss = @. RP.plasma.ne[on_out_wall_nids] * FT(2.0 * pi) * RP.G.Jacob[on_out_wall_nids] * RP.G.dR * RP.G.dZ

        # Track changes in number of electrons
        Ntracker = RP.diagnostics.Ntracker

        Ntracker.cum0D_Ne_src += sum(Ne_iz)
        @. Ntracker.cum2D_Ne_src += Ne_iz

        # One ionization event makes one electron AND one ion, so the ion ledger is
        # fed the SAME number rather than recomputing it. `treat_ion_outside_wall!`
        # used to, and by then `plasma.ne` had already been zeroed outside the wall
        # a few lines below — so ionization in that band was booked for electrons
        # and lost for ions. Guarded by the flag that decides whether the ion pass
        # runs at all, so the pairing of writers is unchanged.
        if RP.flags.update_ni_independently
            Ntracker.cum0D_Ni_src += sum(Ne_iz)
            @. Ntracker.cum2D_Ni_src += Ne_iz
        end

        Ntracker.cum0D_Ne_loss += sum(Ne_loss)
        @. Ntracker.cum2D_Ne_loss[on_out_wall_nids] += Ne_loss

        # Set electron density to zero outside the wall
        RP.plasma.ne[on_out_wall_nids] .= 0.0

        # Damp out electron temperature outside the wall
        out_wall_nids = RP.G.nodes.out_wall_nids
        @. RP.plasma.Te_eV[out_wall_nids] *= RP.damping_func[out_wall_nids]

        # Correct negative densities if enabled
        if RP.flags.negative_n_correction
            neg_n_idx = findall(RP.plasma.ne .< 0)
            if !isempty(neg_n_idx)
                ori_ne = copy(RP.plasma.ne)
                @inbounds for nid in neg_n_idx
                    rid = RP.G.nodes.rid[nid]
                    zid = RP.G.nodes.zid[nid]

                    ngh_rids = max(1, rid - 1):min(RP.G.NR, rid + 1)
                    ngh_zids = max(1, zid - 1):min(RP.G.NZ, zid + 1)

                    RP.plasma.ne[nid] = min(0.1 * abs(ori_ne[nid]), 0.01 * mean(abs.(ori_ne[ngh_rids, ngh_zids])))

                    Ne_loss = (RP.plasma.ne[nid] - ori_ne[nid]) * FT(2.0 * pi) * RP.G.Jacob[nid] * RP.G.dR * RP.G.dZ
                    Ntracker.cum0D_Ne_loss += Ne_loss
                    Ntracker.cum2D_Ne_loss[nid] += Ne_loss
                end
            end
        end

        if !RP.flags.update_ni_independently
            # Same definition the step uses; they disagreed before (ne/Zeff vs ne).
            slave_ions_to_electrons!(RP)
        end

        return RP
    end # @timeit
end


"""
    treat_ion_outside_wall!(RP::RAPID{FT}) where FT<:AbstractFloat

Applies boundary conditions for ions outside the computational domain wall. Tracks ionization
source/loss, sets boundary conditions (zero density, room temperature), generates secondary
electrons from ion wall impacts, and optionally corrects negative densities.

# Arguments
- `RP`: RAPID simulation object containing plasma state and configuration

# Returns
- Modified `RP` object with updated ion density and temperature boundary conditions
"""
function treat_ion_outside_wall!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    @timeit RAPID_TIMER "treat_ion_outside_wall!" begin
        # The ion SOURCE is not booked here. Ne_iz = Ni_iz by construction — one
        # ionization makes one of each — so `treat_electron_outside_wall!` books
        # both from one number, before it zeroes `ne` outside the wall. Recomputing
        # it here read that already-zeroed density.

        # Estimate electron loss outside the wall
        # TODO: How to accurately define the volume outside/on the wall?
        on_out_wall_nids = RP.G.nodes.on_out_wall_nids
        Ni_loss = @. RP.plasma.ni[on_out_wall_nids] * FT(2.0 * pi) * RP.G.Jacob[on_out_wall_nids] * RP.G.dR * RP.G.dZ

        # Track changes in number of ions
        Ntracker = RP.diagnostics.Ntracker

        Ntracker.cum0D_Ni_loss += sum(Ni_loss)
        @. Ntracker.cum2D_Ni_loss[on_out_wall_nids] += Ni_loss

        # Secondary electron generated by ion impacts on wall
        # TODO: needs to improve this part (somehow this should generate them inside wall)
        if RP.flags.secondary_electron
            RP.plasma.ne[on_out_wall_nids] .+= RP.flags.γ_2nd_electron * RP.plasma.ni[on_out_wall_nids]
        end

        # Set ion density to zero outside the wall
        RP.plasma.ni[on_out_wall_nids] .= 0.0

        # Set electron temperature to room temperature outside the wall
        out_wall_nids = RP.G.nodes.out_wall_nids
        @. RP.plasma.Ti_eV[out_wall_nids] *= RP.damping_func[out_wall_nids]

        # Correct negative densities if enabled
        if RP.flags.negative_n_correction
            neg_n_idx = findall(RP.plasma.ni .< 0)
            if !isempty(neg_n_idx)
                Ni_loss = @. RP.plasma.ni[neg_n_idx] * FT(2.0 * pi) * RP.G.Jacob[neg_n_idx] * RP.G.dR * RP.G.dZ
                Ntracker.cum0D_Ni_loss += sum(Ni_loss)
                @. @views Ntracker.cum2D_Ni_loss[neg_n_idx] += Ni_loss
                RP.plasma.ni[neg_n_idx] .= 0.0
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
    calculate_electron_acceleration_by_convection(RP::RAPID{FT}; num_SM::Int=2) where {FT<:AbstractFloat}

Calculate the electron acceleration due to convection

This function computes the convection term [-(ud⋅∇)ud] in the electron momentum equation.
It uses a smoothed parallel electron velocity field to improve numerical stability.

# Arguments
- `RP::RAPID{FT}`: RAPID simulation object containing plasma and grid data
- `num_SM::Int=0`: Number of smoothing iterations to apply to the velocity field
- `flag_upwind::Bool=RP.flags.upwind`: Whether to use upwind differencing for gradient calculation

# Returns
- Electron acceleration due to convection term

# Implementation Details
1. Smooths the parallel electron velocity field using the Jacobian-weighted smoothing
2. Calculates the gradient of the smoothed velocity field
3. Computes the convection term as -(ueR*∇ud_R + ueZ*∇ud_Z)
"""
function calculate_electron_acceleration_by_convection(RP::RAPID{FT}; num_SM::Int = 2, flag_upwind::Bool = RP.flags.upwind) where {FT <: AbstractFloat}

    # Smooth the density field to reduce numerical noise (skip if num_SM is 0)
    ue_para_SM = smooth_data_2D(RP.plasma.ue_para; num_SM, weighting = RP.G.Jacob)

    ∇ud_R, ∇ud_Z = calculate_grad_of_scalar_F(RP, ue_para_SM; upwind = flag_upwind)
    accel_by_convection = @. -(RP.plasma.ueR * ∇ud_R + RP.plasma.ueZ * ∇ud_Z)

    if RP.flags.limit_acceleration.state
        factor = RP.flags.limit_acceleration.factor
        min_max_accel = factor .* extrema(RP.plasma.ue_para[RP.G.nodes.in_wall_nids]) ./ RP.dt
        clamp!(accel_by_convection, min_max_accel...)
    end

    return accel_by_convection
end

"""
    solve_Ampere_equation!(RP::RAPID{FT}, F::Fields{FT}=RP.fields; plasma::Bool=true, coils::Bool=true, update_Eϕ_self::Bool=true) where {FT<:AbstractFloat}

Solve Ampère's equation (Grad-Shafranov equation) to update self-consistent magnetic fields.

Solves: ΔGS * ψ = - μ₀R²Jϕ with boundary conditions from Green's function.
Updates ψ_self, and optionally Eϕ_self.

# Arguments
- `RP::RAPID{FT}`: RAPID simulation object containing plasma, grid, and operators
- `F::Fields{FT}`: Fields object to update (defaults to `RP.fields`)

# Keyword Arguments
- `plasma::Bool=true`: Include plasma current density (Jϕ) in the source term
- `coils::Bool=true`: Include external coil current contributions to the source term
"""
function solve_Ampere_equation!(RP::RAPID{FT}, F::Fields{FT} = RP.fields; plasma::Bool = true, coils::Bool = true, update_Eϕ_self::Bool = true) where {FT <: AbstractFloat}
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

        old_ψ_self = copy(F.ψ_self)

        # solve Ampere's equation
        F.ψ_self = OP.ΔGS \ OP.RHS

        # % calculate the magnetic field from the self-consistent ψ
        calculate_B_from_ψ!(RP.G, F.ψ_self, F.BR_self, F.BZ_self)

        if update_Eϕ_self && RP.flags.E_para_self_EM
            @. F.Eϕ_self = - (F.ψ_self - old_ψ_self) / (RP.G.R2D * RP.flags.Ampere_nstep * RP.dt)
        end

        return RP
    end # @timeit
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
    solve_coupled_momentum_Ampere_equations_with_coils!(RP::RAPID{FT};
                                                        tolerance=1e-3,
                                                        max_iter=10,
                                                        relaxation_w=0.5) where {FT}

Solve coupled electron momentum and Ampère equations with coil interactions using Picard iteration.

Solves the coupled system:
- Electron parallel momentum:
    - Au ≡ [ 𝐈 + Δt*θimp*(ν_sum_mom_iz_ei + 𝐮⋅∇)]
    - Au * ue∥⁽ⁿ⁺¹⁾ = ue∥⁽ⁿ⁾ + Δt*ã∥⁽ⁿ⁾ - (qe*bϕ²/me*R)*ψ_self⁽ⁿ⁺¹⁾
- Implicit Ampere's equation:
    - [Au*ΔGS - μ₀*ne*qe²*bϕ²/me]*ψ_self⁽ⁿ⁺¹⁾ = -μ₀R² J̃ϕ⁽ⁿ⁾
- Coil circuit equations: V = Ic*Rc + Lc(dI/dt) + mutual coupling (coils+plasma)

The electromagnetic induction coupling creates strong nonlinearity requiring iterative solution.

# Arguments
- `tolerance`: Convergence tolerance for Picard iteration (default: 1e-3)
- `max_iter`: Maximum iterations (default: 10)
- `relaxation_w`: Boundary relaxation weight (default: 0.5)

# Updates
Modifies `RP.plasma.ue_para`, `RP.fields.ψ_self`, `RP.fields.Eϕ_self`, and magnetic fields.
"""
function solve_coupled_momentum_Ampere_equations_with_coils!(
        RP::RAPID{FT};
        tolerance::FT = 1.0e-6,
        max_iter::Int = 10,
        relaxation_w::FT = 0.5
    ) where {FT <: AbstractFloat}
    _refuse_exprb_decay(
        RP.flags, "solve_coupled_momentum_Ampere_equations_with_coils!", "fixes θ = 1 for the u∥ friction"
    )

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
    Z_i = FT(bulk_ion_charge(RP))   # scalar: `@.` would call it per element

    θimp = FT(1.0)  # Explicit(=0), Crank-Nicholson(=0.5), Backward Euler(=1)

    # Factor for EM drive contribution
    # derived from [E_para_EM = -qe/me * (ψ^(n+1) - ψ^(n))/(R*Δt)  = -facEM * ((ψ^(n+1) - ψ^(n)))/Δt]
    facEM = (qe / me) * (F.bϕ ./ G.R2D)

    # 1. calculate accel_para_tilde using the information at the current time step
    accel_para_tilde = zeros(FT, G.NR, G.NZ) # Initialize acceleration field

    # pressure gradient contribution: [-∇∥(ne*Te)/(me*ne)]
    if RP.flags.Include_ud_pressure_term
        accel_para_tilde .+= calculate_electron_acceleration_by_pressure(RP)
    end

    # convection contribution:  (1-θimp)*[-(𝐮⋅∇)u∥]
    if RP.flags.Include_ud_convec_term
        accel_para_tilde .+= (one(FT) - θimp) * calculate_electron_acceleration_by_convection(RP)
    end

    # Electric field contributions: [(qe/me)* (E∥_ext + E∥_self_ES)]
    if flags.E_para_self_ES
        @. accel_para_tilde += qe / me * (F.E_para_ext + F.E_para_self_ES)
    else
        @. accel_para_tilde += qe / me * (F.E_para_ext)
    end

    # Effective electron collision frequency
    ν_sum_mom_iz_ei = pla.ν_en_mom_tot + pla.ν_en_iz_tot + pla.ν_ei_eff

    @. accel_para_tilde += (
        facEM / dt * F.ψ_self
            - (one(FT) - θimp) * ν_sum_mom_iz_ei * pla.ue_para
            + pla.ν_ei_eff * pla.ui_para
    )


    # 2. Define Au matrix for the electron parallel momentum equation
    # Au ≡ [ 𝐈 + Δt*θimp*(ν_sum_mom_iz_ei + 𝐮⋅∇)]
    Au = DiscretizedOperator{FT}(dims_rz = (G.NR, G.NZ))
    Au .= OP.II + spdiagm(@views dt * θimp * ν_sum_mom_iz_ei[:])
    if flags.Include_ud_convec_term
        Au .+= dt * θimp * OP.𝐮∇
    end
    Au_X_ui_para = Au * pla.ui_para

    # Jϕ_tilde is the part of prediction of Jϕ at the next time step, using the current information
    Jϕ_tilde = @. (
        pla.ne * qe * (pla.ue_para + dt * accel_para_tilde)
            + pla.ni * (ee * Z_i) * Au_X_ui_para
    ) * F.bϕ


    # Calculate Rue_ei (electron-ion momentum exchange rate) - first part (n-th step)
    if RP.flags.Coulomb_Collision
        @. pla.Rue_ei = pla.ν_ei_eff * (pla.ui_para - (one(FT) - θimp) * pla.ue_para)
    end

    # Toroidal current density Jϕ @ t=(n-th step)
    Jϕ_pla_0 = @. (qe * pla.ne * pla.ue_para + pla.ni * (ee * Z_i) * pla.ui_para) * F.bϕ

    # 6. Initial guess for ψ_self using θ-implicit scheme with extrapolated Eϕ_self
    # Predict Eϕ_self(n+1) by linear extrapolation: 2*E(n) - E(n-1)
    Eϕ_self_np1_pred = @. FT(2.0) * F.Eϕ_self - FT(1.0) * F.Eϕ_self_prev
    # Apply θ-weighting: (1-θ)*E(n) + θ*E(n+1_predicted)
    new_ψ_self_k = @. F.ψ_self - dt * G.R2D * ((one(FT) - θimp) * F.Eϕ_self + θimp * Eϕ_self_np1_pred)

    # Prepare Picard iteration for coupled system
    F.Eϕ_self_prev .= F.Eϕ_self # Store previous Eϕ_self for self-consistency
    old_ψ_self = copy(F.ψ_self) # Store old ψ_self for convergence checking


    ue_para_k = zeros(FT, G.NR, G.NZ) # Initialize ue_para for iterationo
    new_ψ_self_kp1 = zeros(FT, G.NR, G.NZ) # Initialize next ψ_self for iteration
    RHS = zeros(FT, G.NR, G.NZ) # preallocate reusable RHS for efficiency
    Jϕ_pla_k = zeros(FT, G.NR, G.NZ) # Initialize Jϕ for iteration

    # Prepare coil_system for current calculation
    if (RP.dt != csys.Δt || θimp != csys.θimp)
        # If the time step or implicit factor have changed, recalculate the coil system matrices
        csys.Δt = RP.dt
        csys.θimp = θimp
        calculate_circuit_matrices!(csys)
    end

    # Define implicit LHS matrix for the coupled Ampere equation
    # A_imp_ampere ≡ [Au*ΔGS - μ₀*ne*qe²*bϕ²/me]
    induc_shielding_term = @. μ0 * pla.ne * qe^2 * F.bϕ^2 / me
    induc_shielding_term[G.BDY_idx] .= 0.0 # For dirichlet condition of A_imp_ampere
    A_imp_ampere = (Au * OP.ΔGS) - spdiagm(@views induc_shielding_term[:])

    # TODO: need to make it more efficient.. direct indexing is not efficient
    for nid in G.BDY_idx
        A_imp_ampere.matrix[nid, nid] = one(FT) # Dirichlet condition at boundary nodes
    end

    new_coils_I_k = zeros(FT, csys.n_total) # Initialize coil currents for iteration

    iter = 1
    converged = false
    while (true)
        # Step #1: Calculate ue_para, Jphi, coils according to new_psi_self_k
        @. RHS = pla.ue_para + dt * accel_para_tilde - facEM * new_ψ_self_k
        ue_para_k .= Au \ RHS # Solve for ue_para at (k)-th step
        @. Jϕ_pla_k = (qe * pla.ne * ue_para_k + pla.ni * (ee * Z_i) * pla.ui_para) * F.bϕ


        if csys.n_total > 0
            Mcp_dIpla = 2π * csys.Green_grid2coils * (Jϕ_pla_k[:] .- Jϕ_pla_0[:]) * G.dR * G.dZ

            if flags.convec
                # TODO: Is this part needed? Grid is not moving, so plasma movement should not affect coil currents?
                Ipla = @. (θimp * Jϕ_pla_k + (1 - θimp) * Jϕ_pla_0) * G.dR * G.dZ
                pla_displacement_R = pla.ueR * dt + 0.5 * pla.mean_aR_by_JxB * dt^2
                pla_displacement_Z = pla.ueZ * dt + 0.5 * pla.mean_aZ_by_JxB * dt^2
                # change rate of Mcp (mutual inductance between coils and plasma) due to plasma movement
                Ipla_dMcp = 2π * (
                    csys.dGreen_dRg_grid2coils * (Ipla[:] .* pla_displacement_R[:]) +
                        csys.dGreen_dZg_grid2coils * (Ipla[:] .* pla_displacement_Z[:])
                )

                # grad_Ipla_R, grad_Ipla_Z = Cal_grad_of_scalar_F(reshape(Ipla, size(R2D)))
                # dIpla_by_conv = pla_displacement_R .* grad_Ipla_R + pla_displacement_Z .* grad_Ipla_Z
                # Mcp_dIpla_by_conv = 2π * coils.G_grid2coil * dIpla_by_conv[:]
                Mcp_dIpla_by_conv = 0
            else
                Ipla_dMcp = 0
                Mcp_dIpla_by_conv = 0
            end

            coil_flux_change_by_plasma = @. Mcp_dIpla + Ipla_dMcp + Mcp_dIpla_by_conv

            circuit_rhs = calculate_LR_circuit_rhs_by_coils(csys, RP.time_s) - coil_flux_change_by_plasma
            new_coils_I_k = csys.inv_A_LR_circuit * circuit_rhs  # valid if "dt" is constant
        end

        # Step #2: Update Boundary psi by both plasma and coils currents using Green's function
        new_ψ_self_kp1_at_BDY = (G.Green_inWall2bdy * Jϕ_pla_k[G.nodes.in_wall_nids]) * G.dR * G.dZ
        if csys.n_total > 0
            new_ψ_self_kp1_at_BDY .+= csys.Green_coils2bdy * new_coils_I_k
        end

        #  update (k+1)-th boudary psi with some relaxation
        @views new_ψ_self_kp1_at_BDY .= (
            relaxation_w * new_ψ_self_kp1_at_BDY
                + (one(FT) - relaxation_w) * new_ψ_self_k[G.BDY_idx]
        )

        # Step #3: Set RHS of the implicit Ampere equation
        @. RHS = -μ0 * G.R2D * Jϕ_tilde
        if csys.n_total > 0
            inside_Jϕ_coil_k = distribute_coil_currents_to_Jϕ(csys, RP.G; currents = new_coils_I_k)
            @. RHS .+= -μ0 * G.R2D * inside_Jϕ_coil_k
        end
        RHS[G.BDY_idx] .= new_ψ_self_kp1_at_BDY # Set RHS at boundary nodes

        # Step #4: Solve the implicit Ampere equation
        new_ψ_self_kp1 = A_imp_ampere \ RHS

        # Step #5: Check if ψ solution is converged
        convergence_rate = norm(new_ψ_self_kp1 - new_ψ_self_k) / norm(new_ψ_self_k)
        if (convergence_rate < tolerance)
            # println("  ψ_self converged after $iter iterations! convergence_rate: $convergence_rate")
            converged = true
            break
        elseif iter >= max_iter
            converged = false
            println("  Warning: Picard iteration did not converge after $max_iter iterations at step=$(RP.step)")
            println("  Final change: $(norm(new_ψ_self_kp1 - new_ψ_self_k) / norm(new_ψ_self_k))")
            break
        else
            new_ψ_self_k .= new_ψ_self_kp1 # Update for next iteration
            iter += 1
        end
    end

    # 7. Final updates of electromagnetic fields
    @. F.ψ_self = new_ψ_self_kp1

    # Update self-consistent electric field: Eϕ = -∂ψ/∂t/R
    @. F.Eϕ_self = -(F.ψ_self - old_ψ_self) / (G.R2D * dt)

    # Update parallel electron velocity: ue_para = (ψ_self - ψ_self_old)/(R*Δt) + ue_para_k
    @. RHS = pla.ue_para + dt * accel_para_tilde - facEM * F.ψ_self
    pla.ue_para = Au \ RHS # Solve for ue_para at (k)-th step

    # Complete the Rue_ei calculation with second part (n+1 step contribution)
    if RP.flags.Coulomb_Collision
        @. pla.Rue_ei += pla.ν_ei_eff * (-θimp * pla.ue_para)
    end

    @. pla.Jϕ = pla.ne * qe * pla.ue_para * F.bϕ

    # Update coil currents
    if RP.coil_system.n_total > 0
        csys.time_s += csys.Δt
        set_all_currents!(csys, new_coils_I_k)
    end

    # Update magnetic fields from ψ_self
    calculate_B_from_ψ!(G, F.ψ_self, F.BR_self, F.BZ_self)

    return RP
end


"""
    combine_Au_and_ΔGS_sparse_matrices(RP::RAPID{FT}, Au::SparseMatrixCSC{FT,Int}, A_GS::SparseMatrixCSC{FT,Int}) where {FT<:AbstractFloat}

Combine the electron parallel momentum operator and Grad–Shafranov operator into a single block sparse matrix for coupled solves.

Constructs a 2×2 block matrix of size (2N×2N):

    [ Au         diag(inductive_term);
      diag(current_term)   A_GS        ]

where:
- `Au` is the electron parallel momentum operator.
- `A_GS` is the Grad–Shafranov operator.
- `inductive_term = (qe/me) * (bϕ ./ R2D)` couples poloidal flux changes into the momentum equation.
- `current_term = μ0 * R2D * ne * qe * bϕ` couples plasma current into Ampère's equation.

# Arguments
- `RP::RAPID{FT}`: Simulation state, providing grid geometry and physical constants.
- `Au::SparseMatrixCSC{FT,Int}`: Momentum operator matrix.
- `A_GS::SparseMatrixCSC{FT,Int}`: Grad–Shafranov operator matrix.

# Returns
- `SparseMatrixCSC{FT,Int}`: Combined block sparse matrix of size (2N×2N), where N = RP.G.NR * RP.G.NZ.

# Notes
- Boundary conditions are enforced by zeroing coupling terms at boundary nodes.
"""
function combine_Au_and_ΔGS_sparse_matrices(RP::RAPID{FT}, Au::SparseMatrixCSC{FT}, A_GS::SparseMatrixCSC{FT}) where {FT <: AbstractFloat}
    # Get dimensions
    N = RP.G.NR * RP.G.NZ

    # Physical constants
    @unpack qe, me, μ0 = RP.config.constants

    # Calculate coupling terms
    inductive_term = @. qe * RP.fields.bϕ / (me * RP.G.R2D)
    electron_current_term = @. μ0 * RP.G.R2D * RP.plasma.ne * qe * RP.fields.bϕ
    # Zero out boundary terms for proper boundary conditions
    electron_current_term[RP.G.BDY_idx] .= zero(FT)

    # Count non-zero entries for efficient allocation
    # Upper left: A_upara entries
    # Lower right: A_GS entries
    # Upper right: N diagonal entries (inductive coupling)
    # Lower left: N diagonal entries (current coupling)
    nnz_A_upara = nnz(Au)
    nnz_A_GS = nnz(A_GS)
    nnz_coupling = 2 * N  # Two diagonal blocks
    total_nnz = nnz_A_upara + nnz_A_GS + nnz_coupling

    # Pre-allocate arrays for sparse matrix construction
    I_combined = zeros(Int, total_nnz)
    J_combined = zeros(Int, total_nnz)
    V_combined = zeros(FT, total_nnz)

    idx = 1

    # Upper left block: A_upara (rows 1:N, cols 1:N)
    I_up, J_up, V_up = findnz(Au)
    len_upara = length(I_up)
    I_combined[idx:(idx + len_upara - 1)] = I_up
    J_combined[idx:(idx + len_upara - 1)] = J_up
    V_combined[idx:(idx + len_upara - 1)] = V_up
    idx += len_upara

    # Lower right block: A_GS (rows N+1:2N, cols N+1:2N)
    I_gs, J_gs, V_gs = findnz(A_GS)
    len_gs = length(I_gs)
    I_combined[idx:(idx + len_gs - 1)] = I_gs .+ N  # Shift row indices
    J_combined[idx:(idx + len_gs - 1)] = J_gs .+ N  # Shift column indices
    V_combined[idx:(idx + len_gs - 1)] = V_gs
    idx += len_gs

    # Upper right block: inductive coupling (rows 1:N, cols N+1:2N)
    # Diagonal matrix: (i,i) -> value inductive_term[i]
    for i in 1:N
        I_combined[idx] = i      # Row index
        J_combined[idx] = i + N  # Column index (shifted to upper right block)
        V_combined[idx] = inductive_term[i]
        idx += 1
    end

    # Lower left block: current coupling (rows N+1:2N, cols 1:N)
    # Diagonal matrix: (i,i) -> value electron_current_term[i]
    for i in 1:N
        I_combined[idx] = i + N  # Row index (shifted to lower left block)
        J_combined[idx] = i      # Column index
        V_combined[idx] = electron_current_term[i]
        idx += 1
    end

    return sparse(I_combined, J_combined, V_combined, 2 * N, 2 * N)
end

"""
    solve_combined_momentum_Ampere_equations_with_coils!(RP::RAPID{FT};
                                                         tolerance::FT=1e-3,
                                                         max_iter::Int=10,
                                                         relaxation_w::FT=0.5) where {FT<:AbstractFloat}

Solve the coupled electron momentum and Ampère equations with coil interactions using a single block-sparse solver.

This method:
1. Assembles the combined block matrix via `combine_Au_and_ΔGS_sparse_matrices`.
2. Constructs a unified linear system for `ue_para` and `ψ_self`.
3. Performs Picard iteration to update:
   - `RP.plasma.ue_para`
   - `RP.fields.ψ_self`
   - `RP.fields.Eϕ_self`
   - External coil currents and resulting magnetic fields.

# Arguments
- `RP::RAPID{FT}`: Simulation state object, modified in place.
- `tolerance::FT=1e-3`: Convergence tolerance for the Picard iteration.
- `max_iter::Int=10`: Maximum number of Picard iterations.
- `relaxation_w::FT=0.5`: Relaxation weight for boundary ψ updates.

# Returns
- `RP::RAPID{FT}`: The updated simulation object with new plasma and field values.
"""
function solve_combined_momentum_Ampere_equations_with_coils!(
        RP::RAPID{FT};
        tolerance::FT = 1.0e-6,
        max_iter::Int = 10,
        relaxation_w::FT = 0.5
    ) where {FT <: AbstractFloat}
    _refuse_exprb_decay(
        RP.flags, "solve_combined_momentum_Ampere_equations_with_coils!", "fixes θ = 1 for the u∥ friction"
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


        # TODO: diffusion term

        # Physical constants
        @unpack ee, me, μ0, qe = RP.config.constants
        Z_i = FT(bulk_ion_charge(RP))   # scalar: `@.` would call it per element

        θimp = FT(1.0)  # Explicit(=0), Crank-Nicholson(=0.5), Backward Euler(=1)

        # Factor for EM drive contribution
        # derived from [E_para_EM = -qe/me * (ψ^(n+1) - ψ^(n))/(R*Δt)  = -facEM * ((ψ^(n+1) - ψ^(n)))/Δt]
        facEM = (qe / me) * (F.bϕ ./ G.R2D)

        # 1. calculate accel_para_tilde using the information at the current time step
        accel_para_tilde = zeros(FT, G.NR, G.NZ) # Initialize acceleration field

        # pressure gradient contribution: [-∇∥(ne*Te)/(me*ne)]
        if RP.flags.Include_ud_pressure_term
            accel_para_tilde .+= calculate_electron_acceleration_by_pressure(RP)
        end

        # convection contribution:  (1-θ)*[-(𝐮⋅∇)u∥]
        if RP.flags.Include_ud_convec_term
            accel_para_tilde .+= (one(FT) - θimp) * calculate_electron_acceleration_by_convection(RP)
        end

        # Electric field contributions: [(qe/me)* (E∥_ext + E∥_self_ES)]
        if flags.E_para_self_ES
            @. accel_para_tilde += qe / me * (F.E_para_ext + F.E_para_self_ES)
        else
            @. accel_para_tilde += qe / me * (F.E_para_ext)
        end

        # Effective electron collision frequency
        ν_sum_mom_iz_ei = pla.ν_en_mom_tot + pla.ν_en_iz_tot + pla.ν_ei_eff

        @. accel_para_tilde += (
            facEM / dt * F.ψ_self
                - (one(FT) - θimp) * ν_sum_mom_iz_ei * pla.ue_para
                + pla.ν_ei_eff * pla.ui_para
        )

        A_u = OP.II + spdiagm(@views dt * θimp * ν_sum_mom_iz_ei[:])
        if flags.Include_ud_convec_term
            A_u += dt * θimp * (OP.𝐮∇.matrix)
        end

        # Calculate Rue_ei (electron-ion momentum exchange rate) - first part (n-th step)
        if RP.flags.Coulomb_Collision
            @. pla.Rue_ei = pla.ν_ei_eff * (pla.ui_para - (one(FT) - θimp) * pla.ue_para)
        end


        # Toroidal current density Jϕ @ t=(n-th step)
        Jϕ_pla_0 = @. (qe * pla.ne * pla.ue_para + pla.ni * (ee * Z_i) * pla.ui_para) * F.bϕ


        # 6. Initial guess for ψ_self using θ-implicit scheme with extrapolated Eϕ_self
        # Predict Eϕ_self(n+1) by linear extrapolation: 2*E(n) - E(n-1)
        Eϕ_self_np1_pred = @. FT(2.0) * F.Eϕ_self - FT(1.0) * F.Eϕ_self_prev
        # Apply θ-weighting: (1-θ)*E(n) + θ*E(n+1_predicted)
        new_ψ_self_k = @. F.ψ_self - dt * G.R2D * ((one(FT) - θimp) * F.Eϕ_self + θimp * Eϕ_self_np1_pred)

        # Prepare Picard iteration for coupled system
        F.Eϕ_self_prev .= F.Eϕ_self # Store previous Eϕ_self for self-consistency
        old_ψ_self = copy(F.ψ_self) # Store old ψ_self for convergence checking


        ue_para_k = zeros(FT, G.NR, G.NZ) # Initialize ue_para for iterationo
        ue_para_kp1 = zeros(FT, G.NR, G.NZ) # Initialize ue_para for iterationo
        new_ψ_self_kp1 = zeros(FT, G.NR, G.NZ) # Initialize next ψ_self for iteration

        new_coils_I_k = zeros(FT, csys.n_total) # Initialize coil currents for iteration
        coil_flux_change_by_plasma = zeros(FT, csys.n_total)

        RHS_u = zeros(FT, G.NR, G.NZ) # preallocate reusable RHS related to u
        RHS_ψ = zeros(FT, G.NR, G.NZ) # preallocate reusable RHS relatedl to ψ
        Jϕ_pla_k = zeros(FT, G.NR, G.NZ) # Initialize Jϕ for iteration

        # Prepare coil_system for current calculation
        if (RP.dt != csys.Δt || θimp != csys.θimp)
            # If the time step or implicit factor have changed, recalculate the coil system matrices
            csys.Δt = RP.dt
            csys.θimp = θimp
            calculate_circuit_matrices!(csys)
        end

        A_u_ψ = combine_Au_and_ΔGS_sparse_matrices(RP, A_u, OP.ΔGS.matrix)
        @. RHS_u = pla.ue_para + dt * accel_para_tilde - facEM * new_ψ_self_k
        @views ue_para_k[:] .= A_u \ RHS_u[:] # Solve for ue_para at (k)-th step

        iter = 1
        converged = false
        while true
            # Step #1: Calculate ue_para, Jphi, coils according to new_psi_self_k
            @. Jϕ_pla_k = (qe * pla.ne * ue_para_k + pla.ni * (ee * Z_i) * pla.ui_para) * F.bϕ


            if csys.n_total > 0
                Mcp_dIpla = 2π * csys.Green_grid2coils * (Jϕ_pla_k[:] .- Jϕ_pla_0[:]) * G.dR * G.dZ

                if flags.convec
                    # TODO: Is this part needed? Grid is not moving, so plasma movement should not affect coil currents?
                    Ipla = @. (θimp * Jϕ_pla_k + (1 - θimp) * Jϕ_pla_0) * G.dR * G.dZ
                    pla_displacement_R = pla.ueR * dt + 0.5 * pla.mean_aR_by_JxB * dt^2
                    pla_displacement_Z = pla.ueZ * dt + 0.5 * pla.mean_aZ_by_JxB * dt^2
                    # change rate of Mcp (mutual inductance between coils and plasma) due to plasma movement
                    Ipla_dMcp = 2π * (
                        csys.dGreen_dRg_grid2coils * (Ipla[:] .* pla_displacement_R[:]) +
                            csys.dGreen_dZg_grid2coils * (Ipla[:] .* pla_displacement_Z[:])
                    )

                    # grad_Ipla_R, grad_Ipla_Z = Cal_grad_of_scalar_F(reshape(Ipla, size(R2D)))
                    # dIpla_by_conv = pla_displacement_R .* grad_Ipla_R + pla_displacement_Z .* grad_Ipla_Z
                    # Mcp_dIpla_by_conv = 2π * coils.G_grid2coil * dIpla_by_conv[:]
                    Mcp_dIpla_by_conv = 0
                else
                    Ipla_dMcp = 0
                    Mcp_dIpla_by_conv = 0
                end

                @. coil_flux_change_by_plasma = Mcp_dIpla + Ipla_dMcp + Mcp_dIpla_by_conv

                circuit_rhs = calculate_LR_circuit_rhs_by_coils(csys, RP.time_s) - coil_flux_change_by_plasma
                new_coils_I_k = csys.inv_A_LR_circuit * circuit_rhs  # valid if "dt" is constant
            end

            # Step #2: Update Boundary psi by both plasma and coils currents using Green's function
            new_ψ_self_kp1_at_BDY = (G.Green_inWall2bdy * Jϕ_pla_k[G.nodes.in_wall_nids]) * G.dR * G.dZ
            if csys.n_total > 0
                new_ψ_self_kp1_at_BDY .+= csys.Green_coils2bdy * new_coils_I_k
            end

            #  update (k+1)-th boudary psi with some relaxation
            @views new_ψ_self_kp1_at_BDY .= (
                relaxation_w * new_ψ_self_kp1_at_BDY
                    + (one(FT) - relaxation_w) * new_ψ_self_k[G.BDY_idx]
            )

            # Step #3: Set RHS of the implicit Ampere equation
            @. RHS_u = pla.ue_para + dt * accel_para_tilde

            @. RHS_ψ = -μ0 * G.R2D * pla.ni * ee * Z_i * pla.ui_para * F.bϕ
            if csys.n_total > 0
                inside_Jϕ_coil_k = distribute_coil_currents_to_Jϕ(csys, RP.G; currents = new_coils_I_k)
                @. RHS_ψ += -μ0 * G.R2D * inside_Jϕ_coil_k
            end
            RHS_ψ[G.BDY_idx] .= new_ψ_self_kp1_at_BDY

            # Step #4: Solve the implicit Ampere equation
            @views RHS_u_ψ = vcat(RHS_u[:], RHS_ψ[:])
            sol = A_u_ψ \ RHS_u_ψ

            @views ue_para_kp1[:] .= sol[1:(G.NR * G.NZ)]
            @views new_ψ_self_kp1[:] .= sol[(G.NR * G.NZ + 1):end]


            # Step #5: Check if ψ solution is converged
            convergence_rate = norm(new_ψ_self_kp1 - new_ψ_self_k) / norm(new_ψ_self_k)
            if (convergence_rate < tolerance)
                # println("  ψ_self converged after $iter iterations! convergence_rate: $convergence_rate")
                converged = true
                break
            elseif iter >= max_iter
                converged = false
                println("  Warning: Picard iteration did not converge after $max_iter iterations at step=$(RP.step)")
                println("  Final change: $(norm(new_ψ_self_kp1 - new_ψ_self_k) / norm(new_ψ_self_k))")
                break
            else
                new_ψ_self_k .= new_ψ_self_kp1 # Update for next iteration
                ue_para_k .= ue_para_kp1
                iter += 1
            end
        end

        # 7. Final updates of electromagnetic fields
        F.ψ_self .= new_ψ_self_kp1
        pla.ue_para .= ue_para_kp1

        # Update self-consistent electric field: Eϕ = -∂ψ/∂t/R
        @. F.Eϕ_self = -(F.ψ_self - old_ψ_self) / (G.R2D * dt)

        # Complete the Rue_ei calculation with second part (n+1 step contribution)
        if RP.flags.Coulomb_Collision
            @. pla.Rue_ei += pla.ν_ei_eff * (-θimp * pla.ue_para)
        end

        @. pla.Jϕ = (pla.ne * qe * pla.ue_para + pla.ni * (ee * Z_i) * pla.ui_para) * F.bϕ

        # Update coil currents
        if RP.coil_system.n_total > 0
            csys.time_s += csys.Δt
            set_all_currents!(csys, new_coils_I_k)
        end

        # Update magnetic fields from ψ_self
        calculate_B_from_ψ!(G, F.ψ_self, F.BR_self, F.BZ_self)

        return RP
    end # @timeit
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
        mi = bulk_ion_mass(RP)   # the declared species, as transport uses
        nids = vcat(RP.G.nodes.on_wall_nids, RP.G.nodes.in_wall_nids)
        pla = RP.plasma
        F = RP.fields

        # Calculate total plasma mass by integrating over volume
        sum_plasma_mass = sum(@. (mi * pla.ni[nids] + me * pla.ne[nids]) * RP.G.inVol2D[nids])
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
