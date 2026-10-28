"""
Transport module for RAPID2D.

Contains functions related to transport phenomena, including:
- Diffusion coefficients
- Convection terms
- Source and sink terms
"""

# Export public functions
export update_transport_quantities!,
    update_diffusion_tensor!,
    calculate_particle_fluxes!

"""
    update_transport_quantities!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Update all transport-related quantities including diffusion coefficients, velocities, and collision frequencies.

Everything here is rebuilt from the current state, so a second call on the same state
changes nothing; nothing is damped or accumulated in place.
"""
function update_transport_quantities!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    pla = RP.plasma
    tp = RP.transport
    # The module-level `EE`, not `constants.ee`: the Maxwellian speed helpers convert
    # eV→J with `EE`, and the ambipolar line below has to match them bit for bit for
    # its `Te/De ≡ mₑνₑ/e` cancellation to be exact. Unpacking `ee` as well would put
    # two names for one constant in scope.
    @unpack me = RP.config.constants

    # The ion species summed first: the collision rates, the Spitzer factor and the ion
    # mass below read them.
    update_charge_states!(RP)
    # `vp_i` reaches the ion collisional diffusivity and, through the ambipolar
    # average, the ELECTRON one: the mean ion mass node by node, the species' own
    # where there is one.
    mi = pla.mi_mean
    # The electrons each ion brings, Z̄ = nₑ/Σ nₛ: the pair's pressure and drag per ion
    Z_mean = mean_ion_charge(RP)

    # The one RRC evaluation point of the step. Everything downstream — the rest of this
    # function and the whole of the next advance_timestep! — reads plasma.ν_en_* instead
    # of re-querying the tables, so a step cannot mix coefficients from two states.
    update_RRCs!(RP)

    # Sum of the electron collision frequencies that randomize directed momentum.
    # Used below only as the ν in D∥ = T/(mν): ionization is counted as a
    # momentum-randomizing event here because the newborn electron carries none of the
    # parent's drift. The same sum drives the momentum decay in `update_ue_para!`
    # (`physics.jl`) and the combined momentum-Ampère solver.
    ν_sum_mom_iz_ei = zeros(FT, size(pla.ne))
    νi_eff = zeros(FT, size(pla.ne))

    if RP.flags.Atomic_Collision
        @. ν_sum_mom_iz_ei += pla.ν_en_mom_tot + pla.ν_en_iz_tot

        iRRC_elastic = get_H2_ion_RRC(RP, RP.iRRCs, :Elastic)
        iRRC_cx = get_H2_ion_RRC(RP, RP.iRRCs, :Charge_Exchange)
        @. νi_eff += pla.n_H2_gas * (FT(0.5) * iRRC_elastic + iRRC_cx)
    end
    tp.νi_neutral .= νi_eff      # everything so far is ion-neutral
    # The ions born at rest from the gas share the ion fluid's momentum with more mass
    # (`update_ui_para!`): a drag on it, the born mass per ion mass at Z̄ν_iz births per ion and
    # second. Ionization of the gas creates the first species, H₂⁺ (`bulk_ion_mass`): exact for one
    # species. `born_ion_mass`, which the momentum update reads, also counts the ions that other
    # ions turn into (H₃⁺ by conversion); this rate leaves them out.
    if RP.flags.Atomic_Collision && RP.flags.src
        m_born = bulk_ion_mass(RP)
        @. tp.νi_birth = (m_born / mi) * Z_mean * pla.ν_en_iz_tot
    else
        fill!(tp.νi_birth, zero(FT))
    end

    # Calculate total collision frequency
    if RP.flags.Coulomb_Collision
        update_coulomb_collision_parameters!(RP)
        ν_sum_mom_iz_ei .+= pla.ν_ei_eff
        νi_eff .+= pla.ν_ii
        tp.νi_coulomb .= pla.ν_ii
    else
        fill!(tp.νi_coulomb, zero(FT))
    end
    # the ion composition this refresh saw: the species' friction on each other (`ion_transport_channels`)
    # reads it, so that one step's rates come from one state, not from densities moving within the step
    tp.ion_refresh_records = [r for records in RP.store_layout.ion_records for r in records]
    tp.ion_n_refresh = [
        copy(ion_plane(pla, (k, z))) for (k, records) in enumerate(RP.store_layout.ion_records) for (z, _) in enumerate(records)
    ]

    # `vp = √(2T/m)` is the moment that makes the line below come out at D∥ = T/(mν):
    # ½vp² = T/m exactly. Any other Maxwellian speed would need a different
    # numerical prefactor, so the `½` and the `vp` must be read together.
    vp_e = maxwellian_most_probable_speed.(pla.Te_eV, me)
    vp_i = maxwellian_most_probable_speed.(pla.Ti_eV, mi)

    # Collision-based diffusion coefficient: ½·(2T/m)/ν = T/(mν).
    # The zero test is on the SPEED: `½vp²/ν` is `0/0` for a cold collisionless cell,
    # and the old `isnan ⟹ typemax(FT)` patch answered `Inf` where a species that is
    # not moving does not diffuse. `vp > 0, ν = 0` keeps the honest collisionless
    # `Inf`, bounded downstream by the geometric ceiling.
    tp.Dpara_e_coll = @. ifelse(vp_e > 0, FT(0.5) * vp_e^2 / ν_sum_mom_iz_ei, zero(FT))

    # Kept as state for the diagnostics: no coefficient reads it. Each ion species builds its own
    # rate from `νi_neutral`, `νi_birth` and its friction on the other species (`ion_transport_channels`).
    tp.νi_eff .= νi_eff

    # The ambipolar pair, electrons and ions moving together, is dragged by the gas alone: e–i
    # friction vanishes when they move together and ion–ion collisions cannot slow the ion fluid,
    # so neither ν_ei nor ν_ii enters what drags it (coulomb-collision-0d-audit.md).
    ν_en_drag = RP.flags.Atomic_Collision ? (@. pla.ν_en_mom_tot + pla.ν_en_iz_tot) : zeros(FT, size(pla.ne))

    # The bulk ion channel on the gas: ½vp²/ν = Tᵢ/(mᵢν), ν = ν_in + ν_birth, Inf without gas.
    tp.Dpara_i_coll = @. ifelse(vp_i > 0, FT(0.5) * vp_i^2 / (tp.νi_neutral + tp.νi_birth), zero(FT))

    # The pair's drag rate per ion mass, ν_amb = (Z̄mₑν_en' + mᵢ(ν_in + ν_birth))/mᵢ: the drag on
    # its Z̄ electrons and its ion, the births at rest on both sides. Its damping rate, mₑ ≪ mᵢ.
    @. tp.ν_amb = (Z_mean * me * ν_en_drag + mi * (tp.νi_neutral + tp.νi_birth)) / mi

    # ── how D∥ᵉ is assembled ──────────────────────────────────────────────────
    #
    # The pressure-driven parallel flow is carried once (space-charge-layers-v4.md §5): by u∥
    # for the electrons' own share 1 − w, w = `ambipolar_weight` (`ue_pressure_acceleration`),
    # and for the ambipolar share w by D_amb, which both species take alike:
    #
    #   D∥ᵉ = (1 − w)·D_base + w·D_amb
    #
    # D_base is the bare base `Dpara0`: the electrons' own pressure-driven motion is u∥'s.
    # Each coefficient is bounded by the two ceilings of its own speed, from its own D0
    # (internal/docs/src/notes/design/random-step-ceiling.md):
    #
    #   3a D_geom = min(D0, ¼vL_c)       what the GEOMETRY allows
    #   3b D_flux = F₂(D0, ¼vLₙ)         what the FLUX closure allows
    #   4  D = min(D_geom, D_flux)       the tighter answer wins
    #
    # Both bound the SUM (capping inside it would let `Dpara0` escape) and both read the SAME
    # D0: they are one free-streaming limit at two lengths, so feeding D_geom into F₂
    # double-counts it — 29 % low at the crossover. The electrons' speed is v̄ₑ for both. The
    # ambipolar pair's free flight is at the sound speed c_s (3a), and what it carries across a
    # surface is the Bohm flux c_s n (3b at `bohm_flux_mean_speed`), not a Maxwellian's ¼c_s n.

    # `vm_e`, not the `vp_e` above: both ceilings are moments of the mean speed. The geometric
    # one is the speed along one axis, ⟨|v∥|⟩ = v̄/2, over the largest step, half the line on
    # average (¼v̄L_c); the flux one is the one-way flux across a surface, ⟨v_n θ(v_n)⟩ = ¼v̄.
    # The pair (½, vp) two dozen lines up is a different derivation and must not be copied
    # here — see the speed contract at the top of `transport_channels.jl`.
    vm_e = maxwellian_mean_speed.(pla.Te_eV, me)
    c_s = sound_speed.(pla.Te_eV, pla.Ti_eV, mi, Z_mean)

    # Stage 3a. The geometric ceiling: a random step cannot be longer than the
    # distance at which the geometry stops being new — wall distance, circuit, or 2πR
    # at a null (`FLF_TERMINATIONS`), so a closed surface binds at its circuit rather
    # than switching off: ¼vL_c with the connection length L_c (`wall_step_ceiling`), end to
    # end on an open line, one poloidal circuit (≈ 2πqR) on a closed surface. Only a failed trace carries `Inf`,
    # and the FLF validator never lets one reach this line.
    step_ceiling(v) = @. wall_step_ceiling(v, RP.flf.Lc_tot)

    # Stage 3b. The flux ceiling, from the SAME `D0` — never from `D_geom`.
    # `upwind = false`, overriding `flags.upwind`: this gradient describes the PROFILE, not a
    # flow. The upwind stencil picks its side from sign(ueR)/sign(ueZ) — components this
    # function refreshes further down, so the limiter would respond to a flow reversal one
    # update late — and the diffusion operator it is bounding uses centred face gradients
    # regardless.
    #
    # Raw `pla.ne`, not a smoothed copy: the bound has to apply to the field the solve
    # actually transports. The old `n_raw / |∇∥(n_SM)|` mixed two different fields and ran ~5×
    # tighter on the low-density flank of a front.
    ∇para_ne = RP.flags.limit_flux.state ? calculate_para_grad_of_scalar_F(RP, pla.ne; upwind = false) : nothing
    flux_ceiling(D0, v) = isnothing(∇para_ne) ? D0 :
        @. flux_limited_diffusivity(D0, ∇para_ne, pla.ne, v, RP.flags.limit_flux.factor)

    # Stage 4, for a D0, the speed of its free flight and the mean speed of what it carries
    # across a surface
    bounded(D0, v, vm_flux = v) = (D_flux = flux_ceiling(D0, vm_flux); @. min(min(D0, $(step_ceiling(v))), D_flux))

    # The ambipolar diffusivity: the pair's pressure nᵢe(Z̄Tₑ + Tᵢ) against its drag, c_s²/ν_amb, plus
    # the base path, under its free flight at the sound speed and its flux at the Bohm flux. At
    # Z̄ = 1, c_s²/ν_amb is the mobility-weighted harmonic mean of the two drag-only limbs, written as
    # one ratio because each limb's temperature cancels (`Te/De ≡ mₑν/e`). The numerator is tested,
    # not the quotient: zero thermal drive means zero diffusion, not 0/0; no drag at all is Inf,
    # which the ceilings bound.
    D_amb_drag = @. ifelse(
        Z_mean * pla.Te_eV + pla.Ti_eV > 0, EE * (Z_mean * pla.Te_eV + pla.Ti_eV) / (mi * tp.ν_amb), zero(FT)
    )
    tp.Dpara_amb .= bounded((@. tp.Dpara0 + D_amb_drag), c_s, bohm_flux_mean_speed.(c_s))
    D_base = bounded(fill(tp.Dpara0, size(pla.ne)), vm_e)
    w = pla.ambipolar_weight
    @. tp.Dpara = (one(FT) - w) * D_base + w * tp.Dpara_amb

    # The rest of the ambipolar pressure drive, down the temperature gradient: the pair's
    # pressure is nᵢe(Z̄Tₑ + Tᵢ), so Γ = −D_amb (∇∥n + n ∇∥ln(Z̄Tₑ + Tᵢ)) at a uniform
    # composition. D_amb carries the first part as diffusion; the second, the thermodiffusion
    # part in proportion to n, is this drift along b̂ (the ambipolar temperature-gradient drift),
    # for both species alike, at most at the sound speed. The temperature on and beyond the wall is
    # extrapolated from inside: no solve reaches that band, which keeps the room temperature it
    # started with, and a gradient read across the wall would drive the pair into it at c_s.
    T_pair = @. Z_mean * pla.Te_eV + pla.Ti_eV
    extrapolate_field_to_boundary_nodes!(RP.G, T_pair)
    ∇para_lnT = calculate_para_grad_of_scalar_F(RP, log.(max.(T_pair, floatmin(FT))); upwind = false)
    @. tp.u_para_amb_gradT = -w * clamp(tp.Dpara_amb * ∇para_lnT, -c_s, c_s)

    # What u∥ and Te conduct with: the electrons' own collisional limb under their own
    # ceilings. The ions' friction holds back the electrons' particle flux, through the
    # ambipolar field, not the heat and momentum they pass along B.
    D_cond = bounded((@. tp.Dpara0 + tp.Dpara_e_coll), vm_e)

    # What the ions' u∥ and Tᵢ conduct with along B: their own collisional limb, the gas's drag
    # and their self-collisions, which do not slow the fluid's common motion but do limit how far an
    # ion's heat and momentum random-walk; under their free flight along the line alone (no flux
    # ceiling: that one bounds a density flux). Cold ions do not conduct; without collisions the
    # ceiling binds.
    vm_i = maxwellian_mean_speed.(pla.Ti_eV, mi)
    @. tp.Dpara_cond_i = min(
        tp.Dpara0 + ifelse(
            pla.Ti_eV > zero(FT),
            ifelse(
                tp.νi_neutral + tp.νi_birth + tp.νi_coulomb > zero(FT),
                EE * pla.Ti_eV / (mi * (tp.νi_neutral + tp.νi_birth + tp.νi_coulomb)), FT(Inf),
            ),
            zero(FT),
        ),
        wall_step_ceiling(vm_i, RP.flf.Lc_tot),
    )

    # Ions carry no gradient ceiling yet — they have only stage 3a. It is NOT the one
    # line it looks like: an ion ceiling has to bound the total non-convective species
    # flux including the pinch term, it makes otherwise-grouped species operators
    # differ, and `ion_transport_channels` receives an `IonRecord` that carries no
    # density. The neutral gas is owed the same question — panel 2 of the 1-D spec is a
    # GAS problem and the gas channel has no gradient cap either. Both are queued
    # behind the ambipolar-gate work; see
    # `internal/docs/src/notes/TODO/electron-parallel-transport-gating.md`.

    # Bohm, D_B = Tₑ/(16B), shared so that the cross-field flux carries no current: the
    # electrons take the mean of what the ion species take (`ion_transport_channels`), weighted
    # by the charge each brings, Σ zₛnₛ D_B/zₛ / Σ zₛnₛ = D_B/Z̄ with the per-charge 1/Z of
    # `bohm_charge_scaling`, D_B without it
    Dperp_bohm = @. abs((1 / 16) * pla.Te_eV / RP.fields.Bϕ)
    RP.flags.bohm_charge_scaling && (Dperp_bohm ./= Z_mean)
    @. tp.Dperp = tp.Dperp0 + Dperp_bohm

    extrapolate_field_to_boundary_nodes!(RP.G, tp.Dpara)
    extrapolate_field_to_boundary_nodes!(RP.G, tp.Dperp)
    # never less than what the particles diffuse with, so the conduction correction is a
    # parallel diffusion
    extrapolate_field_to_boundary_nodes!(RP.G, D_cond)
    extrapolate_field_to_boundary_nodes!(RP.G, tp.Dpara_cond_i)
    @. tp.Dpara_cond = max(D_cond, tp.Dpara)

    # Project the parallel speeds onto (R, ϕ, Z). `"upara"` is the only representation
    # implemented: a `"uRphiZ"` mode would evolve the components directly and skip this.
    if RP.flags.upara_or_uRphiZ == "upara"
        # Calculate diamagnetic drift if enabled
        if RP.flags.diaMag_drift
            @warn "Not implemented yet: `diaMag_drift`"
            # Placeholder for diamagnetic drift calculation
            # A simplified diamagnetic drift is implemented here
            # In the full implementation, we'd calculate grad_n and grad_T accurately
            n_min = FT(1.0e6)  # Minimum density to avoid division by zero
            n_safe = copy(pla.ne)
            n_safe[n_safe .< n_min] .= n_min

            # Simple approximation of diamagnetic drift
            # In the real implementation, we'd use cal_grad_of_scalar_F
            pla.diaMag_R .= zeros(FT, size(pla.ne))
            pla.diaMag_Z .= zeros(FT, size(pla.ne))
        end

        # Update velocity components
        pla.ueR .= pla.ue_para .* RP.fields.bR
        pla.ueϕ .= pla.ue_para .* RP.fields.bϕ
        pla.ueZ .= pla.ue_para .* RP.fields.bZ

        # the ambipolar temperature-gradient drift, the electrons' and the ions' alike
        @. pla.ueR += tp.u_para_amb_gradT * RP.fields.bR
        @. pla.ueZ += tp.u_para_amb_gradT * RP.fields.bZ

        # Add ExB and diamagnetic drifts if enabled
        if RP.flags.mean_ExB
            pla.ueR .+= pla.mean_ExB_R
            pla.ueZ .+= pla.mean_ExB_Z
        end

        if RP.flags.diaMag_drift
            pla.ueR .+= pla.diaMag_R
            pla.ueZ .+= pla.diaMag_Z
        end

        if RP.flags.Global_JxB_Force
            pla.ueR .+= pla.uMHD_R
            pla.ueZ .+= pla.uMHD_Z
        end

        # Same for ion velocities
        pla.uiR .= pla.ui_para .* RP.fields.bR
        pla.uiϕ .= pla.ui_para .* RP.fields.bϕ
        pla.uiZ .= pla.ui_para .* RP.fields.bZ

        @. pla.uiR += tp.u_para_amb_gradT * RP.fields.bR
        @. pla.uiZ += tp.u_para_amb_gradT * RP.fields.bZ

        # Add ExB drift for ions too if enabled
        if RP.flags.mean_ExB
            pla.uiR .+= pla.mean_ExB_R
            pla.uiZ .+= pla.mean_ExB_Z
        end

        if RP.flags.Global_JxB_Force
            pla.uiR .+= pla.uMHD_R
            pla.uiZ .+= pla.uMHD_Z
        end
    else
        # Refused rather than skipped: nothing else writes the components, so falling
        # through would leave ueR/ueϕ/ueZ and uiR/uiϕ/uiZ at their previous-step values
        # — which `update_diffusion_tensor!` then consumes two lines below, silently.
        throw(
            ArgumentError(
                "upara_or_uRphiZ = \"$(RP.flags.upara_or_uRphiZ)\" is not implemented: " *
                    "no solver evolves the (R, ϕ, Z) velocity components directly, and " *
                    "this is the only place they are written. Use \"upara\"."
            )
        )
    end

    # update diffusion tensor (DRR,DRZ,DZZ) & (CTRR,CTRZ,CTZZ)
    update_diffusion_tensor!(RP)

    # The electron in-wall operators, once per step: nothing else writes `ueR`/`ueZ` or the
    # tensor before the next call, so every consumer within the step reads these.
    cache_electron_operators!(RP)

    return RP
end

"""
    update_diffusion_tensor!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Calculate diffusion coefficients based on field configuration and turbulence models.
"""
function update_diffusion_tensor!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    # compute RR, RZ, ZZ components of the diffusivity tensor
    F = RP.fields
    tp = RP.transport
    @. tp.DRR = tp.Dperp + (tp.Dpara - tp.Dperp) * F.bR^2
    @. tp.DRZ = (tp.Dpara - tp.Dperp) * F.bR * F.bZ
    @. tp.DZZ = tp.Dperp + (tp.Dpara - tp.Dperp) * F.bZ^2

    # Add turbulent diffusion if enabled
    if RP.flags.turb_ExB_mixing
        # In a real implementation, turbulent diffusion would be calculated based on
        # field line connection length, ExB drifts, etc.

        fpara = FT(RP.config.turbulent_diffusion_fraction_along_bpol)
        fperp = one(FT) - fpara

        # 𝐃 = [ (f⟂ 𝐈) + (f∥ - f⟂) * 𝐛𝐛]
        @. tp.DRR_turb = tp.Dpol_turb * (fperp + (fpara - fperp) * F.bpol_R^2)
        @. tp.DRZ_turb = (tp.Dpol_turb) * (fpara - fperp) * (F.bpol_R * F.bpol_Z)
        @. tp.DZZ_turb = tp.Dpol_turb * (fperp + (fpara - fperp) * F.bpol_Z^2)

        # Add turbulent diffusion to base diffusion
        @. tp.DRR .+= tp.DRR_turb
        @. tp.DRZ .+= tp.DRZ_turb
        @. tp.DZZ .+= tp.DZZ_turb
    end

    dR, dZ = RP.G.dR, RP.G.dZ

    @. tp.CTRR = RP.G.Jacob * tp.DRR / (dR * dR)
    @. tp.CTRZ = RP.G.Jacob * tp.DRZ / (dR * dZ)
    @. tp.CTZZ = RP.G.Jacob * tp.DZZ / (dZ * dZ)

    return RP
end


"""
    calculate_particle_fluxes!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Calculate particle fluxes based on density gradients and transport coefficients.
"""
function calculate_particle_fluxes!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    # Initialize arrays for density gradients
    dndR = zeros(FT, RP.G.NR, RP.G.NZ)
    dndZ = zeros(FT, RP.G.NR, RP.G.NZ)

    # Calculate density gradients (using forward/central/backward differences)
    # R-direction
    dndR[:, 1] .= (RP.plasma.ne[:, 2] .- RP.plasma.ne[:, 1]) / RP.G.dR
    dndR[:, 2:(end - 1)] .= (RP.plasma.ne[:, 3:end] .- RP.plasma.ne[:, 1:(end - 2)]) / (2 * RP.G.dR)
    dndR[:, end] .= (RP.plasma.ne[:, end] .- RP.plasma.ne[:, end - 1]) / RP.G.dR

    # Z-direction
    dndZ[1, :] .= (RP.plasma.ne[2, :] .- RP.plasma.ne[1, :]) / RP.G.dZ
    dndZ[2:(end - 1), :] .= (RP.plasma.ne[3:end, :] .- RP.plasma.ne[1:(end - 2), :]) / (2 * RP.G.dZ)
    dndZ[end, :] .= (RP.plasma.ne[end, :] .- RP.plasma.ne[end - 1, :]) / RP.G.dZ

    # Calculate fluxes
    # Diffusive flux: -D⋅∇n
    # Convective flux: n⋅v

    diffusive_flux_R = -RP.transport.DRR .* dndR - RP.transport.DRZ .* dndZ
    diffusive_flux_Z = -RP.transport.DRZ .* dndR - RP.transport.DZZ .* dndZ

    convective_flux_R = RP.plasma.ne .* RP.plasma.ueR
    convective_flux_Z = RP.plasma.ne .* RP.plasma.ueZ

    # Total flux
    RP.plasma.ptl_Flux_R .= FT(0.0)
    RP.plasma.ptl_Flux_Z .= FT(0.0)

    if RP.flags.diffu
        RP.plasma.ptl_Flux_R .+= diffusive_flux_R
        RP.plasma.ptl_Flux_Z .+= diffusive_flux_Z
    end

    if RP.flags.convec
        RP.plasma.ptl_Flux_R .+= convective_flux_R
        RP.plasma.ptl_Flux_Z .+= convective_flux_Z
    end

    return RP
end
