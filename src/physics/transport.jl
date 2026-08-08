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
"""
function update_transport_quantities!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    pla = RP.plasma
    tp = RP.transport
    # The module-level `EE`, not `constants.ee`: the Maxwellian speed helpers convert
    # eV→J with `EE`, and the ambipolar line below has to match them bit for bit for
    # its `Te/De ≡ mₑνₑ/e` cancellation to be exact. Unpacking `ee` as well would put
    # two names for one constant in scope.
    @unpack me = RP.config.constants
    # `vp_i` reaches the ion collisional diffusivity and, through the ambipolar
    # average, the ELECTRON one — so a mass that disagrees with the species the
    # ion equation transports would move both.
    mi = bulk_ion_mass(RP)

    # Charge state first: `Zeff` reaches the Spitzer factor a few lines below. The
    # charge density and the quasineutrality slaving do NOT come through here —
    # they take the species' own charge, which is why nothing downstream depends on
    # this call having run.
    update_charge_states!(RP)

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
        @. ν_sum_mom_iz_ei += pla.ν_en_mom_tot + pla.ν_en_iz

        iRRC_elastic = get_H2_ion_RRC(RP, RP.iRRCs, :Elastic)
        iRRC_cx = get_H2_ion_RRC(RP, RP.iRRCs, :Charge_Exchange)
        @. νi_eff += pla.n_H2_gas * (FT(0.5) * iRRC_elastic + iRRC_cx)
    end
    tp.νi_neutral .= νi_eff      # everything so far is ion-neutral

    # Calculate total collision frequency
    if RP.flags.Coulomb_Collision
        update_coulomb_collision_parameters!(RP)
        ν_sum_mom_iz_ei .+= pla.ν_ei_eff
        νi_eff .+= pla.ν_ii
        tp.νi_coulomb .= pla.ν_ii
    else
        fill!(tp.νi_coulomb, zero(FT))
    end

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

    # Kept as state, not a local: the ion continuity equation rebuilds D∥ per
    # species from its own mass, so it needs the collision frequency rather than
    # the H₂⁺ diffusivity this line happens to produce.
    tp.νi_eff .= νi_eff

    tp.Dpara_i_coll = @. ifelse(vp_i > 0, FT(0.5) * vp_i^2 / νi_eff, zero(FT))

    # Ambipolar diffusivity. The textbook form is a mobility-weighted harmonic mean,
    #
    #     D_a = (Te+Ti)·De·Di/(Ti·De + Te·Di) = (Te+Ti) / (Ti/Di + Te/De)
    #
    # and BOTH limbs of that denominator lose their temperature exactly, because this
    # code builds D from a known rate rather than measuring it: `De = Te·e/(mₑνₑ)`
    # makes `Te/De ≡ mₑνₑ/e`, and likewise `Ti/Di ≡ mᵢνᵢ/e`. So the whole coefficient
    # is a ratio of thermal drive to collisional friction, with no cancelling
    # temperature anywhere:
    # The numerator is tested, not the quotient: `0/0` is reachable when both T's and
    # both ν's vanish, and zero thermal drive means zero diffusion, not undefined.
    tp.Dpara_amb = @. ifelse(
        pla.Te_eV + pla.Ti_eV > 0,
        EE * (pla.Te_eV + pla.Ti_eV) / (me * ν_sum_mom_iz_ei + mi * νi_eff),
        zero(FT)
    )
    #
    # Identical to the product form to 4e-16 relative wherever that form is finite,
    # and finite where it is not. The 0/0 this removes was not hypothetical:
    # `treat_electron_outside_wall!` sets `Te = 0` on every out-wall node, at which
    # point the product form is `(Ti·0·Di)/(Ti·0 + 0·Di)` and the `NaN → typemax(FT)`
    # patch that used to follow turned "cold electrons, so no electron diffusion"
    # into `D_a = Inf`. That is the same inverted sign the flux limiter's old `Lₙ`
    # guard had, and it made `∇𝐃∇` singular one step later.

    # ── how D∥ᵉ is assembled ──────────────────────────────────────────────────
    #
    #   1  1/D_ch = Σ_p 1/D_p            competing termination events
    #   2  D0 = D_ch + Dpara0            independent arrival paths
    #   3a D_geom = min(D0, ¼v̄(Lf+Lb))   what the GEOMETRY allows, from D0
    #   3b D_flux = F₂(D0, ¼v̄Lₙ)         what the FLUX closure allows, from D0
    #   4  D∥ = min(D_geom, D_flux)      the tighter answer wins
    #
    # Both ceilings bound the SUM (capping inside it would let `Dpara0` escape), and
    # both read the SAME D0: they are one free-streaming limit at two lengths, so
    # feeding D_geom into F₂ double-counts it — 29 % low at the crossover. Derived in
    # internal/docs/src/notes/design/random-step-ceiling.md.

    # Stage 1. Harmonic mean of the collisional and ambipolar limbs, written in
    # inverse space — the same move the flux ceiling makes, for the same reason.
    # `inv(0) === Inf` and `inv(Inf) === 0.0` exactly, so a dead limb (`D = 0`, cold
    # electrons) shuts the channel and a frictionless one (`D = Inf`, ν → 0) drops
    # out of it, both without a branch. The product form needed a NaN patch for
    # precisely those two cases and answered `Inf` to both.
    tp.Dpara_e_eff = @. inv(inv(tp.Dpara_e_coll) + inv(tp.Dpara_amb))

    # Stage 2. A base diffusivity is an independent arrival path, so it ADDS rather
    # than joining the harmonic sum — the same rule the ion channel states.
    D0 = @. tp.Dpara0 + tp.Dpara_e_eff

    # `vm_e`, not the `vp_e` above: both ceilings are ONE-WAY fluxes across a surface,
    # ⟨v_n θ(v_n)⟩ = ¼v̄, and only the mean speed has that meaning. The pair (½, vp) two
    # dozen lines up is a different derivation and must not be copied here — see the
    # speed contract at the top of `transport_channels.jl`.
    vm_e = maxwellian_mean_speed.(pla.Te_eV, me)

    # Stage 3a. The geometric ceiling: a random step cannot be longer than the
    # distance at which the geometry stops being new — wall distance, circuit, or 2πR
    # at a null (`FLF_TERMINATIONS`), so a closed surface binds at its circuit rather
    # than switching off. Forward/backward bound disjoint halves of velocity space and
    # enter as an arithmetic mean (`wall_step_ceiling`). Only a failed trace carries
    # `Inf`, and the FLF validator never lets one reach this line.
    D_geom = @. min(D0, wall_step_ceiling(vm_e, RP.flf.Lc_forward, RP.flf.Lc_backward))

    # Stage 3b. The flux ceiling, from the SAME `D0` — never from `D_geom`.
    D_flux = if RP.flags.limit_flux.state
        # `upwind = false`, overriding `flags.upwind`: this gradient describes the
        # PROFILE, not a flow. The upwind stencil picks its side from sign(ueR)/
        # sign(ueZ) — components this function refreshes further down, so the limiter
        # would respond to a flow reversal one update late — and the diffusion
        # operator it is bounding uses centred face gradients regardless.
        #
        # Raw `pla.ne`, not a smoothed copy: the bound has to apply to the field the
        # solve actually transports. The old `n_raw / |∇∥(n_SM)|` mixed two different
        # fields and ran ~5× tighter on the low-density flank of a front.
        ∇para_ne = calculate_para_grad_of_scalar_F(RP, pla.ne; upwind = false)
        @. flux_limited_diffusivity(D0, ∇para_ne, pla.ne, vm_e, RP.flags.limit_flux.factor)
    else
        D0
    end

    # Stage 4. Whichever ceiling is tighter.
    @. tp.Dpara = min(D_geom, D_flux)

    # Ions carry no gradient ceiling yet — they have only stage 3a. It is NOT the one
    # line it looks like: an ion ceiling has to bound the total non-convective species
    # flux including the pinch term, it makes otherwise-grouped species operators
    # differ, and `ion_transport_channels` receives an `IonSpecies` that carries no
    # density. The neutral gas is owed the same question — panel 2 of the 1-D spec is a
    # GAS problem and the gas channel has no gradient cap either. Both are queued
    # behind the ambipolar-gate work; see
    # `internal/docs/src/notes/TODO/electron-parallel-transport-gating.md`.

    # Calculate perpendicular diffusion using Bohm diffusivity
    Dperp_bohm = @. abs((1 / 16) * pla.Te_eV / RP.fields.Bϕ)
    @. tp.Dperp = tp.Dperp0 + Dperp_bohm

    extrapolate_field_to_boundary_nodes!(RP.G, tp.Dpara)
    extrapolate_field_to_boundary_nodes!(RP.G, tp.Dperp)

    # Apply damping function outside wall if enabled
    if RP.flags.Damp_Transp_outWall
        @. tp.Dpara *= RP.damping_func
        @. tp.Dperp *= RP.damping_func
        @. pla.ue_para *= RP.damping_func

        @. pla.mean_ExB_R *= RP.damping_func
        @. pla.mean_ExB_Z *= RP.damping_func

        @. pla.ui_para *= RP.damping_func
    end

    # Convert parallel velocities to R,Z components if needed
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

        # Add ExB drift for ions too if enabled
        if RP.flags.mean_ExB
            pla.uiR .+= pla.mean_ExB_R
            pla.uiZ .+= pla.mean_ExB_Z
        end

        if RP.flags.Global_JxB_Force
            pla.uiR .+= pla.uMHD_R
            pla.uiZ .+= pla.uMHD_Z
        end
    end

    # update diffusion tensor (DRR,DRZ,DZZ) & (CTRR,CTRZ,CTZZ)
    update_diffusion_tensor!(RP)

    update_transport_related_operators!(RP)

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
    update_transport_related_operators!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Update transport-related sparse matrix operators (𝐮∇, ∇𝐮, ∇𝐃∇) based on current transport coefficients and velocity fields.
"""
function update_transport_related_operators!(RP::RAPID{FT}) where {FT <: AbstractFloat}

    OP = RP.operators

    if !isempty(OP.𝐮∇.k2csc)
        update_𝐮∇_operator!(RP)
    end

    if !isempty(OP.∇𝐮.k2csc)
        update_∇𝐮_operator!(RP)
    end

    if !isempty(OP.∇𝐮_i.k2csc)
        update_∇𝐮_operator!(RP, RP.plasma.uiR, RP.plasma.uiZ; ∇𝐮 = OP.∇𝐮_i)
    end

    if !isempty(OP.∇𝐃∇.k2csc)
        update_∇𝐃∇_operator!(RP)
    end

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
