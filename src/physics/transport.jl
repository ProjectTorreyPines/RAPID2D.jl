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
    # No `ee` here: the Maxwellian speed helpers convert eV→J themselves, with the
    # module-level `EE` that `constants.ee` defaults to. Unpacking it as well would
    # put two names for one constant in scope.
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

    # Collision-based diffusion coefficient: ½·(2T/m)/ν = T/(mν)
    tp.Dpara_e_coll = @. FT(0.5) * vp_e^2 / ν_sum_mom_iz_ei
    @. tp.Dpara_e_coll[isnan(tp.Dpara_e_coll)] = typemax(FT) # make NaN to Inf

    # Kept as state, not a local: the ion continuity equation rebuilds D∥ per
    # species from its own mass, so it needs the collision frequency rather than
    # the H₂⁺ diffusivity this line happens to produce.
    tp.νi_eff .= νi_eff

    tp.Dpara_i_coll = @. FT(0.5) * vp_i^2 / νi_eff
    @. tp.Dpara_i_coll[isnan(tp.Dpara_i_coll)] = typemax(FT) # make NaN to Inf

    # Ambipolar diffusion coefficient (Te+Ti)*(De*Di) /(Ti*De + Te*Di)
    tp.Dpara_amb = @. (pla.Te_eV + pla.Ti_eV) * tp.Dpara_e_coll * tp.Dpara_i_coll / (pla.Ti_eV * tp.Dpara_e_coll + pla.Te_eV * tp.Dpara_i_coll)
    @. tp.Dpara_amb[isnan(tp.Dpara_amb)] = typemax(FT) # make NaN to Inf

    # ── how D∥ᵉ is assembled: three stages, and the order is the physics ──────
    #
    #   1  1/D_ch = Σ_p 1/D_p     competing termination events   (PR #11)
    #   2  D∥ = D_ch + Dpara0     independent arrival paths
    #   3  cap(D∥, ¼·v̄·Lₙ)        a causality bound on the TOTAL
    #
    # Stage 3 must come last. The free-streaming ceiling is mechanism-agnostic — a
    # Maxwellian of density n cannot push more than ¼nv̄ across any surface, whatever
    # moves it — so it bounds the sum, not one limb of it. Capping before the addition
    # is what let `Dpara0` transport faster than free streaming.

    # Stage 1. Harmonic average of collision and ambipolar diffusion coefficients.
    tp.Dpara_e_eff = @. (tp.Dpara_e_coll * tp.Dpara_amb) / (tp.Dpara_e_coll + tp.Dpara_amb)
    @. tp.Dpara_e_eff[isnan(tp.Dpara_e_eff)] = typemax(FT) # make NaN to Inf

    # Stage 2. A base diffusivity is an independent arrival path, so it ADDS rather
    # than joining the harmonic sum — the same rule the ion channel states.
    @. tp.Dpara = tp.Dpara0 + tp.Dpara_e_eff

    # Stage 3. The free-streaming ceiling.
    if RP.flags.limit_flux.state
        # `vm_e`, not the `vp_e` above: the ceiling is a ONE-WAY flux across a
        # surface, ⟨v_n θ(v_n)⟩ = ¼v̄, and only the mean speed has that meaning. The
        # pair (½, vp) two dozen lines up is a different derivation and must not be
        # copied here — see the speed contract at the top of `transport_channels.jl`.
        vm_e = maxwellian_mean_speed.(pla.Te_eV, me)
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
        @. tp.Dpara = flux_limited_diffusivity(
            tp.Dpara, ∇para_ne, pla.ne, vm_e, RP.flags.limit_flux.factor
        )

        # Ions carry no gradient ceiling yet. It is NOT the one line it looks like:
        # an ion ceiling has to bound the total non-convective species flux including
        # the pinch term, it makes otherwise-grouped species operators differ, and
        # `ion_transport_channels` receives an `IonSpecies` that carries no density.
        # The neutral gas is owed the same question — panel 2 of the 1-D spec is a
        # GAS problem and the gas channel has no gradient cap either. Both are queued
        # behind the ambipolar-gate work; see
        # `internal/docs/src/notes/TODO/electron-parallel-transport-gating.md`.
    end

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
