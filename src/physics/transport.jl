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
function update_transport_quantities!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    pla = RP.plasma

    # Calculate momentum transfer reaction rate coefficient and collision frequency
    if RP.flags.Atomic_Collision
        RRC_mom = get_electron_RRC(RP, RP.eRRCs, :Momentum)
        RRC_iz = get_electron_RRC(RP, RP.eRRCs, :Ionization)
        @. pla.ν_en_mom = pla.n_H2_gas * RRC_mom
        @. pla.ν_en_iz = pla.n_H2_gas * RRC_iz
    end

    # Calculate total collision frequency
    if RP.flags.Coulomb_Collision
        update_coulomb_collision_parameters!(RP)
    end

    ν_tot = @. pla.ν_en_mom + pla.ν_en_iz + pla.ν_ei_eff

    # Calculate parallel diffusion coefficient based on collision frequency
    # Thermal velocity
    vthe = @. sqrt(pla.Te_eV * RP.config.ee / RP.config.me)

    # Collision-based diffusion coefficient (D = vth²/(3ν))
    # NOTE: factor 1/3 is used to consider the diffusion with 3D isotropic collision
    Dpara_coll = @. vthe^2 / (ν_tot * FT(3.0))
    Dpara_coll[.!isfinite.(Dpara_coll)] .= zero(FT)

    # Flux-limiter scheme to prevent excessive diffusion
    if RP.flags.limit_flux.state
        ne_SM = smooth_data_2D(pla.ne; num_SM=2, weighting=RP.G.inVol2D)
        Lne_para = abs.(pla.ne ./ calculate_para_grad_of_scalar_F(RP,ne_SM)) # gradient-scale length
        Lne_para[isnan.(Lne_para)] .= zero(FT)
        Dmax_para = RP.flags.limit_flux.factor * vthe .* Lne_para

        Dpara_coll = min.(Dpara_coll, Dmax_para)
    end

    # Combine base and collision diffusion
    @. RP.transport.Dpara = RP.transport.Dpara0 + Dpara_coll

    # Calculate perpendicular diffusion using Bohm diffusivity
    Dperp_bohm = @. abs((1/16) * pla.Te_eV / RP.fields.Bϕ)
    @. RP.transport.Dperp = RP.transport.Dperp0 + Dperp_bohm

    extrapolate_field_to_boundary_nodes!(RP.G, RP.transport.Dpara)
    extrapolate_field_to_boundary_nodes!(RP.G, RP.transport.Dperp)

    # Apply damping function outside wall if enabled
    if RP.flags.Damp_Transp_outWall
        @. RP.transport.Dpara *= RP.damping_func
        @. RP.transport.Dperp *= RP.damping_func
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
function update_diffusion_tensor!(RP::RAPID{FT}) where {FT<:AbstractFloat}
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
        @. tp.DRR_turb = tp.Dpol_turb * ( fperp + (fpara - fperp) * F.bpol_R^2 )
        @. tp.DRZ_turb = (tp.Dpol_turb) * (fpara - fperp) * (F.bpol_R * F.bpol_Z)
        @. tp.DZZ_turb = tp.Dpol_turb * ( fperp + (fpara - fperp) * F.bpol_Z^2)

        # Add turbulent diffusion to base diffusion
        @. tp.DRR .+= tp.DRR_turb
        @. tp.DRZ .+= tp.DRZ_turb
        @. tp.DZZ .+= tp.DZZ_turb
    end

    dR, dZ = RP.G.dR, RP.G.dZ

    @. tp.CTRR = RP.G.Jacob*tp.DRR/(dR*dR);
    @. tp.CTRZ = RP.G.Jacob*tp.DRZ/(dR*dZ);
    @. tp.CTZZ = RP.G.Jacob*tp.DZZ/(dZ*dZ);

    return RP
end


"""
    update_transport_related_operators!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Update transport-related sparse matrix operators (𝐮∇, ∇𝐮, ∇𝐃∇) based on current transport coefficients and velocity fields.
"""
function update_transport_related_operators!(RP::RAPID{FT}) where {FT<:AbstractFloat}

    OP = RP.operators

    if !isempty(OP.𝐮∇.k2csc)
        update_𝐮∇_operator!(RP)
    end

    if !isempty(OP.∇𝐮.k2csc)
        update_∇𝐮_operator!(RP)
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
function calculate_particle_fluxes!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Initialize arrays for density gradients
    dndR = zeros(FT, RP.G.NR, RP.G.NZ)
    dndZ = zeros(FT, RP.G.NR, RP.G.NZ)

    # Calculate density gradients (using forward/central/backward differences)
    # R-direction
    dndR[:,1] .= (RP.plasma.ne[:,2] .- RP.plasma.ne[:,1])/RP.G.dR
    dndR[:,2:end-1] .= (RP.plasma.ne[:,3:end] .- RP.plasma.ne[:,1:end-2])/(2*RP.G.dR)
    dndR[:,end] .= (RP.plasma.ne[:,end] .- RP.plasma.ne[:,end-1])/RP.G.dR

    # Z-direction
    dndZ[1,:] .= (RP.plasma.ne[2,:] .- RP.plasma.ne[1,:])/RP.G.dZ
    dndZ[2:end-1,:] .= (RP.plasma.ne[3:end,:] .- RP.plasma.ne[1:end-2,:])/(2*RP.G.dZ)
    dndZ[end,:] .= (RP.plasma.ne[end,:] .- RP.plasma.ne[end-1,:])/RP.G.dZ

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
