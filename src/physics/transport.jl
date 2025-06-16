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

    # Calculate momentum transfer reaction rate coefficient and collision frequency
    if RP.flags.Atomic_Collision
        RRC_mom = get_electron_RRC(RP, RP.eRRCs, :Momentum)
        RRC_iz = get_electron_RRC(RP, RP.eRRCs, :Ionization)
        @. RP.plasma.ν_en_mom = RP.plasma.n_H2_gas * RRC_mom
        @. RP.plasma.ν_en_iz = RP.plasma.n_H2_gas * RRC_iz
    end

    # Calculate total collision frequency
    if RP.flags.Coulomb_Collision
        update_coulomb_collision_parameters!(RP)
    end

    ν_tot = @. RP.plasma.ν_en_mom + RP.plasma.ν_en_iz + RP.plasma.ν_ei_eff

    # Calculate parallel diffusion coefficient based on collision frequency
    # Thermal velocity
    vthe = @. sqrt(RP.plasma.Te_eV * RP.config.ee / RP.config.me)

    # Collision-based diffusion coefficient (D = vth²/(2ν))
    Dpara_coll_1 = @. 0.5 * vthe^2 / ν_tot

    # Field line mixing length-based diffusion coefficient
    Dpara_coll_2 = zeros(FT, size(RP.plasma.Te_eV))
    Dpara_coll_2 = @. 0.5 * vthe * RP.transport.L_mixing * RP.fields.Btot / RP.fields.Bpol
    Dpara_coll_2[RP.flf.closed_surface_nids] .= typemax(FT) # Effectively infinity for closed surfaces

    # Use the minimum of the two diffusion coefficients
    Dpara_coll = min.(Dpara_coll_1, Dpara_coll_2)
    Dpara_coll[.!isfinite.(Dpara_coll)] .= zero(FT)
    Dpara_coll[RP.G.nodes.on_out_wall_nids] .= zero(FT)

    # Combine base and collision diffusion
    @. RP.transport.Dpara = RP.transport.Dpara0 + Dpara_coll

    # Calculate perpendicular diffusion using Bohm diffusivity
    Dperp_bohm = @. abs((1/16) * RP.plasma.Te_eV / RP.fields.Bϕ)
    @. RP.transport.Dperp = RP.transport.Dperp0 + Dperp_bohm

    extrapolate_field_to_boundary_nodes!(RP.G, RP.transport.Dpara)
    extrapolate_field_to_boundary_nodes!(RP.G, RP.transport.Dperp)

    # Apply damping function outside wall if enabled
    if RP.flags.Damp_Transp_outWall
        @. RP.transport.Dpara *= RP.damping_func
        @. RP.transport.Dperp *= RP.damping_func
        @. RP.plasma.ue_para *= RP.damping_func

        @. RP.plasma.mean_ExB_R *= RP.damping_func
        @. RP.plasma.mean_ExB_Z *= RP.damping_func

        @. RP.plasma.ui_para *= RP.damping_func
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
            n_safe = copy(RP.plasma.ne)
            n_safe[n_safe .< n_min] .= n_min

            # Simple approximation of diamagnetic drift
            # In the real implementation, we'd use cal_grad_of_scalar_F
            RP.plasma.diaMag_R .= zeros(FT, size(RP.plasma.ne))
            RP.plasma.diaMag_Z .= zeros(FT, size(RP.plasma.ne))
        end

        # Update velocity components
        RP.plasma.ueR .= RP.plasma.ue_para .* RP.fields.bR
        RP.plasma.ueϕ .= RP.plasma.ue_para .* RP.fields.bϕ
        RP.plasma.ueZ .= RP.plasma.ue_para .* RP.fields.bZ

        # Add ExB and diamagnetic drifts if enabled
        if RP.flags.mean_ExB
            RP.plasma.ueR .+= RP.plasma.mean_ExB_R
            RP.plasma.ueZ .+= RP.plasma.mean_ExB_Z
        end

        if RP.flags.diaMag_drift
            RP.plasma.ueR .+= RP.plasma.diaMag_R
            RP.plasma.ueZ .+= RP.plasma.diaMag_Z
        end

        if RP.flags.Global_Force_Balance
            RP.plasma.ueR .+= RP.plasma.uMHD_R
            RP.plasma.ueZ .+= RP.plasma.uMHD_Z
        end

        # Same for ion velocities
        RP.plasma.uiR .= RP.plasma.ui_para .* RP.fields.bR
        RP.plasma.uiϕ .= RP.plasma.ui_para .* RP.fields.bϕ
        RP.plasma.uiZ .= RP.plasma.ui_para .* RP.fields.bZ

        # Add ExB drift for ions too if enabled
        if RP.flags.mean_ExB
            RP.plasma.uiR .+= RP.plasma.mean_ExB_R
            RP.plasma.uiZ .+= RP.plasma.mean_ExB_Z
        end

        if RP.flags.Global_Force_Balance
            RP.plasma.uiR .+= RP.plasma.uMHD_R
            RP.plasma.uiZ .+= RP.plasma.uMHD_Z
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
