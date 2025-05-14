"""
Transport module for RAPID2D.

Contains functions related to transport phenomena, including:
- Diffusion coefficients
- Convection terms
- Source and sink terms
"""

# Export public functions
export update_transport_quantities!,
       calculate_diffusion_coefficients!,
       calculate_particle_fluxes!,
       calculate_diffusion_term!,
       construct_diffusion_operator!

"""
    update_transport_quantities!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Update all transport-related quantities.
"""
function update_transport_quantities!(RP::RAPID{FT}) where {FT<:AbstractFloat}

    @warn "Not yet implemented: update_transport_quantities!"

    # Update particle fluxes
    calculate_particle_fluxes!(RP)

    return RP
end

"""
    calculate_diffusion_coefficients!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Calculate diffusion coefficients based on field configuration and turbulence models.
"""
function calculate_diffusion_coefficients!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Base diffusion coefficients
    RP.transport.Dpara .= RP.transport.Dpara0 * ones(FT, RP.G.NR, RP.G.NZ)
    RP.transport.Dperp .= RP.transport.Dperp0 * ones(FT, RP.G.NR, RP.G.NZ)

    # Add turbulent diffusion if enabled
    if RP.flags.turb_ExB_mixing
        # In a real implementation, turbulent diffusion would be calculated based on
        # field line connection length, ExB drifts, etc.
        RP.transport.Dturb_para .= zeros(FT, RP.G.NR, RP.G.NZ)
        RP.transport.Dturb_perp .= zeros(FT, RP.G.NR, RP.G.NZ)

        # Add turbulent diffusion to base diffusion
        RP.transport.Dpara .+= RP.transport.Dturb_para
        RP.transport.Dperp .+= RP.transport.Dturb_perp
    end

    # Calculate full diffusivity tensor components
    RP.transport.DRR .= RP.transport.Dperp .+
                         (RP.transport.Dpara .- RP.transport.Dperp) .*
                         (RP.fields.bR).^2

    RP.transport.DRZ .= (RP.transport.Dpara .- RP.transport.Dperp) .*
                         RP.fields.bR .* RP.fields.bZ

    RP.transport.DZZ .= RP.transport.Dperp .+
                         (RP.transport.Dpara .- RP.transport.Dperp) .*
                         (RP.fields.bZ).^2

    # Apply damping outside the wall if enabled
    if RP.flags.Damp_Transp_outWall
        RP.transport.DRR .*= RP.damping_func
        RP.transport.DRZ .*= RP.damping_func
        RP.transport.DZZ .*= RP.damping_func
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

"""
    calculate_diffusion_term!(RP::RAPID{FT}, density::AbstractMatrix{FT}=RP.plasma.ne) where {FT<:AbstractFloat}

Calculate the diffusion term for a given density field using the diffusion coefficients.

# Arguments
- `RP::RAPID{FT}`: The RAPID object containing simulation state
- `density::AbstractMatrix{FT}=RP.plasma.ne`: The density field to calculate diffusion for (defaults to electron density)

# Returns
- `RP`: The updated RAPID object with the calculated diffusion term stored in RP.operators.neRHS_diffu

# Notes
- The calculation is performed only for interior points (2:NR-1, 2:NZ-1)
- Boundary conditions must be handled separately
- The result is stored in RP.operators.neRHS_diffu
"""
function calculate_diffusion_term!(RP::RAPID{FT}, density::AbstractMatrix{FT}=RP.plasma.ne) where {FT<:AbstractFloat}
    # Alias necessary fields from the RP object
    G = RP.G
    inv_Jacob = G.inv_Jacob
    NR, NZ = G.NR, G.NZ

    # geometric factor for Coefficient Tensors (CTRR, CTRZ, CTZZ)
    geoFac = G.Jacob / (G.dR * G.dZ)

    CTRR = @. geoFac*RP.transport.DRR
    CTRZ = @. geoFac*RP.transport.DRZ
    CTZZ = @. geoFac*RP.transport.DZZ

    # Ensure the diffusion term array is properly initialized
    diffu_term = RP.operators.neRHS_diffu
    fill!(diffu_term, zero(FT))

    # Note: Following the Julia convention where first index is R and second is Z
    # We keep i as R-index and j as Z-index but switch the array indexing order
    @inbounds for j in 2:NZ-1
        for i in 2:NR-1
            # Using @fastmath for potential performance improvements
            @fastmath diffu_term[i,j] = (
                +0.5*(CTRR[i+1,j]+CTRR[i,j])*(density[i+1,j]-density[i,j])
                -0.5*(CTRR[i-1,j]+CTRR[i,j])*(density[i,j]-density[i-1,j])
                +0.125*(CTRZ[i+1,j]+CTRZ[i,j])*(density[i,j+1]+density[i+1,j+1]-density[i,j-1]-density[i+1,j-1])
                -0.125*(CTRZ[i-1,j]+CTRZ[i,j])*(density[i,j+1]+density[i-1,j+1]-density[i,j-1]-density[i-1,j-1])
                +0.125*(CTRZ[i,j+1]+CTRZ[i,j])*(density[i+1,j]+density[i+1,j+1]-density[i-1,j]-density[i-1,j+1])
                -0.125*(CTRZ[i,j-1]+CTRZ[i,j])*(density[i+1,j]+density[i+1,j-1]-density[i-1,j]-density[i-1,j-1])
                +0.5*(CTZZ[i,j+1]+CTZZ[i,j])*(density[i,j+1]-density[i,j])
                -0.5*(CTZZ[i,j-1]+CTZZ[i,j])*(density[i,j]-density[i,j-1])
            )
            diffu_term[i,j] = diffu_term[i,j]*inv_Jacob[i,j]
        end
    end

    return RP
end

"""
    construct_diffusion_operator!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Construct the sparse matrix representation of the diffusion operator for implicit time-stepping.

# Arguments
- `RP::RAPID{FT}`: The RAPID object containing simulation state

# Returns
- `SparseMatrixCSC{FT, Int}`: The sparse matrix representation of the diffusion operator
"""
function construct_diffusion_operator!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Alias necessary fields from the RP object
    G = RP.G
    inv_Jacob = G.inv_Jacob
    NR, NZ = G.NR, G.NZ
    nid = G.nodes.nid

    # geometric factor for Coefficient Tensors (CTRR, CTRZ, CTZZ)
    geoFac = G.Jacob / (G.dR * G.dZ)

    CTRR = @. geoFac*RP.transport.DRR
    CTRZ = @. geoFac*RP.transport.DRZ
    CTZZ = @. geoFac*RP.transport.DZZ

    # Pre-allocate arrays for sparse matrix construction
    num_internal_nodes = (NR-2)*(NZ-2)
    num_entries = num_internal_nodes * 9
    I = zeros(Int, num_entries)  # Row indices
    J = zeros(Int, num_entries)  # Column indices
    V = zeros(FT, num_entries)   # Values


    # Fill arrays for sparse matrix construction
    k = 1
    for j in 2:NZ-1
        for i in 2:NR-1
            # Linear index for current node (i,j)
            xg = nid[i,j]

            # Fill the row indices (all entries in this loop have the same row index)
            I[k:k+8] .= xg

            # Fill the column indices for the 8 neighboring nodes
            # Note: We need to adjust the linear indexing for Julia's convention
            J[k:k+7] = [
                nid[i+1,j],       # East
                nid[i-1,j],       # West
                nid[i,j+1],       # North
                nid[i,j-1],       # South
                nid[i+1,j+1],     # Northeast
                nid[i-1,j-1],     # Southwest
                nid[i-1,j+1],     # Northwest
                nid[i+1,j-1]      # Southeast
            ]

            # Fill the coefficient values
            # Note: We need to transpose all indices for Julia's convention
            V[k:k+7] = inv_Jacob[i,j]*[
                0.5*(CTRR[i+1,j]+CTRR[i,j]) + 0.125*(CTRZ[i,j+1]-CTRZ[i,j-1]),
                0.5*(CTRR[i-1,j]+CTRR[i,j]) - 0.125*(CTRZ[i,j+1]-CTRZ[i,j-1]),
                0.5*(CTZZ[i,j+1]+CTZZ[i,j]) + 0.125*(CTRZ[i+1,j]-CTRZ[i-1,j]),
                0.5*(CTZZ[i,j-1]+CTZZ[i,j]) - 0.125*(CTRZ[i+1,j]-CTRZ[i-1,j]),
                0.125*(2*CTRZ[i,j] + CTRZ[i+1,j]+CTRZ[i,j+1]),
                0.125*(2*CTRZ[i,j] + CTRZ[i-1,j]+CTRZ[i,j-1]),
                -0.125*(2*CTRZ[i,j] + CTRZ[i,j+1]+CTRZ[i-1,j]),
                -0.125*(2*CTRZ[i,j] + CTRZ[i,j-1]+CTRZ[i+1,j])
            ]

            # Fill the diagonal entry (center node)
            J[k+8] = xg
            V[k+8] = -sum(V[k:k+7])  # Ensure row sum is zero

            k += 9
        end
    end

    # Construct a sparse matrix of size (NR*NZ)×(NR*NZ) by prepending 1 and appending NR*NZ to the indices
    # and padding with zeros to ensure proper dimensions for the diffusion operator
    A_diffu = sparse([1; I; NR * NZ], [1; J; NR * NZ], [0; V; 0])

    return A_diffu
end