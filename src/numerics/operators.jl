"""
    operators.jl

This file defines numerical operators used in the RAPID simulation framework,
primarily focusing on discretizations of diffusion, convection, and advection terms
for plasma transport equations.

Key functionalities include:
- Calculation of the Right-Hand Side (RHS) for explicit time-stepping schemes:
    - `calculate_ne_diffusion_explicit_RHS!`: Computes the diffusion term [ ∇𝐃∇n ].
    - `calculate_ne_convection_explicit_RHS!`: Computes the convection term [ -∇⋅(nv) ].
- Construction and management of sparse matrix operators for implicit time-stepping:
    - Diffusion operator [ ∇𝐃∇ ]:
        - `initialize_∇𝐃∇_operator!`: Sets up the sparse matrix.
        - `construct_∇𝐃∇_operator`: Builds the operator from scratch.
        - `allocate_∇𝐃∇_operator_pattern`: Defines the sparsity pattern.
        - `update_∇𝐃∇_operator!`: Updates matrix values.
    - Convection operator [ -∇⋅(nv) ]:
        - `initialize_Ane_convection_operator!`: Sets up the sparse matrix.
        - `construct_Ane_convection_operator`: Builds the operator from scratch.
        - `allocate_Ane_convection_operator_pattern!`: Defines the sparsity pattern.
        - `update_Ane_convection_operator!`: Updates matrix values.
    - Advection operator [ (u·∇)f ]:
        - `initialize_𝐮∇_operator!`: Sets up the sparse matrix.
        - `construct_𝐮∇_operator`: Builds the operator from scratch.
        - `allocate_𝐮∇_operator_pattern!`: Defines the sparsity pattern.
        - `update_𝐮∇_operator!`: Updates matrix values.

These operators are designed for 2D cylindrical coordinates (R, Z) and support
different numerical schemes, such as central differencing and upwind schemes,
for stability and accuracy.
"""


# Export public functions
export calculate_diffusion_coefficients!,
       calculate_ne_diffusion_explicit_RHS!,
       construct_∇𝐃∇_operator,
       calculate_ne_convection_explicit_RHS!,
       construct_Ane_convection_operator,
       construct_𝐮∇_operator


"""
    calculate_ne_diffusion_explicit_RHS!(RP::RAPID{FT}, density::AbstractMatrix{FT}=RP.plasma.ne) where {FT<:AbstractFloat}

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
function calculate_ne_diffusion_explicit_RHS!(RP::RAPID{FT}, density::AbstractMatrix{FT}=RP.plasma.ne) where {FT<:AbstractFloat}
    # Alias necessary fields from the RP object
    G = RP.G
    inv_Jacob = G.inv_Jacob
    NR, NZ = G.NR, G.NZ

    CTRR = RP.transport.CTRR
    CTRZ = RP.transport.CTRZ
    CTZZ = RP.transport.CTZZ

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
    initialize_∇𝐃∇_operator!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Initialize the sparse matrix representation of the diffusion operator [∇𝐃∇] with proper structure and values.

# Arguments
- `RP::RAPID{FT}`: The RAPID object containing simulation state

# Returns
- `RP`: The updated RAPID object with initialized diffusion operator

# Notes
- This function first creates the sparsity pattern and then updates the values
- Uses `allocate_∇𝐃∇_operator_pattern` to create the matrix structure
- Uses `update_∇𝐃∇_operator!` to populate the non-zero values
"""
function initialize_∇𝐃∇_operator!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # create a sparse matrix with the sparisty pattern
    allocate_∇𝐃∇_operator_pattern(RP)

    # update the diffusion operator's non-zero entries with the actual values
    update_∇𝐃∇_operator!(RP)

    return RP
end

function construct_∇𝐃∇_operator(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Alias necessary fields from the RP object
    G = RP.G
    NR, NZ = G.NR, G.NZ
    nid = G.nodes.nid
    inv_Jacob = G.inv_Jacob

    CTRR = RP.transport.CTRR
    CTRZ = RP.transport.CTRZ
    CTZZ = RP.transport.CTZZ

    # define constants with FT for type stability
    half = FT(0.5)
    eighth = FT(0.125)

    # Pre-allocate arrays for sparse matrix construction
    num_entries = (NR-2) * (NZ-2) * 9
    I = zeros(Int, num_entries)  # Row indices
    J = zeros(Int, num_entries)  # Column indices
    V = zeros(FT, num_entries)   # Values (all zeros initially)

    # Fill arrays for sparse matrix construction
    k = 1
    for j in 2:NZ-1
        for i in 2:NR-1
            factor = inv_Jacob[i,j]
            two_CTRZ = 2 * CTRZ[i,j]

            # Set row indices (all entries in this loop have the same row index)
            I[k:k+8] .= nid[i, j]

            # Northwest [i-1,j+1]
            J[k]   = nid[i+1, j]
            V[k]   = factor * (half*(CTRR[i+1,j]+CTRR[i,j]) + eighth*(CTRZ[i,j+1]-CTRZ[i,j-1]))

            # Northwest [i-1,j+1]
            J[k+1] = nid[i-1, j]
            V[k+1] = factor * (half*(CTRR[i-1,j]+CTRR[i,j]) - eighth*(CTRZ[i,j+1]-CTRZ[i,j-1]))

            # Northwest [i-1,j+1]
            J[k+2] = nid[i, j+1]
            V[k+2] = factor * (half*(CTZZ[i,j+1]+CTZZ[i,j]) + eighth*(CTRZ[i+1,j]-CTRZ[i-1,j]))

            # Northwest [i-1,j+1]
            J[k+3] = nid[i, j-1]
            V[k+3] = factor * (half*(CTZZ[i,j-1]+CTZZ[i,j]) - eighth*(CTRZ[i+1,j]-CTRZ[i-1,j]))

            # Northwest [i-1,j+1]
            J[k+4] = nid[i+1, j+1]
            V[k+4] = factor * (eighth*( two_CTRZ + CTRZ[i+1,j]+CTRZ[i,j+1]))

            # Northwest [i-1,j+1]
            J[k+5] = nid[i-1, j-1]
            V[k+5] = factor * (eighth*( two_CTRZ + CTRZ[i-1,j]+CTRZ[i,j-1]))

            # Northwest [i-1,j+1]
            J[k+6] = nid[i-1, j+1]
            V[k+6] = factor * (-eighth*( two_CTRZ + CTRZ[i,j+1]+CTRZ[i-1,j]))

            # Southeast [i+1,j-1]
            J[k+7] = nid[i+1, j-1]
            V[k+7] = factor * (-eighth*( two_CTRZ + CTRZ[i,j-1]+CTRZ[i+1,j]))

            # Center [i,j]
            J[k+8] = nid[i, j]
            V[k+8] = zero(FT)
            @inbounds for t in 0:7
                V[k+8] -= V[k+t]
            end

            k += 9
        end
    end

    # Construct a sparse matrix with the explicit size (NR*NZ)×(NR*NZ)
    return sparse(I, J, V, NR*NZ, NR*NZ)

end

"""
    allocate_∇𝐃∇_operator_pattern(RP::RAPID{FT}) where {FT<:AbstractFloat}

Create a sparse matrix with the sparsity pattern for the diffusion operator [∇𝐃∇] without computing coefficient values.

# Arguments
- `RP::RAPID{FT}`: The RAPID object containing simulation state

# Notes
- This function only creates the sparsity pattern (non-zero locations) without computing the actual coefficients
- The function is called by `initialize_∇𝐃∇_operator!` to set up the structure before filling in values
- Creates a 9-point stencil pattern for each interior grid point
"""
function allocate_∇𝐃∇_operator_pattern(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Alias necessary fields from the RP object
    G = RP.G
    NR, NZ = G.NR, G.NZ
    nid = G.nodes.nid

    # Pre-allocate arrays for sparse matrix construction
    num_entries = (NR-2) * (NZ-2) * 9
    I = zeros(Int, num_entries)  # Row indices
    J = zeros(Int, num_entries)  # Column indices
    V = zeros(FT, num_entries)   # Values (all zeros initially)

    # Fill arrays for sparse matrix construction
    k = 1
    for j in 2:NZ-1
        for i in 2:NR-1
            # Set row indices (all entries in this loop have the same row index)
            I[k:k+8] .= nid[i, j]

            # column indices for the 8 neighboring nodes
            J[k]   = nid[i+1, j]    # East
            J[k+1] = nid[i-1, j]    # West
            J[k+2] = nid[i, j+1]    # North
            J[k+3] = nid[i, j-1]    # South
            J[k+4] = nid[i+1, j+1]  # Northeast
            J[k+5] = nid[i-1, j-1]  # Southwest
            J[k+6] = nid[i-1, j+1]  # Northwest
            J[k+7] = nid[i+1, j-1]  # Southeast
            J[k+8] = nid[i, j]      # Center

            k += 9
        end
    end

    # to compute k2csc later
    @. V = 1:num_entries

    # Construct a sparse matrix with the explicit size (NR*NZ)×(NR*NZ)
    RP.operators.An_diffu = sparse(I, J, V, NR*NZ, NR*NZ)

    # Get a mapping from k to the non-zero indices
    V2 = RP.operators.An_diffu.nzval
    k2csc = zeros(Int, length(V2))
    for csc_idx in eachindex(V2)
        orig_k = round(Int, V2[csc_idx])
        k2csc[orig_k] = csc_idx
    end
    RP.operators.map_diffu_k2csc = k2csc

    # Reset the values to zero
    @. RP.operators.An_diffu.nzval = zero(FT)

    return RP
end

"""
    update_∇𝐃∇_operator!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Update the non-zero entries of the diffusion operator matrix based on the current state of the RAPID object.
# Arguments
- `RP::RAPID{FT}`: The RAPID object containing simulation state

# Returns
- `RP`: The updated RAPID object with the diffusion operator matrix updated

# Notes
- The function assumes that the diffusion operator matrix has already been initialized with the correct sparsity pattern.
- The function updates the non-zero entries of the matrix based on the current state of the RAPID object.
"""
function update_∇𝐃∇_operator!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    @assert !isempty(RP.operators.An_diffu.nzval) "Diffusion operator not initialized"

    # Alias necessary fields from the RP object
    inv_Jacob = RP.G.inv_Jacob
    NR, NZ = RP.G.NR, RP.G.NZ

    CTRR = RP.transport.CTRR
    CTRZ = RP.transport.CTRZ
    CTZZ = RP.transport.CTZZ

    # define constants with FT for type stability
    half = FT(0.5)
    eighth = FT(0.125)

    # Alias the existing sparse matrix for readability
    nzV = RP.operators.An_diffu.nzval

    k2csc = RP.operators.map_diffu_k2csc

    k = 1

    @inbounds for j in 2:NZ-1
        for i in 2:NR-1
            factor = inv_Jacob[i,j]

            nzV[k2csc[k]]   = factor * (half*(CTRR[i+1,j]+CTRR[i,j]) + eighth*(CTRZ[i,j+1]-CTRZ[i,j-1])) # East [i+1,j]
            nzV[k2csc[k+1]] = factor * (half*(CTRR[i-1,j]+CTRR[i,j]) - eighth*(CTRZ[i,j+1]-CTRZ[i,j-1])) # West [i-1,j]
            nzV[k2csc[k+2]] = factor * (half*(CTZZ[i,j+1]+CTZZ[i,j]) + eighth*(CTRZ[i+1,j]-CTRZ[i-1,j])) # North [i,j+1]
            nzV[k2csc[k+3]] = factor * (half*(CTZZ[i,j-1]+CTZZ[i,j]) - eighth*(CTRZ[i+1,j]-CTRZ[i-1,j])) # South [i,j-1]

            two_CTRZ = 2 * CTRZ[i,j]
            nzV[k2csc[k+4]] = factor * (eighth*( two_CTRZ + CTRZ[i+1,j]+CTRZ[i,j+1]))  # Northeast [i+1,j+1]
            nzV[k2csc[k+5]] = factor * (eighth*( two_CTRZ + CTRZ[i-1,j]+CTRZ[i,j-1]))  # Southwest [i-1,j-1]
            nzV[k2csc[k+6]] = factor * (-eighth*( two_CTRZ + CTRZ[i,j+1]+CTRZ[i-1,j])) # Northwest [i-1,j+1]
            nzV[k2csc[k+7]] = factor * (-eighth*( two_CTRZ + CTRZ[i,j-1]+CTRZ[i+1,j])) # Southeast [i+1,j-1]

            nzV[k2csc[k+8]] = zero(FT)
            @inbounds for t in 0:7
                nzV[k2csc[k+8]] -= nzV[k2csc[k+t]]
            end

            k += 9
        end
    end

    return RP
end


"""
    calculate_ne_convection_explicit_RHS!(
        RP::RAPID{FT},
        density::AbstractMatrix{FT}=RP.plasma.ne,
        uR::AbstractMatrix{FT}=RP.plasma.ueR,
        uZ::AbstractMatrix{FT}=RP.plasma.ueZ
        ;
        flag_upwind::Bool=RP.flags.upwind) where {FT<:AbstractFloat}

Calculate the convection term [-∇⋅(nv)] for a given density field using the velocity field.

# Arguments
- `RP::RAPID{FT}`: The RAPID object containing simulation state
- `density::AbstractMatrix{FT}=RP.plasma.ne`: The density field to calculate convection for (defaults to electron density)
- `uR::AbstractMatrix{FT}=RP.plasma.ueR`: The R-component of velocity field (defaults to electron fluid velocity)
- `uZ::AbstractMatrix{FT}=RP.plasma.ueZ`: The Z-component of velocity field (defaults to electron fluid velocity)
- `flag_upwind::Bool=RP.flags.upwind`: Flag to use upwind scheme (if false, uses central differencing)

# Returns
- `RP`: The updated RAPID object with the calculated convection term stored in RP.operators.neRHS_convec

# Notes
- The calculation is performed only for interior points (2:NR-1, 2:NZ-1)
- Boundary conditions must be handled separately
- The result is stored in RP.operators.neRHS_convec
- Uses a first-order upwind scheme for numerical stability when upwind=true
- Falls back to central differencing for zero velocity even when upwind=true
- Uses second-order central differencing when upwind=false
"""
function calculate_ne_convection_explicit_RHS!(
    RP::RAPID{FT},
    density::AbstractMatrix{FT}=RP.plasma.ne,
    uR::AbstractMatrix{FT}=RP.plasma.ueR,
    uZ::AbstractMatrix{FT}=RP.plasma.ueZ
    ;
    flag_upwind::Bool=RP.flags.upwind) where {FT<:AbstractFloat}

    # Alias necessary fields from the RP object
    G = RP.G
    Jacob = G.Jacob
    inv_Jacob = G.inv_Jacob
    NR, NZ = G.NR, G.NZ
    dR, dZ = G.dR, G.dZ

    # Precompute inverse values for faster calculation (multiplication instead of division)
    inv_dR = one(FT) / dR
    inv_dZ = one(FT) / dZ

    # Cache common constants with proper type once
    zero_val = zero(FT)
    eps_val = eps(FT)
    half = FT(0.5)  # Define half once with correct type

    # Ensure the convection term array is properly initialized
    if !isdefined(RP.operators, :neRHS_convec) || size(RP.operators.neRHS_convec) != (NR, NZ)
        RP.operators.neRHS_convec = zeros(FT, NR, NZ)
    end

    convec_term = RP.operators.neRHS_convec
    fill!(convec_term, zero_val)

    # Apply appropriate differencing scheme based on upwind flag and velocity
    # Move the upwind flag check outside the loop for better performance
    if flag_upwind
        # Upwind scheme with check for zero velocity
        @inbounds for j in 2:NZ-1
            for i in 2:NR-1
                negative_flux_R = zero_val
                negative_flux_Z = zero_val

                # R-direction convection flux with upwind scheme
                if abs(uR[i, j]) < eps_val
                    # Zero velocity, use central differencing
                    negative_flux_R = -Jacob[i+1, j] * uR[i+1, j] * half * inv_dR * density[i+1, j] +
                                        Jacob[i-1, j] * uR[i-1, j] * half * inv_dR * density[i-1, j]
                elseif uR[i, j] > zero_val
                    # Flow from left to right, use left (upwind) node
                    negative_flux_R = -Jacob[i, j] * uR[i, j] * inv_dR * density[i, j] + Jacob[i-1, j] * uR[i-1, j] * inv_dR * density[i-1, j]
                else
                    # Flow from right to left, use right (upwind) node
                    negative_flux_R = -Jacob[i+1, j] * uR[i+1, j] * inv_dR * density[i+1, j] +
                                      Jacob[i, j] * uR[i, j] * inv_dR * density[i, j]
                end

                # Z-direction convection flux with upwind scheme
                if abs(uZ[i, j]) < eps_val
                    # Zero velocity, use central differencing
                    negative_flux_Z = -Jacob[i, j+1] * uZ[i, j+1] * half * inv_dZ * density[i, j+1] +
                                    Jacob[i, j-1] * uZ[i, j-1] * half * inv_dZ * density[i, j-1]
                elseif uZ[i, j] > zero_val
                    # Flow from bottom to top, use bottom (upwind) node
                    negative_flux_Z = -Jacob[i, j] * uZ[i, j] * inv_dZ * density[i, j] +
                                      Jacob[i, j-1] * uZ[i, j-1] * inv_dZ * density[i, j-1]
                else
                    # Flow from top to bottom, use top (upwind) node
                    negative_flux_Z = -Jacob[i, j+1] * uZ[i, j+1] * inv_dZ * density[i, j+1] +
                                      Jacob[i, j] * uZ[i, j] * inv_dZ * density[i, j]
                end

                # Calculate the total convection contribution
                # Note: The negative sign is already incorporated in the flux calculations
                convec_term[i, j] = (negative_flux_R + negative_flux_Z) * inv_Jacob[i, j]
            end
        end
    else
        # Central differencing for both directions (simpler logic)
        @inbounds for j in 2:NZ-1
            for i in 2:NR-1
                # Calculate fluxes with central differencing
                negative_flux_R = -Jacob[i+1, j] * uR[i+1, j] * half * inv_dR * density[i+1, j] +
                                  Jacob[i-1, j] * uR[i-1, j] * half * inv_dR * density[i-1, j]

                negative_flux_Z = -Jacob[i, j+1] * uZ[i, j+1] * half * inv_dZ * density[i, j+1] +
                                  Jacob[i, j-1] * uZ[i, j-1] * half * inv_dZ * density[i, j-1]

                # Calculate the total convection contribution
                convec_term[i, j] = (negative_flux_R + negative_flux_Z) * inv_Jacob[i, j]
            end
        end
    end

    return RP
end

"""
    construct_Ane_convection_operator(
        RP::RAPID{FT},
        uR::AbstractMatrix{FT}=RP.plasma.ueR,
        uZ::AbstractMatrix{FT}=RP.plasma.ueZ
        ;
        flag_upwind::Bool=RP.flags.upwind) where {FT<:AbstractFloat}

Construct the sparse matrix representation of the convection operator [-∇⋅(nv)] for implicit time-stepping.

# Arguments
- `RP::RAPID{FT}`: The RAPID object containing simulation state
- `uR::AbstractMatrix{FT}=RP.plasma.ueR`: The R-component of velocity field (defaults to electron fluid velocity)
- `uZ::AbstractMatrix{FT}=RP.plasma.ueZ`: The Z-component of velocity field (defaults to electron fluid velocity)
- `flag_upwind::Bool=RP.flags.upwind`: Flag to use upwind scheme (if false, uses central differencing)

# Returns
- `SparseMatrixCSC{FT, Int}`: The sparse matrix representation of the convection operator
"""
function construct_Ane_convection_operator(
    RP::RAPID{FT},
    uR::AbstractMatrix{FT}=RP.plasma.ueR,
    uZ::AbstractMatrix{FT}=RP.plasma.ueZ
    ;
    flag_upwind::Bool=RP.flags.upwind) where {FT<:AbstractFloat}


    # Alias necessary fields from the RP object
    G = RP.G
    Jacob = G.Jacob        # Jacobian matrix
    inv_Jacob = G.inv_Jacob
    NR, NZ = G.NR, G.NZ
    nid = G.nodes.nid

    # Precompute inverse values for faster calculation (multiplication instead of division)
    inv_dR = one(FT) / G.dR
    inv_dZ = one(FT) / G.dZ

    # Cache common constants with proper type once
    zero_val = zero(FT)
    eps_val = eps(FT)
    half = FT(0.5)  # Define half once with correct type

    # Pre-allocate arrays for sparse matrix construction
    num_internal_nodes = (NR-2)*(NZ-2)
    # Each node connects to at most 4 points in convection (center + neighbors)
    num_entries = num_internal_nodes * 4
    I = zeros(Int, num_entries)  # Row indices
    J = zeros(Int, num_entries)  # Column indices
    V = zeros(FT, num_entries)   # Values

    # Fill arrays for sparse matrix construction with different logic based on upwind flag
    k = 1

    if flag_upwind
        # Upwind scheme with special handling for zero velocity
        for j in 2:NZ-1
            for i in 2:NR-1
                # Linear index for current node (i,j)
                row_idx = nid[i,j]
                I[k:k+3] .= row_idx
                inv_Jacob_ij = inv_Jacob[i,j]

                # R-direction
                if abs(uR[i,j]) < eps_val
                    # Zero velocity: use central differencing
                    J[k] = nid[i+1,j]
                    V[k] = -Jacob[i+1,j]*uR[i+1,j]*half*inv_dR*inv_Jacob_ij

                    J[k+1] = nid[i-1,j]
                    V[k+1] = Jacob[i-1,j]*uR[i-1,j]*half*inv_dR*inv_Jacob_ij
                elseif uR[i,j] > zero_val
                    # Flow from left to right
                    J[k] = nid[i,j]
                    V[k] = -Jacob[i,j]*uR[i,j]*inv_dR*inv_Jacob_ij

                    J[k+1] = nid[i-1,j]
                    V[k+1] = Jacob[i-1,j]*uR[i-1,j]*inv_dR*inv_Jacob_ij
                else
                    # Flow from right to left
                    J[k] = nid[i+1,j]
                    V[k] = -Jacob[i+1,j]*uR[i+1,j]*inv_dR*inv_Jacob_ij

                    J[k+1] = nid[i,j]
                    V[k+1] = Jacob[i,j]*uR[i,j]*inv_dR*inv_Jacob_ij
                end

                # Z-direction
                if abs(uZ[i,j]) < eps_val
                    # Zero velocity: use central differencing
                    J[k+2] = nid[i,j+1]
                    V[k+2] = -Jacob[i,j+1]*uZ[i,j+1]*half*inv_dZ*inv_Jacob_ij

                    J[k+3] = nid[i,j-1]
                    V[k+3] = Jacob[i,j-1]*uZ[i,j-1]*half*inv_dZ*inv_Jacob_ij
                elseif uZ[i,j] > zero_val
                    # Flow from bottom to top
                    J[k+2] = nid[i,j]
                    V[k+2] = -Jacob[i,j]*uZ[i,j]*inv_dZ*inv_Jacob_ij

                    J[k+3] = nid[i,j-1]
                    V[k+3] = Jacob[i,j-1]*uZ[i,j-1]*inv_dZ*inv_Jacob_ij
                else
                    # Flow from top to bottom
                    J[k+2] = nid[i,j+1]
                    V[k+2] = -Jacob[i,j+1]*uZ[i,j+1]*inv_dZ*inv_Jacob_ij

                    J[k+3] = nid[i,j]
                    V[k+3] = Jacob[i,j]*uZ[i,j]*inv_dZ*inv_Jacob_ij
                end

                k += 4
            end
        end
    else
        # Central differencing for both directions (simpler logic)
        for j in 2:NZ-1
            for i in 2:NR-1
                @. @views I[k:k+3] = nid[i,j]
                inv_Jacob_ij = inv_Jacob[i,j]

                # Note the sign of [-∇·(nv)] operator
                J[k]   = nid[i+1,j] # East
                J[k+1] = nid[i-1,j] # West
                J[k+2] = nid[i,j+1] # North
                J[k+3] = nid[i,j-1] # South

                V[k]   = -Jacob[i+1,j]*uR[i+1,j]*half*inv_dR*inv_Jacob_ij
                V[k+1] = +Jacob[i-1,j]*uR[i-1,j]*half*inv_dR*inv_Jacob_ij
                V[k+2] = -Jacob[i,j+1]*uZ[i,j+1]*half*inv_dZ*inv_Jacob_ij
                V[k+3] = +Jacob[i,j-1]*uZ[i,j-1]*half*inv_dZ*inv_Jacob_ij

                k += 4
            end
        end
    end

    # Construct a sparse matrix with the explicit size (NR*NZ)×(NR*NZ)
    return sparse(I, J, V, NR*NZ, NR*NZ)
end

"""
    initialize_Ane_convection_operator!(RP::RAPID{FT}; flag_upwind::Bool=RP.flags.upwind) where {FT<:AbstractFloat}

Initialize the sparse matrix representation of the convection operator [-∇⋅(nv)] with proper structure and values.

# Arguments
- `RP::RAPID{FT}`: The RAPID object containing simulation state
- `flag_upwind::Bool=RP.flags.upwind`: Flag to use upwind scheme (if false, uses central differencing)

# Returns
- `RP`: The updated RAPID object with initialized convection operator

# Notes
- This function first creates the sparsity pattern and then updates the values
- Uses `allocate_Ane_convection_operator_pattern!` to create the matrix structure
- Uses `update_Ane_convection_operator` to populate the non-zero values
"""
function initialize_Ane_convection_operator!(RP::RAPID{FT}; flag_upwind::Bool=RP.flags.upwind) where {FT<:AbstractFloat}
    # create a sparse matrix with the sparisty pattern
    allocate_Ane_convection_operator_pattern!(RP)

    # update the convection operator's non-zero entries with the actual values
    update_Ane_convection_operator!(RP; flag_upwind=flag_upwind)

    return RP
end

"""
    allocate_Ane_convection_operator_pattern!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Create a sparse matrix with the sparsity pattern for the convection operator [-∇⋅(nv)]
without computing coefficient values.

# Arguments
- `RP::RAPID{FT}`: The RAPID object containing simulation state

# Returns
- The updated RAPID object with allocated sparsity pattern

# Notes
- This function only creates the sparsity pattern (non-zero locations) without computing the actual coefficients
- The function is called by `initialize_Ane_convection_operator!` to set up the structure before filling in values
- Supports both upwind and central differencing schemes
"""
function allocate_Ane_convection_operator_pattern!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Alias necessary fields
    G = RP.G
    NR, NZ = G.NR, G.NZ
    nid = G.nodes.nid

    # Each interior node has 5 connections (center + E,W,N,S) for upwind scheme
    num_entries = (NR-2)*(NZ-2) * 5

    I = zeros(Int, num_entries)  # Row indices
    J = zeros(Int, num_entries)  # Column indices
    V = zeros(FT, num_entries)   # Values (initially zero)

    k = 1
    for j in 2:NZ-1
        for i in 2:NR-1
            # Set row indices (all entries in this loop have the same row index)
            I[k:k+4] .= nid[i, j]

            # Column indices for the 5 nodes (center + neighbors)
            J[k]   = nid[i,j]     # Center
            J[k+1] = nid[i+1,j]    # East
            J[k+2] = nid[i-1,j]    # West
            J[k+3] = nid[i,j+1]    # North
            J[k+4] = nid[i,j-1]    # South

            k += 5
        end
    end

    # to compute k2csc later
    @. V = 1:num_entries

    # Create sparse matrix
    RP.operators.An_convec = sparse(I, J, V, NR*NZ, NR*NZ)

    # Get a mapping from k to the non-zero indices
    V2 = RP.operators.An_convec.nzval
    k2csc = zeros(Int, length(V2))
    for csc_idx in eachindex(V2)
        orig_k = round(Int, V2[csc_idx])
        k2csc[orig_k] = csc_idx
    end
    RP.operators.map_convec_k2csc = k2csc

    # Reset the values to zero
    @. RP.operators.An_convec.nzval = zero(FT)

    return RP
end


"""
    update_Ane_convection_operator!(RP::RAPID{FT},
                                   uR::AbstractMatrix{FT}=RP.plasma.ueR,
                                   uZ::AbstractMatrix{FT}=RP.plasma.ueZ;
                                   flag_upwind::Bool=RP.flags.upwind) where {FT<:AbstractFloat}

Update the non-zero entries of the convection operator matrix [-∇⋅(nv)] based on the current state of the RAPID object.
# Arguments
- `RP::RAPID{FT}`: The RAPID object containing simulation state

# Returns
- `RP`: The updated RAPID object with the convection operator matrix updated

# Notes
- The function assumes that the convection operator matrix has already been initialized with the correct sparsity pattern.
- The function updates the non-zero entries of the matrix based on the current state of the RAPID object.
"""
function update_Ane_convection_operator!(RP::RAPID{FT},
                                   uR::AbstractMatrix{FT}=RP.plasma.ueR,
                                   uZ::AbstractMatrix{FT}=RP.plasma.ueZ;
                                   flag_upwind::Bool=RP.flags.upwind) where {FT<:AbstractFloat}

    @assert !isempty(RP.operators.An_convec.nzval) "Convection operator not initialized"

    # Alias necessary fields
    G = RP.G
    Jacob = G.Jacob
    inv_Jacob = G.inv_Jacob
    NR, NZ = G.NR, G.NZ

    # Precompute for efficiency
    inv_dR = one(FT) / G.dR
    inv_dZ = one(FT) / G.dZ

    # Constants for type stability
    zero_FT = zero(FT)
    eps_FT = eps(FT)
    half = FT(0.5)

    # Access the sparse matrix values directly
    nzval = RP.operators.An_convec.nzval

    k2csc = RP.operators.map_convec_k2csc

    # Reset values to zero before updating
    fill!(nzval, zero(FT))

    # Follow the established pattern and update values
    k = 1
    if flag_upwind
        @inbounds for j in 2:NZ-1
            for i in 2:NR-1
                # Jacobian factor at current position
                ij_factor = inv_Jacob[i,j]

                # Pattern indices: center(k), east(k+1), west(k+2), north(k+3), south(k+4)

                # R-direction velocity-dependent coefficients
                if abs(uR[i,j]) < eps_FT
                    # Zero velocity: use central differencing
                    nzval[k2csc[k+1]] -= Jacob[i+1,j]*uR[i+1,j]*half*inv_dR*ij_factor  # East
                    nzval[k2csc[k+2]] += Jacob[i-1,j]*uR[i-1,j]*half*inv_dR*ij_factor  # West
                elseif uR[i,j] > zero_FT
                    # Positive velocity: flow from west
                    nzval[k2csc[k]] -= Jacob[i,j]*uR[i,j]*inv_dR*ij_factor          # Center (divergence)
                    nzval[k2csc[k+2]] += Jacob[i-1,j]*uR[i-1,j]*inv_dR*ij_factor   # West (inflow)
                else
                    # Negative velocity: flow from east
                    nzval[k2csc[k+1]] -= Jacob[i+1,j]*uR[i+1,j]*inv_dR*ij_factor   # East (inflow)
                    nzval[k2csc[k]] += Jacob[i,j]*uR[i,j]*inv_dR*ij_factor         # Center (divergence)
                end

                # Z-direction velocity-dependent coefficients
                if abs(uZ[i,j]) < eps_FT
                    # Zero velocity: use central differencing
                    nzval[k2csc[k+3]] -= Jacob[i,j+1]*uZ[i,j+1]*half*inv_dZ*ij_factor  # North
                    nzval[k2csc[k+4]] += Jacob[i,j-1]*uZ[i,j-1]*half*inv_dZ*ij_factor  # South
                elseif uZ[i,j] > zero_FT
                    # Positive velocity: flow from south
                    nzval[k2csc[k]] -= Jacob[i,j]*uZ[i,j]*inv_dZ*ij_factor          # Center (divergence)
                    nzval[k2csc[k+4]] += Jacob[i,j-1]*uZ[i,j-1]*inv_dZ*ij_factor   # South (inflow)
                else
                    # Negative velocity: flow from north
                    nzval[k2csc[k+3]] -= Jacob[i,j+1]*uZ[i,j+1]*inv_dZ*ij_factor   # North (inflow)
                    nzval[k2csc[k]] += Jacob[i,j]*uZ[i,j]*inv_dZ*ij_factor         # Center (divergence)
                end

                k += 5
            end
        end
    else
        @inbounds for j in 2:NZ-1
            for i in 2:NR-1
                # Jacobian factor at current position
                ij_factor = inv_Jacob[i,j]

                # Always use central differencing
                nzval[k2csc[k+1]] -= Jacob[i+1,j]*uR[i+1,j]*half*inv_dR*ij_factor  # East
                nzval[k2csc[k+2]] += Jacob[i-1,j]*uR[i-1,j]*half*inv_dR*ij_factor  # West
                nzval[k2csc[k+3]] -= Jacob[i,j+1]*uZ[i,j+1]*half*inv_dZ*ij_factor  # North
                nzval[k2csc[k+4]] += Jacob[i,j-1]*uZ[i,j-1]*half*inv_dZ*ij_factor  # South

                k += 5
            end
        end
    end

    return RP
end

"""
    construct_𝐮∇_operator(
        RP::RAPID{FT},
        uR::AbstractMatrix{FT}=RP.plasma.ueR,
        uZ::AbstractMatrix{FT}=RP.plasma.ueZ
        ;
        flag_upwind::Bool=RP.flags.upwind) where {FT<:AbstractFloat}

Construct a sparse matrix operator for the advection term (u·∇)f.
This operator calculates the directional derivative of a scalar field f along the velocity field (u).

# Arguments
- `RP::RAPID{FT}`: The RAPID object containing simulation state
- `uR::AbstractMatrix{FT}=RP.plasma.ueR`: The R-component of velocity field (defaults to electron fluid velocity)
- `uZ::AbstractMatrix{FT}=RP.plasma.ueZ`: The Z-component of velocity field (defaults to electron fluid velocity)
- `flag_upwind::Bool=RP.flags.upwind`: Flag to use upwind scheme (if false, uses central differencing)

# Returns
- `SparseMatrixCSC{FT, Int}`: The sparse matrix representation of the advection operator (u·∇)f

# Notes
- This differs from the convection operator which calculates -∇·(nv)
- The advection operator represents the directional derivative (u·∇)f = uR*(df/dR) + uZ*(df/dZ)
- When flag_upwind=true, uses flow direction to choose appropriate differencing scheme
- When flag_upwind=false, uses standard central differencing for all points
"""
function construct_𝐮∇_operator(
    RP::RAPID{FT},
    uR::AbstractMatrix{FT}=RP.plasma.ueR,
    uZ::AbstractMatrix{FT}=RP.plasma.ueZ
    ;
    flag_upwind::Bool=RP.flags.upwind) where {FT<:AbstractFloat}

    # Alias necessary fields from the RP object
    G = RP.G
    NR, NZ = G.NR, G.NZ
    nid = G.nodes.nid

    # Precompute inverse values for faster calculation
    inv_dR = one(FT) / G.dR
    inv_dZ = one(FT) / G.dZ

    # Cache common constants for type stability
    zero_FT = zero(FT)
    eps_val = eps(FT)
    half = FT(0.5)

    # Pre-allocate arrays for sparse matrix construction
    # Each interior node connects to at most 5 points (itself + 4 neighbors)
    num_internal_nodes = (NR-2) * (NZ-2)
    num_entries = num_internal_nodes * 4
    I = zeros(Int, num_entries)  # Row indices
    J = zeros(Int, num_entries)  # Column indices
    V = zeros(FT, num_entries)   # Values

    k = 1
    if flag_upwind
        # Upwind scheme based on velocity direction
        @inbounds for j in 2:NZ-1
            for i in 2:NR-1
                # All entries for this node have same row index
                I[k:k+3] .= nid[i,j]

                # R-direction contribution (uR*df/dR)
                if abs(uR[i,j]) < eps_val
                    # Zero velocity: central difference
                    J[k] = nid[i+1,j]
                    J[k+1] = nid[i-1,j]
                    V[k] = uR[i,j] * half * inv_dR
                    V[k+1] = -uR[i,j] * half * inv_dR
                elseif uR[i,j] > zero_FT
                    # Positive flow: backward difference (upwind)
                    J[k] = nid[i,j]
                    J[k+1] = nid[i-1,j]
                    V[k] = uR[i,j] * inv_dR
                    V[k+1] = -uR[i,j] * inv_dR
                else
                    # Negative flow: forward difference (upwind)
                    J[k] = nid[i+1,j]
                    J[k+1] = nid[i,j]
                    V[k] = uR[i,j] * inv_dR
                    V[k+1] = -uR[i,j] * inv_dR
                end

                # Z-direction contribution (uZ*df/dZ)
                if abs(uZ[i,j]) < eps_val
                    # Zero velocity: central difference
                    J[k+2] = nid[i,j+1]
                    J[k+3] = nid[i,j-1]
                    V[k+2] = uZ[i,j] * half * inv_dZ
                    V[k+3] = -uZ[i,j] * half * inv_dZ
                elseif uZ[i,j] > zero_FT
                    # Positive flow: backward difference (upwind)
                    J[k+2] = nid[i,j]
                    J[k+3] = nid[i,j-1]
                    V[k+2] = uZ[i,j] * inv_dZ
                    V[k+3] = -uZ[i,j] * inv_dZ
                else
                    # Negative flow: forward difference (upwind)
                    J[k+2] = nid[i,j+1]
                    J[k+3] = nid[i,j]
                    V[k+2] = uZ[i,j] * inv_dZ
                    V[k+3] = -uZ[i,j] * inv_dZ
                end

                k += 4
            end
        end
    else
        # Central differencing for all points
        @inbounds for j in 2:NZ-1
            for i in 2:NR-1
                # All entries for this node have same row index
                I[k:k+3] .= nid[i,j]


                # R-direction central difference
                J[k] = nid[i+1,j]  # East
                J[k+1] = nid[i-1,j]  # West
                V[k] = uR[i,j] * half * inv_dR
                V[k+1] = -uR[i,j] * half * inv_dR

                # Z-direction central difference
                J[k+2] = nid[i,j+1]  # North
                J[k+3] = nid[i,j-1]  # South
                V[k+2] = uZ[i,j] * half * inv_dZ
                V[k+3] = -uZ[i,j] * half * inv_dZ

                k += 4
            end
        end
    end

    # Resize arrays to actual number of entries used
    resize!(I, k-1)
    resize!(J, k-1)
    resize!(V, k-1)

    # Add padding for the first and last nodes like in MATLAB
    # This ensures the matrix has the correct dimensions (NR*NZ)×(NR*NZ)
    push!(I, 1)
    push!(J, 1)
    push!(V, zero_FT)
    push!(I, NR*NZ)
    push!(J, NR*NZ)
    push!(V, zero_FT)

    # Construct a sparse matrix with the explicit size (NR*NZ)×(NR*NZ)
    return sparse(I, J, V, NR*NZ, NR*NZ)
end

"""
    initialize_𝐮∇_operator!(RP::RAPID{FT};
                           flag_upwind::Bool=RP.flags.upwind) where {FT<:AbstractFloat}

Initialize the sparse matrix representation of the advection operator (u·∇) with appropriate
sparsity pattern and initial values.

# Arguments
- `RP::RAPID{FT}`: The RAPID object containing simulation state
- `flag_upwind::Bool=RP.flags.upwind`: Flag to use upwind scheme (if false, uses central differencing)

# Returns
- The updated RAPID object with initialized advection operator

# Notes
- This function first creates the sparsity pattern and then updates the values
- Uses `allocate_𝐮∇_operator_pattern!` to create the matrix structure
- Uses `update_𝐮∇_operator!` to populate the non-zero values
"""
function initialize_𝐮∇_operator!(RP::RAPID{FT};
                                flag_upwind::Bool=RP.flags.upwind) where {FT<:AbstractFloat}
    # Create the sparsity patter
    allocate_𝐮∇_operator_pattern!(RP)

    # Update the values based on current velocity field
    update_𝐮∇_operator!(RP; flag_upwind)

    return RP
end

"""
    allocate_𝐮∇_operator_pattern!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Create a sparse matrix with the sparsity pattern for the advection operator (u·∇)
without computing coefficient values.

# Arguments
- `RP::RAPID{FT}`: The RAPID object containing simulation state

# Returns
- The updated RAPID object with allocated sparsity pattern

# Notes
- This function only creates the sparsity pattern (non-zero locations) without computing the actual coefficients
- The created pattern supports both upwind and central differencing schemes
- Stores a mapping in RP.operators.map_𝐮∇_k2csc for efficient updates
"""
function allocate_𝐮∇_operator_pattern!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Alias necessary fields from the RP object
    G = RP.G
    NR, NZ = G.NR, G.NZ
    nid = G.nodes.nid

    # Pre-allocate arrays for sparse matrix construction
    # Each interior node has 4 connections (East,West,North,South) for both upwind and central differencing
    num_entries =  (NR-2) * (NZ-2) * 5
    I = zeros(Int, num_entries)  # Row indices
    J = zeros(Int, num_entries)  # Column indices
    V = zeros(FT, num_entries)   # Values (will be used for mapping)

    k = 1
    for j in 2:NZ-1
        for i in 2:NR-1
            # All entries for this node have same row index
            I[k:k+4] .= nid[i,j]

            # Column indices for the 5 nodes (center + neighbors)
            J[k]   = nid[i,j]    # Center
            J[k+1] = nid[i+1,j]  # East
            J[k+2] = nid[i-1,j]  # West
            J[k+3] = nid[i,j+1]  # North
            J[k+4] = nid[i,j-1]  # South

            k += 5
        end
    end

    # to compute k2csc later
    @. V = 1:num_entries

    # Construct sparse matrix with padding for correct dimensions
    RP.operators.A_𝐮∇ = sparse(I, J, V, NR*NZ, NR*NZ)

    # Create mapping from original indices to CSC format
    V2 = RP.operators.A_𝐮∇.nzval
    k2csc = zeros(Int, length(V2))
    for csc_idx in eachindex(V2)
        orig_k = round(Int, V2[csc_idx])
        k2csc[orig_k] = csc_idx
    end
    RP.operators.map_𝐮∇_k2csc = k2csc

    # Reset values to zero
    @. RP.operators.A_𝐮∇ = zero(FT)

    return RP
end

"""
    update_𝐮∇_operator!(RP::RAPID{FT},
                        uR::AbstractMatrix{FT}=RP.plasma.ueR,
                        uZ::AbstractMatrix{FT}=RP.plasma.ueZ;
                        flag_upwind::Bool=RP.flags.upwind) where {FT<:AbstractFloat}

Update the non-zero entries of the advection operator matrix (u·∇)f based on the current velocity field.
This function updates an existing sparse matrix without changing its structure.

# Arguments
- `RP::RAPID{FT}`: The RAPID object containing simulation state
- `uR::AbstractMatrix{FT}=RP.plasma.ueR`: The R-component of velocity field
- `uZ::AbstractMatrix{FT}=RP.plasma.ueZ`: The Z-component of velocity field
- `flag_upwind::Bool=RP.flags.upwind`: Flag to use upwind scheme

# Returns
- `RP`: The updated RAPID object with advection operator values updated

# Notes
- This function assumes the sparse matrix has already been allocated with the proper sparsity pattern
- For maximum performance, updates values directly without reconstructing the matrix
- Uses the mapping in RP.operators.map_𝐮∇_k2csc to locate entries in the CSC format
"""
function update_𝐮∇_operator!(RP::RAPID{FT},
                            uR::AbstractMatrix{FT}=RP.plasma.ueR,
                            uZ::AbstractMatrix{FT}=RP.plasma.ueZ;
                            flag_upwind::Bool=RP.flags.upwind) where {FT<:AbstractFloat}

    # Alias necessary fields from the RP object
    G = RP.G
    NR, NZ = G.NR, G.NZ

    # Precompute inverse values for faster calculation
    inv_dR = one(FT) / G.dR
    inv_dZ = one(FT) / G.dZ

    # Cache common constants for type stability
    zero_FT = zero(FT)
    eps_val = eps(FT)
    half = FT(0.5)

    # Get direct access to sparse matrix values
    nzval = RP.operators.A_𝐮∇.nzval
    k2csc = RP.operators.map_𝐮∇_k2csc

    # Reset values to zero
    fill!(nzval, zero_FT)

    # Update values based on current velocity and chosen scheme
    # Start index for the main loop entries
    k = 1

    # Number of entries for the center nodes (added after the main loop)
    num_internal_nodes = (NR-2) * (NZ-2)

    if flag_upwind
        # Upwind scheme with special handling for zero velocity
        @inbounds for j in 2:NZ-1
            for i in 2:NR-1
                # R-direction contribution
                if abs(uR[i,j]) < eps_val
                    # Zero velocity: central difference
                    nzval[k2csc[k+1]] += uR[i,j] * half * inv_dR   # East
                    nzval[k2csc[k+2]] -= uR[i,j] * half * inv_dR   # West
                elseif uR[i,j] > zero_FT
                    # Positive flow: backward difference
                    nzval[k2csc[k]] += uR[i,j] * inv_dR            # Center
                    nzval[k2csc[k+2]] -= uR[i,j] * inv_dR          # West
                else
                    # Negative flow: forward difference
                    nzval[k2csc[k+1]] += uR[i,j] * inv_dR          # East
                    nzval[k2csc[k]] -= uR[i,j] * inv_dR            # Center
                end

                # Z-direction contribution
                if abs(uZ[i,j]) < eps_val
                    # Zero velocity: central difference
                    nzval[k2csc[k+3]] += uZ[i,j] * half * inv_dZ   # North
                    nzval[k2csc[k+4]] -= uZ[i,j] * half * inv_dZ   # South
                elseif uZ[i,j] > zero_FT
                    # Positive flow: backward difference
                    nzval[k2csc[k]] += uZ[i,j] * inv_dZ           # Center
                    nzval[k2csc[k+4]] -= uZ[i,j] * inv_dZ         # South
                else
                    # Negative flow: forward difference
                    nzval[k2csc[k+3]] += uZ[i,j] * inv_dZ        # North
                    nzval[k2csc[k]] -= uZ[i,j] * inv_dZ          # Center
                end

                # Move to next node's entries
                k += 5
            end
        end
    else
        # Central differencing for all points (simpler logic)
        @inbounds for j in 2:NZ-1
            for i in 2:NR-1
                # R-direction central difference
                nzval[k2csc[k+1]] = uR[i,j] * half * inv_dR      # East
                nzval[k2csc[k+2]] = -uR[i,j] * half * inv_dR     # West

                # Z-direction central difference
                nzval[k2csc[k+3]] = uZ[i,j] * half * inv_dZ      # North
                nzval[k2csc[k+4]] = -uZ[i,j] * half * inv_dZ     # South

                k += 5
            end
        end
    end

    return RP
end

