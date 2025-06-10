using LinearAlgebra

# =============================================================================
# Mutual Inductance Matrix and Circuit Matrix Calculations
# =============================================================================

"""
    calculate_mutual_inductance_matrix!(csys::CoilSystem{FT}) where FT<:AbstractFloat

Calculate the mutual inductance matrix for all coils in the system.

This function ports the MATLAB calculation:
```matlab
obj.LM_matrix = 2*pi*obj.Cal_psi_by_Green_function(Rdest,Zdest,Rsrc,Zsrc,1);
for idx=1:obj.N
    obj.LM_matrix(idx,idx) = obj.self_L(idx);
end
```

# Arguments
- `csys::CoilSystem{FT}`: The coil system to update

# Side effects
- Updates `csys.mutual_inductance` matrix in-place
"""
function calculate_mutual_inductance_matrix!(csys::CoilSystem{FT}) where FT<:AbstractFloat
    N = csys.n_total
    if N == 0
        return nothing
    end

    # Extract coil positions
    R_positions = [coil.position.r for coil in csys.coils]
    Z_positions = [coil.position.z for coil in csys.coils]

    # Calculate mutual inductance using Green's function
    # Each element [i,j] is the flux at coil i due to unit current in coil j
    ψ_matrix = calculate_ψ_by_green_function(R_positions, Z_positions,
                                             R_positions, Z_positions, ones(FT, N))

    # Convert to mutual inductance: LM = 2π * ψ
    csys.mutual_inductance = FT(2π) * ψ_matrix

    # Set diagonal elements to self-inductance values
    for i in 1:N
        csys.mutual_inductance[i, i] = csys.coils[i].self_inductance
    end

    return nothing
end

"""
    calculate_circuit_matrices!(csys::CoilSystem{FT}, dt::FT) where FT<:AbstractFloat

Calculate the circuit matrices A_circuit and inv_A_circuit for time-stepping.

This function ports the MATLAB calculation:
```matlab
obj.A_circuit = obj.LM_matrix + diag(input_dt*obj.res_R(:));
obj.inv_A_circuit = inv(obj.A_circuit);
```

The circuit equation is: (L + R*dt) * I_new = L * I_old + dt * (V_ext - other_terms)

# Arguments
- `csys::CoilSystem{FT}`: The coil system to update

# Side effects
- Updates `csys.A_circuit` and `csys.inv_A_circuit` matrices in-place

# Notes
- Mutual inductance matrix must be calculated first using `calculate_mutual_inductance_matrix!`
"""
function calculate_circuit_matrices!(csys::CoilSystem{FT}) where FT<:AbstractFloat
    N = csys.n_total
    if N == 0
        return nothing
    end

    # Start with the mutual inductance matrix
    csys.A_circuit = copy(csys.mutual_inductance)

    # Add resistance terms to diagonal: A = Inductance + θimp*Δt*Resistive
    for i in 1:N
        csys.A_circuit[i, i] += csys.θimp * csys.Δt * csys.coils[i].resistance
    end

    # Calculate inverse matrix for efficient solving
    csys.inv_A_circuit = inv(csys.A_circuit)

    return nothing
end

"""
    update_coil_system_matrices!(csys::CoilSystem{FT}) where FT<:AbstractFloat

Update all system matrices: mutual inductance, circuit matrices.

This is a convenience function that calls both:
1. `calculate_mutual_inductance_matrix!`
2. `calculate_circuit_matrices!`

# Arguments
- `csys::CoilSystem{FT}`: The coil system to update

# Notes
- Call this function whenever coils are added/removed or when time step changes
- For efficiency, only call `calculate_circuit_matrices!` if only dt changes
"""
function update_coil_system_matrices!(csys::CoilSystem{FT}) where FT<:AbstractFloat
    calculate_mutual_inductance_matrix!(csys)
    calculate_circuit_matrices!(csys)
    return nothing
end

"""
    get_mutual_inductance(csys::CoilSystem{FT}, i::Int, j::Int) where FT<:AbstractFloat

Get mutual inductance between coil i and coil j.

# Arguments
- `csys::CoilSystem{FT}`: The coil system
- `i::Int`: Index of first coil
- `j::Int`: Index of second coil

# Returns
- `FT`: Mutual inductance value [H]
"""
function get_mutual_inductance(csys::CoilSystem{FT}, i::Int, j::Int) where FT<:AbstractFloat
    @assert 1 <= i <= csys.n_total "Coil index i out of range"
    @assert 1 <= j <= csys.n_total "Coil index j out of range"
    return csys.mutual_inductance[i, j]
end

"""
    get_coil_coupling_matrix(csys::CoilSystem{FT}) where FT<:AbstractFloat

Return the full mutual inductance matrix as a copy.

# Returns
- `Matrix{FT}`: Copy of the mutual inductance matrix [H]
"""
function get_coil_coupling_matrix(csys::CoilSystem{FT}) where FT<:AbstractFloat
    return copy(csys.mutual_inductance)
end

# =============================================================================
# Current Distribution Functions
# =============================================================================

"""
    distribute_coil_currents_to_Jϕ!(
        Jϕ::Matrix{FT},
        csys::CoilSystem{FT},
        grid::GridGeometry{FT};
        coil_mask::Union{Nothing, Vector{Bool}} = nothing
    ) where FT

Distribute coil currents to toroidal current density on the 2D grid using bilinear interpolation.

This function distributes point coil currents to the 2D grid using bilinear interpolation,
which spreads each coil's current to the four surrounding grid nodes based on their distances.

# Arguments
- `Jϕ::Matrix{FT}`: Pre-allocated matrix for toroidal current density [A/m²] (modified in-place)
- `csys::CoilSystem{FT}`: Coil system containing coil positions
- `grid::GridGeometry{FT}`: Grid geometry containing grid spacing and coordinates
- `coil_mask::Union{Nothing, Vector{Bool}}`: Optional mask specifying which coils to include

# Notes
- Only processes coils that are inside the grid domain
- Current density units are [A/m²]

"""
function distribute_coil_currents_to_Jϕ!(
    Jϕ::Matrix{FT},
    csys::CoilSystem{FT},
    grid::GridGeometry{FT};
    coil_mask::Union{Nothing, Vector{Bool}} = nothing
    # currents::Vector{FT} = csys.coils[1:csys.n_total].current
) where FT<:AbstractFloat

    @assert size(Jϕ) == (grid.NR, grid.NZ) "Jϕ matrix size must match grid dimensions"
    # @assert length(currents) == csys.n_total "Number of currents must match number of coils"

    # Clear the output matrix
    fill!(Jϕ, zero(FT))

    # Determine which coils to process
    if coil_mask === nothing
        # Use all coils that are inside the domain
        coil_indices = csys.inside_domain_indices
    else
        @assert length(coil_mask) == csys.n_total "Coil mask length must match number of coils"
        # Use only masked coils that are also inside domain
        coil_indices = [i for i in csys.inside_domain_indices if coil_mask[i]]
    end

    # Early return if no coils to process
    if isempty(coil_indices)
        return Jϕ
    end

    # Pre-compute grid parameters for efficiency
    inv_dR = one(FT) / grid.dR
    inv_dZ = one(FT) / grid.dZ
    inv_dA = inv_dR * inv_dZ  # Inverse cell area

    R_min = grid.R1D[1]
    Z_min = grid.Z1D[1]

    # Process each coil
    for coil_idx in coil_indices
        coil = csys.coils[coil_idx]
        current = coil.current

        # Skip if current is zero
        if abs(current) < eps(FT)
            continue
        end

        # Find grid cell indices (1-based)
        # floor(...) + 1 converts from 0-based to 1-based indexing
        # Ensure indices are within bounds
        rid = clamp(floor(Int, (coil.position.r - R_min) * inv_dR) + 1, 1, grid.NR - 1)
        zid = clamp(floor(Int, (coil.position.z - Z_min) * inv_dZ) + 1, 1, grid.NZ - 1)

        # Calculate fractional positions within the cell
        mr = (coil.position.r - grid.R1D[rid]) * inv_dR
        mz = (coil.position.z - grid.Z1D[zid]) * inv_dZ

        # Distribute current to 4 corner nodes using bilinear interpolation
        # Bottom-left node [rid, zid]
        Jϕ[rid, zid] += (one(FT) - mr) * (one(FT) - mz) * current * inv_dA
        # Bottom-right node [rid+1, zid]
        Jϕ[rid+1, zid] += mr * (one(FT) - mz) * current * inv_dA
        # Top-left node [rid, zid+1]
        Jϕ[rid, zid+1] += (one(FT) - mr) * mz * current * inv_dA
        # Top-right node [rid+1, zid+1]
        Jϕ[rid+1, zid+1] += mr * mz * current * inv_dA
    end

    return Jϕ
end

"""
    distribute_coil_currents_to_Jϕ(
        csys::CoilSystem{FT},
        grid::GridGeometry{FT};
        coil_mask::Union{Nothing, Vector{Bool}} = nothing
    ) where FT

Distribute coil currents to toroidal current density (allocating version).

This is a convenience wrapper that allocates the output matrix and calls the in-place version.

# Arguments
- `csys::CoilSystem{FT}`: Coil system containing coil positions
- `grid::GridGeometry{FT}`: Grid geometry containing grid spacing and coordinates
- `coil_mask::Union{Nothing, Vector{Bool}}`: Optional mask specifying which coils to include

# Returns
- `Matrix{FT}`: Toroidal current density distribution [A/m²]
"""
function distribute_coil_currents_to_Jϕ(
    csys::CoilSystem{FT},
    grid::GridGeometry{FT};
    coil_mask::Union{Nothing, Vector{Bool}} = nothing
) where FT<:AbstractFloat

    Jϕ = zeros(FT, grid.NR, grid.NZ)
    distribute_coil_currents_to_Jϕ!(Jϕ, csys, grid; coil_mask)
    return Jϕ
end

"""
    determine_coils_inside_grid!(csys::CoilSystem{FT}, grid::GridGeometry{FT}) where FT

Update the `inside_domain_indices` field of the coil system based on grid boundaries.

This function determines which coils are positioned within the computational grid
and updates the coil system's `inside_domain_indices` field.

# Arguments
- `csys::CoilSystem{FT}`: Coil system to update (modified in-place)
- `grid::GridGeometry{FT}`: Grid geometry defining the grid boundaries

# Notes
- A coil is considered inside if: R_min ≤ R ≤ R_max and Z_min ≤ Z ≤ Z_max
- Coils exactly on the boundary are considered inside
- This function modifies the coil system in-place
"""
function determine_coils_inside_grid!(csys::CoilSystem{FT}, grid::GridGeometry{FT}) where FT<:AbstractFloat
    # Clear existing indices
    empty!(csys.inside_domain_indices)

    # Grid boundaries
    R_min = grid.R1D[1]
    R_max = grid.R1D[end]
    Z_min = grid.Z1D[1]
    Z_max = grid.Z1D[end]

    # Check each coil
    for (i, coil) in enumerate(csys.coils)
        if R_min ≤ coil.position.r ≤ R_max && Z_min ≤ coil.position.z ≤ Z_max
            push!(csys.inside_domain_indices, i)
        end
    end

    return csys
end
