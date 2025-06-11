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
    R_positions = [coil.location.r for coil in csys.coils]
    Z_positions = [coil.location.z for coil in csys.coils]

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

Calculate the circuit matrices A_LR_circuit and inv_A_LR_circuit for time-stepping.

This function ports the MATLAB calculation:
```matlab
obj.A_LR_circuit = obj.LM_matrix + diag(input_dt*obj.res_R(:));
obj.inv_A_LR_circuit = inv(obj.A_LR_circuit);
```

The circuit equation is: (L + R*dt) * I_new = L * I_old + dt * (V_ext - other_terms)

# Arguments
- `csys::CoilSystem{FT}`: The coil system to update

# Side effects
- Updates `csys.A_LR_circuit` and `csys.inv_A_LR_circuit` matrices in-place

# Notes
- Mutual inductance matrix must be calculated first using `calculate_mutual_inductance_matrix!`
"""
function calculate_circuit_matrices!(csys::CoilSystem{FT}) where FT<:AbstractFloat
    N = csys.n_total
    if N == 0
        return nothing
    end

    # Start with the mutual inductance matrix
    csys.A_LR_circuit = copy(csys.mutual_inductance)

    # Add resistance terms to diagonal: A = Inductance + θimp*Δt*Resistive
    for i in 1:N
        csys.A_LR_circuit[i, i] += csys.θimp * csys.Δt * csys.coils[i].resistance
    end

    # Calculate inverse matrix for efficient solving
    csys.inv_A_LR_circuit = inv(csys.A_LR_circuit)

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
        coil_mask::Union{Nothing, Vector{Bool}} = nothing,
        currents::Union{Nothing, Vector{FT}} = nothing
    ) where FT

Distribute coil currents to toroidal current density on the 2D grid using bilinear interpolation.

This function distributes point coil currents to the 2D grid using bilinear interpolation,
which spreads each coil's current to the four surrounding grid nodes based on their distances.

# Arguments
- `Jϕ::Matrix{FT}`: Pre-allocated matrix for toroidal current density [A/m²] (modified in-place)
- `csys::CoilSystem{FT}`: Coil system containing coil positions
- `grid::GridGeometry{FT}`: Grid geometry containing grid spacing and coordinates
- `coil_mask::Union{Nothing, Vector{Bool}}`: Optional mask specifying which coils to include
- `currents::Union{Nothing, Vector{FT}}`: Optional external currents. If nothing, uses csys.coils[i].current

# Notes
- Only processes coils that are inside the grid domain
- Current density units are [A/m²]

"""
function distribute_coil_currents_to_Jϕ!(
    Jϕ::Matrix{FT},
    csys::CoilSystem{FT},
    grid::GridGeometry{FT};
    coil_mask::Union{Nothing, Vector{Bool}} = nothing,
    currents::Union{Nothing, Vector{FT}} = nothing
) where FT<:AbstractFloat

    @assert size(Jϕ) == (grid.NR, grid.NZ) "Jϕ matrix size must match grid dimensions"
    if currents !== nothing
        @assert length(currents) == csys.n_total "Number of currents must match number of coils"
    end

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
        # Use external currents if provided, otherwise use coil's current
        current = currents === nothing ? coil.current : currents[coil_idx]

        # Skip if current is zero
        if abs(current) < eps(FT)
            continue
        end

        # Find grid cell indices (1-based)
        # floor(...) + 1 converts from 0-based to 1-based indexing
        # Ensure indices are within bounds
        rid = clamp(floor(Int, (coil.location.r - R_min) * inv_dR) + 1, 1, grid.NR - 1)
        zid = clamp(floor(Int, (coil.location.z - Z_min) * inv_dZ) + 1, 1, grid.NZ - 1)

        # Calculate fractional positions within the cell
        mr = (coil.location.r - grid.R1D[rid]) * inv_dR
        mz = (coil.location.z - grid.Z1D[zid]) * inv_dZ

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
        coil_mask::Union{Nothing, Vector{Bool}} = nothing,
        currents::Union{Nothing, Vector{FT}} = nothing
    ) where FT

Distribute coil currents to toroidal current density (allocating version).

This is a convenience wrapper that allocates the output matrix and calls the in-place version.

# Arguments
- `csys::CoilSystem{FT}`: Coil system containing coil positions
- `grid::GridGeometry{FT}`: Grid geometry containing grid spacing and coordinates
- `coil_mask::Union{Nothing, Vector{Bool}}`: Optional mask specifying which coils to include
- `currents::Union{Nothing, Vector{FT}}`: Optional external currents. If nothing, uses csys.coils[i].current

# Returns
- `Matrix{FT}`: Toroidal current density distribution [A/m²]
"""
function distribute_coil_currents_to_Jϕ(
    csys::CoilSystem{FT},
    grid::GridGeometry{FT};
    coil_mask::Union{Nothing, Vector{Bool}} = nothing,
    currents::Union{Nothing, Vector{FT}} = nothing
) where FT<:AbstractFloat

    Jϕ = zeros(FT, grid.NR, grid.NZ)
    distribute_coil_currents_to_Jϕ!(Jϕ, csys, grid; coil_mask, currents)
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
        if R_min ≤ coil.location.r ≤ R_max && Z_min ≤ coil.location.z ≤ Z_max
            push!(csys.inside_domain_indices, i)
        end
    end

    return csys
end

# =============================================================================
# Circuit Equation Time Evolution Solvers
# =============================================================================

"""
    solve_LR_circuit_step!(csys::CoilSystem{FT}, t::FT) where FT

Solve one time step of the circuit equation without plasma contribution.

This function implements the MATLAB circuit equation:
```matlab
circuit_rhs = obj.coils.LM_matrix*obj.coils.I + obj.dt*obj.coils.LV_ext;
new_coil_I_k = obj.coils.inv_A_LR_circuit*circuit_rhs;
```

# Arguments
- `csys::CoilSystem{FT}`: The coil system with current state
- `t::FT`: Current time for evaluating time-dependent voltages

# Side effects
- Updates the current field of all coils in the system

# Notes
- Assumes circuit matrices (A_LR_circuit, inv_A_LR_circuit) are already computed
- Does not include plasma contributions (simplified circuit equation)
"""
function solve_LR_circuit_step!(csys::CoilSystem{FT}, t::FT) where FT<:AbstractFloat
    if csys.n_total == 0
        return nothing
    end

    # Get current voltages at time t
    voltages = get_all_voltages_at_time(csys, t)

    # Get current state
    currents = get_all_currents(csys)
    resistances = get_all_resistances(csys)

    # Circuit equation: (L + R*dt) * I_new = L * I_old + dt * V_ext
    circuit_rhs = csys.mutual_inductance * currents .+ csys.Δt * ( voltages - (one(FT) - csys.θimp) * resistances .* currents)

    # Solve for new currents
    new_currents = csys.inv_A_LR_circuit * circuit_rhs

    # Update coil currents
    set_all_currents!(csys, new_currents)

    return nothing
end


"""
    calculate_circuit_magnetic_energy(csys::CoilSystem{FT}) where FT

Calculate the total magnetic energy stored in the circuit.

Energy = 0.5 * I^T * L * I

# Arguments
- `csys::CoilSystem{FT}`: The coil system

# Returns
- `FT`: Total magnetic energy [J]
"""
function calculate_circuit_magnetic_energy(csys::CoilSystem{FT}) where FT<:AbstractFloat
    if csys.n_total == 0
        return zero(FT)
    end

    currents = get_all_currents(csys)
    return FT(0.5) * dot(currents, csys.mutual_inductance * currents)
end

"""
    calculate_power_dissipation(csys::CoilSystem{FT}) where FT

Calculate the instantaneous power dissipation in all resistances.

Power = I^T * R * I

# Arguments
- `csys::CoilSystem{FT}`: The coil system

# Returns
- `FT`: Total power dissipation [W]
"""
function calculate_power_dissipation(csys::CoilSystem{FT}) where FT<:AbstractFloat
    if csys.n_total == 0
        return zero(FT)
    end

    currents = get_all_currents(csys)
    resistances = [coil.resistance for coil in csys.coils]

    return sum(currents .^ 2 .* resistances)
end
