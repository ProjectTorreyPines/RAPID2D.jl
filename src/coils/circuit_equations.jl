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
    csys.mutual_inductance = 2π * ψ_matrix

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
