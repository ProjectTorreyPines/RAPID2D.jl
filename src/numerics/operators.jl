"""
    operators.jl

This file defines numerical operators used in the RAPID simulation framework,
primarily focusing on discretizations of diffusion, convection, and advection terms
for plasma transport equations.

Key functionalities include:
- Basic differential operators for cylindrical coordinates:
    - `construct_∂R_operator`: Builds the radial derivative operator (∂/∂R).
    - `construct_∂Z_operator`: Builds the vertical derivative operator (∂/∂Z).
    - `construct_𝐽⁻¹∂R_𝐽_operator`: Constructs the divergence-preserving radial derivative.
    - `calculate_divergence`: Computes vector field divergence in cylindrical coordinates.
- Construction and management of sparse matrix operators for implicit time-stepping:
    Implemented operators:
    - Diffusion operator [ ∇𝐃∇ ]
    - Convection operator [ -∇⋅(nv) ]
    - Advection operator [ (𝐮·∇)f ]
    - Convective-flux divergence operator [ ∇⋅(𝐮 f) ]

    Each operator type follows a consistent pattern with these functions:
    - `construct_*`: Builds the operator from scratch and returns a sparse matrix.
    - `initialize_*!`: Sets up the sparse matrix in the RAPID object.
    - `allocate_*_pattern`: Defines the sparsity pattern of the matrix.
    - `update_*!`: Updates matrix values based on current state.

These operators are designed for 2D cylindrical coordinates (R, Z) and support
different numerical schemes, such as central differencing and upwind schemes,
for stability and accuracy.
"""

# Export public functions
export construct_∂R_operator, construct_∂Z_operator,
        calculate_divergence, construct_𝐽⁻¹∂R_𝐽_operator,
        construct_ΔGS_operator,
        update_diffusion_tensor!,
        compute_∇𝐃∇f_directly,
        construct_∇𝐃∇_operator,
        compute_𝐮∇f_directly,
        construct_𝐮∇_operator,
        compute_∇f𝐮_directly,
        construct_∇𝐮_operator,
        update_∇𝐮_operator!



"""
    compute_∇𝐃∇f_directly(RP::RAPID{FT}, f::AbstractMatrix{FT}) where {FT<:AbstractFloat}

Directly compute ∇⋅𝐃⋅∇f using explicit finite difference.

This function applies the anisotropic diffusion operator 𝐃 to the scalar field f using
the diffusion tensor components CTRR, CTRZ, and CTZZ stored in the transport object.
The computation uses second-order central differences with proper handling of
cross-derivative terms.

# Arguments
- `RP::RAPID{FT}`: The RAPID simulation object containing grid geometry and transport coefficients
- `f::AbstractMatrix{FT}`: The input scalar field to which the diffusion operator is applied

# Returns
- `∇𝐃∇f::AbstractMatrix{FT}`: The result of applying the diffusion operator to f

# Mathematical Description
The diffusion operator in cylindrical coordinates (R,Z) with Jacobian is:
```
∇⋅(𝐃∇f) = (1/J) * [∂/∂R(J*D_RR*∂f/∂R + J*D_RZ*∂f/∂Z) + ∂/∂Z(J*D_RZ*∂f/∂R + J*D_ZZ*∂f/∂Z)]
```

where:
- J is the Jacobian of the coordinate transformation
- D_RR, D_RZ, D_ZZ are the diffusion tensor components
- CTRR = J*D_RR/(ΔR)², CTRZ = J*D_RZ/(ΔR*ΔZ), CTZZ = J*D_ZZ/(ΔZ)²

# Notes
- Only interior points (2:NR-1, 2:NZ-1) are computed; boundary values remain unchanged
- Uses explicit finite difference stencils with proper averaging of coefficients
- Cross-derivative terms (CTRZ) use 4-point stencils for second-order accuracy
- Performance is enhanced with @fastmath macro for interior calculations
"""
function compute_∇𝐃∇f_directly(RP::RAPID{FT}, f::AbstractMatrix{FT}) where {FT<:AbstractFloat}
    @timeit RAPID_TIMER "compute_∇𝐃∇f_directly" begin
        # Alias necessary fields from the RP object
        G = RP.G
        inv_Jacob = G.inv_Jacob
        NR, NZ = G.NR, G.NZ

        CTRR = RP.transport.CTRR
        CTRZ = RP.transport.CTRZ
        CTZZ = RP.transport.CTZZ

        ∇𝐃∇f = zeros(FT, size(f))

        @inbounds for j in 2:NZ-1
            for i in 2:NR-1
                # Using @fastmath for potential performance improvements
                @fastmath ∇𝐃∇f[i,j] = inv_Jacob[i,j]*(
                    +0.5*(CTRR[i+1,j]+CTRR[i,j])*(f[i+1,j]-f[i,j])
                    -0.5*(CTRR[i-1,j]+CTRR[i,j])*(f[i,j]-f[i-1,j])
                    +0.125*(CTRZ[i+1,j]+CTRZ[i,j])*(f[i,j+1]+f[i+1,j+1]-f[i,j-1]-f[i+1,j-1])
                    -0.125*(CTRZ[i-1,j]+CTRZ[i,j])*(f[i,j+1]+f[i-1,j+1]-f[i,j-1]-f[i-1,j-1])
                    +0.125*(CTRZ[i,j+1]+CTRZ[i,j])*(f[i+1,j]+f[i+1,j+1]-f[i-1,j]-f[i-1,j+1])
                    -0.125*(CTRZ[i,j-1]+CTRZ[i,j])*(f[i+1,j]+f[i+1,j-1]-f[i-1,j]-f[i-1,j-1])
                    +0.5*(CTZZ[i,j+1]+CTZZ[i,j])*(f[i,j+1]-f[i,j])
                    -0.5*(CTZZ[i,j-1]+CTZZ[i,j])*(f[i,j]-f[i,j-1])
                )
            end
        end

        return ∇𝐃∇f
    end
end

"""
    construct_∂R_operator(G::GridGeometry{FT}) where {FT<:AbstractFloat}

Constructs a sparse matrix operator that computes the first-order partial derivative
with respect to the radial coordinate (∂/∂R) using a central difference scheme.

# Arguments
- `G::GridGeometry{FT}`: Grid geometry containing dimensions, node indices, and spacing information

# Returns
- `DiscretizedOperator`, which contains a sparse matrix of size (NR*NZ)×(NR*NZ) representing the first-order
  radial derivative operator with coefficients ±0.5/dR at interior points
"""
function construct_∂R_operator(G::GridGeometry{FT}) where {FT<:AbstractFloat}
    # Alias necessary fields from the RP object
    NR, NZ = G.NR, G.NZ
    nid = G.nodes.nid
    inv_dR = one(FT) / G.dR

    # define constants with FT for type stability
    half = FT(0.5)

    # Pre-allocate arrays for sparse matrix construction
    num_entries = (NR-2) * (NZ-2) * 2
    I = zeros(Int, num_entries)  # Row indices
    J = zeros(Int, num_entries)  # Column indices
    V = zeros(FT, num_entries)   # Values (all zeros initially)

    # Fill arrays for sparse matrix construction
    k = 1
    for j in 2:NZ-1
        for i in 2:NR-1
            # Set row indices
            I[k:k+1] .= nid[i, j]
            # East [i+1,j]
            J[k] = nid[i+1, j]
            V[k] = half * inv_dR
            # West [i-1,j]
            J[k+1] = nid[i-1, j]
            V[k+1] = -half * inv_dR
            k += 2
        end
    end

    return DiscretizedOperator((NR,NZ), I, J, V)
end

# Convinience dispatch
function construct_∂R_operator(RP::RAPID{FT}) where {FT<:AbstractFloat}
    return construct_∂R_operator(RP.G)
end

"""
    construct_𝐽⁻¹∂R_𝐽_operator(G::GridGeometry{FT}) where {FT<:AbstractFloat}

Construct a sparse matrix operator representing `(1/R)(∂/∂R)*(R f)`.

This function creates a discrete approximation of the radial derivative operator
using central differences, with appropriate Jacobian transformations for the
curvilinear coordinate system (here, cylindrical coordinates).

# Arguments
- `G::GridGeometry{FT}`: Grid geometry containing grid dimensions, node indices,
  and Jacobian information

# Returns
- `DiscretizedOperator`, which contains a sparse matrix of size (NR*NZ)×(NR*NZ) representing the differential operator
"""
function construct_𝐽⁻¹∂R_𝐽_operator(G::GridGeometry{FT}) where {FT<:AbstractFloat}
    # [(1/R)(∂/∂R)*(R f)] operator
    NR, NZ = G.NR, G.NZ
    nid = G.nodes.nid
    Jacob = G.Jacob
    inv_Jacob = G.inv_Jacob
    inv_dR = one(FT) / G.dR

    # define constants with FT for type stability
    half = FT(0.5)

    # Pre-allocate arrays for sparse matrix construction
    num_entries = (NR-2) * (NZ-2) * 2
    I = zeros(Int, num_entries)  # Row indices
    J = zeros(Int, num_entries)  # Column indices
    V = zeros(FT, num_entries)   # Values (all zeros initially)

    # Fill arrays for sparse matrix construction
    k = 1
    for j in 2:NZ-1
        for i in 2:NR-1
            # Set row indices
            I[k:k+1] .= nid[i, j]
            # East [i+1,j]
            J[k] = nid[i+1, j]
            V[k] = (inv_Jacob[i,j]* half * inv_dR) * Jacob[i+1,j]
            # West [i-1,j]
            J[k+1] = nid[i-1, j]
            V[k+1] = -(inv_Jacob[i,j]* half * inv_dR) * Jacob[i-1,j]
            k += 2
        end
    end

    return DiscretizedOperator((NR,NZ), I, J, V)
end
# Convinience dispatch
function construct_𝐽⁻¹∂R_𝐽_operator(RP::RAPID{FT}) where {FT<:AbstractFloat}
    return construct_𝐽⁻¹∂R_𝐽_operator(RP.G)
end



"""
    construct_∂Z_operator(G::GridGeometry{FT}) where {FT<:AbstractFloat}

Constructs a sparse matrix operator that computes the first-order partial derivative
with respect to the vertical coordinate (∂/∂Z) using a central difference scheme.

# Arguments
- `G::GridGeometry{FT}`: Grid geometry containing dimensions, node indices, and spacing information

# Returns
- `DiscretizedOperator`, which contains a sparse matrix of size (NR*NZ)×(NR*NZ) representing the first-order
  radial derivative operator with coefficients ±0.5/dZ at interior points
"""
function construct_∂Z_operator(G::GridGeometry{FT}) where {FT<:AbstractFloat}
    # Alias necessary fields from the RP object
    NR, NZ = G.NR, G.NZ
    nid = G.nodes.nid
    inv_dZ = one(FT) / G.dZ

    # define constants with FT for type stability
    half = FT(0.5)

    # Pre-allocate arrays for sparse matrix construction
    num_entries = (NR-2) * (NZ-2) * 2
    I = zeros(Int, num_entries)  # Row indices
    J = zeros(Int, num_entries)  # Column indices
    V = zeros(FT, num_entries)   # Values (all zeros initially)

    # Fill arrays for sparse matrix construction
    k = 1
    for j in 2:NZ-1
        for i in 2:NR-1
            # Set row indices
            I[k:k+1] .= nid[i, j]
            # North [i,j+1]
            J[k] = nid[i, j+1]
            V[k] = half * inv_dZ
            # South [i,j-1]
            J[k+1] = nid[i, j-1]
            V[k+1] = -half * inv_dZ
            k += 2
        end
    end

    return DiscretizedOperator((NR,NZ), I, J, V)
end

# Convinience dispatch
function construct_∂Z_operator(RP::RAPID{FT}) where{FT<:AbstractFloat}
    return construct_∂Z_operator(RP.G)
end

"""
    calculate_divergence(
        OP::Operators{FT},
        vecR::AbstractVector{FT},
        vecZ::AbstractVector{FT}
    ) where {FT<:AbstractFloat}

Calculate the divergence of a vector field (vecR, vecZ) using pre-constructed matrix operators.

# Arguments
- `OP::Operators{FT}`: Operator struct containing differential operator matrices
- `vecR::AbstractVector{FT}`: Flattened vector of radial components
- `vecZ::AbstractVector{FT}`: Flattened vector of vertical components

# Returns
- `Vector{FT}`: Flattened divergence field

# Notes
- Expects flattened vectors from 2D fields
- Uses matrix multiplication for efficient calculation
- Automatically handles cylindrical coordinate Jacobian factors
"""
@inline function calculate_divergence(
                    OP::Operators{FT},
                    vecR::AbstractVector{FT},
                    vecZ::AbstractVector{FT}) where {FT<:AbstractFloat}
    @assert size(vecR) == size(vecZ) "Vector sizes do not match"
    @assert prod(OP.dims) == length(vecR) "Operator and vector sizes do not match"

    return OP.𝐽⁻¹∂R_𝐽*vecR .+ OP.∂Z*vecZ
end

"""
    calculate_divergence(
        OP::Operators{FT},
        vecR::AbstractMatrix{FT},
        vecZ::AbstractMatrix{FT}
    ) where {FT<:AbstractFloat}

Calculate the divergence of a 2D vector field (vecR, vecZ) using pre-constructed matrix operators.

# Arguments
- `OP::Operators{FT}`: Operator struct containing differential operator matrices
- `vecR::AbstractMatrix{FT}`: 2D matrix of radial components
- `vecZ::AbstractMatrix{FT}`: 2D matrix of vertical components

# Returns
- `Matrix{FT}`: 2D divergence field

# Notes
- Preserves 2D structure of input fields
- Internally flattens matrices for matrix-vector multiplication
- Automatically handles cylindrical coordinate Jacobian factors
"""
@inline function calculate_divergence(
                    OP::Operators{FT},
                    vecR::AbstractMatrix{FT},
                    vecZ::AbstractMatrix{FT}) where {FT<:AbstractFloat}
    @assert size(vecR) == size(vecZ) "Matrix sizes do not match"
    @assert OP.dims == size(vecR) "Operator and vector sizes do not match"

    # return reshape(OP.𝐽⁻¹∂R_𝐽*@view(vecR[:]) .+ OP.∂Z*@view(vecZ[:]), OP.dims)
    return OP.𝐽⁻¹∂R_𝐽*vecR .+ OP.∂Z*vecZ
end


"""
    calculate_divergence(
        G::GridGeometry{FT},
        𝐯R::AbstractMatrix{FT},
        𝐯Z::AbstractMatrix{FT}
        ) where {FT<:AbstractFloat}

Calculate the divergence of a vector field F = [𝐯R, 𝐯Z] in cylindrical coordinates.
div(F) = (1/𝐽)∂(𝐽 𝐯R)/∂R + ∂(𝐯Z)/∂Z, where 𝐽 is the Jacobian.

# Arguments
- `RP::RAPID{FT}`: The RAPID object containing simulation state
- `𝐯R::AbstractMatrix{FT}`: The radial component of the vector field
- `𝐯Z::AbstractMatrix{FT}`: The vertical component of the vector field

# Returns
- `result`: divergence of the vector field F at each grid point

# Notes
- Uses 2nd order central differencing
- Accounts for the Jacobian in the divergence calculation: ∇·F = (1/J)∂(JFᵢ)/∂xᵢ
"""
function calculate_divergence(
        G::GridGeometry{FT},
        𝐯R::AbstractMatrix{FT},
        𝐯Z::AbstractMatrix{FT}
    ) where {FT<:AbstractFloat}

    # Alias necessary fields 𝐯Rom the RP object
    Jacob = G.Jacob
    inv_Jacob = G.inv_Jacob
    NR, NZ = G.NR, G.NZ

    # Precompute inverse values for faster calculation
    half_inv_dR = FT(0.5) / G.dR
    half_inv_dZ = FT(0.5) / G.dZ

    # Ensure the result array is properly initialized
    result = zeros(FT, NR, NZ)

    # 2nd order central differencing
    @inbounds for j in 2:NZ-1
        for i in 2:NR-1
            # Apply central difference formula with Jacobian
            result[i, j] = inv_Jacob[i, j] * (
                    (Jacob[i+1, j] * 𝐯R[i+1, j] - Jacob[i-1, j] * 𝐯R[i-1, j]) * half_inv_dR +
                    (Jacob[i, j+1] * 𝐯Z[i, j+1] - Jacob[i, j-1] * 𝐯Z[i, j-1]) * half_inv_dZ
                )
        end
    end

    return result
end

"""
    construct_ΔGS_operator(G::GridGeometry{FT}) where {FT<:AbstractFloat}

Constructs a sparse matrix operator for the Grad-Shafranov differential operator in cylindrical coordinates (R, Z).

# Mathematical Definition
The Grad-Shafranov operator is defined as:
```
ΔGS ≡ ∂²/∂R² - (1/R)∂/∂R + ∂²/∂Z²
```

This operator appears in the Grad-Shafranov equation for magnetohydrodynamic equilibrium:
```
ΔGS ψ = -μ₀ * R * Jϕ
      = -μ₀ R² p'(ψ) - FF'(ψ)
```
,where ψ is the magnetic flux function, Jϕ is the toroidal current density, p'(ψ) is the pressure gradient, and FF'(ψ) is related to the poloidal current function.

# Discretization Scheme
The operator is discretized using finite differences on a regular cylindrical grid with Dirichlet boundary conditions.

# Arguments
- `G::GridGeometry{FT}`: Grid geometry containing:

# Returns
- `DiscretizedOperator{FT}`: A sparse matrix operator of size (NR×NZ) × (NR×NZ) representing the discretized Grad-Shafranov operator

# See Also
- [`calculate_B_from_ψ!`](@ref): Computes magnetic field components from flux function
"""
function construct_ΔGS_operator(G::GridGeometry{FT}) where {FT<:AbstractFloat}
    @timeit RAPID_TIMER "constrcut_ΔGS_operator" begin
        # ΔGS ≡ (∂R)^2 - (1/R)*∂R + (∂Z)^2
        NR, NZ = G.NR, G.NZ
        nid = G.nodes.nid
        inv_dR = one(FT) / G.dR
        inv_dZ = one(FT) / G.dZ

        # define constants with FT for type stability
        half = FT(0.5)
        twoFT = FT(2.0)

        # Pre-allocate arrays for sparse matrix construction
        num_entries = (NR-2) * (NZ-2) * 5 + 2*NR+2*NZ-4;
        I = zeros(Int, num_entries)  # Row indices
        J = zeros(Int, num_entries)  # Column indices
        V = zeros(FT, num_entries)   # Values (all zeros initially)

        # Fill arrays for sparse matrix construction
        k = 1
        for j in 1:NZ
            for i in 1:NR

                if i == 1 || i == NR || j == 1 || j == NZ
                    # Boundary nodes only have one neighbor, so we skip them
                    I[k] = nid[i, j]
                    J[k] = nid[i, j]
                    V[k] = FT(1.0)  # Dirichlet boundary condition
                    k += 1
                    continue
                end

                # Set row indices
                I[k:k+4] .= nid[i, j]

                # Note the negative sign of -(1/R)*∂R

                # East [i+1, j]
                J[k] = nid[i+1, j]
                V[k] = inv_dR^twoFT -(half*inv_dR / G.R1D[i])

                # West [i-1, j]
                J[k+1] = nid[i-1, j]
                V[k+1] = inv_dR^twoFT +(half*inv_dR / G.R1D[i])

                # North [i, j+1]
                J[k+2] = nid[i, j+1]
                V[k+2] = inv_dZ^twoFT

                # South [i, j-1]
                J[k+3] = nid[i, j-1]
                V[k+3] = inv_dZ^twoFT

                # Center [i, j]
                J[k+4] = nid[i, j]
                V[k+4] = -twoFT*(inv_dR^twoFT + inv_dZ^twoFT)

                k += 5
            end
        end

        return DiscretizedOperator((NR,NZ), I, J, V)
    end
end



"""
    construct_∇𝐃∇_operator!(RP::RAPID{FT}) where {FT<:AbstractFloat}

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
function construct_∇𝐃∇_operator(RP::RAPID{FT}) where {FT<:AbstractFloat}
    @timeit RAPID_TIMER "construct_∇𝐃∇_operator" begin
        ∇𝐃∇ = allocate_∇𝐃∇_operator_pattern(RP)
        update_∇𝐃∇_operator!(RP; ∇𝐃∇)
        return ∇𝐃∇
    end
end

"""
    allocate_∇𝐃∇_operator_pattern(RP::RAPID{FT}) where {FT<:AbstractFloat}

Create a sparse matrix with the sparsity pattern for the diffusion operator [∇𝐃∇] without computing coefficient values.

# Arguments
- `RP::RAPID{FT}`: The RAPID object containing simulation state

# Notes
- This function only creates the sparsity pattern (non-zero locations) without computing the actual coefficients
- The function is called by `construct_∇𝐃∇_operator!` to set up the structure before filling in values
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

    return DiscretizedOperator((NR,NZ), I, J, V)
end

"""
    update_∇𝐃∇_operator!(RP::RAPID{FT}; ∇𝐃∇::DiscretizedOperator=RP.operators.∇𝐃∇) where {FT<:AbstractFloat}

Update the non-zero entries of the diffusion operator matrix based on the current state of the RAPID object.
# Arguments
- `RP::RAPID{FT}`: The RAPID object containing simulation state

# Returns
- `RP`: The updated RAPID object with the diffusion operator matrix updated

# Notes
- The function assumes that the diffusion operator matrix has already been initialized with the correct sparsity pattern.
- The function updates the non-zero entries of the matrix based on the current state of the RAPID object.
"""
function update_∇𝐃∇_operator!(RP::RAPID{FT}; ∇𝐃∇::DiscretizedOperator=RP.operators.∇𝐃∇) where {FT<:AbstractFloat}
    @timeit RAPID_TIMER "update_∇𝐃∇_operator!" begin
        @assert !isempty(∇𝐃∇.matrix.nzval) "Diffusion operator not initialized"

        # Alias necessary fields from the RP object
        inv_Jacob = RP.G.inv_Jacob
        NR, NZ = RP.G.NR, RP.G.NZ

        CTRR = RP.transport.CTRR
        CTRZ = RP.transport.CTRZ
        CTZZ = RP.transport.CTZZ

        # define constants with FT for type stability
        half = FT(0.5)
        eighth = FT(0.125)
        two_FT = FT(2.0)
        zero_FT = zero(FT)

        # Alias the existing sparse matrix for readability
        nzval = ∇𝐃∇.matrix.nzval
        k2csc = ∇𝐃∇.k2csc

        k = 1

        @inbounds for j in 2:NZ-1
            for i in 2:NR-1
                factor = inv_Jacob[i,j]

                nzval[k2csc[k]]   = factor * (half*(CTRR[i+1,j]+CTRR[i,j]) + eighth*(CTRZ[i,j+1]-CTRZ[i,j-1])) # East [i+1,j]
                nzval[k2csc[k+1]] = factor * (half*(CTRR[i-1,j]+CTRR[i,j]) - eighth*(CTRZ[i,j+1]-CTRZ[i,j-1])) # West [i-1,j]
                nzval[k2csc[k+2]] = factor * (half*(CTZZ[i,j+1]+CTZZ[i,j]) + eighth*(CTRZ[i+1,j]-CTRZ[i-1,j])) # North [i,j+1]
                nzval[k2csc[k+3]] = factor * (half*(CTZZ[i,j-1]+CTZZ[i,j]) - eighth*(CTRZ[i+1,j]-CTRZ[i-1,j])) # South [i,j-1]

                two_CTRZ = two_FT * CTRZ[i,j]
                nzval[k2csc[k+4]] = factor * (eighth*( two_CTRZ + CTRZ[i+1,j]+CTRZ[i,j+1]))  # Northeast [i+1,j+1]
                nzval[k2csc[k+5]] = factor * (eighth*( two_CTRZ + CTRZ[i-1,j]+CTRZ[i,j-1]))  # Southwest [i-1,j-1]
                nzval[k2csc[k+6]] = factor * (-eighth*( two_CTRZ + CTRZ[i,j+1]+CTRZ[i-1,j])) # Northwest [i-1,j+1]
                nzval[k2csc[k+7]] = factor * (-eighth*( two_CTRZ + CTRZ[i,j-1]+CTRZ[i+1,j])) # Southeast [i+1,j-1]

                nzval[k2csc[k+8]] = zero_FT
                @inbounds for t in 0:7
                    nzval[k2csc[k+8]] -= nzval[k2csc[k+t]]
                end

                k += 9
            end
        end

        return RP
    end
end


"""
    compute_𝐮∇f_directly(
        RP::RAPID{FT},
        f::AbstractMatrix{FT},
        uR::AbstractMatrix{FT}=RP.plasma.ueR,
        uZ::AbstractMatrix{FT}=RP.plasma.ueZ
        ;
        flag_upwind::Bool=RP.flags.upwind) where {FT<:AbstractFloat}

Directly compute 𝐮⋅∇f using explicit finite difference.

This function computes the parallel gradient of f along 𝐮 [𝐮⋅∇f] where f is a scalar field
and 𝐮 = (uR, uZ) is the (velocity or other) vector field. The computation uses finite differences with
support for both upwind and central differencing schemes.

# Arguments
- `RP::RAPID{FT}`: The RAPID simulation object containing grid geometry
- `f::AbstractMatrix{FT}`: The scalar field to be advected
- `uR::AbstractMatrix{FT}`: Radial velocity component (default: RP.plasma.ueR)
- `uZ::AbstractMatrix{FT}`: Vertical velocity component (default: RP.plasma.ueZ)
- `flag_upwind::Bool`: If true, use upwind differencing; otherwise central (default: RP.flags.upwind)

# Returns
- `𝐮∇f::AbstractMatrix{FT}`: parallel gradient of f along 𝐮

# Mathematical Description
The convective-flux divergence in cylindrical coordinates with Jacobian is:
```
𝐮⋅∇f = [uR*∂f/∂R + uZ*∂f/∂Z]
```

# Notes
- Only interior points (2:NR-1, 2:NZ-1) are computed; boundary values remain unchanged
- Upwind scheme uses one-sided differences in the direction opposite to velocity
- Central scheme uses symmetric differences for second-order accuracy
"""
function compute_𝐮∇f_directly(
    RP::RAPID{FT},
    f::AbstractMatrix{FT},
    uR::AbstractMatrix{FT}=RP.plasma.ueR,
    uZ::AbstractMatrix{FT}=RP.plasma.ueZ
    ;
    flag_upwind::Bool=RP.flags.upwind) where {FT<:AbstractFloat}

    @timeit RAPID_TIMER "compute_𝐮∇f_directly" begin
        # Alias necessary fields from the RP object
        G = RP.G
        NR, NZ = G.NR, G.NZ
        dR, dZ = G.dR, G.dZ

        # Precompute inverse values for faster calculation (multiplication instead of division)
        inv_dR = one(FT) / dR
        inv_dZ = one(FT) / dZ

        # Cache common constants with proper type once
        zero_val = zero(FT)
        eps_val = eps(FT)
        half = FT(0.5)  # Define half once with correct type


        𝐮∇f = zeros(size(f))

        # Apply appropriate differencing scheme based on upwind flag and velocity
        # Move the upwind flag check outside the loop for better performance
        if flag_upwind
            # Upwind scheme with check for zero velocity
            @inbounds for j in 2:NZ-1
                for i in 2:NR-1
                    # R-direction convection flux with upwind scheme
                    if abs(uR[i, j]) < eps_val
                        # Zero velocity, use central differencing
                        uR_∇f = uR[i, j] * half * inv_dR * (f[i+1, j] - f[i-1, j])
                    elseif uR[i, j] > zero_val
                        # Flow from left to right, use left (upwind) node
                        uR_∇f = uR[i, j] * inv_dR * (f[i, j] - f[i-1, j])
                    else
                        # Flow from right to left, use right (upwind) node
                        uR_∇f = uR[i, j] * inv_dR * (f[i+1, j] - f[i, j])
                    end

                    # Z-direction convection flux with upwind scheme
                    if abs(uZ[i, j]) < eps_val
                        # Zero velocity, use central differencing
                        uZ_∇f = uZ[i, j] * half * inv_dZ * (f[i, j+1] - f[i, j-1])
                    elseif uZ[i, j] > zero_val
                        # Flow from bottom to top, use bottom (upwind) node
                        uZ_∇f = uZ[i, j] * inv_dZ * (f[i, j] - f[i, j-1])
                    else
                        # Flow from top to bottom, use top (upwind) node
                        uZ_∇f = uZ[i, j] * inv_dZ * (f[i, j+1] - f[i, j])
                    end

                    # Calculate the convective-flux divergance [∇f𝐮]
                    𝐮∇f[i, j] = uR_∇f + uZ_∇f
                end
            end
        else
            # Central differencing for both directions (simpler logic)
            @inbounds for j in 2:NZ-1
                for i in 2:NR-1
                    𝐮∇f[i, j] = (
                        +uR[i, j] * half * inv_dR * (f[i+1, j] - f[i-1, j])
                        +
                        uZ[i, j] * half * inv_dZ * (f[i, j+1] - f[i, j-1])
                    )
                end
            end
        end

        return 𝐮∇f
    end
end


"""
    construct_𝐮∇_operator(RP::RAPID{FT},
                        uR::AbstractMatrix{FT}=RP.plasma.ueR,
                        uZ::AbstractMatrix{FT}=RP.plasma.ueZ;
                        flag_upwind::Bool=RP.flags.upwind) where {FT<:AbstractFloat}

Construct the sparse matrix representation of the advection operator (u·∇) with appropriate
sparsity pattern and initial values.

# Arguments
- `RP::RAPID{FT}`: The RAPID object containing simulation state
- `flag_upwind::Bool=RP.flags.upwind`: Flag to use upwind scheme (if false, uses central differencing)

# Returns
- The updated RAPID object with initialized advection operator

# Notes
- This function first creates the sparsity pattern and then updates the values
- Uses `allocate_𝐮∇_operator_pattern` to create the matrix structure
- Uses `update_𝐮∇_operator!` to populate the non-zero values
"""
function construct_𝐮∇_operator(RP::RAPID{FT},
                            uR::AbstractMatrix{FT}=RP.plasma.ueR,
                            uZ::AbstractMatrix{FT}=RP.plasma.ueZ;
                            flag_upwind::Bool=RP.flags.upwind) where {FT<:AbstractFloat}
    @timeit RAPID_TIMER "construct_𝐮∇_operator" begin
        # Create the sparsity patter
        𝐮∇ = allocate_𝐮∇_operator_pattern(RP)

        # Update the values based on current velocity field
        update_𝐮∇_operator!(RP, uR, uZ; 𝐮∇, flag_upwind)

        return 𝐮∇
    end
end

"""
    allocate_𝐮∇_operator_pattern(RP::RAPID{FT}) where {FT<:AbstractFloat}

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
function allocate_𝐮∇_operator_pattern(RP::RAPID{FT}) where {FT<:AbstractFloat}
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

    return DiscretizedOperator((NR,NZ), I, J, V)
end

"""
    update_𝐮∇_operator!(RP::RAPID{FT},
                        uR::AbstractMatrix{FT}=RP.plasma.ueR,
                        uZ::AbstractMatrix{FT}=RP.plasma.ueZ;
                        𝐮∇::DiscretizedOperator{FT}=RP.operators.𝐮∇,
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
                            𝐮∇::DiscretizedOperator{FT}=RP.operators.𝐮∇,
                            flag_upwind::Bool=RP.flags.upwind) where {FT<:AbstractFloat}

    @timeit RAPID_TIMER "update_𝐮∇_operator!" begin
        @assert !isempty(𝐮∇.matrix.nzval) "Divergence operator not initialized"

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
        nzval = 𝐮∇.matrix.nzval
        k2csc = 𝐮∇.k2csc

        # Reset values to zero
        fill!(nzval, zero_FT)

        # Update values based on current velocity and chosen scheme
        # Start index for the main loop entries
        k = 1
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
end

"""
    compute_∇f𝐮_directly(
        RP::RAPID{FT},
        f::AbstractMatrix{FT},
        uR::AbstractMatrix{FT}=RP.plasma.ueR,
        uZ::AbstractMatrix{FT}=RP.plasma.ueZ
        ;
        flag_upwind::Bool=RP.flags.upwind) where {FT<:AbstractFloat}

Directly compute ∇⋅(f𝐮) using explicit finite difference.

This function computes the convective-flux divergence ∇⋅(f𝐮) where f is a scalar field
and 𝐮 = (uR, uZ) is the velocity field. The computation uses finite differences with
support for both upwind and central differencing schemes.

# Arguments
- `RP::RAPID{FT}`: The RAPID simulation object containing grid geometry
- `f::AbstractMatrix{FT}`: The scalar field to be advected
- `uR::AbstractMatrix{FT}`: Radial velocity component (default: RP.plasma.ueR)
- `uZ::AbstractMatrix{FT}`: Vertical velocity component (default: RP.plasma.ueZ)
- `flag_upwind::Bool`: If true, use upwind differencing; otherwise central (default: RP.flags.upwind)

# Returns
- `∇f𝐮::AbstractMatrix{FT}`: The convective-flux divergence ∇⋅(f𝐮)

# Mathematical Description
The convective-flux divergence in cylindrical coordinates with Jacobian is:
```
∇⋅(f𝐮) = (1/J) * [∂(J*f*uR)/∂R + ∂(J*f*uZ)/∂Z]
```

# Notes
- Only interior points (2:NR-1, 2:NZ-1) are computed; boundary values remain unchanged
- Upwind scheme uses one-sided differences in the direction opposite to velocity
- Central scheme uses symmetric differences for second-order accuracy
"""
function compute_∇f𝐮_directly(
    RP::RAPID{FT},
    f::AbstractMatrix{FT},
    uR::AbstractMatrix{FT}=RP.plasma.ueR,
    uZ::AbstractMatrix{FT}=RP.plasma.ueZ
    ;
    flag_upwind::Bool=RP.flags.upwind) where {FT<:AbstractFloat}

    @timeit RAPID_TIMER "compute_∇f𝐮_directly" begin
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


        ∇f𝐮 = zeros(size(f))

        # Apply appropriate differencing scheme based on upwind flag and velocity
        # Move the upwind flag check outside the loop for better performance
        if flag_upwind
            # Upwind scheme with check for zero velocity
            @inbounds for j in 2:NZ-1
                for i in 2:NR-1
                    flux_R = zero_val
                    flux_Z = zero_val

                    # R-direction convection flux with upwind scheme
                    if abs(uR[i, j]) < eps_val
                        # Zero velocity, use central differencing
                        flux_R = ( +Jacob[i+1, j] * uR[i+1, j] * half * inv_dR * f[i+1, j]
                                    -Jacob[i-1, j] * uR[i-1, j] * half * inv_dR * f[i-1, j])
                    elseif uR[i, j] > zero_val
                        # Flow from left to right, use left (upwind) node
                        flux_R = ( +Jacob[i, j] * uR[i, j] * inv_dR * f[i, j]
                                    - Jacob[i-1, j] * uR[i-1, j] * inv_dR * f[i-1, j])
                    else
                        # Flow from right to left, use right (upwind) node
                        flux_R = ( +Jacob[i+1, j] * uR[i+1, j] * inv_dR * f[i+1, j]
                                    - Jacob[i, j] * uR[i, j] * inv_dR * f[i, j])
                    end

                    # Z-direction convection flux with upwind scheme
                    if abs(uZ[i, j]) < eps_val
                        # Zero velocity, use central differencing
                        flux_Z = ( +Jacob[i, j+1] * uZ[i, j+1] * half * inv_dZ * f[i, j+1]
                                    - Jacob[i, j-1] * uZ[i, j-1] * half * inv_dZ * f[i, j-1])
                    elseif uZ[i, j] > zero_val
                        # Flow from bottom to top, use bottom (upwind) node
                        flux_Z = ( +Jacob[i, j] * uZ[i, j] * inv_dZ * f[i, j]
                                    - Jacob[i, j-1] * uZ[i, j-1] * inv_dZ * f[i, j-1])
                    else
                        # Flow from top to bottom, use top (upwind) node
                        flux_Z = ( +Jacob[i, j+1] * uZ[i, j+1] * inv_dZ * f[i, j+1]
                                    - Jacob[i, j] * uZ[i, j] * inv_dZ * f[i, j])
                    end

                    # Calculate the convective-flux divergance [∇f𝐮]
                    ∇f𝐮[i, j] = (flux_R + flux_Z) * inv_Jacob[i, j]
                end
            end
        else
            # Central differencing for both directions (simpler logic)
            @inbounds for j in 2:NZ-1
                for i in 2:NR-1
                    ∇f𝐮[i, j] = inv_Jacob[i, j]*(
                        +Jacob[i+1, j] * uR[i+1, j] * half * inv_dR * f[i+1, j]
                        -Jacob[i-1, j] * uR[i-1, j] * half * inv_dR * f[i-1, j]
                        +Jacob[i, j+1] * uZ[i, j+1] * half * inv_dZ * f[i, j+1]
                        -Jacob[i, j-1] * uZ[i, j-1] * half * inv_dZ * f[i, j-1]
                    )
                end
            end
        end

        return ∇f𝐮
    end
end


"""
    construct_∇𝐮_operator(RP::RAPID{FT},
                        uR::AbstractMatrix{FT}=RP.plasma.ueR,
                        uZ::AbstractMatrix{FT}=RP.plasma.ueZ;
                        flag_upwind::Bool=RP.flags.upwind) where {FT<:AbstractFloat}

Initialize the sparse matrix representation of the convective-flux divergence [∇⋅(𝐮 f)],
where 𝐮 is the given velocity vector, and f is the scalar field to apply ∇𝐮 operator

# Arguments
- `RP::RAPID{FT}`: The RAPID object containing simulation state
- `flag_upwind::Bool=RP.flags.upwind`: Flag to use upwind scheme (if false, uses central differencing)

# Returns
- `RP`: The updated RAPID object with initialized divergence operator

# Notes
- This function first creates the sparsity pattern and then updates the values
- Uses `allocate_∇𝐮_operator_pattern` to create the matrix structure
- Uses `update_∇𝐮_operator!` to populate the non-zero values
"""
function construct_∇𝐮_operator(RP::RAPID{FT},
                            uR::AbstractMatrix{FT}=RP.plasma.ueR,
                            uZ::AbstractMatrix{FT}=RP.plasma.ueZ;
                            flag_upwind::Bool=RP.flags.upwind) where {FT<:AbstractFloat}
    @timeit RAPID_TIMER "construct_∇𝐮_operator" begin
        # create a sparse matrix with the sparisty pattern
        ∇𝐮 = allocate_∇𝐮_operator_pattern(RP)
        # update the divergence operator's non-zero entries with the actual values
        update_∇𝐮_operator!(RP, uR, uZ; ∇𝐮, flag_upwind)

        return ∇𝐮
    end
end

"""
    allocate_∇𝐮_operator_pattern(RP::RAPID{FT}) where {FT<:AbstractFloat}

Create a sparse matrix with the sparsity pattern for the convective-flux divergence [∇⋅(𝐮 f)]
without computing coefficient values.

# Arguments
- `RP::RAPID{FT}`: The RAPID object containing simulation state

# Returns
- The updated RAPID object with allocated sparsity pattern

# Notes
- This function only creates the sparsity pattern (non-zero locations) without computing the actual coefficients
- The function is called by `construct_∇𝐮_operator` to set up the structure before filling in values
- Supports both upwind and central differencing schemes
"""
function allocate_∇𝐮_operator_pattern(RP::RAPID{FT}) where {FT<:AbstractFloat}
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

    return DiscretizedOperator((NR,NZ), I, J, V)
end

"""
    update_∇𝐮_operator!(RP::RAPID{FT},
                           uR::AbstractMatrix{FT}=RP.plasma.ueR,
                           uZ::AbstractMatrix{FT}=RP.plasma.ueZ;
                           ∇𝐮::DiscretizedOperator{FT}=RP.operators.∇𝐮,
                           flag_upwind::Bool=RP.flags.upwind) where {FT<:AbstractFloat}

Update the non-zero entries of the convective-flux divergence [∇⋅(𝐮 f)], based on the current state of the RAPID object.
# Arguments
- `RP::RAPID{FT}`: The RAPID object containing simulation state

# Returns
- `RP`: The updated RAPID object with the divergence operator matrix updated

# Notes
- The function assumes that the divergence operator matrix has already been initialized with the correct sparsity pattern.
- The function updates the non-zero entries of the matrix based on the current state of the RAPID object.
- This operator has opposite signs compared to the convection operator [-∇⋅(nv)].
"""
function update_∇𝐮_operator!(RP::RAPID{FT},
                            uR::AbstractMatrix{FT}=RP.plasma.ueR,
                            uZ::AbstractMatrix{FT}=RP.plasma.ueZ;
                            ∇𝐮::DiscretizedOperator{FT}=RP.operators.∇𝐮,
                            flag_upwind::Bool=RP.flags.upwind) where {FT<:AbstractFloat}

    @timeit RAPID_TIMER "update_∇𝐮_operator!" begin
        @assert !isempty(∇𝐮.matrix.nzval) "Divergence operator not initialized"

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
        nzval = ∇𝐮.matrix.nzval
        k2csc = ∇𝐮.k2csc

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
                        nzval[k2csc[k+1]] += Jacob[i+1,j]*uR[i+1,j]*half*inv_dR*ij_factor  # East - Sign flipped from convection
                        nzval[k2csc[k+2]] -= Jacob[i-1,j]*uR[i-1,j]*half*inv_dR*ij_factor  # West - Sign flipped from convection
                    elseif uR[i,j] > zero_FT
                        # Positive velocity: flow from west
                        nzval[k2csc[k]] += Jacob[i,j]*uR[i,j]*inv_dR*ij_factor          # Center - Sign flipped from convection
                        nzval[k2csc[k+2]] -= Jacob[i-1,j]*uR[i-1,j]*inv_dR*ij_factor   # West - Sign flipped from convection
                    else
                        # Negative velocity: flow from east
                        nzval[k2csc[k+1]] += Jacob[i+1,j]*uR[i+1,j]*inv_dR*ij_factor   # East - Sign flipped from convection
                        nzval[k2csc[k]] -= Jacob[i,j]*uR[i,j]*inv_dR*ij_factor         # Center - Sign flipped from convection
                    end

                    # Z-direction velocity-dependent coefficients
                    if abs(uZ[i,j]) < eps_FT
                        # Zero velocity: use central differencing
                        nzval[k2csc[k+3]] += Jacob[i,j+1]*uZ[i,j+1]*half*inv_dZ*ij_factor  # North - Sign flipped from convection
                        nzval[k2csc[k+4]] -= Jacob[i,j-1]*uZ[i,j-1]*half*inv_dZ*ij_factor  # South - Sign flipped from convection
                    elseif uZ[i,j] > zero_FT
                        # Positive velocity: flow from south
                        nzval[k2csc[k]] += Jacob[i,j]*uZ[i,j]*inv_dZ*ij_factor          # Center - Sign flipped from convection
                        nzval[k2csc[k+4]] -= Jacob[i,j-1]*uZ[i,j-1]*inv_dZ*ij_factor   # South - Sign flipped from convection
                    else
                        # Negative velocity: flow from north
                        nzval[k2csc[k+3]] += Jacob[i,j+1]*uZ[i,j+1]*inv_dZ*ij_factor   # North - Sign flipped from convection
                        nzval[k2csc[k]] -= Jacob[i,j]*uZ[i,j]*inv_dZ*ij_factor         # Center - Sign flipped from convection
                    end

                    k += 5
                end
            end
        else
            @inbounds for j in 2:NZ-1
                for i in 2:NR-1
                    # Jacobian factor at current position
                    ij_factor = inv_Jacob[i,j]

                    # Always use central differencing - All signs flipped from convection
                    nzval[k2csc[k+1]] += Jacob[i+1,j]*uR[i+1,j]*half*inv_dR*ij_factor  # East
                    nzval[k2csc[k+2]] -= Jacob[i-1,j]*uR[i-1,j]*half*inv_dR*ij_factor  # West
                    nzval[k2csc[k+3]] += Jacob[i,j+1]*uZ[i,j+1]*half*inv_dZ*ij_factor  # North
                    nzval[k2csc[k+4]] -= Jacob[i,j-1]*uZ[i,j-1]*half*inv_dZ*ij_factor  # South

                    k += 5
                end
            end
        end

        return RP
    end
end