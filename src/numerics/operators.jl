"""
    operators.jl

This file defines the whole-grid differential operators of the RAPID simulation
framework: the basic derivatives in cylindrical coordinates and the Grad–Shafranov
operator.

Key functionalities include:
- Basic differential operators for cylindrical coordinates:
    - `construct_∂R_operator`: Builds the radial derivative operator (∂/∂R).
    - `construct_∂Z_operator`: Builds the vertical derivative operator (∂/∂Z).
    - `construct_𝐽⁻¹∂R_𝐽_operator`: Constructs the divergence-preserving radial derivative.
    - `calculate_divergence`: Computes vector field divergence in cylindrical coordinates.
- The Grad–Shafranov operator for the field solve:
    - `construct_ΔGS_operator`.

The transport operators (diffusion, convection, advection) are not built here: they live
on in-wall rows only, in `wall_diffusion.jl`, `face_flux.jl` and `primitive_advection.jl`.
"""

# Export public functions
export construct_∂R_operator, construct_∂Z_operator,
    calculate_divergence, construct_𝐽⁻¹∂R_𝐽_operator,
    construct_ΔGS_operator,
    update_diffusion_tensor!


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
function construct_∂R_operator(G::GridGeometry{FT}) where {FT <: AbstractFloat}
    # Alias necessary fields from the RP object
    NR, NZ = G.NR, G.NZ
    nid = G.nodes.nid
    inv_dR = one(FT) / G.dR

    # define constants with FT for type stability
    half = FT(0.5)

    # Pre-allocate arrays for sparse matrix construction
    num_entries = (NR - 2) * (NZ - 2) * 2
    I = zeros(Int, num_entries)  # Row indices
    J = zeros(Int, num_entries)  # Column indices
    V = zeros(FT, num_entries)   # Values (all zeros initially)

    # Fill arrays for sparse matrix construction
    k = 1
    for j in 2:(NZ - 1)
        for i in 2:(NR - 1)
            # Set row indices
            I[k:(k + 1)] .= nid[i, j]
            # East [i+1,j]
            J[k] = nid[i + 1, j]
            V[k] = half * inv_dR
            # West [i-1,j]
            J[k + 1] = nid[i - 1, j]
            V[k + 1] = -half * inv_dR
            k += 2
        end
    end

    return DiscretizedOperator((NR, NZ), I, J, V)
end

# Convinience dispatch
function construct_∂R_operator(RP::RAPID{FT}) where {FT <: AbstractFloat}
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
function construct_𝐽⁻¹∂R_𝐽_operator(G::GridGeometry{FT}) where {FT <: AbstractFloat}
    # [(1/R)(∂/∂R)*(R f)] operator
    NR, NZ = G.NR, G.NZ
    nid = G.nodes.nid
    Jacob = G.Jacob
    inv_Jacob = G.inv_Jacob
    inv_dR = one(FT) / G.dR

    # define constants with FT for type stability
    half = FT(0.5)

    # Pre-allocate arrays for sparse matrix construction
    num_entries = (NR - 2) * (NZ - 2) * 2
    I = zeros(Int, num_entries)  # Row indices
    J = zeros(Int, num_entries)  # Column indices
    V = zeros(FT, num_entries)   # Values (all zeros initially)

    # Fill arrays for sparse matrix construction
    k = 1
    for j in 2:(NZ - 1)
        for i in 2:(NR - 1)
            # Set row indices
            I[k:(k + 1)] .= nid[i, j]
            # East [i+1,j]
            J[k] = nid[i + 1, j]
            V[k] = (inv_Jacob[i, j] * half * inv_dR) * Jacob[i + 1, j]
            # West [i-1,j]
            J[k + 1] = nid[i - 1, j]
            V[k + 1] = -(inv_Jacob[i, j] * half * inv_dR) * Jacob[i - 1, j]
            k += 2
        end
    end

    return DiscretizedOperator((NR, NZ), I, J, V)
end
# Convinience dispatch
function construct_𝐽⁻¹∂R_𝐽_operator(RP::RAPID{FT}) where {FT <: AbstractFloat}
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
function construct_∂Z_operator(G::GridGeometry{FT}) where {FT <: AbstractFloat}
    # Alias necessary fields from the RP object
    NR, NZ = G.NR, G.NZ
    nid = G.nodes.nid
    inv_dZ = one(FT) / G.dZ

    # define constants with FT for type stability
    half = FT(0.5)

    # Pre-allocate arrays for sparse matrix construction
    num_entries = (NR - 2) * (NZ - 2) * 2
    I = zeros(Int, num_entries)  # Row indices
    J = zeros(Int, num_entries)  # Column indices
    V = zeros(FT, num_entries)   # Values (all zeros initially)

    # Fill arrays for sparse matrix construction
    k = 1
    for j in 2:(NZ - 1)
        for i in 2:(NR - 1)
            # Set row indices
            I[k:(k + 1)] .= nid[i, j]
            # North [i,j+1]
            J[k] = nid[i, j + 1]
            V[k] = half * inv_dZ
            # South [i,j-1]
            J[k + 1] = nid[i, j - 1]
            V[k + 1] = -half * inv_dZ
            k += 2
        end
    end

    return DiscretizedOperator((NR, NZ), I, J, V)
end

# Convinience dispatch
function construct_∂Z_operator(RP::RAPID{FT}) where {FT <: AbstractFloat}
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
        vecZ::AbstractVector{FT}
    ) where {FT <: AbstractFloat}
    @assert size(vecR) == size(vecZ) "Vector sizes do not match"
    @assert prod(OP.dims) == length(vecR) "Operator and vector sizes do not match"

    return OP.𝐽⁻¹∂R_𝐽 * vecR .+ OP.∂Z * vecZ
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
        vecZ::AbstractMatrix{FT}
    ) where {FT <: AbstractFloat}
    @assert size(vecR) == size(vecZ) "Matrix sizes do not match"
    @assert OP.dims == size(vecR) "Operator and vector sizes do not match"

    # return reshape(OP.𝐽⁻¹∂R_𝐽*@view(vecR[:]) .+ OP.∂Z*@view(vecZ[:]), OP.dims)
    return OP.𝐽⁻¹∂R_𝐽 * vecR .+ OP.∂Z * vecZ
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
    ) where {FT <: AbstractFloat}

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
    @inbounds for j in 2:(NZ - 1)
        for i in 2:(NR - 1)
            # Apply central difference formula with Jacobian
            result[i, j] = inv_Jacob[i, j] * (
                (Jacob[i + 1, j] * 𝐯R[i + 1, j] - Jacob[i - 1, j] * 𝐯R[i - 1, j]) * half_inv_dR +
                    (Jacob[i, j + 1] * 𝐯Z[i, j + 1] - Jacob[i, j - 1] * 𝐯Z[i, j - 1]) * half_inv_dZ
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
function construct_ΔGS_operator(G::GridGeometry{FT}) where {FT <: AbstractFloat}
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
        num_entries = (NR - 2) * (NZ - 2) * 5 + 2 * NR + 2 * NZ - 4
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
                I[k:(k + 4)] .= nid[i, j]

                # Note the negative sign of -(1/R)*∂R

                # East [i+1, j]
                J[k] = nid[i + 1, j]
                V[k] = inv_dR^twoFT - (half * inv_dR / G.R1D[i])

                # West [i-1, j]
                J[k + 1] = nid[i - 1, j]
                V[k + 1] = inv_dR^twoFT + (half * inv_dR / G.R1D[i])

                # North [i, j+1]
                J[k + 2] = nid[i, j + 1]
                V[k + 2] = inv_dZ^twoFT

                # South [i, j-1]
                J[k + 3] = nid[i, j - 1]
                V[k + 3] = inv_dZ^twoFT

                # Center [i, j]
                J[k + 4] = nid[i, j]
                V[k + 4] = -twoFT * (inv_dR^twoFT + inv_dZ^twoFT)

                k += 5
            end
        end

        return DiscretizedOperator((NR, NZ), I, J, V)
    end
end
