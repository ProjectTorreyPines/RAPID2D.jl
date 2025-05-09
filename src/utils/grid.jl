"""
Grid initialization and manipulation functions for RAPID2D.jl
"""

"""
    initialize_grid_geometry!(grid::GridGeometry{FT},
                             R_range::Tuple{<:Real,<:Real},
                             Z_range::Tuple{<:Real,<:Real}) where FT<:AbstractFloat

Initialize a grid geometry with specified coordinate ranges.

# Arguments
- `grid`: The grid geometry to initialize
- `R_range`: Tuple of (R_min, R_max) specifying the radial coordinate range
- `Z_range`: Tuple of (Z_min, Z_max) specifying the vertical coordinate range

# Returns
- The initialized grid geometry (for chaining)
"""
function initialize_grid_geometry!(
    grid::GridGeometry{FT},
    R_range::Tuple{<:Real,<:Real},
    Z_range::Tuple{<:Real,<:Real}
) where FT<:AbstractFloat
    # Calculate grid spacings
    R_min, R_max = R_range
    Z_min, Z_max = Z_range

    grid.dR = FT((R_max - R_min) / (grid.NR - 1))
    grid.dZ = FT((Z_max - Z_min) / (grid.NZ - 1))

    # Generate 1D coordinate arrays
    for i in 1:grid.NR
        grid.R1D[i] = FT(R_min + (i-1)*grid.dR)
    end

    for j in 1:grid.NZ
        grid.Z1D[j] = FT(Z_min + (j-1)*grid.dZ)
    end

    # Generate 2D coordinate arrays
    for j in 1:grid.NZ
        for i in 1:grid.NR
            grid.R2D[j, i] = grid.R1D[i]
            grid.Z2D[j, i] = grid.Z1D[j]
        end
    end

    # Calculate Jacobian - for cylindrical coordinates, Jacob = R
    grid.Jacob .= grid.R2D

    # Calculate inverse Jacobian
    grid.inv_Jacob .= 1.0 ./ grid.Jacob

    # Define boundary indices - the perimeter of the domain using linear indices
    # This combines all four edges of the domain and ensures unique, sorted indices
    grid.BDY_idx = sort(unique([
        LinearIndices((grid.NZ, grid.NR))[1,:];       # Top edge
        LinearIndices((grid.NZ, grid.NR))[end,:];     # Bottom edge
        LinearIndices((grid.NZ, grid.NR))[:,1];       # Left edge
        LinearIndices((grid.NZ, grid.NR))[:,end]      # Right edge
    ]))

    return grid
end

"""
    initialize_grid_geometry(NR::Int,
                            NZ::Int,
                            R_range::Tuple{<:Real,<:Real},
                            Z_range::Tuple{<:Real,<:Real},
                            ::Type{T}=Float64) where T<:AbstractFloat

Create and initialize a new grid geometry with specified dimensions and coordinate ranges.

# Arguments
- `NR`: Number of radial grid points
- `NZ`: Number of vertical grid points
- `R_range`: Tuple of (R_min, R_max) specifying the radial coordinate range
- `Z_range`: Tuple of (Z_min, Z_max) specifying the vertical coordinate range
- `T`: Optional type parameter for floating-point precision (default: Float64)

# Returns
- A new initialized grid geometry
"""
function initialize_grid_geometry(
    NR::Int,
    NZ::Int,
    R_range::Tuple{<:Real,<:Real},
    Z_range::Tuple{<:Real,<:Real},
    ::Type{T}=Float64
) where T<:AbstractFloat
    # Create an empty GridGeometry
    grid = GridGeometry{T}(NR, NZ)

    # Initialize it
    initialize_grid_geometry!(grid, R_range, Z_range)

    return grid
end

# Export functions
export initialize_grid_geometry, initialize_grid_geometry!
