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

"""
    initialize_rapid_grid!(RP::RAPID{FT},
                          R_range::Tuple{<:Real,<:Real},
                          Z_range::Tuple{<:Real,<:Real}) where FT<:AbstractFloat

Initialize the grid geometry within a RAPID struct using specified coordinate ranges.

# Arguments
- `RP`: The RAPID struct to update
- `R_range`: Tuple of (R_min, R_max) specifying the radial coordinate range
- `Z_range`: Tuple of (Z_min, Z_max) specifying the vertical coordinate range

# Returns
- The updated RAPID struct (for chaining)
"""
function initialize_rapid_grid!(
    RP::RAPID{FT},
    R_range::Tuple{<:Real,<:Real},
    Z_range::Tuple{<:Real,<:Real}
) where FT<:AbstractFloat
    # Initialize the grid geometry within the RAPID struct
    initialize_grid_geometry!(RP.G, R_range, Z_range)

    return RP
end

"""
    set_rapid_grid!(RP::RAPID{FT}, grid::GridGeometry{FT}) where FT<:AbstractFloat

Set the grid geometry of a RAPID struct to the provided GridGeometry.
This replaces the current grid with the provided one.

# Arguments
- `RP`: The RAPID struct to update
- `grid`: The source GridGeometry to use

# Returns
- The updated RAPID struct (for chaining)
"""
function set_rapid_grid!(RP::RAPID{FT}, grid::GridGeometry{FT}) where FT<:AbstractFloat
    # Verify dimensions match
    if RP.G.NR != grid.NR || RP.G.NZ != grid.NZ
        error("Grid dimensions ($(grid.NR), $(grid.NZ)) do not match RAPID dimensions ($(RP.G.NR), $(RP.G.NZ))")
    end

    # Replace the grid in the RAPID struct
    RP.G = grid

    return RP
end

"""
    get_rapid_grid(RP::RAPID{FT}) where FT<:AbstractFloat

Get the GridGeometry from a RAPID struct.

# Arguments
- `RP`: The source RAPID struct

# Returns
- The GridGeometry from the RAPID struct
"""
function get_rapid_grid(RP::RAPID{FT}) where FT<:AbstractFloat
    return RP.G
end

"""
    copy_rapid_grid(RP::RAPID{FT}) where FT<:AbstractFloat

Create a deep copy of the GridGeometry from a RAPID struct.

# Arguments
- `RP`: The source RAPID struct

# Returns
- A new GridGeometry with data copied from the RAPID struct
"""
function copy_rapid_grid(RP::RAPID{FT}) where FT<:AbstractFloat
    # Create a new empty grid
    grid = GridGeometry{FT}(RP.G.NR, RP.G.NZ)

    # Copy grid information
    grid.R1D = copy(RP.G.R1D)
    grid.Z1D = copy(RP.G.Z1D)
    grid.R2D = copy(RP.G.R2D)
    grid.Z2D = copy(RP.G.Z2D)
    grid.dR = RP.G.dR
    grid.dZ = RP.G.dZ
    grid.Jacob = copy(RP.G.Jacob)
    grid.inv_Jacob = copy(RP.G.inv_Jacob)
    grid.BDY_idx = copy(RP.G.BDY_idx)

    return grid
end

# Export functions
export initialize_grid_geometry, initialize_grid_geometry!,
       initialize_rapid_grid!, set_rapid_grid!,
       get_rapid_grid, copy_rapid_grid