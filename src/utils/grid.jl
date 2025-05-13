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
            grid.R2D[i, j] = grid.R1D[i]
            grid.Z2D[i, j] = grid.Z1D[j]
        end
    end

    # Calculate Jacobian - for cylindrical coordinates, Jacob = R
    grid.Jacob .= grid.R2D

    # Calculate inverse Jacobian
    grid.inv_Jacob .= 1.0 ./ grid.Jacob

    # Define boundary indices - the perimeter of the domain using linear indices
    # This combines all four edges of the domain and ensures unique, sorted indices
    grid.BDY_idx = sort(unique([
        LinearIndices((grid.NR, grid.NZ))[:,1];       # Bottom edge
        LinearIndices((grid.NR, grid.NZ))[:,end];     # Top edge
        LinearIndices((grid.NR, grid.NZ))[1,:];       # Left edge
        LinearIndices((grid.NR, grid.NZ))[end,:]      # Right edge
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
    trace_zero_contour(matrix::Matrix{T}, start_position::NamedTuple{(:rid, :zid), Tuple{Int, Int}}) where T<:Real

Find a connected contour of zeros in a matrix, starting from a specified position.
Uses a depth-first search algorithm to explore all connected zeros.

# Arguments
- `matrix`: 2D matrix to search for zeros
- `start_position`: Starting position as a named tuple with fields `rid` (radial index)
                   and `zid` (vertical index)

# Returns
- Vector of named tuples with fields `rid` and `zid` representing the contour path
"""
function trace_zero_contour(matrix::Matrix{T}, start_position::NamedTuple{(:rid, :zid), Tuple{Int, Int}}) where T<:Real
    # Get matrix dimensions
    num_r, num_z = size(matrix)

    # Extract starting position
    rid, zid = start_position.rid, start_position.zid

    # Check if starting point is valid
    if matrix[rid, zid] != 0
        error("Starting point (rid=$rid, zid=$zid) does not have a value of 0")
    end

    # Direction array (right, up, left, down)
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    # Array to store visited status
    visited = falses(num_r, num_z)

    # Result path
    path = Vector{NamedTuple{(:rid, :zid), Tuple{Int, Int}}}()

    # Depth-First Search (DFS) stack
    stack = [start_position]

    while !isempty(stack)
        # Current position
        current_pos = pop!(stack)
        current_rid, current_zid = current_pos.rid, current_pos.zid

        # Skip if already visited
        if visited[current_rid, current_zid]
            continue
        end

        # Mark current position as visited
        visited[current_rid, current_zid] = true
        push!(path, current_pos)

        # Explore in all 4 directions
        for (dr, dz) in directions
            new_rid = current_rid + dr
            new_zid = current_zid + dz

            # Check if within boundaries
            if 1 <= new_rid <= num_r && 1 <= new_zid <= num_z
                # If value is 0 and not visited
                if matrix[new_rid, new_zid] == 0 && !visited[new_rid, new_zid]
                    push!(stack, (rid=new_rid, zid=new_zid))
                end
            end
        end
    end

    return path
end

"""
    trace_zero_contour(matrix::Matrix{T}, start_rid::Int, start_zid::Int) where T<:Real

Convenience method that accepts separate r and z indices instead of a named tuple.

# Arguments
- `matrix`: 2D matrix to search for zeros
- `start_rid`: Starting r-index (radial)
- `start_zid`: Starting z-index (vertical)

# Returns
- Vector of named tuples with fields `rid` and `zid` representing the contour path
"""
function trace_zero_contour(matrix::Matrix{T}, start_rid::Int, start_zid::Int) where T<:Real
    return trace_zero_contour(matrix, (rid=start_rid, zid=start_zid))
end

# Export functions
export initialize_grid_geometry, initialize_grid_geometry!, trace_zero_contour
