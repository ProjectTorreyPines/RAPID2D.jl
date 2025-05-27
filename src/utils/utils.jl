"""
Utilities module for RAPID2D.

Contains utility functions for the RAPID2D package.
"""

# Export public functions
export get_wall_status,
       is_inside_wall

"""
    get_wall_status(R::FT, Z::FT, wall_R::Vector{FT}, wall_Z::Vector{FT}) where {FT<:AbstractFloat}

Determine if a point (R,Z) is inside, outside, or on the wall.
Returns:
- 1 if inside the wall
- 0 if on the wall
- -1 if outside the wall
"""
function get_wall_status(R::FT, Z::FT, wall_R::Vector{FT}, wall_Z::Vector{FT}) where {FT<:AbstractFloat}
    # Implementation of point-in-polygon test using ray casting algorithm
    # Count the number of times a ray from the point to the right crosses the polygon edges

    # If there are fewer than 3 points, we don't have a polygon
    if length(wall_R) < 3
        return -1
    end

    # Initialize crossing count
    count = 0
    N = length(wall_R)

    for i in 1:N
        # Get current point and next point (wrap around for last point)
        x1, y1 = wall_R[i], wall_Z[i]
        x2, y2 = wall_R[mod1(i+1, N)], wall_Z[mod1(i+1, N)]

        # Check if point is on this edge
        if is_on_line_segment(R, Z, x1, y1, x2, y2)
            return 0  # On the wall
        end

        # Check if ray crosses this edge
        if ((y1 <= Z && Z < y2) || (y2 <= Z && Z < y1)) &&
           (R < x1 + (Z - y1) * (x2 - x1) / (y2 - y1))
            count += 1
        end
    end

    # If count is odd, point is inside
    return (count % 2 == 1) ? 1 : -1
end

"""
    get_wall_status(R::FT, Z::FT, wall::WallGeometry{FT}) where {FT<:AbstractFloat}

Determine if a point (R,Z) is inside, outside, or on the wall.
Returns:
- 1 if inside the wall
- 0 if on the wall
- -1 if outside the wall
"""
function get_wall_status(R::FT, Z::FT, wall::WallGeometry{FT}) where {FT<:AbstractFloat}
    # Call the existing implementation with the WallGeometry's R and Z vectors
    return get_wall_status(R, Z, wall.R, wall.Z)
end

"""
    is_on_line_segment(x::FT, y::FT, x1::FT, y1::FT, x2::FT, y2::FT) where {FT<:AbstractFloat}

Check if point (x,y) is on the line segment from (x1,y1) to (x2,y2).
"""
function is_on_line_segment(x::FT, y::FT, x1::FT, y1::FT, x2::FT, y2::FT) where {FT<:AbstractFloat}
    # Calculate the distance from the point to the line segment
    # and compare with a small tolerance

    # Length of the line segment squared
    line_length_sq = (x2 - x1)^2 + (y2 - y1)^2

    # If line segment is a point, check distance to that point
    if line_length_sq < FT(1e-10)
        return sqrt((x - x1)^2 + (y - y1)^2) < FT(1e-5)
    end

    # Calculate projection of point onto line segment
    t = max(FT(0.0), min(FT(1.0), ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / line_length_sq))

    # Calculate closest point on line segment
    px = x1 + t * (x2 - x1)
    py = y1 + t * (y2 - y1)

    # Check distance from original point to closest point
    return sqrt((x - px)^2 + (y - py)^2) < FT(1e-5)
end

"""
    is_inside_wall(R::FT, Z::FT, wall_R::Vector{FT}, wall_Z::Vector{FT}) where {FT<:AbstractFloat}

Check if a point (R,Z) is inside the wall.
"""
function is_inside_wall(R::FT, Z::FT, wall_R::Vector{FT}, wall_Z::Vector{FT}) where {FT<:AbstractFloat}
    return get_wall_status(R, Z, wall_R, wall_Z) >= 0
end

"""
    is_inside_wall(R::FT, Z::FT, wall::WallGeometry{FT}) where {FT<:AbstractFloat}

Check if a point (R,Z) is inside the wall.
"""
function is_inside_wall(R::FT, Z::FT, wall::WallGeometry{FT}) where {FT<:AbstractFloat}
    # Call the existing implementation with the WallGeometry's R and Z vectors
    return is_inside_wall(R, Z, wall.R, wall.Z)
end

"""
    is_inside_wall(R::AbstractArray{FT}, Z::AbstractArray{FT}, wall_R::Vector{FT}, wall_Z::Vector{FT}) where {FT<:AbstractFloat}

Check if points in arrays R and Z are inside the wall. Returns an array of the same size as R and Z
with `true` for points inside or on the wall and `false` for points outside.

# Arguments
- `R::AbstractArray{FT}`: Array of R coordinates of points to test
- `Z::AbstractArray{FT}`: Array of Z coordinates of points to test
- `wall_R::Vector{FT}`: Vector of R coordinates of polygon vertices
- `wall_Z::Vector{FT}`: Vector of Z coordinates of polygon vertices

# Returns
- `Array{Bool}`: Array of the same size as R and Z with `true` for points inside or on the wall
  and `false` for points outside

# Example
```julia
R = [1.0, 1.2, 1.5]
Z = [0.0, 0.1, 0.2]
wall_R = [0.5, 1.5, 1.5, 0.5, 0.5]
wall_Z = [-0.5, -0.5, 0.5, 0.5, -0.5]
in_wall = is_inside_wall(R, Z, wall_R, wall_Z)
```
"""
function is_inside_wall(R::AbstractArray{FT}, Z::AbstractArray{FT}, wall_R::Vector{FT}, wall_Z::Vector{FT}) where {FT<:AbstractFloat}
    # Check that R and Z have the same size
    if size(R) != size(Z)
        throw(ArgumentError("R and Z must have the same size"))
    end

    # Create result array with the same shape as input
    result = Array{Bool}(undef, size(R))

    # Process each point in the arrays using linear indexing
    for i in eachindex(R, Z)
        # Use the existing get_wall_status function for consistency
        # If status is ≥ 0, the point is inside or on the wall
        status = get_wall_status(R[i], Z[i], wall_R, wall_Z)
        result[i] = status >= 0
    end

    return result
end

"""
    is_inside_wall(R::AbstractArray{FT}, Z::AbstractArray{FT}, wall::WallGeometry{FT}) where {FT<:AbstractFloat}

Check if points in arrays R and Z are inside the wall. Returns an array of the same size as R and Z
with `true` for points inside or on the wall and `false` for points outside.

# Arguments
- `R::AbstractArray{FT}`: Array of R coordinates of points to test
- `Z::AbstractArray{FT}`: Array of Z coordinates of points to test
- `wall::WallGeometry{FT}`: Wall geometry object containing polygon vertices

# Returns
- `Array{Bool}`: Array of the same size as R and Z with `true` for points inside or on the wall
  and `false` for points outside

# Example
```julia
R = [1.0, 1.2, 1.5]
Z = [0.0, 0.1, 0.2]
wall = WallGeometry{Float64}([0.5, 1.5, 1.5, 0.5, 0.5], [-0.5, -0.5, 0.5, 0.5, -0.5])
in_wall = is_inside_wall(R, Z, wall)
```
"""
function is_inside_wall(R::AbstractArray{FT}, Z::AbstractArray{FT}, wall::WallGeometry{FT}) where {FT<:AbstractFloat}
    # Call the existing implementation with the WallGeometry's R and Z vectors
    return is_inside_wall(R, Z, wall.R, wall.Z)
end
