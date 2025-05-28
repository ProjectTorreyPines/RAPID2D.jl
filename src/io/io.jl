"""
I/O module for RAPID2D.

Contains functions for file input/output, including:
- Reading input data
- Saving simulation results
- Loading external field data
- Reading device geometry
- Data visualization
"""

# Export public functions
export save_snapshot,
       write_output_file,
       read_input_file,
       save_snapshot2D,

       # Wall geometry functions
       read_wall_data_file,
       read_device_wall_data!,

       # External field functions
       read_break_input_file,
       read_external_field_time_series,
       load_external_field_data!

# Required imports
using Printf
using LinearAlgebra
using DelimitedFiles
using Interpolations

# Import from fields module
import RAPID2D: TimeSeriesExternalField, AbstractExternalField
import RAPID2D: calculate_external_fields_at_time  # Import this to avoid duplicate definition

# =============================================================================
# Snapshot and diagnostic functions
# =============================================================================

"""
    save_snapshot(RP::RAPID{FT}, snapshot_type::Symbol) where {FT<:AbstractFloat}

Save a snapshot of the current simulation state.

# Arguments
- `RP::RAPID{FT}`: The RAPID simulation instance
- `snapshot_type::Symbol`: Type of snapshot to save (`:snap1D` or `:snap2D`)
"""
function save_snapshot(RP::RAPID{FT}, snapshot_type::Symbol) where {FT<:AbstractFloat}
    if snapshot_type == :snap1D
        measure_snap0D!(RP)
    elseif snapshot_type == :snap2D
        measure_snap2D!(RP)
    else
        @warn "Unknown snapshot type: $snapshot_type"
    end

    return nothing
end

"""
    save_snapshot2D(RP::RAPID{FT}) where {FT<:AbstractFloat}

Save a 2D snapshot of the current simulation state and write to file if needed.

# Arguments
- `RP::RAPID{FT}`: The RAPID simulation instance
"""
function save_snapshot2D(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # First update the snapshot data
    measure_snap2D!(RP)

    # If we need to save to file (based on RAPID2D configuration)
    if hasfield(typeof(RP), :write_snap2D_to_file) && RP.write_snap2D_to_file
        # Format time with leading zeros
        time_str = @sprintf("%08.6f", RP.time_s)

        # Construct filename
        filename = joinpath(
            RP.config.Output_path,
            "$(RP.config.Output_prefix)snap2D_t=$(time_str)s.h5"
        )

        # Create output directory if it doesn't exist
        mkpath(dirname(filename))

        # Open HDF5 file for writing
        h5open(filename, "w") do file
            # Write time
            write(file, "time_s", RP.time_s)

            # Write grid information
            write(file, "R", RP.grid.R1D)
            write(file, "Z", RP.grid.Z1D)

            # Write plasma fields
            write(file, "ne", RP.plasma.ne)
            write(file, "Te_eV", RP.plasma.Te_eV)
            write(file, "ue_para", RP.plasma.ue_para)

            # Write fields
            write(file, "E_para_tot", RP.fields.E_para_tot)
            write(file, "E_para_ext", RP.fields.E_para_ext)

            # Write simulation metadata
            write(file, "step", RP.step)
            write(file, "dt", RP.dt)
        end

        println("Saved 2D snapshot to: $filename")
    end

    return nothing
end

"""
    write_output_file(RP::RAPID{FT}, filename::String=nothing) where {FT<:AbstractFloat}

Write simulation results to file.

# Arguments
- `RP::RAPID{FT}`: The RAPID simulation instance
- `filename::String=nothing`: Optional filename for output (default generates name based on time)
"""
function write_output_file(RP::RAPID{FT}, filename::String=nothing) where {FT<:AbstractFloat}
    # Placeholder implementation - will be filled in later
    @warn "write_output_file not fully implemented yet"

    # Generate default filename if none provided
    if isnothing(filename)
        # Format time with leading zeros
        time_str = @sprintf("%08.6f", RP.time_s)

        # Construct filename
        filename = joinpath(
            RP.config.Output_path,
            "$(RP.config.Output_prefix)$(RP.config.Output_name)_t=$(time_str)s.jld2"
        )
    end

    # Create output directory if it doesn't exist
    mkpath(dirname(filename))

    # In a real implementation, we would save all relevant data to a JLD2 file
    # For now, just print a message
    println("Would save data to: $filename")

    return nothing
end

"""
    read_input_file(RP::RAPID{FT}, filename::String) where {FT<:AbstractFloat}

Read input data from file.

# Arguments
- `RP::RAPID{FT}`: The RAPID simulation instance to load data into
- `filename::String`: Path to the input file to read

# Returns
- `RP::RAPID{FT}`: The updated RAPID instance
"""
function read_input_file(RP::RAPID{FT}, filename::String) where {FT<:AbstractFloat}
    # Placeholder implementation - will be filled in later
    @warn "read_input_file not fully implemented yet"

    # In a real implementation, we would read data from a file
    # For now, just print a message
    println("Would read data from: $filename")

    return RP
end

# =============================================================================
# Wall geometry functions
# =============================================================================

"""
    read_device_wall_data!(RP::RAPID{FT}, wall_file_name::String=nothing) where {FT<:AbstractFloat}

Read the device wall geometry data from a file. This is a non-mutating function that returns
a new RAPID instance with the wall field updated.

# Arguments
- `RP::RAPID{FT}`: The RAPID simulation instance
- `wall_file_name::String=nothing`: Optional specific wall file to read. If not provided,
  will use "{device_Name}_First_Wall.dat" in the input path.

"""
function read_device_wall_data!(RP::RAPID{FT}, wall_file_name::String="") where {FT<:AbstractFloat}
    # Use provided file name or construct default file path
    file_path = isempty(wall_file_name) ?
        joinpath(RP.config.Input_path, "$(RP.config.device_Name)_First_Wall.dat") :
        wall_file_name

    # Read the wall data and assign to RAPID instance
    RP.wall = read_wall_data_file(file_path, FT)
end


"""
    read_wall_data_file(file_path::String, FT::Type{<:AbstractFloat}=Float64)

Read wall geometry data from a file and return a WallGeometry object.

The file should contain the number of wall points in the first line (formatted as "WALL_NUM X")
followed by the (R,Z) coordinates of each point.

# Arguments
- `file_path::String`: Path to the wall data file
- `FT::Type{<:AbstractFloat}=Float64`: Float type to use (default: Float64)

# Returns
- `WallGeometry{FT}`: Wall geometry object with R and Z vectors

# Example
```julia
wall = read_wall_data_file("path/to/wall.dat")
wall = read_wall_data_file("path/to/wall.dat", Float32) # Using Float32
```
"""
function read_wall_data_file(file_path::String, FT::Type{<:AbstractFloat}=Float64)
    @assert isfile(file_path) "Wall data file not found: $file_path"

    # Open the file for reading
    open(file_path, "r") do file
        # Read lines until we find the WALL_NUM declaration, skipping comments
        wall_num = nothing
        while !eof(file)
            line = readline(file)
            stripped_line = strip(line)

            # Skip empty lines or lines starting with #
            if isempty(stripped_line) || startswith(stripped_line, "#")
                continue
            end

            # Check if this line contains the WALL_NUM declaration
            wall_num_match = match(r"WALL_NUM\s+(\d+)", stripped_line)
            if !isnothing(wall_num_match)
                wall_num = parse(Int, wall_num_match.captures[1])
                break
            end
        end

        if isnothing(wall_num)
            error("Wall data file format error: Could not find 'WALL_NUM X' declaration")
        end

        # Preallocate arrays for wall data
        # We add one extra point to close the loop (last point = first point)
        wall_R = Vector{FT}(undef, wall_num + 1)
        wall_Z = Vector{FT}(undef, wall_num + 1)

        # Read data points, skipping comments and non-data lines
        point_count = 0
        while !eof(file) && point_count < wall_num
            line = readline(file)
            stripped_line = strip(line)

            # Skip empty lines or comment lines
            if isempty(stripped_line) || startswith(stripped_line, "#")
                continue
            end

            # Try to parse as a data point
            if !occursin(r"^\s*[\d.-]+\s+[\d.-]+", stripped_line)
                continue  # Skip non-data lines
            end

            values = try
                parse.(FT, split(stripped_line))
            catch e
                # If we can't parse this line, skip it and continue
                continue
            end

            if length(values) < 2
                continue  # Skip lines without enough values
            end

            # Increment first, so our indices start at 1
            point_count += 1
            wall_R[point_count] = values[1]
            wall_Z[point_count] = values[2]
        end

        if point_count == 0
            error("No valid data points found in wall data file")
        end

        if point_count < wall_num
            @warn "Expected $wall_num points but only read $point_count points from wall data file"
            # Resize arrays to match actual number of points read
            resize!(wall_R, point_count + 1)
            resize!(wall_Z, point_count + 1)
        end

        # Close the loop by setting the last point equal to the first
        wall_R[end] = wall_R[1]
        wall_Z[end] = wall_Z[1]

        # Create and return WallGeometry struct
        return WallGeometry{FT}(wall_R, wall_Z)
    end
end


# =============================================================================
# External field data functions
# =============================================================================

"""
    read_break_input_file(file_path::String, FT::Type{<:AbstractFloat}=Float64)

Read a single BREAK input file that contains field data.

# Arguments
- `file_path::String`: Path to the input file
- `FT::Type{<:AbstractFloat}=Float64`: Float type to use (default: Float64)

# Returns
- `NamedTuple`: A named tuple containing the field data and grid information

# Format of BREAK input files
The file should have the following format:
- Line 1: "Time= X" where X is the time in seconds
- Line 2: "R_NUM= X R_MIN= Y R_MAX= Z" (grid dimensions in R)
- Line 3: "Z_NUM= X Z_MIN= Y Z_MAX= Z" (grid dimensions in Z)
- Remaining lines: Data with 6 columns: R Z BR BZ psi LV
"""
function read_break_input_file(file_path::String, FT::Type{<:AbstractFloat}=Float64)
    # Open the file for reading
    open(file_path, "r") do file
        # Read the first three lines for metadata
        time_line = readline(file)
        r_line = readline(file)
        z_line = readline(file)

        # Extract time information
        time_match = match(r"Time=\s*([0-9.eE+-]+)", time_line)
        if isnothing(time_match)
            error("Invalid time format in file: $file_path")
        end
        time_s = parse(FT, time_match[1])

        # Extract R grid information
        r_match = match(r"R_NUM=\s*(\d+)\s+R_MIN=\s*([0-9.eE+-]+)\s+R_MAX=\s*([0-9.eE+-]+)", r_line)
        if isnothing(r_match)
            error("Invalid R grid format in file: $file_path")
        end
        r_num = parse(Int, r_match[1])
        r_min = parse(FT, r_match[2])
        r_max = parse(FT, r_match[3])

        # Extract Z grid information
        z_match = match(r"Z_NUM=\s*(\d+)\s+Z_MIN=\s*([0-9.eE+-]+)\s+Z_MAX=\s*([0-9.eE+-]+)", z_line)
        if isnothing(z_match)
            error("Invalid Z grid format in file: $file_path")
        end
        z_num = parse(Int, z_match[1])
        z_min = parse(FT, z_match[2])
        z_max = parse(FT, z_match[3])

        # Skip any non-data lines (comments, headers, etc.)
        while !eof(file)
            # Save the current position
            pos = position(file)
            line = readline(file)

            # Try to parse the line as data
            values = split(strip(line))
            if length(values) >= 6
                try
                    # Check if all values are valid numbers (if not, this will throw)
                    map(x -> parse(FT, x), values)
                    # This is a data line - rewind to the start of this line
                    seek(file, pos)
                    break
                catch
                    # Not a data line - continue searching
                    continue
                end
            end
        end

        # Read the data
        data = readdlm(file, FT)

        # Expected data shape
        expected_rows = z_num * r_num
        if size(data, 1) != expected_rows || size(data, 2) < 6
            @warn "Data in file $file_path has unexpected shape: $(size(data)). Expected $expected_rows rows with at least 6 columns."
        end

        # Reshape the data into 2D matrices
        R = transpose(reshape(data[1:expected_rows, 1], (z_num, r_num)))
        Z = transpose(reshape(data[1:expected_rows, 2], (z_num, r_num)))
        BR = transpose(reshape(data[1:expected_rows, 3], (z_num, r_num)))
        BZ = transpose(reshape(data[1:expected_rows, 4], (z_num, r_num)))
        psi = transpose(reshape(data[1:expected_rows, 5], (z_num, r_num)))
        LV = transpose(reshape(data[1:expected_rows, 6], (z_num, r_num)))

        # Return as a named tuple
        return (
            time_s = time_s,
            R_NUM = r_num,
            Z_NUM = z_num,
            R_MIN = r_min,
            R_MAX = r_max,
            Z_MIN = z_min,
            Z_MAX = z_max,
            R = R,
            Z = Z,
            BR = BR,
            BZ = BZ,
            psi = psi,
            LV = LV
        )
    end
end

"""
    read_break_input_file(RP::RAPID{FT}, file_name::String) where {FT<:AbstractFloat}

Read a BREAK input file and convert it to the format used by the RAPID simulation.
This version takes a RAPID instance for type information and returns a standardized format.

# Arguments
- `RP::RAPID{FT}`: The RAPID simulation object (used for type information)
- `file_name::String`: Path to the input file

# Returns
- `NamedTuple`: A named tuple containing the field data
"""
function read_break_input_file(RP::RAPID{FT}, file_name::String) where {FT<:AbstractFloat}
    # Use the generic function to read the file data
    data = read_break_input_file(file_name, FT)

    # Return a consistent format matching what RAPID expects
    return data
end

"""
    meshgrid(x, y)

Create a meshgrid similar to MATLAB's meshgrid function.

# Arguments
- `x`: 1D array of x coordinates
- `y`: 1D array of y coordinates

# Returns
- `Tuple{Matrix, Matrix}`: 2D matrices of x and y coordinates
"""
function meshgrid(x1D::AbstractVector, y1D::AbstractVector)
    return (repeat(x1D, 1, length(y1D)), repeat(y1D', length(x1D), 1))
end

"""
    create_new_grid_with_target_resolution(ori_data, target_r_1d, target_z_1d)

Interpolate field data to a new grid with specified resolution.

# Arguments
- `ori_data`: Original field data (as returned by `read_break_input_file`)
- `target_r_1d`: Target 1D array of R coordinates
- `target_z_1d`: Target 1D array of Z coordinates

# Returns
- `NamedTuple`: Interpolated field data on the new grid
"""
function create_new_grid_with_target_resolution(ori_data, target_r_1d, target_z_1d)
    # Create meshgrid for new coordinates
    new_r, new_z = meshgrid(target_r_1d, target_z_1d)

    # Define interpolation objects
    # Use linear interpolation for stability
    itp_br = interpolate((ori_data.R[:, 1], ori_data.Z[1, :]), ori_data.BR, Gridded(Linear()))
    itp_bz = interpolate((ori_data.R[:, 1], ori_data.Z[1, :]), ori_data.BZ, Gridded(Linear()))
    itp_psi = interpolate((ori_data.R[:, 1], ori_data.Z[1, :]), ori_data.psi, Gridded(Linear()))
    itp_lv = interpolate((ori_data.R[:, 1], ori_data.Z[1, :]), ori_data.LV, Gridded(Linear()))


    new_br = [itp_br(r, z) for r in target_r_1d, z in target_z_1d]
    # Interpolate to new grid
    # Note: Order is important - we need to pass (r, z) to match how the interpolation object was created
    new_br = [itp_br(r, z) for r in target_r_1d, z in target_z_1d]
    new_bz = [itp_bz(r, z) for r in target_r_1d, z in target_z_1d]
    new_psi = [itp_psi(r, z) for r in target_r_1d, z in target_z_1d]
    new_lv = [itp_lv(r, z) for r in target_r_1d, z in target_z_1d]

    # Return interpolated data as a named tuple
    return (
        time_s = ori_data.time_s,
        R_NUM = length(target_r_1d),
        Z_NUM = length(target_z_1d),
        R_MIN = minimum(target_r_1d),
        R_MAX = maximum(target_r_1d),
        Z_MIN = minimum(target_z_1d),
        Z_MAX = maximum(target_z_1d),
        R = new_r,
        Z = new_z,
        BR = new_br,
        BZ = new_bz,
        psi = new_psi,
        LV = new_lv
    )
end

"""
    read_external_field_time_series(dir_path::String="./";
                                    r_num::Union{Int,Nothing}=nothing,
                                    r_min::Union{Float64,Nothing}=nothing,
                                    r_max::Union{Float64,Nothing}=nothing,
                                    z_num::Union{Int,Nothing}=nothing,
                                    z_min::Union{Float64,Nothing}=nothing,
                                    z_max::Union{Float64,Nothing}=nothing,
                                    FT::Type{<:AbstractFloat}=Float64)

Read a time series of external field data from BREAK input files.

# Arguments
- `dir_path::String`: Path to the directory containing field data files (default: "./")
- `r_num::Union{Int,Nothing}`: Number of R grid points (default: use value from first file)
- `r_min::Union{Float64,Nothing}`: Minimum R value (default: use value from first file)
- `r_max::Union{Float64,Nothing}`: Maximum R value (default: use value from first file)
- `z_num::Union{Int,Nothing}`: Number of Z grid points (default: use value from first file)
- `z_min::Union{Float64,Nothing}`: Minimum Z value (default: use value from first file)
- `z_max::Union{Float64,Nothing}`: Maximum Z value (default: use value from first file)
- `FT::Type{<:AbstractFloat}=Float64`: Float type to use (default: Float64)

# Returns
- `TimeSeriesExternalField{FT}`: Time series of external field data
"""
function read_external_field_time_series(dir_path::String="./";
                                         FT::Type{T}=Float64,
                                         r_num::Union{Int,Nothing}=nothing,
                                         r_min::Union{T,Nothing}=nothing,
                                         r_max::Union{T,Nothing}=nothing,
                                         z_num::Union{Int,Nothing}=nothing,
                                         z_min::Union{T,Nothing}=nothing,
                                         z_max::Union{T,Nothing}=nothing) where {T<:AbstractFloat}

    # Ensure dir_path ends with a path separator
    if !endswith(dir_path, Base.Filesystem.path_separator)
        dir_path = dir_path * Base.Filesystem.path_separator
    end

    # Find all .dat files in the directory
    files = filter(f -> endswith(f, ".dat"), readdir(dir_path; join=true))

    if isempty(files)
        error("No .dat files found in directory: $dir_path")
    end

    # Read the time from each file and sort by time
    times = Vector{Tuple{Float64, String}}(undef, length(files))

    for (i, file) in enumerate(files)
        open(file, "r") do f
            line = readline(f)
            time_match = match(r"Time=\s*([0-9.eE+-]+)", line)
            if isnothing(time_match)
                error("Invalid time format in file: $file")
            end
            time_s = parse(Float64, time_match[1])
            times[i] = (time_s, file)
        end
    end

    sort!(times)

    # Read the first file to get field dimensions if not specified
    first_data = read_break_input_file(times[1][2], FT)

    # Use provided values or values from the first file
    nr = isnothing(r_num) ? first_data.R_NUM : r_num
    r_min_val = isnothing(r_min) ? first_data.R_MIN : r_min
    r_max_val = isnothing(r_max) ? first_data.R_MAX : r_max

    nz = isnothing(z_num) ? first_data.Z_NUM : z_num
    z_min_val = isnothing(z_min) ? first_data.Z_MIN : z_min
    z_max_val = isnothing(z_max) ? first_data.Z_MAX : z_max

    # Create target grid
    r_1d = range(r_min_val, r_max_val, length=nr)
    z_1d = range(z_min_val, z_max_val, length=nz)

    # Preallocate arrays for time series data
    n_time = length(times)
    time_series = Vector{FT}(undef, n_time)
    br_series = Array{FT, 3}(undef, nr, nz, n_time)
    bz_series = Array{FT, 3}(undef, nr, nz, n_time)
    psi_series = Array{FT, 3}(undef, nr, nz, n_time)
    lv_series = Array{FT, 3}(undef, nr, nz, n_time)

    # Process each file
    for (t, (time_s, file)) in enumerate(times)
        # Read the file
        data = read_break_input_file(file, FT)

        # Create interpolated data on the target grid
        new_data = create_new_grid_with_target_resolution(data, r_1d, z_1d)

        # Store the time and field data
        time_series[t] = new_data.time_s
        br_series[:, :, t] = new_data.BR
        bz_series[:, :, t] = new_data.BZ
        psi_series[:, :, t] = new_data.psi
        lv_series[:, :, t] = new_data.LV
    end

    # Create and return the TimeSeriesExternalField
    r_grid, z_grid = meshgrid(r_1d, z_1d)

    return TimeSeriesExternalField{FT}(
        time_series,
        br_series,
        bz_series,
        psi_series,
        lv_series,
        r_grid,
        z_grid,
        nr,
        nz,
        r_min_val,
        r_max_val,
        z_min_val,
        z_max_val
    )
end

"""
    load_external_field_data!(RP::RAPID{FT}, dir_path::String="./";
                            r_num::Union{Int,Nothing}=nothing,
                            r_min::Union{FT,Nothing}=nothing,
                            r_max::Union{FT,Nothing}=nothing,
                            z_num::Union{Int,Nothing}=nothing,
                            z_min::Union{FT,Nothing}=nothing,
                            z_max::Union{FT,Nothing}=nothing) where {FT<:AbstractFloat}

Load external field data from files and set it as the external field source for the RAPID simulation.

# Arguments
- `RP::RAPID{FT}`: The RAPID simulation instance
- `dir_path::String`: Path to the directory containing field data files (default: "./")
- `r_num`, `r_min`, `r_max`, `z_num`, `z_min`, `z_max`: Optional grid parameters
  (if not provided, use values from the first file or current grid)

# Returns
- `RP::RAPID{FT}`: The updated RAPID instance
"""
function load_external_field_data!(RP::RAPID{FT}, dir_path::String="./";
                                r_num::Union{Int,Nothing}=nothing,
                                r_min::Union{FT,Nothing}=nothing,
                                r_max::Union{FT,Nothing}=nothing,
                                z_num::Union{Int,Nothing}=nothing,
                                z_min::Union{FT,Nothing}=nothing,
                                z_max::Union{FT,Nothing}=nothing) where {FT<:AbstractFloat}

    # Read the external field data
    ext_field = read_external_field_time_series(
        dir_path;
        r_num,
        r_min,
        r_max,
        z_num,
        z_min,
        z_max,
        FT=FT
    )

    # Set the external field data in the RAPID instance
    RP.external_field = ext_field

    return RP
end