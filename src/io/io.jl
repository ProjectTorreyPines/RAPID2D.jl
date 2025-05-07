"""
I/O module for RAPID2D.

Contains functions for file input/output, including:
- Reading input data
- Saving simulation results
- Data visualization
"""

# Export public functions
export save_snapshot,
       write_output_file,
       read_input_file,
       update_snapshot!,
       save_snapshot2D,
       read_device_wall_data,
       read_wall_data_file

"""
    save_snapshot(RP::RAPID{FT}, snapshot_type::Symbol) where {FT<:AbstractFloat}

Save a snapshot of the current simulation state.
"""
function save_snapshot(RP::RAPID{FT}, snapshot_type::Symbol) where {FT<:AbstractFloat}
    if snapshot_type == :snap1D
        update_snapshot1D!(RP)
    elseif snapshot_type == :snap2D
        update_snapshot2D!(RP)
    else
        @warn "Unknown snapshot type: $snapshot_type"
    end

    return nothing
end

"""
    write_output_file(RP::RAPID{FT}, filename::String=nothing) where {FT<:AbstractFloat}

Write simulation results to file.
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
"""
function read_input_file(RP::RAPID{FT}, filename::String) where {FT<:AbstractFloat}
    # Placeholder implementation - will be filled in later
    @warn "read_input_file not fully implemented yet"

    # In a real implementation, we would read data from a file
    # For now, just print a message
    println("Would read data from: $filename")

    return RP
end

"""
    update_snapshot1D!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Update 1D diagnostic snapshots.
"""
function update_snapshot1D!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Get current index
    idx = RP.diagnostics.snap1D[:idx]

    # Check if we've reached the end of the array
    if idx > length(RP.diagnostics.snap1D[:time_s])
        # Expand arrays
        for key in keys(RP.diagnostics.snap1D)
            if key != :idx
                RP.diagnostics.snap1D[key] = [RP.diagnostics.snap1D[key]; zeros(FT, 100)]
            end
        end
    end

    # Store current time
    RP.diagnostics.snap1D[:time_s][idx] = RP.time_s

    # Calculate and store 1D diagnostics
    # Average electron density
    RP.diagnostics.snap1D[:ne_avg][idx] = sum(RP.plasma.ne .* RP.G.inVol2D) / RP.device_inVolume

    # Average electron energy
    avg_eErg_eV = sum(1.5 * RP.plasma.Te_eV .* RP.plasma.ne .* RP.G.inVol2D) /
                 sum(RP.plasma.ne .* RP.G.inVol2D)
    RP.diagnostics.snap1D[:avg_mean_eErg_eV][idx] = avg_eErg_eV

    # Average parallel electric fields
    RP.diagnostics.snap1D[:avg_Epara_ext][idx] = sum(RP.fields.E_para_ext .* RP.G.inVol2D) / RP.device_inVolume
    RP.diagnostics.snap1D[:avg_Epara_tot][idx] = sum(RP.fields.E_para_tot .* RP.G.inVol2D) / RP.device_inVolume

    # Calculate total toroidal current (simplified)
    RP.diagnostics.snap1D[:I_tor][idx] = sum(RP.plasma.Jϕ .* RP.G.inVol2D)

    # Increment index
    RP.diagnostics.snap1D[:idx] += 1

    return nothing
end

"""
    update_snapshot2D!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Update 2D diagnostic snapshots.
"""
function update_snapshot2D!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Get current index
    idx = RP.diagnostics.snap2D[:idx]

    # Check if we've reached the end of the array
    if idx > length(RP.diagnostics.snap2D[:time_s])
        # Expand time array
        RP.diagnostics.snap2D[:time_s] = [RP.diagnostics.snap2D[:time_s]; zeros(FT, 20)]

        # Initialize new 2D arrays if needed for first snapshot
        if idx == 1
            # Add key plasma quantities
            RP.diagnostics.snap2D[:ne] = Array{Matrix{FT}}(undef, 0)
            RP.diagnostics.snap2D[:Te_eV] = Array{Matrix{FT}}(undef, 0)
            RP.diagnostics.snap2D[:ue_para] = Array{Matrix{FT}}(undef, 0)
            RP.diagnostics.snap2D[:E_para_tot] = Array{Matrix{FT}}(undef, 0)
        end
    end

    # Store current time
    RP.diagnostics.snap2D[:time_s][idx] = RP.time_s

    # Store 2D field snapshots
    if !haskey(RP.diagnostics.snap2D, :ne) || length(RP.diagnostics.snap2D[:ne]) < idx
        # Add new snapshot for first time
        push!(RP.diagnostics.snap2D[:ne], copy(RP.plasma.ne))
        push!(RP.diagnostics.snap2D[:Te_eV], copy(RP.plasma.Te_eV))
        push!(RP.diagnostics.snap2D[:ue_para], copy(RP.plasma.ue_para))
        push!(RP.diagnostics.snap2D[:E_para_tot], copy(RP.fields.E_para_tot))
    else
        # Update existing snapshot
        RP.diagnostics.snap2D[:ne][idx] = copy(RP.plasma.ne)
        RP.diagnostics.snap2D[:Te_eV][idx] = copy(RP.plasma.Te_eV)
        RP.diagnostics.snap2D[:ue_para][idx] = copy(RP.plasma.ue_para)
        RP.diagnostics.snap2D[:E_para_tot][idx] = copy(RP.fields.E_para_tot)
    end

    # Increment index
    RP.diagnostics.snap2D[:idx] += 1

    return nothing
end

"""
    save_snapshot2D(RP::RAPID{FT}) where {FT<:AbstractFloat}

Save a 2D snapshot of the current simulation state and write to file if needed.
"""
function save_snapshot2D(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # First update the snapshot data
    update_snapshot2D!(RP)

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
    read_device_wall_data(RP::RAPID{FT}, wall_file_name::String=nothing) where {FT<:AbstractFloat}

Read the device wall geometry data from a file.

The function reads wall data from a file named "{device_Name}_First_Wall.dat" in the input path
unless a specific wall_file_name is provided.
The file should contain the number of wall points in the first line (formatted as "WALL_NUM X")
followed by the (R,Z) coordinates of each point.

Returns the RAPID object with the wall field updated.
"""
function read_device_wall_data(RP::RAPID{FT}, wall_file_name::String=nothing) where {FT<:AbstractFloat}
    # Use provided file name or construct default file path
    file_path = isnothing(wall_file_name) ?
        joinpath(RP.config.Input_path, "$(RP.config.device_Name)_First_Wall.dat") :
        wall_file_name

    # Read the wall data and assign to RAPID instance
    RP.wall = read_wall_data_file(file_path, FT)

    return RP
end

"""
    read_wall_data_file(file_path::String, ::Type{FT}=Float64) where {FT<:AbstractFloat}

Read wall geometry data from a file and return a WallGeometry object.

The file should contain the number of wall points in the first line (formatted as "WALL_NUM X")
followed by the (R,Z) coordinates of each point.

# Arguments
- `file_path::String`: Path to the wall data file
- `::Type{FT}=Float64`: Float type to use (default: Float64)

# Returns
- `WallGeometry{FT}`: Wall geometry object with R and Z vectors

# Example
```julia
wall = read_wall_data_file("path/to/wall.dat")
wall = read_wall_data_file("path/to/wall.dat", Float32) # Using Float32
```
"""
function read_wall_data_file(file_path::String, ::Type{FT}=Float64) where {FT<:AbstractFloat}
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