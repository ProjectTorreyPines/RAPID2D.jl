using ADIOS2

export write_to_adiosBP!,
		write_latest_snap0D!,
		write_latest_snap2D!

# Convinience dispatches
"""
    write_latest_snap0D!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Write the latest 0D snapshot data to ADIOS2 file.
"""
function write_latest_snap0D!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    snap0D = RP.diagnostics.snaps0D[RP.diagnostics.tid_0D]
    write_to_adiosBP!(RP.Afile_snap0D, snap0D)
    return RP
end

"""
    write_latest_snap2D!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Write the latest 2D snapshot data to ADIOS2 file.
"""
function write_latest_snap2D!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    snap2D = RP.diagnostics.snaps2D[RP.diagnostics.tid_2D]
    write_to_adiosBP!(RP.Afile_snap2D, snap2D)
    return RP
end


"""
    write_to_adiosBP!(Afile::AdiosFile, data; data_name::AbstractString="")

Write a data object to an open ADIOS2 file in BP format.
"""
function write_to_adiosBP!(Afile::AdiosFile, data; data_name::AbstractString="")
	data_type = typeof(data)

	# Schedule the ADIOS2 file for writing
	if data_type <: Union{AbstractString, Number, AbstractArray{<:Number}}
		if isempty(data_name)
			error("Must provide a name for non-struct type data object")
		end
		begin_step(Afile.engine)
		adios_put!(Afile, data_name, data)
		end_step(Afile.engine)
	elseif isstructtype(data_type)
		if data_type <: AbstractArray
			for i in eachindex(data)
				begin_step(Afile.engine)
				_adios_put_recursive!(Afile, data[i], data_name)
				end_step(Afile.engine)
			end
		else
			begin_step(Afile.engine)
			_adios_put_recursive!(Afile, data, data_name)
			end_step(Afile.engine)
		end
	else
		error("Type: $(typeof(data)) is not supported for ADIOS2 writing")
	end

	# Write the data to the ADIOS2 file
	adios_perform_puts!(Afile)
	return Afile
end

"""
    write_to_adiosBP!(fileName::AbstractString, data; data_name::AbstractString="")

Write a data object to a new ADIOS2 BP file, creating the file with the given filename.

This is a convenience function that creates a new ADIOS2 file, writes the data object,
and properly closes the file. Supports all data types: primitives (Number, String, Array),
structs, and arrays of structs.

# Arguments
- `fileName::AbstractString`: Output filename (must end with '.bp')
- `data`: Data object to write (Number, String, Array, or Struct)
- `data_name::AbstractString=""`: Variable name (required for primitive types, optional prefix for structs)

# Requirements
- Filename must end with '.bp' extension
- Filename must not already exist as a file or directory
- For primitive types, `data_name` must be provided


# Examples
```julia
# Write primitive data to new file
write_to_adiosBP!("output/temperature.bp", 298.15; data_name="temp_K")
write_to_adiosBP!("output/metadata.bp", "v2.1.0"; data_name="version")
write_to_adiosBP!("output/results.bp", results_array; data_name="simulation_data")

# Write struct data to new file
write_to_adiosBP!("output/grid_data.bp", grid)
write_to_adiosBP!("output/mesh_data.bp", grid; data_name="computational_mesh")

# Write time series data
write_to_adiosBP!("output/snapshots.bp", snapshot_array; data_name="time_series")
```
"""
function write_to_adiosBP!(fileName::AbstractString, data; data_name::AbstractString="")
	@assert !isempty(fileName) "File name cannot be empty"
	@assert endswith(fileName, ".bp") "File name must end with '.bp'"
	@assert !isfile(fileName) || !isfolder(fileName) "File already exists: $fileName"

	# Create new ADIOS2 file handle (overwriting if exists)
	Afile = adios_open_serial(fileName, mode_write)
	write_to_adiosBP!(Afile, data; data_name)
	close(Afile)
end


"""
    _adios_put_recursive!(Afile::AdiosFile, data, data_name::AbstractString="")

Recursively traverse a struct and write all Number and AbstractArray fields to ADIOS2 file.
Nested structs are represented with "/" path separators.

# Arguments
- `Afile::AdiosFile`: ADIOS2 file handle
- `data`: Object to traverse (can be any struct)
- `data_name::AbstractString=""`: Current path data_name for nested objects

# Examples
```julia
# For G.nodes.state, this would create variable names like:
# "nodes/state" if G is passed with data_name=""
# "grid/nodes/state" if G is passed with data_name="grid"
```
"""
function _adios_put_recursive!(Afile::AdiosFile, data, data_name::AbstractString="")
    obj_type = typeof(data)

    # Skip if this is a primitive type that we can directly write
    if obj_type <: Union{AbstractString, Number, AbstractArray{<:Number}}
        if !isempty(data_name)
            adios_put!(Afile, data_name, data)
        end
        return
    end

    # Skip if this is not a struct (e.g., functions, modules, etc.)
    if !isstructtype(obj_type)
        return
    end

    # Traverse all fields of the struct
    for fname in fieldnames(obj_type)
        field_value = getfield(data, fname)
        field_type = typeof(field_value)

        # Construct the new path
        new_path = isempty(data_name) ? string(fname) : data_name * "/" * string(fname)

        if field_type <: Union{AbstractString, Number, AbstractArray{<:Number}}
            # Direct write for primitive types and arrays
            adios_put!(Afile, new_path, field_value)
        elseif isstructtype(field_type)
            # Recursively traverse nested structs
            _adios_put_recursive!(Afile, field_value, new_path)
        end
        # Skip other types (functions, modules, etc.)
    end
end


