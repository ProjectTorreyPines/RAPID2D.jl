using ADIOS2

export write_to_adiosBP!,
    write_latest_snap0D!,
    write_latest_snap2D!,
    adiosBP_to_snap0D,
    adiosBP_to_snap2D

# Convinience dispatches
"""
    archive_snapshot_files(paths...) -> Vector{String}

Move every existing path aside as `<stem>_<yyyymmdd-HHMMSS><ext>`, the stamp being the
latest last-write time among them (local time), so the files of one run stay paired.
A stamp already taken gets `_2`, `_3`, … appended. Returns the new paths.
"""
function archive_snapshot_files(paths::AbstractString...)
    existing = filter(ispath, collect(String, paths))
    isempty(existing) && return String[]
    stamp = Libc.strftime("%Y%m%d-%H%M%S", maximum(mtime, existing))
    moved = String[]
    for p in existing
        stem, ext = splitext(p)
        dst = stem * "_" * stamp * ext
        k = 1
        while ispath(dst)
            k += 1
            dst = stem * "_" * stamp * "_" * string(k) * ext
        end
        mv(p, dst)
        push!(moved, dst)
    end
    return moved
end

"""
    write_latest_snap0D!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Append the latest 0D snapshot to `RP.snap0D_path`: opens, writes one step, closes. The
file is created if it is not there (`initialize!` moves the previous run's file aside), so
the function can be called by hand any number of times. Each completed call is durable.
"""
function write_latest_snap0D!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    write_to_adiosBP!(RP.snap0D_path, RP.diagnostics.snaps0D[end]; append = true)
    return RP
end

"""
    write_latest_snap2D!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Append the latest 2D snapshot to `RP.snap2D_path`: opens, writes one step, closes. The
file is created if it is not there (`initialize!` moves the previous run's file aside), so
the function can be called by hand any number of times. Each completed call is durable.
"""
function write_latest_snap2D!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    write_to_adiosBP!(RP.snap2D_path, RP.diagnostics.snaps2D[end]; append = true)
    return RP
end


"""
    write_to_adiosBP!(Afile::AdiosFile, data; data_name::AbstractString="")

Write a data object to an open ADIOS2 file in BP format.
"""
function write_to_adiosBP!(Afile::AdiosFile, data; data_name::AbstractString = "")
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
    write_to_adiosBP!(fileName::AbstractString, data; data_name="", append=false)

Open `fileName`, write `data` and close it again: no writer outlives the call, even when
the write throws. With `append = false` (default) an existing file is overwritten; with
`append = true` the new step(s) go after the existing ones. A file that does not exist yet
is created either way. Supports primitives (Number, String, Array), structs, and arrays
of structs.

# Arguments
- `fileName::AbstractString`: Output filename (must end with '.bp')
- `data`: Data object to write (Number, String, Array, or Struct)
- `data_name::AbstractString=""`: Variable name (required for primitive types, optional prefix for structs)
- `append::Bool=false`: add to an existing file instead of overwriting it

# Requirements
- Filename must end with '.bp' extension
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

# Add one more step to an existing file
write_to_adiosBP!("output/snapshots.bp", [snapshot]; data_name="time_series", append=true)
```
"""
function write_to_adiosBP!(
        fileName::AbstractString, data;
        data_name::AbstractString = "", append::Bool = false
    )
    @assert !isempty(fileName) "File name cannot be empty"
    @assert endswith(fileName, ".bp") "File name must end with '.bp'"

    # Appending to a file that is not there yet is a first write.
    mode = (append && ispath(fileName)) ? mode_append : mode_write
    # One open–write–close per call: no writer outlives this function, even on error.
    Afile = adios_open_serial(fileName, mode)
    try
        write_to_adiosBP!(Afile, data; data_name)
    finally
        close(Afile)
    end
    return nothing
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
function _adios_put_recursive!(Afile::AdiosFile, data, data_name::AbstractString = "")
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
            if field_type <: Union{AbstractString, Number}
                adios_put!(Afile, new_path, field_value; global_value = true)
            else
                adios_put!(Afile, new_path, field_value)
            end
        elseif isstructtype(field_type)
            # Recursively traverse nested structs
            _adios_put_recursive!(Afile, field_value, new_path)
        end
        # Skip other types (functions, modules, etc.)
    end
    return
end


## ADIOS2 BP file to RAPID2D data conversion
"""
    adiosBP_to_snap0D(bpPath::AbstractString)

Ultra-optimized version using direct symbol-based field access.
This is the most efficient approach, using pre-compiled symbol paths
and direct `getfield`/`setfield!` operations with no path traversal overhead.

# Performance
- O(N_keys) preprocessing + O(dim_tt × N_keys) main loop
- Zero path parsing or object traversal in time loops
- Direct memory access via symbols

# Arguments
- `bpPath::AbstractString`: Path to the ADIOS2 BP file

# Returns
- `Vector{Snapshot0D}`: Array of 0D snapshots with optimal performance

# Example
```julia
snaps0D = adiosBP_to_snap0D("output.bp")
# Fastest possible implementation for large time series
```
"""
function adiosBP_to_snap0D(bpPath::AbstractString)
    @assert isfile(bpPath) || isdir(bpPath) "$bpPath does not exist"
    @assert endswith(bpPath, ".bp") "File must end with '.bp' extension"

    Adict = adios_load(bpPath)

    # Retrieve basic information
    FT = eltype(Adict["time_s"])
    dim_tt = length(Adict["time_s"])

    # Create snapshot array
    snaps0D = [Snapshot0D{FT}() for _ in 1:dim_tt]

    # Pre-process symbol paths once (O(N_keys) complexity)
    symbol_paths = create_symbol_path_mapping(Snapshot0D{FT}, Adict)

    # Ultra-efficient main loop using common helper function
    process_symbol_based_snapshots!(snaps0D, Adict, symbol_paths, dim_tt)

    return snaps0D
end

"""
    adiosBP_to_snap2D(bpPath::AbstractString)

Ultra-optimized version using direct symbol-based field access for 2D snapshots.
This is the most efficient approach, using pre-compiled symbol paths
and direct `getfield`/`setfield!` operations with no path traversal overhead.

# Performance
- O(N_keys) preprocessing + O(dim_tt × N_keys) main loop
- Zero path parsing or object traversal in time loops
- Direct memory access via symbols

# Arguments
- `bpPath::AbstractString`: Path to the ADIOS2 BP file

# Returns
- `Vector{Snapshot2D}`: Array of 2D snapshots with optimal performance

# Example
```julia
snaps2D = adiosBP_to_snap2D("output.bp")
# Fastest possible implementation for large time series
```
"""
function adiosBP_to_snap2D(bpPath::AbstractString)
    @assert isfile(bpPath) || isdir(bpPath) "$bpPath does not exist"
    @assert endswith(bpPath, ".bp") "File must end with '.bp' extension"

    Adict = adios_load(bpPath)

    # Retrieve basic information
    FT = eltype(Adict["time_s"])
    dim_tt = length(Adict["time_s"])
    dims_RZ = (Adict["dims_RZ/1"][1], Adict["dims_RZ/2"][1])

    # Create snapshot array
    snaps2D = [Snapshot2D{FT}(; dims_RZ) for _ in 1:dim_tt]

    # Pre-process symbol paths once (O(N_keys) complexity)
    symbol_paths = create_symbol_path_mapping(Snapshot2D{FT}, Adict)

    # Ultra-efficient main loop using common helper function
    for i in 1:dim_tt
        snaps2D[i].dims_RZ = dims_RZ
    end

    process_symbol_based_snapshots!(snaps2D, Adict, symbol_paths, dim_tt)

    return snaps2D
end
"""
    is_dict_internal_key(key::String)

Check if an ADIOS key represents Dictionary internal structure that should be skipped
during reading to avoid assignment errors with const fields.

Dictionary internal structure includes keys like:
- "*/keys", "*/vals", "*/slots" (Dictionary main internal fields)
- "*/keys/length", "*/keys/vals", etc. (Dictionary keys field internals)
- "*/vals/length", "*/vals/vals", etc. (Dictionary vals field internals)
- "*/count", "*/age", "*/idxfloor", etc. (other Dictionary internal fields)

# Arguments
- `key::String`: ADIOS key to check

# Returns
- `Bool`: true if this is a Dictionary internal key that should be skipped
"""
function is_dict_internal_key(key::String)
    # Dictionary internal structure patterns to skip
    dict_internal_patterns = [
        # Main Dictionary internal fields
        r"/keys$",
        r"/vals$",
        r"/slots$",
        r"/ndel$",
        r"/count$",
        r"/age$",
        r"/idxfloor$",
        r"/maxprobe$",
        # Nested Dictionary internal fields
        r"/keys/",
        r"/vals/",
        r"/slots/",
    ]

    for pattern in dict_internal_patterns
        if occursin(pattern, key)
            return true
        end
    end

    return false
end

"""
    create_symbol_path_mapping(obj_type, Adict::Dict)

Create a mapping from ADIOS keys to symbol paths for ultra-fast field access.
This pre-processes all available paths once and returns accessor information.

# Arguments
- `obj_type`: Type of the target object (e.g., Snapshot0D{Float64})
- `Adict::Dict`: ADIOS dictionary with available keys

# Returns
- `symbol_paths::Dict{String, Vector{Symbol}}`: Maps ADIOS keys to symbol paths

# Example
```julia
symbol_paths = create_symbol_path_mapping(Snapshot0D{Float64}, Adict)
# Returns: {"ePowers/drag" => [:ePowers, :drag], "time_s" => [:time_s], ...}
```
"""
function create_symbol_path_mapping(obj_type, Adict::Dict)
    symbol_paths = Dict{String, Vector{Symbol}}()

    for key in keys(Adict)
        # Skip special keys
        if key in ["dims_RZ", "dims_RZ/1", "dims_RZ/2"]
            continue
        end

        # Skip Dictionary internal structure keys to avoid assignment errors
        if is_dict_internal_key(key)
            # Debug: print filtered keys to understand what's being skipped
            println("Filtering Dictionary internal key: $key")
            continue
        end

        # Convert path string to symbol array
        if contains(key, "/")
            symbols = Symbol.(split(key, '/'))
        else
            symbols = [Symbol(key)]
        end

        # Validate path exists in the type
        if validate_symbol_path(obj_type, symbols)
            symbol_paths[key] = symbols
        else
            println("Invalid symbol path for key: $key -> $(symbols)")
        end
    end

    return symbol_paths
end

"""
    validate_symbol_path(obj_type, symbols::Vector{Symbol})

Validate that a symbol path exists in the given type at compile time.

# Arguments
- `obj_type`: Type to validate against
- `symbols::Vector{Symbol}`: Symbol path to validate

# Returns
- `Bool`: true if path exists, false otherwise
"""
function validate_symbol_path(obj_type, symbols::Vector{Symbol})
    current_type = obj_type

    for sym in symbols
        if hasfield(current_type, sym)
            current_type = fieldtype(current_type, sym)
        else
            return false
        end
    end

    return true
end


"""
    process_symbol_based_snapshots!(snapshots, Adict, symbol_paths, dim_tt)

Common logic for processing symbol-based snapshots for both 0D and 2D cases.
This function handles the main time loop and field assignment logic.

# Arguments
- `snapshots`: Pre-allocated array of snapshots (Snapshot0D or Snapshot2D)
- `Adict`: Dictionary containing ADIOS data
- `symbol_paths`: Pre-computed symbol path mapping
- `dim_tt`: Number of time steps

# Performance
- O(dim_tt × N_keys) complexity
- Direct memory access via symbols
- Zero path parsing overhead
"""
function process_symbol_based_snapshots!(snapshots, Adict, symbol_paths, dim_tt)
    for tt in 1:dim_tt
        snap = snapshots[tt]

        for (key, symbols) in symbol_paths
            Adata = Adict[key]
            # extracted_value = extract_time_data(Adata, tt)

            Ndim = ndims(Adata)
            if Ndim == 1
                extracted_value = Adata[tt]
            else
                extracted_value = collect(selectdim(Adata, Ndim, tt))
            end

            if extracted_value !== nothing
                try
                    # Convert to appropriate type
                    if length(symbols) == 1
                        # Direct field access: snap.field = value
                        target_type = fieldtype(typeof(snap), symbols[1])
                        setfield!(snap, symbols[1], extracted_value)
                    else
                        # Nested field access: snap.parent.field = value
                        parent = foldl(getfield, symbols[1:(end - 1)]; init = snap)
                        target_type = fieldtype(typeof(parent), symbols[end])
                        setfield!(parent, symbols[end], extracted_value)
                    end
                catch e
                    @warn("Failed to assign $(join(symbols, '.')) from key $key at time $tt: $e")
                end
            end
        end
    end
    return
end
