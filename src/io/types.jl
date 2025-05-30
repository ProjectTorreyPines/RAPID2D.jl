"""
    AdiosFileWrapper

A wrapper around ADIOS2.AdiosFile that ensures proper cleanup through finalizers.
This wrapper solves the Julia GC ordering issue where AdiosFile objects are
finalized before the parent RAPID object finalizer runs.

The wrapper ensures that:
1. Each AdiosFile has its own finalizer that runs when the wrapper is collected
2. The finalizer safely closes the ADIOS file before the underlying AdiosFile is destroyed
3. Manual close operations are tracked to prevent double-closing

# Fields
- `file::ADIOS2.AdiosFile`: The underlying ADIOS file handle
- `is_closed::Ref{Bool}`: Flag to track if the file has been manually closed
"""
mutable struct AdiosFileWrapper
    file::ADIOS2.AdiosFile
    is_closed::Ref{Bool}

    function AdiosFileWrapper(adios_file::ADIOS2.AdiosFile)
        wrapper = new(adios_file, Ref(false))
        # Register finalizer on the wrapper to ensure proper cleanup
        finalizer(close_wrapper!, wrapper)
        return wrapper
    end
end

"""
    close_wrapper!(wrapper::AdiosFileWrapper)

Finalizer function for AdiosFileWrapper that safely closes the underlying ADIOS file.
This function is automatically called by Julia's garbage collector when the wrapper
becomes unreachable.
"""
function close_wrapper!(wrapper::AdiosFileWrapper)
    try
        if !wrapper.is_closed[]
            close(wrapper.file)
            wrapper.is_closed[] = true
        end
    catch e
        @info "AdiosFileWrapper finalizer: error during cleanup `$(name(wrapper.file.engine))`: $e"
    end
end

"""
    Base.close(wrapper::AdiosFileWrapper)

Manually close the wrapped ADIOS file. This allows for explicit resource management
while still providing automatic cleanup through the finalizer.
"""
function Base.close(wrapper::AdiosFileWrapper)
    if !wrapper.is_closed[]
        close(wrapper.file)
        wrapper.is_closed[] = true
        @info "Manually closed wrapped ADIOS file"
    end
    return wrapper
end

# Forward common methods to the underlying AdiosFile
Base.getproperty(wrapper::AdiosFileWrapper, name::Symbol) = begin
    if name in (:file, :is_closed)
        return getfield(wrapper, name)
    else
        return getproperty(getfield(wrapper, :file), name)
    end
end

Base.setproperty!(wrapper::AdiosFileWrapper, name::Symbol, value) = begin
    if name in (:file, :is_closed)
        return setfield!(wrapper, name, value)
    else
        return setproperty!(getfield(wrapper, :file), name, value)
    end
end
