# Struct-based Diagnostics Architecture for RAPID2D.jl
# Using @kwdef for automatic initialization with dimension-based array creation

"""
0D diagnostic snapshot
Contains volume-averaged quantities
"""
@kwdef mutable struct Snapshot0D{FT<:AbstractFloat}
    # Step and time of the snapshot
    step::Int = 0
    time_s::FT = zero(FT)

    dt::FT = zero(FT)

    # Basic electron quantities
    ne::FT = zero(FT)              # Average electron density
    ne_max::FT = zero(FT)          # Maximum electron density
    ue_para::FT = zero(FT)         # Average electron parallel velocity
    Te_eV::FT = zero(FT)           # Average electron temperature
    𝒲e_eV::FT = zero(FT)          # Average electron energy (work)

    # Ion quantities
    ni::FT = zero(FT)              # Average ion density
    ni_max::FT = zero(FT)          # Maximum ion density
    ui_para::FT = zero(FT)         # Average ion parallel velocity
    Ti_eV::FT = zero(FT)           # Average ion temperature
    𝒲i_eV::FT = zero(FT)          # Average ion energy (work)

    I_tor::FT = zero(FT)           # Toroidal current

    # Electric fields
    Epara_tot::FT = zero(FT)       # Total parallel E-field
    Epara_ext::FT = zero(FT)       # External parallel E-field
    Epara_self_ES::FT = zero(FT)   # Self-consistent electrostatic E-field
    Epara_self_EM::FT = zero(FT)   # Self-consistent electromagnetic E-field

    # Transport quantities
    abs_ue_para_RZ::FT = zero(FT)     # Average parallel velocity
    D_RZ::FT = zero(FT)         # Average diffusion coefficient

    # Neutral gas
    n_H2_gas::FT = zero(FT)     # Average H2 density
    n_H2_gas_min::FT = zero(FT)     # Minimum H2 density

    # Collision frequencies
    ν_en_iz::FT = zero(FT)            # Average ionization frequency [1/s]
    ν_en_mom::FT = zero(FT)           # Average momentum transfer frequency [1/s]
    ν_en_Hα::FT = zero(FT)            # Hα emission frequency [1/s]
    ν_ei::FT = zero(FT)            # Electron-ion coulomb collision frequency [1/s]


    # Electron Heating Powers
    Pe_diffu::FT = zero(FT)       # Diffusion power
    Pe_conv::FT = zero(FT)        # Convection power
    Pe_drag::FT = zero(FT)        # Drag power
    Pe_iz::FT = zero(FT)          # Ionization power
    Pe_exc::FT = zero(FT)         # Excitation power
    Pe_dilution::FT = zero(FT)    # Dilution power
    Pe_equi::FT = zero(FT)        # Equilibration power
    Pe_heat::FT = zero(FT)        # Heating power
    Pe_tot::FT = zero(FT)         # Total electron power

    # Ion heating powers
    Pi_atomic::FT = zero(FT)      # Atomic processes power
    Pi_equi::FT = zero(FT)        # Equilibration power
    Pi_tot::FT = zero(FT)         # Total ion power

    # Source/loss tracking
    Ne_src_rate::FT = zero(FT)      # Electron source rate
    Ne_loss_rate::FT = zero(FT)     # Electron loss rate
    eGrowth_rate::FT = zero(FT)     # Electron growth rate
    eLoss_rate::FT = zero(FT)       # Electron loss rate

    Ni_src_rate::FT = zero(FT)      # Ion source rate
    Ni_loss_rate::FT = zero(FT)     # Ion loss rate

    # Plasma center tracking
    ne_cen_R::FT = zero(FT)         # Electron density center R
    ne_cen_Z::FT = zero(FT)         # Electron density center Z
    J_cen_R::FT = zero(FT)          # Current center R
    J_cen_Z::FT = zero(FT)          # Current center Z

    # CFL conditions (for adaptive timestepping)
    CFL::Dict{Symbol, FT} = Dict{Symbol, FT}() # CFL terms

    # Control system (optional)
    I_coils::Union{Nothing, Matrix{FT}} = nothing    # Coil currents (N_coils × time)
    pidFac::Union{Nothing, Vector{FT}} = nothing     # PID control factor
    BR_ctrl::Union{Nothing, Vector{FT}} = nothing # Control field BR
    BZ_ctrl::Union{Nothing, Vector{FT}} = nothing # Control field BZ

    # Growth rates (alternative calculation)
    growth_rate2::FT = zero(FT)     # Alternative growth rate
    loss_rate2::FT = zero(FT)       # Alternative loss rate
end

"""
2D diagnostic snapshots
Contains spatial distributions at specific time points
All 3D array fields are automatically sized based on dim_R, dim_Z and dim_tt
"""
@kwdef mutable struct Snapshot2D{FT<:AbstractFloat}
    # Dimension parameters (must be first)
    dims_RZ::Tuple{Int, Int} # (NR, NZ)

    # Metadata
    step::Int = 0
    time_s::FT = zero(FT)

    dt::FT = zero(FT)

    # Basic plasma quantities
    ne::Matrix{FT} = zeros(FT, dims_RZ)             # Electron density (NR, NZ, time)

    # Transport coefficients
    Dpara::Matrix{FT} = zeros(FT, dims_RZ)          # Parallel diffusion
    D_pol::Matrix{FT} = zeros(FT, dims_RZ)          # Poloidal diffusion
    ue_para::Matrix{FT} = zeros(FT, dims_RZ)        # Parallel electron velocity
    u_pol::Matrix{FT} = zeros(FT, dims_RZ)          # Poloidal velocity magnitude

    # Electron properties
    Te_eV::Matrix{FT} = zeros(FT, dims_RZ)          # Electron temperature
    𝒲e_eV::Matrix{FT} = zeros(FT, dims_RZ)        # Mean electron energy
    ueR::Matrix{FT} = zeros(FT, dims_RZ)            # Electron velocity R component
    ueϕ::Matrix{FT} = zeros(FT, dims_RZ)            # Electron velocity ϕ component
    ueZ::Matrix{FT} = zeros(FT, dims_RZ)            # Electron velocity Z component

    # Source/loss rates (2D)
    Ne_src_rate::Matrix{FT} = zeros(FT, dims_RZ)    # Electron source rate
    Ne_loss_rate::Matrix{FT} = zeros(FT, dims_RZ)   # Electron loss rate

    # Magnetic field
    BR::Matrix{FT} = zeros(FT, dims_RZ)             # Radial magnetic field
    BZ::Matrix{FT} = zeros(FT, dims_RZ)             # Vertical magnetic field
    B_pol::Matrix{FT} = zeros(FT, dims_RZ)          # Poloidal magnetic field magnitude
    BR_self::Matrix{FT} = zeros(FT, dims_RZ)        # Self-consistent BR
    BZ_self::Matrix{FT} = zeros(FT, dims_RZ)        # Self-consistent BZ

    # Electric field
    E_para_tot::Matrix{FT} = zeros(FT, dims_RZ)     # Total parallel electric field
    E_para_ext::Matrix{FT} = zeros(FT, dims_RZ)     # External parallel electric field
    mean_ExB_pol::Matrix{FT} = zeros(FT, dims_RZ)   # ExB drift magnitude
    Epol_self::Matrix{FT} = zeros(FT, dims_RZ)      # Self-consistent poloidal E-field
    Eϕ_self::Matrix{FT} = zeros(FT, dims_RZ)        # Self-consistent toroidal E-field

    # Current density
    Jϕ::Matrix{FT} = zeros(FT, dims_RZ)             # Toroidal current density
    J_para::Matrix{FT} = zeros(FT, dims_RZ)         # Parallel current

    # Magnetic flux
    ψ_ext::Matrix{FT} = zeros(FT, dims_RZ)        # External poloidal flux
    ψ_self::Matrix{FT} = zeros(FT, dims_RZ)       # Self-consistent poloidal flux

    # Ion properties
    ni::Matrix{FT} = zeros(FT, dims_RZ)             # Ion density
    ui_para::Matrix{FT} = zeros(FT, dims_RZ)        # Parallel ion velocity
    uiR::Matrix{FT} = zeros(FT, dims_RZ)            # Ion velocity R component
    uiϕ::Matrix{FT} = zeros(FT, dims_RZ)            # Ion velocity ϕ component
    uiZ::Matrix{FT} = zeros(FT, dims_RZ)            # Ion velocity Z component
    Ti_eV::Matrix{FT} = zeros(FT, dims_RZ)          # Ion temperature
    𝒲i_eV::Matrix{FT} = zeros(FT, dims_RZ)         # Mean ion energy
    Ni_src_rate::Matrix{FT} = zeros(FT, dims_RZ)    # Ion source rate
    Ni_loss_rate::Matrix{FT} = zeros(FT, dims_RZ)   # Ion loss rate

    # MHD-like accelerations
    mean_aR_by_JxB::Matrix{FT} = zeros(FT, dims_RZ) # JxB acceleration R component
    mean_aZ_by_JxB::Matrix{FT} = zeros(FT, dims_RZ) # JxB acceleration Z component

    # Physics parameters
    lnΛ::Matrix{FT} = zeros(FT, dims_RZ)            # Coulomb logarithm
    L_mixing::Matrix{FT} = zeros(FT, dims_RZ)       # Mixing length
    nc_para::Matrix{FT} = zeros(FT, dims_RZ)        # Parallel critical density
    nc_perp::Matrix{FT} = zeros(FT, dims_RZ)        # Perpendicular critical density
    γ_shape_fac::Matrix{FT} = zeros(FT, dims_RZ)    # Gamma coefficient

    # Neutral gas
    n_H2_gas::Matrix{FT} = zeros(FT, dims_RZ)       # H2 neutral density

    # Collision frequencies
    ν_en_iz::Matrix{FT} = zeros(FT, dims_RZ)            # Average ionization frequency [1/s]
    ν_en_mom::Matrix{FT} = zeros(FT, dims_RZ)           # Average momentum transfer frequency [1/s]
    ν_en_Hα::Matrix{FT} = zeros(FT, dims_RZ)            # Hα emission frequency [1/s]
    ν_ei::Matrix{FT} = zeros(FT, dims_RZ)            # Electron-ion coulomb collision frequency [1/s]


    # Electron Heating Powers
    Pe_diffu::Matrix{FT} = zeros(FT, dims_RZ)       # Diffusion power
    Pe_conv::Matrix{FT} = zeros(FT, dims_RZ)        # Convection power
    Pe_drag::Matrix{FT} = zeros(FT, dims_RZ)        # Drag power
    Pe_iz::Matrix{FT} = zeros(FT, dims_RZ)          # Ionization power
    Pe_exc::Matrix{FT} = zeros(FT, dims_RZ)         # Excitation power
    Pe_dilution::Matrix{FT} = zeros(FT, dims_RZ)    # Dilution power
    Pe_equi::Matrix{FT} = zeros(FT, dims_RZ)        # Equilibration power
    Pe_heat::Matrix{FT} = zeros(FT, dims_RZ)        # Heating power
    Pe_tot::Matrix{FT} = zeros(FT, dims_RZ)          # Total electron power

    # Ion heating powers
    Pi_atomic::Matrix{FT} = zeros(FT, dims_RZ)      # Atomic processes power
    Pi_equi::Matrix{FT} = zeros(FT, dims_RZ)        # Equilibration power
    Pi_tot::Matrix{FT} = zeros(FT, dims_RZ)          # Total ion power

    # Control fields (optional)
    BR_ctrl::Union{Nothing, Matrix{FT}} = nothing # Control magnetic field BR
    BZ_ctrl::Union{Nothing, Matrix{FT}} = nothing # Control magnetic field BZ
end

"""
Source and Loss Tracker
Tracks cumulative sources and losses of particles and energy
"""
@kwdef mutable struct SrcLossTracker{FT<:AbstractFloat}
    # Dimension parameters (must be first)
    dims_RZ::Tuple{Int, Int}    # Number of R and Z grid points

    # 0D (volume-integrated) tracking
    cum0D_Ne_src::FT = zero(FT)             # Cumulative electron source
    cum0D_Ne_loss::FT = zero(FT)            # Cumulative electron loss
    cum0D_Ni_src::FT = zero(FT)             # Cumulative ion source
    cum0D_Ni_loss::FT = zero(FT)            # Cumulative ion loss

    # 2D (spatially-resolved) tracking
    cum2D_Ne_src::Matrix{FT} = zeros(FT, dims_RZ)     # Cumulative electron source (R,Z)
    cum2D_Ne_loss::Matrix{FT} = zeros(FT, dims_RZ)    # Cumulative electron loss (R,Z)
    cum2D_Ni_src::Matrix{FT} = zeros(FT, dims_RZ)     # Cumulative ion source (R,Z)
    cum2D_Ni_loss::Matrix{FT} = zeros(FT, dims_RZ)    # Cumulative ion loss (R,Z)

    # Energy tracking (can be added/extended as needed)
    # cum0D_Energy_src::FT = zero(FT)
    # cum0D_Energy_loss::FT = zero(FT)
    # cum2D_Energy_src::Matrix{FT} = zeros(FT, dims_rz...)
    # cum2D_Energy_loss::Matrix{FT} = zeros(FT, dims_rz...)
end

"""
Main Diagnostics container
Pure struct-based approach using @kwdef for automatic initialization
No legacy Dictionary compatibility layer
"""
@kwdef mutable struct Diagnostics{FT<:AbstractFloat}
    dims_RZ::Tuple{Int, Int} # Dimensions for R and Z (NR, NZ)

    # 0D time series snapshots
    tid_0D::Int = 0 # Last recorded time index for snaps0D
    snaps0D::Vector{Snapshot0D{FT}} = Snapshot0D{FT}[]

    # 2D spatial snapshots
    tid_2D::Int = 0 # Last recorded time index for snaps2D
    snaps2D::Vector{Snapshot2D{FT}} = Snapshot2D{FT}[]

     # tracking number of particles (source/loss)
    Ntracker::SrcLossTracker{FT} = SrcLossTracker{FT}(;dims_RZ)
end

function Diagnostics{FT}(dim_R::Int, dim_Z::Int) where FT<:AbstractFloat
    return Diagnostics{FT}(; dims_RZ = (dim_R, dim_Z))
end

function Diagnostics{FT}(dim_R::Int, dim_Z::Int, dim_tt_0D::Int, dim_tt_2D::Int) where FT<:AbstractFloat
    dims_RZ = (dim_R, dim_Z)
    # Create empty snapshots with given sizes to preallocate memory
    snaps0D = [Snapshot0D{FT}() for _ in 1:dim_tt_0D]
    snaps2D = [Snapshot2D{FT}( dims_RZ = (dim_R, dim_Z) ) for _ in 1:dim_tt_2D]

    return Diagnostics{FT}(;
        dims_RZ,
        tid_0D = 0, # tid_0D
        snaps0D,
        tid_2D = 0, # tid_2D
        snaps2D,
        Ntracker =SrcLossTracker{FT}(;dims_RZ)
    )
end

# Export everything for external use
export Diagnostics, Snapshot0D, Snapshot2D, SrcLossTracker

# Equality operators for testing ADIOS BP reading routines
"""
    ==(snap1::Snapshot0D, snap2::Snapshot0D) -> Bool

Compare two Snapshot0D objects for equality. All fields must match exactly.
This is used for testing ADIOS BP reading routines to ensure they produce
identical results to existing methods.

Note: This uses == semantics where NaN != NaN. For NaN-safe comparison, use isequal().
"""
function Base.:(==)(snap1::Snapshot0D{<:AbstractFloat}, snap2::Snapshot0D{<:AbstractFloat})
    # Compare all scalar fields
    for field in fieldnames(Snapshot0D)
        val1 = getfield(snap1, field)
        val2 = getfield(snap2, field)

        # Special handling for different field types
        if val1 isa Dict && val2 isa Dict
            # Compare dictionaries (like CFL)
            if val1 != val2
                return false
            end
        elseif val1 isa Union{Nothing, AbstractArray} && val2 isa Union{Nothing, AbstractArray}
            # Compare optional arrays (like I_coils, pidFac, etc.)
            if (val1 === nothing) != (val2 === nothing)
                return false
            elseif val1 !== nothing && val2 !== nothing
                if val1 != val2
                    return false
                end
            end
        else
            # Compare scalar values - standard == semantics (NaN != NaN)
            if val1 != val2
                return false
            end
        end
    end
    return true
end

"""
    isequal(snap1::Snapshot0D, snap2::Snapshot0D) -> Bool

Check if two Snapshot0D objects are equal, with special handling for NaN values.
Unlike ==, this function treats NaN values as equal to other NaN values.
This is recommended for testing ADIOS BP reading routines where NaN values
should be considered identical.
"""
function Base.isequal(snap1::Snapshot0D{<:AbstractFloat}, snap2::Snapshot0D{<:AbstractFloat})
    # Compare all scalar fields
    for field in fieldnames(Snapshot0D)
        val1 = getfield(snap1, field)
        val2 = getfield(snap2, field)

        # Special handling for different field types
        if val1 isa Dict && val2 isa Dict
            # Compare dictionaries with isequal semantics (NaN-safe)
            if !isequal(val1, val2)
                return false
            end
        elseif val1 isa Union{Nothing, AbstractArray} && val2 isa Union{Nothing, AbstractArray}
            # Compare optional arrays
            if (val1 === nothing) != (val2 === nothing)
                return false
            elseif val1 !== nothing && val2 !== nothing
                if !isequal(val1, val2)  # NaN-safe array comparison
                    return false
                end
            end
        else
            # Compare scalar values with isequal semantics (NaN == NaN)
            if !isequal(val1, val2)
                return false
            end
        end
    end
    return true
end

"""
    ==(snap1::Snapshot2D, snap2::Snapshot2D) -> Bool

Compare two Snapshot2D objects for equality. All fields must match exactly.
This is used for testing ADIOS BP reading routines to ensure they produce
identical results to existing methods.

Note: This uses == semantics where NaN != NaN. For NaN-safe comparison, use isequal().
"""
function Base.:(==)(snap1::Snapshot2D{<:AbstractFloat}, snap2::Snapshot2D{<:AbstractFloat})
    # First compare dimensions
    if snap1.dims_RZ != snap2.dims_RZ
        return false
    end

    # Compare all fields
    for field in fieldnames(Snapshot2D)
        val1 = getfield(snap1, field)
        val2 = getfield(snap2, field)

        # All fields are either scalars, tuples, or matrices - standard == semantics
        if val1 != val2
            return false
        end
    end
    return true
end

"""
    isequal(snap1::Snapshot2D, snap2::Snapshot2D) -> Bool

Check if two Snapshot2D objects are equal, with special handling for NaN values.
Unlike ==, this function treats NaN values as equal to other NaN values.
This is recommended for testing ADIOS BP reading routines where NaN values
should be considered identical.
"""
function Base.isequal(snap1::Snapshot2D{FT}, snap2::Snapshot2D{FT}) where {FT <: AbstractFloat}
    # First compare dimensions
    if snap1.dims_RZ != snap2.dims_RZ
        return false
    end

    # Compare all fields
    for field in fieldnames(Snapshot2D)
        val1 = getfield(snap1, field)
        val2 = getfield(snap2, field)

        # Use isequal for NaN-safe comparison
        if !isequal(val1, val2)
            return false
        end
    end
    return true
end

"""
    isapprox(snap1::Snapshot0D, snap2::Snapshot0D; kwargs...) -> Bool

Check if two Snapshot0D objects are approximately equal within floating-point tolerance.
This is useful for comparing results from different numerical methods that may have
small differences due to rounding errors.

# Arguments
- `rtol::Real=√eps()`: Relative tolerance
- `atol::Real=0`: Absolute tolerance
- `nans::Bool=false`: Whether to treat NaN values as equal (default: false)

Note: When nans=true, NaN values are considered equal. When nans=false (default),
NaN values are not considered equal to anything, including other NaN values.
"""
function Base.isapprox(snap1::Snapshot0D{<:AbstractFloat}, snap2::Snapshot0D{<:AbstractFloat}; kwargs...)
    # Compare all scalar fields
    for field in fieldnames(Snapshot0D)
        val1 = getfield(snap1, field)
        val2 = getfield(snap2, field)

        # Special handling for different field types
        if val1 isa Dict && val2 isa Dict
            # Compare dictionaries approximately
            if Set(keys(val1)) != Set(keys(val2))
                return false
            end
            for key in keys(val1)
                if !isapprox(val1[key], val2[key]; kwargs...)
                    return false
                end
            end
        elseif val1 isa Union{Nothing, AbstractArray} && val2 isa Union{Nothing, AbstractArray}
            # Compare optional arrays
            if (val1 === nothing) != (val2 === nothing)
                return false
            elseif val1 !== nothing && val2 !== nothing
                if !isapprox(val1, val2; kwargs...)
                    return false
                end
            end
        elseif val1 isa AbstractFloat && val2 isa AbstractFloat
            # Compare floating-point values approximately
            if !isapprox(val1, val2; kwargs...)
                return false
            end
        else
            # Compare other scalar values exactly (Int, Bool, etc.)
            if val1 != val2
                return false
            end
        end
    end
    return true
end

"""
    isapprox(snap1::Snapshot2D, snap2::Snapshot2D; kwargs...) -> Bool

Check if two Snapshot2D objects are approximately equal within floating-point tolerance.
This is useful for comparing results from different numerical methods that may have
small differences due to rounding errors.

# Arguments
- `rtol::Real=√eps()`: Relative tolerance
- `atol::Real=0`: Absolute tolerance
- `nans::Bool=false`: Whether to treat NaN values as equal (default: false)

Note: When nans=true, NaN values are considered equal. When nans=false (default),
NaN values are not considered equal to anything, including other NaN values.
"""
function Base.isapprox(snap1::Snapshot2D{<:AbstractFloat}, snap2::Snapshot2D{<:AbstractFloat}; kwargs...)
    # First compare dimensions
    if snap1.dims_RZ != snap2.dims_RZ
        return false
    end

    # Compare all fields
    for field in fieldnames(Snapshot2D)
        val1 = getfield(snap1, field)
        val2 = getfield(snap2, field)

        if val1 isa AbstractFloat && val2 isa AbstractFloat
            # Compare floating-point scalars approximately
            if !isapprox(val1, val2; kwargs...)
                return false
            end
        elseif val1 isa Matrix{<:AbstractFloat} && val2 isa Matrix{<:AbstractFloat}
            # Compare matrices approximately
            if !isapprox(val1, val2; kwargs...)
                return false
            end
        else
            # Compare other values exactly (Int, Tuple, etc.)
            if val1 != val2
                return false
            end
        end
    end
    return true
end

"""
    getproperty(sv::Vector{<:Snapshot0D}, sym::Symbol)

Custom property access for Vector{Snapshot0D} to enable convenient field extraction.
Allows syntax like `snaps0D.time_s` to extract all time_s values from the vector.

# Examples
```julia
times = diagnostics.snaps0D.time_s      # Extract all time values
densities = diagnostics.snaps0D.ne      # Extract all electron densities
temps = diagnostics.snaps0D.Te_eV       # Extract all temperatures
```

Note: Regular vector operations (indexing, slicing) remain unchanged.
"""
function Base.getproperty(sv::Vector{<:Snapshot0D{<:AbstractFloat}}, sym::Symbol)
    # First check if this is a Vector field - delegate to original behavior
    if hasfield(Vector, sym)
        return getfield(sv, sym)
    end

    # Handle empty vector case
    if isempty(sv)
        throw(BoundsError("Cannot access property of empty snapshot vector"))
    end

    # Check if it's a valid Snapshot0D field
    if hasfield(typeof(sv[1]), sym)
        return [getfield(s, sym) for s in sv]
    end

    # If not a snapshot field, throw error
    throw(ArgumentError("Vector{Snapshot0D} has no property $sym"))
end

"""
    Base.propertynames(sv::Vector{<:Snapshot0D})

Enable tab completion for snapshot vector properties in REPL.
Returns the field names of the Snapshot0D type for tab completion.
"""
function Base.propertynames(sv::Vector{<:Snapshot0D{<:AbstractFloat}})
    if isempty(sv)
        return ()  # Return empty tuple for empty vector
    end
    return propertynames(sv[1])  # Return field names of Snapshot0D type
end

"""
    getproperty(sv::Vector{<:Snapshot2D}, sym::Symbol)

Custom property access for Vector{Snapshot2D} to enable convenient field extraction.
Allows syntax like `snaps2D.ne` to extract all density matrices from the vector.

# Examples
```julia
ne_matrices = diagnostics.snaps2D.ne      # Extract all density matrices
temp_matrices = diagnostics.snaps2D.Te_eV # Extract all temperature matrices
```

Note: Regular vector operations (indexing, slicing) remain unchanged.
"""
function Base.getproperty(sv::Vector{<:Snapshot2D{FT}}, sym::Symbol) where {FT <: AbstractFloat}
    # First check if this is a Vector field - delegate to original behavior
    if hasfield(Vector, sym)
        return getfield(sv, sym)
    end

    # Handle empty vector case
    if isempty(sv)
        throw(BoundsError("Cannot access property of empty snapshot vector"))
    end

    # Check if it's a valid Snapshot2D field
    if hasfield(Snapshot2D, sym)
        if fieldtype(Snapshot2D, sym) <: AbstractArray
            Ntt = length(sv)
            result_shape = (sv[1].dims_RZ..., Ntt)

            D = length(result_shape)
            result = Array{FT, D}(undef, result_shape)
            if Ntt == 1
                @views result .= getfield(sv[1], sym)
            else
                # Copy data from each step into the result array
                for i in 1:Ntt
                    selectdim(result, D, i) .= getfield(sv[i], sym)
                end
            end

            return result
        else
            return [getfield(s, sym) for s in sv]
        end
    end

    # If not a snapshot field, throw error
    throw(ArgumentError("Vector{Snapshot2D} has no property $sym"))
end

"""
    Base.propertynames(sv::Vector{<:Snapshot2D})

Enable tab completion for snapshot vector properties in REPL.
Returns the field names of the Snapshot2D type for tab completion.
"""
function Base.propertynames(sv::Vector{<:Snapshot2D{<:AbstractFloat}})
    if isempty(sv)
        return ()  # Return empty tuple for empty vector
    end
    return propertynames(sv[1])  # Return field names of Snapshot2D type
end