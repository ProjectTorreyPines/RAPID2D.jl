# Coil and CoilSystem types for electromagnetic modeling
# Ported from MATLAB c_Coils class

# Export essential types
export Coil, CoilSystem

"""
    Coil{FT <: AbstractFloat}

Represents a single toroidal current loop, which can be either a powered coil or a passive conductor.

# Immutable Fields (geometry and electrical properties)
- `location::NamedTuple{(:r, :z), Tuple{FT, FT}}`: Position in (R, Z) coordinates
- `area::FT`: Cross-sectional area of the conductor
- `resistance::FT`: Electrical resistance
- `self_inductance::FT`: Self-inductance
- `is_powered::Bool`: Whether this coil has a voltage source (true) or is passive conductor (false)
- `is_controllable::Bool`: Whether this coil is available for feedback control (subset of powered coils)
- `name::String`: Identifier name (e.g., "PF1", "CS", "wall_segment_1")
- `max_voltage::Union{FT, Nothing}`: Maximum voltage limit for powered coils
- `max_current::Union{FT, Nothing}`: Maximum current limit

# Mutable Fields (time-evolving state)
- `current::FT`: Current flowing in the coil [A]
- `voltage_ext::Union{FT, Function}`: External applied voltage [V] (for powered coils, 0 for passive)
  Can be either a constant value or a function of time f(t) -> FT

# Note on coil classification:
- Passive coils: is_powered=false, is_controllable=false (e.g., vessel walls)
- Powered coils: is_powered=true, may or may not be controllable
- Controllable coils: is_powered=true, is_controllable=true (available for feedback control)
"""
@kwdef mutable struct Coil{FT <: AbstractFloat}
    # Immutable properties
    location::NamedTuple{(:r, :z), Tuple{FT, FT}} = (r=zero(FT), z=zero(FT))
    area::FT = zero(FT)
    resistance::FT = zero(FT)
    self_inductance::FT = zero(FT)
    is_powered::Bool = false
    is_controllable::Bool = false
    name::String = ""
    max_voltage::Union{FT, Nothing} = nothing
    max_current::Union{FT, Nothing} = nothing

    # Mutable state
    current::FT = zero(FT)
    voltage_ext::Union{FT, Function} = zero(FT)

    # Inner constructor with validation
    function Coil{FT}(location, area, resistance, self_inductance, is_powered, is_controllable, name, max_voltage=nothing, max_current=nothing, current=zero(FT), voltage_ext=zero(FT)) where {FT <: AbstractFloat}
        @assert area > 0 "Coil area must be positive"
        @assert resistance >= 0 "Coil resistance must be non-negative"
        @assert self_inductance >= 0 "Self-inductance must be non-negative"
        @assert location.r > 0 "R coordinate must be positive (toroidal geometry)"
        @assert !is_controllable || is_powered "Controllable coils must be powered (is_powered=true)"

        new{FT}(location, area, resistance, self_inductance, is_powered, is_controllable, name, max_voltage, max_current, current, voltage_ext)
    end
end

# Constructor with positional arguments
function Coil(location::NamedTuple{(:r, :z), Tuple{FT, FT}}, area::FT, resistance::FT,
              self_inductance::FT, is_powered::Bool, is_controllable::Bool, name::String,
              max_voltage::Union{FT, Nothing}=nothing,
              max_current::Union{FT, Nothing}=nothing,
              current::FT=zero(FT), voltage_ext=zero(FT)) where {FT <: AbstractFloat}
    return Coil{FT}(; location, area, resistance, self_inductance,
                    is_powered, is_controllable, name,
                    max_voltage, max_current, current, voltage_ext)
end

function Base.getproperty(coil::Coil{FT}, sym::Symbol) where {FT<:AbstractFloat}
    if hasfield(Coil, sym)
        return getfield(coil, sym)
    else
        if sym === :τ_LR
            # L/R time constant [s]
            return FT(coil.self_inductance / coil.resistance)
        else
            throw(ArgumentError("Coil has no property $sym"))
        end
    end
end

"""
Extend propertynames to include computed properties for tab completion
"""
function Base.propertynames(coil::Coil{FT}) where {FT<:AbstractFloat}
    return (fieldnames(Coil)..., :τ_LR)
end

"""
    Base.propertynames(coils::Vector{<:Coil})

Enable tab completion for coil vector properties in REPL.
Returns the field names of the Coil type for tab completion.
"""
function Base.propertynames(coils::Vector{<:Coil{<:AbstractFloat}})
    if isempty(coils)
        return ()  # Return empty tuple for empty vector
    end
    return propertynames(coils[1])  # Return field names of Coil type
end

"""
    Base.getproperty(coils::Vector{<:Coil}, sym::Symbol)

Enable convenient property access for vectors of coils.

Allows accessing coil properties directly on vectors:
- `coils.current` returns `[coil.current for coil in coils]`
- `coils.location` returns `[coil.location for coil in coils]`

Falls back to standard Vector behavior for Vector-specific fields.
"""
function Base.getproperty(coils::Vector{<:Coil{<:AbstractFloat}}, sym::Symbol)
    # First check if this is a Vector field - delegate to original behavior
    if hasfield(Vector, sym)
        return getfield(coils, sym)
    end

    # Handle empty vector case
    if isempty(coils)
        throw(BoundsError("Cannot access property of empty coil vector"))
    end

    # Check if it's a valid Coil property
    if hasproperty(coils[1], sym)
        return [getproperty(s, sym) for s in coils]
    end

    # If not a coil field, throw error
    throw(ArgumentError("Vector{Coil} has no property $sym"))
end

"""
    Base.setproperty!(coils::Vector{<:Coil}, sym::Symbol, values)

Enable convenient property setting for vectors of coils.

Allows setting coil properties directly on vectors:
- `coils.current = [100.0, 200.0, 300.0]` sets individual coil currents
- `coils.voltage_ext = 500.0` sets all coils to the same voltage
- `coils.voltage_ext = [100.0, 200.0, 300.0]` sets individual voltages

# Arguments
- `coils::Vector{<:Coil}`: Vector of coils to modify
- `sym::Symbol`: Property name to set
- `values`: Either a single value (applied to all coils) or a vector of values (one per coil)

# Examples
```julia
# Set all coil currents individually
coils.current = [1000.0, 2000.0, 3000.0]

# Set all coils to the same voltage
coils.voltage_ext = 500.0

# Set individual voltages
coils.voltage_ext = [100.0, 200.0, 300.0]
```
"""
function Base.setproperty!(coils::Vector{<:Coil{<:AbstractFloat}}, sym::Symbol, values)
    # Handle empty vector case
    if isempty(coils)
        throw(BoundsError("Cannot set property on empty coil vector"))
    end

    # Check if it's a valid mutable Coil field
    if !hasfield(typeof(coils[1]), sym)
        throw(ArgumentError("Vector{Coil} has no property $sym"))
    end

    # Check if the field is mutable (not in the immutable section)
    # The mutable fields in Coil are: current, voltage_ext
    mutable_fields = (:current, :voltage_ext)
    if sym ∉ mutable_fields
        throw(ArgumentError("Property $sym is not mutable. Only $(mutable_fields) can be set."))
    end

    # Handle setting values
    if values isa AbstractVector
        # Vector of values - must match coil count
        if length(values) != length(coils)
            throw(DimensionMismatch("Value vector length ($(length(values))) must match coil count ($(length(coils)))"))
        end

        # Set individual values
        for (i, coil) in enumerate(coils)
            setfield!(coil, sym, values[i])
        end
    else
        # Single value - apply to all coils
        for coil in coils
            setfield!(coil, sym, values)
        end
    end

    return values
end

"""
    CoilSystem{FT <: AbstractFloat}

Manages a collection of coils and their electromagnetic interactions.
Note: Individual coil currents and voltages are stored in each Coil object.

# Fields
- `coils::Vector{Coil{FT}}`: Collection of all coils (powered and passive)
- `n_total::Int`: Total number of coils
- `n_powered::Int`: Number of powered coils (which has non-zero voltage_ext)
- `n_controllable::Int`: Number of coils available for feedback control
- `powered_indices::Vector{Int}`: Indices of powered coils in the coils vector
- `controllable_indices::Vector{Int}`: Indices of controllable coils in the coils vector
- `passive_indices::Vector{Int}`: Indices of passive elements in the coils vector

## System matrices
- `mutual_inductance::Matrix{FT}`: Mutual inductance matrix between all coils [H]
- `circuit_matrix::Matrix{FT}`: Circuit matrix for powered coils (L + R*dt) [H]
- `inv_circuit_matrix::Matrix{FT}`: Inverse of circuit matrix [H^-1]

## Green function coupling matrices
- `Green_coils2bdy::Matrix{FT}`: Green function from coils to boundary points
- `Green_grid2coils::Matrix{FT}`: Green function from coils to plasma grid
- `green_from_grid::Matrix{FT}`: Green function from plasma grid to coils
- `dGreen_dRg_grid2coils::Matrix{FT}`: Derivative of Green function w.r.t. R_grid
- `dGreen_dZg_grid2coils::Matrix{FT}`: Derivative of Green function w.r.t. Z_grid

## Spatial information
- `inside_domain_indices::Vector{Int}`: Indices of coils inside the plasma domain

## Physical constants
- `μ0::FT`: Vacuum permeability [H/m]
- `cu_resistivity::FT`: Copper resistivity [Ω⋅m]
"""
mutable struct CoilSystem{FT <: AbstractFloat}
    # Coil collection
    coils::Vector{Coil{FT}}
    n_total::Int
    n_powered::Int
    n_controllable::Int
    powered_indices::Vector{Int}
    controllable_indices::Vector{Int}
    passive_indices::Vector{Int}

    # System matrices for circuit equations
    mutual_inductance::Matrix{FT}
    time_s::FT
	Δt::FT
	θimp::FT
    A_LR_circuit::Matrix{FT}
    inv_A_LR_circuit::Matrix{FT}

    # Green function coupling matrices
    Green_coils2bdy::Matrix{FT}
    Green_grid2coils::Matrix{FT}
    dGreen_dRg_grid2coils::Matrix{FT}
    dGreen_dZg_grid2coils::Matrix{FT}

    # Spatial information
    inside_domain_indices::Vector{Int}

    # Physical constants
    μ0::FT
    cu_resistivity::FT

    # Inner constructor
    function CoilSystem{FT}(coils::Vector{Coil{FT}},
                           μ0::FT = FT(1.25663706212e-6),
                           cu_resistivity::FT = FT(1.68e-8)) where {FT <: AbstractFloat}
        n_total = length(coils)

        # Separate powered, controllable, and passive coils
        powered_indices = Int[]
        controllable_indices = Int[]
        passive_indices = Int[]
        for (i, coil) in enumerate(coils)
            if coil.is_powered
                push!(powered_indices, i)
                if coil.is_controllable
                    push!(controllable_indices, i)
                end
            else
                push!(passive_indices, i)
            end
        end
        n_powered = length(powered_indices)
        n_controllable = length(controllable_indices)

        # Initialize matrices (will be computed later)
        mutual_inductance = zeros(FT, n_total, n_total)
        time_s = FT(0.0) # Simulation time, to be set later
		Δt = FT(0.0)  # Time step for solving circuit equations, to be set later
		θimp = FT(1.0) # Implicit factor for circuit equations (θimp=1.0 for implicit Euler)
        A_LR_circuit = zeros(FT, n_total, n_total)
        inv_A_LR_circuit = zeros(FT, n_total, n_total)

        # Initialize Green function matrices (sizes to be determined)
        Green_coils2bdy = Matrix{FT}(undef, 0, 0)
        Green_grid2coils = Matrix{FT}(undef, 0, 0)
        dGreen_dRg_grid2coils = Matrix{FT}(undef, 0, 0)
        dGreen_dZg_grid2coils = Matrix{FT}(undef, 0, 0)

        inside_domain_indices = Int[]

        new{FT}(coils, n_total, n_powered, n_controllable, powered_indices, controllable_indices, passive_indices,
                mutual_inductance, time_s, Δt, θimp, A_LR_circuit, inv_A_LR_circuit,
                Green_coils2bdy, Green_grid2coils,
                dGreen_dRg_grid2coils, dGreen_dZg_grid2coils,
                inside_domain_indices, μ0, cu_resistivity)
    end
end

# Convenience constructor with empty coils
function CoilSystem{FT}(μ0::FT = FT(1.25663706212e-6),
                       cu_resistivity::FT = FT(1.68e-8)) where {FT <: AbstractFloat}
    return CoilSystem{FT}(Coil{FT}[], μ0, cu_resistivity)
end

# Type inference constructor
function CoilSystem(coils::Vector{Coil{FT}},
                   μ0::FT = FT(1.25663706212e-6),
                   cu_resistivity::FT = FT(1.68e-8)) where {FT <: AbstractFloat}
    return CoilSystem{FT}(coils, μ0, cu_resistivity)
end
