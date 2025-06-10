# Coil and CoilSystem types for electromagnetic modeling
# Ported from MATLAB c_Coils class

"""
    Coil{FT <: AbstractFloat}

Represents a single toroidal current loop, which can be either a powered coil or a passive conductor.

# Immutable Fields (geometry and electrical properties)
- `position::NamedTuple{(:r, :z), Tuple{FT, FT}}`: Position in (R, Z) coordinates
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
mutable struct Coil{FT <: AbstractFloat}
    # Immutable properties
    position::NamedTuple{(:r, :z), Tuple{FT, FT}}
    area::FT
    resistance::FT
    self_inductance::FT
    is_powered::Bool
    is_controllable::Bool
    name::String
    max_voltage::Union{FT, Nothing}
    max_current::Union{FT, Nothing}

    # Mutable state
    current::FT
    voltage_ext::Union{FT, Function}

    # Inner constructor with validation
    function Coil{FT}(position, area, resistance, self_inductance, is_powered, is_controllable, name,
                      max_voltage=nothing, max_current=nothing,
                      current=zero(FT), voltage_ext=zero(FT)) where {FT <: AbstractFloat}
        @assert area > 0 "Coil area must be positive"
        @assert resistance >= 0 "Coil resistance must be non-negative"
        @assert self_inductance >= 0 "Self-inductance must be non-negative"
        @assert position.r > 0 "R coordinate must be positive (toroidal geometry)"
        @assert !is_controllable || is_powered "Controllable coils must be powered (is_powered=true)"

        new{FT}(position, area, resistance, self_inductance, is_powered, is_controllable, name,
                max_voltage, max_current, current, voltage_ext)
    end
end

# Convenience constructor with type inference
function Coil(position::NamedTuple{(:r, :z), Tuple{FT, FT}}, area::FT, resistance::FT,
              self_inductance::FT, is_powered::Bool, is_controllable::Bool, name::String,
              max_voltage::Union{FT, Nothing}=nothing,
              max_current::Union{FT, Nothing}=nothing,
              current::FT=zero(FT), voltage_ext=zero(FT)) where {FT <: AbstractFloat}
    return Coil{FT}(position, area, resistance, self_inductance, is_powered, is_controllable, name,
                    max_voltage, max_current, current, voltage_ext)
end

# Convenience constructor for backward compatibility (is_controllable = is_powered by default)
function Coil(position::NamedTuple{(:r, :z), Tuple{FT, FT}}, area::FT, resistance::FT,
              self_inductance::FT, is_powered::Bool, name::String,
              max_voltage::Union{FT, Nothing}=nothing,
              max_current::Union{FT, Nothing}=nothing,
              current::FT=zero(FT), voltage_ext=zero(FT)) where {FT <: AbstractFloat}
    return Coil{FT}(position, area, resistance, self_inductance, is_powered, is_powered, name,
                    max_voltage, max_current, current, voltage_ext)
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
	Δt::FT
	θimp::FT
    A_circuit::Matrix{FT}
    inv_A_circuit::Matrix{FT}

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
		Δt = FT(0.0)  # Time step for solving circuit equations, to be set later
		θimp = FT(1.0) # Implicit factor for circuit equations (θimp=1.0 for implicit Euler)
        A_circuit = zeros(FT, n_total, n_total)
        inv_A_circuit = zeros(FT, n_total, n_total)

        # Initialize Green function matrices (sizes to be determined)
        Green_coils2bdy = Matrix{FT}(undef, 0, 0)
        Green_grid2coils = Matrix{FT}(undef, 0, 0)
        dGreen_dRg_grid2coils = Matrix{FT}(undef, 0, 0)
        dGreen_dZg_grid2coils = Matrix{FT}(undef, 0, 0)

        inside_domain_indices = Int[]

        new{FT}(coils, n_total, n_powered, n_controllable, powered_indices, controllable_indices, passive_indices,
                mutual_inductance, Δt, θimp, A_circuit, inv_A_circuit,
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
