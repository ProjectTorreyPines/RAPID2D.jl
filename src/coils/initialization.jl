# Initialization functions for CoilSystem
# Ported from MATLAB c_Coils.Initialize_coils method

using LinearAlgebra

"""
    initialize_four_wall_system!(coil_system::CoilSystem{FT}, n_total::Int) where FT<:AbstractFloat

Initialize a four-wall vessel configuration with wall segments.
This is ported from the MATLAB "4 walls" configuration.

# Arguments
- `coil_system`: CoilSystem to populate
- `n_total`: Total number of wall segments (must be divisible by 4)
"""
function initialize_four_wall_system!(coil_system::CoilSystem{FT}, n_total::Int) where FT<:AbstractFloat
    if n_total % 4 != 0
        error("n_total must be divisible by 4 for four-wall configuration")
    end

    quarter_n = n_total ÷ 4

    # Clear existing coils
    empty!(coil_system.coils)
    coil_system.n_total = 0
    coil_system.n_powered = 0
    coil_system.n_controllable = 0
    empty!(coil_system.powered_indices)
    empty!(coil_system.controllable_indices)
    empty!(coil_system.passive_indices)

    # Right wall (outer radial)
    z_positions = range(-1, 1, length=quarter_n)
    for (i, z) in enumerate(z_positions)
        r = FT(2.01)
        if i < quarter_n
            dl = z_positions[2] - z_positions[1]
        else
            dl = z_positions[end] - z_positions[end-1]
        end
        area = dl * dl
        name = "wall_right_$i"

        coil = create_coil_from_parameters(r, FT(z), area, name, false,
                                         coil_system.μ0, coil_system.cu_resistivity)
        add_coil!(coil_system, coil)
    end

    # Top wall
    r_positions = range(1, 2, length=quarter_n)
    for (i, r) in enumerate(r_positions)
        z = FT(1.01)
        if i < quarter_n
            dl = r_positions[2] - r_positions[1]
        else
            dl = r_positions[end] - r_positions[end-1]
        end
        area = dl * dl
        name = "wall_top_$i"

        coil = create_coil_from_parameters(FT(r), z, area, name, false,
                                         coil_system.μ0, coil_system.cu_resistivity)
        add_coil!(coil_system, coil)
    end

    # Left wall (inner radial)
    z_positions = range(-1, 1, length=quarter_n)
    for (i, z) in enumerate(z_positions)
        r = FT(0.99)
        if i < quarter_n
            dl = z_positions[2] - z_positions[1]
        else
            dl = z_positions[end] - z_positions[end-1]
        end
        area = dl * dl
        name = "wall_left_$i"

        coil = create_coil_from_parameters(r, FT(z), area, name, false,
                                         coil_system.μ0, coil_system.cu_resistivity)
        add_coil!(coil_system, coil)
    end

    # Bottom wall
    r_positions = range(1, 2, length=quarter_n)
    for (i, r) in enumerate(r_positions)
        z = FT(-1.01)
        if i < quarter_n
            dl = r_positions[2] - r_positions[1]
        else
            dl = r_positions[end] - r_positions[end-1]
        end
        area = dl * dl
        name = "wall_bottom_$i"

        coil = create_coil_from_parameters(FT(r), z, area, name, false,
                                         coil_system.μ0, coil_system.cu_resistivity)
        add_coil!(coil_system, coil)
    end

    return nothing
end

"""
    initialize_single_wall_system!(coil_system::CoilSystem{FT}, n_total::Int) where FT<:AbstractFloat

Initialize a single outer radial wall configuration.
This is ported from the commented "out radial wall only" configuration in MATLAB.
"""
function initialize_single_wall_system!(coil_system::CoilSystem{FT}, n_total::Int) where FT<:AbstractFloat
    # Clear existing coils
    empty!(coil_system.coils)
    coil_system.n_total = 0
    coil_system.n_powered = 0
    coil_system.n_controllable = 0
    empty!(coil_system.powered_indices)
    empty!(coil_system.controllable_indices)
    empty!(coil_system.passive_indices)

    # Outer radial wall only
    z_positions = range(-1, 1, length=n_total)
    for (i, z) in enumerate(z_positions)
        r = FT(2.01)
        if i < n_total
            dl = z_positions[2] - z_positions[1]
        else
            dl = z_positions[end] - z_positions[end-1]
        end
        area = dl * dl
        name = "wall_outer_$i"

        coil = create_coil_from_parameters(r, FT(z), area, name, false,
                                         coil_system.μ0, coil_system.cu_resistivity)
        add_coil!(coil_system, coil)
    end

    return nothing
end

"""
    add_control_coils!(coil_system::CoilSystem{FT}, coil_specs::Vector{<:NamedTuple}) where FT<:AbstractFloat

Add powered control coils to the system.

# Arguments
- `coil_system`: CoilSystem to add coils to
- `coil_specs`: Vector of NamedTuples with fields:
  - `r`: Major radius
  - `z`: Z position
  - `area`: Cross-sectional area
  - `name`: Coil name
  - `max_voltage` (optional): Maximum voltage
  - `max_current` (optional): Maximum current

# Example
```julia
pf_coils = [
    (r=2.5, z=0.45, area=π*0.02^2, name="PF1", max_voltage=1000.0),
    (r=2.5, z=0.0, area=π*0.02^2, name="PF2", max_voltage=1000.0),
    (r=2.5, z=-0.45, area=π*0.02^2, name="PF3", max_voltage=1000.0)
]
add_control_coils!(coil_system, pf_coils)
```
"""
function add_control_coils!(coil_system::CoilSystem{FT}, coil_specs::Vector{<:NamedTuple}) where FT<:AbstractFloat
    for spec in coil_specs
        r = FT(spec.r)
        z = FT(spec.z)
        area = FT(spec.area)
        name = String(spec.name)

        max_voltage = haskey(spec, :max_voltage) ? FT(spec.max_voltage) : nothing
        max_current = haskey(spec, :max_current) ? FT(spec.max_current) : nothing
        is_controllable = haskey(spec, :is_controllable) ? spec.is_controllable : true  # Default to controllable for control coils

        coil = create_coil_from_parameters(r, z, area, name, true,
                                         coil_system.μ0, coil_system.cu_resistivity,
                                         is_controllable=is_controllable,
                                         max_voltage=max_voltage, max_current=max_current)
        add_coil!(coil_system, coil)
    end

    return nothing
end

"""
    initialize_example_tokamak_coils!(coil_system::CoilSystem{FT}) where FT<:AbstractFloat

Initialize an example tokamak coil configuration with both control coils and vessel walls.
This creates a realistic setup for testing.
"""
function initialize_example_tokamak_coils!(coil_system::CoilSystem{FT}) where FT<:AbstractFloat
    # First add vessel walls (4-wall configuration with 40 segments total)
    initialize_four_wall_system!(coil_system, 40)

    # Add some example control coils
    control_coils = [
        (r=2.5, z=0.45, area=π*0.02^2, name="PF1", max_voltage=1000.0, max_current=50000.0),
        (r=2.5, z=0.0, area=π*0.02^2, name="CS", max_voltage=2000.0, max_current=100000.0),
        (r=2.5, z=-0.45, area=π*0.02^2, name="PF2", max_voltage=1000.0, max_current=50000.0),
        (r=1.2, z=0.0, area=π*0.015^2, name="OH", max_voltage=500.0, max_current=25000.0)
    ]

    add_control_coils!(coil_system, control_coils)

    @info "Initialized tokamak coil system with $(coil_system.n_total) total coils ($(coil_system.n_powered) powered, $(length(coil_system.passive_indices)) passive)"

    return nothing
end
