# Core functions for CoilSystem - initialization and basic operations
# Ported from MATLAB c_Coils methods

using LinearAlgebra

"""
    add_coil!(csys::CoilSystem{FT}, coil::Coil{FT}) where FT<:AbstractFloat

Add a new coil to the system and update internal indices.
"""
function add_coil!(csys::CoilSystem{FT}, coil::Coil{FT}) where FT<:AbstractFloat
    push!(csys.coils, coil)
    csys.n_total += 1

    if coil.is_powered
        csys.n_powered += 1
        push!(csys.powered_indices, csys.n_total)

        if coil.is_controllable
            csys.n_controllable += 1
            push!(csys.controllable_indices, csys.n_total)
        end
    else
        push!(csys.passive_indices, csys.n_total)
    end

    # Note: System matrices will need to be recomputed
    return nothing
end

"""
    get_powered_coils(csys::CoilSystem{FT}) where FT<:AbstractFloat

Return vector of powered coils only.
"""
function get_powered_coils(csys::CoilSystem{FT}) where FT<:AbstractFloat
    return csys.coils[csys.powered_indices]
end

"""
    get_passive_coils(csys::CoilSystem{FT}) where FT<:AbstractFloat

Return vector of passive coils/conductors only.
"""
function get_passive_coils(csys::CoilSystem{FT}) where FT<:AbstractFloat
    return csys.coils[csys.passive_indices]
end

"""
    get_controllable_coils(csys::CoilSystem{FT}) where FT<:AbstractFloat

Return vector of controllable coils only.
"""
function get_controllable_coils(csys::CoilSystem{FT}) where FT<:AbstractFloat
    return csys.coils[csys.controllable_indices]
end

"""
    find_coil_by_name(csys::CoilSystem, name::String)

Find coil index by name. Returns `nothing` if not found.
"""
function find_coil_by_name(csys::CoilSystem, name::String)
    for (i, coil) in enumerate(csys.coils)
        if coil.name == name
            return i
        end
    end
    return nothing
end

"""
    set_coil_voltage!(csys::CoilSystem{FT}, coil_name::String, voltage::FT) where FT<:AbstractFloat

Set external voltage for a powered coil by name.
"""
function set_coil_voltage!(csys::CoilSystem{FT}, coil_name::String, voltage::FT) where FT<:AbstractFloat
    coil_idx = find_coil_by_name(csys, coil_name)
    if isnothing(coil_idx)
        error("Coil '$coil_name' not found")
    end

    coil = csys.coils[coil_idx]
    if !coil.is_powered
        error("Coil '$coil_name' is not a powered coil")
    end

    # Directly set the voltage in the coil
    coil.voltage_ext = voltage
    return nothing
end

"""
    get_coil_voltage(csys::CoilSystem, coil_name::String)

Get external voltage for a powered coil by name.
For time-dependent voltages, use get_coil_voltage_at_time instead.
"""
function get_coil_voltage(csys::CoilSystem, coil_name::String)
    coil_idx = find_coil_by_name(csys, coil_name)
    if isnothing(coil_idx)
        error("Coil '$coil_name' not found")
    end

    coil = csys.coils[coil_idx]
    if !coil.is_powered
        return zero(Float64)  # Return appropriate zero type
    end

    # If voltage_ext is a function, we can't return a single value
    if isa(coil.voltage_ext, Function)
        @warn "Coil '$coil_name' has time-dependent voltage. Use get_coil_voltage_at_time(coil, t) instead."
        return coil.voltage_ext  # Return the function itself
    end

    return coil.voltage_ext
end

"""
    update_all_voltages!(csys::CoilSystem{FT}, new_voltages::Vector{FT}) where FT<:AbstractFloat

Update all external voltages for powered coils.
Length of `new_voltages` must match number of powered coils.
"""
function update_all_voltages!(csys::CoilSystem{FT}, new_voltages::Vector{FT}) where FT<:AbstractFloat
    if length(new_voltages) != csys.n_powered
        error("Length of new_voltages ($(length(new_voltages))) must match number of powered coils ($(csys.n_powered))")
    end

    for (i, coil_idx) in enumerate(csys.powered_indices)
        csys.coils[coil_idx].voltage_ext = new_voltages[i]
    end
    return nothing
end

"""
    update_controllable_voltages!(csys::CoilSystem{FT}, new_voltages::Vector{FT}) where FT<:AbstractFloat

Update external voltages for controllable coils only.
Length of `new_voltages` must match number of controllable coils.
"""
function update_controllable_voltages!(csys::CoilSystem{FT}, new_voltages::Vector{FT}) where FT<:AbstractFloat
    if length(new_voltages) != csys.n_controllable
        error("Length of new_voltages ($(length(new_voltages))) must match number of controllable coils ($(csys.n_controllable))")
    end

    for (i, coil_idx) in enumerate(csys.controllable_indices)
        csys.coils[coil_idx].voltage_ext = new_voltages[i]
    end
    return nothing
end

"""
    get_coil_positions(csys::CoilSystem{FT}) where FT<:AbstractFloat

Return R and Z coordinates of all coils as separate vectors.
"""
function get_coil_positions(csys::CoilSystem{FT}) where FT<:AbstractFloat
    R = [coil.position.r for coil in csys.coils]
    Z = [coil.position.z for coil in csys.coils]
    return R, Z
end

"""
    get_powered_coil_positions(csys::CoilSystem{FT}) where FT<:AbstractFloat

Return R and Z coordinates of powered coils only.
"""
function get_powered_coil_positions(csys::CoilSystem{FT}) where FT<:AbstractFloat
    powered_coils = get_powered_coils(csys)
    R = [coil.position.r for coil in powered_coils]
    Z = [coil.position.z for coil in powered_coils]
    return R, Z
end

"""
    calculate_coil_resistance(area::FT, major_radius::FT, resistivity::FT) where FT<:AbstractFloat

Calculate resistance of a toroidal coil.
"""
function calculate_coil_resistance(area::FT, major_radius::FT, resistivity::FT) where FT<:AbstractFloat
    path_length = 2π * major_radius
    return resistivity * path_length / area
end

"""
    calculate_self_inductance(area::FT, major_radius::FT, μ0::FT) where FT<:AbstractFloat

Calculate self-inductance of a toroidal coil using Neumann's formula approximation.
"""
function calculate_self_inductance(area::FT, major_radius::FT, μ0::FT) where FT<:AbstractFloat
    # From MATLAB: YY = 1.0; coil_radius = sqrt(area/π)
    # self_L = μ₀ * R * (log(8*R/a) - 2 + 0.25*YY)
    YY = one(FT)
    coil_radius = sqrt(area / π)
    return μ0 * major_radius * (log(8 * major_radius / coil_radius) - 2 + 0.25 * YY)
end

"""
    create_coil_from_parameters(r::FT, z::FT, area::FT, name::String, is_powered::Bool,
                               μ0::FT, cu_resistivity::FT;
                               is_controllable=is_powered,
                               max_voltage=nothing, max_current=nothing,
                               current=zero(FT), voltage_ext=zero(FT)) where FT<:AbstractFloat

Create a coil with calculated resistance and self-inductance.
"""
function create_coil_from_parameters(r::FT, z::FT, area::FT, name::String, is_powered::Bool,
                                   μ0::FT, cu_resistivity::FT;
                                   is_controllable=is_powered,
                                   max_voltage=nothing, max_current=nothing,
                                   current=zero(FT), voltage_ext=zero(FT)) where FT<:AbstractFloat
    position = (r=r, z=z)
    resistance = calculate_coil_resistance(area, r, cu_resistivity)
    self_inductance = calculate_self_inductance(area, r, μ0)

    return Coil(position, area, resistance, self_inductance, is_powered, is_controllable, name,
                max_voltage, max_current, current, voltage_ext)
end

"""
    get_all_currents(csys::CoilSystem{FT}) where FT<:AbstractFloat

Return currents of all coils as a vector.
"""
function get_all_currents(csys::CoilSystem{FT}) where FT<:AbstractFloat
    return [coil.current for coil in csys.coils]
end

"""
    get_powered_currents(csys::CoilSystem{FT}) where FT<:AbstractFloat

Return currents of powered coils only as a vector.
"""
function get_powered_currents(csys::CoilSystem{FT}) where FT<:AbstractFloat
    return [csys.coils[idx].current for idx in csys.powered_indices]
end

"""
    get_controllable_currents(csys::CoilSystem{FT}) where FT<:AbstractFloat

Return currents of controllable coils only as a vector.
"""
function get_controllable_currents(csys::CoilSystem{FT}) where FT<:AbstractFloat
    return [csys.coils[idx].current for idx in csys.controllable_indices]
end

"""
    get_all_voltages(csys::CoilSystem{FT}) where FT<:AbstractFloat

Return external voltages of all coils as a vector at t=0.
For time-dependent voltages, use get_all_voltages_at_time(csys, t) instead.
"""
function get_all_voltages(csys::CoilSystem{FT}) where FT<:AbstractFloat
    return get_all_voltages_at_time(csys, zero(FT))
end

"""
    get_powered_voltages(csys::CoilSystem{FT}) where FT<:AbstractFloat

Return external voltages of powered coils only as a vector at t=0.
For time-dependent voltages, use get_powered_voltages_at_time(csys, t) instead.
"""
function get_powered_voltages(csys::CoilSystem{FT}) where FT<:AbstractFloat
    return get_powered_voltages_at_time(csys, zero(FT))
end

"""
    get_controllable_voltages(csys::CoilSystem{FT}) where FT<:AbstractFloat

Return external voltages of controllable coils only as a vector at t=0.
For time-dependent voltages, use get_controllable_voltages_at_time(csys, t) instead.
"""
function get_controllable_voltages(csys::CoilSystem{FT}) where FT<:AbstractFloat
    return get_controllable_voltages_at_time(csys, zero(FT))
end

"""
    set_all_currents!(csys::CoilSystem{FT}, currents::Vector{FT}) where FT<:AbstractFloat

Set currents for all coils. Length of `currents` must match total number of coils.
"""
function set_all_currents!(csys::CoilSystem{FT}, currents::Vector{FT}) where FT<:AbstractFloat
    if length(currents) != csys.n_total
        error("Length of currents ($(length(currents))) must match total number of coils ($(csys.n_total))")
    end

    for (i, coil) in enumerate(csys.coils)
        coil.current = currents[i]
    end
    return nothing
end

"""
    set_powered_currents!(csys::CoilSystem{FT}, currents::Vector{FT}) where FT<:AbstractFloat

Set currents for powered coils only. Length of `currents` must match number of powered coils.
"""
function set_powered_currents!(csys::CoilSystem{FT}, currents::Vector{FT}) where FT<:AbstractFloat
    if length(currents) != csys.n_powered
        error("Length of currents ($(length(currents))) must match number of powered coils ($(csys.n_powered))")
    end

    for (i, coil_idx) in enumerate(csys.powered_indices)
        csys.coils[coil_idx].current = currents[i]
    end
    return nothing
end

"""
    set_controllable_currents!(csys::CoilSystem{FT}, currents::Vector{FT}) where FT<:AbstractFloat

Set currents for controllable coils only. Length of `currents` must match number of controllable coils.
"""
function set_controllable_currents!(csys::CoilSystem{FT}, currents::Vector{FT}) where FT<:AbstractFloat
    if length(currents) != csys.n_controllable
        error("Length of currents ($(length(currents))) must match number of controllable coils ($(csys.n_controllable))")
    end

    for (i, coil_idx) in enumerate(csys.controllable_indices)
        csys.coils[coil_idx].current = currents[i]
    end
    return nothing
end

"""
    set_coil_current!(csys::CoilSystem{FT}, coil_name::String, current::FT) where FT<:AbstractFloat

Set current for a specific coil by name.
"""
function set_coil_current!(csys::CoilSystem{FT}, coil_name::String, current::FT) where FT<:AbstractFloat
    coil_idx = find_coil_by_name(csys, coil_name)
    if isnothing(coil_idx)
        error("Coil '$coil_name' not found")
    end

    csys.coils[coil_idx].current = current
    return nothing
end

"""
    get_coil_current(csys::CoilSystem, coil_name::String)

Get current for a specific coil by name.
"""
function get_coil_current(csys::CoilSystem, coil_name::String)
    coil_idx = find_coil_by_name(csys, coil_name)
    if isnothing(coil_idx)
        error("Coil '$coil_name' not found")
    end

    return csys.coils[coil_idx].current
end

"""
    evaluate_voltage_ext(voltage_ext::Union{FT, Function}, t::FT) where FT<:AbstractFloat

Evaluate voltage_ext at time t. If voltage_ext is a value, return it directly.
If it's a function, call it with time t.

# Arguments
- `voltage_ext`: Either a constant voltage value or a function f(t) -> voltage
- `t`: Current time

# Returns
- Voltage value at time t
"""
function evaluate_voltage_ext(voltage_ext::FT, t::FT) where {FT <: AbstractFloat}
    return voltage_ext
end

function evaluate_voltage_ext(voltage_ext::Function, t::FT) where {FT <: AbstractFloat}
    return voltage_ext(t)
end

"""
    get_coil_voltage_at_time(coil::Coil{FT}, t::FT) where FT<:AbstractFloat

Get the voltage of a coil at time t, evaluating voltage_ext if it's a function.
"""
function get_coil_voltage_at_time(coil::Coil{FT}, t::FT) where {FT <: AbstractFloat}
    if !coil.is_powered
        return zero(FT)
    end
    return evaluate_voltage_ext(coil.voltage_ext, t)
end

"""
    get_coil_voltage_at_time(csys::CoilSystem, coil_name::String, t::AbstractFloat)

Get external voltage for a powered coil by name at time t.
"""
function get_coil_voltage_at_time(csys::CoilSystem, coil_name::String, t::AbstractFloat)
    coil_idx = find_coil_by_name(csys, coil_name)
    if isnothing(coil_idx)
        error("Coil '$coil_name' not found")
    end

    coil = csys.coils[coil_idx]
    return get_coil_voltage_at_time(coil, t)
end

"""
    get_all_voltages_at_time(csys::CoilSystem{FT}, t::FT) where FT<:AbstractFloat

Return external voltages of all coils as a vector at time t.
For constant voltages, returns the constant value. For time-dependent voltages, evaluates at time t.
"""
function get_all_voltages_at_time(csys::CoilSystem{FT}, t::FT) where FT<:AbstractFloat
    return [get_coil_voltage_at_time(coil, t) for coil in csys.coils]
end

"""
    get_powered_voltages_at_time(csys::CoilSystem{FT}, t::FT) where FT<:AbstractFloat

Return external voltages of powered coils only as a vector at time t.
"""
function get_powered_voltages_at_time(csys::CoilSystem{FT}, t::FT) where FT<:AbstractFloat
    return [get_coil_voltage_at_time(csys.coils[idx], t) for idx in csys.powered_indices]
end

"""
    get_controllable_voltages_at_time(csys::CoilSystem{FT}, t::FT) where FT<:AbstractFloat

Return external voltages of controllable coils only as a vector at time t.
"""
function get_controllable_voltages_at_time(csys::CoilSystem{FT}, t::FT) where FT<:AbstractFloat
    return [get_coil_voltage_at_time(csys.coils[idx], t) for idx in csys.controllable_indices]
end

"""
    set_coil_voltage_function!(csys::CoilSystem, coil_name::String, voltage_func::Function)

Set a time-dependent voltage function for a powered coil.

# Arguments
- `csys`: The coil system
- `coil_name`: Name of the coil to modify
- `voltage_func`: Function f(t) -> voltage that will be called with time argument

# Example
```julia
# Linear ramp: V(t) = rate * (t - t_start) + V0
ramp_func = t -> 100.0 * (t - 0.1) + 50.0
set_coil_voltage_function!(system, "PF1", ramp_func)

# Sinusoidal voltage
sine_func = t -> 1000.0 * sin(2π * 60.0 * t)  # 60 Hz, 1000V amplitude
set_coil_voltage_function!(system, "PF2", sine_func)
```
"""
function set_coil_voltage_function!(csys::CoilSystem, coil_name::String, voltage_func::Function)
    coil_idx = find_coil_by_name(csys, coil_name)
    if isnothing(coil_idx)
        error("Coil '$coil_name' not found")
    end

    coil = csys.coils[coil_idx]
    if !coil.is_powered
        error("Coil '$coil_name' is not a powered coil")
    end

    # Set the voltage function
    coil.voltage_ext = voltage_func
    return nothing
end

"""
    create_linear_voltage_ramp(rate::FT, t_start::FT, V0::FT) where FT<:AbstractFloat

Create a linear voltage ramp function: V(t) = rate * (t - t_start) + V0

# Arguments
- `rate`: Voltage change rate [V/s]
- `t_start`: Start time for the ramp [s]
- `V0`: Initial voltage at t_start [V]

# Returns
- Function f(t) -> voltage
"""
function create_linear_voltage_ramp(rate::FT, t_start::FT, V0::FT) where {FT <: AbstractFloat}
    return t -> rate * (t - t_start) + V0
end

"""
    create_sinusoidal_voltage(amplitude::FT, frequency::FT, phase::FT=zero(FT), offset::FT=zero(FT)) where FT<:AbstractFloat

Create a sinusoidal voltage function: V(t) = amplitude * sin(2π * frequency * t + phase) + offset

# Arguments
- `amplitude`: Voltage amplitude [V]
- `frequency`: Frequency [Hz]
- `phase`: Phase offset [rad] (default: 0)
- `offset`: DC offset [V] (default: 0)

# Returns
- Function f(t) -> voltage
"""
function create_sinusoidal_voltage(amplitude::FT, frequency::FT, phase::FT=zero(FT), offset::FT=zero(FT)) where {FT <: AbstractFloat}
    return t -> amplitude * sin(2π * frequency * t + phase) + offset
end

"""
    create_step_voltage(V_before::FT, V_after::FT, t_step::FT) where FT<:AbstractFloat

Create a step voltage function: V(t) = V_before for t < t_step, V_after for t >= t_step

# Arguments
- `V_before`: Voltage before step [V]
- `V_after`: Voltage after step [V]
- `t_step`: Time of step [s]

# Returns
- Function f(t) -> voltage
"""
function create_step_voltage(V_before::FT, V_after::FT, t_step::FT) where {FT <: AbstractFloat}
    return t -> t < t_step ? V_before : V_after
end

"""
    create_piecewise_linear_voltage(times::Vector{FT}, voltages::Vector{FT}) where FT<:AbstractFloat

Create a piecewise linear voltage function from time-voltage pairs.

# Arguments
- `times`: Time points [s] (must be sorted)
- `voltages`: Corresponding voltage values [V]

# Returns
- Function f(t) -> voltage (linear interpolation between points, constant extrapolation)
"""
function create_piecewise_linear_voltage(times::Vector{FT}, voltages::Vector{FT}) where {FT <: AbstractFloat}
    @assert length(times) == length(voltages) "times and voltages must have same length"
    @assert length(times) >= 2 "Need at least 2 points for piecewise linear"
    @assert issorted(times) "times must be sorted"

    return function(t)
        if t <= times[1]
            return voltages[1]
        elseif t >= times[end]
            return voltages[end]
        else
            # Find the interval
            i = searchsortedlast(times, t)
            # Linear interpolation
            t1, t2 = times[i], times[i+1]
            v1, v2 = voltages[i], voltages[i+1]
            return v1 + (v2 - v1) * (t - t1) / (t2 - t1)
        end
    end
end
