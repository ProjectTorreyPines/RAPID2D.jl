"""
Control types for RAPID2D.

Defines controller structures that can be used for different control purposes
(current control, position control, temperature control, etc.).
"""

using DiscretePIDs

"""
    Controller{FT<:AbstractFloat}

Generic PID controller for plasma control applications.

This structure can be used for various control purposes by setting appropriate
target values and coil sets.

# Fields
- `pid`: DiscretePID controller instance
- `target`: Target value (current, position, temperature, etc.)
- `coils`: Vector of coils controlled by this PID
- `control_type`: String describing what this controller controls (e.g., "current", "position")

# Examples
```julia
# Current controller
current_ctrl = Controller{Float64}(
    pid = create_pid(...),
    target = 1e6,  # 1 MA
    coils = oh_coils,
    control_type = "current"
)

# Position controller
position_ctrl = Controller{Float64}(
    pid = create_pid(...),
    target = 0.65,  # major radius [m]
    coils = pf_coils,
    control_type = "position"
)
```
"""
@kwdef mutable struct Controller{FT<:AbstractFloat}
    pid::DiscretePID{FT}
    target::FT = FT(0.0)
    coils::Vector{Coil{FT}} = Coil{FT}[]
    control_type::String = "generic"
end

# Type aliases for specific controllers
const CurrentController{FT} = Controller{FT}
const PositionController{FT} = Controller{FT}
const TemperatureController{FT} = Controller{FT}

"""
    ControllerSet{FT<:AbstractFloat}

Container for multiple controllers working together.

# Fields
- `current`: Current controller (optional)
- `position`: Position controller (optional)
- `temperature`: Temperature controller (optional)
- `custom`: Dictionary for additional custom controllers
"""
@kwdef mutable struct ControllerSet{FT<:AbstractFloat}
    current::Union{Nothing, Controller{FT}} = nothing
    position::Union{Nothing, Controller{FT}} = nothing
    temperature::Union{Nothing, Controller{FT}} = nothing
    custom::Dict{String, Controller{FT}} = Dict{String, Controller{FT}}()
end

# Export types
export Controller, ControllerSet,
       CurrentController, PositionController, TemperatureController
