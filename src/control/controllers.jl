"""
Controller creation and management functions.

Provides functions to create and configure different types of controllers.
"""

using DiscretePIDs

"""
    create_controller(target::FT, coils::Vector{Coil{FT}};
                     control_type="generic",
                     Kp=FT(1.0), Ti=typemax(FT), Td=zero(FT),
                     dt=zero(FT),
                     umin=typemin(FT), umax=typemax(FT)) where {FT<:AbstractFloat}

Create a generic controller using DiscretePID.

# Arguments
- `target`: Target value (required positional argument)
- `coils`: Vector of coils to control (required positional argument)

# Keyword Arguments
- `control_type`: String describing the control purpose
- `Kp`: Proportional gain
- `Ti`: Integral time constant [s] (typemax(FT) = no integral action)
- `Td`: Derivative time constant [s] (zero(FT) = no derivative action)
- `dt`: Time step [s] (zero(FT) = must be set later)
- `umin, umax`: Control signal limits (typemin/typemax(FT) = no limits)
"""
function create_controller(
			target::FT,
			coils::Vector{Coil{FT}};
			control_type::String="generic",
			Kp::FT=FT(1.0), Ti::FT=typemax(FT), Td::FT=zero(FT),
			dt::FT=zero(FT),
			umin::FT=typemin(FT), umax::FT=typemax(FT)
		) where {FT<:AbstractFloat}

    # Convert PID parameters for DiscretePID
    pid = DiscretePID(;
        K = Kp,
        Ti = Ti,
        Td = Td,
        Ts = dt,
        N = FT(8),         # Derivative filter
        b = FT(1.0),       # Setpoint weighting
        umin = umin,
        umax = umax,
        Tt = sqrt(2.0 * Td)  # Anti-windup time constant
    )

    return Controller{FT}(
        pid = pid,
        target = target,
        coils = copy(coils),
        control_type = control_type
    )
end

"""
    create_current_controller(target_current::FT, coils::Vector{Coil{FT}};
                             Kp=FT(5.0), Ti=FT(0.4), Td=FT(0.02),
                             kwargs...) where {FT<:AbstractFloat}

Create a current controller with default parameters optimized for current control.

# Arguments
- `target_current`: Target plasma current [A] (required positional argument)
- `coils`: Vector of coils to control (required positional argument)

# Keyword Arguments
- `Kp`: Proportional gain (default: 5.0 for current control)
- `Ti`: Integral time constant [s] (default: 0.4s for current control)
- `Td`: Derivative time constant [s] (default: 0.02s for current control)
- Additional kwargs passed to create_controller
"""
function create_current_controller(target_current::FT,
                                  coils::Vector{Coil{FT}};
                                  Kp=FT(5.0), Ti=FT(0.4), Td=FT(0.02),
                                  kwargs...) where {FT<:AbstractFloat}

    return create_controller(target_current, coils;
        control_type = "current",
        Kp = Kp, Ti = Ti, Td = Td,
        kwargs...
    )
end

"""
    create_position_controller(target_position::FT, coils::Vector{Coil{FT}};
                              Kp=FT(1.0), Ti=FT(10.0), Td=FT(0.1),
                              kwargs...) where {FT<:AbstractFloat}

Create a position controller with default parameters optimized for position control.

# Arguments
- `target_position`: Target plasma position [m] (required positional argument)
- `coils`: Vector of coils to control (required positional argument)

# Keyword Arguments
- `Kp`: Proportional gain (default: 1.0 for position control)
- `Ti`: Integral time constant [s] (default: 10.0s for position control)
- `Td`: Derivative time constant [s] (default: 0.1s for position control)
- Additional kwargs passed to create_controller
"""
function create_position_controller(target_position::FT,
                                   coils::Vector{Coil{FT}};
                                   Kp=FT(1.0), Ti=FT(10.0), Td=FT(0.1),
                                   kwargs...) where {FT<:AbstractFloat}

    return create_controller(target_position, coils;
        control_type = "position",
        Kp = Kp, Ti = Ti, Td = Td,
        kwargs...
    )
end

"""
    set_target!(controller::Controller{FT}, new_target::FT) where {FT<:AbstractFloat}

Update controller target value.
"""
function set_target!(controller::Controller{FT}, new_target::FT) where {FT<:AbstractFloat}
    controller.target = new_target
    return controller
end

"""
    reset_controller!(controller::Controller{FT}) where {FT<:AbstractFloat}

Reset PID controller internal state.
"""
function reset_controller!(controller::Controller{FT}) where {FT<:AbstractFloat}
	reset_state!(controller.pid)
    return controller
end

"""
    compute_control_signal!(controller::Controller{FT}, current_value::FT) where {FT<:AbstractFloat}

Compute control signal using PID controller.

Returns the control signal.
"""
function compute_control_signal!(controller::Controller{FT}, current_value::FT) where {FT<:AbstractFloat}
    # Use DiscretePID to compute control signal
    control_signal = controller.pid(controller.target, current_value, FT(0.0))
    return control_signal
end

"""
    apply_control_signal!(controller::Controller{FT}, signal::FT) where {FT<:AbstractFloat}

Apply control signal to all coils controlled by this controller.

# Arguments
- `controller`: Controller with coil information
- `signal`: Control signal to apply

Note: This function directly modifies the coil voltages through references.
"""
function apply_control_signal!(controller::Controller{FT}, signal::FT) where {FT<:AbstractFloat}
    # Apply signal to all coils controlled by this controller
    for coil in controller.coils
        coil.voltage_ext = signal
    end
    return controller
end

# Export controller management functions
export create_controller, create_current_controller, create_position_controller,
       set_target!, reset_controller!,
       compute_control_signal!, apply_control_signal!
