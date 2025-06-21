"""
Controller initialization functions.

Provides functions to automatically initialize controllers by finding appropriate
coils in the coil system.
"""

"""
    find_coils_by_name(coil_system, patterns::Vector{String}; case_sensitive=false)

Find coils in the coil system that match any of the given name patterns.

# Arguments
- `coil_system`: Coil system containing coils
- `patterns`: Vector of string patterns to match
- `case_sensitive`: Whether to perform case-sensitive matching

Returns a vector of coils that match any pattern.
"""
function find_coils_by_name(coil_system, patterns::Vector{String}; case_sensitive=false)
    matched_coils = []

    for coil in coil_system.coils
        coil_name = case_sensitive ? coil.name : uppercase(coil.name)

        for pattern in patterns
            pattern_to_match = case_sensitive ? pattern : uppercase(pattern)
            if occursin(pattern_to_match, coil_name)
                push!(matched_coils, coil)
                break  # Don't add the same coil multiple times
            end
        end
    end

    return matched_coils
end

"""
    initialize_current_controller!(RP::RAPID{FT};
                                  target_current=FT(1e6),
                                  Kp=FT(5.0), Ti=FT(2.5), Td=FT(0.02),
                                  umin=FT(-100.0), umax=FT(100.0)) where {FT<:AbstractFloat}

Initialize current controller by finding Ohmic Heating (OH) coils in the coil system.

This function searches for coils with OH-related names and creates a current controller
to control those coils for plasma current regulation.

# Arguments
- `RP`: RAPID simulation object
- `target_current`: Target plasma current [A]
- `Kp, Ti, Td`: PID controller gains
- `umin, umax`: Voltage limits [V]

Returns the created Controller instance.
"""
function initialize_current_controller!(RP::RAPID{FT};
                                       target_current=FT(1e6),
                                       Kp=FT(5.0), Ti=FT(2.5), Td=FT(0.02),
                                       umin=FT(-100.0), umax=FT(100.0)) where {FT<:AbstractFloat}

    # Primary patterns for OH coils
    primary_patterns = ["OH"]
    oh_coils = find_coils_by_name(RP.coil_system, primary_patterns)

    if isempty(oh_coils)
        @warn "No OH coils found in coil system. Searching for alternative current control coils..."

        # Fallback patterns
        fallback_patterns = ["OHMIC", "CURRENT", "CS"]  # CS = Central Solenoid
        oh_coils = find_coils_by_name(RP.coil_system, fallback_patterns)
    end

    if isempty(oh_coils)
        available_coils = [coil.name for coil in RP.coil_system.coils]
        @error "No suitable coils found for current control. Available coils: " *
               join(available_coils, ", ")
        error("Cannot initialize current controller without suitable coils.")
    end

    coil_names = [coil.name for coil in oh_coils]
    @info "Found $(length(oh_coils)) OH coils for current control: $(join(coil_names, ", "))"

    # Create current controller
    current_controller = create_current_controller(target_current, oh_coils;
        Kp = Kp,
        Ti = Ti,
        Td = Td,
        dt = RP.dt,
        umin = umin,
        umax = umax
    )

    return current_controller
end

"""
    initialize_position_controller!(RP::RAPID{FT};
                                   target_position=FT(0.65),
                                   Kp=FT(1.0), Ti=FT(10.0), Td=FT(0.1),
                                   umin=FT(-10.0), umax=FT(10.0)) where {FT<:AbstractFloat}

Initialize position controller by finding Poloidal Field (PF) coils.

# Arguments
- `RP`: RAPID simulation object
- `target_position`: Target plasma R position [m]
- `Kp, Ti, Td`: PID controller gains
- `umin, umax`: Control signal limits

Returns the created Controller instance.
"""
function initialize_position_controller!(RP::RAPID{FT};
                                        target_position=FT(0.65),
                                        Kp=FT(1.0), Ti=FT(10.0), Td=FT(0.1),
                                        umin=FT(-10.0), umax=FT(10.0)) where {FT<:AbstractFloat}

    # Patterns for position control coils
    patterns = ["PF", "POLOIDAL", "SHAPING"]
    pf_coils = find_coils_by_name(RP.coil_system, patterns)

    if isempty(pf_coils)
        @warn "No PF coils found for position control. Using all available coils as fallback."
        pf_coils = collect(RP.coil_system.coils)
    end

    coil_names = [coil.name for coil in pf_coils]
    @info "Found $(length(pf_coils)) PF coils for position control: $(join(coil_names, ", "))"

    # Create position controller
    position_controller = create_position_controller(target_position, pf_coils;
        Kp = Kp,
        Ti = Ti,
        Td = Td,
        dt = RP.dt,
        umin = umin,
        umax = umax
    )

    return position_controller
end

"""
    initialize_controller_set!(RP::RAPID{FT};
                               enable_current=true,
                               enable_position=false,
                               kwargs...) where {FT<:AbstractFloat}

Initialize a complete controller set with multiple controllers.

# Arguments
- `RP`: RAPID simulation object
- `enable_current`: Whether to initialize current controller
- `enable_position`: Whether to initialize position controller
- `kwargs...`: Additional parameters passed to individual controller initializers

Returns a ControllerSet instance.
"""
function initialize_controller_set!(RP::RAPID{FT};
                                   enable_current=true,
                                   enable_position=false,
                                   kwargs...) where {FT<:AbstractFloat}

    controller_set = ControllerSet{FT}()

    if enable_current
        @info "Initializing current controller..."
        controller_set.current = initialize_current_controller!(RP; kwargs...)
    end

    if enable_position
        @info "Initializing position controller..."
        controller_set.position = initialize_position_controller!(RP; kwargs...)
    end

    @info "Controller set initialized with $(sum([enable_current, enable_position])) controllers."

    return controller_set
end

# Export initialization functions
export find_coils_by_name,
       initialize_current_controller!, initialize_position_controller!,
       initialize_controller_set!
