"""
RAPID2D.jl - Rapid Analysis of Plasma Initiation and Dynamics in 2D

A Julia implementation of the RAPID-2D plasma modeling code for axisymmetric
(R,Z) plasma simulation with a focus on tokamak startup, current drive, and
plasma dynamics.
"""
module RAPID2D

using SparseArrays
using LinearAlgebra
using Statistics
using HDF5
using Printf

# Include type definitions
include("constants.jl")
include("types.jl")
include("initialization.jl")

# Include the field-related functionality
include("physics/fields.jl")
include("physics/transport.jl")

# Include the physics models
include("physics/physics.jl")

# Include the IO-related functionality
include("io/io.jl")

# Include the numerical methods
include("numerics/numerics.jl")

# Include utility functions
include("utils/grid.jl")  # Grid-related utility functions
include("utils/utils.jl")

# Export types from various modules for convenience
export PlasmaConstants  # Physical constants

# Function to initialize RAPID2D simulation
function initialize_simulation(; NR::Int=100, NZ::Int=100,
                               FT::Type{<:AbstractFloat}=Float64,
                               t_start::AbstractFloat=0.0,
                               t_end::AbstractFloat=1.0e-3,
                               dt::AbstractFloat=1.0e-9,
                               R_range::Tuple{<:Real,<:Real}=(1.0, 2.0),
                               Z_range::Tuple{<:Real,<:Real}=(-1.0, 1.0))
    # Convert to correct floating-point type
    t_start_FT = FT(t_start)
    t_end_FT = FT(t_end)
    dt_FT = FT(dt)

    # Create a new RAPID instance with the specified grid
    RP = RAPID{FT}(NR, NZ; t_start=t_start_FT, t_end=t_end_FT, dt=dt_FT)

    # Initialize the grid geometry
    initialize_rapid_grid!(RP, R_range, Z_range)

    # Load physical constants
    load_constants!(RP.config)

    # Return the initialized simulation
    return RP
end

# Main simulation run function
function run_simulation!(RP::RAPID{FT}) where FT<:AbstractFloat
    # Simulation parameters
    dt = RP.dt
    t_end = RP.t_end_s

    # Main time loop
    while RP.time_s < t_end - 0.1*dt

        # Advance simulation one time step
        advance_timestep!(RP, dt)

        # Increment time
        RP.time_s += dt
        RP.step += 1

        # Print progress
        if RP.step % 100 == 0
            @printf("Time: %.6e s, Step: %d\n", RP.time_s, RP.step)
        end

        # Handle snapshots and file outputs if needed
        if hasfield(typeof(RP), :snap2D_Interval_s) && abs(RP.time_s - round(RP.time_s/RP.snap2D_Interval_s)*RP.snap2D_Interval_s) < 0.1*dt
            # Take snapshot of 2D data
            save_snapshot2D(RP)
        end
    end

    println("Simulation completed")
    return RP
end

# Function to advance simulation by one time step
function advance_timestep!(RP::RAPID{FT}, dt::FT) where FT<:AbstractFloat
    # TODO: Implement physics advancement:
    # 1. Update fields
    # 2. Update transport coefficients
    # 3. Solve continuity equations
    # 4. Solve momentum equations
    # 5. Solve temperature equations
end

# Export main functions
export initialize_simulation, run_simulation!, advance_timestep!

end # module RAPID2D