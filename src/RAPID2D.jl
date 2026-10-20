"""
RAPID2D.jl - Rapid Analysis of Plasma Initiation and Dynamics in 2D

A Julia implementation of the RAPID-2D plasma modeling code for axisymmetric
(R,Z) plasma simulation with a focus on tokamak startup, current drive, and
plasma dynamics.

Plotting functionality is provided through package extensions:
- Load Plots.jl to enable RAPID2DPlotsExt with plot_snaps0D, plot_snaps2D, etc.
- Load GLMakie.jl to enable RAPID2DMakieExt for high-performance plotting
"""
module RAPID2D

using StaticArrays
using SparseArrays
using LinearAlgebra
using Statistics
using HDF5
using Printf
using DelimitedFiles
using FastInterpolations
using SimpleUnPack
using ADIOS2
using TimerOutputs
using AdaptiveArrayPools: @with_pool, @safe_with_pool, acquire!, zeros!

LinearAlgebra.BLAS.set_num_threads(1)

# Global timer for performance monitoring
const RAPID_TIMER = TimerOutput()

# Include type definitions
include("constants.jl")
include("numerics/exponential_rosenbrock.jl")   # bern(z) coefficients for the local rates
include("numerics/discretized_operator.jl")
include("numerics/linear_solvers.jl")
include("numerics/coupled_block.jl")         # the coupled solve's u∥–ψ block on a fixed pattern
include("numerics/fixed_point.jl")           # the fixed-point solve, and Newton's step for an affine map
include("numerics/anderson.jl")              # Anderson mixing for the coupled solve's fixed point
include("physics/species.jl")       # species, ion records and the transport policies; types.jl needs them
include("numerics/field_lines/field_line_tracer_policies.jl")   # how a field line is followed: the tracer policies the flags hold
include("numerics/field_lines/closed_surface_flux_types.jl")   # the closed-surface flux constraint: policies, basis, state
include("control/ideal_vertical_field_types.jl")   # the vertical-field control the flags hold, and its state
include("types.jl")

include("diagnostics/diagnostics.jl")
include("diagnostics/column_force_balance.jl")   # the current column's 0D force balance, at the snapshots
include("initialization.jl")
include("numerics/numerics.jl")
include("numerics/operators.jl")

# Include the field-related functionality
include("physics/reactions.jl")             # event rates -> per-species sources
include("physics/neutral_gas.jl")
include("physics/wall.jl")   # wall face geometry; shares the Maxwellian speeds
include("numerics/wall_pattern.jl")        # the one sparsity pattern of the in-wall operators, in-place arithmetic on it
include("numerics/per_particle_operators.jl") # M = N⁻¹(AN − diag(An)): a density operator carried to a per-particle variable; its dissipation rate
include("physics/transport_channels.jl")   # the (v∥, λ∥, v⊥, λ⊥) basis
include("numerics/wall_diffusion.jl")      # wall-aware 9-point tensor operator
include("numerics/face_flux.jl")           # ∇·(u f) on cell faces, wall-aware
include("numerics/wall_advection.jl")      # (u·∇)f from the mass flux; ∇·u and ∇f on in-wall nodes
include("diagnostics/field_moments.jl")    # shape statistics, for direction checks
include("physics/wall_ledger.jl")          # per-face absorbed/emitted inventory
include("physics/ion_transport.jl")        # which equation the ion species share
include("physics/electron_wall.jl")
include("physics/electron_operators.jl")   # the per-step electron in-wall operators; what u∥ and Te solve with
include("physics/fields.jl")
include("physics/transport.jl")

# Include the physics models
include("physics/physics.jl")
include("physics/near_vacuum.jl")          # numerical treatments where the plasma is not resolved: vacuum relaxation, negative-density floor

# Include control system modules
include("control/types.jl")
include("control/controllers.jl")
include("control/interface.jl")
include("control/initialization.jl")
include("control/ideal_vertical_field.jl")   # the ideal vertical-field actuator: target, update, its fields

# Include the cross-section models
include("reactions/electron_Xsec.jl")
include("reactions/reaction_rate_coefficients.jl")
include("reactions/rate_models.jl")   # rate coefficients by model (a table or a fit) and context

# Include simulation workflows
include("step_transaction.jl")   # a step as a transaction: the state it writes, put back when an attempt is rejected
include("workflows.jl")

# Include plotting interface (actual implementations in extensions)
include("plotting.jl")

# Include the IO-related functionality
include("io/io.jl")
include("io/adios_io.jl")

# Include coil-related functionality
include("coils/core_functions.jl")
include("coils/initialization.jl")
include("coils/circuit_equations.jl")

# Include utility functions
include("utils/grid.jl")  # Grid-related utility functions
include("utils/green_function.jl")  # Green's function utilities
include("utils/utils.jl")

# Field lines: the tracers, and what is built on them
include("numerics/field_lines/field_line_tracing.jl")    # following ψ contours: geometry, visitors, the tracers
include("numerics/field_lines/field_line_following.jl")  # field-line lengths and closed lines on the grid (FLF)
include("numerics/field_lines/flux_surface_average.jl")  # averages over flux surfaces, and back to the grid
include("numerics/field_lines/closed_surface_flux.jl")   # surface functions on the closed region, their stream-function flux
include("numerics/field_lines/field_line_average.jl")    # averages along the field line through each node
include("physics/closed_surface_constraint.jl")          # the closed-surface flux constraint in a run

# Export types from various modules for convenience
export PlasmaConstants  # Physical constants
export AbstractExternalField, TimeSeriesExternalField  # External field types

# Export IO functions for external field data
export read_break_input_file, read_external_field_time_series, load_external_field_data!

"""
    create_rapid_object(;
        FT::Type{<:AbstractFloat}=Float64,
        config::SimulationConfig{FT}=SimulationConfig{FT}()
    ) where {FT<:AbstractFloat}

Create and initialize a RAPID object.

# Arguments
- `FT::Type{<:AbstractFloat}=Float64`: Floating-point type to use for calculations.
- `config::SimulationConfig{FT}=SimulationConfig{FT}()`: Configuration parameters for the RAPID object.
  Defines physics parameters, device setup, time stepping, grid dimensions, etc.

# Returns
- `RAPID{FT}`: A fully initialized RAPID instance ready for time advancement.

# Example
```julia
# Using default configuration
rp = create_rapid_object()

# Using custom configuration
config = SimulationConfig{Float64}()
config.device_Name = "KSTAR"
config.shot_Name = "001234"
config.NR = 200  # Higher resolution grid
config.NZ = 200
rp = create_rapid_object(config=config)
```
"""
function create_rapid_object(;
        config::SimulationConfig{FT} = SimulationConfig{Float64}(),
    ) where {FT <: AbstractFloat}

    # Create a RAPID object directly from the configuration
    RP = RAPID{FT}(config)

    # # Load physical constants
    # load_constants!(RP.config)

    # Perform full physics initialization
    initialize!(RP)

    return RP
end

# Export main functions
export create_rapid_object, initialize_simulation

# Export plotting functions (actual implementations in extensions)
export plot_snaps2D, animate_snaps2D, plot_comparison

end # module RAPID2D
