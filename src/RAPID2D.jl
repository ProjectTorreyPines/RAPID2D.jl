"""
RAPID2D.jl - Rapid Analysis of Plasma Initiation and Dynamics in 2D

A Julia implementation of the RAPID-2D plasma modeling code for axisymmetric
(R,Z) plasma simulation with a focus on tokamak startup, current drive, and
plasma dynamics.
"""
module RAPID2D

using Plots
using StaticArrays
using SparseArrays
using LinearAlgebra
using Statistics
using HDF5
using Printf
using DelimitedFiles
using Interpolations
using SimpleUnPack

# Include type definitions
include("constants.jl")
include("numerics/discretized_operator.jl")
include("types.jl")
include("initialization.jl")
include("numerics/numerics.jl")
include("numerics/operators.jl")

# Include the field-related functionality
include("physics/fields.jl")
include("physics/transport.jl")

# Include the physics models
include("physics/physics.jl")

# Include the cross-section models
include("reactions/electron_Xsec.jl")
include("reactions/reaction_rate_coefficients.jl")

# Include simulation workflows
include("workflows.jl")

# Include the IO-related functionality
include("io/io.jl")

# Include utility functions
include("utils/grid.jl")  # Grid-related utility functions
include("utils/utils.jl")

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
    config::SimulationConfig{FT}=SimulationConfig{Float64}(),
) where {FT<:AbstractFloat}

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

end # module RAPID2D