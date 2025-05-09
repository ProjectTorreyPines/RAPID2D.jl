"""
Type definitions for RAPID2D.jl
"""

# Importing the PlasmaConstants from constants.jl
import RAPID2D: PlasmaConstants

"""
    SimulationConfig{FT<:AbstractFloat}

Contains simulation configuration parameters.
"""
@kwdef mutable struct SimulationConfig{FT<:AbstractFloat}
    # Device parameters
    device_Name::String = "manual"     # Device name
    shot_Name::String = "test"         # Shot name

    # Grid dimensions
    NR::Int = 100                      # Number of radial grid points
    NZ::Int = 100                      # Number of vertical grid points
    R_min::FT = FT(1.0)                # Minimum radial coordinate
    R_max::FT = FT(2.0)                # Maximum radial coordinate
    Z_min::FT = FT(-1.0)               # Minimum vertical coordinate
    Z_max::FT = FT(1.0)                # Maximum vertical coordinate

    # Time parameters
    t_start_s::FT = FT(0.0)            # Simulation start time [s]
    t_end_s::FT = FT(1.0e-3)           # Simulation end time [s]
    dt::FT = FT(1.0e-9)                # Time step [s]

    # Physical constants
    constants::PlasmaConstants{FT} = PlasmaConstants{FT}()  # Consolidated physical constants

    # Legacy physical constants (for backward compatibility)
    ee::FT = FT(1.602176634e-19)       # Elementary charge (C)
    me::FT = FT(9.1093837015e-31)      # Electron mass (kg)
    mi::FT = FT(3.34754699166e-27)     # Ion mass (kg)
    eps0::FT = FT(8.8541878128e-12)    # Vacuum permittivity (F/m)
    mu0::FT = FT(1.25663706212e-6)     # Vacuum permeability (H/m)
    kB::FT = FT(1.380649e-23)          # Boltzmann constant (J/K)

    # Field configuration
    R0B0::FT = FT(2.0)                 # On-axis R0*B0 value

    # Initial conditions
    prefilled_gas_pressure::FT = FT(0.1)  # Prefilled gas pressure (Pa)

    # Limits
    min_Te::FT = FT(0.05)              # Minimum electron temperature (eV)
    max_Te::FT = FT(100.0)             # Maximum electron temperature (eV)

    # Transport parameters
    Dpara0::FT = FT(1.0)               # Base parallel diffusion coefficient
    Dperp0::FT = FT(0.1)               # Base perpendicular diffusion coefficient

    # Paths
    Input_path::String = "./input"     # Path to input files
    Output_path::String = "./output"   # Path to output files
    Output_prefix::String = ""         # Prefix for output files
    Output_name::String = "RAPID2D"    # Name for output files

    # Output intervals
    snap1D_Interval_s::FT = FT(1.0e-5)  # Time interval for 1D snapshots
    snap2D_Interval_s::FT = FT(1.0e-4)  # Time interval for 2D snapshots
    write_File_Interval_s::FT = FT(1.0e-3)  # Time interval for file writing
end

"""
    WallGeometry{FT<:AbstractFloat}

Represents the geometry of the device wall.

Fields:
- `R`: Radial coordinates of wall points
- `Z`: Vertical coordinates of wall points
"""
@kwdef struct WallGeometry{FT<:AbstractFloat}
    R::Vector{FT} = Vector{FT}()
    Z::Vector{FT} = Vector{FT}()
end


function WallGeometry(R::Vector{FT}, Z::Vector{FT}, check_wall::Bool) where {FT<:AbstractFloat}
    if check_wall
        @assert length(R) == length(Z) "R and Z must have the same length"
        @assert length(R) >= 3 "At least 3 points needed to define a wall unless creating an empty placeholder"

        new_R, new_Z = copy(R), copy(Z)
        if new_R[1] != new_R[end] || new_Z[1] != new_Z[end]
            push!(new_R, new_R[1])
            push!(new_Z, new_Z[1])
        end

        return WallGeometry{FT}(; R=new_R, Z=new_Z)
    else
        return WallGeometry{FT}(; R, Z)
    end
end

"""
    PlasmaState{FT<:AbstractFloat}

Contains the plasma state variables including density, temperature, and velocity components.
"""
@kwdef mutable struct PlasmaState{FT<:AbstractFloat}
    # Dimensions
    dims::Tuple{Int,Int} # (NR, NZ)

    # Gas temperature (scalar)
    T_gas_eV::FT = FT(0.026)           # Gas temperature [eV]

    # Densities
    ne::Matrix{FT} = zeros(FT, dims)    # Electron density [m^-3]
    ni::Matrix{FT} = zeros(FT, dims)    # Ion density [m^-3]
    n_H2_gas::Matrix{FT} = zeros(FT, dims)  # H2 gas density [m^-3]

    # Temperatures
    Te_eV::Matrix{FT} = zeros(FT, dims) # Electron temperature [eV]
    Ti_eV::Matrix{FT} = zeros(FT, dims) # Ion temperature [eV]

    # Velocities - parallel components
    ue_para::Matrix{FT} = zeros(FT, dims)  # Electron parallel velocity [m/s]
    ui_para::Matrix{FT} = zeros(FT, dims)  # Ion parallel velocity [m/s]

    # Velocities - vector components
    ueR::Matrix{FT} = zeros(FT, dims)   # Electron R velocity [m/s]
    ueZ::Matrix{FT} = zeros(FT, dims)   # Electron Z velocity [m/s]
    ueϕ::Matrix{FT} = zeros(FT, dims)   # Electron ϕ velocity [m/s]
    uiR::Matrix{FT} = zeros(FT, dims)   # Ion R velocity [m/s]
    uiZ::Matrix{FT} = zeros(FT, dims)   # Ion Z velocity [m/s]
    uiϕ::Matrix{FT} = zeros(FT, dims)   # Ion ϕ velocity [m/s]

    # Collision parameters
    lnA::Matrix{FT} = zeros(FT, dims)   # Coulomb logarithm
    nu_ei::Matrix{FT} = zeros(FT, dims) # Electron-ion collision frequency [1/s]
    sptz_fac::Matrix{FT} = zeros(FT, dims) # Spitzer factor for conductivity

    # Current densities
    Jϕ::Matrix{FT} = zeros(FT, dims)    # Toroidal current density [A/m²]

    # Power sources/sinks
    ePowers::Dict{Symbol, Matrix{FT}} = Dict{Symbol, Matrix{FT}}(
        :tot => zeros(FT, dims),
        :diffu => zeros(FT, dims),
        :conv => zeros(FT, dims),
        :heat => zeros(FT, dims),
        :drag => zeros(FT, dims),
        :equi => zeros(FT, dims),
        :iz => zeros(FT, dims),
        :exc => zeros(FT, dims),
        :dilution => zeros(FT, dims)
    )

    iPowers::Dict{Symbol, Matrix{FT}} = Dict{Symbol, Matrix{FT}}(
        :tot => zeros(FT, dims),
        :atomic => zeros(FT, dims),
        :equi => zeros(FT, dims)
    )
end

function PlasmaState{FT}(dimensions::Tuple{Int,Int}) where {FT<:AbstractFloat}
    return PlasmaState{FT}(dims=dimensions)
end
function PlasmaState{FT}(NR::Int, NZ::Int; kwargs...) where {FT<:AbstractFloat}
    return PlasmaState{FT}(dims=(NR, NZ); kwargs...)
end


"""
    Fields{FT<:AbstractFloat}

Contains the electromagnetic field variables.

Fields include components of the magnetic and electric fields.
"""
@kwdef mutable struct Fields{FT<:AbstractFloat}
    # Dimensions
    dims::Tuple{Int,Int} # (NR, NZ)

    # External fields
    BR_ext::Matrix{FT} = zeros(FT, dims)        # External radial magnetic field [T]
    BZ_ext::Matrix{FT} = zeros(FT, dims)        # External vertical magnetic field [T]
    LV_ext::Matrix{FT} = zeros(FT, dims)        # External Loop Voltage [V]
    psi_ext::Matrix{FT} = zeros(FT, dims)       # External magnetic flux [Wb/rad]
    Eϕ_ext::Matrix{FT} = zeros(FT, dims)        # External toroidal electric field [V/m]
    E_para_ext::Matrix{FT} = zeros(FT, dims)    # External parallel electric field [V/m]

    # Self-generated fields
    BR_self::Matrix{FT} = zeros(FT, dims)       # Self-generated radial magnetic field [T]
    BZ_self::Matrix{FT} = zeros(FT, dims)       # Self-generated vertical magnetic field [T]
    psi_self::Matrix{FT} = zeros(FT, dims)      # Self-generated magnetic flux [Wb/rad]
    Eϕ_self::Matrix{FT} = zeros(FT, dims)       # Self-generated toroidal electric field [V/m]
    E_para_self_ES::Matrix{FT} = zeros(FT, dims) # Electrostatic self-generated parallel electric field [V/m]
    E_para_self_EM::Matrix{FT} = zeros(FT, dims) # Electromagnetic self-generated parallel electric field [V/m]

    # Total fields - external + self-generated
    BR::Matrix{FT} = zeros(FT, dims)            # Total radial magnetic field [T]
    BZ::Matrix{FT} = zeros(FT, dims)            # Total vertical magnetic field [T]
    Bϕ::Matrix{FT} = zeros(FT, dims)            # Toroidal magnetic field [T]

    # Derived field quantities
    Bpol::Matrix{FT} = zeros(FT, dims)          # Poloidal magnetic field [T]
    Btot::Matrix{FT} = zeros(FT, dims)          # Total magnetic field [T]

    # Magnetic field unit vectors
    bR::Matrix{FT} = zeros(FT, dims)            # Radial unit vector
    bZ::Matrix{FT} = zeros(FT, dims)            # Vertical unit vector
    bϕ::Matrix{FT} = zeros(FT, dims)            # Toroidal unit vector

    # Electric field components
    ER::Matrix{FT} = zeros(FT, dims)            # Radial electric field [V/m]
    EZ::Matrix{FT} = zeros(FT, dims)            # Vertical electric field [V/m]
    Eϕ::Matrix{FT} = zeros(FT, dims)            # Toroidal electric field [V/m]

    # Parallel electric field
    E_para_ind::Matrix{FT} = zeros(FT, dims)    # Induced parallel electric field [V/m]
    E_para_tot::Matrix{FT} = zeros(FT, dims)    # Total parallel electric field [V/m]

    # Magnetic flux
    psi::Matrix{FT} = zeros(FT, dims)           # Total magnetic flux [Wb/rad]
end

# Constructor with separate dimensions
function Fields{FT}(dimensions::Tuple{Int,Int}) where {FT<:AbstractFloat}
    return Fields{FT}(dims=dimensions)
end
function Fields{FT}(NR::Int, NZ::Int) where {FT<:AbstractFloat}
    return Fields{FT}(dims=(NR, NZ))
end

"""
    Transport{FT<:AbstractFloat}

Contains the transport coefficients for the plasma.

Fields include diffusion coefficients in different directions.
"""
@kwdef mutable struct Transport{FT<:AbstractFloat}
    # Dimensions
    dims::Tuple{Int,Int} # (NR, NZ)

    # Base diffusivity values
    Dpara0::FT = FT(1.0)            # Base parallel diffusion coefficient [m²/s]
    Dperp0::FT = FT(0.1)            # Base perpendicular diffusion coefficient [m²/s]

    # Spatially-varying diffusion coefficients
    Dpara::Matrix{FT} = zeros(FT, dims)  # Parallel diffusion coefficient [m²/s]
    Dperp::Matrix{FT} = zeros(FT, dims)  # Perpendicular diffusion coefficient [m²/s]

    # Diffusion tensor components
    DRR::Matrix{FT} = zeros(FT, dims)    # R-R component of diffusion tensor
    DRZ::Matrix{FT} = zeros(FT, dims)    # R-Z component of diffusion tensor
    DZZ::Matrix{FT} = zeros(FT, dims)    # Z-Z component of diffusion tensor
end

# Constructor with separate dimensions
function Transport{FT}(dimensions::Tuple{Int,Int}; Dpara0::FT=FT(1.0), Dperp0::FT=FT(0.1)) where FT<:AbstractFloat
    return Transport{FT}(dims=dimensions, Dpara0=Dpara0, Dperp0=Dperp0)
end
function Transport{FT}(NR::Int, NZ::Int; Dpara0::FT=FT(1.0), Dperp0::FT=FT(0.1)) where FT<:AbstractFloat
    return Transport{FT}(dims=(NR, NZ), Dpara0=Dpara0, Dperp0=Dperp0)
end


"""
    Operators{FT<:AbstractFloat}

Contains the numerical operators used in the simulation.

Fields include various matrices for solving different parts of the model.
"""
@kwdef mutable struct Operators{FT<:AbstractFloat}
    # Dimensions
    dims::Tuple{Int,Int} # (NR, NZ)

    # Operators for solving equations
    A_GS::SparseMatrixCSC{FT, Int} = spzeros(FT, prod(dims), prod(dims))  # Matrix for Grad-Shafranov equation

    # RHS vectors for various equations
    neRHS_diffu::Matrix{FT} = zeros(FT, dims)  # RHS for diffusion term in electron continuity
    neRHS_convec::Matrix{FT} = zeros(FT, dims) # RHS for convection term in electron continuity
    neRHS_src::Matrix{FT} = zeros(FT, dims)    # RHS for source term in electron continuity
end

# Constructor with separate dimensions
function Operators{FT}(dimensions::Tuple{Int,Int}) where {FT<:AbstractFloat}
    return Operators{FT}(dims=dimensions)
end
function Operators{FT}(NR::Int, NZ::Int) where {FT<:AbstractFloat}
    return Operators{FT}(dims=(NR, NZ))
end


"""
    SimulationFlags

Contains boolean flags that control various aspects of the simulation.
"""
@kwdef mutable struct SimulationFlags
    # Method selection flags
    eRRC_method::String = "EoverP_Erg"        # Electron reaction rate coefficient method
    iRRC_method::String = "ud_T"              # Ion reaction rate coefficient method
    ud_method::String = "Xsec"                # Drift velocity method
    Ionz_method::String = "Xsec"              # Ionization method
    upara_or_uRphiZ::String = "upara"         # Velocity representation

    # Visualization flags
    vis1D::Bool = true                        # Enable 1D visualization
    vis2D::Bool = true                        # Enable 2D visualization

    # Physics flags
    diffu::Bool = true                        # Enable diffusion
    convec::Bool = true                       # Enable convection
    upwind::Bool = true                       # Use upwind scheme for convection
    src::Bool = true                          # Enable particle sources
    mean_ExB::Bool = true                     # Include mean ExB drift
    diaMag_drift::Bool = false                # Include diamagnetic drift
    turb_ExB_mixing::Bool = true              # Include turbulent ExB mixing
    E_para_self_ES::Bool = true               # Include self-electrostatic parallel E-field
    E_para_self_EM::Bool = true               # Include self-electromagnetic parallel E-field
    neg_n_correction::Bool = true             # Correct negative densities
    Te_evolve::Bool = true                    # Evolve electron temperature
    ud_evolve::Bool = true                    # Evolve drift velocity
    Gas_evolve::Bool = true                   # Evolve neutral gas density
    Coulomb_Collision::Bool = true            # Include Coulomb collisions
    Spitzer_Resistivity::Bool = true          # Include Spitzer resistivity
    Update_gFac::Bool = true                  # Update g factor for generalized EDF
    update_ni_independently::Bool = true      # Update ion density independently
    Secondary_Electron::Bool = true           # Include secondary electron emission

    # Field-related flags
    Ampere::Bool = false                      # Enable Ampere's law (magnetic field update)

    # Transport flags
    Include_ud_convec_term::Bool = true       # Include convection term in drift velocity equation
    Include_ud_diffu_term::Bool = true        # Include diffusion term in drift velocity equation
    Include_Te_convec_term::Bool = true       # Include convection term in Te equation
    Include_Te_diffu_term::Bool = true        # Include diffusion term in Te equation
    evolve_ud_inWall_only::Bool = false       # Only evolve drift velocity inside wall
    evolve_Te_inWall_only::Bool = false       # Only evolve Te inside wall
    Damp_Transp_outWall::Bool = true          # Damp transport outside wall

    # Numerical settings
    Ampere_nstep::Int = 10                    # Steps between Ampere's law updates
    FLF_nstep::Int = 10                       # Steps between field line following updates
    Implicit::Bool = true                     # Use implicit methods
    Implicit_weight::Float64 = 0.5            # Weight for implicit scheme
    Adapt_dt::Bool = false                    # Use adaptive time stepping

    # Temperature limits
    min_Te::Float64 = 0.001                   # Minimum electron temperature (eV)
    max_Te::Float64 = 500.0                   # Maximum electron temperature (eV)

    # Global force balance
    Global_Force_Balance::Bool = false        # Include global toroidal force balance

    # Control system
    Control::Dict{Symbol, Any} = Dict{Symbol, Any}(:state => false, :target_R => nothing)

    # Numerical stability controls
    Limit_too_negative_Diffusion::Dict{Symbol, Any} = Dict{Symbol, Any}(
        :state => true,
        :limit_lower_bound_ratio => -0.1  # -0.1*n
    )

    # Current threshold for Ampere's equation
    Ampere_Itor_threshold::Float64 = 0.0      # Current threshold for Ampere equation

    # Debug flags
    tmp_test::Bool = false                    # Enable temporary tests
    tmp_fig::Int = 100                        # Figure number for temporary tests

    # Initial parameters
    ini_gFac::Float64 = 1.0                   # Initial g factor value
    gamma_2nd_electron::Float64 = 0.1         # Secondary electron emission coefficient
end

"""
    NodeState{FT<:AbstractFloat}

Contains information about the grid nodes in relation to the wall.

# Fields
- `rid`: Radial index of each node
- `zid`: Vertical index of each node
- `nid`: Linear index of each node
- `state`: Node state (-1: outside, 0: boundary, 1: inside)
- `in_wall_nids`: Linear indices of nodes inside wall
- `out_wall_nids`: Linear indices of nodes outside wall
- `on_wall_nids`: Linear indices of nodes on the wall
"""
mutable struct NodeState{FT<:AbstractFloat}
    rid::Matrix{Int}          # Radial index of each node
    zid::Matrix{Int}          # Vertical index of each node
    nid::Matrix{Int}      # Linear index of each node
    state::Matrix{FT}         # Node state (-1: outside, 0: boundary, 1: inside)
    in_wall_nids::Vector{Int}  # Linear indices of nodes inside wall
    out_wall_nids::Vector{Int} # Linear indices of nodes outside wall
    on_wall_nids::Vector{Int}  # Linear indices of nodes on the wall

    # Constructor
    function NodeState{FT}(NR::Int, NZ::Int) where FT<:AbstractFloat
        rid = zeros(Int, NR, NZ)
        zid = zeros(Int, NR, NZ)
        nid = zeros(Int, NR, NZ)
        state = fill(NaN, NR, NZ)

        return new{FT}(rid, zid, nid, state, Int[], Int[], Int[])
    end

    # Convenience constructor
    function NodeState(NR::Int, NZ::Int)
        return NodeState{Float64}(NR, NZ)
    end
end


"""
    GridGeometry{FT<:AbstractFloat}

Contains the geometric properties of the computational grid.

# Fields
- `NR`: Number of radial grid points
- `NZ`: Number of vertical grid points
- `R1D`: 1D array of radial grid coordinates
- `Z1D`: 1D array of vertical grid coordinates
- `R2D`: 2D array of radial grid coordinates
- `Z2D`: 2D array of vertical grid coordinates
- `dR`: Radial grid spacing
- `dZ`: Vertical grid spacing
- `Jacob`: Jacobian determinant at grid points
- `inv_Jacob`: Inverse of Jacobian determinant
- `inVol2D`: Volume of each grid cell
- `BDY_idx`: Indices of boundary points
- `nodes`: Node information
"""
mutable struct GridGeometry{FT<:AbstractFloat}
    # Grid dimensions
    NR::Int                  # Number of radial grid points
    NZ::Int                  # Number of vertical grid points

    # Grid coordinates
    R1D::Vector{FT}          # 1D radial coordinates
    Z1D::Vector{FT}          # 1D vertical coordinates
    R2D::Matrix{FT}          # 2D radial coordinates
    Z2D::Matrix{FT}          # 2D vertical coordinates

    # Grid metrics
    dR::FT                   # Radial grid spacing
    dZ::FT                   # Vertical grid spacing
    Jacob::Matrix{FT}        # Jacobian determinant
    inv_Jacob::Matrix{FT}    # Inverse of Jacobian determinant
	inVol2D::Matrix{FT}      # Volume of each grid cell

    # Boundary indices
    BDY_idx::Vector{Int}     # Indices of boundary points

    # Node information
    nodes::NodeState{FT}     # Information about grid nodes

    # Constructor with dimensions
    function GridGeometry{FT}(NR::Int, NZ::Int) where FT<:AbstractFloat
        # Pre-allocate arrays
        R1D = Vector{FT}(undef, NR)
        Z1D = Vector{FT}(undef, NZ)
        R2D = zeros(FT, NR, NZ)
        Z2D = zeros(FT, NR, NZ)
        Jacob = zeros(FT, NR, NZ)
        inv_Jacob = zeros(FT, NR, NZ)
		inVol2D = zeros(FT, NR, NZ)
        BDY_idx = Int[]
        nodes = NodeState{FT}(NR, NZ)

        return new{FT}(
            NR, NZ,
            R1D, Z1D, R2D, Z2D,
            FT(0.0), FT(0.0),
            Jacob, inv_Jacob, inVol2D,
            BDY_idx,
            nodes
        )
    end

    # Convenience constructor
    function GridGeometry(NR::Int, NZ::Int)
        return GridGeometry{Float64}(NR, NZ)
    end
end

"""
    AbstractExternalField{FT<:AbstractFloat}

Abstract type for all external electromagnetic field sources.
Concrete implementations must provide methods to compute or interpolate field values at specified times.
"""
abstract type AbstractExternalField{FT<:AbstractFloat} end

"""
    RAPID{FT<:AbstractFloat}

The main simulation structure containing all simulation data including grid information,
physical fields, and simulation state.
"""
mutable struct RAPID{FT<:AbstractFloat}
    # Grid and wall geometry
    G::GridGeometry{FT}               # Grid geometry
    wall::WallGeometry{FT}            # Wall geometry data
    damping_func::Matrix{FT}          # Damping function outside wall

    # Grid masks
    cell_state::Matrix{Int}           # Cell state (1 inside wall, -1 outside)
    in_wall_nids::Vector{Int}         # Linear indices of cells inside wall
    out_wall_nids::Vector{Int}        # Linear indices of cells outside wall
    device_inVolume::FT               # Total volume inside wall

    # External field source
    external_field::Union{Nothing, AbstractExternalField{FT}}  # External EM field source

    # Reaction rate coefficients
    eRRC::Dict{Symbol, Any}           # Electron reaction rate coefficients
    iRRC::Dict{Symbol, Any}           # Ion reaction rate coefficients

    # Physical components
    config::SimulationConfig{FT}      # Simulation configuration
    flags::SimulationFlags            # Simulation flags
    plasma::PlasmaState{FT}           # Plasma state variables
    fields::Fields{FT}                # Field variables
    transport::Transport{FT}          # Transport coefficients
    operators::Operators{FT}          # Numerical operators

    # Time evolution
    step::Int                         # Current time step
    time_s::FT                        # Current time [s]
    t_start_s::FT                     # Start time [s]
    t_end_s::FT                       # End time [s]
    dt::FT                            # Time step [s]

    # Previous state and diagnostics
    prev_n::Matrix{FT}                # Previous density
    tElap::Dict{Symbol, Float64}      # Elapsed times
    diagnostics::Dict{Symbol, Any}    # Diagnostic data

    # Primary constructor - from config
    function RAPID{FT}(config::SimulationConfig{FT}) where {FT<:AbstractFloat}
        # Get grid dimensions
        NR, NZ = config.NR, config.NZ
        dims = (NR, NZ)

        # Initialize sub-components
        G = GridGeometry{FT}(NR, NZ)
        wall = WallGeometry{FT}()
        plasma = PlasmaState{FT}(dims)
        fields = Fields{FT}(dims)
        transport = Transport{FT}(dims; Dpara0=config.Dpara0, Dperp0=config.Dperp0)
        operators = Operators{FT}(dims)
        flags = SimulationFlags()

        # Initialize matrices
        damping_func = zeros(FT, dims)
        cell_state = zeros(Int, dims)
        prev_n = zeros(FT, dims)

        # Initialize empty containers
        in_wall_nids = Vector{Int}()
        out_wall_nids = Vector{Int}()
        eRRC = Dict{Symbol, Any}()
        iRRC = Dict{Symbol, Any}()
        tElap = Dict{Symbol, Float64}()
        diagnostics = Dict{Symbol, Any}()

        # Create and return new instance
        return new{FT}(
            G, wall, damping_func,
            cell_state, in_wall_nids, out_wall_nids, FT(0.0),
            nothing,  # external_field
            eRRC, iRRC,
            config, flags, plasma, fields, transport, operators,
            0, config.t_start_s, config.t_start_s, config.t_end_s, config.dt,
            prev_n, tElap, diagnostics
        )
    end
end

# Convenience constructors

"""
    RAPID{FT}(NR::Int, NZ::Int; kwargs...)

Create a RAPID instance with the specified grid dimensions.
"""
function RAPID{FT}(NR::Int, NZ::Int;
                  t_start::FT=FT(0.0),
                  t_end::FT=FT(1.0e-3),
                  dt::FT=FT(1.0e-9),
                  kwargs...) where {FT<:AbstractFloat}
    # Create a default config with provided dimensions and time params
    config = SimulationConfig{FT}(;
        NR=NR,
        NZ=NZ,
        t_start_s=t_start,
        t_end_s=t_end,
        dt=dt,
        kwargs...
    )

    # Use the primary constructor
    return RAPID{FT}(config)
end

# Type-inferring constructor
RAPID(NR::Int, NZ::Int; kwargs...) = RAPID{Float64}(NR, NZ; kwargs...)
RAPID(config::SimulationConfig{FT}) where {FT<:AbstractFloat} = RAPID{FT}(config)

# Export types
export SimulationConfig, WallGeometry, PlasmaState, Fields, Transport, Operators, SimulationFlags, RAPID, GridGeometry, NodeState