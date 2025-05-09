"""
Type definitions for RAPID2D.jl
"""

# Importing the PlasmaConstants from constants.jl
import RAPID2D: PlasmaConstants

"""
    SimulationConfig{FT<:AbstractFloat}

Contains simulation configuration parameters.
"""
Base.@kwdef mutable struct SimulationConfig{FT<:AbstractFloat}
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
Base.@kwdef struct WallGeometry{FT<:AbstractFloat}
    R::Vector{FT} = Vector{FT}()  # Radial coordinates
    Z::Vector{FT} = Vector{FT}()  # Vertical coordinates

    # Constructor ensuring the wall is a closed loop
    function WallGeometry{FT}(R::Vector{FT}, Z::Vector{FT}) where {FT<:AbstractFloat}
        @assert length(R) == length(Z) "R and Z must have the same length"
        @assert length(R) >= 3 "At least 3 points needed to define a wall unless creating an empty placeholder"

        # Check if the wall is closed (first point equals last point)
        if R[1] != R[end] || Z[1] != Z[end]
            # Add the first point at the end to close the loop
            push!(R, R[1])
            push!(Z, Z[1])
        end

        new{FT}(R, Z)
    end
end

# Constructor that infers the floating-point type
WallGeometry(R::Vector{FT}, Z::Vector{FT}) where {FT<:AbstractFloat} = WallGeometry{FT}(R, Z)

"""
    PlasmaState{FT<:AbstractFloat}

Contains the plasma state variables.

Fields include density, temperature, and velocity components for electrons and ions.
"""
Base.@kwdef mutable struct PlasmaState{FT<:AbstractFloat}
    # Densities
    ne::Matrix{FT} = Matrix{FT}(undef, 0, 0)      # Electron density [m^-3]
    ni::Matrix{FT} = Matrix{FT}(undef, 0, 0)      # Ion density [m^-3]
    n_H2_gas::Matrix{FT} = Matrix{FT}(undef, 0, 0)  # H2 gas density [m^-3]

    # Temperatures
    Te_eV::Matrix{FT} = Matrix{FT}(undef, 0, 0)    # Electron temperature [eV]
    Ti_eV::Matrix{FT} = Matrix{FT}(undef, 0, 0)    # Ion temperature [eV]
    T_gas_eV::FT = FT(0.026)         # Gas temperature [eV], defaults to room temperature ~300K

    # Velocities - parallel components
    ue_para::Matrix{FT} = Matrix{FT}(undef, 0, 0)  # Electron parallel velocity [m/s]
    ui_para::Matrix{FT} = Matrix{FT}(undef, 0, 0)  # Ion parallel velocity [m/s]

    # Velocities - vector components
    ueR::Matrix{FT} = Matrix{FT}(undef, 0, 0)     # Electron R velocity [m/s]
    ueZ::Matrix{FT} = Matrix{FT}(undef, 0, 0)     # Electron Z velocity [m/s]
    ueϕ::Matrix{FT} = Matrix{FT}(undef, 0, 0)   # Electron ϕ velocity [m/s]

    uiR::Matrix{FT} = Matrix{FT}(undef, 0, 0)     # Ion R velocity [m/s]
    uiZ::Matrix{FT} = Matrix{FT}(undef, 0, 0)     # Ion Z velocity [m/s]
    uiϕ::Matrix{FT} = Matrix{FT}(undef, 0, 0)   # Ion ϕ velocity [m/s]

    # Collision parameters
    lnA::Matrix{FT} = Matrix{FT}(undef, 0, 0)     # Coulomb logarithm
    nu_ei::Matrix{FT} = Matrix{FT}(undef, 0, 0)   # Electron-ion collision frequency [1/s]
    sptz_fac::Matrix{FT} = Matrix{FT}(undef, 0, 0) # Spitzer factor for conductivity

    # Constructor for matrices with specific dimensions
    function PlasmaState{FT}(NR::Int, NZ::Int) where FT<:AbstractFloat
        # Pre-allocate arrays
        ne = zeros(FT, NZ, NR)
        ni = zeros(FT, NZ, NR)
        n_H2_gas = zeros(FT, NZ, NR)

        Te_eV = zeros(FT, NZ, NR)
        Ti_eV = zeros(FT, NZ, NR)
        T_gas_eV = FT(0.026)  # Room temperature ~300K

        ue_para = zeros(FT, NZ, NR)
        ui_para = zeros(FT, NZ, NR)

        ueR = zeros(FT, NZ, NR)
        ueZ = zeros(FT, NZ, NR)
        ueϕ = zeros(FT, NZ, NR)

        uiR = zeros(FT, NZ, NR)
        uiZ = zeros(FT, NZ, NR)
        uiϕ = zeros(FT, NZ, NR)

        lnA = zeros(FT, NZ, NR)
        nu_ei = zeros(FT, NZ, NR)
        sptz_fac = zeros(FT, NZ, NR)

        return new{FT}(
            ne, ni, n_H2_gas,
            Te_eV, Ti_eV, T_gas_eV,
            ue_para, ui_para,
            ueR, ueZ, ueϕ,
            uiR, uiZ, uiϕ,
            lnA, nu_ei, sptz_fac
        )
    end
end

"""
    Fields{FT<:AbstractFloat}

Contains the electromagnetic field variables.

Fields include components of the magnetic and electric fields.
"""
Base.@kwdef mutable struct Fields{FT<:AbstractFloat}
    # External fields
    BR_ext::Matrix{FT} = Matrix{FT}(undef, 0, 0)       # External radial magnetic field [T]
    BZ_ext::Matrix{FT} = Matrix{FT}(undef, 0, 0)       # External vertical magnetic field [T]
    LV_ext::Matrix{FT} = Matrix{FT}(undef, 0, 0)       # External Loop Voltage [V]
    psi_ext::Matrix{FT} = Matrix{FT}(undef, 0, 0)      # External magnetic flux [Wb/rad]
    Eϕ_ext::Matrix{FT} = Matrix{FT}(undef, 0, 0)     # External toroidal electric field [V/m]
    E_para_ext::Matrix{FT} = Matrix{FT}(undef, 0, 0)   # External parallel electric field [V/m]

    # Self-generated fields
    BR_self::Matrix{FT} = Matrix{FT}(undef, 0, 0)      # Self-generated radial magnetic field [T]
    BZ_self::Matrix{FT} = Matrix{FT}(undef, 0, 0)      # Self-generated vertical magnetic field [T]
    psi_self::Matrix{FT} = Matrix{FT}(undef, 0, 0)     # Self-generated magnetic flux [Wb/rad]
    Eϕ_self::Matrix{FT} = Matrix{FT}(undef, 0, 0)    # Self-generated toroidal electric field [V/m]
    E_para_self_ES::Matrix{FT} = Matrix{FT}(undef, 0, 0) # Electrostatic self-generated parallel electric field [V/m]
    E_para_self_EM::Matrix{FT} = Matrix{FT}(undef, 0, 0) # Electromagnetic self-generated parallel electric field [V/m]

    # Total fields - external + self-generated
    BR::Matrix{FT} = Matrix{FT}(undef, 0, 0)        # Total radial magnetic field [T]
    BZ::Matrix{FT} = Matrix{FT}(undef, 0, 0)        # Total vertical magnetic field [T]
    Bϕ::Matrix{FT} = Matrix{FT}(undef, 0, 0)        # Toroidal magnetic field [T]

    # Derived field quantities
    Bpol::Matrix{FT} = Matrix{FT}(undef, 0, 0)      # Poloidal magnetic field [T]
    Btot::Matrix{FT} = Matrix{FT}(undef, 0, 0)      # Total magnetic field [T]

    # Magnetic field unit vectors
    bR::Matrix{FT} = Matrix{FT}(undef, 0, 0)        # Radial unit vector
    bZ::Matrix{FT} = Matrix{FT}(undef, 0, 0)        # Vertical unit vector
    bϕ::Matrix{FT} = Matrix{FT}(undef, 0, 0)        # Toroidal unit vector

    # Electric field components
    ER::Matrix{FT} = Matrix{FT}(undef, 0, 0)        # Radial electric field [V/m]
    EZ::Matrix{FT} = Matrix{FT}(undef, 0, 0)        # Vertical electric field [V/m]
    Eϕ::Matrix{FT} = Matrix{FT}(undef, 0, 0)        # Toroidal electric field [V/m]

    # Parallel electric field
    E_para_ind::Matrix{FT} = Matrix{FT}(undef, 0, 0)  # Induced parallel electric field [V/m]
    E_para_tot::Matrix{FT} = Matrix{FT}(undef, 0, 0)  # Total parallel electric field [V/m]

    # Magnetic flux
    psi::Matrix{FT} = Matrix{FT}(undef, 0, 0)       # Total magnetic flux [Wb/rad]

    # Constructor for matrices with specific dimensions
    function Fields{FT}(NR::Int, NZ::Int) where FT<:AbstractFloat
        # Initialize all fields with zeros
        matrices = [zeros(FT, NZ, NR) for _ in 1:27]

        return new{FT}(
            matrices[1], matrices[2], matrices[3], matrices[4], matrices[5], matrices[6],
            matrices[7], matrices[8], matrices[9], matrices[10], matrices[11], matrices[12],
            matrices[13], matrices[14], matrices[15],
            matrices[16], matrices[17],
            matrices[18], matrices[19], matrices[20],
            matrices[21], matrices[22], matrices[23],
            matrices[24], matrices[25],
            matrices[26]
        )
    end
end

"""
    Transport{FT<:AbstractFloat}

Contains the transport coefficients for the plasma.

Fields include diffusion coefficients in different directions.
"""
Base.@kwdef mutable struct Transport{FT<:AbstractFloat}
    # Base diffusivity values
    Dpara0::FT = FT(1.0)            # Base parallel diffusion coefficient [m²/s]
    Dperp0::FT = FT(0.1)            # Base perpendicular diffusion coefficient [m²/s]

    # Spatially-varying diffusion coefficients
    Dpara::Matrix{FT} = Matrix{FT}(undef, 0, 0)  # Parallel diffusion coefficient [m²/s]
    Dperp::Matrix{FT} = Matrix{FT}(undef, 0, 0)  # Perpendicular diffusion coefficient [m²/s]

    # Diffusion tensor components
    DRR::Matrix{FT} = Matrix{FT}(undef, 0, 0)    # R-R component of diffusion tensor
    DRZ::Matrix{FT} = Matrix{FT}(undef, 0, 0)    # R-Z component of diffusion tensor
    DZZ::Matrix{FT} = Matrix{FT}(undef, 0, 0)    # Z-Z component of diffusion tensor

    # Constructor for matrices with specific dimensions
    function Transport{FT}(NR::Int, NZ::Int) where FT<:AbstractFloat
        # Pre-allocate arrays
        Dpara = zeros(FT, NZ, NR)
        Dperp = zeros(FT, NZ, NR)

        DRR = zeros(FT, NZ, NR)
        DRZ = zeros(FT, NZ, NR)
        DZZ = zeros(FT, NZ, NR)

        return new{FT}(FT(1.0), FT(0.1), Dpara, Dperp, DRR, DRZ, DZZ)
    end
end

"""
    Operators{FT<:AbstractFloat}

Contains the numerical operators used in the simulation.

Fields include various matrices for solving different parts of the model.
"""
mutable struct Operators{FT<:AbstractFloat}
    # Operators for solving equations
    A_GS::SparseMatrixCSC{FT, Int}  # Matrix for Grad-Shafranov equation

    # Constructor
    function Operators{FT}(NR::Int, NZ::Int) where FT<:AbstractFloat
        # Initialize with empty sparse matrix of correct type
        # Will be populated later when needed
        A_GS = spzeros(FT, NR*NZ, NR*NZ)

        return new{FT}(A_GS)
    end
end

"""
    SimulationFlags

Contains boolean flags that control various aspects of the simulation.
"""
Base.@kwdef mutable struct SimulationFlags
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
    function NodeState{FT}(NZ::Int, NR::Int) where FT<:AbstractFloat
        rid = zeros(Int, NZ, NR)
        zid = zeros(Int, NZ, NR)
        nid = zeros(Int, NZ, NR)
        state = fill(NaN, NZ, NR)

        return new{FT}(rid, zid, nid, state, Int[], Int[], Int[])
    end

    # Convenience constructor
    function NodeState(NZ::Int, NR::Int)
        return NodeState{Float64}(NZ, NR)
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
        R2D = zeros(FT, NZ, NR)
        Z2D = zeros(FT, NZ, NR)
        Jacob = zeros(FT, NZ, NR)
        inv_Jacob = zeros(FT, NZ, NR)
		inVol2D = zeros(FT, NZ, NR)
        BDY_idx = Int[]
        nodes = NodeState{FT}(NZ, NR)

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

The main simulation structure containing all simulation data.

Fields include grid information, physical fields, and simulation state.
"""
mutable struct RAPID{FT<:AbstractFloat}
    # Grid geometry
    G::GridGeometry{FT}         # Grid geometry containing all grid-related properties

    # Wall geometry
    wall::WallGeometry{FT}       # Wall geometry data
    damping_func::Matrix{FT}     # Damping function outside wall

    # Grid masks
    cell_state::Matrix{Int}      # Cell state (1 inside wall, -1 outside)
    in_wall_nids::Vector{Int}     # Linear indices of cells inside wall
    out_wall_nids::Vector{Int}    # Linear indices of cells outside wall

    # Volume elements
    device_inVolume::FT          # Total volume inside wall

    # External field source
    external_field::Union{Nothing, AbstractExternalField{FT}}  # External electromagnetic field source

    # Reaction rate coefficients
    eRRC::Dict{Symbol, Any}     # Electron reaction rate coefficients
    iRRC::Dict{Symbol, Any}     # Ion reaction rate coefficients

    # Physical components
    config::SimulationConfig{FT} # Simulation configuration
    flags::SimulationFlags      # Simulation flags
    plasma::PlasmaState{FT}     # Plasma state variables
    fields::Fields{FT}           # Field variables
    transport::Transport{FT}     # Transport coefficients
    operators::Operators{FT}     # Numerical operators

    # Time evolution
    step::Int                   # Current time step
    time_s::FT                   # Current time [s]
    t_start_s::FT                # Start time [s]
    t_end_s::FT                  # End time [s]
    dt::FT                       # Time step [s]

    # Previous state for time stepping
    prev_n::Matrix{FT}           # Previous density

    # Performance tracking
    tElap::Dict{Symbol, Float64} # Elapsed times for different parts

    # Diagnostics
    diagnostics::Dict{Symbol, Any} # Diagnostic data

    # Constructor
    function RAPID{FT}(NR::Int, NZ::Int;
                      t_start::FT=FT(0.0),
                      t_end::FT=FT(1.0e-3),
                      dt::FT=FT(1.0e-9)) where FT<:AbstractFloat
        # Create a new RAPID instance
        RP = new{FT}()

        # Initialize grid geometry
        RP.G = GridGeometry{FT}(NR, NZ)

        # Initialize time parameters
        RP.step = 0
        RP.time_s = t_start
        RP.t_start_s = t_start
        RP.t_end_s = t_end
        RP.dt = dt

        # Create default configuration
        RP.config = SimulationConfig{FT}()

        # Create flags with defaults
        RP.flags = SimulationFlags()

        # Initialize diagnostics
        RP.diagnostics = Dict{Symbol, Any}()

        # Create emtpy wall geometry
        RP.wall = WallGeometry{FT}()
        RP.damping_func = zeros(FT, RP.G.NZ, RP.G.NR)

        # Initialize grid masks with empty or zero-filled arrays
        RP.cell_state = zeros(Int, RP.G.NZ, RP.G.NR)
        RP.in_wall_nids = Vector{Int}()
        RP.out_wall_nids = Vector{Int}()

        # Initialize volume elements
        RP.device_inVolume = FT(0.0)

        # Initialize physical state objects
        RP.plasma = PlasmaState{FT}(RP.G.NR, RP.G.NZ)
        RP.fields = Fields{FT}(RP.G.NR, RP.G.NZ)
        RP.transport = Transport{FT}(RP.G.NR, RP.G.NZ)
        RP.operators = Operators{FT}(RP.G.NR, RP.G.NZ)

        # Initialize previous state
        RP.prev_n = zeros(FT, RP.G.NZ, RP.G.NR)

        # Empty dictionaries
        RP.eRRC = Dict{Symbol, Any}()
        RP.iRRC = Dict{Symbol, Any}()
        RP.tElap = Dict{Symbol, Float64}()

        # Set external field to nothing initially
        RP.external_field = nothing

        return RP
    end

    # Constructor from SimulationConfig
    function RAPID{FT}(config::SimulationConfig{FT}) where FT<:AbstractFloat
        # Create a new RAPID instance
        RP = new{FT}()

        # Extract grid dimensions from config
        NR = config.NR
        NZ = config.NZ

        # Initialize grid geometry
        RP.G = GridGeometry{FT}(NR, NZ)

        # Initialize time parameters from config
        RP.step = 0
        RP.time_s = config.t_start_s
        RP.t_start_s = config.t_start_s
        RP.t_end_s = config.t_end_s
        RP.dt = config.dt

        # Store the provided configuration
        RP.config = config

        # Create flags with defaults
        RP.flags = SimulationFlags()

        # Initialize diagnostics
        RP.diagnostics = Dict{Symbol, Any}()

        # Create empty wall geometry
        RP.wall = WallGeometry{FT}()
        RP.damping_func = zeros(FT, RP.G.NZ, RP.G.NR)

        # Initialize grid masks with empty or zero-filled arrays
        RP.cell_state = zeros(Int, RP.G.NZ, RP.G.NR)
        RP.in_wall_nids = Vector{Int}()
        RP.out_wall_nids = Vector{Int}()

        # Initialize volume elements
        RP.device_inVolume = FT(0.0)

        # Initialize physical state objects
        RP.plasma = PlasmaState{FT}(RP.G.NR, RP.G.NZ)
        RP.fields = Fields{FT}(RP.G.NR, RP.G.NZ)
        RP.transport = Transport{FT}(RP.G.NR, RP.G.NZ)

        # Initialize transport parameters from config
        RP.transport.Dpara0 = config.Dpara0
        RP.transport.Dperp0 = config.Dperp0

        RP.operators = Operators{FT}(RP.G.NR, RP.G.NZ)

        # Initialize previous state
        RP.prev_n = zeros(FT, RP.G.NZ, RP.G.NR)

        # Empty dictionaries
        RP.eRRC = Dict{Symbol, Any}()
        RP.iRRC = Dict{Symbol, Any}()
        RP.tElap = Dict{Symbol, Float64}()

        # Set external field to nothing initially
        RP.external_field = nothing

        return RP
    end
end

# Export types
export SimulationConfig, WallGeometry, PlasmaState, Fields, Transport, Operators, SimulationFlags, RAPID, GridGeometry, NodeState