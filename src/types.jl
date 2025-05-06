"""
Type definitions for RAPID2D.jl
"""

# Importing the PlasmaConstants from constants.jl
import RAPID2D: PlasmaConstants

"""
    SimulationConfig{FT<:AbstractFloat}

Contains simulation configuration parameters.
"""
mutable struct SimulationConfig{FT<:AbstractFloat}
    # Device parameters
    device_Name::String         # Device name
    shot_Name::String           # Shot name

    # Physical constants
    constants::PlasmaConstants{FT}  # Consolidated physical constants

    # Legacy physical constants (for backward compatibility)
    ee::FT                       # Elementary charge (C)
    me::FT                       # Electron mass (kg)
    mi::FT                       # Ion mass (kg)
    eps0::FT                     # Vacuum permittivity (F/m)
    mu0::FT                      # Vacuum permeability (H/m)
    kB::FT                       # Boltzmann constant (J/K)

    # Field configuration
    R0B0::FT                     # On-axis R0*B0 value

    # Initial conditions
    prefilled_gas_pressure::FT   # Prefilled gas pressure (Pa)

    # Limits
    min_Te::FT                   # Minimum electron temperature (eV)
    max_Te::FT                   # Maximum electron temperature (eV)

    # Transport parameters
    Dpara0::FT                   # Base parallel diffusion coefficient
    Dperp0::FT                   # Base perpendicular diffusion coefficient

    # Paths
    Input_path::String          # Path to input files
    Output_path::String         # Path to output files
    Output_prefix::String       # Prefix for output files
    Output_name::String         # Name for output files

    # Output intervals
    snap1D_Interval_s::FT        # Time interval for 1D snapshots
    snap2D_Interval_s::FT        # Time interval for 2D snapshots
    write_File_Interval_s::FT    # Time interval for file writing

    function SimulationConfig{FT}() where FT<:AbstractFloat
        # Create default PlasmaConstants object
        constants = PlasmaConstants{FT}()

        return new{FT}(
            "manual",         # device_Name
            "test",           # shot_Name

            constants,        # constants (initialized default)

            FT(1.602176634e-19),  # ee
            FT(9.1093837015e-31), # me
            FT(3.34754699166e-27),# mi (H2+ ion mass)
            FT(8.8541878128e-12), # eps0
            FT(1.25663706212e-6), # mu0
            FT(1.380649e-23),     # kB

            FT(2.0),              # R0B0

            FT(0.1),              # prefilled_gas_pressure (Pa)

            FT(0.05),             # min_Te (eV)
            FT(100.0),            # max_Te (eV)

            FT(1.0),              # Dpara0 (m²/s)
            FT(0.1),              # Dperp0 (m²/s)

            "./input",           # Input_path
            "./output",          # Output_path
            "",                  # Output_prefix
            "RAPID2D",           # Output_name

            FT(1.0e-5),           # snap1D_Interval_s
            FT(1.0e-4),           # snap2D_Interval_s
            FT(1.0e-3)            # write_File_Interval_s
        )
    end

    # Default constructor for non-parameterized use
    function SimulationConfig()
        return SimulationConfig{Float64}()
    end
end

"""
    WallGeometry{FT<:AbstractFloat}

Represents the geometry of the device wall.

Fields:
- `R`: Radial coordinates of wall points
- `Z`: Vertical coordinates of wall points
"""
struct WallGeometry{FT<:AbstractFloat}
    R::Vector{FT}  # Radial coordinates
    Z::Vector{FT}  # Vertical coordinates
end

"""
    PlasmaState{FT<:AbstractFloat}

Contains the plasma state variables.

Fields include density, temperature, and velocity components for electrons and ions.
"""
mutable struct PlasmaState{FT<:AbstractFloat}
    # Densities
    ne::Matrix{FT}      # Electron density [m^-3]
    ni::Matrix{FT}      # Ion density [m^-3]
    n_H2_gas::Matrix{FT}  # H2 gas density [m^-3]

    # Temperatures
    Te_eV::Matrix{FT}    # Electron temperature [eV]
    Ti_eV::Matrix{FT}    # Ion temperature [eV]
    T_gas_eV::FT         # Gas temperature [eV]

    # Velocities - parallel components
    ue_para::Matrix{FT}  # Electron parallel velocity [m/s]
    ui_para::Matrix{FT}  # Ion parallel velocity [m/s]

    # Velocities - vector components
    ueR::Matrix{FT}     # Electron R velocity [m/s]
    ueZ::Matrix{FT}     # Electron Z velocity [m/s]
    ueϕ::Matrix{FT}   # Electron phi velocity [m/s]

    uiR::Matrix{FT}     # Ion R velocity [m/s]
    uiZ::Matrix{FT}     # Ion Z velocity [m/s]
    uiϕ::Matrix{FT}   # Ion phi velocity [m/s]

    # Collision parameters
    lnA::Matrix{FT}     # Coulomb logarithm
    nu_ei::Matrix{FT}   # Electron-ion collision frequency [1/s]
    sptz_fac::Matrix{FT} # Spitzer factor for conductivity

    # Constructor
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
mutable struct Fields{FT<:AbstractFloat}
    # Magnetic field components
    BR::Matrix{FT}        # Radial magnetic field [T]
    BZ::Matrix{FT}        # Vertical magnetic field [T]
    Bϕ::Matrix{FT}      # Toroidal magnetic field [T]

    # Derived magnetic field quantities
    Bpol::Matrix{FT}      # Poloidal magnetic field [T]
    Btot::Matrix{FT}      # Total magnetic field [T]

    # Magnetic field unit vectors
    bR::Matrix{FT}        # Radial unit vector
    bZ::Matrix{FT}        # Vertical unit vector
    bϕ::Matrix{FT}      # Toroidal unit vector

    # Vacuum magnetic field components
    BR_vac::Matrix{FT}    # Vacuum radial magnetic field [T]
    BZ_vac::Matrix{FT}    # Vacuum vertical magnetic field [T]

    # Electric field components
    ER::Matrix{FT}        # Radial electric field [V/m]
    EZ::Matrix{FT}        # Vertical electric field [V/m]
    Eϕ::Matrix{FT}      # Toroidal electric field [V/m]

    # Parallel electric fields
    E_para_vac::Matrix{FT}  # Vacuum parallel electric field [V/m]
    E_para_ind::Matrix{FT}  # Induced parallel electric field [V/m]
    E_para_tot::Matrix{FT}  # Total parallel electric field [V/m]

    # Flux quantities
    psi::Matrix{FT}       # Poloidal flux [Wb/rad]
    psi_vac::Matrix{FT}   # Vacuum poloidal flux [Wb/rad]

    # Constructor
    function Fields{FT}(NR::Int, NZ::Int) where FT<:AbstractFloat
        # Pre-allocate arrays
        BR = zeros(FT, NZ, NR)
        BZ = zeros(FT, NZ, NR)
        Bϕ = zeros(FT, NZ, NR)

        Bpol = zeros(FT, NZ, NR)
        Btot = zeros(FT, NZ, NR)

        bR = zeros(FT, NZ, NR)
        bZ = zeros(FT, NZ, NR)
        bϕ = zeros(FT, NZ, NR)

        BR_vac = zeros(FT, NZ, NR)
        BZ_vac = zeros(FT, NZ, NR)

        ER = zeros(FT, NZ, NR)
        EZ = zeros(FT, NZ, NR)
        Eϕ = zeros(FT, NZ, NR)

        E_para_vac = zeros(FT, NZ, NR)
        E_para_ind = zeros(FT, NZ, NR)
        E_para_tot = zeros(FT, NZ, NR)

        psi = zeros(FT, NZ, NR)
        psi_vac = zeros(FT, NZ, NR)

        return new{FT}(
            BR, BZ, Bϕ,
            Bpol, Btot,
            bR, bZ, bϕ,
            BR_vac, BZ_vac,
            ER, EZ, Eϕ,
            E_para_vac, E_para_ind, E_para_tot,
            psi, psi_vac
        )
    end
end

"""
    Transport{FT<:AbstractFloat}

Contains the transport coefficients for the plasma.

Fields include diffusion coefficients in different directions.
"""
mutable struct Transport{FT<:AbstractFloat}
    # Base diffusivity values
    Dpara0::FT            # Base parallel diffusion coefficient [m²/s]
    Dperp0::FT            # Base perpendicular diffusion coefficient [m²/s]

    # Spatially-varying diffusion coefficients
    Dpara::Matrix{FT}     # Parallel diffusion coefficient [m²/s]
    Dperp::Matrix{FT}     # Perpendicular diffusion coefficient [m²/s]

    # Diffusion tensor components
    DRR::Matrix{FT}       # R-R component of diffusion tensor
    DRZ::Matrix{FT}       # R-Z component of diffusion tensor
    DZZ::Matrix{FT}       # Z-Z component of diffusion tensor

    # Constructor
    function Transport{FT}(NR::Int, NZ::Int) where FT<:AbstractFloat
        # Pre-allocate arrays
        Dpara = zeros(FT, NZ, NR)
        Dperp = zeros(FT, NZ, NR)

        DRR = zeros(FT, NZ, NR)
        DRZ = zeros(FT, NZ, NR)
        DZZ = zeros(FT, NZ, NR)

        return new{FT}(
            FT(1.0),    # Dpara0 default
            FT(0.1),    # Dperp0 default
            Dpara, Dperp,
            DRR, DRZ, DZZ
        )
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
mutable struct SimulationFlags
    # Physics flags
    diffu::Bool        # Enable diffusion
    convec::Bool       # Enable convection
    Ampere::Bool       # Enable Ampere's law (magnetic field update)
    Implicit::Bool     # Use implicit methods

    # Numerical settings
    Ampere_nstep::Int  # Frequency of Ampere's law updates

    # Constructor with defaults
    function SimulationFlags()
        return new(
            true,   # diffu
            true,   # convec
            false,  # Ampere
            true,   # Implicit
            10      # Ampere_nstep
        )
    end
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

        # Create empty wall data (will be properly initialized later)
        RP.wall = WallGeometry{FT}(Vector{FT}(), Vector{FT}())
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

        return RP
    end
end

# Export types
export SimulationConfig, WallGeometry, PlasmaState, Fields, Transport, Operators, SimulationFlags, RAPID, GridGeometry, NodeState