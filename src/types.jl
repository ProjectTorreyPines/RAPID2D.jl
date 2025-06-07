"""
Type definitions for RAPID2D.jl
"""

# Importing the PlasmaConstants from constants.jl
import RAPID2D: PlasmaConstants


include("diagnostics/types.jl")
include("io/types.jl")
include("utils/types.jl")


# Abstract types for reaction rate coefficients for a specific species
"""
    AbstractElectronRRCs{T<:AbstractFloat}
"""
abstract type AbstractSpeciesRRCs{FT<:AbstractFloat} end

"""
    SimulationConfig{FT<:AbstractFloat}

Contains simulation configuration parameters.
"""
@kwdef mutable struct SimulationConfig{FT<:AbstractFloat}
    # Paths
    Input_path::String = "./input"     # Path to input files
    Output_path::String = "./output"   # Path to output files
    Output_prefix::String = ""         # Prefix for output files
    Output_name::String = "RAPID2D"    # Name for output files

    # Device parameters
    device_Name::String = "manual"     # Device name
    shot_Name::String = "test"         # Shot name

    # Grid dimensions
    NR::Int = 50                       # Number of radial grid points
    NZ::Int = 100                      # Number of vertical grid points
    R_min::Union{FT,Nothing} = nothing       # Minimum radial coordinate
    R_max::Union{FT,Nothing} = nothing       # Maximum radial coordinate
    Z_min::Union{FT,Nothing} = nothing       # Minimum vertical coordinate
    Z_max::Union{FT,Nothing} = nothing       # Maximum vertical coordinate

    # Time parameters
    t_start_s::FT = FT(0.0)            # Simulation start time [s]
    t_end_s::FT = FT(1.0e-3)           # Simulation end time [s]
    dt::FT = FT(10e-6)                # Time step [s]

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
    R0B0::Union{FT,Nothing} = nothing                 # On-axis R0*B0 value

    # Initial conditions
    prefilled_gas_pressure::Union{FT,Nothing} = nothing  # Prefilled gas pressure (Pa)

    # Limits
    min_Te::FT = FT(0.001)              # Minimum electron temperature (eV)
    max_Te::FT = FT(500.0)             # Maximum electron temperature (eV)

    # Transport parameters
    Dpara0::FT = FT(0.0)               # Base parallel diffusion coefficient
    Dperp0::FT = FT(0.0)               # Base perpendicular diffusion coefficient

    turbulent_diffusion_fraction_along_bpol::FT = FT(0.9)  # Fraction of turbulent diffusion along poloidal field lines

    # Output intervals
    snap0D_Δt_s::FT = FT(20e-6)  # Time interval for 1D snapshots
    snap2D_Δt_s::FT = FT(100e-6)  # Time interval for 2D snapshots
    write_File_Interval_s::FT = FT(1e-3)  # Time interval for file writing

    # Wall geometry
    wall_R::Vector{FT}  = Vector{FT}()  # Radial coordinates of wall points
    wall_Z::Vector{FT}  = Vector{FT}()  # Vertical coordinates of wall points
end

"""
    WallGeometry{FT<:AbstractFloat}

Represents the geometry of the device wall.

Fields:
- `R`: Radial coordinates of wall points
- `Z`: Vertical coordinates of wall points
"""
struct WallGeometry{FT<:AbstractFloat}
    R::Vector{FT}
    Z::Vector{FT}

    function WallGeometry{FT}() where {FT<:AbstractFloat}
        return new{FT}(FT[], FT[])
    end

    # Custom constructor that ensures valid wall geometry
    function WallGeometry{FT}(R::Vector{FT}, Z::Vector{FT}) where {FT<:AbstractFloat}
        @assert length(R) == length(Z) "R and Z must have the same length"
        @assert length(R) >= 3 "At least 3 points needed to define a wall unless creating an empty placeholder"

        new_R, new_Z = copy(R), copy(Z)
        if new_R[1] != new_R[end] || new_Z[1] != new_Z[end]
            push!(new_R, new_R[1])
            push!(new_Z, new_Z[1])
        end
        return new{FT}(new_R, new_Z)
    end
end

function WallGeometry(R::Vector{FT}, Z::Vector{FT}) where {FT<:AbstractFloat}
    return WallGeometry{FT}(R, Z)
end


"""
    ElectronHeatingPowers{FT<:AbstractFloat}

Contains the power terms for electron energy equation.

# Fields
- `tot`: Total power density [W/m³]
- `drag`: Power from drag forces [W/m³]
- `conv`: Power from convective transport [W/m³]
- `diffu`: Power from diffusive transport [W/m³]
- `heat`: Power from heating sources (e.g., ohmic) [W/m³]
- `iz`: Power from ionization [W/m³]
- `exc`: Power from excitation [W/m³]
- `dilution`: Power from density dilution [W/m³]
- `equi`: Power from temperature equilibration [W/m³]
"""
@kwdef mutable struct ElectronHeatingPowers{FT<:AbstractFloat}
    dims::Tuple{Int,Int}  # Grid dimensions (NR, NZ)

    # Power terms - all in W/m³
    tot::Matrix{FT} = zeros(FT, dims)        # Total power density
    drag::Matrix{FT} = zeros(FT, dims)       # Power from drag forces
    conv::Matrix{FT} = zeros(FT, dims)       # Power from convective transport
    diffu::Matrix{FT} = zeros(FT, dims)      # Power from diffusive transport
    heat::Matrix{FT} = zeros(FT, dims)       # Power from heating (q)
    iz::Matrix{FT} = zeros(FT, dims)         # Power from ionization
    exc::Matrix{FT} = zeros(FT, dims)        # Power from excitation
    dilution::Matrix{FT} = zeros(FT, dims)   # Power from density dilution
    equi::Matrix{FT} = zeros(FT, dims)       # Power from temperature equilibration
end

# Constructor with dimensions
function ElectronHeatingPowers{FT}(dimensions::Tuple{Int,Int}) where {FT<:AbstractFloat}
    return ElectronHeatingPowers{FT}(dims=dimensions)
end
function ElectronHeatingPowers{FT}(NR::Int, NZ::Int) where {FT<:AbstractFloat}
    return ElectronHeatingPowers{FT}(dims=(NR, NZ))
end

"""
    IonHeatingPowers{FT<:AbstractFloat}

Contains the power terms for ion energy equation.

# Fields
- `tot`: Total power density [W/m³]
- `atomic`: Power from atomic processes [W/m³]
- `equi`: Power from temperature equilibration [W/m³]
"""
@kwdef mutable struct IonHeatingPowers{FT<:AbstractFloat}
    dims::Tuple{Int,Int}  # Grid dimensions (NR, NZ)

    # Power terms - all in W/m³
    tot::Matrix{FT} = zeros(FT, dims)        # Total power density
    atomic::Matrix{FT} = zeros(FT, dims)     # Power from atomic processes
    equi::Matrix{FT} = zeros(FT, dims)       # Power from temperature equilibration
end

# Constructor with dimensions
function IonHeatingPowers{FT}(dimensions::Tuple{Int,Int}) where {FT<:AbstractFloat}
    return IonHeatingPowers{FT}(dims=dimensions)
end
function IonHeatingPowers{FT}(NR::Int, NZ::Int) where {FT<:AbstractFloat}
    return IonHeatingPowers{FT}(dims=(NR, NZ))
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

    # mean ExB transport
    mean_ExB_R::Matrix{FT} = zeros(FT, dims) # Mean ExB drift R component [m/s]
    mean_ExB_Z::Matrix{FT} = zeros(FT, dims) # Mean ExB drift Z component [m/s]

    # Parameters for Self-E field effects
    nc_para::Matrix{FT} = zeros(FT, dims) # Parallel critical density [m^-3]
    nc_perp::Matrix{FT} = zeros(FT, dims) # Perpendicular critical density [m^-3]
    γ_shape_fac::Matrix{FT} = zeros(FT, dims) # shape factor of plasma

    # Collision parameters
    lnΛ::Matrix{FT} = zeros(FT, dims)   # Coulomb logarithm
    ν_ei::Matrix{FT} = zeros(FT, dims) # Electron-ion collision frequency [1/s]
    sptz_fac::Matrix{FT} = zeros(FT, dims) # Spitzer factor for conductivity
    Rue_ei::Matrix{FT} = zeros(FT, dims) # ue change rate by electron-ion collision

    Zeff::Matrix{FT} = ones(FT, dims) # Effective ion charge

    # Current densities
    Jϕ::Matrix{FT} = zeros(FT, dims)    # Toroidal current density [A/m²]

    ν_iz::Matrix{FT} = zeros(FT, dims) # Electron ionization rate [1/s]

    # Power sources/sinks - using new struct-based approach
    ePowers::ElectronHeatingPowers{FT} = ElectronHeatingPowers{FT}(dims)
    iPowers::IonHeatingPowers{FT} = IonHeatingPowers{FT}(dims)
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

    # R0B0
    R0B0::FT = FT(0.0)

    # External fields
    BR_ext::Matrix{FT} = zeros(FT, dims)        # External radial magnetic field [T]
    BZ_ext::Matrix{FT} = zeros(FT, dims)        # External vertical magnetic field [T]
    LV_ext::Matrix{FT} = zeros(FT, dims)        # External Loop Voltage [V]
    ψ_ext::Matrix{FT} = zeros(FT, dims)         # External magnetic flux [Wb/rad]
    Eϕ_ext::Matrix{FT} = zeros(FT, dims)        # External toroidal electric field [V/m]
    E_para_ext::Matrix{FT} = zeros(FT, dims)    # External parallel electric field [V/m]

    # Self-generated fields
    BR_self::Matrix{FT} = zeros(FT, dims)       # Self-generated radial magnetic field [T]
    BZ_self::Matrix{FT} = zeros(FT, dims)       # Self-generated vertical magnetic field [T]
    ψ_self::Matrix{FT} = zeros(FT, dims)      # Self-generated magnetic flux [Wb/rad]
    Eϕ_self::Matrix{FT} = zeros(FT, dims)       # Self-generated toroidal electric field [V/m]
    Epol_self::Matrix{FT} = zeros(FT, dims)       # Self-generated poloidal electric field [V/m]
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

    bpol_R::Matrix{FT} = zeros(FT, dims)        # Radial component of poloidal unit vector
    bpol_Z::Matrix{FT} = zeros(FT, dims)        # Radial component of poloidal unit vector

    # Electric field components
    ER::Matrix{FT} = zeros(FT, dims)            # Radial electric field [V/m]
    EZ::Matrix{FT} = zeros(FT, dims)            # Vertical electric field [V/m]
    Eϕ::Matrix{FT} = zeros(FT, dims)            # Toroidal electric field [V/m]

    # Parallel electric field
    E_para_ind::Matrix{FT} = zeros(FT, dims)    # Induced parallel electric field [V/m]
    E_para_tot::Matrix{FT} = zeros(FT, dims)    # Total parallel electric field [V/m]

    # Magnetic flux
    ψ::Matrix{FT} = zeros(FT, dims)             # Total magnetic flux [Wb/rad]
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

    # turbulent diffusion coefficients
    L_mixing::Matrix{FT} = zeros(FT, dims)          # Length of field line mixing [m]
    Dpol_turb::Matrix{FT} = zeros(FT, dims)       # Turbulent diffusion coefficient on poloidal plane [m²/s]

    DRR_turb::Matrix{FT} = zeros(FT, dims)  # R-R component of turbulent diffusion tensor
    DRZ_turb::Matrix{FT} = zeros(FT, dims)  # R-Z component of turbulent diffusion tensor
    DZZ_turb::Matrix{FT} = zeros(FT, dims)  # Z-Z component of turbulent diffusion tensor

    # Diffusion tensor components
    DRR::Matrix{FT} = zeros(FT, dims)    # R-R component of diffusion tensor
    DRZ::Matrix{FT} = zeros(FT, dims)    # R-Z component of diffusion tensor
    DZZ::Matrix{FT} = zeros(FT, dims)    # Z-Z component of diffusion tensor

    # Coefficient Tensor
    CTRR::Matrix{FT} = zeros(FT, dims)    # R-R component of coefficient tensor
    CTRZ::Matrix{FT} = zeros(FT, dims)    # R-Z component of coefficient tensor
    CTZZ::Matrix{FT} = zeros(FT, dims)    # Z-Z component of coefficient tensor
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

    # Identity matrix
    II::SparseMatrixCSC{FT, Int} = sparse(one(FT) * I, prod(dims), prod(dims))

    # Matrix placeholders to avoid repetitive allocations
    A_LHS::DiscretizedOperator{FT} = DiscretizedOperator{FT}(dims) # LHS for implicit methods

    # Basic differential operators (2nd-order central difference)
    ∂R::DiscretizedOperator{FT} = DiscretizedOperator{FT}(dims) # Radial derivative operator ∂R
    𝐽⁻¹∂R_𝐽::DiscretizedOperator{FT} = DiscretizedOperator{FT}(dims) # [(1/𝐽)(∂/∂R)*(𝐽 f)] operator
    ∂Z::DiscretizedOperator{FT} = DiscretizedOperator{FT}(dims) # Vertical derivative operator ∂Z

    # Operators for solving continuity equations
    ∇𝐃∇::DiscretizedOperator{FT} = DiscretizedOperator{FT}(dims) # Diffusion operator
    ν_iz ::DiscretizedOperator{FT} = DiscretizedOperator{FT}(dims) # Reaction frequency of ionization [1/s]

    𝐮∇::DiscretizedOperator{FT} = DiscretizedOperator{FT}(dims) # advection operator (𝐮·∇)f
    ∇𝐮::DiscretizedOperator{FT} = DiscretizedOperator{FT}(dims) # convective-flux divergence [ ∇⋅(𝐮 * f) ]

    # Mapping from k-index to CSC index (for more efficient update of non-zero elements of CSC matrix)
    # map_diffu_k2csc::Vector{Int} = zeros(Int, prod(dims)) # Mapping from k-index to CSC index

    # Operator for magnetic field solver
    ΔGS::DiscretizedOperator{FT} = DiscretizedOperator{FT}(dims) # Grad-Shafranov operator

    # RHS vectors for electron continuity equation
    RHS::Matrix{FT} = zeros(FT, dims) # Generic RHS placeholder
    neRHS_diffu::Matrix{FT} = zeros(FT, dims)  # Diffusion term
    neRHS_convec::Matrix{FT} = zeros(FT, dims) # Convection term
    neRHS_src::Matrix{FT} = zeros(FT, dims)    # Source term
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
@kwdef mutable struct SimulationFlags{FT<:AbstractFloat}
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
    negative_n_correction::Bool = true             # Correct negative densities
    Te_evolve::Bool = true                    # Evolve electron temperature
    ud_evolve::Bool = true                    # Evolve drift velocity
    Gas_evolve::Bool = true                   # Evolve neutral gas density
    Atomic_Collision::Bool = true             # Include Atomic collisions
    Coulomb_Collision::Bool = true            # Include Coulomb collisions
    Spitzer_Resistivity::Bool = true          # Include Spitzer resistivity
    Update_gFac::Bool = true                  # Update g factor for generalized EDF

    # Ion dynamics
    update_ni_independently::Bool = true      # Update ion density independently
    Ti_evolve::Bool = true                   # Update ion temperature

    # secondary electron emission by ion impact
    secondary_electron::Bool = true           # Include secondary electron emission
    γ_2nd_electron::FT = FT(0.1)         # Secondary electron emission coefficient

    # Field-related flags
    Ampere::Bool = false                      # Enable Ampere's law (magnetic field update)

    # Transport flags
    Include_heat_flux_term::Bool = false      # Include heat flux term in energy equation
    Include_ud_convec_term::Bool = true       # Include convection term in drift velocity equation
    Include_ud_pressure_term::Bool = true    # Include pressure term in drift velocity equation
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
    Implicit_weight::FT = FT(0.5)            # Weight for implicit scheme
    Adapt_dt::Bool = false                    # Use adaptive time stepping

    # Temperature limits
    min_Te::FT = FT(0.001)                   # Minimum electron temperature (eV)
    max_Te::FT = FT(500.0)                   # Maximum electron temperature (eV)

    # Global force balance
    Global_Force_Balance::Bool = false        # Include global toroidal force balance

    # Control system
    Control::Dict{Symbol, Any} = Dict{Symbol, Any}(:state => false, :target_R => nothing)

    # Numerical stability controls
    Limit_too_negative_Diffusion::Dict{Symbol, Any} = Dict{Symbol, Any}(
        :state => true,
        :limit_lower_bound_ratio => FT(-0.1)  # -0.1*n
    )

    # Current threshold for Ampere's equation
    Ampere_Itor_threshold::FT = FT(0.0)      # Current threshold for Ampere equation

    # Debug flags
    tmp_test::Bool = false                    # Enable temporary tests
    tmp_fig::Int = 100                        # Figure number for temporary tests

    # Initial parameters
    ini_gFac::FT = FT(1.0)                   # Initial g factor value
    gamma_2nd_electron::FT = FT(0.1)         # Secondary electron emission coefficient
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
    on_out_wall_nids::Vector{Int}  # Linear indices of nodes on & out the wall

    # Neighbor information for each node (including itself)
    ngh_in_wall_nids::Matrix{Vector{Int}}      # All neighboring in-wall nodes for each node
    ngh_normal_in_wall_nids::Matrix{Vector{Int}}  # Neighboring in-wall nodes in cardinal directions
    ngh_on_wall_nids::Matrix{Vector{Int}}      # Neighboring on-wall nodes for each node

    # Classification based on proximity to wall
    inWall_but_nearWall_nids::Vector{Int}  # In-wall nodes near the wall boundary
    inWall_deepInWall_nids::Vector{Int}    # In-wall nodes deep inside (away from boundary)

    # Constructor
    function NodeState{FT}(NR::Int, NZ::Int) where FT<:AbstractFloat
        rid = zeros(Int, NR, NZ)
        zid = zeros(Int, NR, NZ)
        nid = zeros(Int, NR, NZ)
        state = fill(NaN, NR, NZ)

        # Initialize neighbor information matrices
        ngh_in_wall_nids = Matrix{Vector{Int}}(undef, NR, NZ)
        ngh_normal_in_wall_nids = Matrix{Vector{Int}}(undef, NR, NZ)
        ngh_on_wall_nids = Matrix{Vector{Int}}(undef, NR, NZ)

        # Initialize all vectors to empty
        for i in 1:NR, j in 1:NZ
            ngh_in_wall_nids[i, j] = Int[]
            ngh_normal_in_wall_nids[i, j] = Int[]
            ngh_on_wall_nids[i, j] = Int[]
        end

        return new{FT}(rid, zid, nid, state, Int[], Int[], Int[], Int[],
                       ngh_in_wall_nids, ngh_normal_in_wall_nids, ngh_on_wall_nids, Int[], Int[])
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

    cell_state::Matrix{Int}  # 1: inside fitted wall, -1: outside fitted wall
    device_inVolume::FT      # Total volume inside fitted wall

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
        cell_state = zeros(Int, NR, NZ)
        device_inVolume = FT(0.0)

        return new{FT}(
            NR, NZ,
            R1D, Z1D, R2D, Z2D,
            FT(0.0), FT(0.0),
            Jacob, inv_Jacob, inVol2D,
            BDY_idx,
            nodes, cell_state,
            device_inVolume
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
    fitted_wall::WallGeometry{FT}     # Wall geometry fitted to the grid
    damping_func::Matrix{FT}          # Damping function outside wall

    # External field source
    external_field::Union{Nothing, AbstractExternalField{FT}}  # External EM field source

    # Reaction rate coefficients
    eRRCs::AbstractSpeciesRRCs{FT}    # Electron reaction rate coefficients
    iRRCs::AbstractSpeciesRRCs{FT}    # H2 Ion reaction rate coefficients

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
    diagnostics::Diagnostics   # Diagnostic data

    # Field-Line-Following analysis
    flf::FieldLineFollowingResult{FT}  # Results of field line following analysis

    # File IO
    AW_snap0D::AdiosFileWrapper    # Wrapped AdiosFile for 0D snapshots
    AW_snap2D::AdiosFileWrapper    # Wrapped AdiosFile for 2D snapshots

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
        flags = SimulationFlags{FT}()

        # Initialize matrices
        damping_func = zeros(FT, dims)
        prev_n = zeros(FT, dims)

        # Initialize empty containers
        eRRC = load_electron_RRCs()
        iRRC = load_H2_Ion_RRCs()
        tElap = Dict{Symbol, Float64}()

        dim_tt_0D = Int(ceil((config.t_end_s - config.t_start_s) / config.snap0D_Δt_s)) + 1
        dim_tt_2D = Int(ceil((config.t_end_s - config.t_start_s) / config.snap2D_Δt_s)) + 1
        diagnostics = Diagnostics{FT}(G.NR, G.NZ, dim_tt_0D, dim_tt_2D)

        flf = FieldLineFollowingResult{FT}(NR, NZ)

        # Create AdiosFileWrapper instances for snapshots
        prefixName = joinpath(config.Output_path, config.Output_prefix)
        AW_snap0D = AdiosFileWrapper(adios_open_serial(prefixName * "snap0D.bp", mode_write))
        AW_snap2D = AdiosFileWrapper(adios_open_serial(prefixName * "snap2D.bp", mode_write))

        # Create and return new instance
        return new{FT}(
            G, wall, WallGeometry{FT}(), damping_func,
            nothing,  # external_field
            eRRC, iRRC,
            config, flags, plasma, fields, transport, operators,
            0, config.t_start_s, config.t_start_s, config.t_end_s, config.dt,
            prev_n, tElap, diagnostics,
            flf,
            AW_snap0D, AW_snap2D
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