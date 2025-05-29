# Struct-based Diagnostics Architecture for RAPID2D.jl
# Using @kwdef for automatic initialization with dimension-based array creation

"""
0D diagnostic snapshot
Contains volume-averaged quantities
"""
@kwdef mutable struct Snapshot0D{FT<:AbstractFloat}
    # Step and time of the snapshot
    step::Int = 0
    time_s::FT = zero(FT)

    dt::FT = zero(FT)

    # Basic electron quantities
    ne::FT = zero(FT)              # Average electron density
    ne_max::FT = zero(FT)          # Maximum electron density
    ue_para::FT = zero(FT)         # Average electron parallel velocity
    Te_eV::FT = zero(FT)           # Average electron temperature
    𝒲e_eV::FT = zero(FT)          # Average electron energy (work)

    # Ion quantities
    ni::FT = zero(FT)              # Average ion density
    ni_max::FT = zero(FT)          # Maximum ion density
    ui_para::FT = zero(FT)         # Average ion parallel velocity
    Ti_eV::FT = zero(FT)           # Average ion temperature
    𝒲i_eV::FT = zero(FT)          # Average ion energy (work)

    I_tor::FT = zero(FT)           # Toroidal current

    # Electric fields
    Epara_tot::FT = zero(FT)       # Total parallel E-field
    Epara_ext::FT = zero(FT)       # External parallel E-field
    Epara_self_ES::FT = zero(FT)   # Self-consistent electrostatic E-field
    Epara_self_EM::FT = zero(FT)   # Self-consistent electromagnetic E-field

    # Transport quantities
    abs_ue_para_RZ::FT = zero(FT)     # Average parallel velocity
    D_RZ::FT = zero(FT)         # Average diffusion coefficient

    # Neutral gas
    n_H2_gas::FT = zero(FT)     # Average H2 density
    n_H2_gas_min::FT = zero(FT)     # Minimum H2 density

    # Collision frequencies
    ν_iz::FT = zero(FT)            # Average ionization frequency [1/s]
    ν_mom::FT = zero(FT)           # Average momentum transfer frequency [1/s]
    ν_Hα::FT = zero(FT)            # Hα emission frequency [1/s]
    ν_ei::FT = zero(FT)            # Electron-ion coulomb collision frequency [1/s]


    # Electron Heating Powers
    Pe_diffu::FT = zero(FT)       # Diffusion power
    Pe_conv::FT = zero(FT)        # Convection power
    Pe_drag::FT = zero(FT)        # Drag power
    Pe_iz::FT = zero(FT)          # Ionization power
    Pe_exc::FT = zero(FT)         # Excitation power
    Pe_dilution::FT = zero(FT)    # Dilution power
    Pe_equi::FT = zero(FT)        # Equilibration power
    Pe_heat::FT = zero(FT)        # Heating power
    Pe_tot::FT = zero(FT)         # Total electron power

    # Ion heating powers
    Pi_atomic::FT = zero(FT)      # Atomic processes power
    Pi_equi::FT = zero(FT)        # Equilibration power
    Pi_tot::FT = zero(FT)         # Total ion power

    # Source/loss tracking
    Ne_src_rate::FT = zero(FT)      # Electron source rate
    Ne_loss_rate::FT = zero(FT)     # Electron loss rate
    eGrowth_rate::FT = zero(FT)     # Electron growth rate
    eLoss_rate::FT = zero(FT)       # Electron loss rate

    Ni_src_rate::FT = zero(FT)      # Ion source rate
    Ni_loss_rate::FT = zero(FT)     # Ion loss rate

    # Plasma center tracking
    ne_cen_R::FT = zero(FT)         # Electron density center R
    ne_cen_Z::FT = zero(FT)         # Electron density center Z
    J_cen_R::FT = zero(FT)          # Current center R
    J_cen_Z::FT = zero(FT)          # Current center Z

    # CFL conditions (for adaptive timestepping)
    CFL::Dict{Symbol, FT} = Dict{Symbol, FT}() # CFL terms

    # Control system (optional)
    I_coils::Union{Nothing, Matrix{FT}} = nothing    # Coil currents (N_coils × time)
    pidFac::Union{Nothing, Vector{FT}} = nothing     # PID control factor
    BR_ctrl::Union{Nothing, Vector{FT}} = nothing # Control field BR
    BZ_ctrl::Union{Nothing, Vector{FT}} = nothing # Control field BZ

    # Growth rates (alternative calculation)
    growth_rate2::FT = zero(FT)     # Alternative growth rate
    loss_rate2::FT = zero(FT)       # Alternative loss rate
end

"""
2D diagnostic snapshots
Contains spatial distributions at specific time points
All 3D array fields are automatically sized based on dim_R, dim_Z and dim_tt
"""
@kwdef mutable struct Snapshot2D{FT<:AbstractFloat}
    # Dimension parameters (must be first)
    dims_RZ::Tuple{Int, Int} # (NR, NZ)

    # Metadata
    step::Int = 0
    time_s::FT = zero(FT)

    dt::FT = zero(FT)

    # Basic plasma quantities
    ne::Matrix{FT} = zeros(FT, dims_RZ)             # Electron density (NR, NZ, time)

    # Transport coefficients
    Dpara::Matrix{FT} = zeros(FT, dims_RZ)          # Parallel diffusion
    D_pol::Matrix{FT} = zeros(FT, dims_RZ)          # Poloidal diffusion
    ue_para::Matrix{FT} = zeros(FT, dims_RZ)        # Parallel electron velocity
    u_pol::Matrix{FT} = zeros(FT, dims_RZ)          # Poloidal velocity magnitude

    # Electron properties
    Te_eV::Matrix{FT} = zeros(FT, dims_RZ)          # Electron temperature
    𝒲e_eV::Matrix{FT} = zeros(FT, dims_RZ)        # Mean electron energy
    ueR::Matrix{FT} = zeros(FT, dims_RZ)            # Electron velocity R component
    ueϕ::Matrix{FT} = zeros(FT, dims_RZ)            # Electron velocity ϕ component
    ueZ::Matrix{FT} = zeros(FT, dims_RZ)            # Electron velocity Z component

    # Source/loss rates (2D)
    Ne_src_rate::Matrix{FT} = zeros(FT, dims_RZ)    # Electron source rate
    Ne_loss_rate::Matrix{FT} = zeros(FT, dims_RZ)   # Electron loss rate

    # Magnetic field
    BR::Matrix{FT} = zeros(FT, dims_RZ)             # Radial magnetic field
    BZ::Matrix{FT} = zeros(FT, dims_RZ)             # Vertical magnetic field
    B_pol::Matrix{FT} = zeros(FT, dims_RZ)          # Poloidal magnetic field magnitude
    BR_self::Matrix{FT} = zeros(FT, dims_RZ)        # Self-consistent BR
    BZ_self::Matrix{FT} = zeros(FT, dims_RZ)        # Self-consistent BZ

    # Electric field
    E_para_tot::Matrix{FT} = zeros(FT, dims_RZ)     # Total parallel electric field
    E_para_ext::Matrix{FT} = zeros(FT, dims_RZ)     # External parallel electric field
    mean_ExB_pol::Matrix{FT} = zeros(FT, dims_RZ)   # ExB drift magnitude
    Epol_self::Matrix{FT} = zeros(FT, dims_RZ)      # Self-consistent poloidal E-field
    Eϕ_self::Matrix{FT} = zeros(FT, dims_RZ)        # Self-consistent toroidal E-field

    # Current density
    Jϕ::Matrix{FT} = zeros(FT, dims_RZ)             # Toroidal current density
    J_para::Matrix{FT} = zeros(FT, dims_RZ)         # Parallel current

    # Magnetic flux
    psi_ext::Matrix{FT} = zeros(FT, dims_RZ)        # External poloidal flux
    psi_self::Matrix{FT} = zeros(FT, dims_RZ)       # Self-consistent poloidal flux

    # Ion properties
    ni::Matrix{FT} = zeros(FT, dims_RZ)             # Ion density
    ui_para::Matrix{FT} = zeros(FT, dims_RZ)        # Parallel ion velocity
    uiR::Matrix{FT} = zeros(FT, dims_RZ)            # Ion velocity R component
    uiϕ::Matrix{FT} = zeros(FT, dims_RZ)            # Ion velocity ϕ component
    uiZ::Matrix{FT} = zeros(FT, dims_RZ)            # Ion velocity Z component
    Ti_eV::Matrix{FT} = zeros(FT, dims_RZ)          # Ion temperature
    𝒲i_eV::Matrix{FT} = zeros(FT, dims_RZ)         # Mean ion energy
    Ni_src_rate::Matrix{FT} = zeros(FT, dims_RZ)    # Ion source rate
    Ni_loss_rate::Matrix{FT} = zeros(FT, dims_RZ)   # Ion loss rate

    # MHD-like accelerations
    mean_aR_by_JxB::Matrix{FT} = zeros(FT, dims_RZ) # JxB acceleration R component
    mean_aZ_by_JxB::Matrix{FT} = zeros(FT, dims_RZ) # JxB acceleration Z component

    # Physics parameters
    lnΛ::Matrix{FT} = zeros(FT, dims_RZ)            # Coulomb logarithm
    L_mixing::Matrix{FT} = zeros(FT, dims_RZ)       # Mixing length
    nc_para::Matrix{FT} = zeros(FT, dims_RZ)        # Parallel critical density
    nc_perp::Matrix{FT} = zeros(FT, dims_RZ)        # Perpendicular critical density
    γ_shape_fac::Matrix{FT} = zeros(FT, dims_RZ)    # Gamma coefficient

    # Neutral gas
    n_H2_gas::Matrix{FT} = zeros(FT, dims_RZ)       # H2 neutral density

    # Collision frequencies
    ν_iz::Matrix{FT} = zeros(FT, dims_RZ)            # Average ionization frequency [1/s]
    ν_mom::Matrix{FT} = zeros(FT, dims_RZ)           # Average momentum transfer frequency [1/s]
    ν_Hα::Matrix{FT} = zeros(FT, dims_RZ)            # Hα emission frequency [1/s]
    ν_ei::Matrix{FT} = zeros(FT, dims_RZ)            # Electron-ion coulomb collision frequency [1/s]


    # Electron Heating Powers
    Pe_diffu::Matrix{FT} = zeros(FT, dims_RZ)       # Diffusion power
    Pe_conv::Matrix{FT} = zeros(FT, dims_RZ)        # Convection power
    Pe_drag::Matrix{FT} = zeros(FT, dims_RZ)        # Drag power
    Pe_iz::Matrix{FT} = zeros(FT, dims_RZ)          # Ionization power
    Pe_exc::Matrix{FT} = zeros(FT, dims_RZ)         # Excitation power
    Pe_dilution::Matrix{FT} = zeros(FT, dims_RZ)    # Dilution power
    Pe_equi::Matrix{FT} = zeros(FT, dims_RZ)        # Equilibration power
    Pe_heat::Matrix{FT} = zeros(FT, dims_RZ)        # Heating power
    Pe_tot::Matrix{FT} = zeros(FT, dims_RZ)          # Total electron power

    # Ion heating powers
    Pi_atomic::Matrix{FT} = zeros(FT, dims_RZ)      # Atomic processes power
    Pi_equi::Matrix{FT} = zeros(FT, dims_RZ)        # Equilibration power
    Pi_tot::Matrix{FT} = zeros(FT, dims_RZ)          # Total ion power

    # Control fields (optional)
    BR_ctrl::Union{Nothing, Matrix{FT}} = nothing # Control magnetic field BR
    BZ_ctrl::Union{Nothing, Matrix{FT}} = nothing # Control magnetic field BZ
end

"""
Source and Loss Tracker
Tracks cumulative sources and losses of particles and energy
"""
@kwdef mutable struct SrcLossTracker{FT<:AbstractFloat}
    # Dimension parameters (must be first)
    dims_RZ::Tuple{Int, Int}    # Number of R and Z grid points

    # 0D (volume-integrated) tracking
    cum0D_Ne_src::FT = zero(FT)             # Cumulative electron source
    cum0D_Ne_loss::FT = zero(FT)            # Cumulative electron loss
    cum0D_Ni_src::FT = zero(FT)             # Cumulative ion source
    cum0D_Ni_loss::FT = zero(FT)            # Cumulative ion loss

    # 2D (spatially-resolved) tracking
    cum2D_Ne_src::Matrix{FT} = zeros(FT, dims_RZ)     # Cumulative electron source (R,Z)
    cum2D_Ne_loss::Matrix{FT} = zeros(FT, dims_RZ)    # Cumulative electron loss (R,Z)
    cum2D_Ni_src::Matrix{FT} = zeros(FT, dims_RZ)     # Cumulative ion source (R,Z)
    cum2D_Ni_loss::Matrix{FT} = zeros(FT, dims_RZ)    # Cumulative ion loss (R,Z)

    # Energy tracking (can be added/extended as needed)
    # cum0D_Energy_src::FT = zero(FT)
    # cum0D_Energy_loss::FT = zero(FT)
    # cum2D_Energy_src::Matrix{FT} = zeros(FT, dims_rz...)
    # cum2D_Energy_loss::Matrix{FT} = zeros(FT, dims_rz...)
end

"""
Main Diagnostics container
Pure struct-based approach using @kwdef for automatic initialization
No legacy Dictionary compatibility layer
"""
@kwdef mutable struct Diagnostics{FT<:AbstractFloat}
    dims_RZ::Tuple{Int, Int} # Dimensions for R and Z (NR, NZ)

    # 0D time series snapshots
    tid_0D::Int = 0 # Last recorded time index for snaps0D
    snaps0D::Vector{Snapshot0D{FT}} = Snapshot0D{FT}[]

    # 2D spatial snapshots
    tid_2D::Int = 0 # Last recorded time index for snaps2D
    snaps2D::Vector{Snapshot2D{FT}} = Snapshot2D{FT}[]

     # tracking number of particles (source/loss)
    Ntracker::SrcLossTracker{FT} = SrcLossTracker{FT}(;dims_RZ)
end

function Diagnostics{FT}(dim_R::Int, dim_Z::Int) where FT<:AbstractFloat
    return Diagnostics{FT}(; dims_RZ = (dim_R, dim_Z))
end

function Diagnostics{FT}(dim_R::Int, dim_Z::Int, dim_tt_0D::Int, dim_tt_2D::Int) where FT<:AbstractFloat
    dims_RZ = (dim_R, dim_Z)
    # Create empty snapshots with given sizes to preallocate memory
    snaps0D = [Snapshot0D{FT}() for _ in 1:dim_tt_0D]
    snaps2D = [Snapshot2D{FT}( dims_RZ = (dim_R, dim_Z) ) for _ in 1:dim_tt_2D]

    return Diagnostics{FT}(;
        dims_RZ,
        tid_0D = 0, # tid_0D
        snaps0D,
        tid_2D = 0, # tid_2D
        snaps2D,
        Ntracker =SrcLossTracker{FT}(;dims_RZ)
    )
end

# Export everything
export Diagnostics, Snapshot0D, Snapshot2D, SrcLossTracker
