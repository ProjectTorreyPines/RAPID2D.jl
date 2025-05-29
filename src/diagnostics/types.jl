# Struct-based Diagnostics Architecture for RAPID2D.jl
# Using @kwdef for automatic initialization with dimension-based array creation

"""
0D diagnostic snapshots (renamed from snap1D for clarity)
Contains volume-averaged quantities over time
All vector fields are automatically sized based on dim_tt
"""
@kwdef mutable struct Snap0D{FT<:AbstractFloat}
    # Dimension parameter (must be first)
    dim_tt::Int

    # Metadata
    idx::Int = 1
    time_s::Vector{FT} = zeros(FT, dim_tt)
    dt::Vector{FT} = zeros(FT, dim_tt)
    step::Vector{Int} = zeros(Int, dim_tt)

    # Basic electron quantities
    ne::Vector{FT} = zeros(FT, dim_tt)              # Average electron density
    ne_max::Vector{FT} = zeros(FT, dim_tt)          # Maximum electron density
    ue_para::Vector{FT} = zeros(FT, dim_tt)         # Average electron parallel velocity
    Te_eV::Vector{FT} = zeros(FT, dim_tt)           # Average electron temperature
    𝒲e_eV::Vector{FT} = zeros(FT, dim_tt)          # Average electron energy (work)

    # Ion quantities
    ni::Vector{FT} = zeros(FT, dim_tt)              # Average ion density
    ni_max::Vector{FT} = zeros(FT, dim_tt)          # Maximum ion density
    ui_para::Vector{FT} = zeros(FT, dim_tt)         # Average ion parallel velocity
    Ti_eV::Vector{FT} = zeros(FT, dim_tt)           # Average ion temperature
    𝒲i_eV::Vector{FT} = zeros(FT, dim_tt)          # Average ion energy (work)

    I_tor::Vector{FT} = zeros(FT, dim_tt)           # Toroidal current

    # Electric fields
    Epara_tot::Vector{FT} = zeros(FT, dim_tt)       # Total parallel E-field
    Epara_ext::Vector{FT} = zeros(FT, dim_tt)       # External parallel E-field
    Epara_self_ES::Vector{FT} = zeros(FT, dim_tt)   # Self-consistent electrostatic E-field
    Epara_self_EM::Vector{FT} = zeros(FT, dim_tt)   # Self-consistent electromagnetic E-field

    # Transport quantities
    abs_ue_para_RZ::Vector{FT} = zeros(FT, dim_tt)     # Average parallel velocity
    D_RZ::Vector{FT} = zeros(FT, dim_tt)         # Average diffusion coefficient

    # Neutral gas
    n_H2_gas::Vector{FT} = zeros(FT, dim_tt)     # Average H2 density
    n_H2_gas_min::Vector{FT} = zeros(FT, dim_tt)     # Minimum H2 density

    # Collision frequencies
    ν_iz::Vector{FT} = zeros(FT, dim_tt)            # Average ionization frequency [1/s]
    ν_mom::Vector{FT} = zeros(FT, dim_tt)           # Average momentum transfer frequency [1/s]
    ν_Hα::Vector{FT} = zeros(FT, dim_tt)            # Hα emission frequency [1/s]
    ν_ei::Vector{FT} = zeros(FT, dim_tt)            # Electron-ion coulomb collision frequency [1/s]


    # Electron Heating Powers
    Pe = (
        diffu = zeros(FT, dim_tt),       # Diffusion power
        conv = zeros(FT, dim_tt),        # Convection power
        drag = zeros(FT, dim_tt),        # Drag power
        iz = zeros(FT, dim_tt),          # Ionization power
        exc = zeros(FT, dim_tt),         # Excitation power
        dilution = zeros(FT, dim_tt),    # Dilution power
        equi = zeros(FT, dim_tt),        # Equilibration power
        heat = zeros(FT, dim_tt),        # Heating power
        tot = zeros(FT, dim_tt)          # Total electron power
    )

    # Ion heating powers
    Pi = (
        atomic = zeros(FT, dim_tt),      # Atomic processes power
        equi = zeros(FT, dim_tt),        # Equilibration power
        tot = zeros(FT, dim_tt)          # Total ion power
    )

    # Source/loss tracking
    Ne_src_rate::Vector{FT} = zeros(FT, dim_tt)      # Electron source rate
    Ne_loss_rate::Vector{FT} = zeros(FT, dim_tt)     # Electron loss rate
    eGrowth_rate::Vector{FT} = zeros(FT, dim_tt)     # Electron growth rate
    eLoss_rate::Vector{FT} = zeros(FT, dim_tt)       # Electron loss rate

    Ni_src_rate::Vector{FT} = zeros(FT, dim_tt)      # Ion source rate
    Ni_loss_rate::Vector{FT} = zeros(FT, dim_tt)     # Ion loss rate

    # Plasma center tracking
    ne_cen_R::Vector{FT} = zeros(FT, dim_tt)         # Electron density center R
    ne_cen_Z::Vector{FT} = zeros(FT, dim_tt)         # Electron density center Z
    J_cen_R::Vector{FT} = zeros(FT, dim_tt)          # Current center R
    J_cen_Z::Vector{FT} = zeros(FT, dim_tt)          # Current center Z

    # CFL conditions (for adaptive timestepping)
    CFL::Dict{Symbol, Vector{FT}} = Dict{Symbol, Vector{FT}}() # CFL terms

    # Control system (optional)
    I_coils::Union{Nothing, Matrix{FT}} = nothing    # Coil currents (N_coils × time)
    pidFac::Union{Nothing, Vector{FT}} = nothing     # PID control factor
    BR_ctrl::Union{Nothing, Vector{FT}} = nothing # Control field BR
    BZ_ctrl::Union{Nothing, Vector{FT}} = nothing # Control field BZ

    # Growth rates (alternative calculation)
    growth_rate2::Vector{FT} = zeros(FT, dim_tt)     # Alternative growth rate
    loss_rate2::Vector{FT} = zeros(FT, dim_tt)       # Alternative loss rate
end

"""
2D diagnostic snapshots
Contains spatial distributions at specific time points
All 3D array fields are automatically sized based on dim_R, dim_Z and dim_tt
"""
@kwdef mutable struct Snap2D{FT<:AbstractFloat}
    # Dimension parameters (must be first)
    dims_RZt::Tuple{Int, Int, Int} # (NR, NZ, Ntime)

    # Metadata
    idx::Int = 1
    step::Vector{Int} = zeros(Int, dims_RZt[3])
    dt::Vector{FT} = zeros(FT, dims_RZt[3])
    time_s::Vector{FT} = zeros(FT, dims_RZt[3])

    # Basic plasma quantities
    ne::Array{FT, 3} = zeros(FT, dims_RZt)             # Electron density (NR, NZ, time)

    # Transport coefficients
    Dpara::Array{FT, 3} = zeros(FT, dims_RZt)          # Parallel diffusion
    D_pol::Array{FT, 3} = zeros(FT, dims_RZt)          # Poloidal diffusion
    ue_para::Array{FT, 3} = zeros(FT, dims_RZt)        # Parallel electron velocity
    u_pol::Array{FT, 3} = zeros(FT, dims_RZt)          # Poloidal velocity magnitude

    # Electron properties
    Te_eV::Array{FT, 3} = zeros(FT, dims_RZt)          # Electron temperature
    𝒲e_eV::Array{FT, 3} = zeros(FT, dims_RZt)        # Mean electron energy
    ueR::Array{FT, 3} = zeros(FT, dims_RZt)            # Electron velocity R component
    ueϕ::Array{FT, 3} = zeros(FT, dims_RZt)            # Electron velocity ϕ component
    ueZ::Array{FT, 3} = zeros(FT, dims_RZt)            # Electron velocity Z component

    # Source/loss rates (2D)
    Ne_src_rate::Array{FT, 3} = zeros(FT, dims_RZt)    # Electron source rate
    Ne_loss_rate::Array{FT, 3} = zeros(FT, dims_RZt)   # Electron loss rate

    # Magnetic field
    BR::Array{FT, 3} = zeros(FT, dims_RZt)             # Radial magnetic field
    BZ::Array{FT, 3} = zeros(FT, dims_RZt)             # Vertical magnetic field
    B_pol::Array{FT, 3} = zeros(FT, dims_RZt)          # Poloidal magnetic field magnitude
    BR_self::Array{FT, 3} = zeros(FT, dims_RZt)        # Self-consistent BR
    BZ_self::Array{FT, 3} = zeros(FT, dims_RZt)        # Self-consistent BZ

    # Electric field
    E_para_tot::Array{FT, 3} = zeros(FT, dims_RZt)     # Total parallel electric field
    E_para_ext::Array{FT, 3} = zeros(FT, dims_RZt)     # External parallel electric field
    mean_ExB_pol::Array{FT, 3} = zeros(FT, dims_RZt)   # ExB drift magnitude
    Epol_self::Array{FT, 3} = zeros(FT, dims_RZt)      # Self-consistent poloidal E-field
    Eϕ_self::Array{FT, 3} = zeros(FT, dims_RZt)        # Self-consistent toroidal E-field

    # Current density
    Jϕ::Array{FT, 3} = zeros(FT, dims_RZt)             # Toroidal current density
    J_para::Array{FT, 3} = zeros(FT, dims_RZt)         # Parallel current

    # Magnetic flux
    psi_ext::Array{FT, 3} = zeros(FT, dims_RZt)        # External poloidal flux
    psi_self::Array{FT, 3} = zeros(FT, dims_RZt)       # Self-consistent poloidal flux

    # Ion properties
    ni::Array{FT, 3} = zeros(FT, dims_RZt)             # Ion density
    ui_para::Array{FT, 3} = zeros(FT, dims_RZt)        # Parallel ion velocity
    uiR::Array{FT, 3} = zeros(FT, dims_RZt)            # Ion velocity R component
    uiϕ::Array{FT, 3} = zeros(FT, dims_RZt)            # Ion velocity ϕ component
    uiZ::Array{FT, 3} = zeros(FT, dims_RZt)            # Ion velocity Z component
    Ti_eV::Array{FT, 3} = zeros(FT, dims_RZt)          # Ion temperature
    𝒲i_eV::Array{FT, 3} = zeros(FT, dims_RZt)         # Mean ion energy
    Ni_src_rate::Array{FT, 3} = zeros(FT, dims_RZt)    # Ion source rate
    Ni_loss_rate::Array{FT, 3} = zeros(FT, dims_RZt)   # Ion loss rate

    # MHD-like accelerations
    mean_aR_by_JxB::Array{FT, 3} = zeros(FT, dims_RZt) # JxB acceleration R component
    mean_aZ_by_JxB::Array{FT, 3} = zeros(FT, dims_RZt) # JxB acceleration Z component

    # Physics parameters
    lnΛ::Array{FT, 3} = zeros(FT, dims_RZt)            # Coulomb logarithm
    L_mixing::Array{FT, 3} = zeros(FT, dims_RZt)       # Mixing length
    nc_para::Array{FT, 3} = zeros(FT, dims_RZt)        # Parallel critical density
    nc_perp::Array{FT, 3} = zeros(FT, dims_RZt)        # Perpendicular critical density
    γ_shape_fac::Array{FT, 3} = zeros(FT, dims_RZt)    # Gamma coefficient

    # Neutral gas
    n_H2_gas::Array{FT, 3} = zeros(FT, dims_RZt)       # H2 neutral density

    # Collision frequencies
    ν_iz::Array{FT, 3} = zeros(FT, dims_RZt)            # Average ionization frequency [1/s]
    ν_mom::Array{FT, 3} = zeros(FT, dims_RZt)           # Average momentum transfer frequency [1/s]
    ν_Hα::Array{FT, 3} = zeros(FT, dims_RZt)            # Hα emission frequency [1/s]
    ν_ei::Array{FT, 3} = zeros(FT, dims_RZt)            # Electron-ion coulomb collision frequency [1/s]


    # Electron Heating Powers
    Pe = (
        diffu = zeros(FT, dims_RZt),       # Diffusion power
        conv = zeros(FT, dims_RZt),        # Convection power
        drag = zeros(FT, dims_RZt),        # Drag power
        iz = zeros(FT, dims_RZt),          # Ionization power
        exc = zeros(FT, dims_RZt),         # Excitation power
        dilution = zeros(FT, dims_RZt),    # Dilution power
        equi = zeros(FT, dims_RZt),        # Equilibration power
        heat = zeros(FT, dims_RZt),        # Heating power
        tot = zeros(FT, dims_RZt)          # Total electron power
    )

    # Ion heating powers
    Pi = (
        atomic = zeros(FT, dims_RZt),      # Atomic processes power
        equi = zeros(FT, dims_RZt),        # Equilibration power
        tot = zeros(FT, dims_RZt)          # Total ion power
    )

    # Control fields (optional)
    BR_ctrl::Union{Nothing, Array{FT, 3}} = nothing # Control magnetic field BR
    BZ_ctrl::Union{Nothing, Array{FT, 3}} = nothing # Control magnetic field BZ
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
mutable struct Diagnostics{FT<:AbstractFloat}
    # Structured diagnostic components
    snap0D::Snap0D{FT} # 0D time series (renamed from snap1D)
    snap2D::Snap2D{FT} # 2D spatial snapshots
    Ntracker::SrcLossTracker{FT} # tracking number of particles (source/loss)
end

function Diagnostics{FT}(; dim_R::Int, dim_Z::Int, dim_tt_0D::Int, dim_tt_2D::Int) where FT<:AbstractFloat
    # Create structured diagnostics with default dimensions
    return Diagnostics(
        Snap0D{FT}(; dim_tt = dim_tt_0D),
        Snap2D{FT}(; dims_RZt=(dim_R, dim_Z, dim_tt_2D)),
        SrcLossTracker{FT}(; dims_RZ=(dim_R, dim_Z))
    )
end

# Export everything
export Diagnostics, Snap0D, Snap2D, SrcLossTracker
