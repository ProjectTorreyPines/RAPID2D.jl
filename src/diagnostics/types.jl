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
    mean_eErg_eV::Vector{FT} = zeros(FT, dim_tt)    # Average electron energy

    # Ion quantities
    ni::Vector{FT} = zeros(FT, dim_tt)              # Average ion density
    ni_max::Vector{FT} = zeros(FT, dim_tt)          # Maximum ion density
    ui_para::Vector{FT} = zeros(FT, dim_tt)         # Average ion parallel velocity
    Ti_eV::Vector{FT} = zeros(FT, dim_tt)           # Average ion temperature
    mean_iErg_eV::Vector{FT} = zeros(FT, dim_tt)    # Average ion energy

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


    # Power balance (electron)
    P_diffu::Vector{FT} = zeros(FT, dim_tt)      # Diffusion power
    P_conv::Vector{FT} = zeros(FT, dim_tt)       # Convection power
    P_drag::Vector{FT} = zeros(FT, dim_tt)       # Drag power
    P_iz::Vector{FT} = zeros(FT, dim_tt)         # Ionization power
    P_exc::Vector{FT} = zeros(FT, dim_tt)        # Excitation power
    P_dilution::Vector{FT} = zeros(FT, dim_tt)   # Dilution power
    P_equi::Vector{FT} = zeros(FT, dim_tt)       # Equilibration power
    P_heat::Vector{FT} = zeros(FT, dim_tt)       # Heating power
    P_tot::Vector{FT} = zeros(FT, dim_tt)        # Total power

    # Ion power balance
    Pi_tot::Vector{FT} = zeros(FT, dim_tt)       # Total ion power
    Pi_atomic::Vector{FT} = zeros(FT, dim_tt)    # Atomic processes power
    Pi_equi::Vector{FT} = zeros(FT, dim_tt)      # Equilibration power

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
    dim_R::Int    # Number of R grid points
    dim_Z::Int    # Number of Z grid points
    dim_tt::Int   # Number of time points

    # Metadata
    idx::Int = 1
    step::Vector{Int} = zeros(Int, dim_tt)
    dt::Vector{FT} = zeros(FT, dim_tt)
    time_s::Vector{FT} = zeros(FT, dim_tt)

    # Basic plasma quantities
    ne::Array{FT, 3} = zeros(FT, dim_R, dim_Z, dim_tt)             # Electron density (NR, NZ, time)
    neRHS_diffu::Array{FT, 3} = zeros(FT, dim_R, dim_Z, dim_tt)    # Diffusion term
    neRHS_convec::Array{FT, 3} = zeros(FT, dim_R, dim_Z, dim_tt)   # Convection term
    neRHS_src::Array{FT, 3} = zeros(FT, dim_R, dim_Z, dim_tt)      # Source term

    # Transport coefficients
    Dpara::Array{FT, 3} = zeros(FT, dim_R, dim_Z, dim_tt)          # Parallel diffusion
    D_pol::Array{FT, 3} = zeros(FT, dim_R, dim_Z, dim_tt)          # Poloidal diffusion
    ue_para::Array{FT, 3} = zeros(FT, dim_R, dim_Z, dim_tt)        # Parallel electron velocity
    u_pol::Array{FT, 3} = zeros(FT, dim_R, dim_Z, dim_tt)          # Poloidal velocity magnitude

    # Electron properties
    Te_eV::Array{FT, 3} = zeros(FT, dim_R, dim_Z, dim_tt)          # Electron temperature
    mean_eErg_eV::Array{FT, 3} = zeros(FT, dim_R, dim_Z, dim_tt)   # Mean electron energy
    ueR::Array{FT, 3} = zeros(FT, dim_R, dim_Z, dim_tt)            # Electron velocity R component
    ueϕ::Array{FT, 3} = zeros(FT, dim_R, dim_Z, dim_tt)            # Electron velocity ϕ component
    ueZ::Array{FT, 3} = zeros(FT, dim_R, dim_Z, dim_tt)            # Electron velocity Z component

    # Source/loss rates (2D)
    Ne_src_rate::Array{FT, 3} = zeros(FT, dim_R, dim_Z, dim_tt)    # Electron source rate
    Ne_loss_rate::Array{FT, 3} = zeros(FT, dim_R, dim_Z, dim_tt)   # Electron loss rate

    # Magnetic field
    BR::Array{FT, 3} = zeros(FT, dim_R, dim_Z, dim_tt)             # Radial magnetic field
    BZ::Array{FT, 3} = zeros(FT, dim_R, dim_Z, dim_tt)             # Vertical magnetic field
    B_pol::Array{FT, 3} = zeros(FT, dim_R, dim_Z, dim_tt)          # Poloidal magnetic field magnitude
    BR_self::Array{FT, 3} = zeros(FT, dim_R, dim_Z, dim_tt)        # Self-consistent BR
    BZ_self::Array{FT, 3} = zeros(FT, dim_R, dim_Z, dim_tt)        # Self-consistent BZ

    # Electric field
    E_para_tot::Array{FT, 3} = zeros(FT, dim_R, dim_Z, dim_tt)     # Total parallel electric field
    E_para_ext::Array{FT, 3} = zeros(FT, dim_R, dim_Z, dim_tt)     # External parallel electric field
    mean_ExB_pol::Array{FT, 3} = zeros(FT, dim_R, dim_Z, dim_tt)   # ExB drift magnitude
    Epol_self::Array{FT, 3} = zeros(FT, dim_R, dim_Z, dim_tt)      # Self-consistent poloidal E-field
    Eϕ_self::Array{FT, 3} = zeros(FT, dim_R, dim_Z, dim_tt)        # Self-consistent toroidal E-field

    # Current density
    Jpara_R::Array{FT, 3} = zeros(FT, dim_R, dim_Z, dim_tt)        # Parallel current R component
    Jpara_Z::Array{FT, 3} = zeros(FT, dim_R, dim_Z, dim_tt)        # Parallel current Z component
    Jpara_ϕ::Array{FT, 3} = zeros(FT, dim_R, dim_Z, dim_tt)        # Parallel current phi component

    # Magnetic flux
    psi_ext::Array{FT, 3} = zeros(FT, dim_R, dim_Z, dim_tt)        # External poloidal flux
    psi_self::Array{FT, 3} = zeros(FT, dim_R, dim_Z, dim_tt)       # Self-consistent poloidal flux

    # Ion properties
    ni::Array{FT, 3} = zeros(FT, dim_R, dim_Z, dim_tt)             # Ion density
    ui_para::Array{FT, 3} = zeros(FT, dim_R, dim_Z, dim_tt)        # Parallel ion velocity
    uiR::Array{FT, 3} = zeros(FT, dim_R, dim_Z, dim_tt)            # Ion velocity R component
    uiPhi::Array{FT, 3} = zeros(FT, dim_R, dim_Z, dim_tt)          # Ion velocity phi component
    uiZ::Array{FT, 3} = zeros(FT, dim_R, dim_Z, dim_tt)            # Ion velocity Z component
    Ti_eV::Array{FT, 3} = zeros(FT, dim_R, dim_Z, dim_tt)          # Ion temperature
    mean_iErg_eV::Array{FT, 3} = zeros(FT, dim_R, dim_Z, dim_tt)   # Mean ion energy
    Ni_src_rate::Array{FT, 3} = zeros(FT, dim_R, dim_Z, dim_tt)    # Ion source rate
    Ni_loss_rate::Array{FT, 3} = zeros(FT, dim_R, dim_Z, dim_tt)   # Ion loss rate

    # MHD-like accelerations
    mean_aR_by_JxB::Array{FT, 3} = zeros(FT, dim_R, dim_Z, dim_tt) # JxB acceleration R component
    mean_aZ_by_JxB::Array{FT, 3} = zeros(FT, dim_R, dim_Z, dim_tt) # JxB acceleration Z component

    # Physics parameters
    lnΛ::Array{FT, 3} = zeros(FT, dim_R, dim_Z, dim_tt)            # Coulomb logarithm
    L_mixing::Array{FT, 3} = zeros(FT, dim_R, dim_Z, dim_tt)       # Mixing length
    nc_para::Array{FT, 3} = zeros(FT, dim_R, dim_Z, dim_tt)        # Parallel critical density
    nc_perp::Array{FT, 3} = zeros(FT, dim_R, dim_Z, dim_tt)        # Perpendicular critical density
    gamma_coeff::Array{FT, 3} = zeros(FT, dim_R, dim_Z, dim_tt)    # Gamma coefficient

    # Neutral gas
    n_H2_gas::Array{FT, 3} = zeros(FT, dim_R, dim_Z, dim_tt)       # H2 neutral density
    Halpha::Array{FT, 3} = zeros(FT, dim_R, dim_Z, dim_tt)         # H-alpha emission rate

    # Collision frequencies (2D)
    coll_freq_en_mom::Array{FT, 3} = zeros(FT, dim_R, dim_Z, dim_tt) # Electron-neutral momentum collision frequency
    coll_freq_ei::Array{FT, 3} = zeros(FT, dim_R, dim_Z, dim_tt)     # Electron-ion collision frequency

    # Power densities - electrons
    ePowers::Dict{Symbol, Array{FT, 3}} = Dict{Symbol, Array{FT, 3}}(
        :tot => zeros(FT, dim_R, dim_Z, dim_tt),
        :diffu => zeros(FT, dim_R, dim_Z, dim_tt),
        :conv => zeros(FT, dim_R, dim_Z, dim_tt),
        :drag => zeros(FT, dim_R, dim_Z, dim_tt),
        :dilution => zeros(FT, dim_R, dim_Z, dim_tt),
        :iz => zeros(FT, dim_R, dim_Z, dim_tt),
        :exc => zeros(FT, dim_R, dim_Z, dim_tt)
    )

    # Power densities - ions
    iPowers::Dict{Symbol, Array{FT, 3}} = Dict{Symbol, Array{FT, 3}}(
        :tot => zeros(FT, dim_R, dim_Z, dim_tt),
        :atomic => zeros(FT, dim_R, dim_Z, dim_tt),
        :equi => zeros(FT, dim_R, dim_Z, dim_tt)
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
    dim_R::Int    # Number of R grid points
    dim_Z::Int    # Number of Z grid points

    # 0D (volume-integrated) tracking
    cum1D_Ne_src::FT = zero(FT)             # Cumulative electron source
    cum1D_Ne_loss::FT = zero(FT)            # Cumulative electron loss
    cum1D_Ni_src::FT = zero(FT)             # Cumulative ion source
    cum1D_Ni_loss::FT = zero(FT)            # Cumulative ion loss

    # 2D (spatially-resolved) tracking
    cum2D_Ne_src::Matrix{FT} = zeros(FT, dim_R, dim_Z)     # Cumulative electron source (R,Z)
    cum2D_Ne_loss::Matrix{FT} = zeros(FT, dim_R, dim_Z)    # Cumulative electron loss (R,Z)
    cum2D_Ni_src::Matrix{FT} = zeros(FT, dim_R, dim_Z)     # Cumulative ion source (R,Z)
    cum2D_Ni_loss::Matrix{FT} = zeros(FT, dim_R, dim_Z)    # Cumulative ion loss (R,Z)

    # Energy tracking (can be added/extended as needed)
    # cum1D_Energy_src::FT = zero(FT)
    # cum1D_Energy_loss::FT = zero(FT)
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
    tracker::SrcLossTracker{FT} # Source/loss tracking
end

function Diagnostics{FT}(; dim_R::Int, dim_Z::Int, dim_tt_0D::Int, dim_tt_2D::Int) where FT<:AbstractFloat
    # Create structured diagnostics with default dimensions
    return Diagnostics(
        Snap0D{FT}(; dim_tt=dim_tt_0D),
        Snap2D{FT}(; dim_R=dim_R, dim_Z = dim_Z, dim_tt=dim_tt_2D),
        SrcLossTracker{FT}(; dim_R=dim_R, dim_Z=dim_Z)
    )
end

# Export everything
export Diagnostics, Snap0D, Snap2D, SrcLossTracker
