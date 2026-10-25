# Struct-based Diagnostics Architecture for RAPID2D.jl
# Using @kwdef for automatic initialization with dimension-based array creation

# The snapshot layout: 1, one ion density `ni` and the H₂ gas `n_H2_gas`; 2, the total ion density
# `ni_tot` and each parent's species (`ni_<parent>`, `ng_<parent>`)
const SNAPSHOT_SCHEMA = 2

"""
0D diagnostic snapshot
Contains volume-averaged quantities
"""
@kwdef mutable struct Snapshot0D{FT <: AbstractFloat}
    # Step and time of the snapshot
    step::Int = 0
    time_s::FT = zero(FT)

    dt::FT = zero(FT)

    # Basic electron quantities
    ne::FT = zero(FT)              # Average electron density
    ne_max::FT = zero(FT)          # Maximum electron density
    ue_para::FT = zero(FT)         # Average electron parallel velocity
    Te_eV::FT = zero(FT)           # Average electron temperature
    Ke_eV::FT = zero(FT)           # Average electron kinetic energy

    # Ion quantities
    ni_tot::FT = zero(FT)          # Average ion density, every species
    ni_tot_max::FT = zero(FT)      # Maximum ion density, every species
    ui_para::FT = zero(FT)         # Average ion parallel velocity
    Ti_eV::FT = zero(FT)           # Average ion temperature
    Ki_eV::FT = zero(FT)           # Average ion kinetic energy

    I_tor::FT = zero(FT)           # Toroidal current (the stored Jϕ: from the momentum solve, before continuity)
    I_tor_endpoint::FT = zero(FT)  # the same from the step's end state, e(Z nᵢ uᵢ∥ − nₑ uₑ∥) bϕ

    # Electric fields
    E_para_tot::FT = zero(FT)       # Total parallel E-field
    Eϕ_para_presc::FT = zero(FT)     # Prescribed parallel E-field
    Eϕ_para_ctrl::FT = zero(FT)      # the vertical-field control's parallel E-field
    ERZ_para_pla::FT = zero(FT)   # Self-consistent electrostatic E-field
    Eϕ_para_coils_pla::FT = zero(FT)   # Self-consistent electromagnetic E-field

    # Transport quantities
    abs_ue_para_RZ::FT = zero(FT)     # Average parallel velocity
    D_RZ::FT = zero(FT)         # Average diffusion coefficient

    # Neutral gas (its average is the species' `ng_H2`, also read as `n_H2_gas`)
    n_H2_gas_min::FT = zero(FT)     # Minimum H2 density

    # Collision frequencies
    ν_en_iz::FT = zero(FT)          # Average ionization frequency [1/s]
    ν_en_mom_tot::FT = zero(FT)         # Average drift-friction frequency (v_z-weighted) [1/s]
    ν_en_Hα::FT = zero(FT)          # Hα emission frequency [1/s]
    ν_ei::FT = zero(FT)             # Electron-ion coulomb collision frequency [1/s]
    ν_ei_eff::FT = zero(FT)         # effective electron-ion coulomb collision frequency [1/s]
    ν_ii::FT = zero(FT)             # Ion-ion coulomb collision frequency [1/s]


    # Electron kinetic energy, and the closed-surface flux constraint (SurfaceFluxConstraint)
    Ke_tot::FT = zero(FT)            # ½ mₑ n u∥² over the in-wall volume [J]
    cs_nodes::Int = 0                # nodes of the constraint domain Ω_c (0: no constraint)
    cs_knots::Int = 0                # knots of its basis (one basis function each)
    cs_current_out::FT = zero(FT)    # share of the closed current the constraint leaves out
    cs_E_activation::FT = zero(FT)   # cumulative kinetic energy removed on activation [J]
    cs_E_basis_change::FT = zero(FT) # ... on steps where the basis changed [J]
    cs_E_projection::FT = zero(FT)   # ... on other steps [J]
    cs_E_remap::FT = zero(FT)        # cumulative kinetic-energy change of the remap [J]

    # The near-vacuum treatments (`near_vacuum.jl`, `NearVacuumLedger`)
    vac_relax_nodes::Int = 0         # in-wall nodes where the vacuum relaxation acts (ne < 100·n_vac)
    vac_relax_Ne_share::FT = zero(FT) # share of the electrons on them
    vac_relax_momentum::FT = zero(FT) # cumulative parallel momentum it took out [kg m/s]
    vac_relax_energy::FT = zero(FT)  # cumulative thermal energy it took out [J]
    limited_pairs::Int = 0           # node pairs the pair limiter scaled at the latest refresh
    floor_nodes::Int = 0             # cumulative negative-density floorings
    floor_Ne::FT = zero(FT)          # cumulative electrons the floor created

    # The current column's force balance (`column_force_balance`): a total-current screening
    # estimate on the current's moment ellipse. NaN where a value does not exist, all of them below
    # the current threshold.
    I_closed::FT = zero(FT)          # current on closed field lines [A]
    current_coherence::FT = zero(FT) # |∫Jϕ| / ∫|Jϕ|: 1 for a one-signed current
    R_cen_by_Jϕ::FT = zero(FT)        # current centroid, |Jϕ| over the area: the equivalent filament [m]
    Z_cen_by_Jϕ::FT = zero(FT)        #   (R_cen_by_Jϕ_Vol: signed Jϕ over the volume; R_cen_by_ne: by nₑ)
    a_eff_by_Jϕ::FT = zero(FT)        # effective minor radius √(2⟨r²⟩) of the current [m]
    Bpol_edge::FT = zero(FT)         # μ0 I/(2π a_eff_by_Jϕ), signed with I [T]
    li::FT = zero(FT)                # internal inductance in the column's moment ellipse (plasma's own field)
    beta_p::FT = zero(FT)            # poloidal beta in the column's moment ellipse
    pressure_column_share::FT = zero(FT) # part of ∫p dA in the column's moment ellipse
    Bv_req::FT = zero(FT)            # vertical field the column there needs (Shafranov, ln(8R/ā)) [T]
    BZ_presc_at_cen_by_Jϕ::FT = zero(FT) # prescribed B_Z at the current centroid [T]
    BR_presc_at_cen_by_Jϕ::FT = zero(FT) # prescribed B_R at the current centroid [T]
    decay_index::FT = zero(FT)       # −(R/B_Z) ∂B_Z/∂R of the prescribed field there
    decay_index_radial_limit::FT = zero(FT) # 1 − 1/Λ: fixed-current window 0 < n < this
    force_R_presc::FT = zero(FT)     # ∫ Jϕ B_Z,presc dV [N]
    force_Z_presc::FT = zero(FT)     # −∫ Jϕ B_R,presc dV [N]
    force_R_coils::FT = zero(FT)     # the same with the field of the coils the grid solves [N]
    force_Z_coils::FT = zero(FT)
    force_R_pla::FT = zero(FT)      # the same with the plasma's own field (hoop force) [N]
    force_Z_pla::FT = zero(FT)
    force_R_pressure::FT = zero(FT)  # −∫ ∂p/∂R dV over the column: tire tube less edge pressure [N]
    force_R_toroidal::FT = zero(FT)  # toroidal-field closure (μ0 I²/4)(β_p − 1) [N]
    Bv_to_add::FT = zero(FT)         # uniform B_Z that cancels the net radial force estimate [T]
    R_eq::FT = zero(FT)              # nearest radius where the Shafranov radial force vanishes [m]
    R_eq_stiffness::FT = zero(FT)    # dF_R/dR there, negative is restoring [N/m]
    R_axis::FT = zero(FT)            # magnetic axis: the O-point of the closed region [m]
    Z_axis::FT = zero(FT)
    kappa_by_Jϕ::FT = zero(FT)       # elongation √(σ_ZZ/σ_RR) of the current
    half_width_minor_by_Jϕ::FT = zero(FT) # the column's moment ellipse, half-widths 2√λ [m]
    half_width_major_by_Jϕ::FT = zero(FT) #   (its principal axes; Shafranov's ln(8R/ā) uses their mean ā)
    force_R_pressure_all::FT = zero(FT) # tire tube 2π ∫ p dR dZ of all the pressure on the grid [N]
    Br_to_add::FT = zero(FT)         # uniform B_R that cancels the net vertical force [T]
    force_R_on_closed::FT = zero(FT) # the same reduced radial force estimate on the closed-line current [N]
    force_Z_on_closed::FT = zero(FT) # vertical force on it [N]
    Bv_to_add_closed::FT = zero(FT)  # uniform B_Z that cancels the radial force on it [T]
    column_mass::FT = zero(FT)       # ion mass in the column's moment ellipse [kg]
    nu_in_column::FT = zero(FT)      # ion–neutral drag rate there, ion-weighted [1/s]
    wall_gap_R::FT = zero(FT)        # centroid travel to the wall along the net radial force, on its row [m]
    t_wall_inertial::FT = zero(FT)   # √(2 M gap/|F_R|): time to the wall at constant force, inertial [s]
    t_wall_drag::FT = zero(FT)       # gap M ν/|F_R|: the same, ion–neutral drag-limited [s]

    # The ideal vertical-field control (`IdealVerticalFieldControl`); zero without it
    ctrl_status::Int = 0             # 0 off, 1 waiting, 2 ramping, 3 on, 4 saturated, 5 turning off
    ctrl_a1::FT = zero(FT)           # applied a₁: the uniform B_Z [T]
    ctrl_a2::FT = zero(FT)           # applied a₂: −B_R at R_ref of the R²Z basis [T]
    ctrl_a1_target::FT = zero(FT)    # the target of the last update [T]
    ctrl_a2_target::FT = zero(FT)
    ctrl_BZ_at_cen::FT = zero(FT)    # the control B_Z, B_R at the current centroid [T]
    ctrl_BR_at_cen::FT = zero(FT)
    ctrl_force_R_predictor::FT = zero(FT) # F₀ + A a when the coefficients were set [N]
    ctrl_force_Z_predictor::FT = zero(FT)
    ctrl_force_R_residual::FT = zero(FT)  # the same on the present state [N]
    ctrl_force_Z_residual::FT = zero(FT)
    ctrl_force_scale::FT = zero(FT)  # μ0 I²/4: a force scale that does not vanish [N]
    ctrl_loop_voltage::FT = zero(FT) # current-weighted loop voltage of the control [V]
    ctrl_power::FT = zero(FT)        # ∫Jϕ E_ϕ,ctrl dV over the last accepted step [W]
    ctrl_energy::FT = zero(FT)       # its time integral: energy from the actuator's store [J]
    ctrl_BZ_total_at_cen::FT = zero(FT)   # prescribed + control B_Z at the current centroid [T]
    ctrl_decay_index_total::FT = zero(FT) # −(R/B_Z) ∂B_Z/∂R of that prescribed + control field

    # Electron Heating Powers
    Pe_diffu::FT = zero(FT)       # Diffusion power
    Pe_visc_heat::FT = zero(FT)    # Viscous heating of u∥, mₑ Γ_M(u∥)
    Pe_conv::FT = zero(FT)        # Convection power
    Pe_drag::FT = zero(FT)        # Drag power
    Pe_iz::FT = zero(FT)          # Ionization power
    Pe_exc::FT = zero(FT)         # Excitation power
    Pe_dilution::FT = zero(FT)    # Dilution power
    Pe_equi::FT = zero(FT)        # Given to the ions over the step (not in Pe_tot)
    Pe_heat::FT = zero(FT)        # Heating power
    Pe_tot::FT = zero(FT)         # Total electron power

    # Ion heating powers
    Pi_atomic::FT = zero(FT)      # Atomic processes power
    Pi_equi::FT = zero(FT)        # Taken from the electrons over the step (not in Pi_tot)
    Pi_tot::FT = zero(FT)         # Total ion power

    # Source/loss tracking
    Ne_src_rate::FT = zero(FT)      # Electron source rate
    Ne_loss_rate::FT = zero(FT)     # Electron loss rate
    eGrowth_rate::FT = zero(FT)     # Electron growth rate
    eLoss_rate::FT = zero(FT)       # Electron loss rate

    Ni_src_rate::FT = zero(FT)      # Ion source rate
    Ni_loss_rate::FT = zero(FT)     # Ion loss rate

    # Plasma center tracking
    R_cen_by_ne::FT = zero(FT)      # electron-density centroid, nₑ over the volume [m]
    Z_cen_by_ne::FT = zero(FT)
    R_cen_by_Jϕ_Vol::FT = zero(FT)   # current centroid, signed Jϕ over the volume (R_cen_by_Jϕ: |Jϕ| over the area) [m]
    Z_cen_by_Jϕ_Vol::FT = zero(FT)

    ueR::FT = zero(FT)            # Average electron velocity R component
    ueZ::FT = zero(FT)            # Average electron velocity Z component
    aR_by_JxB::FT = zero(FT)        # JxB acceleration R component
    aZ_by_JxB::FT = zero(FT)        # JxB acceleration Z component

    # CFL conditions (for adaptive timestepping)
    CFL::Dict{Symbol, FT} = Dict{Symbol, FT}() # CFL terms

    # Control system (optional)
    coils_I::Union{Nothing, Vector{FT}} = nothing    # Coil currents (N_coils)
    coils_V_applied::Union{Nothing, Vector{FT}} = nothing    # Coil applied voltages (N_coils)

    # Growth rates (alternative calculation)
    growth_rate2::FT = zero(FT)     # Alternative growth rate
    loss_rate2::FT = zero(FT)       # Alternative loss rate

    # Magnetic field energies
    tot_W_mag::FT = zero(FT)     # magnetic energy by both plasma and coils
    self_inductance_plasma::FT = zero(FT)   # Plasma self-inductance = 𝒲_magnetic_plasma / I_tor² [H]
    resistance_plasma::FT = zero(FT) # Plasma resistance = P_ohmic / I_tor² [Ohm]

    η_resistivity::FT = zero(FT) # Average resistivity [Ohm*m]

    # Ohmic dissipations
    tot_P_input_coils::FT = zero(FT)  # Input power by coils [W]
    tot_P_ohm_plasma::FT = zero(FT) # Ohmic dissipation by plasma [W]
    tot_P_ohm_coils::FT = zero(FT)   # Ohmic dissipation by coils [W]

    # Each parent's species, read by name: `ni_<parent>`, the volume-averaged density of its ions
    # by charge state z = 1…Z, and `ng_<parent>`, that of its neutral gas [m⁻³]
    ion_parents::Vector{String} = String[]
    ni_parents::Vector{Vector{FT}} = Vector{FT}[]
    gas_parents::Vector{String} = String[]
    ng_parents::Vector{FT} = FT[]
    schema::Int = SNAPSHOT_SCHEMA
end

"""
2D diagnostic snapshots
Contains spatial distributions at specific time points
All 3D array fields are automatically sized based on dim_R, dim_Z and dim_tt
"""
@kwdef mutable struct Snapshot2D{FT <: AbstractFloat}
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
    Ke_eV::Matrix{FT} = zeros(FT, dims_RZ)        # Mean electron energy
    ueR::Matrix{FT} = zeros(FT, dims_RZ)            # Electron velocity R component
    ueϕ::Matrix{FT} = zeros(FT, dims_RZ)            # Electron velocity ϕ component
    ueZ::Matrix{FT} = zeros(FT, dims_RZ)            # Electron velocity Z component

    # Source/loss rates (2D)
    Ne_src_rate::Matrix{FT} = zeros(FT, dims_RZ)    # Electron source rate
    Ne_loss_rate::Matrix{FT} = zeros(FT, dims_RZ)   # Electron loss rate

    # Loop voltage
    LV_presc::Matrix{FT} = zeros(FT, dims_RZ)         # Prescribed loop voltage (NR, NZ)
    LV_tot::Matrix{FT} = zeros(FT, dims_RZ)          # Total loop voltage (NR, NZ)

    # Magnetic field
    BR::Matrix{FT} = zeros(FT, dims_RZ)             # Radial magnetic field
    BZ::Matrix{FT} = zeros(FT, dims_RZ)             # Vertical magnetic field
    B_pol::Matrix{FT} = zeros(FT, dims_RZ)          # Poloidal magnetic field magnitude
    BR_coils_pla::Matrix{FT} = zeros(FT, dims_RZ)   # of the coils (walls included) and the plasma together
    BZ_coils_pla::Matrix{FT} = zeros(FT, dims_RZ)
    BR_pla::Matrix{FT} = zeros(FT, dims_RZ)         # of the plasma alone, from the stored Jϕ
    BZ_pla::Matrix{FT} = zeros(FT, dims_RZ)
    BR_coils::Matrix{FT} = zeros(FT, dims_RZ)       # of the coils alone, from their currents
    BZ_coils::Matrix{FT} = zeros(FT, dims_RZ)

    # Electric field
    E_para_tot::Matrix{FT} = zeros(FT, dims_RZ)     # Total parallel electric field
    Eϕ_para_presc::Matrix{FT} = zeros(FT, dims_RZ)   # Prescribed parallel electric field
    mean_ExB_pol::Matrix{FT} = zeros(FT, dims_RZ)   # ExB drift magnitude
    ERZ_pla_mag::Matrix{FT} = zeros(FT, dims_RZ)      # Self-consistent poloidal E-field
    Eϕ_coils_pla::Matrix{FT} = zeros(FT, dims_RZ)   # induced by the coils and the plasma over the last step
    Eϕ_coils::Matrix{FT} = zeros(FT, dims_RZ)       # the coils' part, −ψ_coils(ΔI)/(R Δt); NaN without induction
    Eϕ_pla::Matrix{FT} = zeros(FT, dims_RZ)         # the rest, Eϕ_coils_pla − Eϕ_coils (not −Δψ_pla/(R Δt)); NaN without induction

    # Current density
    Jϕ::Matrix{FT} = zeros(FT, dims_RZ)             # Toroidal current density
    J_para::Matrix{FT} = zeros(FT, dims_RZ)         # Parallel current

    # Magnetic flux
    ψ::Matrix{FT} = zeros(FT, dims_RZ)            # Total poloidal flux
    ψ_presc::Matrix{FT} = zeros(FT, dims_RZ)      # Prescribed poloidal flux
    ψ_coils_pla::Matrix{FT} = zeros(FT, dims_RZ)  # of the coils and the plasma together, as solved
    ψ_ctrl::Matrix{FT} = zeros(FT, dims_RZ)       # The vertical-field control's flux (ψ = ψ_presc + ψ_coils_pla + ψ_ctrl)
    ψ_pla::Matrix{FT} = zeros(FT, dims_RZ)        # of the plasma alone, from the stored Jϕ
    ψ_coils::Matrix{FT} = zeros(FT, dims_RZ)      # of the coils alone, from their currents
    ψ_closure_abs::FT = zero(FT)                  # ‖ψ_coils_pla − ψ_pla − ψ_coils‖∞ [Wb/rad]
    ψ_closure_rel::FT = zero(FT)                  # the same over max(ψ_floor, ‖ψ_coils_pla‖∞, ‖ψ_pla‖∞ + ‖ψ_coils‖∞)

    # The closed region: field-line following's closed in-wall nodes (1, else 0), and its magnetic
    # axis and LCFS (`closed_region_lcfs`; measured as NaN without one)
    closed_lines::Matrix{FT} = zeros(FT, dims_RZ)
    ψ_axis::FT = zero(FT)
    ψ_lcfs::FT = zero(FT)
    R_axis::FT = zero(FT)
    Z_axis::FT = zero(FT)

    # The magnetic LCFS around that axis (`magnetic_lcfs`, from the field and the wall alone; NaN,
    # and code 0, without one): its flux; its configuration, 0 none or the index in
    # `LCFS_CONFIGURATIONS` (1 limited, 2 single null, 3 double null); its geometric centre
    # ((R_max + R_min)/2, (Z_max + Z_min)/2); the X-points on it (the second only in a double null);
    # and the wall node a limited one touches
    ψ_lcfs_magnetic::FT = zero(FT)
    lcfs_configuration::Int = 0
    R_geo::FT = zero(FT)
    Z_geo::FT = zero(FT)
    R_xpoint1::FT = zero(FT)
    Z_xpoint1::FT = zero(FT)
    R_xpoint2::FT = zero(FT)
    Z_xpoint2::FT = zero(FT)
    R_contact::FT = zero(FT)
    Z_contact::FT = zero(FT)

    # Ion properties
    ni_tot::Matrix{FT} = zeros(FT, dims_RZ)         # Ion density, every species
    ui_para::Matrix{FT} = zeros(FT, dims_RZ)        # Parallel ion velocity
    uiR::Matrix{FT} = zeros(FT, dims_RZ)            # Ion velocity R component
    uiϕ::Matrix{FT} = zeros(FT, dims_RZ)            # Ion velocity ϕ component
    uiZ::Matrix{FT} = zeros(FT, dims_RZ)            # Ion velocity Z component
    Ti_eV::Matrix{FT} = zeros(FT, dims_RZ)          # Ion temperature
    Ki_eV::Matrix{FT} = zeros(FT, dims_RZ)         # Mean ion energy
    Ni_src_rate::Matrix{FT} = zeros(FT, dims_RZ)    # Ion source rate
    Ni_loss_rate::Matrix{FT} = zeros(FT, dims_RZ)   # Ion loss rate

    # MHD-like accelerations
    mean_aR_by_JxB::Matrix{FT} = zeros(FT, dims_RZ) # JxB acceleration R component
    mean_aZ_by_JxB::Matrix{FT} = zeros(FT, dims_RZ) # JxB acceleration Z component

    # Physics parameters
    lnΛ::Matrix{FT} = zeros(FT, dims_RZ)            # ELECTRON-ion Coulomb logarithm
    # Ion-ion Coulomb logarithm. It is here because it is the lever behind changes a
    # user would otherwise have to guess at: `lnΛ_ii` sets ν_ii, which moves the ion
    # AND (ambipolarly) the electron parallel diffusivity.
    #
    # No charge average travels with it. One ion species means one charge state, and
    # NR×NZ copies of a scalar carry nothing a reader cannot get from the species
    # itself. A mixture is what makes Z̄ and Z_eff vary in space, and that is when
    # they earn a place here.
    lnΛ_ii::Matrix{FT} = zeros(FT, dims_RZ)
    L_mixing::Matrix{FT} = zeros(FT, dims_RZ)       # Mixing length
    nc_para::Matrix{FT} = zeros(FT, dims_RZ)        # Parallel critical density
    nc_perp::Matrix{FT} = zeros(FT, dims_RZ)        # Perpendicular critical density
    γ_shape_fac::Matrix{FT} = zeros(FT, dims_RZ)    # Gamma coefficient

    # Collision frequencies
    ν_en_iz::Matrix{FT} = zeros(FT, dims_RZ)        # Average ionization frequency [1/s]
    ν_en_mom_tot::Matrix{FT} = zeros(FT, dims_RZ)       # Average drift-friction frequency (v_z-weighted) [1/s]
    ν_en_Hα::Matrix{FT} = zeros(FT, dims_RZ)        # Hα emission frequency [1/s]
    ν_ei::Matrix{FT} = zeros(FT, dims_RZ)           # Electron-ion coulomb collision frequency [1/s]
    ν_ei_eff::Matrix{FT} = zeros(FT, dims_RZ)       # effective electron-ion coulomb collision frequency [1/s]
    ν_ii::Matrix{FT} = zeros(FT, dims_RZ)           # Ion-ion coulomb collision frequency [1/s]

    # Electron Heating Powers
    Pe_diffu::Matrix{FT} = zeros(FT, dims_RZ)       # Diffusion power
    Pe_visc_heat::Matrix{FT} = zeros(FT, dims_RZ)    # Viscous heating of u∥, mₑ Γ_M(u∥)
    Pe_conv::Matrix{FT} = zeros(FT, dims_RZ)        # Convection power
    Pe_drag::Matrix{FT} = zeros(FT, dims_RZ)        # Drag power
    Pe_iz::Matrix{FT} = zeros(FT, dims_RZ)          # Ionization power
    Pe_exc::Matrix{FT} = zeros(FT, dims_RZ)         # Excitation power
    Pe_dilution::Matrix{FT} = zeros(FT, dims_RZ)    # Dilution power
    Pe_equi::Matrix{FT} = zeros(FT, dims_RZ)        # Given to the ions over the step (not in Pe_tot)
    Pe_heat::Matrix{FT} = zeros(FT, dims_RZ)        # Heating power
    Pe_tot::Matrix{FT} = zeros(FT, dims_RZ)          # Total electron power

    # Ion heating powers
    Pi_atomic::Matrix{FT} = zeros(FT, dims_RZ)      # Atomic processes power
    Pi_equi::Matrix{FT} = zeros(FT, dims_RZ)        # Taken from the electrons over the step (not in Pi_tot)
    Pi_tot::Matrix{FT} = zeros(FT, dims_RZ)          # Total ion power

    η_resistivity::Matrix{FT} = zeros(FT, dims_RZ)        # Parallel resistivity

    # Each parent's species, read by name: `ni_<parent>[:, :, z]`, the density of its ions in
    # charge state z = 1…Z, and `ng_<parent>`, that of its neutral gas [m⁻³] (the H₂ gas is also
    # read as `n_H2_gas`)
    ion_parents::Vector{String} = String[]
    ni_parents::Vector{Array{FT, 3}} = Array{FT, 3}[]
    gas_parents::Vector{String} = String[]
    ng_parents::Vector{Matrix{FT}} = Matrix{FT}[]
    schema::Int = SNAPSHOT_SCHEMA
end

# ── a snapshot's species, by name ───────────────────────────────────────────────────────

const AnySnapshot = Union{Snapshot0D, Snapshot2D}

# The hydrogen molecule's gas parent in each isotope (H2, D2, T2): the one `n_H2_gas` names, in a
# snapshot as in the plasma
const HYDROGEN_GAS_PARENTS = Tuple(parent_display_name(species_label(Particles.H2[0], iso)) for iso in (Protium(), Deuterium(), Tritium()))

# A field, or a species: `ni_<parent>`, `ng_<parent>`, and the hydrogen molecule's gas by its old
# name `n_H2_gas`
@inline function Base.getproperty(s::AnySnapshot, name::Symbol)
    hasfield(typeof(s), name) && return getfield(s, name)
    return species_property(s, name)
end

function species_property(s::AnySnapshot, name::Symbol)
    if name === :n_H2_gas
        j = findfirst(in(HYDROGEN_GAS_PARENTS), getfield(s, :gas_parents))
        isnothing(j) || return getfield(s, :ng_parents)[j]
    end
    str = String(name)
    if startswith(str, "ni_")
        k = findfirst(==(chopprefix(str, "ni_")), getfield(s, :ion_parents))
        isnothing(k) || return getfield(s, :ni_parents)[k]
    elseif startswith(str, "ng_")
        j = findfirst(==(chopprefix(str, "ng_")), getfield(s, :gas_parents))
        isnothing(j) || return getfield(s, :ng_parents)[j]
    end
    throw(ArgumentError("$(nameof(typeof(s))) has no property $name; its species are $(join(species_names(s), ", "))"))
end

species_names(s::AnySnapshot) = (
    (Symbol("ni_", p) for p in getfield(s, :ion_parents))...,
    (Symbol("ng_", p) for p in getfield(s, :gas_parents))...,
)

function Base.propertynames(s::AnySnapshot, ::Bool = false)
    h2 = any(in(HYDROGEN_GAS_PARENTS), getfield(s, :gas_parents)) ? (:n_H2_gas,) : ()
    return (fieldnames(typeof(s))..., species_names(s)..., h2...)
end

"""
    snapshot_matrix(snap2D, field[, z]) -> AbstractMatrix

A field or a species of a 2D snapshot as one NR × NZ matrix, the way a plot takes it. An ion
parent's species (`ni_<parent>`) holds its charge states, and `z` picks one.
"""
function snapshot_matrix(snap::Snapshot2D, field::Symbol)
    data = getproperty(snap, field)
    data isa AbstractMatrix && return data
    ndims(data) == 3 && throw(
        ArgumentError("$field holds the charge states z = 1…$(size(data, 3)); pick one: snapshot_matrix(snap, :$field, z)")
    )
    throw(ArgumentError("$field is not a 2D field of the snapshot"))
end

snapshot_matrix(snap::Snapshot2D, field::Symbol, z::Integer) = view(getproperty(snap, field), :, :, z)

"Whether a snapshot field is a density (shown on a log scale): the electrons', an ion or gas species'."
is_density_field(field::Symbol) = field in (:ne, :n_H2_gas) || startswith(String(field), "ni_") || startswith(String(field), "ng_")

"""
Source and Loss Tracker
Tracks cumulative sources and losses of particles and energy
"""
@kwdef mutable struct SrcLossTracker{FT <: AbstractFloat}
    # Dimension parameters (must be first)
    dims_RZ::Tuple{Int, Int}    # Number of R and Z grid points

    # 0D (volume-integrated) tracking
    cum0D_Ne_src::FT = zero(FT)             # Cumulative electron source
    cum0D_Ne_loss::FT = zero(FT)            # Cumulative electron loss
    cum0D_Ni_src::FT = zero(FT)             # Cumulative ion source
    cum0D_Ni_loss::FT = zero(FT)            # Cumulative ion loss
    # The ion source and loss per species (`transport.ion_species` order), recorded beside
    # the totals by their own reductions: they sum to the totals to round-off
    cum0D_Ni_src_s::Vector{FT} = FT[]
    cum0D_Ni_loss_s::Vector{FT} = FT[]

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
    PicardStats

Counts kept over a run for the coupled momentum–Ampère solve: solves, block solves summed
over them, and solves that stopped short of the tolerance (at `max_iter`, or with the mixer's
restarts used up); with the last solve's block solves and the residuals of its accepted
evaluation, of the induced field [V/m] and of the coil currents [A].
"""
@kwdef mutable struct PicardStats
    nsolve::Int = 0
    niter::Int = 0
    nunconverged::Int = 0
    last_niter::Int = 0
    last_E_residual::Float64 = 0.0
    last_I_residual::Float64 = 0.0
end

"""
    NearVacuumLedger{FT}()

What the near-vacuum treatments (`near_vacuum.jl`) did over a run, cumulative and signed:

- `relax_momentum` [kg m/s], `relax_energy` [J]: the parallel momentum and the thermal energy the
  vacuum relaxation took out of the electrons, `Σ V n w (f − f_ref)` with `f = mₑ u∥` and
  `(3/2) e Tₑ`. Exact for the backward-Euler term the solves carry.
- `floor_nodes`, `floor_Ne`: the negative-density floorings, and the electrons they created.
- `limited_pairs`: the node pairs the anti-diffusive pair limiter scaled at the latest operator
  refresh (not cumulative).

A step solved again above the Ampère gate puts it back as it was.
"""
@kwdef mutable struct NearVacuumLedger{FT <: AbstractFloat}
    relax_momentum::FT = zero(FT)
    relax_energy::FT = zero(FT)
    floor_nodes::Int = 0
    floor_Ne::FT = zero(FT)
    limited_pairs::Int = 0
end

"""
    StepRetries()

The attempts at a step the verdict rejected over a run, by its reason, and the steps then taken
in parts ([`take_step!`](@ref)). A rejected attempt leaves no other trace: the state, the
ledgers and the counts above are put back.
"""
@kwdef mutable struct StepRetries
    rejected::Dict{Symbol, Int} = Dict{Symbol, Int}()
    split_steps::Int = 0
end

"""
Main Diagnostics container
Pure struct-based approach using @kwdef for automatic initialization
No legacy Dictionary compatibility layer
"""
@kwdef mutable struct Diagnostics{FT <: AbstractFloat}
    dims_RZ::Tuple{Int, Int} # Dimensions for R and Z (NR, NZ)

    # Storage GROWS as snapshots are taken; nothing is sized from `t_end_s`. So `length`
    # is the number recorded, not a capacity with a zero-filled tail, and `snaps*[end]`
    # is the latest — no separate index to fall out of step with it.

    # 0D time series snapshots
    snaps0D::Vector{Snapshot0D{FT}} = Snapshot0D{FT}[]

    # 2D spatial snapshots
    snaps2D::Vector{Snapshot2D{FT}} = Snapshot2D{FT}[]

    # tracking number of particles (source/loss)
    Ntracker::SrcLossTracker{FT} = SrcLossTracker{FT}(; dims_RZ)

    # Picard iterations of the coupled momentum–Ampère solve
    ampere_picard::PicardStats = PicardStats()

    # what the near-vacuum treatments did (`near_vacuum.jl`)
    near_vacuum::NearVacuumLedger{FT} = NearVacuumLedger{FT}()

    # the attempts at a step the verdict rejected (`take_step!`)
    step_retries::StepRetries = StepRetries()
end

function Diagnostics{FT}(dim_R::Int, dim_Z::Int) where {FT <: AbstractFloat}
    return Diagnostics{FT}(; dims_RZ = (dim_R, dim_Z))
end

# Export everything for external use
export Diagnostics, Snapshot0D, Snapshot2D, SrcLossTracker

# Equality operators for testing ADIOS BP reading routines
"""
    ==(snap1::Snapshot0D, snap2::Snapshot0D) -> Bool

Compare two Snapshot0D objects for equality. All fields must match exactly.
This is used for testing ADIOS BP reading routines to ensure they produce
identical results to existing methods.

Note: This uses == semantics where NaN != NaN. For NaN-safe comparison, use isequal().
"""
function Base.:(==)(snap1::Snapshot0D{<:AbstractFloat}, snap2::Snapshot0D{<:AbstractFloat})
    # Compare all scalar fields
    for field in fieldnames(Snapshot0D)
        val1 = getfield(snap1, field)
        val2 = getfield(snap2, field)

        # Special handling for different field types
        if val1 isa Dict && val2 isa Dict
            # Compare dictionaries (like CFL)
            if val1 != val2
                return false
            end
        elseif val1 isa Union{Nothing, AbstractArray} && val2 isa Union{Nothing, AbstractArray}
            # Compare optional arrays (like coils_I,etc.)
            if (val1 === nothing) != (val2 === nothing)
                return false
            elseif val1 !== nothing && val2 !== nothing
                if val1 != val2
                    return false
                end
            end
        else
            # Compare scalar values - standard == semantics (NaN != NaN)
            if val1 != val2
                return false
            end
        end
    end
    return true
end

"""
    isequal(snap1::Snapshot0D, snap2::Snapshot0D) -> Bool

Check if two Snapshot0D objects are equal, with special handling for NaN values.
Unlike ==, this function treats NaN values as equal to other NaN values.
This is recommended for testing ADIOS BP reading routines where NaN values
should be considered identical.
"""
function Base.isequal(snap1::Snapshot0D{<:AbstractFloat}, snap2::Snapshot0D{<:AbstractFloat})
    # Compare all scalar fields
    for field in fieldnames(Snapshot0D)
        val1 = getfield(snap1, field)
        val2 = getfield(snap2, field)

        # Special handling for different field types
        if val1 isa Dict && val2 isa Dict
            # Compare dictionaries with isequal semantics (NaN-safe)
            if !isequal(val1, val2)
                return false
            end
        elseif val1 isa Union{Nothing, AbstractArray} && val2 isa Union{Nothing, AbstractArray}
            # Compare optional arrays
            if (val1 === nothing) != (val2 === nothing)
                return false
            elseif val1 !== nothing && val2 !== nothing
                if !isequal(val1, val2)  # NaN-safe array comparison
                    return false
                end
            end
        else
            # Compare scalar values with isequal semantics (NaN == NaN)
            if !isequal(val1, val2)
                return false
            end
        end
    end
    return true
end

"""
    ==(snap1::Snapshot2D, snap2::Snapshot2D) -> Bool

Compare two Snapshot2D objects for equality. All fields must match exactly.
This is used for testing ADIOS BP reading routines to ensure they produce
identical results to existing methods.

Note: This uses == semantics where NaN != NaN. For NaN-safe comparison, use isequal().
"""
function Base.:(==)(snap1::Snapshot2D{<:AbstractFloat}, snap2::Snapshot2D{<:AbstractFloat})
    # First compare dimensions
    if snap1.dims_RZ != snap2.dims_RZ
        return false
    end

    # Compare all fields
    for field in fieldnames(Snapshot2D)
        val1 = getfield(snap1, field)
        val2 = getfield(snap2, field)

        # All fields are either scalars, tuples, or matrices - standard == semantics
        if val1 != val2
            return false
        end
    end
    return true
end

"""
    isequal(snap1::Snapshot2D, snap2::Snapshot2D) -> Bool

Check if two Snapshot2D objects are equal, with special handling for NaN values.
Unlike ==, this function treats NaN values as equal to other NaN values.
This is recommended for testing ADIOS BP reading routines where NaN values
should be considered identical.
"""
function Base.isequal(snap1::Snapshot2D{FT}, snap2::Snapshot2D{FT}) where {FT <: AbstractFloat}
    # First compare dimensions
    if snap1.dims_RZ != snap2.dims_RZ
        return false
    end

    # Compare all fields
    for field in fieldnames(Snapshot2D)
        val1 = getfield(snap1, field)
        val2 = getfield(snap2, field)

        # Use isequal for NaN-safe comparison
        if !isequal(val1, val2)
            return false
        end
    end
    return true
end

# Two lists of per-species arrays, parent by parent
species_isapprox(a, b; kwargs...) = length(a) == length(b) && all(((x, y),) -> isapprox(x, y; kwargs...), zip(a, b))

"""
    isapprox(snap1::Snapshot0D, snap2::Snapshot0D; kwargs...) -> Bool

Check if two Snapshot0D objects are approximately equal within floating-point tolerance.
This is useful for comparing results from different numerical methods that may have
small differences due to rounding errors.

# Arguments
- `rtol::Real=√eps()`: Relative tolerance
- `atol::Real=0`: Absolute tolerance
- `nans::Bool=false`: Whether to treat NaN values as equal (default: false)

Note: When nans=true, NaN values are considered equal. When nans=false (default),
NaN values are not considered equal to anything, including other NaN values.
"""
function Base.isapprox(snap1::Snapshot0D{<:AbstractFloat}, snap2::Snapshot0D{<:AbstractFloat}; kwargs...)
    # Compare all scalar fields
    for field in fieldnames(Snapshot0D)
        val1 = getfield(snap1, field)
        val2 = getfield(snap2, field)

        # Special handling for different field types
        if val1 isa Dict && val2 isa Dict
            # Compare dictionaries approximately
            if Set(keys(val1)) != Set(keys(val2))
                return false
            end
            for key in keys(val1)
                if !isapprox(val1[key], val2[key]; kwargs...)
                    return false
                end
            end
        elseif val1 isa AbstractVector{<:AbstractString} && val2 isa AbstractVector{<:AbstractString}
            val1 == val2 || return false            # the species' parents, by name
        elseif val1 isa AbstractVector{<:AbstractArray} && val2 isa AbstractVector{<:AbstractArray}
            species_isapprox(val1, val2; kwargs...) || return false
        elseif val1 isa Union{Nothing, AbstractArray} && val2 isa Union{Nothing, AbstractArray}
            # Compare optional arrays
            if (val1 === nothing) != (val2 === nothing)
                return false
            elseif val1 !== nothing && val2 !== nothing
                if !isapprox(val1, val2; kwargs...)
                    return false
                end
            end
        elseif val1 isa AbstractFloat && val2 isa AbstractFloat
            # Compare floating-point values approximately
            if !isapprox(val1, val2; kwargs...)
                return false
            end
        else
            # Compare other scalar values exactly (Int, Bool, etc.)
            if val1 != val2
                return false
            end
        end
    end
    return true
end

"""
    isapprox(snap1::Snapshot2D, snap2::Snapshot2D; kwargs...) -> Bool

Check if two Snapshot2D objects are approximately equal within floating-point tolerance.
This is useful for comparing results from different numerical methods that may have
small differences due to rounding errors.

# Arguments
- `rtol::Real=√eps()`: Relative tolerance
- `atol::Real=0`: Absolute tolerance
- `nans::Bool=false`: Whether to treat NaN values as equal (default: false)

Note: When nans=true, NaN values are considered equal. When nans=false (default),
NaN values are not considered equal to anything, including other NaN values.
"""
function Base.isapprox(snap1::Snapshot2D{<:AbstractFloat}, snap2::Snapshot2D{<:AbstractFloat}; kwargs...)
    # First compare dimensions
    if snap1.dims_RZ != snap2.dims_RZ
        return false
    end

    # Compare all fields
    for field in fieldnames(Snapshot2D)
        val1 = getfield(snap1, field)
        val2 = getfield(snap2, field)

        if val1 isa AbstractFloat && val2 isa AbstractFloat
            # Compare floating-point scalars approximately
            if !isapprox(val1, val2; kwargs...)
                return false
            end
        elseif val1 isa Matrix{<:AbstractFloat} && val2 isa Matrix{<:AbstractFloat}
            # Compare matrices approximately
            if !isapprox(val1, val2; kwargs...)
                return false
            end
        elseif val1 isa AbstractVector{<:AbstractArray} && val2 isa AbstractVector{<:AbstractArray}
            species_isapprox(val1, val2; kwargs...) || return false
        else
            # Compare other values exactly (Int, Tuple, etc.)
            if val1 != val2
                return false
            end
        end
    end
    return true
end

"""
    getproperty(sv::Vector{<:Snapshot0D}, sym::Symbol)

Custom property access for Vector{Snapshot0D} to enable convenient field extraction.
Allows syntax like `snaps0D.time_s` to extract all time_s values from the vector.

# Examples
```julia
times = diagnostics.snaps0D.time_s      # Extract all time values
densities = diagnostics.snaps0D.ne      # Extract all electron densities
temps = diagnostics.snaps0D.Te_eV       # Extract all temperatures
```

Note: Regular vector operations (indexing, slicing) remain unchanged.
"""
function Base.getproperty(sv::Vector{<:Snapshot0D{<:AbstractFloat}}, sym::Symbol)
    # First check if this is a Vector field - delegate to original behavior
    if hasfield(Vector, sym)
        return getfield(sv, sym)
    end

    # Handle empty vector case
    if isempty(sv)
        throw(BoundsError("Cannot access property of empty snapshot vector"))
    end

    # Check if it's a valid Snapshot0D field
    if sym === :coils_I || sym === :coils_V_applied
        return reduce(hcat, [getfield(s, sym) for s in sv])  # Extract in a matrix form
    elseif hasfield(typeof(sv[1]), sym)
        return [getfield(s, sym) for s in sv]
    end

    # a species (throws for a name that is none): a value per snapshot, or its ions' densities by
    # charge state, a column per snapshot
    values = [getproperty(s, sym) for s in sv]
    return first(values) isa AbstractVector ? reduce(hcat, values) : values
end

"""
    Base.propertynames(sv::Vector{<:Snapshot0D})

Enable tab completion for snapshot vector properties in REPL.
Returns the field names of the Snapshot0D type for tab completion.
"""
function Base.propertynames(sv::Vector{<:Snapshot0D{<:AbstractFloat}})
    if isempty(sv)
        return ()  # Return empty tuple for empty vector
    end
    return propertynames(sv[1])  # Return field names of Snapshot0D type
end

"""
    getproperty(sv::Vector{<:Snapshot2D}, sym::Symbol)

Custom property access for Vector{Snapshot2D} to enable convenient field extraction.
Allows syntax like `snaps2D.ne` to extract all density matrices from the vector.

# Examples
```julia
ne_matrices = diagnostics.snaps2D.ne      # Extract all density matrices
temp_matrices = diagnostics.snaps2D.Te_eV # Extract all temperature matrices
```

Note: Regular vector operations (indexing, slicing) remain unchanged.
"""
function Base.getproperty(sv::Vector{<:Snapshot2D{FT}}, sym::Symbol) where {FT <: AbstractFloat}
    # First check if this is a Vector field - delegate to original behavior
    if hasfield(Vector, sym)
        return getfield(sv, sym)
    end

    # Handle empty vector case
    if isempty(sv)
        throw(BoundsError("Cannot access property of empty snapshot vector"))
    end

    # an array of numbers, a field or a species (`ni_<parent>`, `ng_<parent>`), gains a time axis;
    # anything else is listed per snapshot. A name that is neither throws
    if hasfield(Snapshot2D, sym) && !(fieldtype(Snapshot2D{FT}, sym) <: AbstractArray{<:Number})
        return [getfield(s, sym) for s in sv]
    end
    first_value = getproperty(sv[1], sym)
    first_value isa AbstractArray || return [getproperty(s, sym) for s in sv]
    Ntt = length(sv)
    result = Array{FT}(undef, size(first_value)..., Ntt)
    D = ndims(result)
    for i in 1:Ntt
        selectdim(result, D, i) .= getproperty(sv[i], sym)
    end
    return result
end

"""
    Base.propertynames(sv::Vector{<:Snapshot2D})

Enable tab completion for snapshot vector properties in REPL.
Returns the field names of the Snapshot2D type for tab completion.
"""
function Base.propertynames(sv::Vector{<:Snapshot2D{<:AbstractFloat}})
    if isempty(sv)
        return ()  # Return empty tuple for empty vector
    end
    return propertynames(sv[1])  # Return field names of Snapshot2D type
end
