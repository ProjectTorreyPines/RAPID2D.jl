"""
Initialization functions for RAPID2D simulations.
"""

"""
    initialize!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Initialize all components of the RAPID simulation.
"""
function initialize!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Initialize time tracking
    RP.tElap = Dict{Symbol, Float64}(
        :Main => 0.0,
        :diffu => 0.0,
        :convec => 0.0,
        :src => 0.0,
        :ud_update => 0.0,
        :Te_update => 0.0,
        :Ti_update => 0.0,
        :Ampere => 0.0,
        :snap1D => 0.0,
        :snap2D => 0.0,
        :write_output_file => 0.0,
        :Cal_FLF => 0.0,
        :n_H2_update => 0.0
    )

    # Set up fields based on device
    if RP.config.device_Name == "manual"
        set_RZ_B_E_manually!(RP)
    else
        file_path = joinpath(RP.config.Input_path, RP.config.device_Name,
                            RP.config.shot_Name)
        set_RZ_B_E_from_file!(RP, file_path)
    end

    # Initialize damping function
    RP.damping_func = cal_damping_function_outside_wall(RP,
                                                         RP.G.R1D, RP.G.Z1D,
                                                         RP.wall.R, RP.wall.Z)

    # Set up grid and wall information
    setup_grid_and_wall!(RP)

    # Update vacuum fields
    update_vacuum_fields!(RP)

    # Initialize reaction rate coefficients
    initialize_reaction_rates!(RP)

    # Initialize physical fields
    initialize_physical_fields!(RP)

    # Initialize operators
    initialize_operators!(RP)

    # Setup other components
    initialize_diagnostics!(RP)

    # Set initial time
    RP.time_s = RP.t_start_s

    return RP
end

"""
    initialize_physical_fields!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Initialize all physical field variables.
"""
function initialize_physical_fields!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Create properly sized field objects
    RP.plasma = PlasmaState{FT}(RP.G.NR, RP.G.NZ)
    RP.fields = Fields{FT}(RP.G.NR, RP.G.NZ)
    RP.transport = Transport{FT}(RP.G.NR, RP.G.NZ)

    # Set base diffusivities
    RP.transport.Dpara0 = FT(RP.config.Dpara0)
    RP.transport.Dperp0 = FT(RP.config.Dperp0)

    # Initialize transport coefficients
    RP.transport.Dpara .= RP.transport.Dpara0 * ones(FT, RP.G.NZ, RP.G.NR)
    RP.transport.Dperp .= RP.transport.Dperp0 * ones(FT, RP.G.NZ, RP.G.NR)

    # Set initial gas density
    RP.plasma.n_H2_gas .= RP.config.prefilled_gas_pressure ./
                           (RP.plasma.T_gas_eV * RP.config.ee) .*
                           ones(FT, RP.G.NZ, RP.G.NR)

    # Initialize density and temperature
    initialize_density!(RP)
    initialize_temperature!(RP)

    # Initialize velocities
    initialize_velocities!(RP)

    # Update Coulomb collision parameters
    update_coulomb_logarithm!(RP)

    return RP
end

"""
    initialize_operators!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Initialize the numerical operators for the simulation.
"""
function initialize_operators!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Create properly sized operators object
    RP.operators = Operators{FT}(RP.G.NR, RP.G.NZ)

    # Initialize specific operators based on flags
    if RP.flags.Ampere
        RP.operators.A_GS = construct_A_GS(RP)

        # Calculate Green's function for boundaries if needed
        Rsrc = RP.G.R2D[RP.in_wall_nids]
        Zsrc = RP.G.Z2D[RP.in_wall_nids]
        Rdest = RP.G.R2D[RP.G.BDY_idx]
        Zdest = RP.G.Z2D[RP.G.BDY_idx]

        # Placeholder for the Green's function calculation
        # RP.G_inWall2bdy = cal_psi_by_green_function(RP, Rdest, Zdest, Rsrc, Zsrc, ones(FT, length(Rsrc)))
    end

    # Create the diffusion operator if needed
    if RP.flags.diffu && RP.flags.Implicit
        # Update the diffusion tensor components
        RP.transport.DRR .= RP.transport.Dperp .+
                             (RP.transport.Dpara .- RP.transport.Dperp) .*
                             (RP.fields.bR).^2
        RP.transport.DRZ .= (RP.transport.Dpara .- RP.transport.Dperp) .*
                             RP.fields.bR .* RP.fields.bZ
        RP.transport.DZZ .= RP.transport.Dperp .+
                             (RP.transport.Dpara .- RP.transport.Dperp) .*
                             (RP.fields.bZ).^2

        # Construct diffusion operator
        # RP.operators.An_diffu = construct_An_diffu(RP, ...)
    end

    return RP
end

"""
    initialize_diagnostics!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Initialize diagnostic data structures.
"""
function initialize_diagnostics!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Initialize snapshots
    initialize_snap1D!(RP)
    initialize_snap2D!(RP)

    return RP
end

# Placeholder for detailed implementations
function set_RZ_B_E_manually!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Placeholder implementation - will be filled in later
    @warn "set_RZ_B_E_manually! implementation needed"

    # Basic grid setup
    RP.G.R1D = collect(range(FT(1.0), FT(2.0), length=RP.G.NR))
    RP.G.Z1D = collect(range(FT(-0.5), FT(0.5), length=RP.G.NZ))

    # Create 2D grid
    RP.G.R2D = repeat(RP.G.R1D', RP.G.NZ, 1)
    RP.G.Z2D = repeat(RP.G.Z1D, 1, RP.G.NR)

    # Grid spacing
    RP.G.dR = RP.G.R1D[2] - RP.G.R1D[1]
    RP.G.dZ = RP.G.Z1D[2] - RP.G.Z1D[1]

    # Jacobian - for cylindrical coordinates, this is r
    RP.G.Jacob = copy(RP.G.R2D)
    RP.G.inv_Jacob = 1.0 ./ RP.G.Jacob

    # Create a circular wall
    theta = collect(range(0, 2π, length=100))
    center_R = (RP.G.R1D[1] + RP.G.R1D[end]) / 2
    center_Z = (RP.G.Z1D[1] + RP.G.Z1D[end]) / 2
    radius = min((RP.G.R1D[end] - RP.G.R1D[1]),
                (RP.G.Z1D[end] - RP.G.Z1D[1])) * 0.45

    wall_R = center_R .+ radius .* cos.(theta)
    wall_Z = center_Z .+ radius .* sin.(theta)

    RP.wall = WallGeometry{FT}(wall_R, wall_Z)

    return RP
end

function set_RZ_B_E_from_file!(RP::RAPID{FT}, file_path::String) where {FT<:AbstractFloat}
    # Placeholder implementation - will be filled in later
    @warn "set_RZ_B_E_from_file! implementation needed"

    # Just call the manual method for now
    set_RZ_B_E_manually!(RP)

    return RP
end

function cal_damping_function_outside_wall(RP::RAPID{FT},
                                         R1D::Vector{FT},
                                         Z1D::Vector{FT},
                                         Wall_R::Vector{FT},
                                         Wall_Z::Vector{FT}) where {FT<:AbstractFloat}
    # Placeholder implementation - will create a simple radial damping function
    @warn "cal_damping_function_outside_wall implementation needed"

    NZ = length(Z1D)
    NR = length(R1D)

    # Create 2D grid if not already available
    R2D = repeat(R1D', NZ, 1)
    Z2D = repeat(Z1D, 1, NR)

    # Simplified damping function - 1 inside wall, decaying outside
    damping = ones(FT, NZ, NR)

    # Find center of wall
    center_R = sum(Wall_R) / length(Wall_R)
    center_Z = sum(Wall_Z) / length(Wall_Z)

    # Maximum radius of wall from center
    max_radius = maximum(sqrt.((Wall_R .- center_R).^2 .+ (Wall_Z .- center_Z).^2))

    # Calculate distance from center for each grid point
    for i in 1:NZ
        for j in 1:NR
            r = sqrt((R2D[i,j] - center_R)^2 + (Z2D[i,j] - center_Z)^2)

            # Apply damping outside the wall
            if r > max_radius
                # Exponential decay outside wall
                damping_factor = exp(-(r - max_radius) / (0.1 * max_radius))
                damping[i,j] = max(FT(0.01), damping_factor)
            end
        end
    end

    return damping
end

function update_vacuum_fields!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Placeholder implementation - will be filled in later
    @warn "update_vacuum_fields! implementation needed"

    # Create simple toroidal field - B_phi = B0*R0/R
    R0 = (RP.G.R1D[1] + RP.G.R1D[end]) / 2
    B0 = abs(RP.config.R0B0 / R0)

	F = RP.fields

    F.Bϕ .= RP.config.R0B0 ./ RP.G.R2D
    F.BR .= zeros(FT, RP.G.NZ, RP.G.NR)
    F.BZ .= zeros(FT, RP.G.NZ, RP.G.NR)

    # Simple uniform loop voltage
    F.Eϕ .= FT(0.5) .* ones(FT, RP.G.NZ, RP.G.NR)

    # Copy to vacuum fields
    F.BR_vac .= F.BR
    F.BZ_vac .= F.BZ

    # Calculate field magnitudes
    F.Bpol .= sqrt.(F.BR.^2 .+ F.BZ.^2)
    F.Btot .= sqrt.(F.Bpol.^2 .+ F.Bϕ.^2)

    # Calculate unit vectors
    # Avoid division by zero
    epsilon = FT(1.0e-10)
    mask = F.Btot .< epsilon

    b_denominator = copy(F.Btot)
    b_denominator[mask] .= FT(1.0)

    F.bR .= F.BR ./ b_denominator
    F.bZ .= F.BZ ./ b_denominator
    F.bϕ .= F.Bϕ ./ b_denominator

    # Zero out unit vectors where total field is near zero
    F.bR[mask] .= FT(0.0)
    F.bZ[mask] .= FT(0.0)
    F.bϕ[mask] .= FT(1.0) # Default to toroidal direction

    # Set vacuum parallel E-field
    F.E_para_vac .= F.Eϕ .* F.bϕ

    # Initialize total parallel E-field
    F.E_para_tot .= F.E_para_vac

	Main.@infiltrate

    return RP
end

function initialize_reaction_rates!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Placeholder implementation - will be filled in later
    @warn "initialize_reaction_rates! implementation needed"

    # Just create empty dictionaries for now
    RP.eRRC = Dict{Symbol, Any}()
    RP.iRRC = Dict{Symbol, Any}()

    return RP
end

function setup_grid_and_wall!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Determine boundary indices
    NR = RP.G.NR
    NZ = RP.G.NZ

    # Define boundary indices - the perimeter of the domain
    BDY_Rid = vcat(
        fill(1, NZ),          # Left edge
        2:(NR-1),             # Bottom edge
        fill(NR, NZ),         # Right edge
        (NR-1):-1:2           # Top edge
    )

    BDY_Zid = vcat(
        1:NZ,                 # Left edge
        fill(NZ, NR-2),       # Bottom edge
        NZ:-1:1,              # Right edge
        fill(1, NR-2)         # Top edge
    )

    # Convert to linear indices
    RP.G.BDY_idx = LinearIndices((NZ, NR))[CartesianIndex.(BDY_Zid, BDY_Rid)]

    # Determine wall indices using is_inside_wall with WallGeometry object
    in_Wall_state = is_inside_wall(RP.G.R2D, RP.G.Z2D, RP.wall)

    # Find linear indices of points outside and inside the wall
    RP.out_wall_nids = findall(.!in_Wall_state[:])
    RP.in_wall_nids = findall(in_Wall_state[:])

    # Fill in node information using the NodeState struct
    nodes = RP.G.nodes

    # Fill in node indices information
    for j in 1:NR
        for i in 1:NZ
            nodes.zid[i, j] = i
            nodes.rid[i, j] = j
            nodes.nid[i, j] = LinearIndices((NZ, NR))[i, j]
        end
    end

    # Mark node state (-1 for outside, +1 for inside)
    nodes.state[RP.out_wall_nids] .= -1
    nodes.state[RP.in_wall_nids] .= 1

    # Store indices in the NodeState
    nodes.in_wall_nids = copy(RP.in_wall_nids)
    nodes.out_wall_nids = copy(RP.out_wall_nids)

    # Find nodes on the wall (boundary nodes)
    nodes.on_wall_nids = Int[]
    for k in 1:length(RP.out_wall_nids)
        nid = RP.out_wall_nids[k]
        rid = nodes.rid[nid]
        zid = nodes.zid[nid]

        # Define neighborhood indices, making sure they are within bounds
        ngh_rids = max(1, rid-1):min(NR, rid+1)
        ngh_zids = max(1, zid-1):min(NZ, zid+1)

        # Check if this outside point has any inside neighbors
        # If sum is greater than -N (where N is number of neighbors), some neighbors are inside
        if sum(nodes.state[ngh_zids, ngh_rids]) > -length(ngh_rids)*length(ngh_zids)
            push!(nodes.on_wall_nids, nid)
        end
    end

    # Mark on-wall nodes with state = 0
    nodes.state[nodes.on_wall_nids] .= 0

    # Initialize cell state (1 for inside wall, -1 for outside)
    RP.cell_state = fill(-1, NZ, NR)
    RP.cell_state[RP.in_wall_nids] .= 1

    # Calculate inVol2D - volume elements inside the wall
    RP.G.inVol2D = zeros(FT, NZ, NR)
    RP.G.inVol2D[RP.in_wall_nids] .= RP.G.Jacob[RP.in_wall_nids] * RP.G.dR * RP.G.dZ
    RP.device_inVolume = sum(RP.G.inVol2D)

    return RP
end

function initialize_density!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Placeholder implementation - will be filled in later
    @warn "initialize_density! implementation needed"

    # Set small initial seed density inside wall
    RP.plasma.ne .= FT(1.0e12) * ones(FT, RP.G.NZ, RP.G.NR)

    # Zero outside wall
    RP.plasma.ne[RP.out_wall_nids] .= FT(0.0)

    # Ion density matches electron for now
    RP.plasma.ni .= copy(RP.plasma.ne)

    return RP
end

function initialize_temperature!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Placeholder implementation - will be filled in later
    @warn "initialize_temperature! implementation needed"

    # Set initial electron temperature
    RP.plasma.Te_eV .= FT(2.0) * ones(FT, RP.G.NZ, RP.G.NR)

    # Zero outside wall
    RP.plasma.Te_eV[RP.out_wall_nids] .= RP.config.min_Te

    # Ion temperature matches electron for now
    RP.plasma.Ti_eV .= copy(RP.plasma.Te_eV)

    return RP
end

function initialize_velocities!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Placeholder implementation - will be filled in later
    @warn "initialize_velocities! implementation needed"

    # Set initial velocities to zero
    RP.plasma.ue_para .= zeros(FT, RP.G.NZ, RP.G.NR)
    RP.plasma.ui_para .= zeros(FT, RP.G.NZ, RP.G.NR)

    # Initialize vector components
    RP.plasma.ueR .= RP.plasma.ue_para .* RP.fields.bR
    RP.plasma.ueZ .= RP.plasma.ue_para .* RP.fields.bZ
    RP.plasma.ueϕ .= RP.plasma.ue_para .* RP.fields.bϕ

    RP.plasma.uiR .= RP.plasma.ui_para .* RP.fields.bR
    RP.plasma.uiZ .= RP.plasma.ui_para .* RP.fields.bZ
    RP.plasma.uiϕ .= RP.plasma.ui_para .* RP.fields.bϕ

    return RP
end

function update_coulomb_logarithm!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Placeholder implementation - will be filled in later
    @warn "update_coulomb_logarithm! implementation needed"

    # Set coulomb logarithm to a constant value for now
    RP.plasma.lnA .= FT(10.0) * ones(FT, RP.G.NZ, RP.G.NR)

    # Calculate collision frequency
    # ν_ei = n_e e^4 ln Λ / (4π ε_0^2 m_e^0.5 (kT_e)^1.5)
    # simplified for now
    RP.plasma.nu_ei .= RP.plasma.ne * FT(1.0e-6) ./ (RP.plasma.Te_eV).^(1.5)

    # Spitzer factor - set to 0.51 for Z=1
    RP.plasma.sptz_fac .= FT(0.51) * ones(FT, RP.G.NZ, RP.G.NR)

    return RP
end

function initialize_snap1D!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Placeholder implementation - will be filled in later
    @warn "initialize_snap1D! implementation needed"

    # Calculate the number of snapshots
    n_snapshots = Int(ceil((RP.t_end_s - RP.t_start_s) / RP.config.snap1D_Interval_s)) + 1

    # Create basic snapshot structure
    RP.diagnostics.snap1D = Dict{Symbol, Any}(
        :idx => 1,
        :time_s => zeros(FT, n_snapshots),
        :I_tor => zeros(FT, n_snapshots),
        :ne_avg => zeros(FT, n_snapshots),
        :avg_mean_eErg_eV => zeros(FT, n_snapshots),
        :avg_Epara_vac => zeros(FT, n_snapshots),
        :avg_Epara_tot => zeros(FT, n_snapshots)
    )

    return RP
end

function initialize_snap2D!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Calculate the number of snapshots
    max_snapshots = Int(ceil((RP.t_end_s - RP.t_start_s) / RP.config.snap2D_Interval_s)) + 1

    # Create structure for storing 2D snapshot data
    snap2D = Dict{Symbol, Any}(
        :idx => 1,
        :step => zeros(Int, max_snapshots),
        :dt => zeros(FT, max_snapshots),
        :time_s => zeros(FT, max_snapshots),
        :dims => (RP.G.NZ, RP.G.NR, max_snapshots)
    )

    # Pre-allocate arrays for storing physical quantities
    dims_3d = (RP.G.NZ, RP.G.NR, max_snapshots)

    # Electron properties
    snap2D[:ne] = zeros(FT, dims_3d)
    snap2D[:neRHS_diffu] = zeros(FT, dims_3d)
    snap2D[:neRHS_convec] = zeros(FT, dims_3d)
    snap2D[:neRHS_src] = zeros(FT, dims_3d)
    snap2D[:Dpara] = zeros(FT, dims_3d)
    snap2D[:D_pol] = zeros(FT, dims_3d)
    snap2D[:ue_para] = zeros(FT, dims_3d)
    snap2D[:Te_eV] = zeros(FT, dims_3d)
    snap2D[:mean_eErg_eV] = zeros(FT, dims_3d)
    snap2D[:coll_freq_en_mom] = zeros(FT, dims_3d)
    snap2D[:coll_freq_ei] = zeros(FT, dims_3d)
    snap2D[:ueR] = zeros(FT, dims_3d)
    snap2D[:uePhi] = zeros(FT, dims_3d)
    snap2D[:ueZ] = zeros(FT, dims_3d)
    snap2D[:Ne_src_rate] = zeros(FT, dims_3d)
    snap2D[:Ne_loss_rate] = zeros(FT, dims_3d)

    # Magnetic field and electric field
    snap2D[:BR] = zeros(FT, dims_3d)
    snap2D[:BZ] = zeros(FT, dims_3d)
    snap2D[:B_pol] = zeros(FT, dims_3d)
    snap2D[:u_pol] = zeros(FT, dims_3d)
    snap2D[:E_para_tot] = zeros(FT, dims_3d)
    snap2D[:E_para_vac] = zeros(FT, dims_3d)
    snap2D[:mean_ExB_pol] = zeros(FT, dims_3d)
    snap2D[:E_self_pol] = zeros(FT, dims_3d)
    snap2D[:BR_self] = zeros(FT, dims_3d)
    snap2D[:BZ_self] = zeros(FT, dims_3d)
    snap2D[:Ephi_self] = zeros(FT, dims_3d)

    # Current
    snap2D[:Jpara_R] = zeros(FT, dims_3d)
    snap2D[:Jpara_Z] = zeros(FT, dims_3d)
    snap2D[:Jpara_phi] = zeros(FT, dims_3d)

    # Poloidal flux
    snap2D[:psi_vac] = zeros(FT, dims_3d)
    snap2D[:psi_self] = zeros(FT, dims_3d)

    # Ion properties
    snap2D[:ni] = zeros(FT, dims_3d)
    snap2D[:ui_para] = zeros(FT, dims_3d)
    snap2D[:uiR] = zeros(FT, dims_3d)
    snap2D[:uiPhi] = zeros(FT, dims_3d)
    snap2D[:uiZ] = zeros(FT, dims_3d)
    snap2D[:Ti_eV] = zeros(FT, dims_3d)
    snap2D[:mean_iErg_eV] = zeros(FT, dims_3d)
    snap2D[:Ni_src_rate] = zeros(FT, dims_3d)
    snap2D[:Ni_loss_rate] = zeros(FT, dims_3d)

    # MHD-like velocities
    snap2D[:mean_aR_by_JxB] = zeros(FT, dims_3d)
    snap2D[:mean_aZ_by_JxB] = zeros(FT, dims_3d)

    # Other physics parameters
    snap2D[:lnA] = zeros(FT, dims_3d)
    snap2D[:L_mixing] = zeros(FT, dims_3d)
    snap2D[:nc_para] = zeros(FT, dims_3d)
    snap2D[:nc_perp] = zeros(FT, dims_3d)
    snap2D[:gamma_coeff] = zeros(FT, dims_3d)
    snap2D[:n_H2_gas] = zeros(FT, dims_3d)
    snap2D[:Halpha] = zeros(FT, dims_3d)

    # Power terms - electron
    snap2D[:ePowers] = Dict{Symbol, Array{FT, 3}}(
        :tot => zeros(FT, dims_3d),
        :diffu => zeros(FT, dims_3d),
        :conv => zeros(FT, dims_3d),
        :drag => zeros(FT, dims_3d),
        :dilution => zeros(FT, dims_3d),
        :iz => zeros(FT, dims_3d),
        :exc => zeros(FT, dims_3d)
    )

    # Power terms - ion
    snap2D[:iPowers] = Dict{Symbol, Array{FT, 3}}(
        :tot => zeros(FT, dims_3d),
        :atomic => zeros(FT, dims_3d),
        :equi => zeros(FT, dims_3d)
    )

    # Control coils (if enabled)
    if hasfield(typeof(RP.config), :Control) && RP.config.Control.state
        snap2D[:BR_ctrl] = zeros(FT, dims_3d)
        snap2D[:BZ_ctrl] = zeros(FT, dims_3d)
    end

    # Store initial values
    # Only storing ne and neRHS_src initially as in the MATLAB version
    if hasfield(typeof(RP), :ne) && hasfield(typeof(RP), :neRHS_src)
        snap2D[:ne][:,:,1] = RP.ne
        snap2D[:neRHS_src][:,:,1] = RP.neRHS_src
    end

    # Assign to the diagnostics structure
    RP.diagnostics.snap2D = snap2D

    return RP
end