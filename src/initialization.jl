"""
Initialization functions for RAPID2D simulations.
"""

"""
    validate_config!(config::SimulationConfig{FT}) where {FT<:AbstractFloat}

Validates that all required parameters in the configuration are properly set.
Throws an error if any required parameter is missing or has an invalid value.

# Arguments
- `config::SimulationConfig{FT}`: The simulation configuration to validate

# Raises
- `ArgumentError`: If any required parameter is missing or invalid
"""
function validate_config!(config::SimulationConfig{FT}) where {FT<:AbstractFloat}
    missing_params = String[]

    # Check required physical parameters
    if isnothing(config.prefilled_gas_pressure)
        push!(missing_params, "prefilled_gas_pressure")
    end

    if isnothing(config.R0B0)
        push!(missing_params, "R0B0")
    end

    # Check grid parameters if manual configuration
    if config.device_Name == "manual"
        # # For manual setup, we need R_min, R_max, Z_min, Z_max
        # if isnothing(config.R_min)
        #     push!(missing_params, "R_min")
        # end
    end

    # Raise error if any required parameters are missing
    if !isempty(missing_params)
        error_msg = "Missing required configuration parameters: $(join(missing_params, ", ")). " *
                   "Please set these parameters before creating the RAPID object."
        throw(ArgumentError(error_msg))
    end

    # Additional validation for parameter values
    if !isnothing(config.prefilled_gas_pressure) && config.prefilled_gas_pressure <= 0
        throw(ArgumentError("prefilled_gas_pressure must be positive"))
    end

    if !isnothing(config.R_min) && !isnothing(config.R_max) && config.R_min >= config.R_max
        throw(ArgumentError("R_min must be less than R_max"))
    end

    if !isnothing(config.Z_min) && !isnothing(config.Z_max) && config.Z_min >= config.Z_max
        throw(ArgumentError("Z_min must be less than Z_max"))
    end

    return nothing
end

# Export the function
export validate_config!

"""
    initialize!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Initialize all components of the RAPID simulation.
"""
function initialize!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Validate configuration parameters first
    validate_config!(RP.config)

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
        set_RZ_B_E_from_file!(RP)
    end

    # Initialize damping function
    RP.damping_func = cal_damping_function_outside_wall(RP,
                                                         RP.G.R1D, RP.G.Z1D,
                                                         RP.wall.R, RP.wall.Z)

    # Set up grid and wall information
    setup_grid_nodes_state!(RP)

    # Initialize reaction rate coefficients
    initialize_reaction_rates!(RP)

    # update E,B fields
    update_fields!(RP)

    # Initialize plasma and transport
    initialize_plasma_and_transport!(RP)

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
function initialize_plasma_and_transport!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Create properly sized field objects
    RP.plasma = PlasmaState{FT}(RP.G.NR, RP.G.NZ)
    # RP.fields = Fields{FT}(RP.G.NR, RP.G.NZ)
    RP.transport = Transport{FT}(RP.G.NR, RP.G.NZ)

    # Set base diffusivities
    RP.transport.Dpara0 = FT(RP.config.Dpara0)
    RP.transport.Dperp0 = FT(RP.config.Dperp0)

    # Initialize transport coefficients
    RP.transport.Dpara .= RP.transport.Dpara0 * ones(FT, RP.G.NR, RP.G.NZ)
    RP.transport.Dperp .= RP.transport.Dperp0 * ones(FT, RP.G.NR, RP.G.NZ)

    # Set initial gas density
    RP.plasma.n_H2_gas .= RP.config.prefilled_gas_pressure ./
                           (RP.plasma.T_gas_eV * RP.config.ee) .*
                           ones(FT, RP.G.NR, RP.G.NZ)

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

"""
    set_RZ_B_E_manually!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Set up electromagnetic fields manually for a test case.
This initializes a simple geometry with analytical field configurations.
"""
function set_RZ_B_E_manually!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Set grid dimensions
    NR = RP.G.NR > 0 ? RP.G.NR : 50  # Default if not already set
    NZ = RP.G.NZ > 0 ? RP.G.NZ : 100 # Default if not already set

    # Set domain boundaries
    R_max = FT(2.2)
    R_min = FT(0.8)
    Z_max = FT(1.2)
    Z_min = FT(-1.2)

    RP.G = initialize_grid_geometry(NR, NZ, (R_min, R_max), (Z_min, Z_max));

    # Create rectangular wall
    RP.wall = WallGeometry{FT}(
        [FT(1.0), FT(2.0), FT(2.0), FT(1.0), FT(1.0)],  # Wall R coordinates
        [FT(-1.0), FT(-1.0), FT(1.0), FT(1.0), FT(-1.0)]  # Wall Z coordinates
    )

    # Initialize fields if not already created
    if !isdefined(RP, :fields) || isnothing(RP.fields)
        RP.fields = Fields{FT}(NR, NZ)
    end

    # Set basic field strengths
    Bpol = FT(5e-3)  # Poloidal field strength

    # Create fields
    RP.fields.Bϕ = RP.config.R0B0 ./ RP.G.R2D
    RP.fields.BR = zeros(FT, NR, NZ)
    RP.fields.BZ = Bpol * ones(FT, NR, NZ)

    # Compute derived field quantities
    RP.fields.Bpol = sqrt.(RP.fields.BR.^2 .+ RP.fields.BZ.^2)
    RP.fields.Btot = sqrt.(RP.fields.BR.^2 .+ RP.fields.BZ.^2 .+ RP.fields.Bϕ.^2)

    # Unit vectors for the fields
    RP.fields.bR = RP.fields.BR ./ RP.fields.Btot
    RP.fields.bZ = RP.fields.BZ ./ RP.fields.Btot
    RP.fields.bϕ = RP.fields.Bϕ ./ RP.fields.Btot

    # Electric field
    Eϕ = FT(0.3) * mean(RP.G.R1D) ./ RP.G.R2D  # 0.3 V/m
    RP.fields.Eϕ_ext = Eϕ
    RP.fields.LV_ext = Eϕ .* (2 * π * RP.G.R2D)

    # Parallel component of E
    RP.fields.E_para_ext = Eϕ .* (RP.fields.Bϕ ./ RP.fields.Btot)

    # Initialize other field components
    RP.fields.BR_ext = copy(RP.fields.BR)
    RP.fields.BZ_ext = copy(RP.fields.BZ)
    RP.fields.psi_ext = zeros(FT, NR, NZ)

    # Copy external fields to total fields initially
    RP.fields.E_para_tot = copy(RP.fields.E_para_ext)

    return RP
end

"""
    set_RZ_B_E_from_file!(RP::RAPID{FT}, dir_path::String) where {FT<:AbstractFloat}

Set up electromagnetic fields from external files.
This function loads field data from the specified path and initializes the simulation grid.
"""
function set_RZ_B_E_from_file!(RP::RAPID{FT}, dir_path::String="") where {FT<:AbstractFloat}

    if isempty(dir_path)
        dir_path = joinpath(RP.config.Input_path, RP.config.device_Name, RP.config.shot_Name)
    end


    RP.fields.R0B0 = RP.config.R0B0

    # Load external field data
    load_external_field_data!(RP, dir_path;
            r_num = RP.config.NR,
            r_min = RP.config.R_min,
            r_max = RP.config.R_max,
            z_num = RP.config.NZ,
            z_min = RP.config.Z_min,
            z_max = RP.config.Z_max
    )

    # Set grid dimensions from the loaded data
    NR = RP.external_field.R_NUM
    R_min = RP.external_field.R_MIN
    R_max = RP.external_field.R_MAX

    NZ = RP.external_field.Z_NUM
    Z_min = RP.external_field.Z_MIN
    Z_max = RP.external_field.Z_MAX

    RP.G = initialize_grid_geometry(NR, NZ, (R_min, R_max), (Z_min, Z_max));

    # Read device wall data
    read_device_wall_data!(RP)

    return RP
end

function cal_damping_function_outside_wall(RP::RAPID{FT},
                                         R1D::Vector{FT},
                                         Z1D::Vector{FT},
                                         Wall_R::Vector{FT},
                                         Wall_Z::Vector{FT}) where {FT<:AbstractFloat}
    # Create a 2D grid from R1D and Z1D coordinates
    R2D, Z2D = meshgrid(R1D, Z1D)

    # Determine which grid points are inside the wall using the existing is_inside_wall function
    isInside = is_inside_wall(R2D, Z2D, Wall_R, Wall_Z)

    # Calculate distance to wall for points outside the wall
    distanceToWall = fill(FT(Inf), size(R2D))

    # This is the most computationally intensive part - calculating distances to the wall
    for i in eachindex(R2D)
        if !isInside[i]
            # For points outside the wall, find minimum distance to any wall segment
            for j in 1:length(Wall_R)-1
                edge = [Wall_R[j] Wall_Z[j]; Wall_R[j+1] Wall_Z[j+1]]
                distanceToWall[i] = min(distanceToWall[i],
                                       distance_point_edge([R2D[i], Z2D[i]], edge))
            end
        end
    end

    # Calculate grid parameters and simulation boundaries
    R_min = minimum(R1D)
    R_max = maximum(R1D)
    Z_min = minimum(Z1D)
    Z_max = maximum(Z1D)

    dR = R1D[2] - R1D[1]
    dZ = Z1D[2] - Z1D[1]

    # Create damping characteristic length scale - use cell diagonal as in MATLAB version
    sigma = sqrt(dR^2 + dZ^2)

    # Calculate distance from boundaries for boundary damping
    distanceToBoundary = min.(min.(R2D .- R_min, R_max .- R2D),
                             min.(Z2D .- Z_min, Z_max .- Z2D))

    # Create boundary damping function - 1 inside wall, decaying near boundaries
    boundaryDamping = 1 .- exp.(-(distanceToBoundary.^2) ./ (2 * sigma^2))
    boundaryDamping[isInside] .= 1

    # Create wall damping function - 1 inside wall, exponentially decaying outside
    wallDamping = exp.(-(distanceToWall.^2) ./ (2 * sigma^2))
    wallDamping[isInside] .= 1  # No damping inside wall

    # Combine damping functions (multiply boundary and wall damping)
    Damping_Func = wallDamping .* boundaryDamping

    return Damping_Func
end

"""
    distance_point_edge(point::Vector{FT}, edge::Matrix{FT}) where {FT<:AbstractFloat}

Calculate the minimum distance from a point to a line segment (edge).

# Arguments
- `point`: [x, y] coordinates of the point
- `edge`: [x1 y1; x2 y2] coordinates of the line segment endpoints

# Returns
- The minimum distance from the point to the line segment
"""
function distance_point_edge(point::Vector{FT}, edge::Matrix{FT}) where {FT<:AbstractFloat}
    # Create vectors for calculation
    a = [edge[2, 1] - edge[1, 1], edge[2, 2] - edge[1, 2], FT(0)]  # Vector along edge
    b = [point[1] - edge[1, 1], point[2] - edge[1, 2], FT(0)]     # Vector from edge start to point
    c = [point[1] - edge[2, 1], point[2] - edge[2, 2], FT(0)]     # Vector from edge end to point

    # Check if closest point is one of the endpoints
    if dot(a, b) < 0
        return norm(b)
    elseif dot(-a, c) < 0
        return norm(c)
    else
        # Closest point is on the line segment - use cross product to find distance
        return norm(cross(a, b)) / norm(a)
    end
end

function initialize_reaction_rates!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Placeholder implementation - will be filled in later
    @warn "initialize_reaction_rates! implementation needed"

    # Just create empty dictionaries for now
    RP.eRRC = Dict{Symbol, Any}()
    RP.iRRC = Dict{Symbol, Any}()

    return RP
end

function setup_grid_nodes_state!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Determine boundary indices
    NR = RP.G.NR
    NZ = RP.G.NZ

    # Determine wall indices using is_inside_wall with WallGeometry object
    in_Wall_state = is_inside_wall(RP.G.R2D, RP.G.Z2D, RP.wall)

    # Find linear indices of points outside and inside the wall
    RP.out_wall_nids = findall(.!in_Wall_state[:])
    RP.in_wall_nids = findall(in_Wall_state[:])

    # Fill in node information using the NodeState struct
    nodes = RP.G.nodes

    # Fill in node indices information
    for j in 1:NZ
        for i in 1:NR
            nodes.rid[i, j] = i
            nodes.zid[i, j] = j
            nodes.nid[i, j] = LinearIndices((NR, NZ))[i, j]
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
        if sum(nodes.state[ngh_rids, ngh_zids]) > -length(ngh_rids)*length(ngh_zids)
            push!(nodes.on_wall_nids, nid)
        end
    end

    # Mark on-wall nodes with state = 0
    nodes.state[nodes.on_wall_nids] .= 0

    # Initialize cell state (1 for inside wall, -1 for outside)
    RP.cell_state = fill(-1, NR, NZ)
    RP.cell_state[RP.in_wall_nids] .= 1

    # Calculate inVol2D - volume elements inside the wall
    RP.G.inVol2D = zeros(FT, NR, NZ)
    RP.G.inVol2D[RP.in_wall_nids] .= RP.G.Jacob[RP.in_wall_nids] * RP.G.dR * RP.G.dZ
    RP.device_inVolume = sum(RP.G.inVol2D)

    return RP
end

function initialize_density!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Placeholder implementation - will be filled in later
    @warn "initialize_density! implementation needed"

    # Set small initial seed density inside wall
    RP.plasma.ne .= FT(1.0e12) * ones(FT, RP.G.NR, RP.G.NZ)

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
    RP.plasma.Te_eV .= FT(2.0) * ones(FT, RP.G.NR, RP.G.NZ)

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
    RP.plasma.ue_para .= zeros(FT, RP.G.NR, RP.G.NZ)
    RP.plasma.ui_para .= zeros(FT, RP.G.NR, RP.G.NZ)

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
    RP.plasma.lnA .= FT(10.0) * ones(FT, RP.G.NR, RP.G.NZ)

    # Calculate collision frequency
    # ν_ei = n_e e^4 ln Λ / (4π ε_0^2 m_e^0.5 (kT_e)^1.5)
    # simplified for now
    RP.plasma.nu_ei .= RP.plasma.ne * FT(1.0e-6) ./ (RP.plasma.Te_eV).^(1.5)

    # Spitzer factor - set to 0.51 for Z=1
    RP.plasma.sptz_fac .= FT(0.51) * ones(FT, RP.G.NR, RP.G.NZ)

    return RP
end

function initialize_snap1D!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Calculate the number of snapshots
    n_snapshots = Int(ceil((RP.t_end_s - RP.t_start_s) / RP.config.snap1D_Interval_s)) + 1

    # Create basic snapshot structure
    RP.diagnostics[:snap1D] = Dict{Symbol, Any}(
        :idx => 1,
        :time_s => zeros(FT, n_snapshots),
        :I_tor => zeros(FT, n_snapshots),
        :ne_avg => zeros(FT, n_snapshots),
        :avg_mean_eErg_eV => zeros(FT, n_snapshots),
        :avg_Epara_ext => zeros(FT, n_snapshots),
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
        :dims => (RP.G.NR, RP.G.NZ, max_snapshots)
    )

    # Pre-allocate arrays for storing physical quantities
    dims_3d = (RP.G.NR, RP.G.NZ, max_snapshots)

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
    snap2D[:E_para_ext] = zeros(FT, dims_3d)
    snap2D[:mean_ExB_pol] = zeros(FT, dims_3d)
    snap2D[:E_self_pol] = zeros(FT, dims_3d)
    snap2D[:BR_self] = zeros(FT, dims_3d)
    snap2D[:BZ_self] = zeros(FT, dims_3d)
    snap2D[:Eϕ_self] = zeros(FT, dims_3d)

    # Current
    snap2D[:Jpara_R] = zeros(FT, dims_3d)
    snap2D[:Jpara_Z] = zeros(FT, dims_3d)
    snap2D[:Jpara_ϕ] = zeros(FT, dims_3d)

    # Poloidal flux
    snap2D[:psi_ext] = zeros(FT, dims_3d)
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
    if hasfield(typeof(RP.plasma), :ne) && hasfield(typeof(RP), :neRHS_src)
        snap2D[:ne][:,:,1] = RP.plasma.ne
        snap2D[:neRHS_src][:,:,1] = RP.neRHS_src
    end

    # Assign to the diagnostics structure
    RP.diagnostics[:snap2D] = snap2D

    return RP
end

export initialize!
