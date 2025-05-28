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
    setup_grid_state_and_volumes_with_wall!(RP)

    # Initialize reaction rate coefficients
    initialize_RRCs!(RP)

    # update E,B fields
    update_external_fields!(RP)
    update_self_fields!(RP)
    combine_external_and_self_fields!(RP)

    # Initialize plasma and transport
    initialize_plasma_and_transport!(RP)

    # Initialize operators
    initialize_operators!(RP)


    # Set initial time
    RP.time_s = RP.t_start_s
    RP.step = 0

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
    update_coulomb_collision_parameters!(RP)

    update_transport_quantities!(RP)

    return RP
end

"""
    initialize_operators!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Initialize the numerical operators for the simulation.
"""
function initialize_operators!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Create properly sized operators object
    RP.operators = Operators{FT}(RP.G.NR, RP.G.NZ)

    # Construct basic differntial operators
    RP.operators.∂R = construct_∂R_operator(RP.G)
    RP.operators.𝐽⁻¹∂R_𝐽 = construct_𝐽⁻¹∂R_𝐽_operator(RP.G)
    RP.operators.∂Z = construct_∂Z_operator(RP.G)

    if RP.flags.diffu
        RP.operators.∇𝐃∇ = construct_∇𝐃∇_operator(RP)
    end
    if RP.flags.convec
        RP.operators.∇𝐮 = construct_∇𝐮_operator(RP)
        RP.operators.𝐮∇ = construct_𝐮∇_operator(RP)
    end

    # Initialize specific operators based on flags
    if RP.flags.Ampere
        RP.operators.A_GS = construct_A_GS(RP)

        # Calculate Green's function for boundaries if needed
        Rsrc = RP.G.R2D[RP.G.nodes.in_wall_nids]
        Zsrc = RP.G.Z2D[RP.G.nodes.in_wall_nids]
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
        # RP.operators.∇𝐃∇ = construct_∇𝐃∇(RP, ...)
    end

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
    R_max = isnothing(RP.config.R_max) ? FT(2.2) : RP.config.R_max
    R_min = isnothing(RP.config.R_min) ? FT(0.8) : RP.config.R_min
    Z_max = isnothing(RP.config.Z_max) ? FT(1.2) : RP.config.Z_max
    Z_min = isnothing(RP.config.Z_min) ? FT(-1.2) : RP.config.Z_min

    RP.G = initialize_grid_geometry(NR, NZ, (R_min, R_max), (Z_min, Z_max));

    if isempty(RP.config.wall_R) || isempty(RP.config.wall_Z)
        # Set default wall coordinates if not provided
        # Create rectangular wall a few cells away from the numerical boundary
        # Set wall offset (several grid cells from the boundary)
        offset_R = 3 * RP.G.dR  # 3 cells from R boundary
        offset_Z = 3 * RP.G.dZ  # 3 cells from Z boundary

        # Create wall coordinates with the offset
        wall_R_min = R_min + offset_R
        wall_R_max = R_max - offset_R
        wall_Z_min = Z_min + offset_Z
        wall_Z_max = Z_max - offset_Z

        # Create rectangular wall
        RP.wall = WallGeometry{FT}(
            [wall_R_min, wall_R_max, wall_R_max, wall_R_min, wall_R_min],  # Wall R coordinates
            [wall_Z_min, wall_Z_min, wall_Z_max, wall_Z_max, wall_Z_min]   # Wall Z coordinates
        )
    else
        # Use provided wall coordinates
        RP.wall = WallGeometry{FT}(RP.config.wall_R, RP.config.wall_Z)
    end


    # Initialize fields if not already created
    if !isdefined(RP, :fields) || isnothing(RP.fields)
        RP.fields = Fields{FT}(NR, NZ)
    end

    # Set basic field strengths
    Bpol = FT(5e-3)  # Poloidal field strength

    RP.fields.R0B0 = RP.config.R0B0

    # Create fields
    RP.fields.Bϕ = RP.fields.R0B0 ./ RP.G.R2D
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

    if isempty(RP.config.wall_R) || isempty(RP.config.wall_Z)
        # Read device wall data
        read_device_wall_data!(RP)
    else
        # Use provided wall coordinates
        RP.wall = WallGeometry{FT}(RP.config.wall_R, RP.config.wall_Z)
    end

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

function load_electron_RRCs()
    RRC_data_dir = joinpath(dirname(@__DIR__), "RRC_data")
    eRRCs_EoverP_Erg_file = joinpath(RRC_data_dir, "eRRCs_EoverP_Erg.h5")
    eRRCs_T_ud_file = joinpath(RRC_data_dir, "eRRCs_T_ud.h5")
    eRRCs = Electron_RRCs(eRRCs_EoverP_Erg_file, eRRCs_T_ud_file)
    return eRRCs
end

function load_H2_Ion_RRCs()
    RRC_data_dir = joinpath(dirname(@__DIR__), "RRC_data")
    iRRCs_T_ud_file = joinpath(RRC_data_dir, "iRRCs_T_ud.h5")
    iRRCs = H2_Ion_RRCs(iRRCs_T_ud_file)
    return iRRCs
end

function initialize_RRCs!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    RP.eRRCs = load_electron_RRCs()
    RP.iRRCs = load_H2_Ion_RRCs()
    return RP
end

function setup_grid_state_and_volumes_with_wall!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Determine boundary indices
    NR = RP.G.NR
    NZ = RP.G.NZ

    # Determine wall indices using is_inside_wall with WallGeometry object
    in_Wall_state = is_inside_wall(RP.G.R2D, RP.G.Z2D, RP.wall)

    # Fill in node information using the NodeState struct
    nodes = RP.G.nodes

    # Find linear indices of points outside and inside the wall
    nodes.in_wall_nids = findall(in_Wall_state[:])
    nodes.out_wall_nids = findall(.!in_Wall_state[:])

    # Fill in node indices information
    for j in 1:NZ
        for i in 1:NR
            nodes.rid[i, j] = i
            nodes.zid[i, j] = j
            nodes.nid[i, j] = LinearIndices((NR, NZ))[i, j]
        end
    end

    # Mark node state (-1 for outside, +1 for inside)
    nodes.state[nodes.out_wall_nids] .= -1
    nodes.state[nodes.in_wall_nids] .= 1


    # Find nodes on the wall (boundary nodes)
    nodes.on_wall_nids = Int[]
    for k in 1:length(nodes.out_wall_nids)
        nid = nodes.out_wall_nids[k]
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

    # NOTE: Update out_wall_nids to exclude on-wall nodes
    nodes.out_wall_nids = sort(setdiff(nodes.out_wall_nids, nodes.on_wall_nids))

    # Ensure we have on-wall nodes before proceeding
    if !isempty(nodes.on_wall_nids)
        # Select first on_wall node as starting point
        ini_nid = nodes.on_wall_nids[1]
        ini_rid = nodes.rid[ini_nid]
        ini_zid = nodes.zid[ini_nid]

        # Find path of 0-valued nodes (on-wall nodes) starting from initial point
        # This traces the wall along grid points that lie on the actual boundary
        path = trace_zero_contour(nodes.state, (rid=ini_rid, zid=ini_zid))

        # Extract R and Z coordinates for the fitted wall
        fitted_wall_R = Vector{FT}(undef, length(path) + 1)
        fitted_wall_Z = Vector{FT}(undef, length(path) + 1)

        # Map each path point to its actual spatial coordinates
        for (i, node) in pairs(path)
            fitted_wall_R[i] = RP.G.R2D[node.rid, node.zid]
            fitted_wall_Z[i] = RP.G.Z2D[node.rid, node.zid]
        end

        # Close the path by adding the first point at the end
        fitted_wall_R[end] = fitted_wall_R[1]
        fitted_wall_Z[end] = fitted_wall_Z[1]

        # Create the fitted wall geometry that aligns perfectly with grid points
        RP.fitted_wall = WallGeometry{FT}(fitted_wall_R, fitted_wall_Z)
    end

    # Calculate cell state using the fitted wall
    cell_centers_R = RP.G.R2D .+ 0.5 * RP.G.dR
    cell_centers_Z = RP.G.Z2D .+ 0.5 * RP.G.dZ

    # Initialize cell state (1 for inside wall, -1 for outside)
    RP.G.cell_state = fill(-1, NR, NZ)

    # Use fitted wall coordinates to determine cell states
    for i in 1:NR
        for j in 1:NZ
            if is_inside_wall(cell_centers_R[i, j], cell_centers_Z[i, j], RP.fitted_wall)
                RP.G.cell_state[i, j] = 1
            end
        end
    end

    # Calculate inVol2D - volume elements inside the wall
    RP.G.inVol2D = 2π .* RP.G.Jacob .* RP.G.dR .* RP.G.dZ

    # Set volume to zero for cells outside the wall
    RP.G.inVol2D[nodes.out_wall_nids] .= zero(FT)

    # Adjust volume for cells on the wall boundary
    for nid in nodes.on_wall_nids
        rid = nodes.rid[nid]
        zid = nodes.zid[nid]

        # Define 2x2 cell region around the boundary node
        rr = max(1, rid-1):rid
        zz = max(1, zid-1):zid

        # Calculate fraction of cell inside the wall (similar to MATLAB's sum(...,'all')/4)
        frac = sum(RP.G.cell_state[rr, zz].==1) / 4
        RP.G.inVol2D[rid, zid] = frac * RP.G.inVol2D[rid, zid]
    end

    # Calculate total device volume
    RP.G.device_inVolume = sum(RP.G.inVol2D)

    return RP
end

function initialize_density!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Set small initial seed density inside and on wall
    RP.plasma.ne .= FT(1.0e6)
    # RP.plasma.ne[RP.G.nodes.in_wall_nids] .= FT(1.0e6)
    RP.plasma.ne[RP.G.nodes.out_wall_nids] .= zero(FT)

    # Ion density matches electron for now
    RP.plasma.ni .= copy(RP.plasma.ne)

    return RP
end

function initialize_temperature!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Set initial electron temperature
    RP.plasma.Te_eV .= RP.config.constants.room_T_eV * ones(FT, RP.G.NR, RP.G.NZ)
    RP.plasma.Ti_eV .= RP.config.constants.room_T_eV * ones(FT, RP.G.NR, RP.G.NZ)
    return RP
end

function initialize_velocities!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Set initial velocities to zero
    RP.plasma.ue_para = zeros(FT, RP.G.NR, RP.G.NZ)
    RP.plasma.ui_para = zeros(FT, RP.G.NR, RP.G.NZ)

    # Initialize vector components
    @. RP.plasma.ueR = RP.plasma.ue_para * RP.fields.bR
    @. RP.plasma.ueZ = RP.plasma.ue_para * RP.fields.bZ
    @. RP.plasma.ueϕ = RP.plasma.ue_para * RP.fields.bϕ

    @. RP.plasma.uiR = RP.plasma.ui_para * RP.fields.bR
    @. RP.plasma.uiZ = RP.plasma.ui_para * RP.fields.bZ
    @. RP.plasma.uiϕ = RP.plasma.ui_para * RP.fields.bϕ

    return RP
end

function update_coulomb_collision_parameters!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # NRL formula for Coulomb logarithm
    # Note that NRL uses cgs units for density and eV for temperature

    # Constants needed for calculation
    mass_proton = RP.config.constants.mp # proton mass [kg]
    μ = RP.config.constants.mi / mass_proton   # ion mass / proton mass
    me_over_mi = RP.config.constants.me / RP.config.constants.mi  # electron to ion mass ratio

    # Calculate temperature ratio threshold
    Ti_mass_ratio = RP.plasma.Ti_eV * me_over_mi

    # Create index masks for different temperature regimes
    idx1 = @. (RP.plasma.Te_eV < Ti_mass_ratio)  # very low Te case
    idx2 = @. (!idx1 & (RP.plasma.Te_eV > 10 * RP.plasma.Zeff^2))  # normal Te case
    idx3 = @. (!idx1 & !idx2)  # low Te case

    # Calculate Coulomb logarithm based on different regimes
    # Very low Te case
    @. RP.plasma.lnΛ[idx1] = 16.0 - log(sqrt(RP.plasma.ni[idx1] * 1e-6) *
                                (RP.plasma.Ti_eV[idx1])^(-1.5) *
                                RP.plasma.Zeff[idx1]^2 * μ)

    # Normal Te case (Te_eV > 10*Zeff^2)
    @. RP.plasma.lnΛ[idx2] = 24.0 - log(sqrt(RP.plasma.ne[idx2] * 1e-6) /
                                  RP.plasma.Te_eV[idx2])

    # Low Te case
    @. RP.plasma.lnΛ[idx3] = 23.0 - log(sqrt(RP.plasma.ne[idx3] * 1e-6) *
                                 RP.plasma.Zeff[idx3] *
                                 (RP.plasma.Te_eV[idx3])^(-1.5))

    # Handle non-finite values (NaN, Inf) or non-real values
    not_valid = @. !isfinite(RP.plasma.lnΛ) | !isreal(RP.plasma.lnΛ)
    @. RP.plasma.lnΛ[not_valid] = FT(10.0)  # base value

    # Update collision frequency
    # ν_ei = n_e e^4 lnΛ / (4π ε_0^2 m_e^0.5 (kT_e)^1.5)
    ν_factor_Maxwellian = FT(1.863033936542749e-40)  # sqrt(2)*ee^4/(12π^(1.5)*ϵ0^2*sqrt(me))
    @. RP.plasma.ν_ei = ν_factor_Maxwellian * RP.plasma.Zeff^2 * RP.plasma.ni *
                        RP.plasma.lnΛ * (RP.config.constants.ee * RP.plasma.Te_eV)^(-1.5)

    Zeff = RP.plasma.Zeff
    @. RP.plasma.sptz_fac = (1+1.198*Zeff+0.222*Zeff^2)/(1+2.966*Zeff+0.753*Zeff^2);
    # Set Spitzer factor to 0.51 for Zeff=1
    RP.plasma.sptz_fac[Zeff.==1] .= FT(0.510469472194728);

    return RP
end

export initialize!
