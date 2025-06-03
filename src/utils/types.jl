# Types needed for RAPID2D field line following analysis

"""
    FieldLineFollowingResult{FT<:AbstractFloat}

Results structure for field line following analysis.

# Fields
- `Lpol_forward::Matrix{FT}`: Poloidal length in forward direction
- `Lpol_backward::Matrix{FT}`: Poloidal length in backward direction
- `Lpol_tot::Matrix{FT}`: Total poloidal length
- `Lc_forward::Matrix{FT}`: Connection length in forward direction
- `Lc_backward::Matrix{FT}`: Connection length in backward direction
- `Lc_tot::Matrix{FT}`: Total connection length
- `min_Bpol::Matrix{FT}`: Minimum poloidal field along field line
- `step::Array{Int,2}`: Number of integration steps taken
- `is_closed::Array{Bool,2}`: Whether field line is closed (360° circulation)
- `max_Lpol::FT`: Maximum allowed poloidal length
- `max_step::Int`: Maximum number of integration steps
"""
@kwdef mutable struct FieldLineFollowingResult{FT<:AbstractFloat}
    dims_RZ::Tuple{Int, Int}  # Dimensions of the RZ grid

    Lpol_forward::Matrix{FT} = zeros(FT, dims_RZ)
    Lpol_backward::Matrix{FT} = zeros(FT, dims_RZ)
    Lpol_tot::Matrix{FT} = zeros(FT, dims_RZ)
    Lc_forward::Matrix{FT} = zeros(FT, dims_RZ)
    Lc_backward::Matrix{FT} = zeros(FT, dims_RZ)
    Lc_tot::Matrix{FT} = zeros(FT, dims_RZ)
    min_Bpol::Matrix{FT} = zeros(FT, dims_RZ)
    step::Matrix{Int} = zeros(Int, dims_RZ)
    is_closed::Matrix{Bool} = zeros(Bool, dims_RZ)

    closed_surface_nids::Vector{Int} = Int[]

    max_Lpol::FT = FT(0.0)
    max_step::Int = 0
end

function FieldLineFollowingResult{FT}(NR::Int, NZ::Int) where {FT<:AbstractFloat}
    flf = FieldLineFollowingResult{FT}(dims_RZ=(NR, NZ))
    sizehint!(flf.closed_surface_nids, NR*NZ)  # Preallocate for closed surfaces
    return flf
end


"""
    SingleTraceResult{FT<:AbstractFloat}

Result of tracing a single magnetic field line in one direction.

# Fields
- `Lpol::FT`: Poloidal length traveled
- `Lc::FT`: Connection length traveled
- `min_Bpol::FT`: Minimum poloidal field encountered
- `steps::Int`: Number of integration steps taken
- `is_closed::Bool`: Whether field line closed (360° circulation)
- `hit_wall::Bool`: Whether field line hit the wall
- `final_R::FT`: Final R coordinate
- `final_Z::FT`: Final Z coordinate
"""
@kwdef mutable struct SingleTraceResult{FT<:AbstractFloat}
    Lpol::FT = zero(FT)
    Lc::FT = zero(FT)
    min_Bpol::FT = zero(FT)
    steps::Int = zero(FT)
    is_closed::Bool = false
    hit_wall::Bool = false
    final_R::FT = zero(FT)
    final_Z::FT = zero(FT)
end