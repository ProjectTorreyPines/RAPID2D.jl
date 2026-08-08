# Types needed for RAPID2D field line following analysis

"""
Why a trace ended, and what `Lc` holds for each outcome:

- `:wall`        — distance to the vessel along `B`
- `:closed`      — one circuit length (the geometry repeats after it)
- `:null`        — `2πR`: `Bpol = 0` means purely toroidal, a closed circle
- `:trace_limit` — `Inf`: the budget (`max_Lpol`/`max_steps`) ran out, distance UNKNOWN

`Inf` therefore has exactly one meaning — the measurement failed — and
[`validate_field_line_terminations!`](@ref) refuses to let it reach a consumer:
an unknown distance must never read as a measured one.
"""
const FLF_TERMINATIONS = (:wall, :closed, :null, :trace_limit)

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
- `termination_forward::Matrix{Symbol}`: Why the forward trace ended, see [`FLF_TERMINATIONS`](@ref)
- `termination_backward::Matrix{Symbol}`: Why the backward trace ended
- `max_Lpol::FT`: Maximum allowed poloidal length
- `max_step::Int`: Maximum number of integration steps

`Lpol_*` is distance **travelled** in the poloidal projection; `Lc_*` is the geometric
bound along `B` — wall distance, circuit, or `2πR` at a null. `Lc = Inf` only records a
failed trace; see [`FLF_TERMINATIONS`](@ref).
"""
@kwdef mutable struct FieldLineFollowingResult{FT <: AbstractFloat}
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

    # `:unset` = "no trace has run"; a plausible default would let an unpopulated grid
    # pass a topology check.
    termination_forward::Matrix{Symbol} = fill(:unset, dims_RZ)
    termination_backward::Matrix{Symbol} = fill(:unset, dims_RZ)

    closed_surface_nids::Vector{Int} = Int[]

    max_Lpol::FT = FT(0.0)
    max_step::Int = 0
end

function FieldLineFollowingResult{FT}(NR::Int, NZ::Int) where {FT <: AbstractFloat}
    flf = FieldLineFollowingResult{FT}(dims_RZ = (NR, NZ))
    sizehint!(flf.closed_surface_nids, NR * NZ)  # Preallocate for closed surfaces
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
- `termination::Symbol`: Why the trace ended, see [`FLF_TERMINATIONS`](@ref)
- `is_closed::Bool`: Whether field line closed (360° circulation)
- `hit_wall::Bool`: Whether field line hit the wall
- `final_R::FT`: Final R coordinate
- `final_Z::FT`: Final Z coordinate

`Lc` is finite for every termination except `:trace_limit`; see
[`FLF_TERMINATIONS`](@ref) for the per-outcome values.
"""
@kwdef mutable struct SingleTraceResult{FT <: AbstractFloat}
    Lpol::FT = zero(FT)
    Lc::FT = zero(FT)
    min_Bpol::FT = zero(FT)
    steps::Int = zero(Int)
    termination::Symbol = :unset
    is_closed::Bool = false
    hit_wall::Bool = false
    final_R::FT = zero(FT)
    final_Z::FT = zero(FT)
end
