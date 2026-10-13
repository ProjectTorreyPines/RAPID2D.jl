# The closed-surface flux constraint: its policies, its basis and its state, as types only. They
# sit here, before `types.jl`, so the flags and the run can hold them; the methods are in
# `closed_surface_flux.jl`.

"""
    ClosedSurfaceFlow

How electrons flow along B on closed flux surfaces.
"""
abstract type ClosedSurfaceFlow end

"""
    FreeParallelFlow()

**The default.** The electron parallel velocity is solved on every node alike, and its flux
moves the density on closed surfaces as on open field lines.
"""
struct FreeParallelFlow <: ClosedSurfaceFlow end

"""
    ClosedSurfaceCoupling

How the surface-flux constraint enters the step that solves u∥.
"""
abstract type ClosedSurfaceCoupling end

"""
    PostProjection()

Above the Ampère gate the coupled step is solved as before, then u∥ is solved again with the
constraint at the induced field it found; the step keeps that solve's ψ, so Ampère's law is met
for the unconstrained current (an experiment). Below the gate the constraint is in the u∥ solve.
"""
struct PostProjection <: ClosedSurfaceCoupling end

"""
    SurfaceFluxConstraint(; coupling = PostProjection())

On the closed region the electron parallel particle flux per B is a surface function,
n u∥ / B = K_e(ψ), solved from the momentum equation averaged over each surface, and the
density's parallel flux there is the stream-function flux of K_e, which leaves the density
unmoved.
"""
struct SurfaceFluxConstraint{C <: ClosedSurfaceCoupling} <: ClosedSurfaceFlow
    coupling::C
end
SurfaceFluxConstraint(; coupling::ClosedSurfaceCoupling = PostProjection()) = SurfaceFluxConstraint(coupling)

"""
    ClosedSurfaceBasis{FT}

The constraint domain Ω_c and its surface functions at one ψ:
- `nodes`: the linear ids of Ω_c; `V`: their control volumes 2πR ΔR ΔZ.
- `axis` = (; R, Z, ψ): the O-point. `ψ_cut`: the level where the surface functions end.
- `ψN`, `corner_ψN`: ψN′ = (ψ − ψ_axis)/(ψ_cut − ψ_axis) at the nodes and at the dual corners,
  `corner_ψN[i, j]` at (R_i + ΔR/2, Z_j + ΔZ/2).
  `corner_active`: the corners of Ω_c's control volumes inside the cut, where H varies; at
  every other corner H takes its value at the cut, so no flux appears where ψ elsewhere in the
  plane runs through the region's range.
- `knots`: in ψN′ from the axis (0) to the cut (1); a single knot is one constant function.
- `P`: the hats of `knots` at the nodes, zero beyond the cut, where K_e ends.
  `Pτ`: the same continued as constants beyond the cut, for temperatures, which do not end there.

Build with [`closed_surface_basis`](@ref).
"""
struct ClosedSurfaceBasis{FT <: AbstractFloat}
    nodes::Vector{Int}
    V::Vector{FT}
    axis::@NamedTuple{R::FT, Z::FT, ψ::FT}
    ψ_cut::FT
    ψN::Vector{FT}
    corner_ψN::Matrix{FT}
    corner_active::BitMatrix
    knots::Vector{FT}
    P::SparseMatrixCSC{FT, Int}
    Pτ::SparseMatrixCSC{FT, Int}
    dims::Tuple{Int, Int}
end

"""
    SurfaceFluxSolve{FT}(NR, NZ)

What the constraint's last u∥ solve leaves for the rest of the step: the surface flux `K` at the
knots of the basis with `signature` (empty: none yet on this basis), its stream-function face
fluxes `ΓR`, `ΓZ` laid out as in [`stream_function_fluxes`](@ref), and the `compression` B_pᵀK
on the grid (the pressure operator's adjoint; zero off its domain).
"""
mutable struct SurfaceFluxSolve{FT <: AbstractFloat}
    K::Vector{FT}
    signature::UInt
    ΓR::Matrix{FT}
    ΓZ::Matrix{FT}
    compression::Matrix{FT}
end
SurfaceFluxSolve{FT}(NR::Int, NZ::Int) where {FT <: AbstractFloat} =
    SurfaceFluxSolve{FT}(FT[], UInt(0), zeros(FT, NR - 1, NZ), zeros(FT, NR, NZ - 1), zeros(FT, NR, NZ))

"""
    SurfaceFluxLedger{FT}()

The constraint's cumulative signed energy ledger [J]: the kinetic energy it removed on its first
step (`activation`), on steps where the basis changed (`basis_change`) and on other steps
(`projection`), and the change from remapping u∥ to the step's new density (`remap`).
"""
mutable struct SurfaceFluxLedger{FT <: AbstractFloat}
    activation::FT
    basis_change::FT
    projection::FT
    remap::FT
end
SurfaceFluxLedger{FT}() where {FT <: AbstractFloat} = SurfaceFluxLedger{FT}(zero(FT), zero(FT), zero(FT), zero(FT))

"""
    ClosedSurfaceState{FT}(NR, NZ)

The constraint in a run, by lifetime: the `basis` of the step (`nothing` when there is none),
rebuilt after each field-line analysis; the `solve` of the step ([`SurfaceFluxSolve`](@ref));
and the cumulative `ledger` ([`SurfaceFluxLedger`](@ref)). A step solved again above the Ampère
gate puts `solve` and `ledger` back as they were.
"""
mutable struct ClosedSurfaceState{FT <: AbstractFloat}
    basis::Union{Nothing, ClosedSurfaceBasis{FT}}
    solve::SurfaceFluxSolve{FT}
    ledger::SurfaceFluxLedger{FT}
end
ClosedSurfaceState{FT}(NR::Int, NZ::Int) where {FT <: AbstractFloat} =
    ClosedSurfaceState{FT}(nothing, SurfaceFluxSolve{FT}(NR, NZ), SurfaceFluxLedger{FT}())
