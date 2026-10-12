# The closed-surface flux constraint: its policies, its basis and its state, as types only. They
# sit here, before `types.jl`, so the flags and the run can hold them; the methods are in
# `closed_surface_flux.jl`. internal/docs/src/notes/plans/PLAN_closed-surface-flux.md.

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
unmoved. PLAN_closed-surface-flux.md.
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
    knots::Vector{FT}
    P::SparseMatrixCSC{FT, Int}
    Pτ::SparseMatrixCSC{FT, Int}
    dims::Tuple{Int, Int}
end

"""
    ClosedSurfaceState{FT}(NR, NZ)

The constraint's state in a run: the basis of the step (`nothing` when there is none), the
surface flux `K` of its last solve at the basis knots and the `signature` of the basis it used,
the stream-function face fluxes `ΓR`, `ΓZ` laid out as in [`stream_function_fluxes`](@ref), and the
cumulative signed energy ledger [J]: kinetic energy the constraint removed on its first step, on
steps where the basis changed, and on other steps, and the change from remapping u∥ to the new
density.
"""
mutable struct ClosedSurfaceState{FT <: AbstractFloat}
    basis::Union{Nothing, ClosedSurfaceBasis{FT}}
    K::Vector{FT}
    signature::UInt
    ΓR::Matrix{FT}
    ΓZ::Matrix{FT}
    E_activation::FT
    E_basis_change::FT
    E_projection::FT
    E_remap::FT
end
ClosedSurfaceState{FT}(NR::Int, NZ::Int) where {FT <: AbstractFloat} = ClosedSurfaceState{FT}(
    nothing, FT[], UInt(0), zeros(FT, NR - 1, NZ), zeros(FT, NR, NZ - 1), zero(FT), zero(FT), zero(FT), zero(FT),
)
