# How the anisotropic diffusion ∇·(𝐃∇·) becomes a matrix: the discretization policies. Their
# methods are `assemble_anisotropic_diffusion!` in `wall_diffusion.jl`; the types sit here, before
# `types.jl`, so the simulation flags can hold one.

export AnisotropicDiscretization, LimitedNinePoint, NinePoint

"""
    AnisotropicDiscretization

How every anisotropic diffusion operator of a run is assembled from its tensor: the stencil, and
what is done with the negative couplings the stencil's cross terms bring on an oblique field.
Those couplings are what cancels the diffusion across `b̂`; a linear scheme cannot be both
monotone and consistent there. The operators (the electrons' particle diffusion and conduction,
the ions' groups) are built through [`assemble_anisotropic_diffusion!`](@ref) and never learn
which policy runs. The isotropic gas has no cross terms and no policy
([`build_isotropic_diffusion_matrix`](@ref)). [`LimitedNinePoint`](@ref) is the default; [`NinePoint`](@ref) the plain
stencil.
"""
abstract type AnisotropicDiscretization end

"""
    LimitedNinePoint(; max_ratio = 20)

The nine-point stencil with its cross terms kept, and its anti-diffusive pairs limited where the
density is not resolved: across a pair whose densities differ by more than `max_ratio`
([`limit_antidiffusive_pairs!`](@ref)). A resolved density keeps the stencil's operator bit for
bit. **The default.** Its assembly needs the density the limiter reads.
"""
struct LimitedNinePoint <: AnisotropicDiscretization
    max_ratio::Float64

    function LimitedNinePoint(max_ratio::Real)
        r = Float64(max_ratio)   # checked as stored
        (isfinite(r) && r >= 1) ||
            throw(ArgumentError("LimitedNinePoint: max_ratio = $max_ratio; it must be finite and at least 1"))
        return new(r)
    end
end
LimitedNinePoint(; max_ratio::Real = 20) = LimitedNinePoint(max_ratio)

"""
    NinePoint()

The nine-point stencil with every cross term kept, limited nowhere
([`build_wall_diffusion_matrix!`](@ref)). Consistent on an oblique field at any density, and
not monotone: across an unresolved density cliff a negative coupling can drive the low side
negative.
"""
struct NinePoint <: AnisotropicDiscretization end
