# Green's function utilities for RAPID2D.jl
# Contains Green's function calculations for electromagnetic problems, particularly
# for the Grad-Shafranov equation and mutual inductance calculations

using SpecialFunctions

export calculate_ψ_by_green_function

"""
    calculate_ψ_by_green_function(R_dest, Z_dest, R_src, Z_src, I_src;
                                   compute_derivatives::Bool=false)

Calculate the Green's function for the Grad-Shafranov equation.

This function computes the magnetic flux ψ at destination points from current sources
using elliptic integrals. The Green's function represents the response at destination
points (R_dest, Z_dest) due to toroidal current sources at (R_src, Z_src).

# Mathematical formulation
The Green's function for the Grad-Shafranov equation is:

```
m = 4(R_d R_s) / [(R_d + R_s)² + (Z_d - Z_s)²]
ψ = I_s × 2×10⁻⁷ × √(R_d R_s / m) × [(2-m)K(m) - 2E(m)]
```

where K(m) and E(m) are complete elliptic integrals of the first and second kind.

# Arguments
- `R_dest`: Radial coordinates of destination points [m]
- `Z_dest`: Vertical coordinates of destination points [m]
- `R_src`: Radial coordinates of current sources [m]
- `Z_src`: Vertical coordinates of current sources [m]
- `I_src`: Current values at source points [A] (can be scalar or array)
- `compute_derivatives::Bool=false`: Whether to compute partial derivatives

# Returns
- `ψ`: Magnetic flux at destination points [Wb/rad]
- `derivatives`: Named tuple with partial derivatives (if requested):
  - `dψ_dRdest`: ∂ψ/∂R_dest
  - `dψ_dZdest`: ∂ψ/∂Z_dest
  - `dψ_dRsrc`: ∂ψ/∂R_src
  - `dψ_dZsrc`: ∂ψ/∂Z_src

# Notes
- Output dimensions: (size(dest) × size(src))
"""
function calculate_ψ_by_green_function(R_dest::AbstractArray{FT},
                                        Z_dest::AbstractArray{FT},
                                        R_src::AbstractArray{FT},
                                        Z_src::AbstractArray{FT},
                                        I_src::AbstractArray{FT}
                                        ;
                                        compute_derivatives::Bool=false) where {FT<:AbstractFloat}
    @assert size(R_dest) == size(Z_dest) "R_dest and Z_dest must have the same size"
    @assert size(R_src) == size(Z_src) "R_src and Z_src must have the same size"
    @assert size(R_src) == size(I_src) "I_src size must match size(R_src)"

    # Convert to column and row vectors for broadcasting
    Rs = reshape(R_src, 1, :)  # Row vector (1 × N_src)
    Zs = reshape(Z_src, 1, :)  # Row vector (1 × N_src)
    Is = reshape(I_src, 1, :)  # Row vector (1 × N_src)
    Rd = reshape(R_dest, :, 1) # Column vector (N_dest × 1)
    Zd = reshape(Z_dest, :, 1) # Column vector (N_dest × 1)

    # Calculate parameter m for elliptic integrals
    # m = 4(R_d R_s) / [(R_d + R_s)² + (Z_d - Z_s)²]
    m = @. 4.0 * (Rd * Rs) / ((Rd + Rs)^2 + (Zd - Zs)^2)

    # Calculate complete elliptic integrals
    K = ellipk.(m)
    E = ellipe.(m)

    # Calculate the Green's function
    # ψ = I_s × 2×10⁻⁷ × √(Rd*Rs/m) × [(2-m)K - 2E]
    ψ = @. Is * FT(2e-7) * sqrt((Rd * Rs) / m) * ((FT(2.0) - m) * K - FT(2.0) * E)

    # Reshape to match expected output dimensions
    output_size = (size(R_dest)..., size(R_src)...)
    ψ = reshape(ψ, output_size)

    if !compute_derivatives
        return ψ
    end

    # Calculate derivatives if requested
    dψ_dm = @. Is * 2e-7 * sqrt.((Rd * Rs) / m) * (-K / m + (2.0 - m) / (2.0 * m * (1.0 - m)) * E)

    denominator = @. ((Rd + Rs)^2 + (Zd - Zs)^2)^2

    # Derivatives with respect to destination coordinates
    dmdRd = @. (4.0 * Rs * (-Rd^2 + Rs^2 + (Zd - Zs)^2)) / denominator
    dψ_dRd = @. dmdRd * dψ_dm

    dmdZd = @. -8.0 * (Rd * Rs) * (Zd - Zs) / denominator
    dψ_dZd = @. dmdZd * dψ_dm

    # Derivatives with respect to source coordinates
    dmdRs = @. (4.0 * Rd * (-Rs^2 + Rd^2 + (Zd - Zs)^2)) / denominator
    dψ_dRs = @. dmdRs * dψ_dm

    dmdZs = @. -8.0 * (Rd * Rs) * (Zs - Zd) / denominator
    dψ_dZs = @. dmdZs * dψ_dm

    # Reshape derivatives to match output dimensions
    derivatives = (
        dψ_dRdest = reshape(dψ_dRd, output_size),
        dψ_dZdest = reshape(dψ_dZd, output_size),
        dψ_dRsrc = reshape(dψ_dRs, output_size),
        dψ_dZsrc = reshape(dψ_dZs, output_size)
    )

    return ψ, derivatives
end

"""
    calculate_ψ_by_green_function(R_dest, Z_dest, R_src, Z_src, I_src::Number;
                                   compute_derivatives::Bool=false)

Convenience method for scalar current source. Converts scalar I_src to array and calls main method.
"""
function calculate_ψ_by_green_function(R_dest::AbstractArray{FT},
                                        Z_dest::AbstractArray{FT},
                                        R_src::AbstractArray{FT},
                                        Z_src::AbstractArray{FT},
                                        I_src::Number;
                                        compute_derivatives::Bool=false) where {FT<:AbstractFloat}
    # Convert scalar current to array matching source coordinates
    I_array = fill(FT(I_src), size(R_src))
    return calculate_ψ_by_green_function(R_dest, Z_dest, R_src, Z_src, I_array;
                                         compute_derivatives=compute_derivatives)
end
