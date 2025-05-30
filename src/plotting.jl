"""
Plotting interface for RAPID2D.jl

This module provides fallback functions and documentation for plotting functionality.
The actual implementations are provided by package extensions when plotting packages are loaded.
"""

"""
    plot_snap1D(snap0D::Vector{Snapshot0D{FT}}; kwargs...) where {FT}

Plot 1D/0D time series diagnostics from RAPID2D simulation.

**Requires Plots.jl to be loaded to activate the extension.**

# Usage
```julia
using RAPID2D
using Plots  # This activates the RAPID2DPlotsExt extension

RP = RAPID{Float64}(config)
run_simulation!(RP)
plot_snap1D(RP.diagnostics.snaps0D)
```

For high-performance plotting with large datasets:
```julia
using RAPID2D
using Makie, CairoMakie  # This activates the RAPID2DMakieExt extension

makie_plot_snap1D(RP.diagnostics.snaps0D)
```

# Arguments
- `snap0D::Vector{Snapshot0D{FT}}`: Vector of 0D snapshots containing time series data
- `kwargs...`: Additional keyword arguments passed to the plotting backend

# Returns
- Plot object (type depends on the loaded plotting backend)
"""
function plot_snap1D(snap0D; kwargs...)
    error("""
    plot_snap1D requires a plotting package to be loaded.

    For Plots.jl backend:
        using Plots

    For Makie.jl backend:
        using CairoMakie  # or GLMakie
    """)
end

"""
    plot_snap2D(snap2D::Snapshot2D{FT}, R1D, Z1D; field=:ne, kwargs...) where {FT}

Plot 2D field distribution from RAPID2D simulation.

**Requires Plots.jl to be loaded to activate the extension.**

# Arguments
- `snap2D::Snapshot2D{FT}`: 2D snapshot data
- `R1D`: Radial grid coordinates
- `Z1D`: Vertical grid coordinates
- `field::Symbol`: Field to plot (see available fields below)
- `kwargs...`: Additional keyword arguments

# Available fields
- `:ne` - Electron density [m⁻³]
- `:ni` - Ion density [m⁻³]
- `:Te_eV` - Electron temperature [eV]
- `:Ti_eV` - Ion temperature [eV]
- `:B_pol` - Poloidal magnetic field [T]
- `:BR`, `:BZ` - Magnetic field components [T]
- `:E_para_tot` - Total parallel electric field [V/m]
- `:Jϕ` - Toroidal current density [A/m²]
- `:ue_para`, `:ui_para` - Parallel velocities [m/s]

# Example
```julia
using RAPID2D, Plots
RP = RAPID{Float64}(config)
run_simulation!(RP)

# Plot electron density at last time step
plot_snap2D(RP.diagnostics.snaps2D[end], RP.G.R1D, RP.G.Z1D, field=:ne)
```
"""
function plot_snap2D(snap2D, R1D, Z1D; kwargs...)
    error("""
    plot_snap2D requires a plotting package to be loaded.

    For Plots.jl backend:
        using Plots

    For Makie.jl backend:
        using CairoMakie  # or GLMakie
    """)
end

"""
    animate_snap2D(snaps2D, R1D, Z1D; field=:ne, kwargs...)

Create animation of 2D field evolution.

**Requires Plots.jl to be loaded to activate the extension.**

# Arguments
- `snaps2D::Vector{Snapshot2D{FT}}`: Vector of 2D snapshots
- `R1D`, `Z1D`: Grid coordinates
- `field::Symbol`: Field to animate
- `fps::Int`: Frames per second (default: 5)
- `filename::String`: Output filename (default: "rapid2d_animation.mp4")

# Example
```julia
using RAPID2D, Plots
RP = RAPID{Float64}(config)
run_simulation!(RP)

# Create animation of electron density evolution
animate_snap2D(RP.diagnostics.snaps2D, RP.G.R1D, RP.G.Z1D,
               field=:ne, fps=10, filename="ne_evolution.mp4")
```
"""
function animate_snap2D(snaps2D, R1D, Z1D; kwargs...)
    error("""
    animate_snap2D requires a plotting package to be loaded.

    For Plots.jl backend:
        using Plots

    For Makie.jl backend (faster for large datasets):
        using CairoMakie  # or GLMakie
    """)
end

"""
    plot_comparison(snaps0D_1, snaps0D_2; labels=["Case 1", "Case 2"], kwargs...)

Compare two sets of 1D diagnostic data.

**Requires Plots.jl to be loaded to activate the extension.**

# Example
```julia
using RAPID2D, Plots

# Run two different cases
RP1 = RAPID{Float64}(config1)
RP2 = RAPID{Float64}(config2)
run_simulation!(RP1)
run_simulation!(RP2)

# Compare results
plot_comparison(RP1.diagnostics.snaps0D, RP2.diagnostics.snaps0D,
                labels=["Low pressure", "High pressure"])
```
"""
function plot_comparison(snaps0D_1, snaps0D_2; kwargs...)
    error("""
    plot_comparison requires a plotting package to be loaded.

    For Plots.jl backend:
        using Plots

    For Makie.jl backend:
        using CairoMakie  # or GLMakie
    """)
end

# Export plotting interface functions (implementations provided by extensions)
export plot_snap1D, plot_snap2D, animate_snap2D, plot_comparison
