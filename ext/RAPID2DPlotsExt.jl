"""
RAPID2DPlotsExt.jl - Plots.jl extension for RAPID2D.jl

This extension provides plotting functionality for RAPID2D using Plots.jl.
It includes recipes and convenience functions for visualizing simulation data.
"""
module RAPID2DPlotsExt

using RAPID2D
using Plots

# Import types for function definitions
import RAPID2D: Snapshot0D, Snapshot2D, RAPID, Diagnostics

"""
    RAPID2D.plot_snap1D(snap0D; kwargs...)

Plot 1D/0D time series diagnostics from RAPID2D simulation.

# Arguments
- `snap0D`: Vector of 0D snapshots
- `kwargs...`: Additional keyword arguments for plot customization

# Example
```julia
using RAPID2D, Plots
RP = RAPID{Float64}(config)
run_simulation!(RP)
plot_snap1D(RP.diagnostics.snaps0D)
```
"""
function RAPID2D.plot_snap1D(snap0D; kwargs...)
    if isempty(snap0D)
        error("No snapshot data available for plotting")
    end

    # Extract time series
    times_ms = [s.time_s * 1e3 for s in snap0D]  # Convert to ms

    # Create multi-panel plot
    p1 = plot(times_ms, [s.ne for s in snap0D],
              ylabel="⟨ne⟩ (m⁻³)", label="Electron density",
              yscale=:log10, linewidth=2)

    p2 = plot(times_ms, [s.Te_eV for s in snap0D],
              ylabel="⟨Te⟩ (eV)", label="Electron temperature",
              linewidth=2)

    p3 = plot(times_ms, [abs(s.Epara_tot) for s in snap0D],
              ylabel="|⟨E∥⟩| (V/m)", label="Parallel E-field",
              yscale=:log10, linewidth=2)

    p4 = plot(times_ms, [abs(s.ue_para) for s in snap0D],
              ylabel="|⟨u∥⟩| (m/s)", label="Parallel velocity",
              xlabel="Time (ms)", yscale=:log10, linewidth=2)

    return plot(p1, p2, p3, p4, layout=(2,2), size=(800, 600),
                plot_title="RAPID2D Time Evolution",
                left_margin=5Plots.mm, right_margin=5Plots.mm,
                top_margin=5Plots.mm, bottom_margin=5Plots.mm,
                kwargs...)
end

"""
    plot_snap2D(snap2D::Snapshot2D{FT}, R1D, Z1D; field=:ne, kwargs...) where {FT}

Plot 2D field distribution from RAPID2D simulation.

# Arguments
- `snap2D::Snapshot2D{FT}`: 2D snapshot data
- `R1D`: Radial grid coordinates
- `Z1D`: Vertical grid coordinates
- `field::Symbol`: Field to plot (default: :ne)
- `kwargs...`: Additional keyword arguments

# Available fields
- `:ne` - Electron density
- `:Te_eV` - Electron temperature
- `:B_pol` - Poloidal magnetic field
- `:E_para_tot` - Total parallel electric field
- `:Jϕ` - Toroidal current density
"""
function RAPID2D.plot_snap2D(snap2D, R1D, Z1D;
                     field=:ne, colorscale=:auto, streamlines=true, wall=nothing, kwargs...)

    # Get field data
    field_data = getfield(snap2D, field)
    R2D = R1D' .* ones(length(Z1D))
    Z2D = ones(length(R1D))' .* Z1D

    # Determine color scale and limits
    if colorscale == :auto
        if field in [:ne, :ni]
            colorscale = :log10
            clims = (maximum(field_data) * 1e-3, maximum(field_data))
        elseif field in [:Te_eV, :Ti_eV]
            colorscale = :linear
            clims = (0, maximum(field_data))
        else
            colorscale = :linear
            clims = (minimum(field_data), maximum(field_data))
        end
    end


    # Create base heatmap
    p = heatmap(R1D, Z1D, field_data',
                aspect_ratio=:equal,
                color=:turbo,
                xlabel="R (m)", ylabel="Z (m)",
                title="$(field) at t = $(round(snap2D.time_s*1e3, digits=2)) ms",
                colorbar_title=get_field_label(field),
                left_margin=2Plots.mm, right_margin=8Plots.mm,
                top_margin=3Plots.mm, bottom_margin=3Plots.mm)

    if colorscale == :log10
        if clims[1] <= 0
            new_clims = (min(1e-3, 1e-3*clims[2]), clims[2])  # Avoid log10 of zero or negative
        else
            new_clims = clims
        end
        p = plot!(p, clims=new_clims, colorbar_scale=:log10)
    end


    # Add wall if provided
    if wall !== nothing && hasfield(typeof(wall), :R) && hasfield(typeof(wall), :Z)
        plot!(p, wall.R, wall.Z, color=:red, linewidth=2, label="Wall")
    end

    # Add streamlines for magnetic field
    if streamlines && hasfield(typeof(snap2D), :BR) && hasfield(typeof(snap2D), :BZ)
        # Note: Streamline plotting would need additional implementation
        # This is a placeholder for the concept
    end

    # Set tight limits and return
    plot!(p, xlims=(minimum(R1D), maximum(R1D)),
             ylims=(minimum(Z1D), maximum(Z1D)))
    return plot(p; kwargs...)
end

"""
    get_field_label(field::Symbol)

Get appropriate axis label for a given field symbol.
"""
function get_field_label(field::Symbol)
    labels = Dict(
        :ne => "ne (m⁻³)",
        :ni => "ni (m⁻³)",
        :Te_eV => "Te (eV)",
        :Ti_eV => "Ti (eV)",
        :B_pol => "Bpol (T)",
        :BR => "BR (T)",
        :BZ => "BZ (T)",
        :E_para_tot => "E∥ (V/m)",
        :Jϕ => "Jϕ (A/m²)",
        :ue_para => "ue∥ (m/s)",
        :ui_para => "ui∥ (m/s)"
    )
    return get(labels, field, string(field))
end

"""
    animate_snap2D(snaps2D::Vector{Snapshot2D{FT}}, R1D, Z1D; field=:ne, fps=5, kwargs...) where {FT}

Create animation of 2D field evolution.

# Arguments
- `snaps2D::Vector{Snapshot2D{FT}}`: Vector of 2D snapshots
- `R1D`, `Z1D`: Grid coordinates
- `field::Symbol`: Field to animate
- `fps::Int`: Frames per second
"""
function RAPID2D.animate_snap2D(snaps2D, R1D, Z1D;
                       field=:ne, fps=5, filename="rapid2d_animation.mp4", kwargs...)

    if isempty(snaps2D)
        error("No snapshot data available for animation")
    end

    # Determine global color limits for consistency
    all_data = [getfield(snap, field) for snap in snaps2D]
    global_min = minimum(minimum.(all_data))
    global_max = maximum(maximum.(all_data))

    if field in [:ne, :ni] && global_min > 0
        clims = (global_max * 1e-3, global_max)
        colorscale = :log10
    else
        clims = (global_min, global_max)
        colorscale = :linear
    end

    # Create animation
    anim = @animate for (i, snap) in enumerate(snaps2D)
        RAPID2D.plot_snap2D(snap, R1D, Z1D; field=field, colorscale=colorscale,
                   clims=clims, title="$(get_field_label(field)) at t = $(round(snap.time_s*1e3, digits=2)) ms",
                   kwargs...)
    end

    return mp4(anim, filename, fps=fps)
end

"""
    plot_comparison(snaps0D_1::Vector{Snapshot0D{FT}}, snaps0D_2::Vector{Snapshot0D{FT}};
                   labels=["Case 1", "Case 2"], kwargs...) where {FT}

Compare two sets of 1D diagnostic data.
"""
function RAPID2D.plot_comparison(snaps0D_1, snaps0D_2;
                        labels=["Case 1", "Case 2"], kwargs...)

    times_1 = [s.time_s * 1e3 for s in snaps0D_1]
    times_2 = [s.time_s * 1e3 for s in snaps0D_2]

    p1 = plot(times_1, [s.ne for s in snaps0D_1],
              ylabel="⟨ne⟩ (m⁻³)", label=labels[1],
              yscale=:log10, linewidth=2)
    plot!(p1, times_2, [s.ne for s in snaps0D_2],
          label=labels[2], linewidth=2, linestyle=:dash)

    p2 = plot(times_1, [s.Te_eV for s in snaps0D_1],
              ylabel="⟨Te⟩ (eV)", label=labels[1], linewidth=2)
    plot!(p2, times_2, [s.Te_eV for s in snaps0D_2],
          label=labels[2], linewidth=2, linestyle=:dash)

    p3 = plot(times_1, [abs(s.Epara_tot) for s in snaps0D_1],
              ylabel="|⟨E∥⟩| (V/m)", label=labels[1],
              yscale=:log10, linewidth=2)
    plot!(p3, times_2, [abs(s.Epara_tot) for s in snaps0D_2],
          label=labels[2], linewidth=2, linestyle=:dash)

    p4 = plot(times_1, [s.𝒲e_eV for s in snaps0D_1],
              ylabel="⟨𝒲e⟩ (eV)", label=labels[1],
              xlabel="Time (ms)", linewidth=2)
    plot!(p4, times_2, [s.𝒲e_eV for s in snaps0D_2],
          label=labels[2], linewidth=2, linestyle=:dash)

    return plot(p1, p2, p3, p4, layout=(2,2), size=(800, 600),
                plot_title="RAPID2D Comparison",
                left_margin=5Plots.mm, right_margin=5Plots.mm,
                top_margin=5Plots.mm, bottom_margin=5Plots.mm,
                kwargs...)
end

# Note: Extension functions are automatically available when the extension is loaded
# No need to export since they extend RAPID2D functions

end # module RAPID2DPlotsExt
