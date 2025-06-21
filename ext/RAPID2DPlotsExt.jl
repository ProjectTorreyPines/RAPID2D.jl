__precompile__(false)

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
    RAPID2D.plot_snaps0D(snaps0D; kwargs...)

Plot 1D/0D time series diagnostics from RAPID2D simulation.

# Arguments
- `snaps0D`: Vector of 0D snapshots
- `kwargs...`: Additional keyword arguments for plot customization

# Example
```julia
using RAPID2D, Plots
RP = RAPID{Float64}(config)
run_simulation!(RP)
plot_snaps0D(RP.diagnostics.snaps0D)
```
"""
function RAPID2D.plot_snaps0D(snaps0D; kwargs...)
    if isempty(snaps0D)
        error("No snapshot data available for plotting")
    end


    # Extract time series
    times_ms = [s.time_s * 1e3 for s in snaps0D]  # Convert to ms


	p_vec = Plots.Plot[];


    # Create multi-panel plot
    push!(p_vec, plot(times_ms, [s.ne for s in snaps0D],
              ylabel="⟨ne⟩ (m⁻³)", label="Electron density",
              yscale=:log10, linewidth=2))

    push!(p_vec, plot(times_ms[2:end], abs.(snaps0D.I_tor[2:end]),
              ylabel="⟨I_tor⟩ (A)", label="Toroidal current",
              yscale=:log10, linewidth=2))

    push!(p_vec, plot(times_ms, [s.Te_eV for s in snaps0D],
              ylabel="⟨Te⟩ (eV)", label="Electron temperature",
              linewidth=2))
	push!(p_vec, plot(times_ms, [s.Ke_eV for s in snaps0D],
			  ylabel="⟨We⟩ (eV)", label="Electron energy",
			  linewidth=2))

	push!(p_vec, plot(times_ms, [s.Ti_eV for s in snaps0D],
              ylabel="⟨Ti⟩ (eV)", label="Ion temperature",
              linewidth=2))

	p = plot(times_ms,  [s.ν_en_iz for s in snaps0D],
			  ylabel="⟨νiz⟩ (s⁻¹)", label="Ionization rate",
			  linewidth=2)
	plot!(p, times_ms, [s.eLoss_rate for s in snaps0D],
			  label="loss rate", linestyle=:dash,
			  linewidth=2)
	push!(p_vec, p)

	push!(p_vec, plot(times_ms, [s.Epara_tot for s in snaps0D],
			  ylabel="|⟨E∥⟩| (V/m)", label="Parallel E-field",
			  linewidth=2))
	plot!(p_vec[end], times_ms, [s.Epara_ext for s in snaps0D],
			label="E_{ext}", linewidth=2)
	plot!(p_vec[end], times_ms, [s.Epara_self_ES for s in snaps0D],
			label="E_{self}^{ES}", linewidth=2)
	plot!(p_vec[end], times_ms, [s.Epara_self_EM for s in snaps0D],
			label="E_{self}^{EM}", linewidth=2)

	push!(p_vec, plot(times_ms, [abs(s.ue_para) for s in snaps0D],
			  ylabel="|⟨u∥⟩| (m/s)", label="Parallel velocity",
			  xlabel="Time (ms)", linewidth=2))

	ncols = 2
	nrows = ceil(Int, length(p_vec) / ncols)


    return plot(p_vec..., layout=(nrows, ncols), size=(1000, 1200),
                plot_title="RAPID2D Time Evolution",
                left_margin=5Plots.mm, right_margin=5Plots.mm,
                top_margin=5Plots.mm, bottom_margin=5Plots.mm,
                kwargs...)
end

"""
    plot_snaps2D(snap2D::Snapshot2D{FT}, R1D, Z1D; field=:ne, kwargs...) where {FT}

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
function RAPID2D.plot_snaps2D(snap2D, R1D, Z1D, fields::AbstractArray{Symbol}; colorscale=:auto, streamlines=true, wall=nothing, kwargs...)

    # Get field data
	p_vec = Plots.Plot[];

    for field in fields
        push!(p_vec,
                plot_snaps2D(snap2D, R1D, Z1D, field; colorscale=colorscale,
                   streamlines=streamlines, wall=wall, kwargs...)
            )
    end

	ncols = 2
	nrows = ceil(Int, length(p_vec) / ncols)

    return plot(p_vec...; layout=(nrows, ncols), size=(800, 1200),
                plot_title="RAPID2D 2D fields",
                left_margin=5Plots.mm, right_margin=5Plots.mm,
                top_margin=5Plots.mm, bottom_margin=5Plots.mm,
                kwargs...)
end

function RAPID2D.plot_snaps2D(snap2D, R1D, Z1D, field; colorscale=:auto, streamlines=true, wall=nothing, kwargs...)

    # Get field data
    field_data = getfield(snap2D, field)
    R2D = R1D' .* ones(length(Z1D))
    Z2D = ones(length(R1D))' .* Z1D

    # # Determine color scale and limits
    # if colorscale == :auto
    #     if field in [:ne, :ni]
    #         colorscale = :log10
    #         clims = (maximum(field_data) * 1e-3, maximum(field_data))
    #     elseif field in [:Te_eV, :Ti_eV]
    #         colorscale = :linear
    #         clims = (0, maximum(field_data))
    #     else
    #         colorscale = :linear
    #         clims = (minimum(field_data), maximum(field_data))
    #     end
    # end


    # Create base heatmap
    p = heatmap(R1D, Z1D, field_data',
                aspect_ratio=:equal,
                color=:turbo,
                xlabel="R (m)", ylabel="Z (m)",
                title="$(field) at t = $(round(snap2D.time_s*1e3, digits=2)) ms",
                colorbar_title=get_field_label(field),
                left_margin=2Plots.mm, right_margin=8Plots.mm,
                top_margin=3Plots.mm, bottom_margin=3Plots.mm)

    ψ = getfield(snap2D, :ψ)
    contour!(p, R1D, Z1D, ψ', levels=range(extrema(ψ)...,11), color=:white)

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
        plot!(p, wall.R, wall.Z, color=:red, linewidth=2, label="")
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
    animate_snaps2D(snaps2D::Vector{Snapshot2D{FT}}, R1D, Z1D; field=:ne, fps=5, kwargs...) where {FT}

Create animation of 2D field evolution.

# Arguments
- `snaps2D::Vector{Snapshot2D{FT}}`: Vector of 2D snapshots
- `R1D`, `Z1D`: Grid coordinates
- `field::Symbol`: Field to animate
- `fps::Int`: Frames per second
"""
function RAPID2D.animate_snaps2D(snaps2D, R1D, Z1D, field::Symbol; fps=5, filename="rapid2d_animation.mp4", headless=true, kwargs...)

    if isempty(snaps2D)
        error("No snapshot data available for animation")
    end

    # Optionally set headless mode for animation
    if headless
        original_backend = Plots.backend()
        original_gks = get(ENV, "GKSwstype", nothing)
        Plots.gr(show=false)
        ENV["GKSwstype"] = "nul"
    end

    try
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

        # Create animation without displaying
        anim = @animate for (i, snap) in enumerate(snaps2D)
            RAPID2D.plot_snaps2D(snap, R1D, Z1D, field; colorscale=colorscale,
                       clims=clims, title="$(get_field_label(field)) at t = $(round(snap.time_s*1e3, digits=2)) ms",
                       kwargs...)
        end every 1

        return mp4(anim, filename, fps=fps)
    finally
        # Restore original settings if they were changed
        if headless
            Plots.backend(original_backend)
            if original_gks === nothing
                delete!(ENV, "GKSwstype")
            else
                ENV["GKSwstype"] = original_gks
            end
        end
    end
end

function RAPID2D.animate_snaps2D(snaps2D, R1D, Z1D, fields::AbstractArray{Symbol}; fps=5, filename="rapid2d_animation.mp4", headless=true, kwargs...)

    if isempty(snaps2D)
        error("No snapshot data available for animation")
    end

    # Optionally set headless mode for animation
    if headless
        original_backend = Plots.backend()
        original_gks = get(ENV, "GKSwstype", nothing)
        Plots.gr(show=false)
        ENV["GKSwstype"] = "nul"
    end

    try
        # Create animation without displaying
        anim = @animate for (i, snap) in enumerate(snaps2D)
            RAPID2D.plot_snaps2D(snap, R1D, Z1D, fields; size=(800,1200), dpi=151, kwargs...)
        end every 1

        return mp4(anim, filename, fps=fps)
    finally
        # Restore original settings if they were changed
        if headless
            Plots.backend(original_backend)
            if original_gks === nothing
                delete!(ENV, "GKSwstype")
            else
                ENV["GKSwstype"] = original_gks
            end
        end
    end
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

    p4 = plot(times_1, [s.Ke_eV for s in snaps0D_1],
              ylabel="⟨𝒲e⟩ (eV)", label=labels[1],
              xlabel="Time (ms)", linewidth=2)
    plot!(p4, times_2, [s.Ke_eV for s in snaps0D_2],
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
