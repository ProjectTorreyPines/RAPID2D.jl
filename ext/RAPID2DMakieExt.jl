__precompile__(false)

"""
RAPID2DMakieExt.jl - Makie.jl extension for RAPID2D.jl

This extension provides high-performance plotting functionality for RAPID2D using Makie.jl.
Offers better performance for large datasets and interactive visualization.
"""

module RAPID2DMakieExt

using RAPID2D
# using Makie
# using CairoMakie
using GLMakie

# Import types for method extension
import RAPID2D: Snapshot0D, Snapshot2D, RAPID, Diagnostics

"""
    RAPID2D.plot_snaps0D(snap0D::Vector{Snapshot0D{FT}}; kwargs...) where {FT}

Plot 1D/0D time series diagnostics using Makie for better performance.
"""
function RAPID2D.plot_snaps0D(snap0D; kwargs...)
    if isempty(snap0D)
        error("No snapshot data available for plotting")
    end

    fig = Figure(size=(800, 600),
                figure_padding=5,  # Tight padding around the figure
                )

    times_ms = [s.time_s * 1e3 for s in snap0D]

    # Electron density
    ax1 = Axis(fig[1, 1], ylabel="⟨ne⟩ (m⁻³)", yscale=log10)
    lines!(ax1, times_ms, [s.ne for s in snap0D], linewidth=2, label="Electron density")

    # Electron temperature
    ax2 = Axis(fig[1, 2], ylabel="⟨Te⟩ (eV)")
    lines!(ax2, times_ms, [s.Te_eV for s in snap0D], linewidth=2, label="Electron temperature")

    # Parallel E-field
    ax3 = Axis(fig[2, 1], ylabel="|⟨E∥⟩| (V/m)", yscale=log10)
    lines!(ax3, times_ms, [abs(s.Epara_tot) for s in snap0D], linewidth=2, label="Parallel E-field")

    # Parallel velocity
    ax4 = Axis(fig[2, 2], ylabel="|⟨u∥⟩| (m/s)", xlabel="Time (ms)", yscale=log10)
    lines!(ax4, times_ms, [abs(s.ue_para) for s in snap0D], linewidth=2, label="Parallel velocity")

    return fig
end

"""
    RAPID2D.plot_snaps2D(snap2D::Snapshot2D{FT}, R1D, Z1D; field=:ne, kwargs...) where {FT}

High-performance 2D plotting using Makie.
"""
function RAPID2D.plot_snaps2D(snap2D, R1D, Z1D;
                           field=:ne, colormap=:turbo, kwargs...)

    field_data = getfield(snap2D, field)

    fig = Figure(size=(600, 500), figure_padding=5)
    ax = Axis(fig[1, 1], xlabel="R (m)", ylabel="Z (m)", aspect=DataAspect(),
              title="$(field) at t = $(round(snap2D.time_s*1e3, digits=2)) ms")

    # Use heatmap for better performance with large datasets
    hm = heatmap!(ax, R1D, Z1D, field_data, colormap=colormap)
    Colorbar(fig[1, 2], hm, label=get_makie_field_label(field))

    return fig
end

"""
    RAPID2D.animate_snaps2D(snaps2D::Vector{Snapshot2D{FT}}, R1D, Z1D; field=:ne, kwargs...) where {FT}

Create interactive animation using Makie.
"""
function RAPID2D.animate_snaps2D(snaps2D, R1D, Z1D;
                             field=:ne, colormap=:turbo, fps=10, filename="rapid2d_makie.mp4", kwargs...)

    if isempty(snaps2D)
        error("No snapshot data available for animation")
    end

    fig = Figure(size=(800, 600), figure_padding=5)

    # Observable for animation
    frame_idx = Observable(1)
    field_data = @lift(getfield(snaps2D[$frame_idx], field))

    # Create axis with static title
    ax = Axis(fig[1, 1], xlabel="R (m)", ylabel="Z (m)", aspect=DataAspect())
    # ax = Axis(fig[1, 1], xlabel="R (m)", ylabel="Z (m)", aspect=DataAspect(),
    #           title="$(field) animation")

    # Create a text element that can be updated for the dynamic title
    title_text = text!(ax.scene, 0.5, 1.05, text=@lift("$(field) at t = $(round(snaps2D[$frame_idx].time_s*1e3, digits=2)) ms"),
                      align=(:center, :bottom), space=:relative, fontsize=16)

    hm = heatmap!(ax, R1D, Z1D, field_data, colormap=colormap)
    Colorbar(fig[1, 2], hm, label=get_makie_field_label(field))

    # Record animation
    record(fig, filename, 1:length(snaps2D); framerate=fps) do i
        frame_idx[] = i
    end

    return fig
end

"""
    get_makie_field_label(field::Symbol)

Get appropriate axis label for Makie plots.
"""
function get_makie_field_label(field::Symbol)
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
    RAPID2D.plot_comparison(snaps0D_1::Vector{Snapshot0D{FT}}, snaps0D_2::Vector{Snapshot0D{FT}};
                   labels=["Case 1", "Case 2"], kwargs...) where {FT}

Compare two sets of 1D diagnostic data using Makie.
"""
function RAPID2D.plot_comparison(snaps0D_1, snaps0D_2;
                        labels=["Case 1", "Case 2"], kwargs...)

    fig = Figure(size=(800, 600), figure_padding=5)

    times_1 = [s.time_s * 1e3 for s in snaps0D_1]
    times_2 = [s.time_s * 1e3 for s in snaps0D_2]

    # Electron density comparison
    ax1 = Axis(fig[1, 1], ylabel="⟨ne⟩ (m⁻³)", yscale=log10)
    lines!(ax1, times_1, [s.ne for s in snaps0D_1], linewidth=2, label=labels[1])
    lines!(ax1, times_2, [s.ne for s in snaps0D_2], linewidth=2, linestyle=:dash, label=labels[2])
    axislegend(ax1)

    # Electron temperature comparison
    ax2 = Axis(fig[1, 2], ylabel="⟨Te⟩ (eV)")
    lines!(ax2, times_1, [s.Te_eV for s in snaps0D_1], linewidth=2, label=labels[1])
    lines!(ax2, times_2, [s.Te_eV for s in snaps0D_2], linewidth=2, linestyle=:dash, label=labels[2])

    # Parallel E-field comparison
    ax3 = Axis(fig[2, 1], ylabel="|⟨E∥⟩| (V/m)", yscale=log10)
    lines!(ax3, times_1, [abs(s.Epara_tot) for s in snaps0D_1], linewidth=2, label=labels[1])
    lines!(ax3, times_2, [abs(s.Epara_tot) for s in snaps0D_2], linewidth=2, linestyle=:dash, label=labels[2])

    # Energy comparison
    ax4 = Axis(fig[2, 2], ylabel="⟨𝒲e⟩ (eV)", xlabel="Time (ms)")
    lines!(ax4, times_1, [s.𝒲e_eV for s in snaps0D_1], linewidth=2, label=labels[1])
    lines!(ax4, times_2, [s.𝒲e_eV for s in snaps0D_2], linewidth=2, linestyle=:dash, label=labels[2])

    return fig
end

# Note: Extension functions are automatically available when the extension is loaded
# No need to export since they extend RAPID2D functions

end # module RAPID2DMakieExt
