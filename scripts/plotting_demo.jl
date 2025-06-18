#!/usr/bin/env julia
"""
Demo script showing RAPID2D.jl plotting capabilities with package extensions.

This script demonstrates:
1. Loading RAPID2D without plotting dependencies (lightweight)
2. Adding Plots.jl for basic plotting functionality
3. Adding Makie.jl for high-performance plotting
4. Creating various types of plots and animations
"""

println("=== RAPID2D.jl Plotting Extensions Demo ===\n")

## Step 1: Load core RAPID2D (no plotting dependencies)
println("Step 1: Loading core RAPID2D package...")
using RAPID2D
println("✓ RAPID2D loaded successfully (no plotting dependencies)")

# Test that plotting functions exist but give helpful errors
println("\nTesting fallback plotting functions:")
try
    config = SimulationConfig{Float64}()
    RP = RAPID{Float64}(config)
    plot_snaps0D(RP.diagnostics.snaps0D)
catch e
    println("✓ Expected error: ", split(string(e), '\n')[1])
end

## Step 3: Create a simple simulation for demo
println("\nStep 3: Running a minimal simulation...")
config = SimulationConfig{Float64}()
config.NR, config.NZ = 20, 30
config.t_end_s = 1e-4
config.dt = 1e-6
config.snap0D_Δt_s = 2e-5
config.snap2D_Δt_s = 5e-5
config.prefilled_gas_pressure = 1e-3
config.R0B0=3.0

RP = RAPID{Float64}(config)

# Initialize with some basic setup
initialize!(RP)

# Create some fake data for demonstration
for i in 1:5
    # Create fake snapshot data
    snap0D = Snapshot0D{Float64}()
    snap0D.time_s = i * 2e-5
    snap0D.ne = 1e19 * (1 + 0.1 * sin(i))
    snap0D.Te_eV = 10.0 * (1 + 0.2 * cos(i))
    snap0D.Epara_tot = 1000.0 * exp(-i/10)
    snap0D.ue_para = 1e5 * (1 + 0.3 * sin(i))
    snap0D.Ke_eV = 15.0 * (1 + 0.1 * i)
    push!(RP.diagnostics.snaps0D, snap0D)

    # Create fake 2D data
    snap2D = Snapshot2D{Float64}(dims_RZ=(config.NR, config.NZ))
    snap2D.time_s = i * 5e-5

    # Create some interesting 2D patterns
    for (ir, r) in enumerate(RP.G.R1D), (iz, z) in enumerate(RP.G.Z1D)
        # Gaussian-like distribution with time evolution
        r_center, z_center = 1.5, 0.0
        σ = 0.3
        amp = 1e19 * (1 + 0.2 * sin(i))
        snap2D.ne[ir, iz] = amp * exp(-((r-r_center)^2 + (z-z_center)^2)/(2*σ^2))
        snap2D.Te_eV[ir, iz] = 10.0 + 5.0 * sin(i + r + z)
        snap2D.B_pol[ir, iz] = 0.1 * r * (1 + 0.1 * cos(i + z))
    end
    push!(RP.diagnostics.snaps2D, snap2D)
end

println("✓ Demo simulation data created")

## Step 4: Demonstrate plotting with Plots.jl
using Plots
println("\nStep 4: Creating plots with Plots.jl backend...")

try
    # 1D time series plot
    p1 = plot_snaps0D(RP.diagnostics.snaps0D)
    savefig(p1, "demo_time_series_plots.png")
    println("✓ Time series plot saved as 'demo_time_series_plots.png'")

    # 2D density plot
    p2 = plot_snaps2D(RP.diagnostics.snaps2D[end], RP.G.R1D, RP.G.Z1D, field=:ne)
    savefig(p2, "demo_density_2d_plots.png")
    println("✓ 2D density plot saved as 'demo_density_2d_plots.png'")

    # 2D temperature plot
    p3 = plot_snaps2D(RP.diagnostics.snaps2D[end], RP.G.R1D, RP.G.Z1D, field=:Te_eV)
    savefig(p3, "demo_temperature_2d_plots.png")
    println("✓ 2D temperature plot saved as 'demo_temperature_2d_plots.png'")

    # Animation
    println("Creating animation with Plots.jl (this may take a moment)...")
    animate_snaps2D(RP.diagnostics.snaps2D, RP.G.R1D, RP.G.Z1D,
                   field=:ne, fps=2, filename="demo_evolution_plots..mp4")
    println("✓ Animation saved as 'demo_evolution_plots.mp4'")

catch e
    println("⚠ Plots.jl demo failed: ", e)
end

## Step 5: Demonstrate Makie.jl for high performance
using GLMakie
println("\nStep 5: Loading Makie.jl backend for high-performance plotting...")
try
    # using Makie, CairoMakie
    println("✓ Makie.jl loaded - RAPID2DMakieExt extension activated")

    # High-performance 1D plot
    fig1 = plot_snaps0D(RP.diagnostics.snaps0D)
    save("demo_time_series_makie.png", fig1)
    println("✓ Makie time series saved as 'demo_time_series_makie.png'")

    # High-performance 2D plot
    fig2 = plot_snaps2D(RP.diagnostics.snaps2D[end], RP.G.R1D, RP.G.Z1D, field=:ne)
    save("demo_density_2d_makie.png", fig2)
    println("✓ Makie 2D plot saved as 'demo_density_2d_makie.png'")

    # High-performance animation
    println("Creating animation with Makie.jl (faster for large datasets)...")
    animate_snaps2D(RP.diagnostics.snaps2D, RP.G.R1D, RP.G.Z1D,
                         field=:ne, fps=3, filename="demo_evolution_makie.mp4")
    println("✓ Makie animation saved as 'demo_evolution_makie.mp4'")

catch e
    println("⚠ Makie.jl not available or demo failed: ", e)
    println("  To install: using Pkg; Pkg.add([\"Makie\", \"CairoMakie\"])")
end

# Summary
println("\n=== Demo Complete ===")
println("Files created:")
files = ["demo_time_series_plots.png", "demo_density_2d_plots.png",
         "demo_temperature_2d_plots.png", "demo_evolution_plots..mp4",
         "demo_time_series_makie.png", "demo_density_2d_makie.png",
         "demo_evolution_makie.mp4"]

for file in files
    if isfile(file)
        println("  ✓ $file")
    end
end

println("\nKey benefits of the extension approach:")
println("  • Core RAPID2D.jl remains lightweight")
println("  • Users choose their preferred plotting backend")
println("  • Automatic extension loading when plotting packages are loaded")
println("  • Makie.jl provides better performance for large datasets")
println("  • Plots.jl provides easier syntax for quick analysis")
