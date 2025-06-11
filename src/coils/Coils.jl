# Coil module for RAPID2D.jl
# Electromagnetic coil and conductor modeling


# Re-export key types and functions
export Coil, CoilSystem
export add_coil!, get_powered_coils, get_passive_coils, get_controllable_coils
export find_coil_by_name, set_coil_voltage!, get_coil_voltage, update_all_voltages!, update_controllable_voltages!
export get_coil_positions, get_powered_coil_positions
export calculate_coil_resistance, calculate_self_inductance, create_coil_from_parameters
export initialize_four_wall_system!, initialize_single_wall_system!
export add_control_coils!, initialize_example_tokamak_coils!
# New functions for current and voltage access
export get_all_resistances, get_powered_resistances, get_controllable_resistances
export get_all_currents, get_powered_currents, get_controllable_currents, get_all_voltages, get_powered_voltages, get_controllable_voltages
export set_all_currents!, set_powered_currents!, set_controllable_currents!, set_coil_current!, get_coil_current
# New functions for mutual inductance and circuit matrices
export calculate_mutual_inductance_matrix!, calculate_circuit_matrices!, update_coil_system_matrices!
export get_mutual_inductance, get_coil_coupling_matrix
export get_coil_voltage_at_time, set_coil_voltage_function!
export get_all_voltages_at_time, get_powered_voltages_at_time, get_controllable_voltages_at_time
export create_sinusoidal_voltage, create_linear_voltage_ramp, create_step_voltage, create_piecewise_linear_voltage
export evaluate_voltage_ext
# Current distribution functions
export distribute_coil_currents_to_Jϕ!, distribute_coil_currents_to_Jϕ
export determine_coils_inside_grid!
# Circuit equation solvers
export solve_LR_circuit_step!
export calculate_circuit_magnetic_energy, calculate_power_dissipation

# Include source files
# include("core_functions.jl")
# include("initialization.jl")
# include("circuit_equations.jl")
