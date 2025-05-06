"""
Physical constants and conversion factors used in RAPID2D.
"""

"""
    PlasmaConstants{FT<:AbstractFloat}

Structure containing physical constants used in plasma physics simulations.
This structure consolidates all physical constants needed for simulation in one place,
matching the naming conventions used in the MATLAB version.

Fields:
- `ee`: Elementary charge [C]
- `me`: Electron mass [kg]
- `mi`: Ion mass (H2+) [kg]
- `mp`: Proton mass [kg]
- `eps0`: Vacuum permittivity [F/m]
- `mu0`: Vacuum permeability [H/m]
- `kB`: Boltzmann constant [J/K]
- `c_light`: Speed of light in vacuum [m/s]
- `room_T_eV`: Room temperature in eV [eV]
- `eV_to_J`: Conversion from eV to J [J/eV]
- `eV_to_K`: Conversion from eV to K [K/eV]
- `electron_mass_eV`: Electron mass [eV/c²]
- `proton_mass_eV`: Proton mass [eV/c²]
"""
struct PlasmaConstants{FT<:AbstractFloat}
    # Basic physical constants
    ee::FT         # Elementary charge [C]
    me::FT         # Electron mass [kg]
    mi::FT         # Ion mass (H2+) [kg]
    mp::FT         # Proton mass [kg]
    eps0::FT       # Vacuum permittivity [F/m]
    mu0::FT        # Vacuum permeability [H/m]
    kB::FT         # Boltzmann constant [J/K]
    c_light::FT    # Speed of light in vacuum [m/s]
    room_T_eV::FT  # Room temperature [eV]

    # Derived constants
    eV_to_J::FT    # Conversion from eV to J [J/eV]
    eV_to_K::FT    # Conversion from eV to K [K/eV]
    electron_mass_eV::FT # Electron mass [eV/c²]
    proton_mass_eV::FT   # Proton mass [eV/c²]

    # Constructor with default values
    function PlasmaConstants{FT}() where FT<:AbstractFloat
        # Basic constants
        ee = FT(1.602176634e-19)    # Elementary charge [C]
        me = FT(9.1093837015e-31)   # Electron mass [kg]
        mp = FT(1.67262192369e-27)  # Proton mass [kg]
        mi = FT(3.34754699166e-27)  # H2+ ion mass [kg]
        eps0 = FT(8.8541878128e-12) # Vacuum permittivity [F/m]
        mu0 = FT(1.25663706212e-6)  # Vacuum permeability [H/m]
        kB = FT(1.380649e-23)       # Boltzmann constant [J/K]
        c_light = FT(299792458.0)   # Speed of light in vacuum [m/s]
        room_T_eV = FT(0.026)       # Room temperature [eV]

        # Derived constants
        eV_to_J = ee                  # 1 eV in J
        eV_to_K = ee / kB             # 1 eV in K
        electron_mass_eV = me * c_light^2 / eV_to_J  # Electron mass in eV/c²
        proton_mass_eV = mp * c_light^2 / eV_to_J    # Proton mass in eV/c²

        new{FT}(ee, me, mi, mp, eps0, mu0, kB, c_light, room_T_eV,
                eV_to_J, eV_to_K, electron_mass_eV, proton_mass_eV)
    end
end

"""
    load_constants!(config)

Load physical constants into the configuration struct.

# Arguments
- `config`: Configuration struct to be populated with constants
"""
function load_constants!(config)
    # Check if the config already has a parameterized type, or use Float64 as default
    FT = isa(config, SimulationConfig{<:AbstractFloat}) ? eltype(config.R0B0) : Float64

    # Create physical constants with the appropriate floating-point type
    config.constants = PlasmaConstants{FT}()

    # Update legacy fields for backward compatibility
    config.ee = config.constants.ee
    config.me = config.constants.me
    config.mi = config.constants.mi
    config.eps0 = config.constants.eps0
    config.mu0 = config.constants.mu0
    config.kB = config.constants.kB

    return nothing
end

# Export structures and functions
export PlasmaConstants, load_constants!