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
@kwdef struct PlasmaConstants{FT<:AbstractFloat}
    # Basic physical constants
    ee::FT = FT(1.602176634e-19)    # Elementary charge [C]
    me::FT = FT(9.1093837015e-31)   # Electron mass [kg]
    mi::FT = FT(3.34754699166e-27)  # H2+ ion mass [kg]
    mp::FT = FT(1.67262192369e-27)  # Proton mass [kg]
    eps0::FT = FT(8.8541878128e-12) # Vacuum permittivity [F/m]
    mu0::FT = FT(1.25663706212e-6)  # Vacuum permeability [H/m]
    kB::FT = FT(1.380649e-23)       # Boltzmann constant [J/K]
    c_light::FT = FT(299792458.0)   # Speed of light in vacuum [m/s]
    room_T_eV::FT = FT(0.026)       # Room temperature [eV]

    # Derived constants
    eV_to_J::FT = ee                             # 1 eV in J
    eV_to_K::FT = ee / kB                        # 1 eV in K
    electron_mass_eV::FT = me * c_light^2 / eV_to_J  # Electron mass in eV/c²
    proton_mass_eV::FT = mp * c_light^2 / eV_to_J    # Proton mass in eV/c²
end

# Export structures and functions
export PlasmaConstants