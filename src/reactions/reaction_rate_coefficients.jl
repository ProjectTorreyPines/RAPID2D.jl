using HDF5
using Interpolations

"""
    AbstractReactionRateCoefficient{T<:AbstractFloat}

Abstract type for all reaction rate coefficient models.
Concrete implementations must provide methods to compute or interpolate reaction rates for different conditions.
"""
abstract type AbstractReactionRateCoefficient{T<:AbstractFloat} end

"""
    RRC_EoverP_Erg{FT<:AbstractFloat} <: AbstractReactionRateCoefficient{FT}

Reaction rate coefficient model based on electric field over pressure (E/p) and particle energy.
Stores both raw data and an interpolation object for efficient calculation of rate coefficients.

# Fields
- `EoverP::Vector{FT}`: Electric field over pressure (E/p) coordinates
- `Erg_eV::Vector{FT}`: Particle energy in eV
- `raw_data::AbstractArray{FT}`: Raw reaction rate data as a matrix
- `itp`: Interpolation object with extrapolation for quick access"""
struct RRC_EoverP_Erg{FT<:AbstractFloat} <: AbstractReactionRateCoefficient{FT}
	# 2 variables for given reaction rate coefficient
	EoverP::Vector{FT}  # Electric field over pressure (E/p) coordinates
	Erg_eV::Vector{FT}  # Particle's energy

	raw_data::AbstractArray{FT}
	itp  # Interpolation object with extrapolation

	function RRC_EoverP_Erg(EoverP::Vector{FT}, Erg_eV::Vector{FT}, raw_data::AbstractArray{FT}) where FT<:AbstractFloat
		itp = interpolate((EoverP, Erg_eV), raw_data, Gridded(Linear()))
		itp_extrap = extrapolate(itp, FT(0.0))  # Return 0 outside interpolation bounds
		new{FT}(EoverP, Erg_eV, raw_data, itp_extrap)
	end
end

"""
    RRC_T_ud{FT<:AbstractFloat} <: AbstractReactionRateCoefficient{FT}

Reaction rate coefficient model based on temperature and parallel drift velocity.
Used for reactions where the rate depends on temperature and drift velocity.

# Fields
- `T_eV::Vector{FT}`: Temperature in eV
- `ud_para::Vector{FT}`: Parallel drift velocity
- `raw_data::AbstractArray{FT}`: Raw reaction rate data as a matrix
- `itp`: Interpolation object with extrapolation for quick access
"""
struct RRC_T_ud{FT<:AbstractFloat} <: AbstractReactionRateCoefficient{FT}
	# 2 variables for given reaction rate coefficient
	T_eV::Vector{FT}  # Temperature in eV
	ud_para::Vector{FT}  # parallel velocity

	raw_data::AbstractArray{FT}
	itp  # Interpolation object with extrapolation

	function RRC_T_ud(T_eV::Vector{FT}, ud_para::Vector{FT}, raw_data::AbstractArray{FT}) where FT<:AbstractFloat
		itp = interpolate((T_eV, ud_para), raw_data, Gridded(Linear()))
		itp_extrap = extrapolate(itp, FT(0.0))  # Return 0 outside interpolation bounds
		new{FT}(T_eV, ud_para, raw_data, itp_extrap)
	end
end

"""
    RRC_T_ud_gFac{FT<:AbstractFloat} <: AbstractReactionRateCoefficient{FT}

Reaction rate coefficient model based on temperature, parallel drift velocity, and distribution function g-factor.
Used for more complex reactions where the distribution function shape affects the rate.

# Fields
- `T_eV::Vector{FT}`: Temperature in eV
- `ud_para::Vector{FT}`: Parallel drift velocity
- `gFac::Vector{FT}`: g-factor of the distribution function
- `raw_data::AbstractArray{FT}`: Raw reaction rate data
- `itp`: Interpolation object with extrapolation for quick access
"""
struct RRC_T_ud_gFac{FT<:AbstractFloat} <: AbstractReactionRateCoefficient{FT}
	# 3 variables for given reaction rate coefficient
	T_eV::Vector{FT}  # Temperature in eV
	ud_para::Vector{FT}  # parallel velocity
	gFac::Vector{FT}  # g-factor of Distribution function

	raw_data::AbstractArray{FT}
	itp  # Interpolation object with extrapolation

	function RRC_T_ud_gFac(T_eV::Vector{FT}, ud_para::Vector{FT}, gFac::Vector{FT}, raw_data::AbstractArray{FT}) where FT<:AbstractFloat
		itp = interpolate((T_eV, ud_para, gFac), raw_data, Gridded(Linear()))
		itp_extrap = extrapolate(itp, FT(0.0))  # Return 0 outside interpolation bounds
		new{FT}(T_eV, ud_para, gFac, raw_data, itp_extrap)
	end
end

"""
    Electron_RRCs{FT<:AbstractFloat} <: AbstractSpeciesRRCs{FT}

Container for electron-related reaction rate coefficient models.
Stores various reaction models for electron-neutral and electron-ion interactions.

# Fields
- `Ionization`: Rate coefficient for electron impact ionization
- `Momentum`: Rate coefficient for momentum transfer
- `Total_Excitation`: Rate coefficient for all excitation processes
- `Dissoc_Ionz`: Rate coefficient for dissociative ionization
- `Halpha`: Rate coefficient for Halpha emission
- `Recomb_H2Ion`: Rate coefficient for H2+ recombination
- `Recomb_H3Ion`: Rate coefficient for H3+ recombination
"""
struct Electron_RRCs{FT<:AbstractFloat} <: AbstractSpeciesRRCs{FT}
    Ionization::RRC_EoverP_Erg{FT}
    Momentum::RRC_EoverP_Erg{FT}
    Total_Excitation::RRC_EoverP_Erg{FT}

	Dissoc_Ionz::RRC_T_ud{FT}
	Halpha::RRC_T_ud{FT}
	Recomb_H2Ion::RRC_T_ud{FT}
	Recomb_H3Ion::RRC_T_ud{FT}

	function Electron_RRCs(eRRC_EoverP_Erg_fileName::String, eRRC_T_ud_fileName::String)
		@assert isfile(eRRC_EoverP_Erg_fileName) "File not found: $eRRC_EoverP_Erg_fileName"
		@assert isfile(eRRC_T_ud_fileName) "File not found: $eRRC_T_ud_fileName"

		# Create RRC_EoverP_Erg objects for each reaction type from the given H5 file
		h5fid = h5open(eRRC_EoverP_Erg_fileName,"r");
		EoverP = read(h5fid, "EoverP")
		Erg_eV = read(h5fid, "Erg_eV")
		Ionization = RRC_EoverP_Erg(EoverP, Erg_eV, read(h5fid, "Ionization"))
		Momentum = RRC_EoverP_Erg(EoverP, Erg_eV, read(h5fid, "Momentum"))
		Total_Excitation = RRC_EoverP_Erg(EoverP, Erg_eV, read(h5fid, "Total_Excitation"))
		close(h5fid)

		# Create RRC_T_ud objects for each reaction type from the given H5 file
		h5fid = h5open(eRRC_T_ud_fileName,"r");
		T_eV = read(h5fid, "T_eV")
		ud_para = read(h5fid, "ud_para")
		Dissoc_Ionz = RRC_T_ud(T_eV, ud_para, read(h5fid, "Dissoc_Ionz"))
		Halpha = RRC_T_ud(T_eV, ud_para, read(h5fid, "Halpha"))
		Recomb_H2Ion = RRC_T_ud(T_eV, ud_para, read(h5fid, "Recomb_H2Ion"))
		Recomb_H3Ion = RRC_T_ud(T_eV, ud_para, read(h5fid, "Recomb_H3Ion"))
		close(h5fid)


		FT = eltype(EoverP)  # Determine the floating-point type from the data

		new{FT}(Ionization, Momentum, Total_Excitation,
			Dissoc_Ionz, Halpha, Recomb_H2Ion, Recomb_H3Ion)
	end
end

"""
    H2_Ion_RRCs{FT<:AbstractFloat} <: AbstractSpeciesRRCs{FT}

Container for H2+ ion-related reaction rate coefficient models.
Stores various reaction models for H2+ interactions with background gas.

# Fields
- `Elastic`: Rate coefficient for elastic collisions
- `Charge_Exchange`: Rate coefficient for charge exchange processes
- `Target_Ionization`: Rate coefficient for ionization of target particles
- `Projectile_Dissociation`: Rate coefficient for dissociation of projectile ions
- `Particle_Exchange`: Rate coefficient for particle exchange processes
"""
struct H2_Ion_RRCs{FT<:AbstractFloat} <: AbstractSpeciesRRCs{FT}
    Elastic::RRC_T_ud{FT}
    Charge_Exchange::RRC_T_ud{FT}
    Target_Ionization::RRC_T_ud{FT}
	Projectile_Dissociation::RRC_T_ud{FT}
	Particle_Exchange::RRC_T_ud{FT}

	function H2_Ion_RRCs(iRRCs_T_ud_fileName::String)
		# Create RRC_T_ud objects for each reaction type from the given H5 file
		h5fid = h5open(iRRCs_T_ud_fileName,"r");
		T_eV = read(h5fid, "T_eV")
		ud_para = read(h5fid, "ud_para")
		Elastic = RRC_T_ud(T_eV, ud_para, read(h5fid, "Elastic"))
		Charge_Exchange = RRC_T_ud(T_eV, ud_para, read(h5fid, "Charge_Exchange"))
		Target_Ionization = RRC_T_ud(T_eV, ud_para, read(h5fid, "Target_Ionization"))
		Projectile_Dissociation = RRC_T_ud(T_eV, ud_para, read(h5fid, "Projectile_Dissociation"))
		Particle_Exchange = RRC_T_ud(T_eV, ud_para, read(h5fid, "Particle_Exchange"))
		close(h5fid)

		FT = eltype(T_eV)  # Determine the floating-point type from the data

		new{FT}(Elastic, Charge_Exchange, Target_Ionization, Projectile_Dissociation, Particle_Exchange)
	end
end

"""
    get_electron_RRC(RP::RAPID{FT}, eRRCs::Electron_RRCs{FT}, reaction::Symbol) where FT<:AbstractFloat

Calculate electron reaction rate coefficients for the specified reaction using interpolation.
Automatically selects appropriate physical parameters from the RAPID model.

# Arguments
- `RP::RAPID{FT}`: RAPID plasma model containing physical state variables
- `eRRCs::Electron_RRCs{FT}`: Container of electron reaction rate coefficient models
- `reaction::Symbol`: Symbol specifying which reaction to compute (e.g., :Ionization)

# Returns
- RRC (reaction rate coefficient) values at each spatial point
"""
function get_electron_RRC(RP::RAPID{FT}, eRRCs::Electron_RRCs{FT}, reaction::Symbol) where FT<:AbstractFloat
	if hasfield(typeof(eRRCs), reaction)
		mass = RP.config.constants.me;
		ee = RP.config.constants.ee;

		RRC = getfield(eRRCs, reaction)
		if RRC isa RRC_EoverP_Erg
			mean_Ke_eV = @. 1.5*RP.plasma.Te_eV + 0.5*mass*RP.plasma.ue_para^2/ee;
			abs_Epara_over_pGas = @. abs(RP.fields.E_para_tot/(RP.plasma.n_H2_gas*RP.plasma.T_gas_eV*ee));
			return RRC.itp.(abs_Epara_over_pGas, mean_Ke_eV)
		elseif RRC isa RRC_T_ud
			return RRC.itp.(RP.plasma.Te_eV, abs.(RP.plasma.ue_para))
		end
	else
		throw(ArgumentError("Invalid reaction type: $reaction"))
	end
end

# Convenience dispatch
function get_electron_RRC(RP::RAPID{FT}, reaction::Symbol) where FT<:AbstractFloat
	return get_electron_RRC(RP, RP.eRRCs, reaction)
end

"""
    get_H2_ion_RRC(RP::RAPID{FT}, iRRCs::H2_Ion_RRCs{FT}, reaction::Symbol) where FT<:AbstractFloat

Calculate H2+ ion reaction rate coefficients for the specified reaction using interpolation.
Automatically selects appropriate physical parameters from the RAPID model.

# Arguments
- `RP::RAPID{FT}`: RAPID plasma model containing physical state variables
- `iRRCs::H2_Ion_RRCs{FT}`: Container of H2+ ion reaction rate coefficient models
- `reaction::Symbol`: Symbol specifying which reaction to compute (e.g., :Elastic)

# Returns
- RRC (reaction rate coefficient) values at each spatial point
"""
function get_H2_ion_RRC(RP::RAPID{FT}, iRRCs::H2_Ion_RRCs{FT}, reaction::Symbol) where FT<:AbstractFloat
	if hasfield(typeof(iRRCs), reaction)
		RRC = getfield(iRRCs, reaction)
		if RRC isa RRC_T_ud
			return RRC.itp.(RP.plasma.Ti_eV, abs.(RP.plasma.ui_para))
		end
	else
		throw(ArgumentError("Invalid reaction type: $reaction"))
	end
end

# Convenience dispatch
function get_H2_ion_RRC(RP::RAPID{FT}, reaction::Symbol) where FT<:AbstractFloat
	return get_H2_ion_RRC(RP, RP.iRRCs, reaction)
end

# Export types and functions for reaction rate coefficients
export AbstractReactionRateCoefficient
export RRC_EoverP_Erg, RRC_T_ud, RRC_T_ud_gFac
export Electron_RRCs, H2_Ion_RRCs
export get_electron_RRC, get_H2_ion_RRC