using HDF5
using Interpolations

"""
    AbstractReactionRateCoefficient{T<:AbstractFloat}

Abstract type for all reaction rate coefficient models.
Concrete implementations must provide methods to compute or interpolate reaction rates for different conditions.
"""
abstract type AbstractReactionRateCoefficient{T<:AbstractFloat} end

struct RRC_EoverP_Erg{FT<:AbstractFloat} <: AbstractReactionRateCoefficient{FT}
	# 2 variables for given reaction rate coefficient
	EoverP::Vector{FT}  # Electric field over pressure (E/p) coordinates
	Erg_eV::Vector{FT}  # Particle's energy

	raw_data::AbstractArray{FT}
	itp::Interpolations.GriddedInterpolation

	function RRC_EoverP_Erg(EoverP::Vector{FT}, Erg_eV::Vector{FT}, raw_data::AbstractArray{FT}) where FT<:AbstractFloat
		itp = interpolate((EoverP, Erg_eV), raw_data, Gridded(Linear()))
		new{FT}(EoverP, Erg_eV, raw_data, itp)
	end
end

struct RRC_T_ud{FT<:AbstractFloat} <: AbstractReactionRateCoefficient{FT}
	# 2 variables for given reaction rate coefficient
	T_eV::Vector{FT}  # Temperature in eV
	ud_para::Vector{FT}  # parallel velocity

	raw_data::AbstractArray{FT}
	itp::Interpolations.GriddedInterpolation

	function RRC_T_ud(T_eV::Vector{FT}, ud_para::Vector{FT}, raw_data::AbstractArray{FT}) where FT<:AbstractFloat
		itp = interpolate((T_eV, ud_para), raw_data, Gridded(Linear()))
		new{FT}(T_eV, ud_para, raw_data, itp)
	end
end

struct RRC_T_ud_gFac{FT<:AbstractFloat} <: AbstractReactionRateCoefficient{FT}
	# 3 variables for given reaction rate coefficient
	T_eV::Vector{FT}  # Temperature in eV
	ud_para::Vector{FT}  # parallel velocity
	gFac::Vector{FT}  # g-factor of Distribution function

	raw_data::AbstractArray{FT}
	itp::Interpolations.GriddedInterpolation

	function RRC_T_ud_gFac(T_eV::Vector{FT}, ud_para::Vector{FT}, gFac::Vector{FT}, raw_data::AbstractArray{FT}) where FT<:AbstractFloat
		itp = interpolate((T_eV, ud_para, gFac), raw_data, Gridded(Linear()))
		new{FT}(T_eV, ud_para, gFac, raw_data, itp)
	end
end

struct Electron_RRCs{FT<:AbstractFloat}
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
		Dissoc_Ionz = RRC_EoverP_Erg(T_eV, ud_para, read(h5fid, "Dissoc_Ionz"))
		Halpha = RRC_EoverP_Erg(T_eV, ud_para, read(h5fid, "Halpha"))
		Recomb_H2Ion = RRC_EoverP_Erg(T_eV, ud_para, read(h5fid, "Recomb_H2Ion"))
		Recomb_H3Ion = RRC_EoverP_Erg(T_eV, ud_para, read(h5fid, "Recomb_H3Ion"))
		close(h5fid)


		FT = eltype(EoverP)  # Determine the floating-point type from the data

		new{FT}(Ionization, Momentum, Total_Excitation,
			Dissoc_Ionz, Halpha, Recomb_H2Ion, Recomb_H3Ion)
	end
end

struct H2_Ion_RRCs{FT<:AbstractFloat}
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

function get_electron_RRC(RP::RAPID{FT}, eRRCs::Electron_RRCs{FT}, reaction::Symbol) where FT<:AbstractFloat
	if hasfield(eRRCs, reaction)
		mass = RP.constants.me;
		ee = RP.constants.ee;

		RRC = getfield(eRRCs, reaction)
		if RRC isa RRC_EoverP_Erg
			mean_eErg_eV = 1.5*RP.plasma.Te_eV + 0.5*mass*RP.plasma.ue_para^2/ee;
			abs_Epara_over_pGas = abs(RP.fields.E_para_tot./(RP.plasma.n_H2_gas.*RP.plasma.T_gas_eV*ee));
			return RRC.itp.(mean_eErg_eV, abs_Epara_over_pGas)
		elseif RRC isa RRC_T_ud_gFac
			return RRC.itp.(RP.plasma.Te_eV, abs.(RP.plasma.ue_para), RP.plasma.gFac_e)
		else
			throw(ArgumentError("Invaild eRRCs' Data type: $(typeof(RRC))"))
		end
	else
		throw(ArgumentError("Invalid reaction type: $reaction"))
	end
end

function get_H2_ion_RRC(RP::RAPID{FT}, iRRCs::H2_Ion_RRCs{FT}, reaction::Symbol) where FT<:AbstractFloat
	if hasfield(iRRCs, reaction)
		mass = RP.constants.mi;
		ee = RP.constants.ee;

		RRC = getfield(iRRCs, reaction)
		if RRC isa RRC_EoverP_Erg
			mean_eErg_eV = 1.5*RP.plasma.Ti_eV + 0.5*mass*RP.plasma.ui_para^2/ee;
			abs_Epara_over_pGas = abs(RP.fields.E_para_tot./(RP.plasma.n_H2_gas.*RP.plasma.T_gas_eV*ee));
			return RRC.itp.(mean_eErg_eV, abs_Epara_over_pGas)
		else
			throw(ArgumentError("Invaild iRRCs' Data type: $(typeof(RRC))"))
		end
	else
		throw(ArgumentError("Invalid reaction type: $reaction"))
	end
end
