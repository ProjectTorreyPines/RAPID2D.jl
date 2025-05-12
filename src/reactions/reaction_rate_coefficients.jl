using HDF5
using Interpolations

"""
    AbstractReactionRateCoefficient{T<:AbstractFloat}

Abstract type for all reaction rate coefficient models.
Concrete implementations must provide methods to compute or interpolate reaction rates for different conditions.
"""
abstract type AbstractReactionRateCoefficient{T<:AbstractFloat} end


struct RRC_2vars{FT<:AbstractFloat} <: AbstractReactionRateCoefficient{FT}
	# 2 variables for given reaction rate coefficient
	EoverP::Vector{FT}  # Electric field over pressure (E/p) coordinates
	Erg_eV::Vector{FT}  # Particle's energy

	raw_data::AbstractArray{FT}
	itp::Interpolations.GriddedInterpolation

	function RRC_2vars(EoverP::Vector{FT}, Erg_eV::Vector{FT}, raw_data::AbstractArray{FT}) where FT<:AbstractFloat
		itp = interpolate((EoverP, Erg_eV), raw_data, Gridded(Linear()))
		new{FT}(EoverP, Erg_eV, raw_data, itp)
	end
end

struct RRC_3vars{FT<:AbstractFloat} <: AbstractReactionRateCoefficient{FT}
	# 3 variables for given reaction rate coefficient
	T_eV::Vector{FT}  # Temperature in eV
	ud_para::Vector{FT}  # parallel velocity
	gFac::Vector{FT}  # g-factor of Distribution function

	raw_data::AbstractArray{FT}
	itp::Interpolations.GriddedInterpolation

	function RRC_3vars(T_eV::Vector{FT}, ud_para::Vector{FT}, gFac::Vector{FT}, raw_data::AbstractArray{FT}) where FT<:AbstractFloat
		itp = interpolate((T_eV, ud_para, gFac), raw_data, Gridded(Linear()))
		new{FT}(T_eV, ud_para, gFac, raw_data, itp)
	end
end


struct Electron_RRC{FT<:AbstractFloat}
    Ionization::RRC_2vars{FT}
    Momentum::RRC_2vars{FT}
    TotalExcitation::RRC_2vars{FT}

	function Electron_RRC(RRC2vars_fileName::String)
		# Create RRC objects for each reaction type from the given H5 file
		h5fid = h5open(RRC2vars_fileName,"r");

		# Read the data from the HDF5 file
		EoverP = read(h5fid, "EoverP")
		Erg_eV = read(h5fid, "Erg_eV")

		Ionization = RRC_2vars(EoverP, Erg_eV, read(h5fid, "Ionization"))
		Momentum = RRC_2vars(EoverP, Erg_eV, read(h5fid, "Momentum"))
		TotalExcitation = RRC_2vars(EoverP, Erg_eV, read(h5fid, "TotalExcitation"))

		close(h5fid)

		FT = eltype(EoverP)  # Determine the floating-point type from the data

		new{FT}(Ionization, Momentum, TotalExcitation)
	end
end


