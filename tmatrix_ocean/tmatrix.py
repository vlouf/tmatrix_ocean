import os
import glob
import datetime
import warnings

import dask
import dask.bag as db
from dask.diagnostics import ProgressBar

# Other packages
import netCDF4
import numpy as np
from scipy import interpolate

import pytmatrix
from pytmatrix import orientation, radar, tmatrix_aux, refractive
from pytmatrix.psd import PSDIntegrator, GammaPSD
from pytmatrix.tmatrix import TMatrix, Scatterer
from pytmatrix.tmatrix_psd import TMatrixPSD, GammaPSD


def drop_axis_ratio(D_eq):
    """
    Axis ratio of drops with respect to their diameter.

    Parameter:
    ==========
    D_eq: float
        Drop diameter.

    Return:
    =======
    axratio: float
        Axis ratio of drop.
    """
    if D_eq < 0.7:
        axratio = 1.0  # Spherical
    elif D_eq < 1.5:
        axratio = 1.173 - 0.5165 * D_eq + 0.4698 * D_eq**2 - 0.1317 * D_eq**3 - 8.5e-3 * D_eq**4
    else:
        axratio = 1.065 - 6.25e-2 * D_eq - 3.99e-3 * D_eq**2 + 7.66e-4 * D_eq**3 - 4.095e-5 * D_eq**4

    return 1.0 / axratio


def multi_proc_manager(d_diameters, d_densities):

    if len(d_diameters) != len(d_densities):
        print(len(d_diameters), len(d_densities))
        raise IndexError("Not the same dim")

    try:
        dbz, zdr, kdp, atten_spec, atten_spec_v = scatter_off_2dvd_packed(d_diameters, d_densities)
    except Exception:
        raise

    return dbz, zdr, kdp, atten_spec, atten_spec_v


def radar_band_name(wavelength):
    """
    Get the meteorological frequency band name.

    Parameters:
    ===========
    wavelength: float
        Radar wavelength in mm.

    Returns:
    ========
    freqband: str
        Frequency band name.
    """
    if wavelength >= 100:
        return "S"
    elif wavelength >= 40:
        return "C"
    elif wavelength >= 30:
        return "X"
    elif wavelength >= 20:
        return "Ku"
    elif wavelength >= 7:
        return "Ka"
    else:
        return "W"

    return None


def scatter_off_2dvd_packed(d_diameters, d_densities):
    """
    Computing the scattering properties of homogeneous nonspherical scatterers with the T-Matrix method.

    Parameters:
    ===========
    d_diameters: array
        Drop diameters in mm! (or else returns values won't be with proper units.)
    d_densities: array
        Drop densities.

    Returns:
    ========
    dbz: array
        Horizontal reflectivity.
    zdr: array
        Differential reflectivity.
    kdp: array
        Specific differential phase (deg/km).
    atten_spec: array
        Specific attenuation (dB/km).
    """
    # Function interpolation.
    mypds = interpolate.interp1d(d_diameters, d_densities, bounds_error=False, fill_value=0.0)
    SCATTERER.psd = mypds  # GammaPSD(D0=2.0, Nw=1e3, mu=4)

    # Obtaining reflectivity and ZDR.
    dbz = 10 * np.log10(radar.refl(SCATTERER))  # in dBZ
    zdr = 10 * np.log10(radar.Zdr(SCATTERER))  # in dB

    # Specific attenuation and KDP.
    SCATTERER.set_geometry(tmatrix_aux.geom_horiz_forw)
    atten_spec = radar.Ai(SCATTERER)  # in dB/km
    atten_spec_v = radar.Ai(SCATTERER, h_pol=False)  # in dB/km
    kdp = radar.Kdp(SCATTERER)  # in deg/km

    return dbz, zdr, kdp, atten_spec, atten_spec_v


def write_netcdf(outfilename, time, diameter_bin_size, disdro_data, PSD_raw_count, dbz, zdr, kdp, atten_spec, atten_spec_v):
    # Write netCDF4 file.
    with netCDF4.Dataset(outfilename, "w", format="NETCDF4") as rootgrp:
        # Create dimension
        rootgrp.createDimension("time", len(time))
        rootgrp.createDimension("diameter", len(diameter_bin_size))

        # Create variable for these dimensions.
        nctime = rootgrp.createVariable('time', 'f8', 'time')
        ncdiam = rootgrp.createVariable('diameter', 'f8', 'diameter')

        # Populating these variables.
        nctime[:] = date2num(time, TIME_UNIT)
        nctime.units = TIME_UNIT
        ncdiam[:] = diameter_bin_size
        ncdiam.units = "mm"

        # PSD count
        ncpsd = rootgrp.createVariable("concentration_number", 'f8', ("time", "diameter"), zlib=True)
        ncpsd[:] = PSD_raw_count

        # T-Matrix results:
        ncdbz = rootgrp.createVariable("DBZ", "f8", ("time",), zlib=True)
        nczdr = rootgrp.createVariable("ZDR", "f8", ("time",), zlib=True)
        nckdp = rootgrp.createVariable("KDP", "f8", ("time",), zlib=True)
        ncatten_spec = rootgrp.createVariable("ATTEN_SPEC", "f8", ("time",), zlib=True)
        ncatten_spec_v = rootgrp.createVariable("ATTEN_SPEC_V", "f8", ("time",), zlib=True)

        ncdbz[:] = dbz
        nczdr[:] = zdr
        nckdp[:] = kdp
        ncatten_spec[:] = atten_spec
        ncatten_spec_v[:] = atten_spec_v

        ncdbz.units = "dBZ"
        nczdr.units = "dB"
        nckdp.units = "deg/km"
        ncatten_spec.units = "dB/km"
        ncatten_spec_v.units = "dB/km"

        ncdbz.setncattr_string("description", "Horizontal reflectivity")
        nczdr.setncattr_string("description", "Differential reflectivity")
        nckdp.setncattr_string("description", "Specific differential phase ")
        ncatten_spec.setncattr_string("description", "Specific attenuation for the horizontal reflectivity")
        ncatten_spec_v.setncattr_string("description", "Specific attenuation for the vertical reflectivity")

    return None


def main():
    # List all input files.
    flist = glob.glob("/g/data/kl02/vhl548/data_for_others/disdro2/*psd_na.txt")
#     flist = ['/g/data/kl02/vhl548/data_for_others/disdro2/joint_sonne2DISDRO_SO257-SO267_psd_na.txt']
    if len(flist) == 0:
        print("No file found.")
        return None

    # Getting radar for generating output file name.
    letter_band = radar_band_name(RADAR_BAND)

    for input_file in flist:
        # Generating output file name:
        outfile = os.path.basename(input_file)
        outfile = outfile.replace(".txt", ".nc")
        outfile = outfile.replace("_psd", "_PSD_TMATRIX_{}-band".format(letter_band))
        outfilename = os.path.join(OUTDIR, outfile)

        # Check if output file exists.
        if os.path.exists(outfilename):
            print("Output file already exists. Doing nothing.")
            continue

        # Read data.
        diameter_bin_size, disdro_data, PSD_raw_count = read_ascii_file(input_file)
        print("input file {} read.".format(input_file))

        # Build argument list for multiprocessing.
        myargs = [(diameter_bin_size, PSD_raw_count[cnt, :]) for cnt in range(0, len(PSD_raw_count))]
        bag = db.from_sequence(myargs).starmap(multi_proc_manager)
        with ProgressBar():
            rslt = bag.compute()        
        dbz, zdr, kdp, atten_spec, atten_spec_v = zip(*rslt)
        print("T-Matrix computation finished.")

        time = disdro_data["date"]
        print("The output file will be {}.".format(outfilename))
        write_netcdf(outfilename, time, diameter_bin_size, disdro_data, PSD_raw_count, dbz, zdr, kdp, atten_spec, atten_spec_v)
        print("Output file {} written.".format(outfilename))

    return None


if __name__ == '__main__':
    TIME_UNIT = "seconds since 1970-01-01 00:00"
    OUTDIR = "."

    # Radar band in mm.
    for RADAR_BAND in [tmatrix_aux.wl_S,
                       tmatrix_aux.wl_C,
                       tmatrix_aux.wl_X,
                       tmatrix_aux.wl_Ku,
                       tmatrix_aux.wl_Ka,
                       tmatrix_aux.wl_W]:

        print("Looking at wavelength {} mm.".format(RADAR_BAND))        
        SCATTERER = Scatterer(wavelength=RADAR_BAND, m=refractive.m_w_10C[RADAR_BAND])        
        SCATTERER.psd_integrator = PSDIntegrator()
        SCATTERER.psd_integrator.axis_ratio_func = lambda D: drop_axis_ratio(D)
        SCATTERER.psd_integrator.D_max = 8
        SCATTERER.psd_integrator.geometries = (tmatrix_aux.geom_horiz_back, tmatrix_aux.geom_horiz_forw)
        SCATTERER.or_pdf = orientation.gaussian_pdf(10.0)
        SCATTERER.orient = orientation.orient_averaged_fixed
        SCATTERER.psd_integrator.init_scatter_table(SCATTERER)

        main()
