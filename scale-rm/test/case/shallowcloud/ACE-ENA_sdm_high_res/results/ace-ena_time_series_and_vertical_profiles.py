#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 16:42:12 2024

@author: YCZ
"""
import xarray as xr
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import concurrent
import re

num_cores = 40

def read_value_from_config(key_to_find, file_path):
    """
    Searches the specified configuration file for a given key and returns the numeric value found
    between an equal sign and a comma following the key.

    :param file_path: Path to the configuration file
    :param key_to_find: The key for which the value is sought
    :return: The float or integer value associated with the key, or None if key is not found or format is incorrect
    """
    with open(file_path, 'r') as file:
        for line in file:
            if key_to_find in line:
                parts = line.split('=')
                if len(parts) > 1:
                    value_part = parts[1].split(',')[0]

                    match = re.search(r'[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?', value_part)
                    if match:
                        return float(match.group(0))
                    else:
                        print("No valid number found after the key.")
                        return None
    print(f"Key '{key_to_find}' not found in the file.")
    return None

config_path = '../run.conf'

# Get PRC_NUM_X and PRC_NUM_Y from the configuration dictionary and calculate the total number of processes
PRC_NUM_X = int(read_value_from_config('PRC_NUM_X', config_path))
PRC_NUM_Y = int(read_value_from_config('PRC_NUM_Y', config_path))
total_processes = PRC_NUM_X * PRC_NUM_Y
IMAX = int(read_value_from_config('IMAX', config_path))
JMAX = int(read_value_from_config('JMAX', config_path))
DZ = read_value_from_config('DZ', config_path)

start_tm = 0 # in sec
output_interval1 = 30 # in sec
output_interval2 = 300 # in sec
start_index = start_tm // output_interval2
water_density = 1e6  # water density, g/m3

# Calculation time index: 0, 300, 600, 900, ... etc.
# Assuming the initial time starts at 0 and is recorded every 30 seconds, so every 10th index is 300 seconds
indices = slice(start_index, None, int(output_interval2/output_interval1))

directory = '../'
output_directory = './'

# reading time and z
with xr.open_dataset(directory + 'history.pe000000.nc') as ds:
    z = ds['z']
    selected_data = ds.isel(time=indices)
    time = selected_data['time']

def process_file(file_path):
    with xr.open_dataset(file_path) as ds:
        selected_data = ds.isel(time=indices)
        # calculating LWC
        DMOM3 = selected_data['DMOM3']
        LWC = (4/3) * np.pi * DMOM3 * water_density # g/m3
        LWP = LWC.sum(dim='z') * DZ  # g/m2
        lwp_mean = LWP.mean(dim=['x', 'y'])

        # Criteria for clouds: LWC > 0.01 g/m3
        cloud_criteria = 0.01  # g/m3
        cloud_points = LWC > cloud_criteria
        z_expanded, _ = xr.broadcast(z, cloud_points)
        if cloud_points.any():
            z_cloud = z_expanded.where(cloud_points)
            cloud_top = z_cloud.max(dim='z', skipna=True)  # heights of highest cloudy cells
            cloud_base = z_cloud.min(dim='z', skipna=True)  # heights of lowest cloudy cells
            valid_columns = cloud_points.sum(dim='z')
            number_of_valid_columns = (valid_columns > 0).sum(dim=["y", "x"])

            # Horizontal mean cloud top and base heights
            cloud_top_sum = cloud_top.sum(dim=['x', 'y'], skipna=True) # m
            cloud_base_sum = cloud_base.sum(dim=['x', 'y'], skipna=True) # m
        else:
            number_of_valid_columns = 0
            cloud_top_sum = np.nan
            cloud_base_sum = np.nan

        temp_mean = selected_data['T'].mean(dim=['x', 'y']) # K
        pres_mean = selected_data['PRES'].mean(dim=['x', 'y']) # Pa
        qc_mean = selected_data['qnd'].mean(dim=['x', 'y'])*1e3 # g/kg
        qv_mean = selected_data['QV'].mean(dim=['x', 'y'])*1e3 # g/kg
        total_water = (selected_data['QV'] + selected_data['qnd'] + selected_data['QR_sd']) * 1e3
        qt_sum = total_water.sum(dim=['x', 'y']) # g/kg
        qt2_sum  = (total_water**2).sum(dim=['x', 'y'])
        rh_mean = selected_data['RH'].mean(dim=['x', 'y']) # %
        na_temp = selected_data['Na']*1e-6/selected_data['DENS'] # mg-1
        na_mean = na_temp.mean(dim=['x', 'y'])
        nact_mean = selected_data['Nact'].mean(dim=['x', 'y']) # mg-1 s-1
        ndeact_mean = selected_data['Ndeact'].mean(dim=['x', 'y']) # mg-1 s-1
        nc_temp = selected_data['NC']*1e-6/selected_data['DENS'] # mg-1
        nc_mean = nc_temp.mean(dim=['x', 'y'])
        u_temp = selected_data['U']
        u_sum = u_temp.sum(dim=['x', 'y'])
        u2_sum  = (u_temp**2).sum(dim=['x', 'y'])
        v_temp = selected_data['V']
        v_sum = v_temp.sum(dim=['x', 'y'])
        v2_sum  = (v_temp**2).sum(dim=['x', 'y'])
        w_temp = selected_data['W']
        w_sum = w_temp.sum(dim=['x', 'y'])
        w2_sum  = (w_temp**2).sum(dim=['x', 'y'])
        thetal_temp = selected_data['LWPT']
        thetal_sum = thetal_temp.sum(dim=['x', 'y'])
        thetal2_sum  = (thetal_temp**2).sum(dim=['x', 'y'])
        tke_res_mean = selected_data['TKE_RS'].mean(dim=['x', 'y'])
        tke_sgs_mean = selected_data['TKE_SMG'].mean(dim=['x', 'y'])

        return lwp_mean, cloud_base_sum, cloud_top_sum, number_of_valid_columns, temp_mean, pres_mean, qt_sum, qc_mean, qv_mean, rh_mean, na_mean, nact_mean, ndeact_mean, nc_mean, u_sum, u2_sum, v_sum, v2_sum, w_sum, w2_sum, thetal_sum, thetal2_sum, qt2_sum, tke_res_mean, tke_sgs_mean


def main(directory, total_processes):
    files = [f'{directory}history.pe{str(i).zfill(6)}.nc' for i in range(total_processes)]

    # Initialise variables
    lwp_time_series = 0
    cloud_base_time_series = 0
    cloud_top_time_series = 0
    number_of_valid_columns_sum = 0
    temp_profile = 0
    pres_profile = 0
    qt_profile = 0
    qc_profile = 0
    qv_profile = 0
    rh_profile = 0
    na_profile = 0
    nact_profile = 0
    ndeact_profile = 0
    nc_profile = 0
    u_profile = 0
    u2_mean = 0
    v_profile = 0
    v2_mean = 0
    w_profile = 0
    w2_mean = 0
    thetal_profile = 0
    thetal2_mean = 0
    qt2_mean = 0
    tke_res_profile = 0
    tke_sgs_profile = 0
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Submit tasks to the process pool
        future_to_file = {executor.submit(process_file, file): file for file in files}
        for future in concurrent.futures.as_completed(future_to_file):
            file = future_to_file[future]
            try:
                lwp_mean, cloud_base_sum, cloud_top_sum, number_of_valid_columns, temp_mean, pres_mean, qt_sum, qc_mean, qv_mean, rh_mean, na_mean, nact_mean, ndeact_mean, nc_mean, u_sum, u2_sum, v_sum, v2_sum, w_sum, w2_sum, thetal_sum, thetal2_sum, qt2_sum, tke_res_mean, tke_sgs_mean = future.result()
                lwp_time_series += lwp_mean
                cloud_base_time_series += cloud_base_sum
                cloud_top_time_series += cloud_top_sum
                number_of_valid_columns_sum += number_of_valid_columns
                temp_profile += temp_mean
                pres_profile += pres_mean
                qt_profile += qt_sum
                qc_profile += qc_mean
                qv_profile += qv_mean
                rh_profile += rh_mean
                na_profile += na_mean
                nact_profile += nact_mean
                ndeact_profile += ndeact_mean
                nc_profile += nc_mean
                u_profile += u_sum
                u2_mean += u2_sum
                v_profile += v_sum
                v2_mean += v2_sum
                w_profile += w_sum
                w2_mean += w2_sum
                thetal_profile += thetal_sum
                thetal2_mean += thetal2_sum
                qt2_mean += qt2_sum
                tke_res_profile += tke_res_mean
                tke_sgs_profile += tke_sgs_mean
            except Exception as exc:
                print('%r generated an exception: %s' % (file, exc))
        lwp_time_series /= total_processes
        cloud_base_time_series /= number_of_valid_columns_sum
        cloud_top_time_series /= number_of_valid_columns_sum
        temp_profile /= total_processes
        pres_profile /= total_processes
        qt_profile /= total_processes*IMAX*JMAX
        qc_profile /= total_processes
        qv_profile /= total_processes
        rh_profile /= total_processes
        na_profile /= total_processes
        nact_profile /= total_processes
        ndeact_profile /= total_processes
        nc_profile /= total_processes
        u_profile /= total_processes*IMAX*JMAX
        u2_mean /= total_processes*IMAX*JMAX
        v_profile /= total_processes*IMAX*JMAX
        v2_mean /= total_processes*IMAX*JMAX
        w_profile /= total_processes*IMAX*JMAX
        w2_mean /= total_processes*IMAX*JMAX
        thetal_profile /= total_processes*IMAX*JMAX
        thetal2_mean /= total_processes*IMAX*JMAX
        qt2_mean /= total_processes*IMAX*JMAX
        tke_res_profile /= total_processes
        tke_sgs_profile /= total_processes

        u2_profile = u2_mean - u_profile**2
        v2_profile = v2_mean - v_profile**2
        w2_profile = w2_mean - w_profile**2
        thetal2_profile = thetal2_mean - thetal_profile**2
        qt2_profile = qt2_mean - qt_profile**2

    lwp_time_series_da = xr.DataArray(lwp_time_series,
                               dims=['time'],
                               coords={'time': time},
                               attrs={'units': 'g/m^2', 'description': 'Total mean liquid water path'})
    cloud_base_time_series_da = xr.DataArray(cloud_base_time_series,
                               dims=['time'],
                               coords={'time': time},
                               attrs={'units': 'm', 'description': 'Cloud Base Height'})
    cloud_top_time_series_da = xr.DataArray(cloud_top_time_series,
                               dims=['time'],
                               coords={'time': time},
                               attrs={'units': 'm', 'description': 'Cloud Top Height'})
    temp_profile_da = xr.DataArray(temp_profile,
                               dims=['time', 'z'],
                               coords={'time': time, 'z': z},
                               attrs={'units': 'K', 'description': 'Absolute Temperature profile'})
    pres_profile_da = xr.DataArray(pres_profile,
                               dims=['time', 'z'],
                               coords={'time': time, 'z': z},
                               attrs={'units': 'Pa', 'description': 'Pressure profile'})
    qt_profile_da = xr.DataArray(qt_profile,
                               dims=['time', 'z'],
                               coords={'time': time, 'z': z},
                               attrs={'units': 'g/kg', 'description': 'Total water mixing ratio profile'})
    qc_profile_da = xr.DataArray(qc_profile,
                               dims=['time', 'z'],
                               coords={'time': time, 'z': z},
                               attrs={'units': 'g/kg', 'description': 'Liquid water mixing ratio profile'})
    qv_profile_da = xr.DataArray(qv_profile,
                               dims=['time', 'z'],
                               coords={'time': time, 'z': z},
                               attrs={'units': 'g/kg', 'description': 'Water vapor mixing ratio profile'})
    rh_profile_da = xr.DataArray(rh_profile,
                               dims=['time', 'z'],
                               coords={'time': time, 'z': z},
                               attrs={'units': '%', 'description': 'Relative Humidity profile'})
    na_profile_da = xr.DataArray(na_profile,
                               dims=['time', 'z'],
                               coords={'time': time, 'z': z},
                               attrs={'units': 'mg−1', 'description': 'Aerosol number concentration profile'})
    nact_profile_da = xr.DataArray(nact_profile,
                               dims=['time', 'z'],
                               coords={'time': time, 'z': z},
                               attrs={'units': 'mg−1 s−1', 'description': 'Aerosol activation rate profile'})
    ndeact_profile_da = xr.DataArray(ndeact_profile,
                               dims=['time', 'z'],
                               coords={'time': time, 'z': z},
                               attrs={'units': 'mg−1 s−1', 'description': 'Aerosol deactivation rate profile'})
    nc_profile_da = xr.DataArray(nc_profile,
                               dims=['time', 'z'],
                               coords={'time': time, 'z': z},
                               attrs={'units': 'mg−1', 'description': 'Cloud droplet number concentration profile'})
    u_profile_da = xr.DataArray(u_profile,
                               dims=['time', 'z'],
                               coords={'time': time, 'z': z},
                               attrs={'units': 'm/s', 'description': 'Zonal wind profile'})
    v_profile_da = xr.DataArray(v_profile,
                               dims=['time', 'z'],
                               coords={'time': time, 'z': z},
                               attrs={'units': 'm/s', 'description': 'Meridional wind profile'})
    w_profile_da = xr.DataArray(w_profile,
                               dims=['time', 'z'],
                               coords={'time': time, 'z': z},
                               attrs={'units': 'm/s', 'description': 'Vertical wind profile'})
    u2_profile_da = xr.DataArray(u2_profile,
                               dims=['time', 'z'],
                               coords={'time': time, 'z': z},
                               attrs={'units': 'm2/s2', 'description': 'Resolved variance of zonal wind profile'})
    v2_profile_da = xr.DataArray(v2_profile,
                               dims=['time', 'z'],
                               coords={'time': time, 'z': z},
                               attrs={'units': 'm2/s2', 'description': 'Resolved variance of meridional wind profile'})
    w2_profile_da = xr.DataArray(w2_profile,
                               dims=['time', 'z'],
                               coords={'time': time, 'z': z},
                               attrs={'units': 'm2/s2', 'description': 'Resolved variance of vertical wind profile'})
    thetal_profile_da = xr.DataArray(thetal_profile,
                               dims=['time', 'z'],
                               coords={'time': time, 'z': z},
                               attrs={'units': 'K', 'description': 'Liquid water potential temperature profile'})
    thetal2_profile_da = xr.DataArray(thetal2_profile,
                               dims=['time', 'z'],
                               coords={'time': time, 'z': z},
                               attrs={'units': 'K2', 'description': 'Resolved variance of liquid water potential temperature profile'})
    qt2_profile_da = xr.DataArray(qt2_profile,
                               dims=['time', 'z'],
                               coords={'time': time, 'z': z},
                               attrs={'units': 'g2/kg2', 'description': 'Resolved variance of qt'})
    tke_res_profile_da = xr.DataArray(tke_res_profile,
                               dims=['time', 'z'],
                               coords={'time': time, 'z': z},
                               attrs={'units': 'm2/s2', 'description': 'Resolved TKE'})
    tke_sgs_profile_da = xr.DataArray(tke_sgs_profile,
                               dims=['time', 'z'],
                               coords={'time': time, 'z': z},
                               attrs={'units': 'm2/s2', 'description': 'Subgrid TKE'})
    # Merged into data sets
    output_ds = xr.Dataset({
        'time': time,
        'z': z,
        'LWP': lwp_time_series_da,
        'CB': cloud_base_time_series_da,
        'CT': cloud_top_time_series_da,
        'Temp': temp_profile_da,
        'Pres': pres_profile_da,
        'qt': qt_profile_da,
        'qc': qc_profile_da,
        'qv': qv_profile_da,
        'RH': rh_profile_da,
        'Na': na_profile_da,
        'Nact': nact_profile_da,
        'Ndeact': ndeact_profile_da,
        'Nc': nc_profile_da,
        'U': u_profile_da,
        'V': v_profile_da,
        'W': w_profile_da,
        'U2': u2_profile_da,
        'V2': v2_profile_da,
        'W2': w2_profile_da,
        'thetal': thetal_profile_da,
        'thetal2': thetal2_profile_da,
        'qt2': qt2_profile_da,
        'TKE_res': tke_res_profile_da,
        'TKE_sgs': tke_sgs_profile_da
    })

    # Save as NetCDF file
    output_ds.to_netcdf(output_directory + 'icmw24_aceena_stat_SCALE-SDM_SDM_NUIST_Yin_35x35x2.5.nc')

if __name__ == "__main__":
    main(directory, total_processes)
