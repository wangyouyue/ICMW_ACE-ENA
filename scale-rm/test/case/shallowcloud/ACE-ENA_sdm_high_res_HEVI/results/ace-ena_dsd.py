import xarray as xr
import numpy as np
from time import gmtime, strftime
import re
from concurrent.futures import ProcessPoolExecutor
import concurrent
import os

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
KMAX = int(read_value_from_config('KMAX', config_path))
DX = read_value_from_config('DX', config_path)
DY = read_value_from_config('DY', config_path)
DZ = read_value_from_config('DZ', config_path)

z_centers = np.arange(DZ/2, KMAX*DZ, DZ)
radius_bins = np.logspace(np.log10(1e-6), np.log10(20e-6), num=20)

# Time window for the final 30 minutes of an 8-hour simulation.
start_tm = 27000  # 8 hours minus 30 minutes
output_interval1 = 30  # Output interval in seconds
end_tm = 28800  # 8-hour simulation end time
T_MAX = int((end_tm - start_tm)/output_interval1) + 1  # 61 output times

directory = '../'
output_directory = './'

def process_and_output_cloud_spectra(pnx, pny, current_time):
    mpi = pnx * PRC_NUM_X + pny
    time_str = strftime("%H%M%S", gmtime(current_time))
    input_file = directory  + 'SD_all_NetCDF_00000101-'+time_str+'.000.pe'+str(mpi).zfill(6)
    ds = xr.open_dataset(input_file)

    sd_z_temp = ds['sd_z'].values
    sd_r_temp = ds['sd_r'].values
    sd_n_temp = ds['sd_n'].values

    mask = sd_z_temp > 0
    sd_z = sd_z_temp[mask]
    sd_nk = (sd_z/DZ).astype(int)
    sd_r = sd_r_temp[mask]
    sd_n = sd_n_temp[mask]
    sd_num = len(sd_nk)

    sorted_indices = np.argsort(sd_nk)
    sd_nk = sd_nk[sorted_indices]
    sd_r = sd_r[sorted_indices]
    sd_n = sd_n[sorted_indices]

    spectra = np.zeros((len(z_centers), len(radius_bins) - 1))

    z_st_index = 0

    for zi in range(len(z_centers)):
        if z_st_index >= sd_num:
            break
        z_count = np.count_nonzero(sd_nk == zi)

        r_layer = sd_r[z_st_index:z_st_index+z_count]
        n_layer = sd_n[z_st_index:z_st_index+z_count]

        if r_layer.size > 0:
            hist, _ = np.histogram(r_layer, bins=radius_bins, weights=n_layer)
            log_bin_widths = np.log10(radius_bins[1:]) - np.log10(radius_bins[:-1])
            spectra[zi, :] = hist / (log_bin_widths*DX*DY*DZ*IMAX*JMAX)

        z_st_index = z_st_index + z_count

    # Convert simulation time to the output time index.
    time_index = int((current_time - start_tm)/output_interval1)
    return spectra, pnx, pny, time_index

def main():
    DSD = np.zeros((T_MAX, PRC_NUM_Y, PRC_NUM_X, len(z_centers), len(radius_bins) - 1))

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = [executor.submit(process_and_output_cloud_spectra, pnx, pny, current_time)
                   for current_time in range(start_tm, end_tm + output_interval1, output_interval1)
                   for pnx in range(PRC_NUM_X)
                   for pny in range(PRC_NUM_Y)]
        
        for future in concurrent.futures.as_completed(futures):
            spectra, pnx, pny, time_index = future.result()
            DSD[time_index, pny, pnx, :, :] = spectra

    # --- New implementation using append mode ('a') with corrected encoding ---

    # Define output file path
    output_filename = 'icmw24_aceena_3d_SCALE-SDM_SDM_NUIST_27000_Yin_35x35x2.5_hevi.nc'
    output_filepath = os.path.join(output_directory, output_filename)

    # 1. Check if the file and variables exist to avoid errors on re-runs.
    try:
        with xr.open_dataset(output_filepath, decode_times=False) as existing_ds:
            if 'DSD' in existing_ds.variables:
                print(f"Error: Variable 'DSD' already exists in {output_filepath}.")
                print("If you want to re-run, please delete the file manually or use a different output file.")
                return
            # Use existing time and z coordinates from the file
            time = existing_ds['time'].values
            z = existing_ds['z'].values
    except FileNotFoundError:
        print(f"Error: The file {output_filepath} does not exist. Please run ace-ena_3D_fields.py first.")
        return
    except Exception as e:
        print(f"An error occurred while reading the existing NetCDF file: {e}")
        return

    # 2. Define new coordinates specific to the DSD variable
    y_proc = np.arange(JMAX*DY/2, JMAX*DY/2 + JMAX*DY * PRC_NUM_Y, JMAX*DY)
    x_proc = np.arange(IMAX*DX/2, IMAX*DX/2 + IMAX*DX * PRC_NUM_X, IMAX*DX)
    radius_bin_centers = (radius_bins[:-1] + radius_bins[1:]) / 2

    # 3. Create a NEW dataset containing ONLY the variables to be appended.
    ds_to_append = xr.Dataset(
        {
            'DSD': (
                ["time", "y_proc", "x_proc", "z", "radius_bin_centers"],
                DSD,
                {'long_name': 'Droplet number concentration per log-bin of radius', 'units': 'number m^-3 per log10(m)'}
            ),
            'radius_bins': (
                ["radius_bin_edges"],
                radius_bins,
                {'long_name': 'Radius bin edges', 'units': 'm'}
            )
        },
        coords={
            "time": ("time", time),
            "y_proc": ('y_proc', y_proc, {'long_name': 'Y-coordinate of MPI process center', 'units': 'm'}),
            "x_proc": ('x_proc', x_proc, {'long_name': 'X-coordinate of MPI process center', 'units': 'm'}),
            "z": ("z", z),
            "radius_bin_centers": ('radius_bin_centers', radius_bin_centers, {'long_name': 'Center of radius bin', 'units': 'm'}),
            "radius_bin_edges": ('radius_bin_edges', np.arange(len(radius_bins)))
        }
    )

    # 4. CRITICAL: Define encoding for ALL NEW variables (data and coords) to ensure proper writing.
    encoding = {var_name: {'_FillValue': None, 'dtype': 'float64'} for var_name in ds_to_append.variables}

    # 5. Append the new dataset to the existing file using mode='a'
    print(f"\nAppending DSD data to file: {output_filepath}")
    try:
        ds_to_append.to_netcdf(
            output_filepath,
            mode='a',  # Use append mode
            engine='netcdf4',
            encoding=encoding
        )
        print("File updated successfully with DSD data!")
    except Exception as e:
        print(f"An error occurred while appending to the file: {e}")


if __name__ == "__main__":
    main()
