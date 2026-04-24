import xarray as xr
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import concurrent
import re
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

x = np.arange(DX/2, IMAX*PRC_NUM_X*DX-DX/2+1e-6, DX)
y = np.arange(DY/2, JMAX*PRC_NUM_Y*DY-DY/2+1e-6, DY)
z = np.arange(DZ/2, KMAX*DZ-DZ/2+1e-6, DZ)

# Remove the old conflicting time configuration (lines 58-62)
# start_tm = 21600 # in sec
# output_interval1 = 30 # in sec
# output_interval2 = 1800 # in sec
# start_index = start_tm // output_interval2
# indices = slice(start_index, None, int(output_interval2/output_interval1))

directory = '../'
output_directory = './'

# Clean time configuration with debug output
total_time = 28800  # 8 hours total time in seconds
last_30_min = 1800  # last 30 minutes in seconds
output_interval = 30  # output every 30 seconds
start_time = total_time - last_30_min  # start time: 27000 seconds

print(f"Time configuration:")
print(f"  Total time: {total_time} seconds ({total_time/3600:.1f} hours)")
print(f"  Last 30 minutes: {last_30_min} seconds")
print(f"  Start time: {start_time} seconds ({start_time/3600:.1f} hours)")
print(f"  Output interval: {output_interval} seconds")

# Read and analyze original data time information
with xr.open_dataset(directory + 'history.pe000000.nc') as ds:
    time_coord = ds['time']
    total_time_steps = len(time_coord)
    
    # Use xarray's time handling
    if len(time_coord) > 1:
        dt_seconds = float((time_coord[1] - time_coord[0]).values / np.timedelta64(1, 's'))
    else:
        dt_seconds = 30.0
    
    # Convert to seconds for display
    first_time = float(time_coord[0].values / np.timedelta64(1, 's'))
    last_time = float(time_coord[-1].values / np.timedelta64(1, 's'))
    duration = last_time - first_time
    
    # print(f"\nOriginal data analysis:")
    # print(f"  Total time steps: {total_time_steps}")
    # print(f"  Time step interval (dt): {dt_seconds} seconds")
    # print(f"  First time value: {first_time} seconds ({first_time/3600:.1f} hours)")
    # print(f"  Last time value: {last_time} seconds ({last_time/3600:.1f} hours)")
    # print(f"  Data duration: {duration} seconds ({duration/3600:.1f} hours)")

# Update the dt variable for later use
dt = dt_seconds

# Calculate correct indices
start_index = int(start_time / dt)  # 27000/30 = 900
step_interval = 1  # take every step since original data is already 30-second interval

# print(f"\nIndex calculations:")
# print(f"  Start index: {start_index}")
# print(f"  Step interval: {step_interval}")
# print(f"  Expected end index: {total_time_steps - 1}")
# print(f"  Expected number of output time steps: {total_time_steps - start_index}")

# Verify indices are within bounds
if start_index >= total_time_steps:
    print(f"ERROR: Start index {start_index} exceeds total time steps {total_time_steps}")
    exit(1)

indices = slice(start_index, None, step_interval)  # from index 900 to end

# Read time coordinate with verification
# Time coordinate extraction with proper datetime64 handling
with xr.open_dataset(directory + 'history.pe000000.nc') as ds:
    time_raw = ds['time'].isel(time=indices)
    
    # Convert time to seconds since start for better readability
    # Keep the original datetime64 format but ensure proper units
    time = time_raw.copy()
    
    # For debug output, convert to seconds
    first_time_seconds = float(time_raw.values[0] / np.timedelta64(1, 's'))
    last_time_seconds = float(time_raw.values[-1] / np.timedelta64(1, 's'))
    time_range_seconds = last_time_seconds - first_time_seconds
    
    print(f"\nTime coordinate extraction:")
    print(f"  Selected time steps: {len(time)}")
    print(f"  First selected time: {first_time_seconds} seconds ({first_time_seconds/3600:.1f} hours)")
    print(f"  Last selected time: {last_time_seconds} seconds ({last_time_seconds/3600:.1f} hours)")
    print(f"  Time range: {time_range_seconds} seconds ({time_range_seconds/60:.1f} minutes)")
    
    # Alternative: Create a simple time coordinate in seconds
    # This ensures NetCDF output has readable time values
    time_data = np.arange(first_time_seconds, last_time_seconds + 30, 30)
    time_attrs = {'units': 'seconds since simulation start', 'long_name': 'time'}
    x_attrs = {'units': 'm', 'long_name': 'x coordinate'}
    y_attrs = {'units': 'm', 'long_name': 'y coordinate'}
    z_attrs = {'units': 'm', 'long_name': 'z coordinate'}
    
    time_seconds = xr.DataArray(
    time_data,
    dims=['time'],
    attrs=time_attrs
    )
    
    time = time_data
    
    # Verify time values are not empty or NaN
    if len(time) == 0:
        print("ERROR: No time values selected!")
        exit(1)
    if np.any(np.isnan(time)):
        print("ERROR: Time values contain NaN!")
        exit(1)

def process_file(process_rank):
    file = f'{directory}history.pe{str(process_rank).zfill(6)}.nc'
    yp = int(process_rank / PRC_NUM_X)
    xp = process_rank % PRC_NUM_X
    
    print(f"Processing file: {file} (rank {process_rank}, yp={yp}, xp={xp})")

    with xr.open_dataset(file) as ds:
        # Check if indices are valid for this file
        file_time_steps = len(ds['time'])
        print(f"  File time steps: {file_time_steps}")
        
        if start_index >= file_time_steps:
            print(f"  ERROR: Start index {start_index} exceeds file time steps {file_time_steps}")
            return None
        
        # Extract data with time slicing and convert to numpy arrays
        U = ds['U'].isel(time=indices).values  # Convert to numpy array
        V = ds['V'].isel(time=indices).values
        W = ds['W'].isel(time=indices).values
        Temp = ds['T'].isel(time=indices).values
        qv = (ds['QV'].isel(time=indices)*1e3).values  # g/kg
        Pres = ds['PRES'].isel(time=indices).values
        DENS = ds['DENS'].isel(time=indices).values
        Nc = (ds['NC'].isel(time=indices)/DENS).values # kg-1
        qc = (ds['qnd'].isel(time=indices)*1e3).values  # g/kg
        Na = (ds['Na'].isel(time=indices)/DENS).values # kg-1
        qa = (ds['qna'].isel(time=indices)*1e3).values  # g/kg
        Nact = ds['Nact'].isel(time=indices).values # mg-1 s-1
        Ndeact = ds['Ndeact'].isel(time=indices).values # mg-1 s-1
        
        # Debug: check data shapes and values (fixed - U is now numpy array)
        # print(f"  Data shapes: U={U.shape}, time_steps={U.shape[0]}")
        # print(f"  Sample values: U_mean={np.nanmean(U):.6f}, V_mean={np.nanmean(V):.6f}, W_mean={np.nanmean(W):.6f}, Temp_mean={np.nanmean(Temp):.6f}")
        # print(f"  Sample values: qv_mean={np.nanmean(qv):.6f}, Pres_mean={np.nanmean(Pres):.6f}, DENS_mean={np.nanmean(DENS):.6f}")
        # print(f"  Sample values: Nc_mean={np.nanmean(Nc):.6f}, qc_mean={np.nanmean(qc):.6f}, Na_mean={np.nanmean(Na):.6f}, qa_mean={np.nanmean(qa):.6f}")
        # print(f"  Sample values: Nact_mean={np.nanmean(Nact):.6f}, Ndeact_mean={np.nanmean(Ndeact):.6f}")
        
        return U, V, W, Temp, qv, Pres, Nc, qc, Na, qa, Nact, Ndeact, yp, xp

def main():
    print(f"\nInitializing 3D arrays with time dimension: {len(time)}")
    print(f"Spatial dimensions: z={len(z)}, y={len(y)}, x={len(x)}")
    
    # Calculate the correct shape for all arrays
    shape = (len(time), len(z), len(y), len(x))
    print(f"Creating arrays with shape: {shape}")
    
    # Initialize arrays with correct shape using numpy arrays
    U_3d = np.full(shape, np.nan)
    V_3d = np.full(shape, np.nan)
    W_3d = np.full(shape, np.nan)
    Temp_3d = np.full(shape, np.nan)
    qv_3d = np.full(shape, np.nan)
    Pres_3d = np.full(shape, np.nan)
    Nc_3d = np.full(shape, np.nan)
    qc_3d = np.full(shape, np.nan)
    Na_3d = np.full(shape, np.nan)
    qa_3d = np.full(shape, np.nan)
    Nact_3d = np.full(shape, np.nan)
    Ndeact_3d = np.full(shape, np.nan)

    print(f"\nProcessing {total_processes} files with {num_cores} cores...")
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = [executor.submit(process_file, i) for i in range(total_processes)]
        processed_count = 0
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is None:
                print(f"WARNING: Skipping failed process")
                continue
                
            U, V, W, Temp, qv, Pres, Nc, qc, Na, qa, Nact, Ndeact, yp, xp = result
            
            # Calculate slices for data assignment
            y_slice = slice(yp*JMAX, (yp+1)*JMAX)
            x_slice = slice(xp*IMAX, (xp+1)*IMAX)
            
            # Debug: print assignment info and check data before assignment
            # print(f"  Assigning data for process {processed_count}: yp={yp}, xp={xp}")
            # print(f"    y_slice: {y_slice}, x_slice: {x_slice}")
            # print(f"    Input data shape: U={U.shape}")
            # print(f"    Target slice shape: {U_3d[:, :, y_slice, x_slice].shape}")
            # print(f"    Input U data range: {np.nanmin(U):.6f} to {np.nanmax(U):.6f}")
            # print(f"    Input U non-NaN count: {np.count_nonzero(~np.isnan(U))}")
            
            # Check target area before assignment
            target_before = U_3d[:, :, y_slice, x_slice].copy()
            nan_count_before = np.count_nonzero(np.isnan(target_before))
            print(f"    Target area NaN count before assignment: {nan_count_before}")
            
            # Assign data to the numpy arrays
            U_3d[:, :, y_slice, x_slice] = U
            V_3d[:, :, y_slice, x_slice] = V
            W_3d[:, :, y_slice, x_slice] = W
            Temp_3d[:, :, y_slice, x_slice] = Temp
            qv_3d[:, :, y_slice, x_slice] = qv
            Pres_3d[:, :, y_slice, x_slice] = Pres
            Nc_3d[:, :, y_slice, x_slice] = Nc
            qc_3d[:, :, y_slice, x_slice] = qc
            Na_3d[:, :, y_slice, x_slice] = Na
            qa_3d[:, :, y_slice, x_slice] = qa
            Nact_3d[:, :, y_slice, x_slice] = Nact
            Ndeact_3d[:, :, y_slice, x_slice] = Ndeact
            
            # Check target area after assignment
            target_after = U_3d[:, :, y_slice, x_slice]
            nan_count_after = np.count_nonzero(np.isnan(target_after))
            print(f"    Target area NaN count after assignment: {nan_count_after}")
            print(f"    Target U data range after assignment: {np.nanmin(target_after):.6f} to {np.nanmax(target_after):.6f}")
            
            processed_count += 1
            if processed_count % 10 == 0:
                print(f"  Processed {processed_count}/{total_processes} files")
                # Check overall progress
                total_nan_count = np.count_nonzero(np.isnan(U_3d))
                total_elements = U_3d.size
                filled_percentage = (total_elements - total_nan_count) / total_elements * 100
                print(f"    Overall progress: {filled_percentage:.1f}% filled ({total_elements - total_nan_count}/{total_elements})")
    
    # print(f"\nFinal data verification:")
    # print(f"  U_3d shape: {U_3d.shape}")
    # print(f"  U_3d mean: {np.nanmean(U_3d):.6f}")
    # print(f"  U_3d min/max: {np.nanmin(U_3d):.6f} / {np.nanmax(U_3d):.6f}")
    # print(f"  Non-NaN count: {np.count_nonzero(~np.isnan(U_3d))}")
    # print(f"  Time coordinate length: {len(time)}")
    # print(f"  Time coordinate values: {time[:3]}...{time[-3:]}")
    
    # Create the Dataset with all variables and coordinates defined at the top level
    output_ds = xr.Dataset(
        data_vars={
            'U': (('time', 'z', 'y', 'x'), U_3d, {'units': 'm/s', 'description': 'Zonal wind 3-D field'}),
            'V': (('time', 'z', 'y', 'x'), V_3d, {'units': 'm/s', 'description': 'Meridional wind 3-D field'}),
            'W': (('time', 'z', 'y', 'x'), W_3d, {'units': 'm/s', 'description': 'Vertical wind 3-D field'}),
            'Temp': (('time', 'z', 'y', 'x'), Temp_3d, {'units': 'K', 'description': 'Absolute temperature 3-D field'}),
            'qv': (('time', 'z', 'y', 'x'), qv_3d, {'units': 'g/kg', 'description': 'Water vapor mixing ratio 3-D field'}),
            'Pres': (('time', 'z', 'y', 'x'), Pres_3d, {'units': 'Pa', 'description': 'Pressure 3-D field'}),
            'Nc': (('time', 'z', 'y', 'x'), Nc_3d, {'units': 'num/kg', 'description': 'Cloud droplet number mixing ratio 3-D field'}),
            'qc': (('time', 'z', 'y', 'x'), qc_3d, {'units': 'g/kg', 'description': 'Cloud droplet mass mixing ratio 3-D field'}),
            'Na': (('time', 'z', 'y', 'x'), Na_3d, {'units': 'num/kg', 'description': 'Aerosol number mixing ratio 3-D field'}),
            'qa': (('time', 'z', 'y', 'x'), qa_3d, {'units': 'g/kg', 'description': 'Aerosol mass mixing ratio 3-D field'}),
            'Nact': (('time', 'z', 'y', 'x'), Nact_3d, {'units': 'num/mg·s', 'description': 'Activation rate 3-D field'}),
            'Ndeact': (('time', 'z', 'y', 'x'), Ndeact_3d, {'units': 'num/mg·s', 'description': 'Deactivation rate 3-D field'}),
        },
        coords={
            'time': ('time', time, time_attrs),
            'z': ('z', z, z_attrs),
            'y': ('y', y, y_attrs),
            'x': ('x', x, x_attrs),
        }
    )

    # Print the dataset structure for final verification
    # print("\nFinal Dataset structure:")
    # print(output_ds)

    # Add final verification of all data variables
    # print("\nVerifying data in all variables before saving:")
    # print("\nVerifying data in all variables before saving:")
    # for var_name in output_ds.data_vars:
    #     var_data = output_ds[var_name].values
    #     mean_val = np.nanmean(var_data)
    #     min_val = np.nanmin(var_data)
    #     max_val = np.nanmax(var_data)
    #     non_nan_count = np.count_nonzero(~np.isnan(var_data))
    #     print(f"  - {var_name}: Non-NaN={non_nan_count}, Mean={mean_val:.6f}, Min={min_val:.6f}, Max={max_val:.6f}")

    print(f"\nSaving to: {output_directory}icmw24_aceena_3d_SCALE-SDM_SDM_NUIST_27000_Yin_35x35x2.5_hevi.nc")
    
    encoding = {var: {'_FillValue': None, 'dtype': 'float64'} for var in output_ds.variables}
    
    output_filename = 'icmw24_aceena_3d_SCALE-SDM_SDM_NUIST_27000_Yin_35x35x2.5_hevi.nc'
    output_filepath = os.path.join(output_directory, output_filename)

    print(f"\nSaving to: {output_filepath}")
    
    try:
        output_ds.to_netcdf(
            output_filepath,
            engine='netcdf4',
            encoding=encoding
        )
        print("File saved successfully!")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")

if __name__ == "__main__":
    main()
