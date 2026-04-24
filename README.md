# SCALE-SDM ACE-ENA stratocumulus cases

This repository contains three SCALE-SDM ACE-ENA stratocumulus simulation cases and their post-processing/plotting scripts. The workflow below is written for SQUID supercomputer at the Osaka University, Japan, but the plotting scripts can also be run locally against completed NetCDF outputs.

## Model background

The Super-Droplet Method (SDM) is a particle-based, probabilistic cloud microphysics method introduced by Shima et al. (2009). In SDM, each computational super-droplet represents many real aerosol, cloud, or precipitation particles with similar attributes. This Lagrangian formulation allows condensation/evaporation, activation/deactivation, collision-coalescence, and sedimentation-related microphysical variability to be represented without prescribing a fixed bulk droplet-size distribution.

SCALE-SDM couples SDM with the SCALE regional model framework. SCALE provides a scalable nonhydrostatic dynamical core and infrastructure for high-resolution atmospheric simulations, including large-eddy simulations of boundary-layer clouds. The SCALE framework and cloud-microphysics applications are described by Nishizawa et al. (2015) and Sato et al. (2015).

These ACE-ENA cases are configured for non-precipitating stratocumulus experiments. The `ACE-ENA_sdm` simulation result was used in the model intercomparison study of Yang et al. (2026) on mixing characteristics in non-precipitating stratocumulus clouds.

## Case overview

| Case | Directory | Dynamics | Horizontal grid | Vertical grid | MPI ranks | Main job script |
|---|---|---:|---:|---:|---:|---|
| ACE-ENA SDM | `scale-rm/test/case/shallowcloud/ACE-ENA_sdm` | `HEVE` | 35 m | 5 m | 256 | `ace-ena.sh` |
| ACE-ENA SDM high-res | `scale-rm/test/case/shallowcloud/ACE-ENA_sdm_high_res` | `HEVE` | 35 m | 2.5 m | 1024 | `ace-ena.sh` |
| ACE-ENA SDM high-res HEVI | `scale-rm/test/case/shallowcloud/ACE-ENA_sdm_high_res_HEVI` | `HEVI` | 35 m | 2.5 m | 1024 | `ace-ena.sh` |

The case-specific namelists are:

- `init.conf`: initialization namelist used by `scale-rm_init`
- `run.conf`: model integration namelist used by `scale-rm`
- `Makefile`: case-level build configuration
- `ace-ena.sh`: SQUID batch script that runs initialization and the main model integration

## Required software

SCALE-SDM requires Fortran/C compilers, MPI, HDF5, NetCDF-C, and NetCDF-Fortran. The SQUID scripts in this repository use:

```bash
source /etc/profile.d/modules.sh
module load BaseCPU/2024 inteloneAPI/2023.2 hdf5/1.12.3.mpi netcdf-c/4.9.2 netcdf-fortran/4.6.1
export SCALE_SYS=Linux64-intel-impi
```

The Python post-processing and plotting scripts require:

- `python`
- `numpy`
- `xarray`
- `netCDF4`
- `matplotlib`
- `ffmpeg`

On SQUID, the batch scripts under `results/` currently activate `sdm_env`. On a local workstation, the plotting examples below use the local conda environment `work`.

## Build on SQUID

Always configure the environment before compiling. Enter the target case directory first, then clean and build with SDM enabled.

Example for the HEVI case:

```bash
cd scale-rm/test/case/shallowcloud/ACE-ENA_sdm_high_res_HEVI

source /etc/profile.d/modules.sh
module load BaseCPU/2024 inteloneAPI/2023.2 hdf5/1.12.3.mpi netcdf-c/4.9.2 netcdf-fortran/4.6.1
export SCALE_SYS=Linux64-intel-impi

make allclean
make allclean SCALE_ENABLE_SDM=T SCALE_DISABLE_LOCALBIN=T SCALE_DYCOMS2_RF02_SDM=T
make -j SCALE_ENABLE_SDM=T SCALE_DISABLE_LOCALBIN=T SCALE_DYCOMS2_RF02_SDM=T
ln -fsv `grep ^TOPDIR Makefile | sed s/\)//g | awk '{print $NF}'`/bin/scale-rm* .
```

Use the same build sequence for the other two cases by changing only the case directory:

```bash
cd scale-rm/test/case/shallowcloud/ACE-ENA_sdm
```

or:

```bash
cd scale-rm/test/case/shallowcloud/ACE-ENA_sdm_high_res
```

## Submit simulations on SQUID

After the executable links `scale-rm_init` and `scale-rm` are present in the case directory, submit the case job:

```bash
cd scale-rm/test/case/shallowcloud/ACE-ENA_sdm_high_res_HEVI
qsub ace-ena.sh
```

`ace-ena.sh` runs two MPI commands in sequence:

```bash
mpirun ${NQSV_MPIOPTS} -n 1024 ./scale-rm_init init.conf || exit
mpirun ${NQSV_MPIOPTS} -n 1024 ./scale-rm run.conf || exit
```

For the base `ACE-ENA_sdm` case, the script uses 256 MPI ranks. For the two high-resolution cases, it uses 1024 MPI ranks.

Record the simulation job ID printed by `qsub`; it is needed for dependent post-processing jobs.

## Post-processing workflow

Each case has a `results/` directory with three batch scripts and three Python post-processing scripts:

- `profile.sh` runs `ace-ena_time_series_and_vertical_profiles.py`
- `3d_fields.sh` runs `ace-ena_3D_fields.py`
- `dsd_output.sh` runs `ace-ena_dsd.py`

The recommended SQUID workflow is to submit post-processing jobs with scheduler dependencies. `dsd_output.sh` must be submitted after `3d_fields.sh` has finished because it appends the `DSD` variable into the 3-D NetCDF file created by `ace-ena_3D_fields.py`.

Manual dependency example:

```bash
cd scale-rm/test/case/shallowcloud/ACE-ENA_sdm_high_res_HEVI/results

# Suppose the simulation job ID from qsub ace-ena.sh is 587259.
qsub --after 587259 profile.sh
qsub --after 587259 3d_fields.sh

# Suppose the 3d_fields.sh job ID is 587310.
qsub --after 587310 dsd_output.sh
```

Apply the same dependency pattern to the other two cases:

```bash
cd scale-rm/test/case/shallowcloud/ACE-ENA_sdm/results
qsub --after <SIM_JOB_ID> profile.sh
qsub --after <SIM_JOB_ID> 3d_fields.sh
qsub --after <THREED_JOB_ID> dsd_output.sh
```

```bash
cd scale-rm/test/case/shallowcloud/ACE-ENA_sdm_high_res/results
qsub --after <SIM_JOB_ID> profile.sh
qsub --after <SIM_JOB_ID> 3d_fields.sh
qsub --after <THREED_JOB_ID> dsd_output.sh
```

The expected post-processed NetCDF outputs are:

| Case | Statistics output | 3-D output |
|---|---|---|
| `ACE-ENA_sdm` | `icmw24_aceena_stat_SCALE-SDM_SDM_NUIST_Yin.nc` | `icmw24_aceena_3d_SCALE-SDM_SDM_NUIST_27000_Yin.nc` |
| `ACE-ENA_sdm_high_res` | `icmw24_aceena_stat_SCALE-SDM_SDM_NUIST_Yin_35x35x2.5.nc` | `icmw24_aceena_3d_SCALE-SDM_SDM_NUIST_27000_Yin_35x35x2.5.nc` |
| `ACE-ENA_sdm_high_res_HEVI` | `icmw24_aceena_stat_SCALE-SDM_SDM_NUIST_Yin_35x35x2.5_hevi.nc` | `icmw24_aceena_3d_SCALE-SDM_SDM_NUIST_27000_Yin_35x35x2.5_hevi.nc` |

If `ace-ena_dsd.py` is rerun and the target 3-D file already contains `DSD`, the script exits to avoid appending duplicate variables. Recreate the 3-D file or use a fresh output before rerunning DSD post-processing.

## Plotting

The plotting scripts are:

| Case | Statistics plots | 3-D/DSD animations |
|---|---|---|
| `ACE-ENA_sdm` | `ace_ena_stat_plots.py` | `ace-ena_3d_plot.py` |
| `ACE-ENA_sdm_high_res` | `ace_ena_stat_plots_high_res.py` | `ace-ena_3d_plot_high_res.py` |
| `ACE-ENA_sdm_high_res_HEVI` | `ace_ena_stat_plots_high_res_hevi.py` | `ace-ena_3d_plot_high_res_hevi.py` |

The six scripts above are case-specific entry points. Their common plotting implementation is maintained in `scale-rm/test/case/shallowcloud/ace_ena_plotting.py`; keep that shared file with the case directories when copying the workflow to another machine.

By default, the plotting scripts:

- use every time level in the input NetCDF file
- read NetCDF files from the plotting script directory
- write outputs to `DATA_DIR/plots`, where `DATA_DIR` is the script directory unless `--data-dir` is set
- save one static figure format per run, using vector PDF by default
- create cloud-layer time-series diagnostics, time-height sections, and mean-profile envelope figures from the statistics NetCDF file
- save animations as high-quality H.264 MP4
- keep 3-D field animations in the original two-row layout, with the horizontal slice above the vertical slice
- preserve the true physical aspect ratio of horizontal and vertical slices
- create a vertical-maximum `qc` projection animation from the 3-D NetCDF file
- keep legends outside plotting panels when legends are needed

Use `--max-frames` only for quick local smoke tests. Omit it for production figures and animations.

### Local HEVI plotting example

Put or link the HEVI post-processed NetCDF files next to the plotting scripts, then run:

```bash
cd scale-rm/test/case/shallowcloud/ACE-ENA_sdm_high_res_HEVI/results

conda run -n work python ace_ena_stat_plots_high_res_hevi.py
conda run -n work python ace-ena_3d_plot_high_res_hevi.py
```

The generated products are written to `plots/` under the same directory. If the NetCDF files are stored elsewhere, pass a private data directory at run time:

```bash
conda run -n work python ace_ena_stat_plots_high_res_hevi.py --data-dir <DATA_DIR>
conda run -n work python ace-ena_3d_plot_high_res_hevi.py --data-dir <DATA_DIR>
```

Useful plotting options:

```bash
# Quick smoke test with only the first 3 NetCDF time levels.
--max-frames 3

# Use every second NetCDF time level.
--frame-step 2

# Plot only selected 3-D variables.
--variables U,V,W,Temp

# Change or expand the vertical-maximum projection variables.
--projection-variables qc,Nc

# Increase or reduce animation raster resolution.
--video-dpi 200

# Change the single static image format.
--image-format png
```

## Practical end-to-end example for HEVI

```bash
cd scale-rm/test/case/shallowcloud/ACE-ENA_sdm_high_res_HEVI

source /etc/profile.d/modules.sh
module load BaseCPU/2024 inteloneAPI/2023.2 hdf5/1.12.3.mpi netcdf-c/4.9.2 netcdf-fortran/4.6.1
export SCALE_SYS=Linux64-intel-impi

make allclean
make allclean SCALE_ENABLE_SDM=T SCALE_DISABLE_LOCALBIN=T SCALE_DYCOMS2_RF02_SDM=T
make -j SCALE_ENABLE_SDM=T SCALE_DISABLE_LOCALBIN=T SCALE_DYCOMS2_RF02_SDM=T
ln -fsv `grep ^TOPDIR Makefile | sed s/\)//g | awk '{print $NF}'`/bin/scale-rm* .

qsub ace-ena.sh
```

After the simulation job ID is known:

```bash
cd results
qsub --after <SIM_JOB_ID> profile.sh
qsub --after <SIM_JOB_ID> 3d_fields.sh
qsub --after <THREED_JOB_ID> dsd_output.sh
```

After the post-processing NetCDF files are complete:

```bash
python ace_ena_stat_plots_high_res_hevi.py
python ace-ena_3d_plot_high_res_hevi.py
```

## References

Nishizawa, S., Yashiro, H., Sato, Y., Miyamoto, Y., and Tomita, H.: Influence of grid aspect ratio on planetary boundary layer turbulence in large-eddy simulations, Geoscientific Model Development, 8, 3393-3419, https://doi.org/10.5194/gmd-8-3393-2015, 2015.

Sato, Y., Nishizawa, S., Yashiro, H., Miyamoto, Y., Kajikawa, Y., and Tomita, H.: Impacts of cloud microphysics on trade wind cumulus: which cloud microphysics processes contribute to the diversity in a large eddy simulation?, Progress in Earth and Planetary Science, 2, 1-16, https://doi.org/10.1186/s40645-015-0053-6, 2015.

Shima, S.-i., Kusano, K., Kawano, A., Sugiyama, T., and Kawahara, S.: The super-droplet method for the numerical simulation of clouds and precipitation: a particle-based and probabilistic microphysics model coupled with a non-hydrostatic model, Quarterly Journal of the Royal Meteorological Society, 135, 1307-1320, https://doi.org/10.1002/qj.441, 2009.

Yang, F., Choi, K. O., Chandrakar, K. K., Hoffmann, F., Hou, P., Krueger, S., et al.: A model intercomparison study to investigate mixing characteristics in non-precipitating stratocumulus clouds, Journal of Advances in Modeling Earth Systems, 18, e2025MS005620, https://doi.org/10.1029/2025MS005620, 2026.
