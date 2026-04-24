#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared plotting helpers for the ACE-ENA shallow-cloud cases."""

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(), "fontconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.gridspec import GridSpec


PROFILE_TOP_M = 1440.0
DEFAULT_VIDEO_DPI = 200
DEFAULT_IMAGE_DPI = 600
DEFAULT_IMAGE_FORMAT = "pdf"

COLORS = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#E69F00",
    "gray": "#4D4D4D",
    "black": "#222222",
}

LINE_COLORS = [COLORS["blue"], COLORS["vermillion"], COLORS["green"], COLORS["purple"], COLORS["yellow"]]
SIGNED_VARIABLES = {"U", "V", "W"}
FIELD_VARIABLES = ["U", "V", "W", "Temp", "qv", "Pres", "Nc", "qc", "Na", "qa", "Nact", "Ndeact"]
PROJECTION_VARIABLES = ["qc"]

PROFILE_LABELS = {
    "Temp": "Temp (K)",
    "Pres": "Pres (Pa)",
    "qt": r"q$_{\mathrm{t}}$ (g kg$^{-1}$)",
    "qc": r"q$_{\mathrm{c}}$ (g kg$^{-1}$)",
    "qv": r"q$_{\mathrm{v}}$ (g kg$^{-1}$)",
    "RH": "RH (%)",
    "Na": r"N$_{\mathrm{a}}$ (mg$^{-1}$)",
    "Nact": r"N$_{\mathrm{act}}$ (mg$^{-1}$ s$^{-1}$)",
    "Ndeact": r"N$_{\mathrm{deact}}$ (mg$^{-1}$ s$^{-1}$)",
    "Nc": r"N$_{\mathrm{c}}$ (mg$^{-1}$)",
    "U": r"U (m s$^{-1}$)",
    "V": r"V (m s$^{-1}$)",
    "W": r"W (m s$^{-1}$)",
    "U2": r"Var(U) (m$^2$ s$^{-2}$)",
    "V2": r"Var(V) (m$^2$ s$^{-2}$)",
    "W2": r"Var(W) (m$^2$ s$^{-2}$)",
    "thetal": r"$\theta_{\mathrm{l}}$ (K)",
    "thetal2": r"Var($\theta_{\mathrm{l}}$) (K$^2$)",
    "qt2": r"Var(q$_{\mathrm{t}}$) (g$^2$ kg$^{-2}$)",
    "TKE_res": r"TKE$_{\mathrm{res}}$ (m$^2$ s$^{-2}$)",
    "TKE_sgs": r"TKE$_{\mathrm{sgs}}$ (m$^2$ s$^{-2}$)",
}

FIELD_LABELS = {
    "U": r"U (m s$^{-1}$)",
    "V": r"V (m s$^{-1}$)",
    "W": r"W (m s$^{-1}$)",
    "Temp": "Temp (K)",
    "qv": r"q$_{\mathrm{v}}$ (g kg$^{-1}$)",
    "Pres": "Pres (Pa)",
    "Nc": r"N$_{\mathrm{c}}$ (kg$^{-1}$)",
    "qc": r"q$_{\mathrm{c}}$ (g kg$^{-1}$)",
    "Na": r"N$_{\mathrm{a}}$ (kg$^{-1}$)",
    "qa": r"q$_{\mathrm{a}}$ (g kg$^{-1}$)",
    "Nact": r"N$_{\mathrm{act}}$ (mg$^{-1}$ s$^{-1}$)",
    "Ndeact": r"N$_{\mathrm{deact}}$ (mg$^{-1}$ s$^{-1}$)",
}


def configure_matplotlib():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.unicode_minus": False,
        }
    )


def parse_image_format(text):
    image_format = text.strip().lower().lstrip(".")
    if "," in image_format:
        raise ValueError("Choose a single static figure format, for example 'pdf' or 'png'.")
    if image_format not in {"pdf", "png", "tif", "tiff"}:
        raise ValueError("Static figure format must be one of: pdf, png, tif, tiff.")
    return image_format


def parse_float_list(text):
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def parse_variable_list(text):
    if text.strip().lower() == "all":
        return FIELD_VARIABLES
    return [value.strip() for value in text.split(",") if value.strip()]


def parse_projection_variable_list(text):
    if text.strip().lower() == "all":
        return FIELD_VARIABLES
    return [value.strip() for value in text.split(",") if value.strip()]


def time_values_in_seconds(time_coord):
    values = time_coord.values
    if np.issubdtype(values.dtype, np.timedelta64):
        return values / np.timedelta64(1, "s")
    if np.issubdtype(values.dtype, np.datetime64):
        return (values - values[0]) / np.timedelta64(1, "s")

    numeric_values = values.astype(float)
    units = str(time_coord.attrs.get("units", "")).lower()
    if "hour" in units:
        return numeric_values * 3600.0
    if "minute" in units:
        return numeric_values * 60.0
    return numeric_values


def format_nc_time(time_coord, frame):
    values = time_coord.values
    value = values[int(frame)]
    if np.issubdtype(values.dtype, np.timedelta64):
        seconds = float(value / np.timedelta64(1, "s"))
        return f"{seconds:.0f} s ({seconds / 3600.0:.2f} h)"
    if np.issubdtype(values.dtype, np.datetime64):
        return np.datetime_as_string(value, unit="s")

    number = float(value)
    units = str(time_coord.attrs.get("units", "")).strip().lower()
    if "second" in units or units == "s":
        return f"{number:.0f} s ({number / 3600.0:.2f} h)"
    if "hour" in units:
        return f"{number:.2f} h"
    if "minute" in units:
        return f"{number:.1f} min"
    return f"{number:g}"


def selected_frames(frame_count, frame_step, max_frames):
    step = max(1, frame_step)
    frames = np.arange(0, frame_count, step, dtype=int)
    if max_frames is not None:
        frames = frames[: max(0, max_frames)]
    if frames.size == 0:
        raise ValueError("No animation frames selected.")
    return frames


def padded_limits(values):
    finite_values = np.asarray(values)[np.isfinite(values)]
    if finite_values.size == 0:
        return 0.0, 1.0
    data_min = float(np.nanmin(finite_values))
    data_max = float(np.nanmax(finite_values))
    span = data_max - data_min
    pad = 0.05 * span if span > 0 else max(abs(data_max) * 0.05, 1.0)
    return data_min - pad, data_max + pad


def robust_color_limits(values, force_zero_min=False):
    finite_values = np.asarray(values)[np.isfinite(values)]
    if finite_values.size == 0:
        return 0.0, 1.0

    data_min, data_max = np.nanpercentile(finite_values, [2.0, 98.0])
    if force_zero_min:
        data_min = 0.0
    if not np.isfinite(data_min) or not np.isfinite(data_max) or data_min == data_max:
        return padded_limits(finite_values)
    return float(data_min), float(data_max)


def coordinate_edge_bounds(coord):
    values = np.asarray(coord.values, dtype=float)
    if values.size == 0:
        raise ValueError(f"Coordinate {coord.name!r} is empty.")
    if values.size == 1:
        return float(values[0] - 0.5), float(values[0] + 0.5)

    lower_edge = values[0] - 0.5 * (values[1] - values[0])
    upper_edge = values[-1] + 0.5 * (values[-1] - values[-2])
    return float(lower_edge), float(upper_edge)


def even_pixel_figsize(width_inches, height_inches, dpi):
    width_pixels = int(np.ceil(width_inches * dpi / 2.0) * 2)
    height_pixels = int(np.ceil(height_inches * dpi / 2.0) * 2)
    return width_pixels / dpi, height_pixels / dpi


def color_limits(var_name, xy_plane, xz_plane):
    data_min = min(float(xy_plane.min(skipna=True)), float(xz_plane.min(skipna=True)))
    data_max = max(float(xy_plane.max(skipna=True)), float(xz_plane.max(skipna=True)))
    if var_name in SIGNED_VARIABLES and data_min < 0.0 < data_max:
        bound = max(abs(data_min), abs(data_max))
        return -bound, bound
    span = data_max - data_min
    pad = 0.05 * span if span > 0 else max(abs(data_max) * 0.05, 1.0)
    return data_min - pad, data_max + pad


def field_cmap(var_name):
    return "RdBu_r" if var_name in SIGNED_VARIABLES else "viridis"


def diagnostic_cmap(var_name):
    if var_name in SIGNED_VARIABLES:
        return "RdBu_r"
    if var_name in {"qc", "Nc", "Na", "Nact", "Ndeact"}:
        return "magma"
    if var_name in {"RH"}:
        return "YlGnBu"
    return "viridis"


def video_writer(fps):
    return FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=18000,
        extra_args=["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:white", "-pix_fmt", "yuv420p", "-crf", "18"],
    )


def save_static_figure(fig, output_dir, stem, image_format, image_dpi):
    output_path = output_dir / f"{stem}.{image_format}"
    dpi = image_dpi if image_format in {"png", "tif", "tiff"} else None
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved {output_path}")


def add_stat_args(parser, input_filename, default_data_dir):
    parser.add_argument("--data-dir", type=Path, default=default_data_dir, help="Directory containing the NetCDF input file. Defaults to the script directory.")
    parser.add_argument("--input-file", default=input_filename, help="Input NetCDF file name.")
    parser.add_argument("--output-dir", type=Path, help="Directory for figures and animations. Defaults to DATA_DIR/plots.")
    parser.add_argument("--frame-step", type=int, default=1, help="Use every Nth frame in animations.")
    parser.add_argument("--max-frames", type=int, help="Limit animation frames for local smoke tests only.")
    parser.add_argument("--fps", type=int, default=2, help="Animation frame rate.")
    parser.add_argument("--video-dpi", type=int, default=DEFAULT_VIDEO_DPI, help="Animation raster resolution.")
    parser.add_argument("--image-dpi", type=int, default=DEFAULT_IMAGE_DPI, help="Raster image resolution.")
    parser.add_argument("--image-format", default=DEFAULT_IMAGE_FORMAT, help="Static figure format. Choose one of: pdf, png, tif, tiff.")
    parser.add_argument("--profile-top", type=float, default=PROFILE_TOP_M, help="Upper y-axis limit for profiles in meters.")


def run_stat_plots(input_filename, output_suffix, default_data_dir=None):
    parser = argparse.ArgumentParser(description="Create ACE-ENA time-series and vertical-profile figures.")
    add_stat_args(parser, input_filename, Path(".") if default_data_dir is None else Path(default_data_dir))
    args = parser.parse_args()
    configure_matplotlib()

    data_dir = args.data_dir.expanduser().resolve()
    input_path = data_dir / args.input_file
    output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else data_dir / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(input_path) as ds:
        frames = selected_frames(len(ds["time"]), args.frame_step, args.max_frames)
        print(f"Using {len(frames)} of {len(ds['time'])} NetCDF time levels.")
        image_format = parse_image_format(args.image_format)
        plot_time_series(ds, output_dir, output_suffix, image_format, args.image_dpi)
        plot_cloud_diagnostics(ds, output_dir, output_suffix, image_format, args.image_dpi)
        plot_time_height_sections(ds, output_dir, output_suffix, image_format, args.image_dpi, args.profile_top)
        plot_profile_envelopes(ds, output_dir, output_suffix, image_format, args.image_dpi, args.profile_top)
        plot_vertical_profiles(ds, output_dir, output_suffix, frames, args.fps, args.video_dpi, args.profile_top)


def plot_time_series(ds, output_dir, output_suffix, image_format, image_dpi):
    time_hours = time_values_in_seconds(ds["time"]) / 3600.0

    fig, ax_height = plt.subplots(figsize=(6.0, 3.4))
    ax_lwp = ax_height.twinx()
    ax_lwp.spines["right"].set_visible(True)

    line_cb = ax_height.plot(time_hours, ds["CB"], color=COLORS["blue"], linewidth=1.6, label="Cloud base")
    line_ct = ax_height.plot(time_hours, ds["CT"], color=COLORS["vermillion"], linewidth=1.6, linestyle="--", label="Cloud top")
    line_lwp = ax_lwp.plot(time_hours, ds["LWP"], color=COLORS["gray"], linewidth=1.6, label="LWP")

    ax_height.set_xlabel("Time (h)")
    ax_height.set_ylabel("Height (m)")
    ax_lwp.set_ylabel(r"LWP (g m$^{-2}$)")
    ax_height.set_xlim(float(np.nanmin(time_hours)), float(np.nanmax(time_hours)))
    ax_height.grid(axis="y", color="#D9D9D9", linewidth=0.5)

    lines = line_cb + line_ct + line_lwp
    labels = [line.get_label() for line in lines]
    fig.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=3, handlelength=2.4)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    save_static_figure(fig, output_dir, f"time_series{output_suffix}", image_format, image_dpi)
    plt.close(fig)


def plot_cloud_diagnostics(ds, output_dir, output_suffix, image_format, image_dpi):
    required = {"time", "CB", "CT", "LWP"}
    if not required.issubset(ds.variables):
        return

    time_hours = time_values_in_seconds(ds["time"]) / 3600.0
    cloud_depth = np.asarray(ds["CT"].values - ds["CB"].values)
    fig, (ax_layer, ax_micro) = plt.subplots(2, 1, figsize=(6.6, 5.6), sharex=True)

    ax_lwp = ax_layer.twinx()
    ax_lwp.spines["right"].set_visible(True)
    ax_layer.plot(time_hours, cloud_depth, color=COLORS["blue"], linewidth=1.6, label="Cloud depth")
    ax_lwp.plot(time_hours, ds["LWP"].values, color=COLORS["gray"], linewidth=1.5, label="LWP")
    ax_layer.set_ylabel("Cloud depth (m)")
    ax_lwp.set_ylabel(r"LWP (g m$^{-2}$)")
    ax_layer.grid(axis="y", color="#D9D9D9", linewidth=0.5)

    micro_lines = []
    if "qc" in ds:
        line_qc, = ax_micro.plot(time_hours, ds["qc"].max(dim="z", skipna=True).values, color=COLORS["vermillion"], linewidth=1.5, label=r"max q$_{\mathrm{c}}$")
        micro_lines.append(line_qc)
    if "Nc" in ds:
        line_nc, = ax_micro.plot(time_hours, ds["Nc"].max(dim="z", skipna=True).values, color=COLORS["green"], linewidth=1.5, label=r"max N$_{\mathrm{c}}$")
        micro_lines.append(line_nc)
    if micro_lines:
        ax_micro.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=len(micro_lines), handlelength=2.4)
    ax_micro.set_xlabel("Time (h)")
    ax_micro.set_ylabel("Column maximum")
    ax_micro.set_xlim(float(np.nanmin(time_hours)), float(np.nanmax(time_hours)))
    ax_micro.grid(axis="y", color="#E5E5E5", linewidth=0.45)

    layer_lines = list(ax_layer.lines) + list(ax_lwp.lines)
    layer_labels = [line.get_label() for line in layer_lines]
    fig.legend(layer_lines, layer_labels, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=2, handlelength=2.4)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    save_static_figure(fig, output_dir, f"cloud_diagnostics{output_suffix}", image_format, image_dpi)
    plt.close(fig)


def plot_time_height_sections(ds, output_dir, output_suffix, image_format, image_dpi, profile_top):
    variables = [name for name in ("qc", "Nc", "RH", "TKE_res") if name in ds and {"time", "z"}.issubset(ds[name].dims)]
    if not variables:
        return

    time_hours = time_values_in_seconds(ds["time"]) / 3600.0
    z_values = ds["z"].values
    z_mask = z_values <= profile_top
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, var_name in zip(axes, variables):
        data = ds[var_name].isel(z=z_mask).values.T
        vmin, vmax = robust_color_limits(data, force_zero_min=var_name in {"qc", "Nc", "TKE_res"})
        mesh = ax.pcolormesh(time_hours, z_values[z_mask], data, shading="auto", cmap=diagnostic_cmap(var_name), vmin=vmin, vmax=vmax)
        ax.set_title(PROFILE_LABELS.get(var_name, var_name))
        ax.set_xlabel("Time (h)")
        ax.grid(False)
        colorbar = fig.colorbar(mesh, ax=ax, pad=0.015)
        colorbar.ax.tick_params(labelsize=8)

    for ax in axes[len(variables) :]:
        ax.set_visible(False)
    for ax in axes[::2]:
        ax.set_ylabel("Height (m)")

    fig.tight_layout()
    save_static_figure(fig, output_dir, f"time_height_sections{output_suffix}", image_format, image_dpi)
    plt.close(fig)


def plot_profile_envelopes(ds, output_dir, output_suffix, image_format, image_dpi, profile_top):
    variables = [name for name in ("thetal", "qt", "qc", "Nc", "RH", "TKE_res") if name in ds and {"time", "z"}.issubset(ds[name].dims)]
    if not variables:
        return

    z_values = ds["z"].values
    z_mask = z_values <= profile_top
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.8), sharey=True)
    axes = axes.ravel()

    for ax, var_name in zip(axes, variables):
        data = ds[var_name].isel(z=z_mask).values
        mean_profile = np.nanmean(data, axis=0)
        lower_profile = np.nanpercentile(data, 10.0, axis=0)
        upper_profile = np.nanpercentile(data, 90.0, axis=0)
        ax.fill_betweenx(z_values[z_mask], lower_profile, upper_profile, color=COLORS["sky"], alpha=0.32, linewidth=0)
        ax.plot(mean_profile, z_values[z_mask], color=COLORS["blue"], linewidth=1.5)
        ax.set_xlabel(PROFILE_LABELS.get(var_name, var_name))
        ax.grid(axis="x", color="#E5E5E5", linewidth=0.45)
        ax.set_ylim(0.0, profile_top)

    for ax in axes[len(variables) :]:
        ax.set_visible(False)
    for ax in axes[::3]:
        ax.set_ylabel("Height (m)")

    fig.tight_layout()
    save_static_figure(fig, output_dir, f"profile_envelopes{output_suffix}", image_format, image_dpi)
    plt.close(fig)


def plot_vertical_profiles(ds, output_dir, output_suffix, frame_indices, fps, video_dpi, profile_top):
    variables = [name for name in PROFILE_LABELS if name in ds and "z" in ds[name].dims]
    z_values = ds["z"].values
    ncols = 7
    nrows = int(np.ceil(len(variables) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(12.8, 7.2), sharey=True)
    axes = np.atleast_1d(axes).ravel()

    lines = []
    for ax, var_name in zip(axes, variables):
        line, = ax.plot([], [], color=COLORS["black"], linewidth=1.3)
        lines.append((line, var_name))
        ax.set_xlim(*padded_limits(ds[var_name].values))
        ax.set_ylim(0.0, profile_top)
        ax.set_xlabel(PROFILE_LABELS[var_name])
        ax.grid(axis="x", color="#E5E5E5", linewidth=0.45)

    for ax in axes[len(variables) :]:
        ax.set_visible(False)
    for ax in axes[::ncols]:
        ax.set_ylabel("Height (m)")

    def update(frame):
        fig.suptitle(f"Time = {format_nc_time(ds['time'], frame)}", y=0.995)
        for line, var_name in lines:
            line.set_data(ds[var_name].isel(time=int(frame)).values, z_values)
        return [line for line, _ in lines]

    animation = FuncAnimation(fig, update, frames=frame_indices, blit=False, repeat=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    output_path = output_dir / f"vertical_profiles_animation{output_suffix}.mp4"
    animation.save(output_path, writer=video_writer(fps), dpi=video_dpi)
    plt.close(fig)
    print(f"Saved {output_path}")


def add_3d_args(parser, input_filename, default_data_dir):
    parser.add_argument("--data-dir", type=Path, default=default_data_dir, help="Directory containing the NetCDF input file. Defaults to the script directory.")
    parser.add_argument("--input-file", default=input_filename, help="Input NetCDF file name.")
    parser.add_argument("--output-dir", type=Path, help="Directory for animations. Defaults to DATA_DIR/plots.")
    parser.add_argument("--variables", default=",".join(FIELD_VARIABLES), help="Comma-separated variables to animate.")
    parser.add_argument("--frame-step", type=int, default=1, help="Use every Nth frame in animations.")
    parser.add_argument("--max-frames", type=int, help="Limit animation frames for local smoke tests only.")
    parser.add_argument("--fps", type=int, default=10, help="Field animation frame rate.")
    parser.add_argument("--dsd-fps", type=int, default=2, help="DSD animation frame rate.")
    parser.add_argument("--video-dpi", type=int, default=DEFAULT_VIDEO_DPI, help="Animation raster resolution.")
    parser.add_argument("--z-slice-height", type=float, default=700.5, help="Height for the horizontal slice in meters.")
    parser.add_argument("--y-slice-location", type=float, default=1697.5, help="Y location for the vertical slice in meters.")
    parser.add_argument("--dsd-heights", default="697.5,747.5,797.5,847.5", help="Comma-separated DSD heights in meters.")
    parser.add_argument("--projection-variables", default=",".join(PROJECTION_VARIABLES), help="Comma-separated variables for vertical-maximum projection animations.")
    parser.add_argument("--skip-fields", action="store_true", help="Skip standard 3-D field animations.")
    parser.add_argument("--skip-projections", action="store_true", help="Skip vertical-maximum projection animations.")
    parser.add_argument("--skip-dsd", action="store_true", help="Skip DSD animation.")


def run_3d_plots(input_filename, output_suffix, default_data_dir=None):
    parser = argparse.ArgumentParser(description="Create ACE-ENA 3-D field and droplet-spectrum animations.")
    add_3d_args(parser, input_filename, Path(".") if default_data_dir is None else Path(default_data_dir))
    args = parser.parse_args()
    configure_matplotlib()

    data_dir = args.data_dir.expanduser().resolve()
    input_path = data_dir / args.input_file
    output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else data_dir / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(input_path, decode_times=False) as ds:
        frame_indices = selected_frames(len(ds["time"]), args.frame_step, args.max_frames)
        print(f"Using {len(frame_indices)} of {len(ds['time'])} NetCDF time levels.")
        z_index = int(np.argmin(np.abs(ds["z"].values - args.z_slice_height)))
        y_index = int(np.argmin(np.abs(ds["y"].values - args.y_slice_location)))
        requested_variables = parse_variable_list(args.variables)
        variables = [var_name for var_name in requested_variables if var_name in ds]
        missing = [var_name for var_name in requested_variables if var_name not in ds]
        if missing:
            print(f"Skipping missing variables: {', '.join(missing)}")

        if not args.skip_fields:
            for var_name in variables:
                plot_field_animation(ds, var_name, output_dir, output_suffix, frame_indices, args.fps, args.video_dpi, z_index, y_index)
        if not args.skip_projections:
            projection_variables = [var_name for var_name in parse_projection_variable_list(args.projection_variables) if var_name in ds]
            for var_name in projection_variables:
                plot_vertical_max_projection_animation(ds, var_name, output_dir, output_suffix, frame_indices, args.fps, args.video_dpi)
        if not args.skip_dsd:
            plot_dsd_animation(ds, output_dir, output_suffix, frame_indices, args.dsd_fps, args.video_dpi, parse_float_list(args.dsd_heights))


def plot_field_animation(ds, var_name, output_dir, output_suffix, frame_indices, fps, video_dpi, z_index, y_index):
    print(f"Rendering {var_name}")
    xy_plane = ds[var_name].isel(z=z_index)
    xz_plane = ds[var_name].isel(y=y_index)
    vmin, vmax = color_limits(var_name, xy_plane, xz_plane)
    x_min, x_max = coordinate_edge_bounds(ds["x"])
    y_min, y_max = coordinate_edge_bounds(ds["y"])
    z_min, z_max = coordinate_edge_bounds(ds["z"])
    x_span = max(x_max - x_min, 1.0)
    y_span = max(y_max - y_min, 1.0)
    z_span = max(z_max - z_min, 1.0)
    xy_box_aspect = y_span / x_span
    xz_box_aspect = z_span / x_span
    figure_width = 7.6
    plot_width_fraction = 0.72
    plot_height_fraction = 0.80
    figure_height = figure_width * plot_width_fraction * (xy_box_aspect + xz_box_aspect) / plot_height_fraction
    figure_size = even_pixel_figsize(figure_width, figure_height, video_dpi)

    fig = plt.figure(figsize=figure_size)
    grid = GridSpec(2, 1, figure=fig, height_ratios=[y_span, z_span], hspace=0.26)
    ax_xy = fig.add_subplot(grid[0])
    ax_xz = fig.add_subplot(grid[1], sharex=ax_xy)

    first_frame = int(frame_indices[0])
    image_xy = ax_xy.imshow(
        xy_plane.isel(time=first_frame).values,
        extent=(x_min, x_max, y_min, y_max),
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
        cmap=field_cmap(var_name),
        interpolation="nearest",
    )
    image_xz = ax_xz.imshow(
        xz_plane.isel(time=first_frame).values,
        extent=(x_min, x_max, z_min, z_max),
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
        cmap=field_cmap(var_name),
        interpolation="nearest",
    )

    cax = fig.add_axes([0.86, 0.16, 0.025, 0.68])
    colorbar = fig.colorbar(image_xy, cax=cax, orientation="vertical")
    colorbar.set_label(FIELD_LABELS.get(var_name, var_name))
    colorbar.ax.tick_params(labelsize=8)

    ax_xy.set_box_aspect(xy_box_aspect)
    ax_xz.set_box_aspect(xz_box_aspect)
    ax_xy.set_title(f"Horizontal slice, z = {float(ds['z'].isel(z=z_index)):.1f} m")
    ax_xz.set_title(f"Vertical slice, y = {float(ds['y'].isel(y=y_index)):.1f} m")
    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xz.set_xlabel("X (m)")
    ax_xz.set_ylabel("Z (m)")
    fig.subplots_adjust(left=0.10, right=0.82, bottom=0.08, top=0.91)

    def update(frame):
        frame = int(frame)
        fig.suptitle(f"{var_name}: Time = {format_nc_time(ds['time'], frame)}", y=0.96)
        image_xy.set_data(xy_plane.isel(time=frame).values)
        image_xz.set_data(xz_plane.isel(time=frame).values)
        return image_xy, image_xz

    animation = FuncAnimation(fig, update, frames=frame_indices, blit=False, repeat=False)
    output_path = output_dir / f"{var_name}_animation{output_suffix}.mp4"
    animation.save(output_path, writer=video_writer(fps), dpi=video_dpi)
    plt.close(fig)
    print(f"Saved {output_path}")


def sample_projection_limits(var_name, data_array, frame_indices):
    sample_count = min(5, len(frame_indices))
    sample_positions = np.linspace(0, len(frame_indices) - 1, sample_count, dtype=int)
    sampled = []
    for position in sample_positions:
        projection = data_array.isel(time=int(frame_indices[position])).max(dim="z", skipna=True).values
        sampled.append(np.asarray(projection).ravel())
    return robust_color_limits(np.concatenate(sampled), force_zero_min=var_name not in SIGNED_VARIABLES)


def plot_vertical_max_projection_animation(ds, var_name, output_dir, output_suffix, frame_indices, fps, video_dpi):
    if not {"time", "z", "y", "x"}.issubset(ds[var_name].dims):
        return

    print(f"Rendering {var_name} vertical-maximum projection")
    data_array = ds[var_name]
    vmin, vmax = sample_projection_limits(var_name, data_array, frame_indices)
    x_min, x_max = coordinate_edge_bounds(ds["x"])
    y_min, y_max = coordinate_edge_bounds(ds["y"])
    first_frame = int(frame_indices[0])

    fig, ax = plt.subplots(figsize=even_pixel_figsize(7.2, 6.1, video_dpi))
    fig.subplots_adjust(left=0.10, right=0.84, bottom=0.12, top=0.88)
    projection = data_array.isel(time=first_frame).max(dim="z", skipna=True).values
    image = ax.imshow(
        projection,
        extent=(x_min, x_max, y_min, y_max),
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
        cmap=diagnostic_cmap(var_name),
        interpolation="nearest",
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Vertical maximum of {FIELD_LABELS.get(var_name, var_name)}")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
    colorbar.set_label(FIELD_LABELS.get(var_name, var_name))
    colorbar.ax.tick_params(labelsize=8)

    def update(frame):
        frame = int(frame)
        fig.suptitle(f"{var_name} max projection: Time = {format_nc_time(ds['time'], frame)}", y=0.965)
        projection = data_array.isel(time=frame).max(dim="z", skipna=True).values
        image.set_data(projection)
        return (image,)

    animation = FuncAnimation(fig, update, frames=frame_indices, blit=False, repeat=False)
    output_path = output_dir / f"{var_name}_max_projection_animation{output_suffix}.mp4"
    animation.save(output_path, writer=video_writer(fps), dpi=video_dpi)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_dsd_animation(ds, output_dir, output_suffix, frame_indices, fps, video_dpi, heights):
    if "DSD" not in ds:
        print("Variable 'DSD' is not available; skipping DSD animation.")
        return

    dsd_data = ds["DSD"]
    z_values = ds["z"].values
    radius = ds["radius_bin_centers"].values
    z_indices = [int(np.argmin(np.abs(z_values - height))) for height in heights]
    mean_dims = [dim for dim in ("x_proc", "y_proc") if dim in dsd_data.dims]
    sampled = dsd_data.isel(time=frame_indices, z=z_indices).mean(dim=mean_dims)
    global_max = float(sampled.max(skipna=True))
    y_max = max(1.0, global_max * 1.2)

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    fig.subplots_adjust(left=0.10, right=0.74, bottom=0.14, top=0.84)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(float(np.nanmin(radius)), float(np.nanmax(radius)))
    ax.set_ylim(1e-2, y_max)
    ax.set_xlabel("Radius (m)")
    ax.set_ylabel(r"dN/dlogR (m$^{-3}$)")
    ax.grid(True, which="both", color="#E5E5E5", linewidth=0.45)

    lines = []
    for color, z_index in zip(LINE_COLORS, z_indices):
        line, = ax.plot([], [], color=color, linewidth=1.6, label=f"z = {z_values[z_index]:.1f} m")
        lines.append((line, z_index))
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=9)

    def update(frame):
        frame = int(frame)
        fig.suptitle(f"DSD: Time = {format_nc_time(ds['time'], frame)}", y=0.96)
        for line, z_index in lines:
            mean_dsd = dsd_data.isel(time=frame, z=z_index).mean(dim=mean_dims).values
            line.set_data(radius, np.where(mean_dsd > 0.0, mean_dsd, np.nan))
        return [line for line, _ in lines]

    animation = FuncAnimation(fig, update, frames=frame_indices, blit=False, repeat=False)
    output_path = output_dir / f"dsd_animation{output_suffix}.mp4"
    animation.save(output_path, writer=video_writer(fps), dpi=video_dpi)
    plt.close(fig)
    print(f"Saved {output_path}")
