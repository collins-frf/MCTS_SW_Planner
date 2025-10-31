# plotting.py
"""
Contains all plotting and animation functions for the simulation.
- _plot_simulation_frame: Generates a single, complex frame.
- create_gif_from_frames: Assembles frames into a GIF.
"""

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import imageio.v3 as iio
from scipy.interpolate import griddata


def _plot_simulation_frame(t_idx, vessel_path, motor_history, waypoint_list,
                           env, known_bathymetry,
                           plot_limits, zoom_extents, temp_dir):
    """Generates and saves a single frame of the simulation animation."""
    ny, nx = env.grid_shape
    frame_num = t_idx

    # Unpack plot_limits dictionary
    dt = plot_limits['dt']
    contour_level = plot_limits['contour_level']
    vmax_vel = plot_limits['vmax_vel']
    vmin_bathy = plot_limits['vmin_bathy']
    vmax_bathy = plot_limits['vmax_bathy']
    max_motor_val = plot_limits['max_motor_val']
    full_potential_time = plot_limits['full_potential_time']

    fig = plt.figure(figsize=(24, 10))
    gs = gridspec.GridSpec(2, 3, height_ratios=[3, 1])
    ax1 = fig.add_subplot(gs[0, 0])  # Top-left: Velocity Magnitude
    ax2 = fig.add_subplot(gs[0, 1])  # Top-middle: Known Bathymetry
    ax3 = fig.add_subplot(gs[0, 2])  # Top-right: Full Bathymetry
    ax4 = fig.add_subplot(gs[1, :])  # Bottom: Motor History
    fig.suptitle(f"Vessel Mission Plan - Time: {t_idx * dt:.1f}s", fontsize=16)

    # --- Panel 1: Velocity Magnitude (ax1) ---
    vel_idx = min(t_idx, env.total_timesteps - 1)
    vel_mag = np.sqrt(env.u_all_timesteps[vel_idx] ** 2 + env.v_all_timesteps[vel_idx] ** 2)
    im1 = ax1.imshow(vel_mag, cmap='viridis', origin='lower', extent=[0, nx, 0, ny], vmin=0, vmax=vmax_vel,
                     aspect='auto')
    ax1.set_title("Velocity Magnitude")
    ax1.set_xlabel("Easting (m)")
    ax1.set_ylabel("Northing (m)")
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Velocity Mag (m/s)')

    # --- Panel 2: Known Bathymetry (ax2) ---
    known_bathy_grid = np.full(env.grid_shape, np.nan)

    if len(known_bathymetry) > 3:
        try:
            points = []
            values = []
            for (ix, iy), depth in known_bathymetry.items():
                points.append([ix, iy])
                values.append(depth)

            grid_y, grid_x = np.mgrid[0:ny, 0:nx]
            interpolated_grid = griddata(points, values, (grid_x, grid_y), method='nearest')
            known_bathy_grid = interpolated_grid
        except Exception:
            # Fallback for plotting sparse points
            for (ix, iy), depth in known_bathymetry.items():
                if 0 <= iy < ny and 0 <= ix < nx:
                    known_bathy_grid[iy, ix] = depth
    else:
        for (ix, iy), depth in known_bathymetry.items():
            if 0 <= iy < ny and 0 <= ix < nx:
                known_bathy_grid[iy, ix] = depth

    im2 = ax2.imshow(known_bathy_grid, cmap='terrain', origin='lower', extent=[0, nx, 0, ny],
                     vmin=vmin_bathy, vmax=vmax_bathy, aspect='auto')
    ax2.set_title(f"Known Bathymetry (Surveyed)")
    ax2.set_xlabel("Easting (m)")
    ax2.set_ylabel("Northing (m)")
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Depth (m)')

    # --- Panel 3: Full Bathymetry (ax3) ---
    im3 = ax3.imshow(env.bathymetry_full, cmap='terrain', origin='lower', extent=[0, nx, 0, ny], vmin=vmin_bathy,
                     vmax=vmax_bathy, aspect='auto')
    ax3.set_title(f"Full Bathymetry (Land >= {contour_level}m)")
    ax3.set_xlabel("Easting (m)")
    ax3.set_ylabel("Northing (m)")
    cbar3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.set_label('Depth (m)')

    # --- Plot waypoints and path on TOP axes ---
    if waypoint_list:
        wp_x, wp_y = zip(*waypoint_list)
    else:
        wp_x, wp_y = [], []

    for ax_top in [ax1, ax2, ax3]:
        if wp_x:
            ax_top.plot(wp_x, wp_y, 's--', color='white', markersize=4, linewidth=1, alpha=0.7, label='Waypoints')

        current_path = vessel_path
        if current_path:
            path_x, path_y = zip(*current_path)
            ax_top.plot(path_x, path_y, 'w-', linewidth=2.5, label='Vessel Path')
            ax_top.plot(path_x[-1], path_y[-1], 'o', color='cyan', markersize=8, markeredgecolor='black')

        if zoom_extents:
            ax_top.set_xlim(zoom_extents[0], zoom_extents[1])
            ax_top.set_ylim(zoom_extents[2], zoom_extents[3])
        else:
            ax_top.set_xlim(0, nx)
            ax_top.set_ylim(0, ny)

    ax1.legend(loc='upper right')

    # --- Panel 4: Motor History (ax4) ---
    current_time = t_idx * dt
    motor_arr = np.array(motor_history)
    actual_time_axis_plot = np.arange(len(motor_arr)) * dt

    if motor_arr.size > 0:
        ax4.plot(actual_time_axis_plot, motor_arr[:, 0], 'r-', label='Motor X')
        ax4.plot(actual_time_axis_plot, motor_arr[:, 1], 'b-', label='Motor Y')

    ax4.axvline(current_time, color='cyan', linestyle='--', linewidth=2, label='Current Time')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Motor Velocity (m/s)')
    ax4.set_title('Motor Command History')
    ax4.set_xlim(0, full_potential_time if full_potential_time > 0 else 1.0)
    ax4.set_ylim(-max_motor_val, max_motor_val)
    ax4.legend(loc='upper right')
    ax4.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    frame_path = os.path.join(temp_dir, f'frame_{frame_num:04d}.png')
    plt.savefig(frame_path, dpi=90)
    plt.close(fig)


def create_gif_from_frames(frames_dir, output_filename, fps):
    """Collects saved PNG frames and assembles them into a GIF."""
    print(f"\nAssembling GIF: {output_filename}...")
    frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.png')])

    if not frame_files:
        print("  -> No frames found in temp directory.")
        return

    try:
        # Use imageio.v3 syntax
        frames = [iio.imread(filename) for filename in frame_files]
        iio.imwrite(output_filename, frames, duration=(1000 / fps), loop=0)

        print(f"  -> Successfully saved GIF with {len(frame_files)} frames.")
    except Exception as e:
        print(f"Error creating GIF: {e}")

    # Clean up temporary frames
    if os.path.exists(frames_dir):
        try:
            shutil.rmtree(frames_dir)
            print(f"  -> Cleaned up temp frames directory: {frames_dir}")
        except Exception as e:
            print(f"Error removing temp frames directory: {e}")