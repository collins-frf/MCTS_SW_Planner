# main.py
"""
Main entry point for the Vessel Mission Planner.

This script initializes all components and runs the simulation.
1. Loads configuration from config.py.
2. Initializes the Environment.
3. Loads or prompts user for a survey polygon (UI).
4. Generates the static waypoint route (RouteGenerator).
5. Initializes the reactive MCTSPlanner.
6. Runs the simulation loop.
7. Generates the final GIF animation.
"""

import os
import numpy as np
import config

# Import all custom modules
from environment import Environment
from ui import load_or_select_polygon
from planning import RouteGenerator, MCTSPlanner
from simulation import run_simulation
from plotting import create_gif_from_frames


def main():
    """ Main execution function. """

    # --- 1. Load Environment ---
    print("Initializing environment...")
    env = Environment(
        data_dir=config.OUTPUT_DIR,
        bathy_file=config.FILE_BATHYMETRY,
        nx_file=config.FILE_NX,
        ny_file=config.FILE_NY
    )
    try:
        env.load_data()
    except FileNotFoundError:
        print("Error: Could not load critical environment files. Exiting.")
        return

    ny, nx = env.grid_shape

    # --- 2. Get Survey Area ---
    survey_polygon = load_or_select_polygon(
        config.POLYGON_FILE_PATH,
        env.bathymetry_full,
        env.grid_shape
    )
    if survey_polygon is None:
        print("No survey polygon selected. Exiting.")
        return

    # --- 3. Generate Waypoints (Static Route) ---
    shore_normal_angle_rad = RouteGenerator.find_shoreline_orientation(
        env.bathymetry_full,
        env.grid_shape,
        config.CONTOUR_LEVEL_METERS,
        survey_polygon
    )
    waypoints = RouteGenerator.generate_rotated_waypoints(
        env.bathymetry_full,
        env.grid_shape,
        config.CONTOUR_LEVEL_METERS,
        survey_polygon,
        shore_normal_angle_rad,
        config.TRANSECT_SPACING
    )
    if not waypoints:
        print("Failed to generate waypoints. Exiting.")
        return

    # --- 4. Initialize Reactive Planner ---
    planner = MCTSPlanner(env, survey_polygon)

    # --- 5. Calculate Plotting Limits ---
    print("Calculating plot zoom extents and limits...")
    waypoints_arr = np.array(waypoints)
    if waypoints_arr.size > 0:
        x_min_wp = np.min(waypoints_arr[:, 0]) - config.PLOT_BUFFER
        x_max_wp = np.max(waypoints_arr[:, 0]) + config.PLOT_BUFFER
        y_min_wp = np.min(waypoints_arr[:, 1]) - config.PLOT_BUFFER
        y_max_wp = np.max(waypoints_arr[:, 1]) + config.PLOT_BUFFER
    else:
        # Fallback to polygon bounds
        x_min_wp = np.min(survey_polygon[:, 0]) - config.PLOT_BUFFER
        x_max_wp = np.max(survey_polygon[:, 0]) + config.PLOT_BUFFER
        y_min_wp = np.min(survey_polygon[:, 1]) - config.PLOT_BUFFER
        y_max_wp = np.max(survey_polygon[:, 1]) + config.PLOT_BUFFER

    zoom_extents = [max(0, x_min_wp), min(nx, x_max_wp), max(0, y_min_wp), min(ny, y_max_wp)]
    print(
        f"  -> Zoom extents set to: X[{zoom_extents[0]:.1f}, {zoom_extents[1]:.1f}], Y[{zoom_extents[2]:.1f}, {zoom_extents[3]:.1f}]")

    # Pre-calculate other limits
    full_potential_time = (env.total_timesteps - 1) * config.DT
    max_motor_val = config.MAX_MOTOR_SPEED * 1.1

    vmax_vel = 1.0
    try:
        initial_vel_mag = np.sqrt(env.u_all_timesteps[0] ** 2 + env.v_all_timesteps[0] ** 2)
        vmax_vel = np.percentile(initial_vel_mag, 98)
    except Exception:
        pass  # Keep default

    water_bathy = env.bathymetry_full[env.bathymetry_full < config.CONTOUR_LEVEL_METERS]
    vmin_bathy = np.min(water_bathy) if water_bathy.size > 0 else config.CONTOUR_LEVEL_METERS - 1.0
    vmax_bathy = 0.0

    # Bundle plot limits into a dictionary
    plot_limits = {
        'dt': config.DT,
        'contour_level': config.CONTOUR_LEVEL_METERS,
        'vmax_vel': vmax_vel,
        'vmin_bathy': vmin_bathy,
        'vmax_bathy': vmax_bathy,
        'max_motor_val': max_motor_val,
        'full_potential_time': full_potential_time
    }

    # --- 6. Run Simulation ---
    run_simulation(
        env,
        planner,
        waypoints,
        plot_limits,
        zoom_extents
    )

    # --- 7. Assemble GIF ---
    create_gif_from_frames(
        config.TEMP_FRAMES_DIR,
        config.GIF_FILENAME,
        config.GIF_FPS
    )

    print("\n\nAll processes complete.")


if __name__ == '__main__':
    main()