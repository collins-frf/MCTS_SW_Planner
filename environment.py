# environment.py
"""
Defines the Environment class, which loads and provides access to
bathymetry and velocity (drift) data.
"""

import os
import re
import numpy as np
from scipy.interpolate import interpn
import config


class Environment:
    """
    Loads and contains all static environmental data (grid, bathymetry,
    and time-series velocity fields). Provides helper methods to
    query this data.
    """

    def __init__(self, data_dir, bathy_file, nx_file, ny_file):
        self.data_dir = data_dir
        self.bathy_file_path = os.path.join(data_dir, bathy_file)
        self.nx_file_path = os.path.join(data_dir, nx_file)
        self.ny_file_path = os.path.join(data_dir, ny_file)

        self.nx = 0
        self.ny = 0
        self.grid_shape = (0, 0)
        self.bathymetry_full = None
        self.u_all_timesteps = None
        self.v_all_timesteps = None

        self.total_timesteps = 0
        self.points_full = None  # For interpolation
        self.expected_elements = 0

    def load_data(self):
        """
        Loads all grid, bathymetry, and velocity data from files.
        """
        print("Loading environment data...")
        # --- 1. Load Grid ---
        try:
            with open(self.ny_file_path, 'r') as f:
                self.ny = int(f.read().strip())
            with open(self.nx_file_path, 'r') as f:
                self.nx = int(f.read().strip())
            self.grid_shape = (self.ny, self.nx)
            self.expected_elements = self.ny * self.nx
            print(f"Grid dimensions loaded: {self.grid_shape}")
        except FileNotFoundError as e:
            print(f"Error: Dimension file not found: {e.filename}")
            raise

        # For interpolation
        y_pts_full, x_pts_full = np.arange(self.ny), np.arange(self.nx)
        self.points_full = (y_pts_full, x_pts_full)

        # --- 2. Load Bathymetry ---
        try:
            self.bathymetry_full = np.fromfile(self.bathy_file_path, dtype=config.DATA_TYPE) \
                [-self.expected_elements:].reshape(self.grid_shape)
            print(
                f"Bathymetry loaded. Min: {np.min(self.bathymetry_full):.2f}, Max: {np.max(self.bathymetry_full):.2f}")
        except FileNotFoundError:
            print(f"Warning: Bathymetry file not found: {self.bathy_file_path}. Defaulting to all water.")
            self.bathymetry_full = np.full(self.grid_shape, -10.0)  # Default deep water

        # --- 3. Find and Load Velocity Files ---
        print("Finding and grouping all time-series .bin files...")
        timeseries_files = {}
        for filename in os.listdir(self.data_dir):
            match = re.match(r'([a-zA-Z_]+)_(\d+)\.bin', filename)
            if match:
                var, num = match.groups()
                num = int(num)
                timeseries_files.setdefault(num, {})[var] = os.path.join(self.data_dir, filename)

        if not timeseries_files:
            print("No time-series files found. Using zero drift.")
            self.total_timesteps = 1  # Set to 1 to avoid divide-by-zero
            self.u_all_timesteps = np.zeros((1, self.ny, self.nx))
            self.v_all_timesteps = np.zeros((1, self.ny, self.nx))
        else:
            sorted_timesteps = sorted(timeseries_files.keys())
            self.total_timesteps = len(sorted_timesteps)

            print(f"\nPre-loading all velocity data ({self.total_timesteps} timesteps)...")
            self.u_all_timesteps = np.zeros((self.total_timesteps, self.ny, self.nx))
            self.v_all_timesteps = np.zeros((self.total_timesteps, self.ny, self.nx))

            for i, timestep in enumerate(sorted_timesteps):
                files = timeseries_files[timestep]
                if 'xvelo' in files and 'yvelo' in files:
                    self.u_all_timesteps[i, :, :] = np.fromfile(files['xvelo'], dtype=config.DATA_TYPE)[
                                                    -self.expected_elements:].reshape(self.grid_shape)
                    self.v_all_timesteps[i, :, :] = np.fromfile(files['yvelo'], dtype=config.DATA_TYPE)[
                                                    -self.expected_elements:].reshape(self.grid_shape)
            print("Velocity data loading complete.")

    def get_drift(self, pos, time_idx):
        """
        Get the water drift (u, v) at a specific position and time index.
        """
        t_idx = min(time_idx, self.total_timesteps - 1)
        px, py = pos

        u_drift = interpn(self.points_full, self.u_all_timesteps[t_idx], (py, px),
                          method='nearest', bounds_error=False, fill_value=0)[0]
        v_drift = interpn(self.points_full, self.v_all_timesteps[t_idx], (py, px),
                          method='nearest', bounds_error=False, fill_value=0)[0]

        return np.array([u_drift, v_drift])

    def get_depth(self, pos):
        """
        Get the true bathymetry depth at a specific position.
        """
        px, py = pos
        ix, iy = int(np.round(px)), int(np.round(py))

        if not (0 <= ix < self.nx and 0 <= iy < self.ny):
            return 100.0  # Return "land" if out of bounds

        return self.bathymetry_full[iy, ix]

    def is_collision(self, pos, contour_level):
        """
        Check if a position is a collision (on land or out of bounds).
        """
        px, py = pos
        ix, iy = int(np.round(px)), int(np.round(py))

        # Check 1: Out of Bounds
        if not (0 <= ix < self.nx and 0 <= iy < self.ny):
            return True

        # Check 2: Land Collision
        if self.bathymetry_full[iy, ix] >= contour_level:
            return True

        return False