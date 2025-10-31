# config.py
"""
Contains all configuration constants for the simulation.
"""

import os
import numpy as np

# --- File & Directory Settings ---
OUTPUT_DIR = './output/'
TEMP_FRAMES_DIR = os.path.join(OUTPUT_DIR, 'temp_planner_frames/')
GIF_FILENAME = os.path.join(OUTPUT_DIR, "vessel_mission.gif")
POLYGON_FILE_PATH = os.path.join(OUTPUT_DIR, 'survey_polygon.npy')
GIF_FPS = 5

# --- Data File Names (relative to OUTPUT_DIR) ---
FILE_NX = 'nx.txt'
FILE_NY = 'ny.txt'
FILE_BATHYMETRY = 'bathytopo.bin'
DATA_TYPE = np.dtype('<f')

# --- Simulation Settings ---
DT = 1.0  # Timestep in seconds

# --- Vessel Settings ---
TRANSECT_SPACING = 20  # Spacing between transects (m)
VESSEL_SPEED = 2.0  # Desired vessel speed (m/s)
WAYPOINT_REACH_THRESHOLD = 5.0  # How close to get to a waypoint (m)
MAX_MOTOR_SPEED = 6.0  # Max motor speed (m/s)

# --- Environment Settings ---
CONTOUR_LEVEL_METERS = -2.0  # The "shoreline" depth for collision

# --- MCTS Settings ---
MCTS_ITERATIONS = 100  # Keep this high
MCTS_EXPLORATION_CONSTANT = 1.414
MCTS_SIMULATION_DEPTH = 25  # Keep this high

## --- MCTS Reward Weights (These are for the *real* reward) ---
W_PROGRESS_PRIMARY = 10.0
W_PROGRESS_SECONDARY = 2.0
W_EXPLORE = 0.5
W_OUTSIDE_POLY_PENALTY = 20.0
W_CROSS_TRACK_PENALTY = 1.0

## --- Rollout Policy Weights (These are for the *imagination*) ---
# [REMOVED] W_GRADIENT is no longer needed. Our policy is smarter.
# W_GRADIENT = 2.0
W_TARGET_POLICY = 1.0       # (Rollout policy) How strongly to favor the target

# --- Plotting Settings ---
PLOT_BUFFER = 20.0  # Buffer around waypoints for zoom