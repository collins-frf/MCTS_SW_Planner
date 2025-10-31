# simulation.py
"""
Contains the main simulation loop function `run_simulation`.
"""

import os
import shutil
import numpy as np
import config
import plotting


def run_simulation(env, planner, waypoint_list, plot_limits, zoom_extents):
    """
    - Starts at waypoint_list[0].
    - Runs simulation, calling planner and plotter at each step.
    - Stops on mission complete, collision, or out of bounds.
    """
    vessel_path = []
    motor_history = []

    if len(waypoint_list) < 1:
        print("Not enough waypoints provided to start simulation.")
        return

    # --- Setup for frame generation ---
    if os.path.exists(config.TEMP_FRAMES_DIR):
        shutil.rmtree(config.TEMP_FRAMES_DIR)
    os.makedirs(config.TEMP_FRAMES_DIR)

    # --- Start at the first waypoint ---
    start_waypoint = waypoint_list[0]
    current_waypoint_target_index = 1

    vessel_path = [start_waypoint]
    planner.update_known_bathymetry(start_waypoint)
    motor_history.append((0.0, 0.0))  # Initial motor command is zero

    vessel_active = True  # Whether following standard waypoints
    using_mcts = False  # Flag to switch to MCTS planner

    print(f"Starting simulation at waypoint {start_waypoint}, targeting index {current_waypoint_target_index}")
    print("\nStarting vessel simulation loop...")

    for t_idx in range(env.total_timesteps):

        # --- Generate frame for the *start* of this timestep ---
        plotting._plot_simulation_frame(
            t_idx, vessel_path, motor_history, waypoint_list,
            env, planner.known_bathymetry,
            plot_limits, zoom_extents, config.TEMP_FRAMES_DIR
        )

        # --- Calculate movement for this timestep ---
        current_pos = vessel_path[-1]
        v_motor_actual = np.array([0.0, 0.0])

        # --- Get current drift regardless of planner state ---
        v_drift_vec = env.get_drift(current_pos, t_idx)

        # --- Check for imminent collision based on current drift ONLY ---
        predicted_pos_drift_only = (
            current_pos[0] + v_drift_vec[0] * config.DT,
            current_pos[1] + v_drift_vec[1] * config.DT
        )
        imminent_collision = env.is_collision(predicted_pos_drift_only, config.CONTOUR_LEVEL_METERS)

        if imminent_collision and not using_mcts:
            print(f"  -> Predicted drift collision at t={t_idx * config.DT:.1f}s. Activating MCTS.")
            using_mcts = True
            vessel_active = False  # MCTS takes over

        # --- Planning Logic ---
        if using_mcts:
            # Determine targets for MCTS
            target_waypoint = None
            if current_waypoint_target_index < len(waypoint_list):
                target_waypoint = waypoint_list[current_waypoint_target_index]

            next_target_waypoint = None
            if current_waypoint_target_index + 1 < len(waypoint_list):
                next_target_waypoint = waypoint_list[current_waypoint_target_index + 1]

            transect_start_waypoint = None
            if current_waypoint_target_index > 0:
                transect_start_waypoint = waypoint_list[current_waypoint_target_index - 1]
            else:
                transect_start_waypoint = current_pos

            # Call MCTS planner
            v_motor_actual = planner.plan_step(
                current_pos, t_idx,
                target_waypoint, next_target_waypoint, transect_start_waypoint
            )

            # Apply motor speed constraint
            motor_mag = np.linalg.norm(v_motor_actual)
            if motor_mag > config.MAX_MOTOR_SPEED:
                v_motor_actual = v_motor_actual / motor_mag * config.MAX_MOTOR_SPEED

        elif vessel_active:
            # --- Standard Waypoint Following ---
            if current_waypoint_target_index >= len(waypoint_list):
                print("  -> Vessel has completed all waypoints. Switching to drift mode.")
                vessel_active = False
                v_motor_actual = np.array([0.0, 0.0])
            else:
                target_waypoint = waypoint_list[current_waypoint_target_index]
                dist_to_target = np.linalg.norm(np.array(target_waypoint) - np.array(current_pos))

                if dist_to_target < config.WAYPOINT_REACH_THRESHOLD:
                    print(f"  -> Vessel reached waypoint {current_waypoint_target_index} at t={t_idx * config.DT:.1f}s")
                    current_waypoint_target_index += 1
                    # Check again if mission is complete
                    if current_waypoint_target_index >= len(waypoint_list):
                        print("  -> Vessel has completed all waypoints. Switching to drift mode.")
                        vessel_active = False
                        v_motor_actual = np.array([0.0, 0.0])

            # If still active, calculate motor commands
            # If still active, calculate motor commands
            if vessel_active:
                target_waypoint = waypoint_list[current_waypoint_target_index]
                vec_to_target = np.array(target_waypoint) - np.array(current_pos)
                vec_to_target_norm = vec_to_target / (np.linalg.norm(vec_to_target) + 1e-6)

                v_target = vec_to_target_norm * config.VESSEL_SPEED
                v_motor_required = v_target - v_drift_vec
                required_speed_mag = np.linalg.norm(v_motor_required)

                # Calculate the *intended* motor command
                if required_speed_mag > config.MAX_MOTOR_SPEED:
                    v_motor_intended = v_motor_required / required_speed_mag * config.MAX_MOTOR_SPEED
                else:
                    v_motor_intended = v_motor_required

                # --- [NEW SAFETY CHECK] ---
                # Check if this intended action is safe before committing
                v_total_intended = v_drift_vec + v_motor_intended
                intended_pos = (
                    current_pos[0] + v_total_intended[0] * config.DT,
                    current_pos[1] + v_total_intended[1] * config.DT
                )

                if env.is_collision(intended_pos, config.CONTOUR_LEVEL_METERS):
                    # Waypoint follower is about to crash. Activate MCTS
                    # and command a "stop" (drift-only) for this frame.
                    print(
                        f"  -> Waypoint plan predicts collision at t={t_idx * config.DT:.1f}s. Activating MCTS.")
                    using_mcts = True
                    vessel_active = False
                    v_motor_actual = np.array([0.0, 0.0])  # Override: Stop
                else:
                    # Action is safe, commit to it
                    v_motor_actual = v_motor_intended

        else:
            # Drifting
            v_motor_actual = np.array([0.0, 0.0])

        # --- Physics Update and Collision Check ---
        v_actual = v_drift_vec + v_motor_actual
        new_x = current_pos[0] + v_actual[0] * config.DT
        new_y = current_pos[1] + v_actual[1] * config.DT
        new_pos = (new_x, new_y)

        # Check for collision at the *new* position
        if env.is_collision(new_pos, config.CONTOUR_LEVEL_METERS):
            print(f"  -> Vessel action leads to collision at t={t_idx * config.DT:.1f}s.")

            if not using_mcts:
                # First collision: Waypoint plan failed. Activate MCTS and *stay put*.
                print("     Waypoint plan failed. Activating MCTS...")
                using_mcts = True
                vessel_active = False

                motor_history.append(tuple(v_motor_actual))  # Record the failed command
                vessel_path.append(current_pos)  # Stay at the *previous* safe position
                planner.update_known_bathymetry(current_pos)  # Update bathy here
                continue  # Go to next iteration, where MCTS will be active

            else:
                # MCTS itself failed. Stop the simulation.
                print("     MCTS action still resulted in collision. Stopping.")
                print("     MCTS action still resulted in collision. Stopping.")

                # --- [NEW DEBUGGING] ---
                new_pos_ix, new_pos_iy = int(np.round(new_pos[0])), int(np.round(new_pos[1]))
                start_pos_ix, start_pos_iy = int(np.round(current_pos[0])), int(np.round(current_pos[1]))
                actual_depth = env.get_depth(new_pos)

                print("     --- MCTS FAILURE ANALYSIS ---")
                print(
                    f"     Start Pos (at t={t_idx * config.DT:.1f}s): {current_pos[0]:.2f}, {current_pos[1]:.2f} (Cell: {start_pos_ix}, {start_pos_iy})")
                print(f"     MCTS Command:  Motor=({v_motor_actual[0]:.2f}, {v_motor_actual[1]:.2f})")
                print(f"     Environment:   Drift=({v_drift_vec[0]:.2f}, {v_drift_vec[1]:.2f})")
                print(f"     Resulting Vec: Total=({v_actual[0]:.2f}, {v_actual[1]:.2f})")
                print(
                    f"     End Pos (at t={(t_idx + 1) * config.DT:.1f}s):   {new_pos[0]:.2f}, {new_pos[1]:.2f} (Cell: {new_pos_ix}, {new_pos_iy})")
                print(f"     Collision Check: Actual Depth at ({new_pos_ix}, {new_pos_iy}) is {actual_depth:.2f}m")
                print(f"     Collision Level: {config.CONTOUR_LEVEL_METERS:.2f}m")
                print(
                    f"     Is {actual_depth:.2f}m >= {config.CONTOUR_LEVEL_METERS:.2f}m? -> {actual_depth >= config.CONTOUR_LEVEL_METERS}")
                print("     -----------------------------")
                # --- [END DEBUGGING] ---

                v_motor_actual = np.array([0.0, 0.0])  # Command stop
                motor_history.append(tuple(v_motor_actual))
                vessel_path.append(current_pos)  # Append final safe position

                # Plot final frame
                plotting._plot_simulation_frame(
                    t_idx + 1, vessel_path, motor_history, waypoint_list,
                    env, planner.known_bathymetry,
                    plot_limits, zoom_extents, config.TEMP_FRAMES_DIR
                )
                break  # Exit loop

        # If safe, append new state
        vessel_path.append(new_pos)
        motor_history.append(tuple(v_motor_actual))
        planner.update_known_bathymetry(new_pos)

        # If this was the last timestep, plot the final frame
        if t_idx == env.total_timesteps - 1:
            print("Reached end of simulation time.")
            plotting._plot_simulation_frame(
                t_idx + 1, vessel_path, motor_history, waypoint_list,
                env, planner.known_bathymetry,
                plot_limits, zoom_extents, config.TEMP_FRAMES_DIR
            )

    print(f"Path simulation complete. Total steps simulated: {len(vessel_path)}")