# planning.py (continuation)
"""
Contains all planning logic:
- RouteGenerator: For creating the initial lawnmower path.
- MCTSNode: Data structure for the MCTS tree.
- MCTSPlanner: The real-time, reactive MCTS planner for avoidance.
"""

import numpy as np
from matplotlib.path import Path
from scipy.stats import linregress
import matplotlib.pyplot as plt
import random
import math
import config  # Import config for MCTS settings


class RouteGenerator:
    """
    Utility class for generating the initial survey route (lawnmower path).
    Methods are static as they don't require internal state.
    """

    @staticmethod
    def find_shoreline_orientation(bathymetry_full, grid_shape, contour_level, polygon_vertices):
        """
        Performs linear regression on contour points inside the polygon
        to find the dominant shoreline angle.
        """
        ny, nx = grid_shape
        print("Calculating local shoreline orientation...")

        # 1. Find the specified contour
        x = np.arange(nx)
        y = np.arange(ny)
        fig_temp, ax_temp = plt.subplots()
        cs = ax_temp.contour(x, y, bathymetry_full, levels=[contour_level])
        plt.close(fig_temp)  # Close the temp figure

        shoreline_points = []
        if not cs.allsegs[0]:
            print(f"Warning: No {contour_level}m shoreline found. Defaulting to 0-degree (E-W) normal.")
            return 0.0

        # 2. Check which contour points are inside the user's polygon
        polygon_path = Path(polygon_vertices)
        for path in cs.allsegs[0]:
            if len(path) > 0:
                inside = polygon_path.contains_points(path)
                shoreline_points.extend(path[inside])

        if len(shoreline_points) < 2:
            print(
                f"Warning: No {contour_level}m shoreline found *inside* polygon. Defaulting to 0-degree (E-W) normal.")
            return 0.0

        # 3. Perform linear regression
        shore_pts = np.array(shoreline_points)
        x_shore, y_shore = shore_pts[:, 0], shore_pts[:, 1]

        along_shore_angle = 0.0
        if (np.max(x_shore) - np.min(x_shore)) > (np.max(y_shore) - np.min(y_shore)):
            # More E-W
            slope, _, _, _, _ = linregress(x_shore, y_shore)
            along_shore_angle = np.arctan(slope)
        else:
            # More N-S
            slope, _, _, _, _ = linregress(y_shore, x_shore)
            along_shore_angle = np.pi / 2.0 if slope == 0 else np.arctan2(1.0, slope)

        # 4. Calculate the shore-normal angle
        shore_normal_angle = (along_shore_angle + np.pi / 2.0) % (2 * np.pi)

        print(f"  -> Local shoreline (along-shore) angle: {np.rad2deg(along_shore_angle):.1f} deg")
        print(f"  -> Calculated shore-normal angle: {np.rad2deg(shore_normal_angle):.1f} deg")
        return shore_normal_angle

    @staticmethod
    def generate_rotated_waypoints(bathymetry_full, grid_shape, contour_level,
                                   polygon_vertices, shore_normal_angle, transect_spacing):
        """
        Generates a lawnmower waypoint path, rotated to be normal
        to the shoreline, and starting from the offshore side.
        """
        ny, nx = grid_shape
        print("Generating rotated lawnmower waypoints...")

        # 1. Rotate polygon vertices
        along_shore_angle = shore_normal_angle - np.pi / 2.0
        rotation_matrix = np.array([
            [np.cos(-along_shore_angle), -np.sin(-along_shore_angle)],
            [np.sin(-along_shore_angle), np.cos(-along_shore_angle)]
        ])
        center = np.mean(polygon_vertices, axis=0)
        verts_centered = polygon_vertices - center
        verts_rotated = verts_centered @ rotation_matrix.T

        # 2. Get bounding box in ROTATED frame
        x_min_rot, y_min_rot = np.min(verts_rotated, axis=0)
        x_max_rot, y_max_rot = np.max(verts_rotated, axis=0)

        # 3. Generate waypoints in ROTATED frame
        waypoints_rotated_raw = []
        x_positions = np.arange(x_min_rot, x_max_rot + transect_spacing, transect_spacing)
        for i, x in enumerate(x_positions):
            if i % 2 == 0:  # Go in positive Y-dir
                waypoints_rotated_raw.append([x, y_min_rot])
                waypoints_rotated_raw.append([x, y_max_rot])
            else:  # Go in negative Y-dir
                waypoints_rotated_raw.append([x, y_max_rot])
                waypoints_rotated_raw.append([x, y_min_rot])

        # 4. Rotate waypoints back
        unrotation_matrix = np.array([
            [np.cos(along_shore_angle), -np.sin(along_shore_angle)],
            [np.sin(along_shore_angle), np.cos(along_shore_angle)]
        ])
        waypoints_unrotated_centered = np.array(waypoints_rotated_raw) @ unrotation_matrix.T
        waypoints_final = list(waypoints_unrotated_centered + center)

        if len(waypoints_final) < 2:
            print("Warning: Fewer than 2 waypoints generated.")
            return []

        # 5. Determine Offshore Side
        mid_x_rot = (x_min_rot + x_max_rot) / 2.0
        offshore_check_point_min_y_rot = np.array([mid_x_rot, y_min_rot]) @ unrotation_matrix.T + center
        offshore_check_point_max_y_rot = np.array([mid_x_rot, y_max_rot]) @ unrotation_matrix.T + center
        offshore_side_is_ymax = False
        try:
            # Helper to get bathy at a point, handling OOB
            def get_bathy(point):
                ix, iy = int(np.round(point[0])), int(np.round(point[1]))
                if 0 <= ix < nx and 0 <= iy < ny:
                    return bathymetry_full[iy, ix]
                return np.nan  # OOB

            bathy_min_y = get_bathy(offshore_check_point_min_y_rot)
            bathy_max_y = get_bathy(offshore_check_point_max_y_rot)

            # Logic to find the deeper (more negative) side
            if np.isnan(bathy_min_y) and np.isnan(bathy_max_y):
                offshore_side_is_ymax = True
            elif np.isnan(bathy_min_y):
                offshore_side_is_ymax = True
            elif np.isnan(bathy_max_y):
                offshore_side_is_ymax = False
            elif bathy_min_y < contour_level and bathy_max_y < contour_level:  # Both water
                offshore_side_is_ymax = bathy_max_y < bathy_min_y  # True if max_y is deeper
            elif bathy_min_y >= contour_level:  # min_y is land
                offshore_side_is_ymax = True
            elif bathy_max_y >= contour_level:  # max_y is land
                offshore_side_is_ymax = False
        except Exception:
            offshore_side_is_ymax = True  # Default

        print(f"...Offshore side determined to be {'Max Y (rotated)' if offshore_side_is_ymax else 'Min Y (rotated)'}")

        # 6. Check and Reverse if needed
        first_point_rotated = (np.array(waypoints_final[0]) - center) @ rotation_matrix.T
        y_rot_start = first_point_rotated[1]
        start_is_near_min = abs(y_rot_start - y_min_rot) < abs(y_rot_start - y_max_rot)
        if (offshore_side_is_ymax and start_is_near_min) or \
                (not offshore_side_is_ymax and not start_is_near_min):
            print("  -> First waypoint not offshore. Reversing list.")
            waypoints_final.reverse()

        print(f"Generated {len(waypoints_final)} final waypoints.")
        return [tuple(wp) for wp in waypoints_final]


# ==============================================================================
# --- MCTS PLANNER ---
# ==============================================================================

class MCTSNode:
    """ Represents a node in the Monte Carlo Tree Search. """

    def __init__(self, state, parent=None, action=None, max_motor_speed=1.5):
        self.state = state  # (pos_tuple, time_idx)
        self.parent = parent
        self.action = action  # Action that led to this state
        self.children = []
        self.visits = 0
        self.total_reward = 0.0
        self.max_motor_speed = max_motor_speed
        # Call the new static method, passing it the speed
        self.untried_actions = MCTSNode.get_possible_actions(self.max_motor_speed)
        self.is_terminal = False

    @staticmethod
    def get_possible_actions(max_motor_speed):
        actions = []
        step = max_motor_speed
        # 8 directions + stop
        for angle in np.linspace(0, 2 * np.pi * 7 / 8, 8):
            actions.append(np.array([step * np.cos(angle), step * np.sin(angle)]))
        actions.append(np.array([0.0, 0.0]))  # Stop action
        random.shuffle(actions)
        return actions

    def uct_value(self, exploration_constant):
        if self.visits == 0:
            return float('inf')
        parent_visits = self.parent.visits if self.parent and self.parent.visits > 0 else 1
        avg_reward = self.total_reward / self.visits
        exploration_term = exploration_constant * math.sqrt(math.log(parent_visits) / self.visits)
        return avg_reward + exploration_term

    def add_child(self, child_state, action):
        child_node = MCTSNode(child_state, parent=self, action=action, max_motor_speed=self.max_motor_speed)
        self.children.append(child_node)
        self.untried_actions = [a for a in self.untried_actions if not np.array_equal(a, action)]
        return child_node

    def update(self, reward):
        self.visits += 1
        self.total_reward += reward

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def select_best_child(self, exploration_constant):
        best_score = -float('inf')
        best_child = None
        for child in self.children:
            score = child.uct_value(exploration_constant)
            score += random.uniform(0, 1e-6)  # Tie-breaker
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def __repr__(self):
        pos_str = f"({self.state[0][0]:.1f},{self.state[0][1]:.1f})"
        return f"State({pos_str}, t={self.state[1]}), V={self.visits}, Q={self.total_reward / self.visits if self.visits > 0 else 0:.2f}"


class MCTSPlanner:
    """
    Handles real-time motion planning using MCTS to avoid obstacles
    and seek targets.
    """

    def __init__(self, environment, survey_polygon_vertices):
        self.env = environment  # Reference to the Environment object
        self.survey_path = Path(survey_polygon_vertices)

        # Planner's "fog of war"
        self.known_bathymetry = {}  # Dictionary: {(ix, iy): depth}

        self.grid_shape = self.env.grid_shape
        self.ny, self.nx = self.grid_shape

        print("MCTSPlanner Initialized.")

    def update_known_bathymetry(self, pos):
        """ Called by the main simulation loop to update known bathy. """
        px, py = pos
        ix, iy = int(np.round(px)), int(np.round(py))
        grid_cell = (ix, iy)
        if 0 <= ix < self.nx and 0 <= iy < self.ny:
            if grid_cell not in self.known_bathymetry:
                # Query the true environment for the depth
                depth = self.env.get_depth(pos)
                self.known_bathymetry[grid_cell] = depth
                # print(f"  -> MCTSPlanner revealed bathy at ({ix}, {iy}): {depth:.2f}m")

    def plan_step(self, current_pos, current_time_idx, target_waypoint, next_target_waypoint, transect_start_waypoint):
        """
        Performs MCTS search to find the best motor command.
        """
        # [DEBUG] Removed the verbose print from original planner.py
        # print(f"  -> MCTS Planner called at t={current_time_idx * config.DT:.1f}s")

        initial_state_tuple = (tuple(current_pos), current_time_idx)
        root_node = MCTSNode(state=initial_state_tuple, max_motor_speed=config.MAX_MOTOR_SPEED)

        # Pre-calculate target positions
        target_pos = np.array(target_waypoint) if target_waypoint else None
        next_target_pos = np.array(next_target_waypoint) if next_target_waypoint else None
        transect_start_pos = np.array(transect_start_waypoint) if transect_start_waypoint else None

        for _ in range(config.MCTS_ITERATIONS):

            # --- 1. Selection ---
            node = root_node
            while not node.is_terminal:
                if not node.is_fully_expanded():
                    break
                if not node.children:
                    break  # Reached a leaf
                node = node.select_best_child(config.MCTS_EXPLORATION_CONSTANT)
                if node is None:
                    node = root_node  # Failsafe
                    break

            # --- 2. Expansion ---
            if not node.is_terminal and node.untried_actions:
                action_to_try = node.untried_actions[0]
                next_state_tuple, _, is_terminal_next = self._simulate_physics_step(node.state, action_to_try)
                node = node.add_child(next_state_tuple, action_to_try)
                node.is_terminal = is_terminal_next

            # --- 3. Simulation (Rollout) ---
                # --- 3. Simulation (Rollout) ---
                total_rollout_reward = 0

                if node.is_terminal:
                    # The node we just expanded *is* a terminal state (an immediate crash).
                    # Its reward is simply the terminal penalty. No rollout is needed.
                    total_rollout_reward = self._calculate_terminal_penalty(node.state, True)
                else:
                    # The node is safe. Proceed with a full, deep simulation (rollout).
                    rollout_state = node.state
                    visited_in_rollout = set()

                    for _ in range(config.MCTS_SIMULATION_DEPTH):
                        # Get action from default policy
                        action = self._get_rollout_action(rollout_state, target_pos)

                        # Simulate one step
                        prev_rollout_state = rollout_state
                        rollout_state, terminal_penalty, is_terminal_step = self._simulate_physics_step(
                            prev_rollout_state, action)

                        # Calculate and accumulate reward for this single step
                        reward = self._calculate_rollout_reward(
                            prev_rollout_state, rollout_state, is_terminal_step, terminal_penalty,
                            target_pos, next_target_pos, transect_start_pos,
                            visited_in_rollout
                        )
                        total_rollout_reward += reward

                        if is_terminal_step:
                            # The simulation hit a dead end, stop this rollout
                            break

            # --- 4. Backpropagation ---
            temp_node = node
            while temp_node is not None:
                temp_node.update(total_rollout_reward)  # <<<--- THIS IS WHERE IT FROZE
                temp_node = temp_node.parent

        # --- Action Selection ---
        # Choose the action from the root's children with the highest visit count
        best_child = None
        most_visits = -1

        if not root_node.children:
            print("Warning: MCTS root node has no children after iterations. Returning stop action.")
            return np.array([0.0, 0.0])

        print("     --- MCTS FINAL ACTION SCORES ---")
        for child in root_node.children:
            # --- [NEW DEBUG PRINT] ---
            # This shows the "score" for each of the 9 possible first-level actions
            q_avg = child.total_reward / child.visits if child.visits > 0 else 0
            print(f"     -> Action: {str(child.action):<18} Visits: {child.visits:<5} Avg Reward (Q): {q_avg:<8.2f}")
            # --- [END DEBUG PRINT] ---

            if child.visits > most_visits:
                most_visits = child.visits
                best_child = child
        print("     ------------------------------------")

        if best_child is None:
            print("Warning: MCTS failed to select a best child. Returning stop action.")
            return np.array([0.0, 0.0])

        # print(
        #     f"  -> MCTS selected action: {best_child.action} (Visits: {most_visits}, Avg Reward: {best_child.total_reward / best_child.visits if best_child.visits > 0 else 0:.2f})")
        return best_child.action

    # --- MCTS Helper Methods ---

    def _simulate_physics_step(self, state_tuple, motor_action):
        """
        Simulates one physics timestep forward.
        Returns: (next_state_tuple, terminal_penalty, is_terminal)
        """
        current_pos, current_t_idx = state_tuple

        if current_t_idx + 1 >= self.env.total_timesteps:
            penalty = self._calculate_terminal_penalty(state_tuple, True)
            return state_tuple, penalty, True  # Terminal: out of time

        next_t_idx = current_t_idx + 1

        # 1. Get drift from environment
        v_drift_vec = self.env.get_drift(current_pos, next_t_idx)

        # 2. Apply motor command
        v_actual = v_drift_vec + motor_action

        # 3. Calculate next position
        new_x = current_pos[0] + v_actual[0] * config.DT
        new_y = current_pos[1] + v_actual[1] * config.DT
        next_pos = (new_x, new_y)
        next_state = (next_pos, next_t_idx)

        # 4. Check for terminal state (collision or out of bounds)
        is_terminal = self.env.is_collision(next_pos, config.CONTOUR_LEVEL_METERS)

        # 5. Calculate *only* the terminal penalty
        terminal_penalty = self._calculate_terminal_penalty(next_state, is_terminal)

        return next_state, terminal_penalty, is_terminal

    def _calculate_terminal_penalty(self, state_tuple, is_terminal):
        """ Calculates the large negative reward for a terminal state. """
        if not is_terminal:
            return 0.0

        pos_x, pos_y = state_tuple[0]
        ix, iy = int(np.round(pos_x)), int(np.round(pos_y))

        # Check if it was Out-of-Bounds vs. Land Collision
        if not (0 <= ix < self.nx and 0 <= iy < self.ny):
            return -1000.0  # Heavier OOB penalty
        else:
            return -100.0  # Land collision penalty

    def _get_rollout_action(self, rollout_state, target_pos):
        """
        Selects an action for the rollout (simulation) phase.

        NEW POLICY:
        Performs a 1-step lookahead for all 9 actions.
        - If an action causes an immediate collision, it's given a
          massive penalty (-1000.0).
        - If it's safe, it's scored based on progress to the target.
        This correctly allows moving "uphill" as long as it's not a collision.
        """
        possible_actions = MCTSNode.get_possible_actions(config.MAX_MOTOR_SPEED)
        action_scores = []
        current_pos_arr = np.array(rollout_state[0])

        for action in possible_actions:

            # --- 1. Simulate one step ahead ---
            # We use _simulate_physics_step which uses the TRUE environment
            # to check for collisions.
            _next_state, _terminal_penalty, is_terminal_step = \
                self._simulate_physics_step(rollout_state, action)

            # --- 2. Score the action ---
            if is_terminal_step:
                # This action is an immediate crash. Give it a terrible score.
                action_scores.append(-1000.0)
            else:
                # This action is SAFE.
                # Now, score it based on how well it moves to the target.
                score = 0.0
                action_norm = np.linalg.norm(action)

                if target_pos is not None:
                    vec_to_target = target_pos - current_pos_arr
                    dist_to_target = np.linalg.norm(vec_to_target)

                    if dist_to_target > 1e-5:
                        norm_vec_to_target = vec_to_target / dist_to_target

                        if action_norm < 1e-5:  # 'Stop' action
                            # Stop action is neutral, its score is 0
                            pass
                        else:
                            norm_action_vec = action / action_norm
                            # cos_theta is 1 if aligned, -1 if opposite
                            cos_theta = np.dot(norm_action_vec, norm_vec_to_target)
                            score = cos_theta  # Score is [-1, 1]

                action_scores.append(score * config.W_TARGET_POLICY)

        # --- 3. Choose the best action ---
        # We add a little randomness (epsilon-greedy)
        # to prevent the rollout from being *too* deterministic.
        if random.random() < 0.10:  # 10% chance of random action
            best_action = random.choice(possible_actions)
        else:
            # Find the best action(s)
            best_score = max(action_scores)
            best_indices = [i for i, score in enumerate(action_scores) if score >= best_score - 1e-5]

            if not best_indices:  # Failsafe
                best_action = random.choice(possible_actions)
            else:
                # Pick one randomly from the best
                best_action = possible_actions[random.choice(best_indices)]

        return best_action

    def _calculate_rollout_reward(self, prev_rollout_state, rollout_state,
                                  is_terminal_step, terminal_penalty,
                                  target_pos, next_target_pos, transect_start_pos,
                                  visited_in_rollout):
        """ Calculates the full reward for a single step in a rollout. """

        # 1. Terminal Penalty (already calculated)
        total_reward = terminal_penalty

        if is_terminal_step:
            return total_reward  # No other rewards if terminal

        # 2. Exploration Bonus
        current_ix, current_iy = int(np.round(rollout_state[0][0])), int(np.round(rollout_state[0][1]))
        grid_cell = (current_ix, current_iy)

        if grid_cell not in self.known_bathymetry and grid_cell not in visited_in_rollout:
            if 0 <= current_ix < self.nx and 0 <= current_iy < self.ny:
                total_reward += config.W_EXPLORE
                visited_in_rollout.add(grid_cell)

        # 3. Progress Reward
        # A) Primary Target Reward
        if target_pos is not None:
            dist_prev = np.linalg.norm(np.array(prev_rollout_state[0]) - target_pos)
            dist_new = np.linalg.norm(np.array(rollout_state[0]) - target_pos)
            total_reward += (dist_prev - dist_new) * config.W_PROGRESS_PRIMARY

        # B) Secondary Target Reward
        if next_target_pos is not None:
            dist_prev = np.linalg.norm(np.array(prev_rollout_state[0]) - next_target_pos)
            dist_new = np.linalg.norm(np.array(rollout_state[0]) - next_target_pos)
            total_reward += (dist_prev - dist_new) * config.W_PROGRESS_SECONDARY

        # 4. Polygon Penalty
        (current_x, current_y) = rollout_state[0]
        is_currently_outside = not self.survey_path.contains_point((current_x, current_y))

        if is_currently_outside:
            # We are outside the box. Are we trying to get back in?
            is_target_inside = False
            if target_pos is not None:
                # Check if the primary target is inside the polygon
                is_target_inside = self.survey_path.contains_point(target_pos)

            if is_target_inside:
                # "AMNESTY" RULE:
                # We are OUT, but the target is IN.
                # DO NOT apply the penalty. The W_PROGRESS_PRIMARY reward
                # (which is positive for moving toward the target)
                # will strongly encourage the planner to go back in.
                pass
            else:
                # We are OUT, and the target is also OUT (or non-existent).
                # This is "flying off." Apply the full penalty.
                total_reward -= config.W_OUTSIDE_POLY_PENALTY

        # 5. Cross-Track Error Penalty
        if target_pos is not None and transect_start_pos is not None:
            line_vec = target_pos - transect_start_pos
            line_mag_sq = np.dot(line_vec, line_vec)

            if line_mag_sq > 1e-4:  # Avoid division by zero
                pos_vec = np.array(rollout_state[0]) - transect_start_pos
                projection_scalar = np.dot(pos_vec, line_vec) / line_mag_sq
                projection_scalar_clamped = max(0, min(1, projection_scalar))
                closest_point = transect_start_pos + projection_scalar_clamped * line_vec
                cross_track_error = np.linalg.norm(np.array(rollout_state[0]) - closest_point)
                total_reward -= (cross_track_error * config.W_CROSS_TRACK_PENALTY)

        return total_reward

    def _estimate_local_gradient(self, pos):
        """
        Estimates bathymetry gradient based on TRUE environment data.
        Returns (grad_x, grad_y) or None.
        """
        px, py = pos
        ix, iy = int(np.round(px)), int(np.round(py))

        # Helper to get true depth from the environment.
        # self.env.get_depth() is "all-knowing" and returns
        # a large positive value (land) for out-of-bounds queries.
        def get_true_depth(cell_x, cell_y):
            return self.env.get_depth((cell_x, cell_y))

        # Get depths of neighbors from the TRUE map
        depth_east = get_true_depth(ix + 1, iy)
        depth_west = get_true_depth(ix - 1, iy)
        depth_north = get_true_depth(ix, iy + 1)
        depth_south = get_true_depth(ix, iy - 1)

        # Use central difference (change in depth / change in x)
        # (depth_east is 2 cells away from depth_west)
        grad_x = (depth_east - depth_west) / 2.0

        # Use central difference (change in depth / change in y)
        grad_y = (depth_north - depth_south) / 2.0

        if grad_x == 0.0 and grad_y == 0.0:
            return None

        # This is the gradient vector. A positive value means depth is
        # increasing (e.g., from -5 to -2), which is "uphill"/shallower.
        return np.array([grad_x, grad_y])