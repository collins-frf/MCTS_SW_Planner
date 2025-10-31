# vessel.py
"""
Defines the Vessel class, which tracks its own state (position, history)
and handles its own physics update.
"""

import numpy as np


class Vessel:
    """
    Represents the vessel. Tracks its state (position) and history.
    """

    def __init__(self, start_position):
        self.position = tuple(start_position)
        self.path_history = [self.position]
        self.motor_history = [(0.0, 0.0)]  # Start with zero motor command

    def move_step(self, motor_command, drift_vector, dt):
        """
        Updates the vessel's position based on motor commands,
        environmental drift, and timestep.

        Args:
            motor_command (np.array): (u, v) motor velocity
            drift_vector (np.array): (u, v) water velocity
            dt (float): Timestep
        """
        v_actual = motor_command + drift_vector
        new_x = self.position[0] + v_actual[0] * dt
        new_y = self.position[1] + v_actual[1] * dt

        self.position = (new_x, new_y)

        # Record history
        self.path_history.append(self.position)
        self.motor_history.append(tuple(motor_command))

    def get_current_position(self):
        return self.position

    def get_path_history(self):
        return self.path_history

    def get_motor_history(self):
        return self.motor_history