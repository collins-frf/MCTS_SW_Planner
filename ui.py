# ui.py
"""
Contains UI-related functions, like the Matplotlib polygon selector.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector

def get_survey_polygon_gui(bathymetry, grid_shape):
    """
    Shows a plot of the bathymetry and waits for the user to
    draw a polygon to define the survey area.
    Closes on 'Enter' key press or right-click.
    """
    ny, nx = grid_shape
    print("\n--- Survey Area Selection ---")
    print("Please click the vertices of your survey polygon on the plot.")
    print("Right-click or press 'Enter' when finished.")

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(bathymetry, origin='lower', extent=[0, nx, 0, ny], cmap='terrain')
    ax.set_title('Click to define survey polygon. Press Enter or Right-Click to accept.')

    polygon_vertices = []

    def onselect(verts):
        nonlocal polygon_vertices
        polygon_vertices = verts
        print(f"Selected {len(verts)} vertices.")

    poly_selector = PolygonSelector(ax, onselect, useblit=True)

    def on_key_press(event):
        if event.key == 'enter':
            plt.close(fig)

    def on_button_press(event):
        if event.button == 3: # Right-click
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key_press)
    fig.canvas.mpl_connect('button_press_event', on_button_press)

    plt.show()  # Blocks until closed

    if not polygon_vertices:
        print("No polygon selected.")
        return None

    print("Polygon selection complete.")
    return np.array(polygon_vertices)

def load_or_select_polygon(poly_file_path, bathymetry, grid_shape):
    """
    Loads a saved polygon if it exists, otherwise
    opens the GUI for the user to select one.
    """
    survey_polygon = None
    if os.path.exists(poly_file_path):
        try:
            print(f"Loading existing survey polygon from {poly_file_path}")
            print("  (To re-draw, delete this file and run again.)")
            survey_polygon = np.load(poly_file_path)
            if survey_polygon.size < 3:
                print("  -> Polygon file is invalid. Will re-draw.")
                survey_polygon = None
        except Exception as e:
            print(f"  -> Error loading polygon file: {e}. Will re-draw.")
            survey_polygon = None

    if survey_polygon is None:
        print("Opening GUI selection...")
        survey_polygon = get_survey_polygon_gui(bathymetry, grid_shape)

        if survey_polygon is not None and survey_polygon.size > 0:
            print(f"Saving new survey polygon to {poly_file_path}...")
            np.save(poly_file_path, survey_polygon)
        else:
            print("No polygon selected. Exiting.")
            return None

    return survey_polygon