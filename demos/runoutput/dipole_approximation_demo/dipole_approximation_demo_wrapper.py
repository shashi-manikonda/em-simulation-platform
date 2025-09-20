
import matplotlib
import matplotlib.pyplot as plt
import os
import sys

# Configure matplotlib for non-interactive backend
matplotlib.use('Agg')

# Global figure counter for sequential naming
_figure_counter = 1

# Override plt.show() to save figures instead
def save_figure_as_png(*args, **kwargs):
    global _figure_counter

    # Get current figure
    fig = plt.gcf()

    # Only save if figure has content
    if fig.get_axes():
        # Create filename
        filename = f"dipole_approximation_demo_fig{_figure_counter:03d}.png"
        filepath = os.path.join("/app/demos/runoutput/dipole_approximation_demo", filename)

        # Save the figure
        fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved plot as: {filename}")

        _figure_counter += 1

    # Clear the figure to free memory
    plt.clf()

# Replace plt.show with our custom function
plt.show = save_figure_as_png

# Also handle the case where show() is called on figure objects
original_fig_show = matplotlib.figure.Figure.show
def fig_show_override(self, *args, **kwargs):
    save_figure_as_png()

matplotlib.figure.Figure.show = fig_show_override

# Execute the original script
try:
    exec(open(r"/app/demos/em/dipole_approximation_demo.py").read())
except Exception as e:
    print(f"Error in demo execution: {e}", file=sys.stderr)
    raise

# Save any remaining figures that weren't explicitly shown
if plt.get_fignums():  # Check if there are any figures
    save_figure_as_png()
