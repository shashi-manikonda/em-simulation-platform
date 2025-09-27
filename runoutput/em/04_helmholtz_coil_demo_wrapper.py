
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

# Change working directory to output_dir so all files/plots are created there
os.makedirs(r"/app/runoutput/em", exist_ok=True)
os.chdir(r"/app/runoutput/em")

_figure_counter = 1
def save_figure_as_png(*args, **kwargs):
    global _figure_counter
    fig = plt.gcf()
    if fig.get_axes():
        filename = f"04_helmholtz_coil_demo_fig{_figure_counter:03d}.png"
        fig.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved plot as: {filename}")
        _figure_counter += 1
    plt.clf()

plt.show = save_figure_as_png
original_fig_show = matplotlib.figure.Figure.show
def fig_show_override(self, *args, **kwargs):
    save_figure_as_png()
matplotlib.figure.Figure.show = fig_show_override

try:
    with open(r"/app/demos/em/04_helmholtz_coil_demo.py") as f:
        exec(f.read())
except Exception as e:
    print(f"Error in demo execution: {e}", file=sys.stderr)
    raise

if plt.get_fignums():
    save_figure_as_png()
