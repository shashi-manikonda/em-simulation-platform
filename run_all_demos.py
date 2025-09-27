import os
import platform
import subprocess
import time
from datetime import datetime


def run_demos():
    """
    Finds and runs all .py and .ipynb demos in a predefined order.
    For .ipynb files, it converts them to .py using jupytext first.
    Saves text output and PNG files to 'demos/runoutput' folder.
    Shows execution status and timing on screen.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(project_root, "src")
    demos_dir = os.path.join(project_root, "demos")

    # --- Demo Files - Automatically Generated ---
    em_demo_dir = os.path.join(demos_dir, "em")
    demo_files = []
    for fname in sorted(os.listdir(em_demo_dir)):
        if fname.endswith(".py") or fname.endswith(".ipynb"):
            demo_files.append(f"em/{fname}")

    # Create runoutput directory
    runoutput_dir = os.path.join(project_root, "runoutput")
    os.makedirs(runoutput_dir, exist_ok=True)

    print(f"Demo Runner Started - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output will be saved to: {runoutput_dir}")
    print(f"Source path: {src_path}")
    print("=" * 60)

    python_executable = "python" if platform.system() == "Windows" else "python3"
    total_demos = len(demo_files)
    successful_demos = 0
    failed_demos = 0
    total_start_time = time.time()

    for i, file_rel_path in enumerate(demo_files):
        file_path = os.path.join(demos_dir, file_rel_path)
        file = os.path.basename(file_path)

        if not os.path.exists(file_path):
            print(f"\n[{i+1}/{total_demos}] Skipping: {file_rel_path} (File not found)")
            failed_demos += 1
            continue

        script_path = None
        wrapper_script = None
        demo_start_time = time.time()

        # Create demo-specific output directory
        demo_name = os.path.splitext(file)[0]
        demo_subdir = os.path.dirname(file_rel_path)
        demo_output_dir = os.path.join(runoutput_dir, demo_subdir)
        os.makedirs(demo_output_dir, exist_ok=True)

        print(f"\n[{i+1}/{total_demos}] Running: {file_rel_path}")
        print(f"    Output dir: {demo_output_dir}")

        try:
            if file.endswith(".py"):
                script_to_run = file_path
            elif file.endswith(".ipynb"):
                print("    Converting notebook...")
                subprocess.run(
                    ["jupytext", "--to", "py", file_path],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                script_path = os.path.splitext(file_path)[0] + ".py"
                script_to_run = script_path

            wrapper_script = create_matplotlib_wrapper(
                script_to_run, demo_output_dir, demo_name
            )

            env = os.environ.copy()
            env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")

            print("    Executing (max 5min timeout)...")
            result = subprocess.run(
                [python_executable, wrapper_script],
                capture_output=True,
                text=True,
                env=env,
                timeout=300,
            )

            demo_duration = time.time() - demo_start_time

            output_file = os.path.join(demo_output_dir, f"{demo_name}_output.txt")
            with open(output_file, "w") as f:
                f.write(f"Demo: {file_rel_path}\n")
                f.write(f"Execution Time: {demo_duration:.2f} seconds\n")
                f.write(f"Return Code: {result.returncode}\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\nSTDOUT:\n")
                f.write(result.stdout)
                if result.stderr:
                    f.write("\n" + "=" * 50 + "\nSTDERR:\n")
                    f.write(result.stderr)

            move_temp_files_to_output(
                script_path, wrapper_script, demo_output_dir, demo_name
            )

            if result.returncode == 0:
                successful_demos += 1
                print(f"    ✓ SUCCESS - {demo_duration:.2f}s")
                if result.stdout.strip():
                    stdout_lines = result.stdout.strip().split("\n")
                    for line in stdout_lines[:3]:
                        if line.strip():
                            print(f"      {line}")
                    if len(stdout_lines) > 3:
                        print(f"      ... ({len(stdout_lines) - 3} more lines)")
            else:
                failed_demos += 1
                print(f"    ✗ FAILED - {demo_duration:.2f}s (code: {result.returncode})")
                if result.stderr:
                    error_lines = result.stderr.strip().split("\n")
                    print(f"      Error: {error_lines[-1] if error_lines else 'Unknown'}")

        except subprocess.TimeoutExpired:
            demo_duration = time.time() - demo_start_time
            failed_demos += 1
            print(f"    ✗ TIMEOUT - {demo_duration:.2f}s")
            # Handle timeout file logging
        except FileNotFoundError as e:
            demo_duration = time.time() - demo_start_time
            failed_demos += 1
            error_msg = "jupytext not found" if "jupytext" in str(e) else str(e)
            print(f"    ✗ FAILED - {demo_duration:.2f}s, Error: {error_msg}")
            if "jupytext" in str(e):
                print("      Install with: pip install jupytext")
                break
        except Exception as e:
            demo_duration = time.time() - demo_start_time
            failed_demos += 1
            print(f"    ✗ FAILED - {demo_duration:.2f}s, Error: {e}")

    total_duration = time.time() - total_start_time
    print("\n" + "=" * 60 + "\nEXECUTION SUMMARY\n" + "=" * 60)
    print(f"Total Demos: {total_demos}")
    print(f"Successful: {successful_demos} ✓")
    print(f"Failed: {failed_demos} ✗")
    print(f"Success Rate: {(successful_demos / total_demos * 100):.1f}%" if total_demos > 0 else "N/A")
    print(f"Total Time: {total_duration:.2f} seconds")


def move_temp_files_to_output(script_path, wrapper_script, demo_output_dir, demo_name):
    """Moves temporary files to the demo output directory."""
    try:
        if script_path and os.path.exists(script_path):
            dest = os.path.join(demo_output_dir, f"{demo_name}_converted.py")
            os.rename(script_path, dest)
        if wrapper_script and os.path.exists(wrapper_script):
            dest = os.path.join(demo_output_dir, f"{demo_name}_wrapper.py")
            os.rename(wrapper_script, dest)
    except OSError as e:
        print(f"    Warning: Could not move temp files: {e}")


def create_matplotlib_wrapper(original_script, output_dir, demo_name):
    """Creates a wrapper to save matplotlib plots."""
    wrapper_content = f'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

# Change working directory to output_dir so all files/plots are created there
os.makedirs(r"{output_dir}", exist_ok=True)
os.chdir(r"{output_dir}")

_figure_counter = 1
def save_figure_as_png(*args, **kwargs):
    global _figure_counter
    fig = plt.gcf()
    if fig.get_axes():
        filename = f"{demo_name}_fig{{_figure_counter:03d}}.png"
        fig.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved plot as: {{filename}}")
        _figure_counter += 1
    plt.clf()

plt.show = save_figure_as_png
original_fig_show = matplotlib.figure.Figure.show
def fig_show_override(self, *args, **kwargs):
    save_figure_as_png()
matplotlib.figure.Figure.show = fig_show_override

try:
    with open(r"{original_script}") as f:
        exec(f.read())
except Exception as e:
    print(f"Error in demo execution: {{e}}", file=sys.stderr)
    raise

if plt.get_fignums():
    save_figure_as_png()
'''
    wrapper_path = original_script + "_wrapper_temp.py"
    with open(wrapper_path, "w") as f:
        f.write(wrapper_content)
    return wrapper_path


if __name__ == "__main__":
    run_demos()