import os
import sys
import subprocess
import platform
import time
from datetime import datetime

def run_demos():
    """
    Finds and runs all .py and .ipynb demos in the 'demos' directory.
    For .ipynb files, it converts them to .py using jupytext first.
    Saves text output and PNG files to 'demo/runoutput' folder.
    Shows execution status and timing on screen.
    Keeps all temporary files (converted .py and wrapper scripts) in runoutput.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(project_root, 'src')
    demos_dir = os.path.join(project_root, 'demos')
    
    # Create runoutput directory
    runoutput_dir = os.path.join(demos_dir, 'runoutput')
    os.makedirs(runoutput_dir, exist_ok=True)
    
    print(f"Demo Runner Started - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output will be saved to: {runoutput_dir}")
    print(f"Source path: {src_path}")
    print("=" * 60)

    python_executable = "python" if platform.system() == "Windows" else "python3"
    total_demos = 0
    successful_demos = 0
    failed_demos = 0
    total_start_time = time.time()

    for root, _, files in os.walk(demos_dir):
        # Skip the runoutput directory itself
        if 'runoutput' in root:
            continue
            
        for file in files:
            if not (file.endswith('.py') or file.endswith('.ipynb')):
                continue
                
            total_demos += 1
            file_path = os.path.join(root, file)
            script_path = None
            wrapper_script = None
            demo_start_time = time.time()
            
            # Create demo-specific output directory
            demo_name = os.path.splitext(file)[0]
            demo_output_dir = os.path.join(runoutput_dir, demo_name)
            os.makedirs(demo_output_dir, exist_ok=True)
            
            print(f"\n[{total_demos}] Running: {file}")
            print(f"    Output dir: {demo_output_dir}")

            try:
                if file.endswith('.py'):
                    script_to_run = file_path
                elif file.endswith('.ipynb'):
                    print(f"    Converting notebook...")
                    # Convert notebook to .py script using jupytext
                    result = subprocess.run(['jupytext', '--to', 'py', file_path], 
                                          check=True, capture_output=True, text=True)
                    script_path = os.path.splitext(file_path)[0] + '.py'
                    script_to_run = script_path

                # Create a wrapper script that configures matplotlib and runs the demo
                wrapper_script = create_matplotlib_wrapper(script_to_run, demo_output_dir, demo_name)
                
                # Set up environment
                env = os.environ.copy()
                env['PYTHONPATH'] = src_path + os.pathsep + env.get('PYTHONPATH', '')

                # Run the demo and capture output with 5-minute timeout
                print(f"    Executing (max 5min timeout)...")
                result = subprocess.run([python_executable, wrapper_script], 
                                      capture_output=True, text=True, env=env, timeout=300)
                
                demo_duration = time.time() - demo_start_time
                
                # Save output to file
                output_file = os.path.join(demo_output_dir, f"{demo_name}_output.txt")
                with open(output_file, 'w') as f:
                    f.write(f"Demo: {file}\n")
                    f.write(f"Execution Time: {demo_duration:.2f} seconds\n")
                    f.write(f"Return Code: {result.returncode}\n")
                    f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("=" * 50 + "\n")
                    f.write("STDOUT:\n")
                    f.write(result.stdout)
                    if result.stderr:
                        f.write("\n" + "=" * 50 + "\n")
                        f.write("STDERR:\n")
                        f.write(result.stderr)
                
                # Move temp files to output directory instead of deleting them
                move_temp_files_to_output(script_path, wrapper_script, demo_output_dir, demo_name)
                
                # Display result
                if result.returncode == 0:
                    successful_demos += 1
                    print(f"    ✓ SUCCESS - {demo_duration:.2f}s")
                    if result.stdout.strip():
                        # Show first few lines of output
                        stdout_lines = result.stdout.strip().split('\n')
                        preview_lines = stdout_lines[:3]
                        for line in preview_lines:
                            if line.strip():
                                print(f"      {line}")
                        if len(stdout_lines) > 3:
                            print(f"      ... ({len(stdout_lines)-3} more lines in output file)")
                else:
                    failed_demos += 1
                    print(f"    ✗ FAILED - {demo_duration:.2f}s (code: {result.returncode})")
                    if result.stderr:
                        # Show error preview
                        error_lines = result.stderr.strip().split('\n')
                        print(f"      Error: {error_lines[-1] if error_lines else 'Unknown error'}")

            except subprocess.TimeoutExpired:
                demo_duration = time.time() - demo_start_time
                failed_demos += 1
                print(f"    ✗ TIMEOUT - {demo_duration:.2f}s (exceeded 5min limit)")
                
                # Save timeout info to file
                output_file = os.path.join(demo_output_dir, f"{demo_name}_output.txt")
                with open(output_file, 'w') as f:
                    f.write(f"Demo: {file}\n")
                    f.write(f"Execution Time: {demo_duration:.2f} seconds\n")
                    f.write(f"Status: TIMEOUT (exceeded 5 minute limit)\n")
                    f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                
                # Move temp files even on timeout
                move_temp_files_to_output(script_path, wrapper_script, demo_output_dir, demo_name)
                    
            except FileNotFoundError as e:
                demo_duration = time.time() - demo_start_time
                failed_demos += 1
                error_msg = "jupytext not found" if "jupytext" in str(e) else str(e)
                print(f"    ✗ FAILED - {demo_duration:.2f}s")
                print(f"      Error: {error_msg}")
                
                # Save error to file
                output_file = os.path.join(demo_output_dir, f"{demo_name}_output.txt")
                with open(output_file, 'w') as f:
                    f.write(f"Demo: {file}\n")
                    f.write(f"Execution Time: {demo_duration:.2f} seconds\n")
                    f.write(f"Error: {error_msg}\n")
                    f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                
                # Move temp files even on error (if they exist)
                move_temp_files_to_output(script_path, wrapper_script, demo_output_dir, demo_name)
                
                if "jupytext" in str(e):
                    print("      Install with: pip install jupytext")
                    break
                    
            except Exception as e:
                demo_duration = time.time() - demo_start_time
                failed_demos += 1
                print(f"    ✗ FAILED - {demo_duration:.2f}s")
                print(f"      Error: {str(e)}")
                
                # Save error to file
                output_file = os.path.join(demo_output_dir, f"{demo_name}_output.txt")
                with open(output_file, 'w') as f:
                    f.write(f"Demo: {file}\n")
                    f.write(f"Execution Time: {demo_duration:.2f} seconds\n")
                    f.write(f"Error: {str(e)}\n")
                    f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                
                # Move temp files even on error (if they exist)
                move_temp_files_to_output(script_path, wrapper_script, demo_output_dir, demo_name)

    # Final summary
    total_duration = time.time() - total_start_time
    print("\n" + "=" * 60)
    print("EXECUTION SUMMARY")
    print("=" * 60)
    print(f"Total Demos: {total_demos}")
    print(f"Successful: {successful_demos} ✓")
    print(f"Failed: {failed_demos} ✗")
    print(f"Success Rate: {(successful_demos/total_demos*100):.1f}%" if total_demos > 0 else "N/A")
    print(f"Total Time: {total_duration:.2f} seconds")
    print(f"Average Time per Demo: {(total_duration/total_demos):.2f} seconds" if total_demos > 0 else "N/A")
    print(f"Output saved to: {runoutput_dir}")

def move_temp_files_to_output(script_path, wrapper_script, demo_output_dir, demo_name):
    """
    Moves temporary files (converted .py from notebooks and wrapper scripts) 
    to the demo output directory instead of deleting them.
    """
    try:
        # Move the converted .py file from notebook (if it exists)
        if script_path and os.path.exists(script_path):
            converted_py_dest = os.path.join(demo_output_dir, f"{demo_name}_converted.py")
            os.rename(script_path, converted_py_dest)
            print(f"    Moved converted script to: {os.path.basename(converted_py_dest)}")
        
        # Move the wrapper script (if it exists)
        if wrapper_script and os.path.exists(wrapper_script):
            wrapper_dest = os.path.join(demo_output_dir, f"{demo_name}_wrapper.py")
            os.rename(wrapper_script, wrapper_dest)
            print(f"    Moved wrapper script to: {os.path.basename(wrapper_dest)}")
            
    except Exception as e:
        print(f"    Warning: Could not move temp files: {str(e)}")
        # If moving fails, fall back to deletion to avoid cluttering
        try:
            if script_path and os.path.exists(script_path):
                os.remove(script_path)
            if wrapper_script and os.path.exists(wrapper_script):
                os.remove(wrapper_script)
        except:
            pass  # Ignore cleanup errors

def create_matplotlib_wrapper(original_script, output_dir, demo_name):
    """
    Creates a wrapper script that configures matplotlib for PNG output
    and then executes the original script.
    """
    wrapper_content = f'''
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
        filename = f"{demo_name}_fig{{_figure_counter:03d}}.png"
        filepath = os.path.join("{output_dir}", filename)
        
        # Save the figure
        fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved plot as: {{filename}}")
        
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
    exec(open(r"{original_script}").read())
except Exception as e:
    print(f"Error in demo execution: {{e}}", file=sys.stderr)
    raise

# Save any remaining figures that weren't explicitly shown
if plt.get_fignums():  # Check if there are any figures
    save_figure_as_png()
'''
    
    # Create temporary wrapper script
    wrapper_path = original_script + '_wrapper_temp.py'
    with open(wrapper_path, 'w') as f:
        f.write(wrapper_content)
    
    return wrapper_path

if __name__ == "__main__":
    run_demos()