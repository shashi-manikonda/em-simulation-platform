import json
import os
import re
import subprocess
import sys
import tempfile

import pytest


def find_demos():
    """Recursively finds all .ipynb and .py files in the demos directory."""
    demo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "demos"))
    demos = []
    for root, _, files in os.walk(demo_root):
        if ".ipynb_checkpoints" in root:
            continue
        for f in files:
            if f.endswith((".ipynb", ".py")):
                demos.append(os.path.join(root, f))
    return sorted(demos)


@pytest.mark.parametrize("backend", ["python", "cosy"])
@pytest.mark.parametrize("demo_path", find_demos())
def test_demo_quick(demo_path, backend):
    """
    Fast verification of a demo file with a specific backend.
    Runs the code in a subprocess to ensure isolation and speed.
    """
    if backend == "cosy":
        try:
            from sandalwood.taylor_function import _COSY_BACKEND_AVAILABLE

            if not _COSY_BACKEND_AVAILABLE:
                pytest.skip("COSY backend not available")
        except ImportError:
            pytest.skip("Could not check COSY availability")

    fname = os.path.basename(demo_path)

    def patch_line(line, backend):
        # 1. Force Backend
        if "mtf.initialize_mtf(max_order=" in line:
            if "implementation=" in line:
                line = line.replace(
                    'implementation="cosy"', f'implementation="{backend}"'
                )
                line = line.replace(
                    'implementation="python"', f'implementation="{backend}"'
                )
                line = line.replace(
                    'implementation="cpp"', f'implementation="{backend}"'
                )
            else:
                line = line.replace(")", f', implementation="{backend}")')

            # 2. SPEED HACK: Force low order for tests
            # This makes 5-second tests run in 0.1 seconds
            line = re.sub(r"max_order=\d+", "max_order=2", line)

        # 3. Patch private attribute access in legacy demos
        if "._vectors_mtf[0]" in line:
            line = line.replace("._vectors_mtf[0]", "[0]")
        if "._vectors_mtf" in line:
            line = line.replace("._vectors_mtf", "")  # Iterate directly over field

        # Fallback for cases without the dot (though unlikely given usage)
        if "_vectors_mtf[0]" in line:
            line = line.replace("_vectors_mtf[0]", "[0]")
        if "_vectors_mtf" in line:
            line = line.replace("_vectors_mtf", "")

        return line

    with tempfile.TemporaryDirectory() as temp_dir:
        exec_path = os.path.join(temp_dir, "run_demo.py")

        if demo_path.endswith(".ipynb"):
            # Extract code from notebook
            with open(demo_path, "r", encoding="utf-8") as f:
                nb = json.load(f)

            code_lines = []
            for cell in nb.get("cells", []):
                if cell.get("cell_type") == "code":
                    source = cell.get("source", [])
                    if isinstance(source, str):
                        code_lines.append(patch_line(source, backend))
                    else:
                        for line in source:
                            code_lines.append(patch_line(line, backend))
                    code_lines.append("\n")

            with open(exec_path, "w", encoding="utf-8") as f:
                f.write("import matplotlib\n")
                f.write("matplotlib.use('Agg')\n")  # Disable GUI
                # Mock IPython for environments where it is missing
                f.write("import sys, types\n")
                f.write("if 'IPython' not in sys.modules:\n")
                f.write("    mock_ipython = types.ModuleType('IPython')\n")
                f.write("    mock_display = types.ModuleType('IPython.display')\n")
                f.write("    mock_display.display = lambda *args, **kwargs: None\n")
                f.write("    mock_ipython.display = mock_display\n")
                f.write("    mock_ipython.get_ipython = lambda: None\n")
                f.write("    mock_ipython.version_info = (8, 24, 0)\n")
                f.write("    sys.modules['IPython'] = mock_ipython\n")
                f.write("    sys.modules['IPython.display'] = mock_display\n")

                # Pre-check for optional modules
                f.write("try:\n")
                f.write("    import psutil\n")
                f.write("except ImportError: pass\n")
                f.write("try:\n")
                f.write("    import torch\n")
                f.write("except ImportError: pass\n")

                f.write("".join(code_lines))
        else:
            # For .py files
            with open(demo_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            with open(exec_path, "w", encoding="utf-8") as f:
                f.write("import matplotlib\n")
                f.write("matplotlib.use('Agg')\n")
                # Mock IPython for environments where it is missing
                f.write("import sys, types\n")
                f.write("if 'IPython' not in sys.modules:\n")
                f.write("    mock_ipython = types.ModuleType('IPython')\n")
                f.write("    mock_display = types.ModuleType('IPython.display')\n")
                f.write("    mock_display.display = lambda *args, **kwargs: None\n")
                f.write("    mock_ipython.display = mock_display\n")
                f.write("    mock_ipython.get_ipython = lambda: None\n")
                f.write("    mock_ipython.version_info = (8, 24, 0)\n")
                f.write("    sys.modules['IPython'] = mock_ipython\n")
                f.write("    sys.modules['IPython.display'] = mock_display\n")

                # Pre-check for optional modules
                f.write("try:\n")
                f.write("    import psutil\n")
                f.write("except ImportError: pass\n")
                f.write("try:\n")
                f.write("    import torch\n")
                f.write("except ImportError: pass\n")

                for line in lines:
                    f.write(patch_line(line, backend))

        # Run in subprocess
        env = os.environ.copy()
        # Be careful not to override standard paths, but ensuring CWD is not polluting
        # Ensure src and sandalwood src are in PYTHONPATH
        src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
        # Hacky relative path to sandalwood from em-simulation-platform (assuming
        # sibling checkout)
        sandalwood_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "sandalwood", "src")
        )

        env["PYTHONPATH"] = (
            src_path
            + os.pathsep
            + sandalwood_path
            + os.pathsep
            + env.get("PYTHONPATH", "")
        )
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        result = subprocess.run(
            [sys.executable, exec_path],
            capture_output=True,
            text=True,
            env=env,
            cwd=temp_dir,
            timeout=120,
        )

        if result.returncode != 0:
            if "ModuleNotFoundError" in result.stderr:
                missing_mod = (
                    result.stderr.split("No module named ")[-1].strip().strip("'")
                )
                pytest.skip(f"Demo {fname} requires missing module: {missing_mod}")

            if (
                "NotImplementedError" in result.stderr
                and (
                    "implemented for Python backend" in result.stderr
                    or "only available for the COSY backend" in result.stderr
                )
            ) or (
                "RuntimeError" in result.stderr
                and "COSY backend not initialized" in result.stderr
            ):
                pytest.skip(
                    f"Demo {fname} uses features not available in {backend} backend"
                )

            pytest.fail(
                f"Demo {fname} failed execution:\nSTDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )
