import sys
import subprocess


def run_main(*args):
    """Helper function to run __main__.py and capture output"""
    cmd = [sys.executable, "-m", "yaml_ml", *args]
    result = subprocess.run(cmd, encoding="utf-8", capture_output=True, text=True)

    if result.returncode != 0:  # checking if an error occurred
        raise RuntimeError(
            f"Error running yaml_ml with args {args}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}\n"
        )

    return result.stdout, result.stderr, result.returncode
