import sys
import subprocess


def run_main(*args):
    """Helper function to run __main__.py and capture output"""
    cmd = [sys.executable, "-m", "yaml_ml", *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout, result.stderr, result.returncode
