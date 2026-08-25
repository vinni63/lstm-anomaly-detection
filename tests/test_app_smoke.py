"""Optional integration smoke test: starts Streamlit briefly, then shuts it down."""
import subprocess
import sys
import time
from pathlib import Path


def test_streamlit_server_starts():
    root = Path(__file__).resolve().parents[1]
    process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless=true", "--server.port=8509"],
                               cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        time.sleep(3)
        assert process.poll() is None, process.stdout.read()
    finally:
        process.terminate()
        process.wait(timeout=5)
