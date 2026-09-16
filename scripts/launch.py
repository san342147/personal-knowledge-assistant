"""Own both server lifetimes; bind only to loopback and reject occupied ports."""
import os
import argparse
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-browser", action="store_true", help="Run both servers without opening a browser")
    args = parser.parse_args()
    for port in (8000, 8501):
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                raise SystemExit(f"Port {port} is already in use. Stop the previous server and retry.")
    children = []
    env = dict(os.environ, PYTHONUNBUFFERED="1", HF_HOME=str(ROOT / ".cache/huggingface"),
               HF_HUB_DISABLE_SYMLINKS_WARNING="1")
    try:
        api = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.api:app", "--host", "127.0.0.1",
                                "--port", "8000", "--workers", "1"], cwd=ROOT, env=env)
        children.append(api)
        for _ in range(120):
            if api.poll() is not None:
                raise SystemExit("API stopped during startup. Check the error above.")
            try:
                urllib.request.urlopen("http://127.0.0.1:8000/metrics", timeout=1).close()
                break
            except (urllib.error.URLError, TimeoutError):
                time.sleep(.5)
        else:
            raise SystemExit("API did not become ready within 60 seconds.")
        ui = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app/ui.py",
            "--server.address", "127.0.0.1", "--server.port", "8501", "--server.headless", "true",
            "--browser.gatherUsageStats", "false"], cwd=ROOT, env=env)
        children.append(ui)
        time.sleep(2)
        if not args.no_browser:
            webbrowser.open("http://127.0.0.1:8501")
        while all(child.poll() is None for child in children):
            time.sleep(.5)
    except KeyboardInterrupt:
        pass
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=8)
            except subprocess.TimeoutExpired:
                child.kill()


if __name__ == "__main__":
    main()
