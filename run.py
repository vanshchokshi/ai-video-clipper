import os
import sys
import webbrowser
import threading
import time
import socket

# Ensure virtual environment binaries are in PATH
_bin_dir = os.path.dirname(sys.executable)
if _bin_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _bin_dir + os.pathsep + os.environ.get("PATH", "")

import uvicorn

def find_available_port(start_port=8000, max_attempts=10):
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', port)) != 0:
                return port
    return start_port

def open_browser(url):
    time.sleep(1.5)
    try:
        webbrowser.open(url)
    except Exception:
        pass

if __name__ == "__main__":
    port = find_available_port(8000)
    url = f"http://127.0.0.1:{port}"
    print("=" * 60)
    print("   AI Video Clipper Studio")
    print(f"   Opening web interface at: {url}")
    print("=" * 60)
    threading.Thread(target=open_browser, args=(url,), daemon=True).start()
    uvicorn.run("app:app", host="127.0.0.1", port=port, reload=False)


