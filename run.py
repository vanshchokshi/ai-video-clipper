import os
import sys
import webbrowser
import threading
import time

# Ensure virtual environment binaries are in PATH
_bin_dir = os.path.dirname(sys.executable)
if _bin_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _bin_dir + os.pathsep + os.environ.get("PATH", "")

import uvicorn

def open_browser():
    time.sleep(1.5)
    try:
        webbrowser.open("http://127.0.0.1:8000")
    except Exception:
        pass

if __name__ == "__main__":
    print("=" * 60)
    print("   AI Video Clipper Studio")
    print("   Local & Private Video Repurposing (Whisper + Llama 3)")
    print("   Opening web interface at: http://127.0.0.1:8000")
    print("=" * 60)
    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)

