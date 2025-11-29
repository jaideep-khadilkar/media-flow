import sys
import os

# Check if the process is running inside the Pixi environment (optional check)
if "PIXI_VIRTUAL_PREFIX" in os.environ:
    environment_status = "ACTIVE"
else:
    environment_status = "INACTIVE (Possible Error)"

# Check if FFmpeg is available (system dependency)
try:
    import subprocess

    subprocess.run(
        ["ffmpeg", "-version"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    ffmpeg_status = "FFmpeg is installed and accessible."
except (subprocess.CalledProcessError, FileNotFoundError):
    ffmpeg_status = "FFmpeg is NOT installed or accessible."


print("--------------------------------------------------")
print("✅ Project: MediaFlow Data Pipeline Test Complete")
print(f"✅ Python Version: {sys.version.split()[0]}")
print(f"✅ Pixi Environment Status: {environment_status}")
print(f"✅ System Dependency Check: {ffmpeg_status}")
print("--------------------------------------------------")

# This is where your Ray pipeline would start
# import ray
# ray.init()
# ...
