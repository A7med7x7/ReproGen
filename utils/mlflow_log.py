import import os,sys
import mlflow
import platform
import subprocess
from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetName, nvmlShutdown, nvmlDeviceGetMemoryInfo, nvmlDeviceGetHandleByIndex

def log_gpu():
    """
    This function checks for the availability of GPU spynvml (NVIDIA),
    and logs the GPU Device name, total memory and count of devices.
    """
    try:
        from pynvml import (
            nvmlInit,
            nvmlShutdown,
            nvmlDeviceGetCount,
            nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetName,
            nvmlDeviceGetMemoryInfo,
        )
        nvmlInit()
        count = nvmlDeviceGetCount()
        for i in range(count):
            handle = nvmlDeviceGetHandleByIndex(i)
            name = nvmlDeviceGetName(handle)
            if isinstance(name, bytes):  # Handle legacy pynvml
                name = name.decode()
            mem_info = nvmlDeviceGetMemoryInfo(handle)
            mlflow.set_tag(f"gpu.{i}.name", name)
            mlflow.set_tag(f"gpu.{i}.memory.total_MB", mem_info.total // (1024 ** 2))
        mlflow.set_tag("gpu.count", count)
        nvmlShutdown()
    except ImportError:
        mlflow.set_tag("gpu.info", "pynvml not installed")
    except Exception as e:
        mlflow.set_tag("gpu.error", str(e))

def log_python():
    """
    This function logs the Python version, platform information, 
    and the list of installed packages.
    """

    mlflow.set_tag("python.version", sys.version.split()[0])
    mlflow.set_tag("platform", platform.platform())
    try:
        pip_freeze = subprocess.check_output(["pip", "freeze"]).decode()
        mlflow.log_text(pip_freeze, artifact_file="environment.txt")
    except Exception as e:
        mlflow.set_tag("pip_freeze_error", str(e))
        
def log_git():
    """
    This function logs the diff (uncommited changes), set a dirty tag if TRUE
    and log the diff. 
    """
    try:
        repo_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], stderr=subprocess.DEVNULL
        ).decode().strip()
        is_dirty = subprocess.call(
            ["git", "diff", "--quiet"], cwd=repo_root
        ) != 0
        git_diff = subprocess.check_output(
            ["git", "diff"], cwd=repo_root
        ).decode()

        mlflow.set_tag("git.dirty", str(is_dirty))
        mlflow.log_text(git_diff, artifact_file="git_diff.patch")

    except subprocess.CalledProcessError:
        mlflow.set_tag("git.status", "may not a Git repository or error")