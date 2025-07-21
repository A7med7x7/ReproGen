import os,sys
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
        nvmlInit()
        count = nvmlDeviceGetCount()
        for i in range(count):
            handle = nvmlDeviceGetHandleByIndex(i)
            name = nvmlDeviceGetName(handle).decode() if isinstance(name := nvmlDeviceGetName(handle), bytes) else name
            mem_info = nvmlDeviceGetMemoryInfo(handle)
            mlflow.set_tag(f"gpu.{i}.name", name)
            mlflow.set_tag(f"gpu.{i}.memory.total_MB", mem_info.total // (1024 ** 2))
        mlflow.set_tag("gpu.count", count)
        nvmlShutdown()
    except ImportError:
        mlflow.set_tag("gpu.info", "pynvml not installed")
    except Exception as e:
        mlflow.set_tag("gpu.error", str(e))

    # Log nvidia-smi or rocm-smi output
    try:
        gpu_info = next(
            (
                subprocess.run(cmd, capture_output=True, text=True).stdout
                for cmd in ["nvidia-smi", "rocm-smi"]
                if subprocess.run(f"command -v {cmd}", shell=True, capture_output=True).returncode == 0
            ),
            "No GPU utility (nvidia-smi or rocm-smi) found."
        )
        mlflow.log_text(gpu_info, "gpu-info.txt")
    except Exception as e:
        mlflow.set_tag("gpu_info_dump_error", str(e))

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
        
        git_remote = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"], cwd=repo_root
        ).decode().strip()
        
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root
        ).decode().strip()
        
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root
        ).decode().strip()
        
        git_diff = subprocess.check_output(
            ["git", "diff"], cwd=repo_root
        ).decode()
        
        mlflow.set_tag("git.remote", git_remote)
        mlflow.set_tag("git.branch", branch)
        mlflow.set_tag("git.dirty", str(is_dirty))
        mlflow.log_text("git.commit", commit)
        mlflow.log_text(git_diff, artifact_file="git_diff.txt")

    except subprocess.CalledProcessError:
        mlflow.set_tag("git.status", "may not a Git repository or error")