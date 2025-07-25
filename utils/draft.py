import os,sys
import mlflow
import platform
import subprocess
from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetName, nvmlShutdown, nvmlDeviceGetMemoryInfo, nvmlDeviceGetHandleByIndex

import mlflow
import subprocess
import shutil
from pynvml import (
    nvmlInit,
    nvmlShutdown,
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetName,
    nvmlDeviceGetMemoryInfo,
)


import mlflow
import subprocess
import shutil
from pynvml import (
    nvmlInit,
    nvmlShutdown,
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetName,
    nvmlDeviceGetMemoryInfo,
)


def log_gpu():
    """
    Logs GPU info to MLflow:
    - gpu.count (param)
    - gpu.X.memory.total_MB (param)
    - gpu.X.name (tag)
    - gpu-info.txt (artifact with device names + nvidia-smi/rocm-smi output)
    """
    gpu_summary_lines = []

    try:
        nvmlInit()
        count = nvmlDeviceGetCount()
        mlflow.log_param("gpu.count", count)

        for i in range(count):
            handle = nvmlDeviceGetHandleByIndex(i)
            name = nvmlDeviceGetName(handle).decode()
            mem_info = nvmlDeviceGetMemoryInfo(handle)
            total_mem_mb = mem_info.total // (1024 ** 2)
            mlflow.log_param(f"gpu.{i}.memory.total_MB", total_mem_mb)

            gpu_summary_lines.append(f"GPU {i}: {name}, Total Memory: {total_mem_mb} MB")

        nvmlShutdown()

    except ImportError:
        mlflow.set_tag("gpu.info", "pynvml not installed")
    except Exception as e:
        mlflow.set_tag("gpu.error", str(e))

    try:
        for cmd in ["nvidia-smi", "rocm-smi"]:
            if shutil.which(cmd):
                result = subprocess.run(cmd, capture_output=True, text=True)
                smi_output = result.stdout
                break
        else:
            smi_output = "No GPU utility (nvidia-smi or rocm-smi) found."

        artifact_content = "\n".join(gpu_summary_lines) + "\n\n" + smi_output
        mlflow.log_text(artifact_content, "gpu-info.txt")

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
    This function logs the diff (uncommitted changes), sets a dirty tag if TRUE
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
        
        git_info = f"""
        Remote: {git_remote}
        Commit: {commit}
        Dirty: {is_dirty}
        
        --- Git Diff ---
        {git_diff}
        """
        mlflow.set_tag("Branch", str(branch))
        mlflow.set_tag("git.dirty", str(is_dirty))
        mlflow.log_text(git_info.strip(), "git_info.txt")

    except subprocess.CalledProcessError:
        mlflow.set_tag("git.status", "may not be a Git repository or error")