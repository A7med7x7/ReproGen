import os,sys
import mlflow
import platform
import subprocess
from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlShutdown, NVMLError

def log_gpu():
    """
    This function checks for the availability of GPU spynvml (NVIDIA),
    and logs the GPU count, and nvidia-smi output.
    """
    try:
        try:
            nvmlInit()
            count = nvmlDeviceGetCount()
            mlflow.log_param("gpu.count", count)
        except NVMLError as nvml_err:
            mlflow.set_tag("gpu.status", f"NVML ereror: {str(nvml_err)}")
            return
        except Exception as e:
            mlflow.set_tag("gpu.status", f"NVML init failed: {str(e)}")
            return

        # nvidia-smi
        try:
            smi_output = subprocess.check_output(["nvidia-smi"], text=True)
            mlflow.log_text(smi_output, artifact_file="gpu-info.txt")
        except Exception as smi_err:
            mlflow.set_tag("gpu.status", f"nvidia-smi failed with: {smi_err}")

    except Exception as e:
        mlflow.set_tag("gpu.status", f"unexpected error: {str(e)}")

    finally:
        try:
            nvmlShutdown()
        except Exception as shutdown_err:
            mlflow.set_tag("gpu.shutdown", f"warning: {str(shutdown_err)}")

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
        
        
        git_info = f"""
        Remote: {git_remote}
        Branch: {branch}
        Commit: {commit}

        --- Git Diff ---
        {git_diff}
        """
        
        mlflow.set_tag("Branch", str(branch))
        mlflow.set_tag("git.dirty", str(is_dirty))
        mlflow.log_text(git_info.strip(), "git_info.txt")

    except subprocess.CalledProcessError:
        mlflow.set_tag("git.status", "not a git repo or error")