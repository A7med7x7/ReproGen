import os,sys
import mlflow
import platform
import subprocess
try:
    from pynvml import (
        nvmlInit,
        nvmlDeviceGetCount,
        nvmlShutdown,
        NVMLError
    )
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

def log_nvidia_gpu():
    """
    This function logs the GPU count, and nvidia-smi output.
    """
    try:
        try:
            nvmlInit()
            count = nvmlDeviceGetCount()
            mlflow.set_tag("gpu_count", count)
        except NVMLError as nvml_err:
            mlflow.set_tag("gpu_status", f"NVML error: {str(nvml_err)}")
            return
        except Exception as e:
            mlflow.set_tag("gpu_shutdown", f"warning: {str(shutdown_err)}")
            return

        try:
            smi_output = subprocess.check_output(["nvidia-smi"], text=True)
            mlflow.log_text(smi_output, artifact_file="nvidia_gpu_info.txt")
        except Exception as smi_err:
            mlflow.set_tag("gpu_status", f"nvidia-smi failed: {smi_err}")

    except Exception as e:
        mlflow.set_tag("gpu_status", f"unexpected NVIDIA error: {str(e)}")

    finally:
        try:
            nvmlShutdown()
        except Exception as shutdown_err:
            mlflow.set_tag("gpu_shutdown", f"warning: {str(shutdown_err)}")

def log_amd_gpu():
    pass

def log_gpu():
    """
    Detects GPU type and logs info accordingly.
    """
    try:
        # NVIDIA 
        if NVML_AVAILABLE:
            try:
                log_nvidia_gpu()
                return
            except:
                pass

        # AMD
        try:
            subprocess.run(["rocm-smi"], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
            log_amd_gpu()
            return
        except FileNotFoundError:
            mlflow.set_tag("gpu_status", "No NVIDIA or AMD GPU tools found")
            return

    except Exception as e:
        mlflow.set_tag("gpu_status", f"auto-detect error: {str(e)}")

def log_python():
    """
    This function logs the Python version, platform information, 
    and the list of installed packages.
    """
    mlflow.set_tag("python_version", sys.version.split()[0])
    mlflow.set_tag("platform", platform.platform())
    try:
        pip_freeze = subprocess.check_output(["pip", "freeze"]).decode()
        mlflow.log_text(pip_freeze, artifact_file="environment.txt")
    except Exception as e:
        mlflow.set_tag("pip_freeze_error", str(e))
        
def log_git():
    """
    This function logs the remote repo link, branch name, commit hash, 
    and diff (uncommited changes), set a dirty tag if TRUE. (A repository is considered dirty if
    there are uncommitted changes (i.e., modifications/staged files,  not yet committed)
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
        
        mlflow.set_tag("git_branch", str(branch))
        mlflow.set_tag("git_dirty", str(is_dirty))
        mlflow.log_text(git_info.strip(), "git_info.txt")

    except subprocess.CalledProcessError:
        mlflow.set_tag("git_status", "not a git repo or error")