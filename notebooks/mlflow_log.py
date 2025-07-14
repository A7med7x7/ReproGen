"""
This module contains functions to log details about the
execution environment, like GPU information, Python environment,
and Git repository status.
"""
import subprocess
import platform
import os
import sys
import mlflow

def log_gpu():
    """
    This function checks for the availability of GPUs using PyTorch and TensorFlow,
    and logs the GPU details if available.
    """
    try:
        import torch
        if torch.cuda.is_available():
            mlflow.set_tag("gpu.framework", "torch")
            mlflow.set_tag("gpu.available", True)
            mlflow.set_tag("gpu.name", torch.cuda.get_device_name(0))
        else:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                mlflow.set_tag("gpu.framework", "tensorflow")
                mlflow.set_tag("gpu.available", True)
                mlflow.set_tag("gpu.name", gpus[0].name)
            else:
                mlflow.set_tag("gpu.available", False)
                return
    except ImportError:
        mlflow.set_tag("gpu.available", False)
        mlflow.set_tag("gpu.framework", "none")
        return

def log_python():
    """
    This function logs the Python version, platform information, 
    and the list of installed packages.
    """
    mlflow.set_tag("python.version", sys.version.replace("\n", " "))
    mlflow.set_tag("platform", platform.platform())
    try:
        pip_freeze = subprocess.getoutput("pip freeze")
        with open("environment.txt", "w") as f:
            f.write(pip_freeze)
        mlflow.log_artifact("environment.txt")
    except Exception as e:
        mlflow.set_tag("pip_freeze_error", str(e))

def log_git():
    """
    This function logs the current commit hash, branch name, diff, and the last commit message.
    """
    try:
        commit = subprocess.getoutput("git rev-parse HEAD")
        branch = subprocess.getoutput("git rev-parse --abbrev-ref HEAD")
        diff = subprocess.getoutput("git diff")
        message = subprocess.getoutput("git log -1 --pretty=%B")

        mlflow.set_tag("git.commit", commit)
        mlflow.set_tag("git.branch", branch)
        mlflow.set_tag("git.message", message.strip())

        with open("git_diff.txt", "w") as f:
            f.write(diff)
        mlflow.log_artifact("git_diff.txt")
    except Exception as e:
        mlflow.set_tag("git_error", str(e))