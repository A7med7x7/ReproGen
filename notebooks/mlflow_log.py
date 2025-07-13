import subprocess
import platform
import os
import sys
import mlflow

def log_gpu():
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
    
    try:
        smi_output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free,memory.used", "--format=csv,nounits,noheader"]
        )
        total, free, used = smi_output.decode().strip().split(', ')
        mlflow.log_metrics({
            "gpu_mem_total_mb": int(total),
            "gpu_mem_free_mb": int(free),
            "gpu_mem_used_mb": int(used)
        })
    except Exception as e:
        mlflow.set_tag("gpu.smi_error", str(e))

def log_python():
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