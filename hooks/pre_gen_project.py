import sys

def get_framework_suffix(framework: str) -> str:
    """Maps framework to the Dockerfile suffix."""
    suffix_mapping = {
        'pytorch': 'pt',
        'pytorch lightning': 'pt',
        'tensorflow': 'tf',
        'scikit_learn': 'ds'
    }
    return suffix_mapping.get(framework.lower(), framework)

def get_compute_platform(gpu_type: str) -> str:
    """Maps GPU type to compute platform."""
    platform_mapping = {
        'nvidia': 'cuda',
        'amd': 'rocm',
        'cpu': 'cpu' # applying a cpu version mentioned in issues suggestions
    }
    return platform_mapping.get(gpu_type.lower(), 'cpu')

def validate_cuda_version():
    """Validate CUDA version selection"""
    cuda_version = "{{ cookiecutter.cuda_version }}"
    gpu_type = "{{ cookiecutter.gpu_type }}"
    
    if gpu_type == "nvidia" and cuda_version not in ["11", "12"]:
        print("Error: CUDA version must be either 11 or 12 for NVIDIA GPUs")
        sys.exit(1)

def validate_and_process_names():
    framework = "{{ cookiecutter.framework }}" 
    gpu_type = "{{ cookiecutter.gpu_type }}"
    compute_platform = get_compute_platform(gpu_type)
    framework_suffix = get_framework_suffix(framework)
    
    if framework.lower() == 'scikit_learn':
        return 'Dockerfile.jupyter-ds'
    
    return f'Dockerfile.jupyter-{compute_platform}-{framework_suffix}'

if __name__ == '__main__':
    validate_cuda_version()
    dockerfile_name = validate_and_process_names()