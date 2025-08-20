# ReproGen: A Template Generator for Reproducible Machine Learning Projects on Chameleon Cloud

[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-orange.json)](https://github.com/copier-org/copier)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub release](https://img.shields.io/github/v/release/a7med7x7/chamelab-cli)](https://github.com/a7med7x7/chamelab-cli/releases)


> Use this template generator when you are ready to start a machine learning project on [Chameleon Cloud](https://www.chameleoncloud.org), or if you want to integrate reproducible experiment tracking into your existing project.


## How to use this template generator to create a new project

###  Requirements

- [Chameleon Cloud account](https://www.chameleoncloud.org)
- **[Python](https://www.python.org/downloads/)** ≥ 3.8
- **[Copier](https://copier.readthedocs.io/en/stable/)** ≥ 9.0

---

###  Installation

Install Copier

```sh
pip install copier
```

or 
```sh
pipx install copier 
```

---

## Quick Start

Create a New Project with 
```sh
copier copy --vcs-ref dev https://github.com/A7med7x7/reprogen.git my_project_name
```
* replace my_project_name with the name of the directory you want
* Answer a few questions

Then, you're ready to use your new project!

## Setup Parameters and Their Roles
your answers will generate a project based on your input values, below you can find the variables with their description and implications  

### `setup_mode`

- **Basic**: minmal prompts. you will be asked to input your project_name, remote repository link and framework of your choice it recommend defaults for most options.
- **Advanced**: Lets you control Compute site, GPU type, CUDA/ROCm version, storage site
the rest of the documentation shows what these options are and their implications 

--- 
### `project_name`
- We recommend setting `project name` as the prefix for the lease name 
- it is used everywhere your project is referenced:   
    - S3 bucket names (e.g., `project-name-data`, `project-name-mlflow-artifacts`)
    - Compute instances /servers are going to include the `project_name` in default format 
    
        ```python
			
            # when creating a server
			s = server.Server(
			f"{{ project_name }}-node-{username}"

        ```
			
     
	- Your material on the compute instance will be under a directory named after your `project_name`
    - The containerized environment will look for a directory with the `project_name`
     directory named after your `project_name`
	- Some commands and scripts assume a unified `project_name`
- **Rules**:Oonly letters, numbers, hyphen (-), and underscore (_). no spaces.
- **Tip**: Choose something short and memorable — remember this will show up in multiple commands and URLs
- **Type:** select

---

### `repo_url`

- The Git repository where the generated project will live, we recommend creating  a remote repository (e.g [GitHub](github.com) or [GitLab](gitlab.com)). 
- Accepts HTTPS or SSH URLs (e.g., `https://github.com/user/repo.git` or `git@gitlab.com:user/repo.git`).
- After having your project generated, you need to push the code there. (see [Github](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github) / [Gitlabl](https://forum.gitlab.com/t/how-do-i-push-a-project-to-a-newly-created-git-repo-on-gitlab/68426) Guide)
- **Type:** string

---

### `chameleon_site`

- The site where your leases at and compute resources will be provisioned. 
- This Doesn’t control persistent storage storage location (that’s `bucket_site`).
- CHI@TACC → Most GPU bare metal nodes.
- CHI@UC → University of Chicago resources.
- KVM@TACC → VM(Virtual Machines )-based compute at TACC.
- **Type:** select

--- 

### `bucket_site`
###### *work only under advanced*

- This is where your [object storage contrainers](https://chameleoncloud.readthedocs.io/en/latest/technical/swift/index.html#object-store) (S3 Buckets) for you project will live.
#### **options**

- CHI@TACC: Texas Advanced Computing Center
- CHI@UC: University of Chicago
- **auto** is usually the best choice unless you have a reason to store data in a specific  location. if matches your selected `chameleon_site` if object storage containers are available, if not it defaults to CHI@TACC site. 
- **Note**: note that if your `chamleon_site` for the compute resources is different than your `bucket_site` further configuration might be needed. (placeholder until I update this)
- **Type:** select

--- 
### `gpu_type`
###### *work only under advanced*

- The type of GPU (or CPU-only) you want to configure. this assumes that you have reserved a node and you know which type it is AMD, NVIDIA or CPU.
- configuring a server from a lease require the `gpu_type`, as different `gpus` have different setup process. 
- `nvidia` and `amd` require different container images to. so your decision will result in selecting the appropriate [container images](https://github.com/A7med7x7/ReproGen/tree/dev/template/docker)
- **Type:** Multi-choice - you can select multiple types. 
- **Note**: when selecting `chemeleon_site` = KVM@TACC the GPU flavors run on NVIDIA hardware as there are no AMD hardware. so this question is not going to be prompted when `chemeleon_site` = KVM@TACC
---  

### `ml_framework`

- Selects the primary ML/deep learning framework for your environment.
- It will decide which container image to include and use for your jupyter lab. 
- custom training code for the selected `ml_framework` will be generated
- **pytorch** – Flexible, widely used deep learning library. Supports CUDA (NVIDIA) and ROCm (AMD).
- **pytorch-lightning** – High-level wrapper for PyTorch that simplifies training loops. Supports CUDA (NVIDIA) and ROCm (AMD).
- **tensorflow** – Popular deep learning library with strong ecosystem. 
- **scikit-learn** –  Machine Learning and data science stack (pandas, scikit-learn, matplotlib, etc.) without deep learning frameworks.
- **Note**: _PyTorch_ and _PyTorch Lightning_ will prompt for CUDA/ROCm version if you select GPU types.
- **Type** Multi-choice: you can select multiple frameworks. 
---

### `cuda_version` 
###### *work only under advanced and NVIDIA setup*

- Choose the CUDA version that matches your code and driver requirements.
    - cuda11-latest : highly compatible with most GPUs in chameleon cloud 
    - cuda12-latest : The latest version designed to work with newer GPU architectures
- **Type** select:
---
### `include_huggingface` 
###### *work only under advanced* 

- If enabled, configures the environment to include a hugging face token for seamless Hugging Face Hub access.
- When configuring servers you will be prompted to enter a [Hugging Face Token](https://huggingface.co/settings/tokens) 
- All models/datasets downloaded from Hugging Face will be stored on the mounted point `/mnt/data/`
- **Type** bool
#### Acknowledgements

This project was supported by the 2025 [Summer of Reproducibility](https://ucsc-ospo.github.io/sor/).

Contributors: [Ahmed Alghali](https://ucsc-ospo.github.io/author/ahmed-alghali/), [Mohamed Saeed](https://ucsc-ospo.github.io/author/mohamed-saeed/), [Fraida Fund](https://ucsc-ospo.github.io/author/fraida-fund/).