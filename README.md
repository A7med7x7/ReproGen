# ReproGen: A Template Generator for Reproducible Machine Learning Projects on Chameleon Cloud

## How to use this template generator to create a new template

Use this template generator when you are ready to start a machine learning project on Chameleon Cloud, or if you want to integrate reproducible experiment tracking into your existing project.

To use it:

* Install [cookiecutter](https://github.com/cookiecutter/cookiecutter)
* Run:

```
cookiecutter https://github.com/A7med7x7/ReproGen.git --checkout resource-management 
```


* Answer a few questions

Then, you're read to use your new project!

## Using your template

This repository automates setting up buckets, spinning up VMs and launching a fully configured Jupyter environment with MLFlow tracking server system.

### 0. Prerequisites
- You must have a working [Chameleon Cloud](https://chameleoncloud.org) account.
- You have already reserved a lease on Chameleon Cloud that includes a GPU-enabled bare metal node

### 1. Create S3 Buckets

Run the notebook `0_create_buckets.ipynb` to create S3-compatible buckets for datasets, metrics, and artifacts. to live beyond your instance lifetime.

**In [Chameleon JupyterHub](https://jupyter.chameleoncloud.org/hub/), open and run:**

- [`chi/0_create_buckets.ipynb`](chi/0_create_buckets.ipynb)

---

### 2. Launch and Set Up the Server

Provision your server and configure it for your project.

**In [Chameleon JupyterHub](https://jupyter.chameleoncloud.org/hub/), open and run:**

- For NVIDIA: [`chi/1_create_server_nvidia.ipynb`](chi/1_create_server_nvidia.ipynb)
- For AMD:  [`chi/1_create_server_amd.ipynb`](chi/1_create_server_amd.ipynb)

---

### 3. Generate Environment Variables

On your computer instance (SSH-ing from your local machine via shell), generate the `.env` file required for Docker Compose:
From your **home directory** (`~`), run:

```sh
 ./ReproGen/scripts/generate_env.sh
```

you will be prompted to enter your HuggingFace Token,after inputting.
you should see something like:

`✅ The .env file has been generated successfully at : /home/cc/.env`

---

### 4. Start the Containarized Environment

From your **home directory** (`~`), run:

```sh
docker compose --env-file ~/.env -f ReproGen/docker/docker-compose.yml up -d --build
```

---

### 5. Login to Jupyter Lab and MLFlow UI

1. Access your jupyter lab at:  `<HOST_IP>:8888` you can grap the token from running image using the command:

```sh
docker logs jupyter 2>&1 | grep -oE "http://127.0.0.1:8888[^ ]*token=[^ ]*"
```

- In the Jupyter terminal, log into GitHub using the CLI

```sh
gh auth login
```

Follow the intstructions to authenticate.

2. Access MLFlow UI at `<HOST_IP>:8000`

### 5.5. Stop the Containerized Environment
If you’d like to pause your environment, you can stop the running containers with the command:

```sh
docker compose --env-file ~/.env -f ReproGen/docker/docker-compose.yml down
```

This will stop and remove the containers, but all your data in mounted volumes will remain safe.
When you want to restart later, simply run the docker compose up command again (see Step 4).

---

### 6. Clean Up Resources

When finished, delete your server to free up resources.

**In Chameleon JupyterHub, open and run:**

- [`chi/2_delete_resources.ipynb`](chi/2_delete_resources.ipynb)

#### Acknowledgements

This project was supported by the 2025 [Summer of Reproducibility](https://ucsc-ospo.github.io/sor/).

Contributors: [Ahmed Alghali](https://ucsc-ospo.github.io/author/ahmed-alghali/), [Mohamed Saeed](https://ucsc-ospo.github.io/author/mohamed-saeed/), [Fraida Fund](https://ucsc-ospo.github.io/author/fraida-fund/).
