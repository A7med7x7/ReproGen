# ML Environment on Chameleon Cloud

This Branch uses [Cookiecutter](https://cookiecutter.readthedocs.io/en/latest/) to generate a customized ML environment project for Chameleon Cloud, supporting different frameworks, GPU types, and Hugging Face integration.

---

## 0. Prerequisites

- You must have a working [Chameleon Cloud](https://chameleoncloud.org) account.
- You have already reserved a lease on Chameleon Cloud that includes a GPU-enabled bare metal node.
- [Install Cookiecutter](https://cookiecutter.readthedocs.io/en/latest/installation.html):
  ```sh
  pip install cookiecutter
  ```

---

## 1. Generate Your Project with Cookiecutter

Clone and use the template branch:

```sh
cookiecutter https://github.com/A7med7x7/ReproGen.git --checkout template
```

You will be prompted for:

- Project name
- GPU type (nvidia, amd, none)
- ML framework (pytorch, pytorch lightning, scikit-learn, tensorflow)
- Whether you want to use a Hugging Face token
- (If yes) Your Hugging Face token value

Cookiecutter will generate a new project directory with the structure and files tailored to your choices.

---

## 2. Create S3 Buckets

Run the notebook `0_create_buckets.ipynb` to create S3-compatible buckets for datasets, metrics, and artifacts to live beyond your instance lifetime.

**In [Chameleon JupyterHub](https://jupyter.chameleoncloud.org/hub/), open and run:**

- [`chi/0_create_buckets.ipynb`](chi/0_create_buckets.ipynb)

---

## 3. Launch and Set Up the Server

Provision your server and configure it for your project.

**In [Chameleon JupyterHub](https://jupyter.chameleoncloud.org/hub/), open and run:**

- For NVIDIA: [`chi/1_create_server_nvidia.ipynb`](chi/1_create_server_nvidia.ipynb)
- For AMD: [`chi/1_create_server_amd.ipynb`](chi/1_create_server_amd.ipynb)

---

## 4. Generate Environment Variables

On your computer instance (SSH-ing from your local machine via shell), generate the `.env` file required for Docker Compose:
From your **home directory** (`~`), run:

```sh
./<your_project_name>/scripts/generate_env.sh
```

`✅ The .env file has been generated successfully at : /home/cc/.env`

---

## 5. Start the Containerized Environment

From your **home directory** (`~`), run:

```sh
docker compose --env-file ~/.env -f <your_project_name>/docker/docker-compose.yml up -d --build
```

---

## 6. Login to Jupyter Lab and MLFlow UI

1. Access your Jupyter Lab at: `<HOST_IP>:8888` (grab the token from the running image using the command below):

```sh
docker logs jupyter 2>&1 | grep -oE "http://127.0.0.1:8888[^ ]*token=[^ ]*"
```

- In the Jupyter terminal, log into GitHub using the CLI:

```sh
gh auth login
```

Follow the instructions to authenticate.

2. Access MLFlow UI at `<HOST_IP>:8000`

---

## 7. Clean Up Resources

When finished, delete your server to free up resources.

**In Chameleon JupyterHub, open and run:**

- [`chi/2_delete_resources.ipynb`](chi/2_delete_resources.ipynb)