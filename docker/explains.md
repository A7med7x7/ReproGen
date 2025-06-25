version: "3.9"
```
**Specifies the Compose file format version.**  
- Use the latest version your Docker Compose supports for new features and compatibility.

---

### 2. Services

````yaml
services:
```
**Defines the containers (services) you want Docker Compose to manage.**  
Each service runs a container.

---

#### 2.1. Postgres Service

````yaml
  postgres:
    image: postgres:latest
    restart: always
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_USER=${POSTGRES_USER}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - POSTGRES_DB=${POSTGRES_DB}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./postgres/init.sql:/docker-entrypoint-initdb.d/init.sql
```
- **image:** Which Docker image to use (`postgres:latest`).
- **restart:** Policy to always restart the container if it stops.
- **ports:** Maps port 5432 on your machine to 5432 in the container (Postgres default).
- **environment:** Sets environment variables for the container.  
  - `${POSTGRES_USER}` etc. are read from your `.env` file or shell.
- **volumes:**  
  - `postgres_data:/var/lib/postgresql/data` persists database data.
  - `./postgres/init.sql:/docker-entrypoint-initdb.d/init.sql` runs an init script at startup.

**Use case:**  
Run a persistent Postgres database with custom initialization.

---

#### 2.2. Minio Service

````yaml
  minio:
      image: quay.io/minio/minio
      ports:
        - "9000:9000"
        - "9001:9001"
      environment:
         - MINIO_ROOT_USER=${MINIO_ACCESS_KEY}
         - MINIO_ROOT_PASSWORD=${MINIO_SECRET_ACCESS_KEY}
         - MINIO_STORAGE_USE_HTTPS=false
      command: server /data --console-address ":9000"
      volumes:
        - minio_data:/data
```
- **image:** Minio image from Quay.
- **ports:** Maps Minio server and web UI ports.
- **environment:** Sets Minio root credentials and disables HTTPS.
- **command:** Custom command to start Minio server.
- **volumes:** Persists Minio data.

**Use case:**  
Run an S3-compatible object storage server locally.

---

#### 2.3. (Incorrectly Nested) minio-setup Service

Your file has a **syntax error** here.  
`minio-setup` should be at the same level as `minio`, not nested inside it.

**Corrected Example:**
````yaml
  minio-setup:
    image: quay.io/minio/mc
    depends_on:
      - minio
    volumes:
      - ./minio/create-bucket.sh:/create-bucket.sh
    entrypoint: /bin/sh
    command: -c "chmod +x /create-bucket.sh && /create-bucket.sh"
    environment:
      - MINIO_ROOT_USER=${MINIO_ACCESS_KEY}
      - MINIO_ROOT_PASSWORD=${MINIO_SECRET_ACCESS_KEY}
```
- **image:** Uses Minio client image.
- **depends_on:** Waits for `minio` to be ready.
- **volumes:** Mounts a script to create buckets.
- **entrypoint/command:** Runs the script.
- **environment:** Passes credentials.

**Use case:**  
Automate bucket creation after Minio starts.

---

### 3. Volumes

````yaml
volumes:
  postgres_data:
  minio_data:
```
**Defines named volumes for persistent storage.**  
- `postgres_data` and `minio_data` are used by the services above.

---

## Key Syntax & Use Cases

- **YAML indentation matters!** Use spaces, not tabs.
- **Services** are top-level under `services:`.
- **Volumes** can be defined at the bottom for reuse.
- **Environment variables** can be set inline or via a `.env` file.
- **Use cases:**  
  - Quickly spin up multi-container environments for development, testing, or CI.
  - Ensure data persists across container restarts.
  - Automate setup tasks (like DB migrations or bucket creation).

---

**Tip:**  
If you get errors like `Additional property Volumes is not allowed`, check for typos (should be `volumes`, not `Volumes`) and indentation.

Let me know if you want a corrected version of your file!