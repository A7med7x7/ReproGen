import os
import getpass
from urllib.parse import urlparse

RCLONE_CONF_DIR = os.path.expanduser("~/.config/rclone")
RCLONE_CONF_PATH = os.path.join(RCLONE_CONF_DIR, "rclone.conf")

def get_container_name(link_or_name: str) -> str:
    """
    Extract container name from URL or return the plain name.
    """
    if link_or_name.startswith("http"):
        path = urlparse(link_or_name).path
        return path.rstrip("/").split("/")[-1]
    return link_or_name

def write_rclone_swift_config(ec2_id, ec2_secret, endpoint):
    os.makedirs(RCLONE_CONF_DIR, exist_ok=True)
    config_content = f"""
[rclone_s3]
provider = Ceph
access_key_id = {ec2_id}
secret_access_key = {ec2_secret}
endpoint = {endpoint}
"""
    
    with open(RCLONE_CONF_PATH, "w") as f:
        f.write(config_content)
    print(f"✅ Rclone s3 config written to {RCLONE_CONF_PATH}")

def main():
    print("=== setup rclone and .env for public object store containers ===")
    
    container1 = input("🦎 Enter metrices container name or full URL: ").strip()
    container2 = input("🦎 Enter artifacts container name or full URL: ").strip()
    
    container1_name = get_container_name(container1)
    container2_name = get_container_name(container2)
    
    ec2_id = input("🦎 Enter EC2 credentials access id: ").strip()
    ec2_secret = getpass.getpass("🦎 Enter EC2 credentials access secret (it will not be visible when typing): ").strip()
    endpoint = input("🦎 Enter ENDPOINT URL (e.g., https://chi.uc.chameleoncloud.org:7480): ").strip()
    
    write_rclone_swift_config(ec2_id, ec2_secret, endpoint)
    
    # Generate mount shell script in $HOME
    mount_script = f"""#!/bin/bash
sudo sed -i '/^#user_allow_other/s/^#//' /etc/fuse.conf
sudo mkdir -p /mnt/public-metrics /mnt/public-artifacts
sudo chown $USER:$USER /mnt/public-metrics /mnt/public-artifacts

rclone mount rclone_s3:{container1_name} /mnt/public-metrics \\
    --allow-other \\ 
    --daemon \\
    --vfs-cache-mode writes \\

rclone mount rclone_s3:{container2_name} /mnt/public-artifacts \\
    --allow-other \\ 
    --daemon \\
    --vfs-cache-mode writes \\
    
echo "✅ containers mounted at /mnt/public-metrics and /mnt/public-artifacts successfully"
"""
    generate_env = f"""#!/bin/bash

REMOTE_NAME="rclone_s3"
RCLONE_CONF="${{HOME}}/.config/rclone/rclone.conf"
ENV_DIR="${{HOME}}"
ENV_FILE="${{ENV_DIR}}/.env"

if [[ ! -f "$RCLONE_CONF" ]]; then
    echo "❌ rclone config not found at $RCLONE_CONF"
    exit 1
fi

# Extract values from rclone.conf (robust to whitespace)
S3_ACCESS_KEY=$(awk -F '=' '/\\[rclone_s3\\]/{{a=1}} a && /access_key_id/ {{gsub(/ /,"",$2); print $2; exit}}' "$RCLONE_CONF")
S3_SECRET_ACCESS_KEY=$(awk -F '=' '/\\[rclone_s3\\]/{{a=1}} a && /secret_access_key/ {{gsub(/ /,"",$2); print $2; exit}}' "$RCLONE_CONF")
S3_ENDPOINT_URL=$(awk -F '=' '/\\[rclone_s3\\]/{{a=1}} a && /endpoint/ {{gsub(/ /,"",$2); print $2; exit}}' "$RCLONE_CONF")

CONTAINER_NAME="{container2_name}"

# Get public IP
HOST_IP=$(curl -s ifconfig.me)

# Creating .env file
cat <<EOF > "$ENV_FILE"
S3_ACCESS_KEY=${{S3_ACCESS_KEY}}
S3_SECRET_ACCESS_KEY=${{S3_SECRET_ACCESS_KEY}}
HOST_IP=${{HOST_IP}}
CONTAINER_NAME=${{CONTAINER_NAME}}
S3_ENDPOINT_URL=${{S3_ENDPOINT_URL}}
EOF

echo "✅ The .env file has been generated successfully at: $ENV_FILE"
"""

    repo_root = os.path.dirname(os.path.abspath(__file__))
    scripts_dir = os.path.join(repo_root, "scripts")
    os.makedirs(scripts_dir, exist_ok=True)

    # Save the mount script
    mount_script_path = os.path.join(scripts_dir, "mount_public.sh")
    with open(mount_script_path, "w") as f:
        f.write(mount_script)
    os.chmod(mount_script_path, 0o755)
    print(f"✅ Mount script written to {mount_script_path}")
    print(f"Run it using: bash {mount_script_path}")

    # Save the generate_env script
    env_script_path = os.path.join(scripts_dir, "generate_env.sh")
    with open(env_script_path, "w") as f:
        f.write(generate_env)
    os.chmod(env_script_path, 0o755)
    print(f"✅ Environment script written to {env_script_path}")
    print(f"Run it using: bash {env_script_path}")

if __name__ == "__main__":
    main()