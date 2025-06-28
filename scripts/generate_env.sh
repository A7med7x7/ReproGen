#!/bin/bash

REMOTE_NAME="rclone_s3" 
RCLONE_CONF="${HOME}/.config/rclone/rclone.conf"
ENV_FILE="../docker/.env"

# Extract values of  rclone.conf
S3_ACCESS_KEY=$(grep -A5 "\[$REMOTE_NAME\]" "$RCLONE_CONF" | grep "access_key_id" | cut -d '=' -f2 | xargs)
S3_SECRET_ACCESS_KEY=$(grep -A5 "\[$REMOTE_NAME\]" "$RCLONE_CONF" | grep "secret_access_key" | cut -d '=' -f2 | xargs)

# Get public IP
HOST_IP=$(curl -s ifconfig.me)

# Creatthe e .env file
echo "S3_ACCESS_KEY=${S3_ACCESS_KEY}" > "$ENV_FILE"
echo "S3_SECRET_ACCESS_KEY=${S3_SECRET_ACCESS_KEY}" >> "$ENV_FILE"
echo "HOST_IP=${HOST_IP}" >> "$ENV_FILE"
echo "MLFLOW_TRACKING_URI=http://localhost:5000" >> "$ENV_FILE"

echo "✅ The .env file has been generated successfully."
