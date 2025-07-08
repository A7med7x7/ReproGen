#!/bin/bash

REMOTE_NAME="rclone_s3" 
RCLONE_CONF="${HOME}/.config/rclone/rclone.conf"
ENV_DIR="${HOME}"
ENV_FILE="${ENV_DIR}/.env"

if [[ ! -f "$RCLONE_CONF" ]]; then
  echo "❌ rclone config not found at $RCLONE_CONF"
  exit 1
fi

# Extract values of  rclone.conf
S3_ACCESS_KEY=$(grep -A5 "\[$REMOTE_NAME\]" "$RCLONE_CONF" | grep "access_key_id" | cut -d '=' -f2 | xargs)
S3_SECRET_ACCESS_KEY=$(grep -A5 "\[$REMOTE_NAME\]" "$RCLONE_CONF" | grep "secret_access_key" | cut -d '=' -f2 | xargs)
S3_ENDPOINT_URL=$(grep -A5 "\[$REMOTE_NAME\]" "$RCLONE_CONF" | grep "endpoint" | cut -d '=' -f2 | xargs)

# Get public IP
HOST_IP=$(curl -s ifconfig.me)

# Set constant values 
PROJECT_NAME="fancyproject"

# Creatthe e .env file
{
  echo "S3_ACCESS_KEY=${S3_ACCESS_KEY}"
  echo "S3_SECRET_ACCESS_KEY=${S3_SECRET_ACCESS_KEY}"
  echo "HOST_IP=${HOST_IP}"
  echo "PROJECT_NAME=${PROJECT_NAME}"
  echo "S3_ENDPOINT_URL=${S3_ENDPOINT_URL}"
} > "$ENV_FILE"

echo "✅ The .env file has been generated successfully at : $ENV_FILE"
