#!/bin/bash

# Configure rclone
mkdir -p ~/.config/rclone
cat > ~/.config/rclone/rclone.conf <<EOL
[chi_tacc]
type = swift
user_id = $1
application_credential_id = $2
application_credential_secret = $3
auth = https://chi.tacc.chameleoncloud.org:5000/v3
region = CHI@TACC
EOL

# Mount object storage (run as needed)
rclone mount chi_tacc:$4 /mnt/object --read-only --allow-other --daemon