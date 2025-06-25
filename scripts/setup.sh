#!/bin/bash
#Run this once when you first create your VM in order to get the software you need
#Install required packages
sudo apt-get update
sudo apt-get install -y docker.io docker-compose parted curl python3-pip

# Install rclone
curl https://rclone.org/install.sh | sudo bash

# Configure FUSE
sudo sed -i '/^#user_allow_other/s/^#//' /etc/fuse.conf

# Create mount points
sudo mkdir -p /mnt/object
sudo mkdir -p /mnt/block
sudo chown -R cc:cc /mnt/object
sudo chown -R cc:cc /mnt/block

# Add user to docker group
sudo usermod -aG docker $USER