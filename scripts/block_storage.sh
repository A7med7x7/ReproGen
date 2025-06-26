#!/bin/bash

# Partition and format block storage
sudo parted -s /dev/vdb mklabel gpt
sudo parted -s /dev/vdb mkpart primary ext4 0% 100%
sudo mkfs.ext4 /dev/vdb1

# Mount block storage
sudo mount /dev/vdb1 /mnt/block
sudo chown -R ahmed_offsechq_com:users /mnt/block
