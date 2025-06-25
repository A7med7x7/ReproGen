#!/bin/bash

# Get host IP
export HOST_IP=$(curl --silent http://169.254.169.254/latest/meta-data/public-ipv4)

# Deploy services
docker compose -f ~/data-persist-chi/docker/docker-compose-block.yaml up -d