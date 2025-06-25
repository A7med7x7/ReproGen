#!/bin/sh
set -e

#waiting for the minio to be ready
echo "Waiting for Minio to be ready..."
sleep 60

mc alias set minioserver http://minio:9000 ${MINIO_ROOT_USER} ${MINIO_ROOT_PASSWORD}
mc mb minioserver/mlflow
echo "debug passed, the script is accessible"