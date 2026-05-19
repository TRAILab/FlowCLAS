#!/usr/bin/env bash
set -euo pipefail

DIR=$(cd "$(dirname "$0")" && pwd)
RENDER_GID=$(getent group render | cut -d: -f3)

docker build \
    --network=host \
    --build-arg "USERNAME=$(whoami)" \
    --build-arg CREATE_USER=true \
    --build-arg "WORKDIR_PATH=/home/$(whoami)" \
    --build-arg "RENDER_GID=${RENDER_GID}" \
    -t flowclas \
    -f "${DIR}/../docker/Dockerfile" \
    "${DIR}/../docker"
