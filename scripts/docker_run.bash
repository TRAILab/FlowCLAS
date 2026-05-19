#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]] || [[ $1 == -* ]]; then
    echo "Usage: $(basename "$0") <data-host-path>" >&2
    echo "  Mounts host path as container ~/data alongside project volumes." >&2
    exit 1
fi

DRIVE=$1
ME=$(whoami)

VOLUMES=(
    --volume="${PWD}/misc:/home/${ME}/misc"
    --volume="${PWD}/src:/home/${ME}/src"
    --volume="${PWD}/results:/home/${ME}/results"
    --volume="${DRIVE}:/home/${ME}/data"
)

exec docker run -it \
    --privileged \
    -p 6006:6006 \
    -p 8888:8888 \
    -e "DISPLAY=unix${DISPLAY:-}" \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e "TORCH_HOME=/home/${ME}/misc" \
    -e "HF_HOME=/home/${ME}/misc" \
    -v /tmp/.X11-unix/:/tmp/.X11-unix/ \
    --gpus device=0 \
    --shm-size 32G \
    "${VOLUMES[@]}" \
    --name=flowclas \
    flowclas
