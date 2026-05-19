# Builder: compile MMCV and install Python stack.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 AS builder

ENV TZ=Canada/Eastern \
    DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 12.0+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    FORCE_CUDA=1 \
    CUDA_HOME=/usr/local/cuda

ENV PATH=$CUDA_HOME/bin:$PATH \
    LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

SHELL ["/bin/bash", "-c"]

COPY apt_packages.txt /tmp/apt_packages.txt
RUN apt-get update && \
    xargs -a /tmp/apt_packages.txt apt-get install -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*.deb /tmp/apt_packages.txt

RUN pip3 install --upgrade --no-cache-dir pip setuptools wheel ninja
RUN pip3 install --no-cache-dir torch==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu128

ENV CMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"

RUN pip3 uninstall -y mmcv mmcv-full mmcv-lite || true
RUN git clone --branch v2.1.0 --depth 1 https://github.com/open-mmlab/mmcv.git && \
    cd mmcv && \
    MMCV_WITH_OPS=1 python3 -m pip install -v --no-build-isolation --no-cache-dir . && \
    cd / && rm -rf /mmcv

RUN python3 -m pip install --no-deps --no-cache-dir mmdet==3.3.0 mmsegmentation==1.2.2

COPY constraints-pytorch.txt /tmp/constraints-pytorch.txt
COPY requirements.txt /tmp/requirements.txt
# Constraints keep torch on cu128-compatible pins (see constraints-pytorch.txt).
RUN for attempt in 1 2 3 4 5; do \
      PIP_CONSTRAINT=/tmp/constraints-pytorch.txt pip3 install --no-build-isolation --no-cache-dir -r /tmp/requirements.txt \
        && rm -f /tmp/requirements.txt && break || true; \
      if [ "$attempt" -eq 5 ]; then echo "requirements install failed after 5 attempts"; exit 1; fi; \
      echo "pip requirements attempt ${attempt} failed; retrying in $((attempt * 20))s..."; \
      sleep $((attempt * 20)); \
    done

RUN git clone -n https://github.com/facebookresearch/sam2.git /sam2 && \
    cd /sam2 && \
    git checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4 && \
    PIP_CONSTRAINT=/tmp/constraints-pytorch.txt pip3 install --no-cache-dir . && \
    cd / && rm -rf /sam2 /tmp/constraints-pytorch.txt

RUN ln -sf /usr/bin/python3 /usr/local/bin/python

# Runtime: slim CUDA image + copied site-packages.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV TZ=Canada/Eastern \
    DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 12.0+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    FORCE_CUDA=1 \
    CUDA_HOME=/usr/local/cuda

ENV PATH=$CUDA_HOME/bin:$PATH \
    LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

ARG USERNAME=root
ARG RENDER_GID
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ARG CREATE_USER="false"
# When CREATE_USER=true, pass e.g. /home/$(whoami). When false, default /root is used.
ARG WORKDIR_PATH=/root

SHELL ["/bin/bash", "-c"]

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-tk \
        libgl1-mesa-glx libegl1-mesa libosmesa6 libnvidia-gl-535 \
        libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 \
        libxcb-xinerama0 libxcb-xkb1 libxkbcommon-x11-0 libxcb-randr0 \
        libx11-6 mesa-utils xvfb x11-utils \
        libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*.deb

COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=builder /usr/local/bin /usr/local/bin

RUN ln -sf /usr/bin/python3 /usr/local/bin/python

RUN if [ "$CREATE_USER" = "true" ]; then \
        groupadd --gid "$USER_GID" "$USERNAME" \
        && useradd --uid "$USER_UID" --gid "$USER_GID" -m "$USERNAME" \
        && apt-get update \
        && apt-get install -y --no-install-recommends sudo \
        && echo "$USERNAME ALL=(root) NOPASSWD:ALL" > "/etc/sudoers.d/$USERNAME" \
        && chmod 0440 "/etc/sudoers.d/$USERNAME" \
        && groupadd --gid "$RENDER_GID" render \
        && usermod -aG render "$USERNAME" \
        && usermod -aG video "$USERNAME" \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*.deb; \
    else \
        echo "Skipping user creation (image runs as root)"; \
    fi

USER $USERNAME
WORKDIR ${WORKDIR_PATH}
