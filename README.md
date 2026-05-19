# [WACV 2026] FlowCLAS: Enhancing Normalizing Flow Via Contrastive Learning For Anomaly Segmentation

Official repository for FlowCLAS.

[![arXiv](https://img.shields.io/badge/arXiv-2411.19888-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2411.19888)
[![Paper](https://img.shields.io/badge/Paper-WACV_2026-4b6fa1?logo=googledocs&logoColor=white)](https://openaccess.thecvf.com/content/WACV2026/papers/Lee_FlowCLAS_Enhancing_Normalizing_Flow-Based_Anomaly_Segmentation_Via_Contrastive_Learning_WACV_2026_paper.pdf)
[![Website](https://img.shields.io/badge/Website-FlowCLAS-0a9396?logo=googlechrome&logoColor=white)](https://trailab.github.io/FlowCLAS/)
[![Video](https://img.shields.io/badge/Video-YouTube-FF0000?logo=youtube&logoColor=white)](https://youtu.be/CUTDzFNhzaY)

All commands below assume the working directory is `src/` (where `trainer_cli.py` lives).

## Environment setup

### Docker

The image is defined under [`docker/`](docker/). It uses a multi-stage build (CUDA 12.8, PyTorch 2.7, MMCV, SAM2, and project Python dependencies). GPU access at runtime requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

**Build** (from the repository root):

```bash
bash scripts/docker_build.bash
```

Equivalent manual command (paths are relative to the repo root; the build context must be the `docker/` directory):

```bash
RENDER_GID=$(getent group render | cut -d: -f3)

docker build \
  --network=host \
  --build-arg "USERNAME=$(whoami)" \
  --build-arg CREATE_USER=true \
  --build-arg "WORKDIR_PATH=/home/$(whoami)" \
  --build-arg "RENDER_GID=${RENDER_GID}" \
  -t flowclas \
  -f docker/Dockerfile \
  docker
```

| Flag / argument | Purpose |
| --- | --- |
| `--network=host` | Uses the host network during build (helps reach package indexes). |
| `CREATE_USER=true` | Creates a container user matching your host username (for volume permissions). |
| `WORKDIR_PATH` | Sets the container working directory to your home path inside the image. |
| `RENDER_GID` | Adds the container user to a `render` group with the host’s GID (for GPU/GL access). |

The image is tagged `flowclas`. To start an interactive container with project directories mounted, run from the repository root:

```bash
bash scripts/docker_run.bash /path/to/your/datasets
```

## Results

Pre-trained checkpoints and pixel-level metrics on the official test splits:

| Benchmark | Data config | Model config | Checkpoint | Pixel AP | Pixel FPR95 |
| --- | --- | --- | --- | ---: | ---: |
| ALLO | [config](src/configs/base/allo_coco.yaml) | [config](src/configs/flowclas/flowclas_allo.yaml) | [checkpoint](weights/flowclas/best_allo.ckpt) | 88.4 | 6.6 |
| Fishyscapes | [config](src/configs/base/cityscapes_coco.yaml) | [config](src/configs/flowclas/flowclas_city.yaml) | [checkpoint](weights/flowclas/best_cityscapes.ckpt) | 88.8 | 0.7 |
| Road Anomaly | [config](src/configs/base/cityscapes_coco.yaml) | [config](src/configs/flowclas/flowclas_city.yaml) | [checkpoint](weights/flowclas/best_cityscapes.ckpt) | 93.0 | 3.3 |

Fishyscapes and Road Anomaly are evaluated with the same Cityscapes-trained [checkpoint](weights/flowclas/best_cityscapes.ckpt); use the inference commands in [Road Anomaly Segmentation](#road-anomaly-segmentation-fishyscapes--road-anomaly) below.

## Training

Replace `<num_devices>`, `<num_workers>`, and `<seed>` with values for your setup.

### ALLO

```bash
python3 trainer_cli.py fit \
  --data configs/base/allo_coco.yaml \
  -c configs/base/base.yaml \
  --trainer.devices <num_devices> \
  --data.num_workers <num_workers> \
  -c configs/flowclas/flowclas_allo.yaml \
  --seed_everything <seed> \
  --experiment.name flowclas_allo \
  --experiment.project_name ALLO
```

### Cityscapes (Road Anomaly / Fishyscapes)

```bash
python3 trainer_cli.py fit \
  --data configs/base/cityscapes_coco.yaml \
  -c configs/base/base.yaml \
  --trainer.devices <num_devices> \
  --data.num_workers <num_workers> \
  -c configs/flowclas/flowclas_city.yaml \
  --seed_everything <seed> \
  --experiment.name flowclas_city \
  --metrics_on_cpu \
  --no_sigmoid
```

### Examples

Paper-style runs using alternate experiment configs (same flag set as above):

**ALLO**

```bash
python3 trainer_cli.py fit \
  --data configs/base/allo_coco.yaml \
  -c configs/base/base.yaml \
  --trainer.devices 4 \
  --data.num_workers 8 \
  -c configs/flowclas/flowclas_allo.yaml \
  --seed_everything 42 \
  --experiment.name flowclas_allo \
  --experiment.project_name flowclas_ALLO
```

**Cityscapes**

```bash
python3 trainer_cli.py fit \
  --data configs/base/cityscapes_coco.yaml \
  -c configs/base/base.yaml \
  --trainer.devices 2 \
  --data.num_workers 8 \
  -c configs/flowclas/flowclas_city.yaml \
  --seed_everything 42 \
  --experiment.name flowclas_city \
  --experiment.project_name flowclas_Cityscapes \
  --metrics_on_cpu \
  --no_sigmoid
```

## Inference

Replace `<num_devices>`, `<num_workers>`, `<seed>`, and `<ckpt>` with values for your setup.

### ALLO

```bash
python3 trainer_cli.py test \
  --data configs/base/allo_coco.yaml \
  -c configs/base/base.yaml \
  --trainer.devices <num_devices> \
  --data.num_workers <num_workers> \
  -c configs/flowclas/flowclas_allo.yaml \
  --seed_everything <seed> \
  --experiment.name flowclas_allo \
  --data.eval_batch_size 2 \
  --experiment.logger [csv] \
  --trainer.precision 32 \
  --ckpt <ckpt>
```

### Road Anomaly Segmentation (Fishyscapes & Road Anomaly)

```bash
python3 trainer_cli.py test \
  --data configs/base/cityscapes_coco.yaml \
  -c configs/base/base.yaml \
  --trainer.devices <num_devices> \
  --data.num_workers <num_workers> \
  -c configs/flowclas/flowclas_city.yaml \
  --seed_everything <seed> \
  --experiment.name flowclas_city \
  --data.eval_batch_size 2 \
  --metrics_on_cpu \
  --no_sigmoid \
  --experiment.logger [csv] \
  --trainer.precision 32 \
  --ckpt <ckpt>
```

### Examples

Camera-ready checkpoints (same flag set as above):

**ALLO**

```bash
python3 trainer_cli.py test \
  --data configs/base/allo_coco.yaml \
  -c configs/base/base.yaml \
  --trainer.devices 1 \
  --data.num_workers 8 \
  -c configs/flowclas/flowclas_allo.yaml \
  --seed_everything 42 \
  --experiment.name flowclas_allo_test \
  --data.eval_batch_size 2 \
  --experiment.logger [csv] \
  --trainer.precision 32 \
  --ckpt ../weights/flowclas/best_allo.ckpt
```

**Road Anomaly Segmentation (Fishyscapes & Road Anomaly)**

```bash
python3 trainer_cli.py test \
  --data configs/base/cityscapes_coco.yaml \
  -c configs/base/base.yaml \
  --trainer.devices 1 \
  --data.num_workers 1 \
  -c configs/flowclas/flowclas_city.yaml \
  --seed_everything 42 \
  --experiment.name flowclas_city_test \
  --data.eval_batch_size 2 \
  --metrics_on_cpu \
  --no_sigmoid \
  --experiment.logger [csv] \
  --trainer.precision 32 \
  --ckpt ../weights/flowclas/best_cityscapes.ckpt
```

## Citation

If you use FlowCLAS in your research, please cite:

```bibtex
@inproceedings{lee2026flowclas,
  title={FlowCLAS: Enhancing Normalizing Flow-Based Anomaly Segmentation Via Contrastive Learning},
  author={Lee, Chang Won and Leveugle, Selina and Grouchy, Paul and Langley, Chris and Stolpner, Svetlana and Kelly, Jonathan and Waslander, Steven L},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={6998--7007},
  year={2026}
}
```
