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

Paths in the YAML configs below are relative to `src/` (for example, `../data/cityscapes` → `~/data/cityscapes` inside the container).

## Dataset preparation

### Cityscapes

1. Register and download the **gtFine** split (and matching **leftImg8bit** images) from the [Cityscapes dataset](https://www.cityscapes-dataset.com/).
2. **(Recommended)** Preprocess labels to the common **19-class** `trainId` convention using [cityscapesScripts](https://github.com/mcordts/cityscapesScripts) — install the package, then run [`createTrainIdLabelImgs.py`](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/createTrainIdLabelImgs.py) on your download. FlowCLAS remaps raw `labelIds` to 19 training classes in code; generating `*_gtFine_labelTrainIds.png` keeps your tree aligned with the standard Cityscapes tooling.
3. Set `city_root` in [`src/configs/base/cityscapes_coco.yaml`](src/configs/base/cityscapes_coco.yaml) to the dataset root that contains `leftImg8bit/` and `gtFine/` (Torchvision layout).

Expected layout:

```text
cityscapes/
├── leftImg8bit/
│   ├── train/<city>/*.png
│   └── val/<city>/*.png
└── gtFine/
    ├── train/<city>/*_gtFine_labelIds.png
    └── val/<city>/*_gtFine_labelIds.png
```

### ALLO

A new version of the ALLO dataset will be released soon. For the current benchmark, set `allo_root`, `train_dir`, and `test_dir` in [`src/configs/base/allo_coco.yaml`](src/configs/base/allo_coco.yaml).

### COCO (outlier exposure)

COCO images are used as an OoD proxy during training (paste/mix with in-distribution scenes). You can either prepare them yourself or use the preprocessed archive on [Google Drive](https://drive.google.com/drive/folders/1snL0FNDnmvEEdFQozC2fjsES3Iamms_n?usp=sharing) (same folder as Fishyscapes and Road Anomaly).

**Option A — prepare from scratch** ([Meta-OoD](https://github.com/robin-chan/meta-ood)):

1. Download [COCO 2017 train](https://cocodataset.org/#download) images and instance annotations.
2. Run [`preparation/prepare_coco_segmentation.py`](https://github.com/robin-chan/meta-ood/blob/master/preparation/prepare_coco_segmentation.py) to build binary `ood_seg` masks for images without instances that overlap Cityscapes train classes.

**Option B — download preprocessed data** from the [Google Drive](https://drive.google.com/drive/folders/1snL0FNDnmvEEdFQozC2fjsES3Iamms_n?usp=sharing) folder and extract under your `coco_root`.

In both cases, set `coco_root` in the data config you use ([`cityscapes_coco.yaml`](src/configs/base/cityscapes_coco.yaml) and/or [`allo_coco.yaml`](src/configs/base/allo_coco.yaml)).

Expected layout:

```text
coco/
├── annotations/
│   ├── instances_train2017.json
│   └── ood_seg_train2017/          # optional; masks may also live at repo root
├── ood_seg_train2017/*.png         # required by FlowCLAS loaders
└── train2017/*.jpg
```

You can override any root on the CLI, for example:

```bash
--data.init_args.city_root /path/to/cityscapes \
--data.init_args.coco_root /path/to/coco
```

### Other benchmarks (preprocessed)

Preprocessed **Fishyscapes**, **Road Anomaly**, and **COCO** (see above) are available on [Google Drive](https://drive.google.com/drive/folders/1snL0FNDnmvEEdFQozC2fjsES3Iamms_n?usp=sharing). After extracting, point the paths in [`cityscapes_coco.yaml`](src/configs/base/cityscapes_coco.yaml):

| Config key | Role |
| --- | --- |
| `fishy_root` | Fishyscapes **Lost & Found** split (`images/`, `labels/`) |
| `roadanomaly_root` | Road Anomaly frames and semantic labels |
| `smiyc_root` | Segment Me If You Can (AnomalyTrack + ObstacleTrack) |

Expected layouts (as used by this repo):

```text
fishyscapes/
├── LostAndFound/
│   ├── images/*.png
│   └── labels/*.png
└── Static/
    ├── images/*.png
    └── labels/*.png

RoadAnomaly/
└── frames/
    ├── <clip>.jpg
    └── <clip>.labels/
        └── labels_semantic.png

smiyc/
├── dataset_AnomalyTrack/
│   ├── images/
│   └── labels_masks/
└── dataset_ObstacleTrack/
    ├── images/
    └── labels_masks/
```

Evaluation uses `fishy_root` for Fishyscapes (default: `../data/fishyscapes/LostAndFound`), plus `smiyc_root` and `roadanomaly_root` for the other test benchmarks in the Cityscapes training pipeline.

## Weights and checkpoints

Pre-trained backbones, SAM2, and FlowCLAS checkpoints are available on [Google Drive](https://drive.google.com/drive/folders/17JWOVX-5sauN_qdM3QfgXqpRE1FEQvwh?usp=sharing). After downloading, place files under the repo root as follows (paths are relative to `src/` in the configs):

| File | Destination |
| --- | --- |
| `rein_dinov2l_allo.pth` | `weights/rein_dinov2l_allo.pth` |
| `rein_dinov2l_city.pth` | `weights/rein_dinov2l_city.pth` |
| `sam2.1_hiera_large.pt` | `misc/weights/sam2.1_hiera_large.pt` |
| `best_allo.ckpt` | `weights/flowclas/best_allo.ckpt` |
| `best_cityscapes.ckpt` | `weights/flowclas/best_cityscapes.ckpt` |

Training reads `backbone_ckpt` from [`flowclas_allo.yaml`](src/configs/flowclas/flowclas_allo.yaml) / [`flowclas_city.yaml`](src/configs/flowclas/flowclas_city.yaml); inference passes `--ckpt` to a FlowCLAS checkpoint under `weights/flowclas/`.

## Results

Pre-trained checkpoints and pixel-level metrics on the official test splits (checkpoints from [Google Drive](https://drive.google.com/drive/folders/17JWOVX-5sauN_qdM3QfgXqpRE1FEQvwh?usp=sharing)):

| Benchmark | Data config | Model config | Checkpoint | Pixel AP | Pixel FPR95 |
| --- | --- | --- | --- | ---: | ---: |
| ALLO | [config](src/configs/base/allo_coco.yaml) | [config](src/configs/flowclas/flowclas_allo.yaml) | `best_allo.ckpt` | 88.4 | 6.6 |
| Fishyscapes | [config](src/configs/base/cityscapes_coco.yaml) | [config](src/configs/flowclas/flowclas_city.yaml) | `best_cityscapes.ckpt` | 88.8 | 0.7 |
| Road Anomaly | [config](src/configs/base/cityscapes_coco.yaml) | [config](src/configs/flowclas/flowclas_city.yaml) | `best_cityscapes.ckpt` | 93.0 | 3.3 |

Fishyscapes and Road Anomaly are evaluated with the same Cityscapes-trained `best_cityscapes.ckpt`; use the inference commands in [Road Anomaly Segmentation](#road-anomaly-segmentation-fishyscapes--road-anomaly) below.

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

Download checkpoints from [Google Drive](https://drive.google.com/drive/folders/17JWOVX-5sauN_qdM3QfgXqpRE1FEQvwh?usp=sharing) first (see [Weights and checkpoints](#weights-and-checkpoints)). Example inference commands (same flag set as above):

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
