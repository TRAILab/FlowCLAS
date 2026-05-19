"""COCO auxiliary lists used when pasting OOD ``train_id_out`` regions into driving scenes."""

from pathlib import Path
import json

import numpy as np


class COCO:
    """Paths to COCO ``ood_seg_{split}`` masks and matching RGB images (optional class-based filtering)."""

    train_id_in = 0
    train_id_out = 254

    ALLO_COCO_OVERLAP_CLASSES = [
        "microwave", "oven", "toaster", "refrigerator", "scissors", "hair drier",
    ]
    VISTAS_COCO_OVERLAP_CLASSES = [
        "person", "bicycle", "motorcycle", "car", "bus", "truck", "boat", "traffic light",
        "fire hydrant", "bench", "bird", "train", "stop sign", "parking meter", "potted plant",
        "airplane",
    ]
    ROAD_ANOMALY_COCO_OVERLAP_CLASSES = (
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
        "backpack", "frisbee", "skis", "snowboard", "umbrella", "sports ball", "bottle",
        "potted plant", "vase",
    )

    def __init__(self, root: Path | str, split: str = "train", filter: str | None = None) -> None:
        """Load targets from ``ood_seg_{split}2017`` and optionally drop images with COCO overlap."""
        self.root = Path(root)

        assert split in ["train", "val"], f"Invalid split: {split}"
        self.split = split + "2017"
        target_dir = self.root / f"ood_seg_{self.split}"
        targets = [str(t) for t in target_dir.glob("*.png")]

        if filter is not None:
            if filter == "allo":
                overlap_classes = self.ALLO_COCO_OVERLAP_CLASSES
            elif filter == "vistas":
                overlap_classes = self.VISTAS_COCO_OVERLAP_CLASSES
            elif filter == "road_anomaly":
                overlap_classes = self.ROAD_ANOMALY_COCO_OVERLAP_CLASSES
            else:
                raise ValueError(f"Invalid filter for COCO outlier set: {filter}")

            ann_path = self.root / "annotations" / f"instances_{self.split}.json"
            with ann_path.open(encoding="utf-8") as f:
                annotations = json.load(f)
            cat2id = {cat["name"]: cat["id"] for cat in annotations["categories"]}
            overlap_cat_ids = [cat2id[cat] for cat in overlap_classes]

            imgid2catid: dict[int, list[int]] = {}
            for ann in annotations["annotations"]:
                if ann["image_id"] not in imgid2catid:
                    imgid2catid[ann["image_id"]] = [ann["category_id"]]
                else:
                    imgid2catid[ann["image_id"]].append(ann["category_id"])

            self.images, self.targets = [], []
            for tgt in targets:
                file_id = int(Path(tgt).stem.lstrip("0"))
                cats = imgid2catid[file_id]

                if not any(cat in overlap_cat_ids for cat in cats):
                    self.targets.append(tgt)
                    img_path = tgt.replace("ood_seg_", "").replace("png", "jpg")
                    self.images.append(img_path)
        else:
            self.targets = targets
            self.images = [t.replace("ood_seg_", "").replace("png", "jpg") for t in self.targets]

    def get_idx(self) -> int:
        """Uniform random index into ``images`` / ``targets``."""
        return int(np.random.randint(0, len(self.images)))
