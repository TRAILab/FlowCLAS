"""Lightning module for FlowCLAS: normalizing flows on frozen backbone features."""

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.utilities import rank_zero_warn
from lightning.pytorch.utilities.types import STEP_OUTPUT
from sam2.build_sam import build_sam2

from core.models.components.anomaly import AnomalyLikelihoodMap, AnomalyLogLikelihoodMap
from core.models.components.base import BaseModule
from core.models.components.losses import MixOODContrastLoss
from core.models.components.utils.sam2 import SAM2AutomaticMaskGenerator
from core.utils import LearningType
from core.utils.evaluation import accuracy

from .flowclas_model import FlowCLASModel
from .losses import FlowCLASNLLLoss


class FlowCLAS(BaseModule):
    """Train and evaluate FlowCLAS: coupling flows on DINOv2-style features with optional contrastive loss."""

    def __init__(
        self,
        backbone: str = "dinov2_vitb14",
        backbone_cfg: dict[str, Any] = {},
        proj_dim: int = 256,
        num_classes: int = 2,
        backbone_ckpt: str | None = None,
        pretrained: bool = True,
        flow_steps: int = 8,
        ood_index: int = 254,
        ignore_index: int = 255,
        alpha: float = 1.0, #* weight for nll loss
        temperature: float = 0.10,
        use_gaussian: bool = False,
        use_logp_score: bool = False,
        use_sam: bool = False,
        alpha_annealing: bool = False,
        subnet_type: str = "simple",
        flow_type: str = "simple",
        no_ood: bool = False,
        normal_only: bool = False,
        use_seg: bool = False,
        slide_inf_cfg: dict[str, Any] = {},
        layers: list[int] = [0, 1, 2, 3],
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.backbone = backbone
        self.backbone_cfg = backbone_cfg
        self.proj_dim = proj_dim
        self.backbone_ckpt = backbone_ckpt
        self.pretrained = pretrained
        self.flow_steps = flow_steps
        self.use_gaussian = use_gaussian
        self.ood_index = ood_index
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.subnet_type = subnet_type
        self.flow_type = flow_type
        self.alpha_annealing = alpha_annealing
        self.normal_only = normal_only
        self.use_seg = use_seg
        self.num_classes = num_classes
        if no_ood:
            raise ValueError(
                "no_ood=True is not supported for FlowCLAS under shipped experiment configs (they all use no_ood: False)."
            )
        self.nll_loss = FlowCLASNLLLoss(ignore_index=ignore_index)
        self.con_loss = MixOODContrastLoss(
            temperature=temperature,
            ood_idx=1,
            ignore_idx=ignore_index,
        )
        self.ce_loss = None
        self.lovasz_loss = None
        if use_seg:
            from core.models.components.losses import CESegLoss, LovaszLoss

            self.ce_loss = CESegLoss(ignore_index=ignore_index)
            self.lovasz_loss = LovaszLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.model: FlowCLASModel

        if use_logp_score:
            self.anomaly_map: AnomalyLogLikelihoodMap
        else:
            self.anomaly_map: AnomalyLikelihoodMap
        self.use_logp_score = use_logp_score
        self.use_sam = use_sam

        self.slide_inf_cfg = slide_inf_cfg
        self.layers = layers

    def _setup(self) -> None:
        self.model = FlowCLASModel(
            backbone=self.backbone,
            backbone_cfg=self.backbone_cfg,
            proj_dim=self.proj_dim,
            num_classes=self.num_classes,
            backbone_ckpt=self.backbone_ckpt,
            pretrained=self.pretrained,
            flow_steps=self.flow_steps,
            subnet_type=self.subnet_type,
            flow_type=self.flow_type,
            use_seg=self.use_seg,
            layers=self.layers,
        )
        if self.use_logp_score:
            self.anomaly_map = AnomalyLogLikelihoodMap(use_gaussian=self.use_gaussian)
        else:
            self.anomaly_map = AnomalyLikelihoodMap(use_gaussian=self.use_gaussian)

        if self.use_sam:
            rank_zero_warn("Using SAM 2 for mask generation. Turning off gaussian post-processing.")
            self.use_gaussian = False
            sam2_checkpoint = "../misc/weights/sam2.1_hiera_large.pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
            sam2_model = build_sam2(model_cfg, sam2_checkpoint)
            self.mask_generator = SAM2AutomaticMaskGenerator(sam2_model, 
                                                                points_per_side=32,
                                                                points_per_batch=128)

    def _compute_alpha(self) -> float:
        if self.alpha_annealing:
            #* Cosine annealing
            T_max = self.trainer.estimated_stepping_batches
            return 0.1 * self.alpha + 0.9 * self.alpha / 2 * (1 + np.cos(np.pi * self.trainer.global_step / T_max))
        return self.alpha

    def training_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Perform the training step input and return the loss.

        Args:
            batch (batch: dict[str, str | torch.Tensor]): Input batch
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            STEP_OUTPUT: Dictionary containing the loss value.
        """
        del args, kwargs  # These variables are not used.
        
        if self.normal_only:
            image, mask = batch["id_image"], batch["id_mask"]
            
            #? Remap labels to [0, 1] binary mask
            mask[mask != self.ignore_index] = 0    #* inlier
            assert torch.isin(mask.unique(), torch.tensor([0, 255], device=mask.device)).all().item(), (
                f"Allowed values in `id_mask` are [0, 255]. Got: {mask.unique()}"
            )

            out = self.model(image)
            loss = self.alpha * self.nll_loss(out["log_jacobians"], out["log_p"], mask)
        else:
            mix_image, mix_mask = batch["mix_image"], batch["mix_mask"]
            ood_image, ood_mask = batch["ood_image"], batch["ood_mask"]

            if self.use_seg:
                #* Segmentation mask for inlier and ood
                seg_mask = mix_mask.clone()
                #* Assume last class idx is anomaly
                seg_mask[seg_mask == self.ood_index] = self.ignore_index
            
            #? Remap labels to [0, 1] for inlier and ood respectively
            for mask in [mix_mask, ood_mask]:
                mask[(mask != self.ood_index) & (mask != self.ignore_index)] = 0    #* inlier
                mask[mask == self.ood_index] = 1
                assert torch.isin(mask.unique(), torch.tensor([0, 1, 255], device=mask.device)).all().item(), \
                    f"Allowed values in `mix_mask` are [0, 1, 255]. Got: {mask.unique()}"

            image = torch.cat([mix_image, ood_image], dim=0)
            half_batch_size = mix_image.shape[0]

            out = self.model(image)

            #* NLL Loss for inlier and ood
            jacobians = [jac[:half_batch_size] for jac in out["log_jacobians"]]
            log_ps = [log_p[:half_batch_size] for log_p in out["log_p"]]
            nll_loss = self.nll_loss(jacobians, log_ps, mix_mask)
            self.log("nll_loss", nll_loss.detach(), on_step=True, on_epoch=False, sync_dist=True, prog_bar=True, logger=True)

            #* Contrastive Loss
            con_loss = torch.tensor(0.0, device=image.device)
            for i, z in enumerate(out["z"]):
                z_mix = z[:half_batch_size]  #* (B, C, H, W)
                z_ood = z[half_batch_size:]  #* (B, C, H, W)
                con_loss += self.con_loss(z_mix, mix_mask, z_ood, ood_mask)
            self.log("contrastive_loss", con_loss.detach(), on_step=True, on_epoch=False, sync_dist=True, prog_bar=True, logger=True)

            alpha = self._compute_alpha()
            loss = alpha * nll_loss + con_loss

            #? Segmentation in flow latent space
            if self.use_seg:
                seg_logits = out["seg_logits"][-1][:half_batch_size]    #* mix only
                seg_logits = F.interpolate(
                    seg_logits,
                    size=tuple(seg_mask.shape[1:]),
                    mode="bilinear",
                    align_corners=False,
                    )   #* (B, num_classes, H, W)
                acc_seg = accuracy(seg_logits, seg_mask).item()
                lovasz_loss = self.lovasz_loss(seg_logits, seg_mask)
                ce_loss = self.ce_loss(seg_logits, seg_mask)
                self.log("acc_seg", acc_seg, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True, logger=True)
                self.log("lovasz_loss", lovasz_loss.detach(), on_step=True, on_epoch=False, sync_dist=True, prog_bar=True, logger=True)
                self.log("ce_loss", ce_loss.detach(), on_step=True, on_epoch=False, sync_dist=True, prog_bar=True, logger=True)
                seg_loss = 0.5 * lovasz_loss + 0.5 * ce_loss
                loss += seg_loss

        self.log("train_loss", loss.detach(), on_step=True, on_epoch=False, sync_dist=True, prog_bar=True, logger=True)
        self.prev_loss = loss.detach()
        
        return {"loss": loss}

    def slide_inference(self, input: torch.Tensor) -> torch.Tensor:
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.

        Returns:
            torch.Tensor: The segmentation results, seg_logits from model of each
                input image and the ood_logits from the model of each input image.
        """
        assert "crop_size" in self.slide_inf_cfg, "crop_size must be set in slide_inf_cfg"
        assert "stride" in self.slide_inf_cfg, "stride must be set in slide_inf_cfg"

        h_stride, w_stride = self.slide_inf_cfg["stride"]
        h_crop, w_crop = self.slide_inf_cfg["crop_size"]
        batch_size, _, h_img, w_img = input.shape
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        anomaly_maps = input.new_zeros((batch_size, 1, h_img, w_img))
        count_mat = input.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = input[:, :, y1:y2, x1:x2]
                crop_anomaly_map = self.anomaly_map(self.model(crop_img), shape=crop_img.shape[-2:])
                anomaly_maps += F.pad(crop_anomaly_map,
                               (int(x1), int(anomaly_maps.shape[3] - x2), int(y1),
                                int(anomaly_maps.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        anomaly_maps = anomaly_maps / count_mat

        return anomaly_maps

    def validation_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Perform the validation step and return the anomaly map.

        Args:
            batch (dict[str, str | torch.Tensor]): Input batch
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            STEP_OUTPUT | None: batch dictionary containing anomaly-maps.
        """
        del args, kwargs  #* These variables are not used.
        ori_shape = batch["original_shape"][0].tolist()

        with torch.no_grad():
            if self.slide_inf_cfg.get("enable", False):
                anomaly_map = self.slide_inference(batch["image"])
            else:
                out = self.model(batch["image"])
                anomaly_map = self.anomaly_map(out, shape=batch["image"].shape[-2:])

            batch["anomaly_maps"] = self.crop_padding(anomaly_map, ori_shape)
            batch["image"] = self.crop_padding(batch["image"], ori_shape)
            batch["mask"] = self.crop_padding(batch["mask"], ori_shape)

            if not self.slide_inf_cfg.get("enable", False):
                batch.update(out)
            
            if self.use_sam:
                batch["anomaly_maps"] = self.post_process_w_sam_masks(batch["sam_image"], 
                                                                      batch["anomaly_maps"],
                                                                      batch['original_shape'][0].tolist())
        return batch

    def predict_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Forward-pass the input and return the anomaly map and intermediate feature maps.

        Args:
            batch (dict[str, str | Tensor]): Input batch

        Returns:
            STEP_OUTPUT | None: batch dictionary containing anomaly-maps and other outputs.
        """
        del args, kwargs  # These variables are not used.
        ori_shape = batch["original_shape"][0].tolist()
        
        with torch.no_grad():
            out = self.model(batch["image"])
            anomaly_map = self.anomaly_map(out, shape=batch["image"].shape[-2:])
            batch["anomaly_maps"] = self.crop_padding(anomaly_map, ori_shape)
            batch["image"] = self.crop_padding(batch["image"], ori_shape)
            if "mask" in batch:
                batch["mask"] = self.crop_padding(batch["mask"], ori_shape)
            batch.update(out)
            
            if self.use_sam:
                batch["anomaly_maps"] = self.post_process_w_sam_masks(batch["sam_image"], 
                                                                      batch["anomaly_maps"],
                                                                      batch['original_shape'][0].tolist())
        return batch

    def post_process_w_sam_masks(self, 
                                 image: torch.Tensor, 
                                 anomaly_map: torch.Tensor,
                                 ori_shape: list[int, int]) -> torch.Tensor:
        """Post-process the anomaly map using SAM 2 masks.
        Args:
            image (torch.Tensor): Transformed input image for SAM 2 mask generation.
            anomaly_map (torch.Tensor): Anomaly map (B, 1, H, W).
            ori_shape (list[int, int]): Original shape of the image [h, w].
        """
        B = image.shape[0]
        for i in range(B):
            image_i = image[i].unsqueeze(0)

            #? Get SAM 2 masks
            mask_list = self.mask_generator.generate(image_i, ori_shape)[0]
            
            for instance in mask_list:
                mask = torch.from_numpy(instance["segmentation"])[None, ...].to(anomaly_map.device) #* (1, H, W)
                
                #? Update anomaly map with mean score of highest frequency histogram for each mask
                num_bins = 20
                scores = anomaly_map[i][mask.bool()].cpu().numpy()
                bins = np.linspace(scores.min(), scores.max(), num_bins+1)
                hist, edges = np.histogram(scores, bins=bins)
                mode_bin_index = np.argmax(hist)
                mode_min, mode_max = edges[mode_bin_index], edges[mode_bin_index+1]
                mode_bin_scores = scores[(scores >= mode_min) 
                                            & (scores < mode_max)]
                if len(mode_bin_scores) > 0:
                    anomaly_map[i, mask & ((anomaly_map[i] < mode_min) \
                                        | (anomaly_map[i] > mode_max))] = torch.tensor(mode_bin_scores.mean(), device=anomaly_map.device)

        return anomaly_map

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Return FlowCLAS trainer arguments. Overrides arguments from CLI."""
        return {"num_sanity_val_steps": 0}

    def param_groups(self):
        custom_param_groups = {"nowd": [], "other": []}
        norm_modules = [nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm3d, nn.GroupNorm, nn.LayerNorm]

        for name, param in self.named_parameters():
            if any(["bias" in name, 
                    "norm" in name, 
                    "flow" in name,
                    type(param) in norm_modules]):
                custom_param_groups["nowd"].append(param)
            else:
                custom_param_groups["other"].append(param)
        
        lr = self.optimizer_cfg.init_args.lr
        param_groups = [
            {"name": "nowd", "params": custom_param_groups["nowd"], "weight_decay": 0.0, "lr_scale": 1.0},
            {"name": "other", "params": custom_param_groups["other"], "lr_scale": 1.0},
        ]
        return param_groups

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type of the model.

        Returns:
            LearningType: Learning type of the model.
        """
        return LearningType.ONE_CLASS

    @property
    def score_range(self) -> tuple[float, float]:
        """Return the range of anomaly scores.

        Returns:
            tuple[float, float]: Range of anomaly scores.
        """
        return (0.0, 1.0)
