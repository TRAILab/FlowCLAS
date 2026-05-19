

import logging

import torch
from mmengine.runner.checkpoint import _load_checkpoint

from .reins import LoRAReins
from .dino_v2 import DINOv2

logger = logging.getLogger(__name__)


class ReinsDinoV2Custom(DINOv2):
    def __init__(self, ckpt: str, **kwargs) -> None:
        super().__init__(
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4,
            ffn_layer="mlp",
            init_values=1e-05,
            block_chunks=0,
            qkv_bias=True,
            proj_bias=True,
            ffn_bias=True,
            **kwargs
        )
        assert "patch_size" in kwargs, "patch_size must be provided for Rein with DINOv2."
        assert "img_size" in kwargs, "img_size must be provided for Rein with DINOv2."
        patch_size = kwargs["patch_size"]
        
        self.reins = LoRAReins(
            embed_dims=1024,
            link_token_to_query=True,
            lora_dim=16,
            num_layers=24,
            patch_size=patch_size,
            token_length=100
        )
        self.out_indices = [7, 11, 15, 23]
        
        #? load pretrained weights
        state_dict = _load_checkpoint(ckpt, logger=logger, map_location='cpu')
        self.load_state_dict(state_dict, strict=True)

    def forward(self, x, masks=None):
        B, _, h, w = x.shape

        #* check image dims divisible by patch_size
        assert (h % self.patch_size) == 0
        assert (w % self.patch_size) == 0

        H, W = h // self.patch_size, w // self.patch_size
        x = self.prepare_tokens_with_masks(x, masks)
        outs = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            x = self.reins.forward(
                x,
                idx,
                batch_first=True,
                has_cls_token=True,
            )
            if idx in self.out_indices:
                outs.append(
                    x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
                )
        reins_out = self.reins.return_auto(outs)
        return reins_out[0][-1] #* return the last layer output