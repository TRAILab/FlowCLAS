

import logging
from functools import partial

import torch
from dinov2.models.vision_transformer import DinoVisionTransformer
from mmengine.runner.checkpoint import _load_checkpoint

from .reins import Reins, LoRAReins
from .dinov2_layers import NestedTensorBlock as Block, MemEffAttention

logger = logging.getLogger(__name__)


class ReinsDinoV2(DinoVisionTransformer):

    _MODELS = {
        'l': {
            'embed_dim': 1024,
            'depth': 24,
            'num_heads': 16,
            'mlp_ratio': 4,
            'out_indices': [7, 11, 15, 23]
        },
        'b': {
            'embed_dim': 768,
            'depth': 12,
            'num_heads': 12,
            'mlp_ratio': 4,
            'out_indices': [2, 5, 8, 11]
        }
    }
    def __init__(self, model: str,ckpt: str) -> None:
        model_cfg = self._MODELS[model]
        super().__init__(
            img_size=518,
            patch_size=14,
            embed_dim=model_cfg['embed_dim'],
            depth=model_cfg['depth'],
            num_heads=model_cfg['num_heads'],
            mlp_ratio=model_cfg['mlp_ratio'],
            init_values= 1.0,
            ffn_layer="mlp",
            block_chunks=0,
            block_fn=partial(Block, attn_class=MemEffAttention),
            num_register_tokens=0,
            interpolate_antialias=False,
            interpolate_offset=0.1
        )

        #? load pretrained weights
        try:
            
            state_dict = _load_checkpoint(f"../misc/hub/checkpoints/dinov2_vit{model}14_pretrain.pth", logger=logger, map_location='cpu')
        except:
            torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model}14')
            state_dict = _load_checkpoint(f"../misc/hub/checkpoints/dinov2_vit{model}14_pretrain.pth", logger=logger, map_location='cpu')
        self.load_state_dict(state_dict, strict=True)
        logger.info("Loaded pretrained weights for DINOv2.")
        print("Loaded pretrained weights for DINOv2.")

        try:
            reins_state_dict = _load_checkpoint(ckpt, logger=logger, map_location='cpu')
        except:
            reins_state_dict = torch.load(ckpt, map_location='cpu', weights_only=False)
        if "state_dict" in reins_state_dict:
            reins_state_dict = reins_state_dict["state_dict"]
        reins_state_dict = {k.replace("backbone.reins.", ""): v for k, v in reins_state_dict.items() if k.startswith("backbone.reins")}
        self.reins = LoRAReins(
            embed_dims=model_cfg['embed_dim'],
            link_token_to_query=True,
            lora_dim=model_cfg['num_heads'],
            num_layers=model_cfg['depth'],
            patch_size=14,
            token_length=100
        )
        self.reins.load_state_dict(reins_state_dict, strict=True)
        logger.info("Loaded pretrained weights for Reins.")
        print("Loaded pretrained weights for Reins.")
        self.out_indices = model_cfg['out_indices']

    def forward(self, x, masks=None):
        B, _, h, w = x.shape

        #* check image dims divisible by patch_size
        assert (h % self.patch_size) == 0
        assert (w % self.patch_size) == 0

        H, W = h // self.patch_size, w // self.patch_size
        x = self.prepare_tokens_with_masks(x, masks)
        outs = []
        for idx, blk in enumerate(self.blocks):
            # if idx < len(self.blocks) - 1:
            #     x = blk(x)
            # else:
            #     # return attention of the last block
            #     x, attn = blk(x, return_attention=True)
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