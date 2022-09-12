from .benchmarking_mask_rcnn_base_FPN_100ep_LSJ_mae import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)

train.max_iter = train.max_iter // 4  # 100ep -> 25ep

lr_multiplier.warmup_length *= 4

__all__ = ["dataloader", "lr_multiplier", "model", "optimizer", "train"]


model.backbone.bottom_up.pretrained = "mcbeit_pretrained/checkpoint-799.pth" # mae pretrained weight does not need convert
model.backbone.bottom_up.stop_grad_conv1 = False
model.backbone.bottom_up.sincos_pos_embed = False
model.backbone.bottom_up.zero_pos_embed= True # z`zero init pos embed
model.backbone.bottom_up.init_values = 0.1 # 0.1 for base, 1e-5 for large
model.backbone.bottom_up.beit_qkv_bias = True


train.output_dir = 'work_dirs/mask_rcnn_base_FPN_25ep_LSJ_mc-beit'