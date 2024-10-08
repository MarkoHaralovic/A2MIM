_base_ = [
    '../_base_/models/convnext_nano.py',
    '../_base_/datasets/imagenet100/imagenet100_swin_ft_sz64_1xbs256.py',
    '../_base_/default_runtime.py',
]

# model settings
model = dict(
    head=dict(
        loss=dict(type='LabelSmoothLoss',
            label_smooth_val=0.1, num_classes=100, mode='original', loss_weight=1.0),
        num_classes=100))


# additional hooks
update_interval = 2  # total: 1 x 256 x 2 accumulates = bs512

# additional hooks
custom_hooks = [
    dict(type='EMAHook',  # EMA_W = (1 - m) * EMA_W + m * W
        momentum=0.9999,
        warmup='linear',
        warmup_iters=20 * 626, warmup_ratio=0.9,  # warmup 20 epochs.
        update_interval=update_interval,
    ),
]

# optimizer
optimizer = dict(
    type='AdamW',
    lr=4e-4,  # lr = 0.0004 / bs4096 for fine-tuning
    weight_decay=0.01, eps=1e-8, betas=(0.9, 0.999),
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'norm': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'gamma': dict(weight_decay=0.),
    },
    constructor='TransformerFinetuneConstructor',
    model_type='convnext',
    layer_decay=0.7)

# fp16
use_fp16 = True
fp16 = dict(type='mmcv', loss_scale='dynamic')
optimizer_config = dict(grad_clip=None, update_interval=update_interval)

# lr scheduler
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False, min_lr=1e-6,
    warmup='linear',
    warmup_iters=5, warmup_by_epoch=True,
    warmup_ratio=1e-5,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=300)
