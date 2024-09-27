_base_ = [
    '../../_base_/models/a2mim/convnext_f.py',
    '../../_base_/datasets/imagenet/a2mim_rgb_m_sz64_rrc08_bs256.py',
    '../../_base_/default_runtime.py',
]

# model settings
model = dict(
    backbone=dict(
        mask_layer=3, mask_token="learnable",
        mask_init=1e-6,  # init residual gamma
    ),
    head=dict(
        fft_weight=0., fft_focal=False,
    ),
)

# dataset
data = dict(
    imgs_per_gpu=1024, workers_per_gpu=4,
    train=dict(
        feature_mode=None, feature_args=dict(),
        mask_pipeline=[
            dict(type='BlockwiseMaskGenerator',
                input_size=64, mask_patch_size=32, mask_ratio=0.6, model_patch_size=32,  # stage 3
                mask_color='mean', mask_only=False),
        ],
))

# interval for accumulate gradient
update_interval = 1  # bs1024 x 1gpus = bs1024

# additional hooks
custom_hooks = [
    dict(type='SAVEHook',
        save_interval=626 * 1,  # plot every 1 ep
        iter_per_epoch=626),
]

# optimizer
optimizer = dict(
    type='AdamW',
    lr=3e-4 * 1024 / 512,  # 3e-4 * 4 for bs2048
    betas=(0.9, 0.999), weight_decay=0.05, eps=1e-8,
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'gamma': dict(weight_decay=0.),
        'mask_token': dict(weight_decay=0., lr_mult=1e-1,),
        'mask_gamma': dict(weight_decay=0., lr_mult=1e-1,),
    })

# fp16
use_fp16 = True
fp16 = dict(type='mmcv', loss_scale='dynamic')
# optimizer args
optimizer_config = dict(update_interval=update_interval)

# lr scheduler
lr_config = dict(
    policy='StepFixCosineAnnealing',
    by_epoch=False, min_lr=1e-5,
    warmup='linear',
    warmup_iters=10, warmup_by_epoch=True,
    warmup_ratio=1e-6,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=600)
