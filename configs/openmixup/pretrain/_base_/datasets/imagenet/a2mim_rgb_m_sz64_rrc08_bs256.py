# dataset settings
data_source_cfg = dict(type='ImageNet', return_label=False)
# ImageNet dataset
data_train_list = 'data/meta/ImageNet/train_full.txt'
data_train_root = 'data/ImageNet/train'

dataset_type = 'MaskedImageDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=64, scale=(0.8, 1.0), ratio=(3. / 4., 4. / 3.)),
    dict(type='RandomHorizontalFlip'),
]
train_mask_pipeline = [
    dict(type='BlockwiseMaskGenerator',
        input_size=64, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6,
        mask_only=False, mask_color='mean'),
]

# prefetch
prefetch = False  # turn off prefetch when use RGB (`mean` filling).
if not prefetch:
    train_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])

# dataset summary
data = dict(
    imgs_per_gpu=256, 
    workers_per_gpu=4,  # according to total cpus cores, usually 4 workers per 32~128 imgs
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            **data_source_cfg),
        pipeline=train_pipeline,
        mask_pipeline=train_mask_pipeline,
        feature_mode=None, feature_args=dict(),
        prefetch=prefetch))

# checkpoint
checkpoint_config = dict(interval=1, max_keep_ckpts=3)
