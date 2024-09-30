_base_ = '../imagenet/imagenet_swin_ft_sz64_1xbs1024.py'

# dataset settings
data_source_cfg = dict(type='ImageNet')
# ImageNet100 dataset, 100 class
data_train_list = 'data/imagenet100/train.txt'
data_train_root = 'data/imagenet100/train'
data_test_list = 'data/imagenet100/val.txt'
data_test_root = 'data/imagenet100/val'

data = dict(
    train=dict(
        data_source=dict(
            list_file=data_train_list, root=data_train_root, **data_source_cfg),
    ),
    val=dict(
        data_source=dict(
            list_file=data_test_list, root=data_test_root, **data_source_cfg),
    ))

# checkpoint
checkpoint_config = dict(interval=1, max_keep_ckpts=3)
