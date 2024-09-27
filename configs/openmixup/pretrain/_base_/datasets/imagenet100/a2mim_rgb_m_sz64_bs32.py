_base_ = '../imagenet/a2mim_rgb_m_sz64_bs32.py'

# dataset settings
data_source_cfg = dict(type='ImageNet', return_label=False)
# ImageNet dataset, 100 class
data_train_list = 'data/imagenet100/train.txt'
data_train_root = 'data/imagenet100/train'

# dataset summary
data = dict(
    train=dict(
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            **data_source_cfg),
    ))
