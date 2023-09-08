_base_ = './yolov5_ins_s-v61_syncbn_fast_8xb16-300e_coco_instance.py'  # noqa

data_root = 'C:/DeepSense/data/' # dataset root
# Training set annotation file of json path
train_ann_file = 'train1.json'
train_data_prefix = 'images/'  # Dataset prefix
# Validation set annotation file of json path
val_ann_file = 'val1.json'
val_data_prefix = 'images/'
metainfo = {
    'classes': ('fish-tag', ), # dataset category name
    'palette': [
        (0, 0, 255),
    ]
}
num_classes = 1
# Set batch size to 4
train_batch_size_per_gpu = 4
# dataloader num workers
train_num_workers = 2
log_interval = 1
#####################
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img=train_data_prefix),
        ann_file=train_ann_file))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img=val_data_prefix),
        ann_file=val_ann_file))
test_dataloader = val_dataloader
val_evaluator = dict(ann_file=data_root + val_ann_file)
test_evaluator = val_evaluator
default_hooks = dict(logger=dict(interval=log_interval))
#####################

model = dict(bbox_head=dict(head_module=dict(num_classes=num_classes)))