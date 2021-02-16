# The new config inherits a base config to highlight the necessary modification
_base_ = ['../configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py']

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    pretrained='checkpoints/resnet50-19c8e357.pth',
    roi_head=dict(
        bbox_head=dict(num_classes=1)
        ))


# pipelines
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(640, 360), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='MyDataAugmentation'),
    #dict(type='PencilFilter'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 360),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            #dict(type='PencilFilter'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]

# Modify dataset related settings
dataset_type = 'CocoDataset'
classes = ('object',)
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        img_prefix='/home/leon/git/python-blender-datagen/DATASET/train3obj_adaptive/',
        classes=classes,
        pipeline=train_pipeline,
        ann_file='/home/leon/git/python-blender-datagen/DATASET/train3obj_adaptive/annotation_coco.json'),
    val=dict(
        type=dataset_type,
        img_prefix='/home/leon/git/python-blender-datagen/DATASET/Frames_rgb/',
        classes=classes,
        pipeline=test_pipeline,
        ann_file='/home/leon/git/python-blender-datagen/DATASET/Frames_rgb/annotation_coco.json'),
    test=dict(
        type=dataset_type,
        img_prefix='/home/leon/git/python-blender-datagen/DATASET/Frames_rgb/',
        classes=classes,
        pipeline=test_pipeline,
        ann_file='/home/leon/git/python-blender-datagen/DATASET/Frames_rgb/annotation_coco.json'))



# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

optimizer = dict(type='SGD', lr=0.00001, momentum=0.9, weight_decay=0.0001)
#optimizer_config = dict(grad_clip=None)

lr_config = dict(
	policy='step',
	warmup='linear',
	warmup_iters=500,
	warmup_ratio=0.001,
	step=[100])
total_epochs=12
#log_config = dict(interval=100)
checkpoint_config = dict(interval=1)
log_config = dict(
	interval=1000,
	hooks=[
	dict(type='TextLoggerHook'),
	dict(type='TensorboardLoggerHook')
	])

workflow = [('train', 1), ('val',1)]
work_dir = 'work_dir_15022021_3obj_adaptive'
