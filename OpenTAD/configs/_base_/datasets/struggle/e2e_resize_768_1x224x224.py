# annotation_path = "/media/alexa/WORKSPACE/Shijia-stage-two/new_struggle_dataset/splits/indomain_generalization/Tying_Knots/Tying_Knots_subactivity05_data.json"
# annotation_path = '/media/alexa/WORKSPACE/Shijia-stage-two/new_struggle_dataset/splits/separate_attempts/Tangram/Tangram_sepattempt.json'
annotation_path = '/media/alexa/WORKSPACE/Shijia-stage-two/new_struggle_dataset/splits/crossdomain_generalization/Tying_Knots/Tying_Knots_crossdomain_testonvalonly.json'
class_map = "/media/alexa/WORKSPACE/Shijia-stage-two/new_struggle_dataset/annotations/category_idx.txt" 
data_path = "/media/alexa/WORKSPACE/Shijia-stage-two/new_struggle_dataset/data/360p"
block_list = None

resize_length = 768

dataset = dict(
    train=dict(
        type="StruggleResizeDataset",
        ann_file=annotation_path,
        subset_name=["train"],
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        resize_length=resize_length,
        class_agnostic=True,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="resize"),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 224)),
            dict(type="mmaction.CenterCrop", crop_size=224),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    val=dict(
        type="StruggleResizeDataset",
        ann_file=annotation_path,
        subset_name=["validation"],
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        resize_length=resize_length,
        class_agnostic=True,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="resize"),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 224)),
            dict(type="mmaction.CenterCrop", crop_size=224),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    test=dict(
        type="StruggleResizeDataset",
        ann_file=annotation_path,
        subset_name=["validation"],
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        test_mode=True,
        resize_length=resize_length,
        class_agnostic=True,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="resize"),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 224)),
            dict(type="mmaction.CenterCrop", crop_size=224),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs"]),
            dict(type="Collect", inputs="imgs", keys=["masks"]),
        ],
    ),
)


evaluation = dict(
    type="mAP",
    subset=["validation"],
    tiou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],
    ground_truth_filename=annotation_path,
)
