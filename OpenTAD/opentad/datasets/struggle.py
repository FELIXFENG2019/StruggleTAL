import numpy as np
import json
import os
from copy import deepcopy
from .base import SlidingWindowDataset, PaddingDataset, ResizeDataset, filter_same_annotation
from .builder import DATASETS


@DATASETS.register_module()
class StruggleResizeDataset(ResizeDataset):
    def get_dataset(self):
        with open(self.ann_file, "r") as f:
            anno_database = json.load(f)["database"]

        # some videos might be missed in the features or videos, we need to block them
        if self.block_list != None:
            if isinstance(self.block_list, list):
                blocked_videos = self.block_list
            else:
                with open(self.block_list, "r") as f:
                    blocked_videos = [line.rstrip("\n") for line in f]
        else:
            blocked_videos = []

        self.data_list = []
        for video_name, video_info in anno_database.items():
            if (video_name in blocked_videos) or (video_info["subset"].lower() not in self.subset_name):
                continue

            # get the ground truth annotation
            if self.test_mode:
                video_anno = {}
            else:
                video_anno = self.get_gt(video_info)
                if video_anno == None:  # have no valid gt
                    continue
            self.data_list.append([video_name, video_info, video_anno])
        assert len(self.data_list) > 0, f"No data found in {self.subset_name} subset."

    def get_gt(self, video_info, thresh=0.01):
        gt_segment = []
        gt_label = []
        for anno in video_info["annotations"]:
            gt_start = float(anno["segment"][0]) # Timestamps
            gt_end = float(anno["segment"][1])
            gt_scale = (gt_end - gt_start) / float(video_info["duration"])

            if (not self.filter_gt) or (gt_scale > thresh):
                gt_segment.append([gt_start, gt_end])
                if self.class_agnostic:
                    gt_label.append(0)
                else:
                    gt_label.append(self.class_map.index(anno["label"]))

        if len(gt_segment) == 0:  # have no valid gt
            return None
        else:
            annotation = dict(
                gt_segments=np.array(gt_segment, dtype=np.float32),
                gt_labels=np.array(gt_label, dtype=np.int32),
            )
            return filter_same_annotation(annotation)

    def __getitem__(self, index):
        video_name, video_info, video_anno = self.data_list[index]

        if video_anno != {}:
            video_anno = deepcopy(video_anno)  # avoid modify the original dict

        # check video_name to see if we need to load the data in combined data mode
        if len(video_name.split('-')) > 1:
            # run in combined data mode (e.g. "activity_name-video_name")
            activity_name = video_name.split('-')[0]
            video_name = video_name.split('-')[1]
        else: 
            activity_name = self.ann_file.split('/')[-2]
        
        results = self.pipeline(
            dict(
                video_name=video_name,
                data_path=os.path.join(self.data_path, activity_name),
                resize_length=self.resize_length,
                sample_stride=self.sample_stride,
                # resize post process setting
                fps=-1,
                duration=float(video_info["duration"]),
                **video_anno,
            )
        )
        return results
    

@DATASETS.register_module()
class StruggleSlidingDataset(SlidingWindowDataset):
    def get_dataset(self):
        with open(self.ann_file, "r") as f:
            anno_database = json.load(f)["database"]

        # some videos might be missed in the features or videos, we need to block them
        if self.block_list != None:
            if isinstance(self.block_list, list):
                blocked_videos = self.block_list
            else:
                with open(self.block_list, "r") as f:
                    blocked_videos = [line.rstrip("\n") for line in f]
        else:
            blocked_videos = []

        self.data_list = []
        for video_name, video_info in anno_database.items():
            # Only change: video_info["subset"].lower() not in self.subset_name
            if (video_name in blocked_videos) or (video_info["subset"].lower() not in self.subset_name):
                continue

            # get the ground truth annotation
            if self.test_mode:
                video_anno = {}
            else:
                video_anno = self.get_gt(video_info)
                if video_anno == None:  # have no valid gt
                    continue

            tmp_data_list = self.split_video_to_windows(video_name, video_info, video_anno)
            self.data_list.extend(tmp_data_list)
        assert len(self.data_list) > 0, f"No data found in {self.subset_name} subset."

    def get_gt(self, video_info, thresh=0.0):
        gt_segment = []
        gt_label = []
        for anno in video_info["annotations"]:
            if anno["label"] == "Ambiguous":
                continue
            # gt_start = int(anno["segment"][0] / video_info["duration"] * video_info["frame"]) # frame
            # gt_end = int(anno["segment"][1] / video_info["duration"] * video_info["frame"])
            gt_start = int(anno["segment"][0] * video_info["fps"]) # frame
            gt_end = int(anno["segment"][1] * video_info["fps"])

            if (not self.filter_gt) or (gt_end - gt_start > thresh):
                gt_segment.append([gt_start, gt_end])
                gt_label.append(self.class_map.index(anno["label"]))

        if len(gt_segment) == 0:  # have no valid gt
            return None
        else:
            annotation = dict(
                gt_segments=np.array(gt_segment, dtype=np.float32),
                gt_labels=np.array(gt_label, dtype=np.int32),
            )
            return filter_same_annotation(annotation)

    def __getitem__(self, index):
        video_name, video_info, video_anno, window_snippet_centers = self.data_list[index]

        if video_anno != {}:
            video_anno = deepcopy(video_anno)  # avoid modify the original dict
            # frame divided by snippet stride inside current window
            # this is only valid gt inside this window
            video_anno["gt_segments"] = video_anno["gt_segments"] - window_snippet_centers[0] - self.offset_frames
            video_anno["gt_segments"] = video_anno["gt_segments"] / self.snippet_stride
        
        # check video_name to see if we need to load the data in combined data mode
        if len(video_name.split('-')) > 1:
            # run in combined data mode (e.g. "activity_name-video_name")
            activity_name = video_name.split('-')[0]
            video_name = video_name.split('-')[1]
        else: 
            activity_name = self.ann_file.split('/')[-2]
        
        results = self.pipeline(
            dict(
                video_name=video_name,
                data_path=os.path.join(self.data_path, activity_name),
                window_size=self.window_size,
                # trunc window setting
                feature_start_idx=int(window_snippet_centers[0] / self.snippet_stride),
                feature_end_idx=int(window_snippet_centers[-1] / self.snippet_stride),
                sample_stride=self.sample_stride,
                # sliding post process setting
                fps=video_info["fps"],
                snippet_stride=self.snippet_stride,
                window_start_frame=window_snippet_centers[0],
                duration=video_info["duration"],
                offset_frames=self.offset_frames,
                # training setting
                **video_anno,
            )
        )

        return results


@DATASETS.register_module()
class StrugglePaddingDataset(PaddingDataset):
    def get_dataset(self):
        with open(self.ann_file, "r") as f:
            anno_database = json.load(f)["database"]

        # some videos might be missed in the features or videos, we need to block them
        if self.block_list != None:
            if isinstance(self.block_list, list):
                blocked_videos = self.block_list
            else:
                with open(self.block_list, "r") as f:
                    blocked_videos = [line.rstrip("\n") for line in f]
        else:
            blocked_videos = []

        self.data_list = []
        for video_name, video_info in anno_database.items():
            if (video_name in blocked_videos) or (video_info["subset"].lower() not in self.subset_name):
                continue

            # get the ground truth annotation
            if self.test_mode:
                video_anno = {}
            else:
                video_anno = self.get_gt(video_info)
                if video_anno == None:  # have no valid gt
                    continue
            self.data_list.append([video_name, video_info, video_anno])
        assert len(self.data_list) > 0, f"No data found in {self.subset_name} subset."
        
    def get_gt(self, video_info, thresh=0.0):
        gt_segment = []
        gt_label = []
        for anno in video_info["annotations"]:
            if anno["label"] == "Ambiguous":
                continue
            # gt_start = int(anno["segment"][0] / video_info["duration"] * video_info["frame"])
            # gt_end = int(anno["segment"][1] / video_info["duration"] * video_info["frame"])
            gt_start = int(anno["segment"][0] * video_info["fps"])
            gt_end = int(anno["segment"][1] * video_info["fps"])

            if (not self.filter_gt) or (gt_end - gt_start > thresh):
                gt_segment.append([gt_start, gt_end])
                gt_label.append(self.class_map.index(anno["label"]))

        if len(gt_segment) == 0:  # have no valid gt
            return None
        else:
            annotation = dict(
                gt_segments=np.array(gt_segment, dtype=np.float32),
                gt_labels=np.array(gt_label, dtype=np.int32),
            )
            return filter_same_annotation(annotation)

    def __getitem__(self, index):
        video_name, video_info, video_anno = self.data_list[index]

        if video_anno != {}:
            video_anno = deepcopy(video_anno)  # avoid modify the original dict
            video_anno["gt_segments"] = video_anno["gt_segments"] - self.offset_frames
            video_anno["gt_segments"] = video_anno["gt_segments"] / self.snippet_stride

        # check video_name to see if we need to load the data in combined data mode
        if len(video_name.split('-')) > 1:
            # run in combined data mode (e.g. "activity_name-video_name")
            activity_name = video_name.split('-')[0]
            video_name = video_name.split('-')[1]
        else:
            activity_name = self.ann_file.split('/')[-2]
        
        results = self.pipeline(
            dict(
                video_name=video_name,
                data_path=os.path.join(self.data_path, activity_name),
                sample_stride=self.sample_stride,
                snippet_stride=self.snippet_stride,
                fps=video_info["fps"],
                duration=video_info["duration"],
                offset_frames=self.offset_frames,
                **video_anno,
            )
        )

        return results
