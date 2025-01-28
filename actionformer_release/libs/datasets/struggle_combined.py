import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

from .datasets import register_dataset
from .data_utils import truncate_feats

@register_dataset("struggle_combined")
class CombinedStruggleDataset(Dataset):
    def __init__(
        self,
        is_training,     # if in training mode
        split,           # split, a tuple/list allowing concat of subsets
        feat_folder,     # folder for features
        json_file,       # actually loading a CSV file for annotations in our case!!
        feat_stride,     # temporal stride of the feats
        num_frames,      # number of frames for each feat
        default_fps,     # default fps
        downsample_rate, # downsample rate for feats
        max_seq_len,     # maximum sequence length during training
        trunc_thresh,    # threshold for truncate an action segment
        crop_ratio,      # a tuple (e.g., (0.9, 1.0)) for random cropping
        input_dim,       # input feat dim
        num_classes,     # number of action categories
        file_prefix,     # feature file prefix if any
        file_ext,        # feature file extension if any
        force_upsampling # force to upsample to max_seq_len
    ):
        # file path
        assert os.path.exists(feat_folder) and os.path.exists(json_file)
        assert isinstance(split, tuple) or isinstance(split, list)
        assert crop_ratio == None or len(crop_ratio) == 2
        self.feat_folder = feat_folder
        if file_prefix is not None:
            self.file_prefix = file_prefix
        else:
            self.file_prefix = ''
        self.file_ext = file_ext
        self.json_file = json_file

        # split / training mode
        self.split = split
        self.is_training = is_training

        # features meta info
        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.default_fps = default_fps
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.label_dict = None
        self.crop_ratio = crop_ratio

        # load database and select the subset
        dict_db, label_dict = self._load_json_db(self.json_file)
        assert len(label_dict) == num_classes
        self.data_list = dict_db
        self.label_dict = label_dict

        # dataset specific attributes 
        self.db_attributes = {
            'dataset_name': 'struggle_combined',
            'tiou_thresholds': np.linspace(0.3, 0.7, 5),
            # we will mask out cliff diving 
            'empty_label_ids': [], 
        }

    def get_attributes(self):
        return self.db_attributes

    def _load_json_db(self, json_file):
        # load database and select the subset
        with open(json_file, 'r') as fid:
            json_data = json.load(fid)
        json_db = json_data['database']

        # if label_dict is not available
        if self.label_dict is None:
            label_dict = {}
            for key, value in json_db.items():
                if len(value['annotations']) > 0:
                    for act in value['annotations']:
                        label_dict[act['label']] = act['label_id'] - 1 # 0-indexed
                # else:
                    # label_dict['Non-struggle'] = self.num_classes - 1

        # fill in the db (immutable afterwards)
        dict_db = tuple()
        for key, value in json_db.items():
            # skip the video if not in the split
            if value['subset'].lower() not in self.split:
                continue
            # or does not have the feature file
            feat_file = os.path.join(self.feat_folder, key.split('-')[0],  key.split('-')[1] + self.file_ext)
            if not os.path.exists(feat_file):
                continue

            # get fps if available
            if self.default_fps is not None:
                fps = self.default_fps
            elif 'fps' in value:
                fps = value['fps']
            else:
                assert False, "Unknown video FPS."

            # get video duration if available
            if 'duration' in value:
                duration = value['duration']
            else:
                duration = 1e8

            # get annotations if available
            if ('annotations' in value) and (len(value['annotations']) > 0):
                # a fun fact of THUMOS: cliffdiving (4) is a subset of diving (7)
                # our code can now handle this corner case
                segments, labels = [], []
                for act in value['annotations']:
                    segments.append(act['segment'])
                    labels.append([label_dict[act['label']]])
                segments = np.asarray(segments, dtype=np.float32) # num_seg x 2
                labels = np.squeeze(np.asarray(labels, dtype=np.int64), axis=1) # (num_seg, )
                """
                # for each interval of the segments, add the non-struggle segments
                for i in range(len(segments)-1):
                    # the first interval segment starts from the beginning of the video
                    if i == 0:
                        if segments[i][0] > 0.0:
                            segments.append([0.0, segments[i][0]])
                            labels.append([label_dict['Non-struggle']])
                    # the last interval segment ends at the end of the video
                    if i == len(segments) - 2:
                        if segments[i+1][1] < duration:
                            segments.append([segments[i+1][1], duration])
                            labels.append([label_dict['Non-struggle']])
                    # the interval between two segments
                    # segments_interval = segments[i+1][0] - segments[i][1]
                    segments.append([segments[i][1], segments[i+1][0]])
                    labels.append([label_dict['Non-struggle']])
                segments = np.asarray(segments, dtype=np.float32) # num_seg x 2
                labels = np.squeeze(np.asarray(labels, dtype=np.int64), axis=1) # (num_seg, )
                """
                """
                elif ('annotations' in value) and (len(value['annotations']) == 0):
                    # Added 21/10/2024 by Shijia
                    # this case is specifically for Struggle Action Detection where the background is the non-struggle action
                    segments = [0., duration]
                    labels = [label_dict['Non-struggle']]
                    segments = np.expand_dims(np.asarray(segments, dtype=np.float32), axis=0) # 1 x 2
                    labels = np.asarray(labels, dtype=np.int64) # (num_seg, )
                """
            else:
                continue
                # segments = None
                # labels = None
            dict_db += ({'id': key,
                         'fps' : fps,
                         'duration' : duration,
                         'segments' : segments,
                         'labels' : labels
            }, )

        return dict_db, label_dict

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader 
        # instead the model will need to decide how to batch / preporcess the data 
        video_item = self.data_list[idx]

        # load features
        activity, vid_id = video_item['id'].split('-') # In format {activity_name}-{01_01_01}
        filename = os.path.join(self.feat_folder, activity, vid_id + self.file_ext)
        feats = np.load(filename).astype(np.float32)
        # Add random Gaussian noise as data augmentation
        noise_std = 0.05 # have tried the 0.01, 0.02, 0.05, and 0.07 in which the 0.05 have the best trend in mAP
        noise = np.random.normal(loc=0.0, scale=noise_std, size=feats.shape).astype(np.float32)
        feats = feats + noise

        # deal with downsampling (= increased feat stride)
        feats = feats[::self.downsample_rate, :]
        feat_stride = self.feat_stride * self.downsample_rate
        feat_offset = 0.5 * self.num_frames / feat_stride
        # T x C -> C x T
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))

        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here
        if video_item['segments'] is not None:
            segments = torch.from_numpy(
                video_item['segments'] * video_item['fps'] / feat_stride - feat_offset
            )
            labels = torch.from_numpy(video_item['labels'])
        else:
            segments, labels = None, None
        
        # return a data dict
        data_dict = {'video_id'        : video_item['id'],
                     'feats'           : feats,      # C x T
                     'segments'        : segments,   # N x 2
                     'labels'          : labels,     # N
                     'fps'             : video_item['fps'],
                     'duration'        : video_item['duration'],
                     'feat_stride'     : feat_stride,
                     'feat_num_frames' : self.num_frames}
        
        # truncate the features during training
        if self.is_training and (segments is not None):
            data_dict = truncate_feats(
                data_dict, self.max_seq_len, self.trunc_thresh, feat_offset, self.crop_ratio
            )

        return data_dict


