# python imports
import argparse
import os
import glob
import time
from pprint import pprint
import pandas as pd
import numpy as np

# torch imports
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import valid_one_epoch, ANETdetection, fix_random_seed


class random_model(nn.Module):
    def __init__(self, num_segments_probs):
        super(random_model, self).__init__()
        self.num_segments_probs = num_segments_probs
        self.dropout = nn.Dropout(p=0.5) 
        # This is a temporary fixing to the error 'random_model' object has no attribute '_modules' when calling model.eval() in the validation function

    def forward(self, video_list):
        """
        video_list items: 
        data_dict = {'video_id'        : video_item['id'],
                     'feats'           : feats,      # C x T
                     'segments'        : segments,   # N x 2
                     'labels'          : labels,     # N
                     'fps'             : video_item['fps'],
                     'duration'        : video_item['duration'],
                     'feat_stride'     : feat_stride,
                     'feat_num_frames' : self.num_frames}
        """
        output = self.dropout(torch.randn(1))
        results = []
        for video_data in video_list:
            video_id = video_data['video_id']
            duration = video_data['duration']

            # Step 1: Sample the number of segments according to num_segments_probs
            num_segments = np.random.choice(
                range(1, len(self.num_segments_probs) + 1), 
                p=self.num_segments_probs
            )

            # Step 2: Generate 2 * num_segments boundary times and sort them
            boundaries = np.sort(np.random.uniform(0, duration, 2 * num_segments))

            # Step 3: Pair consecutive boundaries to form segments
            segs = []
            scores = []
            labels = []
            for i in range(0, len(boundaries) - 1, 2):
                start, end = boundaries[i], boundaries[i + 1]
                segs.append([start, end])
                scores.append(np.random.rand())  # Random score for each segment range [0, 1)
                labels.append(0)

            # Convert segments and scores to numpy arrays
            segs = np.array(segs)
            scores = np.array(scores)
            labels = np.array(labels)
            segs = torch.from_numpy(segs)
            scores = torch.from_numpy(scores)
            labels = torch.from_numpy(labels)

            results.append(
                {
                    'video_id': video_id,
                    'segments': segs,
                    't-start': segs[:, 0],
                    't-end': segs[:, 1],
                    'scores': scores,
                    'labels': labels
                }
            )

        return results


################################################################################
def main(args):
    """0. load config"""
    # sanity check
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    assert len(cfg['val_split']) > 0, "Test set must be specified!"
    # pprint(cfg)

    """1. fix all randomness"""
    # fix the random seeds (this will fix everything)
    _ = fix_random_seed(0, include_cuda=True)

    """2. create dataset / dataloader"""
    train_dataset = make_dataset(
        cfg['dataset_name'], True, cfg['train_split'], **cfg['dataset']
    )
    # establish a dataframe according to the train_dataset.data_list
    video_data_list = train_dataset.data_list
    # Convert each video's dictionary into a DataFrame row, then concatenate all rows
    train_df = pd.concat(
        [pd.DataFrame([video_data]) for video_data in video_data_list],
        ignore_index=True
    )
    train_df['num_struggle'] = train_df['segments'].apply(lambda x: x.shape[0])
    hist, bin_edges = np.histogram(train_df['num_struggle'], bins=np.arange(1, 10))
    hist = hist/hist.sum()
    # import pdb; pdb.set_trace()

    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
    )
    # set bs = 1, and disable shuffle
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg['loader']['num_workers']
    )

    """3. create model and evaluator"""
    # model
    model = random_model(num_segments_probs=hist.tolist())
 
    # set up evaluator
    det_eval, output_file = None, None
    val_db_vars = val_dataset.get_attributes()
    det_eval = ANETdetection(
        val_dataset.json_file,
        val_dataset.split, # val_dataset.split[0]
        tiou_thresholds = val_db_vars['tiou_thresholds']
    )

    """5. Test the model"""
    print("\nStart testing model {:s} ...".format(cfg['model_name']))
    start = time.time()
    mAP = valid_one_epoch(
        val_loader,
        model,
        -1,
        evaluator=det_eval,
        output_file=output_file,
        ext_score_file=cfg['test_cfg']['ext_score_file'],
        tb_writer=None,
        print_freq=args.print_freq
    )
    end = time.time()
    print("All done! Total time: {:0.2f} sec".format(end - start))
    return

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    parser.add_argument('config', type=str, metavar='DIR',
                        help='path to a config file')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    args = parser.parse_args()
    main(args)

