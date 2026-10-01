# StruggleTAL

[![arXiv](https://img.shields.io/badge/arXiv-2510.01362-b31b1b.svg)](https://arxiv.org/abs/2510.01362)
[![Dataset: EvoStruggle](https://img.shields.io/badge/Dataset-EvoStruggle-blue.svg?logo=github)](https://github.com/FELIXFENG2019/EvoStruggle)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

This is the code release for the reproduction of the **Struggle Temporal Action Localization (Struggle TAL)** experiments in the paper [EvoStruggle: A Dataset Capturing the Evolution of Struggle across Activities and Skill Levels](https://arxiv.org/abs/2510.01362) (ICPR 2026).

The dataset used is [EvoStruggle](https://github.com/FELIXFENG2019/EvoStruggle/).

The repository contains three temporal action localization codebases, each adapted to the EvoStruggle dataset:

| Directory | Method | Input | Upstream |
|-----------|--------|-------|----------|
| [`actionformer_release/`](actionformer_release) | ActionFormer | Pre-extracted SlowFast features | [happyharrycn/actionformer_release](https://github.com/happyharrycn/actionformer_release) |
| [`TriDet/`](TriDet) | TriDet | Pre-extracted SlowFast features | [dingfengshi/TriDet](https://github.com/dingfengshi/TriDet) |
| [`OpenTAD/`](OpenTAD) | Re2TAL (end-to-end) | Raw videos | [sming256/OpenTAD](https://github.com/sming256/OpenTAD) |

## 1. Installation

Each codebase has its own environment. The versions below are the ones recommended by the upstream codebases.

### ActionFormer / TriDet

Requirements (see [`actionformer_release/INSTALL.md`](actionformer_release/INSTALL.md) and [`TriDet/README.md`](TriDet/README.md)): Linux, PyTorch >= 1.11 (1.11.0 was used upstream), CUDA, GCC, and `tensorboard pyyaml pandas h5py joblib numpy<=1.23`.

```bash
pip install tensorboard pyyaml pandas h5py joblib "numpy<=1.23"
pip install -r TriDet/requirements.txt

# compile the NMS extension of each codebase (again after updating PyTorch)
cd actionformer_release/libs/utils && python setup.py install --user && cd ../../..
cd TriDet/libs/utils && python setup.py install --user && cd ../../..
```

Optional: `pip install wandb` and pass `--wandb` to `trainval.py` to log training curves to Weights & Biases.

### OpenTAD (Re2TAL)

Follow [`OpenTAD/docs/en/install.md`](OpenTAD/docs/en/install.md) (Python 3.10, PyTorch 2.0.1, mmaction2 1.1.0), then:

```bash
cd OpenTAD
pip install -r requirements.txt
```

Download the K400-pretrained [Re2SlowFast-101 weights](https://drive.google.com/file/d/1FJ_7dxYPobheqCdkRKOZ9KaGNTNp6zx1/view?usp=sharing) and put them at `OpenTAD/pretrained/invslowfast101_k400_pre_k400_ft_e30.pth` (see [`OpenTAD/configs/re2tal/README.md`](OpenTAD/configs/re2tal/README.md)).

## 2. Data Preparation

All configs expect the EvoStruggle dataset at `data/EvoStruggle/` in the root of this repository. The paths in the configs are relative (`../data/EvoStruggle/...`), so run the commands from inside the codebase directory (e.g. `cd actionformer_release`).

```bash
# from the root of this repository
mkdir -p data
git clone https://github.com/FELIXFENG2019/EvoStruggle.git data/EvoStruggle
# or, if you already have the dataset elsewhere:
# ln -s /path/to/EvoStruggle data/EvoStruggle
```

Then download the 360p videos (see the [download instructions](https://github.com/FELIXFENG2019/EvoStruggle#how-to-download)) and place them in `data/EvoStruggle/data/360p/`. ActionFormer and TriDet additionally need the SlowFast features, which are extracted from these videos (see below):

```
data/EvoStruggle/
├── annotations/                         # from the EvoStruggle repository
├── splits/                              # from the EvoStruggle repository
├── extracted_features/
│   └── slowfast_features/               # needed by ActionFormer and TriDet
│       ├── Origami/01_01_01.npy ...
│       ├── Shuffle_Cards/
│       ├── Tangram/
│       └── Tying_Knots/
└── data/
    └── 360p/                            # needed by OpenTAD (Re2TAL)
        ├── Origami/01_01_01.mp4 ...
        ├── Shuffle_Cards/
        ├── Tangram/
        └── Tying_Knots/
```

### Extracting the SlowFast features

The features are not distributed with the dataset; extract them from the 360p videos with [`tools/video_feature_extractor.py`](https://github.com/FELIXFENG2019/EvoStruggle/blob/main/tools/video_feature_extractor.py) of the EvoStruggle repository (see [`extracted_features/README.md`](https://github.com/FELIXFENG2019/EvoStruggle/tree/main/extracted_features) there for details). The script uses a Kinetics-pretrained SlowFast-R50 with a window of 32 frames and a stride of 16 frames, and writes one `(num_clips, 2304)` `.npy` file per video to `data/EvoStruggle/extracted_features/slowfast_features/<Activity>/`:

```bash
# from the root of this repository; requires PyTorch, torchvision and PyTorchVideo
for act in Tying_Knots Origami Tangram Shuffle_Cards; do
    python data/EvoStruggle/tools/video_feature_extractor.py --task $act
done
```

## 3. Experiments

There is one config file per experiment of the paper, generated by [`tools/generate_struggle_configs.py`](tools/generate_struggle_configs.py) from the original configs (`configs/struggle_slowfast.yaml`, `configs/struggle_slowfast_combined.yaml` and `configs/re2tal/e2e_struggle_re2tal_actionformer_slowfast101.py`). Only the split file, the feature/video folder and the subsets differ between experiments; all other settings are copied from the original configs.

| Setting (paper) | Config directory | Config names | Train subsets | Reported on | Codebases |
|-----------------|------------------|--------------|---------------|-------------|-----------|
| Within-Activity (Table 2) | `configs/struggle/within_activity/` | `within_<activity>` | `train_attempt01`–`05` | `validation` | ActionFormer, TriDet, OpenTAD |
| Task Generalization (Table 3) | `configs/struggle/task_generalization/` | `taskgen_<activity>_task<XX>` | `train` | `test` (held-out task) | ActionFormer, TriDet, OpenTAD |
| Activity Generalization (Table 4) | `configs/struggle/activity_generalization/` | `activitygen_<activity>` | `train` | `test` (validation set of the held-out activity) | ActionFormer, TriDet, OpenTAD |
| Separate attempts (Fig. 9) | `configs/struggle/separate_attempts/` | `sepattempt_<activity>_attempt<XX>` | `train_attempt<XX>` | `validation` | ActionFormer, TriDet |
| All attempts, sampled (supp.) | `configs/struggle/allattempts_sampled/` | `allattempts_<activity>_sample<XX>` | `train` | `validation` | ActionFormer, TriDet |

`<activity>` is one of `tyingknots`, `origami`, `tangram`, `shufflecards`. See the [EvoStruggle split documentation](https://github.com/FELIXFENG2019/EvoStruggle#4-data-splits) for the content of each split file.

### ActionFormer / TriDet

```bash
cd actionformer_release   # or: cd TriDet

# train; the checkpoint with the best mAP on `val_split` is saved to ./ckpt/<config name>_<output>/
python trainval.py configs/struggle/within_activity/within_origami.yaml --output run1

# evaluate the saved checkpoint on `val_split`
python eval.py configs/struggle/within_activity/within_origami.yaml ./ckpt/within_origami_run1

# random baseline
python random_model_eval.py configs/struggle/within_activity/within_origami.yaml
```

`trainval.py` evaluates the model on `val_split` after every epoch and keeps the checkpoint with the best mAP. In the generated configs, `val_split` is the subset reported in the paper. To select the checkpoint on the `validation` subset instead, set `val_split: ['validation']` and evaluate the result with `eval.py <config> <ckpt> --split test`.

To run, for example, all task generalization experiments:

```bash
for cfg in configs/struggle/task_generalization/*.yaml; do
    python trainval.py "$cfg" --output run1
done
```

### OpenTAD (Re2TAL)

```bash
cd OpenTAD

# train (here with 2 GPUs); the checkpoint with the lowest validation loss is saved to exps/struggle/<config name>/
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    tools/train.py configs/struggle/within_activity/within_origami.py

# evaluate on the test subset of the config
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    tools/test.py configs/struggle/within_activity/within_origami.py \
    --checkpoint exps/struggle/within_origami/gpu2_id0/checkpoint/best.pth
```

To regenerate the configs after changing one of the original configs, run `python tools/generate_struggle_configs.py` from the root of the repository.

## 4. Results in the Paper

mAP (%) at different tIoU thresholds. See the paper for details.

**Within-Activity (Table 2)**, reported on the validation set of each activity:

| Activity | Model | 0.3 | 0.5 | 0.7 | Avg. |
|----------|-------|----:|----:|----:|-----:|
| Tying Knots | Random | 8.80 | 1.79 | 0.19 | 3.17 |
| | ActionFormer | 67.99 | 39.21 | 10.38 | 39.39 |
| | TriDet | 65.44 | 43.91 | 14.53 | 41.93 |
| | Re2TAL | 61.73 | 38.12 | 8.70 | 35.92 |
| Origami | Random | 7.25 | 0.97 | 0.06 | 2.42 |
| | ActionFormer | 52.75 | 27.73 | 5.23 | 27.98 |
| | TriDet | 54.32 | 27.00 | 5.73 | 28.38 |
| | Re2TAL | 57.37 | 29.62 | 8.78 | 32.36 |
| Tangram | Random | 10.29 | 1.90 | 0.20 | 3.46 |
| | ActionFormer | 55.85 | 30.64 | 4.90 | 29.95 |
| | TriDet | 57.21 | 29.77 | 6.21 | 30.70 |
| | Re2TAL | 69.27 | 44.83 | 19.11 | 44.57 |
| Shuffle Cards | Random | 5.07 | 0.84 | 0.08 | 1.66 |
| | ActionFormer | 71.49 | 56.40 | 21.85 | 50.80 |
| | TriDet | 70.94 | 55.26 | 20.28 | 49.55 |
| | Re2TAL | 78.26 | 62.30 | 35.21 | 59.77 |

**Task Generalization (Table 3)**, average mAP on each held-out task:

| Activity | Model | Task 01 | Task 02 | Task 03 | Task 04 | Task 05 | Average |
|----------|-------|--------:|--------:|--------:|--------:|--------:|--------:|
| Tying Knots | Random | 5.59 | 5.39 | 8.17 | 5.90 | 2.94 | 5.60 |
| | ActionFormer | 38.21 | 36.63 | 36.42 | 29.67 | 27.58 | 33.70 |
| | TriDet | 43.70 | 43.31 | 42.43 | 30.79 | 24.71 | 36.99 |
| | Re2TAL | 40.54 | 40.36 | 46.96 | 28.11 | 20.47 | 35.29 |
| Origami | Random | 3.72 | 3.53 | 2.77 | 2.91 | – | 3.23 |
| | ActionFormer | 24.69 | 18.22 | 23.03 | 23.40 | – | 22.34 |
| | TriDet | 23.65 | 21.03 | 20.70 | 21.59 | – | 21.74 |
| | Re2TAL | 34.92 | 25.03 | 23.05 | 26.77 | – | 27.44 |
| Tangram | Random | 6.80 | 5.42 | 4.17 | 4.66 | – | 5.26 |
| | ActionFormer | 29.63 | 28.37 | 20.59 | 33.90 | – | 28.12 |
| | TriDet | 30.50 | 32.02 | 23.27 | 34.08 | – | 29.97 |
| | Re2TAL | 33.38 | 43.50 | 34.11 | 45.97 | – | 39.24 |
| Shuffle Cards | Random | 1.63 | 2.28 | 1.92 | 2.54 | 2.08 | 2.09 |
| | ActionFormer | 11.48 | 29.31 | 31.55 | 34.21 | 33.05 | 27.92 |
| | TriDet | 9.70 | 32.12 | 27.29 | 32.01 | 34.12 | 27.05 |
| | Re2TAL | 15.10 | 38.56 | 33.26 | 36.30 | 49.71 | 34.59 |

**Activity Generalization (Table 4)**, reported on the validation set of the held-out activity:

| Held-out activity | Model | 0.3 | 0.5 | 0.7 | Avg. |
|-------------------|-------|----:|----:|----:|-----:|
| Tying Knots | Random | 8.80 | 1.79 | 0.19 | 3.17 |
| | ActionFormer | 45.13 | 20.37 | 4.30 | 22.58 |
| | TriDet | 34.76 | 14.08 | 2.68 | 16.25 |
| | Re2TAL | 47.14 | 23.45 | 5.19 | 25.05 |
| Origami | Random | 7.25 | 0.97 | 0.06 | 2.42 |
| | ActionFormer | 32.07 | 8.54 | 0.99 | 12.24 |
| | TriDet | 28.78 | 7.08 | 1.12 | 10.72 |
| | Re2TAL | 26.26 | 9.14 | 2.65 | 11.67 |
| Tangram | Random | 10.29 | 1.90 | 0.20 | 3.46 |
| | ActionFormer | 44.58 | 15.60 | 2.83 | 19.60 |
| | TriDet | 47.07 | 18.19 | 3.23 | 21.42 |
| | Re2TAL | 49.53 | 24.12 | 8.97 | 26.98 |
| Shuffle Cards | Random | 5.07 | 0.84 | 0.08 | 1.66 |
| | ActionFormer | 29.15 | 9.86 | 1.40 | 12.69 |
| | TriDet | 28.42 | 7.15 | 0.67 | 10.75 |
| | Re2TAL | 24.80 | 8.00 | 0.98 | 10.53 |

## 5. Implementation Notes

These details affect the results and are kept as used for the paper:

- **Evaluation metric**: mAP at tIoU thresholds 0.3:0.1:0.7 (ActivityNet-style evaluation) for the single action class `Struggle`.
- **Videos without struggle**: the ActionFormer and TriDet dataloaders skip videos without any annotated struggle segment, both for training and for evaluation.
- **Feature noise (ActionFormer only)**: Gaussian noise with std 0.05 is added to the input features as data augmentation. As in the paper, it is also added at evaluation time, so repeated evaluations of the same checkpoint vary slightly. This can be changed with `feat_noise_std` (default `0.05`) and `eval_feat_noise` (default `true`) in the `dataset` section of the config, e.g. `eval_feat_noise: false` for deterministic evaluation.
- **Random seed**: `init_rand_seed` defaults to `1234567891` (see `libs/core/config.py`).
- **Evaluation subset**: `eval.py` and `random_model_eval.py` accept `--split <subset>` to evaluate on a different subset than `val_split` without editing the config.

## Citation

If you use this code or the EvoStruggle dataset, please cite:

```bibtex
@inproceedings{feng2026evostruggle,
  title={{EvoStruggle}: A Dataset Capturing the Evolution of Struggle across Activities and Skill Levels},
  author={Feng, Shijia and Wray, Michael and Mayol-Cuevas, Walterio},
  booktitle={International Conference on Pattern Recognition (ICPR)},
  pages={96--111},
  year={2026},
  organization={Springer}
}
```

Please also cite [ActionFormer](https://github.com/happyharrycn/actionformer_release), [TriDet](https://github.com/dingfengshi/TriDet) and [OpenTAD](https://github.com/sming256/OpenTAD) / [Re2TAL](https://arxiv.org/abs/2211.14053) if you use the corresponding code.

## License

The code in this repository is released under the Apache-2.0 License (see [`LICENSE`](LICENSE)). The codebases in the subdirectories keep their original licenses (see the `LICENSE` file in each directory).
