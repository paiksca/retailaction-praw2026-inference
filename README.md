# RetailAction challenge inference (Samuel Paik-Heintz, Institute for Systems Biology)

This repository runs inference for our entry in the RetailAction Challenge (PRAW Workshop, WACV 2027). One `docker run` command reads the challenge test set and writes `submission.csv` in the Kaggle format. The repository holds inference code and trained weights only.

## Run

1. Download the challenge test set from the Kaggle competition `retail-action-challenge-praw-workshop-wacv-2027`. Inference reads the folder `challenge_test_set/vam-dataset-challenge_test_no_labels/data`. That folder has one subfolder per clip (`000000` to `002499`), and each subfolder holds `metadata.json`, `rank0_video.mp4` and `rank1_video.mp4`.
2. Build the image and run it:

```bash
docker build -t retailaction-infer .
docker run --rm \
  -v /path/to/challenge_test_set/vam-dataset-challenge_test_no_labels/data:/data:ro \
  -v "$PWD/out":/out \
  retailaction-infer
```

The run writes `out/submission.csv`, which has a header and 2,500 rows. It needs no GPU, and it needs no network access after the build. It uses all CPU cores the container can see. On 5 cores of an Apple M-series laptop (Docker, linux/arm64) the full test set took 6.3 hours. Most of that time goes to the image model, which runs in 64-bit floating point.

The build downloads the DINOv2 code at a fixed commit and its pretrained weights. It checks both against SHA-256 hashes.

## Method

For each clip the model reads the 32 sampled frames from both camera views. It combines two inputs:

- the shopper's pose from `metadata.json` (joint positions, hand velocities, arm extension and frame times)
- image features of small crops around both hands, computed by a frozen DINOv2 ViT-S/14 model

A transformer with 4 layers reads one token per frame. It predicts action centers, action start and end times, and the touch point in both views. The touch point is a weighted choice among the hand and wrist joints of all frames, plus a small correction. Up to 5 actions per clip are kept after removing overlapping candidates.

The pose arrays in `metadata.json` hold (row, column). `src/features.py` converts them to (x, y).

## Reproducibility

- Package versions are pinned in `requirements.txt`, and the DINOv2 code and weights are pinned by commit and hash.
- We decode video frames with PyAV and do color conversion and resizing in 64-bit floating point, so pixel values do not depend on the CPU. The networks also run in 64-bit floating point. Output values are rounded to 6 decimals.
- The submitted file was produced by this Docker image on linux/arm64. Rerunning the image on three clips reproduced the submitted rows exactly.

## Data and license

The weights were trained on the RetailAction dataset by Standard Cognition (Standard AI), https://huggingface.co/datasets/standard-cognition/RetailAction. They are a derivative of that dataset and are distributed under its license terms. DINOv2 is by Meta AI (https://github.com/facebookresearch/dinov2).

## AI assistance

Claude (Anthropic) generated the majority of this code under the author's direction. All AI-generated outputs were reviewed and revised by the author, who assumes full responsibility for the final work.
