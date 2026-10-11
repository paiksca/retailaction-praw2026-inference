# RetailAction challenge inference (team Samuel Paik-Heintz, Institute for Systems Biology)

This repository runs inference for our entry in the RetailAction Challenge (PRAW Workshop, WACV 2027). One `docker run` command reads the challenge test set and writes `submission.csv` in the Kaggle format. The repository holds inference code and trained weights only.

## Run

1. Download the challenge test set from the Kaggle competition `retail-action-challenge-praw-workshop-wacv-2027`. Inference reads the folder `challenge_test_set/vam-dataset-challenge_test_no_labels/data`. That folder holds one subfolder per sample (`000000` to `002499`), and each subfolder holds a `metadata.json`.
2. Build and run:

```bash
docker build -t retailaction-infer .
docker run --rm \
  -v /path/to/challenge_test_set/vam-dataset-challenge_test_no_labels/data:/data:ro \
  -v "$PWD/out":/out \
  retailaction-infer
```

The run writes `out/submission.csv` with 2,500 rows plus a header. On 2 CPU cores it takes a few minutes and needs no GPU or network access.

`Dockerfile` reproduces our current Kaggle submission (an ensemble of two pose models, `weights/pose_f3.pt` and `weights/pose_f4.pt`). `Dockerfile.fusion` builds the next model, which adds image features from crops around the hands and also reads the videos. It is still in development.

## Method

The model reads the metadata of each clip and does not decode the videos. Each of the 32 sampled frames becomes one token. A token holds the frame time, the subject's 2D pose in both camera views, hand velocities and arm extension, the motion sampling score and the face position. A 4-layer transformer encoder over the 32 tokens predicts the following for each frame:

- how likely the frame is an action center
- the distance to the action start and end
- the touch point in each view, as a pointer over the hand and wrist joints of all frames plus a small learned offset

Decoding takes local maxima of the center score, applies temporal non-maximum suppression and keeps up to 5 actions per clip. Every output action has a finite touch point for both camera views.

The pose arrays in `metadata.json` are stored as (row, col). `src/features.py` converts them to (x, y).

## Reproducibility

- Package versions are pinned in `requirements.txt`.
- The networks run in float64, and outputs are rounded to 6 decimals. With `Dockerfile`, the output file is byte-identical on linux/arm64 and linux/amd64. The submitted file was produced by this Docker image.

## Data and license

The weights were trained on the RetailAction dataset by Standard Cognition (Standard AI), https://huggingface.co/datasets/standard-cognition/RetailAction. They are a derivative of that dataset and are distributed under its license terms.

## AI assistance

An AI coding assistant (Claude, by Anthropic) wrote the code under the direction of Samuel Paik-Heintz.
