# catdog
A transfer-learning based image classifier to detect cats vs dogs, built with TensorFlow/Keras, deployed via a minimal FastAPI + HTML frontend.


# Data

We use the Kaggle “Cats vs Dogs” dataset.

## Where files live
- `data/raw/`         → zip(s) from Kaggle (ignored by Git)
- `data/processed/`   → train/ val/ test/ (created by `scripts/prepare_data.py`)
- `data/sample/`      → a few tiny images for tests & demo (committed)

## How to get data
1) Install Kaggle CLI and place your `~/.kaggle/kaggle.json`.
2) From repo root:
```bash
bash scripts/get_data.sh
python scripts/prepare_data.py --input data/raw --output data/processed --img-size 224 --val 0.15 --test 0.15 --seed 42 --make-sample

