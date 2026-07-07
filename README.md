# Human Activity Recognition — CNN-LSTM

Human Activity Recognition (HAR) from raw smartphone accelerometer and
gyroscope signals, using a CNN-LSTM hybrid model in PyTorch, with
experiment tracking via MLflow.

## Dataset

UCI HAR Dataset — 30 subjects, 6 activities (WALKING, WALKING_UPSTAIRS,
WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING), 2.56-second windows
(128 timesteps at 50Hz), 9 raw signal channels per window:

- body_acc_x, body_acc_y, body_acc_z
- body_gyro_x, body_gyro_y, body_gyro_z
- total_acc_x, total_acc_y, total_acc_z

Download from: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

Extract into `data/raw/` so the path
`data/raw/UCI HAR Dataset/train/Inertial Signals/` exists.

## Project Structure

```
har-activity-recognition/
├── data/
│   ├── raw/                     # UCI HAR raw download goes here
│   └── data_loader.py           # Dataset class + loading/normalization
├── models/
│   ├── cnn_lstm.py              # CNN-LSTM architecture
│   └── checkpoints/             # saved model weights (created at train time)
├── training/
│   ├── train.py                 # training loop with early stopping
│   └── config.yaml              # hyperparameters
├── requirements.txt
└── README.md
```

## Architecture

```
Input (batch, 128, 9)
    -> Conv1D blocks (feature extraction, downsampling 128 -> 32 timesteps)
    -> Bidirectional LSTM (2 layers, hidden size 128)
    -> Mean-pool over time
    -> Linear(256 -> 64) -> ReLU -> Dropout -> Linear(64 -> 6)
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**1. Sanity-check the data pipeline:**

```bash
python data/data_loader.py
```

**2. Sanity-check the model forward pass:**

```bash
python models/cnn_lstm.py
```

**3. Train:**

```bash
python training/train.py
```

**4. View MLflow experiment dashboard:**

```bash
mlflow ui
```

Then open `http://localhost:5000`.

## Notes

- Train/validation split is done by **subject ID**, not randomly, to
  prevent windows from the same subject leaking across splits.
- Normalization statistics (mean/std per channel) are computed from the
  training set only and applied to both train and test.
- Class weights are computed from the training subset to guard against
  any class imbalance.
- Best model checkpoint (lowest validation loss) is saved to
  `models/checkpoints/cnn_lstm_best.pt`.

## Expected Results

A well-trained CNN-LSTM on UCI HAR typically reaches **90-94% test
accuracy**. The most common confusion pairs are SITTING vs STANDING and
WALKING_UPSTAIRS vs WALKING_DOWNSTAIRS.
