#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ==========================================================
# Cell 1 - Configuration: Small CPU or Full GPU/Narval
# ==========================================================

from pathlib import Path
import os
import random
import numpy as np
import torch

# ----------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ----------------------------------------------------------
# Device
# ----------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ----------------------------------------------------------
# Run Mode
# ----------------------------------------------------------

RUN_MODE = (
    "full_gpu"
    if device.type == "cuda"
    else "small_cpu"
)

# ----------------------------------------------------------
# Processed Dataset Location
# ----------------------------------------------------------

PROCESSED_DIR = Path(
    os.environ.get(
        "IMU_PROCESSED_PATH",
        "../datasets/DatasetIMUandBIOMARKERS/processed"
    )
).expanduser().resolve()

# ----------------------------------------------------------
# Dataset Information
# ----------------------------------------------------------

NUM_ACTIVITIES = 6
NUM_GENDERS = 2
NUM_WEIGHT_CLASSES = 3

# Embedding size used by PrivDiffuser
Z_DIM = 60

# ----------------------------------------------------------
# Hyperparameters
# ----------------------------------------------------------

if RUN_MODE == "small_cpu":

    BATCH_SIZE = 8
    EPOCHS = 5

else:

    BATCH_SIZE = 64
    EPOCHS = 5

LEARNING_RATE = 0.001

# ----------------------------------------------------------
# Display Configuration
# ----------------------------------------------------------

print("=" * 70)
print("SURROGATE MODEL CONFIGURATION")
print("=" * 70)

print("Device               :", device)
print("Run mode             :", RUN_MODE)
print("Processed directory  :", PROCESSED_DIR)
print("Activities           :", NUM_ACTIVITIES)
print("Batch size           :", BATCH_SIZE)
print("Epochs               :", EPOCHS)
print("Learning rate        :", LEARNING_RATE)
print("Embedding dimension  :", Z_DIM)


# In[2]:


# ==========================================================
# Cell 2 - Imports
# ==========================================================

import bisect
import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import (
    Dataset,
    DataLoader
)


# In[3]:


# ==========================================================
# Cell 3 - IMU Dataset
# Preloaded Multi-Subject Dataset for Fast GPU Training
# ==========================================================

class IMUDataset(Dataset):

    def __init__(
        self,
        processed_dir,
        split="train"
    ):

        super().__init__()

        self.processed_dir = Path(
            processed_dir
        ).expanduser().resolve()

        self.split = split

        if split not in ["train", "test"]:
            raise ValueError(
                "split must be either 'train' or 'test'"
            )

        if not self.processed_dir.exists():
            raise FileNotFoundError(
                f"{self.processed_dir} does not exist."
            )

        self.files = sorted(
            self.processed_dir.glob(
                f"*_{split}.npz"
            )
        )

        if len(self.files) == 0:
            raise RuntimeError(
                f"No {split} files found."
            )

        print("\n" + "=" * 70)
        print(
            f"PRELOADING {split.upper()} DATASET"
        )
        print("=" * 70)

        # --------------------------------------------------
        # Store arrays from every subject
        # --------------------------------------------------

        self.windows_by_file = []
        self.activities_by_file = []
        self.genders_by_file = []
        self.weights_by_file = []

        self.file_lengths = []

        for file_index, file_path in enumerate(
            self.files
        ):

            print(
                f"Loading subject "
                f"{file_index + 1}/{len(self.files)}: "
                f"{file_path.name}",
                flush=True
            )

            with np.load(
                file_path,
                allow_pickle=False
            ) as data:

                # IMPORTANT:
                # Copy arrays into RAM ONCE.
                windows = np.asarray(
                    data["windows"],
                    dtype=np.float32
                ).copy()

                activities = np.asarray(
                    data["activity"],
                    dtype=np.int64
                ).copy()

                gender = int(
                    data["gender"][0]
                )

                weight = int(
                    data["weight"][0]
                )

            if windows.ndim != 3:
                raise ValueError(
                    f"Unexpected windows dimensions in "
                    f"{file_path.name}: {windows.shape}"
                )

            if windows.shape[1:] != (128, 30):
                raise ValueError(
                    f"Unexpected window shape in "
                    f"{file_path.name}: {windows.shape}"
                )

            if len(windows) != len(activities):
                raise ValueError(
                    f"Window/activity mismatch in "
                    f"{file_path.name}"
                )

            self.windows_by_file.append(
                windows
            )

            self.activities_by_file.append(
                activities
            )

            self.genders_by_file.append(
                gender
            )

            self.weights_by_file.append(
                weight
            )

            self.file_lengths.append(
                len(windows)
            )

        # --------------------------------------------------
        # Global indexing boundaries
        # --------------------------------------------------

        self.cumulative_lengths = np.cumsum(
            self.file_lengths
        ).tolist()

        self.total_windows = int(
            self.cumulative_lengths[-1]
        )

        print("\n" + "=" * 70)
        print("DATASET PRELOADING COMPLETE")
        print("=" * 70)

        print(
            "Processed directory :",
            self.processed_dir
        )

        print(
            "Split               :",
            self.split
        )

        print(
            "Subject files       :",
            len(self.files)
        )

        print(
            "Total windows       :",
            self.total_windows
        )

    def __len__(self):

        return self.total_windows

    def __getitem__(self, index):

        if index < 0:
            index += self.total_windows

        if (
            index < 0
            or index >= self.total_windows
        ):
            raise IndexError(
                f"Dataset index out of range: {index}"
            )

        # --------------------------------------------------
        # Convert global index -> subject/local index
        # --------------------------------------------------

        file_id = bisect.bisect_right(
            self.cumulative_lengths,
            index
        )

        if file_id == 0:
            previous = 0
        else:
            previous = (
                self.cumulative_lengths[
                    file_id - 1
                ]
            )

        local_index = (
            index - previous
        )

        # --------------------------------------------------
        # IMPORTANT:
        # NO np.load() here.
        # Data is already in RAM.
        # --------------------------------------------------

        window = (
            self.windows_by_file[file_id][
                local_index
            ]
        )

        activity = int(
            self.activities_by_file[file_id][
                local_index
            ]
        )

        gender = int(
            self.genders_by_file[file_id]
        )

        weight = int(
            self.weights_by_file[file_id]
        )

        # --------------------------------------------------
        # Convert to tensors
        # --------------------------------------------------

        window_tensor = torch.from_numpy(
            window
        ).unsqueeze(0)

        activity_tensor = torch.tensor(
            activity,
            dtype=torch.long
        )

        gender_tensor = torch.tensor(
            gender,
            dtype=torch.long
        )

        weight_tensor = torch.tensor(
            weight,
            dtype=torch.long
        )

        return (
            window_tensor,
            activity_tensor,
            gender_tensor,
            weight_tensor
        )


# In[4]:


# ==========================================================
# Cell 4 - Dataset creation handled in Cell 5
# ==========================================================

print(
    "Dataset class ready. "
    "Datasets will be created in Cell 5."
)


# In[ ]:


# ==========================================================
# Cell 5 - Create DataLoaders
# ==========================================================

from torch.utils.data import DataLoader, Subset

# ----------------------------------------------------------
# Build datasets exactly ONCE
# ----------------------------------------------------------

full_train_dataset = IMUDataset(
    processed_dir=PROCESSED_DIR,
    split="train"
)

full_test_dataset = IMUDataset(
    processed_dir=PROCESSED_DIR,
    split="test"
)

# ----------------------------------------------------------
# Select full or small experiment
# ----------------------------------------------------------

if RUN_MODE == "small_cpu":

    train_dataset = Subset(
        full_train_dataset,
        range(
            min(
                5000,
                len(full_train_dataset)
            )
        )
    )

    test_dataset = Subset(
        full_test_dataset,
        range(
            min(
                1000,
                len(full_test_dataset)
            )
        )
    )

else:

    train_dataset = full_train_dataset
    test_dataset = full_test_dataset


# Data is already in RAM.
# Start with zero workers for reliability.
NUM_WORKERS = 0

# ----------------------------------------------------------
# DataLoaders
# ----------------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(
        device.type == "cuda"
    ),
    drop_last=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(
        device.type == "cuda"
    ),
    drop_last=False
)

print("\n" + "=" * 70)
print("DATALOADERS")
print("=" * 70)

print("Run mode         :", RUN_MODE)
print("Training samples :", len(train_dataset))
print("Testing samples  :", len(test_dataset))
print("Training batches :", len(train_loader))
print("Testing batches  :", len(test_loader))
print("Batch size       :", BATCH_SIZE)
print("Workers          :", NUM_WORKERS)
print(
    "Pinned memory    :",
    device.type == "cuda"
)


# In[6]:


# ==========================================================
# Cell 6 - Surrogate Activity Classifier
# ==========================================================

import torch
import torch.nn as nn


class SurrogateClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.feature_extractor = nn.Sequential(

            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.embedding = nn.Sequential(
            nn.Linear(32, Z_DIM),
            nn.ReLU()
        )

        self.classifier = nn.Linear(
            Z_DIM,
            NUM_ACTIVITIES
        )

    def forward(self, x):

        x = self.feature_extractor(x)

        x = x.view(x.size(0), -1)

        z = self.embedding(x)

        logits = self.classifier(z)

        return logits, z


model = SurrogateClassifier().to(device)

print("=" * 70)
print("SURROGATE ACTIVITY CLASSIFIER")
print("=" * 70)

print("Run mode :", RUN_MODE)
print("Device   :", device)
print("Activities:", NUM_ACTIVITIES)
print("Embedding:", Z_DIM)

print("\nModel Architecture\n")
print(model)


# In[9]:


# ==========================================================
# Cell 7 - Verify Model Using One Batch
# ==========================================================

windows, activities, genders, weights = next(
    iter(train_loader)
)

windows = windows.to(
    device,
    dtype=torch.float32
)

with torch.no_grad():

    logits, embedding = model(windows)

print("=" * 70)
print("MODEL VERIFICATION")
print("=" * 70)

print("Input shape      :", tuple(windows.shape))
print("Activity shape   :", tuple(activities.shape))
print("Logits shape     :", tuple(logits.shape))
print("Embedding shape  :", tuple(embedding.shape))

assert logits.shape == (
    windows.size(0),
    NUM_ACTIVITIES
)

assert embedding.shape == (
    windows.size(0),
    Z_DIM
)

print("\n Model verification passed.")


# In[ ]:


# ==========================================================
# Cell 8 - Train the Surrogate Activity Classifier
# ==========================================================

import time

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

training_losses = []
training_accuracies = []

print("=" * 75)
print("SURROGATE MODEL TRAINING")
print("=" * 75)

print("Device        :", device)
print("Run mode      :", RUN_MODE)
print("Epochs        :", EPOCHS)
print("Learning rate :", LEARNING_RATE)
print("Batch size    :", BATCH_SIZE)

start_time = time.time()

for epoch in range(EPOCHS):

    print(f"\n========== Starting Epoch {epoch + 1}/{EPOCHS} ==========")

    model.train()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for batch_idx, (windows, activities, genders, weights) in enumerate(train_loader):

        if batch_idx == 0:
            print("First batch received from DataLoader")

        windows = windows.to(
            device,
            dtype=torch.float32,
            non_blocking=True
        )

        activities = activities.to(
            device,
            dtype=torch.long,
            non_blocking=True
        )

        if batch_idx == 0:
            print("First batch moved to GPU")

        optimizer.zero_grad(set_to_none=True)

        logits, embedding = model(windows)

        if batch_idx == 0:
            print("First forward pass completed")

        loss = criterion(
            logits,
            activities
        )

        loss.backward()

        optimizer.step()

        if batch_idx == 0:
            print("First optimization step completed")

        batch_size = activities.size(0)

        running_loss += (
            loss.item() * batch_size
        )

        predictions = torch.argmax(
            logits,
            dim=1
        )

        correct_predictions += (
            predictions == activities
        ).sum().item()

        total_samples += batch_size

        # --------------------------------------------------
        # Progress every 100 batches
        # --------------------------------------------------

        if batch_idx % 100 == 0:

            print(
                f"Epoch {epoch + 1:02d}/{EPOCHS:02d} | "
                f"Batch {batch_idx:05d}/{len(train_loader):05d} | "
                f"Loss: {loss.item():.4f}"
            )

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples

    training_losses.append(epoch_loss)
    training_accuracies.append(epoch_accuracy)

    print("\n---------------------------------------------")
    print(f"Epoch {epoch + 1:02d} completed")
    print(f"Training Loss     : {epoch_loss:.4f}")
    print(f"Training Accuracy : {epoch_accuracy:.4f}")
    print("---------------------------------------------")

training_time = time.time() - start_time

print("\n" + "=" * 75)
print("SURROGATE TRAINING COMPLETED")
print("=" * 75)

print(f"Final Training Loss     : {training_losses[-1]:.4f}")
print(f"Final Training Accuracy : {training_accuracies[-1]:.4f}")
print(f"Total Training Time     : {training_time:.2f} seconds")


# In[ ]:


# ==========================================================
# Cell 9 - Evaluate the Surrogate Activity Classifier
# ==========================================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():

    for windows, activities, genders, weights in test_loader:

        windows = windows.to(
            device,
            dtype=torch.float32,
            non_blocking=True
        )

        activities = activities.to(
            device,
            dtype=torch.long,
            non_blocking=True
        )

        logits, embedding = model(windows)

        predictions = torch.argmax(
            logits,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            activities.cpu().numpy()
        )

# ----------------------------------------------------------
# Compute metrics
# ----------------------------------------------------------

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

macro_precision = precision_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

conf_matrix = confusion_matrix(
    all_labels,
    all_predictions
)

print("=" * 75)
print("SURROGATE ACTIVITY CLASSIFICATION RESULTS")
print("=" * 75)

print("Run mode            :", RUN_MODE)
print("Device              :", device)
print("Training samples    :", len(train_dataset))
print("Testing samples     :", len(test_dataset))

print(f"\nAccuracy            : {accuracy:.4f}")
print(f"Macro Precision     : {macro_precision:.4f}")
print(f"Macro Recall        : {macro_recall:.4f}")
print(f"Macro F1-score      : {macro_f1:.4f}")
print(f"Weighted F1-score   : {weighted_f1:.4f}")

print("\nConfusion Matrix")
print(conf_matrix)

print("\nClassification Report")
print(
    classification_report(
        all_labels,
        all_predictions,
        digits=4,
        zero_division=0
    )
)


# In[ ]:


# ==========================================================
# Cell 10 - Save Model and Results
# ==========================================================

from pathlib import Path
import pandas as pd
import torch

# ----------------------------------------------------------
# Create output directory
# ----------------------------------------------------------

OUTPUT_DIR = Path("results/surrogate")
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ----------------------------------------------------------
# Save trained model
# ----------------------------------------------------------

checkpoint = {

    "model_state_dict": model.state_dict(),

    "optimizer_state_dict": optimizer.state_dict(),

    "training_losses": training_losses,

    "training_accuracies": training_accuracies,

    "epochs": EPOCHS,

    "learning_rate": LEARNING_RATE,

    "batch_size": BATCH_SIZE,

    "run_mode": RUN_MODE,

    "z_dim": Z_DIM,

    "num_classes": NUM_ACTIVITIES

}

MODEL_PATH = OUTPUT_DIR / "surrogate_activity_classifier.pt"

torch.save(
    checkpoint,
    MODEL_PATH
)

print("Model saved:")
print(MODEL_PATH)

# ----------------------------------------------------------
# Save summary metrics
# ----------------------------------------------------------

results_summary = pd.DataFrame(
    [
        {
            "Run Mode": RUN_MODE,
            "Training Samples": len(train_dataset),
            "Testing Samples": len(test_dataset),
            "Accuracy": accuracy,
            "Macro Precision": macro_precision,
            "Macro Recall": macro_recall,
            "Macro F1": macro_f1,
            "Weighted F1": weighted_f1,
            "Epochs": EPOCHS,
            "Batch Size": BATCH_SIZE,
            "Learning Rate": LEARNING_RATE
        }
    ]
)

summary_path = OUTPUT_DIR / "surrogate_results.csv"

results_summary.to_csv(
    summary_path,
    index=False
)

print("Results summary saved:")
print(summary_path)

# ----------------------------------------------------------
# Save confusion matrix
# ----------------------------------------------------------

confusion_df = pd.DataFrame(
    conf_matrix
)

confusion_path = OUTPUT_DIR / "confusion_matrix.csv"

confusion_df.to_csv(
    confusion_path,
    index=False
)

print("Confusion matrix saved:")
print(confusion_path)

# ----------------------------------------------------------
# Save classification report
# ----------------------------------------------------------

report_dict = classification_report(
    all_labels,
    all_predictions,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(
    report_dict
).transpose()

report_path = OUTPUT_DIR / "classification_report.csv"

report_df.to_csv(
    report_path
)

print("Classification report saved:")
print(report_path)

# ----------------------------------------------------------
# Training history
# ----------------------------------------------------------

history = pd.DataFrame(
    {
        "Epoch": range(
            1,
            EPOCHS + 1
        ),
        "Loss": training_losses,
        "Training Accuracy": training_accuracies
    }
)

history_path = OUTPUT_DIR / "training_history.csv"

history.to_csv(
    history_path,
    index=False
)

print("Training history saved:")
print(history_path)

print("\n" + "=" * 75)
print("SURROGATE MODEL PIPELINE COMPLETE")
print("=" * 75)

print(f"Model            : {MODEL_PATH}")
print(f"Summary          : {summary_path}")
print(f"History          : {history_path}")
print(f"Confusion Matrix : {confusion_path}")
print(f"Classification   : {report_path}")


# In[ ]:




