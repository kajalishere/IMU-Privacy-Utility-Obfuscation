#!/usr/bin/env python
# coding: utf-8

# In[12]:


# ==========================================================
# Cell 1 - Privacy-Guided Diffusion Configuration
# ==========================================================

import os
import random
from pathlib import Path

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
# Run mode and device
# ----------------------------------------------------------

RUN_MODE = os.environ.get(
    "RUN_MODE",
    "small_cpu"
)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ----------------------------------------------------------
# Project paths
# ----------------------------------------------------------

PROJECT_ROOT = Path(
    os.environ.get(
        "PRIVDIFFUSER_ROOT",
        r"C:\ResearchS26\PrivDiffuser_GPU"
    )
)

WORK_DIR = Path(
    os.environ.get(
        "PRIVDIFFUSER_WORK_DIR",
        PROJECT_ROOT / "new_dataset"
    )
)

PROCESSED_DIR = Path(
    os.environ.get(
        "IMU_PROCESSED_PATH",
        PROJECT_ROOT
        / "datasets"
        / "DatasetIMUandBIOMARKERS"
        / "processed"
    )
)


# ----------------------------------------------------------
# Existing trained classifiers
# ----------------------------------------------------------

if RUN_MODE == "small_cpu":

    # Local CPU checkpoints
    SURROGATE_CHECKPOINT = (
        WORK_DIR
        / "small_surrogate_activity_model.pt"
    )

    PRIVACY_CHECKPOINT = (
        WORK_DIR
        / "small_cpu_privacy_gender_model.pt"
    )

    PRIVACY_GUIDED_CHECKPOINT = (
        WORK_DIR
        / "small_privacy_guided_denoiser.pt"
    )

else:

    # Full Narval/GPU checkpoints
    SURROGATE_CHECKPOINT = (
        WORK_DIR
        / "full_surrogate_activity_model.pt"
    )

    PRIVACY_CHECKPOINT = (
        WORK_DIR
        / "full_gpu_privacy_gender_model.pt"
    )

    PRIVACY_GUIDED_CHECKPOINT = (
        WORK_DIR
        / "full_privacy_guided_denoiser.pt"
    )

# ----------------------------------------------------------
# Dataset configuration
# ----------------------------------------------------------

WINDOW_LENGTH = 128
NUM_FEATURES = 30

NUM_ACTIVITIES = 6
NUM_PRIVATE_CLASSES = 2

Z_DIM = 60


# ----------------------------------------------------------
# Training configuration
# ----------------------------------------------------------

if RUN_MODE == "small_cpu":

    BATCH_SIZE = 4
    PRIVACY_GUIDED_EPOCHS = 1
    MAX_BATCHES_PER_EPOCH = 10

else:

    BATCH_SIZE = 16
    PRIVACY_GUIDED_EPOCHS = 5
    MAX_BATCHES_PER_EPOCH = 500


LEARNING_RATE = 1e-4

DIFFUSION_STEPS = 1000


# ----------------------------------------------------------
# Privacy-guided loss weights
# ----------------------------------------------------------

LAMBDA_UTILITY = 1.0
LAMBDA_PRIVACY = 1.0


# ----------------------------------------------------------
# DataLoader configuration
# ----------------------------------------------------------

NUM_WORKERS = int(
    os.environ.get(
        "NUM_WORKERS",
        "0"
    )
)


# ----------------------------------------------------------
# Basic validation
# ----------------------------------------------------------

if not PROJECT_ROOT.exists():
    raise FileNotFoundError(
        "PrivDiffuser project folder was not found:\n"
        f"{PROJECT_ROOT}"
    )

if not PROCESSED_DIR.exists():
    raise FileNotFoundError(
        "Processed IMU dataset was not found:\n"
        f"{PROCESSED_DIR}"
    )


# ----------------------------------------------------------
# Configuration summary
# ----------------------------------------------------------

print("=" * 80)
print("PRIVACY-GUIDED DIFFUSION CONFIGURATION")
print("=" * 80)

print("Run mode                  :", RUN_MODE)
print("Device                    :", device)

print("Project root              :", PROJECT_ROOT)
print("Working directory         :", WORK_DIR)
print("Processed dataset         :", PROCESSED_DIR)

print("Surrogate checkpoint      :", SURROGATE_CHECKPOINT)
print("Privacy checkpoint        :", PRIVACY_CHECKPOINT)
print("Privacy-guided checkpoint :", PRIVACY_GUIDED_CHECKPOINT)

print("Batch size                :", BATCH_SIZE)
print("Training epochs           :", PRIVACY_GUIDED_EPOCHS)
print("Maximum batches/epoch     :", MAX_BATCHES_PER_EPOCH)

print("Learning rate             :", LEARNING_RATE)
print("Diffusion steps           :", DIFFUSION_STEPS)

print("Utility loss weight       :", LAMBDA_UTILITY)
print("Privacy loss weight       :", LAMBDA_PRIVACY)

print("Number of workers         :", NUM_WORKERS)

print("=" * 80)
print("Configuration initialized successfully.")


# In[42]:


# ==========================================================
# Cell 2 - Import Core Diffusion Modules
# ==========================================================

import sys
import importlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 75)
print("IMPORTING CORE DIFFUSION MODULES")
print("=" * 75)

# ----------------------------------------------------------
# Ensure project root is available to Python
# ----------------------------------------------------------

PRIVDIFFUSER_ROOT = PROJECT_ROOT

project_root_string = str(
    PRIVDIFFUSER_ROOT
)

if project_root_string in sys.path:
    sys.path.remove(
        project_root_string
    )

sys.path.insert(
    0,
    project_root_string
)

importlib.invalidate_caches()


# ----------------------------------------------------------
# Import U-Net
# ----------------------------------------------------------

try:
    from unet.unet import Unet

except ImportError:
    from unet import Unet

print("[OK] Unet imported")


# ----------------------------------------------------------
# Import Gaussian diffusion
# ----------------------------------------------------------

try:
    from diffusion.diffusion import GaussianDiffusion

except ImportError:
    from diffusion import GaussianDiffusion

print("[OK] GaussianDiffusion imported")


# ----------------------------------------------------------
# Import conditional embedding
# ----------------------------------------------------------

try:
    from embedding.embedding import ConditionalEmbedding

except ImportError:
    from embedding import ConditionalEmbedding

print("[OK] ConditionalEmbedding imported")


print("-" * 75)
print("Core diffusion modules imported successfully.")


# In[13]:


# ==========================================================
# Cell 3 - Preloaded IMU Dataset for Privacy-Guided Training
# CPU and Narval/GPU Compatible
# ==========================================================

import gc
from pathlib import Path

import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader


# ----------------------------------------------------------
# Notebook 07 configuration compatibility
# ----------------------------------------------------------

WINDOW_SIZE = WINDOW_LENGTH
NUM_GENDERS = NUM_PRIVATE_CLASSES


# ==========================================================
# IMU Dataset
# ==========================================================

class IMUDataset(Dataset):
    """
    Preloaded subject-wise IMU dataset.

    small_cpu:
        Select a limited balanced number of windows from
        each activity within each subject.

    full_gpu:
        Use all available windows from all subject files.

    All selected windows are loaded once during dataset
    initialization. __getitem__ therefore performs no
    repeated np.load() calls during training.
    """

    def __init__(
        self,
        processed_dir,
        split="train",
        windows_per_activity_per_file=None,
        seed=42
    ):
        super().__init__()

        self.processed_dir = Path(
            processed_dir
        ).expanduser().resolve()

        self.split = split

        self.windows_per_activity_per_file = (
            windows_per_activity_per_file
        )

        self.seed = seed

        if split not in {"train", "test"}:
            raise ValueError(
                "split must be 'train' or 'test'."
            )

        if not self.processed_dir.exists():
            raise FileNotFoundError(
                "Processed directory was not found:\n"
                f"{self.processed_dir}"
            )

        # --------------------------------------------------
        # Find subject files
        # --------------------------------------------------

        self.files = sorted(
            self.processed_dir.glob(
                f"*_{split}.npz"
            )
        )

        if not self.files:
            raise FileNotFoundError(
                f"No '*_{split}.npz' files found in:\n"
                f"{self.processed_dir}"
            )

        # --------------------------------------------------
        # Temporary storage
        # --------------------------------------------------

        all_windows = []
        all_activities = []
        all_genders = []
        all_weights = []

        rng = np.random.default_rng(
            seed
        )

        print("\n" + "=" * 75)

        print(
            f"PRELOADING {split.upper()} "
            f"PRIVACY-GUIDED DIFFUSION DATASET"
        )

        print("=" * 75)

        # --------------------------------------------------
        # Load each subject file once
        # --------------------------------------------------

        for file_number, file_path in enumerate(
            self.files,
            start=1
        ):

            with np.load(
                file_path,
                allow_pickle=False
            ) as data:

                windows = np.asarray(
                    data["windows"],
                    dtype=np.float32
                )

                activities = np.asarray(
                    data["activity"],
                    dtype=np.int64
                )

                gender = int(
                    np.asarray(
                        data["gender"]
                    ).reshape(-1)[0]
                )

                weight = int(
                    np.asarray(
                        data["weight"]
                    ).reshape(-1)[0]
                )

                # ------------------------------------------
                # Full GPU:
                # use every available window.
                # ------------------------------------------

                if (
                    self.windows_per_activity_per_file
                    is None
                ):

                    selected_indices = np.arange(
                        len(activities),
                        dtype=np.int64
                    )

                # ------------------------------------------
                # Small CPU:
                # select balanced windows/activity.
                # ------------------------------------------

                else:

                    selected_indices_list = []

                    for activity_id in range(
                        NUM_ACTIVITIES
                    ):

                        activity_indices = (
                            np.flatnonzero(
                                activities
                                == activity_id
                            )
                        )

                        if len(activity_indices) == 0:
                            continue

                        number_to_select = min(
                            self.windows_per_activity_per_file,
                            len(activity_indices)
                        )

                        chosen = rng.choice(
                            activity_indices,
                            size=number_to_select,
                            replace=False
                        )

                        selected_indices_list.extend(
                            chosen.tolist()
                        )

                    selected_indices = np.asarray(
                        selected_indices_list,
                        dtype=np.int64
                    )

                # ------------------------------------------
                # Copy selected data into memory
                # ------------------------------------------

                selected_windows = np.asarray(
                    windows[selected_indices],
                    dtype=np.float32
                ).copy()

                selected_activities = np.asarray(
                    activities[selected_indices],
                    dtype=np.int64
                ).copy()

            number_selected = len(
                selected_indices
            )

            selected_genders = np.full(
                number_selected,
                gender,
                dtype=np.int64
            )

            selected_weights = np.full(
                number_selected,
                weight,
                dtype=np.int64
            )

            all_windows.append(
                selected_windows
            )

            all_activities.append(
                selected_activities
            )

            all_genders.append(
                selected_genders
            )

            all_weights.append(
                selected_weights
            )

            print(
                f"[{file_number:02d}/{len(self.files):02d}] "
                f"{file_path.name}: "
                f"{number_selected} windows"
            )

            del windows
            del activities
            del selected_windows

            gc.collect()

        # --------------------------------------------------
        # Combine all subjects
        # --------------------------------------------------

        print("\nCombining subject arrays...")

        self.windows = np.concatenate(
            all_windows,
            axis=0
        )

        self.activities = np.concatenate(
            all_activities,
            axis=0
        )

        self.genders = np.concatenate(
            all_genders,
            axis=0
        )

        self.weights = np.concatenate(
            all_weights,
            axis=0
        )

        del all_windows
        del all_activities
        del all_genders
        del all_weights

        gc.collect()

        # --------------------------------------------------
        # Validate
        # --------------------------------------------------

        number_of_samples = len(
            self.windows
        )

        assert len(self.activities) == number_of_samples
        assert len(self.genders) == number_of_samples
        assert len(self.weights) == number_of_samples

        if self.windows.shape[1:] != (
            WINDOW_SIZE,
            NUM_FEATURES
        ):
            raise ValueError(
                "Unexpected IMU window shape: "
                f"{self.windows.shape}"
            )

        # --------------------------------------------------
        # Dataset summary
        # --------------------------------------------------

        activity_counts = np.bincount(
            self.activities,
            minlength=NUM_ACTIVITIES
        )

        gender_counts = np.bincount(
            self.genders,
            minlength=NUM_GENDERS
        )

        print("\n" + "-" * 75)

        print(
            f"{split.capitalize()} files     :",
            len(self.files)
        )

        print(
            f"{split.capitalize()} windows   :",
            number_of_samples
        )

        print(
            "Activity counts :",
            activity_counts.tolist()
        )

        print(
            "Gender counts   :",
            gender_counts.tolist()
        )

        memory_gb = (
            self.windows.nbytes
            + self.activities.nbytes
            + self.genders.nbytes
            + self.weights.nbytes
        ) / (1024 ** 3)

        print(
            "Approx RAM used :",
            f"{memory_gb:.2f} GB"
        )

        print(
            f"\n{split.capitalize()} dataset "
            "preloaded successfully."
        )

    def __len__(self):

        return len(
            self.windows
        )

    def __getitem__(
        self,
        index
    ):

        window = torch.from_numpy(
            self.windows[index]
        ).unsqueeze(0)

        activity = torch.tensor(
            self.activities[index],
            dtype=torch.long
        )

        gender = torch.tensor(
            self.genders[index],
            dtype=torch.long
        )

        weight = torch.tensor(
            self.weights[index],
            dtype=torch.long
        )

        return (
            window,
            activity,
            gender,
            weight
        )


# ==========================================================
# Dataset Size Configuration
# ==========================================================

if RUN_MODE == "small_cpu":

    TRAIN_WINDOWS_PER_ACTIVITY = 20
    TEST_WINDOWS_PER_ACTIVITY = 10

else:

    # Full Narval experiment:
    # use all available processed windows.

    TRAIN_WINDOWS_PER_ACTIVITY = None
    TEST_WINDOWS_PER_ACTIVITY = None


# ==========================================================
# Create Datasets
# ==========================================================

train_dataset = IMUDataset(
    processed_dir=PROCESSED_DIR,
    split="train",
    windows_per_activity_per_file=(
        TRAIN_WINDOWS_PER_ACTIVITY
    ),
    seed=42
)

test_dataset = IMUDataset(
    processed_dir=PROCESSED_DIR,
    split="test",
    windows_per_activity_per_file=(
        TEST_WINDOWS_PER_ACTIVITY
    ),
    seed=43
)


# ==========================================================
# Create DataLoaders
# ==========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(
        device.type == "cuda"
    ),
    drop_last=True
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


# ==========================================================
# Dataset Distribution
# ==========================================================

train_activity_counts = torch.bincount(
    torch.from_numpy(
        train_dataset.activities
    ),
    minlength=NUM_ACTIVITIES
)

train_gender_counts = torch.bincount(
    torch.from_numpy(
        train_dataset.genders
    ),
    minlength=NUM_GENDERS
)

test_activity_counts = torch.bincount(
    torch.from_numpy(
        test_dataset.activities
    ),
    minlength=NUM_ACTIVITIES
)

test_gender_counts = torch.bincount(
    torch.from_numpy(
        test_dataset.genders
    ),
    minlength=NUM_GENDERS
)


# ==========================================================
# Verify One Training Batch
# ==========================================================

sample_batch = next(
    iter(train_loader)
)

sample_windows = sample_batch[0]
sample_activities = sample_batch[1]
sample_genders = sample_batch[2]
sample_weights = sample_batch[3]


print("\n" + "=" * 75)
print("PRIVACY-GUIDED DATASET AND DATALOADER CHECK")
print("=" * 75)

print(
    "Training samples       :",
    len(train_dataset)
)

print(
    "Testing samples        :",
    len(test_dataset)
)

print(
    "Training batches       :",
    len(train_loader)
)

print(
    "Testing batches        :",
    len(test_loader)
)

print(
    "\nWindow batch shape     :",
    tuple(sample_windows.shape)
)

print(
    "Activity label shape   :",
    tuple(sample_activities.shape)
)

print(
    "Gender label shape     :",
    tuple(sample_genders.shape)
)

print(
    "Weight label shape     :",
    tuple(sample_weights.shape)
)

print(
    "\nTraining activity counts:",
    train_activity_counts.tolist()
)

print(
    "Training gender counts  :",
    train_gender_counts.tolist()
)

print(
    "\nTesting activity counts :",
    test_activity_counts.tolist()
)

print(
    "Testing gender counts   :",
    test_gender_counts.tolist()
)

print(
    "\nDataset loaders created successfully."
)


# In[15]:


# ==========================================================
# Cell 4 - Load Frozen Surrogate Utility Classifier
# ==========================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------
# Surrogate activity classifier architecture
# Must match the checkpoint trained in the baseline experiment
# ----------------------------------------------------------

class SmallSurrogateClassifier(nn.Module):
    """
    Frozen activity classifier used to preserve utility.

    Input:
        x: (batch, 1, 128, 30)

    Outputs:
        logits:   (batch, 6)
        z_public: (batch, 60)
    """

    def __init__(
        self,
        num_classes=6,
        z_dim=60
    ):
        super().__init__()

        self.feature_extractor = nn.Sequential(

            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2
            ),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                output_size=(1, 1)
            )
        )

        # IMPORTANT:
        # These names match the saved checkpoint:
        # embedding.0.weight
        # embedding.0.bias

        self.embedding = nn.Sequential(
            nn.Linear(
                in_features=32,
                out_features=z_dim
            ),
            nn.ReLU()
        )

        # IMPORTANT:
        # These names match:
        # classifier.weight
        # classifier.bias

        self.classifier = nn.Linear(
            in_features=z_dim,
            out_features=num_classes
        )

    def forward(self, x):

        features = self.feature_extractor(
            x
        )

        features = torch.flatten(
            features,
            start_dim=1
        )

        z_public = self.embedding(
            features
        )

        logits = self.classifier(
            z_public
        )

        return logits, z_public


# ----------------------------------------------------------
# Verify checkpoint exists
# ----------------------------------------------------------

if not SURROGATE_CHECKPOINT.exists():
    raise FileNotFoundError(
        "Surrogate activity checkpoint was not found:\n"
        f"{SURROGATE_CHECKPOINT}"
    )


# ----------------------------------------------------------
# Create model
# ----------------------------------------------------------

surrogate_model = SmallSurrogateClassifier(
    num_classes=NUM_ACTIVITIES,
    z_dim=Z_DIM
).to(device)


# ----------------------------------------------------------
# Load checkpoint
# ----------------------------------------------------------

surrogate_checkpoint = torch.load(
    SURROGATE_CHECKPOINT,
    map_location=device,
    weights_only=False
)

if "model_state_dict" not in surrogate_checkpoint:
    raise KeyError(
        "The surrogate checkpoint does not contain "
        "'model_state_dict'."
    )

# ----------------------------------------------------------
# Load state dictionary
# Support both small-CPU and full-GPU checkpoint names
# ----------------------------------------------------------

surrogate_state_dict = surrogate_checkpoint[
    "model_state_dict"
]

# Small-CPU checkpoint used the older layer names:
# embedding_layer.*
# activity_classifier.*
#
# Notebook 07 uses:
# embedding.0.*
# classifier.*

if "embedding_layer.weight" in surrogate_state_dict:

    print(
        "Detected small-CPU surrogate checkpoint "
        "with legacy layer names."
    )

    surrogate_state_dict = dict(
        surrogate_state_dict
    )

    surrogate_state_dict[
        "embedding.0.weight"
    ] = surrogate_state_dict.pop(
        "embedding_layer.weight"
    )

    surrogate_state_dict[
        "embedding.0.bias"
    ] = surrogate_state_dict.pop(
        "embedding_layer.bias"
    )

    surrogate_state_dict[
        "classifier.weight"
    ] = surrogate_state_dict.pop(
        "activity_classifier.weight"
    )

    surrogate_state_dict[
        "classifier.bias"
    ] = surrogate_state_dict.pop(
        "activity_classifier.bias"
    )

surrogate_model.load_state_dict(
    surrogate_state_dict,
    strict=True
)


# ----------------------------------------------------------
# Freeze surrogate classifier
# ----------------------------------------------------------

surrogate_model.eval()

for parameter in surrogate_model.parameters():
    parameter.requires_grad = False


# ----------------------------------------------------------
# Verify model using one batch
# ----------------------------------------------------------

verification_batch = next(
    iter(train_loader)
)

verification_windows = (
    verification_batch[0]
    .to(device)
)

verification_activities = (
    verification_batch[1]
    .to(device)
)

with torch.no_grad():

    activity_logits, z_public = (
        surrogate_model(
            verification_windows
        )
    )

    activity_predictions = (
        activity_logits.argmax(
            dim=1
        )
    )


# ----------------------------------------------------------
# Final checks
# ----------------------------------------------------------

if activity_logits.shape != (
    verification_windows.shape[0],
    NUM_ACTIVITIES
):
    raise RuntimeError(
        "Unexpected activity logits shape: "
        f"{tuple(activity_logits.shape)}"
    )

if z_public.shape != (
    verification_windows.shape[0],
    Z_DIM
):
    raise RuntimeError(
        "Unexpected public embedding shape: "
        f"{tuple(z_public.shape)}"
    )


# ----------------------------------------------------------
# Output
# ----------------------------------------------------------

print("=" * 75)
print("SURROGATE UTILITY CLASSIFIER CHECK")
print("=" * 75)

print(
    "Checkpoint loaded  :",
    SURROGATE_CHECKPOINT.name
)

print(
    "Model device       :",
    next(
        surrogate_model.parameters()
    ).device
)

print(
    "Model frozen       :",
    all(
        not parameter.requires_grad
        for parameter
        in surrogate_model.parameters()
    )
)

print(
    "Logits shape       :",
    tuple(
        activity_logits.shape
    )
)

print(
    "Public embedding   :",
    tuple(
        z_public.shape
    )
)

print(
    "True activities    :",
    verification_activities
    .detach()
    .cpu()
    .tolist()
)

print(
    "Predicted activities:",
    activity_predictions
    .detach()
    .cpu()
    .tolist()
)

print(
    "\nSurrogate utility classifier "
    "loaded and frozen successfully."
)


# In[16]:


# ==========================================================
# Cell 5 - Load Frozen Gender Privacy Classifier
# ==========================================================

import torch
import torch.nn as nn


# ----------------------------------------------------------
# Privacy classifier architecture
# ----------------------------------------------------------

class PrivacyClassifier(nn.Module):
    """
    Frozen gender classifier.

    Input:
        x: (batch, 1, 128, 30)

    Outputs:
        logits:    (batch, 2)
        z_private: (batch, 60)
    """

    def __init__(
        self,
        num_classes=2,
        z_dim=60
    ):
        super().__init__()

        self.feature_extractor = nn.Sequential(

            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2
            ),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                output_size=(1, 1)
            )
        )

        self.private_embedding = nn.Linear(
            in_features=32,
            out_features=z_dim
        )

        self.private_classifier = nn.Linear(
            in_features=z_dim,
            out_features=num_classes
        )

    def forward(self, x):

        features = self.feature_extractor(
            x
        )

        features = torch.flatten(
            features,
            start_dim=1
        )

        z_private = F.relu(
            self.private_embedding(
                features
            )
        )

        logits = self.private_classifier(
            z_private
        )

        return logits, z_private


# ----------------------------------------------------------
# Verify checkpoint exists
# ----------------------------------------------------------

if not PRIVACY_CHECKPOINT.exists():

    raise FileNotFoundError(
        "Privacy classifier checkpoint was not found:\n"
        f"{PRIVACY_CHECKPOINT}"
    )


# ----------------------------------------------------------
# Create privacy classifier
# ----------------------------------------------------------

privacy_model = PrivacyClassifier(
    num_classes=NUM_GENDERS,
    z_dim=Z_DIM
).to(device)


# ----------------------------------------------------------
# Load checkpoint
# ----------------------------------------------------------

privacy_checkpoint = torch.load(
    PRIVACY_CHECKPOINT,
    map_location=device,
    weights_only=False
)

if "model_state_dict" not in privacy_checkpoint:

    raise KeyError(
        "The privacy checkpoint does not contain "
        "'model_state_dict'."
    )


privacy_state_dict = privacy_checkpoint[
    "model_state_dict"
]


# ----------------------------------------------------------
# Handle possible legacy small-CPU checkpoint names
# ----------------------------------------------------------

print(
    "Privacy checkpoint:",
    PRIVACY_CHECKPOINT.name
)

print(
    "Checkpoint layer names:",
    list(
        privacy_state_dict.keys()
    )
)


# ----------------------------------------------------------
# First attempt: direct checkpoint loading
# ----------------------------------------------------------

try:

    privacy_model.load_state_dict(
        privacy_state_dict,
        strict=True
    )

except RuntimeError as error:

    print(
        "\nDirect privacy checkpoint loading "
        "did not match the current architecture."
    )

    print(
        "We will inspect the checkpoint names "
        "before changing them."
    )

    raise error


# ----------------------------------------------------------
# Freeze privacy classifier
# ----------------------------------------------------------

privacy_model.eval()

for parameter in privacy_model.parameters():

    parameter.requires_grad = False


# ----------------------------------------------------------
# Verify using one training batch
# ----------------------------------------------------------

verification_batch = next(
    iter(train_loader)
)

verification_windows = (
    verification_batch[0]
    .to(device)
)

verification_genders = (
    verification_batch[2]
    .to(device)
)


with torch.no_grad():

    gender_logits, z_private = (
        privacy_model(
            verification_windows
        )
    )

    gender_predictions = (
        gender_logits.argmax(
            dim=1
        )
    )


# ----------------------------------------------------------
# Validate shapes
# ----------------------------------------------------------

if gender_logits.shape != (
    verification_windows.shape[0],
    NUM_GENDERS
):

    raise RuntimeError(
        "Unexpected gender logits shape: "
        f"{tuple(gender_logits.shape)}"
    )


if z_private.shape != (
    verification_windows.shape[0],
    Z_DIM
):

    raise RuntimeError(
        "Unexpected private embedding shape: "
        f"{tuple(z_private.shape)}"
    )


# ----------------------------------------------------------
# Final output
# ----------------------------------------------------------

print("\n" + "=" * 75)
print("PRIVACY CLASSIFIER CHECK")
print("=" * 75)

print(
    "Checkpoint loaded   :",
    PRIVACY_CHECKPOINT.name
)

print(
    "Model device        :",
    next(
        privacy_model.parameters()
    ).device
)

print(
    "Model frozen        :",
    all(
        not parameter.requires_grad
        for parameter
        in privacy_model.parameters()
    )
)

print(
    "Logits shape        :",
    tuple(
        gender_logits.shape
    )
)

print(
    "Private embedding   :",
    tuple(
        z_private.shape
    )
)

print(
    "True genders        :",
    verification_genders
    .detach()
    .cpu()
    .tolist()
)

print(
    "Predicted genders   :",
    gender_predictions
    .detach()
    .cpu()
    .tolist()
)

print(
    "\nPrivacy classifier loaded "
    "and frozen successfully."
)


# In[27]:


# ==========================================================
# Cell 6 - Forward Diffusion Setup
# ==========================================================

import numpy as np
import torch


# ----------------------------------------------------------
# Diffusion configuration
# ----------------------------------------------------------

# Match the diffusion schedule used by the
# small-CPU baseline checkpoint.
if RUN_MODE == "small_cpu":
    DIFFUSION_STEPS = 50
else:
    DIFFUSION_STEPS = 1000

BETA_START = 1e-4
BETA_END = 0.02


# ----------------------------------------------------------
# Linear beta schedule
# ----------------------------------------------------------

betas = torch.linspace(
    BETA_START,
    BETA_END,
    DIFFUSION_STEPS,
    dtype=torch.float32,
    device=device
)


# ----------------------------------------------------------
# Alpha values
# ----------------------------------------------------------

alphas = 1.0 - betas

alphas_cumprod = torch.cumprod(
    alphas,
    dim=0
)


# ----------------------------------------------------------
# Precompute diffusion coefficients
# ----------------------------------------------------------

sqrt_alphas_cumprod = torch.sqrt(
    alphas_cumprod
)

sqrt_one_minus_alphas_cumprod = torch.sqrt(
    1.0 - alphas_cumprod
)


# ----------------------------------------------------------
# Helper:
# extract timestep-dependent coefficients
# ----------------------------------------------------------

def extract(
    coefficient_tensor,
    timesteps,
    target_shape
):
    """
    Selects the diffusion coefficient corresponding
    to each sample's timestep and reshapes it so that
    it can be broadcast across an IMU batch.
    """

    batch_size = timesteps.shape[0]

    selected = coefficient_tensor.gather(
        0,
        timesteps
    )

    return selected.reshape(
        batch_size,
        *((1,) * (len(target_shape) - 1))
    )


# ----------------------------------------------------------
# Forward diffusion function q(x_t | x_0)
# ----------------------------------------------------------

def forward_diffusion_sample(
    clean_windows,
    timesteps,
    noise=None
):
    """
    Adds Gaussian noise to clean IMU windows.

    x_t =
        sqrt(alpha_bar_t) * x_0
        +
        sqrt(1 - alpha_bar_t) * epsilon
    """

    if noise is None:

        noise = torch.randn_like(
            clean_windows
        )

    sqrt_alpha_bar_t = extract(
        sqrt_alphas_cumprod,
        timesteps,
        clean_windows.shape
    )

    sqrt_one_minus_alpha_bar_t = extract(
        sqrt_one_minus_alphas_cumprod,
        timesteps,
        clean_windows.shape
    )

    noisy_windows = (
        sqrt_alpha_bar_t
        * clean_windows
        +
        sqrt_one_minus_alpha_bar_t
        * noise
    )

    return (
        noisy_windows,
        noise
    )


# ----------------------------------------------------------
# Verify using one batch
# ----------------------------------------------------------

verification_batch = next(
    iter(train_loader)
)

clean_windows = (
    verification_batch[0]
    .to(device)
)

batch_size = clean_windows.shape[0]


# Random timestep for every sample
sample_timesteps = torch.randint(
    low=0,
    high=DIFFUSION_STEPS,
    size=(batch_size,),
    device=device,
    dtype=torch.long
)


noisy_windows, sampled_noise = (
    forward_diffusion_sample(
        clean_windows,
        sample_timesteps
    )
)


# ----------------------------------------------------------
# Shape validation
# ----------------------------------------------------------

if noisy_windows.shape != clean_windows.shape:

    raise RuntimeError(
        "Forward diffusion changed the IMU shape.\n"
        f"Clean: {tuple(clean_windows.shape)}\n"
        f"Noisy: {tuple(noisy_windows.shape)}"
    )


if sampled_noise.shape != clean_windows.shape:

    raise RuntimeError(
        "Noise tensor has an unexpected shape."
    )


# ----------------------------------------------------------
# Output
# ----------------------------------------------------------

print("=" * 75)
print("FORWARD DIFFUSION CHECK")
print("=" * 75)

print(
    "Number of diffusion steps :",
    DIFFUSION_STEPS
)

print(
    "Beta schedule shape       :",
    tuple(betas.shape)
)

print(
    "Clean batch shape         :",
    tuple(clean_windows.shape)
)

print(
    "Noisy batch shape         :",
    tuple(noisy_windows.shape)
)

print(
    "Noise tensor shape        :",
    tuple(sampled_noise.shape)
)

print(
    "Sampled timesteps         :",
    sample_timesteps
    .detach()
    .cpu()
    .tolist()
)

print(
    "First beta                :",
    betas[0].item()
)

print(
    "Last beta                 :",
    betas[-1].item()
)

print(
    "\nForward diffusion configured successfully."
)


# In[28]:


# ==========================================================
# Cell 7 - Initialize Conditional U-Net Denoiser
# ==========================================================

import torch
import torch.nn as nn


# ----------------------------------------------------------
# U-Net configuration
# ----------------------------------------------------------

# Keep local CPU testing small.
# Use the larger configuration on Narval.

if RUN_MODE == "small_cpu":

    UNET_BASE_CHANNELS = 32

    UNET_CHANNEL_MULTIPLIERS = [
        1,
        2,
        4
    ]
    UNET_RESIDUAL_BLOCKS = 1
else:

    UNET_BASE_CHANNELS = 64

    UNET_CHANNEL_MULTIPLIERS = [
        1,
        2,
        4,
        8
    ]
    UNET_RESIDUAL_BLOCKS = 2


CONDITION_DIM = Z_DIM


# ----------------------------------------------------------
# IMU padding helpers
# ----------------------------------------------------------

def pad_imu_for_unet(x):
    """
    Pads the feature dimension from 30 to 32.

    Input:
        (B, 1, 128, 30)

    Output:
        (B, 1, 128, 32)
    """

    if x.shape[-1] != NUM_FEATURES:

        raise ValueError(
            "Unexpected IMU feature dimension: "
            f"{x.shape[-1]}"
        )

    padding_required = (
        32 - x.shape[-1]
    )

    if padding_required < 0:

        raise ValueError(
            "IMU feature dimension is larger "
            "than the expected padded size."
        )

    if padding_required == 0:

        return x

    return F.pad(
        x,
        (
            0,
            padding_required,
            0,
            0
        )
    )


def crop_imu_from_unet(x):
    """
    Crops U-Net output back from 32 features
    to the original 30 IMU features.
    """

    return x[
        ...,
        :NUM_FEATURES
    ]


# ----------------------------------------------------------
# Initialize activity-conditioning network
# ----------------------------------------------------------

condition_embedding = ConditionalEmbedding(
    num_labels=CONDITION_DIM,
    d_model=CONDITION_DIM,
    dim=CONDITION_DIM
).to(device)


# ----------------------------------------------------------
# Initialize conditional U-Net
# ----------------------------------------------------------

denoiser_model = Unet(
    in_ch=1,
    mod_ch=UNET_BASE_CHANNELS,
    out_ch=1,
    ch_mul=UNET_CHANNEL_MULTIPLIERS,
    num_res_blocks=UNET_RESIDUAL_BLOCKS,
    cdim=CONDITION_DIM,
    use_conv=True,
    droprate=0.1,
    dtype=torch.float32
).to(device)

# ----------------------------------------------------------
# Initialize from previously trained baseline diffusion
# ----------------------------------------------------------

if RUN_MODE == "small_cpu":

    BASELINE_DIFFUSION_CHECKPOINT = (
        WORK_DIR / "small_conditional_denoiser.pt"
    )

else:

    BASELINE_DIFFUSION_CHECKPOINT = (
        WORK_DIR / "full_conditional_denoiser.pt"
    )


if not BASELINE_DIFFUSION_CHECKPOINT.exists():

    raise FileNotFoundError(
        "Baseline diffusion checkpoint was not found:\n"
        f"{BASELINE_DIFFUSION_CHECKPOINT}"
    )


baseline_checkpoint = torch.load(
    BASELINE_DIFFUSION_CHECKPOINT,
    map_location=device,
    weights_only=False
)


denoiser_model.load_state_dict(
    baseline_checkpoint[
        "denoiser_state_dict"
    ]
)


condition_embedding.load_state_dict(
    baseline_checkpoint[
        "condition_embedding_state_dict"
    ]
)


print(
    "Baseline diffusion loaded from:",
    BASELINE_DIFFUSION_CHECKPOINT.name
)

print(
    "Baseline training epochs:",
    baseline_checkpoint.get(
        "epochs",
        "unknown"
    )
)

print(
    "Baseline diffusion steps:",
    baseline_checkpoint.get(
        "diffusion_steps",
        "unknown"
    )
)
# ----------------------------------------------------------
# Get one verification batch
# ----------------------------------------------------------

verification_batch = next(
    iter(train_loader)
)

verification_windows = (
    verification_batch[0]
    .to(device)
)


batch_size = (
    verification_windows.shape[0]
)


# ----------------------------------------------------------
# Pad IMU input
# ----------------------------------------------------------

verification_padded = pad_imu_for_unet(
    verification_windows
)


# ----------------------------------------------------------
# Obtain frozen public/activity representation
# ----------------------------------------------------------

with torch.no_grad():

    _, verification_z_public = (
        surrogate_model(
            verification_windows
        )
    )


# ----------------------------------------------------------
# Convert public representation into U-Net condition
# ----------------------------------------------------------

verification_condition = (
    condition_embedding(
        verification_z_public
    )
)


# ----------------------------------------------------------
# Random diffusion timestep
# ----------------------------------------------------------

verification_timesteps = torch.randint(
    low=5,
    high=41,
    size=(batch_size,),
    device=device,
    dtype=torch.long
)


# ----------------------------------------------------------
# Add noise to padded IMU
# ----------------------------------------------------------

verification_noise = torch.randn_like(
    verification_padded
)


sqrt_alpha_bar_t = extract(
    sqrt_alphas_cumprod,
    verification_timesteps,
    verification_padded.shape
)


sqrt_one_minus_alpha_bar_t = extract(
    sqrt_one_minus_alphas_cumprod,
    verification_timesteps,
    verification_padded.shape
)


verification_noisy = (
    sqrt_alpha_bar_t
    * verification_padded
    +
    sqrt_one_minus_alpha_bar_t
    * verification_noise
)


# ----------------------------------------------------------
# U-Net forward pass
# ----------------------------------------------------------

with torch.no_grad():

    verification_prediction = (
        denoiser_model(
            verification_noisy,
            verification_timesteps,
            verification_condition
        )
    )


# ----------------------------------------------------------
# Crop prediction back to 30 features
# ----------------------------------------------------------

verification_prediction_cropped = (
    crop_imu_from_unet(
        verification_prediction
    )
)


# ----------------------------------------------------------
# Validate shapes
# ----------------------------------------------------------

if verification_padded.shape[-1] != 32:

    raise RuntimeError(
        "IMU padding failed."
    )


if verification_prediction.shape != (
    verification_padded.shape
):

    raise RuntimeError(
        "U-Net output shape does not match "
        "the padded input shape.\n"
        f"Input:  {tuple(verification_padded.shape)}\n"
        f"Output: {tuple(verification_prediction.shape)}"
    )


if verification_prediction_cropped.shape != (
    verification_windows.shape
):

    raise RuntimeError(
        "Cropped U-Net prediction does not match "
        "the original IMU shape."
    )


# ----------------------------------------------------------
# Parameter counts
# ----------------------------------------------------------

denoiser_parameters = sum(
    parameter.numel()
    for parameter
    in denoiser_model.parameters()
)

condition_parameters = sum(
    parameter.numel()
    for parameter
    in condition_embedding.parameters()
)


# ----------------------------------------------------------
# Final output
# ----------------------------------------------------------

print("=" * 75)
print("PRIVACY-GUIDED CONDITIONAL U-NET CHECK")
print("=" * 75)

print(
    "Run mode                   :",
    RUN_MODE
)

print(
    "U-Net base channels        :",
    UNET_BASE_CHANNELS
)

print(
    "Channel multipliers        :",
    UNET_CHANNEL_MULTIPLIERS
)

print(
    "Original input shape       :",
    tuple(
        verification_windows.shape
    )
)

print(
    "Padded input shape         :",
    tuple(
        verification_padded.shape
    )
)

print(
    "Public embedding shape     :",
    tuple(
        verification_z_public.shape
    )
)

print(
    "Condition embedding shape  :",
    tuple(
        verification_condition.shape
    )
)

print(
    "Raw U-Net output shape     :",
    tuple(
        verification_prediction.shape
    )
)

print(
    "Cropped prediction shape   :",
    tuple(
        verification_prediction_cropped.shape
    )
)

print(
    "Denoiser parameters        :",
    f"{denoiser_parameters:,}"
)

print(
    "Condition parameters       :",
    f"{condition_parameters:,}"
)

print(
    "\nConditional U-Net initialized "
    "successfully."
)


# In[45]:


# ==========================================================
# Cell 8 - Privacy-Guided Loss Validation
# ==========================================================

import torch
import torch.nn.functional as F


# ----------------------------------------------------------
# Loss weights
# ----------------------------------------------------------

LAMBDA_UTILITY = 1.0
LAMBDA_PRIVACY = 1.0


# ----------------------------------------------------------
# Get one training batch
# ----------------------------------------------------------

batch = next(
    iter(train_loader)
)

clean_windows = batch[0].to(
    device
)

activity_labels = batch[1].to(
    device
)

gender_labels = batch[2].to(
    device
)

batch_size = clean_windows.shape[0]


# ----------------------------------------------------------
# Pad clean IMU windows
# ----------------------------------------------------------

clean_padded = pad_imu_for_unet(
    clean_windows
)


# ----------------------------------------------------------
# Sample diffusion timesteps and Gaussian noise
# ----------------------------------------------------------

timesteps = torch.randint(
    low=0,
    high=DIFFUSION_STEPS,
    size=(batch_size,),
    device=device,
    dtype=torch.long
)

true_noise = torch.randn_like(
    clean_padded
)


# ----------------------------------------------------------
# Forward diffusion
# ----------------------------------------------------------

sqrt_alpha_bar_t = extract(
    sqrt_alphas_cumprod,
    timesteps,
    clean_padded.shape
)

sqrt_one_minus_alpha_bar_t = extract(
    sqrt_one_minus_alphas_cumprod,
    timesteps,
    clean_padded.shape
)

noisy_padded = (
    sqrt_alpha_bar_t * clean_padded
    +
    sqrt_one_minus_alpha_bar_t * true_noise
)


# ----------------------------------------------------------
# Activity condition from original clean signal
# ----------------------------------------------------------
#
# The surrogate is frozen, so we do not need gradients
# while extracting the conditioning representation.
# ----------------------------------------------------------

with torch.no_grad():

    _, z_public = surrogate_model(
        clean_windows
    )

activity_condition = condition_embedding(
    z_public
)


# ----------------------------------------------------------
# Predict diffusion noise
# ----------------------------------------------------------

predicted_noise = denoiser_model(
    noisy_padded,
    timesteps,
    activity_condition
)


# ----------------------------------------------------------
# 1. Standard diffusion loss
# ----------------------------------------------------------

diffusion_loss = F.mse_loss(
    predicted_noise,
    true_noise
)


# ----------------------------------------------------------
# Estimate reconstructed clean signal x_0
# ----------------------------------------------------------

predicted_clean_padded = (
    noisy_padded
    -
    sqrt_one_minus_alpha_bar_t
    * predicted_noise
) / torch.clamp(
    sqrt_alpha_bar_t,
    min=1e-8
)

predicted_clean = crop_imu_from_unet(
    predicted_clean_padded
)


# ----------------------------------------------------------
# 2. Utility loss
#
# IMPORTANT:
# Do NOT use torch.no_grad() here.
#
# The surrogate parameters are frozen, but gradients must
# travel through the classifier to predicted_clean and
# therefore back into the denoiser.
# ----------------------------------------------------------

activity_logits_generated, _ = surrogate_model(
    predicted_clean
)

utility_loss = F.cross_entropy(
    activity_logits_generated,
    activity_labels
)


# ----------------------------------------------------------
# 3. Privacy loss
#
# Goal:
# Push gender predictions toward maximum uncertainty.
#
# For two genders, the target distribution is:
#
# [0.5, 0.5]
#
# rather than teaching the model the opposite gender.
# ----------------------------------------------------------

gender_logits_generated, _ = privacy_model(
    predicted_clean
)

gender_log_probabilities = F.log_softmax(
    gender_logits_generated,
    dim=1
)

uniform_gender_target = torch.full_like(
    gender_log_probabilities,
    fill_value=(
        1.0 / NUM_GENDERS
    )
)

privacy_loss = F.kl_div(
    gender_log_probabilities,
    uniform_gender_target,
    reduction="batchmean"
)


# ==========================================================
# NEW - Individual Loss Gradient Contribution Diagnostic
# ==========================================================
#
# This section measures how strongly each loss individually
# pushes the denoiser.
#
# It does NOT update model parameters.
# ==========================================================

def calculate_gradient_norm(
    loss,
    model,
    retain_graph=True
):

    model.zero_grad(
        set_to_none=True
    )

    condition_embedding.zero_grad(
        set_to_none=True
    )

    loss.backward(
        retain_graph=retain_graph
    )

    squared_gradient_norm = 0.0

    for parameter in model.parameters():

        if parameter.grad is not None:

            parameter_norm = (
                parameter.grad
                .detach()
                .norm(2)
                .item()
            )

            squared_gradient_norm += (
                parameter_norm ** 2
            )

    return (
        squared_gradient_norm ** 0.5
    )


# ----------------------------------------------------------
# Gradient produced by diffusion loss
# ----------------------------------------------------------

diffusion_gradient_norm = (
    calculate_gradient_norm(
        diffusion_loss,
        denoiser_model,
        retain_graph=True
    )
)


# ----------------------------------------------------------
# Gradient produced by utility loss
# ----------------------------------------------------------

utility_gradient_norm = (
    calculate_gradient_norm(
        LAMBDA_UTILITY * utility_loss,
        denoiser_model,
        retain_graph=True
    )
)


# ----------------------------------------------------------
# Gradient produced by privacy loss
# ----------------------------------------------------------

privacy_gradient_norm = (
    calculate_gradient_norm(
        LAMBDA_PRIVACY * privacy_loss,
        denoiser_model,
        retain_graph=True
    )
)


# ----------------------------------------------------------
# Clear diagnostic gradients
# ----------------------------------------------------------

denoiser_model.zero_grad(
    set_to_none=True
)

condition_embedding.zero_grad(
    set_to_none=True
)


# ----------------------------------------------------------
# Gradient diagnostic output
# ----------------------------------------------------------

print("\n" + "=" * 75)
print("INDIVIDUAL LOSS GRADIENT CONTRIBUTION")
print("=" * 75)

print(
    "Diffusion gradient norm :",
    f"{diffusion_gradient_norm:.6f}"
)

print(
    "Utility gradient norm   :",
    f"{utility_gradient_norm:.6f}"
)

print(
    "Privacy gradient norm   :",
    f"{privacy_gradient_norm:.6f}"
)

if privacy_gradient_norm > 0:

    print(
        "Utility / Privacy ratio:",
        f"{utility_gradient_norm / privacy_gradient_norm:.2f}"
    )

    print(
        "Diffusion / Privacy ratio:",
        f"{diffusion_gradient_norm / privacy_gradient_norm:.2f}"
    )

else:

    print(
        "WARNING: Privacy gradient is zero."
    )

print("=" * 75)


# ----------------------------------------------------------
# Combined privacy-guided objective
# ----------------------------------------------------------

total_loss = (
    diffusion_loss
    +
    LAMBDA_UTILITY * utility_loss
    +
    LAMBDA_PRIVACY * privacy_loss
)


# ----------------------------------------------------------
# Check whether total loss is connected to denoiser
# ----------------------------------------------------------

if not total_loss.requires_grad:

    raise RuntimeError(
        "Total loss is detached from the "
        "training computation graph."
    )


# ----------------------------------------------------------
# Gradient-flow validation
# ----------------------------------------------------------

denoiser_model.zero_grad(
    set_to_none=True
)

condition_embedding.zero_grad(
    set_to_none=True
)

total_loss.backward()

denoiser_gradient_norm = 0.0

for parameter in denoiser_model.parameters():

    if parameter.grad is not None:

        denoiser_gradient_norm += (
            parameter.grad
            .detach()
            .norm()
            .item()
        )


# ----------------------------------------------------------
# Confirm frozen classifiers did not receive gradients
# ----------------------------------------------------------

surrogate_has_parameter_gradients = any(
    parameter.grad is not None
    for parameter
    in surrogate_model.parameters()
)

privacy_has_parameter_gradients = any(
    parameter.grad is not None
    for parameter
    in privacy_model.parameters()
)


# ----------------------------------------------------------
# Gender probabilities for inspection
# ----------------------------------------------------------

with torch.no_grad():

    gender_probabilities = F.softmax(
        gender_logits_generated,
        dim=1
    )

    mean_gender_probabilities = (
        gender_probabilities.mean(
            dim=0
        )
    )


# ----------------------------------------------------------
# Clear validation gradients
# ----------------------------------------------------------

denoiser_model.zero_grad(
    set_to_none=True
)

condition_embedding.zero_grad(
    set_to_none=True
)


# ----------------------------------------------------------
# Output
# ----------------------------------------------------------

print("\n" + "=" * 75)
print("PRIVACY-GUIDED LOSS CHECK")
print("=" * 75)

print(
    "Diffusion loss      :",
    f"{diffusion_loss.item():.6f}"
)

print(
    "Utility loss        :",
    f"{utility_loss.item():.6f}"
)

print(
    "Privacy loss        :",
    f"{privacy_loss.item():.6f}"
)

print(
    "Total loss          :",
    f"{total_loss.item():.6f}"
)

print(
    "\nLambda utility      :",
    LAMBDA_UTILITY
)

print(
    "Lambda privacy      :",
    LAMBDA_PRIVACY
)

print(
    "\nPredicted clean shape:",
    tuple(
        predicted_clean.shape
    )
)

print(
    "Mean gender probabilities:",
    mean_gender_probabilities
    .detach()
    .cpu()
    .tolist()
)

print(
    "\nTotal loss requires gradient :",
    total_loss.requires_grad
)

print(
    "Denoiser gradient norm       :",
    denoiser_gradient_norm
)

print(
    "Surrogate parameter gradients:",
    surrogate_has_parameter_gradients
)

print(
    "Privacy parameter gradients  :",
    privacy_has_parameter_gradients
)


# ----------------------------------------------------------
# Final validations
# ----------------------------------------------------------

if denoiser_gradient_norm <= 0:

    raise RuntimeError(
        "No gradients reached the denoiser."
    )

if privacy_gradient_norm <= 0:

    raise RuntimeError(
        "Privacy loss produced no gradient "
        "for the denoiser."
    )

if surrogate_has_parameter_gradients:

    raise RuntimeError(
        "Frozen surrogate classifier unexpectedly "
        "received parameter gradients."
    )

if privacy_has_parameter_gradients:

    raise RuntimeError(
        "Frozen privacy classifier unexpectedly "
        "received parameter gradients."
    )

print(
    "\nPrivacy-guided loss is connected "
    "correctly to the denoiser."
)


# In[49]:


# ==========================================================
# Cell 8B - Initialize Adaptive Gender Adversary
# DySan-inspired hybrid privacy experiment
# ==========================================================

import copy
import torch
import torch.nn.functional as F


# ----------------------------------------------------------
# Create a separate gender adversary
# ----------------------------------------------------------
#
# privacy_model:
#     remains the FROZEN reference privacy classifier
#     used for fair final evaluation.
#
# adversary_model:
#     separate TRAINABLE copy used only during
#     adversarial privacy-guided training.
# ----------------------------------------------------------

adversary_model = copy.deepcopy(
    privacy_model
).to(device)


# ----------------------------------------------------------
# Initial adversary state
# ----------------------------------------------------------
#
# Start from the already-trained gender classifier.
# This gives the adversary an initially meaningful
# gender decision boundary instead of random weights.
# ----------------------------------------------------------

adversary_model.train()

for parameter in adversary_model.parameters():

    parameter.requires_grad = True


# ----------------------------------------------------------
# Keep original privacy classifier frozen
# ----------------------------------------------------------

privacy_model.eval()

for parameter in privacy_model.parameters():

    parameter.requires_grad = False


# ----------------------------------------------------------
# Adversary optimizer
# ----------------------------------------------------------

ADVERSARY_LR = 1e-4

adversary_optimizer = torch.optim.Adam(
    adversary_model.parameters(),
    lr=ADVERSARY_LR
)


# ----------------------------------------------------------
# Verification
# ----------------------------------------------------------

adversary_parameters = sum(
    parameter.numel()
    for parameter
    in adversary_model.parameters()
)


trainable_adversary_parameters = sum(
    parameter.numel()
    for parameter
    in adversary_model.parameters()
    if parameter.requires_grad
)


print("=" * 80)
print("ADAPTIVE GENDER ADVERSARY INITIALIZED")
print("=" * 80)

print(
    "Adversary parameters           :",
    f"{adversary_parameters:,}"
)

print(
    "Trainable adversary parameters :",
    f"{trainable_adversary_parameters:,}"
)

print(
    "Adversary learning rate        :",
    ADVERSARY_LR
)

print(
    "Reference privacy classifier   : frozen"
)

print(
    "Adaptive gender adversary      : trainable"
)


# In[51]:


# ==========================================================
# Cell 9 - Hybrid Adversarial Privacy-Guided Diffusion
# Restricted Timesteps + Adaptive Gender Adversary
# ==========================================================

import os
import time
import torch
import torch.nn.functional as F


# ----------------------------------------------------------
# Training configuration
# ----------------------------------------------------------

if RUN_MODE == "small_cpu":

    PRIVACY_GUIDED_EPOCHS = 3
    MAX_BATCHES_PER_EPOCH = 100
    PRIVACY_GUIDED_LR = 1e-5

else:

    PRIVACY_GUIDED_EPOCHS = 5
    MAX_BATCHES_PER_EPOCH = 500
    PRIVACY_GUIDED_LR = 1e-5


# ----------------------------------------------------------
# Restricted training timestep range
# ----------------------------------------------------------

TRAIN_TIMESTEP_MIN = 5
TRAIN_TIMESTEP_MAX = 40


# ----------------------------------------------------------
# Loss weights
# ----------------------------------------------------------

LAMBDA_UTILITY = float(
    os.environ.get(
        "LAMBDA_UTILITY",
        "1.0"
    )
)

LAMBDA_PRIVACY = float(
    os.environ.get(
        "LAMBDA_PRIVACY",
        "100.0"
    )
)


# ----------------------------------------------------------
# Hybrid-specific checkpoint
# ----------------------------------------------------------

if RUN_MODE == "small_cpu":

    PRIVACY_GUIDED_CHECKPOINT = (
        WORK_DIR
        /
        f"small_hybrid_adv_t5_40_lu_"
        f"{LAMBDA_UTILITY:g}_lp_"
        f"{LAMBDA_PRIVACY:g}.pt"
    )

else:

    PRIVACY_GUIDED_CHECKPOINT = (
        WORK_DIR
        /
        f"full_hybrid_adv_t5_40_lu_"
        f"{LAMBDA_UTILITY:g}_lp_"
        f"{LAMBDA_PRIVACY:g}.pt"
    )


# ----------------------------------------------------------
# Configuration output
# ----------------------------------------------------------

print("=" * 80)
print("HYBRID ADVERSARIAL PRIVACY-GUIDED DIFFUSION")
print("=" * 80)

print(
    "Run mode                :",
    RUN_MODE
)

print(
    "Epochs                  :",
    PRIVACY_GUIDED_EPOCHS
)

print(
    "Maximum batches/epoch   :",
    MAX_BATCHES_PER_EPOCH
)

print(
    "Diffusion learning rate :",
    PRIVACY_GUIDED_LR
)

print(
    "Adversary learning rate :",
    ADVERSARY_LR
)

print(
    "Diffusion steps         :",
    DIFFUSION_STEPS
)

print(
    "Training timestep range :",
    f"{TRAIN_TIMESTEP_MIN}-{TRAIN_TIMESTEP_MAX}"
)

print(
    "Lambda utility          :",
    LAMBDA_UTILITY
)

print(
    "Lambda privacy          :",
    LAMBDA_PRIVACY
)

print(
    "Device                  :",
    device
)


# ----------------------------------------------------------
# Model states
# ----------------------------------------------------------
#
# Trainable:
#
#   denoiser_model
#   condition_embedding
#   adversary_model
#
# Frozen:
#
#   surrogate_model
#   privacy_model
#
# privacy_model remains untouched so that it can be used
# as the fixed reference gender classifier during evaluation.
# ----------------------------------------------------------

denoiser_model.train()
condition_embedding.train()

surrogate_model.eval()
privacy_model.eval()

adversary_model.train()


for parameter in surrogate_model.parameters():

    parameter.requires_grad = False


for parameter in privacy_model.parameters():

    parameter.requires_grad = False


for parameter in adversary_model.parameters():

    parameter.requires_grad = True


# ----------------------------------------------------------
# Diffusion optimizer
# ----------------------------------------------------------

optimizer = torch.optim.Adam(
    list(
        denoiser_model.parameters()
    )
    +
    list(
        condition_embedding.parameters()
    ),
    lr=PRIVACY_GUIDED_LR
)


# ----------------------------------------------------------
# Training history
# ----------------------------------------------------------

training_history = {

    "total_loss": [],

    "diffusion_loss": [],

    "utility_loss": [],

    "privacy_loss": [],

    "adversary_loss": []

}


# ----------------------------------------------------------
# Start training
# ----------------------------------------------------------

training_start_time = time.time()


for epoch in range(
    PRIVACY_GUIDED_EPOCHS
):

    epoch_total_loss = 0.0
    epoch_diffusion_loss = 0.0
    epoch_utility_loss = 0.0
    epoch_privacy_loss = 0.0
    epoch_adversary_loss = 0.0

    processed_batches = 0


    for batch_index, batch in enumerate(
        train_loader
    ):

        if (
            MAX_BATCHES_PER_EPOCH is not None
            and
            batch_index >= MAX_BATCHES_PER_EPOCH
        ):

            break


        # --------------------------------------------------
        # Load batch
        # --------------------------------------------------

        windows, activities, genders, _ = batch


        windows = windows.to(
            device,
            non_blocking=True
        )


        activities = activities.to(
            device,
            non_blocking=True
        )


        genders = genders.to(
            device,
            non_blocking=True
        )


        batch_size = windows.shape[0]


        # --------------------------------------------------
        # Pad IMU from 30 -> 32 features
        # --------------------------------------------------

        clean_padded = pad_imu_for_unet(
            windows
        )


        # --------------------------------------------------
        # Restricted random diffusion timestep
        #
        # high is exclusive, therefore 41 gives:
        #
        # 5, 6, ..., 40
        # --------------------------------------------------

        timesteps = torch.randint(
            low=TRAIN_TIMESTEP_MIN,
            high=TRAIN_TIMESTEP_MAX + 1,
            size=(batch_size,),
            device=device,
            dtype=torch.long
        )


        # --------------------------------------------------
        # Forward diffusion
        # --------------------------------------------------

        true_noise = torch.randn_like(
            clean_padded
        )


        sqrt_alpha_bar_t = extract(
            sqrt_alphas_cumprod,
            timesteps,
            clean_padded.shape
        )


        sqrt_one_minus_alpha_bar_t = extract(
            sqrt_one_minus_alphas_cumprod,
            timesteps,
            clean_padded.shape
        )


        noisy_padded = (
            sqrt_alpha_bar_t
            * clean_padded
            +
            sqrt_one_minus_alpha_bar_t
            * true_noise
        )


        # --------------------------------------------------
        # Obtain frozen public/activity representation
        # --------------------------------------------------

        with torch.no_grad():

            _, z_public = surrogate_model(
                windows
            )


        activity_condition = (
            condition_embedding(
                z_public
            )
        )


        # --------------------------------------------------
        # Predict diffusion noise
        # --------------------------------------------------

        predicted_noise = denoiser_model(
            noisy_padded,
            timesteps,
            activity_condition
        )


        # --------------------------------------------------
        # 1. Diffusion loss
        # --------------------------------------------------

        diffusion_loss = F.mse_loss(
            predicted_noise,
            true_noise
        )


        # --------------------------------------------------
        # Estimate reconstructed clean signal x_0
        # --------------------------------------------------

        predicted_clean_padded = (
            noisy_padded
            -
            sqrt_one_minus_alpha_bar_t
            * predicted_noise
        ) / torch.clamp(
            sqrt_alpha_bar_t,
            min=1e-6
        )


        # --------------------------------------------------
        # Crop back from 32 -> 30 features
        # --------------------------------------------------

        predicted_clean = (
            predicted_clean_padded[
                :,
                :,
                :,
                :NUM_FEATURES
            ]
        )


        # ==================================================
        # STAGE A
        # UPDATE THE ADAPTIVE GENDER ADVERSARY
        # ==================================================
        #
        # The adversary attempts to correctly predict
        # gender from the CURRENT obfuscated signal.
        #
        # predicted_clean.detach() is essential:
        #
        # gradients update the adversary here,
        # NOT the diffusion model.
        # ==================================================

        for parameter in adversary_model.parameters():

            parameter.requires_grad = True


        adversary_model.train()


        adversary_logits, _ = adversary_model(
            predicted_clean.detach()
        )


        adversary_loss = F.cross_entropy(
            adversary_logits,
            genders
        )


        adversary_optimizer.zero_grad(
            set_to_none=True
        )


        adversary_loss.backward()


        torch.nn.utils.clip_grad_norm_(
            adversary_model.parameters(),
            max_norm=1.0
        )


        adversary_optimizer.step()


        # ==================================================
        # STAGE B
        # UPDATE THE DIFFUSION SANITIZER
        # ==================================================
        #
        # Freeze adversary parameters.
        #
        # We DO NOT use torch.no_grad() for its forward pass.
        #
        # This allows the privacy gradient to flow:
        #
        # privacy loss
        #       ↓
        # adversary computation
        #       ↓
        # predicted_clean
        #       ↓
        # denoiser_model
        #
        # while preventing the adversary weights themselves
        # from being modified during Stage B.
        # ==================================================

        for parameter in adversary_model.parameters():

            parameter.requires_grad = False


        adversary_model.eval()


        # --------------------------------------------------
        # 2. Utility loss
        # --------------------------------------------------
        #
        # Frozen activity classifier evaluates whether the
        # obfuscated signal still contains enough activity
        # information.
        # --------------------------------------------------

        activity_logits, _ = surrogate_model(
            predicted_clean
        )


        utility_loss = F.cross_entropy(
            activity_logits,
            activities
        )


        # --------------------------------------------------
        # 3. Adaptive adversarial privacy loss
        # --------------------------------------------------
        #
        # Push the adaptive gender adversary toward:
        #
        # [0.5, 0.5]
        #
        # for the two gender classes.
        # --------------------------------------------------

        gender_logits, _ = adversary_model(
            predicted_clean
        )


        gender_log_probabilities = (
            F.log_softmax(
                gender_logits,
                dim=1
            )
        )


        uniform_gender_target = (
            torch.full_like(
                gender_log_probabilities,
                fill_value=(
                    1.0 / NUM_GENDERS
                )
            )
        )


        privacy_loss = F.kl_div(
            gender_log_probabilities,
            uniform_gender_target,
            reduction="batchmean"
        )


        # --------------------------------------------------
        # Combined hybrid objective
        # --------------------------------------------------

        total_loss = (
            diffusion_loss
            +
            LAMBDA_UTILITY
            * utility_loss
            +
            LAMBDA_PRIVACY
            * privacy_loss
        )


        # --------------------------------------------------
        # Safety check
        # --------------------------------------------------

        if not total_loss.requires_grad:

            raise RuntimeError(
                "Hybrid total loss is detached from "
                "the training computation graph."
            )


        # --------------------------------------------------
        # Update diffusion model
        # --------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )


        total_loss.backward()


        torch.nn.utils.clip_grad_norm_(
            list(
                denoiser_model.parameters()
            )
            +
            list(
                condition_embedding.parameters()
            ),
            max_norm=1.0
        )


        optimizer.step()


        # --------------------------------------------------
        # Re-enable adversary for next batch
        # --------------------------------------------------

        for parameter in adversary_model.parameters():

            parameter.requires_grad = True


        adversary_model.train()


        # --------------------------------------------------
        # Accumulate statistics
        # --------------------------------------------------

        epoch_total_loss += (
            total_loss.item()
        )


        epoch_diffusion_loss += (
            diffusion_loss.item()
        )


        epoch_utility_loss += (
            utility_loss.item()
        )


        epoch_privacy_loss += (
            privacy_loss.item()
        )


        epoch_adversary_loss += (
            adversary_loss.item()
        )


        processed_batches += 1


    # ------------------------------------------------------
    # Epoch averages
    # ------------------------------------------------------

    denominator = max(
        processed_batches,
        1
    )


    average_total_loss = (
        epoch_total_loss
        / denominator
    )


    average_diffusion_loss = (
        epoch_diffusion_loss
        / denominator
    )


    average_utility_loss = (
        epoch_utility_loss
        / denominator
    )


    average_privacy_loss = (
        epoch_privacy_loss
        / denominator
    )


    average_adversary_loss = (
        epoch_adversary_loss
        / denominator
    )


    # ------------------------------------------------------
    # Save history
    # ------------------------------------------------------

    training_history[
        "total_loss"
    ].append(
        average_total_loss
    )


    training_history[
        "diffusion_loss"
    ].append(
        average_diffusion_loss
    )


    training_history[
        "utility_loss"
    ].append(
        average_utility_loss
    )


    training_history[
        "privacy_loss"
    ].append(
        average_privacy_loss
    )


    training_history[
        "adversary_loss"
    ].append(
        average_adversary_loss
    )


    elapsed = (
        time.time()
        -
        training_start_time
    )


    # ------------------------------------------------------
    # Epoch output
    # ------------------------------------------------------

    print(
        f"\nEpoch "
        f"{epoch + 1}/"
        f"{PRIVACY_GUIDED_EPOCHS}"
    )


    print(
        f"  Batches         : "
        f"{processed_batches}"
    )


    print(
        f"  Diffusion loss  : "
        f"{average_diffusion_loss:.6f}"
    )


    print(
        f"  Utility loss    : "
        f"{average_utility_loss:.6f}"
    )


    print(
        f"  Privacy loss    : "
        f"{average_privacy_loss:.6f}"
    )


    print(
        f"  Adversary loss  : "
        f"{average_adversary_loss:.6f}"
    )


    print(
        f"  Total loss      : "
        f"{average_total_loss:.6f}"
    )


    print(
        f"  Elapsed         : "
        f"{elapsed:.1f} sec"
    )


# ----------------------------------------------------------
# Save hybrid checkpoint
# ----------------------------------------------------------

hybrid_checkpoint = {

    "denoiser_state_dict":
        denoiser_model.state_dict(),

    "condition_embedding_state_dict":
        condition_embedding.state_dict(),

    "adversary_state_dict":
        adversary_model.state_dict(),

    "optimizer_state_dict":
        optimizer.state_dict(),

    "adversary_optimizer_state_dict":
        adversary_optimizer.state_dict(),

    "training_history":
        training_history,

    "run_mode":
        RUN_MODE,

    "epochs":
        PRIVACY_GUIDED_EPOCHS,

    "diffusion_steps":
        DIFFUSION_STEPS,

    "training_timestep_min":
        TRAIN_TIMESTEP_MIN,

    "training_timestep_max":
        TRAIN_TIMESTEP_MAX,

    "lambda_utility":
        LAMBDA_UTILITY,

    "lambda_privacy":
        LAMBDA_PRIVACY,

    "learning_rate":
        PRIVACY_GUIDED_LR,

    "adversary_learning_rate":
        ADVERSARY_LR,

    "hybrid_adversarial":
        True,

    "unet_base_channels":
        UNET_BASE_CHANNELS,

    "unet_channel_multipliers":
        UNET_CHANNEL_MULTIPLIERS,

    "condition_dim":
        CONDITION_DIM,

  
}


torch.save(
    hybrid_checkpoint,
    PRIVACY_GUIDED_CHECKPOINT
)


# ----------------------------------------------------------
# Final model states
# ----------------------------------------------------------

denoiser_model.eval()

condition_embedding.eval()

adversary_model.eval()


for parameter in adversary_model.parameters():

    parameter.requires_grad = False


total_training_time = (
    time.time()
    -
    training_start_time
)


# ----------------------------------------------------------
# Final output
# ----------------------------------------------------------

print("\n" + "=" * 80)

print(
    "HYBRID ADVERSARIAL TRAINING COMPLETE"
)

print("=" * 80)


print(
    "Total training time       :",
    f"{total_training_time:.1f} seconds"
)


print(
    "Training timestep range   :",
    f"{TRAIN_TIMESTEP_MIN}-{TRAIN_TIMESTEP_MAX}"
)


print(
    "Lambda utility            :",
    LAMBDA_UTILITY
)


print(
    "Lambda privacy            :",
    LAMBDA_PRIVACY
)


print(
    "Adversary learning rate   :",
    ADVERSARY_LR
)


print(
    "Checkpoint saved          :",
    PRIVACY_GUIDED_CHECKPOINT
)


print(
    "\nFinal losses:"
)


print(
    "Diffusion  :",
    f"{training_history['diffusion_loss'][-1]:.6f}"
)


print(
    "Utility    :",
    f"{training_history['utility_loss'][-1]:.6f}"
)


print(
    "Privacy    :",
    f"{training_history['privacy_loss'][-1]:.6f}"
)


print(
    "Adversary  :",
    f"{training_history['adversary_loss'][-1]:.6f}"
)


print(
    "Total      :",
    f"{training_history['total_loss'][-1]:.6f}"
)


print(
    "\nHybrid adversarial privacy-guided diffusion "
    "fine-tuning completed successfully."
)


# In[48]:


# ==========================================================
# Cell 10 - Load Privacy-Guided Model and Define
# Obfuscation / Reconstruction Function
# ==========================================================

import torch
import torch.nn.functional as F


# ----------------------------------------------------------
# Verify privacy-guided checkpoint exists
# ----------------------------------------------------------

if not PRIVACY_GUIDED_CHECKPOINT.exists():

    raise FileNotFoundError(
        "Privacy-guided diffusion checkpoint was not found:\n"
        f"{PRIVACY_GUIDIDED_CHECKPOINT}"
    )


# ----------------------------------------------------------
# Load privacy-guided checkpoint
# ----------------------------------------------------------

privacy_guided_checkpoint = torch.load(
    PRIVACY_GUIDED_CHECKPOINT,
    map_location=device,
    weights_only=False
)


print("=" * 80)
print("LOADING PRIVACY-GUIDED DIFFUSION MODEL")
print("=" * 80)


print(
    "Checkpoint:",
    PRIVACY_GUIDED_CHECKPOINT.name
)


print(
    "Checkpoint run mode:",
    privacy_guided_checkpoint.get(
        "run_mode",
        "unknown"
    )
)


print(
    "Checkpoint lambda privacy:",
    privacy_guided_checkpoint.get(
        "lambda_privacy",
        "unknown"
    )
)


# ----------------------------------------------------------
# Load trained weights
# ----------------------------------------------------------

denoiser_model.load_state_dict(
    privacy_guided_checkpoint[
        "denoiser_state_dict"
    ]
)


condition_embedding.load_state_dict(
    privacy_guided_checkpoint[
        "condition_embedding_state_dict"
    ]
)


denoiser_model.eval()

condition_embedding.eval()


print(
    "Privacy-guided checkpoint loaded successfully."
)


# ==========================================================
# Privacy-Guided Obfuscation Function
# ==========================================================

@torch.no_grad()
def privacy_guided_obfuscate(
    clean_windows,
    activity_labels,
    timestep=20
):
    """
    Obfuscate IMU windows using the trained
    privacy-guided conditional diffusion model.

    Process:

        clean IMU signal x_0
                ↓
        add Gaussian noise to timestep t
                ↓
        noisy signal x_t
                ↓
        privacy-guided U-Net predicts noise
                ↓
        estimate reconstructed clean signal
                ↓
        return obfuscated IMU signal

    Parameters
    ----------
    clean_windows:
        Tensor with shape:
        (batch, 1, 128, 30)

    activity_labels:
        Activity labels for the batch.

    timestep:
        Amount of forward diffusion noise.

    Returns
    -------
    reconstructed_windows:
        Privacy-guided reconstructed /
        obfuscated IMU windows.
    """

    clean_windows = clean_windows.to(
        device
    )

    activity_labels = activity_labels.to(
        device
    )

    batch_size = clean_windows.shape[0]


    # ------------------------------------------------------
    # Create timestep tensor
    # ------------------------------------------------------

    timesteps = torch.full(
        (batch_size,),
        int(timestep),
        device=device,
        dtype=torch.long
    )


    # ------------------------------------------------------
    # Add forward diffusion noise
    # ------------------------------------------------------

    noisy_windows, sampled_noise = (
        forward_diffusion_sample(
            clean_windows,
            timesteps
        )
    )


    # ------------------------------------------------------
    # Pad feature dimension
    #
    # Original:
    #     (B, 1, 128, 30)
    #
    # U-Net:
    #     (B, 1, 128, 32)
    # ------------------------------------------------------

    noisy_windows_padded = F.pad(
        noisy_windows,
        (
            0,
            2,
            0,
            0
        )
    )


    # ------------------------------------------------------
    # Obtain activity/public representation
    # ------------------------------------------------------

    _, z_public = surrogate_model(
        clean_windows
    )


    # ------------------------------------------------------
    # Build conditioning exactly as used during training
    # ------------------------------------------------------
    #
    # z_public is the 60-dimensional activity-preserving
    # representation produced by the frozen surrogate model.
    # ------------------------------------------------------

    model_condition = condition_embedding(
        z_public
    )


    # ------------------------------------------------------
    # Predict noise using privacy-guided U-Net
    # ------------------------------------------------------

    predicted_noise_padded = denoiser_model(
        noisy_windows_padded,
        timesteps,
        model_condition
    )


    # ------------------------------------------------------
    # Crop back to original 30 features
    # ------------------------------------------------------

    predicted_noise = (
        predicted_noise_padded[
            ...,
            :NUM_FEATURES
        ]
    )


    # ------------------------------------------------------
    # Reconstruct x_0
    #
    # From:
    #
    # x_t =
    # sqrt(alpha_bar_t) * x_0
    # +
    # sqrt(1-alpha_bar_t) * epsilon
    #
    # Therefore:
    #
    # x_0_hat =
    # (
    #   x_t
    #   -
    #   sqrt(1-alpha_bar_t) * epsilon_theta
    # )
    # /
    # sqrt(alpha_bar_t)
    # ------------------------------------------------------

    sqrt_alpha_bar_t = extract(
        sqrt_alphas_cumprod,
        timesteps,
        clean_windows.shape
    )


    sqrt_one_minus_alpha_bar_t = extract(
        sqrt_one_minus_alphas_cumprod,
        timesteps,
        clean_windows.shape
    )


    reconstructed_windows = (
        noisy_windows
        -
        (
            sqrt_one_minus_alpha_bar_t
            * predicted_noise
        )
    ) / torch.clamp(
        sqrt_alpha_bar_t,
        min=1e-8
    )


    return reconstructed_windows


# ==========================================================
# Quick Reconstruction Test
# ==========================================================

verification_batch = next(
    iter(test_loader)
)


verification_windows = (
    verification_batch[0]
    .to(device)
)


verification_activities = (
    verification_batch[1]
    .to(device)
)


# ----------------------------------------------------------
# Test at timestep 20
# ----------------------------------------------------------

TEST_TIMESTEP = 20


obfuscated_windows = (
    privacy_guided_obfuscate(
        verification_windows,
        verification_activities,
        timestep=TEST_TIMESTEP
    )
)


# ----------------------------------------------------------
# Validate shape
# ----------------------------------------------------------

if (
    obfuscated_windows.shape
    != verification_windows.shape
):

    raise RuntimeError(
        "Obfuscated signal shape does not match "
        "the original signal shape.\n"
        f"Original: {tuple(verification_windows.shape)}\n"
        f"Obfuscated: {tuple(obfuscated_windows.shape)}"
    )


# ----------------------------------------------------------
# Calculate basic signal distortion
# ----------------------------------------------------------

mean_absolute_change = (
    torch.mean(
        torch.abs(
            obfuscated_windows
            -
            verification_windows
        )
    )
    .item()
)


rmse = torch.sqrt(
    torch.mean(
        (
            obfuscated_windows
            -
            verification_windows
        ) ** 2
    )
).item()


# ----------------------------------------------------------
# Check activity and gender predictions
# ----------------------------------------------------------

activity_logits_before, _ = (
    surrogate_model(
        verification_windows
    )
)


activity_logits_after, _ = (
    surrogate_model(
        obfuscated_windows
    )
)


gender_logits_before, _ = (
    privacy_model(
        verification_windows
    )
)


gender_logits_after, _ = (
    privacy_model(
        obfuscated_windows
    )
)


activity_predictions_before = (
    activity_logits_before.argmax(
        dim=1
    )
)


activity_predictions_after = (
    activity_logits_after.argmax(
        dim=1
    )
)


gender_predictions_before = (
    gender_logits_before.argmax(
        dim=1
    )
)


gender_predictions_after = (
    gender_logits_after.argmax(
        dim=1
    )
)


# ==========================================================
# Output
# ==========================================================

print("\n" + "=" * 80)

print(
    "PRIVACY-GUIDED OBFUSCATION CHECK"
)

print("=" * 80)


print(
    "Test timestep            :",
    TEST_TIMESTEP
)


print(
    "Original shape           :",
    tuple(
        verification_windows.shape
    )
)


print(
    "Obfuscated shape         :",
    tuple(
        obfuscated_windows.shape
    )
)


print(
    "Mean absolute change     :",
    f"{mean_absolute_change:.6f}"
)


print(
    "RMSE                     :",
    f"{rmse:.6f}"
)


print(
    "\nTrue activities          :",
    verification_activities
    .detach()
    .cpu()
    .tolist()
)


print(
    "Activity before          :",
    activity_predictions_before
    .detach()
    .cpu()
    .tolist()
)


print(
    "Activity after           :",
    activity_predictions_after
    .detach()
    .cpu()
    .tolist()
)


print(
    "\nGender before            :",
    gender_predictions_before
    .detach()
    .cpu()
    .tolist()
)


print(
    "Gender after             :",
    gender_predictions_after
    .detach()
    .cpu()
    .tolist()
)


print(
    "\nPrivacy-guided signal "
    "reconstruction completed successfully."
)


# In[41]:


# ==========================================================
# Cell 11 - Balanced Privacy-Utility Evaluation
#           of Privacy-Guided Diffusion
# ==========================================================

import time
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    f1_score
)

# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------

EVALUATION_TIMESTEPS = [
    5,
    10,
    20,
    30,
    40
]

SUBJECTS_PER_GENDER = 4
WINDOWS_PER_ACTIVITY_PER_GENDER = 20

rng = np.random.default_rng(42)


# ==========================================================
# Build Balanced Evaluation Dataset
# ==========================================================

# Group test subject files according to gender.

subjects_by_gender = {
    0: [],
    1: []
}

for file_path in test_dataset.files:

    with np.load(
        file_path,
        allow_pickle=False
    ) as data:

        gender = int(
            np.asarray(
                data["gender"]
            ).reshape(-1)[0]
        )

    subjects_by_gender[
        gender
    ].append(
        file_path
    )


print("=" * 90)
print("BALANCED PRIVACY-GUIDED EVALUATION DATASET")
print("=" * 90)

print(
    "Available test subjects by gender:",
    {
        gender: len(files)
        for gender, files
        in subjects_by_gender.items()
    }
)


# ----------------------------------------------------------
# Select equal number of subjects from each gender
# ----------------------------------------------------------

selected_subjects = {}

for gender in range(NUM_GENDERS):

    available_files = (
        subjects_by_gender[
            gender
        ]
    )

    number_to_select = min(
        SUBJECTS_PER_GENDER,
        len(available_files)
    )

    selected_subjects[
        gender
    ] = available_files[
        :number_to_select
    ]


print(
    "Selected subjects per gender:",
    {
        gender: len(files)
        for gender, files
        in selected_subjects.items()
    }
)


# ----------------------------------------------------------
# Collect windows by gender and activity
# ----------------------------------------------------------

balanced_windows = []
balanced_activities = []
balanced_genders = []

for gender in range(NUM_GENDERS):

    gender_windows = []
    gender_activities = []

    for file_path in selected_subjects[
        gender
    ]:

        with np.load(
            file_path,
            allow_pickle=False
        ) as data:

            windows = np.asarray(
                data["windows"],
                dtype=np.float32
            )

            activities = np.asarray(
                data["activity"],
                dtype=np.int64
            )

        gender_windows.append(
            windows
        )

        gender_activities.append(
            activities
        )

    gender_windows = np.concatenate(
        gender_windows,
        axis=0
    )

    gender_activities = np.concatenate(
        gender_activities,
        axis=0
    )

    # ------------------------------------------------------
    # Select equal number from every activity
    # ------------------------------------------------------

    for activity_id in range(
        NUM_ACTIVITIES
    ):

        activity_indices = np.flatnonzero(
            gender_activities
            == activity_id
        )

        if len(activity_indices) == 0:
            raise RuntimeError(
                f"No samples found for gender "
                f"{gender}, activity {activity_id}."
            )

        number_to_select = min(
            WINDOWS_PER_ACTIVITY_PER_GENDER,
            len(activity_indices)
        )

        selected_indices = rng.choice(
            activity_indices,
            size=number_to_select,
            replace=False
        )

        selected_windows = (
            gender_windows[
                selected_indices
            ]
        )

        balanced_windows.append(
            selected_windows
        )

        balanced_activities.extend(
            [activity_id]
            * number_to_select
        )

        balanced_genders.extend(
            [gender]
            * number_to_select
        )


# ----------------------------------------------------------
# Combine balanced arrays
# ----------------------------------------------------------

balanced_windows = np.concatenate(
    balanced_windows,
    axis=0
)

balanced_activities = np.asarray(
    balanced_activities,
    dtype=np.int64
)

balanced_genders = np.asarray(
    balanced_genders,
    dtype=np.int64
)


# ----------------------------------------------------------
# Convert to tensors
# ----------------------------------------------------------

balanced_windows_tensor = (
    torch.from_numpy(
        balanced_windows
    )
    .unsqueeze(1)
)

balanced_activities_tensor = (
    torch.from_numpy(
        balanced_activities
    )
)

balanced_genders_tensor = (
    torch.from_numpy(
        balanced_genders
    )
)


# ----------------------------------------------------------
# Distribution checks
# ----------------------------------------------------------

activity_counts = np.bincount(
    balanced_activities,
    minlength=NUM_ACTIVITIES
)

gender_counts = np.bincount(
    balanced_genders,
    minlength=NUM_GENDERS
)

gender_activity_counts = np.zeros(
    (
        NUM_GENDERS,
        NUM_ACTIVITIES
    ),
    dtype=np.int64
)

for gender, activity in zip(
    balanced_genders,
    balanced_activities
):

    gender_activity_counts[
        gender,
        activity
    ] += 1


print(
    "\nTotal balanced windows :",
    len(balanced_windows)
)

print(
    "Activity counts        :",
    activity_counts.tolist()
)

print(
    "Gender counts          :",
    gender_counts.tolist()
)

print(
    "\nGender x activity counts:"
)

print(
    gender_activity_counts
)


# ==========================================================
# Evaluation Helper
# ==========================================================

@torch.no_grad()
def evaluate_signal_batch(
    signal_windows,
    true_activities,
    true_genders
):

    signal_windows = signal_windows.to(
        device
    )

    activity_logits, _ = surrogate_model(
        signal_windows
    )

    gender_logits, _ = privacy_model(
        signal_windows
    )

    activity_predictions = (
        activity_logits
        .argmax(dim=1)
        .cpu()
        .numpy()
    )

    gender_predictions = (
        gender_logits
        .argmax(dim=1)
        .cpu()
        .numpy()
    )

    true_activities_np = (
        true_activities
        .cpu()
        .numpy()
    )

    true_genders_np = (
        true_genders
        .cpu()
        .numpy()
    )

    activity_accuracy = accuracy_score(
        true_activities_np,
        activity_predictions
    )

    activity_f1 = f1_score(
        true_activities_np,
        activity_predictions,
        average="macro",
        zero_division=0
    )

    gender_accuracy = accuracy_score(
        true_genders_np,
        gender_predictions
    )

    gender_f1 = f1_score(
        true_genders_np,
        gender_predictions,
        average="macro",
        zero_division=0
    )

    return {
        "activity_accuracy":
            activity_accuracy,

        "activity_f1":
            activity_f1,

        "gender_accuracy":
            gender_accuracy,

        "gender_f1":
            gender_f1
    }


# ==========================================================
# Evaluate Raw Balanced Signals
# ==========================================================

print("\n" + "=" * 90)
print("PRIVACY-GUIDED BALANCED EVALUATION")
print("=" * 90)

print(
    "Evaluation timesteps:",
    EVALUATION_TIMESTEPS
)

print(
    "Evaluation samples  :",
    len(
        balanced_windows_tensor
    )
)


raw_metrics = evaluate_signal_batch(
    balanced_windows_tensor,
    balanced_activities_tensor,
    balanced_genders_tensor
)


print(
    "\nRaw complete | "
    f"Activity accuracy: "
    f"{raw_metrics['activity_accuracy']:.4f} | "
    f"Activity F1: "
    f"{raw_metrics['activity_f1']:.4f} | "
    f"Gender accuracy: "
    f"{raw_metrics['gender_accuracy']:.4f} | "
    f"Gender F1: "
    f"{raw_metrics['gender_f1']:.4f}"
)


# ==========================================================
# Results Storage
# ==========================================================

results = []

results.append({

    "Timestep":
        "Raw",

    "Samples":
        len(
            balanced_windows_tensor
        ),

    "Activity Accuracy":
        raw_metrics[
            "activity_accuracy"
        ],

    "Activity Macro F1":
        raw_metrics[
            "activity_f1"
        ],

    "Gender Accuracy":
        raw_metrics[
            "gender_accuracy"
        ],

    "Gender Macro F1":
        raw_metrics[
            "gender_f1"
        ],

    "Mean Absolute Change":
        0.0,

    "Root Mean Squared Change":
        0.0,

    "Activity Accuracy Change":
        0.0,

    "Gender Accuracy Change":
        0.0
})


# ==========================================================
# Evaluate Privacy-Guided Obfuscation
# ==========================================================

evaluation_start_time = (
    time.time()
)

EVALUATION_BATCH_SIZE = 16


for timestep in EVALUATION_TIMESTEPS:

    print(
        f"\nEvaluating privacy-guided "
        f"timestep {timestep}..."
    )

    timestep_start = time.time()

    all_obfuscated = []

    # ------------------------------------------------------
    # Process balanced data in small batches
    # ------------------------------------------------------

    for start_index in range(
        0,
        len(balanced_windows_tensor),
        EVALUATION_BATCH_SIZE
    ):

        end_index = min(
            start_index
            + EVALUATION_BATCH_SIZE,
            len(balanced_windows_tensor)
        )

        batch_windows = (
            balanced_windows_tensor[
                start_index:end_index
            ]
            .to(device)
        )

        batch_activities = (
            balanced_activities_tensor[
                start_index:end_index
            ]
            .to(device)
        )

        obfuscated_batch = (
            privacy_guided_obfuscate(
                batch_windows,
                batch_activities,
                timestep=timestep
            )
        )

        all_obfuscated.append(
            obfuscated_batch
            .detach()
            .cpu()
        )


    # ------------------------------------------------------
    # Combine reconstructed windows
    # ------------------------------------------------------

    obfuscated_windows_tensor = (
        torch.cat(
            all_obfuscated,
            dim=0
        )
    )


    # ------------------------------------------------------
    # Classification metrics
    # ------------------------------------------------------

    metrics = evaluate_signal_batch(
        obfuscated_windows_tensor,
        balanced_activities_tensor,
        balanced_genders_tensor
    )


    # ------------------------------------------------------
    # Signal distortion metrics
    # ------------------------------------------------------

    difference = (
        obfuscated_windows_tensor
        -
        balanced_windows_tensor
    )

    mean_absolute_change = (
        torch.mean(
            torch.abs(
                difference
            )
        )
        .item()
    )

    rmse = (
        torch.sqrt(
            torch.mean(
                difference ** 2
            )
        )
        .item()
    )


    # ------------------------------------------------------
    # Changes relative to raw data
    # ------------------------------------------------------

    activity_accuracy_change = (
        metrics[
            "activity_accuracy"
        ]
        -
        raw_metrics[
            "activity_accuracy"
        ]
    )

    gender_accuracy_change = (
        metrics[
            "gender_accuracy"
        ]
        -
        raw_metrics[
            "gender_accuracy"
        ]
    )


    timestep_time = (
        time.time()
        - timestep_start
    )


    print(
        f"Timestep {timestep} complete | "
        f"Activity accuracy: "
        f"{metrics['activity_accuracy']:.4f} | "
        f"Activity F1: "
        f"{metrics['activity_f1']:.4f} | "
        f"Gender accuracy: "
        f"{metrics['gender_accuracy']:.4f} | "
        f"Gender F1: "
        f"{metrics['gender_f1']:.4f} | "
        f"MAC: "
        f"{mean_absolute_change:.4f} | "
        f"RMSE: "
        f"{rmse:.4f} | "
        f"Time: "
        f"{timestep_time:.1f} sec"
    )


    results.append({

        "Timestep":
            timestep,

        "Samples":
            len(
                balanced_windows_tensor
            ),

        "Activity Accuracy":
            metrics[
                "activity_accuracy"
            ],

        "Activity Macro F1":
            metrics[
                "activity_f1"
            ],

        "Gender Accuracy":
            metrics[
                "gender_accuracy"
            ],

        "Gender Macro F1":
            metrics[
                "gender_f1"
            ],

        "Mean Absolute Change":
            mean_absolute_change,

        "Root Mean Squared Change":
            rmse,

        "Activity Accuracy Change":
            activity_accuracy_change,

        "Gender Accuracy Change":
            gender_accuracy_change
    })


# ==========================================================
# Create Results Table
# ==========================================================

results_dataframe = pd.DataFrame(
    results
)

numeric_columns = [
    column
    for column
    in results_dataframe.columns
    if column not in {
        "Timestep",
        "Samples"
    }
]

results_dataframe[
    numeric_columns
] = results_dataframe[
    numeric_columns
].round(4)


print("\n" + "=" * 110)

print(
    "PRIVACY-GUIDED BALANCED "
    "PRIVACY-UTILITY RESULTS"
)

print("=" * 110)

print(
    results_dataframe.to_string(
        index=False
    )
)


# ==========================================================
# Save Results
# ==========================================================

PRIVACY_GUIDED_RESULTS_FILE = (
    WORK_DIR
    / "privacy_guided_balanced_results.csv"
)

results_dataframe.to_csv(
    PRIVACY_GUIDED_RESULTS_FILE,
    index=False
)


# ----------------------------------------------------------
# Save detailed tensors too
# ----------------------------------------------------------

PRIVACY_GUIDED_DETAILS_FILE = (
    WORK_DIR
    / "privacy_guided_balanced_details.pt"
)

torch.save(
    {
        "results":
            results,

        "balanced_windows":
            balanced_windows_tensor,

        "balanced_activities":
            balanced_activities_tensor,

        "balanced_genders":
            balanced_genders_tensor,

        "evaluation_timesteps":
            EVALUATION_TIMESTEPS
    },
    PRIVACY_GUIDED_DETAILS_FILE
)


total_evaluation_time = (
    time.time()
    - evaluation_start_time
)


print("-" * 110)

print(
    "Total evaluation time:",
    f"{total_evaluation_time:.1f} seconds"
)

print(
    "Results CSV saved to:",
    PRIVACY_GUIDED_RESULTS_FILE
)

print(
    "Detailed results saved to:",
    PRIVACY_GUIDED_DETAILS_FILE
)

print(
    "\nPrivacy-guided balanced evaluation "
    "completed successfully."
)


# In[ ]:




